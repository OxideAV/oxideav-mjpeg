#![cfg(feature = "registry")]
// Parallel-array index loops are idiomatic in these per-component
// roundtrip checks; skip the lint (same allow as `src/lib.rs`).
#![allow(clippy::needless_range_loop)]
//! Bit-exact encode→decode roundtrips for the public lossless JPEG (SOF3)
//! encoder.
//!
//! The encoder lives in `oxideav_mjpeg::encoder::encode_lossless_jpeg_grayscale`
//! and produces a standalone single-component grayscale JPEG with predictor
//! selectable in `1..=7` (T.81 Table H.1) at any precision `P ∈ 2..=16`. The
//! decoder side has supported P=2..=16 grayscale lossless since round 0.1.0,
//! so these tests exercise the two halves end-to-end.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};
use oxideav_mjpeg::encoder::{
    encode_lossless_jpeg_cmyk, encode_lossless_jpeg_cmyk_with_opts, encode_lossless_jpeg_grayscale,
    encode_lossless_jpeg_grayscale_with_opts, encode_lossless_jpeg_rgb,
    encode_lossless_jpeg_rgb_with_opts,
};
use oxideav_mjpeg::registry::make_decoder;

/// Build a deterministic, textured 8-bit grayscale source so the residuals
/// exercise multiple Huffman categories (smooth-gradient samples would all
/// land in SSSS=1..2).
fn mk_samples_8bit(w: usize, h: usize) -> Vec<u8> {
    let mut samples = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // Mix a smooth gradient with a high-frequency XOR pattern so
            // residuals span a wide range of categories.
            samples[j * w + i] =
                ((i as i32 * 3 + j as i32 * 5 + ((i ^ j) as i32 & 31)) & 0xFF) as u8;
        }
    }
    samples
}

/// Build a deterministic textured source for higher precisions (P>8). Stored
/// as little-endian 16-bit samples with stride = w * 2 bytes.
fn mk_samples_wide(w: usize, h: usize, precision: u8) -> Vec<u8> {
    let max = (1u32 << precision) - 1;
    let mut bytes = vec![0u8; w * h * 2];
    for j in 0..h {
        for i in 0..w {
            // Gradient that exercises full dynamic range plus a bit of
            // high-frequency texture to push some residuals up the SSSS
            // ladder.
            let lin = ((i as u32 * (max / w.max(1) as u32))
                + (j as u32 * (max / h.max(1) as u32) / 2))
                & max;
            let bump = ((i as u32 ^ j as u32) * 7) & max;
            let v = (lin ^ (bump >> 3)) & max;
            let v = v as u16;
            bytes[(j * w + i) * 2] = (v & 0xFF) as u8;
            bytes[(j * w + i) * 2 + 1] = (v >> 8) as u8;
        }
    }
    bytes
}

fn decode_to_frame(jpeg: Vec<u8>, w: u32, h: u32) -> oxideav_core::VideoFrame {
    let mut params = CodecParameters::video(CodecId::new("mjpeg"));
    params.width = Some(w);
    params.height = Some(h);
    let mut dec = make_decoder(&params).expect("make_decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), jpeg))
        .expect("send_packet");
    let Frame::Video(v) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected VideoFrame")
    };
    v
}

#[test]
fn lossless_8bit_every_predictor_is_bit_exact() {
    let w = 24u32;
    let h = 16u32;
    let samples = mk_samples_8bit(w as usize, h as usize);
    for predictor in 1u8..=7 {
        let jpeg = encode_lossless_jpeg_grayscale(w, h, &samples, w as usize, 8, predictor)
            .unwrap_or_else(|e| panic!("encode predictor={predictor}: {e:?}"));
        // SOF3 marker must be present.
        assert!(
            jpeg.windows(2).any(|x| x == [0xFF, 0xC3]),
            "predictor={predictor}: SOF3 marker missing"
        );
        let v = decode_to_frame(jpeg, w, h);
        assert_eq!(v.planes.len(), 1, "predictor={predictor}");
        let stride = v.planes[0].stride;
        for j in 0..h as usize {
            for i in 0..w as usize {
                let got = v.planes[0].data[j * stride + i];
                let want = samples[j * w as usize + i];
                assert_eq!(got, want, "predictor={predictor} mismatch at ({i},{j})");
            }
        }
    }
}

#[test]
fn lossless_12bit_predictor_1_is_bit_exact() {
    let w = 16u32;
    let h = 12u32;
    let precision = 12u8;
    let bytes = mk_samples_wide(w as usize, h as usize, precision);
    let jpeg = encode_lossless_jpeg_grayscale(w, h, &bytes, (w as usize) * 2, precision, 1)
        .expect("encode 12-bit lossless");
    // P=12 should produce a Gray12Le output.
    let v = decode_to_frame(jpeg, w, h);
    assert_eq!(v.planes.len(), 1);
    let stride = v.planes[0].stride;
    assert_eq!(
        stride,
        (w as usize) * 2,
        "expected 16-bit-per-sample stride"
    );
    for j in 0..h as usize {
        for i in 0..w as usize {
            let lo_in = bytes[(j * w as usize + i) * 2] as u16;
            let hi_in = bytes[(j * w as usize + i) * 2 + 1] as u16;
            let want = lo_in | (hi_in << 8);
            let lo = v.planes[0].data[j * stride + i * 2] as u16;
            let hi = v.planes[0].data[j * stride + i * 2 + 1] as u16;
            let got = lo | (hi << 8);
            assert_eq!(got, want, "12-bit mismatch at ({i},{j})");
        }
    }
}

#[test]
fn lossless_16bit_predictor_4_is_bit_exact() {
    // 16-bit precision exercises the SSSS=16 / Di=32768 special case
    // (Annex H.1.2.2). We include the value 0 next to 32768 in the
    // gradient so the residual at that pixel forces the half-modulus
    // path.
    let w = 16u32;
    let h = 8u32;
    let precision = 16u8;
    let mut bytes = vec![0u8; (w * h * 2) as usize];
    for j in 0..h as usize {
        for i in 0..w as usize {
            // Alternate 0 and 0x8000 so adjacent diffs cycle through
            // -32768 / +32768 / 0 — hitting the Di=32768 corner case.
            let v: u16 = if (i + j) % 2 == 0 { 0 } else { 0x8000 };
            bytes[(j * w as usize + i) * 2] = (v & 0xFF) as u8;
            bytes[(j * w as usize + i) * 2 + 1] = (v >> 8) as u8;
        }
    }
    let jpeg = encode_lossless_jpeg_grayscale(w, h, &bytes, (w as usize) * 2, precision, 4)
        .expect("encode 16-bit lossless");
    let v = decode_to_frame(jpeg, w, h);
    assert!(matches!(
        v.planes[0].data.len(),
        n if n == (w * h * 2) as usize
    ));
    let stride = v.planes[0].stride;
    for j in 0..h as usize {
        for i in 0..w as usize {
            let lo_in = bytes[(j * w as usize + i) * 2] as u16;
            let hi_in = bytes[(j * w as usize + i) * 2 + 1] as u16;
            let want = lo_in | (hi_in << 8);
            let lo = v.planes[0].data[j * stride + i * 2] as u16;
            let hi = v.planes[0].data[j * stride + i * 2 + 1] as u16;
            let got = lo | (hi << 8);
            assert_eq!(got, want, "16-bit mismatch at ({i},{j})");
        }
    }
}

/// Build three textured 8-bit planes (one each for R, G, B) so each
/// component independently exercises the predictor + Huffman magnitude
/// ladder. The planes are deliberately decorrelated so a bug in
/// per-component buffer indexing (e.g. predicting R from G's neighbours)
/// would produce a visible mismatch rather than a coincidentally-correct
/// roundtrip on a single-colour source.
fn mk_rgb_8bit(w: usize, h: usize) -> [Vec<u8>; 3] {
    let mut r = vec![0u8; w * h];
    let mut g = vec![0u8; w * h];
    let mut b = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            r[j * w + i] = ((i as i32 * 5 + j as i32 * 3 + ((i ^ j) as i32 & 31)) & 0xFF) as u8;
            g[j * w + i] =
                ((i as i32 * 11 + j as i32 * 7 + ((i.wrapping_mul(j)) as i32 & 63)) & 0xFF) as u8;
            b[j * w + i] = ((255 - (i as i32 * 2 + j as i32 * 9)) & 0xFF) as u8;
        }
    }
    [r, g, b]
}

#[test]
fn lossless_rgb_8bit_every_predictor_is_bit_exact() {
    let w = 24u32;
    let h = 16u32;
    let planes = mk_rgb_8bit(w as usize, h as usize);
    let strides = [w as usize, w as usize, w as usize];
    for predictor in 1u8..=7 {
        let jpeg = encode_lossless_jpeg_rgb(
            w,
            h,
            [&planes[0], &planes[1], &planes[2]],
            strides,
            8,
            predictor,
        )
        .unwrap_or_else(|e| panic!("encode predictor={predictor}: {e:?}"));
        // SOF3 marker must be present.
        assert!(
            jpeg.windows(2).any(|x| x == [0xFF, 0xC3]),
            "predictor={predictor}: SOF3 marker missing"
        );
        let v = decode_to_frame(jpeg, w, h);
        assert_eq!(v.planes.len(), 1, "predictor={predictor}");
        let stride = v.planes[0].stride;
        assert_eq!(
            stride,
            (w as usize) * 3,
            "predictor={predictor}: expected packed Rgb24 stride"
        );
        for j in 0..h as usize {
            for i in 0..w as usize {
                let got_r = v.planes[0].data[j * stride + i * 3];
                let got_g = v.planes[0].data[j * stride + i * 3 + 1];
                let got_b = v.planes[0].data[j * stride + i * 3 + 2];
                assert_eq!(
                    got_r,
                    planes[0][j * w as usize + i],
                    "predictor={predictor} R mismatch at ({i},{j})"
                );
                assert_eq!(
                    got_g,
                    planes[1][j * w as usize + i],
                    "predictor={predictor} G mismatch at ({i},{j})"
                );
                assert_eq!(
                    got_b,
                    planes[2][j * w as usize + i],
                    "predictor={predictor} B mismatch at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn lossless_rgb_rejects_oversized_sample() {
    // Precision = 4 (max value 15) but a sample is 200: encoder must
    // reject up-front rather than producing a stream the decoder can't
    // reconstruct.
    let w = 4u32;
    let h = 4u32;
    let mut r = vec![0u8; (w * h) as usize];
    let g = vec![0u8; (w * h) as usize];
    let b = vec![0u8; (w * h) as usize];
    r[0] = 200; // > 2^4 - 1
    let err = encode_lossless_jpeg_rgb(
        w,
        h,
        [&r, &g, &b],
        [w as usize, w as usize, w as usize],
        4,
        1,
    );
    assert!(
        err.is_err(),
        "expected encoder to reject sample exceeding declared precision"
    );
}

/// Build three textured high-bit-depth planes (one each for the three
/// SOS components) at the requested precision, packed as little-endian
/// 16-bit samples with stride = w * 2 bytes per plane. Each plane is
/// independently textured so a per-component buffer-indexing bug
/// (cross-talk between components) would produce a visible mismatch.
fn mk_rgb_wide(w: usize, h: usize, precision: u8) -> [Vec<u8>; 3] {
    let max = (1u32 << precision) - 1;
    let mut out: [Vec<u8>; 3] = [
        vec![0u8; w * h * 2],
        vec![0u8; w * h * 2],
        vec![0u8; w * h * 2],
    ];
    for j in 0..h {
        for i in 0..w {
            // Three decorrelated textures, one per component. The XOR /
            // multiply combinations keep residuals from collapsing to
            // SSSS=0/1 while staying entirely inside `0..=max`.
            let v0 = ((i as u32 * 5 + j as u32 * 3 + ((i ^ j) as u32)) & max) as u16;
            let v1 =
                ((i as u32 * 11 + j as u32 * 7 + ((i.wrapping_mul(j)) as u32 & 63)) & max) as u16;
            let v2 = ((max ^ ((i as u32 * 2 + j as u32 * 9) & max)) & max) as u16;
            let pix = j * w + i;
            out[0][pix * 2] = (v0 & 0xFF) as u8;
            out[0][pix * 2 + 1] = (v0 >> 8) as u8;
            out[1][pix * 2] = (v1 & 0xFF) as u8;
            out[1][pix * 2 + 1] = (v1 >> 8) as u8;
            out[2][pix * 2] = (v2 & 0xFF) as u8;
            out[2][pix * 2 + 1] = (v2 >> 8) as u8;
        }
    }
    out
}

/// Read a little-endian u16 from a plane at logical `(i, j)`.
fn read_le_u16(plane: &[u8], stride: usize, i: usize, j: usize) -> u16 {
    let lo = plane[j * stride + i * 2] as u16;
    let hi = plane[j * stride + i * 2 + 1] as u16;
    lo | (hi << 8)
}

#[test]
fn lossless_rgb_10bit_every_predictor_planar_gbrp10() {
    // P = 10 three-component decode lands on planar `Gbrp10Le`: three
    // 16-bit-storage planes in SOS-component order. The decoder is
    // colour-agnostic — caller plane-order is preserved end-to-end —
    // so each input plane round-trips bit-exact through the matching
    // output plane.
    let w = 12u32;
    let h = 8u32;
    let precision = 10u8;
    let planes = mk_rgb_wide(w as usize, h as usize, precision);
    let strides = [(w as usize) * 2, (w as usize) * 2, (w as usize) * 2];
    for predictor in 1u8..=7 {
        let jpeg = encode_lossless_jpeg_rgb(
            w,
            h,
            [&planes[0], &planes[1], &planes[2]],
            strides,
            precision,
            predictor,
        )
        .unwrap_or_else(|e| panic!("encode predictor={predictor}: {e:?}"));
        let v = decode_to_frame(jpeg, w, h);
        assert_eq!(
            v.planes.len(),
            3,
            "predictor={predictor}: expected 3 Gbrp10Le planes"
        );
        for ci in 0..3 {
            assert_eq!(
                v.planes[ci].stride,
                (w as usize) * 2,
                "predictor={predictor} comp={ci}: expected 2 bytes/sample stride"
            );
            for j in 0..h as usize {
                for i in 0..w as usize {
                    let want = read_le_u16(&planes[ci], (w as usize) * 2, i, j);
                    let got = read_le_u16(&v.planes[ci].data, v.planes[ci].stride, i, j);
                    assert_eq!(
                        got, want,
                        "predictor={predictor} comp={ci} mismatch at ({i},{j})"
                    );
                }
            }
        }
    }
}

#[test]
fn lossless_rgb_12bit_predictor_4_planar_gbrp12() {
    // P = 12 lands on `Gbrp12Le`; predictor 4 exercises the
    // two-dimensional `Ra + Rb - Rc` form (touches all three neighbour
    // history slots and the `wrapping_add` / `wrapping_sub` path that
    // matters at the modulus boundary).
    let w = 16u32;
    let h = 12u32;
    let precision = 12u8;
    let planes = mk_rgb_wide(w as usize, h as usize, precision);
    let strides = [(w as usize) * 2, (w as usize) * 2, (w as usize) * 2];
    let jpeg = encode_lossless_jpeg_rgb(
        w,
        h,
        [&planes[0], &planes[1], &planes[2]],
        strides,
        precision,
        4,
    )
    .expect("encode 12-bit 3-component");
    let v = decode_to_frame(jpeg, w, h);
    assert_eq!(v.planes.len(), 3, "expected 3 Gbrp12Le planes");
    for ci in 0..3 {
        for j in 0..h as usize {
            for i in 0..w as usize {
                let want = read_le_u16(&planes[ci], (w as usize) * 2, i, j);
                let got = read_le_u16(&v.planes[ci].data, v.planes[ci].stride, i, j);
                assert_eq!(got, want, "comp={ci} mismatch at ({i},{j})");
            }
        }
    }
}

#[test]
fn lossless_rgb_14bit_predictor_7_planar_gbrp14() {
    // P = 14 lands on `Gbrp14Le`; predictor 7 exercises the
    // `(Ra + Rb) >> 1` averaging path.
    let w = 8u32;
    let h = 8u32;
    let precision = 14u8;
    let planes = mk_rgb_wide(w as usize, h as usize, precision);
    let strides = [(w as usize) * 2, (w as usize) * 2, (w as usize) * 2];
    let jpeg = encode_lossless_jpeg_rgb(
        w,
        h,
        [&planes[0], &planes[1], &planes[2]],
        strides,
        precision,
        7,
    )
    .expect("encode 14-bit 3-component");
    let v = decode_to_frame(jpeg, w, h);
    assert_eq!(v.planes.len(), 3, "expected 3 Gbrp14Le planes");
    for ci in 0..3 {
        for j in 0..h as usize {
            for i in 0..w as usize {
                let want = read_le_u16(&planes[ci], (w as usize) * 2, i, j);
                let got = read_le_u16(&v.planes[ci].data, v.planes[ci].stride, i, j);
                assert_eq!(got, want, "comp={ci} mismatch at ({i},{j})");
            }
        }
    }
}

#[test]
fn lossless_rgb_16bit_predictor_1_packed_rgb48() {
    // P = 16: no `Gbrp16Le` variant exists, so the decoder widens onto
    // packed `Rgb48Le` (6 bytes per pixel, scan-component order). This
    // is the "widest container" fallback that mirrors how grayscale
    // P = 14 widens onto `Gray16Le`. Predictor 1 (left-only) keeps the
    // test independent of vertical neighbours.
    let w = 8u32;
    let h = 6u32;
    let precision = 16u8;
    let planes = mk_rgb_wide(w as usize, h as usize, precision);
    let strides = [(w as usize) * 2, (w as usize) * 2, (w as usize) * 2];
    let jpeg = encode_lossless_jpeg_rgb(
        w,
        h,
        [&planes[0], &planes[1], &planes[2]],
        strides,
        precision,
        1,
    )
    .expect("encode 16-bit 3-component");
    let v = decode_to_frame(jpeg, w, h);
    assert_eq!(v.planes.len(), 1, "expected 1 packed Rgb48Le plane");
    let stride = v.planes[0].stride;
    assert_eq!(
        stride,
        (w as usize) * 6,
        "expected 6 bytes/pixel (Rgb48Le) stride"
    );
    for j in 0..h as usize {
        for i in 0..w as usize {
            // Packed layout: c0-low, c0-high, c1-low, c1-high, c2-low, c2-high.
            let base = j * stride + i * 6;
            let got_c0 =
                (v.planes[0].data[base] as u16) | ((v.planes[0].data[base + 1] as u16) << 8);
            let got_c1 =
                (v.planes[0].data[base + 2] as u16) | ((v.planes[0].data[base + 3] as u16) << 8);
            let got_c2 =
                (v.planes[0].data[base + 4] as u16) | ((v.planes[0].data[base + 5] as u16) << 8);
            assert_eq!(
                got_c0,
                read_le_u16(&planes[0], (w as usize) * 2, i, j),
                "c0 mismatch at ({i},{j})"
            );
            assert_eq!(
                got_c1,
                read_le_u16(&planes[1], (w as usize) * 2, i, j),
                "c1 mismatch at ({i},{j})"
            );
            assert_eq!(
                got_c2,
                read_le_u16(&planes[2], (w as usize) * 2, i, j),
                "c2 mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn lossless_rgb_odd_precision_9_widens_to_rgb48() {
    // P = 9 has no exact-bit-depth pixel format, so it widens onto
    // packed `Rgb48Le` just like 11 / 13 / 15. Verify the round-trip
    // is still bit-exact (samples sit in the low 9 bits of each
    // 16-bit word).
    let w = 6u32;
    let h = 4u32;
    let precision = 9u8;
    let planes = mk_rgb_wide(w as usize, h as usize, precision);
    let strides = [(w as usize) * 2, (w as usize) * 2, (w as usize) * 2];
    let jpeg = encode_lossless_jpeg_rgb(
        w,
        h,
        [&planes[0], &planes[1], &planes[2]],
        strides,
        precision,
        1,
    )
    .expect("encode 9-bit 3-component");
    let v = decode_to_frame(jpeg, w, h);
    assert_eq!(v.planes.len(), 1, "P=9 should widen to single packed plane");
    let stride = v.planes[0].stride;
    assert_eq!(stride, (w as usize) * 6, "expected Rgb48Le stride");
    for j in 0..h as usize {
        for i in 0..w as usize {
            let base = j * stride + i * 6;
            for ci in 0..3 {
                let got = (v.planes[0].data[base + ci * 2] as u16)
                    | ((v.planes[0].data[base + ci * 2 + 1] as u16) << 8);
                let want = read_le_u16(&planes[ci], (w as usize) * 2, i, j);
                // P = 9: only the low 9 bits are meaningful; the top 7
                // bits of the output word must be zero (no sign-extension
                // or stray bits from the predictor arithmetic).
                assert_eq!(
                    got >> 9,
                    0,
                    "P=9 comp={ci} stray high bits at ({i},{j}): got={got:#06x}"
                );
                assert_eq!(got, want, "comp={ci} mismatch at ({i},{j})");
            }
        }
    }
}

#[test]
fn lossless_decoder_reports_gray_pixel_format_for_each_precision() {
    // P=8 → Gray8, P=10 → Gray10Le, P=12 → Gray12Le, else Gray16Le.
    let w = 8u32;
    let h = 8u32;
    let cases: &[(u8, PixelFormat)] = &[
        (8, PixelFormat::Gray8),
        (10, PixelFormat::Gray10Le),
        (12, PixelFormat::Gray12Le),
        (14, PixelFormat::Gray16Le),
        (16, PixelFormat::Gray16Le),
    ];
    for &(p, expected_pf) in cases {
        let bytes = if p == 8 {
            mk_samples_8bit(w as usize, h as usize)
        } else {
            mk_samples_wide(w as usize, h as usize, p)
        };
        let stride = if p == 8 { w as usize } else { (w as usize) * 2 };
        let jpeg = encode_lossless_jpeg_grayscale(w, h, &bytes, stride, p, 1)
            .unwrap_or_else(|e| panic!("encode P={p}: {e:?}"));
        let mut params = CodecParameters::video(CodecId::new("mjpeg"));
        params.width = Some(w);
        params.height = Some(h);
        let mut dec = make_decoder(&params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), jpeg))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!()
        };
        // The decoder doesn't surface a per-frame PixelFormat on the
        // slim VideoFrame, but it does pick stride based on the
        // expected output format. Verify byte width matches expectation.
        let bpp = match expected_pf {
            PixelFormat::Gray8 => 1,
            PixelFormat::Gray10Le | PixelFormat::Gray12Le | PixelFormat::Gray16Le => 2,
            _ => unreachable!(),
        };
        assert_eq!(
            v.planes[0].stride,
            (w as usize) * bpp,
            "P={p} expected stride {} bytes/row",
            (w as usize) * bpp
        );
    }
}

#[test]
fn lossless_encoder_rejects_out_of_range_precision() {
    let w = 4u32;
    let h = 4u32;
    let samples = vec![0u8; (w * h) as usize];
    let err = encode_lossless_jpeg_grayscale(w, h, &samples, w as usize, 1, 1).unwrap_err();
    assert!(format!("{err:?}").to_lowercase().contains("unsupported"));
    let err = encode_lossless_jpeg_grayscale(w, h, &samples, w as usize, 17, 1).unwrap_err();
    assert!(format!("{err:?}").to_lowercase().contains("unsupported"));
}

#[test]
fn lossless_encoder_rejects_invalid_predictor() {
    let w = 4u32;
    let h = 4u32;
    let samples = vec![0u8; (w * h) as usize];
    let err = encode_lossless_jpeg_grayscale(w, h, &samples, w as usize, 8, 0).unwrap_err();
    assert!(format!("{err:?}").to_lowercase().contains("predictor"));
    let err = encode_lossless_jpeg_grayscale(w, h, &samples, w as usize, 8, 8).unwrap_err();
    assert!(format!("{err:?}").to_lowercase().contains("predictor"));
}

/// Trait-API end-to-end: configure `MjpegEncoder` with `Gray8` input +
/// `set_lossless(true)`, push a frame, pull the packet, decode it through
/// the matching trait-API decoder, and verify byte-exact recovery.
#[test]
fn registry_encoder_gray8_lossless_roundtrip() {
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{Encoder, VideoFrame};

    let w = 20u32;
    let h = 12u32;
    let samples = mk_samples_8bit(w as usize, h as usize);

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Gray8);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).unwrap();
    enc.set_lossless(true);
    enc.set_lossless_predictor(3); // exercise a 2-D predictor (Rc)

    let frame = Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: w as usize,
            data: samples.clone(),
        }],
    });
    enc.send_frame(&frame).unwrap();
    let pkt = enc.receive_packet().unwrap();
    assert!(
        pkt.data.windows(2).any(|x| x == [0xFF, 0xC3]),
        "registry encoder output missing SOF3 marker"
    );

    let v = decode_to_frame(pkt.data.clone(), w, h);
    assert_eq!(v.planes.len(), 1);
    let stride = v.planes[0].stride;
    for j in 0..h as usize {
        for i in 0..w as usize {
            let got = v.planes[0].data[j * stride + i];
            let want = samples[j * w as usize + i];
            assert_eq!(got, want, "registry roundtrip mismatch at ({i},{j})");
        }
    }
}

/// `Gray8` input without `set_lossless(true)` now takes the baseline
/// (SOF0) single-component DCT path — the encoder emits a complete
/// SOF0 + SOS bitstream and the decoder recovers a `Gray8` frame.
#[test]
fn registry_encoder_gray8_without_lossless_flag_takes_baseline() {
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{Encoder, VideoFrame};

    let w = 8u32;
    let h = 8u32;
    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Gray8);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).unwrap();
    // No `set_lossless(true)` here — default is the new baseline path.
    let frame = Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: w as usize,
            data: vec![100u8; (w * h) as usize],
        }],
    });
    enc.send_frame(&frame).expect("send_frame baseline Gray8");
    let pkt = enc.receive_packet().expect("recv baseline pkt");
    // Output must carry SOF0 (lossy baseline), not SOF3 (lossless).
    assert!(pkt.data.windows(2).any(|w| w == [0xFF, 0xC0]));
    assert!(!pkt.data.windows(2).any(|w| w == [0xFF, 0xC3]));

    // Decode the result back to a `Gray8` frame.
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    dec.send_packet(&oxideav_core::Packet::new(
        0,
        oxideav_core::TimeBase::new(1, 30),
        pkt.data,
    ))
    .expect("send pkt");
    let Frame::Video(v) = dec.receive_frame().expect("decode") else {
        panic!("expected video frame")
    };
    assert_eq!(v.planes.len(), 1);
}

/// Higher-precision grayscale (10 / 12 / 16-bit) still requires
/// `set_lossless(true)` — the baseline DCT path is 8-bit by spec.
#[test]
fn registry_encoder_gray12_without_lossless_flag_errors() {
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{Encoder, VideoFrame};

    let w = 8u32;
    let h = 8u32;
    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Gray12Le);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).unwrap();
    // No set_lossless(true) — 12-bit DCT isn't a thing on this path.
    let frame = Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: (w as usize) * 2,
            data: vec![0u8; (w as usize) * 2 * (h as usize)],
        }],
    });
    let err = enc.send_frame(&frame).unwrap_err();
    assert!(
        matches!(err, oxideav_core::Error::Unsupported(_)),
        "expected Unsupported, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Lossless encoder restart-marker + point-transform coverage (round 89).
//
// The decoder side has supported `RSTn` resets and non-zero `Pt` for the
// lossless path since round 0.1.0; these tests exercise the matching
// encoder options shipped in `*_with_opts` variants.
// ---------------------------------------------------------------------------

/// Confirm DRI + at least one `RSTn` marker land in the bitstream when
/// `restart_interval > 0`. Pure header-level: independent from the decode
/// roundtrip below.
#[test]
fn lossless_grayscale_with_restart_emits_dri_and_rst() {
    let w = 16u32;
    let h = 8u32;
    let samples = mk_samples_8bit(w as usize, h as usize);
    let ri: u16 = 32; // 32 samples per restart on a 128-sample image → 3 RSTn.
    let jpeg = encode_lossless_jpeg_grayscale_with_opts(w, h, &samples, w as usize, 8, 1, ri, 0)
        .expect("encode with restart");
    // DRI = 0xFF 0xDD.
    assert!(
        jpeg.windows(2).any(|x| x == [0xFF, 0xDD]),
        "DRI marker missing"
    );
    // At least one RSTn marker (0xD0..=0xD7) — not stuff-prefixed, so the
    // first 0xFF in the pair must be a real marker introducer.
    let rst_count = jpeg
        .windows(2)
        .filter(|x| x[0] == 0xFF && (0xD0..=0xD7).contains(&x[1]))
        .count();
    assert!(rst_count >= 3, "expected ≥3 RSTn markers, got {rst_count}");
}

/// 8-bit grayscale lossless with a non-trivial restart interval must
/// still round-trip exactly through the SOF3 decoder, for every
/// predictor 1..=7. The restart interval is set so that several
/// boundaries fall inside the image (not just at the end).
#[test]
fn lossless_grayscale_with_restart_roundtrips_every_predictor() {
    let w = 24u32;
    let h = 16u32;
    let samples = mk_samples_8bit(w as usize, h as usize);
    let ri: u16 = 17; // forces ≥22 RSTn on a 384-sample image
    for predictor in 1u8..=7 {
        let jpeg = encode_lossless_jpeg_grayscale_with_opts(
            w, h, &samples, w as usize, 8, predictor, ri, 0,
        )
        .unwrap_or_else(|e| panic!("encode predictor={predictor}: {e:?}"));
        let v = decode_to_frame(jpeg, w, h);
        let stride = v.planes[0].stride;
        for j in 0..h as usize {
            for i in 0..w as usize {
                assert_eq!(
                    v.planes[0].data[j * stride + i],
                    samples[j * w as usize + i],
                    "predictor={predictor} mismatch at ({i},{j})"
                );
            }
        }
    }
}

/// 12-bit grayscale lossless with restart and predictor 4. Confirms the
/// wider-precision path threads the restart bookkeeping correctly.
#[test]
fn lossless_grayscale_12bit_with_restart_roundtrips() {
    let w = 16u32;
    let h = 12u32;
    let precision = 12u8;
    let bytes = mk_samples_wide(w as usize, h as usize, precision);
    let ri: u16 = 23; // 192-sample image → ≥8 RSTn boundaries.
    let jpeg = encode_lossless_jpeg_grayscale_with_opts(
        w,
        h,
        &bytes,
        (w as usize) * 2,
        precision,
        4,
        ri,
        0,
    )
    .expect("encode 12-bit with restart");
    let v = decode_to_frame(jpeg, w, h);
    let stride = v.planes[0].stride;
    for j in 0..h as usize {
        for i in 0..w as usize {
            let lo_in = bytes[(j * w as usize + i) * 2] as u16;
            let hi_in = bytes[(j * w as usize + i) * 2 + 1] as u16;
            let want = lo_in | (hi_in << 8);
            let lo = v.planes[0].data[j * stride + i * 2] as u16;
            let hi = v.planes[0].data[j * stride + i * 2 + 1] as u16;
            let got = lo | (hi << 8);
            assert_eq!(got, want, "12-bit restart mismatch at ({i},{j})");
        }
    }
}

/// Non-zero `Pt` (point transform) shifts the wire samples right by Pt
/// at encode time; the decoder shifts left by Pt on the way out. The
/// round-trip therefore preserves the high `precision − Pt` bits exactly
/// and zeroes the low `Pt` bits.
#[test]
fn lossless_grayscale_with_point_transform_drops_low_bits() {
    let w = 16u32;
    let h = 8u32;
    let samples = mk_samples_8bit(w as usize, h as usize);
    let pt: u8 = 2;
    let jpeg = encode_lossless_jpeg_grayscale_with_opts(w, h, &samples, w as usize, 8, 1, 0, pt)
        .expect("encode with Pt");
    let v = decode_to_frame(jpeg, w, h);
    let stride = v.planes[0].stride;
    for j in 0..h as usize {
        for i in 0..w as usize {
            let got = v.planes[0].data[j * stride + i];
            // Pt rounds toward zero: encoder right-shifts (samples[i] >> Pt),
            // decoder left-shifts back (got = (samples[i] >> Pt) << Pt).
            let want = (samples[j * w as usize + i] >> pt) << pt;
            assert_eq!(got, want, "Pt={pt} mismatch at ({i},{j})");
        }
    }
}

/// Combining Pt + restart on the grayscale path: round-trip must still
/// drop exactly the low Pt bits and reset cleanly at every RSTn.
#[test]
fn lossless_grayscale_with_restart_and_pt_roundtrips() {
    let w = 20u32;
    let h = 12u32;
    let samples = mk_samples_8bit(w as usize, h as usize);
    let pt: u8 = 1;
    let ri: u16 = 13;
    let jpeg = encode_lossless_jpeg_grayscale_with_opts(w, h, &samples, w as usize, 8, 5, ri, pt)
        .expect("encode with restart+Pt");
    let v = decode_to_frame(jpeg, w, h);
    let stride = v.planes[0].stride;
    for j in 0..h as usize {
        for i in 0..w as usize {
            let got = v.planes[0].data[j * stride + i];
            let want = (samples[j * w as usize + i] >> pt) << pt;
            assert_eq!(got, want, "Pt+restart mismatch at ({i},{j})");
        }
    }
}

/// 3-component RGB lossless with restart markers — every predictor must
/// still byte-exact-roundtrip through the SOF3 decoder. The decoder side
/// already exercises per-component predictor reset at RSTn (`reset_pred`
/// applies uniformly across all components in `decode_lossless_scan`).
#[test]
fn lossless_rgb_with_restart_roundtrips_every_predictor() {
    let w = 16u32;
    let h = 12u32;
    let planes = mk_rgb_8bit(w as usize, h as usize);
    let strides = [w as usize, w as usize, w as usize];
    let ri: u16 = 11; // 192-pixel image → ≥17 RSTn boundaries.
    for predictor in 1u8..=7 {
        let jpeg = encode_lossless_jpeg_rgb_with_opts(
            w,
            h,
            [&planes[0], &planes[1], &planes[2]],
            strides,
            8,
            predictor,
            ri,
            0,
        )
        .unwrap_or_else(|e| panic!("encode predictor={predictor}: {e:?}"));
        let v = decode_to_frame(jpeg, w, h);
        let stride = v.planes[0].stride;
        for j in 0..h as usize {
            for i in 0..w as usize {
                let got_r = v.planes[0].data[j * stride + i * 3];
                let got_g = v.planes[0].data[j * stride + i * 3 + 1];
                let got_b = v.planes[0].data[j * stride + i * 3 + 2];
                assert_eq!(
                    got_r,
                    planes[0][j * w as usize + i],
                    "pred={predictor} R mismatch at ({i},{j})"
                );
                assert_eq!(
                    got_g,
                    planes[1][j * w as usize + i],
                    "pred={predictor} G mismatch at ({i},{j})"
                );
                assert_eq!(
                    got_b,
                    planes[2][j * w as usize + i],
                    "pred={predictor} B mismatch at ({i},{j})"
                );
            }
        }
    }
}

/// 3-component RGB lossless with non-zero Pt: every plane must decode
/// back to `(sample >> Pt) << Pt`.
#[test]
fn lossless_rgb_with_point_transform_drops_low_bits() {
    let w = 16u32;
    let h = 12u32;
    let planes = mk_rgb_8bit(w as usize, h as usize);
    let strides = [w as usize, w as usize, w as usize];
    let pt: u8 = 3;
    let jpeg = encode_lossless_jpeg_rgb_with_opts(
        w,
        h,
        [&planes[0], &planes[1], &planes[2]],
        strides,
        8,
        1,
        0,
        pt,
    )
    .expect("encode RGB with Pt");
    let v = decode_to_frame(jpeg, w, h);
    let stride = v.planes[0].stride;
    for j in 0..h as usize {
        for i in 0..w as usize {
            for (ch, plane) in planes.iter().enumerate() {
                let got = v.planes[0].data[j * stride + i * 3 + ch];
                let want = (plane[j * w as usize + i] >> pt) << pt;
                assert_eq!(got, want, "ch={ch} Pt={pt} mismatch at ({i},{j})");
            }
        }
    }
}

/// 3-component RGB lossless with restart + Pt combined.
#[test]
fn lossless_rgb_with_restart_and_pt_roundtrips() {
    let w = 24u32;
    let h = 16u32;
    let planes = mk_rgb_8bit(w as usize, h as usize);
    let strides = [w as usize, w as usize, w as usize];
    let pt: u8 = 2;
    let ri: u16 = 19;
    let jpeg = encode_lossless_jpeg_rgb_with_opts(
        w,
        h,
        [&planes[0], &planes[1], &planes[2]],
        strides,
        8,
        7,
        ri,
        pt,
    )
    .expect("encode RGB restart+Pt");
    let v = decode_to_frame(jpeg, w, h);
    let stride = v.planes[0].stride;
    for j in 0..h as usize {
        for i in 0..w as usize {
            for (ch, plane) in planes.iter().enumerate() {
                let got = v.planes[0].data[j * stride + i * 3 + ch];
                let want = (plane[j * w as usize + i] >> pt) << pt;
                assert_eq!(got, want, "ch={ch} Pt+restart mismatch at ({i},{j})");
            }
        }
    }
}

/// Reject Pt ≥ precision on both encoders (would shift away every bit).
#[test]
fn lossless_encoders_reject_pt_ge_precision() {
    let w = 4u32;
    let h = 4u32;
    let samples = vec![0u8; (w * h) as usize];
    let err = encode_lossless_jpeg_grayscale_with_opts(
        w, h, &samples, w as usize, 8, 1, 0, 8, // Pt == precision
    )
    .unwrap_err();
    assert!(format!("{err:?}")
        .to_lowercase()
        .contains("point_transform"));

    let plane = vec![0u8; (w * h) as usize];
    let err = encode_lossless_jpeg_rgb_with_opts(
        w,
        h,
        [&plane, &plane, &plane],
        [w as usize, w as usize, w as usize],
        8,
        1,
        0,
        9, // Pt > precision
    )
    .unwrap_err();
    assert!(format!("{err:?}")
        .to_lowercase()
        .contains("point_transform"));
}

/// Build four decorrelated 8-bit planes (C, M, Y, K) so a cross-component
/// indexing bug would surface as a visible mismatch rather than coincide
/// with a flat-fill roundtrip.
fn mk_cmyk_8bit(w: usize, h: usize) -> [Vec<u8>; 4] {
    let mut c = vec![0u8; w * h];
    let mut m = vec![0u8; w * h];
    let mut y = vec![0u8; w * h];
    let mut k = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            c[j * w + i] = ((i as i32 * 5 + j as i32 * 3 + ((i ^ j) as i32 & 31)) & 0xFF) as u8;
            m[j * w + i] =
                ((i as i32 * 11 + j as i32 * 7 + ((i.wrapping_mul(j)) as i32 & 63)) & 0xFF) as u8;
            y[j * w + i] = ((255 - (i as i32 * 2 + j as i32 * 9)) & 0xFF) as u8;
            k[j * w + i] =
                ((i as i32 + j as i32 * 13 + (((i + j) as i32 * 17) & 127)) & 0xFF) as u8;
        }
    }
    [c, m, y, k]
}

/// Plain ("regular") CMYK: every Annex H Table H.1 predictor 1..=7 must
/// round-trip bit-exact when no APP14 colour-transform is requested.
#[test]
fn lossless_cmyk_no_app14_every_predictor_is_bit_exact() {
    let w = 24u32;
    let h = 16u32;
    let planes = mk_cmyk_8bit(w as usize, h as usize);
    let strides = [w as usize, w as usize, w as usize, w as usize];
    for predictor in 1u8..=7 {
        let jpeg = encode_lossless_jpeg_cmyk(
            w,
            h,
            [&planes[0], &planes[1], &planes[2], &planes[3]],
            strides,
            predictor,
            None,
        )
        .unwrap_or_else(|e| panic!("encode predictor={predictor}: {e:?}"));
        // SOF3 marker must be present.
        assert!(
            jpeg.windows(2).any(|x| x == [0xFF, 0xC3]),
            "predictor={predictor}: SOF3 marker missing"
        );
        let v = decode_to_frame(jpeg, w, h);
        assert_eq!(v.planes.len(), 1, "predictor={predictor}");
        let stride = v.planes[0].stride;
        assert_eq!(
            stride,
            (w as usize) * 4,
            "predictor={predictor}: expected packed Cmyk stride"
        );
        for j in 0..h as usize {
            for i in 0..w as usize {
                for ci in 0..4 {
                    let got = v.planes[0].data[j * stride + i * 4 + ci];
                    let want = planes[ci][j * w as usize + i];
                    assert_eq!(
                        got, want,
                        "predictor={predictor} plane {ci} mismatch at ({i},{j})"
                    );
                }
            }
        }
    }
}

/// Adobe CMYK (APP14 transform = 0): the encoder pre-inverts on the wire
/// and the decoder un-inverts on output, so the round-trip must still
/// recover the caller's original samples.
#[test]
fn lossless_cmyk_adobe_transform_0_roundtrip_is_bit_exact() {
    let w = 20u32;
    let h = 14u32;
    let planes = mk_cmyk_8bit(w as usize, h as usize);
    let strides = [w as usize, w as usize, w as usize, w as usize];
    for predictor in [1u8, 4, 7] {
        let jpeg = encode_lossless_jpeg_cmyk(
            w,
            h,
            [&planes[0], &planes[1], &planes[2], &planes[3]],
            strides,
            predictor,
            Some(0),
        )
        .unwrap_or_else(|e| panic!("encode predictor={predictor}: {e:?}"));
        let v = decode_to_frame(jpeg, w, h);
        let stride = v.planes[0].stride;
        for j in 0..h as usize {
            for i in 0..w as usize {
                for ci in 0..4 {
                    let got = v.planes[0].data[j * stride + i * 4 + ci];
                    let want = planes[ci][j * w as usize + i];
                    assert_eq!(
                        got, want,
                        "predictor={predictor} adobe=0 plane {ci} mismatch at ({i},{j})"
                    );
                }
            }
        }
    }
}

/// Restart-marker emission: the encoder writes DRI + RSTn at the
/// requested MCU cadence and the decoder reconstructs the same samples.
#[test]
fn lossless_cmyk_restart_interval_roundtrip_is_bit_exact() {
    let w = 16u32;
    let h = 16u32;
    let planes = mk_cmyk_8bit(w as usize, h as usize);
    let strides = [w as usize, w as usize, w as usize, w as usize];
    let jpeg = encode_lossless_jpeg_cmyk_with_opts(
        w,
        h,
        [&planes[0], &planes[1], &planes[2], &planes[3]],
        strides,
        1,
        None,
        13, // restart every 13 pixels — does not divide width × height evenly
        0,
    )
    .expect("encode with restart interval");
    // DRI segment marker FF DD must appear in the byte stream.
    assert!(
        jpeg.windows(2).any(|x| x == [0xFF, 0xDD]),
        "DRI marker missing when restart_interval > 0"
    );
    let v = decode_to_frame(jpeg, w, h);
    let stride = v.planes[0].stride;
    for j in 0..h as usize {
        for i in 0..w as usize {
            for ci in 0..4 {
                let got = v.planes[0].data[j * stride + i * 4 + ci];
                let want = planes[ci][j * w as usize + i];
                assert_eq!(got, want, "restart roundtrip plane {ci} at ({i},{j})");
            }
        }
    }
}

/// Non-zero point transform shifts every sample by Pt on encode; the
/// decoder shifts back on output. With Pt = 2 only the top 6 bits survive,
/// so the round-trip must equal each input byte with its low 2 bits
/// cleared.
#[test]
fn lossless_cmyk_point_transform_roundtrips_with_quantization() {
    let w = 12u32;
    let h = 8u32;
    let planes = mk_cmyk_8bit(w as usize, h as usize);
    let strides = [w as usize, w as usize, w as usize, w as usize];
    let pt: u8 = 2;
    let jpeg = encode_lossless_jpeg_cmyk_with_opts(
        w,
        h,
        [&planes[0], &planes[1], &planes[2], &planes[3]],
        strides,
        1,
        None,
        0,
        pt,
    )
    .expect("encode with Pt");
    let v = decode_to_frame(jpeg, w, h);
    let stride = v.planes[0].stride;
    for j in 0..h as usize {
        for i in 0..w as usize {
            for ci in 0..4 {
                let got = v.planes[0].data[j * stride + i * 4 + ci];
                let want = (planes[ci][j * w as usize + i] >> pt) << pt;
                assert_eq!(got, want, "Pt={pt} plane {ci} at ({i},{j})");
            }
        }
    }
}

#[test]
fn lossless_cmyk_rejects_bad_predictor() {
    let w = 4u32;
    let h = 4u32;
    let planes = mk_cmyk_8bit(w as usize, h as usize);
    let strides = [w as usize, w as usize, w as usize, w as usize];
    for bad in [0u8, 8, 9, 255] {
        let err = encode_lossless_jpeg_cmyk(
            w,
            h,
            [&planes[0], &planes[1], &planes[2], &planes[3]],
            strides,
            bad,
            None,
        );
        assert!(err.is_err(), "expected reject for predictor={bad}");
    }
}

#[test]
fn lossless_cmyk_rejects_invalid_adobe_transform() {
    let w = 4u32;
    let h = 4u32;
    let planes = mk_cmyk_8bit(w as usize, h as usize);
    let strides = [w as usize, w as usize, w as usize, w as usize];
    for bad in [1u8, 3, 4, 255] {
        let err = encode_lossless_jpeg_cmyk(
            w,
            h,
            [&planes[0], &planes[1], &planes[2], &planes[3]],
            strides,
            1,
            Some(bad),
        );
        assert!(err.is_err(), "expected reject for adobe_transform={bad}");
    }
}
