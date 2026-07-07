#![cfg(feature = "registry")]
// Parallel-array index loops are idiomatic in these per-pixel reconstruction
// checks; skip the lint (same allow as the other hierarchical suites).
#![allow(clippy::needless_range_loop)]
//! Hierarchical mode (T.81 Annex J) — **encoder** round-trips.
//!
//! Every test drives a public `encoder::encode_hierarchical_*` entry point
//! and decodes the result through the public `Decoder` trait, asserting the
//! reconstruction is bit-exact (the spatial-lossless progression of
//! §K.7.2.2 has a truly lossless final stage, so exactness is the
//! conformance criterion — no PSNR tolerance).

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_mjpeg::encoder::{
    encode_hierarchical_lossless_jpeg_cmyk, encode_hierarchical_lossless_jpeg_grayscale,
    encode_hierarchical_lossless_jpeg_rgb,
};
use oxideav_mjpeg::registry::make_decoder;

fn decode(jpeg: &[u8], w: u32, h: u32) -> oxideav_core::VideoFrame {
    let mut params = CodecParameters::video(CodecId::new("mjpeg"));
    params.width = Some(w);
    params.height = Some(h);
    let mut dec = make_decoder(&params).expect("make_decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), jpeg.to_vec()))
        .expect("send_packet");
    let Frame::Video(v) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected VideoFrame")
    };
    v
}

/// Deterministic pseudo-random test plane: xorshift noise over a gradient
/// so both the low-resolution stage and the differential residuals carry
/// non-trivial structure.
fn mk_plane(w: usize, h: usize, precision: u8, seed: u32) -> Vec<u32> {
    let max = ((1u64 << precision) - 1) as u32;
    let mut s = seed | 1;
    let mut out = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let grad = ((x * 7 + y * 3) as u32) % (max + 1);
            out[y * w + x] = (grad + (s % 16)).min(max);
        }
    }
    out
}

fn plane_to_bytes(plane: &[u32], precision: u8) -> Vec<u8> {
    if precision <= 8 {
        plane.iter().map(|&v| v as u8).collect()
    } else {
        let mut out = Vec::with_capacity(plane.len() * 2);
        for &v in plane {
            out.extend_from_slice(&(v as u16).to_le_bytes());
        }
        out
    }
}

/// Read back a decoded grayscale frame as u32 samples for any precision.
fn gray_samples(frame: &oxideav_core::VideoFrame, w: usize, h: usize, precision: u8) -> Vec<u32> {
    let plane = &frame.planes[0];
    let mut out = vec![0u32; w * h];
    if precision <= 8 {
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = plane.data[y * plane.stride + x] as u32;
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                let o = y * plane.stride + x * 2;
                out[y * w + x] = plane.data[o] as u32 | ((plane.data[o + 1] as u32) << 8);
            }
        }
    }
    out
}

// ---- Grayscale -------------------------------------------------------------

#[test]
fn hier_gray_p8_two_stage_bit_exact() {
    let (w, h) = (32usize, 24usize);
    for predictor in [1u8, 4, 7] {
        let img = mk_plane(w, h, 8, 0xC0FFEE);
        let bytes = plane_to_bytes(&img, 8);
        let jpeg = encode_hierarchical_lossless_jpeg_grayscale(
            w as u32, h as u32, &bytes, w, 8, predictor, 2,
        )
        .expect("encode");
        let frame = decode(&jpeg, w as u32, h as u32);
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.planes[0].stride, w, "Gray8 stride");
        let got = gray_samples(&frame, w, h, 8);
        assert_eq!(got, img, "predictor {predictor}");
    }
}

#[test]
fn hier_gray_p8_three_stage_bit_exact() {
    let (w, h) = (64usize, 32usize);
    let img = mk_plane(w, h, 8, 0xBEEF01);
    let bytes = plane_to_bytes(&img, 8);
    let jpeg = encode_hierarchical_lossless_jpeg_grayscale(w as u32, h as u32, &bytes, w, 8, 1, 3)
        .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(gray_samples(&frame, w, h, 8), img);
}

#[test]
fn hier_gray_single_stage_is_plain_dhp_lossless() {
    // levels = 1: DHP envelope around one non-differential SOF3 frame,
    // including odd dimensions (no divisibility constraint applies).
    let (w, h) = (13usize, 7usize);
    let img = mk_plane(w, h, 8, 0x1234);
    let bytes = plane_to_bytes(&img, 8);
    let jpeg = encode_hierarchical_lossless_jpeg_grayscale(w as u32, h as u32, &bytes, w, 8, 5, 1)
        .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(gray_samples(&frame, w, h, 8), img);
}

#[test]
fn hier_gray_p12_two_stage_bit_exact() {
    let (w, h) = (16usize, 16usize);
    let img = mk_plane(w, h, 12, 0xABCDEF);
    let bytes = plane_to_bytes(&img, 12);
    let jpeg =
        encode_hierarchical_lossless_jpeg_grayscale(w as u32, h as u32, &bytes, w * 2, 12, 4, 2)
            .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes[0].stride, w * 2, "Gray12Le stride");
    assert_eq!(gray_samples(&frame, w, h, 12), img);
}

#[test]
fn hier_gray_p16_full_range_two_stage_bit_exact() {
    // Full-range 16-bit content: the modulo-2^16 differential residuals
    // span the widest SSSS categories (including negatives wrapping to the
    // top of the modulus range).
    let (w, h) = (16usize, 12usize);
    let mut img = mk_plane(w, h, 16, 0x777AA);
    // Force extreme jumps so the residuals exercise large categories.
    img[0] = 0;
    img[1] = 65_535;
    img[w] = 65_535;
    img[w + 1] = 0;
    let bytes = plane_to_bytes(&img, 16);
    let jpeg =
        encode_hierarchical_lossless_jpeg_grayscale(w as u32, h as u32, &bytes, w * 2, 16, 1, 2)
            .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes[0].stride, w * 2, "Gray16Le stride");
    assert_eq!(gray_samples(&frame, w, h, 16), img);
}

#[test]
fn hier_gray_p2_two_stage_bit_exact() {
    let (w, h) = (8usize, 8usize);
    let img = mk_plane(w, h, 2, 0x99);
    let bytes = plane_to_bytes(&img, 2);
    let jpeg = encode_hierarchical_lossless_jpeg_grayscale(w as u32, h as u32, &bytes, w, 2, 1, 2)
        .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    // P=2 lands in the Gray16Le widen policy of the lossless shaper.
    let got = gray_samples(&frame, w, h, 16);
    assert_eq!(got, img);
}

// ---- Error paths -----------------------------------------------------------

#[test]
fn hier_gray_rejects_zero_levels_and_bad_geometry() {
    let (w, h) = (16usize, 16usize);
    let img = mk_plane(w, h, 8, 1);
    let bytes = plane_to_bytes(&img, 8);
    assert!(
        encode_hierarchical_lossless_jpeg_grayscale(w as u32, h as u32, &bytes, w, 8, 1, 0)
            .is_err(),
        "levels = 0 must be rejected"
    );
    // 10 % 4 != 0 → three levels are impossible.
    let img2 = mk_plane(10, 10, 8, 2);
    let bytes2 = plane_to_bytes(&img2, 8);
    assert!(
        encode_hierarchical_lossless_jpeg_grayscale(10, 10, &bytes2, 10, 8, 1, 3).is_err(),
        "non-divisible dimensions must be rejected"
    );
    // Predictor 0 is reserved for differential frames.
    assert!(
        encode_hierarchical_lossless_jpeg_grayscale(w as u32, h as u32, &bytes, w, 8, 0, 2)
            .is_err(),
        "predictor 0 must be rejected"
    );
}

// ---- Three-component (RGB-class) -------------------------------------------

#[test]
fn hier_rgb_p8_two_stage_bit_exact() {
    let (w, h) = (24usize, 16usize);
    let r = mk_plane(w, h, 8, 11);
    let g = mk_plane(w, h, 8, 22);
    let b = mk_plane(w, h, 8, 33);
    let rb = plane_to_bytes(&r, 8);
    let gb = plane_to_bytes(&g, 8);
    let bb = plane_to_bytes(&b, 8);
    let jpeg = encode_hierarchical_lossless_jpeg_rgb(
        w as u32,
        h as u32,
        [&rb, &gb, &bb],
        [w, w, w],
        8,
        4,
        2,
    )
    .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes.len(), 1);
    assert_eq!(frame.planes[0].stride, w * 3, "packed Rgb24 stride");
    let plane = &frame.planes[0];
    for y in 0..h {
        for x in 0..w {
            let o = y * plane.stride + x * 3;
            assert_eq!(plane.data[o] as u32, r[y * w + x], "R ({x},{y})");
            assert_eq!(plane.data[o + 1] as u32, g[y * w + x], "G ({x},{y})");
            assert_eq!(plane.data[o + 2] as u32, b[y * w + x], "B ({x},{y})");
        }
    }
}

#[test]
fn hier_rgb_p12_two_stage_bit_exact() {
    let (w, h) = (16usize, 8usize);
    // Pass planes in G, B, R order so the canonical Gbrp12Le layout holds.
    let g = mk_plane(w, h, 12, 44);
    let b = mk_plane(w, h, 12, 55);
    let r = mk_plane(w, h, 12, 66);
    let gb = plane_to_bytes(&g, 12);
    let bb = plane_to_bytes(&b, 12);
    let rb = plane_to_bytes(&r, 12);
    let jpeg = encode_hierarchical_lossless_jpeg_rgb(
        w as u32,
        h as u32,
        [&gb, &bb, &rb],
        [w * 2, w * 2, w * 2],
        12,
        1,
        2,
    )
    .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes.len(), 3, "planar Gbrp12Le");
    for (ci, src) in [&g, &b, &r].into_iter().enumerate() {
        let plane = &frame.planes[ci];
        for y in 0..h {
            for x in 0..w {
                let o = y * plane.stride + x * 2;
                let got = plane.data[o] as u32 | ((plane.data[o + 1] as u32) << 8);
                assert_eq!(got, src[y * w + x], "component {ci} ({x},{y})");
            }
        }
    }
}

// ---- Four-component (CMYK-class) --------------------------------------------

#[test]
fn hier_cmyk_two_stage_bit_exact_no_app14_and_adobe() {
    let (w, h) = (16usize, 16usize);
    let c = mk_plane(w, h, 8, 1);
    let m = mk_plane(w, h, 8, 2);
    let y = mk_plane(w, h, 8, 3);
    let k = mk_plane(w, h, 8, 4);
    let cb = plane_to_bytes(&c, 8);
    let mb = plane_to_bytes(&m, 8);
    let yb = plane_to_bytes(&y, 8);
    let kb = plane_to_bytes(&k, 8);
    for transform in [None, Some(0u8)] {
        let jpeg = encode_hierarchical_lossless_jpeg_cmyk(
            w as u32,
            h as u32,
            [&cb, &mb, &yb, &kb],
            [w, w, w, w],
            1,
            transform,
            2,
        )
        .expect("encode");
        let frame = decode(&jpeg, w as u32, h as u32);
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.planes[0].stride, w * 4, "packed Cmyk stride");
        let plane = &frame.planes[0];
        for yy in 0..h {
            for x in 0..w {
                let o = yy * plane.stride + x * 4;
                assert_eq!(plane.data[o] as u32, c[yy * w + x], "C {transform:?}");
                assert_eq!(plane.data[o + 1] as u32, m[yy * w + x], "M {transform:?}");
                assert_eq!(plane.data[o + 2] as u32, y[yy * w + x], "Y {transform:?}");
                assert_eq!(plane.data[o + 3] as u32, k[yy * w + x], "K {transform:?}");
            }
        }
    }
}

#[test]
fn hier_cmyk_ycck_decodes_with_exact_k_plane() {
    // YCCK is a lossy interop convention (BT.601 YCbCr → RGB → CMY clamps
    // on decode) but the K plane round-trips exactly.
    let (w, h) = (16usize, 8usize);
    let y = mk_plane(w, h, 8, 5);
    let cb = mk_plane(w, h, 8, 6);
    let cr = mk_plane(w, h, 8, 7);
    let k = mk_plane(w, h, 8, 8);
    let yb = plane_to_bytes(&y, 8);
    let cbb = plane_to_bytes(&cb, 8);
    let crb = plane_to_bytes(&cr, 8);
    let kb = plane_to_bytes(&k, 8);
    let jpeg = encode_hierarchical_lossless_jpeg_cmyk(
        w as u32,
        h as u32,
        [&yb, &cbb, &crb, &kb],
        [w, w, w, w],
        7,
        Some(2),
        2,
    )
    .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes.len(), 1);
    assert_eq!(frame.planes[0].stride, w * 4, "packed Cmyk stride");
    let plane = &frame.planes[0];
    for yy in 0..h {
        for x in 0..w {
            let o = yy * plane.stride + x * 4;
            assert_eq!(plane.data[o + 3] as u32, k[yy * w + x], "K ({x},{yy})");
        }
    }
}

// ---- Arithmetic (SOF11 + SOF15) progression ---------------------------------

use oxideav_mjpeg::encoder::{
    encode_hierarchical_lossless_arith_jpeg_cmyk,
    encode_hierarchical_lossless_arith_jpeg_grayscale, encode_hierarchical_lossless_arith_jpeg_rgb,
};

#[test]
fn hier_arith_gray_p8_two_stage_bit_exact() {
    let (w, h) = (32usize, 24usize);
    for predictor in [1u8, 4, 7] {
        let img = mk_plane(w, h, 8, 0xA11CE);
        let bytes = plane_to_bytes(&img, 8);
        let jpeg = encode_hierarchical_lossless_arith_jpeg_grayscale(
            w as u32, h as u32, &bytes, w, 8, predictor, 2,
        )
        .expect("encode");
        let frame = decode(&jpeg, w as u32, h as u32);
        assert_eq!(gray_samples(&frame, w, h, 8), img, "predictor {predictor}");
    }
}

#[test]
fn hier_arith_gray_p8_three_stage_bit_exact() {
    let (w, h) = (64usize, 32usize);
    let img = mk_plane(w, h, 8, 0xD0D0);
    let bytes = plane_to_bytes(&img, 8);
    let jpeg =
        encode_hierarchical_lossless_arith_jpeg_grayscale(w as u32, h as u32, &bytes, w, 8, 1, 3)
            .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(gray_samples(&frame, w, h, 8), img);
}

#[test]
fn hier_arith_gray_p16_full_range_two_stage_bit_exact() {
    let (w, h) = (16usize, 12usize);
    let mut img = mk_plane(w, h, 16, 0x5EED5);
    img[0] = 0;
    img[1] = 65_535;
    img[w] = 65_535;
    img[w + 1] = 0;
    let bytes = plane_to_bytes(&img, 16);
    let jpeg = encode_hierarchical_lossless_arith_jpeg_grayscale(
        w as u32,
        h as u32,
        &bytes,
        w * 2,
        16,
        1,
        2,
    )
    .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes[0].stride, w * 2, "Gray16Le stride");
    assert_eq!(gray_samples(&frame, w, h, 16), img);
}

#[test]
fn hier_arith_gray_p12_single_stage_bit_exact() {
    let (w, h) = (11usize, 9usize);
    let img = mk_plane(w, h, 12, 0xF00D);
    let bytes = plane_to_bytes(&img, 12);
    let jpeg = encode_hierarchical_lossless_arith_jpeg_grayscale(
        w as u32,
        h as u32,
        &bytes,
        w * 2,
        12,
        6,
        1,
    )
    .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(gray_samples(&frame, w, h, 12), img);
}

#[test]
fn hier_arith_rgb_p8_two_stage_bit_exact() {
    let (w, h) = (24usize, 16usize);
    let r = mk_plane(w, h, 8, 0x11);
    let g = mk_plane(w, h, 8, 0x22);
    let b = mk_plane(w, h, 8, 0x33);
    let rb = plane_to_bytes(&r, 8);
    let gb = plane_to_bytes(&g, 8);
    let bb = plane_to_bytes(&b, 8);
    let jpeg = encode_hierarchical_lossless_arith_jpeg_rgb(
        w as u32,
        h as u32,
        [&rb, &gb, &bb],
        [w, w, w],
        8,
        4,
        2,
    )
    .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes[0].stride, w * 3, "packed Rgb24 stride");
    let plane = &frame.planes[0];
    for y in 0..h {
        for x in 0..w {
            let o = y * plane.stride + x * 3;
            assert_eq!(plane.data[o] as u32, r[y * w + x], "R ({x},{y})");
            assert_eq!(plane.data[o + 1] as u32, g[y * w + x], "G ({x},{y})");
            assert_eq!(plane.data[o + 2] as u32, b[y * w + x], "B ({x},{y})");
        }
    }
}

#[test]
fn hier_arith_rgb_p14_two_stage_bit_exact() {
    let (w, h) = (8usize, 8usize);
    let g = mk_plane(w, h, 14, 0x44);
    let b = mk_plane(w, h, 14, 0x55);
    let r = mk_plane(w, h, 14, 0x66);
    let gb = plane_to_bytes(&g, 14);
    let bb = plane_to_bytes(&b, 14);
    let rb = plane_to_bytes(&r, 14);
    let jpeg = encode_hierarchical_lossless_arith_jpeg_rgb(
        w as u32,
        h as u32,
        [&gb, &bb, &rb],
        [w * 2, w * 2, w * 2],
        14,
        1,
        2,
    )
    .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes.len(), 3, "planar Gbrp14Le");
    for (ci, src) in [&g, &b, &r].into_iter().enumerate() {
        let plane = &frame.planes[ci];
        for y in 0..h {
            for x in 0..w {
                let o = y * plane.stride + x * 2;
                let got = plane.data[o] as u32 | ((plane.data[o + 1] as u32) << 8);
                assert_eq!(got, src[y * w + x], "component {ci} ({x},{y})");
            }
        }
    }
}

#[test]
fn hier_arith_cmyk_two_stage_bit_exact() {
    let (w, h) = (16usize, 16usize);
    let c = mk_plane(w, h, 8, 0x71);
    let m = mk_plane(w, h, 8, 0x72);
    let y = mk_plane(w, h, 8, 0x73);
    let k = mk_plane(w, h, 8, 0x74);
    let cb = plane_to_bytes(&c, 8);
    let mb = plane_to_bytes(&m, 8);
    let yb = plane_to_bytes(&y, 8);
    let kb = plane_to_bytes(&k, 8);
    for transform in [None, Some(0u8)] {
        let jpeg = encode_hierarchical_lossless_arith_jpeg_cmyk(
            w as u32,
            h as u32,
            [&cb, &mb, &yb, &kb],
            [w, w, w, w],
            1,
            transform,
            2,
        )
        .expect("encode");
        let frame = decode(&jpeg, w as u32, h as u32);
        assert_eq!(frame.planes[0].stride, w * 4, "packed Cmyk stride");
        let plane = &frame.planes[0];
        for yy in 0..h {
            for x in 0..w {
                let o = yy * plane.stride + x * 4;
                assert_eq!(plane.data[o] as u32, c[yy * w + x], "C {transform:?}");
                assert_eq!(plane.data[o + 1] as u32, m[yy * w + x], "M {transform:?}");
                assert_eq!(plane.data[o + 2] as u32, y[yy * w + x], "Y {transform:?}");
                assert_eq!(plane.data[o + 3] as u32, k[yy * w + x], "K {transform:?}");
            }
        }
    }
}

// ---- DCT progression (SOF0 + SOF5, Huffman) ---------------------------------

use oxideav_mjpeg::encoder::{
    encode_hierarchical_dct_jpeg_grayscale, encode_hierarchical_dct_jpeg_yuv444,
};

/// Smooth mid-range plane (30..=220) so DCT-stage PSNR assertions do not
/// flap on clamp-edge wrap effects (a conformant §J.2.1 reconstruction
/// wraps modulo 2^16 rather than clamping between stages).
fn mk_smooth_plane(w: usize, h: usize, seed: u32) -> Vec<u32> {
    let mut s = seed | 1;
    let mut out = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let base = 30.0
                + 90.0 * (1.0 + ((x as f32) * 0.19).sin()) / 2.0
                + 90.0 * (1.0 + ((y as f32) * 0.13).cos()) / 2.0;
            out[y * w + x] = (base as u32 + (s % 8)).clamp(30, 220);
        }
    }
    out
}

fn psnr(orig: &[u32], got: &[u32], peak: f64) -> f64 {
    assert_eq!(orig.len(), got.len());
    let mse: f64 = orig
        .iter()
        .zip(got.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum::<f64>()
        / orig.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    }
}

#[test]
fn hier_dct_gray_two_stage_decodes_with_high_fidelity() {
    let (w, h) = (64usize, 48usize);
    let img = mk_smooth_plane(w, h, 0xDC7);
    let bytes = plane_to_bytes(&img, 8);
    let jpeg = encode_hierarchical_dct_jpeg_grayscale(w as u32, h as u32, &bytes, w, 90, 2)
        .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes.len(), 1);
    assert_eq!(frame.planes[0].stride, w, "Gray8 stride");
    let got = gray_samples(&frame, w, h, 8);
    let db = psnr(&img, &got, 255.0);
    assert!(
        db >= 35.0,
        "two-stage hierarchical DCT PSNR {db:.2} dB < 35"
    );
}

#[test]
fn hier_dct_gray_three_stage_decodes_with_high_fidelity() {
    let (w, h) = (64usize, 64usize);
    let img = mk_smooth_plane(w, h, 0xDC8);
    let bytes = plane_to_bytes(&img, 8);
    let jpeg = encode_hierarchical_dct_jpeg_grayscale(w as u32, h as u32, &bytes, w, 90, 3)
        .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    let got = gray_samples(&frame, w, h, 8);
    let db = psnr(&img, &got, 255.0);
    assert!(
        db >= 35.0,
        "three-stage hierarchical DCT PSNR {db:.2} dB < 35"
    );
}

#[test]
fn hier_dct_gray_single_stage_matches_flat_baseline_quality() {
    // levels = 1 is a DHP envelope around one SOF0 frame; fidelity should
    // be in the same band as the flat baseline encoder at equal quality.
    let (w, h) = (32usize, 32usize);
    let img = mk_smooth_plane(w, h, 0xDC9);
    let bytes = plane_to_bytes(&img, 8);
    let jpeg = encode_hierarchical_dct_jpeg_grayscale(w as u32, h as u32, &bytes, w, 90, 1)
        .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    let got = gray_samples(&frame, w, h, 8);
    let hier_db = psnr(&img, &got, 255.0);

    let flat = oxideav_mjpeg::encoder::encode_jpeg_grayscale(w as u32, h as u32, &bytes, w, 90)
        .expect("flat encode");
    let flat_frame = decode(&flat, w as u32, h as u32);
    let flat_got = gray_samples(&flat_frame, w, h, 8);
    let flat_db = psnr(&img, &flat_got, 255.0);
    assert!(
        (hier_db - flat_db).abs() <= 1.0,
        "single-stage hierarchical ({hier_db:.2} dB) should track the flat baseline ({flat_db:.2} dB)"
    );
}

#[test]
fn hier_dct_refinement_improves_over_truncated_stream() {
    // Decoding only the first (low-resolution) stage upsampled to full
    // resolution must be strictly worse than the full progression — the
    // differential stage really does carry correction energy.
    let (w, h) = (64usize, 48usize);
    let img = mk_smooth_plane(w, h, 0xDCA);
    let bytes = plane_to_bytes(&img, 8);
    let jpeg = encode_hierarchical_dct_jpeg_grayscale(w as u32, h as u32, &bytes, w, 85, 2)
        .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    let full_db = psnr(&img, &gray_samples(&frame, w, h, 8), 255.0);

    // Baseline for comparison: the low-res stage alone, decoded via a flat
    // baseline encode of the 2x-downsampled image, then nearest upsampled.
    let (lw, lh) = (w / 2, h / 2);
    let mut low = vec![0u32; lw * lh];
    for y in 0..lh {
        for x in 0..lw {
            let a = img[(2 * y) * w + 2 * x];
            let b = img[(2 * y) * w + 2 * x + 1];
            let c = img[(2 * y + 1) * w + 2 * x];
            let d = img[(2 * y + 1) * w + 2 * x + 1];
            low[y * lw + x] = (a + b + c + d) / 4;
        }
    }
    let mut up = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            up[y * w + x] = low[(y / 2) * lw + x / 2];
        }
    }
    let low_db = psnr(&img, &up, 255.0);
    assert!(
        full_db > low_db + 3.0,
        "full progression ({full_db:.2} dB) must beat the low-res stage alone ({low_db:.2} dB) by > 3 dB"
    );
}

#[test]
fn hier_dct_yuv444_two_stage_decodes_with_high_fidelity() {
    let (w, h) = (32usize, 32usize);
    let yp = mk_smooth_plane(w, h, 0xE1);
    let cb = mk_smooth_plane(w, h, 0xE2);
    let cr = mk_smooth_plane(w, h, 0xE3);
    let yb = plane_to_bytes(&yp, 8);
    let cbb = plane_to_bytes(&cb, 8);
    let crb = plane_to_bytes(&cr, 8);
    let jpeg = encode_hierarchical_dct_jpeg_yuv444(
        w as u32,
        h as u32,
        [&yb, &cbb, &crb],
        [w, w, w],
        90,
        2,
    )
    .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes.len(), 3, "planar Yuv444P");
    for (ci, src) in [&yp, &cb, &cr].into_iter().enumerate() {
        let plane = &frame.planes[ci];
        assert_eq!(plane.stride, w, "full-resolution plane {ci}");
        let mut got = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                got[y * w + x] = plane.data[y * plane.stride + x] as u32;
            }
        }
        let db = psnr(src, &got, 255.0);
        assert!(db >= 35.0, "YUV444 plane {ci} PSNR {db:.2} dB < 35");
    }
}

// ---- DCT progression terminated by a lossless SOF7 frame (§K.7.2) ----------

use oxideav_mjpeg::encoder::{
    encode_hierarchical_dct_jpeg_grayscale_lossless_final,
    encode_hierarchical_dct_jpeg_yuv444_lossless_final,
};

#[test]
fn hier_dct_lossless_final_gray_two_stage_bit_exact() {
    // Full-range noisy content: exactness must hold regardless of clamp /
    // wrap effects in the lossy stages, because the SOF7 terminator codes
    // the exact residual.
    let (w, h) = (48usize, 32usize);
    let img = mk_plane(w, h, 8, 0x50F7);
    let bytes = plane_to_bytes(&img, 8);
    for quality in [10u8, 75, 95] {
        let jpeg = encode_hierarchical_dct_jpeg_grayscale_lossless_final(
            w as u32, h as u32, &bytes, w, quality, 2,
        )
        .expect("encode");
        let frame = decode(&jpeg, w as u32, h as u32);
        assert_eq!(
            gray_samples(&frame, w, h, 8),
            img,
            "quality {quality} must be bit-exact"
        );
    }
}

#[test]
fn hier_dct_lossless_final_gray_single_stage_bit_exact() {
    // levels = 1: SOF0 first frame + SOF7 terminator, no EXP anywhere.
    let (w, h) = (17usize, 11usize);
    let img = mk_plane(w, h, 8, 0x50F8);
    let bytes = plane_to_bytes(&img, 8);
    let jpeg =
        encode_hierarchical_dct_jpeg_grayscale_lossless_final(w as u32, h as u32, &bytes, w, 75, 1)
            .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(gray_samples(&frame, w, h, 8), img);
}

#[test]
fn hier_dct_lossless_final_yuv444_two_stage_bit_exact() {
    let (w, h) = (32usize, 16usize);
    let yp = mk_plane(w, h, 8, 0xF1);
    let cb = mk_plane(w, h, 8, 0xF2);
    let cr = mk_plane(w, h, 8, 0xF3);
    let yb = plane_to_bytes(&yp, 8);
    let cbb = plane_to_bytes(&cb, 8);
    let crb = plane_to_bytes(&cr, 8);
    let jpeg = encode_hierarchical_dct_jpeg_yuv444_lossless_final(
        w as u32,
        h as u32,
        [&yb, &cbb, &crb],
        [w, w, w],
        85,
        2,
    )
    .expect("encode");
    let frame = decode(&jpeg, w as u32, h as u32);
    assert_eq!(frame.planes.len(), 3, "planar Yuv444P");
    for (ci, src) in [&yp, &cb, &cr].into_iter().enumerate() {
        let plane = &frame.planes[ci];
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    plane.data[y * plane.stride + x] as u32,
                    src[y * w + x],
                    "plane {ci} ({x},{y})"
                );
            }
        }
    }
}
