#![cfg(feature = "registry")]
//! Bit-exact encode→decode roundtrips for the public lossless JPEG (SOF3)
//! encoder.
//!
//! The encoder lives in `oxideav_mjpeg::encoder::encode_lossless_jpeg_grayscale`
//! and produces a standalone single-component grayscale JPEG with predictor
//! selectable in `1..=7` (T.81 Table H.1) at any precision `P ∈ 2..=16`. The
//! decoder side has supported P=2..=16 grayscale lossless since round 0.1.0,
//! so these tests exercise the two halves end-to-end.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};
use oxideav_mjpeg::encoder::encode_lossless_jpeg_grayscale;
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

/// Grayscale without `set_lossless(true)` must surface a clear error
/// rather than silently fall back to a non-functional code path.
#[test]
fn registry_encoder_gray8_without_lossless_flag_errors() {
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{Encoder, VideoFrame};

    let w = 8u32;
    let h = 8u32;
    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Gray8);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).unwrap();
    // No set_lossless(true) here.
    let frame = Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: w as usize,
            data: vec![0u8; (w * h) as usize],
        }],
    });
    let err = enc.send_frame(&frame).unwrap_err();
    assert!(
        format!("{err:?}").to_lowercase().contains("lossless"),
        "expected lossless-hint error, got {err:?}"
    );
}
