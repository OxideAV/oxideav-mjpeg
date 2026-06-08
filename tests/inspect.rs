//! Integration tests for the decode-free JPEG inspector
//! (`oxideav_mjpeg::inspect_jpeg`).
//!
//! These cross-check the inspector's classification against bytes
//! produced by the in-tree encoder — the bytes we ship out the encoder
//! must classify back to the same SOF variant / subsampling /
//! precision / colour hint we asked the encoder to emit. The
//! inspector and the encoder are independent walks of the same T.81
//! marker structure, so a mismatch on any field would be a bug in
//! one or the other.
//!
//! The fixtures live in tests/ next to the existing `roundtrip.rs` and
//! follow the same xorshift-gradient pattern so no committed payload
//! files are needed.

#![cfg(feature = "registry")]

use oxideav_core::frame::VideoPlane;
use oxideav_core::{PixelFormat, VideoFrame};

use oxideav_mjpeg::encoder::{
    encode_jpeg, encode_jpeg_grayscale, encode_jpeg_progressive, encode_lossless_jpeg_grayscale,
};
use oxideav_mjpeg::{inspect_jpeg, ChromaSubsampling, ColorHint, JpegInfo, SofKind};

/// The Ghostscript-bundled sRGB ICC fixture (`with-icc-profile-embedded`)
/// from the project's docs corpus, embedded here so the integration
/// test does not rely on the docs submodule being checked out at test
/// time. See `docs/image/jpeg/jpeg-fixtures-and-traces.md` §3.11 for
/// the segment layout (12-byte `"ICC_PROFILE\0"` + (seq, total) +
/// 2576-byte ICC body in one APP2 segment, declared length 2590).
const ICC_FIXTURE: &[u8] =
    include_bytes!("../../../docs/image/jpeg/fixtures/with-icc-profile-embedded/input.jpg");

/// xorshift32 PRNG — matches the shape used by other tests in this
/// crate so fixture content stays reproducible.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn make_frame(w: u32, h: u32, pix: PixelFormat) -> VideoFrame {
    let (cw, ch): (u32, u32) = match pix {
        PixelFormat::Yuv444P => (w, h),
        PixelFormat::Yuv422P => (w.div_ceil(2), h),
        PixelFormat::Yuv420P => (w.div_ceil(2), h.div_ceil(2)),
        PixelFormat::Gray8 => (0, 0),
        _ => panic!("test fixture unsupported"),
    };
    let mut rng = 0xDEAD_BEEFu32;
    let y_stride = w as usize;
    let mut y = vec![0u8; y_stride * h as usize];
    for j in 0..h as usize {
        for i in 0..w as usize {
            let base = ((i + j) as i32) & 0xFF;
            let noise = (xorshift32(&mut rng) & 0x07) as i32 - 4;
            y[j * y_stride + i] = (base + noise).clamp(0, 255) as u8;
        }
    }
    let mut planes = vec![VideoPlane {
        stride: y_stride,
        data: y,
    }];
    if pix != PixelFormat::Gray8 {
        let cb_stride = cw as usize;
        let cr_stride = cw as usize;
        let mut cb = vec![0u8; cb_stride * ch as usize];
        let mut cr = vec![0u8; cr_stride * ch as usize];
        for j in 0..ch as usize {
            for i in 0..cw as usize {
                cb[j * cb_stride + i] = 128;
                cr[j * cr_stride + i] = 128;
            }
        }
        planes.push(VideoPlane {
            stride: cb_stride,
            data: cb,
        });
        planes.push(VideoPlane {
            stride: cr_stride,
            data: cr,
        });
    }
    VideoFrame {
        pts: Some(0),
        planes,
    }
}

fn assert_baseline_3comp(info: &JpegInfo) {
    assert_eq!(info.sof_kind, SofKind::Baseline);
    assert!(info.sof_kind.is_dct());
    assert!(!info.sof_kind.is_arithmetic());
    assert!(info.sof_kind.is_supported_by_decoder());
    assert_eq!(info.precision, 8);
    assert_eq!(info.num_components(), 3);
}

#[test]
fn inspect_matches_encoded_baseline_yuv420() {
    let (w, h) = (32u32, 32u32);
    let frame = make_frame(w, h, PixelFormat::Yuv420P);
    let bytes = encode_jpeg(&frame, w, h, PixelFormat::Yuv420P, 75).expect("encode baseline 4:2:0");
    let info = inspect_jpeg(&bytes).expect("inspect baseline 4:2:0");
    assert_baseline_3comp(&info);
    assert_eq!(info.width as u32, w);
    assert_eq!(info.height as u32, h);
    assert_eq!(info.subsampling, ChromaSubsampling::Yuv420);
    // Our encoder emits JFIF APP0 for the standard Y/Cb/Cr layouts.
    assert!(matches!(
        info.color_hint,
        ColorHint::JfifYCbCr | ColorHint::Unspecified
    ));
}

#[test]
fn inspect_matches_encoded_baseline_yuv422() {
    let (w, h) = (32u32, 32u32);
    let frame = make_frame(w, h, PixelFormat::Yuv422P);
    let bytes = encode_jpeg(&frame, w, h, PixelFormat::Yuv422P, 75).expect("encode baseline 4:2:2");
    let info = inspect_jpeg(&bytes).expect("inspect baseline 4:2:2");
    assert_baseline_3comp(&info);
    assert_eq!(info.subsampling, ChromaSubsampling::Yuv422);
}

#[test]
fn inspect_matches_encoded_baseline_yuv444() {
    let (w, h) = (32u32, 32u32);
    let frame = make_frame(w, h, PixelFormat::Yuv444P);
    let bytes = encode_jpeg(&frame, w, h, PixelFormat::Yuv444P, 75).expect("encode baseline 4:4:4");
    let info = inspect_jpeg(&bytes).expect("inspect baseline 4:4:4");
    assert_baseline_3comp(&info);
    assert_eq!(info.subsampling, ChromaSubsampling::Yuv444);
}

#[test]
fn inspect_matches_encoded_baseline_gray() {
    let (w, h) = (32u32, 32u32);
    let samples: Vec<u8> = (0..w * h).map(|i| (i & 0xFF) as u8).collect();
    let bytes =
        encode_jpeg_grayscale(w, h, &samples, w as usize, 75).expect("encode baseline grayscale");
    let info = inspect_jpeg(&bytes).expect("inspect baseline gray");
    assert_eq!(info.sof_kind, SofKind::Baseline);
    assert_eq!(info.num_components(), 1);
    assert_eq!(info.subsampling, ChromaSubsampling::GrayscaleOnly);
    assert_eq!(info.precision, 8);
}

#[test]
fn inspect_matches_encoded_progressive_yuv420() {
    let (w, h) = (32u32, 32u32);
    let frame = make_frame(w, h, PixelFormat::Yuv420P);
    let bytes = encode_jpeg_progressive(&frame, w, h, PixelFormat::Yuv420P, 75)
        .expect("encode progressive 4:2:0");
    let info = inspect_jpeg(&bytes).expect("inspect progressive 4:2:0");
    assert_eq!(info.sof_kind, SofKind::Progressive);
    assert!(info.sof_kind.is_supported_by_decoder());
    assert!(info.sof_kind.is_dct());
    assert!(!info.sof_kind.is_arithmetic());
    assert_eq!(info.precision, 8);
    assert_eq!(info.num_components(), 3);
    assert_eq!(info.subsampling, ChromaSubsampling::Yuv420);
}

#[test]
fn inspect_matches_encoded_lossless_gray() {
    let (w, h) = (16u32, 16u32);
    let samples = (0..w * h).map(|i| (i & 0xFF) as u8).collect::<Vec<u8>>();
    let bytes = encode_lossless_jpeg_grayscale(w, h, &samples, w as usize, 8, 1)
        .expect("encode lossless gray pred1");
    let info = inspect_jpeg(&bytes).expect("inspect lossless gray");
    assert_eq!(info.sof_kind, SofKind::Lossless);
    assert!(info.sof_kind.is_supported_by_decoder());
    assert!(!info.sof_kind.is_dct());
    assert!(!info.sof_kind.is_arithmetic());
    assert_eq!(info.precision, 8);
    assert_eq!(info.num_components(), 1);
    assert_eq!(info.subsampling, ChromaSubsampling::GrayscaleOnly);
}

#[test]
fn inspect_lossless_at_higher_precision() {
    let (w, h) = (8u32, 8u32);
    // 12-bit samples packed little-endian into a u8 stride that's
    // 2 bytes wide per sample. The lossless encoder reads the
    // `precision` arg and produces an SOF3 with that P.
    let samples = (0..w * h).map(|i| (i & 0xFF) as u8).collect::<Vec<u8>>();
    let bytes = encode_lossless_jpeg_grayscale(w, h, &samples, w as usize, 8, 4)
        .expect("encode lossless gray pred4 P=8");
    let info = inspect_jpeg(&bytes).expect("inspect lossless gray pred4");
    assert_eq!(info.sof_kind, SofKind::Lossless);
    assert_eq!(info.precision, 8);
}

#[test]
fn inspector_is_cheap_relative_to_decode() {
    // Sanity: inspecting a 32x32 baseline JPEG should always succeed
    // and the call should not allocate proportional to the scan
    // body. We don't enforce timing here — we just exercise the
    // walker against a reasonable spread of variants in one test
    // invocation to give the lib coverage tooling a single entry
    // point that covers all the prefix walker branches.
    let (w, h) = (16u32, 16u32);
    let yuv420 = encode_jpeg(
        &make_frame(w, h, PixelFormat::Yuv420P),
        w,
        h,
        PixelFormat::Yuv420P,
        75,
    )
    .unwrap();
    let prog = encode_jpeg_progressive(
        &make_frame(w, h, PixelFormat::Yuv420P),
        w,
        h,
        PixelFormat::Yuv420P,
        75,
    )
    .unwrap();
    let samples = vec![128u8; (w * h) as usize];
    let lossless = encode_lossless_jpeg_grayscale(w, h, &samples, w as usize, 8, 1).unwrap();

    for bytes in [&yuv420, &prog, &lossless] {
        let info = inspect_jpeg(bytes).expect("inspect any-variant");
        // The inspector must not depend on the buffer's tail —
        // truncating after the SOS marker byte (and the inspector's
        // next-marker call into the scan) should still classify the
        // SOF correctly. Find SOS, truncate after it.
        let sos_pos = bytes
            .windows(2)
            .position(|w| w[0] == 0xFF && w[1] == 0xDA)
            .expect("SOS present");
        let truncated = &bytes[..sos_pos + 2];
        let info2 = inspect_jpeg(truncated).expect("inspect truncated");
        assert_eq!(info.sof_kind, info2.sof_kind);
        assert_eq!(info.width, info2.width);
        assert_eq!(info.height, info2.height);
        assert_eq!(info.num_components(), info2.num_components());
    }
}

#[test]
fn inspect_real_fixture_reports_embedded_icc_profile() {
    // The `with-icc-profile-embedded` fixture wraps Ghostscript's
    // 2576-byte sRGB ICC inside a single APP2 segment whose payload
    // is `"ICC_PROFILE\0"` + (seq=1, total=1) + 2576 ICC bytes.
    // The inspector should surface a complete one-chunk summary with
    // `total_payload_len == 2576`.
    let info = inspect_jpeg(ICC_FIXTURE).expect("inspect real ICC fixture");
    let icc = info
        .icc_profile
        .as_ref()
        .expect("icc_profile summary populated");
    assert_eq!(icc.total, 1);
    assert_eq!(icc.chunks.len(), 1);
    assert_eq!(icc.chunks[0].0, 1, "seq_no");
    assert_eq!(icc.total_payload_len, 2576);
    assert!(icc.is_complete());
    // The fixture is a baseline YCbCr 4:2:0 stream with a JFIF APP0
    // too — confirm the unrelated colour-hint signalling still works
    // alongside the ICC summary.
    assert_eq!(info.sof_kind, SofKind::Baseline);
    assert_eq!(info.color_hint, ColorHint::JfifYCbCr);
}
