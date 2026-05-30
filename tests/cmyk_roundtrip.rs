//! Public-API CMYK / YCCK encoder round-trip tests.
//!
//! Exercises the four public CMYK entry points landed alongside this
//! file:
//!
//!   * `encoder::encode_jpeg_cmyk` (baseline SOF0, packed buffer)
//!   * `encoder::encode_jpeg_cmyk_progressive` (SOF2, packed buffer)
//!   * `encoder::encode_jpeg_cmyk_1111` (baseline, per-plane back-end)
//!   * `encoder::encode_jpeg_progressive_cmyk_1111` (SOF2, per-plane
//!     back-end)
//!
//! and the trait-API `MjpegPixelFormat::Cmyk` path through
//! [`MjpegEncoder`].
//!
//! Each test encodes synthetic CMYK gradients, decodes the result via
//! the public [`Decoder`] trait, and asserts per-component PSNR ≥ 30 dB
//! at Q = 90 — the same tolerance the internal `decoder::cmyk_tests`
//! suite uses, so the public-API surface inherits the same correctness
//! floor as the private back-end paths.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase, VideoFrame};

use oxideav_mjpeg::encoder::{
    encode_jpeg_cmyk, encode_jpeg_cmyk_1111, encode_jpeg_cmyk_progressive,
    encode_jpeg_progressive_cmyk_1111,
};
use oxideav_mjpeg::registry::{make_decoder, MjpegEncoder};

/// PSNR across two equal-length byte buffers treated as 8-bit samples.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: f64 = 0.0;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        sse += d * d;
    }
    if sse == 0.0 {
        return 99.0;
    }
    20.0 * (255.0 / (sse / a.len() as f64).sqrt()).log10()
}

/// Build four CMYK gradient planes (planar layout, `stride = width`).
fn make_planar(w: usize, h: usize) -> [Vec<u8>; 4] {
    let mut c = vec![0u8; w * h];
    let mut m = vec![0u8; w * h];
    let mut y = vec![0u8; w * h];
    let mut k = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            c[j * w + i] = ((i * 255 / w.max(1)) as u32).min(255) as u8;
            m[j * w + i] = ((j * 255 / h.max(1)) as u32).min(255) as u8;
            y[j * w + i] = (((i + j) * 255 / (w + h).max(1)) as u32).min(255) as u8;
            k[j * w + i] = ((((i ^ j) * 7) & 0xFF) as u8) / 2;
        }
    }
    [c, m, y, k]
}

/// Interleave four planar buffers into one packed `[C, M, Y, K]` plane.
fn pack(w: usize, h: usize, planes: &[Vec<u8>; 4]) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 4];
    for j in 0..h {
        for i in 0..w {
            let o = j * w * 4 + i * 4;
            out[o] = planes[0][j * w + i];
            out[o + 1] = planes[1][j * w + i];
            out[o + 2] = planes[2][j * w + i];
            out[o + 3] = planes[3][j * w + i];
        }
    }
    out
}

/// Decode a self-contained CMYK / YCCK JPEG via the trait API.
fn decode(width: u32, height: u32, jpeg: Vec<u8>) -> VideoFrame {
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(width);
    dec_params.height = Some(height);
    let mut dec = make_decoder(&dec_params).expect("make_decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), jpeg))
        .expect("send_packet");
    let Frame::Video(v) = dec.receive_frame().expect("receive_frame") else {
        panic!("expected video frame")
    };
    v
}

/// Assert per-component round-trip closeness on a packed Cmyk frame.
fn assert_per_component_psnr(
    v: &VideoFrame,
    w: u32,
    h: u32,
    src_planar: &[Vec<u8>; 4],
    floor: f64,
    label: &str,
) {
    assert_eq!(v.planes.len(), 1, "{label}: expected one packed Cmyk plane");
    assert_eq!(
        v.planes[0].stride,
        (w * 4) as usize,
        "{label}: packed Cmyk row stride must equal 4 × width"
    );
    for (ci, src) in src_planar.iter().enumerate() {
        let mut got = Vec::with_capacity(src.len());
        for j in 0..h as usize {
            for i in 0..w as usize {
                got.push(v.planes[0].data[j * v.planes[0].stride + i * 4 + ci]);
            }
        }
        let p = psnr(src, &got);
        assert!(
            p >= floor,
            "{label}: component {ci} PSNR {p:.2} dB below floor {floor:.2}"
        );
    }
}

// ---- encode_jpeg_cmyk (packed-buffer convenience wrapper) --------------

#[test]
fn packed_plain_baseline_roundtrip() {
    let w = 32u32;
    let h = 16u32;
    let planar = make_planar(w as usize, h as usize);
    let packed = pack(w as usize, h as usize, &planar);
    let jpeg =
        encode_jpeg_cmyk(w, h, &packed, w as usize * 4, 90, None).expect("encode plain CMYK");
    let v = decode(w, h, jpeg);
    assert_per_component_psnr(&v, w, h, &planar, 30.0, "packed_plain");
}

#[test]
fn packed_adobe_inverted_baseline_roundtrip() {
    let w = 16u32;
    let h = 16u32;
    let planar = make_planar(w as usize, h as usize);
    let packed = pack(w as usize, h as usize, &planar);
    let jpeg =
        encode_jpeg_cmyk(w, h, &packed, w as usize * 4, 90, Some(0)).expect("encode Adobe CMYK");
    let v = decode(w, h, jpeg);
    assert_per_component_psnr(&v, w, h, &planar, 30.0, "packed_adobe_inverted");
}

#[test]
fn packed_plain_progressive_roundtrip() {
    let w = 32u32;
    let h = 16u32;
    let planar = make_planar(w as usize, h as usize);
    let packed = pack(w as usize, h as usize, &planar);
    let jpeg = encode_jpeg_cmyk_progressive(w, h, &packed, w as usize * 4, 90, None)
        .expect("encode plain progressive CMYK");
    let v = decode(w, h, jpeg);
    assert_per_component_psnr(&v, w, h, &planar, 30.0, "packed_plain_progressive");
}

#[test]
fn packed_adobe_inverted_progressive_roundtrip() {
    let w = 16u32;
    let h = 16u32;
    let planar = make_planar(w as usize, h as usize);
    let packed = pack(w as usize, h as usize, &planar);
    let jpeg = encode_jpeg_cmyk_progressive(w, h, &packed, w as usize * 4, 90, Some(0))
        .expect("encode Adobe progressive CMYK");
    let v = decode(w, h, jpeg);
    assert_per_component_psnr(&v, w, h, &planar, 30.0, "packed_adobe_progressive");
}

#[test]
fn packed_rejects_short_stride() {
    let w = 8u32;
    let h = 8u32;
    let packed = vec![0u8; (w as usize * 3) * h as usize];
    let err =
        encode_jpeg_cmyk(w, h, &packed, w as usize * 3, 90, None).expect_err("stride too short");
    assert!(format!("{err:?}").contains("stride"));
}

#[test]
fn packed_rejects_short_buffer() {
    let w = 8u32;
    let h = 8u32;
    let packed = vec![0u8; (w as usize * 4) * h as usize / 2];
    let err =
        encode_jpeg_cmyk(w, h, &packed, w as usize * 4, 90, None).expect_err("buffer too short");
    assert!(format!("{err:?}").contains("shorter than"));
}

#[test]
fn packed_rejects_unknown_adobe_transform() {
    let w = 8u32;
    let h = 8u32;
    let packed = vec![0u8; (w as usize * 4) * h as usize];
    let err = encode_jpeg_cmyk(w, h, &packed, w as usize * 4, 90, Some(1))
        .expect_err("transform=1 rejected");
    assert!(format!("{err:?}").contains("transform must be 0"));
    let err = encode_jpeg_cmyk_progressive(w, h, &packed, w as usize * 4, 90, Some(99))
        .expect_err("transform=99 rejected on progressive");
    assert!(format!("{err:?}").contains("transform must be 0"));
}

// ---- per-plane back-end entry points (now public) ----------------------

#[test]
fn planar_baseline_public_back_end() {
    let w = 32u32;
    let h = 16u32;
    let planar = make_planar(w as usize, h as usize);
    let refs: [&[u8]; 4] = [&planar[0], &planar[1], &planar[2], &planar[3]];
    let strides = [w as usize; 4];
    let jpeg = encode_jpeg_cmyk_1111(w, h, &refs, &strides, 90, None)
        .expect("encode planar baseline CMYK");
    let v = decode(w, h, jpeg);
    assert_per_component_psnr(&v, w, h, &planar, 30.0, "planar_baseline");
}

#[test]
fn planar_progressive_public_back_end() {
    let w = 32u32;
    let h = 16u32;
    let planar = make_planar(w as usize, h as usize);
    let refs: [&[u8]; 4] = [&planar[0], &planar[1], &planar[2], &planar[3]];
    let strides = [w as usize; 4];
    let jpeg = encode_jpeg_progressive_cmyk_1111(w, h, &refs, &strides, 90, None)
        .expect("encode planar progressive CMYK");
    let v = decode(w, h, jpeg);
    assert_per_component_psnr(&v, w, h, &planar, 30.0, "planar_progressive");
}

// ---- trait-API MjpegEncoder path --------------------------------------

fn trait_encode_decode(
    w: u32,
    h: u32,
    packed: &[u8],
    adobe_transform: Option<u8>,
    progressive: bool,
) -> VideoFrame {
    let mut params = CodecParameters::video(CodecId::new("mjpeg"));
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::Cmyk);
    let mut enc = MjpegEncoder::from_params(&params).expect("MjpegEncoder::from_params");
    enc.set_adobe_transform(adobe_transform)
        .expect("set_adobe_transform");
    enc.set_progressive(progressive);
    let plane = VideoPlane {
        stride: w as usize * 4,
        data: packed.to_vec(),
    };
    let frame = Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![plane],
    });
    use oxideav_core::Encoder;
    enc.send_frame(&frame).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    decode(w, h, pkt.data)
}

#[test]
fn trait_api_cmyk_baseline_plain() {
    let w = 32u32;
    let h = 16u32;
    let planar = make_planar(w as usize, h as usize);
    let packed = pack(w as usize, h as usize, &planar);
    let v = trait_encode_decode(w, h, &packed, None, false);
    assert_per_component_psnr(&v, w, h, &planar, 30.0, "trait_baseline_plain");
}

#[test]
fn trait_api_cmyk_baseline_adobe_inverted() {
    let w = 16u32;
    let h = 16u32;
    let planar = make_planar(w as usize, h as usize);
    let packed = pack(w as usize, h as usize, &planar);
    let v = trait_encode_decode(w, h, &packed, Some(0), false);
    assert_per_component_psnr(&v, w, h, &planar, 30.0, "trait_baseline_adobe");
}

#[test]
fn trait_api_cmyk_progressive_plain() {
    let w = 32u32;
    let h = 16u32;
    let planar = make_planar(w as usize, h as usize);
    let packed = pack(w as usize, h as usize, &planar);
    let v = trait_encode_decode(w, h, &packed, None, true);
    assert_per_component_psnr(&v, w, h, &planar, 30.0, "trait_progressive_plain");
}

#[test]
fn trait_api_rejects_invalid_adobe_transform() {
    let mut params = CodecParameters::video(CodecId::new("mjpeg"));
    params.width = Some(8);
    params.height = Some(8);
    params.pixel_format = Some(PixelFormat::Cmyk);
    let mut enc = MjpegEncoder::from_params(&params).expect("from_params");
    let err = enc
        .set_adobe_transform(Some(1))
        .expect_err("transform=1 rejected");
    assert!(format!("{err:?}").contains("APP14"));
    // Successful set (0) followed by a roundtrip read-back.
    enc.set_adobe_transform(Some(0))
        .expect("transform=0 accepted");
    assert_eq!(enc.adobe_transform(), Some(0));
    enc.set_adobe_transform(None)
        .expect("transform=None accepted");
    assert_eq!(enc.adobe_transform(), None);
}

#[test]
fn trait_api_rejects_short_cmyk_stride() {
    use oxideav_core::Encoder;
    let mut params = CodecParameters::video(CodecId::new("mjpeg"));
    params.width = Some(8);
    params.height = Some(8);
    params.pixel_format = Some(PixelFormat::Cmyk);
    let mut enc = MjpegEncoder::from_params(&params).expect("from_params");
    // stride = width * 3 — too short for packed CMYK.
    let plane = VideoPlane {
        stride: 24,
        data: vec![0u8; 24 * 8],
    };
    let frame = Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![plane],
    });
    let err = enc.send_frame(&frame).expect_err("short stride rejected");
    assert!(format!("{err:?}").contains("stride"));
}
