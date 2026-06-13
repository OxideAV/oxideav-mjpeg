#![cfg(feature = "registry")]
//! DNL (Define Number of Lines) decode tests — T.81 §B.2.2 / §B.2.5.
//!
//! A JPEG frame header may code the number of lines `Y` as 0, in which
//! case the real line count is carried by a mandatory DNL segment that
//! "shall immediately follow the first scan" (T.81 §B.2.5). These tests
//! synthesise such streams from the in-crate encoder's normal output by
//! (a) zeroing the SOF `Y` field and (b) splicing a `FF DC 00 04 NL`
//! segment in just before EOI, then confirm the decoder recovers the
//! line count and decodes pixel-identically to the unmodified stream.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase, VideoFrame};

/// Build a flat-ish gradient frame for the requested chroma layout.
fn make_frame(w: u32, h: u32, pix: PixelFormat) -> VideoFrame {
    let (cw, ch): (u32, u32) = match pix {
        PixelFormat::Yuv444P => (w, h),
        PixelFormat::Yuv422P => (w.div_ceil(2), h),
        PixelFormat::Yuv420P => (w.div_ceil(2), h.div_ceil(2)),
        _ => panic!("unsupported for this test"),
    };
    let y_stride = w as usize;
    let mut y = vec![0u8; y_stride * h as usize];
    for j in 0..h as usize {
        for i in 0..w as usize {
            y[j * y_stride + i] = (((i + j) * 3) % 255) as u8;
        }
    }
    let cb_stride = cw as usize;
    let cr_stride = cw as usize;
    let mut cb = vec![0u8; cb_stride * ch as usize];
    let mut cr = vec![0u8; cr_stride * ch as usize];
    for j in 0..ch as usize {
        for i in 0..cw as usize {
            cb[j * cb_stride + i] = ((128 + (i as i32 - cw as i32 / 2)) as u8).clamp(0, 255);
            cr[j * cr_stride + i] = ((128 + (j as i32 - ch as i32 / 2)) as u8).clamp(0, 255);
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: y_stride,
                data: y,
            },
            VideoPlane {
                stride: cb_stride,
                data: cb,
            },
            VideoPlane {
                stride: cr_stride,
                data: cr,
            },
        ],
    }
}

/// Find the byte offset of the first SOF marker (`FF C0..C3 / C9..CB`).
fn find_sof(stream: &[u8]) -> usize {
    let mut i = 0;
    while i + 1 < stream.len() {
        if stream[i] == 0xFF {
            let m = stream[i + 1];
            if matches!(m, 0xC0..=0xC3 | 0xC9..=0xCB) {
                return i;
            }
        }
        i += 1;
    }
    panic!("no SOF marker found");
}

/// Rewrite a baseline/single-scan stream so the SOF codes `Y = 0` and a
/// DNL segment (`FF DC 00 04 NL_hi NL_lo`) is spliced in immediately
/// before the trailing EOI (`FF D9`). For the encoder's single-scan
/// output the entropy data runs straight up to EOI, so "just before EOI"
/// is exactly "immediately after the first scan".
fn make_dnl_variant(stream: &[u8], nl: u16) -> Vec<u8> {
    let sof = find_sof(stream);
    // SOF payload layout: FF Cn Lf_hi Lf_lo P Y_hi Y_lo X_hi X_lo Nf ...
    // The Y field is at sof+5 (high) and sof+6 (low).
    let mut out = stream.to_vec();
    out[sof + 5] = 0;
    out[sof + 6] = 0;

    // Locate the trailing EOI (the final FF D9).
    let eoi = out
        .windows(2)
        .rposition(|w| w == [0xFF, 0xD9])
        .expect("EOI present");
    let dnl = [0xFF, 0xDC, 0x00, 0x04, (nl >> 8) as u8, (nl & 0xFF) as u8];
    out.splice(eoi..eoi, dnl);
    out
}

fn decode(stream: Vec<u8>) -> VideoFrame {
    let params = CodecParameters::video(CodecId::new("mjpeg"));
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&params).expect("make decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), stream))
        .expect("send_packet");
    match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v,
        _ => panic!("expected a video frame"),
    }
}

fn try_decode(stream: Vec<u8>) -> Result<VideoFrame, String> {
    let params = CodecParameters::video(CodecId::new("mjpeg"));
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&params).map_err(|e| e.to_string())?;
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), stream))
        .map_err(|e| e.to_string())?;
    match dec.receive_frame().map_err(|e| e.to_string())? {
        Frame::Video(v) => Ok(v),
        _ => Err("not a video frame".into()),
    }
}

/// A DNL-resolved stream must decode to the same dimensions and the same
/// pixels as the equivalent stream whose SOF already carries the height.
fn dnl_roundtrip_for(w: u32, h: u32, pix: PixelFormat) {
    use oxideav_mjpeg::encoder::encode_jpeg;
    let frame = make_frame(w, h, pix);
    let base = encode_jpeg(&frame, w, h, pix, 80).expect("encode");
    let variant = make_dnl_variant(&base, h as u16);

    // The variant really does carry Y = 0 and a DNL segment now.
    let sof = find_sof(&variant);
    assert_eq!(
        u16::from_be_bytes([variant[sof + 5], variant[sof + 6]]),
        0,
        "variant SOF must code Y = 0"
    );
    assert!(
        variant.windows(2).any(|x| x == [0xFF, 0xDC]),
        "variant must carry a DNL marker"
    );

    let a = decode(base);
    let b = decode(variant);

    // The core `VideoFrame` carries no width/height field — the resolved
    // line count is reflected in the plane sizes. For the luma plane,
    // rows = data.len() / stride must equal `h` (the DNL value).
    let y = &b.planes[0];
    assert!(y.stride >= w as usize, "luma stride must cover the width");
    assert_eq!(
        y.data.len() / y.stride,
        h as usize,
        "DNL-resolved luma plane must have {h} rows ({w}x{h} {pix:?})"
    );

    // The DNL-resolved stream must decode pixel-identically to the
    // stream whose SOF already carried the height.
    assert_eq!(a.planes.len(), b.planes.len());
    for p in 0..a.planes.len() {
        assert_eq!(
            a.planes[p].stride, b.planes[p].stride,
            "plane {p} stride must match with and without DNL ({w}x{h} {pix:?})"
        );
        assert_eq!(
            a.planes[p].data, b.planes[p].data,
            "plane {p} must be identical with and without DNL ({w}x{h} {pix:?})"
        );
    }
}

#[test]
fn dnl_yuv420_resolves_height() {
    dnl_roundtrip_for(48, 32, PixelFormat::Yuv420P);
}

#[test]
fn dnl_yuv422_resolves_height() {
    dnl_roundtrip_for(40, 24, PixelFormat::Yuv422P);
}

#[test]
fn dnl_yuv444_resolves_height() {
    dnl_roundtrip_for(32, 16, PixelFormat::Yuv444P);
}

/// A non-square height that is not a multiple of the MCU height still
/// resolves: the encoder pads internally, the DNL carries the real
/// (unpadded) line count, and the decoder crops to it.
#[test]
fn dnl_non_mcu_aligned_height() {
    dnl_roundtrip_for(64, 30, PixelFormat::Yuv420P);
}

/// T.81 §B.2.5: when SOF codes `Y = 0` the DNL segment is *mandatory*.
/// A stream that zeroes Y but omits DNL must be rejected, not silently
/// decoded to a zero-height (or wrong-height) image.
#[test]
fn dnl_missing_is_rejected() {
    use oxideav_mjpeg::encoder::encode_jpeg;
    let (w, h) = (48u32, 32u32);
    let pix = PixelFormat::Yuv420P;
    let frame = make_frame(w, h, pix);
    let base = encode_jpeg(&frame, w, h, pix, 80).expect("encode");

    // Zero the SOF Y field but do NOT add a DNL segment.
    let sof = find_sof(&base);
    let mut broken = base.clone();
    broken[sof + 5] = 0;
    broken[sof + 6] = 0;

    let res = try_decode(broken);
    assert!(
        res.is_err(),
        "Y = 0 without a DNL segment must be rejected (DNL is mandatory)"
    );
}

/// A malformed DNL carrying `NL = 0` is rejected — a DNL segment exists
/// to define a non-zero line count (T.81 Table B.10: NL ∈ 1..65535).
#[test]
fn dnl_zero_nl_is_rejected() {
    use oxideav_mjpeg::encoder::encode_jpeg;
    let (w, h) = (48u32, 32u32);
    let pix = PixelFormat::Yuv420P;
    let frame = make_frame(w, h, pix);
    let base = encode_jpeg(&frame, w, h, pix, 80).expect("encode");
    let variant = make_dnl_variant(&base, 0);
    let res = try_decode(variant);
    assert!(res.is_err(), "DNL with NL = 0 must be rejected");
}
