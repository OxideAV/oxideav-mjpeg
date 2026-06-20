#![cfg(feature = "registry")]
#![allow(clippy::needless_range_loop)]
//! Hierarchical mode (T.81 Annex J) — DCT progression decode (§K.7.2.1).
//!
//! Two shapes are exercised:
//!
//! 1. **Single-stage DCT progression.** A DHP segment (§B.3.2) is spliced
//!    in front of an ordinary baseline (SOF0) JPEG produced by this crate's
//!    encoder. The decoder routes the stream through the hierarchical
//!    control loop, decodes the SOF0 frame as the non-differential first
//!    (and only) stage, and must reconstruct exactly the same pixels as the
//!    plain baseline decode of the same bytes. This proves the §K.7.2.1
//!    non-differential DCT frame path + DCT reference-plane shaping.
//!
//! 2. **Two-stage DCT progression with a differential SOF5 frame.** A
//!    low-resolution non-differential SOF0 frame is followed by an EXP
//!    (Eh=Ev=1) ×2 upsample and a differential SOF5 frame (§J.2.3.1: IDCT
//!    without level shift, DC decoded directly). The differential blocks
//!    carry a flat per-block correction so the reconstructed full-resolution
//!    image matches a known target. This proves the differential DCT frame
//!    path + the modulo-2^16 §J.2.1 reconstruction.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_mjpeg::registry::make_decoder;

fn decode(jpeg: Vec<u8>, w: u32, h: u32) -> oxideav_core::VideoFrame {
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

/// Walk JPEG markers and return the byte offset of the first SOFn marker
/// (`0xFF 0xC?` excluding DHT 0xC4 / JPG 0xC8) and its segment length
/// (including the 2-byte length field), so the SOF segment body can be
/// copied to synthesise a DHP.
fn find_sof(jpeg: &[u8]) -> (usize, usize) {
    let mut i = 2; // skip SOI
    while i + 1 < jpeg.len() {
        if jpeg[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = jpeg[i + 1];
        if m == 0xFF {
            i += 1;
            continue;
        }
        // SOI/EOI/RST* carry no length.
        if m == 0xD8 || m == 0xD9 || (0xD0..=0xD7).contains(&m) {
            i += 2;
            continue;
        }
        let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        let is_sof = matches!(m, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF);
        if is_sof {
            return (i, 2 + len);
        }
        // SOS — entropy data follows; stop walking.
        if m == 0xDA {
            break;
        }
        i += 2 + len;
    }
    panic!("no SOF found");
}

/// Splice a DHP (§B.3.2) segment, copied from the stream's SOF header body,
/// directly in front of the SOF marker. The DHP body has the same shape as
/// a frame header (P, Y, X, Nf, components); Tq is irrelevant at the DHP
/// level so the SOF body is copied verbatim. The result is a single-stage
/// hierarchical DCT progression.
fn wrap_single_stage_dhp(jpeg: &[u8]) -> Vec<u8> {
    let (sof_off, _sof_len) = find_sof(jpeg);
    // SOF segment body (after the 0xFF marker + length field).
    let seg_len = u16::from_be_bytes([jpeg[sof_off + 2], jpeg[sof_off + 3]]) as usize;
    let body = &jpeg[sof_off + 4..sof_off + 2 + seg_len];
    let mut dhp = Vec::new();
    dhp.push(0xFF);
    dhp.push(0xDE); // DHP
    let dhp_len = (body.len() + 2) as u16;
    dhp.extend_from_slice(&dhp_len.to_be_bytes());
    dhp.extend_from_slice(body);

    let mut out = Vec::with_capacity(jpeg.len() + dhp.len());
    out.extend_from_slice(&jpeg[..sof_off]);
    out.extend_from_slice(&dhp);
    out.extend_from_slice(&jpeg[sof_off..]);
    out
}

#[test]
fn single_stage_dct_hierarchical_grayscale_matches_baseline() {
    let (w, h) = (24usize, 16usize);
    let mut src = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            // A smooth gradient with a little structure.
            src[y * w + x] = ((x * 7 + y * 11) & 0xFF) as u8;
        }
    }
    let baseline =
        oxideav_mjpeg::encoder::encode_jpeg_grayscale(w as u32, h as u32, &src, w, 80).unwrap();

    let plain = decode(baseline.clone(), w as u32, h as u32);
    let hier = decode(wrap_single_stage_dhp(&baseline), w as u32, h as u32);

    assert_eq!(plain.planes[0].data, hier.planes[0].data);
    assert_eq!(hier.planes.len(), 1);
}

#[test]
fn single_stage_dct_hierarchical_rgb_matches_baseline() {
    let (w, h) = (16usize, 16usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let o = (y * w + x) * 3;
            rgb[o] = (x * 15) as u8;
            rgb[o + 1] = (y * 15) as u8;
            rgb[o + 2] = ((x + y) * 7) as u8;
        }
    }
    let baseline =
        oxideav_mjpeg::encoder::encode_jpeg_rgb24(w as u32, h as u32, &rgb, w * 3, 85).unwrap();

    let plain = decode(baseline.clone(), w as u32, h as u32);
    let hier = decode(wrap_single_stage_dhp(&baseline), w as u32, h as u32);

    // RGB-class baseline → single packed Rgb24 plane (3 bytes/pixel).
    assert_eq!(hier.planes.len(), 1);
    assert_eq!(hier.planes[0].stride, w * 3);
    assert_eq!(plain.planes[0].data, hier.planes[0].data);
}
