#![cfg(feature = "registry")]
#![allow(clippy::needless_range_loop)]
//! Hierarchical mode (T.81 Annex J) — **arithmetic** spatial-lossless
//! progression decode (SOF11 non-differential + SOF15 differential).
//!
//! The crate's encoder does not emit hierarchical streams, so these tests
//! hand-assemble a conformant two-stage arithmetic spatial progression
//! byte-by-byte from the T.81 syntax (§B.3.2 DHP, §B.3.3 EXP, Annex H
//! §H.1.2.3 arithmetic lossless scan, §J.2 differential reconstruction) and
//! verify the decoder reconstructs the original full-resolution image
//! bit-exactly. The Q-coder scan bytes are produced with the crate's own
//! `jpeg::arith` encoder primitives (`ArithEncoder`, `LosslessStats`,
//! `encode_lossless_diff`) — the exact mirror of the decode-side model — so
//! the bytes the decoder consumes are sample-exact by construction.
//!
//!   SOI
//!   DHP            (completed image = W × H, 1 component, P = 8)
//!   SOF11          (non-differential frame: low-res W/2 × H/2, arith lossless)
//!   SOS / scan     (§H.1.2.3 predictor-1 arithmetic lossless coding)
//!   EXP (Eh=Ev=1)  (expand reference ×2 both axes)
//!   SOF15          (differential frame: full-res W × H, arith lossless)
//!   SOS / scan     (§J.2.3.2 — difference coded directly, predictor 0)
//!   EOI

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_mjpeg::jpeg::arith::{encode_lossless_diff, ArithEncoder, LosslessStats};
use oxideav_mjpeg::registry::make_decoder;

fn push_marker(out: &mut Vec<u8>, m: u8) {
    out.push(0xFF);
    out.push(m);
}

fn push_seg(out: &mut Vec<u8>, marker: u8, body: &[u8]) {
    push_marker(out, marker);
    let len = (body.len() + 2) as u16;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(body);
}

/// SOFn / DHP body: P=8, Y, X, Nf, components (all id=1.., H=V=1, Tq=0).
fn frame_body(w: u16, h: u16, nf: u8) -> Vec<u8> {
    let mut v = vec![8, (h >> 8) as u8, h as u8, (w >> 8) as u8, w as u8, nf];
    for id in 1..=nf {
        v.extend_from_slice(&[id, 0x11, 0]);
    }
    v
}

/// SOS body for an interleaved lossless scan: Ns, then (id, Td=0/Ta=0) per
/// component, then Ss=predictor, Se=0, Ah=0/Al=0.
fn sos_body(predictor: u8, nf: u8) -> Vec<u8> {
    let mut v = vec![nf];
    for id in 1..=nf {
        v.push(id);
        v.push(0x00);
    }
    v.extend_from_slice(&[predictor, 0, 0x00]);
    v
}

/// Bilinear ×2 upsampling per T.81 §J.1.1.2 (truncating midpoints, edge
/// replication). Horizontal first then vertical — mirrors the decoder.
fn upsample2x(plane: &[u32], w: usize, h: usize) -> (Vec<u32>, usize, usize) {
    let hw = w * 2;
    let mut hor = vec![0u32; hw * h];
    for y in 0..h {
        for x in 0..w {
            let ra = plane[y * w + x];
            let rb = if x + 1 < w { plane[y * w + x + 1] } else { ra };
            hor[y * hw + 2 * x] = ra;
            hor[y * hw + 2 * x + 1] = (ra + rb) / 2;
        }
    }
    let vh = h * 2;
    let mut out = vec![0u32; hw * vh];
    for y in 0..h {
        for x in 0..hw {
            let ra = hor[y * hw + x];
            let rb = if y + 1 < h { hor[(y + 1) * hw + x] } else { ra };
            out[(2 * y) * hw + x] = ra;
            out[(2 * y + 1) * hw + x] = (ra + rb) / 2;
        }
    }
    (out, hw, vh)
}

/// Encode `nc` interleaved component planes into one arithmetic lossless scan.
/// `pred` selects the §H.1.2.1 reconstruction: for a non-differential frame
/// (`differential = false`) predictor 1 (Ra, with the origin / first-line /
/// first-column edge rules); for a differential frame (`differential = true`)
/// the value itself is the difference (no prediction, §J.2.3.2). The Q-coder
/// model and the `Da`/`Db` conditioning history mirror the decoder exactly.
fn encode_arith_lossless_scan(
    planes: &[Vec<i32>],
    w: usize,
    h: usize,
    differential: bool,
) -> Vec<u8> {
    let nc = planes.len();
    let origin: i32 = 1 << 7; // 2^(P-1), P=8
    let mut enc = ArithEncoder::new();
    let mut stats: Vec<LosslessStats> = (0..nc).map(|_| LosslessStats::new()).collect();
    let mut prev_diff: Vec<Vec<i32>> = (0..nc).map(|_| vec![0i32; w]).collect();
    let mut cur_diff: Vec<Vec<i32>> = (0..nc).map(|_| vec![0i32; w]).collect();
    for y in 0..h {
        for x in 0..w {
            for ci in 0..nc {
                let plane = &planes[ci];
                let val = plane[y * w + x];
                let pred = if differential {
                    0
                } else if x == 0 && y == 0 {
                    origin
                } else if y == 0 {
                    plane[y * w + x - 1]
                } else if x == 0 {
                    plane[(y - 1) * w + x]
                } else {
                    plane[y * w + x - 1] // predictor 1 = Ra
                };
                let diff = val - pred;
                let da = if x == 0 { 0 } else { cur_diff[ci][x - 1] };
                let db = prev_diff[ci][x];
                encode_lossless_diff(&mut enc, &mut stats[ci], da, db, diff).unwrap();
                cur_diff[ci][x] = diff;
            }
        }
        std::mem::swap(&mut prev_diff, &mut cur_diff);
    }
    enc.finish()
}

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

/// Build a two-stage arithmetic spatial-lossless hierarchical stream
/// (SOF11 + SOF15) from `nc` full-resolution planes, reconstructing exactly.
fn build_arith_hierarchical(full: &[Vec<u32>], w: usize, h: usize) -> Vec<u8> {
    assert!(w % 2 == 0 && h % 2 == 0);
    let nc = full.len();
    let lw = w / 2;
    let lh = h / 2;
    // Low-resolution reference: decimate (top-left of each 2×2 block).
    let low: Vec<Vec<u32>> = full
        .iter()
        .map(|plane| {
            let mut l = vec![0u32; lw * lh];
            for y in 0..lh {
                for x in 0..lw {
                    l[y * lw + x] = plane[(2 * y) * w + (2 * x)];
                }
            }
            l
        })
        .collect();
    // Differential delta = full − (decoder's ×2 upsample of low).
    let diff: Vec<Vec<i32>> = full
        .iter()
        .zip(low.iter())
        .map(|(plane, lo)| {
            let (up, uw, uh) = upsample2x(lo, lw, lh);
            assert_eq!((uw, uh), (w, h));
            (0..w * h)
                .map(|i| {
                    let d = plane[i] as i32 - up[i] as i32;
                    let m = ((d % 256) + 256) % 256;
                    if m >= 128 {
                        m - 256
                    } else {
                        m
                    }
                })
                .collect()
        })
        .collect();

    let low_i: Vec<Vec<i32>> = low
        .iter()
        .map(|p| p.iter().map(|&v| v as i32).collect())
        .collect();

    let mut out = Vec::new();
    push_marker(&mut out, 0xD8); // SOI
    push_seg(&mut out, 0xDE, &frame_body(w as u16, h as u16, nc as u8)); // DHP
                                                                         // Stage 1: non-differential SOF11 (low-res), predictor 1.
    push_seg(&mut out, 0xCB, &frame_body(lw as u16, lh as u16, nc as u8));
    push_seg(&mut out, 0xDA, &sos_body(1, nc as u8));
    out.extend_from_slice(&encode_arith_lossless_scan(&low_i, lw, lh, false));
    // Stage 2: EXP ×2 + differential SOF15 (full-res), predictor 0.
    push_seg(&mut out, 0xDF, &[0x11]);
    push_seg(&mut out, 0xCF, &frame_body(w as u16, h as u16, nc as u8));
    push_seg(&mut out, 0xDA, &sos_body(0, nc as u8));
    out.extend_from_slice(&encode_arith_lossless_scan(&diff, w, h, true));
    push_marker(&mut out, 0xD9); // EOI
    out
}

fn mk_image(w: usize, h: usize, seed: u32) -> Vec<u32> {
    let mut v = vec![0u32; w * h];
    let mut s = seed;
    for px in v.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        *px = (s >> 16) & 0xFF;
    }
    v
}

#[test]
fn arith_hierarchical_grayscale_roundtrip() {
    let (w, h) = (8usize, 8usize);
    let img = mk_image(w, h, 0x1234);
    let jpeg = build_arith_hierarchical(std::slice::from_ref(&img), w, h);
    let frame = decode(jpeg, w as u32, h as u32);
    assert_eq!(frame.planes.len(), 1, "expected single Gray8 plane");
    let stride = frame.planes[0].stride;
    let data = &frame.planes[0].data;
    for y in 0..h {
        for x in 0..w {
            assert_eq!(
                data[y * stride + x] as u32,
                img[y * w + x],
                "pixel ({x},{y})"
            );
        }
    }
}

#[test]
fn arith_hierarchical_rgb_roundtrip() {
    let (w, h) = (8usize, 6usize);
    let r = mk_image(w, h, 0x1111);
    let g = mk_image(w, h, 0x2222);
    let b = mk_image(w, h, 0x3333);
    let full = vec![r.clone(), g.clone(), b.clone()];
    let jpeg = build_arith_hierarchical(&full, w, h);
    let frame = decode(jpeg, w as u32, h as u32);
    // 3-component P=8 lossless → packed Rgb24 (one plane, 3 bytes/pixel).
    assert_eq!(frame.planes.len(), 1, "expected packed Rgb24 plane");
    let stride = frame.planes[0].stride;
    assert_eq!(stride, w * 3);
    let data = &frame.planes[0].data;
    for y in 0..h {
        for x in 0..w {
            let o = y * stride + x * 3;
            assert_eq!(data[o] as u32, r[y * w + x], "R ({x},{y})");
            assert_eq!(data[o + 1] as u32, g[y * w + x], "G ({x},{y})");
            assert_eq!(data[o + 2] as u32, b[y * w + x], "B ({x},{y})");
        }
    }
}
