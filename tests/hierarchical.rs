#![cfg(feature = "registry")]
// Parallel-array index loops are idiomatic in these per-pixel reconstruction
// checks; skip the lint (same allow as the lossless roundtrip suite).
#![allow(clippy::needless_range_loop)]
//! Hierarchical mode (T.81 Annex J) — spatial lossless progression decode.
//!
//! The crate's encoder does not emit hierarchical streams, so these tests
//! hand-assemble a conformant two-stage spatial progression byte-by-byte
//! from the T.81 syntax (§B.3.2 DHP, §B.3.3 EXP, Annex H lossless scan,
//! §J.2 differential reconstruction) and verify the decoder reconstructs
//! the original full-resolution image bit-exactly.
//!
//! The stream shape exercised here is:
//!
//!   SOI
//!   DHP            (completed image = W × H, 1 component, P = 8)
//!   DHT            (standard lossless DC table, SSSS 0..=16)
//!   SOF3           (non-differential frame: low-res W/2 × H/2)
//!   SOS / scan     (Annex H predictor-1 lossless coding)
//!   EXP (Eh=Ev=1)  (expand reference ×2 both axes)
//!   SOF7           (differential frame: full-res W × H)
//!   SOS / scan     (§J.2.3.2 — difference coded directly, predictor 0)
//!   EOI
//!
//! The differential samples are the per-pixel two's-complement delta
//! between the original full-resolution image and the bilinearly upsampled
//! low-resolution reference, so the final reconstruction is exactly the
//! original.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_mjpeg::registry::make_decoder;

// ---- Standard lossless DC Huffman table (mirrors the encoder's
// STD_DC_LOSSLESS_* layout: 15 codes of length 4 for SSSS 0..=14, then 2
// codes of length 5 for SSSS 15 and 16; Kraft sum 1.0). -----------------

const DC_BITS: [u8; 16] = [0, 0, 0, 15, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const DC_VALS: [u8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

/// Build the canonical-order (len, code) for each symbol value per T.81
/// Annex C: codes are assigned in increasing code length, in the order the
/// symbols appear in the VALS list.
fn canonical_codes() -> std::collections::HashMap<u8, (u8, u32)> {
    let mut map = std::collections::HashMap::new();
    let mut code: u32 = 0;
    let mut k = 0usize;
    for len in 1..=16u8 {
        for _ in 0..DC_BITS[(len - 1) as usize] {
            map.insert(DC_VALS[k], (len, code));
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    map
}

/// MSB-first bit writer with §B.1.1.5 0xFF byte stuffing.
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }
    fn put(&mut self, code: u32, len: u8) {
        for i in (0..len).rev() {
            let bit = (code >> i) & 1;
            self.acc = (self.acc << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                let b = (self.acc & 0xFF) as u8;
                self.out.push(b);
                if b == 0xFF {
                    self.out.push(0x00);
                }
                self.acc = 0;
                self.nbits = 0;
            }
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            // Pad the final partial byte with 1-bits (Annex F convention).
            let pad = 8 - self.nbits;
            self.acc = (self.acc << pad) | ((1 << pad) - 1);
            let b = (self.acc & 0xFF) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0x00);
            }
        }
        self.out
    }
}

/// Magnitude category (SSSS) + the `ssss`-bit magnitude code for a signed
/// value per T.81 Annex F / Table H.2. Value 0 → SSSS 0 (no extra bits).
fn category(v: i32) -> (u8, u32) {
    if v == 0 {
        return (0, 0);
    }
    let m = v.unsigned_abs();
    let mut ssss = 0u8;
    let mut t = m;
    while t > 0 {
        ssss += 1;
        t >>= 1;
    }
    // For negative values the magnitude bits are the one's complement of
    // |v| in `ssss` bits (Annex F: V is in the range 2^(ssss-1)..2^ssss-1
    // for positive, and the negative mapping is V = value - (2^ssss - 1)).
    let bits = if v > 0 {
        v as u32
    } else {
        (v + ((1 << ssss) - 1)) as u32
    };
    (ssss, bits)
}

/// Encode one single-component lossless plane (predictor 1 = Ra, with the
/// H.1.2.1 edge fall-backs) into an entropy-coded scan body. `origin` is
/// the per-component prediction seed `2^(P-1)`.
fn encode_pred1_scan(plane: &[u32], w: usize, h: usize, origin: u32) -> Vec<u8> {
    let codes = canonical_codes();
    let mut bw = BitWriter::new();
    for y in 0..h {
        for x in 0..w {
            let pred: u32 = if x == 0 && y == 0 {
                origin
            } else if y == 0 {
                plane[y * w + x - 1]
            } else if x == 0 {
                plane[(y - 1) * w + x]
            } else {
                plane[y * w + x - 1] // predictor 1 = Ra
            };
            let diff = plane[y * w + x] as i32 - pred as i32;
            let (ssss, bits) = category(diff);
            let (len, code) = codes[&ssss];
            bw.put(code, len);
            if ssss > 0 {
                bw.put(bits, ssss);
            }
        }
    }
    bw.finish()
}

/// Encode a differential plane (§J.2.3.2: difference coded directly, no
/// prediction — every sample value IS the two's-complement difference).
fn encode_diff_scan(diff: &[i32]) -> Vec<u8> {
    let codes = canonical_codes();
    let mut bw = BitWriter::new();
    for &d in diff {
        let (ssss, bits) = category(d);
        let (len, code) = codes[&ssss];
        bw.put(code, len);
        if ssss > 0 {
            bw.put(bits, ssss);
        }
    }
    bw.finish()
}

fn push_marker(out: &mut Vec<u8>, m: u8) {
    out.push(0xFF);
    out.push(m);
}

/// A frame-header / DHP body: P, Y, X, Nf=1, component (id=1, H=V=1, Tq=0).
fn frame_body(p: u8, w: u16, h: u16) -> Vec<u8> {
    let mut v = Vec::new();
    v.push(p);
    v.extend_from_slice(&h.to_be_bytes());
    v.extend_from_slice(&w.to_be_bytes());
    v.push(1); // Nf
    v.push(1); // component id
    v.push(0x11); // H=1 V=1
    v.push(0); // Tq
    v
}

fn push_seg(out: &mut Vec<u8>, marker: u8, body: &[u8]) {
    push_marker(out, marker);
    let len = (body.len() + 2) as u16;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(body);
}

/// SOS body for a single-component lossless scan: Ns=1, component(id=1,
/// Td=0/Ta=0), Ss=predictor, Se=0, Ah=0/Al=pt.
fn sos_body(predictor: u8, pt: u8) -> Vec<u8> {
    vec![1, 1, 0x00, predictor, 0x00, pt]
}

/// DHT body for the standard lossless DC table (class 0, id 0).
fn dht_body() -> Vec<u8> {
    let mut v = Vec::new();
    v.push(0x00); // Tc=0 (DC), Th=0
    v.extend_from_slice(&DC_BITS);
    v.extend_from_slice(&DC_VALS);
    v
}

/// Bilinear ×2 upsampling per T.81 §J.1.1.2 (truncating midpoints, edge
/// replication). Horizontal first then vertical — mirrors the decoder.
fn upsample2x(plane: &[u32], w: usize, h: usize) -> (Vec<u32>, usize, usize) {
    // Horizontal.
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
    // Vertical.
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

/// Build a complete two-stage spatial-lossless hierarchical stream from a
/// full-resolution image, reconstructing exactly on decode.
fn build_hierarchical(full: &[u32], w: usize, h: usize) -> Vec<u8> {
    assert!(w % 2 == 0 && h % 2 == 0);
    let lw = w / 2;
    let lh = h / 2;
    // Low-resolution reference: simple decimation (take the top-left sample
    // of each 2×2 block). Any downsampling works — the differential frame
    // corrects whatever it is.
    let mut low = vec![0u32; lw * lh];
    for y in 0..lh {
        for x in 0..lw {
            low[y * lw + x] = full[(2 * y) * w + (2 * x)];
        }
    }
    // Upsample the reference the way the decoder will, so the differential
    // delta we encode lands exactly.
    let (up, uw, uh) = upsample2x(&low, lw, lh);
    assert_eq!((uw, uh), (w, h));
    let diff: Vec<i32> = (0..w * h)
        .map(|i| {
            // modulo-2^8 two's-complement difference
            let d = full[i] as i32 - up[i] as i32;
            ((d % 256) + 256) % 256 // canonical 0..255 then re-signed below
        })
        .map(|d| if d >= 128 { d - 256 } else { d })
        .collect();

    let origin = 1u32 << 7; // 2^(P-1), P=8

    let mut out = Vec::new();
    push_marker(&mut out, 0xD8); // SOI
    push_seg(&mut out, 0xDE, &frame_body(8, w as u16, h as u16)); // DHP (completed size)
    push_seg(&mut out, 0xC4, &dht_body()); // DHT (shared)

    // Stage 1: non-differential low-res frame (SOF3).
    push_seg(&mut out, 0xC3, &frame_body(8, lw as u16, lh as u16));
    push_seg(&mut out, 0xDA, &sos_body(1, 0)); // SOS predictor 1
    out.extend_from_slice(&encode_pred1_scan(&low, lw, lh, origin));

    // Stage 2: EXP (×2 both axes) + differential full-res frame (SOF7).
    push_seg(&mut out, 0xDF, &[0x11]); // EXP Eh=Ev=1
    push_seg(&mut out, 0xC7, &frame_body(8, w as u16, h as u16));
    push_seg(&mut out, 0xDA, &sos_body(0, 0)); // SOS predictor 0 (differential)
    out.extend_from_slice(&encode_diff_scan(&diff));

    push_marker(&mut out, 0xD9); // EOI
    out
}

fn mk_image(w: usize, h: usize, seed: u32) -> Vec<u32> {
    let mut v = vec![0u32; w * h];
    let mut s = seed;
    for px in v.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        // Mix a smooth gradient with pseudo-random texture so the
        // differential frame carries non-trivial residuals.
        *px = (s >> 16) & 0xFF;
    }
    v
}

#[test]
fn spatial_hierarchical_two_stage_reconstructs_exactly() {
    let (w, h) = (16usize, 12usize);
    let full = mk_image(w, h, 0xC0FFEE);
    let jpeg = build_hierarchical(&full, w, h);

    // Sanity: the stream carries DHP, SOF3, EXP and SOF7 markers.
    assert!(jpeg.windows(2).any(|x| x == [0xFF, 0xDE]), "DHP missing");
    assert!(jpeg.windows(2).any(|x| x == [0xFF, 0xC3]), "SOF3 missing");
    assert!(jpeg.windows(2).any(|x| x == [0xFF, 0xDF]), "EXP missing");
    assert!(jpeg.windows(2).any(|x| x == [0xFF, 0xC7]), "SOF7 missing");

    let frame = decode(jpeg, w as u32, h as u32);
    assert_eq!(frame.planes.len(), 1);
    let plane = &frame.planes[0];
    assert_eq!(plane.stride, w);
    for y in 0..h {
        for x in 0..w {
            let got = plane.data[y * plane.stride + x] as u32;
            let want = full[y * w + x];
            assert_eq!(got, want, "pixel ({x},{y}) mismatch: got {got} want {want}");
        }
    }
}

#[test]
fn spatial_hierarchical_flat_image_is_lossless() {
    // A flat field upsamples exactly, so the differential frame is all
    // zeros — exercises the SSSS=0 path end-to-end.
    let (w, h) = (8usize, 8usize);
    let full = vec![123u32; w * h];
    let jpeg = build_hierarchical(&full, w, h);
    let frame = decode(jpeg, w as u32, h as u32);
    let plane = &frame.planes[0];
    for i in 0..w * h {
        assert_eq!(plane.data[i] as u32, full[i]);
    }
}

#[test]
fn non_differential_only_hierarchical_decodes() {
    // A hierarchical stream with just the DHP + one SOF3 frame (no
    // differential refinement) decodes to that single frame. The DHP size
    // equals the frame size.
    let (w, h) = (8usize, 8usize);
    let img = mk_image(w, h, 0x1234);
    let origin = 1u32 << 7;
    let mut out = Vec::new();
    push_marker(&mut out, 0xD8);
    push_seg(&mut out, 0xDE, &frame_body(8, w as u16, h as u16));
    push_seg(&mut out, 0xC4, &dht_body());
    push_seg(&mut out, 0xC3, &frame_body(8, w as u16, h as u16));
    push_seg(&mut out, 0xDA, &sos_body(1, 0));
    out.extend_from_slice(&encode_pred1_scan(&img, w, h, origin));
    push_marker(&mut out, 0xD9);

    let frame = decode(out, w as u32, h as u32);
    let plane = &frame.planes[0];
    for i in 0..w * h {
        assert_eq!(plane.data[i] as u32, img[i], "pixel {i}");
    }
}
