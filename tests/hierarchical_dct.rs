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

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase, VideoFrame};
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

/// Build a packed `[C, M, Y, K]` gradient buffer (`stride = width * 4`).
fn make_packed_cmyk(w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 4];
    for j in 0..h {
        for i in 0..w {
            let o = (j * w + i) * 4;
            out[o] = (i * 255 / w.max(1)).min(255) as u8;
            out[o + 1] = (j * 255 / h.max(1)).min(255) as u8;
            out[o + 2] = ((i + j) * 255 / (w + h).max(1)).min(255) as u8;
            out[o + 3] = (((i ^ j) * 7) & 0xFF) as u8 / 2;
        }
    }
    out
}

#[test]
fn single_stage_dct_hierarchical_cmyk_matches_baseline() {
    // 4-component (CMYK-class) baseline → component IDs 1/2/3/4, every
    // component H = V = 1, Adobe APP14 transform = 0 (plain CMYK). The
    // hierarchical DCT loop decodes it as the sole non-differential stage
    // and shapes the four reconstructed reference planes to packed Cmyk
    // (honouring the Adobe transform), matching the plain baseline decode.
    let (w, h) = (16usize, 16usize);
    let packed = make_packed_cmyk(w, h);
    let baseline =
        oxideav_mjpeg::encoder::encode_jpeg_cmyk(w as u32, h as u32, &packed, w * 4, 90, Some(0))
            .unwrap();

    let plain = decode(baseline.clone(), w as u32, h as u32);
    let hier = decode(wrap_single_stage_dhp(&baseline), w as u32, h as u32);

    // CMYK-class → single packed Cmyk plane (4 bytes/pixel).
    assert_eq!(hier.planes.len(), 1, "expected one packed Cmyk plane");
    assert_eq!(hier.planes[0].stride, w * 4);
    assert_eq!(
        plain.planes[0].data, hier.planes[0].data,
        "CMYK hierarchical decode must match the plain baseline decode"
    );
}

/// Build a 4:4:4 YUV `VideoFrame` (three full-resolution planar Y/Cb/Cr
/// planes) carrying a smooth gradient with a little structure on every
/// component — exercised by the YUV-class hierarchical DCT path.
fn make_yuv444_frame(w: usize, h: usize) -> VideoFrame {
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; w * h];
    let mut cr = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let o = j * w + i;
            y[o] = ((i * 9 + j * 5) & 0xFF) as u8;
            cb[o] = (128 + (i as i32 - w as i32 / 2)).clamp(0, 255) as u8;
            cr[o] = (128 + (j as i32 - h as i32 / 2)).clamp(0, 255) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: w,
                data: cb,
            },
            VideoPlane {
                stride: w,
                data: cr,
            },
        ],
    }
}

#[test]
fn single_stage_dct_hierarchical_yuv_matches_baseline() {
    let (w, h) = (16usize, 16usize);
    let frame = make_yuv444_frame(w, h);
    // 4:4:4 baseline → component IDs 1/2/3 (YUV-class, every component
    // H = V = 1), so the hierarchical DCT loop routes it to the new
    // YUV-class shaping → planar Yuv444P.
    let baseline =
        oxideav_mjpeg::encoder::encode_jpeg(&frame, w as u32, h as u32, PixelFormat::Yuv444P, 85)
            .unwrap();

    let plain = decode(baseline.clone(), w as u32, h as u32);
    let hier = decode(wrap_single_stage_dhp(&baseline), w as u32, h as u32);

    // YUV-class → three planar Yuv444P planes, full resolution.
    assert_eq!(hier.planes.len(), 3, "expected 3 planar Y/Cb/Cr planes");
    assert_eq!(plain.planes.len(), 3);
    for p in 0..3 {
        assert_eq!(hier.planes[p].stride, w, "plane {p} stride");
        assert_eq!(
            plain.planes[p].data, hier.planes[p].data,
            "plane {p} mismatch between baseline and hierarchical decode"
        );
    }
}

#[test]
fn single_stage_progressive_dct_hierarchical_yuv_matches_baseline() {
    // Non-differential first frame = SOF2 (progressive) YUV-class 4:4:4. The
    // hierarchical control loop decodes the SOF2 frame as the sole stage via
    // the spectral-selection accumulator, then shapes the YUV-class
    // reference to planar Yuv444P. Proves the SOF2-first DCT progression
    // routes through the new YUV-class shaping.
    let (w, h) = (16usize, 16usize);
    let frame = make_yuv444_frame(w, h);
    let prog = oxideav_mjpeg::encoder::encode_jpeg_progressive(
        &frame,
        w as u32,
        h as u32,
        PixelFormat::Yuv444P,
        85,
    )
    .unwrap();

    let plain = decode(prog.clone(), w as u32, h as u32);
    let hier = decode(wrap_single_stage_dhp(&prog), w as u32, h as u32);

    assert_eq!(hier.planes.len(), 3);
    for p in 0..3 {
        assert_eq!(
            plain.planes[p].data, hier.planes[p].data,
            "plane {p} mismatch (SOF2 YUV hierarchical vs plain)"
        );
    }
}

// ---- Two-stage differential DCT progression (SOF0 + EXP + SOF5) ----------
//
// The frames here use flat (DC-only) 8×8 blocks so the DCT round-trip is
// exact: a constant `c` image has DCT coefficient `8*c` (level-shifted)
// with all AC = 0, and the IDCT of a DC-only block `C` is the constant
// `C/8`. With a quant table of all 1s the stored coefficient is the DC
// value directly. This lets us hand-assemble a verifiable differential
// progression without an FDCT.

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
/// value (T.81 §F.1.2.1 Table F.1 / the Annex J extension for SSSS up to 15).
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
    let bits = if v > 0 {
        v as u32
    } else {
        (v + ((1 << ssss) - 1)) as u32
    };
    (ssss, bits)
}

// DC Huffman table covering categories 0..=15 (one code of each length
// 1..=16 is impossible; use the canonical 16-symbol layout: SSSS i gets a
// code of length i+1 for i in 0..=15 → BITS has one symbol at each length
// 1..16). Kraft sum = sum 2^-(i+1) for i 0..15 < 1, valid prefix code.
const DC_BITS: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
const DC_VALS: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
// AC Huffman table with a single symbol: 0x00 (EOB), code length 1.
const AC_BITS: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const AC_VALS: [u8; 1] = [0x00];

fn canonical_codes(bits: &[u8; 16], vals: &[u8]) -> std::collections::HashMap<u8, (u8, u32)> {
    let mut map = std::collections::HashMap::new();
    let mut code: u32 = 0;
    let mut k = 0usize;
    for len in 1..=16u8 {
        for _ in 0..bits[(len - 1) as usize] {
            map.insert(vals[k], (len, code));
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    map
}

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

/// SOFn / DHP body: P=8, Y, X, Nf=1, component (id=1, H=V=1, Tq=0).
fn frame_body(w: u16, h: u16) -> Vec<u8> {
    vec![
        8,
        (h >> 8) as u8,
        h as u8,
        (w >> 8) as u8,
        w as u8,
        1,
        1,
        0x11,
        0,
    ]
}

/// DQT body: 8-bit precision, table 0, all 64 entries = 1.
fn dqt_body() -> Vec<u8> {
    let mut v = vec![0x00]; // Pq=0, Tq=0
    v.extend_from_slice(&[1u8; 64]);
    v
}

/// DHT body for a (class, id) table.
fn dht_body(tc: u8, th: u8, bits: &[u8; 16], vals: &[u8]) -> Vec<u8> {
    let mut v = vec![(tc << 4) | th];
    v.extend_from_slice(bits);
    v.extend_from_slice(vals);
    v
}

/// SOS body: Ns=1, component(id=1, Td=0/Ta=0), Ss=0, Se=63, Ah=0/Al=0.
fn sos_body() -> Vec<u8> {
    vec![1, 1, 0x00, 0, 63, 0x00]
}

/// SOFn / DHP body: P=8, Y, X, Nf=3, three components (ids 1/2/3, all
/// H=V=1, Tq=0). Component IDs 1/2/3 (not R/G/B, no Adobe APP14) make this a
/// YUV-class three-component frame.
fn frame_body_3comp(w: u16, h: u16) -> Vec<u8> {
    let mut v = vec![8, (h >> 8) as u8, h as u8, (w >> 8) as u8, w as u8, 3];
    for id in 1u8..=3 {
        v.extend_from_slice(&[id, 0x11, 0]);
    }
    v
}

/// SOS body: Ns=3, components 1/2/3 (Td=0/Ta=0 each), Ss=0, Se=63,
/// Ah=0/Al=0 — an interleaved three-component scan.
fn sos_body_3comp() -> Vec<u8> {
    vec![3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0x00]
}

/// Encode an interleaved three-component DCT scan of flat DC-only 8×8
/// blocks, all components at H=V=1. `blocks[c]` is the raster-ordered list
/// of flat block values for component `c`; every component must supply the
/// same number of blocks (one per MCU). Within each MCU the three component
/// blocks are emitted in order (§A.2.3). DC prediction is tracked
/// independently per component.
fn encode_flat_dct_scan_3comp(blocks: &[Vec<i32>; 3], differential: bool) -> Vec<u8> {
    let dc_codes = canonical_codes(&DC_BITS, &DC_VALS);
    let ac_codes = canonical_codes(&AC_BITS, &AC_VALS);
    let nmcu = blocks[0].len();
    assert_eq!(blocks[1].len(), nmcu);
    assert_eq!(blocks[2].len(), nmcu);
    let mut bw = BitWriter::new();
    let mut prev_dc = [0i32; 3];
    let shift = if differential { 0 } else { 128 };
    for mcu in 0..nmcu {
        for c in 0..3 {
            let dc_coef = 8 * (blocks[c][mcu] - shift);
            let diff = if differential {
                dc_coef
            } else {
                dc_coef - prev_dc[c]
            };
            prev_dc[c] = dc_coef;
            let (ssss, bitsv) = category(diff);
            let (len, code) = dc_codes[&ssss];
            bw.put(code, len);
            if ssss > 0 {
                bw.put(bitsv, ssss);
            }
            let (l, cc) = ac_codes[&0x00];
            bw.put(cc, l);
        }
    }
    bw.finish()
}

/// SOFn / DHP body: P=8, Y, X, Nf=4, four components (ids 1/2/3/4, all
/// H=V=1, Tq=0) — a CMYK-class four-component frame.
fn frame_body_4comp(w: u16, h: u16) -> Vec<u8> {
    let mut v = vec![8, (h >> 8) as u8, h as u8, (w >> 8) as u8, w as u8, 4];
    for id in 1u8..=4 {
        v.extend_from_slice(&[id, 0x11, 0]);
    }
    v
}

/// SOS body: Ns=4, components 1/2/3/4 (Td=0/Ta=0 each), Ss=0, Se=63,
/// Ah=0/Al=0 — an interleaved four-component scan.
fn sos_body_4comp() -> Vec<u8> {
    vec![4, 1, 0x00, 2, 0x00, 3, 0x00, 4, 0x00, 0, 63, 0x00]
}

/// Adobe APP14 segment body declaring colour `transform` (0 = CMYK).
fn adobe_app14_body(transform: u8) -> Vec<u8> {
    // "Adobe" + version(2) + flags0(2) + flags1(2) + transform(1).
    vec![
        b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, transform,
    ]
}

/// Encode an interleaved four-component DCT scan of flat DC-only 8×8
/// blocks, all components at H=V=1. Mirrors `encode_flat_dct_scan_3comp`.
fn encode_flat_dct_scan_4comp(blocks: &[Vec<i32>; 4], differential: bool) -> Vec<u8> {
    let dc_codes = canonical_codes(&DC_BITS, &DC_VALS);
    let ac_codes = canonical_codes(&AC_BITS, &AC_VALS);
    let nmcu = blocks[0].len();
    for b in blocks.iter() {
        assert_eq!(b.len(), nmcu);
    }
    let mut bw = BitWriter::new();
    let mut prev_dc = [0i32; 4];
    let shift = if differential { 0 } else { 128 };
    for mcu in 0..nmcu {
        for c in 0..4 {
            let dc_coef = 8 * (blocks[c][mcu] - shift);
            let diff = if differential {
                dc_coef
            } else {
                dc_coef - prev_dc[c]
            };
            prev_dc[c] = dc_coef;
            let (ssss, bitsv) = category(diff);
            let (len, code) = dc_codes[&ssss];
            bw.put(code, len);
            if ssss > 0 {
                bw.put(bitsv, ssss);
            }
            let (l, cc) = ac_codes[&0x00];
            bw.put(cc, l);
        }
    }
    bw.finish()
}

/// Encode a single-component DCT scan of flat DC-only 8×8 blocks. Each block
/// `b` carries constant value `vals[b]`; the stored DC coefficient is
/// `8*vals[b]` (after the optional level shift), AC all zero (EOB only).
/// `differential` selects the §J.2.3.1 model: no level shift, DC coded
/// directly (no inter-block prediction).
fn encode_flat_dct_scan(blocks: &[i32], differential: bool) -> Vec<u8> {
    let dc_codes = canonical_codes(&DC_BITS, &DC_VALS);
    let ac_codes = canonical_codes(&AC_BITS, &AC_VALS);
    let mut bw = BitWriter::new();
    let mut prev_dc = 0i32;
    for &b in blocks {
        // DC coefficient = 8 * (value - level_shift_offset). For
        // non-differential P=8 the level shift is 128; for differential it
        // is 0.
        let shift = if differential { 0 } else { 128 };
        let dc_coef = 8 * (b - shift);
        let diff = if differential {
            dc_coef // coded directly
        } else {
            dc_coef - prev_dc
        };
        prev_dc = dc_coef;
        let (ssss, bitsv) = category(diff);
        let (len, code) = dc_codes[&ssss];
        bw.put(code, len);
        if ssss > 0 {
            bw.put(bitsv, ssss);
        }
        // AC: single EOB.
        let (l, c) = ac_codes[&0x00];
        bw.put(c, l);
    }
    bw.finish()
}

#[test]
fn two_stage_differential_dct_hierarchical_grayscale() {
    // Stage 1 (non-differential SOF0): 8×8 flat block, value 100.
    // → upsample ×2 both axes → flat 16×16 = 100.
    // Stage 2 (differential SOF5): 16×16 = four 8×8 blocks, each a flat
    // correction d. Final block pixels = 100 + d.
    let base: i32 = 100;
    let corrections: [i32; 4] = [10, -5, 20, -30];

    let mut out = Vec::new();
    push_marker(&mut out, 0xD8); // SOI
    push_seg(&mut out, 0xDE, &frame_body(16, 16)); // DHP (completed image 16×16)
    push_seg(&mut out, 0xDB, &dqt_body()); // DQT (all 1s)
    push_seg(&mut out, 0xC4, &dht_body(0, 0, &DC_BITS, &DC_VALS)); // DC DHT
    push_seg(&mut out, 0xC4, &dht_body(1, 0, &AC_BITS, &AC_VALS)); // AC DHT

    // Stage 1: non-differential SOF0, low-res 8×8.
    push_seg(&mut out, 0xC0, &frame_body(8, 8));
    push_seg(&mut out, 0xDA, &sos_body());
    out.extend_from_slice(&encode_flat_dct_scan(&[base], false));

    // EXP ×2 both axes.
    push_seg(&mut out, 0xDF, &[0x11]);

    // Stage 2: differential SOF5, full-res 16×16 (2×2 blocks).
    push_seg(&mut out, 0xC5, &frame_body(16, 16));
    push_seg(&mut out, 0xDA, &sos_body());
    out.extend_from_slice(&encode_flat_dct_scan(&corrections, true));

    push_marker(&mut out, 0xD9); // EOI

    let frame = decode(out, 16, 16);
    assert_eq!(frame.planes.len(), 1);
    let stride = frame.planes[0].stride;
    let data = &frame.planes[0].data;
    // Block layout: block index = (by, bx) covers rows by*8.. cols bx*8.. .
    // corrections array is raster block order: [TL, TR, BL, BR].
    let block_corr = |bx: usize, by: usize| corrections[by * 2 + bx];
    for y in 0..16usize {
        for x in 0..16usize {
            let bx = x / 8;
            let by = y / 8;
            let expect = (base + block_corr(bx, by)).clamp(0, 255) as u8;
            assert_eq!(
                data[y * stride + x],
                expect,
                "pixel ({x},{y}) block ({bx},{by})"
            );
        }
    }
}

#[test]
fn two_stage_differential_dct_hierarchical_yuv() {
    // Three-component (YUV-class) two-stage DCT progression.
    //
    // Stage 1 (non-differential SOF0): one 8×8 flat block per component →
    //   Y = 100, Cb = 120, Cr = 60. EXP ×2 both axes → flat 16×16 per
    //   component.
    // Stage 2 (differential SOF5): 16×16 = 2×2 blocks per component, each a
    //   flat per-block correction. Final per-component pixel = base + d,
    //   added modulo 2^16 then folded into 0..256 (§J.2.1).
    //
    // The output must be planar Yuv444P (3 full-resolution planes) carrying
    // the reconstructed Y/Cb/Cr samples verbatim — no colour conversion.
    let base = [100i32, 120, 60];
    // Raster block order per component: [TL, TR, BL, BR].
    let corr: [[i32; 4]; 3] = [[10, -5, 20, -30], [-8, 12, -4, 16], [5, -15, 25, -10]];

    let mut out = Vec::new();
    push_marker(&mut out, 0xD8); // SOI
    push_seg(&mut out, 0xDE, &frame_body_3comp(16, 16)); // DHP (16×16, Nf=3)
    push_seg(&mut out, 0xDB, &dqt_body()); // DQT (all 1s, table 0)
    push_seg(&mut out, 0xC4, &dht_body(0, 0, &DC_BITS, &DC_VALS)); // DC DHT
    push_seg(&mut out, 0xC4, &dht_body(1, 0, &AC_BITS, &AC_VALS)); // AC DHT

    // Stage 1: non-differential SOF0, low-res 8×8, one block per component.
    push_seg(&mut out, 0xC0, &frame_body_3comp(8, 8));
    push_seg(&mut out, 0xDA, &sos_body_3comp());
    let stage1: [Vec<i32>; 3] = [vec![base[0]], vec![base[1]], vec![base[2]]];
    out.extend_from_slice(&encode_flat_dct_scan_3comp(&stage1, false));

    // EXP ×2 both axes.
    push_seg(&mut out, 0xDF, &[0x11]);

    // Stage 2: differential SOF5, full-res 16×16 (2×2 blocks per component).
    push_seg(&mut out, 0xC5, &frame_body_3comp(16, 16));
    push_seg(&mut out, 0xDA, &sos_body_3comp());
    let stage2: [Vec<i32>; 3] = [corr[0].to_vec(), corr[1].to_vec(), corr[2].to_vec()];
    out.extend_from_slice(&encode_flat_dct_scan_3comp(&stage2, true));

    push_marker(&mut out, 0xD9); // EOI

    let frame = decode(out, 16, 16);
    assert_eq!(frame.planes.len(), 3, "expected planar Yuv444P (3 planes)");
    for c in 0..3 {
        let stride = frame.planes[c].stride;
        assert_eq!(stride, 16, "plane {c} stride");
        let data = &frame.planes[c].data;
        let block_corr = |bx: usize, by: usize| corr[c][by * 2 + bx];
        for y in 0..16usize {
            for x in 0..16usize {
                let bx = x / 8;
                let by = y / 8;
                let expect = (base[c] + block_corr(bx, by)).clamp(0, 255) as u8;
                assert_eq!(
                    data[y * stride + x],
                    expect,
                    "plane {c} pixel ({x},{y}) block ({bx},{by})"
                );
            }
        }
    }
}

#[test]
fn two_stage_differential_dct_hierarchical_cmyk() {
    // Four-component (CMYK-class, Adobe transform = 0) two-stage DCT
    // progression. Each component accumulates modulo 2^16 (§J.2.1) exactly
    // like the 1/3-component cases; the EOI shaping packs the four planes
    // into packed Cmyk and un-inverts the Adobe-CMYK convention
    // (output = 255 − reconstructed sample).
    let base = [80i32, 140, 40, 200];
    // Raster block order per component: [TL, TR, BL, BR].
    let corr: [[i32; 4]; 4] = [
        [10, -5, 20, -30],
        [-8, 12, -4, 16],
        [5, -15, 25, -10],
        [-12, 6, -18, 9],
    ];

    let mut out = Vec::new();
    push_marker(&mut out, 0xD8); // SOI
    push_seg(&mut out, 0xDE, &frame_body_4comp(16, 16)); // DHP (16×16, Nf=4)
    push_seg(&mut out, 0xEE, &adobe_app14_body(0)); // APP14 Adobe transform=0
    push_seg(&mut out, 0xDB, &dqt_body()); // DQT (all 1s)
    push_seg(&mut out, 0xC4, &dht_body(0, 0, &DC_BITS, &DC_VALS)); // DC DHT
    push_seg(&mut out, 0xC4, &dht_body(1, 0, &AC_BITS, &AC_VALS)); // AC DHT

    // Stage 1: non-differential SOF0, low-res 8×8, one block per component.
    push_seg(&mut out, 0xC0, &frame_body_4comp(8, 8));
    push_seg(&mut out, 0xDA, &sos_body_4comp());
    let stage1: [Vec<i32>; 4] = [vec![base[0]], vec![base[1]], vec![base[2]], vec![base[3]]];
    out.extend_from_slice(&encode_flat_dct_scan_4comp(&stage1, false));

    // EXP ×2 both axes.
    push_seg(&mut out, 0xDF, &[0x11]);

    // Stage 2: differential SOF5, full-res 16×16 (2×2 blocks per component).
    push_seg(&mut out, 0xC5, &frame_body_4comp(16, 16));
    push_seg(&mut out, 0xDA, &sos_body_4comp());
    let stage2: [Vec<i32>; 4] = [
        corr[0].to_vec(),
        corr[1].to_vec(),
        corr[2].to_vec(),
        corr[3].to_vec(),
    ];
    out.extend_from_slice(&encode_flat_dct_scan_4comp(&stage2, true));

    push_marker(&mut out, 0xD9); // EOI

    let frame = decode(out, 16, 16);
    assert_eq!(frame.planes.len(), 1, "expected one packed Cmyk plane");
    let stride = frame.planes[0].stride;
    assert_eq!(stride, 16 * 4);
    let data = &frame.planes[0].data;
    for y in 0..16usize {
        for x in 0..16usize {
            let bx = x / 8;
            let by = y / 8;
            for c in 0..4 {
                let recon = (base[c] + corr[c][by * 2 + bx]).clamp(0, 255);
                // Adobe transform = 0 → output = 255 − reconstructed sample.
                let expect = (255 - recon) as u8;
                assert_eq!(
                    data[y * stride + x * 4 + c],
                    expect,
                    "CMYK channel {c} pixel ({x},{y}) block ({bx},{by})"
                );
            }
        }
    }
}
