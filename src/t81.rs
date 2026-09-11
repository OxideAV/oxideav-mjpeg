//! General ITU-T T.81 | ISO/IEC 10918-1 JPEG datastream **writer** — the
//! one encoder surface sibling crates build on (`oxideav-tiff`'s
//! `Compression = 7` writer re-points here).
//!
//! Where the historical `encoder::encode_jpeg_*` entry points each fix
//! one layout, this module codes an arbitrary frame description:
//!
//! * **Sequential DCT** (`SOF0` baseline at `P = 8` with table
//!   destinations 0/1 and 8-bit `Qk`; `SOF1` extended sequential
//!   otherwise, including `P = 12`), T.81 Annex F.1.
//! * **Progressive DCT** (`SOF2`, spectral selection: one interleaved
//!   DC scan, then per component the AC bands `1..=5` and `6..=63`),
//!   T.81 Annex G.1 — `P = 8` or `12`.
//! * **Lossless** (`SOF3`, Table H.1 predictors 1..=7, point transform),
//!   T.81 Annex H.1 — `P ∈ 2..=16`.
//!
//! with 1..=4 components, per-component sampling factors `Hi, Vi ∈
//! 1..=4` (any §A.1.1 combination under the §B.2.3 `Σ Hi × Vi ≤ 10`
//! interleave bound — 4×2 luma included), per-component quantisation /
//! entropy table destinations, restart intervals on every process
//! (§B.2.4.4 / §E.1.3), Annex K.3 "typical" **or** Annex K.2 *optimal*
//! Huffman tables derived from the frame's own symbol statistics, and
//! the §B.5 abbreviated formats (a tables-only stream plus table-less
//! frame streams, the `JPEGTables` carriage of TIFF Technical Note 2).
//!
//! Clause map (all from the staged `docs/image/jpeg/T-REC-T.81-199209-I.pdf`):
//! A.1.1 (component dimensions), A.2.2 / A.2.3 (non-interleaved and
//! interleaved data-unit order), A.2.4 (partial-MCU completion by edge
//! replication), A.3.1 (level shift `2^(P−1)`), A.3.3 / A.3.4 (FDCT +
//! uniform quantiser), A.3.6 (zig-zag), B.1.1.5 (byte stuffing),
//! B.2.2 / B.2.3 (frame / scan headers, Tables B.2 / B.3), B.2.4.1 /
//! B.2.4.2 / B.2.4.4 (DQT / DHT / DRI), B.5 (abbreviated formats),
//! Annex C (code tables from `BITS` / `HUFFVAL`, §C.2 all-ones reserve),
//! E.1.3 / E.1.4 (restart interval control), F.1.1.5.1 (DC prediction
//! reset), F.1.2.1 / F.1.2.2 (DC / AC symbol coding, ZRL, EOB),
//! G.1.2 (progressive DC / AC-first scans), H.1.2.1 / H.1.2.2 (lossless
//! prediction incl. the first-line / restart rule, Table H.2 categories),
//! K.1 (Tables K.1 / K.2), K.2 (Figures K.1–K.4), K.3.3 (Tables K.3–K.6).
//! The numeric tables match the staged CSV transcriptions under
//! `docs/image/jpeg/tables/`.
//!
//! ```
//! use oxideav_mjpeg::t81::{HuffmanTables, JpegEncodeOptions, JpegProcess};
//!
//! // 12-bit extended-sequential 4:2:0 with optimal tables and restarts.
//! let (w, h) = (16u32, 8u32);
//! let y: Vec<u16> = (0..w * h).map(|i| (i * 37 % 4096) as u16).collect();
//! let c: Vec<u16> = vec![2048; 8 * 4];
//! let opts = JpegEncodeOptions {
//!     precision: 12,
//!     tables: HuffmanTables::Optimal,
//!     sampling: vec![(2, 2), (1, 1), (1, 1)],
//!     restart_interval: 2,
//!     ..JpegEncodeOptions::default()
//! };
//! let out = opts.encode(w, h, &[&y, &c, &c]).unwrap();
//! assert!(out.data.windows(2).any(|m| m == [0xFF, 0xC1])); // SOF1
//! # let _ = JpegProcess::Sequential;
//! ```

use crate::error::{MjpegError as Error, Result};
use crate::jpeg::dct::fdct8x8;
use crate::jpeg::markers;
use crate::jpeg::zigzag::ZIGZAG;

// ---------------------------------------------------------------------------
// Annex K.1 quantisation tables (natural row-major order).
// ---------------------------------------------------------------------------

/// T.81 Table K.1 — luminance quantisation table (natural order).
#[rustfmt::skip]
pub const QUANT_LUMINANCE_K1: [u16; 64] = [
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
];

/// T.81 Table K.2 — chrominance quantisation table (natural order).
#[rustfmt::skip]
pub const QUANT_CHROMINANCE_K2: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

// ---------------------------------------------------------------------------
// Annex K.3.3 "typical" Huffman table specifications (BITS / HUFFVAL).
// ---------------------------------------------------------------------------

/// Table K.3 (luminance DC) — K.3.3.1 `BITS`.
const BITS_DC_LUMINANCE: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
/// Table K.3 (luminance DC) — K.3.3.1 `HUFFVAL`.
const VAL_DC_LUMINANCE: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
/// Table K.4 (chrominance DC) — K.3.3.1 `BITS`.
const BITS_DC_CHROMINANCE: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
/// Table K.4 (chrominance DC) — K.3.3.1 `HUFFVAL`.
const VAL_DC_CHROMINANCE: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
/// Table K.5 (luminance AC) — K.3.3.2 `BITS`.
const BITS_AC_LUMINANCE: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
/// Table K.5 (luminance AC) — K.3.3.2 `HUFFVAL`.
#[rustfmt::skip]
const VAL_AC_LUMINANCE: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];
/// Table K.6 (chrominance AC) — K.3.3.2 `BITS`.
const BITS_AC_CHROMINANCE: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
/// Table K.6 (chrominance AC) — K.3.3.2 `HUFFVAL`.
#[rustfmt::skip]
const VAL_AC_CHROMINANCE: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];
/// Lossless DC-difference table covering every Table H.2 category
/// `SSSS ∈ 0..=16`: 14 four-bit codes + 3 five-bit codes (Kraft sum
/// 31/32, longest code `11110` — §C.2 reserves the all-ones word).
const BITS_DC_LOSSLESS: [u8; 16] = [0, 0, 0, 14, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const VAL_DC_LOSSLESS: [u8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

// ---------------------------------------------------------------------------
// Huffman table specification (B.2.4.2 BITS + HUFFVAL), Annex C encoder
// code tables, and the K.2 optimal-table procedure.
// ---------------------------------------------------------------------------

/// A Huffman table in its B.2.4.2 specification form: `bits[i]` is the
/// number of codes of length `i + 1` (`L1..L16`), `vals` the symbol
/// values in code order (`HUFFVAL`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuffSpec {
    pub bits: [u8; 16],
    pub vals: Vec<u8>,
}

impl HuffSpec {
    fn new(bits: [u8; 16], vals: &[u8]) -> Self {
        HuffSpec {
            bits,
            vals: vals.to_vec(),
        }
    }

    /// Table K.3 — luminance DC differences.
    pub fn k3_dc_luminance() -> Self {
        Self::new(BITS_DC_LUMINANCE, &VAL_DC_LUMINANCE)
    }
    /// Table K.4 — chrominance DC differences.
    pub fn k4_dc_chrominance() -> Self {
        Self::new(BITS_DC_CHROMINANCE, &VAL_DC_CHROMINANCE)
    }
    /// Table K.5 — luminance AC coefficients.
    pub fn k5_ac_luminance() -> Self {
        Self::new(BITS_AC_LUMINANCE, &VAL_AC_LUMINANCE)
    }
    /// Table K.6 — chrominance AC coefficients.
    pub fn k6_ac_chrominance() -> Self {
        Self::new(BITS_AC_CHROMINANCE, &VAL_AC_CHROMINANCE)
    }
    /// A DC-class table covering every lossless difference category
    /// `SSSS ∈ 0..=16` (Table H.2) — valid at every precision.
    pub fn lossless_dc() -> Self {
        Self::new(BITS_DC_LOSSLESS, &VAL_DC_LOSSLESS)
    }

    /// Byte length of this table inside a DHT segment: `17 + m_t`
    /// (Table B.5).
    fn dht_len(&self) -> usize {
        17 + self.vals.len()
    }

    /// Check the B.2.4.2 / Annex C constraints: `Σ BITS = |HUFFVAL| ≤
    /// 256`, a prefix-free (not over-subscribed) code, and the all-ones
    /// code word of every length left unused (§C.2).
    pub fn validate(&self) -> Result<()> {
        let total: usize = self.bits.iter().map(|&b| b as usize).sum();
        if total != self.vals.len() || total > 256 {
            return Err(Error::invalid(format!(
                "JPEG encode: Huffman table BITS sum {total} does not match {} HUFFVAL entries",
                self.vals.len()
            )));
        }
        // Kraft accounting over 2^16: the code space must not be
        // over-subscribed, and it must not be completely filled either
        // (a full code space means the last canonical code is all ones).
        let mut used: u32 = 0;
        for (l, &n) in self.bits.iter().enumerate() {
            used += u32::from(n) << (15 - l);
        }
        if used > 1 << 16 {
            return Err(Error::invalid(
                "JPEG encode: Huffman BITS list over-subscribes the code space",
            ));
        }
        if total > 0 && used == 1 << 16 {
            return Err(Error::invalid(
                "JPEG encode: Huffman BITS list fills the code space (T.81 §C.2 reserves the all-ones code word)",
            ));
        }
        Ok(())
    }
}

/// Encoder code tables `EHUFCO` / `EHUFSI` (Annex C, Figure C.3),
/// indexed by symbol value.
#[derive(Debug, Clone)]
struct HuffCodes {
    code: [u16; 256],
    size: [u8; 256],
}

impl HuffCodes {
    /// Annex C.2: Figures C.1 (HUFFSIZE), C.2 (HUFFCODE), C.3 (order by
    /// symbol value).
    fn from_spec(spec: &HuffSpec) -> Result<Self> {
        spec.validate()?;
        // Figure C.1 — Generate_size_table.
        let mut huffsize: Vec<u8> = Vec::with_capacity(spec.vals.len());
        for (i, &count) in spec.bits.iter().enumerate() {
            for _ in 0..count {
                huffsize.push(i as u8 + 1);
            }
        }
        // Figure C.2 — Generate_code_table.
        let mut huffcode: Vec<u16> = Vec::with_capacity(huffsize.len());
        let mut code: u32 = 0;
        let mut si = huffsize.first().copied().unwrap_or(0);
        for &size in &huffsize {
            while size != si {
                code <<= 1;
                si += 1;
            }
            huffcode.push(code as u16);
            code += 1;
        }
        // Figure C.3 — Order_codes.
        let mut out = HuffCodes {
            code: [0; 256],
            size: [0; 256],
        };
        for (k, &v) in spec.vals.iter().enumerate() {
            if out.size[v as usize] != 0 {
                return Err(Error::invalid(format!(
                    "JPEG encode: Huffman HUFFVAL lists symbol {v:#04x} twice"
                )));
            }
            out.code[v as usize] = huffcode[k];
            out.size[v as usize] = huffsize[k];
        }
        Ok(out)
    }
}

/// Symbol frequency counter for one Huffman table destination (K.2:
/// `FREQ(V)` for `V = 0..=255`; `FREQ(256)` is the reserved code point
/// that guarantees no all-ones code word).
#[derive(Debug, Clone)]
pub struct HuffStats {
    freq: [u32; 257],
}

impl Default for HuffStats {
    fn default() -> Self {
        HuffStats { freq: [0; 257] }
    }
}

/// K.2 tie-break helper: `v` displaces the current candidate when its
/// frequency is less than or equal (so the largest index wins ties).
fn least_so_far(best: Option<usize>, freq: &[u64; 257], v: usize) -> bool {
    match best {
        None => true,
        Some(b) => freq[v] <= freq[b],
    }
}

impl HuffStats {
    /// Count one occurrence of `symbol`.
    pub fn count(&mut self, symbol: u8) {
        self.freq[symbol as usize] = self.freq[symbol as usize].saturating_add(1);
    }

    /// Merge another counter into this one (statistics gathered over
    /// several frames sharing one table set).
    pub fn merge(&mut self, other: &HuffStats) {
        for (d, &s) in self.freq.iter_mut().zip(other.freq.iter()) {
            *d = d.saturating_add(s);
        }
    }

    /// True when no symbol was ever counted.
    pub fn is_empty(&self) -> bool {
        self.freq[..256].iter().all(|&f| f == 0)
    }

    /// K.2 Figures K.1–K.4: derive a table specification whose code
    /// lengths are optimal for the collected statistics, limited to
    /// 16 bits with the Figure K.3 adjustment.
    pub fn to_spec(&self) -> HuffSpec {
        let mut freq: [u64; 257] = [0; 257];
        for (d, &s) in freq.iter_mut().zip(self.freq.iter()) {
            *d = s as u64;
        }
        // Reserve one code point (K.2: "FREQ value for V = 256 is set to 1").
        freq[256] = 1;
        let mut codesize: [u32; 257] = [0; 257];
        let mut others: [i32; 257] = [-1; 257];

        // Figure K.1 — Code_size. "Find V1 for least value of FREQ(V1) > 0"
        // selects the largest V on ties; V2 is the next least, likewise.
        loop {
            let mut v1: Option<usize> = None;
            for v in 0..257 {
                if freq[v] > 0 && least_so_far(v1, &freq, v) {
                    v1 = Some(v);
                }
            }
            let Some(v1) = v1 else { break };
            let mut v2: Option<usize> = None;
            for v in 0..257 {
                if v != v1 && freq[v] > 0 && least_so_far(v2, &freq, v) {
                    v2 = Some(v);
                }
            }
            let Some(v2) = v2 else { break };
            freq[v1] += freq[v2];
            freq[v2] = 0;
            let mut a = v1;
            loop {
                codesize[a] += 1;
                if others[a] == -1 {
                    break;
                }
                a = others[a] as usize;
            }
            others[a] = v2 as i32;
            let mut b = v2;
            loop {
                codesize[b] += 1;
                if others[b] == -1 {
                    break;
                }
                b = others[b] as usize;
            }
        }

        // Figure K.2 — Count_BITS (lengths up to 32 before adjustment).
        let mut bits: [i32; 33] = [0; 33];
        for &cs in codesize.iter() {
            if cs != 0 {
                bits[cs.min(32) as usize] += 1;
            }
        }

        // Figure K.3 — Adjust_BITS: no code longer than 16 bits, then
        // drop the reserved code point from the longest length.
        let mut i = 32usize;
        while i > 16 {
            if bits[i] > 0 {
                // J starts at I − 1 and is decremented *before* each
                // BITS(J) > 0 test, so the search begins at I − 2.
                let mut j = i - 1;
                loop {
                    j -= 1;
                    if bits[j] > 0 {
                        break;
                    }
                }
                bits[i] -= 2;
                bits[i - 1] += 1;
                bits[j + 1] += 2;
                bits[j] -= 1;
            } else {
                i -= 1;
            }
        }
        while bits[i] == 0 {
            i -= 1;
        }
        bits[i] -= 1;

        // Figure K.4 — Sort_input (symbols 0..=255 by code size).
        let mut vals: Vec<u8> = Vec::new();
        for size in 1..=32u32 {
            for (j, &cs) in codesize.iter().enumerate().take(256) {
                if cs == size {
                    vals.push(j as u8);
                }
            }
        }
        let mut out_bits = [0u8; 16];
        for (k, slot) in out_bits.iter_mut().enumerate() {
            *slot = bits[k + 1].max(0) as u8;
        }
        HuffSpec {
            bits: out_bits,
            vals,
        }
    }
}

// ---------------------------------------------------------------------------
// Quantisation tables and the quality knob.
// ---------------------------------------------------------------------------

/// Scale a K.1 / K.2 table by the `quality` knob (1..=100).
///
/// Anchors follow the Annex K.1 guidance: `quality = 50` leaves the
/// printed table unchanged, `quality = 75` halves every step ("If these
/// quantization values are divided by 2, the resulting reconstructed
/// image is usually nearly indistinguishable from the source image"),
/// and `quality = 100` collapses every step to 1. The mapping is
/// `scale = 50 / q` below 50 and `scale = (100 − q) / 50` at or above
/// 50, applied as `max(1, round(Qk × scale))`. For 12-bit sample
/// precision every step is additionally multiplied by 16 so the
/// quantiser keeps the same *relative* coarseness the tables were
/// designed for on 8-bit data (12-bit coefficients carry four more
/// bits); the resulting entries exceed 255 and are written with
/// `Pq = 1` (16-bit `Qk`, Table B.4), which the extended and
/// progressive processes permit.
pub fn scaled_quant_table(base: &[u16; 64], quality: u8, precision: u8) -> [u16; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale_pct: u32 = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [0u16; 64];
    let max = if precision > 8 { 65535u32 } else { 255u32 };
    let mul = if precision > 8 { 16u32 } else { 1u32 };
    for (o, &b) in out.iter_mut().zip(base.iter()) {
        let v = ((b as u32 * scale_pct + 50) / 100).max(1) * mul;
        *o = v.min(max) as u16;
    }
    out
}

// ---------------------------------------------------------------------------
// Table set.
// ---------------------------------------------------------------------------

/// The quantisation + Huffman table destinations a frame references.
/// Slot `i` of each array is destination `i` (`Tq` / `Th` = i); unused
/// slots stay `None` and are neither written nor referenced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JpegTableSet {
    /// Quantisation tables, natural (row-major) order.
    pub quant: [Option<[u16; 64]>; 4],
    /// DC (or lossless) Huffman tables — `Tc = 0`.
    pub dc: [Option<HuffSpec>; 4],
    /// AC Huffman tables — `Tc = 1`.
    pub ac: [Option<HuffSpec>; 4],
}

impl JpegTableSet {
    /// The Annex K "typical" set: destination 0 = Table K.1 luminance
    /// quantiser + Tables K.3 / K.5, destination 1 (only when
    /// `components > 1`) = Table K.2 chrominance quantiser + Tables K.4 /
    /// K.6, every quantiser scaled per [`scaled_quant_table`]. For the
    /// lossless process the DC slots carry [`HuffSpec::lossless_dc`] and
    /// no quantiser / AC table is populated.
    pub fn typical(quality: u8, precision: u8, lossless: bool, components: usize) -> Self {
        let mut t = JpegTableSet::default();
        let slots = if components > 1 { 2 } else { 1 };
        for slot in 0..slots {
            if lossless {
                t.dc[slot] = Some(HuffSpec::lossless_dc());
            } else {
                let (q, dc, ac) = if slot == 0 {
                    (
                        &QUANT_LUMINANCE_K1,
                        HuffSpec::k3_dc_luminance(),
                        HuffSpec::k5_ac_luminance(),
                    )
                } else {
                    (
                        &QUANT_CHROMINANCE_K2,
                        HuffSpec::k4_dc_chrominance(),
                        HuffSpec::k6_ac_chrominance(),
                    )
                };
                t.quant[slot] = Some(scaled_quant_table(q, quality, precision));
                t.dc[slot] = Some(dc);
                t.ac[slot] = Some(ac);
            }
        }
        t
    }

    /// Replace the Huffman tables with K.2 optimal ones derived from
    /// `frame`'s own symbol statistics (one pass over the entropy coder
    /// in counting mode). Destinations no component references are
    /// left untouched; a referenced destination whose statistics are
    /// empty keeps its previous table.
    pub fn optimise_huffman(
        &mut self,
        frame: &JpegFrame,
        comps: &[JpegComponent<'_>],
    ) -> Result<()> {
        let mut dc: [HuffStats; 4] = Default::default();
        let mut ac: [HuffStats; 4] = Default::default();
        gather_stats(frame, comps, self, &mut dc, &mut ac)?;
        for t in 0..4 {
            if !dc[t].is_empty() {
                self.dc[t] = Some(dc[t].to_spec());
            }
            if !ac[t].is_empty() {
                self.ac[t] = Some(ac[t].to_spec());
            }
        }
        Ok(())
    }

    /// Serialise the DQT / DHT marker segments for every populated
    /// slot (B.2.4.1 / B.2.4.2). Quantisation tables go first, one
    /// table per DQT segment; then the DC tables, then the AC tables,
    /// one table per DHT segment. `dct = false` (lossless) suppresses
    /// the quantisation tables and the AC tables, neither of which the
    /// lossless process references.
    pub fn write_tables(&self, out: &mut Vec<u8>, dct: bool) {
        if dct {
            for (tq, table) in self.quant.iter().enumerate() {
                if let Some(q) = table {
                    let pq: u8 = if q.iter().any(|&v| v > 255) { 1 } else { 0 };
                    let len = 2 + 1 + 64 * (1 + pq as usize);
                    out.extend_from_slice(&[0xFF, markers::DQT]);
                    out.extend_from_slice(&(len as u16).to_be_bytes());
                    out.push((pq << 4) | tq as u8);
                    for k in 0..64 {
                        let v = q[ZIGZAG[k]];
                        if pq == 1 {
                            out.extend_from_slice(&v.to_be_bytes());
                        } else {
                            out.push(v as u8);
                        }
                    }
                }
            }
        }
        for (class, tables) in [(0u8, &self.dc), (1u8, &self.ac)] {
            if class == 1 && !dct {
                continue;
            }
            for (th, table) in tables.iter().enumerate() {
                if let Some(h) = table {
                    let len = 2 + h.dht_len();
                    out.extend_from_slice(&[0xFF, markers::DHT]);
                    out.extend_from_slice(&(len as u16).to_be_bytes());
                    out.push((class << 4) | th as u8);
                    out.extend_from_slice(&h.bits);
                    out.extend_from_slice(&h.vals);
                }
            }
        }
    }

    /// A complete §B.5 "abbreviated format for table-specification
    /// data" stream: `SOI`, the table segments, `EOI` — the payload of a
    /// TIFF `JPEGTables` field.
    pub fn tables_stream(&self, dct: bool) -> Vec<u8> {
        let mut out = vec![0xFF, markers::SOI];
        self.write_tables(&mut out, dct);
        out.extend_from_slice(&[0xFF, markers::EOI]);
        out
    }
}

// ---------------------------------------------------------------------------
// Frame description.
// ---------------------------------------------------------------------------

/// One frame component: its samples at the component's own resolution
/// (A.1.1: `xi = ceil(X × Hi / Hmax)`, `yi = ceil(Y × Vi / Vmax)`), its
/// identifier, the sampling factors, and the table destinations it
/// selects.
#[derive(Debug, Clone)]
pub struct JpegComponent<'a> {
    /// Component identifier `Ci` (B.2.2) — distinct per component.
    pub id: u8,
    /// Row-major samples, `width × height`, each `< 2^precision`.
    pub samples: &'a [u16],
    pub width: usize,
    pub height: usize,
    /// Horizontal sampling factor `Hi` (1..=4).
    pub h: u8,
    /// Vertical sampling factor `Vi` (1..=4).
    pub v: u8,
    /// Quantisation table destination `Tqi` (DCT processes).
    pub quant_id: u8,
    /// Entropy table destination (`Tdj` and `Taj` — the same slot is
    /// used for both classes).
    pub huff_id: u8,
}

/// The coding process of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegProcess {
    /// Sequential DCT, Huffman: `SOF0` (baseline) when `P = 8`, every
    /// table destination is 0 or 1 and every quantiser fits 8 bits;
    /// `SOF1` (extended sequential) otherwise.
    Sequential,
    /// Progressive DCT, Huffman (`SOF2`): one interleaved DC scan, then
    /// per component the AC bands `1..=5` and `6..=63` (spectral
    /// selection, `Ah = Al = 0`).
    Progressive,
    /// Lossless, Huffman (`SOF3`) with the Table H.1 predictor
    /// selection value (1..=7) and the point transform `Pt`
    /// (`0..precision`).
    Lossless { predictor: u8, point_transform: u8 },
}

impl JpegProcess {
    /// True for the two DCT-based processes.
    pub fn is_dct(self) -> bool {
        !matches!(self, JpegProcess::Lossless { .. })
    }
}

/// Frame-level parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegFrame {
    /// Frame width `X` (samples per line of the highest-resolution
    /// component).
    pub width: u16,
    /// Frame height `Y`.
    pub height: u16,
    /// Sample precision `P`: 8 or 12 for the DCT processes, 2..=16 for
    /// [`JpegProcess::Lossless`].
    pub precision: u8,
    pub process: JpegProcess,
    /// Restart interval `Ri` in MCUs (B.2.4.4); 0 disables restarts.
    /// When enabled a `DRI` segment precedes the first scan and an
    /// `RSTm` marker (`m` cycling 0..=7, restarting at 0 for every
    /// scan) terminates every interval but the last (E.1.3 / E.1.4:
    /// the entropy-coded segment is padded with 1-bits, the DC
    /// predictions — or the lossless predictor state — are reset). For
    /// the lossless process Table B.7 requires `Ri` to be an integer
    /// multiple of the number of MCUs in an MCU-row.
    pub restart_interval: u16,
}

// ---------------------------------------------------------------------------
// Entropy-coded segment sink: either emits bits or counts symbols.
// ---------------------------------------------------------------------------

/// Bit writer implementing C.3 (MSB-first) and F.1.2.3 / B.1.1.5 (byte
/// stuffing, 1-bit padding).
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }

    fn put(&mut self, code: u32, size: u32) {
        if size == 0 {
            return;
        }
        debug_assert!(size <= 24);
        self.acc = (self.acc << size) | (code & ((1u32 << size) - 1));
        self.nbits += size;
        while self.nbits >= 8 {
            let byte = ((self.acc >> (self.nbits - 8)) & 0xFF) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0x00);
            }
            self.nbits -= 8;
        }
        self.acc &= (1u32 << self.nbits).wrapping_sub(1);
    }

    /// F.1.2.3: pad the current byte with 1-bits (stuffing a zero
    /// after an `X'FF'` produced by the padding) — E.1.4
    /// "Prepare_for_marker".
    fn pad_to_byte(&mut self) {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.put((1u32 << pad) - 1, pad);
        }
    }

    /// Terminate the current restart interval with `RSTm` (E.1.3).
    fn restart(&mut self, m: u8) {
        self.pad_to_byte();
        self.out.extend_from_slice(&[0xFF, markers::RST0 + (m & 7)]);
    }

    fn finish(mut self) -> Vec<u8> {
        self.pad_to_byte();
        self.out
    }
}

/// Per-scan entropy tables (up to four destinations of each class).
struct ScanTables {
    dc: [Option<HuffCodes>; 4],
    ac: [Option<HuffCodes>; 4],
}

enum Sink<'a> {
    Emit {
        writer: &'a mut BitWriter,
        tables: &'a ScanTables,
    },
    Count {
        dc: &'a mut [HuffStats; 4],
        ac: &'a mut [HuffStats; 4],
    },
}

impl Sink<'_> {
    /// Code one DC-class symbol (`SSSS` category for DC differences and
    /// lossless differences) followed by `extra_bits` additional bits.
    fn dc_symbol(&mut self, table: u8, symbol: u8, extra: u32, extra_bits: u32) -> Result<()> {
        match self {
            Sink::Emit { writer, tables } => {
                let codes = tables.dc[table as usize].as_ref().ok_or_else(|| {
                    Error::invalid(format!("JPEG encode: DC table {table} not defined"))
                })?;
                let size = codes.size[symbol as usize];
                if size == 0 {
                    return Err(Error::invalid(format!(
                        "JPEG encode: DC table {table} has no code for category {symbol}"
                    )));
                }
                writer.put(codes.code[symbol as usize] as u32, size as u32);
                writer.put(extra, extra_bits);
            }
            Sink::Count { dc, .. } => dc[table as usize].count(symbol),
        }
        Ok(())
    }

    /// Code one AC-class composite symbol (`RRRRSSSS`) plus additional
    /// bits.
    fn ac_symbol(&mut self, table: u8, symbol: u8, extra: u32, extra_bits: u32) -> Result<()> {
        match self {
            Sink::Emit { writer, tables } => {
                let codes = tables.ac[table as usize].as_ref().ok_or_else(|| {
                    Error::invalid(format!("JPEG encode: AC table {table} not defined"))
                })?;
                let size = codes.size[symbol as usize];
                if size == 0 {
                    return Err(Error::invalid(format!(
                        "JPEG encode: AC table {table} has no code for run/size {symbol:#04x}"
                    )));
                }
                writer.put(codes.code[symbol as usize] as u32, size as u32);
                writer.put(extra, extra_bits);
            }
            Sink::Count { ac, .. } => ac[table as usize].count(symbol),
        }
        Ok(())
    }

    /// E.1.3: end the current restart interval with `RSTm` (the
    /// statistics sink has nothing to emit).
    fn restart(&mut self, m: u8) {
        if let Sink::Emit { writer, .. } = self {
            writer.restart(m);
        }
    }
}

// ---------------------------------------------------------------------------
// Magnitude categories and additional bits (F.1.2.1.1 / F.1.2.2.1).
// ---------------------------------------------------------------------------

/// `SSSS` for a two's-complement value: the number of bits needed for
/// its magnitude (Tables F.1 / F.2 / F.6 / F.7 / H.2).
fn category(v: i32) -> u32 {
    let m = v.unsigned_abs();
    32 - m.leading_zeros()
}

/// The additional bits: "When DIFF is positive, the SSSS low order bits
/// of DIFF are appended. When DIFF is negative, the SSSS low order bits
/// of (DIFF – 1) are appended."
fn extra_bits(v: i32, ssss: u32) -> u32 {
    if ssss == 0 {
        return 0;
    }
    let raw = if v < 0 { v - 1 } else { v };
    (raw as u32) & ((1u32 << ssss) - 1)
}

// ---------------------------------------------------------------------------
// Validation (A.1.1, B.2.2 / B.2.3).
// ---------------------------------------------------------------------------

/// Geometry of the frame's MCU grid.
struct Geometry {
    h_max: usize,
    v_max: usize,
    /// True when the scan is interleaved (`Nf > 1`; A.2.3). A single
    /// component is always coded non-interleaved (A.2.2).
    interleaved: bool,
}

/// Validate the component geometry against the frame per A.1.1 and the
/// interleave rule of B.2.3 (`Σ Hj × Vj ≤ 10` when `Ns > 1`).
fn validate(frame: &JpegFrame, comps: &[JpegComponent<'_>]) -> Result<Geometry> {
    if comps.is_empty() || comps.len() > 4 {
        return Err(Error::invalid(format!(
            "JPEG encode: {} components (a single interleaved scan carries 1..=4)",
            comps.len()
        )));
    }
    if frame.width == 0 || frame.height == 0 {
        return Err(Error::invalid(
            "JPEG encode: frame dimensions must be non-zero (B.2.2: X ≥ 1)",
        ));
    }
    match frame.process {
        JpegProcess::Sequential | JpegProcess::Progressive => {
            if frame.precision != 8 && frame.precision != 12 {
                return Err(Error::invalid(format!(
                    "JPEG encode: DCT sample precision {} (Table B.2 allows 8 or 12)",
                    frame.precision
                )));
            }
        }
        JpegProcess::Lossless {
            predictor,
            point_transform,
        } => {
            if !(2..=16).contains(&frame.precision) {
                return Err(Error::invalid(format!(
                    "JPEG encode: lossless sample precision {} (Table B.2 allows 2..=16)",
                    frame.precision
                )));
            }
            if !(1..=7).contains(&predictor) {
                return Err(Error::invalid(format!(
                    "JPEG encode: lossless predictor selection {predictor} (Table H.1 defines 1..=7 for non-differential frames)"
                )));
            }
            if point_transform >= frame.precision || point_transform > 15 {
                return Err(Error::invalid(format!(
                    "JPEG encode: point transform {point_transform} must be below the precision {} (and ≤ 15, Table B.3)",
                    frame.precision
                )));
            }
        }
    }
    let h_max = comps.iter().map(|c| c.h).max().unwrap_or(1);
    let v_max = comps.iter().map(|c| c.v).max().unwrap_or(1);
    let mut hv_sum = 0u32;
    for (i, c) in comps.iter().enumerate() {
        if !(1..=4).contains(&c.h) || !(1..=4).contains(&c.v) {
            return Err(Error::invalid(format!(
                "JPEG encode: component {i} sampling factors {}x{} (Table B.2 allows 1..=4)",
                c.h, c.v
            )));
        }
        if c.quant_id > 3 || c.huff_id > 3 {
            return Err(Error::invalid(format!(
                "JPEG encode: component {i} table destination out of range (0..=3)"
            )));
        }
        if comps.iter().filter(|o| o.id == c.id).count() != 1 {
            return Err(Error::invalid(format!(
                "JPEG encode: component identifier {} is not unique",
                c.id
            )));
        }
        let want_w = (frame.width as usize * c.h as usize).div_ceil(h_max as usize);
        let want_h = (frame.height as usize * c.v as usize).div_ceil(v_max as usize);
        if c.width != want_w || c.height != want_h {
            return Err(Error::invalid(format!(
                "JPEG encode: component {i} is {}x{} but A.1.1 requires {want_w}x{want_h} (X={} Y={} H={} V={} Hmax={h_max} Vmax={v_max})",
                c.width, c.height, frame.width, frame.height, c.h, c.v
            )));
        }
        if c.samples.len() != c.width * c.height {
            return Err(Error::invalid(format!(
                "JPEG encode: component {i} carries {} samples for {}x{}",
                c.samples.len(),
                c.width,
                c.height
            )));
        }
        let limit = 1u32 << frame.precision;
        if c.samples.iter().any(|&s| (s as u32) >= limit) {
            return Err(Error::invalid(format!(
                "JPEG encode: component {i} has a sample outside 0..2^{}",
                frame.precision
            )));
        }
        hv_sum += c.h as u32 * c.v as u32;
    }
    if comps.len() > 1 && hv_sum > 10 {
        return Err(Error::invalid(format!(
            "JPEG encode: Σ Hj × Vj = {hv_sum} exceeds the B.2.3 interleave limit of 10"
        )));
    }
    if comps.len() == 1 && (comps[0].h != 1 || comps[0].v != 1) {
        return Err(Error::invalid(
            "JPEG encode: a single-component frame must use sampling factors 1x1 (A.2.2 orders its data units independently of H/V)",
        ));
    }
    Ok(Geometry {
        h_max: h_max as usize,
        v_max: v_max as usize,
        interleaved: comps.len() > 1,
    })
}

// ---------------------------------------------------------------------------
// Marker segment writers.
// ---------------------------------------------------------------------------

/// Baseline (`SOF0`) needs `P = 8`, table destinations ≤ 1 and 8-bit
/// quantisers (Tables B.2 / B.3 / B.4); anything else in the sequential
/// process is `SOF1`.
fn sequential_is_baseline(
    frame: &JpegFrame,
    comps: &[JpegComponent<'_>],
    tables: &JpegTableSet,
) -> bool {
    frame.precision == 8
        && comps.iter().all(|c| {
            c.quant_id <= 1
                && c.huff_id <= 1
                && tables.quant[c.quant_id as usize]
                    .map(|q| q.iter().all(|&v| v <= 255))
                    .unwrap_or(true)
        })
}

/// Write the frame header (B.2.2, Figure B.3).
fn write_sof(
    out: &mut Vec<u8>,
    frame: &JpegFrame,
    comps: &[JpegComponent<'_>],
    tables: &JpegTableSet,
) {
    let marker = match frame.process {
        JpegProcess::Sequential if sequential_is_baseline(frame, comps, tables) => markers::SOF0,
        JpegProcess::Sequential => markers::SOF1,
        JpegProcess::Progressive => markers::SOF2,
        JpegProcess::Lossless { .. } => markers::SOF3,
    };
    out.extend_from_slice(&[0xFF, marker]);
    let lf = 8 + 3 * comps.len();
    out.extend_from_slice(&(lf as u16).to_be_bytes());
    out.push(frame.precision);
    out.extend_from_slice(&frame.height.to_be_bytes());
    out.extend_from_slice(&frame.width.to_be_bytes());
    out.push(comps.len() as u8);
    for c in comps {
        out.push(c.id);
        out.push((c.h << 4) | c.v);
        out.push(if frame.process.is_dct() {
            c.quant_id
        } else {
            0
        });
    }
}

/// Write a scan header (B.2.3, Figure B.4) for the listed components.
fn write_sos(
    out: &mut Vec<u8>,
    comps: &[&JpegComponent<'_>],
    dct: bool,
    ss: u8,
    se: u8,
    ah_al: u8,
) {
    out.extend_from_slice(&[0xFF, markers::SOS]);
    let ls = 6 + 2 * comps.len();
    out.extend_from_slice(&(ls as u16).to_be_bytes());
    out.push(comps.len() as u8);
    for c in comps {
        out.push(c.id);
        let ta = if dct { c.huff_id } else { 0 };
        out.push((c.huff_id << 4) | ta);
    }
    out.extend_from_slice(&[ss, se, ah_al]);
}

fn write_dri(out: &mut Vec<u8>, ri: u16) {
    out.extend_from_slice(&[0xFF, markers::DRI, 0, 4]);
    out.extend_from_slice(&ri.to_be_bytes());
}

// ---------------------------------------------------------------------------
// Sample preparation: quantised DCT blocks / padded lossless grids.
// ---------------------------------------------------------------------------

/// One component's quantised coefficient blocks on the MCU-padded block
/// grid (`bx_pad × by_pad`); `bx_true × by_true` is the A.2.2
/// non-interleaved extent `ceil(xi / 8) × ceil(yi / 8)`.
struct DctComponent {
    blocks: Vec<[i32; 64]>,
    bx_pad: usize,
    bx_true: usize,
    by_true: usize,
}

/// A.2.4 + A.3.1 + A.3.3 + A.3.4: level-shift, transform and quantise
/// every block of every component once; the scan coders then only walk
/// the block grid (statistics pass and emission pass see identical
/// coefficients).
fn prepare_dct(
    frame: &JpegFrame,
    comps: &[JpegComponent<'_>],
    geo: &Geometry,
    tables: &JpegTableSet,
) -> Result<Vec<DctComponent>> {
    let level = 1i32 << (frame.precision - 1);
    let mcus_x = (frame.width as usize).div_ceil(8 * geo.h_max);
    let mcus_y = (frame.height as usize).div_ceil(8 * geo.v_max);
    let mut out = Vec::with_capacity(comps.len());
    for c in comps {
        let quant = tables.quant[c.quant_id as usize].ok_or_else(|| {
            Error::invalid(format!(
                "JPEG encode: quantisation table {} not defined",
                c.quant_id
            ))
        })?;
        let inv_q: Vec<f32> = quant.iter().map(|&q| 1.0 / q.max(1) as f32).collect();
        let (bx_pad, by_pad) = if geo.interleaved {
            (mcus_x * c.h as usize, mcus_y * c.v as usize)
        } else {
            (c.width.div_ceil(8), c.height.div_ceil(8))
        };
        let (bx_true, by_true) = (c.width.div_ceil(8), c.height.div_ceil(8));
        let mut blocks = Vec::with_capacity(bx_pad * by_pad);
        let mut block = [0f32; 64];
        for by in 0..by_pad {
            for bx in 0..bx_pad {
                // A.2.4: complete partial blocks / MCUs by replicating the
                // right-most column and the bottom line.
                for y in 0..8 {
                    let sy = (by * 8 + y).min(c.height - 1);
                    let row = &c.samples[sy * c.width..(sy + 1) * c.width];
                    for x in 0..8 {
                        let sx = (bx * 8 + x).min(c.width - 1);
                        block[y * 8 + x] = (row[sx] as i32 - level) as f32;
                    }
                }
                fdct8x8(&mut block);
                let mut q = [0i32; 64];
                for k in 0..64 {
                    // A.3.4: Sq = round(S / Q), half away from zero.
                    q[k] = (block[k] * inv_q[k]).round() as i32;
                }
                blocks.push(q);
            }
        }
        debug_assert_eq!(blocks.len(), bx_pad * by_pad);
        out.push(DctComponent {
            blocks,
            bx_pad,
            bx_true,
            by_true,
        });
    }
    Ok(out)
}

/// One component's point-transformed samples on the MCU-padded sample
/// grid (`w_pad × h_pad`, A.2.4 edge replication).
struct LosslessComponent {
    grid: Vec<i32>,
    w_pad: usize,
    h_pad: usize,
}

fn prepare_lossless(
    comps: &[JpegComponent<'_>],
    geo: &Geometry,
    mcus_x: usize,
    mcus_y: usize,
    pt: u8,
) -> Vec<LosslessComponent> {
    comps
        .iter()
        .map(|c| {
            let (w_pad, h_pad) = if geo.interleaved {
                (mcus_x * c.h as usize, mcus_y * c.v as usize)
            } else {
                (c.width, c.height)
            };
            let mut grid = vec![0i32; w_pad * h_pad];
            for y in 0..h_pad {
                let sy = y.min(c.height - 1);
                let row = &c.samples[sy * c.width..(sy + 1) * c.width];
                for x in 0..w_pad {
                    let sx = x.min(c.width - 1);
                    grid[y * w_pad + x] = (row[sx] >> pt) as i32;
                }
            }
            LosslessComponent { grid, w_pad, h_pad }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Block coders (F.1.2 / G.1.2).
// ---------------------------------------------------------------------------

/// F.1.2.1.3: code the DC difference of one block.
fn code_dc(q: &[i32; 64], pred: &mut i32, table: u8, sink: &mut Sink<'_>) -> Result<()> {
    let dc = q[0];
    let diff = dc - *pred;
    *pred = dc;
    let ssss = category(diff);
    sink.dc_symbol(table, ssss as u8, extra_bits(diff, ssss), ssss)
}

/// F.1.2.2.3 (Figures F.2 / F.3) restricted to the zig-zag band
/// `ss..=se` (G.1.2.2 AC first scans with `Al = 0`; the full `1..=63`
/// band is the sequential case): run/size composites in zig-zag order,
/// `ZRL = X'F0'` for 16 zeros, `EOB = X'00'` when the band ends in
/// zeros.
fn code_ac_band(q: &[i32; 64], ss: usize, se: usize, table: u8, sink: &mut Sink<'_>) -> Result<()> {
    let mut run: u32 = 0;
    for k in ss..=se {
        let coef = q[ZIGZAG[k]];
        if coef == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            sink.ac_symbol(table, 0xF0, 0, 0)?;
            run -= 16;
        }
        let ssss = category(coef);
        let rs = ((run << 4) | ssss) as u8;
        sink.ac_symbol(table, rs, extra_bits(coef, ssss), ssss)?;
        run = 0;
    }
    if run > 0 {
        sink.ac_symbol(table, 0x00, 0, 0)?;
    }
    Ok(())
}

/// Restart-interval bookkeeping shared by every scan coder (E.1.3):
/// after `ri` data units (with more to come) close the interval with
/// `RSTm`, `m` cycling 0..=7 from 0 at the start of the scan.
struct Restarts {
    ri: usize,
    done: usize,
    m: u8,
}

impl Restarts {
    fn new(ri: u16) -> Self {
        Restarts {
            ri: ri as usize,
            done: 0,
            m: 0,
        }
    }

    /// Call before coding the next unit; returns true when an `RSTm`
    /// was emitted (the caller resets its predictions).
    fn before_unit(&mut self, sink: &mut Sink<'_>) -> bool {
        let boundary = self.ri > 0 && self.done > 0 && self.done % self.ri == 0;
        if boundary {
            sink.restart(self.m);
            self.m = (self.m + 1) & 7;
        }
        self.done += 1;
        boundary
    }
}

/// A.2.3 interleaved MCU walk over the padded block grids: sequential
/// scans code DC + AC of every block, progressive DC scans (`dc_only`)
/// code DC only.
fn code_dct_interleaved(
    comps: &[JpegComponent<'_>],
    dct: &[DctComponent],
    mcus_x: usize,
    mcus_y: usize,
    ri: u16,
    dc_only: bool,
    sink: &mut Sink<'_>,
) -> Result<()> {
    let mut pred: Vec<i32> = vec![0; comps.len()];
    let mut restarts = Restarts::new(ri);
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if restarts.before_unit(sink) {
                pred.fill(0);
            }
            for (ci, c) in comps.iter().enumerate() {
                let d = &dct[ci];
                for v in 0..c.v as usize {
                    for h in 0..c.h as usize {
                        let q =
                            &d.blocks[(my * c.v as usize + v) * d.bx_pad + mx * c.h as usize + h];
                        code_dc(q, &mut pred[ci], c.huff_id, sink)?;
                        if !dc_only {
                            code_ac_band(q, 1, 63, c.huff_id, sink)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// A.2.2 non-interleaved walk over one component's true block extent.
fn code_dct_single(
    c: &JpegComponent<'_>,
    d: &DctComponent,
    ri: u16,
    ss: usize,
    se: usize,
    sink: &mut Sink<'_>,
) -> Result<()> {
    let mut pred = 0i32;
    let mut restarts = Restarts::new(ri);
    for by in 0..d.by_true {
        for bx in 0..d.bx_true {
            if restarts.before_unit(sink) {
                pred = 0;
            }
            let q = &d.blocks[by * d.bx_pad + bx];
            if ss == 0 {
                code_dc(q, &mut pred, c.huff_id, sink)?;
            }
            if se >= 1 {
                code_ac_band(q, ss.max(1), se, c.huff_id, sink)?;
            }
        }
    }
    Ok(())
}

/// Annex H lossless scan (interleaved when `Nf > 1`), with the
/// §H.1.2.1 prediction rules: the first sample of the scan and of every
/// restart interval is predicted as `2^(P − Pt − 1)`, the rest of that
/// first line uses `Ra`, later lines use `Rb` at the line start and the
/// selected predictor elsewhere; differences are taken modulo `2^16`
/// and coded per Table H.2 (`SSSS = 16` carries no extra bits).
fn code_lossless(
    frame: &JpegFrame,
    comps: &[JpegComponent<'_>],
    geo: &Geometry,
    predictor: u8,
    pt: u8,
    sink: &mut Sink<'_>,
) -> Result<()> {
    let (mcus_x, mcus_y) = if geo.interleaved {
        (
            (frame.width as usize).div_ceil(geo.h_max),
            (frame.height as usize).div_ceil(geo.v_max),
        )
    } else {
        (comps[0].width, comps[0].height)
    };
    let ri = frame.restart_interval as usize;
    if ri > 0 && ri % mcus_x != 0 {
        return Err(Error::invalid(format!(
            "JPEG encode: lossless restart interval {ri} is not a multiple of the {mcus_x} MCUs per MCU-row (T.81 Table B.7)"
        )));
    }
    let grids = prepare_lossless(comps, geo, mcus_x, mcus_y, pt);
    let initial = 1i32 << (frame.precision - pt - 1);
    let mut restarts = Restarts::new(frame.restart_interval);
    // Grid row on which the current interval began, per component.
    let mut first_row: Vec<usize> = vec![0; comps.len()];
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if restarts.before_unit(sink) {
                for (ci, c) in comps.iter().enumerate() {
                    first_row[ci] = if geo.interleaved {
                        my * c.v as usize
                    } else {
                        my
                    };
                }
            }
            for (ci, c) in comps.iter().enumerate() {
                let g = &grids[ci];
                let (bh, bv) = if geo.interleaved {
                    (c.h as usize, c.v as usize)
                } else {
                    (1, 1)
                };
                for v in 0..bv {
                    for h in 0..bh {
                        let x = mx * bh + h;
                        let y = my * bv + v;
                        let at = |xx: usize, yy: usize| g.grid[yy * g.w_pad + xx];
                        let cur = at(x, y);
                        let px = if y == first_row[ci] {
                            if x == 0 {
                                initial
                            } else {
                                at(x - 1, y)
                            }
                        } else if x == 0 {
                            at(x, y - 1)
                        } else {
                            let ra = at(x - 1, y);
                            let rb = at(x, y - 1);
                            let rc = at(x - 1, y - 1);
                            match predictor {
                                1 => ra,
                                2 => rb,
                                3 => rc,
                                4 => ra + rb - rc,
                                5 => ra + ((rb - rc) >> 1),
                                6 => rb + ((ra - rc) >> 1),
                                _ => (ra + rb) >> 1,
                            }
                        };
                        let diff = ((cur - px) as i64).rem_euclid(65536) as i32;
                        let diff = if diff >= 32768 { diff - 65536 } else { diff };
                        if diff == -32768 {
                            sink.dc_symbol(c.huff_id, 16, 0, 0)?;
                        } else {
                            let ssss = category(diff);
                            sink.dc_symbol(c.huff_id, ssss as u8, extra_bits(diff, ssss), ssss)?;
                        }
                    }
                }
            }
        }
        debug_assert!(grids.iter().all(|g| g.h_pad >= mcus_y));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scan plans.
// ---------------------------------------------------------------------------

/// One scan of the frame: which components, and the spectral band.
struct ScanPlan {
    comps: Vec<usize>,
    ss: u8,
    se: u8,
}

/// The scans of a frame in emission order (G.1.1.1.1: the first scan of
/// every component in a progressive frame is a DC scan; AC scans are
/// non-interleaved).
fn scan_plans(frame: &JpegFrame, comps: &[JpegComponent<'_>]) -> Vec<ScanPlan> {
    let all: Vec<usize> = (0..comps.len()).collect();
    match frame.process {
        JpegProcess::Sequential => vec![ScanPlan {
            comps: all,
            ss: 0,
            se: 63,
        }],
        JpegProcess::Lossless { predictor, .. } => vec![ScanPlan {
            comps: all,
            ss: predictor,
            se: 0,
        }],
        JpegProcess::Progressive => {
            let mut plans = vec![ScanPlan {
                comps: all,
                ss: 0,
                se: 0,
            }];
            for ci in 0..comps.len() {
                plans.push(ScanPlan {
                    comps: vec![ci],
                    ss: 1,
                    se: 5,
                });
                plans.push(ScanPlan {
                    comps: vec![ci],
                    ss: 6,
                    se: 63,
                });
            }
            plans
        }
    }
}

/// Drive the entropy coder over one scan, feeding `sink`.
fn code_scan(
    frame: &JpegFrame,
    comps: &[JpegComponent<'_>],
    geo: &Geometry,
    dct: Option<&[DctComponent]>,
    plan: &ScanPlan,
    sink: &mut Sink<'_>,
) -> Result<()> {
    match frame.process {
        JpegProcess::Lossless {
            predictor,
            point_transform,
        } => code_lossless(frame, comps, geo, predictor, point_transform, sink),
        JpegProcess::Sequential | JpegProcess::Progressive => {
            let dct = dct.expect("DCT blocks prepared");
            let dc_only = plan.se == 0;
            if plan.comps.len() > 1 {
                let mcus_x = (frame.width as usize).div_ceil(8 * geo.h_max);
                let mcus_y = (frame.height as usize).div_ceil(8 * geo.v_max);
                code_dct_interleaved(
                    comps,
                    dct,
                    mcus_x,
                    mcus_y,
                    frame.restart_interval,
                    dc_only,
                    sink,
                )
            } else {
                let ci = plan.comps[0];
                code_dct_single(
                    &comps[ci],
                    &dct[ci],
                    frame.restart_interval,
                    plan.ss as usize,
                    plan.se as usize,
                    sink,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public frame API.
// ---------------------------------------------------------------------------

/// Gather the DC / AC symbol statistics of one frame (K.2 input) into
/// `dc_stats` / `ac_stats` (indexed by table destination), so that
/// optimal tables can be derived — possibly across many frames sharing
/// one table set. `tables` only needs its quantisation tables populated.
pub fn gather_stats(
    frame: &JpegFrame,
    comps: &[JpegComponent<'_>],
    tables: &JpegTableSet,
    dc_stats: &mut [HuffStats; 4],
    ac_stats: &mut [HuffStats; 4],
) -> Result<()> {
    let geo = validate(frame, comps)?;
    let dct = if frame.process.is_dct() {
        Some(prepare_dct(frame, comps, &geo, tables)?)
    } else {
        None
    };
    let mut sink = Sink::Count {
        dc: dc_stats,
        ac: ac_stats,
    };
    for plan in scan_plans(frame, comps) {
        code_scan(frame, comps, &geo, dct.as_deref(), &plan, &mut sink)?;
    }
    Ok(())
}

/// Encode one complete frame as a JPEG datastream: `SOI`, the table
/// segments (only when `emit_tables`), `SOFn`, `DRI` (when a restart
/// interval is set), one or more `SOS` + entropy-coded segments, `EOI`.
/// With `emit_tables = false` the stream is a §B.5 abbreviated image
/// segment that relies on the same [`JpegTableSet`] having been
/// installed from a [`JpegTableSet::tables_stream`] (TIFF `JPEGTables`).
pub fn encode_frame(
    frame: &JpegFrame,
    comps: &[JpegComponent<'_>],
    tables: &JpegTableSet,
    emit_tables: bool,
) -> Result<Vec<u8>> {
    encode_frame_with_meta(frame, comps, tables, emit_tables, &[])
}

/// [`encode_frame`] with caller-supplied marker segments (`APPn` /
/// `COM`, already framed as `FF xx Lp …`) inserted directly after `SOI`.
pub fn encode_frame_with_meta(
    frame: &JpegFrame,
    comps: &[JpegComponent<'_>],
    tables: &JpegTableSet,
    emit_tables: bool,
    meta: &[u8],
) -> Result<Vec<u8>> {
    let geo = validate(frame, comps)?;
    let dct = frame.process.is_dct();
    let mut scan_tables = ScanTables {
        dc: [None, None, None, None],
        ac: [None, None, None, None],
    };
    for c in comps {
        let id = c.huff_id as usize;
        if scan_tables.dc[id].is_none() {
            let spec = tables.dc[id].as_ref().ok_or_else(|| {
                Error::invalid(format!("JPEG encode: DC Huffman table {id} not defined"))
            })?;
            scan_tables.dc[id] = Some(HuffCodes::from_spec(spec)?);
        }
        if dct && scan_tables.ac[id].is_none() {
            let spec = tables.ac[id].as_ref().ok_or_else(|| {
                Error::invalid(format!("JPEG encode: AC Huffman table {id} not defined"))
            })?;
            scan_tables.ac[id] = Some(HuffCodes::from_spec(spec)?);
        }
    }
    let blocks = if dct {
        Some(prepare_dct(frame, comps, &geo, tables)?)
    } else {
        None
    };

    let mut out = vec![0xFF, markers::SOI];
    out.extend_from_slice(meta);
    if emit_tables {
        tables.write_tables(&mut out, dct);
    }
    write_sof(&mut out, frame, comps, tables);
    if frame.restart_interval > 0 {
        write_dri(&mut out, frame.restart_interval);
    }
    let ah_al = match frame.process {
        JpegProcess::Lossless {
            point_transform, ..
        } => point_transform & 0x0F,
        _ => 0,
    };
    for plan in scan_plans(frame, comps) {
        let scan_comps: Vec<&JpegComponent<'_>> = plan.comps.iter().map(|&i| &comps[i]).collect();
        write_sos(&mut out, &scan_comps, dct, plan.ss, plan.se, ah_al);
        let mut writer = BitWriter::new();
        {
            let mut sink = Sink::Emit {
                writer: &mut writer,
                tables: &scan_tables,
            };
            code_scan(frame, comps, &geo, blocks.as_deref(), &plan, &mut sink)?;
        }
        out.extend_from_slice(&writer.finish());
    }
    out.extend_from_slice(&[0xFF, markers::EOI]);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Typed options driver.
// ---------------------------------------------------------------------------

/// Which Huffman tables the frame is coded with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HuffmanTables {
    /// Annex K.3 "typical" tables (K.3–K.6; the lossless DC table for
    /// `SOF3`). At `P = 12` the DC / AC categories exceed the alphabet
    /// of the typical tables (Tables F.6 / F.7), so the DCT processes
    /// fall back to [`HuffmanTables::Optimal`] there.
    #[default]
    Typical,
    /// Annex K.2 optimal tables derived from the frame's own symbol
    /// statistics (a counting pass precedes the emission pass).
    Optimal,
}

/// How the colour space is signalled in the stream (marker segments +
/// component identifiers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSignalling {
    /// One component → JFIF grayscale; three → JFIF YCbCr (ids 1/2/3);
    /// four → plain ("regular", no APP14) CMYK.
    #[default]
    Auto,
    /// JFIF APP0 (T.871): grayscale or YCbCr with ids 1 / 2 / 3.
    Jfif,
    /// Three components of untransformed R/G/B: Adobe APP14
    /// `transform = 0` and component ids `'R' / 'G' / 'B'`, no JFIF.
    Rgb,
    /// Four components. `adobe_transform = None` writes no APP14
    /// ("regular" CMYK, samples coded as given); `Some(0)` writes Adobe
    /// APP14 `transform = 0` and inverts every sample on the wire;
    /// `Some(2)` writes `transform = 2` (YCCK — the caller supplies
    /// `Y / Cb / Cr / K`) and inverts only the fourth component.
    Cmyk { adobe_transform: Option<u8> },
    /// No colour-signalling segment at all; component ids 1..=Nf.
    None,
}

/// Typed encoder options — the single knob set behind the direct
/// factories and the registry encoder's `CodecOptions`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegEncodeOptions {
    /// Quality factor 1..=100 (DCT processes; see [`scaled_quant_table`]).
    pub quality: u8,
    /// Typical (Annex K.3) or optimal (Annex K.2) Huffman tables.
    pub tables: HuffmanTables,
    /// Sequential / progressive / lossless.
    pub process: JpegProcess,
    /// Sample precision `P`: 8 or 12 for the DCT processes, 2..=16 for
    /// lossless.
    pub precision: u8,
    /// Restart interval in MCUs (0 = none). Lossless: a multiple of the
    /// MCUs per MCU-row (Table B.7).
    pub restart_interval: u16,
    /// Emit the §B.5 abbreviated pair: a table-less frame stream plus a
    /// separate tables-only stream (TIFF `JPEGTables`).
    pub abbreviated: bool,
    /// Colour-space signalling (markers + component ids).
    pub signalling: ColorSignalling,
    /// Per-component sampling factors `(Hi, Vi)`; empty = every
    /// component `1×1`. Any §A.1.1 combination under the §B.2.3 bound.
    pub sampling: Vec<(u8, u8)>,
    /// Per-component table destinations `(Tq, Td = Ta)`; empty = the
    /// first component uses destination 0 and the others 1.
    pub table_ids: Vec<(u8, u8)>,
}

impl Default for JpegEncodeOptions {
    fn default() -> Self {
        JpegEncodeOptions {
            quality: crate::encoder::DEFAULT_QUALITY,
            tables: HuffmanTables::Typical,
            process: JpegProcess::Sequential,
            precision: 8,
            restart_interval: 0,
            abbreviated: false,
            signalling: ColorSignalling::Auto,
            sampling: Vec::new(),
            table_ids: Vec::new(),
        }
    }
}

/// The output of [`JpegEncodeOptions::encode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedJpeg {
    /// The frame stream (`SOI … EOI`). Complete interchange format
    /// unless `abbreviated` was set, in which case it carries no table
    /// segments.
    pub data: Vec<u8>,
    /// The §B.5 tables-only stream (`SOI`, DQT / DHT, `EOI`) when
    /// `abbreviated` was set, `None` otherwise.
    pub tables: Option<Vec<u8>>,
}

/// JFIF APP0 (T.871 §10.1): version 1.01, no density, no thumbnail.
fn write_jfif_app0(out: &mut Vec<u8>) {
    out.extend_from_slice(&[
        0xFF,
        markers::APP0,
        0,
        16,
        b'J',
        b'F',
        b'I',
        b'F',
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
    ]);
}

/// Adobe APP14: `"Adobe"`, version 100, flags 0, `transform`.
fn write_adobe_app14(out: &mut Vec<u8>, transform: u8) {
    out.extend_from_slice(&[0xFF, markers::APP14, 0, 14]);
    out.extend_from_slice(b"Adobe");
    out.extend_from_slice(&[0, 100, 0, 0, 0, 0, transform]);
}

impl JpegEncodeOptions {
    /// The effective signalling for `nf` components.
    fn resolve_signalling(&self, nf: usize) -> Result<ColorSignalling> {
        let s = match self.signalling {
            ColorSignalling::Auto => match nf {
                1 | 3 => ColorSignalling::Jfif,
                4 => ColorSignalling::Cmyk {
                    adobe_transform: None,
                },
                _ => ColorSignalling::None,
            },
            other => other,
        };
        let ok = match s {
            ColorSignalling::Jfif => nf == 1 || nf == 3,
            ColorSignalling::Rgb => nf == 3,
            ColorSignalling::Cmyk { adobe_transform } => {
                nf == 4 && matches!(adobe_transform, None | Some(0) | Some(2))
            }
            ColorSignalling::None | ColorSignalling::Auto => true,
        };
        if !ok {
            return Err(Error::invalid(format!(
                "JPEG encode: colour signalling {s:?} does not fit {nf} component(s) (JFIF: 1 or 3, RGB: 3, CMYK: 4 with Adobe transform 0 / 2)"
            )));
        }
        Ok(s)
    }

    /// Build the table set these options imply for `frame` / `comps`:
    /// the typical set, with the Huffman tables replaced by K.2 optimal
    /// ones when requested (or required, at `P = 12` on the DCT
    /// processes).
    pub fn tables_for(
        &self,
        frame: &JpegFrame,
        comps: &[JpegComponent<'_>],
    ) -> Result<JpegTableSet> {
        let lossless = !frame.process.is_dct();
        let mut t = JpegTableSet::typical(self.quality, frame.precision, lossless, comps.len());
        let need_optimal =
            self.tables == HuffmanTables::Optimal || (!lossless && frame.precision > 8);
        if need_optimal {
            t.optimise_huffman(frame, comps)?;
        }
        Ok(t)
    }

    /// Encode `planes` — one row-major `u16` plane per component at the
    /// component's own A.1.1 resolution (`ceil(width × Hi / Hmax) ×
    /// ceil(height × Vi / Vmax)`), samples below `2^precision` — as one
    /// JPEG frame. 1, 3 or 4 planes.
    pub fn encode(&self, width: u32, height: u32, planes: &[&[u16]]) -> Result<EncodedJpeg> {
        let nf = planes.len();
        if !matches!(nf, 1 | 3 | 4) {
            return Err(Error::invalid(format!(
                "JPEG encode: {nf} planes (1, 3 or 4 components are supported)"
            )));
        }
        if width == 0 || width > 65535 || height == 0 || height > 65535 {
            return Err(Error::invalid(
                "JPEG encode: frame dimensions must be in 1..=65535 (B.2.2)",
            ));
        }
        if !self.sampling.is_empty() && self.sampling.len() != nf {
            return Err(Error::invalid(format!(
                "JPEG encode: {} sampling entries for {nf} components",
                self.sampling.len()
            )));
        }
        if !self.table_ids.is_empty() && self.table_ids.len() != nf {
            return Err(Error::invalid(format!(
                "JPEG encode: {} table-id entries for {nf} components",
                self.table_ids.len()
            )));
        }
        let signalling = self.resolve_signalling(nf)?;
        let frame = JpegFrame {
            width: width as u16,
            height: height as u16,
            precision: self.precision,
            process: self.process,
            restart_interval: self.restart_interval,
        };
        let sampling: Vec<(u8, u8)> = if self.sampling.is_empty() {
            vec![(1, 1); nf]
        } else {
            self.sampling.clone()
        };
        let h_max = sampling.iter().map(|s| s.0).max().unwrap_or(1).max(1) as usize;
        let v_max = sampling.iter().map(|s| s.1).max().unwrap_or(1).max(1) as usize;
        let ids: Vec<u8> = match signalling {
            ColorSignalling::Rgb => b"RGB".to_vec(),
            _ => (1..=nf as u8).collect(),
        };
        // Wire representation of Adobe-flagged CMYK (inverted ink).
        let max = (1u32 << self.precision.min(16)) - 1;
        let invert: [bool; 4] = match signalling {
            ColorSignalling::Cmyk {
                adobe_transform: Some(0),
            } => [true; 4],
            ColorSignalling::Cmyk {
                adobe_transform: Some(2),
            } => [false, false, false, true],
            _ => [false; 4],
        };
        let owned: Vec<Option<Vec<u16>>> = planes
            .iter()
            .enumerate()
            .map(|(i, p)| {
                invert[i].then(|| {
                    p.iter()
                        .map(|&v| (max - u32::from(v).min(max)) as u16)
                        .collect()
                })
            })
            .collect();
        let comps: Vec<JpegComponent<'_>> = (0..nf)
            .map(|i| {
                let (h, v) = sampling[i];
                let (quant_id, huff_id) = if self.table_ids.is_empty() {
                    if i == 0 {
                        (0, 0)
                    } else {
                        (1, 1)
                    }
                } else {
                    self.table_ids[i]
                };
                JpegComponent {
                    id: ids[i],
                    samples: owned[i].as_deref().unwrap_or(planes[i]),
                    width: (width as usize * h.max(1) as usize).div_ceil(h_max),
                    height: (height as usize * v.max(1) as usize).div_ceil(v_max),
                    h,
                    v,
                    quant_id,
                    huff_id,
                }
            })
            .collect();
        let tables = self.tables_for(&frame, &comps)?;
        let mut meta = Vec::new();
        match signalling {
            ColorSignalling::Jfif => write_jfif_app0(&mut meta),
            ColorSignalling::Rgb => write_adobe_app14(&mut meta, 0),
            ColorSignalling::Cmyk {
                adobe_transform: Some(t),
            } => write_adobe_app14(&mut meta, t),
            _ => {}
        }
        let data = encode_frame_with_meta(&frame, &comps, &tables, !self.abbreviated, &meta)?;
        let tables_stream = self
            .abbreviated
            .then(|| tables.tables_stream(frame.process.is_dct()));
        Ok(EncodedJpeg {
            data,
            tables: tables_stream,
        })
    }

    /// [`JpegEncodeOptions::encode`] for 8-bit planes (`precision` must
    /// be 8, or at most 8 for the lossless process).
    pub fn encode_u8(&self, width: u32, height: u32, planes: &[&[u8]]) -> Result<EncodedJpeg> {
        if self.precision > 8 {
            return Err(Error::invalid(format!(
                "JPEG encode: 8-bit planes cannot carry precision {}",
                self.precision
            )));
        }
        let wide: Vec<Vec<u16>> = planes
            .iter()
            .map(|p| p.iter().map(|&v| u16::from(v)).collect())
            .collect();
        let refs: Vec<&[u16]> = wide.iter().map(|v| v.as_slice()).collect();
        self.encode(width, height, &refs)
    }
}

// ---------------------------------------------------------------------------
// Tests: table transcription cross-checks + structural properties.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typical_tables_match_the_crate_decoder_constants() {
        use crate::jpeg::huffman as h;
        use crate::jpeg::quant as q;
        assert_eq!(QUANT_LUMINANCE_K1, q::DEFAULT_LUMA_Q50);
        assert_eq!(QUANT_CHROMINANCE_K2, q::DEFAULT_CHROMA_Q50);
        assert_eq!(BITS_DC_LUMINANCE, h::STD_DC_LUMA_BITS);
        assert_eq!(VAL_DC_LUMINANCE, h::STD_DC_LUMA_VALS);
        assert_eq!(BITS_DC_CHROMINANCE, h::STD_DC_CHROMA_BITS);
        assert_eq!(VAL_DC_CHROMINANCE, h::STD_DC_CHROMA_VALS);
        assert_eq!(BITS_AC_LUMINANCE, h::STD_AC_LUMA_BITS);
        assert_eq!(VAL_AC_LUMINANCE, h::STD_AC_LUMA_VALS);
        assert_eq!(BITS_AC_CHROMINANCE, h::STD_AC_CHROMA_BITS);
        assert_eq!(VAL_AC_CHROMINANCE, h::STD_AC_CHROMA_VALS);
    }

    /// Table K.3 prints the luminance DC code words: category 0 =
    /// `00` (2 bits), 1..=5 = `010`..`110` (3 bits), 6 = `1110`, …,
    /// 11 = `111111110` (9 bits); Table K.4: 0 = `00`, 2 = `10`,
    /// 3 = `110`, 11 = `11111111110`.
    #[test]
    fn k3_k4_code_words_match_the_printed_tables() {
        let codes = HuffCodes::from_spec(&HuffSpec::k3_dc_luminance()).unwrap();
        assert_eq!((codes.code[0], codes.size[0]), (0b00, 2));
        assert_eq!((codes.code[1], codes.size[1]), (0b010, 3));
        assert_eq!((codes.code[5], codes.size[5]), (0b110, 3));
        assert_eq!((codes.code[6], codes.size[6]), (0b1110, 4));
        assert_eq!((codes.code[11], codes.size[11]), (0b111111110, 9));
        let codes = HuffCodes::from_spec(&HuffSpec::k4_dc_chrominance()).unwrap();
        assert_eq!((codes.code[0], codes.size[0]), (0b00, 2));
        assert_eq!((codes.code[2], codes.size[2]), (0b10, 2));
        assert_eq!((codes.code[3], codes.size[3]), (0b110, 3));
        assert_eq!((codes.code[11], codes.size[11]), (0b11111111110, 11));
    }

    /// No code word may be all ones (§C.2) in any built-in table.
    #[test]
    fn builtin_tables_reserve_the_all_ones_code() {
        for spec in [
            HuffSpec::k3_dc_luminance(),
            HuffSpec::k4_dc_chrominance(),
            HuffSpec::k5_ac_luminance(),
            HuffSpec::k6_ac_chrominance(),
            HuffSpec::lossless_dc(),
        ] {
            let codes = HuffCodes::from_spec(&spec).unwrap();
            for v in 0..256 {
                let s = codes.size[v] as u32;
                if s > 0 {
                    assert_ne!(codes.code[v] as u32, (1u32 << s) - 1, "symbol {v}");
                }
            }
        }
        // A Kraft-complete list is rejected outright.
        let full = HuffSpec {
            bits: [0, 0, 0, 15, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vals: (0..17).collect(),
        };
        assert!(full.validate().is_err());
    }

    #[test]
    fn quality_scaling_anchors() {
        let q50 = scaled_quant_table(&QUANT_LUMINANCE_K1, 50, 8);
        assert_eq!(q50, QUANT_LUMINANCE_K1);
        let q75 = scaled_quant_table(&QUANT_LUMINANCE_K1, 75, 8);
        assert_eq!(q75[0], 8);
        assert_eq!(q75[63], 50);
        let q100 = scaled_quant_table(&QUANT_LUMINANCE_K1, 100, 8);
        assert!(q100.iter().all(|&v| v == 1));
        let q1 = scaled_quant_table(&QUANT_LUMINANCE_K1, 1, 8);
        assert!(q1.iter().all(|&v| v == 255));
        let q12 = scaled_quant_table(&QUANT_LUMINANCE_K1, 50, 12);
        assert_eq!(q12[0], 16 * 16);
    }

    #[test]
    fn categories_and_extra_bits() {
        assert_eq!(category(0), 0);
        assert_eq!(category(1), 1);
        assert_eq!(category(-1), 1);
        assert_eq!(category(3), 2);
        assert_eq!(category(-4), 3);
        assert_eq!(category(2047), 11);
        assert_eq!(category(-32767), 15);
        assert_eq!(extra_bits(5, 3), 0b101);
        assert_eq!(extra_bits(-5, 3), 0b010);
        assert_eq!(extra_bits(-1, 1), 0);
        assert_eq!(extra_bits(1, 1), 1);
    }

    #[test]
    fn bit_writer_stuffs_ff_and_pads_with_ones() {
        let mut w = BitWriter::new();
        w.put(0xFF, 8);
        w.put(0b101, 3);
        assert_eq!(w.finish(), vec![0xFF, 0x00, 0b1011_1111]);
        let mut w = BitWriter::new();
        w.put(0b111_1111, 7);
        assert_eq!(w.finish(), vec![0xFF, 0x00]);
    }

    #[test]
    fn optimal_tables_cover_every_counted_symbol_and_limit_lengths() {
        let mut st = HuffStats::default();
        for v in 0..=255u32 {
            for _ in 0..(1 << (v % 12)) {
                st.count(v as u8);
            }
        }
        let spec = st.to_spec();
        spec.validate().unwrap();
        assert_eq!(spec.vals.len(), 256);
        let codes = HuffCodes::from_spec(&spec).unwrap();
        for v in 0..256 {
            assert!(codes.size[v] >= 1 && codes.size[v] <= 16);
            let s = codes.size[v] as u32;
            assert_ne!(codes.code[v] as u32, (1u32 << s) - 1);
        }
        let kraft: f64 = (0..256).map(|v| 2f64.powi(-(codes.size[v] as i32))).sum();
        assert!(kraft < 1.0);
        // Single-symbol alphabet degenerates to one 1-bit code.
        let mut st = HuffStats::default();
        st.count(0);
        let spec = st.to_spec();
        assert_eq!(spec.bits[0], 1);
        assert_eq!(spec.vals, vec![0]);
        // Steep geometric statistics force the K.3 length limiting.
        let mut st = HuffStats::default();
        let mut n: u64 = 1 << 40;
        for v in 0..40u8 {
            for _ in 0..(n.min(1 << 20)) {
                st.count(v);
            }
            n >>= 1;
        }
        let spec = st.to_spec();
        spec.validate().unwrap();
        assert_eq!(spec.vals.len(), 40);
    }

    fn gray_frame(w: u16, h: u16) -> (Vec<u16>, JpegFrame) {
        let mut s = Vec::with_capacity(w as usize * h as usize);
        for y in 0..h {
            for x in 0..w {
                s.push((x * 3 + y * 5) % 256);
            }
        }
        (
            s,
            JpegFrame {
                width: w,
                height: h,
                precision: 8,
                process: JpegProcess::Sequential,
                restart_interval: 0,
            },
        )
    }

    fn comp(s: &[u16], w: usize, h: usize) -> JpegComponent<'_> {
        JpegComponent {
            id: 1,
            samples: s,
            width: w,
            height: h,
            h: 1,
            v: 1,
            quant_id: 0,
            huff_id: 0,
        }
    }

    #[test]
    fn frame_stream_has_the_b2_marker_skeleton() {
        let (s, frame) = gray_frame(13, 9);
        let c = comp(&s, 13, 9);
        let t = JpegTableSet::typical(75, 8, false, 3);
        let out = encode_frame(&frame, std::slice::from_ref(&c), &t, true).unwrap();
        assert_eq!(&out[..2], &[0xFF, markers::SOI]);
        assert_eq!(&out[out.len() - 2..], &[0xFF, markers::EOI]);
        let mut i = 2;
        let mut seen = Vec::new();
        while i + 4 <= out.len() {
            assert_eq!(out[i], 0xFF);
            let m = out[i + 1];
            seen.push(m);
            if m == markers::SOS {
                break;
            }
            let len = u16::from_be_bytes([out[i + 2], out[i + 3]]) as usize;
            i += 2 + len;
        }
        use markers::{DHT, DQT, SOF0, SOS};
        assert_eq!(seen, vec![DQT, DQT, DHT, DHT, DHT, DHT, SOF0, SOS]);
        // Abbreviated form drops the table segments.
        let abbr = encode_frame(&frame, &[c], &t, false).unwrap();
        assert_eq!(&abbr[2..4], &[0xFF, SOF0]);
        assert!(abbr.len() < out.len());
        // No unstuffed FF inside the entropy segment except the EOI.
        let sos_at = out.windows(2).position(|w| w == [0xFF, SOS]).unwrap();
        let body = &out[sos_at + 2 + 6 + 2..out.len() - 2];
        for w in body.windows(2) {
            if w[0] == 0xFF {
                assert_eq!(w[1], 0x00);
            }
        }
    }

    #[test]
    fn geometry_validation_rejects_a1_1_mismatch() {
        let (s, frame) = gray_frame(16, 16);
        let c = comp(&s[..15 * 16], 15, 16);
        let t = JpegTableSet::typical(75, 8, false, 1);
        assert!(encode_frame(&frame, &[c], &t, true).is_err());
    }

    #[test]
    fn twelve_bit_frames_are_sof1_with_16_bit_quantisers() {
        let (s, mut frame) = gray_frame(24, 17);
        let s12: Vec<u16> = s.iter().map(|&v| v * 16).collect();
        frame.precision = 12;
        let c = comp(&s12, 24, 17);
        let opts = JpegEncodeOptions {
            precision: 12,
            quality: 90,
            ..Default::default()
        };
        let t = opts.tables_for(&frame, std::slice::from_ref(&c)).unwrap();
        let out = encode_frame(&frame, &[c], &t, true).unwrap();
        assert!(out.windows(2).any(|w| w == [0xFF, markers::SOF1]));
        let dqt = out
            .windows(2)
            .position(|w| w == [0xFF, markers::DQT])
            .unwrap();
        assert_eq!(out[dqt + 4] >> 4, 1);
    }

    #[test]
    fn lossless_stream_uses_sof3_and_predictor_in_sos() {
        let (s, mut frame) = gray_frame(7, 5);
        frame.process = JpegProcess::Lossless {
            predictor: 4,
            point_transform: 0,
        };
        let c = comp(&s, 7, 5);
        let mut t = JpegTableSet::default();
        let mut dc: [HuffStats; 4] = Default::default();
        let mut ac: [HuffStats; 4] = Default::default();
        gather_stats(&frame, std::slice::from_ref(&c), &t, &mut dc, &mut ac).unwrap();
        assert!(ac[0].is_empty());
        t.dc[0] = Some(dc[0].to_spec());
        let out = encode_frame(&frame, &[c], &t, true).unwrap();
        let sof = out
            .windows(2)
            .position(|w| w == [0xFF, markers::SOF3])
            .unwrap();
        assert_eq!(out[sof + 4], 8);
        let sos = out
            .windows(2)
            .position(|w| w == [0xFF, markers::SOS])
            .unwrap();
        // FF DA | Ls (2) | Ns (1) | Cs1 Td/Ta (2) | Ss | Se | Ah/Al.
        assert_eq!(out[sos + 7], 4);
        assert_eq!(out[sos + 8], 0);
        assert!(!out.windows(2).any(|w| w == [0xFF, markers::DQT]));
    }

    #[test]
    fn lossless_restart_interval_must_be_row_aligned() {
        let (s, mut frame) = gray_frame(7, 5);
        frame.process = JpegProcess::Lossless {
            predictor: 1,
            point_transform: 0,
        };
        frame.restart_interval = 5;
        let c = comp(&s, 7, 5);
        let t = JpegTableSet::typical(75, 8, true, 1);
        assert!(encode_frame(&frame, std::slice::from_ref(&c), &t, true).is_err());
        frame.restart_interval = 14;
        assert!(encode_frame(&frame, &[c], &t, true).is_ok());
    }

    #[test]
    fn options_reject_bad_signalling_and_plane_counts() {
        let p = [0u16; 4];
        let o = JpegEncodeOptions {
            signalling: ColorSignalling::Rgb,
            ..Default::default()
        };
        assert!(o.encode(2, 2, &[&p]).is_err());
        assert!(JpegEncodeOptions::default()
            .encode(2, 2, &[&p, &p])
            .is_err());
        let o = JpegEncodeOptions {
            signalling: ColorSignalling::Cmyk {
                adobe_transform: Some(1),
            },
            ..Default::default()
        };
        assert!(o.encode(2, 2, &[&p, &p, &p, &p]).is_err());
    }
}
