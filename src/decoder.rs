//! JPEG packet decoder — baseline (SOF0) and progressive (SOF2).

use crate::error::{MjpegError as Error, Result};

// When the `registry` feature is on, the public `decode_jpeg` returns
// `oxideav_core::VideoFrame` directly so the trait-side `Decoder`
// impl can hand it to `Frame::Video(...)` without an extra
// conversion. The inner decoder logic only touches plane
// `(stride, data)` tuples and YUV / Gray / Cmyk pixel-format
// discriminants — both type families share that shape, so the same
// helpers compile against either alias unchanged.
//
// With `--no-default-features` the same names resolve to the
// crate-local [`MjpegFrame`] / [`MjpegPixelFormat`] / [`MjpegPlane`]
// types so the standalone build never references `oxideav-core`.
#[cfg(feature = "registry")]
use oxideav_core::frame::VideoPlane;
#[cfg(feature = "registry")]
use oxideav_core::{PixelFormat, VideoFrame};

#[cfg(not(feature = "registry"))]
use crate::image::{
    MjpegFrame as VideoFrame, MjpegPixelFormat as PixelFormat, MjpegPlane as VideoPlane,
};

// Re-export the framework-side `Decoder` factory at its historical
// path so consumers (and integration tests) that import
// `oxideav_mjpeg::decoder::make_decoder` keep compiling.
#[cfg(feature = "registry")]
pub use crate::registry::make_decoder;

use crate::jpeg::arith::{
    decode_ac as arith_decode_ac, decode_dc_diff as arith_decode_dc_diff, AcStats, ArithDecoder,
    DcStats,
};
use crate::jpeg::dct::idct8x8;
use crate::jpeg::huffman::{parse_dht, HuffTable};
use crate::jpeg::markers::{self, *};
use crate::jpeg::parser::{
    parse_dac, parse_dri, parse_sof, parse_sos, MarkerWalker, SofInfo, SosInfo,
};
use crate::jpeg::quant::{parse_dqt, QuantTable};
use crate::jpeg::zigzag::ZIGZAG;

// ---- Decoding state ------------------------------------------------------

struct JpegState {
    quant: [Option<QuantTable>; 4],
    dc_huff: [Option<HuffTable>; 4],
    ac_huff: [Option<HuffTable>; 4],
    restart_interval: u16,
    sof: Option<SofInfo>,
    /// True when SOF2 (progressive) was parsed.
    progressive: bool,
    /// True when a baseline/extended-sequential scan has been accumulated
    /// into the coefficient buffer because it was non-interleaved. Once set,
    /// all subsequent scans also accumulate and we render at EOI (same path
    /// as progressive, just with single-pass coefficients).
    seq_accum: bool,
    /// True when SOF3 (lossless) was parsed. Mutually exclusive with
    /// `progressive` / `seq_accum` — lossless JPEGs use predictor-based
    /// coding rather than DCT and take their own scan decoder.
    lossless: bool,
    /// Adobe APP14 transform flag. `None` if no Adobe marker was seen;
    /// `Some(0)` for direct (CMYK / RGB, samples stored as-is but
    /// Adobe-inverted for CMYK); `Some(1)` for YCbCr (3-component);
    /// `Some(2)` for YCCK (4-component Adobe colour transform).
    adobe_transform: Option<u8>,
    /// True when SOF9 (extended sequential, arithmetic-coded) was parsed.
    /// Mutually exclusive with `progressive` / `lossless`. The scan
    /// dispatcher takes the arithmetic Q-coder path instead of Huffman.
    arithmetic: bool,
    /// Per-destination DC arithmetic conditioning (L/U bounds + per-scan
    /// statistics). Indexed by Tb (0..3). Defaults are L=0, U=1.
    arith_dc: [Option<ArithDcConditioning>; 4],
    /// Per-destination AC arithmetic conditioning (Kx threshold + stats).
    /// Indexed by Tb (0..3). Default Kx=5.
    arith_ac: [Option<ArithAcConditioning>; 4],
}

/// Container holding both the conditioning parameters (L, U) and the live
/// statistics-bin storage that persists across blocks within a scan.
#[derive(Clone, Debug)]
struct ArithDcConditioning {
    pub l: u8,
    pub u: u8,
}

#[derive(Clone, Debug)]
struct ArithAcConditioning {
    pub kx: u8,
}

impl JpegState {
    fn new() -> Self {
        Self {
            quant: Default::default(),
            dc_huff: Default::default(),
            ac_huff: Default::default(),
            restart_interval: 0,
            sof: None,
            progressive: false,
            seq_accum: false,
            lossless: false,
            adobe_transform: None,
            arithmetic: false,
            arith_dc: Default::default(),
            arith_ac: Default::default(),
        }
    }
}

/// Bound the total number of luma samples a single frame may declare
/// in its SOF. T.81 nominally permits 65535 × 65535 × Nf, which a
/// naive implementation would happily attempt to allocate as
/// `width × height × Nf × per-sample-size` bytes — easily tens of
/// gigabytes from a 16-byte SOF segment. The cap here is generous
/// enough to cover any realistic JPEG (8K = 33 Mpx) while keeping the
/// largest possible decoder allocation in the low hundreds of MiB.
const MAX_PIXEL_BUDGET: u64 = 64 * 1024 * 1024;

/// Centralised SOF validator. Run on every freshly-parsed SOF before
/// it's stored as `state.sof`. Catches the panic surfaces a SOF can
/// open downstream:
///   * `Nf = 0` → empty component list → divisions by zero / empty
///     vec indexing.
///   * `Hi / Vi` outside `1..=4` (T.81 §B.2.2) → MCU geometry
///     arithmetic overflows + zero-MCU loops.
///   * `Tq > 3` → out-of-bounds index into the 4-wide quant table.
///   * `Wt × Ht × Nf > MAX_PIXEL_BUDGET` → unbounded
///     `vec![0u8; W × H]` allocation.
fn validate_sof(sof: &SofInfo) -> Result<()> {
    if sof.components.is_empty() {
        return Err(Error::invalid("SOF: Nf = 0"));
    }
    if sof.components.len() > 4 {
        return Err(Error::unsupported("SOF: Nf > 4"));
    }
    for c in &sof.components {
        if !(1..=4).contains(&c.h_factor) || !(1..=4).contains(&c.v_factor) {
            return Err(Error::invalid("SOF: Hi/Vi outside 1..=4"));
        }
        if c.qt_id >= 4 {
            return Err(Error::invalid("SOF: Tq > 3"));
        }
    }
    let w = sof.width as u64;
    let h = sof.height as u64;
    let nf = sof.components.len() as u64;
    if w.saturating_mul(h).saturating_mul(nf) > MAX_PIXEL_BUDGET {
        return Err(Error::unsupported("SOF: pixel budget exceeded"));
    }
    Ok(())
}

/// Centralised SOS validator. Run on every freshly-parsed SOS before
/// dispatch to the scan decoder. Catches the panic surfaces a SOS
/// can open:
///   * `Ns = 0` → empty scan-component list → empty `prev_dc` /
///     `dc_tables` and out-of-bounds Tdj/Taj indexing later.
///   * `Ns > 4` → larger than the spec's 4-component cap; the per-MCU
///     accumulator loops assume at most 4 components.
///   * `Tdj / Taj > 3` → out-of-bounds index into the 4-wide DC / AC
///     Huffman table arrays.
fn validate_sos(sos: &SosInfo) -> Result<()> {
    if sos.components.is_empty() {
        return Err(Error::invalid("SOS: Ns = 0"));
    }
    if sos.components.len() > 4 {
        return Err(Error::invalid("SOS: Ns > 4"));
    }
    for sc in &sos.components {
        if sc.dc_table >= 4 {
            return Err(Error::invalid("SOS: Tdj > 3"));
        }
        if sc.ac_table >= 4 {
            return Err(Error::invalid("SOS: Taj > 3"));
        }
    }
    Ok(())
}

pub(crate) fn decode_jpeg(data: &[u8], pts: Option<i64>) -> Result<VideoFrame> {
    // Verify SOI.
    if data.len() < 2 || data[0] != 0xFF || data[1] != markers::SOI {
        return Err(Error::invalid("JPEG: missing SOI"));
    }

    let mut walker = MarkerWalker::new(&data[2..]);
    let mut state = JpegState::new();

    // Coefficient accumulator, populated on progressive (SOF2) or when a
    // baseline scan turns out to be non-interleaved. One [i32;64] per block
    // per component, in natural order.
    let mut coef_buf: Vec<Vec<[i32; 64]>> = Vec::new();

    loop {
        let Some(marker) = walker.next_marker()? else {
            return Err(Error::invalid("JPEG: unexpected EOF before EOI"));
        };
        match marker {
            EOI => {
                if state.progressive || state.seq_accum || state.arithmetic {
                    return render_from_coefs(&state, &coef_buf, pts);
                }
                return Err(Error::invalid("JPEG: EOI before SOS"));
            }
            SOI => continue,
            m if markers::is_rst(m) => continue,
            DQT => {
                let p = walker.read_segment_payload()?;
                parse_dqt(p, &mut state.quant)?;
            }
            DHT => {
                let p = walker.read_segment_payload()?;
                parse_dht(p, &mut state.dc_huff, &mut state.ac_huff)?;
            }
            DAC => {
                let p = walker.read_segment_payload()?;
                let entries = parse_dac(p)?;
                for e in entries {
                    if e.tc == 0 {
                        // DC conditioning: cs packs L (low nibble) and U (high nibble).
                        let l = e.cs & 0x0F;
                        let u = e.cs >> 4;
                        if l > u || u > 15 {
                            return Err(Error::invalid("DAC: invalid L/U bounds"));
                        }
                        state.arith_dc[e.tb as usize] = Some(ArithDcConditioning { l, u });
                    } else {
                        // AC conditioning: cs is Kx in 1..=63.
                        state.arith_ac[e.tb as usize] = Some(ArithAcConditioning { kx: e.cs });
                    }
                }
            }
            DRI => {
                let p = walker.read_segment_payload()?;
                state.restart_interval = parse_dri(p)?;
            }
            // SOF0 (baseline) and SOF1 (extended sequential) share the same
            // Huffman sequential scan structure; for 8-bit precision the
            // decode path is identical, and the extended-sequential
            // allowance of up to 4 DC/AC Huffman tables falls out of our
            // existing 4-entry table arrays. Treat SOF1 as SOF0.
            SOF0 | SOF1 => {
                if state.sof.is_some() {
                    return Err(Error::invalid("JPEG: multiple SOF segments"));
                }
                let p = walker.read_segment_payload()?;
                let sof = parse_sof(p)?;
                validate_sof(&sof)?;
                state.sof = Some(sof);
            }
            SOF2 => {
                if state.sof.is_some() {
                    return Err(Error::invalid("JPEG: multiple SOF segments"));
                }
                let p = walker.read_segment_payload()?;
                let sof = parse_sof(p)?;
                validate_sof(&sof)?;
                // T.81 §G.1.1 permits SOF2 at P = 8 or P = 12. The progressive
                // scan path operates on i32 coefficient planes, so the
                // increased magnitude range from a 12-bit DC/AC residual fits
                // without code changes; the coefficient accumulator + the
                // EOI render dispatcher both branch on `sof.precision` and
                // already handle the 12-bit case (`render_from_coefs_12bit`).
                //
                // 4-component progressive (CMYK / YCCK) is permitted at
                // `P = 8`: `decode_progressive_scan` is component-count
                // agnostic (interleaved DC walks every SOS component,
                // AC scans are always non-interleaved), the coefficient
                // accumulator already sizes for up to 4 components, and the
                // EOI render path (`render_from_coefs`) already produces a
                // packed `Cmyk` plane for 4-component scans (honouring the
                // Adobe APP14 transform flag for plain CMYK / inverted CMYK /
                // YCCK). 4-component progressive at `P = 12` stays
                // unsupported because the workspace `PixelFormat` enum has
                // no 12-bit CMYK variant.
                if sof.precision != 8 && sof.precision != 12 {
                    return Err(Error::unsupported(format!(
                        "progressive JPEG: precision {} (only 8 and 12 are supported)",
                        sof.precision
                    )));
                }
                if sof.components.len() == 4 && sof.precision != 8 {
                    return Err(Error::unsupported(
                        "progressive JPEG: 4-component scans only at P = 8",
                    ));
                }
                coef_buf = init_coef_buffers(&sof)?;
                state.sof = Some(sof);
                state.progressive = true;
            }
            SOF3 => {
                if state.sof.is_some() {
                    return Err(Error::invalid("JPEG: multiple SOF segments"));
                }
                let p = walker.read_segment_payload()?;
                let sof = parse_sof(p)?;
                validate_sof(&sof)?;
                if !(2..=16).contains(&sof.precision) {
                    return Err(Error::unsupported(format!(
                        "lossless JPEG: precision {} out of range 2..=16",
                        sof.precision
                    )));
                }
                // Single-component grayscale, three-component RGB-class and
                // four-component CMYK-class interleaved scans (each component
                // declared `H_i = V_i = 1`, per T.81 §H.1.2) are supported.
                // Components with non-unit sampling factors are rejected —
                // Annex H pairs lossless with `data unit = one sample`
                // (E.1.1), so non-1:1 sampling would not be a well-defined
                // MCU. The four-component path is `P = 8` only because the
                // workspace `PixelFormat` enum has no high-bit-depth CMYK
                // variant — wider precisions are rejected here.
                if !matches!(sof.components.len(), 1 | 3 | 4) {
                    return Err(Error::unsupported(format!(
                        "lossless JPEG: {} component(s) — only 1 (grayscale), 3 (RGB-class) and 4 (CMYK-class) are supported",
                        sof.components.len()
                    )));
                }
                if sof.components.len() == 4 && sof.precision != 8 {
                    return Err(Error::unsupported(format!(
                        "lossless JPEG: 4-component scans require precision 8, got {}",
                        sof.precision
                    )));
                }
                if sof.components.len() > 1 {
                    for c in &sof.components {
                        if c.h_factor != 1 || c.v_factor != 1 {
                            return Err(Error::unsupported(
                                "lossless JPEG: multi-component scans require H_i = V_i = 1",
                            ));
                        }
                    }
                }
                state.sof = Some(sof);
                state.lossless = true;
            }
            // SOF9 — extended sequential, arithmetic-coded (T.81 §F.1.4).
            // Same DCT machinery as SOF1, but the entropy coder is the
            // Q-coder from Annex D instead of Huffman. Coefficients are
            // accumulated into the same per-block buffer as the
            // progressive / non-interleaved baseline path so that
            // `render_from_coefs` can do the dequant + IDCT pass at EOI.
            SOF9 => {
                if state.sof.is_some() {
                    return Err(Error::invalid("JPEG: multiple SOF segments"));
                }
                let p = walker.read_segment_payload()?;
                let sof = parse_sof(p)?;
                validate_sof(&sof)?;
                if sof.precision != 8 {
                    return Err(Error::unsupported(format!(
                        "arithmetic JPEG: precision {} (only 8 is supported)",
                        sof.precision
                    )));
                }
                if sof.components.len() > 3 {
                    return Err(Error::unsupported(
                        "arithmetic JPEG: 4+ component scans not supported",
                    ));
                }
                coef_buf = init_coef_buffers(&sof)?;
                state.sof = Some(sof);
                state.arithmetic = true;
            }
            0xC5..=0xC7 | 0xCA..=0xCB | 0xCD..=0xCF => {
                let _ = walker.read_segment_payload();
                return Err(Error::unsupported(
                    "JPEG: hierarchical and SOF10..15 arithmetic variants are not supported",
                ));
            }
            SOS => {
                let p = walker.read_segment_payload()?;
                let sos = parse_sos(p)?;
                validate_sos(&sos)?;
                let scan = walker.read_scan_data()?;
                if state.lossless {
                    return decode_lossless_scan(&state, &sos, scan, pts);
                }
                if state.arithmetic {
                    decode_arith_scan(&state, &sos, scan, &mut coef_buf)?;
                    // Continue — more scans or EOI follow.
                } else if state.progressive {
                    decode_progressive_scan(&state, &sos, scan, &mut coef_buf)?;
                    // Continue — more scans or EOI follow.
                } else {
                    let sof = state
                        .sof
                        .as_ref()
                        .ok_or_else(|| Error::invalid("SOS before SOF"))?;
                    let fully_interleaved = sos.components.len() == sof.components.len();
                    // decode_scan only knows 1-3 components at 8-bit
                    // precision. 4-component (CMYK/YCCK) and 12-bit
                    // sample precision must take the accumulator path,
                    // where the wider output machinery (packed CMYK / u16
                    // sample buffers) lives.
                    let fast_path_ok = fully_interleaved
                        && !state.seq_accum
                        && sof.components.len() <= 3
                        && sof.precision == 8;
                    if fast_path_ok {
                        return decode_scan(&state, &sos, scan, pts);
                    }
                    if !state.seq_accum {
                        coef_buf = init_coef_buffers(sof)?;
                        state.seq_accum = true;
                    }
                    decode_sequential_scan_accum(&state, &sos, scan, &mut coef_buf)?;
                }
            }
            COM => {
                let _ = walker.read_segment_payload()?;
            }
            // APP14 "Adobe" marker: `"Adobe"` magic + 5 bytes of metadata
            // whose last byte is the colour-transform flag (0, 1, or 2).
            // Needed to disambiguate 3-component RGB from YCbCr and
            // 4-component CMYK from YCCK.
            markers::APP14 => {
                let p = walker.read_segment_payload()?;
                if p.len() >= 12 && &p[0..5] == b"Adobe" {
                    state.adobe_transform = Some(p[11]);
                }
            }
            m if markers::is_app(m) => {
                let _ = walker.read_segment_payload()?;
            }
            _ => {
                // Ignore unknown length-prefixed markers but still skip the segment.
                let _ = walker.read_segment_payload();
            }
        }
    }
}

/// Allocate a coefficient-accumulator plane for each component in the SOF.
/// Dimensions match the MCU grid so that blocks line up with decoded scans.
/// Shared by progressive (SOF2, multi-scan successive-approximation) and
/// non-interleaved baseline (SOF0/SOF1 with ns < nf) code paths.
fn init_coef_buffers(sof: &SofInfo) -> Result<Vec<Vec<[i32; 64]>>> {
    if sof.precision != 8 && sof.precision != 12 {
        return Err(Error::unsupported(format!(
            "coef accumulator: precision {} (only 8 and 12 are supported)",
            sof.precision
        )));
    }
    if sof.components.is_empty() {
        return Err(Error::invalid("SOF: no components"));
    }
    if sof.components.len() > 4 {
        return Err(Error::unsupported("coef accumulator: >4 components"));
    }
    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1);
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1);
    if h_max == 0 || v_max == 0 {
        return Err(Error::invalid("SOF: sampling factor = 0"));
    }
    let width = sof.width as usize;
    let height = sof.height as usize;
    let mcu_w_px = 8 * h_max as usize;
    let mcu_h_px = 8 * v_max as usize;
    let mcus_x = width.div_ceil(mcu_w_px);
    let mcus_y = height.div_ceil(mcu_h_px);

    let mut out = Vec::with_capacity(sof.components.len());
    for c in &sof.components {
        let blocks_x = mcus_x * c.h_factor as usize;
        let blocks_y = mcus_y * c.v_factor as usize;
        out.push(vec![[0i32; 64]; blocks_x * blocks_y]);
    }
    Ok(out)
}

// ---- Bit reader with 0xFF00 stuff handling -------------------------------

struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    /// 32-bit bit-buffer (MSB-aligned).
    bits: u32,
    /// Number of valid bits currently in `bits`.
    nbits: u32,
    /// Set when we hit a restart marker while refilling.
    pub saw_rst: Option<u8>,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            bits: 0,
            nbits: 0,
            saw_rst: None,
        }
    }

    fn next_byte_with_stuff(&mut self) -> Result<Option<u8>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        if b == 0xFF {
            // Collapse any run of 0xFF fill bytes.
            while self.pos < self.buf.len() && self.buf[self.pos] == 0xFF {
                self.pos += 1;
            }
            if self.pos >= self.buf.len() {
                return Err(Error::invalid("scan: 0xFF at end without followup"));
            }
            let next = self.buf[self.pos];
            self.pos += 1;
            if next == 0x00 {
                // Stuffed zero → literal 0xFF.
                return Ok(Some(0xFF));
            }
            if markers::is_rst(next) {
                self.saw_rst = Some(next);
                // Stop feeding bits; caller must handle restart.
                return Ok(None);
            }
            // Any other marker means scan ended. Back up and stop.
            self.pos -= 2;
            return Ok(None);
        }
        Ok(Some(b))
    }

    fn fill(&mut self, needed: u32) -> Result<()> {
        // The shift `24 - self.nbits` in the refill below underflows
        // for `self.nbits > 24`. JPEG entropy tokens never need more
        // than 16 bits at a time (`extend` symbols cap at 16, and the
        // Annex F lossless residual is the only 16-bit slot — handled
        // out-of-band), so requesting more than 24 here is always a
        // caller bug. Refuse rather than panic.
        if needed > 24 {
            return Err(Error::invalid("BitReader: requested > 24 bits"));
        }
        while self.nbits < needed {
            match self.next_byte_with_stuff()? {
                Some(b) => {
                    self.bits |= (b as u32) << (24 - self.nbits);
                    self.nbits += 8;
                }
                None => {
                    // If we ran out of bits mid-decode, pad with zeros —
                    // a conventional tolerance at scan end. The caller is
                    // responsible for noticing `saw_rst` / EOI and stopping.
                    self.bits |= 0;
                    self.nbits = needed;
                    break;
                }
            }
        }
        Ok(())
    }

    fn get_bits(&mut self, n: u32) -> Result<u32> {
        // `n == 0` would compute `self.bits >> 32` which is UB on u32
        // (debug-panic, release-wraparound). Both fuzz and real-world
        // Huffman tables can decode a SSSS of 0 ("magnitude zero" → no
        // extra bits to read), so short-circuit cleanly here rather
        // than push the guard onto every caller.
        if n == 0 {
            return Ok(0);
        }
        if n > 24 {
            return Err(Error::invalid("BitReader: get_bits(n > 24)"));
        }
        self.fill(n)?;
        let v = self.bits >> (32 - n);
        self.bits <<= n;
        self.nbits -= n;
        Ok(v)
    }

    fn reset_at_restart(&mut self) {
        self.bits = 0;
        self.nbits = 0;
        self.saw_rst = None;
    }
}

/// Read a Huffman symbol using a linear walk up the `min_code` table (fine
/// for a textbook implementation — no fast path).
fn decode_huff(br: &mut BitReader<'_>, t: &HuffTable) -> Result<u8> {
    let mut code: i32 = 0;
    for l in 0..16 {
        let bit = br.get_bits(1)? as i32;
        code = (code << 1) | bit;
        if code <= t.max_code[l] {
            let idx = (t.val_offset[l] + code) as usize;
            if idx >= t.values.len() {
                return Err(Error::invalid("huffman: value index OOB"));
            }
            return Ok(t.values[idx]);
        }
    }
    Err(Error::invalid("huffman: no matching code (length > 16)"))
}

/// Read `size` bits and sign-extend per JPEG Annex F rules.
fn extend(value: i32, size: u32) -> i32 {
    if size == 0 {
        return 0;
    }
    let vt = 1 << (size - 1);
    if value < vt {
        value - ((1 << size) - 1)
    } else {
        value
    }
}

fn decode_scan(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("SOS before SOF"))?;
    if sof.precision != 8 {
        return Err(Error::unsupported("precision != 8"));
    }
    if sof.components.is_empty() {
        return Err(Error::invalid("SOF: no components"));
    }
    if sof.components.len() > 3 {
        return Err(Error::unsupported("4+ components"));
    }
    // We only handle an SOS that lists every SOF component (interleaved scan).
    if sos.components.len() != sof.components.len() {
        return Err(Error::unsupported("non-interleaved scan"));
    }

    let n_comp = sof.components.len();
    let grayscale = n_comp == 1;
    let is_rgb = detect_rgb_3comp(sof, state.adobe_transform);

    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1);
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1);
    if h_max == 0 || v_max == 0 {
        return Err(Error::invalid("SOF: sampling factor = 0"));
    }

    // Output pixel format (subsampling is implied).
    let pix_fmt = if grayscale {
        PixelFormat::Gray8
    } else if is_rgb {
        // 3-component RGB baseline: every component is sampled at
        // `H = V = 1` (no chroma subsampling concept), output is packed
        // `Rgb24`. The encoder we ship enforces 1×1 on every component;
        // reject any conformant-but-exotic RGB JPEG that mixes
        // sampling factors so we never silently reinterpret a
        // subsampled "RGB" stream as 1:1.
        for c in &sof.components {
            if c.h_factor != 1 || c.v_factor != 1 {
                return Err(Error::unsupported(
                    "RGB baseline JPEG: every component must declare H = V = 1",
                ));
            }
        }
        PixelFormat::Rgb24
    } else if n_comp == 3 {
        let y = sof.components[0];
        let cb = sof.components[1];
        let cr = sof.components[2];
        if cb.h_factor != cr.h_factor || cb.v_factor != cr.v_factor {
            return Err(Error::unsupported(
                "chroma components have different sampling factors",
            ));
        }
        if cb.h_factor != 1 || cb.v_factor != 1 {
            return Err(Error::unsupported(
                "chroma components must have factor 1 (luma carries the oversampling)",
            ));
        }
        match (y.h_factor, y.v_factor) {
            (1, 1) => PixelFormat::Yuv444P,
            (2, 1) => PixelFormat::Yuv422P,
            (2, 2) => PixelFormat::Yuv420P,
            (4, 1) => PixelFormat::Yuv411P,
            _ => {
                return Err(Error::unsupported(format!(
                    "luma sampling {}x{}",
                    y.h_factor, y.v_factor
                )))
            }
        }
    } else {
        return Err(Error::unsupported("2-component JPEG"));
    };

    let width = sof.width as usize;
    let height = sof.height as usize;
    let mcu_w_px = 8 * h_max as usize;
    let mcu_h_px = 8 * v_max as usize;
    let mcus_x = width.div_ceil(mcu_w_px);
    let mcus_y = height.div_ceil(mcu_h_px);

    // Component-level output buffers sized to full MCU coverage.
    let mut comp_buf: Vec<Vec<u8>> = Vec::with_capacity(n_comp);
    let mut comp_stride: Vec<usize> = Vec::with_capacity(n_comp);
    let mut comp_w_full: Vec<usize> = Vec::with_capacity(n_comp);
    let mut comp_h_full: Vec<usize> = Vec::with_capacity(n_comp);
    for c in &sof.components {
        let w_full = mcus_x * 8 * c.h_factor as usize;
        let h_full = mcus_y * 8 * c.v_factor as usize;
        comp_buf.push(vec![0u8; w_full * h_full]);
        comp_stride.push(w_full);
        comp_w_full.push(w_full);
        comp_h_full.push(h_full);
    }

    // Map SOS component id → index in SOF component vector.
    let sos_map: Vec<usize> = sos
        .components
        .iter()
        .map(|sc| {
            sof.components
                .iter()
                .position(|fc| fc.id == sc.id)
                .ok_or_else(|| Error::invalid("SOS: component id not in SOF"))
        })
        .collect::<Result<Vec<_>>>()?;

    // Resolve Huffman tables.
    let dc_tables: Vec<&HuffTable> = sos
        .components
        .iter()
        .map(|sc| {
            state.dc_huff[sc.dc_table as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("SOS: DC Huffman table missing"))
        })
        .collect::<Result<Vec<_>>>()?;
    let ac_tables: Vec<&HuffTable> = sos
        .components
        .iter()
        .map(|sc| {
            state.ac_huff[sc.ac_table as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("SOS: AC Huffman table missing"))
        })
        .collect::<Result<Vec<_>>>()?;

    let quant_tables: Vec<&QuantTable> = sof
        .components
        .iter()
        .map(|c| {
            state.quant[c.qt_id as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("quant table missing for component"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut br = BitReader::new(scan);
    let mut prev_dc = vec![0i32; n_comp];
    let mut mcus_since_restart: u32 = 0;
    let mut expected_rst: u8 = RST0;

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if state.restart_interval != 0
                && mcus_since_restart != 0
                && mcus_since_restart % state.restart_interval as u32 == 0
            {
                // Expect a restart marker before this MCU.
                // Drain bits until the reader hits one. If we already consumed
                // the marker during fill, br.saw_rst is set.
                // Drain any partial bits (align to byte).
                br.bits = 0;
                br.nbits = 0;
                // Scan forward for a marker byte.
                while br.saw_rst.is_none() {
                    // Try to pull one byte; if it's a restart, next_byte records it.
                    let prev = br.pos;
                    match br.next_byte_with_stuff()? {
                        Some(_) => {
                            // Plain byte — unexpected, but keep looking.
                            // (Some encoders pad with zeros before the RST marker.)
                        }
                        None => {
                            if br.saw_rst.is_none() {
                                // No marker visible; maybe off the end.
                                if prev == br.pos {
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                if let Some(m) = br.saw_rst {
                    if m != expected_rst {
                        // Some encoders mis-order; accept anyway but warn with InvalidData
                        // only if you want strictness. We'll accept silently.
                    }
                    expected_rst = if expected_rst == RST7 {
                        RST0
                    } else {
                        expected_rst + 1
                    };
                    for p in prev_dc.iter_mut() {
                        *p = 0;
                    }
                    br.reset_at_restart();
                }
            }

            // For each component, decode H*V blocks.
            for (sidx, sof_idx) in sos_map.iter().enumerate() {
                let c = sof.components[*sof_idx];
                for by in 0..c.v_factor as usize {
                    for bx in 0..c.h_factor as usize {
                        let mut block = [0i32; 64];
                        decode_block(
                            &mut br,
                            dc_tables[sidx],
                            ac_tables[sidx],
                            &mut prev_dc[*sof_idx],
                            &mut block,
                        )?;
                        // Dequantise.
                        let qt = quant_tables[*sof_idx];
                        let mut fblock = [0.0f32; 64];
                        for k in 0..64 {
                            // `block` is natural-order; `qt.values` is natural-order.
                            // Multiply in f32 to avoid i32 overflow when Pq=1 (16-bit
                            // quantiser, values up to 65535) meets a coefficient at the
                            // top of its range — the IDCT input is f32 either way.
                            fblock[k] = block[k] as f32 * qt.values[k] as f32;
                        }
                        idct8x8(&mut fblock);
                        // Write 8×8 to component buffer at position (mx * H + bx, my * V + by).
                        let dst_x0 = mx * 8 * c.h_factor as usize + bx * 8;
                        let dst_y0 = my * 8 * c.v_factor as usize + by * 8;
                        let stride = comp_stride[*sof_idx];
                        let buf = &mut comp_buf[*sof_idx];
                        for j in 0..8 {
                            for i in 0..8 {
                                let v = fblock[j * 8 + i] + 128.0;
                                let px = if v <= 0.0 {
                                    0
                                } else if v >= 255.0 {
                                    255
                                } else {
                                    v.round() as u8
                                };
                                buf[(dst_y0 + j) * stride + dst_x0 + i] = px;
                            }
                        }
                    }
                }
            }
            mcus_since_restart += 1;
        }
    }

    // Build output VideoFrame.
    let out_format = pix_fmt;
    let mut planes: Vec<VideoPlane> = Vec::new();
    match out_format {
        PixelFormat::Gray8 => {
            let stride = width;
            let mut data = vec![0u8; stride * height];
            let src_stride = comp_stride[0];
            for y in 0..height {
                data[y * stride..y * stride + width]
                    .copy_from_slice(&comp_buf[0][y * src_stride..y * src_stride + width]);
            }
            planes.push(VideoPlane { stride, data });
        }
        PixelFormat::Rgb24 => {
            // Pack the three full-resolution component buffers into a
            // single packed-RGB output plane (`stride = width * 3`).
            // Sample ordering is `byte 0 = R`, `byte 1 = G`, `byte 2 = B`
            // — i.e. the SOS scan order from the encoder, which writes
            // components 1/2/3 = R/G/B. Per the SOF validation above
            // every component is sampled at H = V = 1, so no upsampling
            // is required.
            let stride = width * 3;
            let mut data = vec![0u8; stride * height];
            let src_strides = [comp_stride[0], comp_stride[1], comp_stride[2]];
            for y in 0..height {
                let off = y * stride;
                for x in 0..width {
                    data[off + x * 3] = comp_buf[0][y * src_strides[0] + x];
                    data[off + x * 3 + 1] = comp_buf[1][y * src_strides[1] + x];
                    data[off + x * 3 + 2] = comp_buf[2][y * src_strides[2] + x];
                }
            }
            planes.push(VideoPlane { stride, data });
        }
        PixelFormat::Yuv444P
        | PixelFormat::Yuv422P
        | PixelFormat::Yuv420P
        | PixelFormat::Yuv411P => {
            let (c_w, c_h) = match out_format {
                PixelFormat::Yuv444P => (width, height),
                PixelFormat::Yuv422P => (width.div_ceil(2), height),
                PixelFormat::Yuv420P => (width.div_ceil(2), height.div_ceil(2)),
                PixelFormat::Yuv411P => (width.div_ceil(4), height),
                _ => unreachable!(),
            };
            // Y plane.
            let y_stride = width;
            let mut y_data = vec![0u8; y_stride * height];
            let src_stride_y = comp_stride[0];
            for y in 0..height {
                y_data[y * y_stride..y * y_stride + width]
                    .copy_from_slice(&comp_buf[0][y * src_stride_y..y * src_stride_y + width]);
            }
            planes.push(VideoPlane {
                stride: y_stride,
                data: y_data,
            });
            // Cb, Cr planes (each same size).
            for ci in [1usize, 2] {
                let src_stride = comp_stride[ci];
                let stride = c_w;
                let mut data = vec![0u8; stride * c_h];
                for y in 0..c_h {
                    data[y * stride..y * stride + c_w]
                        .copy_from_slice(&comp_buf[ci][y * src_stride..y * src_stride + c_w]);
                }
                planes.push(VideoPlane { stride, data });
            }
        }
        _ => unreachable!(),
    }

    Ok(VideoFrame { pts, planes })
}

/// True when a 3-component baseline / sequential SOF should be decoded as
/// packed RGB (`PixelFormat::Rgb24`) instead of planar YCbCr. RGB is
/// signalled in either of two ways:
///
/// * Adobe APP14 segment with `transform = 0` — declares "no colour
///   transform applied", i.e. samples are R/G/B in scan order.
/// * Component IDs `'R' / 'G' / 'B'` (`82 / 71 / 66`) in the SOF, the
///   convention exercised by the `baseline-rgb-32x32` clean-room fixture
///   under `docs/image/jpeg/fixtures/baseline-rgb-32x32/`. The encoder's
///   `encode_jpeg_rgb24_*` entry points also emit this pair.
///
/// Returns `false` for every YUV / monochrome / unknown shape so the
/// existing YCbCr path stays the default.
fn detect_rgb_3comp(sof: &SofInfo, adobe_transform: Option<u8>) -> bool {
    if sof.components.len() != 3 {
        return false;
    }
    if adobe_transform == Some(0) {
        return true;
    }
    let ids: [u8; 3] = [
        sof.components[0].id,
        sof.components[1].id,
        sof.components[2].id,
    ];
    ids == [b'R', b'G', b'B']
}

fn decode_block(
    br: &mut BitReader<'_>,
    dc: &HuffTable,
    ac: &HuffTable,
    prev_dc: &mut i32,
    out_natural: &mut [i32; 64],
) -> Result<()> {
    // DC.
    let t = decode_huff(br, dc)? as u32;
    let dc_diff = if t == 0 {
        0
    } else {
        let bits = br.get_bits(t)? as i32;
        extend(bits, t)
    };
    *prev_dc = prev_dc.wrapping_add(dc_diff);
    out_natural[0] = *prev_dc;
    // AC: zigzag run-length decode into natural order.
    let mut k: usize = 1;
    while k < 64 {
        let rs = decode_huff(br, ac)?;
        let run = (rs >> 4) as usize;
        let size = (rs & 0x0F) as u32;
        if size == 0 {
            if run == 15 {
                // ZRL — 16 zeros.
                k += 16;
                continue;
            }
            // EOB.
            break;
        }
        k += run;
        if k >= 64 {
            return Err(Error::invalid("JPEG AC: run out of block"));
        }
        let bits = br.get_bits(size)? as i32;
        let val = extend(bits, size);
        out_natural[ZIGZAG[k]] = val;
        k += 1;
    }
    Ok(())
}

// ---- Non-interleaved baseline / extended-sequential ---------------------
//
// Sequential JPEGs may split their components across multiple SOS segments
// — one per component, each a full `[0..=63]` Ah=Al=0 scan. Decode each
// segment's blocks into the per-component coefficient accumulator, same
// shape as the progressive path. DC prediction runs independently within
// each scan (reset at the start of every SOS and at each RSTn). Once EOI
// arrives the shared `render_from_coefs` path emits the frame.
fn decode_sequential_scan_accum(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    coefs: &mut [Vec<[i32; 64]>],
) -> Result<()> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("SOS before SOF"))?;
    if sof.precision != 8 && sof.precision != 12 {
        return Err(Error::unsupported(format!(
            "scan: unsupported precision {}",
            sof.precision
        )));
    }

    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1) as usize;
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1) as usize;
    if h_max == 0 || v_max == 0 {
        return Err(Error::invalid("SOF: sampling factor = 0"));
    }
    let width = sof.width as usize;
    let height = sof.height as usize;
    let mcus_x = width.div_ceil(8 * h_max);
    let mcus_y = height.div_ceil(8 * v_max);

    let sos_map: Vec<usize> = sos
        .components
        .iter()
        .map(|sc| {
            sof.components
                .iter()
                .position(|fc| fc.id == sc.id)
                .ok_or_else(|| Error::invalid("SOS: component id not in SOF"))
        })
        .collect::<Result<Vec<_>>>()?;

    let dc_tables: Vec<&HuffTable> = sos
        .components
        .iter()
        .map(|sc| {
            state.dc_huff[sc.dc_table as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("SOS: DC Huffman table missing"))
        })
        .collect::<Result<Vec<_>>>()?;
    let ac_tables: Vec<&HuffTable> = sos
        .components
        .iter()
        .map(|sc| {
            state.ac_huff[sc.ac_table as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("SOS: AC Huffman table missing"))
        })
        .collect::<Result<Vec<_>>>()?;

    let interleaved = sos.components.len() == sof.components.len();

    // For an interleaved scan iterate the MCU grid; a non-interleaved scan
    // iterates the single component's full block grid — per T.81 E.1.4 the
    // scan's MCU is then a single data unit.
    let (scan_mcus_x, scan_mcus_y) = if interleaved {
        (mcus_x, mcus_y)
    } else {
        let sof_idx = sos_map[0];
        let c = sof.components[sof_idx];
        (mcus_x * c.h_factor as usize, mcus_y * c.v_factor as usize)
    };

    let mut br = BitReader::new(scan);
    let mut prev_dc = vec![0i32; sos.components.len()];
    let mut mcus_since_restart: u32 = 0;
    let mut expected_rst: u8 = RST0;

    for my in 0..scan_mcus_y {
        for mx in 0..scan_mcus_x {
            if state.restart_interval != 0
                && mcus_since_restart != 0
                && mcus_since_restart % state.restart_interval as u32 == 0
            {
                br.bits = 0;
                br.nbits = 0;
                while br.saw_rst.is_none() {
                    let prev = br.pos;
                    match br.next_byte_with_stuff()? {
                        Some(_) => {}
                        None => {
                            if prev == br.pos {
                                break;
                            }
                            break;
                        }
                    }
                }
                if br.saw_rst.is_some() {
                    expected_rst = if expected_rst == RST7 {
                        RST0
                    } else {
                        expected_rst + 1
                    };
                    for p in prev_dc.iter_mut() {
                        *p = 0;
                    }
                    br.reset_at_restart();
                }
            }

            if interleaved {
                for (sidx, &sof_idx) in sos_map.iter().enumerate() {
                    let c = sof.components[sof_idx];
                    let blocks_x = mcus_x * c.h_factor as usize;
                    for by in 0..c.v_factor as usize {
                        for bx in 0..c.h_factor as usize {
                            let bidx_x = mx * c.h_factor as usize + bx;
                            let bidx_y = my * c.v_factor as usize + by;
                            let bi = bidx_y * blocks_x + bidx_x;
                            decode_block(
                                &mut br,
                                dc_tables[sidx],
                                ac_tables[sidx],
                                &mut prev_dc[sidx],
                                &mut coefs[sof_idx][bi],
                            )?;
                        }
                    }
                }
            } else {
                // Non-interleaved: one block per "MCU" in this component's grid.
                let sof_idx = sos_map[0];
                let c = sof.components[sof_idx];
                let blocks_x = mcus_x * c.h_factor as usize;
                let bi = my * blocks_x + mx;
                decode_block(
                    &mut br,
                    dc_tables[0],
                    ac_tables[0],
                    &mut prev_dc[0],
                    &mut coefs[sof_idx][bi],
                )?;
            }
            mcus_since_restart += 1;
        }
    }
    Ok(())
}

// ---- SOF9 (extended sequential, arithmetic-coded) -----------------------
//
// Same MCU/component layout as Huffman SOF1, but the entropy coder is the
// Q-coder from T.81 Annex D and the per-coefficient binary-decision trees
// from §F.1.4.3 / decode procedures from §F.2.4. We accumulate the decoded
// DCT coefficients into the shared `coef_buf` (one [i32;64] per block) so
// that `render_from_coefs` handles dequant + IDCT at EOI.
//
// Restart intervals: at each RSTn boundary the spec mandates re-initialising
// both the arithmetic decoder (Initdec) and the per-component statistics
// areas, plus zeroing the DC predictor. We honour that by re-creating the
// `ArithDecoder` with a fresh slice positioned past the marker.
fn decode_arith_scan(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    coefs: &mut [Vec<[i32; 64]>],
) -> Result<()> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("SOS before SOF"))?;
    if sof.precision != 8 {
        return Err(Error::unsupported(format!(
            "arithmetic scan: precision {} (only 8 supported)",
            sof.precision
        )));
    }

    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1) as usize;
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1) as usize;
    if h_max == 0 || v_max == 0 {
        return Err(Error::invalid("SOF: sampling factor = 0"));
    }
    let width = sof.width as usize;
    let height = sof.height as usize;
    let mcus_x = width.div_ceil(8 * h_max);
    let mcus_y = height.div_ceil(8 * v_max);

    let sos_map: Vec<usize> = sos
        .components
        .iter()
        .map(|sc| {
            sof.components
                .iter()
                .position(|fc| fc.id == sc.id)
                .ok_or_else(|| Error::invalid("SOS: component id not in SOF"))
        })
        .collect::<Result<Vec<_>>>()?;

    let interleaved = sos.components.len() == sof.components.len();

    // Per-component DC + AC statistics areas. These are SHARED across all
    // blocks of a component within this scan and reset only at RSTn (per
    // §F.2.4 / F.1.4.4 initial-conditions text).
    let n_comp = sof.components.len();
    let mut dc_stats: Vec<DcStats> = (0..n_comp)
        .map(|i| {
            let mut s = DcStats::new();
            // Resolve DAC conditioning override for this component's DC
            // table destination (sos.components[?].dc_table). For an
            // interleaved scan the same DC destination may be shared across
            // SOS components; we simply look up the SOS component matching
            // this SOF index. If no SOS entry is present (i.e. component
            // not part of this scan), defaults stay.
            if let Some(sc) = sos.components.iter().find(|sc| {
                sof.components
                    .iter()
                    .position(|fc| fc.id == sc.id)
                    .map(|j| j == i)
                    .unwrap_or(false)
            }) {
                if let Some(cond) = state.arith_dc[sc.dc_table as usize].as_ref() {
                    s.l = cond.l;
                    s.u = cond.u;
                }
            }
            s
        })
        .collect();
    let mut ac_stats: Vec<AcStats> = (0..n_comp)
        .map(|i| {
            let mut s = AcStats::new();
            if let Some(sc) = sos.components.iter().find(|sc| {
                sof.components
                    .iter()
                    .position(|fc| fc.id == sc.id)
                    .map(|j| j == i)
                    .unwrap_or(false)
            }) {
                if let Some(cond) = state.arith_ac[sc.ac_table as usize].as_ref() {
                    s.kx = cond.kx;
                }
            }
            s
        })
        .collect();

    // The arithmetic decoder owns its byte source (with 0xFF 0x00 unstuff
    // and marker trapping baked in). When we hit a restart boundary, we
    // need to advance the source past the RSTn marker and re-Initdec.
    // We track the byte offset within `scan` ourselves and rebuild the
    // decoder as needed.
    let mut scan_pos = 0usize;
    let mut decoder = ArithDecoder::new(&scan[scan_pos..]);

    let mut mcus_since_restart: u32 = 0;

    let (scan_mcus_x, scan_mcus_y) = if interleaved {
        (mcus_x, mcus_y)
    } else {
        let sof_idx = sos_map[0];
        let c = sof.components[sof_idx];
        (mcus_x * c.h_factor as usize, mcus_y * c.v_factor as usize)
    };

    for my in 0..scan_mcus_y {
        for mx in 0..scan_mcus_x {
            // Restart-interval boundary?
            if state.restart_interval != 0
                && mcus_since_restart != 0
                && mcus_since_restart % state.restart_interval as u32 == 0
            {
                // The decoder may already have consumed the RSTn during a
                // Byte_in (decoder.marker() is set), or we may need to
                // scan ahead from the current cursor to find it. Either
                // way the next byte after the marker pair is the start
                // of fresh entropy data — so advance `scan_pos` and
                // re-Initdec there. Statistics + DC predictor are reset
                // per F.1.4.4.1.5 / F.2.4.4.
                scan_pos = locate_next_marker_after(scan, scan_pos);
                if scan_pos >= scan.len() {
                    return Err(Error::invalid(
                        "arithmetic scan: missing restart marker mid-scan",
                    ));
                }
                for s in dc_stats.iter_mut() {
                    s.restart_reset();
                }
                for s in ac_stats.iter_mut() {
                    s.restart_reset();
                }
                decoder = ArithDecoder::new(&scan[scan_pos..]);
            }

            if interleaved {
                for &sof_idx in sos_map.iter() {
                    let c = sof.components[sof_idx];
                    let blocks_x = mcus_x * c.h_factor as usize;
                    for by in 0..c.v_factor as usize {
                        for bx in 0..c.h_factor as usize {
                            let bidx_x = mx * c.h_factor as usize + bx;
                            let bidx_y = my * c.v_factor as usize + by;
                            let bi = bidx_y * blocks_x + bidx_x;
                            decode_arith_block(
                                &mut decoder,
                                &mut dc_stats[sof_idx],
                                &mut ac_stats[sof_idx],
                                &mut coefs[sof_idx][bi],
                            )?;
                        }
                    }
                }
            } else {
                let sof_idx = sos_map[0];
                let c = sof.components[sof_idx];
                let blocks_x = mcus_x * c.h_factor as usize;
                let bi = my * blocks_x + mx;
                decode_arith_block(
                    &mut decoder,
                    &mut dc_stats[sof_idx],
                    &mut ac_stats[sof_idx],
                    &mut coefs[sof_idx][bi],
                )?;
            }
            mcus_since_restart += 1;
        }
    }
    Ok(())
}

/// Decode one DCT block (DC diff + AC zigzag) using the arithmetic Q-coder
/// and the per-component DC/AC statistics areas. Writes natural-order
/// dequant-input coefficients into `block`.
fn decode_arith_block(
    d: &mut ArithDecoder<'_>,
    dc: &mut DcStats,
    ac: &mut AcStats,
    block: &mut [i32; 64],
) -> Result<()> {
    // Zero-fill — the AC band may exit early via EOB.
    for v in block.iter_mut() {
        *v = 0;
    }
    let diff = arith_decode_dc_diff(d, dc)?;
    dc.pred = dc.pred.wrapping_add(diff);
    block[0] = dc.pred;
    arith_decode_ac(d, ac, block, 1, 63)?;
    Ok(())
}

/// Locate the next non-stuff JPEG marker in `scan` starting at `from`.
/// Returns the byte offset just AFTER the marker pair (i.e. past 0xFF Mn).
/// Used by the arithmetic-scan restart handler to position the decoder for
/// a fresh Initdec.
fn locate_next_marker_after(scan: &[u8], from: usize) -> usize {
    let mut i = from;
    while i + 1 < scan.len() {
        if scan[i] == 0xFF && scan[i + 1] != 0x00 {
            // Skip 0xFF runs.
            let mut j = i + 1;
            while j < scan.len() && scan[j] == 0xFF {
                j += 1;
            }
            if j < scan.len() {
                return j + 1; // byte AFTER the marker code
            }
            return scan.len();
        }
        i += 1;
    }
    scan.len()
}

// ---- Progressive JPEG (SOF2) --------------------------------------------
//
// A progressive JPEG ships its DCT coefficients in multiple scans. Each
// scan specifies a spectral range `[Ss, Se]` plus successive-approximation
// shift values `Ah`/`Al`:
//
//   * Scans with `Ss == 0 && Se == 0` are DC scans. A DC scan with `Ah == 0`
//     is the first DC pass (decodes `category + sign-magnitude bits`, shifted
//     left by `Al`). A subsequent DC scan with `Ah > 0` refines one more bit.
//   * Scans with `Ss > 0` are AC scans for a single component (non-interleaved
//     per T.81) across `[Ss..=Se]`. `Ah == 0` is the first pass; `Ah > 0` is
//     a refinement scan that adds a correction bit to every already-nonzero
//     AC coefficient and, optionally, creates new coefficients from runs.
//
// The output of each scan is accumulated into the per-component coefficient
// plane. Once EOI is reached, we dequantise + inverse-DCT each block.

fn decode_progressive_scan(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    coefs: &mut [Vec<[i32; 64]>],
) -> Result<()> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("SOS before SOF"))?;

    // Validate spectral range + Ah/Al.
    if sos.ss > 63 || sos.se > 63 || sos.ss > sos.se {
        return Err(Error::invalid("progressive: invalid Ss/Se"));
    }
    let is_dc_scan = sos.ss == 0;
    if is_dc_scan && sos.se != 0 {
        return Err(Error::invalid("progressive: DC scan must have Se=0"));
    }
    if !is_dc_scan && sos.components.len() != 1 {
        return Err(Error::invalid(
            "progressive: AC scans must be non-interleaved",
        ));
    }
    if sos.ah > 13 || sos.al > 13 {
        return Err(Error::invalid("progressive: Ah/Al out of range"));
    }

    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1) as usize;
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1) as usize;
    let width = sof.width as usize;
    let height = sof.height as usize;
    let mcus_x = width.div_ceil(8 * h_max);
    let mcus_y = height.div_ceil(8 * v_max);

    // Map SOS component id → SOF index.
    let sos_map: Vec<usize> = sos
        .components
        .iter()
        .map(|sc| {
            sof.components
                .iter()
                .position(|fc| fc.id == sc.id)
                .ok_or_else(|| Error::invalid("progressive SOS: unknown component"))
        })
        .collect::<Result<Vec<_>>>()?;

    // Resolve Huffman tables per selected component.
    let dc_tables: Vec<Option<&HuffTable>> = sos
        .components
        .iter()
        .map(|sc| {
            if is_dc_scan {
                state.dc_huff[sc.dc_table as usize].as_ref()
            } else {
                None
            }
        })
        .collect();
    let ac_tables: Vec<Option<&HuffTable>> = sos
        .components
        .iter()
        .map(|sc| {
            if is_dc_scan {
                None
            } else {
                state.ac_huff[sc.ac_table as usize].as_ref()
            }
        })
        .collect();
    if is_dc_scan && dc_tables.iter().any(|t| t.is_none()) {
        return Err(Error::invalid("progressive DC: Huffman table missing"));
    }
    if !is_dc_scan && ac_tables[0].is_none() {
        return Err(Error::invalid("progressive AC: Huffman table missing"));
    }

    let mut br = BitReader::new(scan);
    let mut prev_dc = vec![0i32; sos.components.len()];
    let mut eob_run: u32 = 0;
    let mut mcus_since_restart: u32 = 0;
    let mut expected_rst: u8 = RST0;

    // How many MCUs does this scan step through?
    let (scan_mcus_x, scan_mcus_y) = if is_dc_scan && sos.components.len() > 1 {
        (mcus_x, mcus_y)
    } else {
        // Non-interleaved scan: a "MCU" is one data-unit for the component.
        let ci = sos_map[0];
        let c = sof.components[ci];
        if sos.components.len() == 1 && sof.components.len() == 1 {
            // Single-component image: MCU grid equals block grid exactly.
            (mcus_x * c.h_factor as usize, mcus_y * c.v_factor as usize)
        } else {
            // For a non-interleaved scan, iterate the component's full block grid.
            (mcus_x * c.h_factor as usize, mcus_y * c.v_factor as usize)
        }
    };

    for my in 0..scan_mcus_y {
        for mx in 0..scan_mcus_x {
            if state.restart_interval != 0
                && mcus_since_restart != 0
                && mcus_since_restart % state.restart_interval as u32 == 0
            {
                br.bits = 0;
                br.nbits = 0;
                while br.saw_rst.is_none() {
                    let prev = br.pos;
                    match br.next_byte_with_stuff()? {
                        Some(_) => {}
                        None => {
                            if prev == br.pos {
                                break;
                            }
                            break;
                        }
                    }
                }
                if br.saw_rst.is_some() {
                    expected_rst = if expected_rst == RST7 {
                        RST0
                    } else {
                        expected_rst + 1
                    };
                    for p in prev_dc.iter_mut() {
                        *p = 0;
                    }
                    eob_run = 0;
                    br.reset_at_restart();
                }
            }

            if is_dc_scan {
                // DC scan may be interleaved across components.
                if sos.components.len() > 1 {
                    // Interleaved: iterate components with H*V blocks each.
                    for (sidx, &sof_idx) in sos_map.iter().enumerate() {
                        let c = sof.components[sof_idx];
                        for by in 0..c.v_factor as usize {
                            for bx in 0..c.h_factor as usize {
                                let blocks_x = mcus_x * c.h_factor as usize;
                                let bidx_x = mx * c.h_factor as usize + bx;
                                let bidx_y = my * c.v_factor as usize + by;
                                let bi = bidx_y * blocks_x + bidx_x;
                                prog_decode_dc(
                                    &mut br,
                                    dc_tables[sidx].unwrap(),
                                    &mut prev_dc[sidx],
                                    &mut coefs[sof_idx][bi],
                                    sos.ah,
                                    sos.al,
                                )?;
                            }
                        }
                    }
                } else {
                    // Non-interleaved single-component DC scan: one block per MCU.
                    let sof_idx = sos_map[0];
                    let c = sof.components[sof_idx];
                    let blocks_x = mcus_x * c.h_factor as usize;
                    let bi = my * blocks_x + mx;
                    prog_decode_dc(
                        &mut br,
                        dc_tables[0].unwrap(),
                        &mut prev_dc[0],
                        &mut coefs[sof_idx][bi],
                        sos.ah,
                        sos.al,
                    )?;
                }
            } else {
                // AC scan is always non-interleaved.
                let sof_idx = sos_map[0];
                let c = sof.components[sof_idx];
                let blocks_x = mcus_x * c.h_factor as usize;
                let bi = my * blocks_x + mx;
                let ac = ac_tables[0].unwrap();
                if sos.ah == 0 {
                    prog_decode_ac_first(
                        &mut br,
                        ac,
                        &mut coefs[sof_idx][bi],
                        sos.ss as usize,
                        sos.se as usize,
                        sos.al,
                        &mut eob_run,
                    )?;
                } else {
                    prog_decode_ac_refine(
                        &mut br,
                        ac,
                        &mut coefs[sof_idx][bi],
                        sos.ss as usize,
                        sos.se as usize,
                        sos.al,
                        &mut eob_run,
                    )?;
                }
            }
            mcus_since_restart += 1;
        }
    }
    Ok(())
}

/// Progressive DC decode for one block. `Ah == 0` decodes a category + sign
/// bits and stores `diff << Al` into coefficient 0. `Ah > 0` reads one extra
/// bit and ORs it into the existing DC value at bit `Al`.
fn prog_decode_dc(
    br: &mut BitReader<'_>,
    dc: &HuffTable,
    prev_dc: &mut i32,
    block: &mut [i32; 64],
    ah: u8,
    al: u8,
) -> Result<()> {
    if ah == 0 {
        let t = decode_huff(br, dc)? as u32;
        let dc_diff = if t == 0 {
            0
        } else {
            let bits = br.get_bits(t)? as i32;
            extend(bits, t)
        };
        *prev_dc = prev_dc.wrapping_add(dc_diff);
        block[0] = *prev_dc << al;
    } else {
        // Successive-approximation DC refinement: read 1 bit, shift into the
        // existing DC at position `Al`.
        let bit = br.get_bits(1)? as i32;
        block[0] |= bit << al;
    }
    Ok(())
}

/// First-pass AC scan over `[ss..=se]`. Uses an EOBn symbol scheme where the
/// low-nibble `s` with high-nibble `r` can also indicate the run-length of
/// all-zero blocks (`r=15, s=0` → ZRL; `r<15, s=0` → EOBn where `n = r`).
fn prog_decode_ac_first(
    br: &mut BitReader<'_>,
    ac: &HuffTable,
    block: &mut [i32; 64],
    ss: usize,
    se: usize,
    al: u8,
    eob_run: &mut u32,
) -> Result<()> {
    if *eob_run > 0 {
        *eob_run -= 1;
        return Ok(());
    }
    let mut k = ss;
    while k <= se {
        let rs = decode_huff(br, ac)?;
        let r = (rs >> 4) as usize;
        let s = (rs & 0x0F) as u32;
        if s == 0 {
            if r != 15 {
                // EOBn: skip 2^r - 1 additional blocks.
                let extra = if r == 0 { 0 } else { br.get_bits(r as u32)? };
                *eob_run = (1u32 << r) + extra - 1;
                return Ok(());
            }
            // ZRL → 16 zeros.
            k += 16;
        } else {
            k += r;
            if k > se {
                return Err(Error::invalid("progressive AC: run out of band"));
            }
            let bits = br.get_bits(s)? as i32;
            let val = extend(bits, s) << al;
            block[ZIGZAG[k]] = val;
            k += 1;
        }
    }
    Ok(())
}

/// Successive-approximation AC refinement scan. Each already-nonzero AC in
/// `[ss..=se]` gains one correction bit; new coefficients appear at positions
/// where the first pass coded a zero but this pass decodes a fresh sign.
fn prog_decode_ac_refine(
    br: &mut BitReader<'_>,
    ac: &HuffTable,
    block: &mut [i32; 64],
    ss: usize,
    se: usize,
    al: u8,
    eob_run: &mut u32,
) -> Result<()> {
    let p1: i32 = 1 << al;
    let m1: i32 = -1 << al;

    let mut k = ss;
    if *eob_run == 0 {
        while k <= se {
            let rs = decode_huff(br, ac)?;
            let mut r = (rs >> 4) as usize;
            let s = (rs & 0x0F) as u32;
            let new_val: i32;
            if s == 0 {
                if r != 15 {
                    // EOBn: refine remaining non-zeros then schedule n-1 EOB blocks.
                    let extra = if r == 0 { 0 } else { br.get_bits(r as u32)? };
                    *eob_run = (1u32 << r) + extra;
                    break;
                }
                // ZRL: skip 16 zero-history positions (refining any current nonzeros).
                new_val = 0;
            } else if s == 1 {
                // A new nonzero: read the sign bit.
                let sign_bit = br.get_bits(1)?;
                new_val = if sign_bit == 0 { m1 } else { p1 };
            } else {
                return Err(Error::invalid("progressive AC refine: bad s"));
            }

            // Walk through zeros until we've skipped `r` zero-history positions,
            // refining any existing non-zero we pass along the way.
            loop {
                if k > se {
                    return Err(Error::invalid("progressive AC refine: k past se"));
                }
                let pos = ZIGZAG[k];
                if block[pos] != 0 {
                    // Refine existing coefficient with one additional bit.
                    let bit = br.get_bits(1)? as i32;
                    if bit != 0 && (block[pos] & p1) == 0 {
                        if block[pos] >= 0 {
                            block[pos] += p1;
                        } else {
                            block[pos] += m1;
                        }
                    }
                } else if r == 0 {
                    break;
                } else {
                    r -= 1;
                }
                k += 1;
            }

            if new_val != 0 && k <= se {
                block[ZIGZAG[k]] = new_val;
            }
            k += 1;
        }
    }

    // Handle EOB-run tail: continue refining nonzero history for the rest of
    // the band (no new coefficients allowed).
    if *eob_run > 0 {
        while k <= se {
            let pos = ZIGZAG[k];
            if block[pos] != 0 {
                let bit = br.get_bits(1)? as i32;
                if bit != 0 && (block[pos] & p1) == 0 {
                    if block[pos] >= 0 {
                        block[pos] += p1;
                    } else {
                        block[pos] += m1;
                    }
                }
            }
            k += 1;
        }
        *eob_run -= 1;
    }

    Ok(())
}

/// After all scans land, dequantise + IDCT each component block and emit a
/// VideoFrame. Shared by progressive and non-interleaved baseline paths.
fn render_from_coefs(
    state: &JpegState,
    coefs: &[Vec<[i32; 64]>],
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("render: EOI before SOF"))?;
    // 12-bit precision JPEGs take their own render path — different level
    // shift, different clamp range, 16-bit-LE output planes.
    if sof.precision == 12 {
        return render_from_coefs_12bit(state, coefs, pts);
    }
    let n_comp = sof.components.len();
    let grayscale = n_comp == 1;
    let width = sof.width as usize;
    let height = sof.height as usize;
    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1) as usize;
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1) as usize;
    let mcus_x = width.div_ceil(8 * h_max);
    let mcus_y = height.div_ceil(8 * v_max);

    // Determine output pixel format (same rules as baseline).
    let is_rgb = detect_rgb_3comp(sof, state.adobe_transform);
    let out_format = if grayscale {
        PixelFormat::Gray8
    } else if is_rgb {
        // 3-component RGB baseline / progressive: every component is
        // sampled at H = V = 1, output is packed `Rgb24` (see
        // `decode_scan` for the matching constraint and rationale).
        for c in &sof.components {
            if c.h_factor != 1 || c.v_factor != 1 {
                return Err(Error::unsupported(
                    "RGB JPEG: every component must declare H = V = 1",
                ));
            }
        }
        PixelFormat::Rgb24
    } else if n_comp == 3 {
        let y = sof.components[0];
        let cb = sof.components[1];
        let cr = sof.components[2];
        if cb.h_factor != cr.h_factor || cb.v_factor != cr.v_factor {
            return Err(Error::unsupported(
                "chroma components have different sampling factors",
            ));
        }
        if cb.h_factor != 1 || cb.v_factor != 1 {
            return Err(Error::unsupported("chroma components must have factor 1"));
        }
        match (y.h_factor, y.v_factor) {
            (1, 1) => PixelFormat::Yuv444P,
            (2, 1) => PixelFormat::Yuv422P,
            (2, 2) => PixelFormat::Yuv420P,
            (4, 1) => PixelFormat::Yuv411P,
            _ => {
                return Err(Error::unsupported(format!(
                    "luma sampling {}x{}",
                    y.h_factor, y.v_factor
                )))
            }
        }
    } else if n_comp == 4 {
        // 4-component scans: plain CMYK (no APP14, or APP14 transform=0) or
        // Adobe YCCK (APP14 transform=2). Either way the output is packed
        // `Cmyk` — YCCK / Adobe-inverted pixel transforms are applied
        // during the final pack. Any component sampling layout is allowed;
        // planes at less-than-full resolution are upsampled by nearest-
        // neighbour pixel replication.
        PixelFormat::Cmyk
    } else {
        return Err(Error::unsupported("2-component JPEG"));
    };

    // Resolve quant tables.
    let quant_tables: Vec<&QuantTable> = sof
        .components
        .iter()
        .map(|c| {
            state.quant[c.qt_id as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("quant table missing for component"))
        })
        .collect::<Result<Vec<_>>>()?;

    // Per-component decoded sample buffer (full MCU coverage).
    let mut comp_buf: Vec<Vec<u8>> = Vec::with_capacity(n_comp);
    let mut comp_stride: Vec<usize> = Vec::with_capacity(n_comp);
    for c in &sof.components {
        let w_full = mcus_x * 8 * c.h_factor as usize;
        let h_full = mcus_y * 8 * c.v_factor as usize;
        comp_buf.push(vec![0u8; w_full * h_full]);
        comp_stride.push(w_full);
    }

    for (ci, c) in sof.components.iter().enumerate() {
        let blocks_x = mcus_x * c.h_factor as usize;
        let blocks_y = mcus_y * c.v_factor as usize;
        let qt = quant_tables[ci];
        let stride = comp_stride[ci];
        let buf = &mut comp_buf[ci];
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block = &coefs[ci][by * blocks_x + bx];
                let mut fblock = [0.0f32; 64];
                for k in 0..64 {
                    // Multiply in f32 to avoid i32 overflow when Pq=1 (16-bit
                    // quantiser, values up to 65535) meets a coefficient at the
                    // top of its range — the IDCT input is f32 either way.
                    fblock[k] = block[k] as f32 * qt.values[k] as f32;
                }
                idct8x8(&mut fblock);
                let dst_x0 = bx * 8;
                let dst_y0 = by * 8;
                for j in 0..8 {
                    for i in 0..8 {
                        let v = fblock[j * 8 + i] + 128.0;
                        let px = if v <= 0.0 {
                            0
                        } else if v >= 255.0 {
                            255
                        } else {
                            v.round() as u8
                        };
                        buf[(dst_y0 + j) * stride + dst_x0 + i] = px;
                    }
                }
            }
        }
    }

    // Build output.
    let mut planes: Vec<VideoPlane> = Vec::new();
    match out_format {
        PixelFormat::Gray8 => {
            let stride = width;
            let mut data = vec![0u8; stride * height];
            let src_stride = comp_stride[0];
            for y in 0..height {
                data[y * stride..y * stride + width]
                    .copy_from_slice(&comp_buf[0][y * src_stride..y * src_stride + width]);
            }
            planes.push(VideoPlane { stride, data });
        }
        PixelFormat::Rgb24 => {
            // Pack the three full-resolution component buffers into a
            // single packed-RGB output plane. SOS scan order on the
            // encoder side is R/G/B, so plane[0]/[1]/[2] line up to the
            // byte triples here.
            let stride = width * 3;
            let mut data = vec![0u8; stride * height];
            let src_strides = [comp_stride[0], comp_stride[1], comp_stride[2]];
            for y in 0..height {
                let off = y * stride;
                for x in 0..width {
                    data[off + x * 3] = comp_buf[0][y * src_strides[0] + x];
                    data[off + x * 3 + 1] = comp_buf[1][y * src_strides[1] + x];
                    data[off + x * 3 + 2] = comp_buf[2][y * src_strides[2] + x];
                }
            }
            planes.push(VideoPlane { stride, data });
        }
        PixelFormat::Yuv444P
        | PixelFormat::Yuv422P
        | PixelFormat::Yuv420P
        | PixelFormat::Yuv411P => {
            let (c_w, c_h) = match out_format {
                PixelFormat::Yuv444P => (width, height),
                PixelFormat::Yuv422P => (width.div_ceil(2), height),
                PixelFormat::Yuv420P => (width.div_ceil(2), height.div_ceil(2)),
                PixelFormat::Yuv411P => (width.div_ceil(4), height),
                _ => unreachable!(),
            };
            let y_stride = width;
            let mut y_data = vec![0u8; y_stride * height];
            let src_stride_y = comp_stride[0];
            for y in 0..height {
                y_data[y * y_stride..y * y_stride + width]
                    .copy_from_slice(&comp_buf[0][y * src_stride_y..y * src_stride_y + width]);
            }
            planes.push(VideoPlane {
                stride: y_stride,
                data: y_data,
            });
            for ci in [1usize, 2] {
                let src_stride = comp_stride[ci];
                let stride = c_w;
                let mut data = vec![0u8; stride * c_h];
                for y in 0..c_h {
                    data[y * stride..y * stride + c_w]
                        .copy_from_slice(&comp_buf[ci][y * src_stride..y * src_stride + c_w]);
                }
                planes.push(VideoPlane { stride, data });
            }
        }
        PixelFormat::Cmyk => {
            // Pack the 4 per-component planes into a single row-major
            // `C M Y K` output, upsampling each plane to (width, height) by
            // nearest-neighbour replication where sampling factors differ.
            // Then apply the JPEG 4-component colour transform implied by
            // the Adobe APP14 marker (if any) to produce "regular" CMYK
            // where 0 = no ink.
            let stride = width * 4;
            let mut data = vec![0u8; stride * height];
            let fh: [usize; 4] = [
                sof.components[0].h_factor as usize,
                sof.components[1].h_factor as usize,
                sof.components[2].h_factor as usize,
                sof.components[3].h_factor as usize,
            ];
            let fv: [usize; 4] = [
                sof.components[0].v_factor as usize,
                sof.components[1].v_factor as usize,
                sof.components[2].v_factor as usize,
                sof.components[3].v_factor as usize,
            ];
            let transform = state.adobe_transform;
            for y in 0..height {
                for x in 0..width {
                    // Component sample index: scale output pixel by h/v_max
                    // ratio to land on the correct spot in the full-MCU
                    // sample buffer (nearest-neighbour).
                    let mut s = [0u8; 4];
                    for ci in 0..4 {
                        let sx = x * fh[ci] / h_max;
                        let sy = y * fv[ci] / v_max;
                        s[ci] = comp_buf[ci][sy * comp_stride[ci] + sx];
                    }
                    let (c, m, yy, k) = match transform {
                        Some(2) => {
                            // YCCK (Adobe). Decode YCbCr→RGB via BT.601
                            // full-range, then C/M/Y = 255 − RGB. Adobe
                            // stores the K component inverted alongside
                            // YCbCr, so flip it too. Coefficients are the
                            // BT.601 conversion constants in 16-bit
                            // fixed-point (1.40200 / 0.34414 / 0.71414 /
                            // 1.77200 multiplied by 65536 and rounded).
                            // The green expression keeps the negation
                            // inside the shift so rounding lands on the
                            // same 32768 boundary as R and B; moving the
                            // negation outside would emit g 1 LSB low.
                            let y_s = s[0] as i32;
                            let cb = s[1] as i32 - 128;
                            let cr = s[2] as i32 - 128;
                            let r = (y_s + ((cr * 91881 + 32768) >> 16)).clamp(0, 255);
                            let g =
                                (y_s + ((-22554 * cb - 46802 * cr + 32768) >> 16)).clamp(0, 255);
                            let b = (y_s + ((cb * 116130 + 32768) >> 16)).clamp(0, 255);
                            (
                                (255 - r) as u8,
                                (255 - g) as u8,
                                (255 - b) as u8,
                                255 - s[3],
                            )
                        }
                        Some(0) => {
                            // Adobe CMYK — samples stored inverted.
                            (255 - s[0], 255 - s[1], 255 - s[2], 255 - s[3])
                        }
                        _ => {
                            // No APP14 (or unexpected transform): assume
                            // plain/regular CMYK, pass components through.
                            (s[0], s[1], s[2], s[3])
                        }
                    };
                    let o = y * stride + x * 4;
                    data[o] = c;
                    data[o + 1] = m;
                    data[o + 2] = yy;
                    data[o + 3] = k;
                }
            }
            planes.push(VideoPlane { stride, data });
        }
        _ => unreachable!(),
    }

    Ok(VideoFrame { pts, planes })
}

/// 12-bit precision render path. Mirrors `render_from_coefs` but keeps
/// sample buffers in `u16` and applies a level shift of 2048 (spec value
/// for P=12 per T.81 §A.3.1). Grayscale and three-component planar YUV at
/// 4:2:0 / 4:2:2 / 4:4:4 chroma sampling are supported on output (the
/// shared `PixelFormat` enum carries `Gray12Le` / `Yuv420P12Le` /
/// `Yuv422P12Le` / `Yuv444P12Le` 16-bit-LE variants).
fn render_from_coefs_12bit(
    state: &JpegState,
    coefs: &[Vec<[i32; 64]>],
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("render: EOI before SOF"))?;
    let n_comp = sof.components.len();
    let grayscale = n_comp == 1;
    let width = sof.width as usize;
    let height = sof.height as usize;
    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1) as usize;
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1) as usize;
    let mcus_x = width.div_ceil(8 * h_max);
    let mcus_y = height.div_ceil(8 * v_max);

    let out_format = if grayscale {
        PixelFormat::Gray12Le
    } else if n_comp == 3 {
        let y = sof.components[0];
        let cb = sof.components[1];
        let cr = sof.components[2];
        if cb.h_factor != cr.h_factor || cb.v_factor != cr.v_factor {
            return Err(Error::unsupported(
                "12-bit: chroma components have different sampling factors",
            ));
        }
        if cb.h_factor != 1 || cb.v_factor != 1 {
            return Err(Error::unsupported(
                "12-bit: chroma components must have factor 1",
            ));
        }
        match (y.h_factor, y.v_factor) {
            (1, 1) => PixelFormat::Yuv444P12Le,
            (2, 1) => PixelFormat::Yuv422P12Le,
            (2, 2) => PixelFormat::Yuv420P12Le,
            _ => {
                return Err(Error::unsupported(format!(
                    "12-bit: only 4:4:4 / 4:2:2 / 4:2:0 chroma sampling supported (got {}x{})",
                    y.h_factor, y.v_factor
                )))
            }
        }
    } else {
        return Err(Error::unsupported(format!(
            "12-bit: {n_comp}-component JPEGs not supported"
        )));
    };

    let quant_tables: Vec<&QuantTable> = sof
        .components
        .iter()
        .map(|c| {
            state.quant[c.qt_id as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("quant table missing for component"))
        })
        .collect::<Result<Vec<_>>>()?;

    // Per-component u16 sample buffer (full MCU coverage).
    let mut comp_buf: Vec<Vec<u16>> = Vec::with_capacity(n_comp);
    let mut comp_stride: Vec<usize> = Vec::with_capacity(n_comp);
    for c in &sof.components {
        let w_full = mcus_x * 8 * c.h_factor as usize;
        let h_full = mcus_y * 8 * c.v_factor as usize;
        comp_buf.push(vec![0u16; w_full * h_full]);
        comp_stride.push(w_full);
    }

    for (ci, c) in sof.components.iter().enumerate() {
        let blocks_x = mcus_x * c.h_factor as usize;
        let blocks_y = mcus_y * c.v_factor as usize;
        let qt = quant_tables[ci];
        let stride = comp_stride[ci];
        let buf = &mut comp_buf[ci];
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block = &coefs[ci][by * blocks_x + bx];
                let mut fblock = [0.0f32; 64];
                for k in 0..64 {
                    // Multiply in f32 to avoid i32 overflow when Pq=1 (16-bit
                    // quantiser, values up to 65535) meets a coefficient at the
                    // top of its range — the IDCT input is f32 either way.
                    fblock[k] = block[k] as f32 * qt.values[k] as f32;
                }
                idct8x8(&mut fblock);
                let dst_x0 = bx * 8;
                let dst_y0 = by * 8;
                for j in 0..8 {
                    for i in 0..8 {
                        let v = fblock[j * 8 + i] + 2048.0;
                        let px = if v <= 0.0 {
                            0
                        } else if v >= 4095.0 {
                            4095
                        } else {
                            v.round() as u16
                        };
                        buf[(dst_y0 + j) * stride + dst_x0 + i] = px;
                    }
                }
            }
        }
    }

    // Build output — all 12-bit variants store samples as little-endian u16
    // (high 4 bits zero-padded). One plane per component.
    let mut planes: Vec<VideoPlane> = Vec::new();
    let emit_plane = |src: &[u16], src_stride: usize, w: usize, h: usize| -> VideoPlane {
        let stride = w * 2;
        let mut data = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let v = src[y * src_stride + x];
                data[y * stride + x * 2] = (v & 0xFF) as u8;
                data[y * stride + x * 2 + 1] = ((v >> 8) & 0xFF) as u8;
            }
        }
        VideoPlane { stride, data }
    };

    match out_format {
        PixelFormat::Gray12Le => {
            planes.push(emit_plane(&comp_buf[0], comp_stride[0], width, height));
        }
        PixelFormat::Yuv420P12Le | PixelFormat::Yuv422P12Le | PixelFormat::Yuv444P12Le => {
            let (c_w, c_h) = match out_format {
                PixelFormat::Yuv444P12Le => (width, height),
                PixelFormat::Yuv422P12Le => (width.div_ceil(2), height),
                PixelFormat::Yuv420P12Le => (width.div_ceil(2), height.div_ceil(2)),
                _ => unreachable!(),
            };
            planes.push(emit_plane(&comp_buf[0], comp_stride[0], width, height));
            planes.push(emit_plane(&comp_buf[1], comp_stride[1], c_w, c_h));
            planes.push(emit_plane(&comp_buf[2], comp_stride[2], c_w, c_h));
        }
        _ => unreachable!(),
    }

    Ok(VideoFrame { pts, planes })
}

// ---- Lossless JPEG (SOF3) ------------------------------------------------
//
// Lossless JPEG replaces the DCT + quant pipeline with predictive coding
// (Annex H). Each sample `Px` is predicted from its already-decoded
// neighbours Ra (left), Rb (above), and Rc (above-left) via one of seven
// predictors (1..=7), and the residual `Di = Px - Pred(Ra,Rb,Rc)` is
// Huffman-coded with the same category / magnitude scheme as a DCT DC
// coefficient. Precision ranges 2..=16 bits; the point-transform `Al`
// shifts samples by `Pt` bits on the wire.
//
// This implementation handles single-component grayscale at any precision
// in `2..=16`, and three-component (RGB-class) interleaved scans across
// every precision in `2..=16`. Output format for three-component scans:
//   * P = 8                     → packed `Rgb24` (R, G, B bytes per pixel).
//   * P = 10                    → planar `Gbrp10Le` (3 planes G, B, R).
//   * P = 12                    → planar `Gbrp12Le`.
//   * P = 14                    → planar `Gbrp14Le`.
//   * P ∈ {2..=7, 9, 11, 13, 15, 16} → packed `Rgb48Le` (16-bit LE per
//     channel, low bits carry the sample; matches the grayscale path's
//     "no-exact-width-format → widest container" policy).
// T.81 §H.1.2 specifies that
// "each component in the scan is modeled independently, using predictions
// derived from neighbouring samples of that component"; with each
// component declared `H_i = V_i = 1` (E.1.1: lossless data unit is one
// sample) the MCU at position (y, x) is exactly one residual per component
// in scan-header order. Point transform (Pt) is honoured. Restart markers
// re-initialise every component's predictor to the image-origin value as
// required by T.81.
fn decode_lossless_scan(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("SOS before SOF"))?;
    if !matches!(sos.components.len(), 1 | 3 | 4) {
        return Err(Error::unsupported(format!(
            "lossless: {} component(s) — only 1, 3 and 4 are supported",
            sos.components.len()
        )));
    }
    if sos.components.len() != sof.components.len() {
        return Err(Error::unsupported(
            "lossless: non-interleaved multi-component scans are not supported",
        ));
    }
    if sos.components.len() == 4 && sof.precision != 8 {
        return Err(Error::unsupported(
            "lossless: 4-component scans require precision 8",
        ));
    }
    let predictor = sos.ss;
    if !(1..=7).contains(&predictor) {
        return Err(Error::invalid("lossless: predictor Ss must be in 1..=7"));
    }
    let pt = sos.al as u32; // point transform
    let precision = sof.precision as u32;
    if pt >= precision {
        return Err(Error::invalid("lossless: Pt >= precision"));
    }
    let width = sof.width as usize;
    let height = sof.height as usize;
    let nc = sos.components.len();

    // Multi-component output: P = 8 → packed Rgb24, P ∈ {10, 12, 14} →
    // planar `Gbrp{10,12,14}Le`, every other precision in 2..=16 →
    // packed `Rgb48Le`. The decoder already validated `precision ∈
    // 2..=16` at SOF3 parse time (see the `SOF3` arm in `decode_to_eoi`),
    // so no upper-bound recheck is needed here.

    // Resolve one DC Huffman table per scan component (selectors are
    // independent per Annex H; the encoder may share the same table).
    let mut dc_tables: Vec<&HuffTable> = Vec::with_capacity(nc);
    for sc in &sos.components {
        let t = state.dc_huff[sc.dc_table as usize]
            .as_ref()
            .ok_or_else(|| Error::invalid("lossless: DC Huffman table missing"))?;
        dc_tables.push(t);
    }

    // All arithmetic is on the pre-Pt-shift sample range (0..2^(P-Pt)).
    let sample_bits = precision - pt;
    let sample_max: u32 = 1u32 << sample_bits;
    let sample_mask: u32 = sample_max - 1;
    let origin: u32 = 1u32 << (sample_bits - 1);

    // One sample buffer per component; each component is modeled
    // independently per H.1.2.
    let mut samples: Vec<Vec<u32>> = (0..nc).map(|_| vec![0u32; width * height]).collect();
    let mut br = BitReader::new(scan);
    let mut mcus_since_restart: u32 = 0;
    let mut expected_rst: u8 = RST0;
    let mut reset_pred = true; // true at image start and after each RSTn.

    for y in 0..height {
        for x in 0..width {
            if state.restart_interval != 0
                && mcus_since_restart != 0
                && mcus_since_restart % state.restart_interval as u32 == 0
            {
                br.bits = 0;
                br.nbits = 0;
                while br.saw_rst.is_none() {
                    let prev = br.pos;
                    match br.next_byte_with_stuff()? {
                        Some(_) => {}
                        None => {
                            if prev == br.pos {
                                break;
                            }
                            break;
                        }
                    }
                }
                if br.saw_rst.is_some() {
                    expected_rst = if expected_rst == RST7 {
                        RST0
                    } else {
                        expected_rst + 1
                    };
                    br.reset_at_restart();
                    reset_pred = true;
                }
            }

            for ci in 0..nc {
                let plane = &samples[ci];
                let pred: u32 = if reset_pred {
                    origin
                } else if y == 0 {
                    plane[y * width + x - 1]
                } else if x == 0 {
                    plane[(y - 1) * width + x]
                } else {
                    let ra = plane[y * width + x - 1];
                    let rb = plane[(y - 1) * width + x];
                    let rc = plane[(y - 1) * width + x - 1];
                    match predictor {
                        1 => ra,
                        2 => rb,
                        3 => rc,
                        4 => ra.wrapping_add(rb).wrapping_sub(rc),
                        5 => ra.wrapping_add(rb.wrapping_sub(rc) >> 1),
                        6 => rb.wrapping_add(ra.wrapping_sub(rc) >> 1),
                        7 => (ra.wrapping_add(rb)) >> 1,
                        _ => unreachable!(),
                    }
                };

                let s = decode_huff(&mut br, dc_tables[ci])? as u32;
                if s > 16 {
                    // Annex H Table H.2: SSSS = magnitude in 0..16. A
                    // Huffman table that produces a value > 16 here is
                    // either corrupt or maliciously crafted; the
                    // existing extend / get_bits machinery has no
                    // defined behaviour for it.
                    return Err(Error::invalid("lossless: SSSS > 16"));
                }
                let residual: i32 = if s == 0 {
                    0
                } else if s == 16 {
                    32_768
                } else {
                    let bits = br.get_bits(s)? as i32;
                    extend(bits, s)
                };

                let sv = ((pred as i32).wrapping_add(residual) as u32) & sample_mask;
                samples[ci][y * width + x] = sv;
            }
            reset_pred = false;
            mcus_since_restart += 1;
        }
    }

    // Four-component decode: pack the per-component planes into a single
    // row-major `C M Y K` output (packed `PixelFormat::Cmyk`, 4
    // bytes/pixel). All components are `H_i = V_i = 1` in the lossless
    // four-component path, so no upsampling is needed; samples line up
    // one-per-pixel. The Adobe APP14 colour-transform flag is honoured
    // identically to the lossy CMYK paths: `transform = 0` un-inverts the
    // Adobe-CMYK convention, `transform = 2` (YCCK) decodes YCbCr → RGB
    // via BT.601 and un-inverts K, and the no-APP14 case passes the four
    // sample bytes through as plain "regular" CMYK. `precision = 8` was
    // enforced above so each post-`<< pt` byte fits in `u8` without
    // overflow; with `pt = 0` (the default) the shift is a no-op.
    if nc == 4 {
        let stride = width * 4;
        let mut data = vec![0u8; stride * height];
        let transform = state.adobe_transform;
        for i in 0..width * height {
            let s0 = (samples[0][i] << pt) as u8;
            let s1 = (samples[1][i] << pt) as u8;
            let s2 = (samples[2][i] << pt) as u8;
            let s3 = (samples[3][i] << pt) as u8;
            let (c, m, yy, k) = match transform {
                Some(2) => {
                    // YCCK: BT.601 full-range YCbCr → RGB, then C/M/Y =
                    // 255 − RGB, K inverted.
                    let y_s = s0 as i32;
                    let cb = s1 as i32 - 128;
                    let cr = s2 as i32 - 128;
                    let r = (y_s + ((cr * 91881 + 32768) >> 16)).clamp(0, 255);
                    let g = (y_s + ((-22554 * cb - 46802 * cr + 32768) >> 16)).clamp(0, 255);
                    let b = (y_s + ((cb * 116130 + 32768) >> 16)).clamp(0, 255);
                    ((255 - r) as u8, (255 - g) as u8, (255 - b) as u8, 255 - s3)
                }
                Some(0) => {
                    // Adobe CMYK: samples stored inverted.
                    (255 - s0, 255 - s1, 255 - s2, 255 - s3)
                }
                _ => {
                    // No APP14: plain "regular" CMYK pass-through.
                    (s0, s1, s2, s3)
                }
            };
            let o = i * 4;
            data[o] = c;
            data[o + 1] = m;
            data[o + 2] = yy;
            data[o + 3] = k;
        }
        return Ok(VideoFrame {
            pts,
            planes: vec![VideoPlane { stride, data }],
        });
    }

    // Three-component decode: shape the output planes by precision.
    if nc == 3 {
        // P = 8: packed `Rgb24`, one plane, three bytes per pixel
        // ordered R, G, B (matching `oxideav_core::PixelFormat::Rgb24`'s
        // R-G-B byte layout and the encoder's component ID = 1/2/3 →
        // R/G/B scan-order convention).
        if precision == 8 {
            let stride = width * 3;
            let mut data = vec![0u8; stride * height];
            for i in 0..width * height {
                data[i * 3] = (samples[0][i] << pt) as u8;
                data[i * 3 + 1] = (samples[1][i] << pt) as u8;
                data[i * 3 + 2] = (samples[2][i] << pt) as u8;
            }
            return Ok(VideoFrame {
                pts,
                planes: vec![VideoPlane { stride, data }],
            });
        }

        // P ∈ {10, 12, 14}: planar `Gbrp{10,12,14}Le`. The encoder is
        // colour-agnostic — it preserves the caller's plane order onto
        // SOS component IDs 1, 2, 3 — and the decoder mirrors that:
        // output plane `i` carries the samples for SOS component
        // `i + 1`. Callers that want canonical G-B-R ordering (matching
        // the `oxideav_core::PixelFormat::Gbrp*Le` semantic) pass G, B,
        // R planes to the encoder; the decoder hands them back in the
        // same order. Each sample is stored as a 16-bit little-endian
        // word; the low `precision` bits carry the post-Pt-shift
        // sample, top bits zero (mirroring the grayscale Gray16Le
        // policy for P = 14).
        if matches!(precision, 10 | 12 | 14) {
            let stride = width * 2;
            let mut out_planes: Vec<VideoPlane> = Vec::with_capacity(3);
            for si in 0..3 {
                let mut data = vec![0u8; stride * height];
                for i in 0..width * height {
                    let v = (samples[si][i] << pt) as u16;
                    data[i * 2] = (v & 0xFF) as u8;
                    data[i * 2 + 1] = (v >> 8) as u8;
                }
                out_planes.push(VideoPlane { stride, data });
            }
            return Ok(VideoFrame {
                pts,
                planes: out_planes,
            });
        }

        // Every remaining precision in 2..=16 (i.e. 2..=7, 9, 11, 13,
        // 15, 16): packed `Rgb48Le`, one plane, six bytes per pixel
        // ordered c0-low, c0-high, c1-low, c1-high, c2-low, c2-high
        // where `cN` is the Nth scan-order SOS component. Samples
        // narrower than 16 bits sit in the low bits of each 16-bit
        // word — the same policy the grayscale path uses to widen
        // P = 14 into `Gray16Le`.
        let stride = width * 6;
        let mut data = vec![0u8; stride * height];
        for i in 0..width * height {
            let c0 = (samples[0][i] << pt) as u16;
            let c1 = (samples[1][i] << pt) as u16;
            let c2 = (samples[2][i] << pt) as u16;
            data[i * 6] = (c0 & 0xFF) as u8;
            data[i * 6 + 1] = (c0 >> 8) as u8;
            data[i * 6 + 2] = (c1 & 0xFF) as u8;
            data[i * 6 + 3] = (c1 >> 8) as u8;
            data[i * 6 + 4] = (c2 & 0xFF) as u8;
            data[i * 6 + 5] = (c2 >> 8) as u8;
        }
        return Ok(VideoFrame {
            pts,
            planes: vec![VideoPlane { stride, data }],
        });
    }

    // Single-component grayscale: select an output PixelFormat by
    // effective output precision (precision − Pt shifted samples fill
    // `precision` bits). Gray16Le covers every bit depth that is not
    // exactly 8 / 10 / 12.
    let out_format = match precision {
        8 => PixelFormat::Gray8,
        10 => PixelFormat::Gray10Le,
        12 => PixelFormat::Gray12Le,
        _ => PixelFormat::Gray16Le,
    };

    let plane = if out_format == PixelFormat::Gray8 {
        let stride = width;
        let mut data = vec![0u8; stride * height];
        for i in 0..width * height {
            data[i] = (samples[0][i] << pt) as u8;
        }
        VideoPlane { stride, data }
    } else {
        let stride = width * 2;
        let mut data = vec![0u8; stride * height];
        for i in 0..width * height {
            let v = (samples[0][i] << pt) as u16;
            data[i * 2] = (v & 0xFF) as u8;
            data[i * 2 + 1] = (v >> 8) as u8;
        }
        VideoPlane { stride, data }
    };

    Ok(VideoFrame {
        pts,
        planes: vec![plane],
    })
}

#[cfg(all(test, feature = "registry"))]
mod prog_tests {
    use super::*;
    use crate::jpeg::huffman::{HuffTable, STD_DC_LUMA_BITS, STD_DC_LUMA_VALS};
    use crate::jpeg::parser::SofComponent;

    fn make_state(progressive: bool, width: u16, height: u16) -> JpegState {
        let mut s = JpegState::new();
        s.sof = Some(SofInfo {
            precision: 8,
            height,
            width,
            components: vec![SofComponent {
                id: 1,
                h_factor: 1,
                v_factor: 1,
                qt_id: 0,
            }],
        });
        s.progressive = progressive;
        s.quant[0] = Some(QuantTable { values: [1u16; 64] });
        s.dc_huff[0] = Some(HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS).unwrap());
        s
    }

    /// `init_coef_buffers` should size the accumulator to the MCU grid.
    #[test]
    fn accumulator_sizes() {
        let sof = SofInfo {
            precision: 8,
            height: 16,
            width: 16,
            components: vec![
                SofComponent {
                    id: 1,
                    h_factor: 2,
                    v_factor: 2,
                    qt_id: 0,
                },
                SofComponent {
                    id: 2,
                    h_factor: 1,
                    v_factor: 1,
                    qt_id: 1,
                },
                SofComponent {
                    id: 3,
                    h_factor: 1,
                    v_factor: 1,
                    qt_id: 1,
                },
            ],
        };
        let coefs = init_coef_buffers(&sof).unwrap();
        // 16x16 image, h_max=v_max=2 → 1 MCU. Y is 2x2 blocks, chroma is 1.
        assert_eq!(coefs[0].len(), 4);
        assert_eq!(coefs[1].len(), 1);
        assert_eq!(coefs[2].len(), 1);
        for blk in &coefs[0] {
            assert!(blk.iter().all(|&v| v == 0));
        }
    }

    /// Synthesize a DC-first scan buffer for a single-component 8×8 image
    /// with DC diff = 5 and feed it through `prog_decode_dc`. `Al = 2`, so
    /// the stored coefficient should be `5 << 2 = 20`.
    #[test]
    fn dc_first_pass_shifts_by_al() {
        let t = HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS).unwrap();
        // Emit category=3 (encoded with Huffman) followed by 3-bit value 0b101 (=5).
        // Category=3 in the standard luma DC table: (per Annex K) the code for
        // symbol 3 is "100" (3 bits).
        let code = t.encode[3];
        let mut bw = ProgTestBitWriter::new();
        bw.put(code.code as u32, code.len as u32);
        bw.put(0b101, 3);
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let mut prev_dc = 0i32;
        let mut block = [0i32; 64];
        prog_decode_dc(&mut br, &t, &mut prev_dc, &mut block, 0, 2).unwrap();
        assert_eq!(prev_dc, 5);
        assert_eq!(block[0], 20);
    }

    /// DC refinement (Ah>0) just appends a single bit at position `Al`.
    #[test]
    fn dc_refine_appends_bit() {
        // Build a buffer with a single `1` bit MSB-aligned.
        let mut bw = ProgTestBitWriter::new();
        bw.put(1, 1);
        let buf = bw.finish();

        let t = HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS).unwrap();
        let mut br = BitReader::new(&buf);
        let mut prev_dc = 0i32;
        let mut block = [0i32; 64];
        block[0] = 0b1000;
        prog_decode_dc(&mut br, &t, &mut prev_dc, &mut block, 1, 0).unwrap();
        assert_eq!(block[0], 0b1001);
    }

    /// `make_state` is currently used only by the test below.
    #[test]
    fn state_helper_is_coherent() {
        let s = make_state(true, 8, 8);
        assert!(s.progressive);
        assert!(s.sof.is_some());
    }

    // ---- tiny bit-stream writer (MSB first, byte-stuffing for 0xFF) --------

    struct ProgTestBitWriter {
        out: Vec<u8>,
        bits: u32,
        nbits: u32,
    }

    impl ProgTestBitWriter {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                bits: 0,
                nbits: 0,
            }
        }
        fn put(&mut self, val: u32, n: u32) {
            self.bits = (self.bits << n) | (val & ((1u32 << n) - 1));
            self.nbits += n;
            while self.nbits >= 8 {
                self.nbits -= 8;
                let b = ((self.bits >> self.nbits) & 0xFF) as u8;
                self.out.push(b);
                if b == 0xFF {
                    self.out.push(0x00);
                }
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.nbits > 0 {
                let pad = 8 - self.nbits;
                let b = (((self.bits << pad) | ((1u32 << pad) - 1)) & 0xFF) as u8;
                self.out.push(b);
                if b == 0xFF {
                    self.out.push(0x00);
                }
            }
            self.out
        }
    }

    /// First-pass AC decode with an explicit stream: symbol (r=0,s=1) + sign
    /// bit 1, then EOB (r=0,s=0) → one nonzero coefficient at `k=1`.
    #[test]
    fn ac_first_single_coef_then_eob() {
        use crate::jpeg::huffman::{STD_AC_LUMA_BITS, STD_AC_LUMA_VALS};
        let ac = HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS).unwrap();
        let mut bw = ProgTestBitWriter::new();
        // Symbol 0x01 (r=0, s=1) encoded.
        let c = ac.encode[0x01];
        bw.put(c.code as u32, c.len as u32);
        bw.put(1, 1); // coefficient value = +1
                      // EOB (r=0, s=0) → symbol 0x00.
        let c0 = ac.encode[0x00];
        bw.put(c0.code as u32, c0.len as u32);
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let mut block = [0i32; 64];
        let mut eob = 0u32;
        prog_decode_ac_first(&mut br, &ac, &mut block, 1, 63, 0, &mut eob).unwrap();
        // Coefficient at zigzag k=1 → natural position ZIGZAG[1] = 1.
        assert_eq!(block[ZIGZAG[1]], 1);
        for i in 0..64 {
            if i != ZIGZAG[1] {
                assert_eq!(block[i], 0, "unexpected nonzero at {i}");
            }
        }
    }

    /// `prog_decode_ac_first` with Al=2 must left-shift the decoded value.
    #[test]
    fn ac_first_pass_shifts_by_al() {
        use crate::jpeg::huffman::{STD_AC_LUMA_BITS, STD_AC_LUMA_VALS};
        let ac = HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS).unwrap();
        let mut bw = ProgTestBitWriter::new();
        let c = ac.encode[0x01];
        bw.put(c.code as u32, c.len as u32);
        bw.put(1, 1); // +1
        let c0 = ac.encode[0x00];
        bw.put(c0.code as u32, c0.len as u32);
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let mut block = [0i32; 64];
        let mut eob = 0u32;
        prog_decode_ac_first(&mut br, &ac, &mut block, 1, 63, 2, &mut eob).unwrap();
        assert_eq!(block[ZIGZAG[1]], 4); // 1 << 2
    }

    /// Refinement of an already-nonzero coefficient must add one correction
    /// bit at position `Al`.
    #[test]
    fn ac_refine_extends_existing_coef() {
        use crate::jpeg::huffman::{STD_AC_LUMA_BITS, STD_AC_LUMA_VALS};
        let ac = HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS).unwrap();
        // Refinement stream: EOBn with r=0, s=0 → EOB for just this block,
        // but we also need to feed refinement bits for existing nonzeros
        // beforehand. Simplest is: EOBn r=0 (eob_run = 1), which sets up the
        // "refine remaining non-zeros" tail in the same call.
        let mut bw = ProgTestBitWriter::new();
        // Symbol 0x00: r=0, s=0 → EOB0: eob_run = 1. Then refinement tail
        // will consume one bit per existing nonzero in [ss..=se]. Block has
        // one existing nonzero at k=1; give it a `1` refinement bit.
        let c0 = ac.encode[0x00];
        bw.put(c0.code as u32, c0.len as u32);
        bw.put(1, 1); // refine bit for the existing coefficient
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let mut block = [0i32; 64];
        block[ZIGZAG[1]] = 4; // existing coef = +4 (from Al=2 first pass)
        let mut eob = 0u32;
        prog_decode_ac_refine(&mut br, &ac, &mut block, 1, 63, 1, &mut eob).unwrap();
        // 4 (binary 0b100) + 0b10 (1 << Al=1) = 0b110 = 6.
        assert_eq!(block[ZIGZAG[1]], 6);
    }
}

#[cfg(all(test, feature = "registry"))]
mod non_interleaved_tests {
    use super::*;
    use crate::encoder::{encode_jpeg, encode_jpeg_non_interleaved};
    use crate::registry::make_decoder;
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

    fn make_frame(w: u32, h: u32, pix: PixelFormat) -> VideoFrame {
        let (cw, ch) = match pix {
            PixelFormat::Yuv444P => (w, h),
            PixelFormat::Yuv422P => (w.div_ceil(2), h),
            PixelFormat::Yuv420P => (w.div_ceil(2), h.div_ceil(2)),
            _ => panic!("unsupported"),
        };
        let mut y = vec![0u8; (w * h) as usize];
        for j in 0..h as usize {
            for i in 0..w as usize {
                y[j * w as usize + i] = (((i + j * 3) * 7) % 255) as u8;
            }
        }
        let mut cb = vec![0u8; (cw * ch) as usize];
        let mut cr = vec![0u8; (cw * ch) as usize];
        for j in 0..ch as usize {
            for i in 0..cw as usize {
                cb[j * cw as usize + i] = ((128 + i as i32 / 2) as u8).clamp(0, 255);
                cr[j * cw as usize + i] = ((128 + j as i32 / 2) as u8).clamp(0, 255);
            }
        }
        VideoFrame {
            pts: Some(0),
            planes: vec![
                VideoPlane {
                    stride: w as usize,
                    data: y,
                },
                VideoPlane {
                    stride: cw as usize,
                    data: cb,
                },
                VideoPlane {
                    stride: cw as usize,
                    data: cr,
                },
            ],
        }
    }

    /// A non-interleaved-scan encoding of the same frame must yield the
    /// same decoded pixels as the interleaved encoding, since the scan
    /// ordering only affects how blocks are transported — not the coded
    /// coefficient values.
    fn assert_matches_interleaved(w: u32, h: u32, pix: PixelFormat) {
        let frame = make_frame(w, h, pix);
        let base = encode_jpeg(&frame, w, h, pix, 75).expect("interleaved encode");
        let non =
            encode_jpeg_non_interleaved(&frame, w, h, pix, 75).expect("non-interleaved encode");

        // The non-interleaved stream must contain 3 SOS segments.
        let sos_count = non.windows(2).filter(|w| w == &[0xFF, 0xDA]).count();
        assert_eq!(sos_count, 3, "expected 3 SOS segments");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec_a = make_decoder(&dec_params).unwrap();
        let mut dec_b = make_decoder(&dec_params).unwrap();
        dec_a
            .send_packet(&Packet::new(0, TimeBase::new(1, 30), base))
            .unwrap();
        dec_b
            .send_packet(&Packet::new(0, TimeBase::new(1, 30), non))
            .unwrap();
        let Frame::Video(va) = dec_a.receive_frame().unwrap() else {
            panic!()
        };
        let Frame::Video(vb) = dec_b.receive_frame().unwrap() else {
            panic!()
        };
        assert_eq!(va.planes.len(), vb.planes.len());
        for (pi, (pa, pb)) in va.planes.iter().zip(vb.planes.iter()).enumerate() {
            assert_eq!(
                pa.data, pb.data,
                "plane {pi} mismatch between interleaved and non-interleaved decodes"
            );
        }
    }

    #[test]
    fn non_interleaved_yuv420p_matches_interleaved() {
        assert_matches_interleaved(32, 16, PixelFormat::Yuv420P);
    }

    #[test]
    fn non_interleaved_yuv422p_matches_interleaved() {
        assert_matches_interleaved(24, 24, PixelFormat::Yuv422P);
    }

    #[test]
    fn non_interleaved_yuv444p_matches_interleaved() {
        assert_matches_interleaved(16, 16, PixelFormat::Yuv444P);
    }
}

#[cfg(all(test, feature = "registry"))]
mod cmyk_tests {
    use crate::encoder::{encode_jpeg_cmyk_1111, encode_jpeg_progressive_cmyk_1111};
    use crate::registry::make_decoder;
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

    /// PSNR across all bytes of two buffers, treating them as 8-bit samples.
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

    fn make_cmyk_planes(w: usize, h: usize) -> [Vec<u8>; 4] {
        // Deterministic gradients — one per component — so the roundtrip
        // has enough variation across the plane to catch alignment /
        // upsampling errors, while staying smooth enough to survive
        // Q=85 DCT losses.
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

    /// Decoder should output a `Cmyk` frame for a 4-component JPEG and,
    /// with no APP14 marker, pass samples through unchanged (up to DCT
    /// quantisation noise).
    #[test]
    fn cmyk_plain_roundtrip() {
        let w = 32u32;
        let h = 16u32;
        let planes = make_cmyk_planes(w as usize, h as usize);
        let refs: [&[u8]; 4] = [&planes[0], &planes[1], &planes[2], &planes[3]];
        let strides = [w as usize; 4];
        let data =
            encode_jpeg_cmyk_1111(w, h, &refs, &strides, 90, None).expect("encode plain CMYK");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!()
        };
        assert_eq!(v.planes.len(), 1);
        assert_eq!(v.planes[0].stride, (w * 4) as usize);

        // Unpack each component out of the packed plane and compare to
        // the source. PSNR threshold is set to a realistic roundtrip
        // target for Q=90.
        for (ci, src) in planes.iter().enumerate() {
            let mut got = Vec::with_capacity(src.len());
            for j in 0..h as usize {
                for i in 0..w as usize {
                    got.push(v.planes[0].data[j * v.planes[0].stride + i * 4 + ci]);
                }
            }
            let p = psnr(src, &got);
            assert!(p >= 30.0, "component {ci} PSNR too low: {p:.2}");
        }
    }

    /// Adobe CMYK (APP14 transform=0) — the encoder inverts samples on the
    /// way out; the decoder must invert them back so downstream bytes
    /// match the original "regular"-convention inputs.
    #[test]
    fn cmyk_adobe_inverted_roundtrip() {
        let w = 16u32;
        let h = 16u32;
        let planes = make_cmyk_planes(w as usize, h as usize);
        let refs: [&[u8]; 4] = [&planes[0], &planes[1], &planes[2], &planes[3]];
        let strides = [w as usize; 4];
        let data =
            encode_jpeg_cmyk_1111(w, h, &refs, &strides, 90, Some(0)).expect("encode Adobe CMYK");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!()
        };

        for (ci, src) in planes.iter().enumerate() {
            let mut got = Vec::with_capacity(src.len());
            for j in 0..h as usize {
                for i in 0..w as usize {
                    got.push(v.planes[0].data[j * v.planes[0].stride + i * 4 + ci]);
                }
            }
            let p = psnr(src, &got);
            assert!(p >= 30.0, "Adobe CMYK component {ci} PSNR too low: {p:.2}");
        }
    }

    /// Adobe YCCK (APP14 transform=2): the encoder stores YCbCr normally
    /// and K inverted. The decoder should undo both the YCbCr→RGB→CMY
    /// transform and the K inversion to recover the original C/M/Y/K
    /// samples (modulo colour-space roundtrip loss for C/M/Y and DCT
    /// loss for K).
    #[test]
    fn ycck_roundtrip_k_plane_matches() {
        let w = 16u32;
        let h = 16u32;
        // For YCCK we interpret the four input planes as (Y, Cb, Cr, K)
        // — the encoder will inject APP14 transform=2 and invert K.
        // We only verify K recovery here: CMY values go through a
        // YCbCr→RGB lossy transform, while K is straightforward.
        let mut yp = vec![0u8; (w * h) as usize];
        let mut cb = vec![128u8; (w * h) as usize];
        let mut cr = vec![128u8; (w * h) as usize];
        let mut k = vec![0u8; (w * h) as usize];
        for j in 0..h as usize {
            for i in 0..w as usize {
                let idx = j * w as usize + i;
                yp[idx] = 128;
                cb[idx] = 128;
                cr[idx] = 128;
                k[idx] = ((i * 255 / (w as usize - 1).max(1)) as u32).min(255) as u8;
            }
        }
        let refs: [&[u8]; 4] = [&yp, &cb, &cr, &k];
        let strides = [w as usize; 4];
        let data = encode_jpeg_cmyk_1111(w, h, &refs, &strides, 90, Some(2)).expect("encode YCCK");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!()
        };

        // YCbCr=(128,128,128) → RGB≈(128,128,128) → CMY≈(127,127,127).
        // That plus the K gradient should produce a recoverable K plane.
        let mut got_k = Vec::with_capacity((w * h) as usize);
        for j in 0..h as usize {
            for i in 0..w as usize {
                got_k.push(v.planes[0].data[j * v.planes[0].stride + i * 4 + 3]);
            }
        }
        let p = psnr(&k, &got_k);
        assert!(p >= 30.0, "YCCK K plane PSNR too low: {p:.2}");
    }

    // ---- Progressive (SOF2) 4-component CMYK / YCCK roundtrip --------
    //
    // T.81 §G.1.1 permits the progressive coding process at every
    // component-count the spec admits (Nf ∈ 1..=4). The decoder was
    // previously rejecting SOF2 with `Nf = 4` even though
    // `decode_progressive_scan` is component-count agnostic
    // (interleaved DC walks every SOS component, AC scans are always
    // non-interleaved per spec), the coefficient accumulator already
    // sizes for 4 components, and `render_from_coefs` already produces
    // a packed `Cmyk` plane for `Nf = 4` honouring the Adobe APP14
    // colour-transform flag. The three tests below mirror the
    // baseline (`encode_jpeg_cmyk_1111`-based) CMYK tests above but
    // route the bitstream through `encode_jpeg_progressive_cmyk_1111`
    // (SOF2 + 9-scan spectral-selection decomposition: 1 interleaved
    // DC + 4×2 per-component AC bands).

    /// Plain CMYK (no APP14): every component should round-trip through
    /// the progressive path at PSNR ≥ 30 dB at Q=90.
    #[test]
    fn cmyk_progressive_plain_roundtrip() {
        let w = 32u32;
        let h = 16u32;
        let planes = make_cmyk_planes(w as usize, h as usize);
        let refs: [&[u8]; 4] = [&planes[0], &planes[1], &planes[2], &planes[3]];
        let strides = [w as usize; 4];
        let data = encode_jpeg_progressive_cmyk_1111(w, h, &refs, &strides, 90, None)
            .expect("encode progressive plain CMYK");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!()
        };
        assert_eq!(v.planes.len(), 1, "expected one packed Cmyk plane");
        assert_eq!(
            v.planes[0].stride,
            (w * 4) as usize,
            "packed Cmyk row stride = 4 × width"
        );

        for (ci, src) in planes.iter().enumerate() {
            let mut got = Vec::with_capacity(src.len());
            for j in 0..h as usize {
                for i in 0..w as usize {
                    got.push(v.planes[0].data[j * v.planes[0].stride + i * 4 + ci]);
                }
            }
            let p = psnr(src, &got);
            assert!(
                p >= 30.0,
                "progressive plain CMYK component {ci} PSNR too low: {p:.2}"
            );
        }
    }

    /// Adobe CMYK (APP14 transform=0) — encoder inverts every component
    /// on the wire, decoder un-inverts on output. Progressive path must
    /// match the baseline path's behaviour byte-for-byte modulo DCT
    /// quantisation noise.
    #[test]
    fn cmyk_progressive_adobe_inverted_roundtrip() {
        let w = 16u32;
        let h = 16u32;
        let planes = make_cmyk_planes(w as usize, h as usize);
        let refs: [&[u8]; 4] = [&planes[0], &planes[1], &planes[2], &planes[3]];
        let strides = [w as usize; 4];
        let data = encode_jpeg_progressive_cmyk_1111(w, h, &refs, &strides, 90, Some(0))
            .expect("encode progressive Adobe CMYK");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!()
        };

        for (ci, src) in planes.iter().enumerate() {
            let mut got = Vec::with_capacity(src.len());
            for j in 0..h as usize {
                for i in 0..w as usize {
                    got.push(v.planes[0].data[j * v.planes[0].stride + i * 4 + ci]);
                }
            }
            let p = psnr(src, &got);
            assert!(
                p >= 30.0,
                "progressive Adobe CMYK component {ci} PSNR too low: {p:.2}"
            );
        }
    }

    /// Adobe YCCK (APP14 transform=2): K plane recovery should mirror
    /// the baseline path. As in the baseline test we only verify K
    /// (C/M/Y go through a lossy YCbCr→RGB round-trip irrelevant to
    /// the scan-decomposition change being tested).
    #[test]
    fn ycck_progressive_k_plane_matches() {
        let w = 16u32;
        let h = 16u32;
        let mut yp = vec![0u8; (w * h) as usize];
        let mut cb = vec![128u8; (w * h) as usize];
        let mut cr = vec![128u8; (w * h) as usize];
        let mut k = vec![0u8; (w * h) as usize];
        for j in 0..h as usize {
            for i in 0..w as usize {
                let idx = j * w as usize + i;
                yp[idx] = 128;
                cb[idx] = 128;
                cr[idx] = 128;
                k[idx] = ((i * 255 / (w as usize - 1).max(1)) as u32).min(255) as u8;
            }
        }
        let refs: [&[u8]; 4] = [&yp, &cb, &cr, &k];
        let strides = [w as usize; 4];
        let data = encode_jpeg_progressive_cmyk_1111(w, h, &refs, &strides, 90, Some(2))
            .expect("encode progressive YCCK");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!()
        };

        let mut got_k = Vec::with_capacity((w * h) as usize);
        for j in 0..h as usize {
            for i in 0..w as usize {
                got_k.push(v.planes[0].data[j * v.planes[0].stride + i * 4 + 3]);
            }
        }
        let p = psnr(&k, &got_k);
        assert!(p >= 30.0, "progressive YCCK K plane PSNR too low: {p:.2}");
    }

    /// SOF2 frames with `Nf = 4` and `P = 12` must still be rejected with
    /// `Unsupported` — the workspace `PixelFormat` enum has no 12-bit
    /// CMYK variant. Encoder helper isn't available for that combo, so
    /// we hand-craft a minimal SOF2 segment with `P = 12, Nf = 4`
    /// preceded by SOI and verify the parser path returns Unsupported
    /// before any scan is read.
    #[test]
    fn cmyk_progressive_p12_rejected() {
        // SOI + SOF2 with P=12, 4 components, each H=V=1, all qt=0.
        // We don't need DQT/DHT/SOS — the SOF2 rejection happens at
        // segment-parse time, before any scan walks.
        let mut data: Vec<u8> = vec![
            0xFF,
            crate::jpeg::markers::SOI,
            // SOF2 segment: marker + length + payload.
            0xFF,
            crate::jpeg::markers::SOF2,
        ];
        // Length = 2 (length itself) + 1 (P) + 2 (Y) + 2 (X) + 1 (Nf) +
        //          4 × 3 (per-component triplet) = 20.
        let length: u16 = 2 + 1 + 2 + 2 + 1 + 4 * 3;
        data.extend_from_slice(&length.to_be_bytes());
        data.push(12); // precision
        data.extend_from_slice(&16u16.to_be_bytes()); // height
        data.extend_from_slice(&16u16.to_be_bytes()); // width
        data.push(4); // Nf
        for id in 1u8..=4 {
            data.push(id);
            data.push(0x11); // H=V=1
            data.push(0); // qt id
        }
        // EOI — give the marker walker something legal to terminate on,
        // although the SOF2 reject should fire before EOI is reached.
        data.push(0xFF);
        data.push(crate::jpeg::markers::EOI);

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(16);
        dec_params.height = Some(16);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let err = dec.receive_frame().expect_err("expected Unsupported");
        assert!(
            matches!(err, oxideav_core::Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }
}

#[cfg(all(test, feature = "registry"))]
mod precision_12_tests {
    use crate::encoder::{
        encode_grayscale_jpeg_12bit, encode_yuv_jpeg_12bit, encode_yuv_jpeg_progressive_12bit,
    };
    use crate::registry::make_decoder;
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};

    /// Decoder should accept a 12-bit precision grayscale JPEG and output
    /// `Gray12Le` with 16-bit LE samples. We feed a smooth gradient
    /// centred near 2048 so Huffman categories stay small (we reuse the
    /// 8-bit Annex K Huffman tables).
    #[test]
    fn gray_12bit_roundtrip() {
        let w = 16u32;
        let h = 16u32;
        // Values in [2000, 2100] keep DC/AC categories well under 12.
        let mut samples = vec![0u16; (w * h) as usize];
        for j in 0..h as usize {
            for i in 0..w as usize {
                samples[j * w as usize + i] = 2000 + ((i + j) as u16);
            }
        }
        let stride = w as usize;
        let data =
            encode_grayscale_jpeg_12bit(w, h, &samples, stride, 90).expect("encode 12-bit gray");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!()
        };
        assert_eq!(v.planes.len(), 1);
        assert_eq!(v.planes[0].stride, (w * 2) as usize);

        // Unpack LE u16 samples and check they're within a small tolerance
        // of the originals (DCT + quant at Q=90 introduces a few LSBs of
        // noise).
        let mut got = Vec::with_capacity((w * h) as usize);
        for j in 0..h as usize {
            for i in 0..w as usize {
                let o = j * v.planes[0].stride + i * 2;
                got.push(v.planes[0].data[o] as u16 | ((v.planes[0].data[o + 1] as u16) << 8));
            }
        }
        for (orig, dec) in samples.iter().zip(got.iter()) {
            let diff = (*orig as i32 - *dec as i32).abs();
            assert!(diff < 16, "12-bit roundtrip diff too large: {diff}");
        }
    }

    /// Build a smooth 12-bit YUV test image whose DCT/AC categories stay
    /// well under 11 (so the encoder helper can reuse the 8-bit Annex K
    /// Huffman tables without overflowing). Y / Cb / Cr are gentle
    /// gradients centred near 2048.
    fn build_yuv_12bit(
        w: usize,
        h: usize,
        h_factor: u8,
        v_factor: u8,
    ) -> (Vec<u16>, Vec<u16>, Vec<u16>, usize, usize) {
        let c_w = w.div_ceil(h_factor as usize);
        let c_h = h.div_ceil(v_factor as usize);
        let mut y = vec![0u16; w * h];
        let mut cb = vec![0u16; c_w * c_h];
        let mut cr = vec![0u16; c_w * c_h];
        for j in 0..h {
            for i in 0..w {
                // Luma in [2000..2100].
                y[j * w + i] = 2000 + ((i + j) as u16);
            }
        }
        for j in 0..c_h {
            for i in 0..c_w {
                // Chroma near the 2048 mid-point with a different slope so
                // a wrong plane lookup would be obvious.
                cb[j * c_w + i] = 2040 + ((i ^ j) as u16 & 0x07);
                cr[j * c_w + i] = 2056 + (((i + 2 * j) as u16) & 0x07);
            }
        }
        (y, cb, cr, c_w, c_h)
    }

    fn unpack_le_u16_plane(data: &[u8], stride: usize, w: usize, h: usize) -> Vec<u16> {
        let mut out = Vec::with_capacity(w * h);
        for j in 0..h {
            for i in 0..w {
                let o = j * stride + i * 2;
                out.push(data[o] as u16 | ((data[o + 1] as u16) << 8));
            }
        }
        out
    }

    fn assert_plane_close(label: &str, orig: &[u16], dec: &[u16]) {
        assert_eq!(orig.len(), dec.len(), "{label}: length mismatch");
        for (k, (o, d)) in orig.iter().zip(dec.iter()).enumerate() {
            let diff = (*o as i32 - *d as i32).abs();
            assert!(
                diff < 24,
                "{label}: idx {k} diff too large (orig={o} dec={d})"
            );
        }
    }

    fn run_yuv_12bit_roundtrip(
        w: u32,
        h: u32,
        h_factor: u8,
        v_factor: u8,
        expect_pix: PixelFormat,
    ) {
        let (y, cb, cr, c_w, c_h) = build_yuv_12bit(w as usize, h as usize, h_factor, v_factor);
        let data = encode_yuv_jpeg_12bit(w, h, &y, &cb, &cr, h_factor, v_factor, 90)
            .expect("encode 12-bit yuv");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("decoder did not emit a video frame")
        };
        assert_eq!(v.planes.len(), 3, "expected three planes");
        // Each plane is 16-bit LE, so stride is the plane width × 2.
        assert_eq!(v.planes[0].stride, (w * 2) as usize, "Y stride");
        assert_eq!(v.planes[1].stride, c_w * 2, "Cb stride");
        assert_eq!(v.planes[2].stride, c_w * 2, "Cr stride");

        // The crate-public Frame doesn't carry a pixel-format tag (the
        // shape-only slim VideoFrame), but the per-plane geometry uniquely
        // identifies the format. Use `expect_pix` to size the chroma plane
        // for cross-checks.
        let _ = expect_pix;

        let got_y = unpack_le_u16_plane(
            &v.planes[0].data,
            v.planes[0].stride,
            w as usize,
            h as usize,
        );
        let got_cb = unpack_le_u16_plane(&v.planes[1].data, v.planes[1].stride, c_w, c_h);
        let got_cr = unpack_le_u16_plane(&v.planes[2].data, v.planes[2].stride, c_w, c_h);

        assert_plane_close("Y", &y, &got_y);
        assert_plane_close("Cb", &cb, &got_cb);
        assert_plane_close("Cr", &cr, &got_cr);
    }

    /// 12-bit 4:4:4 YUV (`Yuv444P12Le`) end-to-end roundtrip via the
    /// crate's own encoder helper and the registry-side decoder.
    #[test]
    fn yuv444_12bit_roundtrip() {
        run_yuv_12bit_roundtrip(16, 16, 1, 1, PixelFormat::Yuv444P12Le);
    }

    /// 12-bit 4:2:2 YUV (`Yuv422P12Le`) end-to-end roundtrip.
    #[test]
    fn yuv422_12bit_roundtrip() {
        run_yuv_12bit_roundtrip(16, 16, 2, 1, PixelFormat::Yuv422P12Le);
    }

    /// 12-bit 4:2:0 YUV (`Yuv420P12Le`) is the previously-supported case;
    /// keep it in the new roundtrip-test surface as a regression guard
    /// against the render-path refactor.
    #[test]
    fn yuv420_12bit_roundtrip_via_yuv_helper() {
        run_yuv_12bit_roundtrip(16, 16, 2, 2, PixelFormat::Yuv420P12Le);
    }

    /// Non-2x luma sampling at P=12 (e.g. 4:1:1) is still rejected with
    /// `Unsupported` — the matrix only covers the three common YUV
    /// subsamplings the workspace `PixelFormat` enum carries at 12 bits.
    #[test]
    fn yuv_12bit_4x1_luma_rejected() {
        let w = 16u32;
        let h = 16u32;
        let (y, cb, cr, _c_w, _c_h) = build_yuv_12bit(w as usize, h as usize, 4, 1);
        let data =
            encode_yuv_jpeg_12bit(w, h, &y, &cb, &cr, 4, 1, 90).expect("encode 12-bit yuv 4:1:1");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let err = dec.receive_frame().expect_err("expected Unsupported");
        assert!(
            matches!(err, oxideav_core::Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }

    // ---- Progressive (SOF2) 12-bit precision roundtrip ------------------
    //
    // T.81 §G.1.1 permits the progressive coding process at precision 8 or
    // 12. The decoder originally rejected SOF2 at P=12 with `Unsupported`
    // even though `init_coef_buffers` already allocated 12-bit-shaped
    // accumulator planes and `render_from_coefs` already routed P=12 to
    // `render_from_coefs_12bit`. The three tests below cover the three
    // YUV subsamplings the workspace `PixelFormat` enum carries at 12
    // bits (4:4:4 / 4:2:2 / 4:2:0). Each one builds a smooth YUV image
    // centred near the 2048 midpoint, encodes through
    // `encode_yuv_jpeg_progressive_12bit` (SOF2 with `P = 12`,
    // spectral-selection-only scan layout: interleaved DC + Y/Cb/Cr AC
    // bands [1..=5] then [6..=63]), then decodes back to a planar
    // 12-bit-LE u16 frame and asserts per-sample closeness against the
    // originals.

    fn run_progressive_yuv_12bit_roundtrip(
        w: u32,
        h: u32,
        h_factor: u8,
        v_factor: u8,
        expect_pix: PixelFormat,
    ) {
        let (y, cb, cr, c_w, c_h) = build_yuv_12bit(w as usize, h as usize, h_factor, v_factor);
        let data = encode_yuv_jpeg_progressive_12bit(w, h, &y, &cb, &cr, h_factor, v_factor, 90)
            .expect("encode 12-bit progressive yuv");

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("decoder did not emit a video frame")
        };
        assert_eq!(v.planes.len(), 3, "expected three planes");
        assert_eq!(v.planes[0].stride, (w * 2) as usize, "Y stride");
        assert_eq!(v.planes[1].stride, c_w * 2, "Cb stride");
        assert_eq!(v.planes[2].stride, c_w * 2, "Cr stride");
        let _ = expect_pix;

        let got_y = unpack_le_u16_plane(
            &v.planes[0].data,
            v.planes[0].stride,
            w as usize,
            h as usize,
        );
        let got_cb = unpack_le_u16_plane(&v.planes[1].data, v.planes[1].stride, c_w, c_h);
        let got_cr = unpack_le_u16_plane(&v.planes[2].data, v.planes[2].stride, c_w, c_h);

        assert_plane_close("Y", &y, &got_y);
        assert_plane_close("Cb", &cb, &got_cb);
        assert_plane_close("Cr", &cr, &got_cr);
    }

    #[test]
    fn yuv444_12bit_progressive_roundtrip() {
        run_progressive_yuv_12bit_roundtrip(16, 16, 1, 1, PixelFormat::Yuv444P12Le);
    }

    #[test]
    fn yuv422_12bit_progressive_roundtrip() {
        run_progressive_yuv_12bit_roundtrip(16, 16, 2, 1, PixelFormat::Yuv422P12Le);
    }

    #[test]
    fn yuv420_12bit_progressive_roundtrip() {
        run_progressive_yuv_12bit_roundtrip(16, 16, 2, 2, PixelFormat::Yuv420P12Le);
    }
}

#[cfg(all(test, feature = "registry"))]
mod lossless_tests {
    use crate::encoder::encode_lossless_grayscale_jpeg_8bit;
    use crate::registry::make_decoder;
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

    /// Lossless JPEG is, by definition, bit-exact. The decoder should
    /// recover every sample of the input.
    #[test]
    fn lossless_8bit_gray_exact_roundtrip() {
        let w = 24u32;
        let h = 16u32;
        let mut samples = vec![0u8; (w * h) as usize];
        for j in 0..h as usize {
            for i in 0..w as usize {
                // Mix of smooth gradient + a bit of texture to exercise
                // non-zero residuals through multiple Huffman categories.
                samples[j * w as usize + i] =
                    ((i as i32 * 3 + j as i32 * 5 + ((i ^ j) as i32 & 7)) & 0xFF) as u8;
            }
        }
        let data = encode_lossless_grayscale_jpeg_8bit(w, h, &samples, w as usize)
            .expect("encode lossless");
        // SOF3 marker must appear in the output bitstream.
        assert!(
            data.windows(2).any(|x| x == [0xFF, 0xC3]),
            "SOF3 marker missing from lossless output"
        );

        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(w);
        dec_params.height = Some(h);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data))
            .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!()
        };
        assert_eq!(v.planes.len(), 1);
        assert_eq!(v.planes[0].stride, w as usize);

        for j in 0..h as usize {
            for i in 0..w as usize {
                let got = v.planes[0].data[j * v.planes[0].stride + i];
                let want = samples[j * w as usize + i];
                assert_eq!(got, want, "mismatch at ({i},{j})");
            }
        }
    }

    /// Hierarchical (SOF5/6/7) and SOF10..15 arithmetic variants must
    /// still be rejected with Unsupported — make sure we didn't widen
    /// the SOF accept matcher too far while adding SOF3 or SOF9.
    /// SOF9 (extended sequential arithmetic) is now handled separately.
    #[test]
    fn hierarchical_arithmetic_still_rejected() {
        // Hand-construct a minimal stream with SOF5 (hierarchical). The
        // walker stops at that marker with an Unsupported error; nothing
        // further in the stream matters.
        let bytes = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC5, 0x00, 0x08, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, // SOF5 stub
            0xFF, 0xD9, // EOI
        ];
        let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
        dec_params.width = Some(8);
        dec_params.height = Some(8);
        let mut dec = make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), bytes))
            .unwrap();
        let err = dec.receive_frame().expect_err("expected decode error");
        assert!(
            matches!(err, oxideav_core::Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }
}
