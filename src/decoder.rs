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
    decode_ac as arith_decode_ac, decode_ac_refine as arith_decode_ac_refine,
    decode_dc_diff as arith_decode_dc_diff, decode_fixed_bit as arith_decode_fixed_bit,
    decode_lossless_diff as arith_decode_lossless_diff, AcRefineStats, AcStats, ArithDecoder,
    DcStats, LosslessStats,
};
use crate::jpeg::dct::idct8x8;
use crate::jpeg::huffman::{parse_dht, HuffTable};
use crate::jpeg::markers::{self, *};
use crate::jpeg::parser::{
    parse_dac, parse_dnl, parse_dri, parse_sof, parse_sos, MarkerWalker, SofInfo, SosInfo,
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
    /// True when SOF3 or SOF11 (lossless) was parsed. Mutually exclusive
    /// with `progressive` / `seq_accum` — lossless JPEGs use
    /// predictor-based coding rather than DCT and take their own scan
    /// decoder. SOF11 additionally sets `lossless_arith`.
    lossless: bool,
    /// True when SOF11 (lossless, arithmetic-coded) was parsed. The
    /// lossless scan dispatcher takes the §H.1.2.3 Q-coder path instead
    /// of the Annex H Huffman path.
    lossless_arith: bool,
    /// Adobe APP14 transform flag. `None` if no Adobe marker was seen;
    /// `Some(0)` for direct (CMYK / RGB, samples stored as-is but
    /// Adobe-inverted for CMYK); `Some(1)` for YCbCr (3-component);
    /// `Some(2)` for YCCK (4-component Adobe colour transform).
    adobe_transform: Option<u8>,
    /// True when SOF9 (extended sequential, arithmetic-coded) was parsed.
    /// Mutually exclusive with `progressive` / `lossless`. The scan
    /// dispatcher takes the arithmetic Q-coder path instead of Huffman.
    arithmetic: bool,
    /// True when SOF10 (progressive, arithmetic-coded) was parsed. The
    /// scan dispatcher takes the §G.1.3 progressive Q-coder path; the
    /// coefficient accumulator + EOI render are shared with SOF2.
    progressive_arith: bool,
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
            lossless_arith: false,
            adobe_transform: None,
            arithmetic: false,
            progressive_arith: false,
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

/// Shared frame-header constraints for the lossless processes (SOF3
/// Huffman and SOF11 arithmetic — T.81 Annex H):
///   * precision 2..=16 (§H.1.1);
///   * single-component grayscale, three-component RGB-class /
///     subsampled YUV-class, and four-component CMYK-class interleaved
///     frames. Annex H pairs lossless with `data unit = one sample`
///     (E.1.1), so subsampling is expressed as the §A.2.3 interleaved-MCU
///     ordering: a three-component scan may oversample the luma component
///     (`1×1` / `2×1` / `2×2` / `4×1`) with both chroma at `1×1`; every
///     other component must be `1×1` (a four-component scan is all-1×1);
///   * the four-component path is `P = 8` only because the workspace
///     `PixelFormat` enum has no high-bit-depth CMYK variant.
fn validate_lossless_sof(sof: &SofInfo) -> Result<()> {
    if !(2..=16).contains(&sof.precision) {
        return Err(Error::unsupported(format!(
            "lossless JPEG: precision {} out of range 2..=16",
            sof.precision
        )));
    }
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
    // Sampling-factor policy for the multi-component lossless processes:
    //   * Four-component (CMYK-class) scans require `H_i = V_i = 1` on
    //     every component. The packed `PixelFormat::Cmyk` output is one
    //     byte-quadruple per pixel with no subsampled variant, so a
    //     subsampled fourth component would have nowhere to land.
    //   * Three-component (YUV-class) scans permit non-unit sampling on
    //     the *first* (luma) component, with both chroma components held
    //     at `H = V = 1` — exactly the convention the baseline/extended-
    //     sequential decoders enforce (A.2.3 interleaved MCU ordering, luma
    //     carries the oversampling). The supported luma factors are 1×1,
    //     2×1, 2×2 and 4×1, mapping to `Yuv444P` / `Yuv422P` / `Yuv420P` /
    //     `Yuv411P`. The all-1×1 RGB-class case keeps its packed `Rgb24`
    //     (P=8) / `Gbrp*Le` / `Rgb48Le` output unchanged.
    if sof.components.len() == 4 {
        for c in &sof.components {
            if c.h_factor != 1 || c.v_factor != 1 {
                return Err(Error::unsupported(
                    "lossless JPEG: four-component scans require H_i = V_i = 1",
                ));
            }
        }
    } else if sof.components.len() == 3 {
        let cb = &sof.components[1];
        let cr = &sof.components[2];
        if cb.h_factor != cr.h_factor || cb.v_factor != cr.v_factor {
            return Err(Error::unsupported(
                "lossless JPEG: chroma components have different sampling factors",
            ));
        }
        if cb.h_factor != 1 || cb.v_factor != 1 {
            return Err(Error::unsupported(
                "lossless JPEG: chroma components must declare H = V = 1 (luma carries oversampling)",
            ));
        }
        let y = &sof.components[0];
        if !matches!((y.h_factor, y.v_factor), (1, 1) | (2, 1) | (2, 2) | (4, 1)) {
            return Err(Error::unsupported(format!(
                "lossless JPEG: unsupported luma sampling {}x{} (supported: 1x1, 2x1, 2x2, 4x1)",
                y.h_factor, y.v_factor
            )));
        }
        // Subsampled (luma-oversampled) YUV-class output is planar
        // `Yuv*P`, which the workspace `PixelFormat` enum only models at
        // 8-bit precision. The all-1×1 RGB-class case keeps its existing
        // P ∈ 2..=16 support (packed Rgb24 / planar Gbrp*Le / Rgb48Le).
        if (y.h_factor != 1 || y.v_factor != 1) && sof.precision != 8 {
            return Err(Error::unsupported(format!(
                "lossless JPEG: subsampled three-component scans require precision 8, got {}",
                sof.precision
            )));
        }
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

/// Resolve a frame's number of lines when the SOF header coded `Y = 0`
/// (T.81 §B.2.2 / §B.2.5).
///
/// When `Y = 0` the real line count is carried by a mandatory DNL
/// segment that "shall immediately follow the first scan" (§B.2.5). This
/// helper performs an independent forward walk over the marker stream —
/// `data` is the post-SOI byte slice (the same slice the main
/// [`decode_jpeg`] walker sees) — to recover `NL` without disturbing the
/// primary decode loop's position. It:
///
///   * finds the SOFn segment and reads its `Y` field; if `Y != 0` there
///     is nothing to resolve and it returns `Ok(None)`;
///   * otherwise consumes segments up to and including the first SOS plus
///     that scan's entropy-coded data, then reads the next marker, which
///     must be DNL, and returns `Ok(Some(NL))`.
///
/// Returns `Err(Invalid)` when `Y = 0` but no well-formed DNL follows the
/// first scan (the segment is mandatory in that case), matching the
/// spec's "this marker segment is mandatory if the number of lines (Y)
/// specified in the frame header has the value zero".
fn resolve_dnl_height(data: &[u8]) -> Result<Option<u16>> {
    let mut walker = MarkerWalker::new(data);
    // Walk to the SOFn to read Y. Skip table / misc segments along the way.
    loop {
        let Some(marker) = walker.next_marker()? else {
            // No SOF at all — let the main loop report the real error.
            return Ok(None);
        };
        match marker {
            markers::SOI => continue,
            m if markers::is_rst(m) => continue,
            markers::EOI => return Ok(None),
            // Hierarchical / unsupported SOFs (SOF5..7, SOF13..15) are
            // left entirely to the main loop, which rejects them with
            // `Unsupported`. Resolving DNL for them would be pointless and
            // could surface a spurious error ahead of that rejection.
            0xC5..=0xC7 | 0xCD..=0xCF => return Ok(None),
            m if markers::is_sof(m) => {
                // This pre-pass never reports SOF errors itself — a
                // malformed SOF (e.g. a truncated component list) is left
                // for the main decode loop to classify with its canonical
                // error. We only need a *parseable* SOF here to read `Y`;
                // if it doesn't parse, there is nothing to resolve and we
                // defer.
                let Ok(p) = walker.read_segment_payload() else {
                    return Ok(None);
                };
                let Ok(sof) = parse_sof(p) else {
                    return Ok(None);
                };
                if sof.height != 0 {
                    // Y is already known; no DNL resolution needed.
                    return Ok(None);
                }
                break;
            }
            markers::SOS => {
                // SOS before any SOF — malformed, but not our concern here.
                return Ok(None);
            }
            _ => {
                // Any other length-prefixed segment: skip it.
                let _ = walker.read_segment_payload()?;
            }
        }
    }
    // Y = 0: advance to the first SOS, consume its scan data, then the
    // DNL segment must follow (T.81 §B.2.5).
    loop {
        let Some(marker) = walker.next_marker()? else {
            return Err(Error::invalid(
                "JPEG: SOF Y = 0 but stream ended before the first scan",
            ));
        };
        match marker {
            markers::SOI => continue,
            m if markers::is_rst(m) => continue,
            markers::EOI => {
                return Err(Error::invalid(
                    "JPEG: SOF Y = 0 but no scan precedes EOI (DNL required)",
                ));
            }
            markers::SOS => {
                let _ = walker.read_segment_payload()?;
                let _ = walker.read_scan_data()?;
                break;
            }
            _ => {
                let _ = walker.read_segment_payload()?;
            }
        }
    }
    // The marker immediately after the first scan must be DNL.
    let Some(marker) = walker.next_marker()? else {
        return Err(Error::invalid(
            "JPEG: SOF Y = 0 but no DNL marker follows the first scan",
        ));
    };
    if marker != markers::DNL {
        return Err(Error::invalid(
            "JPEG: SOF Y = 0 but the marker after the first scan is not DNL",
        ));
    }
    let p = walker.read_segment_payload()?;
    let nl = parse_dnl(p)?;
    Ok(Some(nl))
}

/// Apply a DNL-recovered line count to a freshly-parsed SOF whose `Y`
/// field was coded as 0 (T.81 §B.2.5). When `dnl_height` is `Some`, the
/// SOF's height is replaced by `NL`; when it is `None` the SOF is left
/// untouched. A `Some` patch is only meaningful for an SOF with
/// `height == 0`, but applying it unconditionally to a `height == 0` SOF
/// keeps the call-sites uniform.
fn apply_dnl_height(sof: &mut SofInfo, dnl_height: Option<u16>) {
    if sof.height == 0 {
        if let Some(nl) = dnl_height {
            sof.height = nl;
        }
    }
}

pub(crate) fn decode_jpeg(data: &[u8], pts: Option<i64>) -> Result<VideoFrame> {
    // Verify SOI.
    if data.len() < 2 || data[0] != 0xFF || data[1] != markers::SOI {
        return Err(Error::invalid("JPEG: missing SOI"));
    }

    // T.81 §B.2.2: a frame header may code Y = 0, in which case the real
    // line count is supplied by a mandatory DNL segment immediately after
    // the first scan (§B.2.5). Resolve it up-front so every downstream
    // scan decoder works against a concrete height; the per-SOF handlers
    // below patch `sof.height` with the recovered value.
    let dnl_height = resolve_dnl_height(&data[2..])?;

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
                if state.progressive
                    || state.seq_accum
                    || state.arithmetic
                    || state.progressive_arith
                {
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
                let mut sof = parse_sof(p)?;
                apply_dnl_height(&mut sof, dnl_height);
                validate_sof(&sof)?;
                state.sof = Some(sof);
            }
            SOF2 => {
                if state.sof.is_some() {
                    return Err(Error::invalid("JPEG: multiple SOF segments"));
                }
                let p = walker.read_segment_payload()?;
                let mut sof = parse_sof(p)?;
                apply_dnl_height(&mut sof, dnl_height);
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
            // SOF3 — lossless, Huffman-coded; SOF11 — lossless,
            // arithmetic-coded (T.81 §H.1.2.3 two-dimensional statistical
            // model over the Annex D Q-coder). Both share the Annex H
            // predictor-based coding model and the same frame-header
            // constraints; only the entropy layer of the scan decoder
            // differs.
            SOF3 | markers::SOF11 => {
                if state.sof.is_some() {
                    return Err(Error::invalid("JPEG: multiple SOF segments"));
                }
                let p = walker.read_segment_payload()?;
                let mut sof = parse_sof(p)?;
                apply_dnl_height(&mut sof, dnl_height);
                validate_sof(&sof)?;
                validate_lossless_sof(&sof)?;
                state.sof = Some(sof);
                state.lossless = true;
                state.lossless_arith = marker == markers::SOF11;
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
                let mut sof = parse_sof(p)?;
                apply_dnl_height(&mut sof, dnl_height);
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
            // SOF10 — progressive, arithmetic-coded (T.81 §G.1.3). The
            // SOF2 multi-scan spectral-selection / successive-approximation
            // structure with the Annex D Q-coder as the entropy layer.
            // Same frame constraints as SOF2: P = 8 or P = 12 (Annex G
            // processes 4 and 8), 4-component CMYK / YCCK at P = 8 only
            // (no 12-bit CMYK `PixelFormat` variant in the workspace).
            markers::SOF10 => {
                if state.sof.is_some() {
                    return Err(Error::invalid("JPEG: multiple SOF segments"));
                }
                let p = walker.read_segment_payload()?;
                let mut sof = parse_sof(p)?;
                apply_dnl_height(&mut sof, dnl_height);
                validate_sof(&sof)?;
                if sof.precision != 8 && sof.precision != 12 {
                    return Err(Error::unsupported(format!(
                        "progressive arithmetic JPEG: precision {} (only 8 and 12 are supported)",
                        sof.precision
                    )));
                }
                if sof.components.len() == 4 && sof.precision != 8 {
                    return Err(Error::unsupported(
                        "progressive arithmetic JPEG: 4-component scans only at P = 8",
                    ));
                }
                coef_buf = init_coef_buffers(&sof)?;
                state.sof = Some(sof);
                state.progressive_arith = true;
            }
            // DHP (T.81 §B.3.2) — Define Hierarchical Progression. Its
            // presence before the first SOF selects the hierarchical
            // decoder (§J.2.1 "hierarchical?" decision point). Hand the
            // remaining stream to the dedicated control loop, inheriting
            // any table-specification segments already parsed into `state`.
            markers::DHP => {
                if state.sof.is_some() {
                    return Err(Error::invalid("JPEG: DHP after first SOF"));
                }
                let dhp_payload = walker.read_segment_payload()?;
                return decode_hierarchical(dhp_payload, &mut walker, state, pts);
            }
            0xC5..=0xC7 | 0xCD..=0xCF => {
                let _ = walker.read_segment_payload();
                return Err(Error::unsupported(
                    "JPEG: hierarchical and SOF13..15 arithmetic variants are not supported",
                ));
            }
            SOS => {
                let p = walker.read_segment_payload()?;
                let sos = parse_sos(p)?;
                validate_sos(&sos)?;
                let scan = walker.read_scan_data()?;
                if state.lossless {
                    if state.lossless_arith {
                        return decode_lossless_arith_scan(&state, &sos, scan, pts);
                    }
                    return decode_lossless_scan(&state, &sos, scan, pts);
                }
                if state.progressive_arith {
                    decode_progressive_arith_scan(&state, &sos, scan, &mut coef_buf)?;
                    // Continue — more scans or EOI follow.
                } else if state.arithmetic {
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
            // DNL (T.81 §B.2.5): the number-of-lines value was already
            // recovered up-front by `resolve_dnl_height` and patched into
            // the SOF, so here the segment is consumed and discarded. We
            // still skip its payload via the length field to keep the
            // walker aligned for any trailing markers.
            markers::DNL => {
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

// ---- Progressive arithmetic JPEG (SOF10) ---------------------------------
//
// T.81 Annex G pairs the SOF2 progressive scan structure (spectral
// selection within successive approximation, §G.1.1) with the Annex D
// Q-coder:
//
//   * DC first scans (Ss = Se = 0, Ah = 0) use the sequential DC
//     statistical model of §F.1.4.1 on the point-transformed values; the
//     decoded difference accumulates into the per-component prediction and
//     lands left-shifted by Al (§G.1.3.1).
//   * DC refinement scans (Ah > 0) code one binary decision per block with
//     the fixed 0.5 probability estimate; the decoded bit is ORed into the
//     existing DC value at bit position Al (§G.1.3.1).
//   * AC first scans (Ss > 0, Ah = 0) are the §F.1.4 sequential AC
//     procedure with Kmin = Ss and "EOB" meaning end-of-band rather than
//     end-of-block (§G.1.3.2); decoded values land left-shifted by Al.
//     Kx conditioning comes from the DAC marker (default 5).
//   * AC refinement scans (Ah > 0) follow the §G.1.3.3 coding model
//     (Figures G.10 / G.11, Table G.2) — see `arith::decode_ac_refine`.
//
// Statistics are re-initialised at scan start and at every restart marker;
// the coefficient accumulator + EOI render path are shared with SOF2.

fn decode_progressive_arith_scan(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    coefs: &mut [Vec<[i32; 64]>],
) -> Result<()> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("SOS before SOF"))?;

    // Same scan-header constraints as the Huffman progressive path.
    if sos.ss > 63 || sos.se > 63 || sos.ss > sos.se {
        return Err(Error::invalid("progressive arith: invalid Ss/Se"));
    }
    let is_dc_scan = sos.ss == 0;
    if is_dc_scan && sos.se != 0 {
        return Err(Error::invalid("progressive arith: DC scan must have Se=0"));
    }
    if !is_dc_scan && sos.components.len() != 1 {
        return Err(Error::invalid(
            "progressive arith: AC scans must be non-interleaved",
        ));
    }
    if sos.ah > 13 || sos.al > 13 {
        return Err(Error::invalid("progressive arith: Ah/Al out of range"));
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
                .ok_or_else(|| Error::invalid("progressive arith SOS: unknown component"))
        })
        .collect::<Result<Vec<_>>>()?;

    // Per-SOS-component statistics areas, re-initialised at scan start and
    // every restart. DC conditioning (L, U) and AC conditioning (Kx) come
    // from the DAC destinations named by this scan's Tdj / Taj selectors.
    // Refinement scans track their own bins (DC refinement none at all —
    // the fixed estimate is stateless).
    let mut dc_stats: Vec<DcStats> = sos
        .components
        .iter()
        .map(|sc| {
            let mut s = DcStats::new();
            if let Some(cond) = state.arith_dc[sc.dc_table as usize].as_ref() {
                s.l = cond.l;
                s.u = cond.u;
            }
            s
        })
        .collect();
    let mut ac_stats = AcStats::new();
    if let Some(sc) = sos.components.first() {
        if let Some(cond) = state.arith_ac[sc.ac_table as usize].as_ref() {
            ac_stats.kx = cond.kx;
        }
    }
    let mut ac_refine_stats = AcRefineStats::new();

    let mut scan_pos = 0usize;
    let mut decoder = ArithDecoder::new(scan);
    let mut mcus_since_restart: u32 = 0;

    // Scan-MCU grid: interleaved DC scans step the frame MCU grid; all
    // other scans step the selected component's own block grid.
    let interleaved = is_dc_scan && sos.components.len() > 1;
    let (scan_mcus_x, scan_mcus_y) = if interleaved {
        (mcus_x, mcus_y)
    } else {
        let c = sof.components[sos_map[0]];
        (mcus_x * c.h_factor as usize, mcus_y * c.v_factor as usize)
    };

    for my in 0..scan_mcus_y {
        for mx in 0..scan_mcus_x {
            if state.restart_interval != 0
                && mcus_since_restart != 0
                && mcus_since_restart % state.restart_interval as u32 == 0
            {
                // Advance past the RSTn marker and re-Initdec there.
                // Statistics + DC predictors reset per F.1.4.4.1.5.
                scan_pos = locate_next_marker_after(scan, scan_pos);
                if scan_pos >= scan.len() {
                    return Err(Error::invalid(
                        "progressive arith: missing restart marker mid-scan",
                    ));
                }
                for s in dc_stats.iter_mut() {
                    s.restart_reset();
                }
                ac_stats.restart_reset();
                ac_refine_stats.restart_reset();
                decoder = ArithDecoder::new(&scan[scan_pos..]);
            }

            if is_dc_scan {
                if interleaved {
                    for (sidx, &sof_idx) in sos_map.iter().enumerate() {
                        let c = sof.components[sof_idx];
                        let blocks_x = mcus_x * c.h_factor as usize;
                        for by in 0..c.v_factor as usize {
                            for bx in 0..c.h_factor as usize {
                                let bidx_x = mx * c.h_factor as usize + bx;
                                let bidx_y = my * c.v_factor as usize + by;
                                let bi = bidx_y * blocks_x + bidx_x;
                                prog_arith_decode_dc(
                                    &mut decoder,
                                    &mut dc_stats[sidx],
                                    &mut coefs[sof_idx][bi],
                                    sos.ah,
                                    sos.al,
                                )?;
                            }
                        }
                    }
                } else {
                    let sof_idx = sos_map[0];
                    let c = sof.components[sof_idx];
                    let blocks_x = mcus_x * c.h_factor as usize;
                    let bi = my * blocks_x + mx;
                    prog_arith_decode_dc(
                        &mut decoder,
                        &mut dc_stats[0],
                        &mut coefs[sof_idx][bi],
                        sos.ah,
                        sos.al,
                    )?;
                }
            } else {
                let sof_idx = sos_map[0];
                let c = sof.components[sof_idx];
                let blocks_x = mcus_x * c.h_factor as usize;
                let bi = my * blocks_x + mx;
                let ss = sos.ss as usize;
                let se = sos.se as usize;
                if sos.ah == 0 {
                    // First scan of the band: sequential AC procedure with
                    // Kmin = Ss; decoded values land shifted by Al.
                    let mut tmp = [0i32; 64];
                    arith_decode_ac(&mut decoder, &mut ac_stats, &mut tmp, ss, se)?;
                    let block = &mut coefs[sof_idx][bi];
                    for k in ss..=se {
                        let pos = ZIGZAG[k];
                        if tmp[pos] != 0 {
                            block[pos] = tmp[pos] << sos.al;
                        }
                    }
                } else {
                    arith_decode_ac_refine(
                        &mut decoder,
                        &mut ac_refine_stats,
                        &mut coefs[sof_idx][bi],
                        ss,
                        se,
                        sos.al,
                    )?;
                }
            }
            mcus_since_restart += 1;
        }
    }
    Ok(())
}

/// Progressive arithmetic DC decode for one block (§G.1.3.1). `Ah == 0`
/// decodes a §F.1.4.1 difference in the point-transformed domain and
/// stores the accumulated prediction left-shifted by `Al`. `Ah > 0`
/// decodes one fixed-estimate decision and ORs it into the existing DC
/// value at bit position `Al`.
fn prog_arith_decode_dc(
    d: &mut ArithDecoder<'_>,
    dc: &mut DcStats,
    block: &mut [i32; 64],
    ah: u8,
    al: u8,
) -> Result<()> {
    if ah == 0 {
        let diff = arith_decode_dc_diff(d, dc)?;
        dc.pred = dc.pred.wrapping_add(diff);
        block[0] = dc.pred << al;
    } else {
        let bit = arith_decode_fixed_bit(d) as i32;
        block[0] |= bit << al;
    }
    Ok(())
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
/// Per-component reconstructed sample planes from a (possibly subsampled)
/// Huffman lossless scan, plus the geometry needed to shape them into a
/// frame or feed them back as hierarchical reference components.
///
/// `samples[ci]` is the padded sample grid for component `ci`, `comp_w[ci]`
/// wide. The values are the post-prediction, pre-`<< pt` reconstructed
/// samples (range `0..2^(P-Pt)`). `subsampled` distinguishes the
/// MCU-padded YUV-class grid from the flat width × height grids.
struct LosslessPlanes {
    samples: Vec<Vec<u32>>,
    comp_w: Vec<usize>,
    comp_true_w: Vec<usize>,
    comp_true_h: Vec<usize>,
    subsampled: bool,
}

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
    let nc = sos.components.len();
    let pt = sos.al as u32;
    let precision = sof.precision as u32;
    let width = sof.width as usize;
    let height = sof.height as usize;
    let planes = decode_lossless_scan_planes(state, sos, scan, false)?;
    if planes.subsampled {
        return shape_lossless_yuv_frame(
            &planes.samples,
            &planes.comp_w,
            &planes.comp_true_w,
            &planes.comp_true_h,
            width,
            height,
            pt,
            pts,
        );
    }
    shape_lossless_frame(
        &planes.samples,
        nc,
        width,
        height,
        pt,
        precision,
        state,
        pts,
    )
}

/// Core of the Huffman lossless (SOF3) scan decoder: reconstruct the
/// per-component sample planes without shaping them into a `VideoFrame`.
/// Shared by the standalone lossless decode path and the hierarchical
/// (Annex J) spatial progression. When `differential` is true the Annex J
/// §J.2.3.2 modification applies — the difference is decoded directly
/// without spatial prediction (the predictor selector Ss must be 0), so
/// every reconstructed sample is the modulo-2^(P-Pt) two's-complement
/// difference itself rather than `pred + diff`.
fn decode_lossless_scan_planes(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    differential: bool,
) -> Result<LosslessPlanes> {
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
    if differential {
        // T.81 §J.1.3.2 / §J.2.3.2: in a differential lossless frame the
        // difference is coded directly without prediction, and "the
        // prediction selection parameter in the scan header shall be set
        // to zero".
        if predictor != 0 {
            return Err(Error::invalid(
                "differential lossless: predictor Ss must be 0",
            ));
        }
    } else if !(1..=7).contains(&predictor) {
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

    // Resolve one DC Huffman table per scan component (selectors are
    // independent per Annex H; the encoder may share the same table).
    let mut dc_tables: Vec<&HuffTable> = Vec::with_capacity(nc);
    for sc in &sos.components {
        let t = state.dc_huff[sc.dc_table as usize]
            .as_ref()
            .ok_or_else(|| Error::invalid("lossless: DC Huffman table missing"))?;
        dc_tables.push(t);
    }

    // Per-component sampling factors (T.81 A.2.3 / E.1.1: the lossless
    // data unit is one sample, so a component with H_i × V_i contributes
    // exactly that many samples per MCU). `validate_lossless_sof` has
    // already constrained the legal combinations (single component, all-
    // 1×1 multi-component, or 3-component YUV-class with luma oversampling
    // and 1×1 chroma).
    let h_factors: Vec<usize> = sof.components.iter().map(|c| c.h_factor as usize).collect();
    let v_factors: Vec<usize> = sof.components.iter().map(|c| c.v_factor as usize).collect();
    let h_max = *h_factors.iter().max().unwrap_or(&1);
    let v_max = *v_factors.iter().max().unwrap_or(&1);
    // The MCU-block (A.2.3) ordering only applies to interleaved scans
    // (Ns > 1). For a single-component scan T.81 A.2.2 fixes the order as
    // plain raster "regardless of the values of H1 and V1", so a
    // grayscale component with non-unit factors still decodes left-to-
    // right / top-to-bottom over the full image grid — never the block
    // path below.
    let subsampled = nc > 1 && (h_max != 1 || v_max != 1);

    // Per-component sample-grid dimensions. For the all-1×1 cases this is
    // exactly width × height; for a subsampled 3-component scan each
    // component is `ceil(width × H_i / H_max) × ceil(height × V_i / V_max)`
    // samples (the spec's "small rectangular arrays" partitioning). The
    // grid is padded out to a whole number of MCUs (A.2.4: the encoder
    // extends to a multiple of H_i / V_i, the decoder removes the added
    // samples on output by cropping to the component's true extent).
    let mcus_x = width.div_ceil(h_max);
    let mcus_y = height.div_ceil(v_max);
    // Padded per-component grid dimensions. In the non-subsampled (flat
    // raster) path every component is decoded directly on the width ×
    // height image grid (the all-1×1 multi-component cases have
    // comp_w == width anyway, and a single-component grayscale with
    // non-unit factors still rasters over the full image per A.2.2), so we
    // collapse the grid to width × height there. Only the interleaved
    // subsampled path uses the MCU-padded `ceil × factor` grids.
    let (comp_w, comp_h): (Vec<usize>, Vec<usize>) = if subsampled {
        (
            h_factors.iter().map(|&h| mcus_x * h).collect(),
            v_factors.iter().map(|&v| mcus_y * v).collect(),
        )
    } else {
        (vec![width; nc], vec![height; nc])
    };
    // True (un-padded) per-component extent used to size the output planes.
    let comp_true_w: Vec<usize> = h_factors
        .iter()
        .map(|&h| (width * h).div_ceil(h_max))
        .collect();
    let comp_true_h: Vec<usize> = v_factors
        .iter()
        .map(|&v| (height * v).div_ceil(v_max))
        .collect();

    // All arithmetic is on the pre-Pt-shift sample range (0..2^(P-Pt)).
    let sample_bits = precision - pt;
    let sample_max: u32 = 1u32 << sample_bits;
    let sample_mask: u32 = sample_max - 1;
    let origin: u32 = 1u32 << (sample_bits - 1);

    // One padded sample grid per component; each component is modeled
    // independently per H.1.2 using its own neighbour grid.
    let mut samples: Vec<Vec<u32>> = (0..nc)
        .map(|ci| vec![0u32; comp_w[ci] * comp_h[ci]])
        .collect();
    let mut br = BitReader::new(scan);
    let mut mcus_since_restart: u32 = 0;
    let mut expected_rst: u8 = RST0;
    let mut reset_pred = true; // true at image start and after each RSTn.

    // Decode one sample at component-grid position (cy, cx) for component
    // `ci`. `decode_one` predicts from the component's own grid per Table
    // H.1, honouring the first-line / first-column / restart fall-backs of
    // H.1.2.1, then reads the modulo difference and stores the result.
    // The closure borrows `samples` mutably through an index so the same
    // body serves both the flat (all-1×1) and MCU-ordered paths.
    macro_rules! decode_one {
        ($ci:expr, $cx:expr, $cy:expr) => {{
            let ci = $ci;
            let cx = $cx;
            let cy = $cy;
            let cw = comp_w[ci];
            let plane = &samples[ci];
            // T.81 §J.2.3.2: a differential lossless frame codes the
            // difference directly — no spatial prediction — so the
            // "prediction" contribution is fixed at zero and the
            // reconstructed plane carries the raw modulo-2^(P-Pt) two's
            // complement differences. The non-differential path predicts
            // per Annex H Table H.1 with the H.1.2.1 edge / restart
            // fall-backs.
            let pred: u32 = if differential {
                0
            } else if reset_pred {
                origin
            } else if cy == 0 {
                plane[cy * cw + cx - 1]
            } else if cx == 0 {
                plane[(cy - 1) * cw + cx]
            } else {
                let ra = plane[cy * cw + cx - 1];
                let rb = plane[(cy - 1) * cw + cx];
                let rc = plane[(cy - 1) * cw + cx - 1];
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
                // Annex H Table H.2: SSSS = magnitude in 0..16. A Huffman
                // table that produces a value > 16 here is either corrupt
                // or maliciously crafted; the existing extend / get_bits
                // machinery has no defined behaviour for it.
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
            samples[ci][cy * cw + cx] = sv;
        }};
    }

    // Consume an `RSTn` marker at a restart boundary and re-arm the
    // predictor reset (H.1.2.1: the prediction value at the beginning of
    // each restart interval is the origin again). Shared by both scan
    // orderings.
    macro_rules! consume_restart {
        () => {{
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
        }};
    }

    if !subsampled {
        // All-1×1 fast path (single component, RGB-class, or CMYK-class):
        // one sample per component per pixel, scanned left-to-right /
        // top-to-bottom. The component grid equals the image grid, so cx
        // = x and cy = y. The restart interval counts MCUs (= pixels).
        for y in 0..height {
            for x in 0..width {
                if state.restart_interval != 0
                    && mcus_since_restart != 0
                    && mcus_since_restart % state.restart_interval as u32 == 0
                {
                    consume_restart!();
                }
                for ci in 0..nc {
                    decode_one!(ci, x, y);
                }
                reset_pred = false;
                mcus_since_restart += 1;
            }
        }
    } else {
        // Subsampled 3-component (YUV-class) path. T.81 A.2.3: each MCU
        // contributes H_i × V_i samples from component `i`, in left-to-
        // right / top-to-bottom order within the MCU's H_i × V_i block,
        // and the MCUs themselves walk left-to-right / top-to-bottom. The
        // restart interval counts whole MCUs (H.1.1).
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                if state.restart_interval != 0
                    && mcus_since_restart != 0
                    && mcus_since_restart % state.restart_interval as u32 == 0
                {
                    consume_restart!();
                }
                for ci in 0..nc {
                    let h = h_factors[ci];
                    let v = v_factors[ci];
                    for sy in 0..v {
                        for sx in 0..h {
                            let cx = mx * h + sx;
                            let cy = my * v + sy;
                            decode_one!(ci, cx, cy);
                        }
                    }
                }
                reset_pred = false;
                mcus_since_restart += 1;
            }
        }
    }

    Ok(LosslessPlanes {
        samples,
        comp_w,
        comp_true_w,
        comp_true_h,
        subsampled,
    })
}

// ---- Hierarchical mode (T.81 Annex J) -------------------------------------
//
// A hierarchical-mode stream begins with a DHP marker segment declaring the
// completed-image geometry, followed by a sequence of frames: one
// non-differential frame per component group (the first, lowest-resolution
// stage) and zero or more differential frames that refine resolution /
// quality. Each differential frame codes the two's-complement difference
// between its input components and the reconstructed reference components
// from the preceding stage; an EXP segment ahead of the frame signals that
// the reference is to be ×2 upsampled (J.1.1.2 bi-linear) before the
// difference is added back modulo 2^16 (J.2.1).
//
// This implementation covers the **spatial (lossless) hierarchical
// progression** of §K.7.2.2: the non-differential frame is SOF3 (lossless
// Huffman) and every differential frame is SOF7 (differential lossless
// Huffman). That is the progression with a "truly lossless final stage"
// and — unlike the DCT hierarchical progression — does not depend on the
// encoder and decoder sharing a bit-exact IDCT, so a conformant
// round-trip is well defined. The current slice is restricted to
// single-component (grayscale) frames; multi-component spatial
// progressions and the DCT hierarchical path return `Unsupported`.

/// One reconstructed reference component: a full-resolution sample plane
/// plus its dimensions. Samples are stored at full precision (modulo
/// 2^16), matching the reference-component model of §J.2.1.
#[derive(Clone)]
struct RefComponent {
    width: usize,
    height: usize,
    samples: Vec<u32>,
}

/// ×2 bi-linear upsampling of a reference component per T.81 §J.1.1.2.
///
/// `Px = (Ra + Rb) / 2` with truncating division. The left column / top
/// line of the upsampled image match the source; the right column / bottom
/// line of the source are replicated to supply the missing right/bottom
/// interpolation neighbours. Horizontal and vertical expansion are applied
/// independently (the caller does horizontal first, then vertical, when
/// both are signalled), so this helper expands a single axis.
fn upsample_axis(rc: &RefComponent, horizontal: bool, modulus: u32) -> RefComponent {
    if horizontal {
        let out_w = rc.width * 2;
        let out_h = rc.height;
        let mut samples = vec![0u32; out_w * out_h];
        for y in 0..out_h {
            for x in 0..rc.width {
                let ra = rc.samples[y * rc.width + x];
                // Right edge replicates the boundary sample.
                let rb = if x + 1 < rc.width {
                    rc.samples[y * rc.width + x + 1]
                } else {
                    ra
                };
                // Even output column matches the source column; the odd
                // column is the interpolated midpoint.
                samples[y * out_w + 2 * x] = ra;
                samples[y * out_w + 2 * x + 1] = ra.wrapping_add(rb) / 2 % modulus.max(1);
            }
        }
        RefComponent {
            width: out_w,
            height: out_h,
            samples,
        }
    } else {
        let out_w = rc.width;
        let out_h = rc.height * 2;
        let mut samples = vec![0u32; out_w * out_h];
        for y in 0..rc.height {
            for x in 0..rc.width {
                let ra = rc.samples[y * rc.width + x];
                let rb = if y + 1 < rc.height {
                    rc.samples[(y + 1) * rc.width + x]
                } else {
                    ra
                };
                samples[(2 * y) * out_w + x] = ra;
                samples[(2 * y + 1) * out_w + x] = ra.wrapping_add(rb) / 2 % modulus.max(1);
            }
        }
        RefComponent {
            width: out_w,
            height: out_h,
            samples,
        }
    }
}

/// Hierarchical-mode (Annex J) decode control loop. Entered from
/// `decode_jpeg` when a DHP marker is seen before the first SOF; `walker`
/// is positioned immediately after the DHP segment payload, and `state`
/// carries any table-specification segments parsed before the DHP.
///
/// Covers the spatial lossless progression only (SOF3 non-differential
/// first frame + SOF7 differential frames). Frames may be single-component
/// (grayscale) or multi-component (RGB-class `Nf = 3` / CMYK-class
/// `Nf = 4`) with all components at `H = V = 1`. Anything else — a
/// DCT-based non-differential frame, an arithmetic differential frame, or a
/// subsampled multi-component frame — returns `Unsupported`.
fn decode_hierarchical(
    dhp_payload: &[u8],
    walker: &mut MarkerWalker<'_>,
    mut state: JpegState,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    // The DHP body is a frame header (§B.3.2): precision, Y, X, component
    // list. It fixes the completed-image precision and component identities.
    let dhp = parse_sof(dhp_payload)?;
    let nc = dhp.components.len();
    if !matches!(nc, 1 | 3 | 4) {
        return Err(Error::unsupported(format!(
            "hierarchical JPEG: {nc} component(s) — only 1, 3 and 4 are supported"
        )));
    }
    let precision = dhp.precision as u32;
    if !(2..=16).contains(&precision) {
        return Err(Error::unsupported(format!(
            "hierarchical JPEG: precision {precision} out of range 2..=16"
        )));
    }
    // The four-component (CMYK-class) output shaping is P = 8 only, matching
    // the non-hierarchical lossless path (no high-bit-depth CMYK
    // `PixelFormat`).
    if nc == 4 && precision != 8 {
        return Err(Error::unsupported(
            "hierarchical JPEG: 4-component progressions require P = 8",
        ));
    }
    // Reference modulus: differential reconstruction is modulo 2^16 for the
    // 16-bit lossless case (§J.1) and, more generally, the difference + the
    // reference are kept within the sample range of the precision. We track
    // it as 2^precision so an n-bit progression wraps consistently.
    let ref_modulus: u32 = 1u32 << precision;

    // One reconstructed reference plane per component; populated by the
    // non-differential frame and refined by each differential frame.
    // table-specification segments (DQT/DHT/DRI) parsed before the DHP are
    // inherited by the first frame per §B.2.4 / §J.2.1.
    let mut reference: Option<Vec<RefComponent>> = None;
    // Pending EXP expansion flags for the *next* frame (cleared after use).
    let mut pending_exp: Option<(bool, bool)> = None;

    loop {
        let Some(marker) = walker.next_marker()? else {
            return Err(Error::invalid(
                "hierarchical JPEG: unexpected EOF before EOI",
            ));
        };
        match marker {
            EOI => {
                let rc = reference
                    .ok_or_else(|| Error::invalid("hierarchical JPEG: EOI before any frame"))?;
                return shape_hierarchical_frame(&rc, precision, &state, pts);
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
            DRI => {
                let p = walker.read_segment_payload()?;
                state.restart_interval = parse_dri(p)?;
            }
            markers::APP14 => {
                let p = walker.read_segment_payload()?;
                if p.len() >= 12 && &p[0..5] == b"Adobe" {
                    state.adobe_transform = Some(p[11]);
                }
            }
            // EXP (§B.3.3): applies to the next (differential) frame only.
            markers::EXP => {
                let p = walker.read_segment_payload()?;
                if p.len() != 1 {
                    return Err(Error::invalid("hierarchical JPEG: EXP length != 1"));
                }
                let eh = (p[0] >> 4) & 0x0F;
                let ev = p[0] & 0x0F;
                if eh > 1 || ev > 1 {
                    return Err(Error::invalid(
                        "hierarchical JPEG: EXP Eh/Ev must be 0 or 1",
                    ));
                }
                if pending_exp.is_some() {
                    return Err(Error::invalid(
                        "hierarchical JPEG: more than one EXP before a frame",
                    ));
                }
                pending_exp = Some((eh == 1, ev == 1));
            }
            // Non-differential lossless frame (the first stage).
            SOF3 => {
                if reference.is_some() {
                    return Err(Error::unsupported(
                        "hierarchical JPEG: a second non-differential frame is not supported",
                    ));
                }
                if pending_exp.take().is_some() {
                    return Err(Error::invalid(
                        "hierarchical JPEG: EXP precedes a non-differential frame",
                    ));
                }
                let p = walker.read_segment_payload()?;
                let sof = parse_sof(p)?;
                validate_sof(&sof)?;
                validate_lossless_sof(&sof)?;
                check_hier_frame(&sof, nc, precision)?;
                state.sof = Some(sof.clone());
                reference = Some(decode_hierarchical_frame(&state, walker, false)?);
            }
            // Differential lossless frame (a refinement stage).
            SOF7 => {
                let prev = reference.take().ok_or_else(|| {
                    Error::invalid(
                        "hierarchical JPEG: differential frame before a non-differential frame",
                    )
                })?;
                let p = walker.read_segment_payload()?;
                let sof = parse_sof(p)?;
                validate_sof(&sof)?;
                validate_lossless_sof(&sof)?;
                check_hier_frame(&sof, nc, precision)?;
                // Upsample every reference component if an EXP segment
                // preceded this frame. Horizontal expansion is applied
                // first, then vertical (§J.1.1.2).
                let mut upsampled = prev;
                if let Some((eh, ev)) = pending_exp.take() {
                    for rc in upsampled.iter_mut() {
                        if eh {
                            *rc = upsample_axis(rc, true, ref_modulus);
                        }
                        if ev {
                            *rc = upsample_axis(rc, false, ref_modulus);
                        }
                    }
                }
                state.sof = Some(sof.clone());
                let diff = decode_hierarchical_frame(&state, walker, true)?;
                if diff.len() != upsampled.len() {
                    return Err(Error::invalid(
                        "hierarchical JPEG: differential frame component count mismatch",
                    ));
                }
                // Reconstruct each component: add the difference modulo
                // 2^precision (§J.2.1).
                let mut next = Vec::with_capacity(diff.len());
                for (u, d) in upsampled.iter().zip(diff.iter()) {
                    if d.width != u.width || d.height != u.height {
                        return Err(Error::invalid(
                            "hierarchical JPEG: differential frame size != reference size",
                        ));
                    }
                    let mut samples = vec![0u32; d.width * d.height];
                    for i in 0..samples.len() {
                        samples[i] = u.samples[i].wrapping_add(d.samples[i]) % ref_modulus;
                    }
                    next.push(RefComponent {
                        width: d.width,
                        height: d.height,
                        samples,
                    });
                }
                reference = Some(next);
            }
            // Any DCT-based or arithmetic SOF inside a hierarchical stream
            // is outside the spatial-lossless slice this decoder covers.
            m if markers::is_sof(m) => {
                let _ = walker.read_segment_payload();
                return Err(Error::unsupported(
                    "hierarchical JPEG: only the spatial lossless progression (SOF3 + SOF7) is supported",
                ));
            }
            COM => {
                let _ = walker.read_segment_payload()?;
            }
            m if markers::is_app(m) => {
                let _ = walker.read_segment_payload()?;
            }
            _ => {
                let _ = walker.read_segment_payload();
            }
        }
    }
}

/// Per-frame constraint check for a hierarchical spatial-lossless frame:
/// the component count and precision must match the DHP, and every
/// component must be `H = V = 1` (the spatial progression refines
/// resolution via EXP upsampling, not via in-frame subsampling).
fn check_hier_frame(sof: &SofInfo, nc: usize, precision: u32) -> Result<()> {
    if sof.components.len() != nc {
        return Err(Error::invalid(
            "hierarchical JPEG: frame component count differs from DHP",
        ));
    }
    if sof.precision as u32 != precision {
        return Err(Error::invalid(
            "hierarchical JPEG: frame precision differs from DHP",
        ));
    }
    if sof
        .components
        .iter()
        .any(|c| c.h_factor != 1 || c.v_factor != 1)
    {
        return Err(Error::unsupported(
            "hierarchical JPEG: subsampled (H/V != 1) frames are not supported",
        ));
    }
    Ok(())
}

/// Decode one hierarchical-mode lossless frame and return its reconstructed
/// per-component sample planes. Reads the frame's single SOS + entropy-coded
/// scan from `walker`. `differential` selects the §J.2.3.2 modification
/// (difference coded directly, no spatial prediction).
fn decode_hierarchical_frame(
    state: &JpegState,
    walker: &mut MarkerWalker<'_>,
    differential: bool,
) -> Result<Vec<RefComponent>> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("hierarchical JPEG: missing SOF"))?;
    let width = sof.width as usize;
    let height = sof.height as usize;
    // Locate the frame's SOS, consuming any table/misc segments that appear
    // between the frame header and the scan header (§B.2 / §E.1: tables may
    // precede a scan as well as a frame).
    loop {
        let Some(marker) = walker.next_marker()? else {
            return Err(Error::invalid("hierarchical JPEG: EOF before SOS"));
        };
        match marker {
            SOS => {
                let p = walker.read_segment_payload()?;
                let sos = parse_sos(p)?;
                validate_sos(&sos)?;
                // A non-zero point transform (Al) would put the
                // reconstructed samples on the `2^(P-Pt)` range and require
                // a Pt-aware add-back in the §J.2.1 reconstruction; the
                // current slice handles Pt = 0 only (the common lossless
                // hierarchical case).
                if sos.al != 0 {
                    return Err(Error::unsupported(
                        "hierarchical JPEG: non-zero point transform is not supported",
                    ));
                }
                let scan = walker.read_scan_data()?;
                let planes = decode_lossless_scan_planes(state, &sos, scan, differential)?;
                if planes.subsampled {
                    return Err(Error::unsupported(
                        "hierarchical JPEG: subsampled frames are not supported",
                    ));
                }
                // Every component shares the frame's width × height (all
                // factors are 1×1, checked in `check_hier_frame`).
                return Ok(planes
                    .samples
                    .into_iter()
                    .map(|samples| RefComponent {
                        width,
                        height,
                        samples,
                    })
                    .collect());
            }
            DQT => {
                let _ = walker.read_segment_payload()?;
            }
            DHT => {
                // A scan-local DHT would need to mutate `state`, but the
                // hierarchical slice keeps tables frame-global; reject
                // rather than silently ignore a table the scan needs.
                return Err(Error::unsupported(
                    "hierarchical JPEG: DHT between frame header and scan is not supported",
                ));
            }
            DRI => {
                let _ = walker.read_segment_payload()?;
            }
            COM => {
                let _ = walker.read_segment_payload()?;
            }
            m if markers::is_app(m) => {
                let _ = walker.read_segment_payload()?;
            }
            _ => {
                return Err(Error::invalid(
                    "hierarchical JPEG: unexpected marker before SOS",
                ));
            }
        }
    }
}

/// Shape the reconstructed hierarchical-mode reference components into the
/// output `VideoFrame`. All components share the same full-resolution
/// `width × height` grid (the spatial progression refines resolution via
/// EXP upsampling, not in-frame subsampling), so the precision-driven
/// output policy of `shape_lossless_frame` applies directly: grayscale
/// (`Gray*`), 3-component (packed `Rgb24` / planar `Gbrp*Le` / packed
/// `Rgb48Le`), or 4-component (packed `Cmyk`, Adobe APP14 transform
/// honoured). The point transform was constrained to zero when each scan
/// was decoded, so the reconstructed samples already span the full
/// `2^P` range and no `<< pt` is needed.
fn shape_hierarchical_frame(
    refs: &[RefComponent],
    precision: u32,
    state: &JpegState,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let nc = refs.len();
    let width = refs[0].width;
    let height = refs[0].height;
    let samples: Vec<Vec<u32>> = refs.iter().map(|rc| rc.samples.clone()).collect();
    shape_lossless_frame(&samples, nc, width, height, 0, precision, state, pts)
}

/// Shape the reconstructed per-component lossless sample planes into the
/// output `VideoFrame`. Shared by the Huffman (SOF3) and arithmetic
/// (SOF11) lossless scan decoders — the Annex H coding model is identical
/// past the entropy layer, so the precision-driven output policy is too.
#[allow(clippy::too_many_arguments)]
fn shape_lossless_frame(
    samples: &[Vec<u32>],
    nc: usize,
    width: usize,
    height: usize,
    pt: u32,
    precision: u32,
    state: &JpegState,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    // Four-component decode: pack the per-component planes into a single
    // row-major `C M Y K` output (packed `PixelFormat::Cmyk`, 4
    // bytes/pixel). All components are `H_i = V_i = 1` in the lossless
    // four-component path, so no upsampling is needed; samples line up
    // one-per-pixel. The Adobe APP14 colour-transform flag is honoured
    // identically to the lossy CMYK paths: `transform = 0` un-inverts the
    // Adobe-CMYK convention, `transform = 2` (YCCK) decodes YCbCr → RGB
    // via BT.601 and un-inverts K, and the no-APP14 case passes the four
    // sample bytes through as plain "regular" CMYK. `precision = 8` was
    // enforced at SOF time so each post-`<< pt` byte fits in `u8` without
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

/// Shape a subsampled three-component (YUV-class) lossless decode into a
/// planar `Yuv444P` / `Yuv422P` / `Yuv420P` / `Yuv411P` frame.
///
/// The scan decoder reconstructed each component onto its own MCU-padded
/// sample grid (`comp_w[ci]` wide). Per T.81 A.2.4 the decoding process
/// removes any samples the encoder added to round each component up to a
/// whole number of MCUs, so here we crop component `ci` to its true extent
/// `comp_true_w[ci] × comp_true_h[ci]` when copying into the output plane.
/// Precision is fixed at `P = 8` for the YUV-class path (the only subsampled
/// combination `validate_lossless_sof` admits), so each post-`<< pt` sample
/// fits in a `u8`. Plane order is Y, Cb, Cr — the SOS scan order, matching
/// the lossy decoder's planar layout.
#[allow(clippy::too_many_arguments)]
fn shape_lossless_yuv_frame(
    samples: &[Vec<u32>],
    comp_w: &[usize],
    comp_true_w: &[usize],
    comp_true_h: &[usize],
    width: usize,
    height: usize,
    pt: u32,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let mut planes: Vec<VideoPlane> = Vec::with_capacity(3);
    // Y plane is always full-resolution (width × height); the chroma
    // planes carry their subsampled extent. We size each output plane to
    // the component's true (un-padded) dimensions and copy row-by-row out
    // of the wider padded grid.
    for ci in 0..3 {
        let cw = comp_w[ci];
        let out_w = if ci == 0 { width } else { comp_true_w[ci] };
        let out_h = if ci == 0 { height } else { comp_true_h[ci] };
        let stride = out_w;
        let mut data = vec![0u8; stride * out_h];
        for y in 0..out_h {
            let src_row = &samples[ci][y * cw..y * cw + out_w];
            let dst_row = &mut data[y * stride..y * stride + out_w];
            for (d, &s) in dst_row.iter_mut().zip(src_row) {
                *d = (s << pt) as u8;
            }
        }
        planes.push(VideoPlane { stride, data });
    }
    Ok(VideoFrame { pts, planes })
}

// ---- Lossless arithmetic JPEG (SOF11) -------------------------------------
//
// Same Annex H predictor-based coding model as SOF3, but the modulo-2^16
// prediction differences are coded with the Annex D Q-coder under the
// two-dimensional statistical model of T.81 §H.1.2.3: each binary decision
// is conditioned on the classifications of Da (the difference coded for
// the sample to the left) and Db (the difference coded for the sample in
// the line above) via the 5 × 5 array of Figure H.2, with the magnitude
// bins selected by X1_Context(Db). Per §H.2.2 the decoder mirrors the
// encoder's coding model; per §H.1.2.3.4 / §H.2.1 all statistics are
// re-initialised at the start of the scan and at each restart, and per
// §H.1.2.3.1 the line-above conditioning is zero for the first line of
// the scan and of each restart interval, with the left conditioning zero
// at the start of each line.
//
// Prediction (§H.1.2.1): the first sample of the scan and of each restart
// interval is predicted as 2^(P − Pt − 1); the rest of the first line of
// the scan / interval uses the one-dimensional horizontal predictor (Ra);
// the first sample of every other line uses Rb; everywhere else the
// scan-header-selected predictor applies. Restart intervals are an
// integer multiple of the samples per MCU-row (§H.1.1), so interval
// boundaries are line-aligned in conformant streams.
fn decode_lossless_arith_scan(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("SOS before SOF"))?;
    if sos.components.len() != sof.components.len() {
        return Err(Error::unsupported(
            "lossless arith: non-interleaved multi-component scans are not supported",
        ));
    }
    let predictor = sos.ss;
    if !(1..=7).contains(&predictor) {
        return Err(Error::invalid(
            "lossless arith: predictor Ss must be in 1..=7",
        ));
    }
    let pt = sos.al as u32; // point transform
    let precision = sof.precision as u32;
    if pt >= precision {
        return Err(Error::invalid("lossless arith: Pt >= precision"));
    }
    let width = sof.width as usize;
    let height = sof.height as usize;
    let nc = sos.components.len();

    // Detect a subsampled YUV-class scan (luma oversampled, chroma 1×1). The
    // A.2.3 interleaved-MCU ordering applies here; the flat per-pixel path
    // below only handles all-1×1 components. `validate_lossless_sof` has
    // already constrained the legal combinations.
    let h_factors: Vec<usize> = sof.components.iter().map(|c| c.h_factor as usize).collect();
    let v_factors: Vec<usize> = sof.components.iter().map(|c| c.v_factor as usize).collect();
    let h_max = *h_factors.iter().max().unwrap_or(&1);
    let v_max = *v_factors.iter().max().unwrap_or(&1);
    if nc > 1 && (h_max != 1 || v_max != 1) {
        if precision != 8 {
            return Err(Error::unsupported(
                "lossless arith: subsampled three-component scans require precision 8",
            ));
        }
        return decode_lossless_arith_scan_subsampled(
            state, sos, scan, pts, predictor, pt, width, height, &h_factors, &v_factors, h_max,
            v_max,
        );
    }

    // One statistics area per scan component (§H.1.2.3.2), with the L / U
    // conditioning bounds taken from the DAC DC-conditioning entry the
    // component's Td selector points at (§H.1.2.3.3; defaults L=0, U=1).
    let mut stats: Vec<LosslessStats> = sos
        .components
        .iter()
        .map(|sc| {
            let mut s = LosslessStats::new();
            if let Some(cond) = state.arith_dc[sc.dc_table as usize].as_ref() {
                s.l = cond.l;
                s.u = cond.u;
            }
            s
        })
        .collect();

    // All arithmetic is on the pre-Pt-shift sample range (0..2^(P-Pt)).
    let sample_bits = precision - pt;
    let sample_max: u32 = 1u32 << sample_bits;
    let sample_mask: u32 = sample_max - 1;
    let origin: u32 = 1u32 << (sample_bits - 1);

    let mut samples: Vec<Vec<u32>> = (0..nc).map(|_| vec![0u32; width * height]).collect();

    // Per-component conditioning history: the differences coded for the
    // line above (Db source) and for the current line so far (Da via
    // column x-1). Zeroed at scan start and at each restart (§H.1.2.3.1).
    let mut prev_diff: Vec<Vec<i32>> = (0..nc).map(|_| vec![0i32; width]).collect();
    let mut cur_diff: Vec<Vec<i32>> = (0..nc).map(|_| vec![0i32; width]).collect();

    // Restart bookkeeping mirrors decode_arith_scan: track the byte offset
    // of the current entropy segment within `scan` and re-Initdec past
    // each RSTn marker.
    let mut scan_pos = 0usize;
    let mut decoder = ArithDecoder::new(scan);
    let mut samples_since_restart: u32 = 0;
    let mut reset_pred = true; // true at scan start and after each RSTn
    let mut first_line_y = 0usize; // first line of the current restart interval

    for y in 0..height {
        for x in 0..width {
            if state.restart_interval != 0
                && samples_since_restart != 0
                && samples_since_restart % state.restart_interval as u32 == 0
            {
                scan_pos = locate_next_marker_after(scan, scan_pos);
                if scan_pos >= scan.len() {
                    return Err(Error::invalid(
                        "lossless arith: missing restart marker mid-scan",
                    ));
                }
                for s in stats.iter_mut() {
                    s.reset();
                }
                for row in prev_diff.iter_mut() {
                    row.fill(0);
                }
                for row in cur_diff.iter_mut() {
                    row.fill(0);
                }
                decoder = ArithDecoder::new(&scan[scan_pos..]);
                reset_pred = true;
                first_line_y = y;
            }

            for ci in 0..nc {
                let plane = &samples[ci];
                let pred: u32 = if reset_pred {
                    origin
                } else if y == first_line_y {
                    // First line of the scan / restart interval: 1-D
                    // horizontal predictor (Ra) per §H.1.2.1.
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

                let da = if x == 0 { 0 } else { cur_diff[ci][x - 1] };
                let db = prev_diff[ci][x];
                let diff = arith_decode_lossless_diff(&mut decoder, &mut stats[ci], da, db)?;
                cur_diff[ci][x] = diff;

                let sv = ((pred as i32).wrapping_add(diff) as u32) & sample_mask;
                samples[ci][y * width + x] = sv;
            }
            reset_pred = false;
            samples_since_restart += 1;
        }
        std::mem::swap(&mut prev_diff, &mut cur_diff);
    }

    shape_lossless_frame(&samples, nc, width, height, pt, precision, state, pts)
}

/// Decode a subsampled three-component (YUV-class) SOF11 lossless scan — the
/// arithmetic-coded counterpart of the Huffman subsampled path in
/// [`decode_lossless_scan`].
///
/// The luma component is oversampled (`1×1` / `2×1` / `2×2` / `4×1`) with both
/// chroma components at `1×1`; the entropy data is walked in T.81 §A.2.3
/// interleaved-MCU order (luma's `H_1 × V_1` block, then one Cb sample, then
/// one Cr). Each component is modelled independently per §H.1.2 over its own
/// MCU-padded sample grid (its own `Ra` / `Rb` / `Rc` predictor neighbours and
/// its own statistics area), and the §H.1.2.3 `L_Context(Da, Db)` conditioning
/// reads the differences from a full per-component **difference grid** indexed
/// by absolute grid coordinate — because the §A.2.3 walk does not visit a
/// component's samples in plain raster order, a sliding two-row window would
/// mis-address the `Db` (above) neighbour. The grids are padded to a whole
/// number of MCUs and cropped to each component's true extent on output
/// (§A.2.4). Precision is fixed at `P = 8`.
#[allow(clippy::too_many_arguments)]
fn decode_lossless_arith_scan_subsampled(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    pts: Option<i64>,
    predictor: u8,
    pt: u32,
    width: usize,
    height: usize,
    h_factors: &[usize],
    v_factors: &[usize],
    h_max: usize,
    v_max: usize,
) -> Result<VideoFrame> {
    let nc = sos.components.len();

    // One statistics area per scan component (§H.1.2.3.2); the DAC L / U
    // conditioning bounds default to (0, 1) when no DAC segment is present.
    let mut stats: Vec<LosslessStats> = sos
        .components
        .iter()
        .map(|sc| {
            let mut s = LosslessStats::new();
            if let Some(cond) = state.arith_dc[sc.dc_table as usize].as_ref() {
                s.l = cond.l;
                s.u = cond.u;
            }
            s
        })
        .collect();

    let sample_bits = 8u32 - pt;
    let sample_mask: u32 = (1u32 << sample_bits) - 1;
    let origin: u32 = 1u32 << (sample_bits - 1);

    // Per-component MCU-padded sample grids and their true (un-padded) extent.
    let mcus_x = width.div_ceil(h_max);
    let mcus_y = height.div_ceil(v_max);
    let comp_w: Vec<usize> = h_factors.iter().map(|&hf| mcus_x * hf).collect();
    let comp_h: Vec<usize> = v_factors.iter().map(|&vf| mcus_y * vf).collect();
    let comp_true_w: Vec<usize> = h_factors
        .iter()
        .map(|&hf| (width * hf).div_ceil(h_max))
        .collect();
    let comp_true_h: Vec<usize> = v_factors
        .iter()
        .map(|&vf| (height * vf).div_ceil(v_max))
        .collect();

    let mut samples: Vec<Vec<u32>> = (0..nc)
        .map(|ci| vec![0u32; comp_w[ci] * comp_h[ci]])
        .collect();
    // Full per-component difference grid for the §H.1.2.3 conditioning.
    let mut diff_grid: Vec<Vec<i32>> = (0..nc)
        .map(|ci| vec![0i32; comp_w[ci] * comp_h[ci]])
        .collect();

    let mut scan_pos = 0usize;
    let mut decoder = ArithDecoder::new(scan);
    let mut mcus_since_restart: u32 = 0;
    let mut reset_pred = true; // true at scan start and after each RSTn
    let mut first_row: Vec<usize> = vec![0usize; nc]; // first grid row of the current interval

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if state.restart_interval != 0
                && mcus_since_restart != 0
                && mcus_since_restart % state.restart_interval as u32 == 0
            {
                scan_pos = locate_next_marker_after(scan, scan_pos);
                if scan_pos >= scan.len() {
                    return Err(Error::invalid(
                        "lossless arith: missing restart marker mid-scan",
                    ));
                }
                for s in stats.iter_mut() {
                    s.reset();
                }
                for g in diff_grid.iter_mut() {
                    g.fill(0);
                }
                decoder = ArithDecoder::new(&scan[scan_pos..]);
                reset_pred = true;
                for ci in 0..nc {
                    first_row[ci] = my * v_factors[ci];
                }
            }

            for ci in 0..nc {
                let gw = comp_w[ci];
                let hf = h_factors[ci];
                let vf = v_factors[ci];
                for sy in 0..vf {
                    for sx in 0..hf {
                        let gx = mx * hf + sx;
                        let gy = my * vf + sy;
                        let plane = &samples[ci];
                        let pred: u32 = if reset_pred && gx == 0 && gy == first_row[ci] {
                            origin
                        } else if gy == first_row[ci] {
                            // First grid line of the scan / interval uses Ra.
                            plane[gy * gw + gx - 1]
                        } else if gx == 0 {
                            plane[(gy - 1) * gw + gx]
                        } else {
                            let ra = plane[gy * gw + gx - 1];
                            let rb = plane[(gy - 1) * gw + gx];
                            let rc = plane[(gy - 1) * gw + gx - 1];
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

                        let dg = &diff_grid[ci];
                        let da = if gx == 0 { 0 } else { dg[gy * gw + gx - 1] };
                        let db = if gy == 0 { 0 } else { dg[(gy - 1) * gw + gx] };
                        let diff =
                            arith_decode_lossless_diff(&mut decoder, &mut stats[ci], da, db)?;
                        diff_grid[ci][gy * gw + gx] = diff;

                        let sv = ((pred as i32).wrapping_add(diff) as u32) & sample_mask;
                        samples[ci][gy * gw + gx] = sv;
                    }
                }
            }
            reset_pred = false;
            mcus_since_restart += 1;
        }
    }

    shape_lossless_yuv_frame(
        &samples,
        &comp_w,
        &comp_true_w,
        &comp_true_h,
        width,
        height,
        pt,
        pts,
    )
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
    use super::{decode_jpeg, Error};
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

    /// Append one length-prefixed marker segment (T.81 §B.1.1.4).
    fn put_seg(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        out.extend_from_slice(&[0xFF, marker]);
        out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(payload);
    }

    /// Build a complete SOF11 (lossless, arithmetic-coded) JPEG from
    /// pre-Pt-shift component planes — the encoder-side mirror of
    /// `decode_lossless_arith_scan` per T.81 §H.1.2 (prediction) +
    /// §H.1.2.3 (two-dimensional statistical model), used as test
    /// scaffolding for round-trip verification.
    #[allow(clippy::too_many_arguments)]
    fn encode_sof11_jpeg(
        width: usize,
        height: usize,
        planes: &[Vec<u32>],
        precision: u8,
        predictor: u8,
        pt: u8,
        restart_interval: u16,
        dac_lu: Option<(u8, u8)>,
    ) -> Vec<u8> {
        use crate::jpeg::arith::{encode_lossless_diff, ArithEncoder, LosslessStats};
        let nc = planes.len();
        let mut out = vec![0xFF, 0xD8]; // SOI
        if let Some((l, u)) = dac_lu {
            // DAC entry: Tc = 0 (DC / lossless conditioning), Tb = 0,
            // Cs = (U << 4) | L (T.81 §B.2.4.3).
            put_seg(&mut out, 0xCC, &[0x00, (u << 4) | l]);
        }
        if restart_interval != 0 {
            put_seg(&mut out, 0xDD, &restart_interval.to_be_bytes());
        }
        let mut sof = vec![precision];
        sof.extend_from_slice(&(height as u16).to_be_bytes());
        sof.extend_from_slice(&(width as u16).to_be_bytes());
        sof.push(nc as u8);
        for ci in 0..nc {
            sof.extend_from_slice(&[ci as u8 + 1, 0x11, 0]);
        }
        put_seg(&mut out, 0xCB, &sof); // SOF11
        let mut sos = vec![nc as u8];
        for ci in 0..nc {
            sos.extend_from_slice(&[ci as u8 + 1, 0x00]);
        }
        // Ss = predictor, Se = 0, Ah = 0, Al = Pt (§B.2.3 for lossless).
        sos.extend_from_slice(&[predictor, 0, pt]);
        put_seg(&mut out, 0xDA, &sos);

        let sample_bits = (precision - pt) as u32;
        let origin: u32 = 1 << (sample_bits - 1);
        let (l, u) = dac_lu.unwrap_or((0, 1));
        let mut stats: Vec<LosslessStats> = (0..nc)
            .map(|_| {
                let mut s = LosslessStats::new();
                s.l = l;
                s.u = u;
                s
            })
            .collect();
        let mut prev_diff = vec![vec![0i32; width]; nc];
        let mut cur_diff = vec![vec![0i32; width]; nc];
        let mut enc = ArithEncoder::new();
        let mut since_restart = 0u32;
        let mut rst = 0u8;
        let mut reset_pred = true;
        let mut first_line_y = 0usize;
        for y in 0..height {
            for x in 0..width {
                if restart_interval != 0
                    && since_restart != 0
                    && since_restart % restart_interval as u32 == 0
                {
                    // Flush the entropy segment, append RSTn, restart the
                    // coder + statistics + conditioning + prediction
                    // (§H.1.1 / §H.1.2.3.4).
                    out.extend_from_slice(&std::mem::take(&mut enc).finish());
                    out.extend_from_slice(&[0xFF, 0xD0 + rst]);
                    rst = (rst + 1) % 8;
                    for s in stats.iter_mut() {
                        s.reset();
                    }
                    for r in prev_diff.iter_mut() {
                        r.fill(0);
                    }
                    for r in cur_diff.iter_mut() {
                        r.fill(0);
                    }
                    reset_pred = true;
                    first_line_y = y;
                }
                for ci in 0..nc {
                    let plane = &planes[ci];
                    let pred: u32 = if reset_pred {
                        origin
                    } else if y == first_line_y {
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
                    let px = plane[y * width + x];
                    // Modulo-2^16 difference (§H.1.2.1) reduced to the
                    // -32768..=32767 representative.
                    let dm = (px.wrapping_sub(pred) & 0xFFFF) as i32;
                    let dm = if dm >= 0x8000 { dm - 0x10000 } else { dm };
                    let da = if x == 0 { 0 } else { cur_diff[ci][x - 1] };
                    let db = prev_diff[ci][x];
                    encode_lossless_diff(&mut enc, &mut stats[ci], da, db, dm).unwrap();
                    cur_diff[ci][x] = dm;
                }
                reset_pred = false;
                since_restart += 1;
            }
            std::mem::swap(&mut prev_diff, &mut cur_diff);
        }
        out.extend_from_slice(&enc.finish());
        out.extend_from_slice(&[0xFF, 0xD9]); // EOI
        out
    }

    /// SOF11 grayscale 8-bit: every Annex H Table H.1 predictor must
    /// round-trip bit-exact through the arithmetic lossless decoder.
    #[test]
    fn sof11_gray8_exact_roundtrip_all_predictors() {
        let w = 24usize;
        let h = 16usize;
        let mut plane = vec![0u32; w * h];
        for j in 0..h {
            for i in 0..w {
                plane[j * w + i] =
                    ((i as i32 * 3 + j as i32 * 5 + ((i ^ j) as i32 & 7)) & 0xFF) as u32;
            }
        }
        for predictor in 1..=7u8 {
            let data = encode_sof11_jpeg(w, h, &[plane.clone()], 8, predictor, 0, 0, None);
            assert!(
                data.windows(2).any(|x| x == [0xFF, 0xCB]),
                "SOF11 marker missing"
            );
            let v = decode_jpeg(&data, None).unwrap();
            assert_eq!(v.planes.len(), 1);
            assert_eq!(v.planes[0].stride, w);
            for j in 0..h {
                for i in 0..w {
                    assert_eq!(
                        v.planes[0].data[j * w + i] as u32,
                        plane[j * w + i],
                        "pred {predictor} mismatch at ({i},{j})"
                    );
                }
            }
        }
    }

    /// SOF11 grayscale at the full 16-bit precision: pseudorandom samples
    /// drive large modulo-2^16 differences through the deep end of the
    /// Table H.3 magnitude tree.
    #[test]
    fn sof11_gray16_exact_roundtrip() {
        let w = 16usize;
        let h = 12usize;
        let mut s = 0x9E37_79B9u32;
        let mut plane = vec![0u32; w * h];
        for v in plane.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *v = s & 0xFFFF;
        }
        let data = encode_sof11_jpeg(w, h, &[plane.clone()], 16, 4, 0, 0, None);
        let v = decode_jpeg(&data, None).unwrap();
        assert_eq!(v.planes.len(), 1);
        assert_eq!(v.planes[0].stride, w * 2);
        for i in 0..w * h {
            let got =
                u16::from_le_bytes([v.planes[0].data[i * 2], v.planes[0].data[i * 2 + 1]]) as u32;
            assert_eq!(got, plane[i], "mismatch at sample {i}");
        }
    }

    /// SOF11 three-component 8-bit (RGB-class): decodes to one packed
    /// `Rgb24` plane carrying the SOS scan-order components, bit-exact.
    #[test]
    fn sof11_rgb8_exact_roundtrip() {
        let w = 20usize;
        let h = 14usize;
        let mut planes = vec![vec![0u32; w * h]; 3];
        for j in 0..h {
            for i in 0..w {
                planes[0][j * w + i] = ((i * 11 + j * 3) & 0xFF) as u32;
                planes[1][j * w + i] = ((i * 2 + j * 17) & 0xFF) as u32;
                planes[2][j * w + i] = (((i ^ j) * 29) & 0xFF) as u32;
            }
        }
        let data = encode_sof11_jpeg(w, h, &planes, 8, 1, 0, 0, None);
        let v = decode_jpeg(&data, None).unwrap();
        assert_eq!(v.planes.len(), 1);
        assert_eq!(v.planes[0].stride, w * 3);
        for i in 0..w * h {
            for (ci, plane) in planes.iter().enumerate() {
                assert_eq!(
                    v.planes[0].data[i * 3 + ci] as u32,
                    plane[i],
                    "component {ci} mismatch at sample {i}"
                );
            }
        }
    }

    /// SOF11 with a DRI restart interval (one MCU-row per interval —
    /// §H.1.1 requires the interval be a multiple of the MCUs per row):
    /// each RSTn re-seeds the coder, statistics, conditioning and
    /// prediction, and the stream still round-trips bit-exact.
    #[test]
    fn sof11_restart_interval_roundtrip() {
        let w = 16usize;
        let h = 24usize;
        let mut plane = vec![0u32; w * h];
        for j in 0..h {
            for i in 0..w {
                plane[j * w + i] = ((i * 7 + j * 13 + (i & j)) & 0xFF) as u32;
            }
        }
        // Interval = 2 MCU-rows; 24 lines → 11 restart markers.
        let data = encode_sof11_jpeg(w, h, &[plane.clone()], 8, 5, 0, (w * 2) as u16, None);
        assert!(
            data.windows(2)
                .any(|x| x[0] == 0xFF && (0xD0..=0xD7).contains(&x[1])),
            "no RSTn marker found in the scan"
        );
        let v = decode_jpeg(&data, None).unwrap();
        for i in 0..w * h {
            assert_eq!(v.planes[0].data[i] as u32, plane[i], "mismatch at {i}");
        }
    }

    /// SOF11 with DAC-overridden conditioning bounds (L = 2, U = 5):
    /// both sides must classify Da / Db with the same non-default
    /// small/large boundaries or the bin streams desynchronise.
    #[test]
    fn sof11_dac_conditioning_roundtrip() {
        let w = 24usize;
        let h = 16usize;
        let mut plane = vec![0u32; w * h];
        for j in 0..h {
            for i in 0..w {
                plane[j * w + i] = ((i * 19 + j * 31 + ((i * j) & 15)) & 0xFF) as u32;
            }
        }
        let data = encode_sof11_jpeg(w, h, &[plane.clone()], 8, 2, 0, 0, Some((2, 5)));
        let v = decode_jpeg(&data, None).unwrap();
        for i in 0..w * h {
            assert_eq!(v.planes[0].data[i] as u32, plane[i], "mismatch at {i}");
        }
    }

    /// SOF11 with a non-zero point transform: samples are coded in
    /// `P − Pt` bits and the decoder output is left-shifted by Pt
    /// (§H.2.2).
    #[test]
    fn sof11_point_transform_roundtrip() {
        let w = 12usize;
        let h = 10usize;
        let pt = 2u8;
        // Pre-shift samples: 6 significant bits at P = 8, Pt = 2.
        let mut plane = vec![0u32; w * h];
        for j in 0..h {
            for i in 0..w {
                plane[j * w + i] = ((i * 5 + j * 9) & 0x3F) as u32;
            }
        }
        let data = encode_sof11_jpeg(w, h, &[plane.clone()], 8, 1, pt, 0, None);
        let v = decode_jpeg(&data, None).unwrap();
        for i in 0..w * h {
            assert_eq!(
                v.planes[0].data[i] as u32,
                plane[i] << pt,
                "mismatch at {i}"
            );
        }
    }

    /// SOF13 (differential sequential, arithmetic) is still rejected —
    /// adding SOF10 must not have widened the accept matcher into the
    /// hierarchical-arithmetic neighbours.
    #[test]
    fn sof13_differential_arithmetic_still_rejected() {
        let bytes = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xCD, 0x00, 0x08, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, // SOF13 stub
            0xFF, 0xD9, // EOI
        ];
        let err = decode_jpeg(&bytes, None).expect_err("expected decode error");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }

    /// Hierarchical (SOF5/6/7) and SOF13..15 arithmetic variants must
    /// still be rejected with Unsupported — make sure we didn't widen
    /// the SOF accept matcher too far while adding SOF3 or SOF9.
    /// SOF9 / SOF10 / SOF11 (arithmetic) are now handled separately.
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

#[cfg(test)]
mod sof10_tests {
    use super::decode_jpeg;
    use crate::jpeg::arith::{encode_magnitude, AcStats, ArithEncoder, Context, DcStats};
    use crate::jpeg::dct::idct8x8;
    use crate::jpeg::zigzag::ZIGZAG;

    /// Append one length-prefixed marker segment (T.81 §B.1.1.4).
    fn put_seg(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        out.extend_from_slice(&[0xFF, marker]);
        out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(payload);
    }

    /// Code one decision with the fixed 0.5 estimate (Qe = 0x5A1D,
    /// MPS = 0, never adapts) — encoder mirror of
    /// `arith::decode_fixed_bit`.
    fn code_fixed_bit(e: &mut ArithEncoder, bit: u8) {
        let mut ctx = Context { idx: 0, mps: 0 };
        e.code_bit(&mut ctx, bit);
    }

    /// Encoder-side §F.1.4.1 DC difference (mirror of
    /// `arith::decode_dc_diff`).
    fn encode_dc_diff(e: &mut ArithEncoder, dc: &mut DcStats, diff: i32) {
        let s0 = dc.dc_context();
        if diff == 0 {
            e.code_bit(&mut dc.bins[s0], 0);
            dc.prev_diff = 0;
            return;
        }
        e.code_bit(&mut dc.bins[s0], 1);
        let sign = u8::from(diff < 0);
        e.code_bit(&mut dc.bins[s0 + 1], sign);
        let sx = s0 + 2 + sign as usize;
        // X1 base for DC is bin 20 (Table F.4); M-bins shadow X-bins at +14.
        encode_magnitude(e, &mut dc.bins, sx, 20, diff.unsigned_abs() - 1);
        dc.prev_diff = diff;
    }

    /// DC progressive coding for one block (§G.1.3.1). `full` is the
    /// full-precision DC value; first scans code the point-transformed
    /// difference, refinement scans code one fixed-estimate LSB.
    fn encode_dc_unit(e: &mut ArithEncoder, dc: &mut DcStats, full: i32, ah: u8, al: u8) {
        if ah == 0 {
            let vt = full >> al; // DC point transform: arithmetic shift
            let diff = vt - dc.pred;
            encode_dc_diff(e, dc, diff);
            dc.pred = vt;
        } else {
            code_fixed_bit(e, ((full >> al) & 1) as u8);
        }
    }

    /// Encoder-side AC magnitude with the Table F.5 bin layout
    /// (SP = SN = X1 = S0 + 1; X2.. at 189 / 217 by `K <= Kx`; M-bins
    /// shadow X-bins at +14) — mirror of the AC arm of
    /// `arith::decode_magnitude`.
    fn encode_ac_mag(e: &mut ArithEncoder, bins: &mut [Context], k: usize, kx: u8, sz: u32) {
        let s_first = 3 * (k - 1) + 2;
        if sz == 0 {
            e.code_bit(&mut bins[s_first], 0);
            return;
        }
        e.code_bit(&mut bins[s_first], 1);
        if sz < 2 {
            // X1 coincides with the first-magnitude bin for AC.
            e.code_bit(&mut bins[s_first], 0);
            return;
        }
        e.code_bit(&mut bins[s_first], 1);
        let mut m = 4u32;
        let mut s = if (k as u8) <= kx { 189 } else { 217 };
        while sz >= m {
            e.code_bit(&mut bins[s], 1);
            m <<= 1;
            s += 1;
        }
        e.code_bit(&mut bins[s], 0);
        let m_bin = s + 14;
        let mut bit = m >> 2;
        while bit != 0 {
            e.code_bit(&mut bins[m_bin], u8::from(sz & bit != 0));
            bit >>= 1;
        }
    }

    /// First scan of a band for one block (§G.1.3.2 — the §F.1.4 AC
    /// procedure with Kmin = Ss and EOB = end-of-band). `vals` holds the
    /// point-transformed coefficients in zigzag-index order.
    fn encode_ac_band(
        e: &mut ArithEncoder,
        ac: &mut AcStats,
        vals: &[i32; 64],
        ss: usize,
        se: usize,
    ) {
        let mut eob = ss;
        for k in ss..=se {
            if vals[k] != 0 {
                eob = k + 1;
            }
        }
        let mut k = ss;
        loop {
            let se_bin = 3 * (k - 1);
            if k >= eob {
                e.code_bit(&mut ac.bins[se_bin], 1);
                return;
            }
            e.code_bit(&mut ac.bins[se_bin], 0);
            while vals[k] == 0 {
                e.code_bit(&mut ac.bins[3 * (k - 1) + 1], 0);
                k += 1;
            }
            e.code_bit(&mut ac.bins[3 * (k - 1) + 1], 1);
            let v = vals[k];
            code_fixed_bit(e, u8::from(v < 0));
            encode_ac_mag(e, &mut ac.bins, k, ac.kx, v.unsigned_abs() - 1);
            if k == se {
                return;
            }
            k += 1;
        }
    }

    /// Refinement scan of a band for one block (Figures G.10 / G.11,
    /// Table G.2) — encoder mirror of `arith::decode_ac_refine`. `full`
    /// holds the full-precision coefficients in zigzag-index order.
    fn encode_ac_refine_band(
        e: &mut ArithEncoder,
        bins: &mut [Context; 189],
        full: &[i32; 64],
        ss: usize,
        se: usize,
        al: u8,
    ) {
        let shifted = |k: usize| (full[k].unsigned_abs() >> al) as i32;
        let hist = |k: usize| (full[k].unsigned_abs() >> (al + 1)) as i32;
        let mut eob = ss;
        let mut eobx = ss;
        for k in ss..=se {
            if shifted(k) != 0 {
                eob = k + 1;
            }
            if hist(k) != 0 {
                eobx = k + 1;
            }
        }
        let mut k = ss;
        loop {
            if k >= eobx {
                let se_bin = 3 * (k - 1);
                if k >= eob {
                    e.code_bit(&mut bins[se_bin], 1);
                    return;
                }
                e.code_bit(&mut bins[se_bin], 0);
            }
            loop {
                if hist(k) != 0 {
                    // Nonzero history → one correction bit (SC).
                    let t = ((full[k].unsigned_abs() >> al) & 1) as u8;
                    e.code_bit(&mut bins[3 * (k - 1) + 2], t);
                    break;
                }
                if shifted(k) != 0 {
                    // Newly nonzero at this precision: V = 0 decision then
                    // the fixed-estimate sign.
                    e.code_bit(&mut bins[3 * (k - 1) + 1], 1);
                    code_fixed_bit(e, u8::from(full[k] < 0));
                    break;
                }
                e.code_bit(&mut bins[3 * (k - 1) + 1], 0);
                k += 1;
            }
            if k == se {
                return;
            }
            k += 1;
        }
    }

    /// One scan-header description: which SOF component indices it
    /// covers, plus (Ss, Se, Ah, Al).
    type ScanDesc = (Vec<usize>, u8, u8, u8, u8);

    /// Build a complete SOF10 (progressive, arithmetic-coded) JPEG from
    /// full-precision zigzag-order coefficient blocks — the encoder-side
    /// mirror of `decode_progressive_arith_scan` per T.81 §G.1.3, used as
    /// test scaffolding for round-trip verification. All quantiser values
    /// are 1 so the decoded pixels are a pure IDCT of the coefficients.
    #[allow(clippy::too_many_arguments)]
    fn encode_sof10_jpeg(
        width: usize,
        height: usize,
        precision: u8,
        comps: &[(u8, u8)],
        blocks: &[Vec<[i32; 64]>],
        scans: &[ScanDesc],
        restart_interval: u16,
        kx_dac: Option<u8>,
    ) -> Vec<u8> {
        let nc = comps.len();
        let mut out = vec![0xFF, 0xD8]; // SOI
                                        // DQT: table 0, Pq = 0, all values 1 (identity dequantise).
        let mut dqt = vec![0u8];
        dqt.extend(std::iter::repeat(1u8).take(64));
        put_seg(&mut out, 0xDB, &dqt);
        if let Some(kx) = kx_dac {
            // DAC entry: Tc = 1 (AC), Tb = 0, Cs = Kx (T.81 §B.2.4.3).
            put_seg(&mut out, 0xCC, &[0x10, kx]);
        }
        if restart_interval != 0 {
            put_seg(&mut out, 0xDD, &restart_interval.to_be_bytes());
        }
        let mut sof = vec![precision];
        sof.extend_from_slice(&(height as u16).to_be_bytes());
        sof.extend_from_slice(&(width as u16).to_be_bytes());
        sof.push(nc as u8);
        for (ci, (h, v)) in comps.iter().enumerate() {
            sof.extend_from_slice(&[ci as u8 + 1, (h << 4) | v, 0]);
        }
        put_seg(&mut out, 0xCA, &sof); // SOF10

        let h_max = comps.iter().map(|c| c.0).max().unwrap() as usize;
        let v_max = comps.iter().map(|c| c.1).max().unwrap() as usize;
        let mcus_x = width.div_ceil(8 * h_max);
        let mcus_y = height.div_ceil(8 * v_max);
        let kx = kx_dac.unwrap_or(5);

        for (scomps, ss, se, ah, al) in scans {
            let mut sos = vec![scomps.len() as u8];
            for &ci in scomps {
                sos.extend_from_slice(&[ci as u8 + 1, 0x00]);
            }
            sos.extend_from_slice(&[*ss, *se, (ah << 4) | al]);
            put_seg(&mut out, 0xDA, &sos);

            let is_dc = *ss == 0;
            let interleaved = is_dc && scomps.len() > 1;
            let (sm_x, sm_y) = if interleaved {
                (mcus_x, mcus_y)
            } else {
                let (h, v) = comps[scomps[0]];
                (mcus_x * h as usize, mcus_y * v as usize)
            };

            let mut dc_stats: Vec<DcStats> = (0..scomps.len()).map(|_| DcStats::new()).collect();
            let mut ac = AcStats::new();
            ac.kx = kx;
            let mut refine_bins = [Context::default(); 189];
            let mut enc = ArithEncoder::new();
            let mut since_restart = 0u32;
            let mut rst = 0u8;

            for my in 0..sm_y {
                for mx in 0..sm_x {
                    if restart_interval != 0
                        && since_restart != 0
                        && since_restart % restart_interval as u32 == 0
                    {
                        out.extend_from_slice(&std::mem::take(&mut enc).finish());
                        out.extend_from_slice(&[0xFF, 0xD0 + rst]);
                        rst = (rst + 1) % 8;
                        for s in dc_stats.iter_mut() {
                            *s = DcStats::new();
                        }
                        ac.reset();
                        refine_bins = [Context::default(); 189];
                    }
                    if is_dc {
                        for (sidx, &ci) in scomps.iter().enumerate() {
                            let (h, v) = comps[ci];
                            let blocks_x = mcus_x * h as usize;
                            if interleaved {
                                for by in 0..v as usize {
                                    for bx in 0..h as usize {
                                        let bi = (my * v as usize + by) * blocks_x
                                            + mx * h as usize
                                            + bx;
                                        encode_dc_unit(
                                            &mut enc,
                                            &mut dc_stats[sidx],
                                            blocks[ci][bi][0],
                                            *ah,
                                            *al,
                                        );
                                    }
                                }
                            } else {
                                let bi = my * blocks_x + mx;
                                encode_dc_unit(
                                    &mut enc,
                                    &mut dc_stats[sidx],
                                    blocks[ci][bi][0],
                                    *ah,
                                    *al,
                                );
                            }
                        }
                    } else {
                        let ci = scomps[0];
                        let (h, _) = comps[ci];
                        let blocks_x = mcus_x * h as usize;
                        let bi = my * blocks_x + mx;
                        let vals = &blocks[ci][bi];
                        if *ah == 0 {
                            // Point transform: the AC transform divides the
                            // magnitude (§G.1.2.1 / §G.1.3.2).
                            let mut tv = [0i32; 64];
                            for k in *ss as usize..=*se as usize {
                                let m = (vals[k].unsigned_abs() >> al) as i32;
                                tv[k] = if vals[k] < 0 { -m } else { m };
                            }
                            encode_ac_band(&mut enc, &mut ac, &tv, *ss as usize, *se as usize);
                        } else {
                            encode_ac_refine_band(
                                &mut enc,
                                &mut refine_bins,
                                vals,
                                *ss as usize,
                                *se as usize,
                                *al,
                            );
                        }
                    }
                    since_restart += 1;
                }
            }
            out.extend_from_slice(&enc.finish());
        }
        out.extend_from_slice(&[0xFF, 0xD9]); // EOI
        out
    }

    /// xorshift32 — deterministic coefficient fixtures, no payload files.
    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
    }

    /// Generate `n` coefficient blocks (zigzag-index order): bounded DC
    /// plus `ac_count` sparse AC values within ±`ac_amp`.
    fn gen_blocks(
        n: usize,
        seed: u32,
        dc_amp: i32,
        ac_amp: i32,
        ac_count: usize,
    ) -> Vec<[i32; 64]> {
        let mut rng = Rng(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut b = [0i32; 64];
            b[0] = (rng.next() % (2 * dc_amp as u32 + 1)) as i32 - dc_amp;
            for _ in 0..ac_count {
                let k = 1 + (rng.next() as usize) % 63;
                let v = (rng.next() % (2 * ac_amp as u32 + 1)) as i32 - ac_amp;
                b[k] = v;
            }
            out.push(b);
        }
        out
    }

    /// Reference render: identity dequantise + IDCT + level shift, exactly
    /// as `render_from_coefs` does for an 8-bit component, cropped to
    /// (w, h).
    fn expected_plane_8(blocks: &[[i32; 64]], blocks_x: usize, w: usize, h: usize) -> Vec<u8> {
        let bw = blocks_x * 8;
        let blocks_y = blocks.len() / blocks_x;
        let mut full = vec![0u8; bw * blocks_y * 8];
        for (bi, vals) in blocks.iter().enumerate() {
            let mut nat = [0.0f32; 64];
            for k in 0..64 {
                nat[ZIGZAG[k]] = vals[k] as f32;
            }
            idct8x8(&mut nat);
            let bx = bi % blocks_x;
            let by = bi / blocks_x;
            for j in 0..8 {
                for i in 0..8 {
                    let v = nat[j * 8 + i] + 128.0;
                    let px = if v <= 0.0 {
                        0
                    } else if v >= 255.0 {
                        255
                    } else {
                        v.round() as u8
                    };
                    full[(by * 8 + j) * bw + bx * 8 + i] = px;
                }
            }
        }
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            out[y * w..y * w + w].copy_from_slice(&full[y * bw..y * bw + w]);
        }
        out
    }

    /// 12-bit variant: level shift 2048, clamp 0..=4095.
    fn expected_plane_12(blocks: &[[i32; 64]], blocks_x: usize, w: usize, h: usize) -> Vec<u16> {
        let bw = blocks_x * 8;
        let blocks_y = blocks.len() / blocks_x;
        let mut full = vec![0u16; bw * blocks_y * 8];
        for (bi, vals) in blocks.iter().enumerate() {
            let mut nat = [0.0f32; 64];
            for k in 0..64 {
                nat[ZIGZAG[k]] = vals[k] as f32;
            }
            idct8x8(&mut nat);
            let bx = bi % blocks_x;
            let by = bi / blocks_x;
            for j in 0..8 {
                for i in 0..8 {
                    let v = nat[j * 8 + i] + 2048.0;
                    let px = if v <= 0.0 {
                        0
                    } else if v >= 4095.0 {
                        4095
                    } else {
                        v.round() as u16
                    };
                    full[(by * 8 + j) * bw + bx * 8 + i] = px;
                }
            }
        }
        let mut out = vec![0u16; w * h];
        for y in 0..h {
            out[y * w..y * w + w].copy_from_slice(&full[y * bw..y * bw + w]);
        }
        out
    }

    fn assert_gray8_exact(jpeg: &[u8], blocks: &[[i32; 64]], blocks_x: usize, w: usize, h: usize) {
        let v = decode_jpeg(jpeg, None).expect("decode SOF10");
        assert_eq!(v.planes.len(), 1);
        assert_eq!(v.planes[0].stride, w);
        let want = expected_plane_8(blocks, blocks_x, w, h);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    v.planes[0].data[y * w + x],
                    want[y * w + x],
                    "pixel mismatch at ({x},{y})"
                );
            }
        }
    }

    /// Spectral selection only (Annex G process 2): DC scan + two AC band
    /// scans, no successive approximation. Bit-exact coefficient recovery
    /// means the decoded pixels equal a direct IDCT of the source blocks.
    #[test]
    fn sof10_gray8_spectral_selection_roundtrip() {
        let (w, h) = (24usize, 16usize);
        let blocks = gen_blocks(3 * 2, 0xC0FFEE11, 200, 60, 10);
        let scans: Vec<ScanDesc> = vec![
            (vec![0], 0, 0, 0, 0),
            (vec![0], 1, 5, 0, 0),
            (vec![0], 6, 63, 0, 0),
        ];
        let jpeg = encode_sof10_jpeg(
            w,
            h,
            8,
            &[(1, 1)],
            std::slice::from_ref(&blocks),
            &scans,
            0,
            None,
        );
        assert!(
            jpeg.windows(2).any(|x| x == [0xFF, 0xCA]),
            "SOF10 marker missing"
        );
        assert_gray8_exact(&jpeg, &blocks, 3, w, h);
    }

    /// Full progression (Annex G process 4): 1-bit point transform on
    /// every first scan, then DC + AC refinement scans restore the LSBs.
    #[test]
    fn sof10_gray8_full_progression_roundtrip() {
        let (w, h) = (24usize, 16usize);
        let blocks = gen_blocks(3 * 2, 0xDEC0DE22, 180, 50, 12);
        let scans: Vec<ScanDesc> = vec![
            (vec![0], 0, 0, 0, 1),
            (vec![0], 1, 5, 0, 1),
            (vec![0], 6, 63, 0, 1),
            (vec![0], 0, 0, 1, 0),
            (vec![0], 1, 5, 1, 0),
            (vec![0], 6, 63, 1, 0),
        ];
        let jpeg = encode_sof10_jpeg(
            w,
            h,
            8,
            &[(1, 1)],
            std::slice::from_ref(&blocks),
            &scans,
            0,
            None,
        );
        assert_gray8_exact(&jpeg, &blocks, 3, w, h);
    }

    /// Two successive-approximation levels (Al = 2 first scans, then two
    /// refinement passes) — exercises EOBx growth across refinement scans.
    #[test]
    fn sof10_gray8_two_level_sa_roundtrip() {
        let (w, h) = (16usize, 16usize);
        let blocks = gen_blocks(2 * 2, 0x5EED3333, 120, 40, 14);
        let scans: Vec<ScanDesc> = vec![
            (vec![0], 0, 0, 0, 2),
            (vec![0], 1, 63, 0, 2),
            (vec![0], 0, 0, 2, 1),
            (vec![0], 1, 63, 2, 1),
            (vec![0], 0, 0, 1, 0),
            (vec![0], 1, 63, 1, 0),
        ];
        let jpeg = encode_sof10_jpeg(
            w,
            h,
            8,
            &[(1, 1)],
            std::slice::from_ref(&blocks),
            &scans,
            0,
            None,
        );
        assert_gray8_exact(&jpeg, &blocks, 2, w, h);
    }

    /// 4:2:0 three-component: interleaved DC scan (4 luma + 1 Cb + 1 Cr
    /// blocks per MCU) followed by per-component full-band AC scans.
    /// Every output plane is compared sample-exact against the IDCT
    /// reference at its own resolution.
    #[test]
    fn sof10_yuv420_interleaved_dc_roundtrip() {
        let (w, h) = (32usize, 16usize);
        // mcus 2x1: luma 4x2 blocks, chroma 2x1 blocks each.
        let comps = [(2u8, 2u8), (1, 1), (1, 1)];
        let blocks = vec![
            gen_blocks(4 * 2, 0xAAAA0001, 150, 40, 8),
            gen_blocks(2, 0xBBBB0002, 100, 30, 6),
            gen_blocks(2, 0xCCCC0003, 100, 30, 6),
        ];
        let scans: Vec<ScanDesc> = vec![
            (vec![0, 1, 2], 0, 0, 0, 0),
            (vec![0], 1, 63, 0, 0),
            (vec![1], 1, 63, 0, 0),
            (vec![2], 1, 63, 0, 0),
        ];
        let jpeg = encode_sof10_jpeg(w, h, 8, &comps, &blocks, &scans, 0, None);
        let v = decode_jpeg(&jpeg, None).expect("decode SOF10 4:2:0");
        assert_eq!(v.planes.len(), 3);
        let dims = [(w, h, 4usize), (w / 2, h / 2, 2), (w / 2, h / 2, 2)];
        for ci in 0..3 {
            let (cw, ch, bx) = dims[ci];
            assert_eq!(v.planes[ci].stride, cw, "plane {ci} stride");
            let want = expected_plane_8(&blocks[ci], bx, cw, ch);
            assert_eq!(v.planes[ci].data, want, "plane {ci} samples");
        }
    }

    /// Restart markers: stats + DC prediction + Q-coder re-initialise at
    /// every RSTn within every scan, in lockstep with the encoder.
    #[test]
    fn sof10_restart_interval_roundtrip() {
        let (w, h) = (32usize, 16usize);
        let blocks = gen_blocks(4 * 2, 0x12345678, 160, 45, 9);
        let scans: Vec<ScanDesc> = vec![
            (vec![0], 0, 0, 0, 1),
            (vec![0], 1, 63, 0, 1),
            (vec![0], 0, 0, 1, 0),
            (vec![0], 1, 63, 1, 0),
        ];
        let jpeg = encode_sof10_jpeg(
            w,
            h,
            8,
            &[(1, 1)],
            std::slice::from_ref(&blocks),
            &scans,
            3,
            None,
        );
        assert!(
            jpeg.windows(2)
                .any(|x| x[0] == 0xFF && (0xD0..=0xD7).contains(&x[1])),
            "no RSTn marker found in the scan"
        );
        assert_gray8_exact(&jpeg, &blocks, 4, w, h);
    }

    /// DAC-overridden AC conditioning (Kx = 20): both sides must place the
    /// X2.. magnitude bins on the same low/high side of the threshold or
    /// the bin streams desynchronise.
    #[test]
    fn sof10_dac_kx_conditioning_roundtrip() {
        let (w, h) = (16usize, 16usize);
        let blocks = gen_blocks(2 * 2, 0x0BAD5EED, 140, 50, 16);
        let scans: Vec<ScanDesc> = vec![(vec![0], 0, 0, 0, 0), (vec![0], 1, 63, 0, 0)];
        let jpeg = encode_sof10_jpeg(
            w,
            h,
            8,
            &[(1, 1)],
            std::slice::from_ref(&blocks),
            &scans,
            0,
            Some(20),
        );
        assert_gray8_exact(&jpeg, &blocks, 2, w, h);
    }

    /// 12-bit grayscale full progression (Annex G process 8): wider
    /// coefficients through the deeper end of the magnitude tree, output
    /// as little-endian `u16` samples with the 2048 level shift.
    #[test]
    fn sof10_gray12_full_progression_roundtrip() {
        let (w, h) = (16usize, 16usize);
        let blocks = gen_blocks(2 * 2, 0x600DCAFE, 4000, 900, 10);
        let scans: Vec<ScanDesc> = vec![
            (vec![0], 0, 0, 0, 1),
            (vec![0], 1, 63, 0, 1),
            (vec![0], 0, 0, 1, 0),
            (vec![0], 1, 63, 1, 0),
        ];
        let jpeg = encode_sof10_jpeg(
            w,
            h,
            12,
            &[(1, 1)],
            std::slice::from_ref(&blocks),
            &scans,
            0,
            None,
        );
        let v = decode_jpeg(&jpeg, None).expect("decode SOF10 12-bit");
        assert_eq!(v.planes.len(), 1);
        assert_eq!(v.planes[0].stride, w * 2);
        let want = expected_plane_12(&blocks, 2, w, h);
        for i in 0..w * h {
            let got = u16::from_le_bytes([v.planes[0].data[i * 2], v.planes[0].data[i * 2 + 1]]);
            assert_eq!(got, want[i], "sample mismatch at {i}");
        }
    }

    /// 4-component (CMYK-class, no APP14 → plain pass-through) progressive
    /// arithmetic: interleaved DC + per-component AC scans, packed `Cmyk`
    /// output compared sample-exact.
    #[test]
    fn sof10_cmyk_spectral_selection_roundtrip() {
        let (w, h) = (16usize, 8usize);
        let comps = [(1u8, 1u8); 4];
        let blocks: Vec<Vec<[i32; 64]>> = (0..4)
            .map(|ci| gen_blocks(2, 0x4444_0000 + ci as u32, 120, 35, 7))
            .collect();
        let mut scans: Vec<ScanDesc> = vec![(vec![0, 1, 2, 3], 0, 0, 0, 0)];
        for ci in 0..4 {
            scans.push((vec![ci], 1, 63, 0, 0));
        }
        let jpeg = encode_sof10_jpeg(w, h, 8, &comps, &blocks, &scans, 0, None);
        let v = decode_jpeg(&jpeg, None).expect("decode SOF10 CMYK");
        assert_eq!(v.planes.len(), 1);
        assert_eq!(v.planes[0].stride, w * 4);
        let want: Vec<Vec<u8>> = blocks
            .iter()
            .map(|b| expected_plane_8(b, 2, w, h))
            .collect();
        for i in 0..w * h {
            for ci in 0..4 {
                assert_eq!(
                    v.planes[0].data[i * 4 + ci],
                    want[ci][i],
                    "component {ci} mismatch at sample {i}"
                );
            }
        }
    }

    /// SOF10 only accepts P = 8 / P = 12 — a 10-bit frame is rejected
    /// with `Unsupported`, not a desynchronised decode.
    #[test]
    fn sof10_unsupported_precision_rejected() {
        let (w, h) = (8usize, 8usize);
        let blocks = gen_blocks(1, 1, 50, 10, 3);
        let scans: Vec<ScanDesc> = vec![(vec![0], 0, 0, 0, 0)];
        let jpeg = encode_sof10_jpeg(w, h, 10, &[(1, 1)], &[blocks], &scans, 0, None);
        let err = decode_jpeg(&jpeg, None).expect_err("expected decode error");
        assert!(
            matches!(err, crate::error::MjpegError::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }
}

#[cfg(test)]
mod dnl_unit_tests {
    use super::resolve_dnl_height;
    use crate::jpeg::markers;

    /// Append one length-prefixed marker segment (T.81 §B.1.1.4).
    fn put_seg(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        out.extend_from_slice(&[0xFF, marker]);
        out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(payload);
    }

    /// Build a minimal single-component frame whose SOF codes the given
    /// `y` value, optionally followed by a `DNL nl` segment after the
    /// (empty placeholder) first scan. Returns the post-SOI byte slice
    /// that `resolve_dnl_height` consumes.
    fn build(y: u16, dnl: Option<u16>) -> Vec<u8> {
        let mut out = Vec::new();
        // SOF0: P=8, Y, X=16, Nf=1, one component 1x1 / Tq=0.
        let mut sof = vec![8u8];
        sof.extend_from_slice(&y.to_be_bytes());
        sof.extend_from_slice(&16u16.to_be_bytes());
        sof.push(1); // Nf
        sof.extend_from_slice(&[1, 0x11, 0]);
        put_seg(&mut out, markers::SOF0, &sof);
        // SOS: Ns=1, Cs=1 Td/Ta=0, Ss=0 Se=63 Ah/Al=0.
        put_seg(&mut out, markers::SOS, &[1, 1, 0, 0, 63, 0]);
        // A token entropy byte so the scan is non-empty.
        out.push(0x00);
        if let Some(nl) = dnl {
            put_seg(&mut out, markers::DNL, &nl.to_be_bytes());
        }
        out.extend_from_slice(&[0xFF, markers::EOI]);
        out
    }

    #[test]
    fn non_zero_y_needs_no_resolution() {
        let data = build(16, None);
        assert_eq!(resolve_dnl_height(&data).unwrap(), None);
    }

    #[test]
    fn zero_y_with_dnl_resolves() {
        let data = build(0, Some(123));
        assert_eq!(resolve_dnl_height(&data).unwrap(), Some(123));
    }

    #[test]
    fn zero_y_without_dnl_errors() {
        let data = build(0, None);
        assert!(resolve_dnl_height(&data).is_err());
    }

    #[test]
    fn zero_y_zero_nl_errors() {
        let data = build(0, Some(0));
        assert!(resolve_dnl_height(&data).is_err());
    }
}
