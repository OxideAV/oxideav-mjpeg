//! Decode-free JPEG inspector.
//!
//! Walks the marker structure of a JPEG byte stream (T.81 §B.1) just
//! far enough to classify the frame variant and pull out the descriptive
//! metadata in the SOF segment. **No entropy decoding, no DCT, no
//! quantization, no colour conversion.** The inspector reads only:
//!
//! * The SOI marker (T.81 §B.1.1.3) at offset 0.
//! * The first APP0/APP14 segments encountered before SOS — these carry
//!   the JFIF magic (T.871) and the Adobe colour-transform tag (T.872
//!   §6.5.3) and let the inspector report a richer colour hint without
//!   guessing from component IDs.
//! * A single DRI segment, if one appears before SOS (T.81 §B.2.4.4).
//! * The SOF segment (T.81 §B.2.2 — Table B.2): precision `P`, lines
//!   `Y`, samples-per-line `X`, number of components `Nf`, and each
//!   component's horizontal / vertical sampling factors plus its
//!   destination quantization-table selector.
//!
//! The walker stops at the **first** SOS marker. The scan body is never
//! touched, restart markers in the scan are never followed, and any
//! second SOF/SOS that a multi-frame hierarchical stream might contain
//! is invisible to the inspector by design — the function reports the
//! variant of the *first* frame, which is the variant the matching
//! decoder will see when handed the same bytes.
//!
//! ## Why this exists
//!
//! Application code that just needs to triage a JPEG (pick a target
//! pixel format, decide whether to fall back to a different decoder,
//! emit a thumbnail-pipeline routing decision, log a corpus summary)
//! shouldn't have to spin up the full Huffman / arithmetic / DCT
//! pipeline. The decoder happily reports `Unsupported` for SOF5/SOF7
//! hierarchical, SOF10..SOF12 / SOF14..SOF15 arithmetic, etc., but
//! that costs a `make_decoder` + `send_packet` + `receive_frame`
//! round trip plus an error-path allocation. The inspector returns
//! the same classification from a single linear marker walk.
//!
//! ## Cost
//!
//! O(prefix-length) — the walker stops at SOS. For a typical baseline
//! JPEG with one DQT and one DHT before SOS, that's ~200 bytes
//! regardless of the image dimensions.

use crate::error::{MjpegError as Error, Result};

use super::markers;
use super::parser::{parse_dri, parse_jfif_app0, parse_sof, JfifApp0, MarkerWalker};

pub use super::parser::JfifUnits;

/// Coarse classification of the SOF marker that opened the frame.
///
/// Matches the T.81 §B.1.1.3 / Table B.1 SOFn enumeration. The two
/// `Hierarchical` variants collapse SOF5/SOF7 and SOF13/SOF15 into
/// "Hierarchical (DCT)" / "Hierarchical (arithmetic / lossless)" — the
/// hierarchical-frame-set differentiation matters for routing but
/// not for the inspector's other fields, which describe the first
/// (lowest-resolution) hierarchical frame only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SofKind {
    /// SOF0 (0xFFC0) — baseline DCT, sequential, 8-bit Huffman.
    Baseline,
    /// SOF1 (0xFFC1) — extended sequential DCT, 8-bit or 12-bit
    /// Huffman.
    ExtendedSequential,
    /// SOF2 (0xFFC2) — progressive DCT, 8-bit or 12-bit Huffman.
    Progressive,
    /// SOF3 (0xFFC3) — lossless (per-sample predictor), Huffman.
    Lossless,
    /// SOF9 (0xFFC9) — extended sequential DCT, arithmetic-coded.
    ExtendedSequentialArith,
    /// SOF10 (0xFFCA) — progressive DCT, arithmetic-coded.
    ProgressiveArith,
    /// SOF11 (0xFFCB) — lossless, arithmetic-coded.
    LosslessArith,
    /// SOF5 / SOF6 / SOF7 — differential / hierarchical DCT (Huffman).
    HierarchicalDct,
    /// SOF13 / SOF14 / SOF15 — differential / hierarchical
    /// arithmetic-coded.
    HierarchicalArith,
}

impl SofKind {
    /// Map the SOF marker byte (T.81 Table B.1, second byte of the
    /// `FF Cn` pair) to a `SofKind`. Returns `None` for non-SOF bytes
    /// — the inspector caller has already filtered via `is_sof` so this
    /// is a total function over the legal SOF subset and a `None`
    /// elsewhere serves as a "should not happen" sentinel.
    fn from_marker(b: u8) -> Option<Self> {
        match b {
            0xC0 => Some(Self::Baseline),
            0xC1 => Some(Self::ExtendedSequential),
            0xC2 => Some(Self::Progressive),
            0xC3 => Some(Self::Lossless),
            0xC5..=0xC7 => Some(Self::HierarchicalDct),
            0xC9 => Some(Self::ExtendedSequentialArith),
            0xCA => Some(Self::ProgressiveArith),
            0xCB => Some(Self::LosslessArith),
            0xCD..=0xCF => Some(Self::HierarchicalArith),
            _ => None,
        }
    }

    /// True for the SOF variants the in-tree decoder is documented to
    /// accept (`lib.rs` module docstring: SOF0 / SOF1 / SOF2 / SOF3 /
    /// SOF9). The "supported" line is data, not a promise; callers
    /// that want to negotiate fallback can read it.
    pub fn is_supported_by_decoder(self) -> bool {
        matches!(
            self,
            Self::Baseline
                | Self::ExtendedSequential
                | Self::Progressive
                | Self::Lossless
                | Self::ExtendedSequentialArith
        )
    }

    /// True for the DCT-based variants (SOF0, SOF1, SOF2, SOF5..SOF7,
    /// SOF9, SOF10, SOF13..SOF15). False for lossless predictor-based
    /// variants (SOF3, SOF11). Callers building a pipeline ("do I
    /// need the DCT path or the predictor path?") read this to skip
    /// IDCT allocation when the answer is no.
    pub fn is_dct(self) -> bool {
        !matches!(self, Self::Lossless | Self::LosslessArith)
    }

    /// True for the arithmetic-coded variants. The two entropy paths
    /// have disjoint table-management state machines (DHT vs DAC).
    pub fn is_arithmetic(self) -> bool {
        matches!(
            self,
            Self::ExtendedSequentialArith
                | Self::ProgressiveArith
                | Self::LosslessArith
                | Self::HierarchicalArith
        )
    }
}

/// Standard chroma-subsampling discriminator inferred from the SOF
/// component sampling factors (T.81 §A.1.1, Table A.4).
///
/// The mapping is the universally-understood one: for a three-component
/// SOF whose first component (luma) has sampling `(Hmax, Vmax)` and the
/// other two have `(H_i, V_i)`, the subsampling is named by the per-axis
/// ratio:
///
/// * `Hmax/H_chroma = 1`, `Vmax/V_chroma = 1` → 4:4:4
/// * `Hmax/H_chroma = 2`, `Vmax/V_chroma = 1` → 4:2:2
/// * `Hmax/H_chroma = 2`, `Vmax/V_chroma = 2` → 4:2:0
/// * `Hmax/H_chroma = 4`, `Vmax/V_chroma = 1` → 4:1:1
///
/// Any other combination — including streams where the two chroma
/// components have unequal sampling — falls into `Custom` because
/// the conventional name doesn't apply.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChromaSubsampling {
    /// Single-component frame; no chroma to subsample.
    GrayscaleOnly,
    /// Three-component 4:4:4 — luma and chroma at the same resolution.
    Yuv444,
    /// Three-component 4:2:2 — chroma half-width.
    Yuv422,
    /// Three-component 4:2:0 — chroma half-width and half-height.
    Yuv420,
    /// Three-component 4:1:1 — chroma quarter-width, full-height.
    Yuv411,
    /// Three-component with unconventional sampling factors (e.g.
    /// asymmetric chroma, 2:1:1 vertical), or any component-count
    /// other than 1 or 3.
    Custom,
}

/// Colour-space hint pulled from the in-band marker segments.
///
/// JPEG itself does not carry a normative colour space — the spec
/// transmits samples and lets the application know what they mean.
/// JFIF (T.871) and the Adobe APP14 tag (T.872 §6.5.3) are the two
/// historical conventions that closed that gap; if neither is present
/// we report `Unspecified` and the caller must infer from component
/// IDs / Nf.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorHint {
    /// No APP0 JFIF magic and no APP14 Adobe tag — colour space
    /// unknown.
    Unspecified,
    /// APP0 `JFIF\0` magic present — components 1/2/3 are Y/Cb/Cr at
    /// BT.601 full range per T.871.
    JfifYCbCr,
    /// APP14 transform = 0 — components are not colour-transformed
    /// (RGB for `Nf=3`, CMYK for `Nf=4`).
    AdobeUntransformed,
    /// APP14 transform = 1 — components are YCbCr (`Nf=3`) per the
    /// Adobe ColorTransform tag.
    AdobeYCbCr,
    /// APP14 transform = 2 — components are YCCK (`Nf=4`), the
    /// Adobe-defined CMYK colour-transform variant.
    AdobeYcck,
}

/// Per-component sampling + table-selector descriptor copied verbatim
/// from the SOF segment (T.81 Table B.2).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InspectedComponent {
    /// Component identifier `Ci`. JFIF convention is 1/2/3 for
    /// Y/Cb/Cr; Adobe RGB JPEGs from photoshop use 'R'/'G'/'B' (0x52
    /// / 0x47 / 0x42). Treated as opaque by the inspector.
    pub id: u8,
    /// Horizontal sampling factor `Hi ∈ 1..=4`.
    pub h_sampling: u8,
    /// Vertical sampling factor `Vi ∈ 1..=4`.
    pub v_sampling: u8,
    /// Quantization-table destination selector `Tqi ∈ 0..=3`. (For
    /// SOF3 lossless, T.81 says this is always 0; the inspector
    /// reports the literal byte regardless.)
    pub quant_table: u8,
}

/// Result of a successful `inspect_jpeg` call.
#[derive(Clone, Debug)]
pub struct JpegInfo {
    /// Which SOFn marker opened the frame.
    pub sof_kind: SofKind,
    /// Sample precision `P` from the SOF segment (T.81 Table B.2):
    /// 8 or 12 for SOF0/SOF1/SOF2/SOF9, 2..=16 for SOF3, 8 or 12 for
    /// the other DCT variants. The inspector does not validate that
    /// `P` is in the legal set for `sof_kind`; it reports the byte.
    pub precision: u8,
    /// Image width `X` (samples per line). T.81 allows `X=0` when the
    /// real width is signalled later by a DNL segment; the inspector
    /// reports the SOF byte without applying DNL.
    pub width: u16,
    /// Image height `Y` (lines).
    pub height: u16,
    /// Sampling / quantization-table descriptor per component.
    pub components: Vec<InspectedComponent>,
    /// Chroma-subsampling discriminator inferred from the sampling
    /// factors. Single-component frames report `GrayscaleOnly`;
    /// four-component CMYK / YCCK frames report `Custom` (the
    /// inspector deliberately does not try to name the K-channel
    /// arrangement — read `color_hint` for that).
    pub subsampling: ChromaSubsampling,
    /// Colour-space hint from APP0 JFIF / APP14 Adobe, or
    /// `Unspecified` if neither was present in the marker prefix.
    pub color_hint: ColorHint,
    /// Restart interval in MCUs from the most recent DRI segment
    /// before SOS (T.81 §B.2.4.4), or `0` if no DRI was seen —
    /// matching the spec's "DRI absent ⇒ no restarts" default.
    pub restart_interval: u16,
    /// Typed view of the JFIF APP0 marker (T.871 §10.1) when one is
    /// present in the prefix. `None` if no APP0 carried the literal
    /// `"JFIF\0"` identifier (the older `JFXX` extension marker, the
    /// Adobe APP14 tag, and every other APPn segment are silent here).
    pub jfif: Option<JfifApp0>,
}

impl JpegInfo {
    /// Total component count from the SOF. Convenience accessor; the
    /// same number is `components.len()`.
    pub fn num_components(&self) -> usize {
        self.components.len()
    }
}

/// JFIF identifier from the start of an APP0 payload, T.871 §6.1.
/// Five bytes: `"JFIF\0"`.
const JFIF_MAGIC: &[u8; 5] = b"JFIF\0";

/// Adobe identifier from the start of an APP14 payload, T.872 §6.5.3.
/// Five bytes: `"Adobe"`. The 6th byte and onwards carry version /
/// flags / colour-transform.
const ADOBE_MAGIC: &[u8; 5] = b"Adobe";

/// Walk a JPEG byte buffer's marker prefix and report the variant +
/// descriptive metadata, without entropy decoding.
///
/// Errors:
/// * `Invalid` if the buffer does not start with SOI (T.81 §B.1.1.3).
/// * `Invalid` if a marker segment's length field is malformed
///   (delegated to the existing `MarkerWalker`).
/// * `Invalid` if SOS appears before SOF (a stream with no frame
///   header is unparseable).
/// * `Invalid` if EOF is reached before SOS (no scan would be
///   readable even by a real decoder).
/// * `Invalid` for an SOF marker byte that doesn't map to a `SofKind`
///   (DHT 0xC4 / JPG 0xC8 are excluded from the SOF range by the
///   walker's `is_sof` filter, so this branch is unreachable in
///   practice but kept defensive).
pub fn inspect_jpeg(buf: &[u8]) -> Result<JpegInfo> {
    // SOI must be the first two bytes. `MarkerWalker::next_marker`
    // would happily skip 0xFF fill bytes ahead of SOI, but a JPEG
    // file that does not literally begin `FF D8` is malformed per
    // T.81 §B.1.1.3 — refuse it here so the inspector's return
    // matches what container-level probes report.
    if buf.len() < 2 || buf[0] != 0xFF || buf[1] != markers::SOI {
        return Err(Error::invalid("inspect: missing SOI"));
    }

    let mut walker = MarkerWalker::new(buf);
    walker.pos = 2;

    let mut sof_kind: Option<SofKind> = None;
    let mut precision: u8 = 0;
    let mut width: u16 = 0;
    let mut height: u16 = 0;
    let mut components: Vec<InspectedComponent> = Vec::new();
    let mut color_hint = ColorHint::Unspecified;
    let mut restart_interval: u16 = 0;
    let mut jfif: Option<JfifApp0> = None;

    loop {
        let Some(marker) = walker.next_marker()? else {
            return Err(Error::invalid("inspect: EOF before SOS"));
        };

        if marker == markers::SOI || markers::is_rst(marker) || marker == markers::EOI {
            // SOI inside the prefix is malformed; RST inside the
            // prefix (before any SOS) has no payload but is also out
            // of place; EOI before SOS means the file has no frame.
            if marker == markers::EOI {
                return Err(Error::invalid("inspect: EOI before SOS"));
            }
            // SOI / RST: no length, no payload. Keep scanning so the
            // walker can recover on benign duplicates (we still
            // refuse SOS-before-SOF below).
            continue;
        }

        if marker == markers::SOS {
            // Reached the scan header. Stop. Validation that an SOF
            // was seen happens after the loop.
            break;
        }

        // Every remaining marker (SOFn / DHT / DQT / DAC / DRI / DNL
        // / APPn / COM / …) is length-prefixed per T.81 §B.1.1.4.
        let payload = walker.read_segment_payload()?;

        if markers::is_sof(marker) {
            // Only the first SOF wins; a hierarchical frame set
            // (SOF5/SOF7 + per-differential SOFs) has many but the
            // inspector reports the first.
            if sof_kind.is_none() {
                let kind = SofKind::from_marker(marker)
                    .ok_or_else(|| Error::invalid("inspect: SOF marker not classifiable"))?;
                let sof = parse_sof(payload)?;
                sof_kind = Some(kind);
                precision = sof.precision;
                width = sof.width;
                height = sof.height;
                components = sof
                    .components
                    .iter()
                    .map(|c| InspectedComponent {
                        id: c.id,
                        h_sampling: c.h_factor,
                        v_sampling: c.v_factor,
                        quant_table: c.qt_id,
                    })
                    .collect();
            }
            continue;
        }

        if marker == markers::DRI {
            restart_interval = parse_dri(payload)?;
            continue;
        }

        if markers::is_app(marker) {
            // APP0 = JFIF, APP14 = Adobe. The two carry colour-space
            // hints; other APPn segments (APP1 EXIF / XMP, APP2 ICC,
            // APP13 IPTC) are silent on colour transform and skipped.
            if marker == markers::APP0 && payload.len() >= 5 && &payload[..5] == JFIF_MAGIC {
                // Don't overwrite an earlier Adobe tag — both should
                // not appear together but if they do, the Adobe tag
                // is the more specific signal.
                if color_hint == ColorHint::Unspecified {
                    color_hint = ColorHint::JfifYCbCr;
                }
                // Surface the typed APP0 view. Only the first
                // JFIF APP0 wins; T.871 §6.3 requires the JFIF
                // marker to immediately follow SOI and §6.4 limits
                // subsequent APP0 segments to the `JFXX` extension
                // identifier, so a second JFIF APP0 in the prefix
                // would be malformed. Parse failure (e.g. unknown
                // units byte) is suppressed here so the inspector
                // still returns the other fields the caller asked
                // for; callers that need a strict JFIF check call
                // `parse_jfif_app0` directly.
                if jfif.is_none() {
                    if let Ok(parsed) = parse_jfif_app0(payload) {
                        jfif = Some(parsed);
                    }
                }
                continue;
            }
            if marker == markers::APP14 && payload.len() >= 12 && &payload[..5] == ADOBE_MAGIC {
                // APP14 layout: 5 bytes magic, 2 bytes version,
                // 2 bytes flags0, 2 bytes flags1, 1 byte transform.
                let transform = payload[11];
                color_hint = match transform {
                    0 => ColorHint::AdobeUntransformed,
                    1 => ColorHint::AdobeYCbCr,
                    2 => ColorHint::AdobeYcck,
                    // Other values are reserved / unknown — fall back
                    // to "untransformed" the way a typical decoder
                    // does, but only if we hadn't seen JFIF already.
                    _ => {
                        if color_hint == ColorHint::Unspecified {
                            ColorHint::AdobeUntransformed
                        } else {
                            color_hint
                        }
                    }
                };
                continue;
            }
            // Other APPn segments: ignored. Their payloads do not
            // affect any field the inspector reports.
            continue;
        }

        // DHT / DQT / DAC / DNL / COM / reserved: skipped. The
        // inspector does not care about Huffman / quant / arithmetic
        // table contents; it only needs the SOF + DRI + APP hints to
        // produce its summary.
    }

    let sof_kind = sof_kind.ok_or_else(|| Error::invalid("inspect: SOS before SOF"))?;

    let subsampling = classify_subsampling(&components);

    Ok(JpegInfo {
        sof_kind,
        precision,
        width,
        height,
        components,
        subsampling,
        color_hint,
        restart_interval,
        jfif,
    })
}

/// Derive a `ChromaSubsampling` discriminator from the SOF sampling
/// factors. See the enum's docstring for the mapping rules.
fn classify_subsampling(comps: &[InspectedComponent]) -> ChromaSubsampling {
    match comps.len() {
        1 => ChromaSubsampling::GrayscaleOnly,
        3 => {
            let y = comps[0];
            let cb = comps[1];
            let cr = comps[2];
            // Asymmetric chroma → Custom.
            if cb.h_sampling != cr.h_sampling || cb.v_sampling != cr.v_sampling {
                return ChromaSubsampling::Custom;
            }
            // Luma must be the dominant sampler for the four
            // conventional names; if a chroma sampler is wider than
            // luma the stream is unconventional.
            if y.h_sampling < cb.h_sampling || y.v_sampling < cb.v_sampling {
                return ChromaSubsampling::Custom;
            }
            // Zero samplers are invalid per T.81 but the inspector
            // doesn't reject — it just falls into Custom.
            if cb.h_sampling == 0 || cb.v_sampling == 0 {
                return ChromaSubsampling::Custom;
            }
            let hr = y.h_sampling / cb.h_sampling;
            let vr = y.v_sampling / cb.v_sampling;
            // Require exact division — non-integer ratios are unconventional.
            if y.h_sampling % cb.h_sampling != 0 || y.v_sampling % cb.v_sampling != 0 {
                return ChromaSubsampling::Custom;
            }
            match (hr, vr) {
                (1, 1) => ChromaSubsampling::Yuv444,
                (2, 1) => ChromaSubsampling::Yuv422,
                (2, 2) => ChromaSubsampling::Yuv420,
                (4, 1) => ChromaSubsampling::Yuv411,
                _ => ChromaSubsampling::Custom,
            }
        }
        _ => ChromaSubsampling::Custom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal JPEG marker prefix that begins with SOI and
    /// ends with the literal SOS marker (no scan body), wrapping a
    /// single SOF segment of the given marker / precision / dims /
    /// components. Used by the unit tests below.
    fn build_prefix(
        sof_marker: u8,
        precision: u8,
        width: u16,
        height: u16,
        components: &[(u8, u8, u8, u8)],
        extras: &[(u8, &[u8])],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        // SOI.
        out.extend_from_slice(&[0xFF, markers::SOI]);
        // Extras (APP / DRI / DQT / …) inserted between SOI and SOF.
        for (marker, payload) in extras {
            out.push(0xFF);
            out.push(*marker);
            let len = (payload.len() + 2) as u16;
            out.extend_from_slice(&len.to_be_bytes());
            out.extend_from_slice(payload);
        }
        // SOF: payload = P | Y(2) | X(2) | Nf | (Ci,Hi/Vi,Tqi)*Nf
        let nf = components.len() as u8;
        let payload_len = 6 + (nf as usize) * 3;
        out.push(0xFF);
        out.push(sof_marker);
        let seg_len = (payload_len + 2) as u16;
        out.extend_from_slice(&seg_len.to_be_bytes());
        out.push(precision);
        out.extend_from_slice(&height.to_be_bytes());
        out.extend_from_slice(&width.to_be_bytes());
        out.push(nf);
        for (id, h, v, tq) in components {
            out.push(*id);
            out.push((h << 4) | (v & 0x0F));
            out.push(*tq);
        }
        // SOS marker — the inspector stops here without reading its
        // payload, so we only need the marker byte. Append a zero-
        // length placeholder so MarkerWalker wouldn't get confused
        // if the inspector ever reads past `next_marker`.
        out.push(0xFF);
        out.push(markers::SOS);
        out
    }

    #[test]
    fn rejects_missing_soi() {
        let data = [0x00, 0x01, 0x02, 0x03];
        assert!(inspect_jpeg(&data).is_err());
    }

    #[test]
    fn rejects_empty_buffer() {
        assert!(inspect_jpeg(&[]).is_err());
    }

    #[test]
    fn rejects_eof_before_sos() {
        // SOI followed by an SOF with a length saying "more
        // payload" but no SOS afterwards.
        let buf = [0xFF, markers::SOI];
        assert!(inspect_jpeg(&buf).is_err());
    }

    #[test]
    fn rejects_eoi_before_sof() {
        let buf = [0xFF, markers::SOI, 0xFF, markers::EOI];
        assert!(inspect_jpeg(&buf).is_err());
    }

    #[test]
    fn baseline_yuv420_jfif() {
        let extras = [(
            markers::APP0,
            &b"JFIF\0\x01\x02\x00\x00\x01\x00\x01\x00\x00"[..],
        )];
        let buf = build_prefix(
            0xC0,
            8,
            640,
            480,
            &[(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect baseline 4:2:0");
        assert_eq!(info.sof_kind, SofKind::Baseline);
        assert!(info.sof_kind.is_supported_by_decoder());
        assert!(info.sof_kind.is_dct());
        assert!(!info.sof_kind.is_arithmetic());
        assert_eq!(info.precision, 8);
        assert_eq!(info.width, 640);
        assert_eq!(info.height, 480);
        assert_eq!(info.num_components(), 3);
        assert_eq!(info.subsampling, ChromaSubsampling::Yuv420);
        assert_eq!(info.color_hint, ColorHint::JfifYCbCr);
        assert_eq!(info.restart_interval, 0);
        // The same APP0 supplies the typed JFIF surface: v1.02,
        // units=0 (aspect-ratio), 1:1 density, no thumbnail.
        let jfif = info.jfif.expect("APP0 carried JFIF magic → typed view");
        assert_eq!(jfif.version_major, 1);
        assert_eq!(jfif.version_minor, 2);
        assert_eq!(jfif.units, JfifUnits::AspectRatio);
        assert_eq!(jfif.h_density, 1);
        assert_eq!(jfif.v_density, 1);
        assert!(!jfif.has_thumbnail());
    }

    #[test]
    fn jfif_app0_dpi_surface() {
        // v1.02 @ 96 dpi, no thumbnail — the typed JFIF view should
        // report DotsPerInch.
        let mut app0 = Vec::new();
        app0.extend_from_slice(b"JFIF\0");
        app0.push(1);
        app0.push(2);
        app0.push(1); // units = dpi
        app0.extend_from_slice(&96u16.to_be_bytes());
        app0.extend_from_slice(&96u16.to_be_bytes());
        app0.push(0);
        app0.push(0);
        let extras = [(markers::APP0, app0.as_slice())];
        let buf = build_prefix(
            0xC0,
            8,
            32,
            32,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect dpi");
        let jfif = info.jfif.expect("JFIF v1.02 dpi");
        assert_eq!(jfif.units, JfifUnits::DotsPerInch);
        assert_eq!(jfif.h_density, 96);
        assert_eq!(jfif.v_density, 96);
    }

    #[test]
    fn jfif_app0_absent_when_no_jfif_magic() {
        // APP0 with a non-JFIF identifier (e.g. AVI1) — the typed
        // surface must stay `None`.
        let app0 = b"AVI1\0\0\0\0\0\0\0\0\0\0";
        let extras = [(markers::APP0, &app0[..])];
        let buf = build_prefix(
            0xC0,
            8,
            32,
            32,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect AVI1");
        assert!(info.jfif.is_none());
        assert_eq!(info.color_hint, ColorHint::Unspecified);
    }

    #[test]
    fn jfif_app0_malformed_units_silently_drops_typed_view() {
        // APP0 carries the JFIF magic but the units byte is 3 (not
        // in {0,1,2}). The typed parser refuses; the inspector
        // suppresses the failure to keep producing the SOF / DRI
        // summary it owes the caller. ColorHint still reports JFIF
        // because the magic was present.
        let mut app0 = Vec::new();
        app0.extend_from_slice(b"JFIF\0");
        app0.push(1);
        app0.push(2);
        app0.push(3); // invalid units
        app0.extend_from_slice(&72u16.to_be_bytes());
        app0.extend_from_slice(&72u16.to_be_bytes());
        app0.push(0);
        app0.push(0);
        let extras = [(markers::APP0, app0.as_slice())];
        let buf = build_prefix(
            0xC0,
            8,
            32,
            32,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect malformed-units JFIF");
        assert_eq!(info.color_hint, ColorHint::JfifYCbCr);
        assert!(info.jfif.is_none());
    }

    #[test]
    fn baseline_yuv422() {
        let buf = build_prefix(
            0xC0,
            8,
            320,
            240,
            &[(1, 2, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &[],
        );
        let info = inspect_jpeg(&buf).expect("inspect baseline 4:2:2");
        assert_eq!(info.subsampling, ChromaSubsampling::Yuv422);
        assert_eq!(info.color_hint, ColorHint::Unspecified);
    }

    #[test]
    fn baseline_yuv444() {
        let buf = build_prefix(
            0xC0,
            8,
            8,
            8,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &[],
        );
        let info = inspect_jpeg(&buf).expect("inspect baseline 4:4:4");
        assert_eq!(info.subsampling, ChromaSubsampling::Yuv444);
    }

    #[test]
    fn baseline_yuv411() {
        let buf = build_prefix(
            0xC0,
            8,
            64,
            64,
            &[(1, 4, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &[],
        );
        let info = inspect_jpeg(&buf).expect("inspect baseline 4:1:1");
        assert_eq!(info.subsampling, ChromaSubsampling::Yuv411);
    }

    #[test]
    fn baseline_grayscale() {
        let buf = build_prefix(0xC0, 8, 16, 16, &[(1, 1, 1, 0)], &[]);
        let info = inspect_jpeg(&buf).expect("inspect baseline grayscale");
        assert_eq!(info.num_components(), 1);
        assert_eq!(info.subsampling, ChromaSubsampling::GrayscaleOnly);
    }

    #[test]
    fn baseline_cmyk_is_custom_subsampling() {
        let buf = build_prefix(
            0xC0,
            8,
            32,
            32,
            &[(1, 1, 1, 0), (2, 1, 1, 0), (3, 1, 1, 0), (4, 1, 1, 0)],
            &[],
        );
        let info = inspect_jpeg(&buf).expect("inspect 4-comp");
        assert_eq!(info.num_components(), 4);
        assert_eq!(info.subsampling, ChromaSubsampling::Custom);
    }

    #[test]
    fn asymmetric_chroma_is_custom() {
        let buf = build_prefix(
            0xC0,
            8,
            32,
            32,
            // Cb = 2x1, Cr = 1x1 — not a conventional ratio name.
            &[(1, 2, 2, 0), (2, 2, 1, 1), (3, 1, 1, 1)],
            &[],
        );
        let info = inspect_jpeg(&buf).expect("inspect asymmetric");
        assert_eq!(info.subsampling, ChromaSubsampling::Custom);
    }

    #[test]
    fn progressive_kind() {
        let buf = build_prefix(
            0xC2,
            8,
            16,
            16,
            &[(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &[],
        );
        let info = inspect_jpeg(&buf).expect("inspect progressive");
        assert_eq!(info.sof_kind, SofKind::Progressive);
        assert!(info.sof_kind.is_supported_by_decoder());
        assert!(info.sof_kind.is_dct());
        assert!(!info.sof_kind.is_arithmetic());
    }

    #[test]
    fn lossless_kind() {
        let buf = build_prefix(0xC3, 12, 100, 100, &[(1, 1, 1, 0)], &[]);
        let info = inspect_jpeg(&buf).expect("inspect lossless");
        assert_eq!(info.sof_kind, SofKind::Lossless);
        assert!(info.sof_kind.is_supported_by_decoder());
        assert!(!info.sof_kind.is_dct());
        assert!(!info.sof_kind.is_arithmetic());
        assert_eq!(info.precision, 12);
    }

    #[test]
    fn arith_kind() {
        let buf = build_prefix(
            0xC9,
            8,
            16,
            16,
            &[(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &[],
        );
        let info = inspect_jpeg(&buf).expect("inspect SOF9");
        assert_eq!(info.sof_kind, SofKind::ExtendedSequentialArith);
        assert!(info.sof_kind.is_supported_by_decoder());
        assert!(info.sof_kind.is_dct());
        assert!(info.sof_kind.is_arithmetic());
    }

    #[test]
    fn hierarchical_dct_kind_not_supported() {
        let buf = build_prefix(0xC5, 8, 16, 16, &[(1, 1, 1, 0)], &[]);
        let info = inspect_jpeg(&buf).expect("inspect SOF5");
        assert_eq!(info.sof_kind, SofKind::HierarchicalDct);
        assert!(!info.sof_kind.is_supported_by_decoder());
    }

    #[test]
    fn progressive_arith_kind_not_supported() {
        let buf = build_prefix(0xCA, 8, 16, 16, &[(1, 2, 2, 0)], &[]);
        let info = inspect_jpeg(&buf).expect("inspect SOF10");
        assert_eq!(info.sof_kind, SofKind::ProgressiveArith);
        assert!(!info.sof_kind.is_supported_by_decoder());
        assert!(info.sof_kind.is_dct());
        assert!(info.sof_kind.is_arithmetic());
    }

    #[test]
    fn dri_before_sof_reported() {
        let dri_payload = [0x00, 0x10]; // restart interval = 16 MCUs
        let extras = [(markers::DRI, &dri_payload[..])];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect DRI");
        assert_eq!(info.restart_interval, 16);
    }

    #[test]
    fn dri_after_sof_also_reported() {
        // Place the DRI between SOF and SOS by appending it as an
        // "extra" — `build_prefix` puts extras before SOF, so we
        // manually craft this one.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0xFF, markers::SOI]);
        // SOF.
        buf.extend_from_slice(&[0xFF, 0xC0]);
        // length = 11 (P=8, X/Y, Nf=1, one component triple)
        buf.extend_from_slice(&11u16.to_be_bytes());
        buf.push(8);
        buf.extend_from_slice(&8u16.to_be_bytes()); // height
        buf.extend_from_slice(&8u16.to_be_bytes()); // width
        buf.push(1);
        buf.extend_from_slice(&[1, 0x11, 0]);
        // DRI = 32.
        buf.extend_from_slice(&[0xFF, markers::DRI, 0x00, 0x04, 0x00, 0x20]);
        // SOS.
        buf.extend_from_slice(&[0xFF, markers::SOS]);
        let info = inspect_jpeg(&buf).expect("inspect DRI-after-SOF");
        assert_eq!(info.restart_interval, 32);
    }

    #[test]
    fn adobe_app14_yccc_color_hint() {
        // 14 bytes: "Adobe" + version (2) + flags0 (2) + flags1 (2)
        // + transform (1) = 12 bytes.
        let mut adobe = Vec::new();
        adobe.extend_from_slice(b"Adobe");
        adobe.extend_from_slice(&[0x00, 0x65]); // version
        adobe.extend_from_slice(&[0x00, 0x00]); // flags0
        adobe.extend_from_slice(&[0x00, 0x00]); // flags1
        adobe.push(2); // transform = YCCK
        let extras = [(markers::APP14, adobe.as_slice())];
        let buf = build_prefix(
            0xC0,
            8,
            32,
            32,
            &[(1, 1, 1, 0), (2, 1, 1, 0), (3, 1, 1, 0), (4, 1, 1, 0)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect Adobe YCCK");
        assert_eq!(info.color_hint, ColorHint::AdobeYcck);
        assert_eq!(info.num_components(), 4);
    }

    #[test]
    fn adobe_app14_untransformed_color_hint() {
        let mut adobe = Vec::new();
        adobe.extend_from_slice(b"Adobe");
        adobe.extend_from_slice(&[0x00, 0x65]);
        adobe.extend_from_slice(&[0x00, 0x00]);
        adobe.extend_from_slice(&[0x00, 0x00]);
        adobe.push(0);
        let extras = [(markers::APP14, adobe.as_slice())];
        let buf = build_prefix(
            0xC0,
            8,
            32,
            32,
            &[(1, 1, 1, 0), (2, 1, 1, 0), (3, 1, 1, 0)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect Adobe untransformed");
        assert_eq!(info.color_hint, ColorHint::AdobeUntransformed);
    }

    #[test]
    fn unknown_app_segment_skipped_no_color_hint() {
        // APP1 (EXIF/XMP) ahead of SOF — must not affect color hint.
        let exif = b"Exif\0\0junk";
        let extras = [(0xE1u8, &exif[..])];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect EXIF prefix");
        assert_eq!(info.color_hint, ColorHint::Unspecified);
    }

    #[test]
    fn malformed_sof_returns_err() {
        // SOF length says 4 bytes of payload but the SOF parser
        // expects at least 6 — the inner parse_sof returns Err.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0xFF, markers::SOI]);
        buf.extend_from_slice(&[0xFF, 0xC0]);
        // payload length = 4 (segment length = 6 including the two
        // length bytes themselves).
        buf.extend_from_slice(&6u16.to_be_bytes());
        buf.extend_from_slice(&[0, 0, 0, 0]);
        buf.extend_from_slice(&[0xFF, markers::SOS]);
        assert!(inspect_jpeg(&buf).is_err());
    }

    #[test]
    fn second_sof_does_not_overwrite() {
        // The inspector reports only the first SOF. Construct a
        // baseline SOF followed by a progressive SOF before SOS;
        // the first must win.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0xFF, markers::SOI]);
        // SOF0 (Baseline)
        buf.extend_from_slice(&[0xFF, 0xC0]);
        buf.extend_from_slice(&11u16.to_be_bytes());
        buf.push(8);
        buf.extend_from_slice(&8u16.to_be_bytes());
        buf.extend_from_slice(&8u16.to_be_bytes());
        buf.push(1);
        buf.extend_from_slice(&[1, 0x11, 0]);
        // SOF2 (Progressive) with a larger geometry.
        buf.extend_from_slice(&[0xFF, 0xC2]);
        buf.extend_from_slice(&11u16.to_be_bytes());
        buf.push(8);
        buf.extend_from_slice(&64u16.to_be_bytes());
        buf.extend_from_slice(&64u16.to_be_bytes());
        buf.push(1);
        buf.extend_from_slice(&[1, 0x11, 0]);
        // SOS.
        buf.extend_from_slice(&[0xFF, markers::SOS]);
        let info = inspect_jpeg(&buf).expect("inspect dup SOF");
        assert_eq!(info.sof_kind, SofKind::Baseline);
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);
    }
}
