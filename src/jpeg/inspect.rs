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
use super::parser::{parse_dri, parse_sof, MarkerWalker};
use super::zigzag::ZIGZAG;

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
    /// SOF9 / SOF10 / SOF11). The "supported" line is data, not a
    /// promise; callers that want to negotiate fallback can read it.
    pub fn is_supported_by_decoder(self) -> bool {
        matches!(
            self,
            Self::Baseline
                | Self::ExtendedSequential
                | Self::Progressive
                | Self::Lossless
                | Self::ExtendedSequentialArith
                | Self::ProgressiveArith
                | Self::LosslessArith
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

/// Units field of the JFIF APP0 segment (T.871 §10.1).
///
/// JFIF's `units` byte selects what the `Hdensity` / `Vdensity`
/// numbers mean. The three encodings are exhaustive — every other
/// value is illegal and produces an `Err` from `parse_jfif_app0`,
/// matching the spec's "shall be one of" wording.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JfifUnits {
    /// `units = 0x00` — densities express only the pixel aspect
    /// ratio (`width:height = Vdensity:Hdensity`). The numerical
    /// values are in arbitrary units.
    AspectRatio,
    /// `units = 0x01` — densities are dots per inch (1 inch = 2.54
    /// cm).
    DotsPerInch,
    /// `units = 0x02` — densities are dots per cm.
    DotsPerCm,
}

impl JfifUnits {
    /// The literal `units` byte (T.871 §10.1) this variant represents.
    /// Provided so callers building a JFIF APP0 segment from a typed
    /// view can re-encode the byte they parsed without a side table.
    pub fn as_byte(self) -> u8 {
        match self {
            Self::AspectRatio => 0x00,
            Self::DotsPerInch => 0x01,
            Self::DotsPerCm => 0x02,
        }
    }
}

/// Typed view of a JFIF APP0 marker segment (T.871 §10.1).
///
/// Parsed from the bytes that follow the APP0 marker's two-byte
/// length field, **including** the `"JFIF\0"` identifier — i.e. the
/// `payload` slice the inspector's marker walker hands to
/// `parse_jfif_app0`. Decode-only: the inspector never builds one
/// from caller input.
///
/// All numeric fields are reported as they appeared on the wire; the
/// only validation is the structural one the spec mandates (identifier
/// equals `"JFIF\0"`, `units ∈ {0, 1, 2}`, both densities non-zero,
/// `Lp` accounts for the optional thumbnail RGB triples). The
/// thumbnail RGB payload itself is **not** copied out — callers that
/// want it can read it from the source buffer using the offsets
/// computed from `thumbnail_width × thumbnail_height × 3`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JfifApp0 {
    /// Major version byte (T.871 §10.1 `version` field, MSB). The
    /// recommendation requires `0x01`; observed JFIF files in the
    /// wild always carry `0x01` here and a parser that rejects
    /// otherwise would lose interoperability with the long tail of
    /// pre-T.871 1.0x writers — so the field is reported, not
    /// validated.
    pub version_major: u8,
    /// Minor version byte (T.871 §10.1 `version` field, LSB).
    /// Common values are `0x00` (JFIF 1.00), `0x01` (1.01), and
    /// `0x02` (1.02 — the version T.871 normalises).
    pub version_minor: u8,
    /// Units selector (T.871 §10.1 `units` field).
    pub units: JfifUnits,
    /// Horizontal pixel density. T.871 §10.1 says this "must be
    /// non-zero"; `parse_jfif_app0` enforces it.
    pub h_density: u16,
    /// Vertical pixel density. Must also be non-zero per T.871
    /// §10.1.
    pub v_density: u16,
    /// Horizontal thumbnail pixel count (`HthumbnailA`, T.871
    /// §10.1). May be zero — `0` together with `thumbnail_height = 0`
    /// signals "no thumbnail" and the trailing `(R, G, B) * k`
    /// payload is empty.
    pub thumbnail_width: u8,
    /// Vertical thumbnail pixel count (`VthumbnailA`).
    pub thumbnail_height: u8,
}

impl JfifApp0 {
    /// True when both thumbnail-pixel-count fields are zero, i.e. the
    /// segment carries no embedded RGB thumbnail (the common case for
    /// real-world JFIF files — most writers emit the extension APP0
    /// `"JFXX"` thumbnail instead).
    pub fn has_thumbnail(self) -> bool {
        self.thumbnail_width != 0 && self.thumbnail_height != 0
    }

    /// Total bytes of trailing RGB-thumbnail payload that follow the
    /// fixed header (T.871 §10.1: `3 * HthumbnailA * VthumbnailA`). Zero
    /// when no thumbnail is present.
    pub fn thumbnail_payload_len(self) -> usize {
        (self.thumbnail_width as usize) * (self.thumbnail_height as usize) * 3
    }

    /// `(version_major, version_minor)` as a `(u8, u8)` tuple, in case
    /// the caller wants to pattern-match against `(1, 2)` etc. without
    /// reading both fields.
    pub fn version(self) -> (u8, u8) {
        (self.version_major, self.version_minor)
    }

    /// Horizontal density in dots-per-inch, if the segment's `units`
    /// permits the conversion. Returns the raw `h_density` for
    /// `DotsPerInch`, the converted value for `DotsPerCm` (`× 2.54`,
    /// rounded), and `None` for `AspectRatio` since aspect-only
    /// density numbers have no DPI meaning.
    pub fn h_density_dpi(self) -> Option<u32> {
        match self.units {
            JfifUnits::AspectRatio => None,
            JfifUnits::DotsPerInch => Some(self.h_density as u32),
            // 1 inch = 2.54 cm → dpi = dpcm × 2.54. Compute with
            // integer arithmetic, rounding half to even via the
            // standard `(a * 254 + 50) / 100` trick.
            JfifUnits::DotsPerCm => Some(((self.h_density as u32).saturating_mul(254) + 50) / 100),
        }
    }

    /// Vertical density in dots-per-inch — see `h_density_dpi`.
    pub fn v_density_dpi(self) -> Option<u32> {
        match self.units {
            JfifUnits::AspectRatio => None,
            JfifUnits::DotsPerInch => Some(self.v_density as u32),
            JfifUnits::DotsPerCm => Some(((self.v_density as u32).saturating_mul(254) + 50) / 100),
        }
    }

    /// Width:height pixel aspect ratio expressed as the raw
    /// `(v_density, h_density)` pair. T.871 §10.1 explicitly states
    /// "pixel aspect ratio = Vdensity:Hdensity"; the helper returns
    /// the same numbers in source order so a caller computing a
    /// floating-point ratio (`w as f32 / h as f32 * v as f32 /
    /// h_density as f32`) doesn't have to remember which way the
    /// spec writes it.
    pub fn pixel_aspect_ratio(self) -> (u16, u16) {
        (self.v_density, self.h_density)
    }
}

/// Thumbnail-storage variant of a JFIF extension APP0 segment
/// (T.871 §10.2 `extension_code`).
///
/// The `extension_code` byte that follows the `"JFXX\0"` identifier
/// selects one of three thumbnail storage formats. T.871 §10.2 defines
/// exactly these three codes (`0x10`, `0x11`, `0x13`); every other byte
/// is reserved and produces an `Err` from `parse_jfxx_app0`, matching
/// the spec's enumerated "the following extensions are defined" wording.
///
/// The thumbnail pixel bytes themselves are **not** copied into the
/// typed view — the inspector reports the geometry and the on-wire byte
/// budget so a caller can locate the payload in the source buffer
/// without the inspector allocating a second copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JfxxThumbnail {
    /// `extension_code = 0x10` — the thumbnail is a complete baseline
    /// JPEG (`SOI .. EOI`) embedded in `extension_data` (T.871 §10.3).
    /// `jpeg_len` is the number of `extension_data` bytes carrying that
    /// embedded stream (the whole payload tail after the 6-byte
    /// identifier + code header).
    JpegEncoded {
        /// Length in bytes of the embedded baseline-JPEG `extension_data`.
        jpeg_len: usize,
    },
    /// `extension_code = 0x11` — an uncompressed 8-bit-palette RGB
    /// thumbnail (T.871 §10.4): a 768-byte (256 × RGB) palette followed
    /// by `width × height` one-byte palette indices.
    PaletteRgb {
        /// `HthumbnailB` — horizontal thumbnail pixel count (non-zero
        /// per §10.4).
        width: u8,
        /// `VthumbnailB` — vertical thumbnail pixel count (non-zero).
        height: u8,
    },
    /// `extension_code = 0x13` — an uncompressed packed 24-bit RGB
    /// thumbnail (T.871 §10.5): `width × height × 3` interleaved
    /// `R, G, B` bytes. Same on-wire layout as the optional thumbnail
    /// carried directly in the JFIF APP0 (§10.1).
    Rgb24 {
        /// `HthumbnailC` — horizontal thumbnail pixel count (non-zero
        /// per §10.5).
        width: u8,
        /// `VthumbnailC` — vertical thumbnail pixel count (non-zero).
        height: u8,
    },
}

impl JfxxThumbnail {
    /// The literal `extension_code` byte (T.871 §10.2) this variant
    /// represents. Provided so a caller re-serialising a JFXX segment
    /// from a typed view can recover the code without a side table.
    pub fn extension_code(self) -> u8 {
        match self {
            Self::JpegEncoded { .. } => 0x10,
            Self::PaletteRgb { .. } => 0x11,
            Self::Rgb24 { .. } => 0x13,
        }
    }
}

/// Typed view of a JFIF extension APP0 marker segment (T.871 §10.2).
///
/// One or more JFIF extension APP0 segments may immediately follow the
/// JFIF APP0 segment; each carries a thumbnail in one of the three
/// formats T.871 §10.3-10.5 defines. The segment is identified by the
/// zero-terminated `"JFXX\0"` string in its first five payload bytes,
/// followed by a one-byte `extension_code`.
///
/// Parsed from the bytes that follow the APP0 marker's two-byte length
/// field, **including** the `"JFXX\0"` identifier — i.e. the `payload`
/// slice the inspector's marker walker hands to `parse_jfxx_app0`.
/// Decode-only: the inspector never builds one from caller input.
///
/// The thumbnail pixel / JPEG bytes are not copied out; the typed view
/// reports the storage variant and its geometry only. Callers that want
/// the thumbnail itself can re-walk the source buffer with the offsets
/// the variant implies (`palette + indices` for §10.4, `3 × w × h` RGB
/// for §10.5, or the embedded baseline stream for §10.3).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JfxxApp0 {
    /// The thumbnail-storage variant decoded from the `extension_code`
    /// byte plus the geometry / length fields that follow it.
    pub thumbnail: JfxxThumbnail,
}

/// Colour-transform byte from the Adobe APP14 segment (T.872 §6.5.3).
///
/// The same three values that `ColorHint`'s `Adobe*` variants surface,
/// but exposed as a self-contained typed enum so downstream code can
/// pattern-match on the transform without first having to disambiguate
/// JFIF-vs-Adobe at the colour-hint level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdobeColorTransform {
    /// `transform = 0x00` — components are not colour-transformed.
    /// For `Nf = 3` the samples are RGB; for `Nf = 4` they are CMYK.
    Unknown,
    /// `transform = 0x01` — `Nf = 3` components are YCbCr per the
    /// Adobe ColorTransform convention. Equivalent to JFIF's BT.601
    /// full-range tag for the YCbCr arrangement.
    YCbCr,
    /// `transform = 0x02` — `Nf = 4` components are YCCK (Y, Cb, Cr,
    /// K) — Adobe's CMYK colour-transform variant.
    Ycck,
}

impl AdobeColorTransform {
    /// The literal `transform` byte (T.872 §6.5.3) this variant
    /// represents. Provided so callers building an APP14 segment
    /// from a typed view can re-encode the byte they parsed without
    /// a side table.
    pub fn as_byte(self) -> u8 {
        match self {
            Self::Unknown => 0x00,
            Self::YCbCr => 0x01,
            Self::Ycck => 0x02,
        }
    }
}

/// Typed view of an Adobe APP14 marker segment (T.872 §6.5.3 /
/// Adobe Technical Note 5116 §18).
///
/// Parsed from the bytes that follow the APP14 marker's two-byte
/// length field, **including** the `"Adobe"` identifier — i.e. the
/// `payload` slice the inspector's marker walker hands to
/// `parse_adobe_app14`. Decode-only: the inspector never builds one
/// from caller input.
///
/// All numeric fields are reported as they appeared on the wire; the
/// only validation is the structural one the spec mandates (identifier
/// equals `"Adobe"`, payload is at least 12 bytes, `transform ∈
/// {0, 1, 2}`). Unknown transform bytes — Photoshop has emitted
/// reserved values historically — produce an `Err` so direct callers
/// can fall back to the `ColorHint`-level inference (which tolerates
/// the byte by treating reserved values as `AdobeUntransformed`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdobeApp14 {
    /// DCT encoding version (T.872 §6.5.3 `DCTEncodeVersion`).
    /// Common values: `100` (Adobe Technical Note 5116, the
    /// near-universal default) and `101` (a Photoshop revision).
    /// Reported as the raw big-endian `u16` without validation —
    /// the field is informational.
    pub dct_encode_version: u16,
    /// APP14 encoder hint flags 0 (T.872 §6.5.3 `APP14Flags0`).
    /// Bit 0x4000 indicates the encoder applied chroma blurring;
    /// bit 0x8000 indicates the encoder used the dampened-edge
    /// quantization. The inspector reports the raw word; callers
    /// that care can bit-test.
    pub flags_0: u16,
    /// APP14 encoder hint flags 1 (T.872 §6.5.3 `APP14Flags1`).
    /// Currently unused; reserved-zero in conformant writers.
    pub flags_1: u16,
    /// Colour-transform byte (T.872 §6.5.3 `ColorTransform`).
    pub transform: AdobeColorTransform,
}

impl AdobeApp14 {
    /// True when the segment declares the universally-used Adobe
    /// Technical Note 5116 DCT encoding version (`100`). Streams with
    /// other versions remain valid; this is a convenience predicate
    /// for diagnostics.
    pub fn is_standard_version(self) -> bool {
        self.dct_encode_version == 100
    }

    /// Equivalent `ColorHint` projection for callers that want to
    /// unify the two colour-convention conventions into the single
    /// inspector-level enum (which is what `inspect_jpeg`'s
    /// `color_hint` field is already exposing).
    pub fn as_color_hint(self) -> ColorHint {
        match self.transform {
            AdobeColorTransform::Unknown => ColorHint::AdobeUntransformed,
            AdobeColorTransform::YCbCr => ColorHint::AdobeYCbCr,
            AdobeColorTransform::Ycck => ColorHint::AdobeYcck,
        }
    }
}

/// Typed view of one ICC profile APP2 marker segment (T.872 / Annex L of
/// T.871; see `docs/image/jpeg/jpeg-fixtures-and-traces.md` §3.11).
///
/// ICC profiles are conventionally embedded in APP2 segments whose
/// payloads start with the 12-byte ASCII identifier `"ICC_PROFILE\0"`
/// followed by a one-byte sequence number `seq_no ∈ 1..=total`, a one-
/// byte total chunk count `total ∈ 1..=255`, and then the next slice of
/// the ICC profile bytes. Profiles longer than ~64 KB are split across
/// multiple consecutive APP2 segments; smaller profiles fit in one. The
/// JPEG decoder never parses the ICC content — it is passed through to
/// the application as an opaque byte run.
///
/// The typed view reports the segment-level chunk header (`seq_no`,
/// `total`, and the byte length of the profile slice this segment
/// carries) and a borrowed slice into the source payload that holds the
/// profile bytes themselves. The inspector's higher-level accumulator
/// (`JpegInfo::icc_profile`) joins consecutive segments into the
/// complete profile blob when one is present.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IccProfileApp2Chunk<'a> {
    /// Chunk-sequence number (`seq_no`) from the APP2 header. Spec
    /// convention numbers chunks `1..=total` (one-based); the inspector
    /// reports the literal byte without remapping.
    pub seq_no: u8,
    /// Total chunk count (`total`) declared by this segment's header.
    /// Every APP2 ICC segment in a well-formed stream agrees on the
    /// same `total`; the inspector reports the byte from this segment
    /// only.
    pub total: u8,
    /// Borrowed slice of the ICC profile bytes carried by this segment
    /// — the payload tail after the 12-byte signature and the two-byte
    /// chunk header. The inspector does not interpret the bytes; the
    /// slice's lifetime is the source buffer's.
    pub profile_bytes: &'a [u8],
}

/// ICC profile identifier from the start of an APP2 payload (T.872 /
/// Annex L of T.871). Twelve bytes: `"ICC_PROFILE\0"` (eleven ASCII
/// characters plus one NUL terminator).
const ICC_PROFILE_MAGIC: &[u8; 12] = b"ICC_PROFILE\0";

/// Parse an APP2 ICC_PROFILE payload (T.872 / Annex L of T.871) into a
/// typed chunk view borrowing from the input buffer.
///
/// `payload` is the byte slice that the marker walker hands to the
/// inspector for the APP2 segment — the bytes after the marker's
/// two-byte length field, starting with the `"ICC_PROFILE\0"`
/// identifier. The function returns `Ok(IccProfileApp2Chunk)` when the
/// segment is structurally valid:
///
/// * Identifier equals `b"ICC_PROFILE\0"`.
/// * Payload is at least 14 bytes long (12 identifier + 1 seq_no + 1
///   total).
/// * `total ≥ 1` and `1 ≤ seq_no ≤ total` (one-based, inclusive
///   bounds; a chunk with `seq_no = 0` or `seq_no > total` is malformed).
///
/// Errors:
/// * `Invalid` when the payload is shorter than the 14-byte fixed
///   header.
/// * `Invalid` when the identifier doesn't match `"ICC_PROFILE\0"`
///   (the caller is expected to gate on the magic first, but the
///   validator re-checks so direct calls aren't a footgun).
/// * `Invalid` when `total = 0` (a profile must have at least one
///   chunk).
/// * `Invalid` when `seq_no = 0` or `seq_no > total`.
///
/// The returned `profile_bytes` slice borrows from `payload`; the
/// function never copies the ICC body. The validator never allocates.
pub fn parse_icc_profile_app2(payload: &[u8]) -> Result<IccProfileApp2Chunk<'_>> {
    // §3.11 layout, byte offsets relative to the start of the APP2
    // payload (i.e. *after* the marker + length):
    //
    //   0..12   identifier "ICC_PROFILE\0"
    //  12       seq_no   (1..=total)
    //  13       total    (>=1)
    //  14..     ICC profile bytes for this chunk
    //
    // Hence 14 bytes is the absolute minimum for a (degenerate)
    // zero-body ICC chunk.
    if payload.len() < 14 {
        return Err(Error::invalid("parse_icc_profile_app2: payload too short"));
    }
    if &payload[..12] != ICC_PROFILE_MAGIC {
        return Err(Error::invalid(
            "parse_icc_profile_app2: identifier != ICC_PROFILE\\0",
        ));
    }
    let seq_no = payload[12];
    let total = payload[13];
    if total == 0 {
        return Err(Error::invalid("parse_icc_profile_app2: total = 0"));
    }
    if seq_no == 0 || seq_no > total {
        return Err(Error::invalid(
            "parse_icc_profile_app2: seq_no outside 1..=total",
        ));
    }
    Ok(IccProfileApp2Chunk {
        seq_no,
        total,
        profile_bytes: &payload[14..],
    })
}

/// Aggregated view of every APP2 `"ICC_PROFILE\0"` segment seen in the
/// marker prefix, in order of appearance.
///
/// The chunks vector preserves the on-wire order — the typed view does
/// not sort by `seq_no` because well-formed writers already emit them in
/// order, and reporting the source order is more useful for diagnostics
/// (a stream with a re-ordered or repeated chunk reveals the issue
/// directly). Each entry carries its segment's `seq_no` / `total` so a
/// caller can confirm contiguity itself.
///
/// The `total` advertised by every entry must agree — `inspect_jpeg`
/// rejects a stream whose APP2 segments declare different totals.
/// Beyond that the inspector is permissive: missing or duplicate
/// chunks are reported via `is_complete()` rather than refused, since
/// the spec leaves application-level recovery to the caller and
/// surfacing a partial summary is more useful than a hard refusal.
#[derive(Clone, Debug)]
pub struct IccProfileChunks {
    /// Total chunk count declared by the segments (every chunk's
    /// `total` byte agrees).
    pub total: u8,
    /// Number of profile bytes summed across every collected chunk.
    pub total_payload_len: usize,
    /// Per-segment `(seq_no, payload_len)` pairs in source order.
    pub chunks: Vec<(u8, usize)>,
}

impl IccProfileChunks {
    /// True when every sequence number from `1..=total` appears
    /// exactly once across the collected chunks. A `false` return
    /// signals a missing / duplicate / re-ordered chunk; the typed
    /// view itself remains usable but the assembled profile would be
    /// suspect.
    pub fn is_complete(&self) -> bool {
        if self.total == 0 {
            return false;
        }
        if self.chunks.len() != self.total as usize {
            return false;
        }
        let mut seen = [false; 256];
        for (seq, _) in &self.chunks {
            let idx = *seq as usize;
            if idx == 0 || idx > self.total as usize || seen[idx] {
                return false;
            }
            seen[idx] = true;
        }
        true
    }
}

/// Typed view of one COM (comment) marker segment (T.81 §B.2.4.5,
/// Figure B.10 / Table B.8).
///
/// A COM segment is `COM | Lc | Cm_1 .. Cm_{Lc-2}`: a two-byte
/// big-endian length `Lc ∈ [2, 65535]` followed by `Lc-2` comment
/// bytes whose interpretation T.81 leaves entirely to the application.
/// The inspector therefore makes no charset assumption — it copies the
/// raw bytes verbatim and exposes them alongside two convenience
/// projections (`as_str_lossy`, `is_ascii_text`) for the common case
/// of a human-readable ASCII / UTF-8 annotation. A zero-length comment
/// (`Lc = 2`, no comment bytes) is legal per the `2..=65535` range and
/// is reported with an empty `bytes` vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JpegComment {
    /// The `Lc-2` comment bytes exactly as they appeared on the wire,
    /// with no charset interpretation or trimming. Empty when the
    /// segment carried only its length field (`Lc = 2`).
    pub bytes: Vec<u8>,
}

impl JpegComment {
    /// Number of comment bytes (`Lc - 2`). A convenience accessor; the
    /// same value is `bytes.len()`.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// True when the segment carried no comment bytes (`Lc = 2`).
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Lossy UTF-8 projection of the comment bytes. Invalid sequences
    /// are replaced with U+FFFD. Most real-world COM segments are plain
    /// ASCII (encoder watermarks like `"Created with …"`), so this is
    /// the convenient accessor for diagnostics; callers that need the
    /// exact bytes read `bytes`.
    pub fn as_str_lossy(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.bytes)
    }

    /// True when every comment byte is a printable-or-whitespace ASCII
    /// character (`0x09`, `0x0A`, `0x0D`, or `0x20..=0x7E`). Lets a
    /// caller decide whether `as_str_lossy` is a faithful rendering
    /// without an embedded replacement character. An empty comment is
    /// reported as text (vacuously true).
    pub fn is_ascii_text(&self) -> bool {
        self.bytes
            .iter()
            .all(|&b| matches!(b, 0x09 | 0x0A | 0x0D | 0x20..=0x7E))
    }
}

/// Typed view of one quantization table defined by a DQT marker
/// segment (T.81 §B.2.4.1, Figure B.6 / Table B.4).
///
/// A DQT segment packs one or more tables; each table is a one-byte
/// `Pq | Tq` selector (the high nibble `Pq` is the element precision —
/// `0` ⇒ 8-bit, `1` ⇒ 16-bit; the low nibble `Tq` is the destination
/// identifier in `0..=3`) followed by 64 quantizer elements `Qk` in
/// the Figure A.6 zigzag order. The inspector unshuffles the elements
/// into natural (row-major) `[row*8 + col]` order on the way in — the
/// same convention the decoder's stored tables use — so two streams
/// that quote the same table at different precisions compare equal in
/// `values`.
///
/// This is pure prefix metadata: no entropy decode, no dequantization.
/// Useful for re-encode fidelity (replay the exact source tables),
/// corpus dedup / table-fingerprinting, and quality estimation against
/// the Annex K base tables.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantTableInfo {
    /// Destination identifier `Tq` (low nibble of the `Pq | Tq` byte),
    /// in `0..=3`. Components in the SOF reference a table by this id
    /// via `InspectedComponent::quant_table`.
    pub id: u8,
    /// Element precision flag `Pq` (high nibble): `false` ⇒ 8-bit
    /// elements (`Qk ∈ 0..=255`), `true` ⇒ 16-bit elements
    /// (`Qk ∈ 0..=65535`). 16-bit tables are only legal for non-baseline
    /// frames per T.81 §B.2.4.1, but the inspector reports the flag as
    /// coded without cross-checking it against `sof_kind`.
    pub sixteen_bit: bool,
    /// The 64 quantizer elements in natural (row-major) order. The DC
    /// quantizer is `values[0]`; element `[row*8 + col]` quantizes the
    /// `(row, col)` DCT coefficient. Already un-zigzagged from the wire
    /// order.
    pub values: [u16; 64],
}

impl QuantTableInfo {
    /// The DC quantizer element (`values[0]`). Convenience accessor for
    /// the common quality-estimation entry point.
    pub fn dc(&self) -> u16 {
        self.values[0]
    }

    /// True when every element equals `1` — a "no quantization" table
    /// (lossless-quality DCT, e.g. an `IJG Q=100`-style encode that
    /// floors every divisor at 1). A quick triage predicate; callers
    /// wanting the exact divisors read `values`.
    pub fn is_all_ones(&self) -> bool {
        self.values.iter().all(|&v| v == 1)
    }
}

/// Parse a DQT marker segment payload (T.81 §B.2.4.1) into zero or more
/// [`QuantTableInfo`] views, appending them to `out`.
///
/// `payload` is the byte slice the marker walker hands the inspector for
/// the DQT segment — the bytes after the marker's two-byte `Lq` length
/// field. A single segment may pack several tables back to back; each is
/// a `Pq | Tq` selector byte followed by 64 elements (1 byte each for
/// `Pq = 0`, 2 big-endian bytes each for `Pq = 1`).
///
/// Errors:
/// * `Invalid` when a table's destination `Tq` exceeds `3`.
/// * `Invalid` when a table's `Pq` nibble is neither `0` nor `1`.
/// * `Invalid` when the payload is truncated mid-table.
///
/// A successful return guarantees the whole payload was consumed exactly
/// (no trailing slack), matching the decoder's stricter DQT reader.
pub fn parse_dqt_segment(payload: &[u8], out: &mut Vec<QuantTableInfo>) -> Result<()> {
    let mut i = 0;
    while i < payload.len() {
        let pq_tq = payload[i];
        let pq = pq_tq >> 4;
        let tq = pq_tq & 0x0F;
        if tq > 3 {
            return Err(Error::invalid("DQT: table id > 3"));
        }
        if pq > 1 {
            return Err(Error::invalid("DQT: precision nibble > 1"));
        }
        i += 1;
        let n_bytes = if pq == 0 { 64 } else { 128 };
        if i + n_bytes > payload.len() {
            return Err(Error::invalid("DQT: truncated table"));
        }
        let mut nat = [0u16; 64];
        if pq == 0 {
            for k in 0..64 {
                nat[ZIGZAG[k]] = payload[i + k] as u16;
            }
        } else {
            for k in 0..64 {
                let hi = payload[i + k * 2] as u16;
                let lo = payload[i + k * 2 + 1] as u16;
                nat[ZIGZAG[k]] = (hi << 8) | lo;
            }
        }
        out.push(QuantTableInfo {
            id: tq,
            sixteen_bit: pq == 1,
            values: nat,
        });
        i += n_bytes;
    }
    Ok(())
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
    /// Typed view of the JFIF APP0 segment (T.871 §10.1) when one
    /// was present at the head of the marker prefix and structurally
    /// well-formed. `None` for streams with no JFIF magic, a
    /// truncated APP0 payload, an illegal `units` byte, or a zero
    /// density.
    ///
    /// Disjoint from `color_hint`: an Adobe-tagged stream that lacks
    /// JFIF reports `color_hint = AdobeYCbCr` and `jfif = None`;
    /// a JFIF + Adobe stream reports both with the colour hint
    /// preferring Adobe (the existing inspector policy).
    pub jfif: Option<JfifApp0>,
    /// Typed view of the first JFIF extension APP0 segment (T.871
    /// §10.2-10.5) seen in the marker prefix, when one was present and
    /// structurally well-formed. `None` for streams with no `"JFXX\0"`
    /// extension segment, a truncated payload, a reserved
    /// `extension_code`, or a thumbnail whose declared geometry
    /// overflows its `extension_data`.
    ///
    /// Independent of `jfif`: a conformant writer that emits a thumbnail
    /// via the extension mechanism (the common case — see the `jfif`
    /// note) reports `jfif = Some(..)` with `has_thumbnail() == false`
    /// *and* `jfxx = Some(..)`. The extension segment carries no
    /// colour-convention signal, so it does not influence `color_hint`.
    pub jfxx: Option<JfxxApp0>,
    /// Typed view of the Adobe APP14 segment (T.872 §6.5.3) when one
    /// was present in the marker prefix and structurally well-formed
    /// (identifier `"Adobe"`, payload ≥ 12 bytes, `transform ∈
    /// {0, 1, 2}`). `None` for streams with no APP14 magic, a
    /// truncated payload, or a reserved transform byte.
    ///
    /// Independent of `jfif`: a stream may carry both segments,
    /// either one, or neither. The `color_hint` field aggregates the
    /// two signals (preferring Adobe when both are present); the
    /// typed views are reported individually so callers building a
    /// faithful re-encoder can replay the originals.
    pub adobe: Option<AdobeApp14>,
    /// Aggregated summary of every APP2 `"ICC_PROFILE\0"` segment
    /// the inspector encountered in the marker prefix (T.872 / Annex L
    /// of T.871). `None` for streams with no APP2 ICC segments; a
    /// `Some(IccProfileChunks)` otherwise reports the declared chunk
    /// total, the cumulative profile-body length, and the per-chunk
    /// `(seq_no, payload_len)` ordering. The ICC profile bytes are
    /// not copied into `JpegInfo`; callers that need the assembled
    /// blob can re-walk the buffer with `parse_icc_profile_app2`.
    pub icc_profile: Option<IccProfileChunks>,
    /// Every COM (comment) marker segment (T.81 §B.2.4.5) seen in the
    /// marker prefix, in source order. Empty when the stream carried no
    /// comments. Each entry holds the raw comment bytes verbatim (T.81
    /// leaves their interpretation to the application); see
    /// `JpegComment` for the convenience text projections. A stream may
    /// legally carry several COM segments interspersed among the table
    /// / frame markers — all of them are reported.
    pub comments: Vec<JpegComment>,
    /// Every quantization table defined by a DQT segment (T.81
    /// §B.2.4.1) in the marker prefix, in the order the tables appear
    /// on the wire (a single DQT segment may pack several). Empty for
    /// streams with no DQT before SOS — including every lossless
    /// (SOF3 / SOF11) frame, which carries no quantization at all.
    ///
    /// A later DQT may legally redefine a destination `Tq` an earlier
    /// one already filled (T.81 §B.2.4.1 — "the quantization tables …
    /// shall be defined prior to … the scan"); the inspector reports
    /// *every* definition in source order rather than collapsing to the
    /// last-wins view a decoder would apply, so a re-encoder can replay
    /// the exact segment sequence. Read `quant_table_for(id)` for the
    /// last-wins lookup that mirrors decode-time resolution.
    pub quant_tables: Vec<QuantTableInfo>,
}

impl JpegInfo {
    /// Total component count from the SOF. Convenience accessor; the
    /// same number is `components.len()`.
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// The quantization table a decoder would resolve for destination
    /// `id` — the *last* DQT definition of that `Tq` in the prefix,
    /// mirroring T.81 §B.2.4.1 redefinition semantics (a later DQT
    /// overrides an earlier one for the same destination). Returns
    /// `None` when no DQT in the prefix defined `id`. Use the raw
    /// `quant_tables` vector when the full redefinition history matters.
    pub fn quant_table_for(&self, id: u8) -> Option<&QuantTableInfo> {
        self.quant_tables.iter().rev().find(|t| t.id == id)
    }
}

/// JFIF identifier from the start of an APP0 payload, T.871 §6.1.
/// Five bytes: `"JFIF\0"`.
const JFIF_MAGIC: &[u8; 5] = b"JFIF\0";

/// JFIF extension identifier from the start of an extension APP0
/// payload, T.871 §10.2. Five bytes: `"JFXX\0"`.
const JFXX_MAGIC: &[u8; 5] = b"JFXX\0";

/// Adobe identifier from the start of an APP14 payload, T.872 §6.5.3.
/// Five bytes: `"Adobe"`. The 6th byte and onwards carry version /
/// flags / colour-transform.
const ADOBE_MAGIC: &[u8; 5] = b"Adobe";

/// Parse a JFIF APP0 payload (T.871 §10.1) into a typed view.
///
/// `payload` is the byte slice that the marker walker hands to the
/// inspector for the APP0 segment — the bytes after the marker's
/// two-byte length field, starting with the `"JFIF\0"` identifier.
/// The function returns `Ok(JfifApp0)` when the segment is
/// structurally valid per T.871 §10.1:
///
/// * Identifier equals `b"JFIF\0"`.
/// * `Lp = 16 + 3 * HthumbnailA * VthumbnailA`, i.e. the fixed-header
///   portion (identifier + version + units + densities + thumbnail
///   dims) is followed by exactly `3 * Hthumb * Vthumb` bytes of
///   trailing RGB.
/// * `units ∈ {0x00, 0x01, 0x02}`.
/// * Both `Hdensity` and `Vdensity` are non-zero.
///
/// Errors:
/// * `Invalid` when the payload is shorter than the 14-byte fixed
///   header.
/// * `Invalid` when the identifier doesn't match `"JFIF\0"` (the
///   caller is expected to gate on the magic first, but the validator
///   re-checks so direct calls aren't a footgun).
/// * `Invalid` for an illegal `units` byte.
/// * `Invalid` for a zero density on either axis.
/// * `Invalid` when the declared trailing thumbnail size doesn't fit
///   in the remaining payload bytes.
///
/// The `version` field is reported as `(major, minor)` without
/// rejecting non-1.02 writers: the wild-corpus 1.00 / 1.01 streams
/// are too common for a parser that refuses them to be useful.
///
/// The validator never allocates; the returned `JfifApp0` is `Copy`.
pub fn parse_jfif_app0(payload: &[u8]) -> Result<JfifApp0> {
    // T.871 §10.1 fixed-header layout, byte offsets relative to the
    // start of the APP0 payload (i.e. *after* the marker + length):
    //
    //   0..5    identifier "JFIF\0"
    //   5..7    version    (major, minor)
    //   7       units      (0 | 1 | 2)
    //   8..10   Hdensity   (big-endian u16, non-zero)
    //  10..12   Vdensity   (big-endian u16, non-zero)
    //  12       HthumbnailA
    //  13       VthumbnailA
    //  14..     trailing (R, G, B) * (Hthumb * Vthumb)
    //
    // Hence 14 bytes is the absolute minimum for a thumbnail-less
    // JFIF APP0.
    if payload.len() < 14 {
        return Err(Error::invalid("parse_jfif_app0: payload too short"));
    }
    if &payload[..5] != JFIF_MAGIC {
        return Err(Error::invalid("parse_jfif_app0: identifier != JFIF\\0"));
    }

    let version_major = payload[5];
    let version_minor = payload[6];

    let units = match payload[7] {
        0x00 => JfifUnits::AspectRatio,
        0x01 => JfifUnits::DotsPerInch,
        0x02 => JfifUnits::DotsPerCm,
        _ => return Err(Error::invalid("parse_jfif_app0: illegal units byte")),
    };

    let h_density = u16::from_be_bytes([payload[8], payload[9]]);
    let v_density = u16::from_be_bytes([payload[10], payload[11]]);
    if h_density == 0 || v_density == 0 {
        return Err(Error::invalid("parse_jfif_app0: zero density"));
    }

    let thumbnail_width = payload[12];
    let thumbnail_height = payload[13];

    // Thumbnail body length must fit in the payload tail. T.871
    // §10.1's `Lp = 16 + 3 * k` is equivalent to "payload length =
    // 14 (fixed) + 3 * k", since Lp excludes the marker itself but
    // includes its own 2-byte length field, so payload-bytes ==
    // Lp - 2 == 14 + 3*k.
    let thumb_bytes = (thumbnail_width as usize) * (thumbnail_height as usize) * 3;
    if payload.len() < 14 + thumb_bytes {
        return Err(Error::invalid(
            "parse_jfif_app0: declared thumbnail overflows payload",
        ));
    }

    Ok(JfifApp0 {
        version_major,
        version_minor,
        units,
        h_density,
        v_density,
        thumbnail_width,
        thumbnail_height,
    })
}

/// Parse a JFIF extension APP0 payload (T.871 §10.2-10.5) into a typed
/// view.
///
/// `payload` is the byte slice the marker walker hands to the inspector
/// for the APP0 segment — the bytes after the marker's two-byte length
/// field, starting with the `"JFXX\0"` identifier. The function returns
/// `Ok(JfxxApp0)` when the segment is structurally valid per
/// T.871 §10.2:
///
/// * Identifier equals `b"JFXX\0"`.
/// * The `extension_code` byte (payload offset 5) is one of the three
///   defined codes (`0x10` JPEG, `0x11` palette RGB, `0x13` packed RGB).
/// * The `extension_data` that follows is large enough for the geometry
///   the extension declares:
///   - `0x10`: any (possibly empty) tail is accepted as the embedded
///     baseline-JPEG body — the inspector does not recurse into it.
///   - `0x11`: `width` / `height` are non-zero and the tail holds the
///     768-byte palette plus `width × height` index bytes.
///   - `0x13`: `width` / `height` are non-zero and the tail holds
///     `3 × width × height` packed-RGB bytes.
///
/// Errors:
/// * `Invalid` when the payload is shorter than the 6-byte
///   identifier + `extension_code` header.
/// * `Invalid` when the identifier doesn't match `"JFXX\0"` (the caller
///   is expected to gate on the magic first, but the validator re-checks
///   so direct calls aren't a footgun).
/// * `Invalid` for a reserved `extension_code` byte.
/// * `Invalid` when a `0x11` / `0x13` thumbnail declares a zero
///   dimension (both spec sections require non-zero counts) or when its
///   declared body overflows the remaining payload.
///
/// The thumbnail body is never copied; the typed view carries the
/// geometry only and is `Copy`. The validator never allocates.
pub fn parse_jfxx_app0(payload: &[u8]) -> Result<JfxxApp0> {
    // T.871 §10.2 layout, byte offsets relative to the start of the
    // APP0 payload (i.e. *after* the marker + length):
    //
    //   0..5    identifier "JFXX\0"
    //   5       extension_code (0x10 | 0x11 | 0x13)
    //   6..     extension_data (varies by code, see §10.3-10.5)
    if payload.len() < 6 {
        return Err(Error::invalid("parse_jfxx_app0: payload too short"));
    }
    if &payload[..5] != JFXX_MAGIC {
        return Err(Error::invalid("parse_jfxx_app0: identifier != JFXX\\0"));
    }

    let data = &payload[6..];
    let thumbnail = match payload[5] {
        // §10.3 — embedded baseline JPEG. The whole tail is the
        // SOI..EOI stream; the inspector does not validate or recurse
        // into it (T.871 §10.3 forbids nested JFIF/JFXX markers there,
        // but checking that would mean a second marker walk — out of
        // scope for a prefix summary).
        0x10 => JfxxThumbnail::JpegEncoded {
            jpeg_len: data.len(),
        },
        // §10.4 — 8-bit palette RGB. extension_data layout:
        //   0       HthumbnailB (non-zero)
        //   1       VthumbnailB (non-zero)
        //   2..770  768-byte palette (256 × RGB)
        //   770..   width * height index bytes
        0x11 => {
            if data.len() < 2 {
                return Err(Error::invalid(
                    "parse_jfxx_app0: palette thumbnail missing dimensions",
                ));
            }
            let width = data[0];
            let height = data[1];
            if width == 0 || height == 0 {
                return Err(Error::invalid(
                    "parse_jfxx_app0: palette thumbnail has zero dimension",
                ));
            }
            // 2 dim bytes + 768 palette + (w * h) indices.
            let need = 2usize + 768 + (width as usize) * (height as usize);
            if data.len() < need {
                return Err(Error::invalid(
                    "parse_jfxx_app0: palette thumbnail overflows payload",
                ));
            }
            JfxxThumbnail::PaletteRgb { width, height }
        }
        // §10.5 — packed 24-bit RGB. extension_data layout:
        //   0       HthumbnailC (non-zero)
        //   1       VthumbnailC (non-zero)
        //   2..     3 * width * height packed R,G,B bytes
        0x13 => {
            if data.len() < 2 {
                return Err(Error::invalid(
                    "parse_jfxx_app0: rgb thumbnail missing dimensions",
                ));
            }
            let width = data[0];
            let height = data[1];
            if width == 0 || height == 0 {
                return Err(Error::invalid(
                    "parse_jfxx_app0: rgb thumbnail has zero dimension",
                ));
            }
            let need = 2usize + 3 * (width as usize) * (height as usize);
            if data.len() < need {
                return Err(Error::invalid(
                    "parse_jfxx_app0: rgb thumbnail overflows payload",
                ));
            }
            JfxxThumbnail::Rgb24 { width, height }
        }
        _ => {
            return Err(Error::invalid(
                "parse_jfxx_app0: reserved extension_code byte",
            ));
        }
    };

    Ok(JfxxApp0 { thumbnail })
}

/// Parse an Adobe APP14 payload (T.872 §6.5.3) into a typed view.
///
/// `payload` is the byte slice that the marker walker hands to the
/// inspector for the APP14 segment — the bytes after the marker's
/// two-byte length field, starting with the `"Adobe"` identifier.
/// The function returns `Ok(AdobeApp14)` when the segment is
/// structurally valid:
///
/// * Identifier equals `b"Adobe"`.
/// * Payload is at least 12 bytes long (5 identifier + 2 version +
///   2 flags0 + 2 flags1 + 1 transform).
/// * `transform ∈ {0x00, 0x01, 0x02}`.
///
/// Errors:
/// * `Invalid` when the payload is shorter than the 12-byte fixed
///   header.
/// * `Invalid` when the identifier doesn't match `"Adobe"` (the
///   caller is expected to gate on the magic first, but the validator
///   re-checks so direct calls aren't a footgun).
/// * `Invalid` for a reserved `transform` byte. Real-world streams
///   occasionally carry `3` (some Photoshop revisions); a strict
///   typed view refuses while the inspector's coarse `ColorHint`
///   path tolerates the byte by defaulting to `AdobeUntransformed`.
///
/// The `dct_encode_version` and the two `flags_*` words are reported
/// as raw big-endian `u16`s without validation: the fields are
/// informational and writers do not always agree on their values.
///
/// The validator never allocates; the returned `AdobeApp14` is `Copy`.
pub fn parse_adobe_app14(payload: &[u8]) -> Result<AdobeApp14> {
    // T.872 §6.5.3 / Adobe Technical Note 5116 §18 layout, byte
    // offsets relative to the start of the APP14 payload (i.e.
    // *after* the marker + length):
    //
    //   0..5    identifier "Adobe"
    //   5..7    DCTEncodeVersion (big-endian u16)
    //   7..9    APP14Flags0      (big-endian u16)
    //   9..11   APP14Flags1      (big-endian u16)
    //  11       ColorTransform   (0 | 1 | 2)
    //
    // Exactly 12 bytes when no implementation-specific trailing data
    // is appended; some encoders pad with zeroes, which we tolerate.
    if payload.len() < 12 {
        return Err(Error::invalid("parse_adobe_app14: payload too short"));
    }
    if &payload[..5] != ADOBE_MAGIC {
        return Err(Error::invalid("parse_adobe_app14: identifier != Adobe"));
    }

    let dct_encode_version = u16::from_be_bytes([payload[5], payload[6]]);
    let flags_0 = u16::from_be_bytes([payload[7], payload[8]]);
    let flags_1 = u16::from_be_bytes([payload[9], payload[10]]);

    let transform = match payload[11] {
        0x00 => AdobeColorTransform::Unknown,
        0x01 => AdobeColorTransform::YCbCr,
        0x02 => AdobeColorTransform::Ycck,
        _ => return Err(Error::invalid("parse_adobe_app14: reserved transform byte")),
    };

    Ok(AdobeApp14 {
        dct_encode_version,
        flags_0,
        flags_1,
        transform,
    })
}

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
    let mut jfxx: Option<JfxxApp0> = None;
    let mut adobe: Option<AdobeApp14> = None;
    let mut icc_total: Option<u8> = None;
    let mut icc_payload_len: usize = 0;
    let mut icc_chunks: Vec<(u8, usize)> = Vec::new();
    let mut comments: Vec<JpegComment> = Vec::new();
    let mut quant_tables: Vec<QuantTableInfo> = Vec::new();

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

        if marker == markers::DQT {
            // DQT (T.81 §B.2.4.1): one or more quantization tables.
            // Collect every definition in source order; redefinition
            // semantics are exposed via `JpegInfo::quant_table_for`. A
            // structurally malformed DQT (bad Tq / Pq nibble, truncated
            // table) is a hard error here — unlike the tolerant APP
            // parsers, a quant table the inspector cannot make sense of
            // means the prefix is corrupt and any reported table would
            // be misleading.
            parse_dqt_segment(payload, &mut quant_tables)?;
            continue;
        }

        if marker == markers::COM {
            // COM (T.81 §B.2.4.5): the marker walker already consumed
            // the `Lc` length field, so `payload` is exactly the
            // `Lc-2` comment bytes (empty when `Lc = 2`). Copy them
            // verbatim — the interpretation is left to the application.
            comments.push(JpegComment {
                bytes: payload.to_vec(),
            });
            continue;
        }

        if markers::is_app(marker) {
            // APP0 = JFIF, APP14 = Adobe, APP2 = (optional) ICC profile.
            // APP2 does not affect `color_hint` directly (it carries the
            // colour-management profile separately from the YCbCr/RGB
            // mapping signalled by APP0/APP14); the inspector reports
            // it as a separate aggregated `icc_profile` summary.
            if marker == markers::APP2 && payload.len() >= 12 && &payload[..12] == ICC_PROFILE_MAGIC
            {
                if let Ok(chunk) = parse_icc_profile_app2(payload) {
                    // First-seen segment pins `total`. Later segments
                    // with a disagreeing `total` are dropped from the
                    // aggregate to keep the summary self-consistent —
                    // a stream with mismatched totals is malformed and
                    // not something the inspector should silently
                    // average away.
                    let accept = match icc_total {
                        None => {
                            icc_total = Some(chunk.total);
                            true
                        }
                        Some(t) => t == chunk.total,
                    };
                    if accept {
                        icc_chunks.push((chunk.seq_no, chunk.profile_bytes.len()));
                        icc_payload_len = icc_payload_len.saturating_add(chunk.profile_bytes.len());
                    }
                }
                continue;
            }
            if marker == markers::APP0 && payload.len() >= 5 && &payload[..5] == JFIF_MAGIC {
                // Don't overwrite an earlier Adobe tag — both should
                // not appear together but if they do, the Adobe tag
                // is the more specific signal.
                if color_hint == ColorHint::Unspecified {
                    color_hint = ColorHint::JfifYCbCr;
                }
                // Attempt to also extract the typed JFIF view. A
                // structurally malformed JFIF segment is reported as
                // `jfif = None` but the colour hint above is left in
                // place — the JFIF magic is enough to confirm the
                // colour convention even when downstream fields are
                // truncated, and refusing the entire stream over a
                // bad-density JFIF would be more disruptive than
                // useful. Only the first JFIF segment wins.
                if jfif.is_none() {
                    if let Ok(parsed) = parse_jfif_app0(payload) {
                        jfif = Some(parsed);
                    }
                }
                continue;
            }
            if marker == markers::APP0 && payload.len() >= 5 && &payload[..5] == JFXX_MAGIC {
                // JFIF extension APP0 (T.871 §10.2) — carries a thumbnail,
                // not a colour-convention signal, so `color_hint` is left
                // untouched. A structurally malformed JFXX segment is
                // reported as `jfxx = None`. Only the first JFXX segment
                // wins (a stream may legally carry several extension
                // segments; the inspector summarises the first).
                if jfxx.is_none() {
                    if let Ok(parsed) = parse_jfxx_app0(payload) {
                        jfxx = Some(parsed);
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
                // Attempt to also extract the typed Adobe view. A
                // payload that fails strict validation (reserved
                // transform byte, identifier mismatch after the
                // gate above) is reported as `adobe = None` while
                // the colour hint above remains in place — the
                // coarse signal is more tolerant by design. Only
                // the first APP14 segment wins.
                if adobe.is_none() {
                    if let Ok(parsed) = parse_adobe_app14(payload) {
                        adobe = Some(parsed);
                    }
                }
                continue;
            }
            // Other APPn segments: ignored. Their payloads do not
            // affect any field the inspector reports.
            continue;
        }

        // DHT / DAC / DNL / reserved: skipped. The inspector does not
        // surface Huffman / arithmetic-conditioning table contents; the
        // DQT path above is the one table type it reports.
    }

    let sof_kind = sof_kind.ok_or_else(|| Error::invalid("inspect: SOS before SOF"))?;

    let subsampling = classify_subsampling(&components);

    let icc_profile = icc_total.map(|total| IccProfileChunks {
        total,
        total_payload_len: icc_payload_len,
        chunks: icc_chunks,
    });

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
        jfxx,
        adobe,
        icc_profile,
        comments,
        quant_tables,
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
        // The fixture's APP0 payload — version 1.02, units = 0
        // (aspect ratio only), Hdensity = Vdensity = 1, no thumbnail.
        let jfif = info.jfif.expect("baseline JFIF view populated");
        assert_eq!(jfif.version(), (1, 2));
        assert_eq!(jfif.units, JfifUnits::AspectRatio);
        assert_eq!(jfif.h_density, 1);
        assert_eq!(jfif.v_density, 1);
        assert_eq!(jfif.thumbnail_width, 0);
        assert_eq!(jfif.thumbnail_height, 0);
        assert!(!jfif.has_thumbnail());
        assert_eq!(jfif.thumbnail_payload_len(), 0);
        // AspectRatio units => no DPI conversion is possible.
        assert_eq!(jfif.h_density_dpi(), None);
        assert_eq!(jfif.v_density_dpi(), None);
        assert_eq!(jfif.pixel_aspect_ratio(), (1, 1));
        // No APP14 segment → no typed Adobe view.
        assert!(info.adobe.is_none());
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
        // No APP0 → no typed JFIF view.
        assert!(info.jfif.is_none());
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
    fn progressive_arith_kind() {
        let buf = build_prefix(0xCA, 8, 16, 16, &[(1, 2, 2, 0)], &[]);
        let info = inspect_jpeg(&buf).expect("inspect SOF10");
        assert_eq!(info.sof_kind, SofKind::ProgressiveArith);
        assert!(info.sof_kind.is_supported_by_decoder());
        assert!(info.sof_kind.is_dct());
        assert!(info.sof_kind.is_arithmetic());
    }

    #[test]
    fn lossless_arith_kind() {
        let buf = build_prefix(0xCB, 8, 16, 16, &[(1, 1, 1, 0)], &[]);
        let info = inspect_jpeg(&buf).expect("inspect SOF11");
        assert_eq!(info.sof_kind, SofKind::LosslessArith);
        assert!(info.sof_kind.is_supported_by_decoder());
        assert!(!info.sof_kind.is_dct());
        assert!(info.sof_kind.is_arithmetic());
    }

    #[test]
    fn no_com_segment_reports_empty() {
        let buf = build_prefix(0xC0, 8, 16, 16, &[(1, 1, 1, 0)], &[]);
        let info = inspect_jpeg(&buf).expect("inspect no-COM");
        assert!(info.comments.is_empty());
    }

    #[test]
    fn single_ascii_com_segment() {
        let text = b"Created with oxideav";
        let extras = [(markers::COM, &text[..])];
        let buf = build_prefix(0xC0, 8, 16, 16, &[(1, 1, 1, 0)], &extras);
        let info = inspect_jpeg(&buf).expect("inspect single COM");
        assert_eq!(info.comments.len(), 1);
        let com = &info.comments[0];
        assert_eq!(com.bytes, text);
        assert_eq!(com.len(), text.len());
        assert!(!com.is_empty());
        assert!(com.is_ascii_text());
        assert_eq!(com.as_str_lossy(), "Created with oxideav");
    }

    #[test]
    fn zero_length_com_segment() {
        // Lc = 2 (length field only, no comment bytes) is legal per the
        // 2..=65535 range in Table B.8.
        let extras = [(markers::COM, &[][..])];
        let buf = build_prefix(0xC0, 8, 16, 16, &[(1, 1, 1, 0)], &extras);
        let info = inspect_jpeg(&buf).expect("inspect empty COM");
        assert_eq!(info.comments.len(), 1);
        assert!(info.comments[0].is_empty());
        assert_eq!(info.comments[0].len(), 0);
        // An empty comment is vacuously text and renders to "".
        assert!(info.comments[0].is_ascii_text());
        assert_eq!(info.comments[0].as_str_lossy(), "");
    }

    #[test]
    fn multiple_com_segments_in_source_order() {
        let extras = [
            (markers::COM, &b"first"[..]),
            (markers::COM, &b"second"[..]),
            (markers::COM, &b"third"[..]),
        ];
        let buf = build_prefix(0xC0, 8, 16, 16, &[(1, 1, 1, 0)], &extras);
        let info = inspect_jpeg(&buf).expect("inspect multi COM");
        assert_eq!(info.comments.len(), 3);
        assert_eq!(info.comments[0].as_str_lossy(), "first");
        assert_eq!(info.comments[1].as_str_lossy(), "second");
        assert_eq!(info.comments[2].as_str_lossy(), "third");
    }

    #[test]
    fn binary_com_segment_not_ascii_text() {
        // Comment bytes whose interpretation T.81 leaves to the
        // application: a non-printable payload is preserved verbatim
        // and flagged as not-ascii-text.
        let raw = [0x00u8, 0xFF, 0x80, 0x01];
        let extras = [(markers::COM, &raw[..])];
        let buf = build_prefix(0xC0, 8, 16, 16, &[(1, 1, 1, 0)], &extras);
        let info = inspect_jpeg(&buf).expect("inspect binary COM");
        assert_eq!(info.comments.len(), 1);
        assert_eq!(info.comments[0].bytes, raw);
        assert!(!info.comments[0].is_ascii_text());
        // Lossy rendering substitutes U+FFFD for the invalid bytes.
        assert!(info.comments[0].as_str_lossy().contains('\u{FFFD}'));
    }

    #[test]
    fn com_after_sof_before_sos_reported() {
        // COM is legal anywhere in the prefix, including between SOF
        // and SOS. `build_prefix` puts extras before SOF, so craft a
        // SOF→COM→SOS layout manually.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0xFF, markers::SOI]);
        // SOF: P=8, Y=16, X=16, Nf=1, one component triple.
        buf.extend_from_slice(&[0xFF, 0xC0]);
        buf.extend_from_slice(&11u16.to_be_bytes());
        buf.push(8);
        buf.extend_from_slice(&16u16.to_be_bytes());
        buf.extend_from_slice(&16u16.to_be_bytes());
        buf.push(1);
        buf.extend_from_slice(&[1, 0x11, 0]);
        // COM between SOF and SOS.
        let text = b"post-frame note";
        buf.extend_from_slice(&[0xFF, markers::COM]);
        buf.extend_from_slice(&((text.len() + 2) as u16).to_be_bytes());
        buf.extend_from_slice(text);
        // SOS marker (inspector stops here).
        buf.extend_from_slice(&[0xFF, markers::SOS]);
        let info = inspect_jpeg(&buf).expect("inspect SOF→COM→SOS");
        assert_eq!(info.comments.len(), 1);
        assert_eq!(info.comments[0].as_str_lossy(), "post-frame note");
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
        let adobe = info.adobe.expect("typed Adobe view");
        assert_eq!(adobe.dct_encode_version, 0x0065);
        assert_eq!(adobe.flags_0, 0);
        assert_eq!(adobe.flags_1, 0);
        assert_eq!(adobe.transform, AdobeColorTransform::Ycck);
        assert_eq!(adobe.transform.as_byte(), 0x02);
        assert_eq!(adobe.as_color_hint(), ColorHint::AdobeYcck);
        assert!(!adobe.is_standard_version()); // 0x65 = 101
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
        let adobe = info.adobe.expect("typed Adobe view");
        assert_eq!(adobe.transform, AdobeColorTransform::Unknown);
        assert_eq!(adobe.transform.as_byte(), 0x00);
        assert_eq!(adobe.as_color_hint(), ColorHint::AdobeUntransformed);
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
    fn jfif_dots_per_inch_parsed() {
        // version 1.01, units = 0x01 (DPI), 72 dpi both axes, no thumbnail.
        let payload = b"JFIF\0\x01\x01\x01\x00\x48\x00\x48\x00\x00";
        let extras = [(markers::APP0, &payload[..])];
        let buf = build_prefix(
            0xC0,
            8,
            32,
            32,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect JFIF dpi");
        let jfif = info.jfif.expect("JFIF view present");
        assert_eq!(jfif.units, JfifUnits::DotsPerInch);
        assert_eq!(jfif.h_density, 72);
        assert_eq!(jfif.v_density, 72);
        assert_eq!(jfif.version(), (1, 1));
        assert_eq!(jfif.h_density_dpi(), Some(72));
        assert_eq!(jfif.v_density_dpi(), Some(72));
        assert_eq!(jfif.units.as_byte(), 0x01);
    }

    #[test]
    fn jfif_dots_per_cm_converted_to_dpi() {
        // units = 0x02 (DPCM). 100 dpcm = 254 dpi exactly. 39 dpcm
        // tests the rounding path: 39 × 2.54 = 99.06 → expected 99.
        let payload = b"JFIF\0\x01\x02\x02\x00\x64\x00\x27\x00\x00";
        let parsed = parse_jfif_app0(payload).expect("dpcm");
        assert_eq!(parsed.units, JfifUnits::DotsPerCm);
        assert_eq!(parsed.h_density, 100);
        assert_eq!(parsed.v_density, 39);
        assert_eq!(parsed.h_density_dpi(), Some(254));
        assert_eq!(parsed.v_density_dpi(), Some(99));
        assert_eq!(parsed.units.as_byte(), 0x02);
    }

    #[test]
    fn jfif_rejects_illegal_units() {
        let payload = b"JFIF\0\x01\x02\x05\x00\x48\x00\x48\x00\x00"; // units = 5
        assert!(parse_jfif_app0(payload).is_err());
    }

    #[test]
    fn jfif_rejects_zero_density() {
        let payload = b"JFIF\0\x01\x02\x01\x00\x00\x00\x48\x00\x00"; // Hdensity = 0
        assert!(parse_jfif_app0(payload).is_err());
        let payload = b"JFIF\0\x01\x02\x01\x00\x48\x00\x00\x00\x00"; // Vdensity = 0
        assert!(parse_jfif_app0(payload).is_err());
    }

    #[test]
    fn jfif_rejects_truncated_header() {
        // 13 bytes — one byte short of the minimum fixed header.
        let payload = b"JFIF\0\x01\x02\x01\x00\x48\x00\x48\x00";
        assert!(parse_jfif_app0(payload).is_err());
    }

    #[test]
    fn jfif_rejects_bad_identifier() {
        // "JFXX" extension marker is NOT the JFIF APP0 marker.
        let payload = b"JFXX\0\x01\x02\x01\x00\x48\x00\x48\x00\x00";
        assert!(parse_jfif_app0(payload).is_err());
    }

    #[test]
    fn jfif_with_2x2_thumbnail() {
        // 2×2 thumbnail = 12 RGB bytes appended. Total payload =
        // 14 (header) + 12 = 26 bytes.
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(b"JFIF\0");
        payload.push(0x01);
        payload.push(0x02);
        payload.push(0x01); // dpi
        payload.extend_from_slice(&96u16.to_be_bytes());
        payload.extend_from_slice(&96u16.to_be_bytes());
        payload.push(2); // Hthumb
        payload.push(2); // Vthumb
        for i in 0..12 {
            payload.push(i as u8);
        }
        let parsed = parse_jfif_app0(&payload).expect("parse 2x2 thumb");
        assert!(parsed.has_thumbnail());
        assert_eq!(parsed.thumbnail_width, 2);
        assert_eq!(parsed.thumbnail_height, 2);
        assert_eq!(parsed.thumbnail_payload_len(), 12);
        assert_eq!(parsed.h_density_dpi(), Some(96));
    }

    #[test]
    fn jfxx_jpeg_encoded_thumbnail() {
        // §10.3 — extension_code 0x10, a tail standing in for an
        // embedded baseline JPEG. The inspector reports its byte length
        // without recursing into it.
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(b"JFXX\0");
        payload.push(0x10);
        // Stand-in embedded stream (SOI..EOI shell, 4 bytes here).
        payload.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xD9]);
        let parsed = parse_jfxx_app0(&payload).expect("parse JFXX jpeg thumb");
        assert_eq!(parsed.thumbnail, JfxxThumbnail::JpegEncoded { jpeg_len: 4 });
        assert_eq!(parsed.thumbnail.extension_code(), 0x10);
    }

    #[test]
    fn jfxx_palette_rgb_thumbnail() {
        // §10.4 — extension_code 0x11: 2 dim bytes + 768 palette +
        // (w * h) index bytes.
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(b"JFXX\0");
        payload.push(0x11);
        payload.push(4); // HthumbnailB
        payload.push(3); // VthumbnailB
        payload.extend_from_slice(&[0u8; 768]); // palette
        payload.extend_from_slice(&[0u8; 12]); // 4 * 3 indices
        let parsed = parse_jfxx_app0(&payload).expect("parse JFXX palette thumb");
        assert_eq!(
            parsed.thumbnail,
            JfxxThumbnail::PaletteRgb {
                width: 4,
                height: 3
            }
        );
        assert_eq!(parsed.thumbnail.extension_code(), 0x11);
    }

    #[test]
    fn jfxx_rgb24_thumbnail() {
        // §10.5 — extension_code 0x13: 2 dim bytes + 3 * w * h RGB bytes.
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(b"JFXX\0");
        payload.push(0x13);
        payload.push(2); // HthumbnailC
        payload.push(2); // VthumbnailC
        payload.extend_from_slice(&[0u8; 12]); // 2 * 2 * 3 RGB
        let parsed = parse_jfxx_app0(&payload).expect("parse JFXX rgb thumb");
        assert_eq!(
            parsed.thumbnail,
            JfxxThumbnail::Rgb24 {
                width: 2,
                height: 2
            }
        );
        assert_eq!(parsed.thumbnail.extension_code(), 0x13);
    }

    #[test]
    fn jfxx_rejects_short_payload() {
        // Only the identifier, no extension_code byte.
        assert!(parse_jfxx_app0(b"JFXX\0").is_err());
    }

    #[test]
    fn jfxx_rejects_bad_identifier() {
        // The JFIF APP0 magic is not the extension magic.
        let payload = b"JFIF\0\x10";
        assert!(parse_jfxx_app0(payload).is_err());
    }

    #[test]
    fn jfxx_rejects_reserved_extension_code() {
        // 0x12 is not one of the three defined codes (0x10/0x11/0x13).
        let payload = b"JFXX\0\x12\x02\x02";
        assert!(parse_jfxx_app0(payload).is_err());
    }

    #[test]
    fn jfxx_rejects_zero_dimension() {
        // §10.4 / §10.5 require non-zero pixel counts.
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(b"JFXX\0");
        payload.push(0x13);
        payload.push(0); // zero width
        payload.push(4);
        assert!(parse_jfxx_app0(&payload).is_err());
    }

    #[test]
    fn jfxx_rejects_thumbnail_overflowing_payload() {
        // Declare a 2×2 packed-RGB thumbnail (needs 12 bytes) but only
        // attach 5.
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(b"JFXX\0");
        payload.push(0x13);
        payload.push(2);
        payload.push(2);
        payload.extend_from_slice(&[0, 0, 0, 0, 0]);
        assert!(parse_jfxx_app0(&payload).is_err());
    }

    #[test]
    fn inspect_jfif_plus_jfxx_dual_app0() {
        // Conformant common case: a JFIF APP0 with no inline thumbnail
        // immediately followed by a JFXX extension APP0 carrying one.
        let jfif: Vec<u8> = b"JFIF\0\x01\x02\x01\x00\x48\x00\x48\x00\x00".to_vec();
        let mut jfxx: Vec<u8> = Vec::new();
        jfxx.extend_from_slice(b"JFXX\0");
        jfxx.push(0x13);
        jfxx.push(2);
        jfxx.push(2);
        jfxx.extend_from_slice(&[0u8; 12]);
        let extras = [
            (markers::APP0, jfif.as_slice()),
            (markers::APP0, jfxx.as_slice()),
        ];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect JFIF+JFXX");
        // Both typed views populate; the JFIF one carries no inline thumb.
        let jfif_view = info.jfif.expect("jfif view");
        assert!(!jfif_view.has_thumbnail());
        let jfxx_view = info.jfxx.expect("jfxx view");
        assert_eq!(
            jfxx_view.thumbnail,
            JfxxThumbnail::Rgb24 {
                width: 2,
                height: 2
            }
        );
        // The extension segment carries no colour-convention signal —
        // the hint comes from the JFIF APP0 only.
        assert_eq!(info.color_hint, ColorHint::JfifYCbCr);
    }

    #[test]
    fn inspect_no_jfxx_when_absent() {
        let buf = build_prefix(0xC0, 8, 8, 8, &[(1, 1, 1, 0)], &[]);
        let info = inspect_jpeg(&buf).expect("inspect plain grayscale");
        assert!(info.jfxx.is_none());
    }

    #[test]
    fn jfif_rejects_thumbnail_overflowing_payload() {
        // Declare a 2×2 thumbnail (needs 12 bytes) but only attach 5.
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(b"JFIF\0");
        payload.push(0x01);
        payload.push(0x02);
        payload.push(0x01);
        payload.extend_from_slice(&72u16.to_be_bytes());
        payload.extend_from_slice(&72u16.to_be_bytes());
        payload.push(2);
        payload.push(2);
        payload.extend_from_slice(&[0, 0, 0, 0, 0]);
        assert!(parse_jfif_app0(&payload).is_err());
    }

    #[test]
    fn jfif_view_disjoint_from_adobe_when_only_adobe_present() {
        let mut adobe = Vec::new();
        adobe.extend_from_slice(b"Adobe");
        adobe.extend_from_slice(&[0x00, 0x65, 0x00, 0x00, 0x00, 0x00, 1]);
        let extras = [(markers::APP14, adobe.as_slice())];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect Adobe-only");
        assert_eq!(info.color_hint, ColorHint::AdobeYCbCr);
        // No JFIF segment → no typed view, even though Adobe is set.
        assert!(info.jfif.is_none());
        // Adobe view IS populated for an Adobe-only stream.
        let adobe = info.adobe.expect("Adobe typed view");
        assert_eq!(adobe.transform, AdobeColorTransform::YCbCr);
    }

    #[test]
    fn jfif_malformed_segment_still_sets_color_hint() {
        // APP0 payload starts with "JFIF\0" magic but is truncated to
        // 8 bytes — the colour hint should still flip to JfifYCbCr
        // (the magic is the colour-convention signal), but the typed
        // view stays `None`.
        let payload = b"JFIF\0\x01\x02\x01"; // 8 bytes
        let extras = [(markers::APP0, &payload[..])];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect truncated JFIF");
        assert_eq!(info.color_hint, ColorHint::JfifYCbCr);
        assert!(info.jfif.is_none());
    }

    #[test]
    fn jfif_only_first_segment_wins() {
        // Two APP0 JFIF segments back-to-back; the second has
        // different density. The inspector picks the first.
        let payload1 = &b"JFIF\0\x01\x02\x01\x00\x48\x00\x48\x00\x00"[..]; // 72 dpi
        let payload2 = &b"JFIF\0\x01\x02\x01\x01\x90\x01\x90\x00\x00"[..]; // 400 dpi
        let extras = [(markers::APP0, payload1), (markers::APP0, payload2)];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect dup JFIF");
        let jfif = info.jfif.expect("typed view");
        assert_eq!(jfif.h_density, 72);
        assert_eq!(jfif.v_density, 72);
    }

    #[test]
    fn parse_adobe_app14_standard_version() {
        // "Adobe" + version 100 + flags0 0 + flags1 0 + transform 1.
        let payload = b"Adobe\x00\x64\x00\x00\x00\x00\x01";
        let parsed = parse_adobe_app14(payload).expect("parse standard Adobe");
        assert!(parsed.is_standard_version());
        assert_eq!(parsed.dct_encode_version, 100);
        assert_eq!(parsed.flags_0, 0);
        assert_eq!(parsed.flags_1, 0);
        assert_eq!(parsed.transform, AdobeColorTransform::YCbCr);
        assert_eq!(parsed.transform.as_byte(), 0x01);
        assert_eq!(parsed.as_color_hint(), ColorHint::AdobeYCbCr);
    }

    #[test]
    fn parse_adobe_app14_with_flags() {
        // Flags0 = 0xC000 (chroma-blur + dampened-edge bits set).
        let payload = b"Adobe\x00\x64\xC0\x00\x00\x00\x00";
        let parsed = parse_adobe_app14(payload).expect("parse flags");
        assert_eq!(parsed.flags_0, 0xC000);
        assert_eq!(parsed.transform, AdobeColorTransform::Unknown);
        assert_eq!(parsed.as_color_hint(), ColorHint::AdobeUntransformed);
    }

    #[test]
    fn parse_adobe_app14_rejects_too_short() {
        // 11 bytes — one short of the 12-byte fixed header.
        let payload = b"Adobe\x00\x64\x00\x00\x00\x00";
        assert!(parse_adobe_app14(payload).is_err());
    }

    #[test]
    fn parse_adobe_app14_rejects_bad_identifier() {
        let payload = b"Other\x00\x64\x00\x00\x00\x00\x01";
        assert!(parse_adobe_app14(payload).is_err());
    }

    #[test]
    fn parse_adobe_app14_rejects_reserved_transform() {
        let payload = b"Adobe\x00\x64\x00\x00\x00\x00\x05"; // transform = 5
        assert!(parse_adobe_app14(payload).is_err());
    }

    #[test]
    fn inspect_adobe_with_reserved_transform_byte_keeps_color_hint() {
        // The inspector's coarse colour-hint path tolerates a
        // reserved transform byte by falling back to
        // AdobeUntransformed, but the typed view must refuse it
        // (so `info.adobe` is `None`).
        let mut adobe = Vec::new();
        adobe.extend_from_slice(b"Adobe");
        adobe.extend_from_slice(&[0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x05]);
        let extras = [(markers::APP14, adobe.as_slice())];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect reserved transform");
        assert_eq!(info.color_hint, ColorHint::AdobeUntransformed);
        assert!(info.adobe.is_none());
    }

    #[test]
    fn inspect_jfif_and_adobe_both_populated() {
        // A stream with both APP0 JFIF and APP14 Adobe — typed views
        // for both are populated, and the colour hint takes the
        // Adobe one per the inspector's existing precedence policy.
        let jfif_payload = &b"JFIF\0\x01\x02\x01\x00\x48\x00\x48\x00\x00"[..];
        let mut adobe = Vec::new();
        adobe.extend_from_slice(b"Adobe");
        adobe.extend_from_slice(&[0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x01]);
        let extras = [
            (markers::APP0, jfif_payload),
            (markers::APP14, adobe.as_slice()),
        ];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect dual");
        let jfif = info.jfif.expect("typed JFIF");
        let adobe_view = info.adobe.expect("typed Adobe");
        assert_eq!(jfif.h_density, 72);
        assert!(adobe_view.is_standard_version());
        assert_eq!(adobe_view.transform, AdobeColorTransform::YCbCr);
        // Inspector precedence: Adobe wins when JFIF appears first.
        // Existing behaviour — APP0 sets JfifYCbCr, then APP14
        // overrides to AdobeYCbCr.
        assert_eq!(info.color_hint, ColorHint::AdobeYCbCr);
    }

    #[test]
    fn inspect_adobe_first_segment_wins() {
        // Two APP14 Adobe segments back-to-back with different
        // transform bytes; the first one populates the typed view.
        let mut a1 = Vec::new();
        a1.extend_from_slice(b"Adobe");
        a1.extend_from_slice(&[0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x01]);
        let mut a2 = Vec::new();
        a2.extend_from_slice(b"Adobe");
        a2.extend_from_slice(&[0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x02]);
        let extras = [
            (markers::APP14, a1.as_slice()),
            (markers::APP14, a2.as_slice()),
        ];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect dup Adobe");
        let adobe = info.adobe.expect("typed Adobe view");
        assert_eq!(adobe.transform, AdobeColorTransform::YCbCr);
    }

    // --- APP2 ICC_PROFILE typed view ---

    /// Build an APP2 payload carrying the `"ICC_PROFILE\0"` signature,
    /// followed by the (seq_no, total) chunk header, then `body`.
    /// Returns the bytes the inspector's marker walker would hand to
    /// `parse_icc_profile_app2`.
    fn icc_payload(seq_no: u8, total: u8, body: &[u8]) -> Vec<u8> {
        let mut p = Vec::with_capacity(14 + body.len());
        p.extend_from_slice(ICC_PROFILE_MAGIC);
        p.push(seq_no);
        p.push(total);
        p.extend_from_slice(body);
        p
    }

    #[test]
    fn parse_icc_profile_app2_minimal() {
        // 14 bytes — identifier + (seq=1, total=1) + empty body.
        let payload = icc_payload(1, 1, &[]);
        let chunk = parse_icc_profile_app2(&payload).expect("parse minimal ICC");
        assert_eq!(chunk.seq_no, 1);
        assert_eq!(chunk.total, 1);
        assert_eq!(chunk.profile_bytes.len(), 0);
    }

    #[test]
    fn parse_icc_profile_app2_with_body() {
        let body: Vec<u8> = (0..64u8).collect();
        let payload = icc_payload(1, 1, &body);
        let chunk = parse_icc_profile_app2(&payload).expect("parse body ICC");
        assert_eq!(chunk.seq_no, 1);
        assert_eq!(chunk.total, 1);
        assert_eq!(chunk.profile_bytes, body.as_slice());
    }

    #[test]
    fn parse_icc_profile_app2_rejects_too_short() {
        // 13 bytes — one short of the 14-byte fixed header.
        let mut payload = Vec::new();
        payload.extend_from_slice(ICC_PROFILE_MAGIC);
        payload.push(1);
        assert!(parse_icc_profile_app2(&payload).is_err());
    }

    #[test]
    fn parse_icc_profile_app2_rejects_bad_identifier() {
        let mut payload = Vec::new();
        payload.extend_from_slice(b"OtherProfile"); // 12 bytes, wrong
        payload.push(1);
        payload.push(1);
        assert!(parse_icc_profile_app2(&payload).is_err());
    }

    #[test]
    fn parse_icc_profile_app2_rejects_zero_total() {
        // seq_no = 1, total = 0 (invalid: a profile must have ≥ 1 chunk).
        let payload = icc_payload(1, 0, &[]);
        assert!(parse_icc_profile_app2(&payload).is_err());
    }

    #[test]
    fn parse_icc_profile_app2_rejects_zero_seq_no() {
        // seq_no = 0 (one-based numbering — zero is invalid).
        let payload = icc_payload(0, 1, &[]);
        assert!(parse_icc_profile_app2(&payload).is_err());
    }

    #[test]
    fn parse_icc_profile_app2_rejects_seq_no_above_total() {
        let payload = icc_payload(3, 2, &[]);
        assert!(parse_icc_profile_app2(&payload).is_err());
    }

    #[test]
    fn inspect_single_icc_chunk_populates_summary() {
        let body: Vec<u8> = (0..128u8).collect();
        let payload = icc_payload(1, 1, &body);
        let extras = [(markers::APP2, payload.as_slice())];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect single ICC");
        let icc = info.icc_profile.expect("typed ICC summary");
        assert_eq!(icc.total, 1);
        assert_eq!(icc.total_payload_len, body.len());
        assert_eq!(icc.chunks.len(), 1);
        assert_eq!(icc.chunks[0], (1, body.len()));
        assert!(icc.is_complete());
        // APP2 ICC does not influence colour-hint signalling.
        assert_eq!(info.color_hint, ColorHint::Unspecified);
    }

    #[test]
    fn inspect_multi_chunk_icc_concatenates_lengths() {
        // A 256-byte profile split across three chunks: 100 + 100 + 56.
        let p1 = icc_payload(1, 3, &[0xAA; 100]);
        let p2 = icc_payload(2, 3, &[0xBB; 100]);
        let p3 = icc_payload(3, 3, &[0xCC; 56]);
        let extras = [
            (markers::APP2, p1.as_slice()),
            (markers::APP2, p2.as_slice()),
            (markers::APP2, p3.as_slice()),
        ];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect multi-chunk ICC");
        let icc = info.icc_profile.expect("typed ICC summary");
        assert_eq!(icc.total, 3);
        assert_eq!(icc.total_payload_len, 256);
        assert_eq!(icc.chunks, vec![(1u8, 100usize), (2, 100), (3, 56)]);
        assert!(icc.is_complete());
    }

    #[test]
    fn inspect_missing_icc_chunk_marks_incomplete() {
        // total=3 but only seq 1 and 3 present — `is_complete` is false
        // and the summary still aggregates what's there.
        let p1 = icc_payload(1, 3, &[0x11; 10]);
        let p3 = icc_payload(3, 3, &[0x33; 30]);
        let extras = [
            (markers::APP2, p1.as_slice()),
            (markers::APP2, p3.as_slice()),
        ];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect partial ICC");
        let icc = info.icc_profile.expect("partial ICC summary");
        assert_eq!(icc.total, 3);
        assert_eq!(icc.chunks.len(), 2);
        assert!(!icc.is_complete());
    }

    #[test]
    fn inspect_duplicate_icc_seq_marks_incomplete() {
        // Two chunks both numbered seq=1 of total=2; the aggregate
        // refuses to call this complete.
        let p1 = icc_payload(1, 2, &[0xAA; 10]);
        let p1_dup = icc_payload(1, 2, &[0xBB; 10]);
        let extras = [
            (markers::APP2, p1.as_slice()),
            (markers::APP2, p1_dup.as_slice()),
        ];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect dup ICC");
        let icc = info.icc_profile.expect("typed ICC summary");
        assert_eq!(icc.total, 2);
        assert_eq!(icc.chunks.len(), 2);
        assert!(!icc.is_complete());
    }

    #[test]
    fn inspect_mismatched_icc_totals_drops_second() {
        // Two APP2 ICC segments declaring different totals — the
        // inspector pins `total` to the first and drops the second
        // from the aggregate (its bytes are not double-counted).
        let p1 = icc_payload(1, 2, &[0xAA; 10]);
        let p_bad = icc_payload(1, 5, &[0xCC; 20]);
        let extras = [
            (markers::APP2, p1.as_slice()),
            (markers::APP2, p_bad.as_slice()),
        ];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect mismatched totals");
        let icc = info.icc_profile.expect("ICC summary");
        // First segment pinned total to 2; the second's payload was
        // dropped, so total_payload_len reflects only the first chunk.
        assert_eq!(icc.total, 2);
        assert_eq!(icc.total_payload_len, 10);
        assert_eq!(icc.chunks.len(), 1);
        assert_eq!(icc.chunks[0], (1, 10));
        assert!(!icc.is_complete()); // missing seq=2
    }

    #[test]
    fn inspect_app2_without_icc_magic_is_ignored() {
        // APP2 carrying an unrelated identifier ("FPXR\0" — FlashPix
        // — for example). The inspector ignores it and reports no
        // ICC summary.
        let mut payload = Vec::new();
        payload.extend_from_slice(b"FPXR\0extra-bytes-go-here");
        let extras = [(markers::APP2, payload.as_slice())];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect non-ICC APP2");
        assert!(info.icc_profile.is_none());
    }

    #[test]
    fn inspect_no_icc_segment_leaves_field_none() {
        // Sanity: the existing baseline-grayscale path still reports
        // `icc_profile = None` so callers can use it as the no-ICC
        // sentinel.
        let buf = build_prefix(0xC0, 8, 8, 8, &[(1, 1, 1, 0)], &[]);
        let info = inspect_jpeg(&buf).expect("inspect no-ICC");
        assert!(info.icc_profile.is_none());
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

    // ---- DQT (quantization table) inspector ----

    /// Build an 8-bit DQT payload for one table: `Pq|Tq` byte followed
    /// by 64 elements in **zigzag** order (the wire order). `nat` is the
    /// natural-order table the caller expects back out.
    fn dqt8_payload(tq: u8, nat: &[u16; 64]) -> Vec<u8> {
        let mut p = vec![tq & 0x0F]; // Pq = 0 (8-bit)
        for k in 0..64 {
            p.push(nat[ZIGZAG[k]] as u8);
        }
        p
    }

    #[test]
    fn dqt_single_8bit_table_natural_order() {
        // A recognisable natural-order table: value == natural index.
        let mut nat = [0u16; 64];
        for i in 0..64 {
            nat[i] = i as u16;
        }
        let payload = dqt8_payload(0, &nat);
        let extras = [(markers::DQT, &payload[..])];
        let buf = build_prefix(0xC0, 8, 16, 16, &[(1, 1, 1, 0)], &extras);
        let info = inspect_jpeg(&buf).expect("inspect single DQT");
        assert_eq!(info.quant_tables.len(), 1);
        let t = &info.quant_tables[0];
        assert_eq!(t.id, 0);
        assert!(!t.sixteen_bit);
        // Un-zigzagged back to natural order.
        assert_eq!(t.values, nat);
        assert_eq!(t.dc(), 0);
        assert!(!t.is_all_ones());
        // Last-wins lookup resolves the same single definition.
        assert_eq!(info.quant_table_for(0), Some(t));
        assert_eq!(info.quant_table_for(1), None);
    }

    #[test]
    fn dqt_all_ones_table_flagged() {
        let nat = [1u16; 64];
        let payload = dqt8_payload(2, &nat);
        let extras = [(markers::DQT, &payload[..])];
        let buf = build_prefix(0xC0, 8, 8, 8, &[(1, 1, 1, 2)], &extras);
        let info = inspect_jpeg(&buf).expect("inspect all-ones DQT");
        assert_eq!(info.quant_tables.len(), 1);
        assert_eq!(info.quant_tables[0].id, 2);
        assert!(info.quant_tables[0].is_all_ones());
        assert_eq!(info.quant_tables[0].dc(), 1);
    }

    #[test]
    fn dqt_two_tables_one_segment() {
        // One DQT segment packing a luma (Tq=0) and a chroma (Tq=1)
        // table back to back — the multi-table-per-segment layout
        // every JFIF writer uses.
        let mut payload = dqt8_payload(0, &DEFAULT_LUMA_NAT());
        payload.extend_from_slice(&dqt8_payload(1, &DEFAULT_CHROMA_NAT()));
        let extras = [(markers::DQT, &payload[..])];
        let buf = build_prefix(
            0xC0,
            8,
            16,
            16,
            &[(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
            &extras,
        );
        let info = inspect_jpeg(&buf).expect("inspect packed DQT");
        assert_eq!(info.quant_tables.len(), 2);
        assert_eq!(info.quant_tables[0].id, 0);
        assert_eq!(info.quant_tables[1].id, 1);
        // Component 0 → table 0, components 1/2 → table 1.
        assert_eq!(
            info.quant_table_for(info.components[0].quant_table)
                .unwrap()
                .dc(),
            DEFAULT_LUMA_NAT()[0]
        );
        assert_eq!(
            info.quant_table_for(info.components[1].quant_table)
                .unwrap(),
            &info.quant_tables[1]
        );
    }

    #[test]
    fn dqt_16bit_table_round_trips() {
        // Pq = 1 (16-bit): each element is two big-endian bytes.
        let mut nat = [0u16; 64];
        for i in 0..64 {
            nat[i] = 256 + (i as u16) * 3; // values that overflow 8-bit
        }
        let mut payload = vec![0x10]; // Pq=1 (16-bit), Tq=0
        for k in 0..64 {
            payload.extend_from_slice(&nat[ZIGZAG[k]].to_be_bytes());
        }
        let extras = [(markers::DQT, &payload[..])];
        let buf = build_prefix(0xC2, 12, 16, 16, &[(1, 1, 1, 0)], &extras);
        let info = inspect_jpeg(&buf).expect("inspect 16-bit DQT");
        assert_eq!(info.quant_tables.len(), 1);
        assert!(info.quant_tables[0].sixteen_bit);
        assert_eq!(info.quant_tables[0].values, nat);
    }

    #[test]
    fn dqt_redefinition_last_wins_but_history_kept() {
        // Two DQT segments redefining Tq = 0. `quant_tables` keeps both
        // in source order; `quant_table_for` resolves the later one.
        let first = dqt8_payload(0, &[7u16; 64]);
        let second = dqt8_payload(0, &[9u16; 64]);
        let extras = [(markers::DQT, &first[..]), (markers::DQT, &second[..])];
        let buf = build_prefix(0xC0, 8, 8, 8, &[(1, 1, 1, 0)], &extras);
        let info = inspect_jpeg(&buf).expect("inspect redefined DQT");
        assert_eq!(info.quant_tables.len(), 2);
        assert_eq!(info.quant_tables[0].dc(), 7);
        assert_eq!(info.quant_tables[1].dc(), 9);
        // Decode-time resolution = last definition wins.
        assert_eq!(info.quant_table_for(0).unwrap().dc(), 9);
    }

    #[test]
    fn dqt_absent_on_lossless_frame() {
        // A SOF3 (lossless) frame carries no quantization at all.
        let buf = build_prefix(0xC3, 8, 16, 16, &[(1, 1, 1, 0)], &[]);
        let info = inspect_jpeg(&buf).expect("inspect lossless");
        assert!(info.quant_tables.is_empty());
        assert_eq!(info.quant_table_for(0), None);
    }

    #[test]
    fn dqt_rejects_bad_table_id() {
        // Tq = 5 is out of the 0..=3 range.
        let mut payload = vec![0x05];
        payload.extend(std::iter::repeat(1u8).take(64));
        let extras = [(markers::DQT, &payload[..])];
        let buf = build_prefix(0xC0, 8, 8, 8, &[(1, 1, 1, 0)], &extras);
        assert!(inspect_jpeg(&buf).is_err());
    }

    #[test]
    fn dqt_rejects_truncated_table() {
        // Pq = 0 promises 64 bytes but only 10 follow.
        let mut payload = vec![0x00];
        payload.extend(std::iter::repeat(1u8).take(10));
        let extras = [(markers::DQT, &payload[..])];
        let buf = build_prefix(0xC0, 8, 8, 8, &[(1, 1, 1, 0)], &extras);
        assert!(inspect_jpeg(&buf).is_err());
    }

    #[test]
    fn parse_dqt_segment_standalone() {
        // The standalone validator works without an enclosing JPEG.
        let payload = dqt8_payload(3, &[42u16; 64]);
        let mut out = Vec::new();
        parse_dqt_segment(&payload, &mut out).expect("standalone DQT");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 3);
        assert_eq!(out[0].dc(), 42);
    }

    // Small natural-order helpers for the packed-DQT test — recognisable
    // distinct DC values so the per-component lookup is unambiguous.
    #[allow(non_snake_case)]
    fn DEFAULT_LUMA_NAT() -> [u16; 64] {
        let mut t = [16u16; 64];
        t[0] = 8;
        t
    }
    #[allow(non_snake_case)]
    fn DEFAULT_CHROMA_NAT() -> [u16; 64] {
        let mut t = [24u16; 64];
        t[0] = 17;
        t
    }
}
