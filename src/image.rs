//! Crate-local image, frame, plane, pixel-format, colour, metadata and
//! option types — the standalone (no `oxideav-core`) vocabulary of the
//! workspace image-crate API contract, plus the Motion-JPEG video frame
//! shape.
//!
//! Defined here (rather than reusing `oxideav_core::VideoFrame` /
//! `oxideav_core::frame::VideoPlane` / `oxideav_core::PixelFormat`) so
//! the crate can be built with the default `registry` feature off —
//! i.e. without depending on `oxideav-core` at all. When the
//! `registry` feature is on the `crate::registry` module provides
//! `From<JpegImage> for oxideav_core::VideoFrame`, `From<MjpegFrame>
//! for oxideav_core::VideoFrame`, the pixel-format conversions and the
//! colour-signal conversion so the `Decoder` / `Encoder` trait surface
//! interoperates cleanly.
//!
//! Two shapes live here:
//!
//! * [`JpegImage`] — one decoded (or to-be-encoded) still picture with
//!   its dimensions, native pixel format, planes, colour description
//!   and metadata. This is what [`crate::decode`] returns and what
//!   [`crate::encode`] consumes, with or without `oxideav-core`.
//! * [`MjpegFrame`] — the slim Motion-JPEG video frame (`pts` +
//!   planes), layout-compatible with the framework's `VideoFrame`. It
//!   carries no dimensions or format because, like the framework
//!   frame, those travel in the stream's codec parameters.

use crate::error::{MjpegError, Result};

// ---------------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------------

/// One image plane: row-major bytes plus the row stride in bytes.
/// Layout-compatible with `oxideav_core::frame::VideoPlane`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Plane {
    /// Bytes per row in `data` (may be larger than the logical row width).
    pub stride: usize,
    /// Raw plane bytes, packed `stride` × number of rows.
    pub data: Vec<u8>,
}

impl Plane {
    /// Wrap a row-major byte buffer with its row stride.
    pub fn new(stride: usize, data: Vec<u8>) -> Self {
        Self { stride, data }
    }

    /// Number of complete rows in the plane (`data.len() / stride`;
    /// `0` for a zero stride).
    pub fn rows(&self) -> usize {
        self.data.len().checked_div(self.stride).unwrap_or(0)
    }
}

/// Historical name of [`Plane`] — the Motion-JPEG video frame's plane
/// type. Same struct; kept so `MjpegFrame { planes: Vec<MjpegPlane> }`
/// reads as before.
pub type MjpegPlane = Plane;

// ---------------------------------------------------------------------------
// Motion-JPEG video frame
// ---------------------------------------------------------------------------

/// Decoded JPEG / MJPEG **video** frame: planes plus an optional PTS.
///
/// Layout-compatible with `oxideav_core::VideoFrame` (slim shape: a
/// `Vec<MjpegPlane>` plus an optional PTS; dimensions and pixel format
/// are carried by the stream's codec parameters, not the frame). For
/// still images use [`JpegImage`], which carries its own geometry —
/// `MjpegFrame::from(image)` converts.
#[derive(Debug, Clone)]
pub struct MjpegFrame {
    /// Optional presentation timestamp, in the surrounding container's
    /// time base when known.
    pub pts: Option<i64>,
    /// One entry per plane. The number and meaning depends on the
    /// pixel format the frame was decoded into (or is being encoded
    /// from).
    pub planes: Vec<MjpegPlane>,
}

impl From<JpegImage> for MjpegFrame {
    /// Drop the geometry and keep the planes (`pts = None`).
    fn from(img: JpegImage) -> Self {
        MjpegFrame {
            pts: None,
            planes: img.planes,
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel format
// ---------------------------------------------------------------------------

/// Subset of `oxideav_core::PixelFormat` the JPEG decoder/encoder
/// produces or accepts. Variant names mirror the framework enum exactly
/// (the `registry` feature maps them 1:1 by name). Defined here so the
/// standalone build does not need to pull in `oxideav-core`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MjpegPixelFormat {
    /// 8-bit single-channel grayscale.
    Gray8,
    /// 10-bit single-channel grayscale, little-endian (16-bit storage).
    Gray10Le,
    /// 12-bit single-channel grayscale, little-endian (16-bit storage).
    Gray12Le,
    /// 16-bit single-channel grayscale, little-endian. Also the carrier
    /// for every lossless precision without a format of its own
    /// (`P ∈ {2..=7, 9, 11, 13, 14, 15}`): the sample sits in the low
    /// `P` bits of each word — [`JpegImage::precision`] says which.
    Gray16Le,
    /// 8-bit packed CMYK (4 bytes per pixel), plain ink amounts:
    /// `(0, 0, 0, 0)` is white (T.872 §6.1). The decoder has already
    /// undone the Adobe APP14 complement / YCCK transform.
    Cmyk,
    /// 8-bit packed RGB (3 bytes per pixel, R-G-B order). Produced for
    /// RGB-coded three-component frames (Adobe APP14 `transform = 0` or
    /// component identifiers `'R' 'G' 'B'`).
    Rgb24,
    /// 16-bit-per-channel packed RGB, little-endian (6 bytes per pixel,
    /// R-G-B order). Produced by the lossless decoder for
    /// three-component scans at precisions that do not map onto a
    /// `Gbrp*Le` width (P = 2..=7 / 9 / 11 / 13 / 15 / 16). Samples
    /// shorter than 16 bits sit in the low bits of each 16-bit word.
    Rgb48Le,
    /// 10-bit planar GBR (3 planes ordered G, B, R, 16-bit LE storage
    /// per sample). Produced by the lossless decoder for
    /// three-component P = 10 scans.
    Gbrp10Le,
    /// 12-bit planar GBR (3 planes ordered G, B, R, 16-bit LE storage
    /// per sample). Produced by the lossless decoder for
    /// three-component P = 12 scans.
    Gbrp12Le,
    /// 14-bit planar GBR (3 planes ordered G, B, R, 16-bit LE storage
    /// per sample). Produced by the lossless decoder for
    /// three-component P = 14 scans.
    Gbrp14Le,
    /// 8-bit planar 4:1:1 YCbCr (luma 4× chroma horizontally). Full
    /// range like every YCbCr JPEG; no `YuvJ411P` label exists.
    Yuv411P,
    /// 8-bit planar 4:2:0 YCbCr — accepted on encode; the decoder
    /// reports [`YuvJ420P`](Self::YuvJ420P).
    Yuv420P,
    /// 8-bit planar 4:2:2 YCbCr — accepted on encode; the decoder
    /// reports [`YuvJ422P`](Self::YuvJ422P).
    Yuv422P,
    /// 8-bit planar 4:4:4 YCbCr — accepted on encode; the decoder
    /// reports [`YuvJ444P`](Self::YuvJ444P).
    Yuv444P,
    /// 8-bit planar 4:2:0 YCbCr, explicitly full-range ("J" = JPEG
    /// range). Every YCbCr JPEG is full range (T.871 §7; T.872 §6.1
    /// extends the relationship to streams without a JFIF segment), so
    /// the decoder labels every 8-bit 4:2:0 frame this way; the
    /// range-agnostic `Yuv420P` is accepted on input as the same layout.
    YuvJ420P,
    /// 8-bit planar 4:2:2 YCbCr, explicitly full-range (every YCbCr JPEG).
    YuvJ422P,
    /// 8-bit planar 4:4:4 YCbCr, explicitly full-range (every YCbCr JPEG).
    YuvJ444P,
    /// 12-bit planar 4:2:0 YCbCr (16-bit storage per sample, little-endian).
    Yuv420P12Le,
    /// 12-bit planar 4:2:2 YCbCr (16-bit storage per sample, little-endian).
    Yuv422P12Le,
    /// 12-bit planar 4:4:4 YCbCr (16-bit storage per sample, little-endian).
    Yuv444P12Le,
}

/// Contract alias: the crate's native pixel-format tag.
pub type PixelFormat = MjpegPixelFormat;

impl MjpegPixelFormat {
    /// Every variant, in declaration order.
    pub const ALL: [MjpegPixelFormat; 20] = [
        Self::Gray8,
        Self::Gray10Le,
        Self::Gray12Le,
        Self::Gray16Le,
        Self::Cmyk,
        Self::Rgb24,
        Self::Rgb48Le,
        Self::Gbrp10Le,
        Self::Gbrp12Le,
        Self::Gbrp14Le,
        Self::Yuv411P,
        Self::Yuv420P,
        Self::Yuv422P,
        Self::Yuv444P,
        Self::YuvJ420P,
        Self::YuvJ422P,
        Self::YuvJ444P,
        Self::Yuv420P12Le,
        Self::Yuv422P12Le,
        Self::Yuv444P12Le,
    ];

    /// Number of planes a [`JpegImage`] in this format carries.
    pub fn plane_count(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray10Le | Self::Gray12Le | Self::Gray16Le => 1,
            Self::Cmyk | Self::Rgb24 | Self::Rgb48Le => 1,
            _ => 3,
        }
    }

    /// `true` for the single-plane interleaved layouts (`Rgb24`,
    /// `Rgb48Le`, `Cmyk`); grayscale counts as packed too (one plane,
    /// one sample per pixel).
    pub fn is_packed(self) -> bool {
        self.plane_count() == 1
    }

    /// Bytes per sample in storage: 1 for the 8-bit formats, 2 for the
    /// little-endian 16-bit carriers.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Gray8
            | Self::Cmyk
            | Self::Rgb24
            | Self::Yuv411P
            | Self::Yuv420P
            | Self::Yuv422P
            | Self::Yuv444P
            | Self::YuvJ420P
            | Self::YuvJ422P
            | Self::YuvJ444P => 1,
            _ => 2,
        }
    }

    /// Sample precision the format nominally carries (8 / 10 / 12 / 14 /
    /// 16 bits). A lossless frame may hold fewer significant bits than
    /// its carrier (see [`JpegImage::precision`]).
    pub fn nominal_bits(self) -> u8 {
        match self {
            Self::Gray10Le | Self::Gbrp10Le => 10,
            Self::Gray12Le
            | Self::Gbrp12Le
            | Self::Yuv420P12Le
            | Self::Yuv422P12Le
            | Self::Yuv444P12Le => 12,
            Self::Gbrp14Le => 14,
            Self::Gray16Le | Self::Rgb48Le => 16,
            _ => 8,
        }
    }

    /// Number of interleaved channels per pixel in a packed layout
    /// (1 for grayscale, 3 for RGB, 4 for CMYK); `None` for planar.
    pub fn packed_channels(self) -> Option<usize> {
        match self {
            Self::Gray8 | Self::Gray10Le | Self::Gray12Le | Self::Gray16Le => Some(1),
            Self::Rgb24 | Self::Rgb48Le => Some(3),
            Self::Cmyk => Some(4),
            _ => None,
        }
    }

    /// Chroma subsampling divisors `(horizontal, vertical)` of the
    /// planar YCbCr layouts; `(1, 1)` for everything else.
    pub fn chroma_divisors(self) -> (usize, usize) {
        match self {
            Self::Yuv420P | Self::YuvJ420P | Self::Yuv420P12Le => (2, 2),
            Self::Yuv422P | Self::YuvJ422P | Self::Yuv422P12Le => (2, 1),
            Self::Yuv411P => (4, 1),
            _ => (1, 1),
        }
    }

    /// `true` for the planar YCbCr layouts (any range, any depth).
    pub fn is_yuv(self) -> bool {
        matches!(
            self,
            Self::Yuv411P
                | Self::Yuv420P
                | Self::Yuv422P
                | Self::Yuv444P
                | Self::YuvJ420P
                | Self::YuvJ422P
                | Self::YuvJ444P
                | Self::Yuv420P12Le
                | Self::Yuv422P12Le
                | Self::Yuv444P12Le
        )
    }

    /// `true` for the single-channel layouts.
    pub fn is_gray(self) -> bool {
        matches!(
            self,
            Self::Gray8 | Self::Gray10Le | Self::Gray12Le | Self::Gray16Le
        )
    }

    /// `true` for the RGB-class layouts (packed RGB and planar GBR).
    pub fn is_rgb(self) -> bool {
        matches!(
            self,
            Self::Rgb24 | Self::Rgb48Le | Self::Gbrp10Le | Self::Gbrp12Le | Self::Gbrp14Le
        )
    }

    /// `true` for the explicitly full-range `YuvJ*` labels.
    pub fn is_full_range_label(self) -> bool {
        matches!(self, Self::YuvJ420P | Self::YuvJ422P | Self::YuvJ444P)
    }

    /// The `YuvJ*` label of an 8-bit planar YCbCr layout when one
    /// exists (`Yuv420P → YuvJ420P`, …); other formats unchanged.
    pub fn full_range_label(self) -> Self {
        match self {
            Self::Yuv420P => Self::YuvJ420P,
            Self::Yuv422P => Self::YuvJ422P,
            Self::Yuv444P => Self::YuvJ444P,
            other => other,
        }
    }

    /// The range-agnostic label of a `YuvJ*` layout (`YuvJ420P →
    /// Yuv420P`, …); other formats unchanged.
    pub fn range_agnostic_label(self) -> Self {
        match self {
            Self::YuvJ420P => Self::Yuv420P,
            Self::YuvJ422P => Self::Yuv422P,
            Self::YuvJ444P => Self::Yuv444P,
            other => other,
        }
    }

    /// Sample dimensions `(samples per row, rows)` of plane `index` for
    /// a `width × height` picture in this format. Chroma planes of the
    /// subsampled layouts are `ceil(width / h) × ceil(height / v)`
    /// (T.81 §A.1.1). Packed layouts have one plane of `width` pixels
    /// per row.
    pub fn plane_dimensions(self, width: u32, height: u32, index: usize) -> (usize, usize) {
        let (w, h) = (width as usize, height as usize);
        if index == 0 || !self.is_yuv() {
            return (w, h);
        }
        let (dh, dv) = self.chroma_divisors();
        (w.div_ceil(dh), h.div_ceil(dv))
    }

    /// Bytes per row of plane `index` when tightly packed.
    pub fn tight_stride(self, width: u32, height: u32, index: usize) -> usize {
        let (w, _) = self.plane_dimensions(width, height, index);
        w * self.packed_channels().unwrap_or(1) * self.bytes_per_sample()
    }

    /// Short lower-case name mirroring the framework's spelling
    /// (`"yuv420p"`, `"gray8"`, `"rgb24"`, …).
    pub fn name(self) -> &'static str {
        match self {
            Self::Gray8 => "gray8",
            Self::Gray10Le => "gray10le",
            Self::Gray12Le => "gray12le",
            Self::Gray16Le => "gray16le",
            Self::Cmyk => "cmyk",
            Self::Rgb24 => "rgb24",
            Self::Rgb48Le => "rgb48le",
            Self::Gbrp10Le => "gbrp10le",
            Self::Gbrp12Le => "gbrp12le",
            Self::Gbrp14Le => "gbrp14le",
            Self::Yuv411P => "yuv411p",
            Self::Yuv420P => "yuv420p",
            Self::Yuv422P => "yuv422p",
            Self::Yuv444P => "yuv444p",
            Self::YuvJ420P => "yuvj420p",
            Self::YuvJ422P => "yuvj422p",
            Self::YuvJ444P => "yuvj444p",
            Self::Yuv420P12Le => "yuv420p12le",
            Self::Yuv422P12Le => "yuv422p12le",
            Self::Yuv444P12Le => "yuv444p12le",
        }
    }
}

impl core::fmt::Display for MjpegPixelFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Colour description
// ---------------------------------------------------------------------------

/// Nominal sample range (H.273 `VideoFullRangeFlag`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ColorRange {
    /// No range was signalled.
    #[default]
    Unspecified,
    /// Limited (video / studio) range: `VideoFullRangeFlag == 0`.
    Limited,
    /// Full (PC / JPEG) range: `VideoFullRangeFlag == 1`.
    Full,
}

/// Colour description: sample range plus the H.273 `ColourPrimaries` /
/// `TransferCharacteristics` / `MatrixCoefficients` code points (raw
/// `u8` values; `2` means "unspecified" for each).
///
/// JPEG carries no colour-signalling fields of its own; the decoder
/// fills this from the JFIF / Adobe APP14 conventions (see
/// [`ColorInfo::jfif_ycbcr`], [`ColorInfo::srgb`]). An embedded ICC
/// profile ([`Metadata::icc`]) takes precedence over these code points
/// for colour-managed consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ColorInfo {
    /// Nominal sample range.
    pub range: ColorRange,
    /// H.273 `ColourPrimaries` code point (`1` = BT.709 / sRGB, `2` =
    /// unspecified, `5` = BT.470 BG / BT.601 625, …).
    pub primaries: u8,
    /// H.273 `TransferCharacteristics` code point (`13` = sRGB /
    /// IEC 61966-2-1, `6` = BT.601, `2` = unspecified, …).
    pub transfer: u8,
    /// H.273 `MatrixCoefficients` code point (`0` = identity / RGB, `5`
    /// = BT.601 625 / sYCC, `2` = unspecified, …).
    pub matrix: u8,
}

impl ColorInfo {
    /// H.273 "unspecified" code point.
    pub const UNSPECIFIED: u8 = 2;

    /// Build from a range and three H.273 code points.
    pub const fn new(range: ColorRange, primaries: u8, transfer: u8, matrix: u8) -> Self {
        Self {
            range,
            primaries,
            transfer,
            matrix,
        }
    }

    /// Every field unspecified — identical to `Default`.
    pub const fn unspecified() -> Self {
        Self::new(
            ColorRange::Unspecified,
            Self::UNSPECIFIED,
            Self::UNSPECIFIED,
            Self::UNSPECIFIED,
        )
    }

    /// The JFIF / T.871 YCbCr convention as H.273 code points: full
    /// range, BT.601 matrix (`KR = 0.299`, `KB = 0.114`; H.273 Table 4
    /// value 5, which also names IEC 61966-2-1 sYCC), sRGB primaries
    /// (1) and transfer (13). T.871 §7 derives its Y/CB/CR from the
    /// 625-line BT.601 signals but its NOTE 3 records that common
    /// practice follows sYCC with negligible difference; this crate
    /// reports the sYCC code points so consumers treat JPEG RGB as sRGB
    /// unless an ICC profile says otherwise.
    pub const fn jfif_ycbcr() -> Self {
        Self::new(ColorRange::Full, 1, 13, 5)
    }

    /// sRGB (IEC 61966-2-1): BT.709 primaries (1), sRGB transfer (13),
    /// identity matrix (0), full range. The description of RGB-coded
    /// (Adobe APP14 `transform = 0`) frames.
    pub const fn srgb() -> Self {
        Self::new(ColorRange::Full, 1, 13, 0)
    }

    /// Full-range grayscale: sRGB primaries / transfer, matrix
    /// unspecified (a single channel has no matrix).
    pub const fn gray() -> Self {
        Self::new(ColorRange::Full, 1, 13, Self::UNSPECIFIED)
    }

    /// Plain CMYK: full range, every H.273 code point unspecified —
    /// T.872 §3.1 leaves the ink values device dependent.
    pub const fn cmyk() -> Self {
        Self::new(
            ColorRange::Full,
            Self::UNSPECIFIED,
            Self::UNSPECIFIED,
            Self::UNSPECIFIED,
        )
    }

    /// `true` when the range is [`ColorRange::Full`].
    pub const fn is_full_range(&self) -> bool {
        matches!(self.range, ColorRange::Full)
    }
}

impl Default for ColorInfo {
    fn default() -> Self {
        Self::unspecified()
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Embedded metadata blobs, as found in (or to be written into) the
/// APPn marker segments.
///
/// * `icc` — the ICC profile reassembled from the `APP2 "ICC_PROFILE\0"`
///   chunk sequence (ICC.1 Annex B.4), in chunk order.
/// * `exif` — the `APP1 "Exif\0\0"` payload **after** the six-byte
///   identifier: a TIFF header (`II*\0` / `MM\0*`) followed by the IFDs.
/// * `xmp` — the `APP1 "http://ns.adobe.com/xap/1.0/\0"` payload after
///   the identifier: the XMP packet bytes (UTF-8 XML).
/// * `gamma` — never set for JPEG (no gAMA-like field); present so the
///   shape matches the other image crates.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Metadata {
    /// ICC profile bytes.
    pub icc: Option<Vec<u8>>,
    /// Exif TIFF structure (without the `Exif\0\0` identifier).
    pub exif: Option<Vec<u8>>,
    /// XMP packet (without the namespace identifier).
    pub xmp: Option<Vec<u8>>,
    /// Display gamma; always `None` for JPEG.
    pub gamma: Option<f32>,
}

impl Metadata {
    /// No metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ICC profile to embed / report.
    pub fn with_icc(mut self, icc: Vec<u8>) -> Self {
        self.icc = Some(icc);
        self
    }

    /// Set the Exif TIFF structure (without `Exif\0\0`).
    pub fn with_exif(mut self, exif: Vec<u8>) -> Self {
        self.exif = Some(exif);
        self
    }

    /// Set the XMP packet.
    pub fn with_xmp(mut self, xmp: Vec<u8>) -> Self {
        self.xmp = Some(xmp);
        self
    }

    /// `true` when no field is set.
    pub fn is_empty(&self) -> bool {
        self.icc.is_none() && self.exif.is_none() && self.xmp.is_none() && self.gamma.is_none()
    }
}

// ---------------------------------------------------------------------------
// The image
// ---------------------------------------------------------------------------

/// One decoded (or to-be-encoded) JPEG picture in its native layout.
///
/// `planes` holds exactly one plane for the packed layouts (`Gray*`,
/// `Rgb24`, `Rgb48Le`, `Cmyk`) and three for the planar YCbCr / GBR
/// layouts, each `stride` bytes per row with the sample geometry of
/// [`MjpegPixelFormat::plane_dimensions`]. Multi-byte samples are
/// little-endian. [`JpegImage::to_rgb8`] / [`JpegImage::to_rgba8`]
/// convert any layout to tightly packed 8-bit RGB(A) with the exact
/// T.871 kernels (see [`crate::convert`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct JpegImage {
    /// Picture width in pixels (T.81 `X`, `1..=65535`).
    pub width: u32,
    /// Picture height in pixels (T.81 `Y` after DNL resolution).
    pub height: u32,
    /// Native pixel layout of `planes`.
    pub format: MjpegPixelFormat,
    /// The sample planes (one for packed layouts, three for planar).
    pub planes: Vec<Plane>,
    /// Colour description inferred from the JFIF / Adobe conventions.
    pub color: ColorInfo,
    /// ICC / Exif / XMP blobs found in the APPn segments.
    pub metadata: Metadata,
    /// Sample precision `P` (T.81 frame header), `2..=16`. Equals
    /// [`MjpegPixelFormat::nominal_bits`] for every format except the
    /// 16-bit carriers of odd lossless precisions (`Gray16Le` /
    /// `Rgb48Le` holding `P < 16` samples in their low bits).
    pub precision: u8,
}

impl JpegImage {
    /// Build an image from its geometry and planes. `color` is
    /// [`ColorInfo::unspecified`], `metadata` empty, `precision` the
    /// format's nominal depth.
    ///
    /// Rejects with [`MjpegError::InvalidData`] a zero dimension or one
    /// past T.81's 65535, a plane count other than
    /// [`MjpegPixelFormat::plane_count`], a plane whose stride is
    /// shorter than its tight row
    /// ([`MjpegPixelFormat::tight_stride`]), or a plane buffer shorter
    /// than the rows it must hold (`(rows − 1) × stride + tight row`;
    /// the last row may be unpadded) — so an image that exists is
    /// always consistent and [`Self::to_rgb8`] / [`Self::to_rgba8`]
    /// never need to fail.
    pub fn new(
        width: u32,
        height: u32,
        format: MjpegPixelFormat,
        planes: Vec<Plane>,
    ) -> Result<Self> {
        if width == 0 || height == 0 || width > 65535 || height > 65535 {
            return Err(MjpegError::invalid(format!(
                "JPEG image: {width}×{height} is outside T.81's 1..=65535 per axis"
            )));
        }
        if planes.len() != format.plane_count() {
            return Err(MjpegError::invalid(format!(
                "JPEG image: {} plane(s) do not fit {format} ({} expected)",
                planes.len(),
                format.plane_count()
            )));
        }
        for (i, p) in planes.iter().enumerate() {
            let (_, rows) = format.plane_dimensions(width, height, i);
            let row = format.tight_stride(width, height, i);
            if p.stride < row {
                return Err(MjpegError::invalid(format!(
                    "JPEG image: plane {i} stride {} is shorter than its {row}-byte row",
                    p.stride
                )));
            }
            let needed = (rows - 1)
                .checked_mul(p.stride)
                .and_then(|v| v.checked_add(row))
                .ok_or_else(|| MjpegError::invalid("JPEG image: plane size overflows usize"))?;
            if p.data.len() < needed {
                return Err(MjpegError::invalid(format!(
                    "JPEG image: plane {i} holds {} bytes but {rows} rows at stride {} need {needed}",
                    p.data.len(),
                    p.stride
                )));
            }
        }
        Ok(Self::new_unchecked(width, height, format, planes))
    }

    /// [`Self::new`] without the geometry checks, for images the crate
    /// assembles itself from already-validated buffers.
    pub(crate) fn new_unchecked(
        width: u32,
        height: u32,
        format: MjpegPixelFormat,
        planes: Vec<Plane>,
    ) -> Self {
        Self {
            width,
            height,
            format,
            planes,
            color: ColorInfo::unspecified(),
            metadata: Metadata::new(),
            precision: format.nominal_bits(),
        }
    }

    /// Wrap tightly packed 8-bit RGB (`3 × width` bytes per row) as an
    /// `Rgb24` image with the sRGB colour description;
    /// [`MjpegError::InvalidData`] when `data` is shorter than `3 ×
    /// width × height` (or the geometry is outside T.81's range).
    pub fn from_rgb8(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let stride = width as usize * 3;
        Ok(Self::new(
            width,
            height,
            MjpegPixelFormat::Rgb24,
            vec![Plane::new(stride, data)],
        )?
        .with_color(ColorInfo::srgb()))
    }

    /// Wrap tightly packed 8-bit RGBA as an `Rgb24` image, **dropping
    /// the alpha channel** — JPEG has no alpha mechanism, so the three
    /// colour bytes of every pixel are kept and the fourth discarded.
    /// The result carries the sRGB colour description;
    /// [`MjpegError::InvalidData`] when `data` is shorter than `4 ×
    /// width × height`.
    pub fn from_rgba8(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let needed = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| MjpegError::invalid("JPEG image: pixel count overflows usize"))?;
        if data.len() < needed {
            return Err(MjpegError::invalid(format!(
                "JPEG image: {} RGBA bytes supplied, {width}×{height} needs {needed}",
                data.len()
            )));
        }
        let rgb: Vec<u8> = data[..needed]
            .chunks_exact(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect();
        Self::from_rgb8(width, height, rgb)
    }

    /// Replace the colour description.
    pub fn with_color(mut self, color: ColorInfo) -> Self {
        self.color = color;
        self
    }

    /// Replace the metadata.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Override the sample precision (`2..=16`) carried by the planes
    /// — needed only when a 16-bit carrier holds fewer significant
    /// bits.
    pub fn with_precision(mut self, precision: u8) -> Self {
        self.precision = precision;
        self
    }

    /// Picture width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Picture height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Native pixel layout.
    pub fn format(&self) -> MjpegPixelFormat {
        self.format
    }

    /// The single plane's bytes for packed layouts (`Gray*`, `Rgb24`,
    /// `Rgb48Le`, `Cmyk`); `None` for the planar layouts — use
    /// [`into_raw`](Self::into_raw) or index `planes` directly.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        if self.format.is_packed() && self.planes.len() == 1 {
            Some(&self.planes[0].data)
        } else {
            None
        }
    }

    /// Consume the image and return its plane bytes: the single plane
    /// for packed layouts; the planes concatenated in order (each at
    /// its own stride, as reported in `planes[i].stride`) for planar
    /// layouts.
    pub fn into_raw(self) -> Vec<u8> {
        let mut planes = self.planes.into_iter();
        let Some(first) = planes.next() else {
            return Vec::new();
        };
        let mut out = first.data;
        for p in planes {
            out.extend_from_slice(&p.data);
        }
        out
    }

    /// Convert to tightly packed 8-bit RGB (`3 × width` bytes per row),
    /// whatever the native layout — see [`crate::convert`] for the
    /// kernels (T.871 §7 YCbCr→RGB, nearest-neighbour chroma
    /// upsampling, CMYK ink inversion, deep samples rescaled to 8 bits).
    pub fn to_rgb8(&self) -> Vec<u8> {
        crate::convert::to_rgb8(self)
    }

    /// Convert to tightly packed 8-bit RGBA with opaque alpha (`255`).
    pub fn to_rgba8(&self) -> Vec<u8> {
        crate::convert::to_rgba8(self)
    }

    /// Rebuild an image from a Motion-JPEG video frame plus the
    /// geometry the stream's parameters carry. Fails when the plane
    /// count or geometry does not match `format` (see [`Self::new`]).
    pub fn from_frame(
        frame: MjpegFrame,
        width: u32,
        height: u32,
        format: MjpegPixelFormat,
    ) -> Result<Self> {
        if frame.planes.len() != format.plane_count() {
            return Err(MjpegError::invalid(format!(
                "JPEG image: {} plane(s) do not fit {format} ({} expected)",
                frame.planes.len(),
                format.plane_count()
            )));
        }
        Self::new(width, height, format, frame.planes)
    }
}

// ---------------------------------------------------------------------------
// Raw RGB / RGBA results
// ---------------------------------------------------------------------------

/// Tightly packed 8-bit RGB: `3 × width` bytes per row, row-major, no
/// padding. Identical definition in every OxideAV image crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RgbImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width × height × 3` bytes, R G B per pixel.
    pub data: Vec<u8>,
}

impl RgbImage {
    /// Wrap a buffer (expected `width × height × 3` bytes).
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    /// The pixel bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume into the pixel buffer.
    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }
}

/// Tightly packed 8-bit RGBA: `4 × width` bytes per row, row-major, no
/// padding. Identical definition in every OxideAV image crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RgbaImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width × height × 4` bytes, R G B A per pixel.
    pub data: Vec<u8>,
}

impl RgbaImage {
    /// Wrap a buffer (expected `width × height × 4` bytes).
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    /// The pixel bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume into the pixel buffer.
    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }
}

// ---------------------------------------------------------------------------
// Header summary
// ---------------------------------------------------------------------------

/// What [`crate::info`] reports without running the entropy decoder:
/// the frame header plus the presence of the metadata segments.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageInfo {
    /// Picture width (T.81 `X`).
    pub width: u32,
    /// Picture height (T.81 `Y`, or the DNL-supplied line count when
    /// the frame header codes `Y = 0`).
    pub height: u32,
    /// The layout [`crate::decode`] will produce for this stream.
    pub format: MjpegPixelFormat,
    /// Number of pictures; always `1` for JPEG.
    pub frames: u32,
    /// Always `false`: JPEG carries no alpha.
    pub has_alpha: bool,
    /// Colour description as the decoder will report it.
    pub color: ColorInfo,
    /// An `APP2 "ICC_PROFILE\0"` chunk sequence is present.
    pub has_icc: bool,
    /// An `APP1 "Exif\0\0"` segment is present.
    pub has_exif: bool,
    /// An `APP1 "http://ns.adobe.com/xap/1.0/\0"` segment is present.
    pub has_xmp: bool,
    /// Sample precision `P` from the frame header (`2..=16`).
    pub precision: u8,
    /// Number of components `Nf` in the frame header (1, 3 or 4).
    pub components: u8,
    /// Progressive DCT frame (SOF2 / SOF6 / SOF10 / SOF14).
    pub progressive: bool,
    /// Lossless (spatial-predictive) frame (SOF3 / SOF7 / SOF11 / SOF15).
    pub lossless: bool,
    /// Arithmetic entropy coding (SOF9..SOF15).
    pub arithmetic: bool,
    /// Hierarchical sequence (a DHP segment precedes the first frame).
    pub hierarchical: bool,
    /// A JFIF APP0 segment is present.
    pub has_jfif: bool,
    /// An Adobe APP14 segment is present.
    pub has_adobe: bool,
}

// ---------------------------------------------------------------------------
// Decode options
// ---------------------------------------------------------------------------

/// Limits and strictness for [`crate::decode_with`].
///
/// Limits are checked against the frame header **before** any sample
/// buffer is allocated; a breach is [`MjpegError::LimitExceeded`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodeOptions {
    /// Largest accepted width; `None` = unlimited (default `65535`, the
    /// T.81 maximum).
    pub max_width: Option<u32>,
    /// Largest accepted height; `None` = unlimited (default `65535`).
    pub max_height: Option<u32>,
    /// Largest accepted `width × height`; `None` = unlimited (default
    /// `1 << 28`, 268 Mpixel — about 1 GiB of decoded 4:4:4 samples).
    pub max_pixels: Option<u64>,
    /// Largest accepted input length in bytes; `None` = unlimited (the
    /// default).
    pub max_bytes: Option<u64>,
    /// Strict mode: reject trailing bytes after `EOI`, and treat a
    /// malformed JFIF / Adobe / ICC metadata segment as an error
    /// instead of ignoring it. Default `false`.
    pub strict: bool,
    /// A T.81 §B.5 abbreviated table-specification stream (`SOI`, DQT /
    /// DHT / DAC / DRI …, `EOI`) preloaded ahead of the image — the TIFF
    /// `JPEGTables` carriage. `None` (default) decodes interchange
    /// streams only.
    pub tables: Option<Vec<u8>>,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            max_width: Some(65535),
            max_height: Some(65535),
            max_pixels: Some(1 << 28),
            max_bytes: None,
            strict: false,
            tables: None,
        }
    }
}

impl DecodeOptions {
    /// The defaults (see the field docs).
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap the accepted width (`None` = unlimited).
    pub fn with_max_width(mut self, max_width: Option<u32>) -> Self {
        self.max_width = max_width;
        self
    }

    /// Cap the accepted height (`None` = unlimited).
    pub fn with_max_height(mut self, max_height: Option<u32>) -> Self {
        self.max_height = max_height;
        self
    }

    /// Cap the accepted pixel count (`None` = unlimited).
    pub fn with_max_pixels(mut self, max_pixels: Option<u64>) -> Self {
        self.max_pixels = max_pixels;
        self
    }

    /// Cap the accepted input length (`None` = unlimited).
    pub fn with_max_bytes(mut self, max_bytes: Option<u64>) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Enable / disable strict mode.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Preload a §B.5 tables-only stream (TIFF `JPEGTables`).
    pub fn with_tables(mut self, tables: Vec<u8>) -> Self {
        self.tables = Some(tables);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_dimensions_follow_a_1_1() {
        assert_eq!(
            MjpegPixelFormat::Yuv420P.plane_dimensions(33, 17, 1),
            (17, 9)
        );
        assert_eq!(
            MjpegPixelFormat::Yuv422P.plane_dimensions(33, 17, 2),
            (17, 17)
        );
        assert_eq!(
            MjpegPixelFormat::Yuv411P.plane_dimensions(33, 17, 1),
            (9, 17)
        );
        assert_eq!(
            MjpegPixelFormat::Yuv444P12Le.plane_dimensions(33, 17, 1),
            (33, 17)
        );
        assert_eq!(
            MjpegPixelFormat::Rgb24.plane_dimensions(33, 17, 0),
            (33, 17)
        );
        assert_eq!(MjpegPixelFormat::Rgb48Le.tight_stride(33, 17, 0), 33 * 6);
        assert_eq!(MjpegPixelFormat::Gbrp12Le.tight_stride(33, 17, 1), 66);
    }

    #[test]
    fn full_range_labels_round_trip() {
        for f in MjpegPixelFormat::ALL {
            let j = f.full_range_label();
            assert_eq!(j.range_agnostic_label().full_range_label(), j);
            assert_eq!(j.plane_count(), f.plane_count());
            assert_eq!(j.chroma_divisors(), f.chroma_divisors());
            assert_eq!(f.name().parse::<String>().unwrap(), f.to_string());
        }
        assert_eq!(
            MjpegPixelFormat::Yuv420P.full_range_label(),
            MjpegPixelFormat::YuvJ420P
        );
        assert_eq!(
            MjpegPixelFormat::Yuv411P.full_range_label(),
            MjpegPixelFormat::Yuv411P
        );
    }

    #[test]
    fn from_rgba8_drops_alpha() {
        let img = JpegImage::from_rgba8(2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        assert_eq!(img.format, MjpegPixelFormat::Rgb24);
        assert_eq!(img.as_bytes(), Some(&[1u8, 2, 3, 5, 6, 7][..]));
        assert_eq!(img.planes[0].stride, 6);
        assert_eq!(img.color, ColorInfo::srgb());
        assert_eq!(img.precision, 8);
    }

    #[test]
    fn constructors_reject_bad_geometry() {
        let bad = |r: Result<JpegImage>| assert!(matches!(r, Err(MjpegError::InvalidData(_))));
        bad(JpegImage::from_rgb8(2, 1, vec![0; 5]));
        bad(JpegImage::from_rgba8(1, 2, vec![0; 7]));
        bad(JpegImage::from_rgb8(0, 1, vec![]));
        bad(JpegImage::from_rgb8(65536, 1, vec![0; 65536 * 3]));
        bad(JpegImage::new(
            4,
            4,
            MjpegPixelFormat::Yuv420P,
            vec![Plane::new(4, vec![0; 16])],
        ));
        bad(JpegImage::new(
            4,
            4,
            MjpegPixelFormat::Gray8,
            vec![Plane::new(3, vec![0; 16])],
        ));
        bad(JpegImage::new(
            4,
            4,
            MjpegPixelFormat::Gray8,
            vec![Plane::new(4, vec![0; 15])],
        ));
        // The last row may be unpadded; chroma planes use their own geometry.
        assert!(JpegImage::new(
            3,
            2,
            MjpegPixelFormat::Yuv420P,
            vec![
                Plane::new(8, vec![0; 11]),
                Plane::new(2, vec![0; 2]),
                Plane::new(2, vec![0; 2]),
            ],
        )
        .is_ok());
        assert!(JpegImage::from_rgba8(1, 1, vec![0; 4]).is_ok());
    }

    #[test]
    fn into_raw_concatenates_planes() {
        let img = JpegImage::new(
            2,
            2,
            MjpegPixelFormat::Yuv420P,
            vec![
                Plane::new(2, vec![1, 2, 3, 4]),
                Plane::new(1, vec![5]),
                Plane::new(1, vec![6]),
            ],
        )
        .unwrap();
        assert!(img.as_bytes().is_none());
        assert_eq!(img.clone().into_raw(), vec![1, 2, 3, 4, 5, 6]);
        let frame = MjpegFrame::from(img.clone());
        assert_eq!(frame.planes.len(), 3);
        assert!(frame.pts.is_none());
        let back = JpegImage::from_frame(frame, 2, 2, MjpegPixelFormat::Yuv420P).unwrap();
        assert_eq!(back.planes, img.planes);
        assert!(JpegImage::from_frame(
            MjpegFrame {
                pts: None,
                planes: vec![]
            },
            2,
            2,
            MjpegPixelFormat::Gray8
        )
        .is_err());
    }

    #[test]
    fn decode_options_defaults_and_builders() {
        let d = DecodeOptions::default();
        assert_eq!(d.max_width, Some(65535));
        assert_eq!(d.max_pixels, Some(1 << 28));
        assert_eq!(d.max_bytes, None);
        assert!(!d.strict);
        assert!(d.tables.is_none());
        let o = DecodeOptions::new()
            .with_max_width(Some(10))
            .with_max_height(None)
            .with_max_pixels(Some(12))
            .with_max_bytes(Some(13))
            .with_strict(true)
            .with_tables(vec![0xFF, 0xD8, 0xFF, 0xD9]);
        assert_eq!(
            (
                o.max_width,
                o.max_height,
                o.max_pixels,
                o.max_bytes,
                o.strict
            ),
            (Some(10), None, Some(12), Some(13), true)
        );
        assert_eq!(o.tables.as_deref(), Some(&[0xFF, 0xD8, 0xFF, 0xD9][..]));
    }

    #[test]
    fn color_presets() {
        assert_eq!(ColorInfo::default(), ColorInfo::unspecified());
        assert!(ColorInfo::jfif_ycbcr().is_full_range());
        assert_eq!(ColorInfo::jfif_ycbcr().matrix, 5);
        assert_eq!(ColorInfo::srgb().matrix, 0);
        assert_eq!(ColorInfo::cmyk().primaries, ColorInfo::UNSPECIFIED);
        assert!(Metadata::new().is_empty());
        assert!(!Metadata::new().with_icc(vec![0]).is_empty());
    }
}
