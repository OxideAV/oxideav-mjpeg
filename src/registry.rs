//! `oxideav-core` integration layer for `oxideav-mjpeg`.
//!
//! Gated behind the default-on `registry` feature so image-library
//! consumers can depend on `oxideav-mjpeg` with `default-features = false`
//! and skip the `oxideav-core` dependency entirely.
//!
//! The module exposes:
//! * [`register`] / [`register_codecs`] / [`register_containers`] — the
//!   `CodecRegistry` / `ContainerRegistry` entry points the umbrella
//!   `oxideav` crate calls during framework initialisation.
//! * The [`MjpegEncoder`] struct that implements the framework
//!   `Encoder` trait, plus the corresponding `MjpegDecoder` that
//!   implements the `Decoder` trait. Both wrap the framework-free
//!   [`crate::decoder::decode_jpeg`] / `encode_jpeg_*` entry points
//!   defined in [`crate::encoder`].
//! * The `From<MjpegError> for oxideav_core::Error` and
//!   `From<MjpegFrame> for oxideav_core::VideoFrame` /
//!   `From<MjpegPixelFormat> for oxideav_core::PixelFormat`
//!   conversions used by the trait impls below.

use std::collections::VecDeque;

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    parse_options, CodecCapabilities, CodecId, CodecInfo, CodecOptionsStruct, CodecParameters,
    CodecRegistry, CodecTag, ContainerRegistry, Decoder, Encoder, Error, Frame, MediaType,
    OptionField, OptionKind, OptionValue, Packet, PixelFormat, Result, RuntimeContext, TimeBase,
    VideoFrame,
};

use crate::container;
use crate::encoder::{
    encode_arith_jpeg_grayscale, encode_arith_jpeg_rgb24, encode_arith_jpeg_yuv, encode_jpeg_cmyk,
    encode_jpeg_cmyk_progressive, encode_jpeg_grayscale_with_opts, encode_jpeg_progressive,
    encode_jpeg_progressive_grayscale, encode_jpeg_rgb24_with_opts, encode_jpeg_with_opts,
    encode_lossless_jpeg_grayscale, DEFAULT_QUALITY,
};
use crate::error::MjpegError;
use crate::image::{ColorInfo, ColorRange, JpegImage, MjpegFrame, MjpegPixelFormat, MjpegPlane};
use crate::jpeg::inspect::ChromaSubsampling;
use crate::mjpeg_container;
use crate::t81::{ColorSignalling, EncodeOptions, HuffmanTables, JpegProcess, JpegTableSet};
use crate::CODEC_ID_STR;

// ---- Error / pixel-format / frame conversions --------------------------

impl From<MjpegError> for Error {
    fn from(e: MjpegError) -> Self {
        match e {
            MjpegError::InvalidData(s) => Error::InvalidData(s),
            MjpegError::Unsupported(s) => Error::Unsupported(s),
            MjpegError::LimitExceeded(s) => Error::InvalidData(format!("limit exceeded: {s}")),
            MjpegError::Io(e) => Error::Io(e),
            MjpegError::Other(s) => Error::Other(s),
            MjpegError::Eof => Error::Eof,
            MjpegError::NeedMore => Error::NeedMore,
        }
    }
}

impl From<MjpegPixelFormat> for PixelFormat {
    fn from(p: MjpegPixelFormat) -> Self {
        match p {
            MjpegPixelFormat::Gray8 => PixelFormat::Gray8,
            MjpegPixelFormat::Gray10Le => PixelFormat::Gray10Le,
            MjpegPixelFormat::Gray12Le => PixelFormat::Gray12Le,
            MjpegPixelFormat::Gray16Le => PixelFormat::Gray16Le,
            MjpegPixelFormat::Cmyk => PixelFormat::Cmyk,
            MjpegPixelFormat::Rgb24 => PixelFormat::Rgb24,
            MjpegPixelFormat::Rgb48Le => PixelFormat::Rgb48Le,
            MjpegPixelFormat::Gbrp10Le => PixelFormat::Gbrp10Le,
            MjpegPixelFormat::Gbrp12Le => PixelFormat::Gbrp12Le,
            MjpegPixelFormat::Gbrp14Le => PixelFormat::Gbrp14Le,
            MjpegPixelFormat::Yuv411P => PixelFormat::Yuv411P,
            MjpegPixelFormat::Yuv420P => PixelFormat::Yuv420P,
            MjpegPixelFormat::Yuv422P => PixelFormat::Yuv422P,
            MjpegPixelFormat::Yuv444P => PixelFormat::Yuv444P,
            MjpegPixelFormat::YuvJ420P => PixelFormat::YuvJ420P,
            MjpegPixelFormat::YuvJ422P => PixelFormat::YuvJ422P,
            MjpegPixelFormat::YuvJ444P => PixelFormat::YuvJ444P,
            MjpegPixelFormat::Yuv420P12Le => PixelFormat::Yuv420P12Le,
            MjpegPixelFormat::Yuv422P12Le => PixelFormat::Yuv422P12Le,
            MjpegPixelFormat::Yuv444P12Le => PixelFormat::Yuv444P12Le,
        }
    }
}

impl TryFrom<PixelFormat> for MjpegPixelFormat {
    type Error = Error;

    /// Inverse of [`From<MjpegPixelFormat> for PixelFormat`]:
    /// `Error::Unsupported` for any framework pixel format the JPEG
    /// codec neither produces nor accepts.
    fn try_from(p: PixelFormat) -> Result<Self> {
        pix_to_local(p).ok_or_else(|| {
            Error::unsupported(format!("JPEG: pixel format {p:?} is not a JPEG layout"))
        })
    }
}

/// Inverse of [`From<MjpegPixelFormat> for PixelFormat`]. Returns
/// `None` for any pixel format the JPEG codec does not produce or
/// accept (so the encoder can reject unsupported `CodecParameters`
/// up-front rather than failing inside `encode_jpeg_*`).
fn pix_to_local(p: PixelFormat) -> Option<MjpegPixelFormat> {
    Some(match p {
        PixelFormat::Gray8 => MjpegPixelFormat::Gray8,
        PixelFormat::Gray10Le => MjpegPixelFormat::Gray10Le,
        PixelFormat::Gray12Le => MjpegPixelFormat::Gray12Le,
        PixelFormat::Gray16Le => MjpegPixelFormat::Gray16Le,
        PixelFormat::Cmyk => MjpegPixelFormat::Cmyk,
        PixelFormat::Rgb24 => MjpegPixelFormat::Rgb24,
        PixelFormat::Rgb48Le => MjpegPixelFormat::Rgb48Le,
        PixelFormat::Gbrp10Le => MjpegPixelFormat::Gbrp10Le,
        PixelFormat::Gbrp12Le => MjpegPixelFormat::Gbrp12Le,
        PixelFormat::Gbrp14Le => MjpegPixelFormat::Gbrp14Le,
        PixelFormat::Yuv411P => MjpegPixelFormat::Yuv411P,
        PixelFormat::Yuv420P => MjpegPixelFormat::Yuv420P,
        PixelFormat::Yuv422P => MjpegPixelFormat::Yuv422P,
        PixelFormat::Yuv444P => MjpegPixelFormat::Yuv444P,
        PixelFormat::YuvJ420P => MjpegPixelFormat::YuvJ420P,
        PixelFormat::YuvJ422P => MjpegPixelFormat::YuvJ422P,
        PixelFormat::YuvJ444P => MjpegPixelFormat::YuvJ444P,
        PixelFormat::Yuv420P12Le => MjpegPixelFormat::Yuv420P12Le,
        PixelFormat::Yuv422P12Le => MjpegPixelFormat::Yuv422P12Le,
        PixelFormat::Yuv444P12Le => MjpegPixelFormat::Yuv444P12Le,
        _ => return None,
    })
}

impl From<MjpegFrame> for VideoFrame {
    fn from(f: MjpegFrame) -> Self {
        VideoFrame {
            pts: f.pts,
            planes: f
                .planes
                .into_iter()
                .map(|p| VideoPlane {
                    stride: p.stride,
                    data: p.data,
                })
                .collect(),
        }
    }
}

impl From<MjpegPlane> for VideoPlane {
    fn from(p: MjpegPlane) -> Self {
        VideoPlane {
            stride: p.stride,
            data: p.data,
        }
    }
}

impl From<VideoPlane> for MjpegPlane {
    fn from(p: VideoPlane) -> Self {
        MjpegPlane::new(p.stride, p.data)
    }
}

impl From<JpegImage> for VideoFrame {
    /// The planes, verbatim, `pts = None`. Geometry, pixel format and
    /// colour travel in the stream's `CodecParameters` on the framework
    /// side (`PixelFormat::from(image.format)`,
    /// `ColorSignal::from(image.color)`); no side-channel record is
    /// attached so the Motion-JPEG frame shape is unchanged.
    fn from(img: JpegImage) -> Self {
        VideoFrame {
            pts: None,
            planes: img.planes.into_iter().map(VideoPlane::from).collect(),
        }
    }
}

impl JpegImage {
    /// Rebuild an image from a framework frame plus the stream's codec
    /// parameters: `width`, `height` and `pixel_format` must be set
    /// (`Error::InvalidData` otherwise, or when the pixel format is not
    /// a JPEG layout or the image-plane count does not fit it); the
    /// frame's own colour-signal side channel wins over
    /// `params.color_signal`. Side-channel records are not copied —
    /// only the image planes become `planes`.
    pub fn from_video_frame(frame: &VideoFrame, params: &CodecParameters) -> Result<JpegImage> {
        let width = params
            .width
            .ok_or_else(|| Error::invalid("JPEG image: codec parameters carry no width"))?;
        let height = params
            .height
            .ok_or_else(|| Error::invalid("JPEG image: codec parameters carry no height"))?;
        let pix = params
            .pixel_format
            .ok_or_else(|| Error::invalid("JPEG image: codec parameters carry no pixel format"))?;
        let format = MjpegPixelFormat::try_from(pix)
            .map_err(|e| Error::invalid(format!("JPEG image: {e}")))?;
        let planes: Vec<MjpegPlane> = frame
            .image_planes()
            .iter()
            .map(|p| MjpegPlane::new(p.stride, p.data.clone()))
            .collect();
        if planes.len() != format.plane_count() {
            return Err(Error::invalid(format!(
                "JPEG image: {} image plane(s) do not fit {format} ({} expected)",
                planes.len(),
                format.plane_count()
            )));
        }
        let signal = frame.color_signal().unwrap_or(params.color_signal);
        Ok(JpegImage::new(width, height, format, planes)?.with_color(ColorInfo::from(signal)))
    }
}

impl TryFrom<(&VideoFrame, &CodecParameters)> for JpegImage {
    type Error = Error;

    fn try_from((frame, params): (&VideoFrame, &CodecParameters)) -> Result<Self> {
        JpegImage::from_video_frame(frame, params)
    }
}

impl From<ColorInfo> for oxideav_core::ColorSignal {
    fn from(c: ColorInfo) -> Self {
        let range = match c.range {
            ColorRange::Full => oxideav_core::ColorRange::Full,
            ColorRange::Limited => oxideav_core::ColorRange::Limited,
            _ => oxideav_core::ColorRange::Unspecified,
        };
        oxideav_core::ColorSignal::new(
            range,
            oxideav_core::ColorPrimaries::new(c.primaries),
            oxideav_core::TransferCharacteristics::new(c.transfer),
            oxideav_core::MatrixCoefficients::new(c.matrix),
        )
    }
}

impl From<oxideav_core::ColorSignal> for ColorInfo {
    fn from(c: oxideav_core::ColorSignal) -> Self {
        let range = match c.range {
            oxideav_core::ColorRange::Full => ColorRange::Full,
            oxideav_core::ColorRange::Limited => ColorRange::Limited,
            _ => ColorRange::Unspecified,
        };
        ColorInfo::new(
            range,
            c.primaries.code_point(),
            c.transfer.code_point(),
            c.matrix.code_point(),
        )
    }
}

// ---- CodecRegistry / ContainerRegistry entry points --------------------

/// Register the JPEG / MJPEG codec (decoder + encoder) into the
/// supplied [`CodecRegistry`].
///
/// Kept as a free function (rather than a method on a registry handle)
/// so it matches the registration shape used by the umbrella
/// `oxideav` crate.
pub fn register_codecs(reg: &mut CodecRegistry) {
    // The declared input layouts let a framework caller (the
    // `oxideav-image` gateway) convert to one of them before encoding
    // instead of guessing; every entry is proved by the registry matrix
    // test (`tests/registry_matrix.rs`).
    let caps = CodecCapabilities::video("mjpeg_sw")
        .with_lossy(true)
        .with_lossless(true)
        .with_intra_only(true)
        .with_max_size(16384, 16384)
        .with_pixel_formats(ACCEPTED_PIXEL_FORMATS.to_vec());
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder)
            .encoder_options::<MjpegEncoderOptions>()
            .tags([
                // AVI FourCC claims — all unambiguous MJPEG variants.
                CodecTag::fourcc(b"MJPG"),
                CodecTag::fourcc(b"AVRN"),
                CodecTag::fourcc(b"LJPG"),
                CodecTag::fourcc(b"JPGL"),
            ]),
    );
}

/// Register both JPEG-family containers:
///
/// - `jpeg` — still-image (`.jpg` / `.jpeg` / `.jpe` / `.jfif`), single
///   packet per file.
/// - `mjpeg-raw` — raw Motion-JPEG (`.mjpeg` / `.mjpg`), concatenated
///   SOI..EOI frames, one packet per frame, with seek support.
///
/// Must be called alongside [`register_codecs`] when wiring up a
/// pipeline that expects to read or write JPEG-family files.
pub fn register_containers(reg: &mut ContainerRegistry) {
    container::register(reg);
    mjpeg_container::register(reg);
}

/// Unified entry point: install every codec and container provided by
/// `oxideav-mjpeg` into a [`RuntimeContext`].
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    register_containers(&mut ctx.containers);
}

oxideav_core::register!("mjpeg", register);

// ---- Decoder trait impl ------------------------------------------------

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let codec_id = params.codec_id.clone();
    // A T.81 §B.5 abbreviated table-specification stream in `extradata`
    // (TIFF `JPEGTables`) is preloaded ahead of every packet, so
    // table-less abbreviated image streams decode.
    let tables = (params.extradata.len() >= 2 && params.extradata[..2] == [0xFF, 0xD8])
        .then(|| params.extradata.clone());
    Ok(Box::new(MjpegDecoder {
        codec_id,
        tables,
        pending: None,
        eof: false,
    }))
}

struct MjpegDecoder {
    codec_id: CodecId,
    /// §B.5 tables-only stream shared by every packet (`extradata`).
    tables: Option<Vec<u8>>,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for MjpegDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "MJPEG decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        // One implementation: the registry decoder is a thin adapter
        // over the standalone `decode_with` (the §B.5 tables stream
        // travels in `DecodeOptions::tables`); the `JpegImage` becomes
        // a framework frame by `From`, `pts` passed through.
        let mut opts = crate::DecodeOptions::new();
        if let Some(t) = &self.tables {
            opts = opts.with_tables(t.clone());
        }
        let img = crate::decode_with(&pkt.data, &opts)?;
        let mut vf = VideoFrame::from(img);
        vf.pts = pkt.pts;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ---- Encoder trait impl ------------------------------------------------

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Ok(MjpegEncoder::from_params(params)?)
}

/// The input layouts the registry encoder declares through
/// `CodecCapabilities::accepted_pixel_formats` — every layout it codes
/// faithfully (the stream carries the samples as given), each with the
/// layout the decoder reports for the stream (see
/// [`coded_pixel_format`]):
///
/// | Input | Stream (no options) | Decodes as |
/// |---|---|---|
/// | `YuvJ444P` / `YuvJ422P` / `YuvJ420P` | JFIF YCbCr, luma `1×1` / `2×1` / `2×2` | the same label |
/// | `Yuv411P` | JFIF YCbCr, luma `4×1` | `Yuv411P` |
/// | `Yuv440P` | JFIF YCbCr, luma `1×2` | `YuvJ444P` (the decoder replicates 4:4:0 chroma onto the frame grid) |
/// | `Gray8` | JFIF, one component | `Gray8` |
/// | `Gray12Le` / `Yuv444P12Le` / `Yuv422P12Le` / `Yuv420P12Le` | 12-bit extended sequential (`SOF1`) | the same label |
/// | `Gray10Le` / `Gray16Le` | lossless (`SOF3`) at `P = 10` / `16` — the only process that carries them | the same label |
/// | `Rgb24` | RGB-coded (T.872 Adobe APP14 `transform = 0`, ids `'R' 'G' 'B'`), no colour conversion | `Rgb24` |
/// | `Rgb48Le` | lossless at `P = 16`, RGB-coded | `Rgb48Le` |
/// | `Gbrp10Le` / `Gbrp12Le` / `Gbrp14Le` | lossless, planes in order, ids `1 2 3`, no colour signalling | the same label |
/// | `Cmyk` | four components, no APP14 (T.872 §6.1 plain CMYK) | `Cmyk` |
/// | `CmykInverted` | complemented into plain CMYK | `Cmyk` |
///
/// The range-agnostic `Yuv444P` / `Yuv422P` / `Yuv420P` labels are
/// accepted too (same layouts; JPEG YCbCr is full range by definition,
/// T.871 §7, and the encoder does not scale ranges) but not declared, so
/// a framework caller that converts through the declaration reaches the
/// explicit full-range label. Alpha layouts are refused (JPEG has no
/// alpha): drop the channel first.
pub const ACCEPTED_PIXEL_FORMATS: &[PixelFormat] = &[
    PixelFormat::YuvJ444P,
    PixelFormat::YuvJ422P,
    PixelFormat::YuvJ420P,
    PixelFormat::Yuv411P,
    PixelFormat::Yuv440P,
    PixelFormat::Gray8,
    PixelFormat::Gray10Le,
    PixelFormat::Gray12Le,
    PixelFormat::Gray16Le,
    PixelFormat::Yuv444P12Le,
    PixelFormat::Yuv422P12Le,
    PixelFormat::Yuv420P12Le,
    PixelFormat::Rgb24,
    PixelFormat::Rgb48Le,
    PixelFormat::Gbrp10Le,
    PixelFormat::Gbrp12Le,
    PixelFormat::Gbrp14Le,
    PixelFormat::Cmyk,
    PixelFormat::CmykInverted,
];

/// [`ACCEPTED_PIXEL_FORMATS`] as a function (for callers that want the
/// slice without naming the constant).
pub fn accepted_pixel_formats() -> &'static [PixelFormat] {
    ACCEPTED_PIXEL_FORMATS
}

/// The layout the decoder reports for the stream the registry encoder
/// writes from `input` with an empty option bag (sequential DCT for the
/// DCT-capable layouts, lossless for the lossless-only carriers); `None`
/// when `input` is not accepted. The encoder's
/// `output_params().pixel_format` says the same thing for the options
/// actually in force.
pub fn coded_pixel_format(input: PixelFormat) -> Option<PixelFormat> {
    let layout = InputLayout::of(input).ok()?;
    let shape = layout.shape(None);
    Some(coded_label(
        &shape,
        layout.default_process(1),
        shape.precision,
    ))
}

/// The `CodecOptions` schema of the registry encoder — the string-bag
/// twin of [`EncodeOptions`]. Any key present in
/// `CodecParameters::options` routes the encoder through the general
/// T.81 writer (`oxideav_mjpeg::t81`), byte-identical to the standalone
/// [`crate::encode`] for the same image and options; an empty bag keeps
/// the historical per-format paths where one exists (8-bit YCbCr 4:4:4 /
/// 4:2:2 / 4:2:0, `Gray8`, `Rgb24`, `Cmyk`) and the general writer with
/// its defaults everywhere else.
///
/// | key               | kind                                   | default      |
/// |-------------------|----------------------------------------|--------------|
/// | `quality`         | 1..=100                                | 75           |
/// | `tables`          | `typical` \| `optimal`                 | `typical`    |
/// | `process`         | `auto` \| `sequential` \| `progressive` \| `lossless` | `auto` (sequential; lossless for the lossless-only carriers) |
/// | `precision`       | 8 / 12 (DCT), 2..=16 (lossless); must match the pixel format, except on the 16-bit carriers | from the pixel format |
/// | `restart`         | restart interval in MCUs               | 0            |
/// | `abbreviated`     | §B.5 table-less frames, tables in `extradata` | false  |
/// | `predictor`       | Table H.1 selector 1..=7 (lossless)    | 1            |
/// | `point_transform` | `Pt` (lossless)                        | 0            |
/// | `sampling`        | `HxV[,HxV…]` per component; must match the pixel format | from the pixel format |
/// | `chroma`          | `yuv444` \| `yuv422` \| `yuv420` \| `yuv411` \| `gray`: convert `Rgb24` input to that JFIF layout (T.871 §7 forward equations, as [`crate::encode_rgb8`]); `gray` keeps the luma of a YCbCr input | keep the input layout |
///
/// `#[non_exhaustive]`: build with [`MjpegEncoderOptions::new`] /
/// [`Default`] and the `with_*` setters; the fields stay readable.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MjpegEncoderOptions {
    pub quality: u32,
    pub tables: String,
    /// `"auto"` / empty: sequential DCT where the layout allows it,
    /// lossless for the lossless-only carriers.
    pub process: String,
    /// `0` = take the precision the pixel format implies.
    pub precision: u32,
    pub restart: u32,
    pub abbreviated: bool,
    pub predictor: u32,
    pub point_transform: u32,
    pub sampling: String,
    /// Empty = keep the input layout; see the table above.
    pub chroma: String,
}

impl Default for MjpegEncoderOptions {
    fn default() -> Self {
        MjpegEncoderOptions {
            quality: u32::from(DEFAULT_QUALITY),
            tables: "typical".into(),
            process: "auto".into(),
            precision: 0,
            restart: 0,
            abbreviated: false,
            predictor: 1,
            point_transform: 0,
            sampling: String::new(),
            chroma: String::new(),
        }
    }
}

impl MjpegEncoderOptions {
    /// The defaults (the table on the type).
    pub fn new() -> Self {
        Self::default()
    }

    /// Quality factor `1..=100`.
    pub fn with_quality(mut self, quality: u32) -> Self {
        self.quality = quality;
        self
    }

    /// `"typical"` or `"optimal"` Huffman tables.
    pub fn with_tables(mut self, tables: impl Into<String>) -> Self {
        self.tables = tables.into();
        self
    }

    /// `"auto"`, `"sequential"`, `"progressive"` or `"lossless"`.
    pub fn with_process(mut self, process: impl Into<String>) -> Self {
        self.process = process.into();
        self
    }

    /// Sample precision (`0` = from the pixel format).
    pub fn with_precision(mut self, precision: u32) -> Self {
        self.precision = precision;
        self
    }

    /// Restart interval in MCUs (`0` = none).
    pub fn with_restart(mut self, restart: u32) -> Self {
        self.restart = restart;
        self
    }

    /// §B.5 abbreviated frames (tables in `extradata`).
    pub fn with_abbreviated(mut self, abbreviated: bool) -> Self {
        self.abbreviated = abbreviated;
        self
    }

    /// Lossless predictor (Table H.1, `1..=7`).
    pub fn with_predictor(mut self, predictor: u32) -> Self {
        self.predictor = predictor;
        self
    }

    /// Lossless point transform `Pt`.
    pub fn with_point_transform(mut self, point_transform: u32) -> Self {
        self.point_transform = point_transform;
        self
    }

    /// Per-component sampling factors `"HxV[,HxV…]"`.
    pub fn with_sampling(mut self, sampling: impl Into<String>) -> Self {
        self.sampling = sampling.into();
        self
    }

    /// `"yuv444"` / `"yuv422"` / `"yuv420"` / `"yuv411"` / `"gray"`
    /// (empty = keep the input layout).
    pub fn with_chroma(mut self, chroma: impl Into<String>) -> Self {
        self.chroma = chroma.into();
        self
    }
}

impl CodecOptionsStruct for MjpegEncoderOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "quality",
            kind: OptionKind::U32,
            default: OptionValue::U32(DEFAULT_QUALITY as u32),
            help: "quality factor 1..=100 (Annex K.1 quantiser scaling)",
        },
        OptionField {
            name: "tables",
            kind: OptionKind::Enum(&["typical", "optimal"]),
            default: OptionValue::String(String::new()),
            help: "Huffman tables: Annex K.3 typical or Annex K.2 optimal (per-frame statistics)",
        },
        OptionField {
            name: "process",
            kind: OptionKind::Enum(&["auto", "sequential", "progressive", "lossless"]),
            default: OptionValue::String(String::new()),
            help: "coding process: auto (sequential DCT; lossless for Gray10Le / Gray16Le / Rgb48Le / Gbrp*Le), sequential DCT (SOF0/SOF1), progressive DCT (SOF2) or lossless (SOF3)",
        },
        OptionField {
            name: "precision",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "sample precision P (8/12 DCT, 2..=16 lossless); 0 = from the pixel format",
        },
        OptionField {
            name: "restart",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "restart interval in MCUs (0 = none; lossless: a multiple of the MCUs per row)",
        },
        OptionField {
            name: "abbreviated",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "emit table-less frames; the shared tables stream is published in extradata",
        },
        OptionField {
            name: "predictor",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "lossless predictor selection value (T.81 Table H.1, 1..=7)",
        },
        OptionField {
            name: "point_transform",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "lossless point transform Pt (0..precision)",
        },
        OptionField {
            name: "sampling",
            kind: OptionKind::String,
            default: OptionValue::String(String::new()),
            help: "per-component sampling factors 'HxV[,HxV...]'; must match the pixel format",
        },
        OptionField {
            name: "chroma",
            kind: OptionKind::Enum(&["yuv444", "yuv422", "yuv420", "yuv411", "gray"]),
            default: OptionValue::String(String::new()),
            help: "convert Rgb24 input to this JFIF YCbCr / grayscale layout (T.871 forward equations); 'gray' keeps the luma of a YCbCr input; unset = keep the input layout",
        },
    ];

    fn apply(&mut self, key: &str, v: &OptionValue) -> Result<()> {
        match key {
            "quality" => self.quality = v.as_u32()?,
            "tables" => self.tables = v.as_str()?.to_owned(),
            "process" => self.process = v.as_str()?.to_owned(),
            "precision" => self.precision = v.as_u32()?,
            "restart" => self.restart = v.as_u32()?,
            "abbreviated" => self.abbreviated = v.as_bool()?,
            "predictor" => self.predictor = v.as_u32()?,
            "point_transform" => self.point_transform = v.as_u32()?,
            "sampling" => self.sampling = v.as_str()?.to_owned(),
            "chroma" => self.chroma = v.as_str()?.to_owned(),
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// Parse `HxV[,HxV…]` into sampling factor pairs.
fn parse_sampling(s: &str) -> Result<Vec<(u8, u8)>> {
    let mut out = Vec::new();
    for item in s.split(',').map(str::trim).filter(|i| !i.is_empty()) {
        let (h, v) = item.split_once('x').ok_or_else(|| {
            Error::invalid(format!("MJPEG encoder: sampling entry '{item}' is not HxV"))
        })?;
        let h: u8 = h
            .trim()
            .parse()
            .map_err(|_| Error::invalid("MJPEG encoder: bad sampling H"))?;
        let v: u8 = v
            .trim()
            .parse()
            .map_err(|_| Error::invalid("MJPEG encoder: bad sampling V"))?;
        out.push((h, v));
    }
    Ok(out)
}

impl MjpegEncoderOptions {
    /// The typed options these string options denote. `precision = 0`
    /// / empty `sampling` are resolved against the pixel format by the
    /// encoder; `process = "auto"` is sequential here (the encoder,
    /// which knows the layout, picks lossless for the lossless-only
    /// carriers — see [`Self::to_encode_options_for`]). `chroma` has
    /// no field in [`EncodeOptions`] the plane-level writer reads; the
    /// encoder applies it to the frame ([`Self::chroma_request`]).
    pub fn to_encode_options(&self) -> Result<EncodeOptions> {
        self.to_encode_options_for(false)
    }

    /// [`Self::to_encode_options`] with `process = "auto"` resolved for
    /// a layout that only the lossless process carries
    /// (`lossless_only`).
    pub fn to_encode_options_for(&self, lossless_only: bool) -> Result<EncodeOptions> {
        if !(1..=100).contains(&self.quality) {
            return Err(Error::invalid("MJPEG encoder: quality must be in 1..=100"));
        }
        if self.restart > u32::from(u16::MAX) {
            return Err(Error::invalid(
                "MJPEG encoder: restart interval exceeds 65535",
            ));
        }
        let tables = match self.tables.as_str() {
            "typical" | "" => HuffmanTables::Typical,
            "optimal" => HuffmanTables::Optimal,
            other => return Err(Error::invalid(format!("MJPEG encoder: tables '{other}'"))),
        };
        let lossless = || -> Result<JpegProcess> {
            if !(1..=7).contains(&self.predictor) {
                return Err(Error::invalid("MJPEG encoder: predictor must be in 1..=7"));
            }
            if self.point_transform > 15 {
                return Err(Error::invalid(
                    "MJPEG encoder: point_transform must be ≤ 15",
                ));
            }
            Ok(JpegProcess::Lossless {
                predictor: self.predictor as u8,
                point_transform: self.point_transform as u8,
            })
        };
        let process = match self.process.as_str() {
            "auto" | "" if lossless_only => lossless()?,
            "auto" | "" | "sequential" => JpegProcess::Sequential,
            "progressive" => JpegProcess::Progressive,
            "lossless" => lossless()?,
            other => return Err(Error::invalid(format!("MJPEG encoder: process '{other}'"))),
        };
        Ok(EncodeOptions {
            quality: self.quality as u8,
            tables,
            process,
            precision: self.precision.min(16) as u8,
            restart_interval: self.restart as u16,
            abbreviated: self.abbreviated,
            signalling: ColorSignalling::Auto,
            sampling: parse_sampling(&self.sampling)?,
            ..EncodeOptions::default()
        })
    }

    /// The conversion `chroma` asks for (`None` = keep the input
    /// layout).
    pub fn chroma_request(&self) -> Result<Option<ChromaSubsampling>> {
        Ok(Some(match self.chroma.as_str() {
            "" => return Ok(None),
            "yuv444" | "444" => ChromaSubsampling::Yuv444,
            "yuv422" | "422" => ChromaSubsampling::Yuv422,
            "yuv420" | "420" => ChromaSubsampling::Yuv420,
            "yuv411" | "411" => ChromaSubsampling::Yuv411,
            "gray" | "grey" | "grayscale" => ChromaSubsampling::GrayscaleOnly,
            other => {
                return Err(Error::invalid(format!(
                    "MJPEG encoder: chroma '{other}' (yuv444 / yuv422 / yuv420 / yuv411 / gray)"
                )))
            }
        }))
    }
}

/// Colour class of an input layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// One component, JFIF.
    Gray,
    /// Three components YCbCr, JFIF.
    Ycc,
    /// Three components R, G, B: T.872 Adobe APP14 `transform = 0` +
    /// ids `'R' 'G' 'B'`.
    Rgb,
    /// Three 16-bit planes in order (G, B, R), ids `1 2 3`, no colour
    /// signalling, lossless only.
    Gbr,
    /// Four components, plain CMYK unless `set_adobe_transform`.
    Cmyk,
}

/// What an accepted input pixel format means for the encoder.
#[derive(Clone, Debug, PartialEq, Eq)]
struct InputLayout {
    /// The caller's label.
    input: PixelFormat,
    class: Class,
    /// Bits per sample the frame nominally carries (8 / 10 / 12 / 16).
    precision: u8,
    /// Bytes per sample in the frame (1 or 2, little-endian).
    bps: usize,
    /// Component count.
    nf: usize,
    /// One interleaved plane (`Rgb24`, `Rgb48Le`, `Cmyk`).
    packed: bool,
    /// Samples stored complemented (`CmykInverted`).
    complemented: bool,
    /// Per-component sampling factors; empty = every component `1×1`.
    sampling: Vec<(u8, u8)>,
    /// Only the lossless process carries the samples.
    lossless_only: bool,
    /// The range-agnostic local label the historical per-format
    /// encoders take; `None` when only the general writer serves the
    /// layout.
    legacy: Option<MjpegPixelFormat>,
}

/// The layout the writer sees once the `chroma` conversion (if any) is
/// applied.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Shape {
    class: Class,
    precision: u8,
    nf: usize,
    sampling: Vec<(u8, u8)>,
}

impl InputLayout {
    /// The layout of an accepted pixel format, or the typed refusal
    /// (`Error::Unsupported`, naming the layout) for anything else.
    fn of(p: PixelFormat) -> Result<InputLayout> {
        use MjpegPixelFormat as L;
        use PixelFormat as P;
        let mk = |class: Class,
                  precision: u8,
                  bps: usize,
                  nf: usize,
                  packed: bool,
                  sampling: &[(u8, u8)],
                  lossless_only: bool,
                  legacy: Option<L>| InputLayout {
            input: p,
            class,
            precision,
            bps,
            nf,
            packed,
            complemented: false,
            sampling: sampling.to_vec(),
            lossless_only,
            legacy,
        };
        const S422: &[(u8, u8)] = &[(2, 1), (1, 1), (1, 1)];
        const S420: &[(u8, u8)] = &[(2, 2), (1, 1), (1, 1)];
        const S411: &[(u8, u8)] = &[(4, 1), (1, 1), (1, 1)];
        const S440: &[(u8, u8)] = &[(1, 2), (1, 1), (1, 1)];
        Ok(match p {
            P::Gray8 => mk(Class::Gray, 8, 1, 1, false, &[], false, Some(L::Gray8)),
            // The lossless-only carriers always run the general writer
            // (one stream whatever the path; the historical lossless
            // grayscale function stays for direct callers).
            P::Gray10Le => mk(Class::Gray, 10, 2, 1, false, &[], true, None),
            P::Gray12Le => mk(Class::Gray, 12, 2, 1, false, &[], false, Some(L::Gray12Le)),
            P::Gray16Le => mk(Class::Gray, 16, 2, 1, false, &[], true, None),
            P::YuvJ444P | P::Yuv444P => {
                mk(Class::Ycc, 8, 1, 3, false, &[], false, Some(L::Yuv444P))
            }
            P::YuvJ422P | P::Yuv422P => {
                mk(Class::Ycc, 8, 1, 3, false, S422, false, Some(L::Yuv422P))
            }
            P::YuvJ420P | P::Yuv420P => {
                mk(Class::Ycc, 8, 1, 3, false, S420, false, Some(L::Yuv420P))
            }
            P::Yuv411P => mk(Class::Ycc, 8, 1, 3, false, S411, false, None),
            P::Yuv440P => mk(Class::Ycc, 8, 1, 3, false, S440, false, None),
            P::Yuv444P12Le => mk(Class::Ycc, 12, 2, 3, false, &[], false, None),
            P::Yuv422P12Le => mk(Class::Ycc, 12, 2, 3, false, S422, false, None),
            P::Yuv420P12Le => mk(Class::Ycc, 12, 2, 3, false, S420, false, None),
            P::Rgb24 => mk(Class::Rgb, 8, 1, 3, true, &[], false, Some(L::Rgb24)),
            P::Rgb48Le => mk(Class::Rgb, 16, 2, 3, true, &[], true, None),
            P::Gbrp10Le => mk(Class::Gbr, 10, 2, 3, false, &[], true, None),
            P::Gbrp12Le => mk(Class::Gbr, 12, 2, 3, false, &[], true, None),
            P::Gbrp14Le => mk(Class::Gbr, 14, 2, 3, false, &[], true, None),
            P::Cmyk => mk(Class::Cmyk, 8, 1, 4, true, &[], false, Some(L::Cmyk)),
            P::CmykInverted => {
                let mut l = mk(Class::Cmyk, 8, 1, 4, true, &[], false, Some(L::Cmyk));
                l.complemented = true;
                l
            }
            other => return Err(Error::unsupported(refusal(other))),
        })
    }

    /// The process an empty option bag means for this layout.
    fn default_process(&self, predictor: u8) -> JpegProcess {
        if self.lossless_only {
            JpegProcess::Lossless {
                predictor,
                point_transform: 0,
            }
        } else {
            JpegProcess::Sequential
        }
    }

    /// The writer-side layout once `chroma` is applied (validated by
    /// [`Self::check_chroma`]).
    fn shape(&self, chroma: Option<ChromaSubsampling>) -> Shape {
        let own = Shape {
            class: self.class,
            precision: self.precision,
            nf: self.nf,
            sampling: self.sampling.clone(),
        };
        let gray = Shape {
            class: Class::Gray,
            precision: self.precision,
            nf: 1,
            sampling: Vec::new(),
        };
        let ycc = |sampling: &[(u8, u8)]| Shape {
            class: Class::Ycc,
            precision: 8,
            nf: 3,
            sampling: sampling.to_vec(),
        };
        match (self.class, chroma) {
            (_, None) => own,
            (_, Some(ChromaSubsampling::GrayscaleOnly)) => gray,
            (Class::Rgb, Some(ChromaSubsampling::Yuv444)) => ycc(&[]),
            (Class::Rgb, Some(ChromaSubsampling::Yuv422)) => ycc(&[(2, 1), (1, 1), (1, 1)]),
            (Class::Rgb, Some(ChromaSubsampling::Yuv420)) => ycc(&[(2, 2), (1, 1), (1, 1)]),
            (Class::Rgb, Some(ChromaSubsampling::Yuv411)) => ycc(&[(4, 1), (1, 1), (1, 1)]),
            _ => own,
        }
    }

    /// Whether `chroma` is a conversion this layout admits.
    fn check_chroma(&self, chroma: Option<ChromaSubsampling>) -> Result<()> {
        let Some(c) = chroma else { return Ok(()) };
        let ok = match (self.class, c) {
            (Class::Rgb, ChromaSubsampling::Custom) => false,
            (Class::Rgb, _) if self.precision == 8 => true,
            (Class::Gray, ChromaSubsampling::GrayscaleOnly) => true,
            (Class::Ycc, ChromaSubsampling::GrayscaleOnly) => true,
            // A YCbCr input in the layout asked for: nothing to do.
            (Class::Ycc, c) => {
                self.precision == 8
                    && self.sampling
                        == match c {
                            ChromaSubsampling::Yuv444 => Vec::new(),
                            ChromaSubsampling::Yuv422 => vec![(2, 1), (1, 1), (1, 1)],
                            ChromaSubsampling::Yuv420 => vec![(2, 2), (1, 1), (1, 1)],
                            ChromaSubsampling::Yuv411 => vec![(4, 1), (1, 1), (1, 1)],
                            _ => vec![(0, 0)],
                        }
            }
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err(Error::unsupported(format!(
                "MJPEG encoder: chroma {c:?} does not apply to {:?} input (Rgb24 converts to any JFIF layout; a YCbCr or grey input only to gray)",
                self.input
            )))
        }
    }
}

/// Why `p` is refused, naming the layout.
fn refusal(p: PixelFormat) -> String {
    if p.has_alpha() {
        format!(
            "MJPEG encoder: pixel format {p:?} carries an alpha channel and JPEG has none — drop the alpha first (Rgb24 / YuvJ444P / Gray8)"
        )
    } else {
        format!(
            "MJPEG encoder: pixel format {p:?} is not a JPEG layout; accepted: {}",
            ACCEPTED_PIXEL_FORMATS
                .iter()
                .map(|f| format!("{f:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// The layout the decoder reports for a stream of `shape` written with
/// `process` at sample precision `precision`.
fn coded_label(shape: &Shape, process: JpegProcess, precision: u8) -> PixelFormat {
    use PixelFormat as P;
    let dct = process.is_dct();
    match shape.class {
        Class::Gray => match precision {
            8 => P::Gray8,
            10 if !dct => P::Gray10Le,
            12 => P::Gray12Le,
            _ => P::Gray16Le,
        },
        Class::Ycc => {
            let luma = shape.sampling.first().copied().unwrap_or((1, 1));
            match (luma, precision) {
                ((2, 1), 8) => P::YuvJ422P,
                ((2, 2), 8) => P::YuvJ420P,
                ((4, 1), 8) => P::Yuv411P,
                (_, 8) => P::YuvJ444P,
                ((2, 1), _) => P::Yuv422P12Le,
                ((2, 2), _) => P::Yuv420P12Le,
                _ => P::Yuv444P12Le,
            }
        }
        Class::Rgb | Class::Gbr => match precision {
            8 => P::Rgb24,
            10 => P::Gbrp10Le,
            12 => P::Gbrp12Le,
            14 => P::Gbrp14Le,
            _ => P::Rgb48Le,
        },
        Class::Cmyk => P::Cmyk,
    }
}

/// JPEG encoder. Emits one self-contained JPEG bitstream (baseline SOF0
/// or progressive SOF2) per video frame.
pub struct MjpegEncoder {
    output_params: CodecParameters,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// The input layout (`params.pixel_format`).
    layout: InputLayout,
    /// The `chroma` conversion in force (`None` = keep the layout).
    chroma: Option<ChromaSubsampling>,
    quality: u8,
    /// MCU-per-restart-interval count. 0 disables DRI / `RSTn` emission.
    /// Restart intervals are only honoured on the baseline (SOF0) path
    /// for now; progressive emission ignores this field.
    restart_interval: u16,
    /// When true, emit SOF2 + multi-scan (spectral selection only).
    progressive: bool,
    /// When true, take the lossless (SOF3) path for single-component
    /// grayscale input. Ignored for any non-grayscale `pix`.
    lossless: bool,
    /// When true, emit a sequential arithmetic-coded DCT frame (SOF9)
    /// instead of the Huffman baseline (SOF0). Honoured for `Gray8`,
    /// `Yuv*P` and `Rgb24` input; ignored on the lossless and progressive
    /// paths (which take precedence) and for CMYK / high-bit-depth input.
    arithmetic: bool,
    /// Lossless predictor selector (T.81 Table H.1, 1..=7). Only
    /// consulted on the lossless path. Defaults to 1 (Ra / left).
    lossless_predictor: u8,
    /// Adobe APP14 colour-transform marker for 4-component
    /// (`MjpegPixelFormat::Cmyk`) input.
    ///
    /// * `None`     — no APP14 segment (plain "regular" CMYK).
    /// * `Some(0)`  — Adobe CMYK: encoder inverts every component on
    ///   the wire; decoder un-inverts on output.
    /// * `Some(2)`  — Adobe YCCK: the packed input is interpreted as
    ///   `[Y, Cb, Cr, K]` and only the K plane is inverted on the
    ///   wire; the decoder performs YCbCr→RGB→CMY and flips K to
    ///   recover CMYK.
    ///
    /// Defaults to `None`. Ignored for non-CMYK pixel formats.
    cmyk_adobe_transform: Option<u8>,
    /// When set, every frame goes through the general T.81 writer with
    /// these options (see [`MjpegEncoder::set_encode_options`]).
    general: Option<EncodeOptions>,
    /// `general` was derived from the knobs (`set_progressive`, …) for a
    /// layout without a historical path, not set by the caller: it is
    /// re-derived on every frame so the knobs keep working.
    general_implied: bool,
    /// The table set shared by every abbreviated frame (its §B.5 stream
    /// is `output_params().extradata`).
    shared_tables: Option<JpegTableSet>,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl MjpegEncoder {
    /// Build a concrete `MjpegEncoder` from video codec parameters.
    /// Preferred over `make_encoder` when the caller wants to tweak
    /// encoder-specific knobs (e.g. progressive mode, restart interval)
    /// before feeding frames.
    ///
    /// `pixel_format` must be one of [`ACCEPTED_PIXEL_FORMATS`] or a
    /// range-agnostic `Yuv444P` / `Yuv422P` / `Yuv420P` (default
    /// `Yuv420P`); anything else is `Error::Unsupported` naming the
    /// layout. The output parameters carry the layout the decoder
    /// reports for the stream ([`coded_pixel_format`]) and its colour
    /// signal, as the `jpeg` demuxer would declare them.
    pub fn from_params(params: &CodecParameters) -> Result<Box<Self>> {
        let width = params
            .width
            .ok_or_else(|| Error::invalid("MJPEG encoder: missing width"))?;
        let height = params
            .height
            .ok_or_else(|| Error::invalid("MJPEG encoder: missing height"))?;
        if width == 0 || height == 0 || width > 65535 || height > 65535 {
            return Err(Error::invalid(format!(
                "MJPEG encoder: {width}×{height} is outside T.81's 1..=65535 per axis"
            )));
        }
        let pix_core = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
        let layout = InputLayout::of(pix_core)?;

        let mut output_params = params.clone();
        output_params.media_type = MediaType::Video;
        output_params.codec_id = CodecId::new(CODEC_ID_STR);
        output_params.width = Some(width);
        output_params.height = Some(height);

        let mut enc = Self {
            output_params,
            width,
            height,
            layout,
            chroma: None,
            quality: DEFAULT_QUALITY,
            restart_interval: 0,
            progressive: false,
            // Lossless mode is opt-in even for grayscale. `Gray8` input
            // takes the baseline (SOF0) single-component DCT path by
            // default; flip `set_lossless(true)` to switch to the
            // bit-exact SOF3 path instead.
            lossless: false,
            arithmetic: false,
            lossless_predictor: 1,
            cmyk_adobe_transform: None,
            general: None,
            general_implied: false,
            shared_tables: None,
            time_base: params
                .frame_rate
                .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num)),
            pending: VecDeque::new(),
            eof: false,
        };
        let (process, precision) = (enc.layout.default_process(1), enc.layout.precision);
        enc.label_output(process, precision);
        // Any option key routes through the general T.81 writer with the
        // typed options the string bag denotes (strict: unknown keys and
        // malformed values are rejected here).
        if !params.options.is_empty() {
            let opts = parse_options::<MjpegEncoderOptions>(&params.options)?;
            let chroma = opts.chroma_request()?;
            enc.layout.check_chroma(chroma)?;
            enc.chroma = chroma;
            let lossless_only = enc.layout.lossless_only && chroma.is_none();
            enc.set_encode_options(opts.to_encode_options_for(lossless_only)?)?;
        } else if enc.layout.legacy.is_none() {
            // No historical per-format path: the general writer with
            // its defaults (re-derived from the knobs on every frame).
            let o = enc.implied_options();
            enc.set_encode_options(o)?;
            enc.general_implied = true;
        }
        Ok(Box::new(enc))
    }

    /// Stamp the output parameters with what the decoder will report.
    fn label_output(&mut self, process: JpegProcess, precision: u8) {
        let shape = self.layout.shape(self.chroma);
        let coded = coded_label(&shape, process, precision);
        self.output_params.pixel_format = Some(coded);
        if let Some(local) = pix_to_local(coded) {
            self.output_params.color_signal =
                oxideav_core::ColorSignal::from(crate::api::color_for(local));
        }
    }

    /// The general-writer options the knobs imply for a layout without
    /// a historical path.
    fn implied_options(&self) -> EncodeOptions {
        let mut o = EncodeOptions::new()
            .with_quality(self.quality)
            .with_restart_interval(self.restart_interval)
            // Resolved against the layout by `set_encode_options`.
            .with_precision(0);
        o.process = if self.lossless || self.layout.lossless_only {
            JpegProcess::Lossless {
                predictor: self.lossless_predictor,
                point_transform: 0,
            }
        } else if self.progressive {
            JpegProcess::Progressive
        } else {
            JpegProcess::Sequential
        };
        o
    }

    /// Route every frame through the general T.81 writer
    /// (`oxideav_mjpeg::t81`) with `opts` — sequential / progressive /
    /// lossless, 8- or 12-bit, optimal tables, restarts, abbreviated
    /// streams. `precision = 0` and an empty `sampling` take the values
    /// the pixel format implies; non-zero / non-empty values must match
    /// them (the 16-bit carriers `Gray16Le` / `Rgb48Le` admit any
    /// lossless precision `2..=16` — the samples sit in the low bits —
    /// except `10 / 12 / 14` on `Rgb48Le`, which the decoder labels as
    /// the planar `Gbrp*Le` carriers: code those from `Gbrp*Le` planes).
    /// Combinations the decoder would relabel or refuse are refused here
    /// (`Error::Unsupported`): a DCT process on a lossless-only carrier,
    /// the lossless process on 4:4:4 YCbCr (T.81 Annex H has no colour
    /// transform, so the stream decodes as the RGB-class carrier) or on
    /// 12-bit subsampled YCbCr. With `abbreviated` the shared §B.5 tables
    /// stream is published in `output_params().extradata` — immediately
    /// for typical tables, after the first frame for optimal ones (the
    /// statistics of the first frame define the shared set).
    pub fn set_encode_options(&mut self, mut opts: EncodeOptions) -> Result<()> {
        let shape = self.layout.shape(self.chroma);
        let input = self.layout.input;
        let carrier16 =
            self.chroma.is_none() && matches!(input, PixelFormat::Gray16Le | PixelFormat::Rgb48Le);
        if opts.precision == 0 {
            opts.precision = shape.precision;
        } else if opts.precision != shape.precision {
            let low_bits = carrier16
                && !opts.process.is_dct()
                && (2..=16).contains(&opts.precision)
                && !(input == PixelFormat::Rgb48Le && matches!(opts.precision, 10 | 12 | 14));
            if !low_bits {
                return Err(Error::invalid(format!(
                    "MJPEG encoder: precision {} does not match the {input:?} input (P = {})",
                    opts.precision, shape.precision
                )));
            }
        }
        if opts.sampling.is_empty() {
            opts.sampling = shape.sampling.clone();
        } else if opts.sampling != shape.sampling
            && !(shape.sampling.is_empty() && opts.sampling.iter().all(|&s| s == (1, 1)))
        {
            return Err(Error::invalid(format!(
                "MJPEG encoder: sampling {:?} does not match the {input:?} input",
                opts.sampling
            )));
        }
        if opts.signalling == ColorSignalling::Auto {
            opts.signalling = match shape.class {
                Class::Rgb => ColorSignalling::Rgb,
                Class::Gbr => ColorSignalling::None,
                Class::Cmyk => ColorSignalling::Cmyk {
                    adobe_transform: self.cmyk_adobe_transform,
                },
                Class::Gray | Class::Ycc => ColorSignalling::Jfif,
            };
        }
        if opts.process.is_dct() {
            if self.layout.lossless_only && self.chroma.is_none() {
                return Err(Error::unsupported(format!(
                    "MJPEG encoder: {input:?} is lossless-only (the DCT processes carry P = 8 or 12 and have no RGB-class signalling above 8 bits); set process = lossless or convert first"
                )));
            }
            if !matches!(opts.precision, 8 | 12) {
                return Err(Error::unsupported(format!(
                    "MJPEG encoder: the DCT processes need P = 8 or 12 (input {input:?} is P = {})",
                    opts.precision
                )));
            }
        } else if shape.class == Class::Ycc {
            let subsampled = opts.sampling.iter().any(|&s| s != (1, 1));
            if !subsampled {
                return Err(Error::unsupported(format!(
                    "MJPEG encoder: lossless 4:4:4 YCbCr from {input:?} is not labelled by the decoder (T.81 Annex H carries no colour transform, so a lossless 1×1 three-component stream is the RGB-class carrier); use Rgb24 / Gbrp12Le for an exact picture or a subsampled YCbCr layout"
                )));
            }
            if opts.precision != 8 {
                return Err(Error::unsupported(format!(
                    "MJPEG encoder: lossless subsampled YCbCr from {input:?} needs P = 8 (the decoder shapes subsampled lossless scans at 8 bits only)"
                )));
            }
        }
        self.shared_tables = None;
        self.output_params.extradata.clear();
        if opts.abbreviated
            && opts.tables == HuffmanTables::Typical
            && !(opts.process.is_dct() && opts.precision > 8)
        {
            let t = JpegTableSet::typical(
                opts.quality,
                opts.precision,
                !opts.process.is_dct(),
                shape.nf,
            );
            self.output_params.extradata = t.tables_stream(opts.process.is_dct());
            self.shared_tables = Some(t);
        }
        self.label_output(opts.process, opts.precision);
        self.general = Some(opts);
        self.general_implied = false;
        Ok(())
    }

    /// The general-writer options in force, if any.
    pub fn encode_options(&self) -> Option<&EncodeOptions> {
        self.general.as_ref()
    }

    /// The input layouts the encoder accepts ([`ACCEPTED_PIXEL_FORMATS`]).
    pub fn accepted_pixel_formats() -> &'static [PixelFormat] {
        ACCEPTED_PIXEL_FORMATS
    }

    /// Every plane the input layout needs is there, with a stride of at
    /// least its tight row and enough bytes for its rows (the last row
    /// may be unpadded) — `InvalidData` naming the layout otherwise, so
    /// no path indexes past a short plane.
    fn check_frame(&self, v: &VideoFrame) -> Result<()> {
        let l = &self.layout;
        let (w, h) = (self.width as usize, self.height as usize);
        let need = if l.packed { 1 } else { l.nf };
        if v.planes.len() < need {
            return Err(Error::invalid(format!(
                "MJPEG encoder: {:?} frame needs {need} plane(s), got {}",
                l.input,
                v.planes.len()
            )));
        }
        let hm = l.sampling.iter().map(|s| s.0).max().unwrap_or(1) as usize;
        let vm = l.sampling.iter().map(|s| s.1).max().unwrap_or(1) as usize;
        for (c, pl) in v.planes.iter().enumerate().take(need) {
            let (cw, ch) = if l.packed || c == 0 {
                (w, h)
            } else {
                let (hi, vi) = l.sampling.get(c).copied().unwrap_or((1, 1));
                (
                    (w * hi as usize).div_ceil(hm),
                    (h * vi as usize).div_ceil(vm),
                )
            };
            let row = cw * if l.packed { l.nf } else { 1 } * l.bps;
            let needed = (ch - 1)
                .checked_mul(pl.stride)
                .and_then(|n| n.checked_add(row))
                .ok_or_else(|| Error::invalid("MJPEG encoder: plane size overflows usize"))?;
            if pl.stride < row || pl.data.len() < needed {
                return Err(Error::invalid(format!(
                    "MJPEG encoder: {:?} plane {c} ({cw}×{ch} samples, {row}-byte rows) is {} bytes at stride {}; {needed} needed",
                    l.input,
                    pl.data.len(),
                    pl.stride
                )));
            }
        }
        Ok(())
    }

    /// Split a frame into per-component `u16` planes at the A.1.1
    /// resolutions the input layout implies, then apply the `chroma`
    /// conversion (RGB → JFIF YCbCr / grey by the T.871 §7 forward
    /// equations, as [`crate::encode_rgb8`]; YCbCr → luma only).
    fn planes_u16(&self, v: &VideoFrame) -> Result<Vec<Vec<u16>>> {
        let l = &self.layout;
        let (w, h) = (self.width as usize, self.height as usize);
        let (nf, bps) = (l.nf, l.bps);
        let need_planes = if l.packed { 1 } else { nf };
        if v.planes.len() < need_planes {
            return Err(Error::invalid(format!(
                "MJPEG encoder: {:?} frame needs {need_planes} plane(s), got {}",
                l.input,
                v.planes.len()
            )));
        }
        let sample = |data: &[u8], o: usize| -> u16 {
            if bps == 2 {
                u16::from(data[o]) | u16::from(data[o + 1]) << 8
            } else {
                u16::from(data[o])
            }
        };
        let mut out = Vec::with_capacity(nf);
        if l.packed {
            let pl = &v.planes[0];
            if pl.stride < w * nf * bps || pl.data.len() < pl.stride * (h - 1) + w * nf * bps {
                return Err(Error::invalid(format!(
                    "MJPEG encoder: packed {:?} plane too small for {w}x{h}",
                    l.input
                )));
            }
            for c in 0..nf {
                let mut p = Vec::with_capacity(w * h);
                for y in 0..h {
                    for x in 0..w {
                        p.push(sample(&pl.data, y * pl.stride + (x * nf + c) * bps));
                    }
                }
                out.push(p);
            }
        } else {
            let hm = l.sampling.iter().map(|s| s.0).max().unwrap_or(1) as usize;
            let vm = l.sampling.iter().map(|s| s.1).max().unwrap_or(1) as usize;
            for c in 0..nf {
                let (hi, vi) = l.sampling.get(c).copied().unwrap_or((1, 1));
                let cw = (w * hi as usize).div_ceil(hm);
                let ch = (h * vi as usize).div_ceil(vm);
                let pl = &v.planes[c];
                if pl.stride < cw * bps || pl.data.len() < pl.stride * (ch - 1) + cw * bps {
                    return Err(Error::invalid(format!(
                        "MJPEG encoder: {:?} plane {c} too small for {cw}x{ch}",
                        l.input
                    )));
                }
                let mut p = Vec::with_capacity(cw * ch);
                for y in 0..ch {
                    for x in 0..cw {
                        p.push(sample(&pl.data, y * pl.stride + x * bps));
                    }
                }
                out.push(p);
            }
        }
        if l.complemented {
            let max = (1u16 << l.precision) - 1;
            for p in &mut out {
                for s in p.iter_mut() {
                    *s = max - (*s).min(max);
                }
            }
        }
        match (l.class, self.chroma) {
            (Class::Rgb, Some(c)) => {
                // Tightly packed 8-bit RGB → the JFIF layout asked for,
                // exactly as `encode_rgb8` builds it.
                let mut rgb = Vec::with_capacity(w * h * 3);
                for i in 0..w * h {
                    rgb.push(out[0][i] as u8);
                    rgb.push(out[1][i] as u8);
                    rgb.push(out[2][i] as u8);
                }
                let o = EncodeOptions::new().with_chroma(c);
                let img = crate::api::rgb8_to_native(self.width, self.height, &rgb, &o)?;
                out = img
                    .planes
                    .iter()
                    .map(|p| p.data.iter().map(|&b| u16::from(b)).collect())
                    .collect();
            }
            (Class::Ycc, Some(ChromaSubsampling::GrayscaleOnly)) => out.truncate(1),
            _ => {}
        }
        Ok(out)
    }

    /// Encode one frame through the general writer.
    fn encode_general(&mut self, opts: &EncodeOptions, v: &VideoFrame) -> Result<Vec<u8>> {
        let planes = self.planes_u16(v)?;
        let refs: Vec<&[u16]> = planes.iter().map(|p| p.as_slice()).collect();
        if opts.abbreviated {
            if self.shared_tables.is_none() {
                let t = opts.tables_for_planes(self.width, self.height, &refs)?;
                self.output_params.extradata = t.tables_stream(opts.process.is_dct());
                self.shared_tables = Some(t);
            }
            let t = self.shared_tables.as_ref().expect("shared tables");
            Ok(opts
                .encode_with_tables(self.width, self.height, &refs, t)?
                .data)
        } else {
            Ok(opts.encode(self.width, self.height, &refs)?.data)
        }
    }

    /// Set the restart interval in MCUs (JPEG DRI field). `0` disables
    /// restart marker emission (matches the default).
    ///
    /// Values are clamped to `u16::MAX` since the JPEG DRI field is a
    /// 16-bit big-endian unsigned integer.
    ///
    /// Currently only applied on the baseline encode path; enabling
    /// progressive output via [`Self::set_progressive`] suppresses
    /// restart-marker emission.
    pub fn set_restart_interval(&mut self, mcus: u32) {
        self.restart_interval = mcus.min(u16::MAX as u32) as u16;
    }

    /// Current restart interval (MCUs between `RSTn` markers; 0 = off).
    pub fn restart_interval(&self) -> u16 {
        self.restart_interval
    }

    /// Enable or disable progressive (SOF2) JPEG emission. When enabled
    /// the encoder produces one DC-first scan plus two per-component AC
    /// band scans (Ss=1..5 then Ss=6..63). See module-level docs.
    pub fn set_progressive(&mut self, on: bool) {
        self.progressive = on;
    }

    /// True when progressive emission is enabled.
    pub fn progressive(&self) -> bool {
        self.progressive
    }

    /// Enable or disable lossless (SOF3) emission. Only honoured when
    /// the input pixel format is `Gray8` / `Gray10Le` / `Gray12Le` /
    /// `Gray16Le`; ignored for YUV inputs (which always take the
    /// baseline / progressive DCT path).
    ///
    /// For `Gray8` input the flag is a real toggle: `false` takes the
    /// baseline (SOF0) single-component DCT path (lossy, scaled by
    /// `quality`), `true` takes the bit-exact lossless (SOF3) path.
    /// `Gray12Le` codes 12-bit sequential DCT without the flag;
    /// `Gray10Le` / `Gray16Le` are lossless whatever the flag says (the
    /// DCT processes carry 8 or 12 bits only).
    ///
    /// The lossless path is bit-exact and reuses the predictor selected
    /// by [`Self::set_lossless_predictor`] (default 1 = Ra / left). It
    /// ignores [`Self::set_progressive`] and [`Self::set_restart_interval`].
    pub fn set_lossless(&mut self, on: bool) {
        self.lossless = on;
    }

    /// True when lossless (SOF3) emission is enabled.
    pub fn lossless(&self) -> bool {
        self.lossless
    }

    /// Enable or disable sequential arithmetic-coded DCT (SOF9) emission.
    /// Honoured for `Gray8`, `Yuv420P` / `Yuv422P` / `Yuv444P` and `Rgb24`
    /// input. The lossless and progressive flags take precedence — when
    /// either is set the arithmetic flag is ignored — and CMYK /
    /// high-bit-depth grayscale input always uses its existing path. The
    /// SOF9 output is the Q-coder counterpart of the baseline SOF0 path:
    /// same forward DCT + quantiser, so it decodes to identical pixels.
    /// `set_restart_interval` is honoured on this path.
    pub fn set_arithmetic(&mut self, on: bool) {
        self.arithmetic = on;
    }

    /// True when sequential arithmetic-coded DCT (SOF9) emission is enabled.
    pub fn arithmetic(&self) -> bool {
        self.arithmetic
    }

    /// Set the lossless predictor selector (T.81 Table H.1, 1..=7).
    /// Values outside `1..=7` are silently clamped to 1 so the setter
    /// can't fail; the value is consulted only when [`Self::set_lossless`]
    /// has been enabled and the input is grayscale.
    pub fn set_lossless_predictor(&mut self, predictor: u8) {
        self.lossless_predictor = if (1..=7).contains(&predictor) {
            predictor
        } else {
            1
        };
    }

    /// Current lossless predictor selector.
    pub fn lossless_predictor(&self) -> u8 {
        self.lossless_predictor
    }

    /// Configure the Adobe APP14 colour-transform marker for 4-component
    /// (`MjpegPixelFormat::Cmyk`) input. Only honoured when the input
    /// pixel format is `Cmyk`; ignored for every other format.
    ///
    /// * `None`     — emit no APP14 segment (the decoder treats the
    ///   result as plain "regular" CMYK).
    /// * `Some(0)`  — Adobe CMYK: every component is inverted on the
    ///   wire; the decoder un-inverts on output.
    /// * `Some(2)`  — Adobe YCCK: the packed input is interpreted as
    ///   `[Y, Cb, Cr, K]`, and only the K plane is inverted on the
    ///   wire. The decoder converts YCbCr→RGB→CMY (BT.601, full-range)
    ///   and flips K to recover CMYK.
    ///
    /// Any other `Some(t)` value is rejected with `Error::InvalidData`
    /// (only `0` and `2` round-trip through this crate's decoder).
    pub fn set_adobe_transform(&mut self, transform: Option<u8>) -> Result<()> {
        if let Some(t) = transform {
            if t != 0 && t != 2 {
                return Err(Error::invalid(
                    "MJPEG encoder: Adobe APP14 transform must be 0 (CMYK) or 2 (YCCK)",
                ));
            }
        }
        self.cmyk_adobe_transform = transform;
        if let Some(o) = &mut self.general {
            if let ColorSignalling::Cmyk { adobe_transform } = &mut o.signalling {
                *adobe_transform = transform;
            }
        }
        Ok(())
    }

    /// Current Adobe APP14 colour-transform marker selection.
    pub fn adobe_transform(&self) -> Option<u8> {
        self.cmyk_adobe_transform
    }

    /// The historical per-format path for the layouts that have one
    /// (`layout.legacy`), driven by the knobs.
    fn encode_legacy(&self, pix: MjpegPixelFormat, v: &VideoFrame) -> Result<Vec<u8>> {
        let plane0 = || -> Result<&VideoPlane> {
            v.planes.first().ok_or_else(|| {
                Error::invalid(format!(
                    "MJPEG encoder: {:?} frame missing plane 0",
                    self.layout.input
                ))
            })
        };
        let out: crate::Result<Vec<u8>> = match (pix, self.lossless) {
            // Grayscale + lossless → SOF3 path (`Gray10Le` / `Gray16Le`
            // always: nothing else carries them). Precision is implied
            // by the pixel format and we read row bytes straight from
            // plane 0.
            (MjpegPixelFormat::Gray8, true)
            | (MjpegPixelFormat::Gray10Le, _)
            | (MjpegPixelFormat::Gray12Le, true)
            | (MjpegPixelFormat::Gray16Le, _) => {
                let plane = plane0()?;
                encode_lossless_jpeg_grayscale(
                    self.width,
                    self.height,
                    &plane.data,
                    plane.stride,
                    self.layout.precision,
                    self.lossless_predictor,
                )
            }
            // 8-bit grayscale without lossless mode takes the baseline
            // (SOF0), progressive (SOF2) or arithmetic (SOF9) single-
            // component DCT path. `restart_interval` is ignored on the
            // progressive path because the 3-component progressive
            // encoder doesn't expose DRI emission either.
            (MjpegPixelFormat::Gray8, false) => {
                let plane = plane0()?;
                if self.progressive {
                    encode_jpeg_progressive_grayscale(
                        self.width,
                        self.height,
                        &plane.data,
                        plane.stride,
                        self.quality,
                    )
                } else if self.arithmetic {
                    encode_arith_jpeg_grayscale(
                        self.width,
                        self.height,
                        &plane.data,
                        plane.stride,
                        self.quality,
                        self.restart_interval,
                    )
                } else {
                    encode_jpeg_grayscale_with_opts(
                        self.width,
                        self.height,
                        &plane.data,
                        plane.stride,
                        self.quality,
                        self.restart_interval,
                    )
                }
            }
            // 12-bit grayscale without the lossless flag: 12-bit
            // sequential / progressive DCT through the general writer.
            (MjpegPixelFormat::Gray12Le, false) => {
                let mut o = self.implied_options();
                o.precision = 12;
                return self.encode_general_stateless(&o, v);
            }
            // Packed `Rgb24` input takes the baseline-SOF0 RGB path (or
            // its SOF9 counterpart). The single plane is laid out as
            // `[R, G, B]` at 3 bytes per pixel, matching the decoder's
            // `Rgb24` output shape.
            (MjpegPixelFormat::Rgb24, _) => {
                let plane = plane0()?;
                let min_stride = (self.width as usize) * 3;
                if plane.stride < min_stride {
                    return Err(Error::invalid(
                        "MJPEG encoder: RGB24 plane stride must be at least width * 3",
                    ));
                }
                if self.arithmetic && !self.lossless {
                    encode_arith_jpeg_rgb24(
                        self.width,
                        self.height,
                        &plane.data,
                        plane.stride,
                        self.quality,
                        self.restart_interval,
                    )
                } else {
                    encode_jpeg_rgb24_with_opts(
                        self.width,
                        self.height,
                        &plane.data,
                        plane.stride,
                        self.quality,
                        self.restart_interval,
                    )
                }
            }
            // 4-component CMYK / YCCK input takes the dedicated CMYK
            // encode path. The single packed plane is laid out as
            // `[C, M, Y, K]` (or `[Y, Cb, Cr, K]` for
            // `set_adobe_transform(Some(2))`) at 4 bytes per pixel,
            // matching the decoder's output shape; `CmykInverted` is
            // complemented into plain ink amounts first.
            (MjpegPixelFormat::Cmyk, _) => {
                let plane = plane0()?;
                let min_stride = (self.width as usize) * 4;
                if plane.stride < min_stride {
                    return Err(Error::invalid(
                        "MJPEG encoder: CMYK plane stride must be at least width * 4",
                    ));
                }
                let complemented: Vec<u8>;
                let data: &[u8] = if self.layout.complemented {
                    complemented = plane.data.iter().map(|&b| 255 - b).collect();
                    &complemented
                } else {
                    &plane.data
                };
                if self.progressive {
                    encode_jpeg_cmyk_progressive(
                        self.width,
                        self.height,
                        data,
                        plane.stride,
                        self.quality,
                        self.cmyk_adobe_transform,
                    )
                } else {
                    encode_jpeg_cmyk(
                        self.width,
                        self.height,
                        data,
                        plane.stride,
                        self.quality,
                        self.cmyk_adobe_transform,
                    )
                }
            }
            // 8-bit YCbCr 4:4:4 / 4:2:2 / 4:2:0 takes the baseline /
            // progressive / arithmetic DCT path.
            _ => {
                let pix = pix.into();
                if self.progressive {
                    encode_jpeg_progressive(v, self.width, self.height, pix, self.quality)
                } else if self.arithmetic {
                    encode_arith_jpeg_yuv(
                        v,
                        self.width,
                        self.height,
                        pix,
                        self.quality,
                        self.restart_interval,
                    )
                } else {
                    encode_jpeg_with_opts(
                        v,
                        self.width,
                        self.height,
                        pix,
                        self.quality,
                        self.restart_interval,
                    )
                }
            }
        };
        out.map_err(Error::from)
    }

    /// [`Self::encode_general`] for an interchange stream (no shared
    /// tables), usable from `&self`.
    fn encode_general_stateless(&self, opts: &EncodeOptions, v: &VideoFrame) -> Result<Vec<u8>> {
        let mut o = opts.clone();
        o.abbreviated = false;
        if o.sampling.is_empty() {
            o.sampling = self.layout.shape(self.chroma).sampling;
        }
        if o.signalling == ColorSignalling::Auto {
            o.signalling = ColorSignalling::Jfif;
        }
        let planes = self.planes_u16(v)?;
        let refs: Vec<&[u16]> = planes.iter().map(|p| p.as_slice()).collect();
        Ok(o.encode(self.width, self.height, &refs)?.data)
    }
}

impl Encoder for MjpegEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(v) => {
                self.check_frame(v)?;
                if self.general_implied {
                    // Knobs may have moved since `from_params`.
                    let o = self.implied_options();
                    self.set_encode_options(o)?;
                    self.general_implied = true;
                }
                let data = match (self.general.clone(), self.layout.legacy) {
                    (Some(opts), _) => self.encode_general(&opts, v)?,
                    (None, Some(pix)) => self.encode_legacy(pix, v)?,
                    (None, None) => {
                        let o = self.implied_options();
                        self.encode_general_stateless(&o, v)?
                    }
                };
                let mut pkt = Packet::new(0, self.time_base, data);
                pkt.pts = v.pts;
                pkt.dts = v.pts;
                pkt.flags.keyframe = true;
                self.pending.push_back(pkt);
                Ok(())
            }
            _ => Err(Error::invalid("MJPEG encoder: video frames only")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}
