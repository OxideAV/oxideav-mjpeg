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
use crate::decoder::{decode_jpeg, decode_jpeg_with_tables};
use crate::encoder::{
    encode_arith_jpeg_grayscale, encode_arith_jpeg_rgb24, encode_arith_jpeg_yuv, encode_jpeg_cmyk,
    encode_jpeg_cmyk_progressive, encode_jpeg_grayscale_with_opts, encode_jpeg_progressive,
    encode_jpeg_progressive_grayscale, encode_jpeg_rgb24_with_opts, encode_jpeg_with_opts,
    encode_lossless_jpeg_grayscale, DEFAULT_QUALITY,
};
use crate::error::MjpegError;
use crate::image::{MjpegFrame, MjpegPixelFormat, MjpegPlane};
use crate::mjpeg_container;
use crate::t81::{ColorSignalling, HuffmanTables, JpegEncodeOptions, JpegProcess, JpegTableSet};
use crate::CODEC_ID_STR;

// ---- Error / pixel-format / frame conversions --------------------------

impl From<MjpegError> for Error {
    fn from(e: MjpegError) -> Self {
        match e {
            MjpegError::InvalidData(s) => Error::InvalidData(s),
            MjpegError::Unsupported(s) => Error::Unsupported(s),
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
            MjpegPixelFormat::Yuv420P12Le => PixelFormat::Yuv420P12Le,
            MjpegPixelFormat::Yuv422P12Le => PixelFormat::Yuv422P12Le,
            MjpegPixelFormat::Yuv444P12Le => PixelFormat::Yuv444P12Le,
        }
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

// ---- CodecRegistry / ContainerRegistry entry points --------------------

/// Register the JPEG / MJPEG codec (decoder + encoder) into the
/// supplied [`CodecRegistry`].
///
/// Kept as a free function (rather than a method on a registry handle)
/// so it matches the registration shape used by the umbrella
/// `oxideav` crate.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("mjpeg_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_size(16384, 16384);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder)
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
        // With the `registry` feature on, `decode_jpeg` already
        // returns `oxideav_core::VideoFrame` (see the conditional
        // alias in `decoder.rs`), so the trait surface needs nothing
        // more than wrapping it in `Frame::Video`.
        let vf = match &self.tables {
            Some(t) => decode_jpeg_with_tables(t, &pkt.data, pkt.pts)?,
            None => decode_jpeg(&pkt.data, pkt.pts)?,
        };
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

/// The `CodecOptions` schema of the registry encoder — the string-bag
/// twin of [`JpegEncodeOptions`]. Any key present in
/// `CodecParameters::options` routes the encoder through the general
/// T.81 writer (`oxideav_mjpeg::t81`); an empty bag keeps the historical
/// per-format paths.
///
/// | key               | kind                                   | default      |
/// |-------------------|----------------------------------------|--------------|
/// | `quality`         | 1..=100                                | 75           |
/// | `tables`          | `typical` \| `optimal`                 | `typical`    |
/// | `process`         | `sequential` \| `progressive` \| `lossless` | `sequential` |
/// | `precision`       | 8 / 12 (DCT), 2..=16 (lossless); must match the pixel format | from the pixel format |
/// | `restart`         | restart interval in MCUs               | 0            |
/// | `abbreviated`     | §B.5 table-less frames, tables in `extradata` | false  |
/// | `predictor`       | Table H.1 selector 1..=7 (lossless)    | 1            |
/// | `point_transform` | `Pt` (lossless)                        | 0            |
/// | `sampling`        | `HxV[,HxV…]` per component; must match the pixel format | from the pixel format |
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MjpegEncoderOptions {
    pub quality: u32,
    pub tables: String,
    pub process: String,
    /// `0` = take the precision the pixel format implies.
    pub precision: u32,
    pub restart: u32,
    pub abbreviated: bool,
    pub predictor: u32,
    pub point_transform: u32,
    pub sampling: String,
}

impl Default for MjpegEncoderOptions {
    fn default() -> Self {
        MjpegEncoderOptions {
            quality: u32::from(DEFAULT_QUALITY),
            tables: "typical".into(),
            process: "sequential".into(),
            precision: 0,
            restart: 0,
            abbreviated: false,
            predictor: 1,
            point_transform: 0,
            sampling: String::new(),
        }
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
            kind: OptionKind::Enum(&["sequential", "progressive", "lossless"]),
            default: OptionValue::String(String::new()),
            help: "coding process: sequential DCT (SOF0/SOF1), progressive DCT (SOF2) or lossless (SOF3)",
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
    /// encoder.
    pub fn to_encode_options(&self) -> Result<JpegEncodeOptions> {
        if !(1..=100).contains(&self.quality) {
            return Err(Error::invalid("MJPEG encoder: quality must be in 1..=100"));
        }
        if self.restart > u32::from(u16::MAX) {
            return Err(Error::invalid(
                "MJPEG encoder: restart interval exceeds 65535",
            ));
        }
        let tables = match self.tables.as_str() {
            "typical" => HuffmanTables::Typical,
            "optimal" => HuffmanTables::Optimal,
            other => return Err(Error::invalid(format!("MJPEG encoder: tables '{other}'"))),
        };
        let process = match self.process.as_str() {
            "sequential" => JpegProcess::Sequential,
            "progressive" => JpegProcess::Progressive,
            "lossless" => {
                if !(1..=7).contains(&self.predictor) {
                    return Err(Error::invalid("MJPEG encoder: predictor must be in 1..=7"));
                }
                if self.point_transform > 15 {
                    return Err(Error::invalid(
                        "MJPEG encoder: point_transform must be ≤ 15",
                    ));
                }
                JpegProcess::Lossless {
                    predictor: self.predictor as u8,
                    point_transform: self.point_transform as u8,
                }
            }
            other => return Err(Error::invalid(format!("MJPEG encoder: process '{other}'"))),
        };
        Ok(JpegEncodeOptions {
            quality: self.quality as u8,
            tables,
            process,
            precision: self.precision.min(16) as u8,
            restart_interval: self.restart as u16,
            abbreviated: self.abbreviated,
            signalling: ColorSignalling::Auto,
            sampling: parse_sampling(&self.sampling)?,
            table_ids: Vec::new(),
        })
    }
}

/// What a pixel format implies for the general writer: sample
/// precision, per-component sampling factors, component count and
/// bytes per sample.
fn pix_layout(pix: MjpegPixelFormat) -> (u8, Vec<(u8, u8)>, usize, usize) {
    use MjpegPixelFormat as P;
    match pix {
        P::Gray8 => (8, vec![], 1, 1),
        P::Gray10Le => (10, vec![], 1, 2),
        P::Gray12Le => (12, vec![], 1, 2),
        P::Gray16Le => (16, vec![], 1, 2),
        P::Rgb24 => (8, vec![], 3, 1),
        P::Cmyk => (8, vec![], 4, 1),
        P::Yuv444P => (8, vec![], 3, 1),
        P::Yuv422P => (8, vec![(2, 1), (1, 1), (1, 1)], 3, 1),
        P::Yuv420P => (8, vec![(2, 2), (1, 1), (1, 1)], 3, 1),
        P::Yuv411P => (8, vec![(4, 1), (1, 1), (1, 1)], 3, 1),
        P::Yuv444P12Le => (12, vec![], 3, 2),
        P::Yuv422P12Le => (12, vec![(2, 1), (1, 1), (1, 1)], 3, 2),
        P::Yuv420P12Le => (12, vec![(2, 2), (1, 1), (1, 1)], 3, 2),
        // Never reaches the general writer (from_params rejects them).
        P::Rgb48Le | P::Gbrp10Le | P::Gbrp12Le | P::Gbrp14Le => (16, vec![], 3, 2),
    }
}

/// JPEG encoder. Emits one self-contained JPEG bitstream (baseline SOF0
/// or progressive SOF2) per video frame.
pub struct MjpegEncoder {
    output_params: CodecParameters,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pix: MjpegPixelFormat,
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
    general: Option<JpegEncodeOptions>,
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
    pub fn from_params(params: &CodecParameters) -> Result<Box<Self>> {
        let width = params
            .width
            .ok_or_else(|| Error::invalid("MJPEG encoder: missing width"))?;
        let height = params
            .height
            .ok_or_else(|| Error::invalid("MJPEG encoder: missing height"))?;
        let pix_core = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
        let pix = pix_to_local(pix_core).ok_or_else(|| {
            Error::unsupported(format!(
                "MJPEG encoder: pixel format {pix_core:?} not supported"
            ))
        })?;
        match pix {
            MjpegPixelFormat::Yuv420P | MjpegPixelFormat::Yuv422P | MjpegPixelFormat::Yuv444P => {}
            // Grayscale takes the lossless (SOF3) path when requested via
            // `set_lossless(true)`. Accepting it here lets callers wire a
            // grayscale `CodecParameters` through the trait API directly
            // rather than dropping to the standalone `encode_lossless_*`
            // function.
            MjpegPixelFormat::Gray8
            | MjpegPixelFormat::Gray10Le
            | MjpegPixelFormat::Gray12Le
            | MjpegPixelFormat::Gray16Le => {}
            // 4-component CMYK / YCCK input takes the dedicated CMYK
            // encode path (baseline SOF0 by default, SOF2 when
            // `set_progressive(true)` is used). Adobe APP14 transform
            // selection comes from `set_adobe_transform`; default is
            // no APP14 (plain "regular" CMYK).
            MjpegPixelFormat::Cmyk => {}
            // Packed `Rgb24` input takes the baseline-SOF0 RGB encode
            // path: three components at IDs 'R'/'G'/'B', all H = V = 1,
            // all bound to one quant table + one DC/AC Huffman pair.
            // Adobe APP14 with transform = 0 is emitted so any
            // conformant decoder honouring the colour-transform flag
            // round-trips the samples as plain R/G/B. Progressive (SOF2)
            // emission of RGB stays a follow-up for now; the lossless
            // (SOF3) path is already available via `set_lossless(true)`
            // on the existing 3-component lossless encoder if a caller
            // needs bit-exactness.
            MjpegPixelFormat::Rgb24 => {}
            _ => {
                return Err(Error::unsupported(format!(
                    "MJPEG encoder: pixel format {pix_core:?} not supported"
                )))
            }
        }

        let mut output_params = params.clone();
        output_params.media_type = MediaType::Video;
        output_params.codec_id = CodecId::new(CODEC_ID_STR);
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(pix.into());

        let mut enc = Self {
            output_params,
            width,
            height,
            pix,
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
            shared_tables: None,
            time_base: params
                .frame_rate
                .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num)),
            pending: VecDeque::new(),
            eof: false,
        };
        // Any option key routes through the general T.81 writer with the
        // typed options the string bag denotes (strict: unknown keys and
        // malformed values are rejected here).
        if !params.options.is_empty() {
            let opts = parse_options::<MjpegEncoderOptions>(&params.options)?;
            enc.set_encode_options(opts.to_encode_options()?)?;
        }
        Ok(Box::new(enc))
    }

    /// Route every frame through the general T.81 writer
    /// (`oxideav_mjpeg::t81`) with `opts` — sequential / progressive /
    /// lossless, 8- or 12-bit, optimal tables, restarts, abbreviated
    /// streams. `precision = 0` and an empty `sampling` take the values
    /// the pixel format implies; non-zero / non-empty values must match
    /// them. With `abbreviated` the shared §B.5 tables stream is
    /// published in `output_params().extradata` — immediately for
    /// typical tables, after the first frame for optimal ones (the
    /// statistics of the first frame define the shared set).
    pub fn set_encode_options(&mut self, mut opts: JpegEncodeOptions) -> Result<()> {
        let (precision, sampling, nf, _) = pix_layout(self.pix);
        if opts.precision == 0 {
            opts.precision = precision;
        } else if opts.precision != precision {
            return Err(Error::invalid(format!(
                "MJPEG encoder: precision {} does not match the {:?} input (P = {precision})",
                opts.precision, self.pix
            )));
        }
        if opts.sampling.is_empty() {
            opts.sampling = sampling;
        } else if opts.sampling != sampling
            && !(sampling.is_empty() && opts.sampling.iter().all(|&s| s == (1, 1)))
        {
            return Err(Error::invalid(format!(
                "MJPEG encoder: sampling {:?} does not match the {:?} input",
                opts.sampling, self.pix
            )));
        }
        if opts.signalling == ColorSignalling::Auto {
            opts.signalling = match self.pix {
                MjpegPixelFormat::Rgb24 => ColorSignalling::Rgb,
                MjpegPixelFormat::Cmyk => ColorSignalling::Cmyk {
                    adobe_transform: self.cmyk_adobe_transform,
                },
                _ => ColorSignalling::Jfif,
            };
        }
        if opts.process.is_dct() && !matches!(opts.precision, 8 | 12) {
            return Err(Error::unsupported(format!(
                "MJPEG encoder: the DCT processes need P = 8 or 12 (input {:?} is P = {})",
                self.pix, opts.precision
            )));
        }
        self.shared_tables = None;
        self.output_params.extradata.clear();
        if opts.abbreviated
            && opts.tables == HuffmanTables::Typical
            && !(opts.process.is_dct() && opts.precision > 8)
        {
            let t = JpegTableSet::typical(opts.quality, opts.precision, !opts.process.is_dct(), nf);
            self.output_params.extradata = t.tables_stream(opts.process.is_dct());
            self.shared_tables = Some(t);
        }
        self.general = Some(opts);
        Ok(())
    }

    /// The general-writer options in force, if any.
    pub fn encode_options(&self) -> Option<&JpegEncodeOptions> {
        self.general.as_ref()
    }

    /// Split a frame into per-component `u16` planes at the A.1.1
    /// resolutions the pixel format implies.
    fn planes_u16(&self, v: &VideoFrame) -> Result<Vec<Vec<u16>>> {
        let (_, sampling, nf, bps) = pix_layout(self.pix);
        let (w, h) = (self.width as usize, self.height as usize);
        let packed = matches!(self.pix, MjpegPixelFormat::Rgb24 | MjpegPixelFormat::Cmyk);
        let need_planes = if packed { 1 } else { nf };
        if v.planes.len() < need_planes {
            return Err(Error::invalid(format!(
                "MJPEG encoder: {:?} frame needs {need_planes} plane(s), got {}",
                self.pix,
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
        if packed {
            let pl = &v.planes[0];
            if pl.stride < w * nf * bps || pl.data.len() < pl.stride * (h - 1) + w * nf * bps {
                return Err(Error::invalid("MJPEG encoder: packed plane too small"));
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
            let hm = sampling.iter().map(|s| s.0).max().unwrap_or(1) as usize;
            let vm = sampling.iter().map(|s| s.1).max().unwrap_or(1) as usize;
            for c in 0..nf {
                let (hi, vi) = sampling.get(c).copied().unwrap_or((1, 1));
                let cw = (w * hi as usize).div_ceil(hm);
                let ch = (h * vi as usize).div_ceil(vm);
                let pl = &v.planes[c];
                if pl.stride < cw * bps || pl.data.len() < pl.stride * (ch - 1) + cw * bps {
                    return Err(Error::invalid(format!(
                        "MJPEG encoder: plane {c} too small for {cw}x{ch}"
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
        Ok(out)
    }

    /// Encode one frame through the general writer.
    fn encode_general(&mut self, opts: &JpegEncodeOptions, v: &VideoFrame) -> Result<Vec<u8>> {
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
    /// The three higher-precision grayscale variants
    /// (`Gray10Le` / `Gray12Le` / `Gray16Le`) require `set_lossless(true)`
    /// — the DCT path is 8-bit by spec.
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
        Ok(())
    }

    /// Current Adobe APP14 colour-transform marker selection.
    pub fn adobe_transform(&self) -> Option<u8> {
        self.cmyk_adobe_transform
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
                // With the `registry` feature on, the public
                // `encode_jpeg_*` functions already accept
                // `&oxideav_core::VideoFrame` directly (see the
                // conditional alias in `encoder.rs`), so we can pass
                // the frame through without local-type bounce.
                let pix = self.pix.into();
                let data = if let Some(opts) = self.general.clone() {
                    self.encode_general(&opts, v)?
                } else {
                    match (self.pix, self.lossless) {
                        // Grayscale + lossless → SOF3 path. Precision is
                        // implied by the pixel format and we read row bytes
                        // straight from plane 0.
                        (MjpegPixelFormat::Gray8, true)
                        | (MjpegPixelFormat::Gray10Le, true)
                        | (MjpegPixelFormat::Gray12Le, true)
                        | (MjpegPixelFormat::Gray16Le, true) => {
                            if v.planes.is_empty() {
                                return Err(Error::invalid(
                                    "MJPEG encoder: grayscale frame missing plane 0",
                                ));
                            }
                            let plane = &v.planes[0];
                            let precision: u8 = match self.pix {
                                MjpegPixelFormat::Gray8 => 8,
                                MjpegPixelFormat::Gray10Le => 10,
                                MjpegPixelFormat::Gray12Le => 12,
                                MjpegPixelFormat::Gray16Le => 16,
                                _ => unreachable!(),
                            };
                            encode_lossless_jpeg_grayscale(
                                self.width,
                                self.height,
                                &plane.data,
                                plane.stride,
                                precision,
                                self.lossless_predictor,
                            )?
                        }
                        // 8-bit grayscale without lossless mode takes the
                        // baseline (SOF0) or progressive (SOF2) single-
                        // component DCT path. The baseline bitstream layout
                        // mirrors `encode_jpeg` reduced to one luma component
                        // (one DQT + DC/AC luma Huffman tables + a one-entry
                        // SOS); flipping `set_progressive(true)` takes the
                        // matching SOF2 path (DC + AC-low + AC-high scans,
                        // spectral-selection decomposition). Either way any
                        // conformant decoder produces a `Gray8` frame
                        // round-tripping with the usual DCT-quantise
                        // distortion floor. `restart_interval` is ignored
                        // on the progressive path because the 3-component
                        // progressive encoder doesn't expose DRI emission
                        // either — kept consistent so the flag has the same
                        // meaning across every progressive variant.
                        (MjpegPixelFormat::Gray8, false) => {
                            if v.planes.is_empty() {
                                return Err(Error::invalid(
                                    "MJPEG encoder: grayscale frame missing plane 0",
                                ));
                            }
                            let plane = &v.planes[0];
                            if self.progressive {
                                encode_jpeg_progressive_grayscale(
                                    self.width,
                                    self.height,
                                    &plane.data,
                                    plane.stride,
                                    self.quality,
                                )?
                            } else if self.arithmetic {
                                encode_arith_jpeg_grayscale(
                                    self.width,
                                    self.height,
                                    &plane.data,
                                    plane.stride,
                                    self.quality,
                                    self.restart_interval,
                                )?
                            } else {
                                encode_jpeg_grayscale_with_opts(
                                    self.width,
                                    self.height,
                                    &plane.data,
                                    plane.stride,
                                    self.quality,
                                    self.restart_interval,
                                )?
                            }
                        }
                        // Higher-precision grayscale (10 / 12 / 16-bit)
                        // still requires `set_lossless(true)` — the
                        // baseline DCT path is 8-bit by spec. Surface a
                        // clear error rather than silently downgrading.
                        (
                            MjpegPixelFormat::Gray10Le
                            | MjpegPixelFormat::Gray12Le
                            | MjpegPixelFormat::Gray16Le,
                            false,
                        ) => {
                            return Err(Error::unsupported(
                            "MJPEG encoder: high-bit-depth grayscale input requires set_lossless(true)",
                        ));
                        }
                        // Packed `Rgb24` input takes the baseline-SOF0 RGB
                        // path. The single plane is laid out as
                        // `[R, G, B]` at 3 bytes per pixel, matching the
                        // decoder's `Rgb24` output shape. Progressive
                        // (SOF2) RGB is not yet wired in here — flipping
                        // `set_progressive(true)` with `Rgb24` input still
                        // takes the baseline path.
                        (MjpegPixelFormat::Rgb24, _) => {
                            if v.planes.is_empty() {
                                return Err(Error::invalid(
                                    "MJPEG encoder: RGB24 frame missing plane 0",
                                ));
                            }
                            let plane = &v.planes[0];
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
                                )?
                            } else {
                                encode_jpeg_rgb24_with_opts(
                                    self.width,
                                    self.height,
                                    &plane.data,
                                    plane.stride,
                                    self.quality,
                                    self.restart_interval,
                                )?
                            }
                        }
                        // 4-component CMYK / YCCK input takes the dedicated
                        // CMYK encode path. The single packed plane is laid
                        // out as `[C, M, Y, K]` (or `[Y, Cb, Cr, K]` for
                        // `set_adobe_transform(Some(2))`) at 4 bytes per
                        // pixel, matching the decoder's output shape.
                        (MjpegPixelFormat::Cmyk, _) => {
                            if v.planes.is_empty() {
                                return Err(Error::invalid(
                                    "MJPEG encoder: CMYK frame missing plane 0",
                                ));
                            }
                            let plane = &v.planes[0];
                            let min_stride = (self.width as usize) * 4;
                            if plane.stride < min_stride {
                                return Err(Error::invalid(
                                    "MJPEG encoder: CMYK plane stride must be at least width * 4",
                                ));
                            }
                            if self.progressive {
                                encode_jpeg_cmyk_progressive(
                                    self.width,
                                    self.height,
                                    &plane.data,
                                    plane.stride,
                                    self.quality,
                                    self.cmyk_adobe_transform,
                                )?
                            } else {
                                encode_jpeg_cmyk(
                                    self.width,
                                    self.height,
                                    &plane.data,
                                    plane.stride,
                                    self.quality,
                                    self.cmyk_adobe_transform,
                                )?
                            }
                        }
                        // YUV inputs take the baseline / progressive / arithmetic
                        // DCT path.
                        _ => {
                            if self.progressive {
                                encode_jpeg_progressive(
                                    v,
                                    self.width,
                                    self.height,
                                    pix,
                                    self.quality,
                                )?
                            } else if self.arithmetic {
                                encode_arith_jpeg_yuv(
                                    v,
                                    self.width,
                                    self.height,
                                    pix,
                                    self.quality,
                                    self.restart_interval,
                                )?
                            } else {
                                encode_jpeg_with_opts(
                                    v,
                                    self.width,
                                    self.height,
                                    pix,
                                    self.quality,
                                    self.restart_interval,
                                )?
                            }
                        }
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
