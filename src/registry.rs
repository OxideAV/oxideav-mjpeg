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
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag,
    ContainerRegistry, Decoder, Encoder, Error, Frame, MediaType, Packet, PixelFormat, Result,
    TimeBase, VideoFrame,
};

use crate::container;
use crate::decoder::decode_jpeg;
use crate::encoder::{encode_jpeg_progressive, encode_jpeg_with_opts, DEFAULT_QUALITY};
use crate::error::MjpegError;
use crate::image::{MjpegFrame, MjpegPixelFormat, MjpegPlane};
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
            MjpegPixelFormat::Yuv411P => PixelFormat::Yuv411P,
            MjpegPixelFormat::Yuv420P => PixelFormat::Yuv420P,
            MjpegPixelFormat::Yuv422P => PixelFormat::Yuv422P,
            MjpegPixelFormat::Yuv444P => PixelFormat::Yuv444P,
            MjpegPixelFormat::Yuv420P12Le => PixelFormat::Yuv420P12Le,
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
        PixelFormat::Yuv411P => MjpegPixelFormat::Yuv411P,
        PixelFormat::Yuv420P => MjpegPixelFormat::Yuv420P,
        PixelFormat::Yuv422P => MjpegPixelFormat::Yuv422P,
        PixelFormat::Yuv444P => MjpegPixelFormat::Yuv444P,
        PixelFormat::Yuv420P12Le => MjpegPixelFormat::Yuv420P12Le,
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

/// Backward-compatible alias for [`register_codecs`]. The crate has
/// shipped this name through 0.1.x and downstream callers (the
/// umbrella `oxideav` crate, oxideav-cli, etc.) reference it
/// unchanged.
pub fn register(reg: &mut CodecRegistry) {
    register_codecs(reg);
}

/// Register the still-image JPEG container (`.jpg` / `.jpeg`). Must be
/// called alongside [`register`] when wiring up a pipeline that expects
/// to read or write raw JPEG files.
pub fn register_containers(reg: &mut ContainerRegistry) {
    container::register(reg);
}

// ---- Decoder trait impl ------------------------------------------------

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let codec_id = params.codec_id.clone();
    Ok(Box::new(MjpegDecoder {
        codec_id,
        pending: None,
        eof: false,
    }))
}

struct MjpegDecoder {
    codec_id: CodecId,
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
        let vf = decode_jpeg(&pkt.data, pkt.pts)?;
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

        Ok(Box::new(Self {
            output_params,
            width,
            height,
            pix,
            quality: DEFAULT_QUALITY,
            restart_interval: 0,
            progressive: false,
            time_base: params
                .frame_rate
                .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num)),
            pending: VecDeque::new(),
            eof: false,
        }))
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
                let data = if self.progressive {
                    encode_jpeg_progressive(v, self.width, self.height, pix, self.quality)?
                } else {
                    encode_jpeg_with_opts(
                        v,
                        self.width,
                        self.height,
                        pix,
                        self.quality,
                        self.restart_interval,
                    )?
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
