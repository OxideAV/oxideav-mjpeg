//! Crate-local frame and pixel-format types used by `oxideav-mjpeg`.
//!
//! Defined here (rather than reusing `oxideav_core::VideoFrame` /
//! `oxideav_core::frame::VideoPlane` / `oxideav_core::PixelFormat`) so
//! the crate can be built with the default `registry` feature off —
//! i.e. without depending on `oxideav-core` at all. When the
//! `registry` feature is on the [`crate::registry`] module provides
//! `From<MjpegFrame> for oxideav_core::VideoFrame` (and the reverse
//! conversion needed by the encoder side) so the `Decoder`/`Encoder`
//! trait surface still interoperates cleanly.
//!
//! The shape of [`MjpegFrame`] / [`MjpegPlane`] mirrors the
//! `oxideav-core` slim-VideoFrame layout (planes are `(stride, data)`
//! tuples; the frame carries an optional PTS and a `Vec` of planes).

/// One image plane: row-major bytes plus the row stride in bytes.
/// Layout-compatible with `oxideav_core::frame::VideoPlane`.
#[derive(Debug, Clone)]
pub struct MjpegPlane {
    /// Bytes per row in `data` (may be larger than the logical row width).
    pub stride: usize,
    /// Raw plane bytes, packed `stride` × number of rows.
    pub data: Vec<u8>,
}

/// Decoded JPEG / MJPEG frame.
///
/// Layout-compatible with `oxideav_core::VideoFrame` (slim shape: a
/// `Vec<MjpegPlane>` plus an optional PTS).
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

/// Subset of `oxideav_core::PixelFormat` the JPEG decoder/encoder
/// produces or accepts. Defined here so the standalone build does not
/// need to pull in `oxideav-core`.
///
/// When the `registry` feature is on, [`crate::registry`] provides a
/// `From<MjpegPixelFormat> for oxideav_core::PixelFormat` impl plus
/// the reverse direction so the trait-side `CodecParameters` /
/// `VideoFrame` flow keeps using the framework's canonical type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MjpegPixelFormat {
    /// 8-bit single-channel grayscale.
    Gray8,
    /// 10-bit single-channel grayscale, little-endian (16-bit storage).
    Gray10Le,
    /// 12-bit single-channel grayscale, little-endian (16-bit storage).
    Gray12Le,
    /// 16-bit single-channel grayscale, little-endian.
    Gray16Le,
    /// 8-bit packed CMYK (4 bytes per pixel).
    Cmyk,
    /// 8-bit planar 4:1:1 YUV (luma 4× chroma horizontally).
    Yuv411P,
    /// 8-bit planar 4:2:0 YUV.
    Yuv420P,
    /// 8-bit planar 4:2:2 YUV.
    Yuv422P,
    /// 8-bit planar 4:4:4 YUV.
    Yuv444P,
    /// 12-bit planar 4:2:0 YUV (16-bit storage per sample, little-endian).
    Yuv420P12Le,
}
