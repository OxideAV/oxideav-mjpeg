//! Crate-local error type used by `oxideav-mjpeg`'s standalone
//! (no `oxideav-core`) public API.
//!
//! Defined as a small std-only enum so the crate can be built with the
//! default `registry` feature off — i.e. without depending on
//! `oxideav-core` at all. When the `registry` feature is on (the
//! default) a `From<MjpegError> for oxideav_core::Error` impl is
//! enabled in [`crate::registry`] so the `Decoder`/`Encoder` trait
//! surface still interoperates cleanly.

use core::fmt;

/// `Result` alias scoped to `oxideav-mjpeg`. Standalone (no
/// `oxideav-core`) callers see this; framework callers convert via the
/// gated `From<MjpegError> for oxideav_core::Error` impl.
pub type Result<T> = core::result::Result<T, MjpegError>;

/// Crate-local error type for the JPEG decoder/encoder pipeline.
///
/// Variants mirror the subset of `oxideav_core::Error` the codec can
/// hit. Transport (`Io`) and framework-specific (`FormatNotFound`,
/// `CodecNotFound`) errors are intentionally absent — they originate
/// in callers that are already linking `oxideav-core`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MjpegError {
    /// The bitstream is malformed (bad marker, truncated segment,
    /// invalid Huffman code length, etc.).
    InvalidData(String),
    /// The bitstream uses a feature this codec doesn't implement
    /// (hierarchical SOF, lossless arithmetic, etc.) or the encoder
    /// was asked to emit a frame in a format it doesn't support.
    Unsupported(String),
    /// Catch-all for misuse errors that aren't bitstream-level
    /// (e.g. trait-API contract violations).
    Other(String),
    /// End of stream — no more packets / frames forthcoming.
    Eof,
    /// More input is required before another frame can be produced
    /// (decoder) or another packet can be flushed (encoder).
    NeedMore,
}

impl MjpegError {
    /// Construct an [`MjpegError::InvalidData`] from a stringy message.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Construct an [`MjpegError::Unsupported`] from a stringy message.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }

    /// Construct an [`MjpegError::Other`] from a stringy message.
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

impl fmt::Display for MjpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(s) => write!(f, "invalid data: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Other(s) => write!(f, "other: {s}"),
            Self::Eof => write!(f, "end of stream"),
            Self::NeedMore => write!(f, "need more data"),
        }
    }
}

impl std::error::Error for MjpegError {}
