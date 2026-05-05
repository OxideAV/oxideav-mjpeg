// Parallel-array index loops are idiomatic in codec code; skip the lint.
#![allow(clippy::needless_range_loop)]
// When built without the `registry` feature, large swathes of the
// decoder/encoder (the JPEG entry-point + render helpers) have no
// callers — they're only reachable via the `Decoder` / `Encoder`
// trait implementations that live behind the `registry` feature.
// Suppress the resulting dead-code warnings rather than gating every
// helper.
#![cfg_attr(not(feature = "registry"), allow(dead_code))]

//! JPEG / Motion-JPEG codec, pure Rust.
//!
//! Each video packet is a standalone JPEG (one full SOI..EOI). The decoder
//! recognises baseline (SOF0), extended-sequential (SOF1), progressive
//! (SOF2), and lossless (SOF3) JPEGs with 4:2:0 / 4:2:2 / 4:4:4 chroma
//! subsampling (DCT variants) and outputs `MjpegFrame`s in the matching
//! `Yuv*P` pixel format or `Gray8` for 1-component streams. SOF0 and
//! SOF1 share the same Huffman sequential scan structure at 8-bit
//! precision; both are handled by the same scan decoder, and
//! non-interleaved sequential scans (one SOS per component) fall back
//! to the same coefficient-accumulator path used by the progressive
//! decoder. Progressive scans accumulate DCT coefficients across
//! multiple SOS segments using both spectral selection and successive
//! approximation; the inverse DCT runs once after EOI. Restart markers
//! (`RSTn`) and DRI segments are honoured on both paths. APP0..APP15
//! segments (JFIF, EXIF, ICC, XMP, …) are skipped without parsing. The
//! encoder accepts the same pixel formats and produces a standalone
//! baseline JPEG using the Annex K "typical" Huffman tables, so its
//! output is interoperable with any compliant JPEG decoder. A
//! progressive (SOF2) output mode is available via
//! `encoder::MjpegEncoder::set_progressive` or
//! `encoder::encode_jpeg_progressive`; it emits a DC-first scan
//! followed by two per-component AC band scans (spectral selection
//! only, `Ah = Al = 0`).
//!
//! 4-component (CMYK / Adobe YCCK) JPEGs decode to packed `Cmyk` — the
//! decoder inspects the APP14 Adobe transform flag to choose among plain
//! CMYK, Adobe-inverted CMYK, and Adobe YCCK (which is colour-converted
//! back to CMYK via BT.601 full-range YCbCr→RGB→CMY plus K inversion).
//! 12-bit precision sequential JPEGs (SOF0/SOF1 with `P=12`) decode to
//! `Gray12Le` or `Yuv420P12Le`; sample buffers stay 16-bit throughout the
//! inverse DCT and the level shift uses 2048. Lossless JPEGs (SOF3)
//! decode single-component grayscale at any precision in 2..=16 bits
//! via Annex H predictor reconstruction (bit-exact, no DCT); output is
//! `Gray8` at P=8 and `Gray16Le` / `Gray10Le` / `Gray12Le` at wider
//! depths.
//!
//! Extended-sequential arithmetic (SOF9) is decoded via the Q-coder /
//! arithmetic entropy decoder from T.81 Annex D + F.2.4. The DAC marker
//! (Define Arithmetic Conditioning) is parsed when present; if absent the
//! decoder uses the spec defaults `(L=0, U=1)` for DC and `Kx=5` for AC.
//!
//! **Not supported** (will return `Error::Unsupported`):
//! - Hierarchical (SOF5..SOF7, SOF13..SOF15) JPEGs
//! - SOF10..SOF12 / SOF14..SOF15 arithmetic variants (progressive
//!   arithmetic, lossless arithmetic, and the 12-bit arithmetic
//!   precisions)
//! - 12-bit progressive (SOF2 with `P=12`)
//! - 12-bit 4:2:2 / 4:4:4 YUV (no matching output `PixelFormat`)
//! - Progressive 4-component JPEGs
//! - Multi-component lossless JPEGs (only grayscale is supported)
//!
//! ## Standalone vs registry-integrated
//!
//! The crate's default `registry` Cargo feature pulls in `oxideav-core`
//! and exposes the `Decoder` / `Encoder` trait surface, the JPEG-still
//! container, and the [`registry::register`] / [`registry::register_codecs`]
//! / [`registry::register_containers`] entry points. Disable the feature (`default-features = false`) for
//! an oxideav-core-free build that still exposes the standalone
//! [`decoder::decode_jpeg`] API plus crate-local [`MjpegFrame`] /
//! [`MjpegPlane`] / [`MjpegPixelFormat`] / [`MjpegError`] types built
//! only on `std`.

pub mod decoder;
pub mod encoder;
pub mod error;
pub mod image;
pub mod jpeg;

#[cfg(feature = "registry")]
pub mod container;

#[cfg(feature = "registry")]
pub mod registry;

pub const CODEC_ID_STR: &str = "mjpeg";

// Standalone, framework-free API. Available regardless of the
// `registry` feature.
pub use error::{MjpegError, Result};
pub use image::{MjpegFrame, MjpegPixelFormat, MjpegPlane};

// Framework-integrated API (`oxideav-core`-dependent). Gated behind
// `registry` so image-library callers can build the crate without
// dragging in `oxideav-core`.
#[cfg(feature = "registry")]
pub use registry::{register, register_codecs, register_containers};

#[cfg(all(test, feature = "registry"))]
mod register_tests {
    use oxideav_core::RuntimeContext;

    #[test]
    fn register_via_runtime_context_installs_factories() {
        let mut ctx = RuntimeContext::new();
        super::register(&mut ctx);
        assert!(
            ctx.codecs.decoder_ids().next().is_some(),
            "register(ctx) should install codec decoder factories"
        );
        assert!(
            ctx.containers.demuxer_names().next().is_some(),
            "register(ctx) should install container demuxer factories"
        );
    }
}
