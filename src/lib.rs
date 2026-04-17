// Parallel-array index loops are idiomatic in codec code; skip the lint.
#![allow(clippy::needless_range_loop)]

//! JPEG / Motion-JPEG codec, pure Rust.
//!
//! Each video packet is a standalone JPEG (one full SOI..EOI). The decoder
//! recognises both baseline (SOF0) and progressive (SOF2) 8-bit JPEGs with
//! 4:2:0 / 4:2:2 / 4:4:4 chroma subsampling and outputs `VideoFrame`s in
//! the corresponding `Yuv*P` pixel format (or `Gray8` for 1-component
//! streams). Progressive scans accumulate DCT coefficients across multiple
//! SOS segments using both spectral selection and successive
//! approximation; the inverse DCT runs once after EOI. The encoder accepts
//! the same pixel formats and produces a standalone baseline JPEG using
//! the Annex K "typical" Huffman tables, so its output is interoperable
//! with any compliant JPEG decoder.
//!
//! **Not supported** (will return `Error::Unsupported`):
//! - Hierarchical, arithmetic coding, lossless
//! - 12-bit precision
//! - CMYK / 4-component scans
//! - Non-interleaved baseline scans (progressive permits them, as
//!   required by the spec)

pub mod container;
pub mod decoder;
pub mod encoder;
pub mod jpeg;

use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

pub const CODEC_ID_STR: &str = "mjpeg";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::video("mjpeg_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_size(16384, 16384);
    reg.register_both(cid, caps, decoder::make_decoder, encoder::make_encoder);
}

/// Register the still-image JPEG container (`.jpg` / `.jpeg`). Must be
/// called alongside [`register`] when wiring up a pipeline that expects
/// to read or write raw JPEG files.
pub fn register_containers(reg: &mut ContainerRegistry) {
    container::register(reg);
}
