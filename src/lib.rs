// Parallel-array index loops are idiomatic in codec code; skip the lint.
#![allow(clippy::needless_range_loop)]

//! JPEG / Motion-JPEG codec, pure Rust.
//!
//! Each video packet is a standalone JPEG (one full SOI..EOI). The decoder
//! recognises baseline (SOF0), extended-sequential (SOF1), and progressive
//! (SOF2) 8-bit JPEGs with 4:2:0 / 4:2:2 / 4:4:4 chroma subsampling and
//! outputs `VideoFrame`s in the corresponding `Yuv*P` pixel format (or
//! `Gray8` for 1-component streams). SOF0 and SOF1 share the same Huffman
//! sequential scan structure at 8-bit precision; both are handled by the
//! same scan decoder, and non-interleaved sequential scans (one SOS per
//! component) fall back to the same coefficient-accumulator path used by
//! the progressive decoder. Progressive scans accumulate DCT coefficients across
//! multiple SOS segments using both spectral selection and successive
//! approximation; the inverse DCT runs once after EOI. Restart markers
//! (`RSTn`) and DRI segments are honoured on both paths. APP0..APP15
//! segments (JFIF, EXIF, ICC, XMP, …) are skipped without parsing. The
//! encoder accepts the same pixel formats and produces a standalone
//! baseline JPEG using the Annex K "typical" Huffman tables, so its
//! output is interoperable with any compliant JPEG decoder.
//!
//! **Not supported** (will return `Error::Unsupported`):
//! - Lossless JPEG (SOF3), hierarchical (SOF5+), arithmetic coding
//!   (SOF9..SOF15)
//! - 12-bit precision
//! - CMYK / 4-component scans

pub mod container;
pub mod decoder;
pub mod encoder;
pub mod jpeg;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_container::ContainerRegistry;
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

pub const CODEC_ID_STR: &str = "mjpeg";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("mjpeg_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_size(16384, 16384);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .encoder(encoder::make_encoder)
            .tags([
                // AVI FourCC claims — all unambiguous MJPEG variants.
                CodecTag::fourcc(b"MJPG"),
                CodecTag::fourcc(b"AVRN"),
                CodecTag::fourcc(b"LJPG"),
                CodecTag::fourcc(b"JPGL"),
            ]),
    );
}

/// Register the still-image JPEG container (`.jpg` / `.jpeg`). Must be
/// called alongside [`register`] when wiring up a pipeline that expects
/// to read or write raw JPEG files.
pub fn register_containers(reg: &mut ContainerRegistry) {
    container::register(reg);
}
