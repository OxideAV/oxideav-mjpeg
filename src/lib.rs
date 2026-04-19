// Parallel-array index loops are idiomatic in codec code; skip the lint.
#![allow(clippy::needless_range_loop)]

//! JPEG / Motion-JPEG codec, pure Rust.
//!
//! Each video packet is a standalone JPEG (one full SOI..EOI). The decoder
//! recognises baseline (SOF0), extended-sequential (SOF1), progressive
//! (SOF2), and lossless (SOF3) JPEGs with 4:2:0 / 4:2:2 / 4:4:4 chroma
//! subsampling (DCT variants) and outputs `VideoFrame`s in the matching
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
//! output is interoperable with any compliant JPEG decoder.
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
//! **Not supported** (will return `Error::Unsupported`):
//! - Hierarchical (SOF5+) and arithmetic-coded (SOF9..SOF15) JPEGs
//! - 12-bit progressive (SOF2 with `P=12`)
//! - 12-bit 4:2:2 / 4:4:4 YUV (no matching output `PixelFormat`)
//! - Progressive 4-component JPEGs
//! - Multi-component lossless JPEGs (only grayscale is supported)

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
