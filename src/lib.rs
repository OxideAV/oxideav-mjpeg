// Parallel-array index loops are idiomatic in codec code; skip the lint.
#![allow(clippy::needless_range_loop)]
// When built without the `registry` feature, parts of the encoder
// (helpers only reachable from the `Encoder` trait implementation that
// lives behind the `registry` feature) have no callers. Suppress the
// resulting dead-code warnings rather than gating every helper.
#![cfg_attr(not(feature = "registry"), allow(dead_code))]

//! JPEG / Motion-JPEG codec, pure Rust.
//!
//! ## Still images — the standalone API
//!
//! The crate root follows the OxideAV image-crate contract and builds
//! with `default-features = false` (no `oxideav-core`):
//!
//! ```no_run
//! # fn main() -> Result<(), oxideav_mjpeg::Error> {
//! let bytes = std::fs::read("in.jpg").map_err(oxideav_mjpeg::Error::from)?;
//! if oxideav_mjpeg::probe(&bytes) {
//!     let info = oxideav_mjpeg::info(&bytes)?;          // header only
//!     let img = oxideav_mjpeg::decode(&bytes)?;         // JpegImage, native layout
//!     let rgba: Vec<u8> = img.to_rgba8();               // packed RGBA, 4 × width bytes per row
//!     let (w, h) = (img.width(), img.height());
//!     let _ = info;
//!
//!     let opts = oxideav_mjpeg::EncodeOptions::default().with_quality(90);
//!     let out: Vec<u8> = oxideav_mjpeg::encode_rgba8(w, h, &rgba, &opts)?; // alpha dropped
//!     std::fs::write("out.jpg", out).map_err(oxideav_mjpeg::Error::from)?;
//! }
//! # Ok(()) }
//! ```
//!
//! [`probe`] / [`info`] / [`decode`] / [`decode_with`] / [`decode_rgb8`]
//! / [`decode_rgba8`] / [`decode_from`] and [`encode`] / [`encode_rgb8`]
//! / [`encode_rgba8`] / [`encode_to`] over [`JpegImage`], [`RgbImage`],
//! [`RgbaImage`], [`Plane`], [`ColorInfo`], [`Metadata`], [`ImageInfo`],
//! [`EncodeOptions`], [`DecodeOptions`], [`PixelFormat`] and
//! [`Error`]. The decoder below is the same one the Motion-JPEG video
//! path uses; the still-image API is a second door to it, and the
//! `registry` adapter is a thin layer over these functions.
//!
//! ## Motion-JPEG — the framework video codec
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
//! only, `Ah = Al = 0`). A lossless (SOF3) grayscale output mode is
//! available via `encoder::encode_lossless_jpeg_grayscale` (or
//! `MjpegEncoder::set_lossless(true)` on the trait-API encoder):
//! every precision `P ∈ 2..=16` and every Annex H Table H.1 spatial
//! predictor `1..=7` are supported, and the bitstream round-trips
//! bit-exact through the matching SOF3 decoder.
//!
//! 4-component (CMYK / Adobe YCCK) JPEGs decode to packed `Cmyk` — the
//! decoder inspects the APP14 Adobe transform flag to choose among plain
//! CMYK, Adobe-inverted CMYK, and Adobe YCCK (which is colour-converted
//! back to CMYK via BT.601 full-range YCbCr→RGB→CMY plus K inversion).
//! The 4-component path covers both the sequential (SOF0 / SOF1) and the
//! progressive (SOF2) scan decompositions at `P = 8`. 12-bit precision
//! sequential JPEGs (SOF0/SOF1 with `P=12`) decode to `Gray12Le` /
//! `Yuv444P12Le` / `Yuv422P12Le` / `Yuv420P12Le`; sample buffers stay
//! 16-bit throughout the inverse DCT and the level shift uses 2048.
//! Lossless JPEGs (SOF3) decode at every precision `P ∈ 2..=16` via
//! Annex H predictor reconstruction (bit-exact, no DCT). Single-component
//! grayscale output: `Gray8` at `P = 8`, `Gray10Le` / `Gray12Le` at
//! `P = 10`/12, `Gray16Le` everywhere else. Three-component (RGB-class,
//! `H_i = V_i = 1`) output: packed `Rgb24` at `P = 8`, planar
//! `Gbrp10Le` / `Gbrp12Le` / `Gbrp14Le` at `P = 10`/12/14, packed
//! `Rgb48Le` at every other precision in the valid range.
//!
//! Extended-sequential arithmetic (SOF9) is decoded via the Q-coder /
//! arithmetic entropy decoder from T.81 Annex D + F.2.4. The DAC marker
//! (Define Arithmetic Conditioning) is parsed when present; if absent the
//! decoder uses the spec defaults `(L=0, U=1)` for DC / lossless
//! conditioning and `Kx=5` for AC. Progressive arithmetic (SOF10) is
//! decoded via the same Q-coder under the T.81 §G.1.3 procedures: DC
//! first scans reuse the §F.1.4.1 model on the point-transformed
//! values, DC refinement bits use the fixed 0.5 estimate, AC first
//! scans run the §F.1.4 procedure per band (`Kmin = Ss`, EOB =
//! end-of-band), and AC refinement scans follow the §G.1.3.3 model
//! (Figures G.10 / G.11, Table G.2) — at `P = 8` and `P = 12` with
//! the same output shaping as the Huffman progressive (SOF2) path,
//! 4-component CMYK / YCCK included at `P = 8`. Lossless arithmetic
//! (SOF11) is decoded via the same Q-coder under the two-dimensional
//! statistical model of §H.1.2.3 (binary decisions conditioned on the
//! left / above difference classifications through the Figure H.2
//! array), covering the full Annex H surface the SOF3 path handles:
//! every precision, every predictor, point transform and restart
//! intervals.
//!
//! Hierarchical-mode (DHP-introduced) JPEGs are decoded across both
//! progression families (T.81 Annex J / §K.7.2): the spatial-lossless
//! progression (non-differential SOF3 / SOF11 first frame + differential
//! SOF7 / SOF15 refinements) and the DCT progression (non-differential
//! SOF0 / SOF1 / SOF2 Huffman or SOF9 / SOF10 arithmetic first frame +
//! differential SOF5 / SOF6 Huffman or SOF13 / SOF14 arithmetic
//! refinements, optionally SOF7 / SOF15-terminated). EXP ×2 upsampling,
//! the §J.2.1 modulo-2^16 reference reconstruction and the §J.2.3.1
//! differential coding model (direct DC, no level shift) are all handled.
//!
//! On the **encode** path, SOF9 (sequential arithmetic DCT) grayscale is
//! produced by [`encoder::encode_arith_jpeg_grayscale`] (with optional
//! restart-interval framing) alongside the baseline / progressive /
//! lossless (Huffman and arithmetic) encoders. **Hierarchical mode is
//! encoded too** (T.81 Annex J): the spatial-lossless progression
//! (`DHP` + `SOF3`/`SOF11` non-differential + `EXP` + `SOF7`/`SOF15`
//! differential — bit-exact, 1/3/4-component, `P ∈ 2..=16`) via
//! [`encoder::encode_hierarchical_lossless_jpeg_grayscale`] and friends,
//! and the DCT progression (§K.7.2.1 — sequential `SOF0`+`SOF5`,
//! progressive `SOF2`+`SOF6`, arithmetic `SOF9`+`SOF13` and
//! `SOF10`+`SOF14` stage families, each optionally terminated by a
//! differential lossless `SOF7`/`SOF15` frame for bit-exact output) via
//! [`encoder::encode_hierarchical_dct_jpeg_grayscale`] and friends — the
//! encoder emits every SOFn family the hierarchical decoder consumes.
//!
//! **Not supported** (will return `Error::Unsupported`):
//! - Bare differential frames (SOF5..SOF7 / SOF13..SOF15) **without** a
//!   preceding DHP marker — a differential frame is only meaningful inside
//!   a hierarchical sequence.
//! - 12-bit progressive 4-component JPEGs (the workspace `PixelFormat`
//!   enum has no 12-bit CMYK variant; `P=8` 4-component CMYK / YCCK
//!   *is* supported on both the sequential and progressive scan
//!   decompositions).
//!
//! **The general T.81 writer** lives in [`t81`] and is the one JPEG
//! encoder sibling crates build on: [`t81::EncodeOptions`] (typed
//! options → [`t81::EncodeOptions::encode`]) and the frame-level
//! [`t81::encode_frame`] / [`t81::gather_stats`] / [`t81::JpegTableSet`]
//! API code sequential (`SOF0` / `SOF1`, `P = 8` / `12`), progressive
//! (`SOF2`) and lossless (`SOF3`, `P ∈ 2..=16`) frames with any §A.1.1
//! sampling layout, per-component table destinations, restart
//! intervals, Annex K.2 optimal Huffman tables and the §B.5 abbreviated
//! (`JPEGTables`) streams; [`decoder::decode_jpeg_with_tables`] is the
//! matching abbreviated-stream decode entry point.
//!
//! Motion-JPEG carried over RTP (RFC 2435) is supported on the decode
//! path via [`rtp::JpegDepacketizer`], which reassembles fragmented
//! RTP/JPEG payloads and reconstructs the absent SOI / DQT / SOF0 / DHT /
//! SOS / EOI marker segments (from the §3.1 main header's Q field or an
//! in-band §3.1.8 quantization-table header) so the result is a complete
//! JPEG interchange stream the [`decoder`] consumes directly.
//!
//! ## Standalone vs registry-integrated
//!
//! The crate's default `registry` Cargo feature pulls in `oxideav-core`
//! and exposes the `Decoder` / `Encoder` trait surface, the JPEG-still
//! container, and the [`registry::register`] / [`registry::register_codecs`]
//! / [`registry::register_containers`] entry points, plus the
//! `From<JpegImage> for VideoFrame` / pixel-format / colour-signal
//! conversions. Disable the feature (`default-features = false`) for an
//! oxideav-core-free build that still exposes the whole still-image API
//! above plus the crate-local [`MjpegFrame`] / [`MjpegPlane`] /
//! [`MjpegPixelFormat`] / [`MjpegError`] types built only on `std`.

mod api;
pub mod convert;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod image;
pub mod jpeg;
pub mod rtp;
pub mod t81;

#[cfg(feature = "registry")]
pub mod container;

#[cfg(feature = "registry")]
pub mod mjpeg_container;

#[cfg(feature = "registry")]
pub mod registry;

pub const CODEC_ID_STR: &str = "mjpeg";

// Standalone, framework-free API. Available regardless of the
// `registry` feature.
pub use api::{
    decode, decode_from, decode_rgb8, decode_rgba8, decode_with, encode, encode_rgb8, encode_rgba8,
    encode_to, info, probe,
};
pub use error::{Error, MjpegError, Result};
pub use image::{
    ColorInfo, ColorRange, DecodeOptions, ImageInfo, JpegImage, Metadata, MjpegFrame,
    MjpegPixelFormat, MjpegPlane, PixelFormat, Plane, RgbImage, RgbaImage,
};

// General T.81 writer surface (the one JPEG encoder sibling crates
// build on): typed options plus the frame / table-set primitives.
pub use t81::{
    ColorSignalling, EncodeOptions, EncodedJpeg, HuffSpec, HuffStats, HuffmanTables, JpegComponent,
    JpegFrame, JpegProcess, JpegTableSet,
};

/// Historical name of [`EncodeOptions`].
#[deprecated(since = "0.1.10", note = "renamed to `EncodeOptions`")]
pub type JpegEncodeOptions = EncodeOptions;

// Decode-free JPEG inspector — classifies the SOF variant + reports
// dimensions / components / chroma-subsampling / colour hint without
// running the entropy decoder. Standalone surface, no `oxideav-core`
// dependency.
pub use jpeg::inspect::{
    inspect_jpeg, parse_adobe_app14, parse_dqt_segment, parse_icc_profile_app2, parse_jfif_app0,
    parse_jfxx_app0, AdobeApp14, AdobeColorTransform, ChromaSubsampling, ColorHint,
    IccProfileApp2Chunk, IccProfileChunks, InspectedComponent, JfifApp0, JfifUnits, JfxxApp0,
    JfxxThumbnail, JpegComment, JpegInfo, QuantTableInfo, SofKind,
};

// Framework-integrated API (`oxideav-core`-dependent). Gated behind
// `registry` so image-library callers can build the crate without
// dragging in `oxideav-core`.
#[cfg(feature = "registry")]
pub use registry::{
    __oxideav_entry, register, register_codecs, register_containers, MjpegEncoderOptions,
};

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
