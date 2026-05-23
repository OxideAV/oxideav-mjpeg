# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `rtp` module: RFC 2435 RTP/JPEG depacketization. `rtp::JpegDepacketizer`
  reassembles fragmented RTP/JPEG payloads (keyed on the §3.1.2 Fragment
  Offset, so misordered intra-frame delivery is tolerated) and
  reconstructs the absent SOI / DQT / SOF0 / DHT / [DRI] / SOS / EOI
  marker segments into a complete JPEG interchange stream the existing
  decoder consumes directly. Covers the well-known fixed type mappings
  0/64 (4:2:2, `H=2 V=1` luma) and 1/65 (4:2:0, `H=2 V=2` luma) with the
  three-component YUV interleaved scan (§4.1). Quantization tables are
  recovered from the Q field via the Independent JPEG Group scale formula
  over Annex K.1 / K.2 for `Q ∈ 1..=99` (§4.2), or read in-band from the
  §3.1.8 Quantization Table header for `Q ∈ 128..=255` (8- and 16-bit
  precision, the latter saturated to the emitted 8-bit DQT). Types
  64..=127 consume the §3.1.7 Restart Marker header and emit a DRI
  segment with the carried interval. Annex K typical Huffman tables are
  written for the abbreviated-format scan. The standalone helpers
  `rtp::parse_main_header` / `rtp::parse_restart_header` expose the wire
  header layout (`MainHeader` / `RestartHeader`). End-to-end tests encode
  a frame, strip it to an RTP/JPEG payload (both Q-field and in-band-table
  paths), depacketize, and decode the result back to a valid frame.
  Out-of-band table negotiation (Q ≥ 128 with no in-band tables) and the
  non-well-known dynamic types 128..=255 return `Unsupported`. RTP
  transport framing (the 12-byte RTP fixed header, sequence ordering)
  stays the caller's responsibility.
- `encoder::encode_lossless_jpeg_grayscale_with_opts(width, height,
  samples, stride, precision, predictor, restart_interval,
  point_transform)` and `encoder::encode_lossless_jpeg_rgb_with_opts(...)`:
  public lossless (SOF3) encoder variants that emit DRI + `RST0..=RST7`
  every `restart_interval` MCUs (cycling modulo 8 per T.81 §F.1.1.5.2)
  and honour a non-zero `point_transform` (`Pt`, the SOS `Al` nibble).
  On every restart boundary the encoder byte-aligns the stream and
  re-seeds every component's predictor history to the per-component
  origin `2^(P − Pt − 1)` per T.81 §H.1.2.1; with `Pt > 0` every input
  sample is right-shifted by `Pt` before prediction, and the decoder
  side reconstructs `(sample >> Pt) << Pt`. Bit-exact roundtrips at
  every supported predictor / restart interval / Pt combination across
  the grayscale 2..=16-bit precisions and the 8-bit three-component
  path. The default-options entry points
  `encode_lossless_jpeg_grayscale` / `encode_lossless_jpeg_rgb` are
  unchanged (now thin wrappers passing `restart_interval = 0`,
  `point_transform = 0`).
- `encoder::encode_lossless_jpeg_rgb(width, height, [r, g, b], strides,
  precision, predictor)`: public three-component (RGB-class) lossless
  JPEG (SOF3) encoder. Emits a standalone SOF3 stream with one
  interleaved SOS scan, each component declared `H_i = V_i = 1` per
  T.81 §H.1.2 (lossless data unit is one sample, so the natural MCU is
  one residual per component at each pixel position). Each component
  is modeled with its own independent predictor buffer (per H.1.2:
  "each component in the scan is modeled independently, using
  predictions derived from neighbouring samples of that component").
  Supports every precision `P ∈ 2..=16` and every Annex H Table H.1
  predictor `1..=7`. Bit-exact roundtrip at `P = 8` against the
  packed `Rgb24` decoder output.
- SOF3 lossless decoder: accept three-component interleaved scans at
  `P = 8` and emit packed `PixelFormat::Rgb24`. Per-component sample
  buffers track independent predictor history; restart markers reset
  every component's predictor together. Four-component CMYK-class
  lossless and non-unit sampling factors stay rejected with
  `Unsupported`.
- `MjpegPixelFormat::Rgb24` (8-bit packed R-G-B) added to the
  standalone-build pixel-format enum, with `From` conversions in both
  directions against `oxideav_core::PixelFormat::Rgb24`.
- `encoder::encode_lossless_jpeg_grayscale(width, height, samples,
  stride, precision, predictor)`: public single-component grayscale
  lossless JPEG (SOF3) encoder. Supports every precision `P ∈ 2..=16`
  and every Annex H Table H.1 predictor `1..=7`. Output is bit-exact —
  the existing SOF3 grayscale decoder recovers every sample verbatim,
  including the special SSSS=16 / Di=32768 half-modulus case (T.81
  §H.1.2.2). Uses a single canonical wide-symbol DC Huffman table
  (`STD_DC_LOSSLESS_*`) that is Kraft-complete over symbols 0..=16, so
  the same table is valid at every supported precision without
  per-image tuning.
- `MjpegEncoder::set_lossless(bool)` /
  `MjpegEncoder::set_lossless_predictor(u8)` on the registry-side
  encoder. With `set_lossless(true)` the trait-API encoder accepts
  `Gray8` / `Gray10Le` / `Gray12Le` / `Gray16Le` `VideoFrame` input
  and dispatches to the lossless path; without the flag, grayscale
  input is rejected with a clear error so the historical YUV-only
  contract stays explicit.
- Raw Motion-JPEG container demuxer (`mjpeg-raw`, owns the `.mjpeg` /
  `.mjpg` extensions). One packet per JPEG frame in the stream, marker-
  aware boundary scanner (T.81 §B.1.1.2 / §B.1.1.4) that honours
  length-prefixed segment bodies — APP1 thumbnails, COM segments, etc.
  cannot false-trigger an SOI / EOI match. Default time base is `1/25`
  (frame `i` carries `pts = i`); callers that know the real rate can
  post-process the emitted `StreamInfo::time_base`.
- `Demuxer::seek_to(stream_index, pts)` on the raw MJPEG demuxer.
  Lazy `(pts, byte_offset)` index pushed every 5 frames (anchor at
  frame 0 seeded at open time); binary-search-then-linear-scan to the
  exact target frame. Past-end targets clamp to the last frame and
  surface `Error::Eof` from the following `next_packet`. Integration
  tests in `tests/seek.rs` cover zero-reset, mid-stream seek, past-end
  clamp, byte-stuffed `FF D8` false-positives, and byte-for-byte
  parity with a baseline drain.

## [0.1.6](https://github.com/OxideAV/oxideav-mjpeg/compare/v0.1.5...v0.1.6) - 2026-05-06

### Other

- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- re-export __oxideav_entry from registry sub-module
- add SA progressive + metadata pass-through API
- auto-register via oxideav_core::register! macro (linkme distributed slice)

### Added

- `encode_jpeg_progressive_sa` / `encode_jpeg_progressive_sa_with_meta`: full
  successive-approximation (SA) progressive JPEG encoder using a 1-bit point
  transform (`Al=1` initial, `Ah=1,Al=0` refinement). Produces 14 SOS scans
  (1 DC initial + 6 AC initial + 1 DC refine + 6 AC refine) that round-trip
  through ffmpeg, libjpeg, and ImageMagick with PSNR ≥ 40 dB relative to the
  source. Implements T.81 §G.1.2.3 AC refinement with correction bits
  interleaved inline during the decoder's zero-history walk.
- `encode_jpeg_with_meta` / `encode_jpeg_progressive_with_meta`: variants of
  the baseline and spectral-selection progressive encoders that accept a `meta`
  byte slice of pre-serialised APP0-APP15 and COM segments to embed verbatim,
  replacing the default JFIF APP0 marker.
- `extract_app_segments(jpeg: &[u8]) -> Vec<u8>`: walks a JPEG byte stream and
  returns a contiguous buffer of all APP (0xFF 0xEn) and COM (0xFF 0xFE)
  segment bytes, ready to pass as `meta` to the `*_with_meta` encoder family.

## [0.1.5](https://github.com/OxideAV/oxideav-mjpeg/compare/v0.1.4...v0.1.5) - 2026-05-05

### Other

- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-mjpeg/pull/502))

## [0.1.4](https://github.com/OxideAV/oxideav-mjpeg/compare/v0.1.3...v0.1.4) - 2026-05-04

### Fixed

- *(clippy)* drop unused Decoder import + div_ceil + doc lints

### Other

- cargo fmt: collapse import + match-arm wrap in registry.rs
- add default-on `registry` cargo feature for standalone-friendly builds
- end-to-end cross-codec roundtrip via libturbojpeg

### Added

- Default-on `registry` Cargo feature gates the `oxideav-core`
  dependency, the `Decoder` / `Encoder` trait implementations, the
  still-image JPEG container demuxer / muxer / probe, and the
  `register_codecs` / `register` / `register_containers` entry points.
  Image-library consumers can now depend on `oxideav-mjpeg` with
  `default-features = false` and skip the `oxideav-core` dep tree
  entirely; the standalone path exposes `decoder::decode_jpeg` and the
  `encoder::encode_jpeg_*` family plus crate-local `MjpegFrame` /
  `MjpegPlane` / `MjpegPixelFormat` / `MjpegError` types built only on
  `std`.
- Inline `ci-standalone` CI job verifies `cargo build --lib
  --no-default-features` and `cargo test --no-default-features` stay
  green on every change.

## [0.1.3](https://github.com/OxideAV/oxideav-mjpeg/compare/v0.1.2...v0.1.3) - 2026-05-03

### Other

- allow dead Tier::Ignored variant
- drop no-op u32 mask + replace MSRV-1.87 is_multiple_of
- cargo-fuzz harness with libjpeg-turbo cross-decode oracle
- drop unused loop binding in K.4 test
- align QE_TABLE comments to column 28 to satisfy rustfmt
- fmt QE_TABLE column alignment to single-space
- add SOF9 arithmetic-coded JPEG entropy decoder
- cargo fmt rustfmt 1.95 if-expression wrap
- fancy chroma upsampling + green-channel rounding fix
- emit Yuv411P for JPEG 4:1:1 sampling (luma 4x1)
- cargo fmt: collapse single-line panic! in tests/docs_corpus.rs
- wire docs/image/jpeg/ fixture corpus into integration suite

## [0.1.2](https://github.com/OxideAV/oxideav-mjpeg/compare/v0.1.1...v0.1.2) - 2026-05-03

### Other

- drop unused PixelFormat imports + ignore unused let-else binds
- cargo fmt: collapse double-blank lines after let-else (tests/)
- replace never-match regex with semver_check = false
- disable semver_check to prevent 0.2 bump
- migrate to centralized OxideAV/.github reusable workflows
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

## [0.1.1](https://github.com/OxideAV/oxideav-mjpeg/compare/v0.1.0...v0.1.1) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- add progressive (SOF2) encoder
- bump usage example to oxideav-mjpeg = "0.1"
- drop Cargo.lock — this crate is a library
- release v0.0.4

## [0.1.0](https://github.com/OxideAV/oxideav-mjpeg/compare/v0.0.3...v0.1.0) - 2026-04-19

### Other

- set version to 0.1.0
- bump oxideav-container dep to 0.1
- bump to oxideav-core 0.1.2
- decode SOF3 lossless grayscale JPEGs
- decode 12-bit precision sequential JPEGs
- decode 4-component CMYK / Adobe YCCK JPEGs
- support non-interleaved baseline/extended-sequential scans
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()
- claim AVI FourCCs via oxideav-codec CodecTag registry
- emit DRI + RSTn restart markers in encoder
- drop dead branch in encoder `category`
- decode SOF1 (extended sequential), refresh docs
