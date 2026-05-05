# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
