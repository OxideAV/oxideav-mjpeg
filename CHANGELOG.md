# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/OxideAV/oxideav-mjpeg/compare/v0.1.1...v0.2.0) - 2026-04-26

### Other

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
