# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `fuzz/fuzz_targets/rtp_depacketize.rs`: cargo-fuzz harness covering
  the RFC 2435 RTP/JPEG depacketizer (`oxideav_mjpeg::rtp`). Feeds
  arbitrary bytes through `parse_main_header`, `parse_restart_header`,
  and `JpegDepacketizer::push` — the latter as a sequence of
  synthetic "packets" so the §3.1.2 fragment-offset reassembly
  buffer, the §3.1.7 Restart Marker header parser, the §3.1.8
  Quantization Table header parser, the §4.2 static-Q table cache,
  the marker-bit close path, and the `reset()` cache-retention
  invariant are all exercised on every iteration. Contract: no
  panic, slice OOB, debug-build integer overflow, or buffer
  allocation the input couldn't plausibly back. Assembled frames
  (when the marker bit closes a frame) are asserted to begin with
  SOI and end with EOI; their interior bytes are not validated
  (round-trip correctness is owned by the unit tests in
  `src/rtp.rs`). This is now the sixth fuzz harness in `fuzz/`
  alongside `decode`, `jpeg_self_roundtrip`,
  `jpeg_progressive_self_roundtrip`,
  `libjpeg_encode_oxideav_decode`, and
  `oxideav_encode_libjpeg_decode`.
- `tests/docs_corpus.rs`: the fixture-corpus harness now gates on
  numerical floors instead of merely reporting. Two new `Tier` variants:
  - `Tier::Exact` asserts every sample matches the reference. Five
    fixtures land here today — `tiny-baseline-1x1`,
    `baseline-grayscale-32x32`, `lossless-1986-mode`,
    `arithmetic-coded` (SOF9 Q-coder Decode path), and
    `baseline-yuv411-32x32` — all already at 100 % exact.
  - `Tier::PsnrFloor { db, exact_pct }` asserts both `total.psnr >= db`
    and `total.match_pct() >= exact_pct`. Eleven lossy fixtures land here
    with floors set ~0.5–2 dB below the observed PSNR and ~1–2 pp
    below the observed exact-sample percentage; tight enough to catch
    a real regression, loose enough to absorb normal floating-point
    jitter.
  Both new tiers fail CI on a regression rather than silently
  degrading the corpus baseline. `ReportOnly` and `Ignored` remain for
  forward-flexibility but are unused at present.
- 12-bit precision decoder: 4:2:2 (`Yuv422P12Le`) and 4:4:4 (`Yuv444P12Le`)
  chroma sampling, alongside the previously-supported 4:2:0
  (`Yuv420P12Le`). All three formats run through the shared
  `render_from_coefs_12bit` path with the spec's `P=12` level shift of
  `2^(P-1) = 2048` and a `[0, 4095]` output clamp (T.81 §A.3.1). The
  decoder accepts any sequential 12-bit JPEG (SOF0/SOF1) declaring
  three components with `Cb`/`Cr` at `H=V=1` and luma at `(1,1)`,
  `(2,1)`, or `(2,2)`; non-2x luma sampling at 12-bit (e.g. 4:1:1)
  still returns `Error::Unsupported`. `MjpegPixelFormat::Yuv422P12Le`
  and `MjpegPixelFormat::Yuv444P12Le` join the standalone enum with
  `From` conversions against `oxideav_core::PixelFormat` in both
  directions. New roundtrip tests in `decoder::precision_12_tests`
  cover all three sampling factors plus the 4:1:1-rejection contract.
- `encoder::encode_yuv_jpeg_12bit` (test-only crate helper): emits a
  standalone three-component SOF1 JPEG at `P=12` for any sampling
  factor in `{(1,1), (2,1), (2,2), (4,1)}`. Reuses the Annex K luma /
  chroma Huffman tables (callers must keep per-block DC/AC categories
  ≤ 11). Drives the decoder-side roundtrip tests; not exposed in the
  public API.
- `fuzz/fuzz_targets/decode.rs`: cargo-fuzz robustness target that drives
  arbitrary bytes (capped at 64 KiB) through the public `Decoder` trait
  (`make_decoder` → `send_packet` → `receive_frame`). The contract is
  "no panic": any malformed input must yield `Err(_)` rather than an
  unwrap, slice-OOB, integer overflow, or unbounded `Vec::with_capacity`
  / `vec![0; n]` allocation. 60 000 runs reach coverage 1 830 / 6 667
  without a crash. The harness is registered as a fifth bin in
  `fuzz/Cargo.toml` alongside the existing round-trip / cross-decode
  targets, so the daily reusable fuzz workflow's auto-discovery picks
  it up without further wiring.

### Fixed

- **Decoder dequantise i32 overflow** (`render_from_coefs` ×3 sites): when a
  DQT carries `Pq = 1` (16-bit precision, T.81 §B.2.4.1) the quant value can
  be up to 65535, and a coefficient at the high end of the DCT range
  (progressive `Al`-shifted, or accumulated DPCM DC) can multiply past
  `i32::MAX`. The 8-bit, 12-bit, and progressive render paths now perform the
  dequantise multiplication in `f32` directly — the IDCT input is `f32`
  either way, and `f32` carries the product well past 24 bits of mantissa
  without overflow. Fuzz-found regression (`crash-ee0cdd45`); replaying the
  artifact through the patched `decode` target now completes in 0 ms.

- Decoder panic surfaces uncovered during fuzz harness bring-up:
  - **SOS `Tdj` / `Taj` selectors** outside `0..=3` no longer panic
    indexing the 4-wide `dc_huff` / `ac_huff` / `arith_dc` / `arith_ac`
    arrays; a new `validate_sos` rejects them as `Error::Invalid`
    before scan dispatch.
  - **SOF `Tq` selectors** outside `0..=3` no longer panic indexing
    `state.quant`; a new `validate_sof` rejects them up-front.
  - **SOS `Ns = 0` / `Ns > 4`** rejected by `validate_sos` (an empty
    component list otherwise produced an empty `prev_dc` whose first
    index panicked).
  - **SOF `Nf = 0` / `Nf > 4`** and **`Hi/Vi` outside `1..=4`**
    rejected by `validate_sof` (zero sampling factors previously hit
    `unwrap_or(1)` fallbacks before downstream MCU arithmetic divided
    by them).
  - **Repeated SOF segments** in a single JPEG now return
    `Error::Invalid("JPEG: multiple SOF segments")` rather than
    overwriting `state.sof` while a stale `coef_buf` allocated against
    the prior SOF stayed live — the geometry mismatch would later
    OOB the per-block accumulator.
  - **SOF pixel-budget DoS**: `Wt × Ht × Nf > 64 Mpx` rejected as
    `Error::Unsupported("SOF: pixel budget exceeded")`. A 16-byte SOF
    segment could previously request `~17 GiB` of per-component output
    buffers.
  - **`BitReader::get_bits(n)` underflow**: a Huffman-decoded SSSS of
    `0` was a `>> 32` UB on `u32` (debug-panic, release-wrap);
    `n > 24` underflowed the `24 - self.nbits` shift in the refill
    loop. Both bounds are now checked: `n == 0` short-circuits to
    `Ok(0)`, `n > 24` returns `Error::Invalid`.
  - **Lossless SSSS > 16**: Annex H Table H.2 limits SSSS to `0..=16`,
    but a crafted DHT can deliver any byte; the lossless scan decoder
    now rejects out-of-range SSSS rather than calling `get_bits(s)`
    with a value the `extend` / shift machinery has no defined
    behaviour for.

- `rtp::JpegDepacketizer`: cross-frame in-band quantization-table caching
  (RFC 2435 §4.2). For a *static* Q value (128..=254) the sender may carry the
  Quantization Table header once and then omit the tables (`Length = 0`) on
  subsequent frames; the depacketizer now caches the tables per Q value and
  reuses them when a later frame's header is empty, so a multi-frame static-Q
  stream decodes past the first frame. A `Length = 0` frame with no cached
  tables for that Q (e.g. a receiver that joined mid-stream) still returns
  `Unsupported`, matching the §4.2 startup caveat. Q = 255 is dynamic and never
  populates the cache (the spec forbids depending on a previous frame's tables
  for Q = 255). `reset()` clears the in-progress reassembly buffer but retains
  the table cache; `new()` starts fully fresh. Five tests cover cache reuse,
  the no-prior-cache error, per-Q keying, the Q = 255 non-caching rule, and
  cache survival across `reset()`.

- `rtp` module: RFC 2435 RTP/JPEG **packetization** (the encode-side inverse
  of the depacketizer). `rtp::packetize(jpeg, max_payload, qmode)` parses a
  complete baseline (SOF0/SOF1) three-component YUV JPEG, strips the frame and
  scan headers, and fragments the entropy-coded scan into `rtp::JpegPacket`
  RTP/JPEG payloads. Luma sampling `2x1` maps to the well-known type 0 (4:2:2)
  and `2x2` to type 1 (4:2:0); a DRI segment promotes the type to 64/65 and
  emits the §3.1.7 Restart Marker header (whole-frame reassembly, F=L=1,
  count=0x3FFF). `rtp::QMode` selects table carriage: `Quality(1..=99)` puts an
  IJG-quality Q value in the Q field (receiver regenerates Annex K tables), or
  `InBand(128..=255)` carries the JPEG's own two DQT tables in a §3.1.8
  Quantization Table header on the first fragment (offset 0). Fragments are
  byte-contiguous; the first has fragment offset 0, the last has
  `JpegPacket::marker == true` (caller sets the RTP marker bit). The caller
  still owns RTP transport (12-byte fixed header, sequence numbers, 90 kHz
  timestamp). Progressive/lossless/grayscale/CMYK, non-2:x luma sampling, and
  16-bit DQT return `Unsupported` — RTP/JPEG has no well-known type for them.
  Tests cover header layout, type 0/1 selection, restart-bit promotion, scan
  fragmentation, the unsupported-input rejections, a structural
  packetize→depacketize scan round trip, and end-to-end
  encode→packetize→depacketize→decode for both the in-band and Q-field paths.
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
