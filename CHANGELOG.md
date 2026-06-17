# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Other

- SOF11 subsampled three-component (YUV-class) lossless **arithmetic**
  encode + decode (T.81 A.2.3 interleaved-MCU ordering, §H.1.2.3
  per-component statistical model over each component's own padded grid)
  — luma 1×1/2×1/2×2/4×1, planar Yuv444P/Yuv422P/Yuv420P/Yuv411P output;
  new `encoder::encode_lossless_arith_jpeg_yuv` / `_with_opts`
- SOF3 subsampled three-component (YUV-class) lossless encode + decode
  (T.81 A.2.3 interleaved-MCU ordering) — luma 1×1/2×1/2×2/4×1, planar
  Yuv444P/Yuv422P/Yuv420P/Yuv411P output
- typed COM (comment) segments view (T.81 §B.2.4.5) on JpegInfo

## [0.1.8](https://github.com/OxideAV/oxideav-mjpeg/compare/v0.1.7...v0.1.8) - 2026-06-15

### Other

- JFIF extension APP0 (JFXX) thumbnail view (T.871 §10.2-10.5)
- SOF11 four-component (CMYK-class) lossless arithmetic encode
- SOF11 three-component (RGB-class) lossless arithmetic encode (T.81 §H.1.2.3)
- lossless arithmetic (SOF11) grayscale encode via T.81 §H.1.2.3
- DNL (Define Number of Lines) support for SOF Y = 0 (T.81 §B.2.5)
- progressive arithmetic JPEG (SOF10) via T.81 §G.1.3 scan procedures
- lossless arithmetic JPEG (SOF11) via T.81 §H.1.2.3 statistical model
- skip ICC-fixture inspect test when docs/ absent (CI fix)
- typed APP2 ICC_PROFILE chunks view (T.872 / Annex L) on JpegInfo
- typed Adobe APP14 view (T.872 §6.5.3) on JpegInfo
- typed JFIF APP0 view (T.871 §10.1) on JpegInfo
- drop release-plz.toml — use release-plz defaults across the workspace
- progressive (SOF2) single-component grayscale encode
- decode-free JPEG SOF discriminator + metadata inspector
- baseline (SOF0) packed-Rgb24 lossy encode + decoder RGB tag
- baseline (SOF0) single-component Gray8 lossy encode
- four-component lossless (SOF3, P=8) round trip
- restart-interval-aligned scan splitting for the packetizer
- criterion harness for encode + decode hot paths
- scrub decorative external-implementation attribution
- add arith_decode cargo-fuzz target for SOF9 Q-coder surface
- gate cmyk_roundtrip on the `registry` feature
- promote 4-component CMYK / YCCK helpers to the public API

### Added

- **JFIF extension APP0 (JFXX) inspector view (T.871 §10.2-10.5)** — the
  decode-free inspector now surfaces the JFIF *extension* APP0 segment
  (identifier `"JFXX\0"`) that conformant writers use to carry a
  thumbnail (the JFIF APP0's own inline thumbnail is rarely populated —
  most files emit a JFXX segment instead). A new `JfxxApp0` typed view
  on `JpegInfo::jfxx` reports the thumbnail-storage variant via a
  `JfxxThumbnail` enum exhaustive over the three `extension_code` bytes
  T.871 §10.2 defines: `JpegEncoded { jpeg_len }` (`0x10`, §10.3 — an
  embedded baseline JPEG, length reported without recursion),
  `PaletteRgb { width, height }` (`0x11`, §10.4 — a 768-byte palette +
  one-byte indices), and `Rgb24 { width, height }` (`0x13`, §10.5 —
  packed 24-bit RGB, same layout as the §10.1 inline thumbnail).
  `JfxxThumbnail::extension_code()` recovers the literal byte for
  re-serialisation. A top-level `parse_jfxx_app0(payload) ->
  Result<JfxxApp0>` validator is exported for callers that already hold
  the APP0 payload bytes; it enforces the structural invariants
  (identifier `"JFXX\0"`, defined `extension_code`, non-zero thumbnail
  dimensions for `0x11`/`0x13`, declared body fits the payload) and
  never copies the thumbnail body. `inspect_jpeg` populates the view
  automatically when a JFXX extension APP0 follows the JFIF APP0; the
  extension segment carries no colour-convention signal so it does not
  influence `color_hint` (independent of `jfif`, which keeps reporting
  the leading JFIF segment). Standalone surface — no `registry` feature,
  no `oxideav-core` dep. Eleven new tests cover the three storage
  variants, the short-payload / bad-identifier / reserved-code /
  zero-dimension / body-overflow rejection paths, the JFIF+JFXX
  dual-APP0 inspector aggregation, and the no-JFXX baseline.

- **Lossless arithmetic (SOF11) four-component (CMYK-class) encode (T.81
  Annex H + §H.1.2.3)** — `encoder::encode_lossless_arith_jpeg_cmyk(width,
  height, planes, strides, predictor, adobe_transform)` and its
  `_with_opts(..., restart_interval, point_transform)` companion emit a
  standalone four-component interleaved SOF11 (lossless, arithmetic-coded)
  JPEG at `P = 8`. This is the Q-coder counterpart of the existing Huffman
  `encode_lossless_jpeg_cmyk` and the four-component extension of
  `encode_lossless_arith_jpeg_rgb`: each component is modelled independently
  (§H.1.2) with its own `LosslessStats` area and `L_Context(Da, Db)` /
  `X1_Context(Db)` difference history (§H.1.2.3.2), while a single
  arithmetic-coded segment carries one residual per component per pixel
  position in scan-component order (each component declared `H_i = V_i = 1`).
  The Adobe APP14 colour-transform flag is honoured identically to the
  Huffman CMYK helper — `None` (no APP14, plain CMYK), `Some(0)` (Adobe
  CMYK, samples inverted on the wire), `Some(2)` (Adobe YCCK, K inverted) —
  with the segment emitted before SOF11 so the decoder's existing
  four-component un-inversion / YCCK → CMYK path applies on output. Output
  decodes to packed `PixelFormat::Cmyk` (4 bytes/pixel), bit-exact on the
  no-APP14 / Adobe-CMYK paths for every predictor, the half-modulus
  `Di = 32768` case (§H.1.2.2), non-zero point transforms, and
  restart-interval emission (each `RSTn` boundary flushes the Q-coder,
  cycles `RST0..=RST7` modulo 8, and re-seeds every component's statistical
  model + difference history + predictor to `2^(P − Pt − 1)`, §H.1.1 /
  §H.1.2.3.4). The matching SOF11 four-component decode path already
  existed. Covered by six new round-trips in `tests/lossless_roundtrip.rs`.

- **Lossless arithmetic (SOF11) three-component encode (T.81 Annex H +
  §H.1.2.3)** — `encoder::encode_lossless_arith_jpeg_rgb(width, height,
  planes, strides, precision, predictor)` and its
  `_with_opts(..., restart_interval, point_transform)` companion emit a
  standalone three-component interleaved SOF11 (lossless, arithmetic-coded)
  JPEG. This is the Q-coder counterpart of the existing Huffman
  `encode_lossless_jpeg_rgb` and the multi-component extension of the
  single-component `encode_lossless_arith_jpeg_grayscale`: each component is
  modelled independently (§H.1.2) with its own `LosslessStats` area and its
  own `L_Context(Da, Db)` / `X1_Context(Db)` difference history (§H.1.2.3.2),
  while a single arithmetic-coded entropy segment carries one residual per
  component per pixel position in scan-component order (each component
  declared `H_i = V_i = 1`, so a lossless MCU is one pixel). The Annex H
  Table H.1 predictors `1..=7` are applied per component over that
  component's own `Ra` / `Rb` / `Rc`; no DAC segment is emitted so the
  decoder uses the default conditioning bounds `(L, U) = (0, 1)`
  (§H.1.2.3.3). Output is bit-exact for every precision `P ∈ 2..=16`
  (decode maps `P = 8` → packed `Rgb24`, `P ∈ {10, 12, 14}` → planar
  `Gbrp*Le`, every other `P` → packed `Rgb48Le`), every predictor, the
  half-modulus `Di = 32768` case (§H.1.2.2), non-zero point transforms, and
  restart-interval emission (each `RSTn` boundary flushes the Q-coder,
  byte-aligns, cycles `RST0..=RST7` modulo 8, and re-seeds every component's
  statistical model + difference history + predictor to the scan-origin
  default `2^(P − Pt − 1)`, §H.1.1 / §H.1.2.3.4). The matching SOF11
  multi-component decode path already existed. Covered by six new
  round-trips in `tests/lossless_roundtrip.rs`.

- **Lossless arithmetic (SOF11) grayscale encode (T.81 Annex H + §H.1.2.3)** —
  `encoder::encode_lossless_arith_jpeg_grayscale(width, height, samples,
  stride, precision, predictor)` and its
  `_with_opts(..., restart_interval, point_transform)` companion emit a
  standalone single-component SOF11 (lossless, arithmetic-coded) JPEG.
  The spatial model reuses the Annex H Table H.1 predictors `1..=7` over
  `Ra` / `Rb` / `Rc`, but each prediction difference is coded with the
  Q-coder arithmetic statistical model of §H.1.2.3 (Table H.3 —
  `L_Context(Da, Db)` / `X1_Context(Db)` conditioning over neighbouring
  differences) rather than a Huffman magnitude category. No DAC segment
  is emitted, so the decoder applies the default conditioning bounds
  `(L, U) = (0, 1)` per §H.1.2.3.3. Output is bit-exact for every
  precision `P ∈ 2..=16`, every predictor, the half-modulus
  `Di = 32768` case (§H.1.2.2), non-zero point transforms, and
  restart-interval emission (each `RSTn` boundary flushes the Q-coder,
  byte-aligns, cycles `RST0..=RST7` modulo 8, and re-seeds the
  statistical model + difference history + predictor to the scan-origin
  default `2^(P − Pt − 1)`, §H.1.1 / §H.1.2.3.4). This is the
  encoder-side counterpart to the existing SOF11 decode path and the
  first arithmetic-coded entry point on the encoder side. Covered by
  six new round-trips in `tests/lossless_roundtrip.rs`.

- **DNL (Define Number of Lines) decode support (T.81 §B.2.2 / §B.2.5)** —
  JPEG frames may code the number of lines `Y = 0` in the SOF header, in
  which case the real line count is supplied by a mandatory DNL segment
  (`0xFFDC`) immediately after the first scan. The decoder now performs an
  up-front marker-stream pre-pass (`resolve_dnl_height`) that, when it
  sees `Y = 0`, walks to the first scan, reads `NL` from the following
  DNL segment, and patches the frame height before any scan decoder runs —
  so every path (baseline fast path, sequential / progressive / arithmetic
  accumulators, lossless) decodes at the correct height with no per-path
  changes. A `Y = 0` stream with no following DNL is rejected (the segment
  is mandatory there), as is a malformed DNL carrying `NL = 0`
  (Table B.10 constrains `NL ∈ 1..=65535`). New `parse_dnl` parser entry,
  `markers::DNL` constant, and explicit DNL handling in the main marker
  loop. Covered by `tests/dnl.rs` (YUV 4:4:4 / 4:2:2 / 4:2:0 round-trips,
  a non-MCU-aligned height, plus the two negative cases) and four
  `decoder::dnl_unit_tests` unit tests.
- **Progressive arithmetic JPEG (SOF10) decode** — the SOF2 multi-scan
  spectral-selection / successive-approximation structure with the
  Annex D Q-coder as the entropy layer, per T.81 §G.1.3:
  - DC first scans (`Ss = Se = 0`, `Ah = 0`) reuse the sequential
    §F.1.4.1 DC statistical model on the point-transformed values
    (DC point transform = arithmetic shift right); the decoded
    difference accumulates into the per-component prediction and lands
    left-shifted by `Al` (§G.1.3.1).
  - DC refinement scans (`Ah > 0`) decode one binary decision per
    block with the fixed 0.5 probability estimate (`Qe = 0x5A1D`,
    `MPS = 0`, non-adapting) and OR the bit into the existing DC value
    at bit position `Al`.
  - AC first scans (`Ss > 0`, `Ah = 0`) run the §F.1.4 sequential AC
    procedure with `Kmin = Ss` and the EOB decision meaning
    end-of-*band* (§G.1.3.2); decoded values land left-shifted by
    `Al`. The DAC marker's `Kx` conditioning is honoured (default 5).
  - AC refinement scans (`Ah > 0`) follow the §G.1.3.3 coding model
    (Figures G.10 / G.11) under the Table G.2 statistics layout — a
    new 189-bin `jpeg::arith::AcRefineStats` area with `SE / S0 / SC`
    bins per coefficient index, the end-of-band decision bypassed
    while `K < EOBx` (recovered from the coefficient history), newly
    nonzero coefficients signed by the fixed estimate, and correction
    bits growing existing magnitudes by `2^Al`
    (`jpeg::arith::decode_ac_refine`).
  - Restart intervals re-initialise the coder, the statistics areas
    and the DC predictions at every `RSTn`, in every scan kind.
  - Same frame constraints and output shaping as the Huffman
    progressive (SOF2) path: `P = 8` and `P = 12` (Annex G processes
    4 and 8), 1- and 3-component plus 4-component CMYK / YCCK at
    `P = 8` (Adobe APP14 transform flag honoured), shared coefficient
    accumulator + EOI render.
  - Round-trip tests drive every scan kind from an encoder-side
    mirror of the §G.1.3 procedures (spectral selection only, full
    progression, two successive-approximation levels, interleaved
    4:2:0 DC, restart intervals, DAC `Kx` override, 12-bit, and
    4-component CMYK), comparing the decoded pixels sample-exact
    against a direct IDCT of the source coefficients.
- `SofKind::is_supported_by_decoder` now reports `ProgressiveArith`
  (SOF10) and `LosslessArith` (SOF11) as supported — the SOF11 decode
  path landed previously but the inspector helper had not been
  updated alongside it.
- **Lossless arithmetic JPEG (SOF11) decode** — the Annex H predictor
  coding model with the modulo-2^16 prediction differences
  entropy-coded by the Annex D Q-coder under the T.81 §H.1.2.3
  two-dimensional statistical model. Each binary decision is
  conditioned on the classifications of the differences coded for the
  sample to the left (Da) and the sample in the line above (Db) via
  the 5 × 5 `L_Context(Da, Db)` array of Figure H.2, with the
  magnitude bins selected by `X1_Context(Db)` — 158 statistics bins
  per scan component per §H.1.2.3.2 / Table H.3
  (`jpeg::arith::LosslessStats`). The DAC marker's DC-conditioning
  `(L, U)` bounds are honoured (defaults `(0, 1)` per §H.1.2.3.3;
  small/zero boundary `2^(L−1)` exclusive, small/large boundary `2^U`
  inclusive per F.1.4.4.1.2). Coverage matches the SOF3 Huffman path:
  single-component grayscale and three-component RGB-class at every
  precision `P ∈ 2..=16`, four-component CMYK-class at `P = 8`, all
  Table H.1 predictors, point transform, and restart intervals
  (statistics + conditioning + prediction re-initialised at each RSTn
  per §H.1.2.3.4 / §H.2.1, with the coder re-initialised past the
  marker). Prediction follows §H.1.2.1: origin `2^(P−Pt−1)` at
  scan/interval start, the 1-D horizontal predictor across the first
  line of the scan *and of each restart interval*, `Rb` at the start
  of every other line. The precision-driven output shaping is shared
  with SOF3 via the extracted `shape_lossless_frame` helper, so the
  pixel-format policy is identical (`Gray8` / `Gray10Le` / `Gray12Le`
  / `Gray16Le`, packed `Rgb24` / planar `Gbrp*Le` / packed `Rgb48Le`,
  packed `Cmyk`).
- **Q-coder arithmetic *encoder*** (`jpeg::arith::ArithEncoder`) per
  T.81 Annex D §D.1: Initenc (Figure D.12), Code_MPS / Code_LPS with
  conditional MPS/LPS exchange (Figures D.3 / D.4), Renorm_e
  (Figure D.7), Byte_out with carry resolution, `0xFF` stacking and
  `0xFF 0x00` stuffing (Figures D.8–D.11), and the Flush /
  Clear_final_bits / Discard_final_zeros termination sequence
  (Figures D.13–D.15) — validated byte-exactly against the Annex K.4.1
  256-bit test sequence (the encoder reproduces the spec's listed
  compressed stream, stuffed byte included), plus
  `encode_lossless_diff` / `encode_magnitude` mirrors of the Table H.3
  decision tree used by the SOF11 round-trip tests.
- Eleven new tests: K.4.1 encode reproduction, multi-context
  encode/decode self-consistency, lossless-diff round-trip across
  default + DAC-overridden bounds, Figure H.2 context-base /
  classification checks, and SOF11 decode round-trips (grayscale
  `P = 8` across all seven predictors, grayscale `P = 16` with
  pseudorandom samples through the deep end of the magnitude tree,
  three-component RGB `P = 8`, line-aligned restart intervals,
  DAC `(L = 2, U = 5)` conditioning, non-zero point transform) plus a
  SOF10 still-rejected guard.

- `IccProfileChunks` aggregated view of every APP2 `"ICC_PROFILE\0"`
  marker segment seen in the prefix (T.872 / Annex L of T.871; see
  `docs/image/jpeg/jpeg-fixtures-and-traces.md` §3.11) on
  `JpegInfo::icc_profile`. The summary reports the declared chunk
  `total` (every segment must agree — mismatches drop the dissenting
  chunks), the cumulative `total_payload_len` of profile bytes seen
  across the segments, and the per-segment `(seq_no, payload_len)`
  ordering in source order, plus `is_complete()` returning true when
  the chunks cover every sequence number `1..=total` exactly once. A
  borrowing `IccProfileApp2Chunk<'a>` (`seq_no`, `total`,
  `profile_bytes: &'a [u8]`) plus a new top-level
  `parse_icc_profile_app2(payload) -> Result<IccProfileApp2Chunk<'_>>`
  validator are exported for callers that already hold the APP2 payload
  bytes; the validator enforces the structural invariants (identifier
  equals `"ICC_PROFILE\0"`, payload ≥ 14 bytes, `total ≥ 1`,
  `1 ≤ seq_no ≤ total`) and never allocates or copies the ICC body.
  `inspect_jpeg` populates the summary automatically; APP2 segments
  whose identifier is not `"ICC_PROFILE\0"` (FPXR, IPTC-bearing APP2,
  etc.) are silently ignored and APP2 does not influence `color_hint`
  (the ICC profile is colour-management metadata, separate from the
  YCbCr/RGB mapping signalled by APP0 JFIF / APP14 Adobe). Twelve new
  tests cover the minimal one-chunk success path, body propagation,
  payload-too-short / bad-identifier / `total = 0` / zero-seq /
  seq-above-total rejection, the inspector's aggregation across one
  / three / partial / duplicate / mismatched-total streams, the
  non-ICC APP2 ignore path, the no-APP2 baseline, plus an integration
  test against the docs corpus' `with-icc-profile-embedded` Ghostscript
  sRGB fixture (one chunk of `total_payload_len = 2576`).

- `markers::APP2 = 0xE2` constant, the standard JPEG APPn byte that
  carries the embedded ICC profile (in addition to the existing APP0
  / APP14 constants).

- `AdobeApp14` typed view of the Adobe APP14 marker segment (T.872
  §6.5.3 / Adobe Technical Note 5116 §18) on `JpegInfo::adobe`.
  Carries the raw `dct_encode_version` `u16` (commonly `100`), the
  two encoder-hint flag words `flags_0` and `flags_1`, and an
  `AdobeColorTransform` enum (`Unknown` / `YCbCr` / `Ycck`,
  exhaustive over the spec's three legal `transform` bytes with
  `as_byte()` for re-encoding), plus `is_standard_version()` (true
  for the universally-used `100`) and an `as_color_hint()`
  projection back to the inspector-level `ColorHint` enum. A new
  top-level `parse_adobe_app14(payload) -> Result<AdobeApp14>`
  validator is exported for callers that already hold the APP14
  payload bytes; it enforces the three structural invariants
  (`identifier == "Adobe"`, payload ≥ 12 bytes, `transform ∈
  {0, 1, 2}`) and never allocates. `inspect_jpeg` populates
  `JpegInfo.adobe` automatically when an APP14 carries a structurally
  valid Adobe segment; reserved `transform` bytes leave the typed
  view as `None` but the inspector's coarse `ColorHint` path still
  flips to `AdobeUntransformed` as before, since the colour-hint
  signal is more tolerant by design. Independent of the JFIF view —
  streams with both an APP0 JFIF and an APP14 Adobe populate both
  typed views, and the colour hint continues to prefer Adobe when
  both are present (existing inspector precedence). Eight new tests
  cover the standard-version success path, the encoder-flag bits,
  payload-too-short / bad-identifier / reserved-transform rejection,
  the inspector's reserved-transform tolerance, the JFIF+Adobe
  dual-segment case, and the only-first-segment-wins rule for
  duplicate APP14s.

- `JfifApp0` typed view of the JFIF APP0 marker segment (T.871 §10.1)
  on `JpegInfo::jfif`. Carries the `version_major` / `version_minor`
  bytes, a `JfifUnits` enum (`AspectRatio` / `DotsPerInch` /
  `DotsPerCm`, exhaustive per the spec's "shall be one of" wording
  with `as_byte()` for re-encoding), `h_density` / `v_density`, and
  the `thumbnail_width` / `thumbnail_height` pair, plus
  `has_thumbnail()`, `thumbnail_payload_len()`, `version()`,
  `pixel_aspect_ratio()`, and `h_density_dpi()` / `v_density_dpi()`
  unit-aware accessors that convert `DotsPerCm` to dots-per-inch via
  integer `(d × 254 + 50) / 100` and return `None` for the
  aspect-ratio case where DPI has no meaning. A new top-level
  `parse_jfif_app0(payload) -> Result<JfifApp0>` validator is
  exported for callers that already hold the APP0 payload bytes; it
  enforces the four T.871 §10.1 invariants (`identifier == "JFIF\0"`,
  `units ∈ {0, 1, 2}`, both densities non-zero, trailing
  `3 × Hthumb × Vthumb` bytes fit in the payload) and never
  allocates. `inspect_jpeg` populates `JpegInfo.jfif` automatically
  when the leading APP0 carries valid JFIF; structurally malformed
  JFIF segments still flip the existing `ColorHint::JfifYCbCr` hint
  but leave the typed view as `None` (the magic alone is a sufficient
  colour-convention signal). Nine new tests cover the DPI / DPCM /
  aspect-ratio variants, illegal-units / zero-density / truncated-
  header / bad-identifier / thumbnail-overflow rejection paths, the
  2×2-thumbnail success case, the only-first-segment-wins rule for
  duplicate APP0s, the malformed-but-magic-present hint-still-set
  case, and a JFIF-disjoint Adobe-only stream.

- `encoder::encode_jpeg_progressive_grayscale(width, height, samples,
  stride, quality)` emits a standalone progressive (SOF2)
  single-component grayscale JPEG at 8-bit precision. T.81 §G.1.1
  permits the progressive coding process at every `Nf ∈ 1..=4`; the
  single-component case ships every block's DC and AC coefficients
  across three spectral-selection scans with `(Ss, Se) = (0, 0)` /
  `(1, 5)` / `(6, 63)`, all at `Ah = 0, Al = 0`. The bitstream layout
  is `SOI / JFIF APP0 / DQT (luma) / SOF2 (Nf = 1, H = V = 1, P = 8)
  / DHT (Annex K luma DC + AC) / SOS_DC / scan / SOS_AC_low / scan /
  SOS_AC_high / scan / EOI` — one DQT, one DC + one AC DHT, no chroma
  table, no DRI / `RSTn`. The companion variant
  `encode_jpeg_progressive_grayscale_with_meta(.., meta)` replaces
  the default JFIF APP0 with caller-supplied APP/COM segments
  harvested via [`extract_app_segments`]. The trait-API encoder
  (`MjpegEncoder::send_frame`) now routes `Gray8` input +
  `set_progressive(true)` through the new path; `set_lossless(true)`
  continues to win over progressive (SOF3 lossless takes priority) and
  `set_restart_interval` is ignored on the progressive path (the
  3-component progressive encoder doesn't expose DRI emission either,
  kept consistent so the flag has the same meaning across every
  progressive variant). Six new unit tests cover the SOF2 single-
  component header walker (SOF2 + single DQT + luma DC+AC DHT only +
  exactly three SOS scans at `(0,0)`/`(1,5)`/`(6,63)`), Q=100 ±4 LSB
  near-lossless ceiling, Q=75 ≥30 dB PSNR floor, short-stride and
  short-buffer rejection, and the `_with_meta` APP1 pass-through; one
  new integration test in `tests/roundtrip.rs` covers the trait-API
  routing (SOF2 present, SOF0 + SOF3 absent, round-trip ≥ 20 dB).

- `inspect_jpeg(bytes) -> Result<JpegInfo>` — decode-free typed
  inspector. Walks the JPEG marker prefix (T.81 §B.1) up to the first
  SOS and returns a `JpegInfo` carrying a `SofKind` discriminator
  (Baseline / ExtendedSequential / Progressive / Lossless /
  ExtendedSequentialArith / ProgressiveArith / LosslessArith /
  HierarchicalDct / HierarchicalArith) plus precision / width / height /
  per-component sampling and quant-table selectors, a
  `ChromaSubsampling` discriminator (4:4:4 / 4:2:2 / 4:2:0 / 4:1:1 /
  GrayscaleOnly / Custom) derived from the SOF sampling factors per
  T.81 §A.1.1, a `ColorHint` from the APP0 JFIF (T.871) and APP14
  Adobe (T.872 §6.5.3) tags, and the `restart_interval` from any DRI
  segment before SOS. No entropy decoding, no DCT, no allocation
  proportional to the scan body — the walk is O(prefix-length) and
  stops at the first SOS marker. `SofKind::is_supported_by_decoder` /
  `is_dct` / `is_arithmetic` expose the routing-relevant predicates so
  callers can negotiate fallback without matching every variant by
  hand. Standalone surface — built without the `registry` Cargo
  feature, no `oxideav-core` dep. Exercised by 23 new unit tests in
  `src/jpeg/inspect.rs` covering all SOFn variants + all chroma-
  subsampling classes + APP0/APP14 colour-hint extraction +
  malformed-input rejection (missing SOI, EOI before SOS, SOS before
  SOF, malformed SOF length), plus 8 new integration tests in
  `tests/inspect.rs` that round-trip the inspector against the
  in-tree encoder's baseline / progressive / lossless outputs at
  multiple chroma subsamplings.
- `encoder::encode_jpeg_rgb24(width, height, samples, stride, quality)`
  emits a standalone baseline (SOF0) three-component RGB JPEG at 8-bit
  precision from a packed RGB triple buffer. Components are written with
  IDs `'R' / 'G' / 'B'` (`82 / 71 / 66`), each declared `H = V = 1`, and
  all three bind the single luma quantiser table — the chroma table is
  never emitted. An Adobe APP14 segment with `transform = 0` flags the
  stream as plain R/G/B for conformant decoders. The companion variants
  `encode_jpeg_rgb24_with_opts(.., restart_interval)` and
  `encode_jpeg_rgb24_with_meta(.., restart_interval, meta)` add `DRI +
  RSTn` emission (same `RST0..=RST7` cycling and per-component predictor
  reset the YUV / grayscale paths use, per T.81 §F.1.1.5.2) and APP /
  COM pass-through respectively. The companion baseline decoder now
  detects 3-component RGB via either the Adobe APP14 `transform = 0`
  flag or the `'R'/'G'/'B'` component-id triple and emits a single
  packed `PixelFormat::Rgb24` plane (`stride = width * 3`) instead of
  reinterpreting the planes as YCbCr. The matching `tests/docs_corpus.rs`
  helpers (`infer_pix_fmt` + `flatten_frame`) gain an `Rgb24` branch so
  the `baseline-rgb-32x32` corpus fixture passes its `PsnrFloor`
  threshold without the planar-YUV reinterpretation fallback. The
  registry-side trait API (`MjpegEncoder::send_frame`) accepts
  `PixelFormat::Rgb24` input and routes it through the new baseline RGB
  path; `set_lossless(true)` is ignored for RGB input (lossless mode
  stays grayscale-only). Seven new unit tests cover the encoder shape
  (SOF0 RGB header walker, APP14 transform=0 emission, single DQT, luma
  DC + AC DHT only), Q=100 ±4 LSB near-lossless ceiling, Q=75 ≥30 dB
  PSNR floor, short stride / short buffer rejection, DRI + RSTn
  emission round-trip, and APP1 pass-through (with component-id
  fallback signalling RGB to the decoder). Three new integration tests
  in `tests/roundtrip.rs` cover the trait-API default-quality round
  trip, the lossless-flag-ignored-on-RGB path, and the short-stride
  rejection.
- `encoder::encode_jpeg_grayscale(width, height, samples, stride, quality)`
  emits a standalone baseline (SOF0) single-component grayscale JPEG at
  8-bit precision. The bitstream layout is the usual `SOI / JFIF APP0 /
  DQT (one luma table, scaled by quality) / SOF0 (Nf=1, H=V=1, P=8) /
  DHT (Annex K luma DC + AC) / SOS (Ns=1) / scan / EOI`. The companion
  variants `encode_jpeg_grayscale_with_opts(..., restart_interval)` and
  `encode_jpeg_grayscale_with_meta(..., restart_interval, meta)` add DRI
  + `RSTn` emission (same once-per-`restart_interval`-MCUs scheme the
  YUV baseline path uses, with predictor reset and `RST0..=RST7`
  cycling per T.81 §F.1.1.5.2) and APP/COM pass-through respectively.
  Bitstreams round-trip through the existing SOF0 single-component
  decode path: high quality stays within ±4 LSB on smooth content,
  default quality (75) sits above 30 dB PSNR on synthetic gradients,
  and `Q = 100` collapses to the all-1 luma quantiser so any residual
  error comes from f32 DCT/IDCT rounding alone.
- `MjpegEncoder::send_frame` (registry-side trait API) now accepts
  `PixelFormat::Gray8` without `set_lossless(true)` and routes it to
  the new baseline grayscale path. `set_lossless(true)` keeps the
  bit-exact SOF3 path for the same input. Higher-precision grayscale
  (`Gray10Le` / `Gray12Le` / `Gray16Le`) still requires
  `set_lossless(true)` — the DCT path is 8-bit by spec — and a clear
  `Unsupported` error surfaces when callers forget. Three new
  integration tests in `tests/roundtrip.rs` cover the trait-API
  baseline path (default quality round-trip with PSNR floor), the
  lossless-flag-still-bit-exact path, and the
  high-bit-depth-without-lossless rejection. The existing
  `registry_encoder_gray8_without_lossless_flag_errors` test is
  rewritten as `registry_encoder_gray8_without_lossless_flag_takes_baseline`
  to assert the new SOF0 emission shape, paired with a fresh
  `registry_encoder_gray12_without_lossless_flag_errors` to lock in
  the high-bit-depth rejection contract.

### Changed

- docs: scrub decorative external-implementation attribution from
  `src/encoder.rs` (DEFAULT_QUALITY, `encode_jpeg`, `encode_jpeg_progressive`,
  `encode_jpeg_progressive_sa`), `src/jpeg/quant.rs` (`scale_for_quality`),
  and `src/mjpeg_container.rs` (DEFAULT_FRAME_RATE). Quality-factor scaling
  is described against the Annex K Q=50 base tables; conformant-SOF2 round
  trip phrased neutrally.
- README: paraphrase the residual decorative external-implementation
  attribution from the encoder + progressive sections so the JPEG
  quality-factor scaling and SOF2 round-trip claim match the language used
  in `src/`.

### Added

- `encoder::encode_lossless_jpeg_cmyk(width, height, [c, m, y, k], strides,
  predictor, adobe_transform)` and its
  `encode_lossless_jpeg_cmyk_with_opts(.., restart_interval,
  point_transform)` companion expose a public four-component lossless
  (SOF3) encoder at 8-bit precision. The four planes share one DC Huffman
  table and one predictor selector, all components are declared
  `H_i = V_i = 1` per T.81 §H.1.2, and the Adobe APP14 colour-transform
  flag is honoured identically to the lossy CMYK helpers (`None` → no
  APP14 / plain "regular" CMYK, `Some(0)` → Adobe CMYK with on-the-wire
  inversion, `Some(2)` → Adobe YCCK with K-only on-the-wire inversion).
  Output: a standalone SOF3 JPEG with one interleaved SOS scan.
- Decoder side: SOF3 now accepts a 4-component scan at `P = 8` and the
  shared `decode_lossless_scan` packs the resulting four sample planes
  into a `PixelFormat::Cmyk` `VideoFrame` (4 bytes/pixel `C M Y K`).
  The Adobe APP14 colour transform on the resulting frame is applied
  identically to the existing lossy CMYK render: no APP14 passes the
  four bytes through, transform=0 un-inverts the Adobe-CMYK convention,
  and transform=2 (YCCK) decodes BT.601 YCbCr → RGB → CMY and
  un-inverts K. Wider precisions (`P > 8`) on a 4-component SOF3 are
  rejected with `Error::Unsupported` because the workspace
  `PixelFormat` enum has no high-bit-depth CMYK variant.
- New `lossless_cmyk_*` tests exercise predictor 1..=7 bit-exact
  roundtrip with no APP14, Adobe-CMYK (transform = 0) bit-exact
  roundtrip on a representative predictor sample, the DRI + `RSTn`
  emission path on a width-not-evenly-divided restart interval, the
  point-transform quantisation path (Pt = 2 / output equals input with
  the low Pt bits cleared), and the invalid-predictor + invalid-
  Adobe-transform-byte rejection paths.

- `rtp::packetize_with_opts` + `rtp::PacketizeOpts` add restart-interval
  -aligned scan splitting (opt-in via
  `PacketizeOpts::new(qmode).with_restart_align(true)`) to the RFC 2435
  packetizer. When the source JPEG carries DRI > 0 the aligned path walks
  the entropy-coded scan for `RSTn` boundaries (T.81 §B.1.1.2 byte
  stuffing respected), packs as many complete intervals per fragment as
  the MTU allows, and writes a §3.1.7 Restart Marker header with `F = L
  = 1` plus the index of each fragment's first interval in the 14-bit
  Restart Count (wrapping modulo `0x3FFF`, the value reserved for
  whole-frame reassembly). A single oversize interval returns
  `MjpegError::Unsupported` instead of falling back silently to byte
  boundaries. When the source has no DRI the flag is a no-op and the
  output equals `rtp::packetize(jpeg, max_payload, qmode)`. New tests
  cover the interval walker (3-interval whole-scan walk + 0xFF-stuffed
  intra-interval byte), the tight-MTU one-interval-per-packet path, the
  loose-MTU multiple-intervals-per-packet path, the no-DRI fallthrough,
  the oversize-interval rejection, the qtable-header carriage on the
  first fragment, and a round-trip through `JpegDepacketizer` that
  shows the reassembled scan preserves every source `RSTn` position.

- New `benches/codec.rs` Criterion harness (`cargo bench -p oxideav-mjpeg
  --bench codec`) measures the baseline SOF0 encode (4:2:0 256x256 q75,
  4:4:4 64x64 q75), baseline SOF0 decode (4:2:0 256x256 q75 through
  the `Decoder` trait), progressive SOF2 encode (4:2:0 64x64 q75), and
  SOF3 lossless grayscale encode with predictors 1 (Ra) and 4
  (Ra + Rb − Rc) at 256x256. Every fixture is built deterministically
  in-bench from an xorshift32 seed plus a low-amplitude triangle-wave
  gradient (so the entropy coder sees realistic run-length patterns
  rather than degenerate random-noise worst cases) — no committed
  payload files, no `docs/` reads, no third-party library calls. Pinned
  to `criterion = "0.5"` to match the existing cross-codec
  bench fleet (flac / tta / tiff / magicyuv / huffyuv / pcx / qoi).

- New `arith_decode` cargo-fuzz target wraps fuzz-supplied bytes in a
  minimal SOF9 (extended-sequential arithmetic-coded) JPEG envelope so
  the `src/jpeg/arith.rs` Q-coder (`ArithDecoder::new` / `Initdec` /
  `Renorm_d` / `Byte_in` / `decode_dc_diff` / `decode_ac` /
  `decode_magnitude`) and the `decode_arith_scan` per-component
  statistics + restart bookkeeping execute on every iteration. A
  fuzz-driven control nibble varies component count (1 vs 3), optional
  DAC conditioning, optional DRI (restart interval = 1 MCU), the
  luma sampling factor (4:4:4 vs 4:2:2), and the image dimension
  (8..=64 px square). Bar is "no panic", same as the existing six
  robustness targets in `fuzz/fuzz_targets/`.

## [0.1.7](https://github.com/OxideAV/oxideav-mjpeg/compare/v0.1.6...v0.1.7) - 2026-05-30

### Other

- 4-component CMYK / YCCK progressive (SOF2) decode
- 12-bit precision progressive (SOF2 P=12) decode
- encoder docstring: refresh stale "P=8 only" note for lossless RGB
- lossless (SOF3) three-component decode at every P in 2..=16
- rustfmt — split InBand q if-else across lines
- fuzz packetizer + close five wire-length panic surfaces
- add rtp_depacketize cargo-fuzz target for RFC 2435 surface
- gate fixture corpus on Tier::Exact + Tier::PsnrFloor
- fix i32 overflow in dequantise when Pq=1 (16-bit quantiser)
- 12-bit 4:2:2 (Yuv422P12Le) + 4:4:4 (Yuv444P12Le) YUV
- add decode robustness target + fix seven panic surfaces
- cache static-Q in-band quantization tables across frames (RFC 2435 §4.2)
- add RFC 2435 RTP/JPEG packetizer (encode side)
- add RFC 2435 RTP/JPEG depacketizer
- lossless encoder: restart markers + non-zero point transform
- rewrite library-citation comments to remove external-library references
- lossless (SOF3) three-component encoder + decoder
- lossless (SOF3) grayscale encoder: P=2..=16 + all 7 predictors
- add multi-frame demuxer with seek_to + marker-aware scanner
- compare libjpeg cross-decode in YUV space, not RGB

### Added

- Public 4-component CMYK / YCCK encoder API. The two
  back-end entry points landed for the round-186 CMYK decoder tests —
  `encode_jpeg_cmyk_1111` (baseline SOF0) and
  `encode_jpeg_progressive_cmyk_1111` (SOF2) — were `pub(crate)` and
  gated behind `#[cfg(test)]`, so the production surface only allowed
  CMYK input through the *decode* side. They are now `pub` and
  always-on, joined by two packed-buffer convenience wrappers that
  accept the same `[C, M, Y, K]` interleaved layout (4 bytes per pixel,
  `stride` bytes per row) the decoder produces:
  - `encoder::encode_jpeg_cmyk(width, height, packed, stride, quality,
    adobe_transform)` — baseline SOF0 path.
  - `encoder::encode_jpeg_cmyk_progressive(width, height, packed,
    stride, quality, adobe_transform)` — SOF2 path with the 9-segment
    spectral-selection-only scan decomposition (1 interleaved DC scan
    + 2 per-component AC bands `[1..=5]` then `[6..=63]` for each of
    the four components, `Ah = Al = 0`).

  The `adobe_transform: Option<u8>` argument carries the Adobe APP14
  colour-transform marker: `None` writes no APP14 segment (plain
  "regular" CMYK); `Some(0)` selects Adobe CMYK and inverts every
  sample on the wire; `Some(2)` selects Adobe YCCK, interpreting the
  packed input as `[Y, Cb, Cr, K]` and inverting only the K plane
  (the decoder un-does both transforms on output). Any other `Some(t)`
  is rejected with `Error::InvalidData` since the decoder only
  round-trips `0` and `2`.

  The `MjpegEncoder` trait-API path now accepts
  `PixelFormat::Cmyk` input as well. A new
  `MjpegEncoder::set_adobe_transform(Option<u8>)` knob configures the
  APP14 marker (defaults to `None`); `set_progressive(true)` switches
  the CMYK path from SOF0 to SOF2. The plane stride must be at least
  `width * 4` (the decoder's output stride); shorter strides are
  rejected with a clear `Error::InvalidData`.

  A new `tests/cmyk_roundtrip.rs` integration suite (14 tests) drives
  each combination of `{baseline, progressive} × {None, Some(0)} ×
  {packed wrapper, planar back-end, trait API}` through encode→decode
  and asserts per-component PSNR ≥ 30 dB at Q = 90 — the same
  tolerance the internal `decoder::cmyk_tests` suite enforces, so the
  public-API surface inherits the same correctness floor as the
  previously test-only back-end paths. Error paths (short stride,
  short buffer, invalid Adobe transform on both the free-function and
  trait-API surfaces) are also covered.

- 4-component (CMYK / Adobe YCCK) progressive (SOF2 with `P = 8`)
  decode. T.81 §G.1.1 permits the progressive coding process at every
  component-count the spec admits (`Nf ∈ 1..=4`), but the decoder was
  previously rejecting `SOF2` with `Nf = 4` even though every
  downstream stage already supported the geometry:
  `decode_progressive_scan` is component-count agnostic (interleaved
  DC walks every SOS component, AC scans are always non-interleaved
  per the spec), `init_coef_buffers` already sizes for up to 4
  components, and `render_from_coefs` already produces a packed
  `PixelFormat::Cmyk` plane for `Nf = 4` honouring the Adobe APP14
  colour-transform flag (plain CMYK, Adobe-inverted CMYK at
  transform=0, YCCK at transform=2). The SOF2 4-component path
  therefore lights up by removing the `Nf > 3` rejection; the
  `Nf = 4 & P = 12` combination is still rejected with
  `Error::Unsupported` because the workspace `PixelFormat` enum
  carries no 12-bit CMYK variant.

  A new test-only encoder helper `encode_jpeg_progressive_cmyk_1111`
  emits a 4-component SOF2 progressive JPEG at `P = 8` with the same
  spectral-selection-only scan decomposition the existing
  three-component progressive YUV helpers use: one interleaved DC
  scan (`Ss = Se = 0, Ah = Al = 0`) followed by per-component AC
  bands `[1..=5]` then `[6..=63]` for each of the four components
  (1 + 4 + 4 = 9 SOS segments total). Each component is declared
  `H_i = V_i = 1` so the MCU equals one data unit per component;
  component 1 binds quant table 0 (`luma_q`), components 2/3/4 share
  quant table 1 (`chroma_q`), mirroring the baseline
  `encode_jpeg_cmyk_1111` policy. The helper accepts the same
  `adobe_transform: Option<u8>` parameter as the baseline path
  (`None` → no APP14, `Some(0)` → Adobe CMYK inverted-on-wire,
  `Some(2)` → Adobe YCCK with K-inversion).

  Three new tests under `decoder::cmyk_tests`
  (`cmyk_progressive_plain_roundtrip`,
  `cmyk_progressive_adobe_inverted_roundtrip`,
  `ycck_progressive_k_plane_matches`) drive each transform variant
  through the helper and decode it back, asserting per-component
  PSNR ≥ 30 dB at Q = 90 (same tolerance as the baseline tests they
  mirror). A fourth test (`cmyk_progressive_p12_rejected`) hand-
  crafts a minimal SOF2 segment with `P = 12, Nf = 4` and confirms
  the parser still rejects the unsupported combo with
  `Error::Unsupported` before any scan is read.

- 12-bit precision progressive (SOF2 with `P = 12`) decode. T.81 §G.1.1
  permits the progressive coding process at precision 8 or 12, but the
  decoder previously rejected `SOF2` at `P = 12` with `Error::Unsupported`
  even though `init_coef_buffers` already allocated 12-bit-shaped
  coefficient accumulator planes and `render_from_coefs` already routed
  `P = 12` to the dedicated `render_from_coefs_12bit` path (level shift
  2048, clamp `[0, 4095]`, 16-bit-LE output planes). The progressive
  scan path itself (`decode_progressive_scan` + `prog_decode_dc` /
  `prog_decode_ac_first` / `prog_decode_ac_refine`) operates entirely on
  `i32` coefficient planes, so the increased DC/AC residual magnitude
  range at `P = 12` fits without numeric changes. The `BitReader::get_bits`
  24-bit ceiling accommodates the wider DC categories (up to 15) and AC
  magnitudes (up to 14) the spec admits at `P = 12`.

  Output shape matches the sequential `P = 12` path:
  - Grayscale → `Gray12Le` (one 16-bit-LE plane, low 12 bits carry the
    sample).
  - Three-component YUV at 4:4:4 → `Yuv444P12Le`.
  - Three-component YUV at 4:2:2 → `Yuv422P12Le`.
  - Three-component YUV at 4:2:0 → `Yuv420P12Le`.
  Non-2x luma sampling factors at `P = 12` continue to be rejected with
  `Error::Unsupported` (no `PixelFormat` enum entry for, e.g., 4:1:1 at
  12-bit).

  A new test-only encoder helper `encode_yuv_jpeg_progressive_12bit`
  emits a three-component SOF2 progressive JPEG at `P = 12` with the
  spectral-selection-only scan decomposition (interleaved DC pass +
  Y/Cb/Cr AC bands `[1..=5]` then `[6..=63]`, `Ah = Al = 0`), reusing
  the same Annex K Huffman tables and `DEFAULT_LUMA_Q50` /
  `DEFAULT_CHROMA_Q50` quant tables as the existing 12-bit baseline
  helper `encode_yuv_jpeg_12bit`. Three new tests under
  `decoder::precision_12_tests` (`yuv444_12bit_progressive_roundtrip`,
  `yuv422_12bit_progressive_roundtrip`, `yuv420_12bit_progressive_roundtrip`)
  drive a smooth gradient through the helper and decode it back,
  asserting per-sample closeness against the originals (`diff < 24`,
  same tolerance as the existing baseline 12-bit YUV roundtrip tests).

- High-bit-depth lossless (SOF3) three-component decode. Previously the
  decoder accepted SOF3 RGB-class scans only at `P = 8` (output:
  packed `Rgb24`); decoding at higher precisions raised
  `Error::Unsupported`. The decoder now covers every precision in
  `2..=16` for three-component lossless, with output shape selected by
  precision:
  - `P = 8`      → packed `Rgb24` (one plane, 3 bytes/pixel) —
    unchanged.
  - `P = 10`     → planar `Gbrp10Le` (3 planes, 16-bit LE per sample,
    low 10 bits carry the sample).
  - `P = 12`     → planar `Gbrp12Le`.
  - `P = 14`     → planar `Gbrp14Le`.
  - any other `P` in `2..=16` (i.e. 2..=7, 9, 11, 13, 15, 16)
                 → packed `Rgb48Le` (one plane, 6 bytes/pixel; samples
    narrower than 16 bits sit in the low bits of each 16-bit word).
  Per-component buffer ordering is preserved end-to-end: planes pass
  through encoder → decoder in the same SOS scan order (mirroring the
  existing colour-agnostic `Rgb24` behaviour), so a caller that wants
  canonical `Gbrp*Le` G-B-R layout passes its G, B, R planes to the
  encoder in that order. Roundtrip bit-exactness verified by five new
  integration tests in `tests/lossless_roundtrip.rs`:
  `lossless_rgb_10bit_every_predictor_planar_gbrp10` (every Annex H
  predictor 1..=7), `lossless_rgb_12bit_predictor_4_planar_gbrp12`,
  `lossless_rgb_14bit_predictor_7_planar_gbrp14`,
  `lossless_rgb_16bit_predictor_1_packed_rgb48`, and
  `lossless_rgb_odd_precision_9_widens_to_rgb48`. The previous
  `lossless_rgb_rejects_higher_precision_decode` test (which asserted
  the `Error::Unsupported` for P > 8) is removed in the same commit
  per the workspace "rewrite, don't `#[ignore]`" guardrail.
- `MjpegPixelFormat` (standalone API) gains `Rgb48Le`, `Gbrp10Le`,
  `Gbrp12Le`, and `Gbrp14Le` variants, with bidirectional
  `From<MjpegPixelFormat> for oxideav_core::PixelFormat` / reverse
  mapping in `registry.rs` so the registry-side `CodecParameters`
  surface accepts and produces them.
- `fuzz/fuzz_targets/rtp_packetize.rs`: cargo-fuzz harness covering
  the RFC 2435 RTP/JPEG packetizer (`oxideav_mjpeg::rtp::packetize`).
  Drives arbitrary bytes (≤ 16 KiB) through the encode-side JPEG
  segment walker (`fn parse_jpeg`), which indexes into the input by
  the big-endian length field of each SOI / SOF / DQT / DRI / SOS /
  catch-all segment. The harness samples both `QMode::Quality(1..=99)`
  and `QMode::InBand(128..=255)` (and the rejection paths outside
  those ranges) and a range of `max_payload` MTUs so the
  header-room rejection and the fragment-split loop both run on
  every iteration. Contract: no panic, slice OOB, debug-build
  integer overflow, or infinite loop from a zero-length segment.
  Successful returns are shape-checked: first fragment offset 0,
  marker bit set on the final packet, no payload exceeds the
  caller's `max_payload`. This is now the seventh fuzz harness in
  `fuzz/` alongside `decode`, `jpeg_self_roundtrip`,
  `jpeg_progressive_self_roundtrip`, `libjpeg_encode_oxideav_decode`,
  `oxideav_encode_libjpeg_decode`, and `rtp_depacketize`. Last
  local 15 s baseline (debug binary, no sanitizer instrumentation):
  21 819 067 runs, 0 crashes; the daily reusable fuzz workflow runs
  the release-instrumented build with proper coverage and pulls the
  new harness in automatically via `fuzz/Cargo.toml` auto-discovery.

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

- **RTP/JPEG packetize parser panic surfaces** in `fn parse_jpeg`
  (`oxideav_mjpeg::rtp`):
  - **SOF0/SOF1 `len < 2`** previously underflowed `len - 2` in the
    payload bounds calculation (`body + len - 2 > jpeg.len()`); the
    check now refuses `len < 8` (the SOF fixed-header minimum, T.81
    §B.2.2) up-front before any subtraction.
  - **SOF0/SOF1 with `Nf = 3`** but a declared length too short to
    carry the three 3-byte component records (`8 + 3 * Nf = 17`)
    previously indexed `jpeg[body + 13]` past the segment end. A
    new `len < 8 + 3 * nc` check rejects the truncated header before
    the component-records read.
  - **DQT `len < 2`** previously underflowed `len - 2` when computing
    the table-body slice end; explicit `len < 2` guard added.
  - **SOS `len < 2`** previously yielded a `scan_start = pos + len`
    underflow (and the subsequent `&jpeg[scan_start..scan_end]` slice
    panic inside `packetize`). Bounds-checked.
  - **Catch-all length-prefixed segment with `len == 0`** (any
    unsupported marker — APPn, COM, DHT, …) previously caused
    `pos += 0`, an infinite loop. The arm now requires `len >= 2`
    (the segment must at least carry its own length field) and
    refuses any `pos + len > jpeg.len()`. Five regression tests
    cover the SOF length/component, DQT, SOS, and APP0 cases.

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
