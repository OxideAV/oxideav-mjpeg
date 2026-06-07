# oxideav-mjpeg

Pure-Rust **JPEG / Motion-JPEG** codec and still-image container —
decodes baseline (SOF0), extended-sequential (SOF1), progressive (SOF2)
and lossless (SOF3) JPEGs (single-component grayscale at any precision
`P ∈ 2..=16` plus three-component RGB-class at `P = 8`), encodes
baseline, progressive **and** lossless JPEG (the lossless path covers
single-component grayscale at every precision `P ∈ 2..=16` and
three-component interleaved RGB at every precision `P ∈ 2..=16`, with
every Annex H Table H.1 predictor). YUV 4:4:4 / 4:2:2 / 4:2:0 and
grayscale. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-container = "0.1"
oxideav-mjpeg = "0.1"
```

## Quick use

A JPEG file is a single SOI..EOI byte stream, so the still-image
container is a pass-through: open the file, pull one packet, decode.
Motion-JPEG streams (inside AVI / MOV / AMV / etc.) reuse the same
codec — each video packet is a full JPEG.

```rust
use oxideav_core::{Frame, RuntimeContext};

let mut ctx = RuntimeContext::new();
oxideav_mjpeg::register(&mut ctx);
let codecs = &ctx.codecs;
let containers = &ctx.containers;

let input: Box<dyn oxideav_core::ReadSeek> = Box::new(
    std::io::Cursor::new(std::fs::read("photo.jpg")?),
);
let mut dmx = containers.open("jpeg", input)?;
let stream = &dmx.streams()[0];
let mut dec = codecs.make_decoder(&stream.params)?;

let pkt = dmx.next_packet()?;
dec.send_packet(&pkt)?;
if let Ok(Frame::Video(vf)) = dec.receive_frame() {
    // vf.format is Yuv444P / Yuv422P / Yuv420P / Gray8
    // vf.planes[0..] carry the planar samples.
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Encoder

```rust
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat};

let mut params = CodecParameters::video(CodecId::new("mjpeg"));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Yuv420P);
let mut enc = codecs.make_encoder(&params)?;
enc.send_frame(&Frame::Video(frame_yuv420))?;
let pkt = enc.receive_packet()?;
```

The encoder accepts `Yuv444P`, `Yuv422P`, `Yuv420P`, `Gray8`, or packed
`Rgb24` planar input and emits a standalone baseline JPEG per frame:
SOI, JFIF APP0, DQT, SOF0, DHT (Annex K typical tables), optional DRI,
SOS, entropy scan, EOI. Default quality factor is 75 on the Annex K
Q=50 base-table scaling (see `oxideav_mjpeg::encoder::DEFAULT_QUALITY`);
`encoder::encode_jpeg(frame, quality)` is also exposed for sibling
crates that wrap the same bitstream in custom containers. For
single-component `Gray8` callers that already hold a flat row-major
byte buffer, `encoder::encode_jpeg_grayscale(width, height, samples,
stride, quality)` is the direct entry point; the corresponding
`encode_jpeg_grayscale_with_opts(..., restart_interval)` and
`encode_jpeg_grayscale_with_meta(..., restart_interval, meta)` variants
add DRI + `RSTn` emission and APP/COM pass-through respectively. The
matching `encoder::encode_jpeg_rgb24(width, height, samples, stride,
quality)` entry point + its `_with_opts` / `_with_meta` companions emit
a baseline RGB JPEG from a packed RGB triple buffer: three components
at IDs `'R' / 'G' / 'B'`, every component at `H = V = 1`, every
component bound to the single luma quantiser table, and an Adobe APP14
`transform = 0` segment alongside the JFIF APP0 to flag the stream as
plain R/G/B. The decoder mirrors the convention — RGB JPEGs (signalled
by either the Adobe APP14 flag or the `'R'/'G'/'B'` component-id
triple) round-trip as a single packed `Rgb24` plane with no YCbCr
conversion.

Restart markers (`RSTn` + DRI) are supported for interop and bitstream
resiliency. They are **off by default** — call
`MjpegEncoder::set_restart_interval(n_mcus)` (or use
`encoder::encode_jpeg_with_opts(frame, quality, n_mcus)`) to enable
them. A non-zero value writes a DRI segment before SOS and cycles
`RST0..=RST7` every `n_mcus` macroblocks in the scan, resetting DC
predictors at each marker. Passing `0` preserves the historical
no-restart behaviour.

### Progressive (SOF2) encode

Toggle progressive emission via `MjpegEncoder::set_progressive(true)`
on the concrete encoder (construct it with
`MjpegEncoder::from_params`), or call
`encoder::encode_jpeg_progressive(frame, quality)` directly. The
output is a standalone progressive JPEG with this scan decomposition:

1. Interleaved DC-first scan (`Ss=0, Se=0, Ah=0, Al=0`) covering every
   component.
2. Per-component low-band AC scan (`Ss=1, Se=5, Ah=0, Al=0`) — luma,
   Cb, Cr.
3. Per-component high-band AC scan (`Ss=6, Se=63, Ah=0, Al=0`).

That's 1 + 3 + 3 = 7 `SOS` segments. Restart markers are not emitted
on the progressive path. Compressed size is typically ~10% larger than
the equivalent baseline encode due to the extra SOS/DHT overhead and
per-scan EOB handling (no EOBn runs).

### Progressive with Successive Approximation (SA)

For full T.81 §G.1 compliance call
`encoder::encode_jpeg_progressive_sa(frame, quality)`. This 14-scan
decomposition uses a 1-bit point transform:

- Phase 1 — initial scans (`Al=1`): one interleaved DC scan + 3
  per-component AC low-band scans + 3 per-component AC high-band scans.
  Each coefficient is encoded as `coef >> 1`, dropping the LSB.
- Phase 2 — refinement scans (`Ah=1, Al=0`): DC and AC correction
  scans send the dropped LSB to the decoder, with AC correction bits
  for pre-existing nonzeros interleaved inline during the decoder's
  zero-history walk (T.81 §G.1.2.3).

Output round-trips through any conformant SOF2 decoder with PSNR
≥ 40 dB relative to the equivalent spectral-selection-only encode.

### Progressive (SOF2) single-component grayscale encode

For single-component (`Gray8`) input call
`encoder::encode_jpeg_progressive_grayscale(width, height, samples,
stride, quality)` directly. The bitstream layout is `SOI / JFIF APP0 /
DQT (luma) / SOF2 (Nf = 1, H = V = 1, P = 8) / DHT (Annex K luma DC +
AC) / SOS_DC (Ss=0, Se=0) / scan / SOS_AC_low (Ss=1, Se=5) / scan /
SOS_AC_high (Ss=6, Se=63) / scan / EOI` — three spectral-selection
scans, no successive approximation, no DRI / `RSTn`. The output
round-trips through any conformant SOF2 decoder as a single `Gray8`
plane (max-diff ≤ 4 LSBs at `Q=100` on smooth content; PSNR ≥ 30 dB
at the default `Q=75`). The companion `_with_meta` variant
(`encode_jpeg_progressive_grayscale_with_meta(..., meta)`) replaces
the default JFIF APP0 with caller-supplied APP/COM segments harvested
via [`extract_app_segments`](#metadata-pass-through).

The trait-API encoder routes `Gray8` input + `set_progressive(true)`
to the same path:

```rust
let mut params = CodecParameters::video(CodecId::new("mjpeg"));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Gray8);
let mut enc = MjpegEncoder::from_params(&params)?;
enc.set_progressive(true);
enc.send_frame(&Frame::Video(frame_gray8))?;
```

`set_lossless(true)` continues to override `set_progressive` for
grayscale (the SOF3 lossless path wins), and `set_restart_interval`
is ignored on the progressive path — neither the 3-component nor the
1-component SOF2 encoder emits DRI / `RSTn`.

### Lossless (SOF3) encode

For single-component grayscale input call
`encoder::encode_lossless_jpeg_grayscale(width, height, samples,
stride, precision, predictor)` directly:

- `precision` must be in `2..=16`. Samples for `P ≤ 8` are one byte
  each (`stride` = bytes per row); for `P > 8` they are 16-bit
  little-endian (`stride` = `width * 2`).
- `predictor` selects one of the Annex H Table H.1 spatial
  predictors `1..=7` (1 = Ra / left is the safest default; 4..7 are
  two-dimensional and can compress better on smooth images).
- Output is bit-exact: the decoder side recovers every input sample
  verbatim, including the special `Di = 32768` half-modulus case
  (T.81 §H.1.2.2). Point transform is fixed at `Pt = 0` and no
  restart markers are emitted by the default entry point; for non-zero
  `Pt` or DRI + `RSTn` emission call
  `encode_lossless_jpeg_grayscale_with_opts(width, height, samples,
  stride, precision, predictor, restart_interval, point_transform)`.
  On each restart boundary the encoder byte-aligns the stream, writes
  `RST0..=RST7` cycling modulo 8 per T.81 §F.1.1.5.2, and re-seeds the
  predictor history to the per-component origin `2^(P − Pt − 1)`
  (§H.1.2.1). With `Pt > 0` every input sample is right-shifted by `Pt`
  before prediction; the decoder side then left-shifts the
  reconstructed sample by the same `Pt` on output.

The same path is available through the trait-API encoder:

```rust
let mut params = CodecParameters::video(CodecId::new("mjpeg"));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Gray12Le);
let mut enc = MjpegEncoder::from_params(&params)?;
enc.set_lossless(true);
enc.set_lossless_predictor(4);
enc.send_frame(&frame)?;
```

Without `set_lossless(true)` the trait-API encoder rejects grayscale
input rather than silently downgrading the bitstream.

### Lossless (SOF3) RGB / three-component encode

For three-component (R, G, B / or any three independent monochrome
planes) lossless output call `encoder::encode_lossless_jpeg_rgb(width,
height, [r, g, b], strides, precision, predictor)` directly:

- The three planes share one shared DC Huffman table (Td = 0) and one
  predictor selector. Each component is modeled independently per
  T.81 §H.1.2 — neighbours come from the same plane only.
- Each component is declared `H_i = V_i = 1`, so the MCU at every
  pixel position is exactly one residual per component in scan order
  (component IDs 1, 2, 3). Output: a standalone SOF3 JPEG with one
  interleaved SOS scan.
- `precision` is the same `2..=16` range as the grayscale entry point.
  Decode output is shaped by precision:
  - `P = 8`     → packed `Rgb24` (one plane, 3 bytes/pixel).
  - `P = 10`    → planar `Gbrp10Le` (3 planes, 16-bit LE storage).
  - `P = 12`    → planar `Gbrp12Le`.
  - `P = 14`    → planar `Gbrp14Le`.
  - any other P → packed `Rgb48Le` (one plane, 6 bytes/pixel — samples
    narrower than 16 bits sit in the low bits of each 16-bit word).
- The codec is colour-agnostic on both sides: callers pass planes in
  whatever channel order they want (R-G-B, G-B-R, etc.) to the encoder,
  and the decoder hands them back in the same SOS scan order — both for
  the 8-bit packed-`Rgb24` path and the high-bit-depth planar paths.
  Callers that want the canonical G-B-R plane order of `Gbrp*Le` should
  pass G, B, R to the encoder in that order.
- For DRI + `RSTn` emission or a non-zero point transform call
  `encode_lossless_jpeg_rgb_with_opts(width, height, [r, g, b],
  strides, precision, predictor, restart_interval, point_transform)`.
  Both options behave identically to the grayscale variant; restarts
  reset every component's predictor in lockstep, and `Pt` shifts every
  sample of every plane uniformly.

### Lossless (SOF3) CMYK / four-component encode

For four-component (C, M, Y, K — or any four independent monochrome
planes) lossless output at 8-bit precision call
`encoder::encode_lossless_jpeg_cmyk(width, height, [c, m, y, k], strides,
predictor, adobe_transform)` directly:

- Each component is modeled independently per T.81 §H.1.2; the four
  planes share one DC Huffman table and one predictor selector. Each
  component is declared `H_i = V_i = 1`, so the MCU at every pixel
  position is exactly one residual per component in scan order
  (component IDs 1, 2, 3, 4). Output: a standalone SOF3 JPEG with one
  interleaved SOS scan.
- `precision` is fixed at 8 bits — the workspace `PixelFormat` enum
  has no high-bit-depth CMYK variant, so the four-component lossless
  path is `P = 8` only. Output: packed `PixelFormat::Cmyk` (one plane,
  4 bytes/pixel in `C, M, Y, K` order).
- `adobe_transform` selects the APP14 colour-transform marker, identical
  to the lossy CMYK helpers: `None` writes no APP14 (plain "regular"
  CMYK), `Some(0)` selects Adobe CMYK and inverts every sample on the
  wire before predictive coding, `Some(2)` selects Adobe YCCK
  (interpret the packed input as `[Y, Cb, Cr, K]` and invert only K
  before coding). The decoder un-does both transforms on output, so a
  no-APP14 or Adobe-CMYK round-trip is bit-exact.
- For DRI + `RSTn` emission or a non-zero point transform call
  `encode_lossless_jpeg_cmyk_with_opts(width, height, [c, m, y, k],
  strides, predictor, adobe_transform, restart_interval, point_transform)`.
  Both options behave identically to the grayscale and three-component
  variants; restarts reset every component's predictor in lockstep, and
  `Pt` shifts every sample of every plane uniformly.

### 4-component CMYK / YCCK encode

The 4-component (CMYK / Adobe YCCK) decode paths landed in earlier
rounds are now matched by a public encoder API. Both a baseline
(SOF0) and a progressive (SOF2) variant accept the same packed
`[C, M, Y, K]` interleaved buffer the decoder produces (4 bytes per
pixel, `stride` bytes per row), so round-tripping a decoded CMYK
frame back into a JPEG is a single call:

```rust
use oxideav_mjpeg::encoder::{encode_jpeg_cmyk, encode_jpeg_cmyk_progressive};

let jpeg = encode_jpeg_cmyk(width, height, &packed, width as usize * 4, 90, None)?;
let prog = encode_jpeg_cmyk_progressive(width, height, &packed, width as usize * 4, 90, None)?;
# Ok::<(), oxideav_mjpeg::MjpegError>(())
```

`adobe_transform` selects the Adobe APP14 colour-transform marker:
`None` writes no APP14 (plain "regular" CMYK), `Some(0)` selects
Adobe CMYK and inverts every sample on the wire, `Some(2)` selects
Adobe YCCK, interpreting the packed input as `[Y, Cb, Cr, K]` and
inverting only the K plane (the decoder un-does both transforms on
output). The two per-plane back-end entry points
`encoder::encode_jpeg_cmyk_1111` / `encode_jpeg_progressive_cmyk_1111`
are also `pub` for callers that already hold four separate component
buffers.

The trait-API encoder accepts CMYK input as well:

```rust
let mut params = CodecParameters::video(CodecId::new("mjpeg"));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Cmyk);
let mut enc = MjpegEncoder::from_params(&params)?;
enc.set_adobe_transform(Some(2))?; // None / Some(0) / Some(2)
enc.set_progressive(true);         // optional — SOF2 instead of SOF0
enc.send_frame(&frame)?;
```

The plane stride must be at least `width * 4`; shorter strides are
rejected with `Error::InvalidData`.

### Metadata pass-through

All encoder entry points have `*_with_meta` variants that accept a
`meta: &[u8]` byte slice of pre-serialised APP/COM segments to embed
immediately after SOI (replacing the default JFIF APP0). Use
`encoder::extract_app_segments(jpeg)` to harvest APP0-APP15 and COM
segments from an existing JPEG for pass-through to the re-encoded
output.

### RTP/JPEG depacketization (RFC 2435)

Motion-JPEG carried over RTP omits the JPEG frame and scan headers from
the wire (abbreviated table-specification format) and fragments the
entropy-coded scan across packets. `rtp::JpegDepacketizer` reassembles
those fragments and reconstructs the absent SOI / DQT / SOF0 / DHT /
[DRI] / SOS / EOI marker segments into a complete JPEG interchange
stream the decoder consumes directly.

```rust
use oxideav_mjpeg::rtp::{JpegDepacketizer, Progress};

let mut dp = JpegDepacketizer::new();
// `payload` = one RTP packet body with the 12-byte RTP fixed header
// already stripped; `marker` = the RTP marker bit (set on the last
// fragment of a frame).
# let payload: &[u8] = &[];
# let marker = false;
match dp.push(payload, marker)? {
    Progress::NeedMore => { /* await further fragments */ }
    Progress::Frame(jpeg) => { /* `jpeg` is a complete SOI..EOI stream */ }
}
# Ok::<(), oxideav_mjpeg::MjpegError>(())
```

Coverage:

- Well-known fixed type mappings 0/64 (4:2:2-class, `H=2 V=1` luma) and
  1/65 (4:2:0-class, `H=2 V=2` luma), three-component YUV interleaved
  scan (§4.1).
- Quantization tables recovered from the Q field via the Independent
  JPEG Group scale formula over Annex K.1 / K.2 for `Q ∈ 1..=99` (§4.2),
  or read in-band from the §3.1.8 Quantization Table header for
  `Q ∈ 128..=255` (8-bit, plus 16-bit saturated to the emitted 8-bit
  DQT).
- Cross-frame in-band table caching (§4.2): a static `Q ∈ 128..=254` may
  carry its tables once and omit them (`Length = 0`) on later frames; the
  depacketizer caches them per Q value and reuses the cached pair, so a
  multi-frame static-Q stream keeps decoding. `Q = 255` is dynamic and
  never cached (tables reload every frame). `reset()` keeps the cache;
  `new()` starts fresh.
- Types 64..=127 consume the §3.1.7 Restart Marker header and emit a DRI
  segment with the carried interval.
- Fragment reassembly keyed on the §3.1.2 Fragment Offset, so misordered
  intra-frame delivery is tolerated as long as the marker-bit fragment
  arrives.

### RTP/JPEG packetization (RFC 2435)

`rtp::packetize(jpeg, max_payload, qmode)` is the encode-side inverse:
it parses a complete baseline JPEG, strips the frame/scan headers, and
emits a `Vec<rtp::JpegPacket>` of RTP/JPEG payloads ready to drop after
the RTP fixed header.

```rust
use oxideav_mjpeg::rtp::{packetize, QMode};

# let jpeg: &[u8] = &[];
// `jpeg` = a complete baseline SOF0/SOF1 4:2:2 or 4:2:0 YUV stream.
let packets = packetize(jpeg, 1400, QMode::InBand(255))?;
for pkt in &packets {
    // Prepend a 12-byte RTP header: same 90 kHz timestamp across the
    // frame, ascending sequence numbers, and the marker bit set when
    // `pkt.marker` is true.
    send_rtp(&pkt.payload, pkt.marker);
}
# fn send_rtp(_p: &[u8], _m: bool) {}
# Ok::<(), oxideav_mjpeg::MjpegError>(())
```

Coverage:

- Luma sampling `2x1` → type 0 (4:2:2), `2x2` → type 1 (4:2:0); chroma must
  be `1x1` (the well-known §4.1 layout).
- A source DRI promotes the type to 64/65 and writes the §3.1.7 Restart
  Marker header. By default the chunks span arbitrary byte boundaries and
  the header signals whole-frame reassembly (`F = L = 1`, Restart Count
  `0x3FFF`); pass `PacketizeOpts::new(qmode).with_restart_align(true)` to
  `packetize_with_opts` to split the scan on restart-interval boundaries
  instead — each emitted fragment then carries one or more complete
  intervals, sets `F = L = 1`, and reports its first interval's index in
  the 14-bit Restart Count (wrapping modulo `0x3FFF`).
- `QMode::Quality(1..=99)` carries an IJG-quality Q value (receiver
  regenerates the Annex K tables); `QMode::InBand(128..=255)` carries the
  JPEG's own two DQT tables in a §3.1.8 Quantization Table header on the
  first fragment.
- The scan is fragmented at `max_payload` (header bytes counted); the first
  fragment has offset 0, the last has `JpegPacket::marker == true`.

Lacks: RTP transport framing itself (the 12-byte RTP fixed header,
sequence numbering, 90 kHz timestamping stay the caller's job),
packetization of progressive / lossless / grayscale / CMYK JPEGs (no
well-known RTP/JPEG type — `Unsupported`), out-of-band table negotiation
via a session-setup protocol on depacketize (a static `Q ≥ 128` frame
whose tables were never sent in-band, nor cached from an earlier frame,
→ `Unsupported`), and the dynamic non-well-known types 128..=255.

### Codec / container IDs

- Codec: `"mjpeg"`. Decoder output / encoder input pixel formats:
  `Yuv444P`, `Yuv422P`, `Yuv420P`, plus `Gray8` on the decode side.
- Container: `"jpeg"`, matches `.jpg` / `.jpeg` / `.jpe` / `.jfif` by
  extension and by `FF D8 FF` magic bytes. One frame per file; muxing
  is a pass-through of the codec packet.
- Container: `"mjpeg-raw"`, matches `.mjpeg` / `.mjpg` by extension.
  Raw concatenated SOI..EOI JPEG frames, one packet per frame.
  Default time base is `1/25` so frame `i` carries `pts = i`; the
  demuxer implements `seek_to(stream, pts)` with a marker-aware
  scanner (no SOI false-positives from APP1 thumbnails / stuffed
  entropy bytes) and a lazy `(pts, byte_offset)` waypoint index
  (one entry every 5 frames).

### Decode-free inspector

`oxideav_mjpeg::inspect_jpeg(bytes) -> Result<JpegInfo>` walks the
marker prefix of a JPEG buffer up to the first SOS (T.81 §B.1) and
returns a typed summary — `SofKind` (Baseline / ExtendedSequential /
Progressive / Lossless / ExtendedSequentialArith /  ProgressiveArith /
LosslessArith / HierarchicalDct / HierarchicalArith), `precision`,
`width`, `height`, per-component sampling / quant-table descriptors,
a `ChromaSubsampling` discriminator (4:4:4 / 4:2:2 / 4:2:0 / 4:1:1 /
GrayscaleOnly / Custom), a `ColorHint` from JFIF (T.871) and Adobe
APP14 (T.872 §6.5.3) tags, the `restart_interval` from a DRI
segment if present, and — when the leading APP0 is a structurally
valid JFIF segment per T.871 §10.1 — an optional `JfifApp0` typed
view (`version_major`/`_minor`, `units: JfifUnits` ∈
{`AspectRatio`, `DotsPerInch`, `DotsPerCm`}, `h_density`/`v_density`,
`thumbnail_width`/`_height`, plus `has_thumbnail()`,
`thumbnail_payload_len()`, `h_density_dpi()` / `v_density_dpi()`
unit-normalised accessors and `pixel_aspect_ratio()` for the
units-= 0 case). When an APP14 Adobe segment is present and
structurally valid per T.872 §6.5.3, an optional `AdobeApp14`
typed view is also exposed (`dct_encode_version`, `flags_0`,
`flags_1`, `transform: AdobeColorTransform` ∈ {`Unknown`,
`YCbCr`, `Ycck`}, plus `is_standard_version()` and
`as_color_hint()` projections). No entropy decoding, no DCT, no
allocation proportional to the scan body — O(prefix-length).
Useful for pipeline triage (pick a target pixel format),
fallback-decoder routing without spinning up the full decode
path, DPI-aware thumbnail sizing, and corpus summarisation. The
`SofKind` exposes `is_supported_by_decoder()`, `is_dct()`, and
`is_arithmetic()` helpers so callers can negotiate without
matching on every variant by hand. Standalone
`parse_jfif_app0(payload) -> Result<JfifApp0>` and
`parse_adobe_app14(payload) -> Result<AdobeApp14>` validators are
also re-exported for callers that already have the APP0 / APP14
payload bytes in hand. Standalone surface — the inspector
requires neither the `registry` feature nor an `oxideav-core` dep.

```rust
use oxideav_mjpeg::{inspect_jpeg, SofKind, ChromaSubsampling};

let info = inspect_jpeg(&jpeg_bytes)?;
println!(
    "{}x{} P={} comps={} subsampling={:?} kind={:?}",
    info.width, info.height, info.precision,
    info.num_components(), info.subsampling, info.sof_kind,
);
if !info.sof_kind.is_supported_by_decoder() {
    // fall back to a different decoder before allocating
}
# Ok::<(), oxideav_mjpeg::MjpegError>(())
```

## Format coverage

Decoder:

- **SOF0** (baseline sequential, Huffman, 8-bit).
- **SOF1** (extended sequential, Huffman, 8-bit) — same scan structure
  as SOF0 at 8-bit, so the same code path handles it.
- **SOF2** (progressive, Huffman) — multi-scan spectral selection and
  successive approximation (DC first + refinement, AC first +
  refinement with EOB-run). Accepts both `P = 8` and `P = 12`; the
  scan path is precision-agnostic (i32 coefficient planes) and the
  EOI render dispatcher routes `P = 12` to the same `Gray12Le` /
  `Yuv444P12Le` / `Yuv422P12Le` / `Yuv420P12Le` shape as the
  sequential 12-bit path below. 4-component CMYK / YCCK is supported
  at `P = 8` and produces the same packed `Cmyk` output the
  sequential path emits (Adobe APP14 transform flag honoured).
- **Non-interleaved sequential scans** (SOF0/SOF1 with one SOS per
  component) — transparently routed through the shared coefficient
  accumulator.
- **12-bit precision** sequential JPEGs (SOF0/SOF1, `P=12`) → 16-bit-LE
  `Gray12Le` for grayscale and `Yuv444P12Le` / `Yuv422P12Le` /
  `Yuv420P12Le` for three-component YUV at 4:4:4 / 4:2:2 / 4:2:0 chroma
  sampling. Level shift is 2048 as per the spec.
- **Lossless JPEG (SOF3)** — single-component grayscale at any
  precision `P ∈ 2..=16`. Annex H predictor reconstruction (bit-exact).
  Output: `Gray8` at P=8, `Gray10Le` / `Gray12Le` at P=10/12, else
  `Gray16Le`. Point transform (`Pt = Al`) honoured.
- **Lossless JPEG (SOF3) three-component** — every precision
  `P ∈ 2..=16`, interleaved scan with each component declared
  `H_i = V_i = 1` (the natural RGB-class layout). Independent
  per-component predictor buffers per Annex H §H.1.2. Output is
  precision-shaped: packed `Rgb24` at `P = 8`, planar `Gbrp10Le` /
  `Gbrp12Le` / `Gbrp14Le` at `P = 10`/12/14, packed `Rgb48Le` for
  every other precision in the valid range (the low `P` bits carry
  the post-Pt-shift sample, top bits zero — same widen policy the
  grayscale path uses to land `P = 14` in `Gray16Le`).
- **Lossless JPEG (SOF3) four-component** — `P = 8` only (the
  workspace `PixelFormat` enum has no high-bit-depth CMYK variant),
  interleaved scan with each component declared `H_i = V_i = 1`.
  Independent per-component predictor buffers per Annex H §H.1.2.
  Output: packed `PixelFormat::Cmyk` (4 bytes/pixel). Adobe APP14
  colour-transform flag honoured identically to the lossy CMYK
  paths (no APP14 → plain "regular" CMYK, transform=0 → Adobe CMYK
  un-inverted on output, transform=2 → YCCK converted back to CMYK
  via BT.601).
- **CMYK / YCCK** 4-component JPEGs → packed `PixelFormat::Cmyk`.
  Adobe APP14 transform flag honoured: transform=0 (Adobe CMYK, stored
  inverted) un-inverts on decode; transform=2 (YCCK) converts back to
  CMYK via BT.601 YCbCr→RGB→CMY and K inversion; no APP14 → plain
  ("regular", C=0 = no ink) pass-through.
- Chroma subsampling: 4:4:4, 4:2:2, 4:2:0.
- Grayscale (single-component → `Gray8`).
- **Baseline RGB** (3-component SOF0 at `H = V = 1`, signalled by either
  an Adobe APP14 `transform = 0` segment or component IDs `'R'/'G'/'B'`
  in the SOF) → packed `PixelFormat::Rgb24` (single plane,
  `stride = width * 3`). The encoder's matching `encode_jpeg_rgb24_*`
  entry points emit both signals (APP14 + component-id triple) by
  default; the decoder accepts either, so a caller-supplied APP-segment
  override that drops the APP14 still round-trips.
- Restart markers (`RSTn`) + DRI.
- **RTP/JPEG (RFC 2435)** depacketization via `rtp::JpegDepacketizer` —
  reassembles fragmented RTP/JPEG payloads and reconstructs the absent
  frame/scan headers (from the Q field or an in-band quantization-table
  header) into a complete JPEG the decoder consumes. The encode-side
  inverse `rtp::packetize` fragments a baseline JPEG into RTP/JPEG
  payloads. See the RTP/JPEG sections below.
- APP0..APP15 segments skipped cleanly (EXIF/XMP/ICC preserved at the
  container level, not parsed).
- Trailing garbage past EOI is stripped by the demuxer.

Encoder:

- **SOF0** (baseline sequential) — 8-bit Huffman, Annex K tables.
  3-component YUV at 4:4:4 / 4:2:2 / 4:2:0, single-component `Gray8`
  (`H = V = 1`, one DQT + DC/AC luma Huffman pair + one-entry SOS),
  3-component packed `Rgb24` at `H = V = 1` (component IDs
  `'R'/'G'/'B'`, single DQT + DC/AC luma Huffman pair, Adobe APP14
  `transform = 0` emitted alongside JFIF APP0), plus 4-component
  CMYK / YCCK at `H_i = V_i = 1` with the Adobe APP14 colour-transform
  flag configurable via the dedicated public CMYK entry points (and
  the trait API's `set_adobe_transform`).
- **SOF2** (progressive) — spectral-selection decomposition (default:
  7 SOS scans, `Ah=0`, `Al=0`) for 3-component YUV, and a 3-scan
  variant (DC + AC-low + AC-high, `Ss/Se ∈ {(0,0), (1,5), (6,63)}`)
  for single-component `Gray8`. The CMYK / YCCK variant uses a
  9-segment spectral-selection scan decomposition over four components.
  Full successive-approximation decomposition (14 SOS scans, 1-bit
  point transform) is available on the YUV path via
  `encode_jpeg_progressive_sa`. See above.
- **SOF3** (lossless) — single-component grayscale at any precision
  `P ∈ 2..=16`, three-component interleaved (RGB-class) at any
  precision `P ∈ 2..=16`, and four-component interleaved (CMYK-class)
  at `P = 8`, all with `H_i = V_i = 1` per component and every
  Annex H Table H.1 predictor `1..=7`. Bit-exact roundtrip on the
  grayscale, RGB and no-APP14 / Adobe-CMYK four-component paths
  (YCCK is a lossy interop convention by construction — BT.601
  YCbCr → RGB → CMY clamps), including the SSSS=16 / Di=32768
  half-modulus case. Optional DRI + `RSTn` emission and non-zero
  point transform via `encode_lossless_jpeg_grayscale_with_opts` /
  `encode_lossless_jpeg_rgb_with_opts` /
  `encode_lossless_jpeg_cmyk_with_opts`. Restart boundaries re-seed
  every component's predictor to `2^(P − Pt − 1)` per T.81 §H.1.2.1.
- 4:4:4 / 4:2:2 / 4:2:0 YUV input on the lossy paths, plus single-
  component `Gray8` and packed `Rgb24` on the baseline SOF0 path;
  `Gray8` / `Gray10Le` / `Gray12Le` / `Gray16Le` input on the lossless
  path.
- Optional DRI + `RSTn` emission on the baseline path (off by default;
  see the Encoder section above).

Not supported (decoder returns `Error::Unsupported`):

- Hierarchical (SOF5+), arithmetic-coded SOF10..SOF15. SOF9 (extended
  sequential, arithmetic) is supported at `P=8`.
- 12-bit 4-component progressive (SOF2 `Nf = 4, P = 12`) — the
  workspace `PixelFormat` enum has no 12-bit CMYK variant. `P = 8`
  4-component CMYK / YCCK *is* supported on both the sequential
  (SOF0 / SOF1) and the progressive (SOF2) scan decompositions, with
  the Adobe APP14 transform flag honoured on both paths.
- 4-component lossless above `P = 8` (the workspace `PixelFormat`
  enum has no high-bit-depth CMYK variant — wider precisions are
  rejected with `Unsupported`). `P = 8` 4-component lossless *is*
  supported on both encode and decode with the Adobe APP14 transform
  flag honoured on output.
- Lossless with non-unit sampling factors (the spec permits this
  but no real-world corpus exercises it; rejected with
  `Unsupported`).

## Fuzzing

The `fuzz/` sub-crate runs eight cargo-fuzz harnesses against the
public encoder + decoder + RTP surface, executed daily by the
org-wide reusable fuzz workflow:

- `decode` — feeds arbitrary bytes (≤ 64 KiB) through the public
  `Decoder` trait (`make_decoder` → `send_packet` → `receive_frame`).
  Contract: never panic. Covers the SOF / SOS validators (`Tdj`/`Taj`
  / `Tq` selectors, `Nf` / `Ns` bounds, `Hi`/`Vi` factors), the
  multi-SOF rejection, the `Wt × Ht × Nf ≤ 64 Mpx` pixel-budget cap,
  the `BitReader::get_bits(n)` guards (`n == 0` short-circuit, `n > 24`
  rejection), and the `Pq = 1` (16-bit quantiser) × coefficient dequantise
  multiplication (now in `f32` to skip i32 overflow). Last local 60 s
  baseline: 25 694 runs, 0 crashes (cov 2023 / ft 7670).
- `arith_decode` — wraps fuzz-supplied bytes (≤ 16 KiB) in a minimal
  SOF9 (extended-sequential arithmetic-coded) JPEG envelope and pushes
  the result through the same `Decoder` trait. A control nibble drives
  component count (1 vs 3), optional DAC conditioning, optional DRI
  (restart interval = 1 MCU), the luma sampling factor (4:4:4 vs 4:2:2),
  and the image dimension (8..=64 px square), so the
  `src/jpeg/arith.rs` Q-coder (`ArithDecoder::new` / `Initdec` /
  `Renorm_d` / `Byte_in` / `decode_dc_diff` / `decode_ac` /
  `decode_magnitude`) and the `decode_arith_scan` per-component
  statistics + restart-interval bookkeeping execute on every iteration.
  Contract: never panic; see `fuzz_targets/arith_decode.rs` for the
  enumerated panic surfaces (per-component bin indexing in
  `DcStats::bins[0..49]` / `AcStats::bins[0..245]`, the
  `category > 15` magnitude guard, the `decode_ac` `k > se` bound, and
  the restart-mid-scan `Err` path).
- `rtp_depacketize` — feeds arbitrary bytes (≤ 16 KiB) through the
  RFC 2435 RTP/JPEG depacketizer (`rtp::parse_main_header`,
  `rtp::parse_restart_header`, `rtp::JpegDepacketizer::push`),
  splitting the input into up to 8 synthetic packets per iteration
  so the §3.1.2 24-bit fragment-offset reassembly buffer, the
  §3.1.7 Restart Marker header, the §3.1.8 in-band Quantization
  Table header, the §4.2 static-Q cache, the marker-bit close
  path, and the `reset()` cache-retention invariant all run on
  every iteration. Contract: never panic. Assembled frames are
  asserted SOI..EOI; interior correctness is owned by the unit
  tests in `src/rtp.rs`.
- `rtp_packetize` — feeds arbitrary bytes (≤ 16 KiB) through the
  RFC 2435 RTP/JPEG packetizer (`rtp::packetize`). The packetizer
  walks a complete external JPEG byte stream and indexes into it
  by big-endian segment lengths; the harness exercises SOF /
  DQT / DRI / SOS / catch-all length-field bounds checks, the
  `QMode::Quality(1..=99)` and `QMode::InBand(128..=255)`
  validation branches, and a range of `max_payload` MTU buckets
  (16 / 256 / 1400 / 8192). Contract: never panic. Successful
  returns are shape-checked (first fragment offset 0, last
  fragment marker bit set, no payload exceeds `max_payload`).
  Round-trip correctness is owned by the unit tests in
  `src/rtp.rs`. Last local 15 s baseline: 21 819 067 runs, 0
  crashes (debug build, no instrumentation; daily CI runs the
  release-instrumented binary).
- `jpeg_self_roundtrip` / `jpeg_progressive_self_roundtrip` —
  oxideav-mjpeg encode → oxideav-mjpeg decode round-trip with ±2 LSB
  YUV tolerance.
- `libjpeg_encode_oxideav_decode` / `oxideav_encode_libjpeg_decode` —
  cross-decode against system `libturbojpeg` (loaded via `libloading`
  at runtime; no `*-sys` crate in the dep tree).

## Fixture corpus

`tests/docs_corpus.rs` decodes every fixture under
`docs/image/jpeg/fixtures/<name>/` and compares the result against the
reference PGM/PPM. Each fixture is classified into one of two enforced
tiers (no more silent reporting):

- **`Tier::Exact`** (5 fixtures): every sample must equal the reference.
  Covers `tiny-baseline-1x1`, `baseline-grayscale-32x32`,
  `lossless-1986-mode`, `arithmetic-coded` (the SOF9 Q-coder path), and
  `baseline-yuv411-32x32`.
- **`Tier::PsnrFloor { db, exact_pct }`** (11 fixtures): total PSNR and
  total exact-sample percentage must both meet a floor recorded ~0.5–2 dB
  / ~1–2 pp below the observed value. A real regression (worse IDCT
  rounding, sloppier YCbCr→RGB) trips the assert; normal floating-point
  jitter does not flap the suite. Covers `baseline-rgb-32x32`,
  `baseline-yuv422-32x32`, `baseline-yuv420-128x128-q75`,
  `baseline-q1-low-quality`, `baseline-q100-no-loss`,
  `progressive-yuv420-128x128`, `multi-scan-non-interleaved`,
  `extended-sequential-12bit`, `with-restart-interval-8`,
  `with-icc-profile-embedded`, and `without-jfif-marker`.

The two remaining variants `Tier::ReportOnly` and `Tier::Ignored` stay
in the enum for future fixtures that haven't earned a baseline yet.

## Benchmarks

`benches/codec.rs` is a Criterion harness for the encode + decode hot
paths. Run with:

```text
cargo bench -p oxideav-mjpeg --bench codec
```

Six scenarios, each fed by a deterministically-built in-bench fixture
(xorshift32 + low-amplitude triangle-wave gradient — no committed
payload files, no `docs/` reads, no third-party library calls):

- `baseline_encode/yuv420_256x256_q75` — full SOF0 path: forward DCT,
  AAN-style quantise, Huffman run-length encode, marker emission.
- `baseline_encode/yuv444_64x64_q75` — same path on a small 4:4:4
  fixture; isolates per-call header / Huffman-table-construction
  overhead from the per-block cost.
- `baseline_decode/yuv420_256x256_q75` — the inverse, driven through
  the `Decoder` trait so the bench tracks the same code path
  application callers exercise.
- `progressive_encode/yuv420_64x64_q75` — SOF2 spectral-selection
  decomposition (7 SOS scans).
- `lossless_encode/gray_pred1_256x256` — SOF3 grayscale encode with
  predictor 1 (Ra / left), the simplest case.
- `lossless_encode/gray_pred4_256x256` — SOF3 grayscale encode with
  predictor 4 (Ra + Rb − Rc), the most expensive 2-D Table H.1
  variant; A/B against `pred1` measures the predictor-loop cost.

Headline numbers on the round-209 dev machine (Apple Silicon, release
profile, criterion `--quick`): baseline 4:2:0 encode 256x256 q75 runs
~185 µs / call (≈ 353 Melem/s); the matching decode runs ~248 µs /
call (≈ 264 Melem/s). The 256x256 lossless grayscale encode runs
~370 µs / call independent of predictor choice (the magnitude /
Huffman emission dominates the per-sample cost — the four extra
predictor arithmetic ops in pred=4 disappear into the noise).

## License

MIT — see [LICENSE](LICENSE).
