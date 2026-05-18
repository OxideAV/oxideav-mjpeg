# oxideav-mjpeg

Pure-Rust **JPEG / Motion-JPEG** codec and still-image container —
decodes baseline (SOF0), extended-sequential (SOF1), progressive (SOF2)
and lossless (SOF3) 8-bit JPEGs, encodes baseline, progressive **and**
lossless JPEG (the lossless path covers single-component grayscale at
every precision `P ∈ 2..=16` and every Annex H Table H.1 predictor).
YUV 4:4:4 / 4:2:2 / 4:2:0 and grayscale. Zero C dependencies.

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

The encoder accepts `Yuv444P`, `Yuv422P`, or `Yuv420P` planar input and
emits a standalone baseline JPEG per frame: SOI, JFIF APP0, DQT, SOF0,
DHT (Annex K typical tables), optional DRI, SOS, entropy scan, EOI.
Default quality factor is 75 (libjpeg style); `encoder::encode_jpeg(frame,
quality)` is also exposed for sibling crates that wrap the same
bitstream in custom containers.

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

Output round-trips through ffmpeg, libjpeg, and ImageMagick with PSNR
≥ 40 dB relative to the equivalent spectral-selection-only encode.

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
  restart markers are emitted.

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

### Metadata pass-through

All encoder entry points have `*_with_meta` variants that accept a
`meta: &[u8]` byte slice of pre-serialised APP/COM segments to embed
immediately after SOI (replacing the default JFIF APP0). Use
`encoder::extract_app_segments(jpeg)` to harvest APP0-APP15 and COM
segments from an existing JPEG for pass-through to the re-encoded
output.

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

## Format coverage

Decoder:

- **SOF0** (baseline sequential, Huffman, 8-bit).
- **SOF1** (extended sequential, Huffman, 8-bit) — same scan structure
  as SOF0 at 8-bit, so the same code path handles it.
- **SOF2** (progressive, Huffman, 8-bit) — multi-scan spectral
  selection and successive approximation (DC first + refinement, AC
  first + refinement with EOB-run).
- **Non-interleaved sequential scans** (SOF0/SOF1 with one SOS per
  component) — transparently routed through the shared coefficient
  accumulator.
- **12-bit precision** sequential JPEGs (SOF0/SOF1, `P=12`) → 16-bit-LE
  `Gray12Le` for grayscale and `Yuv420P12Le` for 4:2:0 YUV. Level shift
  is 2048 as per the spec.
- **Lossless JPEG (SOF3)** — single-component grayscale at any
  precision `P ∈ 2..=16`. Annex H predictor reconstruction (bit-exact).
  Output: `Gray8` at P=8, `Gray10Le` / `Gray12Le` at P=10/12, else
  `Gray16Le`. Point transform (`Pt = Al`) honoured.
- **CMYK / YCCK** 4-component JPEGs → packed `PixelFormat::Cmyk`.
  Adobe APP14 transform flag honoured: transform=0 (Adobe CMYK, stored
  inverted) un-inverts on decode; transform=2 (YCCK) converts back to
  CMYK via BT.601 YCbCr→RGB→CMY and K inversion; no APP14 → plain
  ("regular", C=0 = no ink) pass-through.
- Chroma subsampling: 4:4:4, 4:2:2, 4:2:0.
- Grayscale (single-component → `Gray8`).
- Restart markers (`RSTn`) + DRI.
- APP0..APP15 segments skipped cleanly (EXIF/XMP/ICC preserved at the
  container level, not parsed).
- Trailing garbage past EOI is stripped by the demuxer.

Encoder:

- **SOF0** (baseline sequential) — 8-bit Huffman, Annex K tables.
- **SOF2** (progressive) — spectral-selection decomposition (default:
  7 SOS scans, `Ah=0`, `Al=0`) and full successive-approximation
  decomposition (14 SOS scans, 1-bit point transform). See above.
- **SOF3** (lossless) — single-component grayscale at any precision
  `P ∈ 2..=16` and any Annex H Table H.1 predictor `1..=7`. Bit-exact
  roundtrip including the SSSS=16 / Di=32768 half-modulus case.
- 4:4:4 / 4:2:2 / 4:2:0 YUV input on the lossy paths; `Gray8` /
  `Gray10Le` / `Gray12Le` / `Gray16Le` input on the lossless path.
- Optional DRI + `RSTn` emission on the baseline path (off by default;
  see the Encoder section above).

Not supported (decoder returns `Error::Unsupported`):

- Hierarchical (SOF5+), arithmetic-coded (SOF9..SOF15).
- 12-bit progressive (SOF2 with `P=12`), 12-bit 4:2:2 / 4:4:4 YUV.
- Progressive 4-component JPEGs.
- Multi-component lossless JPEGs (encoder + decoder; only grayscale).
- Lossless encoder restart markers and non-zero point transform on
  the encode side (the decoder already supports both).

## License

MIT — see [LICENSE](LICENSE).
