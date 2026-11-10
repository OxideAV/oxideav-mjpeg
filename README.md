# oxideav-mjpeg

[![CI](https://github.com/OxideAV/oxideav-mjpeg/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-mjpeg/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-mjpeg.svg)](https://crates.io/crates/oxideav-mjpeg) [![docs.rs](https://docs.rs/oxideav-mjpeg/badge.svg)](https://docs.rs/oxideav-mjpeg) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust **JPEG** still-image codec and **Motion-JPEG** video codec,
zero C dependencies. One decoder behind two doors:

* a **standalone still-image API** at the crate root — `probe` / `info`
  / `decode` / `decode_rgb8` / `decode_rgba8` / `encode` / `encode_rgb8`
  / `encode_rgba8` over `JpegImage`, usable with
  `default-features = false` and no `oxideav-core` (the workspace
  image-crate API contract, `IMAGE_CRATE_API.md`);
* the **framework video codec** (`registry` feature, default-on): the
  `mjpeg` codec + `jpeg` / `mjpeg-raw` containers in the `oxideav-core`
  registry, each video packet one complete JPEG.

Decodes every T.81 SOFn family — baseline (SOF0), extended-sequential
(SOF1 / SOF9), progressive (SOF2 / SOF10), lossless (SOF3 / SOF11) and
the full hierarchical mode (Annex J, DHP-introduced spatial-lossless and
DCT progressions) — at 8 / 12-bit DCT precision and `P ∈ 2..=16`
lossless, with any §A.1.1 sampling layout, Huffman or arithmetic entropy
coding, JFIF / Adobe APP14 colour signalling and CMYK / YCCK. Encodes
sequential (8 / 12-bit), progressive, lossless, arithmetic and
hierarchical streams; **the general T.81 writer** (`oxideav_mjpeg::t81`)
is the one JPEG encoder sibling crates build on.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Standalone use

```toml
[dependencies]
oxideav-mjpeg = { version = "0.1", default-features = false }
```

```rust
# fn main() -> Result<(), oxideav_mjpeg::Error> {
let bytes = std::fs::read("in.jpg").map_err(oxideav_mjpeg::Error::from)?;
if oxideav_mjpeg::probe(&bytes) {
    let info = oxideav_mjpeg::info(&bytes)?;      // header only: dims, layout, precision, metadata presence
    let img  = oxideav_mjpeg::decode(&bytes)?;    // JpegImage, native layout (e.g. YuvJ420P)
    let rgba: Vec<u8> = img.to_rgba8();           // tightly packed RGBA, 4 × width bytes per row
    let (w, h) = (img.width(), img.height());
    assert_eq!(info.format, img.format());

    let opts = oxideav_mjpeg::EncodeOptions::default().with_quality(90);
    let out: Vec<u8> = oxideav_mjpeg::encode_rgba8(w, h, &rgba, &opts)?; // alpha dropped (JPEG has none)
    std::fs::write("out.jpg", out).map_err(oxideav_mjpeg::Error::from)?;
}
# Ok(()) }
```

| Item | What it does |
|---|---|
| `probe(&[u8]) -> bool` | `SOI` + marker-prefix sniff; no allocation, never panics. |
| `info(&[u8]) -> Result<ImageInfo>` | Header walk up to the first `SOS`: `width`, `height`, `format` (the layout `decode` will produce), `precision`, `components`, `progressive` / `lossless` / `arithmetic` / `hierarchical`, `color`, `has_jfif` / `has_adobe` / `has_icc` / `has_exif` / `has_xmp`, `comments` (the `COM` count); `frames = 1`, `has_alpha = false`. |
| `decode(&[u8]) -> Result<JpegImage>` | Native layout, `color` + `metadata` filled. `decode_with(&[u8], &DecodeOptions)` adds limits, strict mode and a §B.5 tables stream. |
| `decode_rgb8` / `decode_rgba8` | One call to `RgbImage { width, height, data }` / `RgbaImage` (3 / 4 bytes per pixel, alpha `255`). |
| `decode_from<R: Read>(r)` | Reads to end, then `decode`. |
| `encode(&JpegImage, &EncodeOptions) -> Result<Vec<u8>>` | Writes the image **as given** (layout → stream table below); layouts the chosen process cannot carry are `Error::Unsupported`, never converted silently. |
| `encode_rgb8(w, h, &[u8], &opts)` | RGB → full-range YCbCr (T.871 §7), chroma box-filtered to `opts.chroma` (4:2:0 default), JFIF APP0 written. |
| `encode_rgba8(w, h, &[u8], &opts)` | Same, **alpha dropped** — JPEG has no alpha mechanism. |
| `encode_to<W: Write>(&img, &opts, w)` | Streaming variant of `encode`. |
| `read_coefficients(&[u8]) -> Result<JpegCoefficients>` | The quantised DCT coefficients and quantisation tables of a DCT frame (sequential / progressive, Huffman / arithmetic, `P` = 8 / 12), entropy-decoded only: no dequantisation, no IDCT (see *Coefficient level*). `read_coefficients_with` takes `DecodeOptions` limits / tables. |
| `decode_coefficients(&JpegCoefficients, &DecodeOptions)` / `encode_coefficients(&JpegCoefficients, &EncodeOptions)` | Render a record with this crate's dequantisation + IDCT; entropy-code one (Huffman sequential / progressive, typical or optimal tables) without touching a coefficient — a lossless transcode. |

`JpegImage { width, height, format: PixelFormat, planes: Vec<Plane>,
color: ColorInfo, metadata: Metadata, precision }` with `new` /
`from_rgb8` / `from_rgba8` (alpha dropped) — each a `Result`, refusing
a geometry outside T.81's `1..=65535`, a wrong plane count, a short
stride or a short buffer with `InvalidData` — / `with_color` /
`with_metadata` / `with_precision`, `as_bytes()` (packed layouts),
`into_raw()` (planes concatenated), `to_rgb8()` / `to_rgba8()`.
`Plane { stride, data }`; `PixelFormat = MjpegPixelFormat`; `Error =
MjpegError` (`InvalidData`, `Unsupported`, `LimitExceeded`,
`Io(std::io::Error)`, plus
the video path's `Eof` / `NeedMore`). Every public record is
`#[non_exhaustive]` with constructors / `with_*` builders.

## Framework use

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-mjpeg = "0.1"          # `registry` feature on by default
```

```rust
use oxideav_core::{Frame, RuntimeContext};

let mut ctx = RuntimeContext::new();
oxideav_mjpeg::register(&mut ctx);           // codec `mjpeg` + containers `jpeg`, `mjpeg-raw`

let input: Box<dyn oxideav_core::ReadSeek> = Box::new(
    std::io::Cursor::new(std::fs::read("photo.jpg")?),
);
let mut dmx = ctx.containers.open_demuxer("jpeg", input, &ctx.codecs)?;
let params = dmx.streams()[0].params.clone();
let mut dec = ctx.codecs.first_decoder(&params)?;

let pkt = dmx.next_packet()?;
dec.send_packet(&pkt)?;
if let Ok(Frame::Video(vf)) = dec.receive_frame() {
    // vf.planes[..] carry the planar samples; the pixel format and
    // dimensions travel in `params` (slim VideoFrame).
    let _ = vf;
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`register(&mut RuntimeContext)` (also `register_codecs` /
`register_containers`), `make_decoder` / `make_encoder`, the
`MjpegEncoder` knobs (`set_progressive`, `set_lossless`,
`set_arithmetic`, `set_restart_interval`, `set_adobe_transform`,
`set_jfif_policy`, `set_stream_metadata`,
`set_encode_options(EncodeOptions)`) and the `MjpegEncoderOptions`
`CodecOptions` schema. Metadata rides as core blobs: the demuxer
declares the picture's ICC / Exif / XMP on `params.blobs`, the decoder
puts each frame's own on `vf.blobs()`, the encoder embeds
`params.blobs` into every frame with a frame's blobs overriding per
kind — see *Metadata and colour*. The encoder declares every input layout it
accepts through `CodecCapabilities::accepted_pixel_formats`
(`registry::ACCEPTED_PIXEL_FORMATS`, 19 layouts; `coded_pixel_format`
says what the decoder reports for each) and stamps its output
parameters with that coded label and colour signal — see *Registry
encoder contract* below. The registry `Decoder` is a thin adapter over
the standalone `decode_with` — one implementation — and the conversions
`From<JpegImage> for VideoFrame`, `From<MjpegPixelFormat> for
PixelFormat` / `TryFrom<PixelFormat>` (1:1 by name) and
`From<ColorInfo> for ColorSignal` connect the two layers; the frame
bridge back is `JpegImage::from_video_frame(&VideoFrame,
&CodecParameters)` (geometry, pixel format and colour signal from the
parameters, plane count validated) or `JpegImage::try_from((&frame,
&params))`. The Motion-JPEG path keeps the crate-local
`MjpegFrame::from(img)` / `JpegImage::from_frame(frame, w, h, format)`.

**Motion-JPEG.** The same decoder serves video: AVI / MOV / AMV `MJPG`
packets are complete JPEG interchange streams, the `mjpeg-raw`
container splits concatenated `SOI … EOI` frames (with seek), TIFF
`JPEGTables` abbreviated streams arrive through `CodecParameters::
extradata`, and RFC 2435 RTP/JPEG is reassembled by `rtp::
JpegDepacketizer`. The historical `decoder::decode_jpeg(bytes, pts)` /
`decode_jpeg_with_tables` and `t81::JpegEncodeOptions` names remain as
`#[deprecated]` wrappers / alias for one release.

**Mid-stream shape changes.** Every Motion-JPEG frame carries its own
frame header, so a stream may change size, component count or sampling
(and with them the output pixel format) at any frame. The decoder
states the shape up front through `output_params()` (from the first
packet's frame header) and, at each change, returns one
`Frame::StreamParams` event immediately before the first frame of the
new shape — `output_params()` and `output_params_generation()` have
moved by then and the event's payload equals `output_params()`. No
event for the first shape or for a constant stream; the same rule
holds after `reset()` (which drops the undecoded packet and a frame
held behind its event) and while draining after `flush()`. A packet
carrying an interlaced field pair (two interchange streams of one
frame header) is one picture of the first field's shape.
`receive_arena_frame` skips the event; a consumer that only watches
the counter skips it with `Frame::is_stream_params()`. The encoder
accepts the event in `send_frame`: MJPEG is intra-only, so it
re-configures for the new size / input layout from the next frame on
(knobs and option bag kept, `output_params()` relabelled); an
abbreviated stream refuses it with `Error::Unsupported` (its tables
are published once in `extradata`).

## Supported layouts

Decode — the layout `decode` / `info` report for a stream. Every YCbCr
JPEG is full range (T.871 §7; T.872 §6.1 extends the relationship to
streams without a JFIF segment), so the 8-bit planar layouts are always
the `YuvJ*` family, JFIF or not; the range-agnostic `Yuv*` names are
accepted on encode as the same layouts:

| Stream | `PixelFormat` | Notes |
|---|---|---|
| 1 component, `P = 8` | `Gray8` | |
| 1 component, `P = 12` DCT; lossless `P = 10 / 12` | `Gray12Le` / `Gray10Le` | 16-bit LE storage |
| 1 component, lossless other `P` | `Gray16Le` | sample in the low `P` bits; `JpegImage::precision` says which |
| 3 components YCbCr, `P = 8` | `YuvJ444P` / `YuvJ422P` / `YuvJ420P` / `Yuv411P` | luma `(Hmax, Vmax)` over `1×1` chroma; every other §A.1.1 combination is resampled to 4:4:4 (`DecodeOptions::upsampling`) |
| 3 components YCbCr, `P = 12` | `Yuv444P12Le` / `Yuv422P12Le` / `Yuv420P12Le` | |
| 3 components RGB-coded (Adobe `transform = 0` or ids `R G B`), `P = 8` | `Rgb24` | packed |
| 3 components lossless, `P = 10 / 12 / 14` | `Gbrp10Le` / `Gbrp12Le` / `Gbrp14Le` | planes in scan order |
| 3 components lossless, other `P` | `Rgb48Le` | low `P` bits |
| 4 components (CMYK / Adobe-inverted CMYK / YCCK), `P = 8` | `Cmyk` | plain ink amounts, `(0,0,0,0)` = white; Adobe complement / YCCK undone |

Encode (`encode(&JpegImage, …)`, `Auto` signalling):

| Layout | Components | Signalling | Processes |
|---|---|---|---|
| `Gray8` | 1 | JFIF | sequential / progressive / lossless |
| `Gray12Le` | 1 | JFIF | sequential / progressive / lossless (`P = 12`) |
| `Gray10Le` / `Gray16Le` | 1 | JFIF | lossless only |
| `Yuv4xxP` / `YuvJ4xxP` | 3 subsampled | JFIF | sequential / progressive / lossless |
| `Yuv4xxP12Le` | 3 subsampled | JFIF | sequential / progressive / lossless (`P = 12`) |
| `Rgb24` | 3 | Adobe APP14 `transform = 0`, ids `R G B` | sequential / progressive / lossless |
| `Rgb48Le` | 3 | Adobe RGB | lossless only |
| `Gbrp10Le` / `Gbrp12Le` / `Gbrp14Le` | 3 (order kept, ids 1..3) | none | lossless only |
| `Cmyk` | 4 | Adobe APP14 `transform = 0`, samples complemented on the wire | sequential / progressive / lossless |

Lossless round trips (`decode(encode(img))`) reproduce planes and
metadata exactly for every layout above (`tests/standalone_api.rs`,
`src/api.rs` tests).

## Options

`EncodeOptions` (`Default` + `with_*`; the renamed `t81::JpegEncodeOptions`):

| Field | Default | Meaning |
|---|---|---|
| `quality` | 75 | `1..=100`, Annex K quantiser scaling (DCT processes). |
| `process` | `Sequential` | `with_progressive(bool)`, `with_lossless(predictor)`, `with_process(JpegProcess)`. |
| `tables` | `Optimal` | Annex K.2 `Optimal` tables from the frame's own statistics (one quantisation pass feeds both the counting and the emission) or K.3 `Typical` (`with_optimal_tables(false)`; RTP/JPEG carries only these). |
| `quantisation` | `RateDistortion` | `RateDistortion`: AC levels chosen per block for least squared error + λ · bits (see *Encoder quality*); `Nearest`: T.81 §A.3.4 rounding. |
| `chroma` | `Yuv420` | Layout `encode_rgb8` / `encode_rgba8` convert to (`Yuv444` / `Yuv422` / `Yuv420` / `Yuv411` / `GrayscaleOnly`). `encode(&JpegImage)` uses the image's own layout. |
| `restart_interval` | 0 | `DRI` + `RSTm` every *n* MCUs. |
| `signalling` | `Auto` | `Jfif`, `Rgb`, `Cmyk { adobe_transform }`, `None`. |
| `metadata` | empty | ICC / Exif / XMP blobs and `COM` comments to embed (merged with `JpegImage::metadata`, the option winning per blob — `Metadata::with_overrides`). |
| `jfif` | `Auto` | JFIF `APP0` / Exif `APP1` slot rule (`JfifPolicy::Auto` / `Always` / `Never`, see *Metadata and colour*). |
| `entropy` | `Huffman` | `EntropyCoding::Arithmetic` (`with_arithmetic(true)`) codes the DCT processes with the Annex D arithmetic coder: `SOF9` sequential, `SOF10` progressive, no `DHT`. |
| `conditioning` | `L = 0`, `U = 1`, `Kx = 5` | `ArithConditioning` of the arithmetic models (§F.1.4.4.1.4 / §F.1.4.4.2.1); a non-default value is written as a `DAC` segment. |
| `marker_order` | `QuantFrameHuffman` | Where the tables sit relative to the frame header (§B.2.1 allows all three): `QuantFrameHuffman` = `DQT SOFn DHT`, `TablesBeforeFrame` = `DQT DHT SOFn`, `FrameBeforeTables` = `SOFn DQT DHT`. Scan data is identical. |
| `threads` | 1 | Workers for one picture: the RGB → YCbCr conversion and the transform + quantisation run in that many row bands (entropy coding stays serial). Byte-identical output for every value. The registry encoder takes it from `Encoder::set_execution_context`. |
| `precision`, `sampling`, `table_ids`, `abbreviated` | 8, `[]`, `[]`, `false` | Plane-level writer knobs (`EncodeOptions::encode(w, h, planes)`); `encode(&JpegImage)` derives precision and sampling from the image. |

The registry encoder's `MjpegEncoderOptions` (`CodecOptions` keys
`quality`, `tables`, `process` (`auto` / `sequential` / `progressive` /
`lossless`), `precision`, `restart`, `abbreviated`, `predictor`,
`point_transform`, `sampling`, `chroma`, `adobe_transform`, `jfif`,
`quantisation` (`nearest` / `rd`), `entropy` (`huffman` / `arithmetic`),
`marker_order` (`quant_frame_huffman` / `tables_first` / `frame_first`)) is the string-bag twin of this
record (`#[non_exhaustive]`, `new()` / `with_*`). Its defaults stay
`typical` tables and `nearest` levels, the streams RTP/JPEG and
table-less Motion-JPEG consumers expect. `chroma` applies the `encode_rgb8` conversion to `Rgb24`
frames (or keeps the luma of YCbCr ones) and `adobe_transform`
(`none` / `cmyk` / `ycck`) selects the four-component signalling —
neither has a plane-level field. Metadata is not an option: the
registry encoder embeds the stream's `CodecParameters::blobs`.

`DecodeOptions` (`Default` + `with_*`):

| Field | Default | Meaning |
|---|---|---|
| `max_width` / `max_height` | `Some(65535)` | Frame-header caps (checked before allocation, on every frame of a hierarchical sequence); `None` = unlimited. |
| `max_pixels` | `Some(1 << 28)` | `width × height` cap (≈ 1 GiB of decoded 4:4:4 samples); `None` = unlimited. |
| `max_bytes` | `None` | Input length cap; `None` = unlimited. |
| `strict` | `false` | Reject stray bytes after `SOI`, trailing bytes after `EOI`, truncated segment lengths and malformed JFIF / Adobe / ICC segments. |
| `tables` | `None` | A §B.5 tables-only stream (TIFF `JPEGTables`) preloaded ahead of the image. |
| `idct` | `IdctMethod::Float` | Inverse DCT of 8- and 12-bit DCT frames: `Float` (factorised separable f32 transform, the most accurate and the fastest: 13 samples of 2 073 600 off the rounded f64 definition on a 1080p q85 frame) or `AccurateInteger` (13-bit fixed point, two passes, 2 extra fraction bits between them). |
| `upsampling` | `ChromaUpsampling::LumaGuided` | Chroma reconstruction: `LumaGuided` (centred linear + luma-guided detail, see *Chroma reconstruction*), `Linear` (centred linear, every ratio), `Triangular` (the reference-compatible ¾ / ¼ kernel for 2:1, 1:2 and 2 × 2) or `Replicate`. Used by the decoder for layouts it resamples to 4:4:4 and by `decode_rgb8` / `decode_rgba8` / `to_rgb8` and their `_with` forms. |
| `ycbcr` | `YcbcrKernel::Exact` | YCbCr → RGB of `decode_rgb8_with` / `decode_rgba8_with`: `Exact` (T.871 §7, one rounding) or `FixedPoint16` (coefficients scaled by 2^16, each product rounded). |
| `deblock` | `false` | Opt-in block-edge smoothing of 8-bit DCT frames (see *Deblocking*). |
| `threads` | 1 | Workers per picture: an interleaved sequential Huffman scan with a restart interval decodes its intervals in parallel (T.81 §E.1.4), byte-identical to the serial decode, which is used whenever the `RSTm` markers do not delimit the intervals exactly; `decode_rgb8_with` / `decode_rgba8_with` also convert to RGB in that many row bands (`ConvertOptions::threads`), with the same pixels. The registry decoder takes it from `Decoder::set_execution_context`. |

Breaches are `Error::LimitExceeded`.

### Chroma reconstruction

T.871 §9 sites subsampled chroma at the centre of the luma samples it
covers and leaves the reconstruction filter to the decoder. The default,
`ChromaUpsampling::LumaGuided`, interpolates linearly at that siting
(every ratio, edges replicated), then adds the luma detail the
subsampling removed. That detail is scaled by the local chroma-on-luma
regression slope over a 3 × 3 chroma neighbourhood,
`a = cov(Ŷ, C) / (var(Ŷ) + 16)`, where `Ŷ` is luma averaged over each
chroma cell. The result is clamped to that neighbourhood's chroma range.
Where chroma does not follow luma, `a ≈ 0` and the output is the linear
interpolation. It applies to YCbCr components at an integer ratio to a
full-resolution luma. Other components (non-integer ratios, CMYK / YCCK)
use `Linear`.

The kernel was chosen by measurement. Reference RGB images were encoded
with this crate's encoder and with `cjpeg` at quality 50 / 75 / 90 and
decoded with each kernel. PSNR and SSIM are measured against the
original RGB (SSIM: 11 × 11 Gaussian window, σ = 1.5, mean of R, G, B).
The photographic set is three 1024 × 768 / 768 × 768 crops of a 6016²
landscape photograph, downscaled 4×. The synthetic set is test-card,
colour-bar and fractal frames from `ffmpeg -f lavfi`. Means over
images and qualities, this crate's encoder:

| Set, layout | `Replicate` | `Triangular` (= `djpeg`) | `Linear` | **`LumaGuided`** |
|---|---|---|---|---|
| photo 4:2:0 | 43.50 dB / 0.9816 | 44.11 / 0.9837 | 44.24 / 0.9845 | **45.03 / 0.9854** |
| photo 4:2:2 | 44.72 / 0.9839 | 45.02 / 0.9845 | 45.09 / 0.9849 | **45.26 / 0.9851** |
| photo 4:1:1 | 43.49 / 0.9816 | 43.49 / 0.9816 | 44.07 / 0.9833 | **44.75 / 0.9841** |
| synthetic 4:2:0 | **34.17** / 0.9407 | 29.28 / 0.9390 | 29.28 / 0.9397 | 33.11 / **0.9518** |

`cjpeg`-encoded input ranks the same way (photo 4:2:0: 43.53 / 44.15 /
44.28 / **45.08** dB). `LumaGuided` has the best SSIM in every cell and
the best PSNR on photographs. Replication keeps the PSNR lead only on
synthetic graphics whose colour edges fall on the 2-pixel chroma grid,
where it is exact and every interpolator blurs. Sharper separable
kernels lose to linear on JPEG chroma, because the encoder's box
prefilter and the quantiser leave nothing for them to restore and they
overshoot. On the 4:2:0 daylight crop at quality 75, Catmull-Rom loses
0.17 dB to linear, Lanczos-3 loses 0.28 dB and Mitchell–Netravali loses
0.03 dB. A back-projection step on top of either kernel also lost
(−0.2 to −0.7 dB). `ε = 16` and the clamp were picked on the same sweep (`ε` 4…64,
window 3 × 3 vs 5 × 5).

### Deblocking

`DecodeOptions::with_deblock(true)` (registry decoder key `deblock`) is
off by default. It smooths each 8 × 8 block edge of every component
whose step is below the quantiser step of the first AC coefficient
along that axis, and only where both sides are flat (each within half
that step). The two samples on each side take a quarter / an eighth of
the step. Larger steps are image content and are left alone. Measured
on the photographic set plus the fractal (this crate's encoder, 4:2:0,
default chroma reconstruction):

| quality | off | **on** | gain on `cjpeg -progressive` input |
|---|---|---|---|
| 10 | 30.39 dB / 0.9120 | 30.93 / 0.9377 | +0.54 dB |
| 30 | 35.60 / 0.9522 | 36.26 / 0.9638 | +0.62 dB |
| 50 | 37.48 / 0.9636 | 38.04 / 0.9701 | +0.40 dB |
| 75 | 39.84 / 0.9744 | 40.15 / 0.9768 | +0.15 dB |

On the synthetic set the effect is within ±0.08 dB. On a noisy 1080p
photograph it changes PSNR by under 0.07 dB, while SSIM drops by up to
0.005. That is why it stays opt-in. Thresholds of 0.5 × and 2 × the
step measured up to 0.06 dB below 1 ×.

### Encoder quality

The still-image defaults (`EncodeOptions::new()`) are chosen by size at
equal PSNR. The measurement is a quality 20…95 sweep over the five
reference images of *Chroma reconstruction* (three photographs, a test
card and a fractal), 4:2:0, decoded with this crate's defaults. The
BD-rate (mean log-rate difference over the shared PSNR range, negative
= smaller) is taken against the previous defaults (typical tables,
§A.3.4 rounding, chroma rounded per sample before the box average):

| Change | BD-rate (PSNR) | BD-rate (SSIM) |
|---|---|---|
| chroma averaged exactly, rounded once | −1.0 % | −0.4 % |
| + K.2 optimal Huffman tables | −20.7 % | −20.6 % |
| + rate-distortion levels (**default**) | **−24.0 %** | **−26.6 %** |
| `cjpeg -optimize` (black-box sanity row) | −20.0 % | −21.8 % |

That puts the new defaults 5.1 % below `cjpeg -optimize` at equal PSNR
(4.7 % at equal SSIM). Mean bits per pixel / PSNR at three quality
settings:

| quality | previous | new default | `cjpeg -optimize` |
|---|---|---|---|
| 50 | 0.301 bpp / 36.43 dB | 0.214 / 36.12 | 0.236 / 36.45 |
| 75 | 0.422 / 38.43 | 0.323 / 38.18 | 0.362 / 38.46 |
| 90 | 0.647 / 40.38 | 0.553 / 40.24 | 0.604 / 40.42 |

Optimal tables cost nothing in quality and are smaller at every point,
so they are the default. They cost about 35 % more encode time, now that
one transform pass feeds both the statistics and the emission.
Rate-distortion levels keep each block's DC level and choose its AC
levels by a dynamic programme over the run-length coding. Each level
keeps its rounded value, moves one step towards zero, or becomes zero.
The objective is `squared error + λ · bits`, with `λ = 0.0085 · mean(Q_ac)²`
picked on the same sweep (0.005 to 0.15 tried). The rate model is the
Annex K table of the component's class. With it a given `quality` gives
a slightly lower PSNR in a smaller file. Uniform dead-zone rounding
(offsets 0.1 to 0.3) reached at most −4.0 %.

The chroma downsampling filter stays the T.871 §9 NOTE 1 two-tap box
(centred siting). Wider prefilters (a 4-tap triangle, 8-tap Catmull-Rom
and Lanczos-2) changed photographs by under ±0.03 dB. On the synthetic
set they lost 0.9 to 3.5 dB, because they smear the colour edges the
luma-guided decoder restores.

Encode time, 1080p / 4K RGB at quality 85 (noisy photograph, ≈ 2 bpp):
18 / 71 ms for typical tables and `Nearest` (previously 21.5 /
83 ms), 25 / 96 ms with optimal tables (32 / 124 ms before), and 43 /
165 ms for the new defaults.

### Reference-compatible output (bit-exact against `djpeg`)

An opt-in decode that reproduces `djpeg`'s default output (3.2.0)
sample for sample; the default decode is unchanged.
`DecodeOptions::with_reference_compat(true)` (equivalently the
`libjpeg_compatible()` preset) sets the three knobs to
`AccurateInteger` / `Triangular` / `FixedPoint16`:

```rust,ignore
let opts = oxideav_mjpeg::DecodeOptions::new().with_reference_compat(true);
let rgb = oxideav_mjpeg::decode_rgb8_with(&jpeg, &opts)?;      // == djpeg -ppm
let img = oxideav_mjpeg::decode_with(&jpeg, &opts)?;           // planes from the integer IDCT
let rgb = img.to_rgb8_with(&opts.convert_options());           // same bytes
```

Each knob is independent (`with_idct` / `with_upsampling` /
`with_ycbcr`; `ConvertOptions` carries the last two for
`JpegImage::to_rgb8_with`, with its own `with_reference_compat`); the
defaults are unchanged
(`decode_rgb8(b) == decode_rgb8_with(b, &DecodeOptions::new())`).
Through the registry the decoder takes the same switch in
`CodecParameters::options` (`MjpegDecoderOptions`):

| key | values | default |
|---|---|---|
| `reference_compat` | `true` / `false` | `false` |
| `idct` | `float` / `integer` (overrides `reference_compat`) | from `reference_compat` |
| `upsampling` | `replicate` / `triangular` / `linear` / `guided` (overrides `reference_compat`; used for the layouts the decoder resamples to 4:4:4) | from `reference_compat` |

Other keys in the bag (an encoder's output parameters carry the encoder
options) are ignored; a bad value is an error.

Fitted parameters (every one measured against `djpeg` as a black box;
the experiments and their mismatch tables are in
`notes/reference-compat.md`):

* **IDCT** — the T.81 §A.3.3 transform as two 1-D passes with 13-bit
  constants, 2 fraction bits kept between the passes (1 for 12-bit
  frames), round-half-up shifts, and a measured range limit (clamp for
  overshoot inside −896…1151). Only this `(13, 2)` choice matches all
  1 408 crafted impulse blocks and 2 048 random blocks.
* **YCbCr → RGB** — the T.871 §7 equations at `2^16`, each chroma
  contribution rounded half-up (the two G products summed first). The
  G coefficients are `0.344146` / `0.714142`, measured over every
  `(Cb, Cr)` pair.
* **Chroma upsampling** — ¾ / ¼ interpolation at T.871 §9 siting for
  2:1, 1:2 and 2 × 2 ratios, with edge replication. The rounding bias
  alternates by output phase: 1 / 2 (`/ 4`) on one axis, 8 / 7 (`/ 16`)
  for 2 × 2. Components at most two samples wide are replicated, and so
  are the other ratios.

Verified with 0 differing bytes on 2 621 streams: 1 092 `cjpeg` encodes
(4:4:4 / 4:2:2 / 4:2:0 / 4:4:0 / 4:1:1 / 4×2 / mixed factors, grey, RGB,
restart intervals, progressive, arithmetic, q10–q100, sizes 1×1–101×67),
1 300 of this crate's own encodes, 208 `ffmpeg` encodes, the 8-bit
`docs/` fixtures, and crafted coefficient-level streams. 87 of them are
committed with `djpeg`'s output hashes (`tests/fixtures/libjpeg/`). On
1080p 4:2:0 the compatible RGB decode costs about the same as the
default (22.8 vs 22.1 ms, Apple M5 Max, release build). The integer IDCT
alone is faster than the float one (15.3 vs 16.4 ms for the planes).

12-bit DCT frames get the same IDCT with one fraction bit between the
passes, bit-exact against `djpeg -grayscale`'s 16-bit luma; their RGB is
this crate's 8-bit rescale (`djpeg` writes 16-bit PNM), so it is not
compared.

Not covered:

* **Incomplete progressive streams.** If the five lowest AC
  coefficients are never fully coded (not sent, or left at `Al > 0`),
  `djpeg` alters the reconstruction across neighbouring blocks. This mode
  reconstructs the coded coefficients as they are, so such streams differ
  (for example 1 394–2 788 of 3 072 samples on a 64×48 grey image; see
  the notes).
* Lossless frames with layouts resampled to 4:4:4 (replicated).
* Streams whose dequantised coefficients exceed 16 bits
  (`|S · Q| > 32767`, beyond any 8-bit FDCT output).

## Coefficient level

`read_coefficients(bytes)` stops after the entropy decoder (#21):
`JpegCoefficients { width, height, precision, kind, quant_tables,
components, adobe_transform, jfif, metadata }` with, per component,
`CoefficientComponent { id, h, v, quant_id, blocks }`. `blocks` is the
§A.2.4 MCU-padded block grid in raster order (`block_grid(ci)`;
`block_extent(ci)` is the part over real samples), each block 64
quantised coefficients `Sq_vu` in **natural** order (`v × 8 + u`, `[0]`
= DC — not zig-zag); `quant_tables` use the same order. Multiply by the
table and run any inverse DCT to get samples, or call
`decode_coefficients` for this crate's. `encode_coefficients` codes a
record back (`EncodeOptions::process` sequential or progressive,
`tables` typical or optimal, `restart_interval`, `metadata`, `jfif`),
keeping ids, sampling, tables and the Adobe flag, so coefficient-domain
transforms work: a horizontal mirror (reverse the block columns, negate
the odd `u`) or a transpose (swap block rows / columns, `u` / `v` and
the tables) reproduces `jpegtran -flip horizontal` / `-transpose`
coefficient for coefficient (`tests/coefficients.rs`). Over the 115
fixture streams, rendering equals `decode` and every re-code reads back
to the same coefficients; `djpeg` decodes the re-codes of the 92 8-bit ones
byte-identically (seven re-codes each: Huffman sequential typical /
optimal / restart, Huffman progressive, arithmetic sequential / restart /
progressive). `EncodeOptions::entropy = Arithmetic` writes `SOF9` /
`SOF10`; re-coding a black-box transcoder's sequential arithmetic stream
reproduces its entropy-coded segment byte for byte. `marker_order` and
`conditioning` apply as in `encode`. Lossless and hierarchical frames
are `Error::Unsupported`.

## Metadata and colour

* `Metadata { icc, exif, xmp, gamma, comments }` — the three carried blobs,
  one set of bytes seen from both layers: `read_metadata(bytes)` ==
  `decode(bytes).metadata` == `Metadata::from_blobs(frame.blobs())`,
  and `Metadata::to_blobs()` is what the `jpeg` / `mjpeg-raw`
  demuxers put on `CodecParameters::blobs` and the registry decoder on
  the frame record. `gamma` is always `None`.

  | Blob | Segment | Identifier | Payload (the blob bytes) | Limit (typed `Unsupported`) |
  |---|---|---|---|---|
  | `icc` / `BlobKind::ICC` | `APP2`, one per chunk | `"ICC_PROFILE\0"`, chunk number from 1, chunk count | the profile, split in sequence order; dropped when a chunk is missing, repeated or disagrees on the count (an error in strict mode) | 65 519 bytes per chunk, 255 chunks, 16 707 345 bytes (ICC.1 Annex B.4) |
  | `exif` / `BlobKind::EXIF` | `APP1` | `"Exif\0\0"` | the TIFF structure from its byte-order mark; the first segment wins | 65 527 bytes, one segment (Exif §4.7.2) |
  | `xmp` / `BlobKind::XMP` (`mime` `application/rdf+xml`) | `APP1` | `"http://ns.adobe.com/xap/1.0/\0"` | the standard packet (UTF-8 RDF/XML); the first wins; extended-XMP continuation segments are neither written nor reassembled | 65 502 bytes |

  `comments` (#21) is the payload of every `COM` segment (T.81
  §B.2.4.5) before the first scan, verbatim and in order —
  `read_metadata` / `decode(bytes).metadata` report them, `info` counts
  them (`ImageInfo::comments`), and `Metadata::with_comment("…")` /
  `with_comments(…)` on `EncodeOptions::metadata` writes them (one `COM`
  each, at most 65 533 bytes, after the `APPn` segments and ahead of the
  tables; a decoded image re-encodes with its own comments). They have
  no framework blob form: a Motion-JPEG writer stamps a `COM` into
  every frame, and carrying it would attach a side-channel record to
  each decoded frame.

  JFIF `APP0` and Adobe `APP14` are not blobs — they signal the colour
  model and the decoder interprets them. Blobs of other kinds (IPTC,
  cover art, custom) have no JPEG carriage here and are ignored on
  write.
* **Marker order** (`oxideav_mjpeg::metadata`, the one rule every
  writer finishes through — general T.81 writer, historical per-layout
  encoders, arithmetic knob, abbreviated frames): `SOI`, then **`APP1
  Exif` when present** (Exif §4.5.4 / §4.7.2.1: immediately after
  `SOI`), **else `APP0 JFIF`** (T.871 §6.3: the same claim), then
  `APP1 XMP`, the `APP2 ICC` chunks, then every other segment (Adobe
  `APP14`, `COM`, foreign `APPn`, tables, frame header) in its original
  order. The two carriage specs cannot both be satisfied, so
  `JfifPolicy` decides: `Auto` (default) drops the JFIF segment when
  Exif takes the slot — the file is an Exif file; `Always` keeps JFIF
  right after the Exif `APP1` (Exif §4.5.5 Table 2 row 6 placement,
  not T.871's); `Never` writes none. Every policy decodes to the same
  picture here (the colour-model segments are read wherever they sit
  before `SOS`); the policy only changes which outside reader
  recognises the file by its first segment. `embed_metadata(jpeg,
  &meta, policy)` applies the rule to any stream (same-kind segments
  replaced wherever they sat, the picture untouched; a non-empty
  `comments` list replaces the `COM` segments), `strip_metadata`
  removes the three kinds and the `COM` segments.
* **Precedence** (`Metadata::with_overrides`): per kind, a frame's
  blobs over the stream's `params.blobs` over the `EncodeOptions`
  metadata of a direct caller; on the contract side the option's
  `metadata` over `JpegImage::metadata`. Motion-JPEG frames may each
  carry their own; a still's frame blobs equal its stream blobs.
* `ColorInfo { range, primaries, transfer, matrix }` — range plus H.273
  code points. YCbCr-coded frames report `jfif_ycbcr()` = full range,
  matrix 5 (BT.601 / sYCC, `KR = 0.299`, `KB = 0.114`), primaries 1 and
  transfer 13 (sRGB): T.871 §7 derives its Y/CB/CR from BT.601-625 and
  its NOTE 3 records that practice follows sYCC with negligible
  difference, so consumers treat decoded JPEG RGB as sRGB. RGB-coded
  frames report `srgb()` (matrix 0), grayscale `gray()`, CMYK `cmyk()`
  (code points unspecified — T.872 §3.1 leaves ink values device
  dependent). The range is full for every YCbCr frame, JFIF or not
  (T.872 §6.1 extends the T.871 relationship to every three-component
  JPEG without an Adobe RGB flag). An embedded ICC profile takes
  precedence for colour-managed consumers.
* `to_rgb8` / `to_rgba8` — T.871 §7 inverse equations in exact integer
  arithmetic (`Round(x) = ⌊x + 0.5⌋`, one rounding, clamped; the
  interpolated chroma enters with its fraction to 1/256 of a sample);
  chroma **luma-guided reconstruction** at the T.871 §9 centred siting
  (*Chroma reconstruction* below; T.871 leaves the filter to the
  decoder, and the result is deterministic — a stream always yields the
  same bytes); grayscale replicated; CMYK by `R = Round((255
  − C)(255 − K) / 255)` (a documented convention, not a spec formula);
  deep samples rescaled by `Round(v · 255 / (2^P − 1))` with the
  frame's `precision`; `Gbrp*` re-ordered to R, G, B. Grayscale, RGB
  and 12-bit corpus fixtures reproduce the reference PPMs bit-exactly.
  `to_rgb8_with` / `to_rgba8_with` take a `ConvertOptions` (triangular
  upsampling, 16-bit fixed-point YCbCr — see *Reference-compatible
  output*).

## Limits

* Dimensions `1..=65535` (T.81 `X`, `Y`; `Y = 0` resolved through
  `DNL`). Default `DecodeOptions` accept up to 268 Mpixel; the decoder
  additionally caps `X × Y × Nf` at 64 M samples per frame.
* `encode` rejects (never converts): plane count / size mismatches,
  `sampling` disagreeing with the image layout, `Gbrp*` / `Rgb48Le` /
  `Gray10Le` / `Gray16Le` under a DCT process, an Exif block over
  65 527 bytes, an XMP packet over 65 502 bytes, an ICC profile over
  16 707 345 bytes (255 chunks).
* Hostile input returns `Error`, never panics — fourteen fuzz targets
  (`fuzz/`), `standalone_api` and `metadata_embed` among them, run
  daily.
* Not decoded: bare differential frames (SOF5–7 / SOF13–15) outside a
  DHP sequence; 12-bit 4-component frames (no 12-bit CMYK layout).

## JPEG specifics

### General T.81 writer (`oxideav_mjpeg::t81`)

The one JPEG encoder in the workspace. **Consumers (`oxideav-tiff`'s
`Compression = 7` writer, and anything else that needs a T.81
datastream) re-point here** instead of carrying their own writer;
every item is re-exported at the crate root and the surface is
additive.

Typed options, one call:

```rust
use oxideav_mjpeg::t81::{EncodeOptions, HuffmanTables, JpegProcess};

// 12-bit extended-sequential (SOF1) 4:2:0, K.2 optimal tables, a
// restart every 4 MCUs, emitted as the §B.5 abbreviated pair.
let (w, h) = (640u32, 480u32);
# let y: Vec<u16> = vec![2048; (w * h) as usize];
# let c: Vec<u16> = vec![2048; ((w / 2) * (h / 2)) as usize];
let opts = EncodeOptions::new()
    .with_precision(12)
    .with_process(JpegProcess::Sequential)
    .with_tables(HuffmanTables::Optimal)
    .with_sampling(vec![(2, 2), (1, 1), (1, 1)])
    .with_restart_interval(4)
    .with_abbreviated(true);
let out = opts.encode(w, h, &[&y, &c, &c])?;
let strip: Vec<u8> = out.data;              // SOI, SOF1, DRI, SOS…, EOI — no tables
let jpegtables: Vec<u8> = out.tables.unwrap(); // SOI, DQT/DHT, EOI
# Ok::<(), oxideav_mjpeg::MjpegError>(())
```

An interchange stream (`abbreviated = false`) writes its tables around the
frame header: every `DQT` ahead of `SOFn`, the `DHT` segments after it
(in front of the first scan that uses each destination), then `DRI` and
the first `SOS` — `SOI APPn… DQT… SOFn DHT… [DRI] SOS …`. Only the
destinations the frame references are written.

`EncodeOptions { quality, tables, process, precision, restart_interval,
abbreviated, signalling, sampling, table_ids, chroma, metadata }`
(`#[non_exhaustive]`, `Default` + `with_*` builders; the `chroma` and
`metadata` fields are described under *Options* above):

- `process` — `Sequential` (`SOF0` when `P = 8`, destinations ≤ 1 and
  8-bit quantisers; `SOF1` otherwise, incl. `P = 12`), `Progressive`
  (`SOF2`, one interleaved DC scan + per-component AC bands `1..=5` /
  `6..=63`, `P = 8` / `12`), `Lossless { predictor, point_transform }`
  (`SOF3`, `P ∈ 2..=16`, Table H.1 predictors 1..=7).
- `sampling` — per-component `(Hi, Vi)`; any T.81 §A.1.1 combination
  under the §B.2.3 `Σ Hi × Vi ≤ 10` bound (4×2 luma, mixed chroma,
  chroma oversampled relative to luma, …). Input planes are supplied
  at the A.1.1 resolution `ceil(X × Hi / Hmax) × ceil(Y × Vi / Vmax)`.
- `table_ids` — per-component `(Tq, Td = Ta)` destinations (0..=3).
- `tables` — Annex K.3 typical, or Annex K.2 optimal from the frame's
  own symbol statistics (Figures K.1–K.4, 16-bit length limit, the
  all-ones code word reserved per §C.2). 12-bit DCT frames always use
  optimal tables (the typical alphabet stops at `SSSS = 11`).
- `restart_interval` — `DRI` + `RSTm` on every process (lossless: a
  multiple of the MCUs per MCU-row, Table B.7).
- `abbreviated` — table-less frame streams plus a separate tables-only
  stream (§B.5), the TIFF Technical Note 2 `JPEGTables` carriage;
  `tables_for_planes` + `encode_with_tables` code a whole strip / tile
  sequence against one shared table set.
- `signalling` — `Auto` (JFIF for 1 / 3 components, Adobe CMYK —
  APP14 `transform = 0` — for 4),
  `Jfif`, `Rgb` (Adobe APP14 `transform = 0` + `'R'/'G'/'B'` ids),
  `Cmyk { adobe_transform }` (`None` / `Some(0)` inverted / `Some(2)`
  YCCK), `None`.

Frame-level API (what a TIFF writer calls per strip):

```rust
use oxideav_mjpeg::t81::{
    encode_frame, gather_stats, HuffStats, JpegComponent, JpegFrame, JpegProcess, JpegTableSet,
};

# let (w, h) = (16u32, 8u32);
# let a: Vec<u16> = vec![100; (w * h) as usize];
# let b: Vec<u16> = vec![200; (w * h) as usize];
let frame = JpegFrame { width: w as u16, height: h as u16, precision: 8,
                        process: JpegProcess::Sequential, restart_interval: 0 };
fn comp(s: &[u16], w: u32, h: u32) -> JpegComponent<'_> {
    JpegComponent { id: 1, samples: s, width: w as usize, height: h as usize,
                    h: 1, v: 1, quant_id: 0, huff_id: 0 }
}
// One table set shared by every strip: typical quantisers, optimal
// Huffman tables from the statistics of all strips.
let mut tables = JpegTableSet::typical(90, 8, false, 1);
let mut dc: [HuffStats; 4] = Default::default();
let mut ac: [HuffStats; 4] = Default::default();
gather_stats(&frame, &[comp(&a, w, h)], &tables, &mut dc, &mut ac)?;
gather_stats(&frame, &[comp(&b, w, h)], &tables, &mut dc, &mut ac)?;
tables.dc[0] = Some(dc[0].to_spec());
tables.ac[0] = Some(ac[0].to_spec());
let jpegtables = tables.tables_stream(true);           // JPEGTables field
let strip_a = encode_frame(&frame, &[comp(&a, w, h)], &tables, false)?; // abbreviated
let strip_b = encode_frame(&frame, &[comp(&b, w, h)], &tables, false)?;
# let _ = (jpegtables, strip_a, strip_b);
# Ok::<(), oxideav_mjpeg::MjpegError>(())
```

The decode side of the pair is `decode_with(&strip,
&DecodeOptions::new().with_tables(jpegtables))` — or, through the
registry, `CodecParameters::extradata = jpegtables`.

Every stream the writer emits is checked against this crate's own
decoder (bit-exact for lossless, PSNR for the DCT processes) and
black-box against `djpeg` (lossless byte-exact at 8 / 12 / 16 bits);
see `tests/t81_encoder.rs`.

### Encoder

```rust
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, RuntimeContext};
# use oxideav_core::{VideoFrame, VideoPlane};
# let (w, h) = (64u32, 48u32);
# let plane = |pw: u32, ph: u32| VideoPlane { stride: pw as usize, data: vec![128; (pw * ph) as usize] };
# let frame_yuv420 = VideoFrame { pts: Some(0), planes: vec![plane(w, h), plane(w / 2, h / 2), plane(w / 2, h / 2)] };

let mut ctx = RuntimeContext::new();
oxideav_mjpeg::register(&mut ctx);

let mut params = CodecParameters::video(CodecId::new("mjpeg"));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Yuv420P);
let mut enc = ctx.codecs.first_encoder(&params)?;
enc.send_frame(&Frame::Video(frame_yuv420))?;
let pkt = enc.receive_packet()?;
# let _ = pkt;
# Ok::<(), Box<dyn std::error::Error>>(())
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

Any key in `CodecParameters::options` routes `make_encoder` through the
general T.81 writer with the `MjpegEncoderOptions` schema (`quality`,
`tables = typical | optimal`, `process = sequential | progressive |
lossless`, `precision`, `restart`, `abbreviated`, `predictor`,
`point_transform`, `sampling = "HxV,…"`; unknown keys are rejected),
and `MjpegEncoder::set_encode_options(EncodeOptions)` does the same
from typed code — precision and sampling are taken from (and checked
against) the pixel format, colour signalling from the pixel format and
`set_adobe_transform`. With `abbreviated` the shared tables stream is
published in `output_params().extradata`.

Restart markers (`RSTn` + DRI) are supported for interop and bitstream
resiliency. They are **off by default** — call
`MjpegEncoder::set_restart_interval(n_mcus)` (or use
`encoder::encode_jpeg_with_opts(frame, quality, n_mcus)`) to enable
them. A non-zero value writes a DRI segment before SOS and cycles
`RST0..=RST7` every `n_mcus` macroblocks in the scan, resetting DC
predictors at each marker. Passing `0` preserves the historical
no-restart behaviour.

### Registry encoder contract

The framework encoder (`make_encoder` / `MjpegEncoder::from_params`)
declares the input layouts it accepts through
`CodecCapabilities::accepted_pixel_formats`
(`registry::ACCEPTED_PIXEL_FORMATS`), so a framework caller — the
`oxideav-image` gateway's pixel-format ladder — converts to one of them
before encoding instead of guessing. Every declared layout is coded
faithfully (the stream carries the samples as given) and the encoder's
`output_params()` carry the layout the decoder reports for the stream
(`pixel_format`, `coded_pixel_format(input)` for the knob-free case)
plus its colour signal, exactly as the `jpeg` demuxer declares them.
`tests/registry_matrix.rs` proves every cell (layout × 1×1 / odd /
MCU-unaligned / aligned sizes × option sets: 385 cells, 355 of them
also through the black-box reference decoder with the declared
geometry, component count and precision; plus every layout as a §B.5
abbreviated pair decoded through `extradata` tables, and the 16-bit
carriers at every lossless precision).

| Input | Stream (empty option bag) | Decodes as |
|---|---|---|
| `YuvJ444P` / `YuvJ422P` / `YuvJ420P` | JFIF YCbCr, luma `1×1` / `2×1` / `2×2` | the same label |
| `Yuv411P` | JFIF YCbCr, luma `4×1` | `Yuv411P` |
| `Yuv440P` | JFIF YCbCr, luma `1×2` | `YuvJ444P` (the decoder resamples 4:4:0 chroma onto the frame grid) |
| `Gray8` | JFIF, one component, `SOF0` | `Gray8` |
| `Gray12Le` / `Yuv444P12Le` / `Yuv422P12Le` / `Yuv420P12Le` | 12-bit extended sequential (`SOF1`) | the same label |
| `Gray10Le` / `Gray16Le` | lossless (`SOF3`) at `P = 10` / `16` — the only process that carries them | the same label |
| `Rgb24` | RGB-coded: T.872 Adobe APP14 `transform = 0` + ids `'R' 'G' 'B'`, no colour conversion | `Rgb24` |
| `Rgb48Le` | lossless at `P = 16`, RGB-coded | `Rgb48Le` |
| `Gbrp10Le` / `Gbrp12Le` / `Gbrp14Le` | lossless, planes in order, ids `1 2 3`, no colour signalling | the same label |
| `Cmyk` | four components, Adobe APP14 `transform = 0` (T.872 §6.5.3, samples complemented; `adobe_transform = none / ycck` or `set_adobe_transform` for marker-less plain CMYK / YCCK) | `Cmyk` |
| `CmykInverted` | complemented into plain CMYK | `Cmyk` |

Rulings:

- **Aliases, not declared.** The range-agnostic `Yuv444P` / `Yuv422P` /
  `Yuv420P` labels are accepted as the same layouts (JPEG YCbCr is full
  range by definition, T.871 §7; the encoder scales no ranges, so a
  limited-range picture must be expanded first) and labelled `YuvJ*` on
  output, but they are not declared: a caller converting through the
  declaration reaches the explicit full-range label.
- **`Rgb24` is RGB-coded.** The spec-grounded three-component RGB form
  is T.872's Adobe APP14 `transform = 0`; the encoder also writes the
  `'R' 'G' 'B'` component identifiers and no JFIF segment (T.871 binds
  a three-component JFIF stream to YCbCr). The picture survives exactly
  up to quantisation. A JFIF YCbCr stream from RGB input is one option
  away: `chroma = yuv444 / yuv422 / yuv420 / yuv411 / gray` converts
  with the T.871 §7 forward equations exactly as `encode_rgb8` does
  (byte-identical), and `chroma = gray` keeps the luma of a YCbCr
  input.
- **Refusals are typed and name the layout.** An undeclared pixel
  format is `Error::Unsupported` naming it (alpha layouts say so: JPEG
  has no alpha — drop it first); a frame with too few planes is
  `InvalidData` naming the layout and the count. Combinations the
  decoder would relabel or refuse are refused up front: a DCT process on
  a lossless-only carrier; the lossless process on 4:4:4 YCbCr (T.81
  Annex H carries no colour transform, so a lossless `1×1`
  three-component stream is the RGB-class carrier on decode — code
  `Rgb24` / `Gbrp12Le` for an exact picture, or a subsampled layout) and
  on 12-bit subsampled YCbCr (the decoder shapes subsampled lossless
  scans at 8 bits); `precision` must match the layout except on the
  16-bit carriers (`Gray16Le` / `Rgb48Le` take any lossless `2..=16`,
  the samples in the low bits — but not `10 / 12 / 14` on `Rgb48Le`,
  whose streams the decoder labels as the planar `Gbrp*Le` carriers:
  hand those planes over as `Gbrp*Le`).
- **One writer for every option.** Any option key routes through the
  general T.81 writer and the bytes are identical to
  `encode(&JpegImage::from_video_frame(..), &EncodeOptions)` for the
  same request (49 requests pinned). The empty bag keeps the historical
  per-format streams for `Gray8`, 8-bit YCbCr 4:4:4 / 4:2:2 / 4:2:0,
  `Rgb24` and `Cmyk` (same entropy-coded data;
  `tests/golden.rs` pins them) and the general writer with its defaults
  everywhere else, so every declared layout encodes without options.
  `process = auto` (the default) is sequential DCT, or lossless on the
  lossless-only carriers.

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
# use oxideav_core::{CodecId, CodecParameters, Encoder, Frame, PixelFormat, VideoFrame, VideoPlane};
# use oxideav_mjpeg::encoder::MjpegEncoder;
# let (w, h) = (64u32, 48u32);
# let frame_gray8 = VideoFrame { pts: Some(0), planes: vec![VideoPlane { stride: w as usize, data: vec![128; (w * h) as usize] }] };
let mut params = CodecParameters::video(CodecId::new("mjpeg"));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Gray8);
let mut enc = MjpegEncoder::from_params(&params)?;
enc.set_progressive(true);
enc.send_frame(&Frame::Video(frame_gray8))?;
# Ok::<(), Box<dyn std::error::Error>>(())
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
# use oxideav_core::{CodecId, CodecParameters, Encoder, Frame, PixelFormat, VideoFrame, VideoPlane};
# use oxideav_mjpeg::encoder::MjpegEncoder;
# let (w, h) = (64u32, 48u32);
# let frame = Frame::Video(VideoFrame { pts: Some(0), planes: vec![VideoPlane { stride: w as usize * 2, data: vec![0; (w * h * 2) as usize] }] });
let mut params = CodecParameters::video(CodecId::new("mjpeg"));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Gray12Le);
let mut enc = MjpegEncoder::from_params(&params)?;
enc.set_lossless(true);
enc.set_lossless_predictor(4);
enc.send_frame(&frame)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Without `set_lossless(true)` the trait-API encoder codes `Gray8` and
`Gray12Le` with the sequential DCT (8- and 12-bit); `Gray10Le` /
`Gray16Le` are lossless whatever the flag says — nothing else carries
them — and asking for a DCT process on them (`process = sequential`)
is a typed `Unsupported` refusal rather than a silent downgrade.

### Lossless arithmetic (SOF11) grayscale encode

For single-component grayscale input the lossless **arithmetic-coded**
counterpart of the SOF3 path is exposed directly:

```rust
use oxideav_mjpeg::encoder::encode_lossless_arith_jpeg_grayscale;

# let (width, height, stride) = (16u32, 16u32, 16usize);
# let samples = vec![0u8; stride * height as usize];
// precision ∈ 2..=16, predictor ∈ 1..=7 (Annex H Table H.1).
let jpeg = encode_lossless_arith_jpeg_grayscale(width, height, &samples, stride, 8, 1)?;
# Ok::<(), oxideav_mjpeg::MjpegError>(())
```

The spatial model is identical to the Huffman lossless path (Annex H
predictors `1..=7` over `Ra` / `Rb` / `Rc`), but each prediction
difference is coded with the Q-coder arithmetic statistical model of
T.81 §H.1.2.3 (Table H.3) — the `L_Context(Da, Db)` / `X1_Context(Db)`
conditioning over neighbouring differences — instead of a Huffman
magnitude category. The bitstream is `SOI / JFIF APP0 / SOF11 (Nf = 1,
H = V = 1) / [DRI] / SOS (Ss = predictor, Al = Pt) / arith scan / EOI`;
no DAC segment is emitted, so the SOF11 decoder applies the default
conditioning bounds `(L, U) = (0, 1)` per §H.1.2.3.3. Output is
bit-exact for every precision `P ∈ 2..=16`, every predictor, and the
half-modulus `Di = 32768` corner case (§H.1.2.2). The
`encode_lossless_arith_jpeg_grayscale_with_opts(..., restart_interval,
point_transform)` variant adds DRI + `RSTn` emission (each interval
flushes the Q-coder, byte-aligns, writes `RST0..=RST7` cycling modulo 8,
and re-seeds the statistical model + difference history + predictor to
the scan-origin default `2^(P − Pt − 1)`, §H.1.1 / §H.1.2.3.4) and a
non-zero point transform `Pt` (the low `Pt` bits are discarded on both
sides). The SOF11 decoder consumes these encode outputs end-to-end
(`tests/lossless_roundtrip.rs`).

The three-component (RGB-class) counterpart
`encode_lossless_arith_jpeg_rgb(width, height, [c0, c1, c2], strides,
precision, predictor)` (plus its `_with_opts(..., restart_interval,
point_transform)` companion) emits a `SOF11 (Nf = 3, every component
H = V = 1)` interleaved scan — the Q-coder counterpart of
`encode_lossless_jpeg_rgb`. Each component is modelled independently per
§H.1.2 (its own statistics area + `L_Context(Da, Db)` /
`X1_Context(Db)` difference history), and one residual per component is
emitted per pixel position in scan order into a single arithmetic-coded
segment. Bit-exact for every precision `P ∈ 2..=16` (decode shape:
`P = 8` → packed `Rgb24`, `P ∈ {10, 12, 14}` → planar `Gbrp*Le`, every
other `P` → packed `Rgb48Le`), every predictor, the half-modulus
`Di = 32768` case, non-zero `Pt`, and per-interval restart re-seeding of
all three components.

The four-component (CMYK-class) counterpart
`encode_lossless_arith_jpeg_cmyk(width, height, [c0, c1, c2, c3], strides,
predictor, adobe_transform)` (plus its `_with_opts(..., restart_interval,
point_transform)` companion) emits a `SOF11 (Nf = 4, every component
H = V = 1)` interleaved scan at `P = 8` — the Q-coder counterpart of
`encode_lossless_jpeg_cmyk`. Each component is modelled independently per
§H.1.2 (its own statistics area + `L_Context(Da, Db)` / `X1_Context(Db)`
difference history), and one residual per component is emitted per pixel
position in scan order into a single arithmetic-coded segment. The
`adobe_transform` flag matches the Huffman CMYK helpers: `None` writes no
APP14 (plain "regular" CMYK), `Some(0)` selects Adobe CMYK and inverts
every sample on the wire before predictive coding, `Some(2)` selects Adobe
YCCK (interpret the packed input as `[Y, Cb, Cr, K]` and invert only K
before coding). Decode shape is the same packed `PixelFormat::Cmyk` the
SOF3 four-component path produces (4 bytes/pixel). Bit-exact for every
predictor, the half-modulus `Di = 32768` case, non-zero `Pt`, and
per-interval restart re-seeding of all four components on the no-APP14 /
Adobe-CMYK paths (YCCK is a lossy interop convention by construction —
BT.601 YCbCr → RGB → CMY clamps; the K plane round-trips exactly).

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

### Hierarchical spatial-lossless (Annex J) encode

The decoder's hierarchical coverage is matched by an encode path for the
spatial-lossless progression (T.81 §K.7.2.2):

```rust
use oxideav_mjpeg::encoder::encode_hierarchical_lossless_jpeg_grayscale;

# let (width, height, stride) = (16u32, 16u32, 16usize);
# let samples = vec![0u8; stride * height as usize];
// precision ∈ 2..=16, predictor ∈ 1..=7, levels >= 1 pyramid stages.
let jpeg = encode_hierarchical_lossless_jpeg_grayscale(
    width, height, &samples, stride, 8, 1, /* levels = */ 3,
)?;
# Ok::<(), oxideav_mjpeg::MjpegError>(())
```

The output is a `DHP`-introduced resolution pyramid (§B.3.2): the lowest
stage is a non-differential lossless `SOF3` frame (Annex H predictors
`1..=7`), and each higher stage is an `EXP` segment (`Eh = Ev = 1`,
§B.3.3) followed by a differential lossless `SOF7` frame whose samples
are the modulo-2^P difference between the stage's source and the ×2
bi-linearly upsampled previous stage (§J.1.1.2 / §J.2.1), coded directly
with `Ss = 0` per §J.2.3.2. Because the upsampling filter mirrors the
decoder's normative one exactly, reconstruction is **bit-exact** at every
stage. The pyramid downsampler is the truncating 2×2 mean (T.81 leaves
the downsampling filter to the encoder). Full-resolution dimensions must
be divisible by `2^(levels − 1)`; `levels = 1` emits a single
non-differential frame inside the DHP envelope. Companions:
`encode_hierarchical_lossless_jpeg_rgb(width, height, [c0, c1, c2],
strides, precision, predictor, levels)` (three-component interleaved,
`P ∈ 2..=16`, same precision-shaped decode output as the flat RGB path)
and `encode_hierarchical_lossless_jpeg_cmyk(width, height,
[c0, c1, c2, c3], strides, predictor, adobe_transform, levels)`
(four-component at `P = 8`, Adobe APP14 conventions identical to
`encode_lossless_jpeg_cmyk`). Point transform is fixed at `Pt = 0` and no
restart markers are emitted, matching the decoder's hierarchical slice.

The **arithmetic-coded** progression (`SOF11` non-differential +
`SOF15` differential — the Q-coder counterpart under the T.81
§H.1.2.3 two-dimensional statistical model) is emitted by the matching
`encode_hierarchical_lossless_arith_jpeg_grayscale` /
`_arith_jpeg_rgb` / `_arith_jpeg_cmyk` entry points: same pyramid
shape, one arithmetic-coded entropy segment per stage with independent
per-component statistics areas and `L_Context(Da, Db)` /
`X1_Context(Db)` difference-history conditioning, no DAC segment
(default `(L, U) = (0, 1)` bounds per §H.1.2.3.3), and the same
bit-exact reconstruction guarantee.

### Hierarchical DCT-progression (Annex J) encode

The DCT progression of §K.7.2.1 is emitted by
`encode_hierarchical_dct_jpeg_grayscale(width, height, samples, stride,
quality, levels)` and `encode_hierarchical_dct_jpeg_yuv444(width,
height, [y, cb, cr], strides, quality, levels)` (`P = 8`, every
component `H = V = 1`, component ids 1/2/3 so three-component output
shapes as planar `Yuv444P`). The lowest stage is a non-differential
baseline `SOF0` frame (Annex K luma quantiser scaled by `quality`, one
shared DC/AC table pair); each refinement stage is `EXP` + a
differential sequential `SOF5` frame coding the quantised forward DCT
of the signed stage residual under the §J.2.3.1 model — no level
shift, DC coded directly. The encoder mirrors the decoder's
reconstruction between stages (same dequantise / `idct8x8` / rounding /
modulo-2^16 fold), so every refinement corrects the accumulated
quantisation error below it: a single-stage progression tracks the
flat baseline encoder at equal quality, and each added stage carries
real correction energy. The progression is inherently lossy — for
bit-exact hierarchical output use the spatial-lossless entry points
above, or the **lossless-terminated** variants
`encode_hierarchical_dct_jpeg_grayscale_lossless_final` /
`encode_hierarchical_dct_jpeg_yuv444_lossless_final` (§K.7.2): they
append one differential lossless `SOF7` frame after the last DCT stage
(same resolution, no EXP) coding the exact residual per §J.2.3.2, so
the whole DCT pyramid reconstructs bit-exact at any quality while
keeping the progressive-display stages.

The **arithmetic-coded** DCT progression (`SOF9` non-differential +
`SOF13` differential, §K.7.2.1) is emitted by
`encode_hierarchical_dct_arith_jpeg_grayscale` /
`_arith_jpeg_yuv444(…, quality, levels, lossless_final)`: one Q-coder
segment per stage under the default conditioning (no DHT / DAC), the
§J.2.3.1 direct-DC differential model (per-block DC prediction zeroed;
§J.2.4 keeps the statistics conditioning unchanged), and pixel-identical
reconstruction to the Huffman progression at equal quality. Passing
`lossless_final = true` terminates with a differential lossless `SOF15`
frame coding the exact residual — bit-exact output on a single entropy
coder.

The **progressive** stage variant
(`encode_hierarchical_dct_progressive_jpeg_grayscale` /
`_progressive_jpeg_yuv444(…, quality, levels, lossless_final)`) emits
`SOF2` + differential `SOF6` frames instead: each stage is one
interleaved DC-first scan plus one full-band AC scan per component
(progressive AC scans are single-component, §B.2.3), with the
differential frame's DC scan coded directly per §J.2.3.1 — decoding
pixel-identical to the sequential progression at equal quality, with
the same optional `SOF7` lossless terminator. The arithmetic
progressive variant
(`encode_hierarchical_dct_progressive_arith_jpeg_grayscale` /
`_yuv444`) emits `SOF10` + differential `SOF14` stages (fresh Q-coder
segment + statistics per scan, §G.1.3) with the optional `SOF15`
terminator — completing the encode side of **every SOFn family the
hierarchical decoder consumes**.

### 4-component CMYK / YCCK encode

The 4-component (CMYK / Adobe YCCK) decode paths are matched by a
public encoder API. Both a baseline (SOF0) and a progressive (SOF2)
variant accept the same packed
`[C, M, Y, K]` interleaved buffer the decoder produces (4 bytes per
pixel, `stride` bytes per row), so round-tripping a decoded CMYK
frame back into a JPEG is a single call:

```rust
use oxideav_mjpeg::encoder::{encode_jpeg_cmyk, encode_jpeg_cmyk_progressive};

# let (width, height) = (16u32, 16u32);
# let packed = vec![0u8; (width * height * 4) as usize];
let jpeg = encode_jpeg_cmyk(width, height, &packed, width as usize * 4, 90, None)?;
let prog = encode_jpeg_cmyk_progressive(width, height, &packed, width as usize * 4, 90, None)?;
# Ok::<(), oxideav_mjpeg::MjpegError>(())
```

`adobe_transform` selects the Adobe APP14 colour-transform marker:
`None` writes no APP14 (plain "regular" CMYK), `Some(0)` selects
Adobe CMYK and inverts every sample on the wire, `Some(2)` selects
Adobe YCCK, interpreting the packed input as `[Y, Cb, Cr, K]` and
inverting only the K plane (the decoder un-does both transforms on
output). The typed writer (`EncodeOptions`, `ColorSignalling::Auto`) and
the registry encoder default to `Some(0)` (#21): T.872 §6.1 reads a
marker-less four-component stream as plain CMYK, but `djpeg` and
ImageMagick (black-box) read it as complemented, so only the Adobe form
renders the same everywhere. YCCK decodes per T.872 §7: YCbCr → RGB
gives C / M / Y, K is complemented. The two per-plane back-end entry points
`encoder::encode_jpeg_cmyk_1111` / `encode_jpeg_progressive_cmyk_1111`
are also `pub` for callers that already hold four separate component
buffers.

The trait-API encoder accepts CMYK input as well:

```rust
# use oxideav_core::{CodecId, CodecParameters, Encoder, Frame, PixelFormat, VideoFrame, VideoPlane};
# use oxideav_mjpeg::encoder::MjpegEncoder;
# let (w, h) = (64u32, 48u32);
# let frame = Frame::Video(VideoFrame { pts: Some(0), planes: vec![VideoPlane { stride: w as usize * 4, data: vec![0; (w * h * 4) as usize] }] });
let mut params = CodecParameters::video(CodecId::new("mjpeg"));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Cmyk);
let mut enc = MjpegEncoder::from_params(&params)?;
enc.set_adobe_transform(Some(2))?; // None / Some(0) / Some(2)
enc.set_progressive(true);         // optional — SOF2 instead of SOF0
enc.send_frame(&frame)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The plane stride must be at least `width * 4`; shorter strides are
rejected with `Error::InvalidData`.

### Metadata pass-through

The typed way is `Metadata` on `JpegImage` / `EncodeOptions` /
`CodecParameters::blobs`, or `embed_metadata` on a finished stream
(see *Metadata and colour*). The historical raw path remains: the
encoder entry points have `*_with_meta` variants that accept a `meta:
&[u8]` byte slice of pre-serialised APP/COM segments to embed
immediately after SOI (replacing the default JFIF APP0), and
`encoder::extract_app_segments(jpeg)` harvests APP0-APP15 and COM
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

- Codec: `"mjpeg"`. Decoder output: the native layouts of the
  *Supported layouts* table (`YuvJ444P` / `YuvJ422P` / `YuvJ420P` /
  `Yuv411P`, `Rgb24`, `Gray8`, `Cmyk`, the 12-bit `*12Le` and the
  lossless `Gray10Le` / `Gray16Le` / `Gbrp*Le` / `Rgb48Le` carriers);
  encoder input: the same set. The registry entry declares the
  `MjpegEncoderOptions` schema (`encoder_options_schema("mjpeg")`:
  `quality`, `tables`, `process`, `precision`, `restart`,
  `abbreviated`, `predictor`, `point_transform`, `sampling`).
- Container: `"jpeg"`, matches `.jpg` / `.jpeg` / `.jpe` / `.jfif` by
  extension and by `FF D8 FF` magic bytes. One frame per file; muxing
  is a pass-through of the codec packet. The stream's `pixel_format`
  and `color_signal` are exactly what the decoder emits for the file
  (the header rules of `info` / `decode`), for every process and
  colour case; a layout the decoder refuses leaves `pixel_format`
  unset.
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
units-= 0 case). When a JFIF *extension* APP0 segment (identifier
`"JFXX\0"`, T.871 §10.2) follows the JFIF APP0 — the segment most
writers use to carry the thumbnail — an optional `JfxxApp0` typed
view on `JpegInfo::jfxx` reports the thumbnail-storage variant via a
`JfxxThumbnail` enum exhaustive over the three defined
`extension_code` bytes: `JpegEncoded { jpeg_len }` (`0x10`, §10.3 —
embedded baseline JPEG), `PaletteRgb { width, height }` (`0x11`,
§10.4 — 768-byte palette + indices), and `Rgb24 { width, height }`
(`0x13`, §10.5 — packed 24-bit RGB). It carries no colour-convention
signal, so it leaves `color_hint` untouched. When an APP14 Adobe segment is present and
structurally valid per T.872 §6.5.3, an optional `AdobeApp14`
typed view is also exposed (`dct_encode_version`, `flags_0`,
`flags_1`, `transform: AdobeColorTransform` ∈ {`Unknown`,
`YCbCr`, `Ycck`}, plus `is_standard_version()` and
`as_color_hint()` projections). When one or more APP2 segments
carrying the `"ICC_PROFILE\0"` signature (T.872 / Annex L of
T.871) appear, an `IccProfileChunks` summary on
`JpegInfo::icc_profile` reports the declared chunk `total`, the
cumulative `total_payload_len`, the per-segment
`(seq_no, payload_len)` ordering, and an `is_complete()`
predicate that returns true when the sequence numbers cover
`1..=total` exactly once. Every COM (comment) marker segment
(T.81 §B.2.4.5) seen in the prefix is collected — in source
order — into `JpegInfo::comments` as a `Vec<JpegComment>`; each
entry holds the `Lc-2` comment bytes verbatim (T.81 leaves their
interpretation to the application), with `len()` / `is_empty()`
accessors plus an `as_str_lossy()` UTF-8 projection and an
`is_ascii_text()` predicate for the common human-readable
annotation case (zero-length `Lc = 2` comments and COM segments
appearing anywhere in the prefix, including between SOF and SOS,
are all captured). Every DQT (Define Quantization Table, T.81
§B.2.4.1) segment in the prefix is parsed into
`JpegInfo::quant_tables` as a `Vec<QuantTableInfo>` — one entry
per defined table in wire order (a single DQT segment may pack
several). Each `QuantTableInfo` carries the destination `id`
(`Tq ∈ 0..=3`), a `sixteen_bit` precision flag (`Pq`), and the
64 quantizer elements un-zigzagged into natural (row-major)
order, plus `dc()` (the `[0]` element) and `is_all_ones()`
(no-quantization / lossless-quality DCT) helpers. A later DQT
that redefines a destination is reported *in addition to* the
earlier one (source-order history for faithful re-encode replay);
`JpegInfo::quant_table_for(id)` does the last-wins lookup that
mirrors decode-time resolution. Lossless (SOF3 / SOF11) frames
carry no DQT, so `quant_tables` is empty there. The standalone
`parse_dqt_segment(payload, &mut Vec<QuantTableInfo>)` validator
is also re-exported. No entropy decoding, no DCT, no
allocation proportional to the scan body — O(prefix-length).
Useful for pipeline triage (pick a target pixel format),
fallback-decoder routing without spinning up the full decode
path, DPI-aware thumbnail sizing, and corpus summarisation. The
`SofKind` exposes `is_supported_by_decoder()`, `is_dct()`, and
`is_arithmetic()` helpers so callers can negotiate without
matching on every variant by hand. Standalone
`parse_jfif_app0(payload) -> Result<JfifApp0>`,
`parse_jfxx_app0(payload) -> Result<JfxxApp0>`,
`parse_adobe_app14(payload) -> Result<AdobeApp14>`,
`parse_icc_profile_app2(payload) -> Result<IccProfileApp2Chunk<'_>>`,
and `parse_dqt_segment(payload, &mut Vec<QuantTableInfo>)`
validators are also re-exported for callers that already have the
APP0 / APP14 / APP2 / DQT payload bytes in hand. Standalone surface —
the inspector requires neither the `registry` feature nor an
`oxideav-core` dep.

```rust
use oxideav_mjpeg::{inspect_jpeg, SofKind, ChromaSubsampling};

# let jpeg_bytes: Vec<u8> = std::fs::read("in.jpg").unwrap();
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
- **SOF10** (progressive, arithmetic) — the SOF2 scan structure with
  the Annex D Q-coder as the entropy layer per T.81 §G.1.3. DC first
  scans reuse the §F.1.4.1 sequential model on the point-transformed
  values; DC refinement bits use the fixed 0.5 estimate; AC first
  scans run the §F.1.4 procedure per band (`Kmin = Ss`, EOB =
  end-of-band, DAC `Kx` honoured); AC refinement scans follow the
  §G.1.3.3 model (Figures G.10 / G.11, Table G.2 — 189 statistics
  bins, end-of-band decision bypassed below the prior scan's EOBx).
  Restart intervals re-seed coder + statistics + DC prediction.
  `P = 8` and `P = 12`, 4-component CMYK / YCCK at `P = 8` — same
  output shaping as SOF2 via the shared coefficient accumulator.
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
- **Lossless arithmetic JPEG (SOF11)** — the same Annex H coding model
  (grayscale `P ∈ 2..=16`, three-component RGB-class `P ∈ 2..=16`,
  four-component CMYK-class `P = 8`, all predictors, point transform,
  restart intervals) with the modulo-2^16 prediction differences
  entropy-coded by the Annex D Q-coder under the T.81 §H.1.2.3
  two-dimensional statistical model: each binary decision is
  conditioned on the classifications of the differences coded for the
  sample to the left and the sample in the line above (the 5 × 5
  `L_Context(Da, Db)` array of Figure H.2, 158 statistics bins per
  component per Table H.3), with the DAC marker's DC-conditioning
  `(L, U)` bounds honoured (defaults `(0, 1)` per §H.1.2.3.3). The
  first line of the scan and of each restart interval uses the 1-D
  horizontal predictor per §H.1.2.1. Output shaping is shared with
  the SOF3 path (bit-exact reconstruction, same precision-driven
  pixel-format policy).
- **CMYK / YCCK** 4-component JPEGs → packed `PixelFormat::Cmyk`.
  Adobe APP14 transform flag honoured: transform=0 (Adobe CMYK, stored
  inverted) un-inverts on decode; transform=2 (YCCK) converts back to
  CMYK via BT.601 YCbCr→RGB→CMY and K inversion; no APP14 → plain
  ("regular", C=0 = no ink) pass-through.
- **Every T.81 §A.1.1 sampling layout.** Luma at `1×1` / `2×1` / `2×2`
  / `4×1` over `1×1` chroma decodes to the native planar `Yuv444P` /
  `Yuv422P` / `Yuv420P` / `Yuv411P` (`…P12Le` at `P = 12`); every other
  legal combination (`4×2` or `1×2` luma, mixed chroma factors, chroma
  oversampled relative to luma, …) is resampled onto the frame grid
  (`DecodeOptions::upsampling`; lossless frames replicate) and emitted
  as 4:4:4. With `ChromaUpsampling::Replicate` the result is within ±1
  of `djpeg -nosmooth` (`tests/fixtures/sampling/`).
- **Non-interleaved scans** (one `SOS` per component, sequential or
  progressive) cover the component's own `ceil(xi / 8) × ceil(yi / 8)`
  block extent (§A.2.2 / §A.2.4), not the MCU-padded grid.
- **§B.5 abbreviated streams**: `decode_with(data,
  &DecodeOptions::new().with_tables(tables))` preloads a tables-only
  stream (DQT / DHT / DAC / DRI — TIFF `JPEGTables`) and decodes a
  table-less image stream against it; the registry decoder takes the
  tables stream from `CodecParameters::extradata`.
- Grayscale (single-component → `Gray8`).
- **Baseline RGB** (3-component SOF0 at `H = V = 1`, signalled by either
  an Adobe APP14 `transform = 0` segment or component IDs `'R'/'G'/'B'`
  in the SOF) → packed `PixelFormat::Rgb24` (single plane,
  `stride = width * 3`). The encoder's matching `encode_jpeg_rgb24_*`
  entry points emit both signals (APP14 + component-id triple) by
  default; the decoder accepts either, so a caller-supplied APP-segment
  override that drops the APP14 still round-trips.
- Restart markers (`RSTn`) + DRI.
- **Hierarchical spatial lossless progression (T.81 Annex J)** — a
  `DHP` marker (§B.3.2) before the first frame routes the stream to
  the hierarchical control loop (§J.2.1). The lowest-resolution stage
  is a non-differential `SOF3` (Huffman) **or `SOF11` (arithmetic)**
  lossless frame; each higher-resolution
  stage is a matching `SOF7` (Huffman) **or `SOF15` (arithmetic)**
  differential lossless frame whose reference is ×2
  bi-linearly upsampled per the `EXP` segment (§B.3.3 / §J.1.1.2) and
  whose decoded difference is added back modulo `2^P` (§J.2.1). The
  differential frames decode with the §J.2.3.2 model (difference
  coded directly, predictor `Ss = 0`); the arithmetic frames use the
  §H.1.2.3 two-dimensional Q-coder statistical model (DAC
  DC-conditioning `(L, U)` honoured, default `(0, 1)`). Frames may be
  single-component
  (grayscale, any precision `P ∈ 2..=16` → `Gray8` / `Gray*Le`),
  three-component (RGB-class, `P ∈ 2..=16` → packed `Rgb24` /
  planar `Gbrp*Le` / packed `Rgb48Le`), or four-component
  (CMYK-class, `P = 8` → packed `Cmyk`, Adobe APP14 transform
  honoured), all with every component at `H = V = 1`. Reconstruction
  is bit-exact. The point transform is constrained to `Pt = 0` on
  this path; subsampled frames and the differential DCT arithmetic
  variants (SOF13 / SOF14) still return `Unsupported` (see "Not
  supported" below).
- **Hierarchical DCT progression (T.81 §K.7.2.1)** — the same `DHP` /
  `EXP` control loop also decodes a DCT progression: the
  non-differential first frame is `SOF0` (baseline), `SOF1` (extended
  sequential) or `SOF2` (progressive) — its reconstructed image
  samples (dequant + IDCT + §A.3.1 level shift, clamped to `0..2^P`)
  seed the reference component planes. Each higher-resolution stage is
  a `SOF5` differential sequential **or `SOF6` differential progressive**
  DCT frame decoded under the
  §J.2.3.1 model (IDCT computed *without* the level shift, DC
  coefficient decoded directly with no inter-block prediction — the
  `SOF6` path applies that direct-DC rule to the first DC scan of the
  same spectral-selection / successive-approximation decomposition the
  non-hierarchical `SOF2` path uses); the
  resulting signed difference is added to the ×2 EXP-upsampled
  reference modulo `2^16` (§J.2.1) and folded into the displayable
  `0..2^P` range. Frames are single-component (grayscale, `P = 8` /
  `P = 12` → `Gray8` / `Gray12Le`), three-component RGB-class
  (component IDs `R`/`G`/`B` or Adobe APP14 `transform = 0` → packed
  `Rgb24`, `P = 8` / `P = 12`), three-component YUV-class
  (component IDs other than `R`/`G`/`B`, no Adobe `transform = 0` →
  planar `Yuv444P` at `P = 8`, planar `Yuv444P12Le` at `P = 12`), or
  four-component CMYK-class
  (→ packed `Cmyk`, Adobe APP14 transform honoured, `P = 8`), all
  `H = V = 1`. Reconstruction is colour-blind — every component
  accumulates in its own sample space modulo `2^16` regardless of
  colour class, and the YCbCr / RGB / CMYK interpretation is deferred
  entirely to the display-time output shaping (no in-loop colour
  conversion), so the YUV-class Y/Cb/Cr planes pass through verbatim
  exactly as in the non-hierarchical sequential path. The frame mode
  is fixed by the first frame (T.81 §K.7.2 forbids mixing DCT and
  lossless non-differential frames). A DCT progression may also be
  **terminated by a differential lossless `SOF7` (Huffman) or `SOF15`
  (arithmetic) frame** (§K.7.2): the
  difference decodes under the §J.2.3.2 lossless model (coded
  directly, predictor `Ss = 0`), is added to the ×2 EXP-upsampled
  reference modulo `2^16` (§J.2.1), and folds into the displayable
  `0..2^P` range — 1 / 3 / 4 components, colour class inherited from the
  DCT first frame (grayscale → `Gray8`/`Gray*Le`, RGB-class → `Rgb24`,
  YUV-class → `Yuv444P`, CMYK-class → `Cmyk`). Still `Unsupported`:
  `P = 12` 4-component DCT progressions (no high-bit-depth packed CMYK
  `PixelFormat`) and the differential DCT arithmetic
  variants (SOF13 / SOF14).
- **DNL (Define Number of Lines, T.81 §B.2.5)** — when the SOF frame
  header codes the number of lines `Y = 0`, the real line count is
  recovered from the mandatory DNL segment (`0xFFDC`) that immediately
  follows the first scan, and the frame is decoded at that height. The
  `Y = 0` case without a following DNL (the segment is mandatory there
  per §B.2.5), and a malformed `NL = 0` DNL, are both rejected. Applies
  to every scan-decomposition path (baseline fast path, the
  sequential / progressive / arithmetic accumulator paths, and
  lossless).
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

- **General T.81 writer** (`oxideav_mjpeg::t81`, see "General T.81
  writer" above) — sequential `SOF0` / `SOF1` at `P = 8` / `12`,
  progressive `SOF2` at `P = 8` / `12`, lossless `SOF3` at `P ∈
  2..=16`; 1..=4 components with any §A.1.1 sampling layout and
  per-component table destinations; restart intervals on every
  process; Annex K.3 typical or K.2 optimal Huffman tables; §B.5
  abbreviated pairs; JFIF / RGB / CMYK / YCCK signalling. The
  per-format entry points below predate it and stay for compatibility.
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
- **SOF11** (lossless, arithmetic-coded) — single-component grayscale
  and three-component interleaved (RGB-class) at any precision
  `P ∈ 2..=16`, three-component **subsampled YUV-class** at `P = 8`
  (luma `1×1` / `2×1` / `2×2` / `4×1`, chroma `1×1` — the §A.2.3
  interleaved-MCU layout), plus four-component interleaved (CMYK-class)
  at `P = 8`, with every Annex H Table H.1 predictor `1..=7`. The
  Q-coder counterpart of the SOF3 path: the modulo-2^16 prediction
  differences are entropy-coded by the Annex D arithmetic coder under
  the §H.1.2.3 two-dimensional statistical model (each component
  modelled independently, no DAC segment → default conditioning
  `(L, U) = (0, 1)`). Bit-exact for every predictor, the half-modulus
  `Di = 32768` case, non-zero point transform, and per-interval restart
  re-seeding via `encode_lossless_arith_jpeg_grayscale_with_opts` /
  `encode_lossless_arith_jpeg_rgb_with_opts` /
  `encode_lossless_arith_jpeg_yuv_with_opts` /
  `encode_lossless_arith_jpeg_cmyk_with_opts`. The four-component path
  honours the Adobe APP14 colour-transform flag identically to the
  Huffman SOF3 CMYK encoder (no-APP14 / Adobe-CMYK round-trips are
  bit-exact; YCCK is a lossy interop convention).
- 4:4:4 / 4:2:2 / 4:2:0 YUV input on the lossy paths, plus single-
  component `Gray8` and packed `Rgb24` on the baseline SOF0 path;
  `Gray8` / `Gray10Le` / `Gray12Le` / `Gray16Le` input on the lossless
  path.
- Optional DRI + `RSTn` emission on the baseline path (off by default;
  see the Encoder section above).

Not supported (decoder returns `Error::Unsupported`):

- Hierarchical mode (T.81 Annex J) is **partially** supported. Both
  the **spatial lossless progression** (§K.7.2.2) and the **DCT
  progression** (§K.7.2.1) decode through the same `DHP` (§B.3.2) /
  `EXP` (§B.3.3) control loop with ×2 bi-linear reference upsampling
  (§J.1.1.2) — see the decoder coverage list above for the full
  description. The **lossless** path covers `SOF3` (Huffman) / `SOF11`
  (arithmetic) non-differential +
  `SOF7` / `SOF15` differential frames (1/3/4-component, `P` up to 16,
  modulo-2^P reconstruction, §J.2.3.2 model; the arithmetic frames use
  the §H.1.2.3 Q-coder model with DAC `(L, U)` conditioning). The
  **DCT** path covers `SOF0`
  / `SOF1` / `SOF2` non-differential + `SOF5` differential sequential
  **and `SOF6` differential progressive**
  frames (1-component grayscale or 3-component RGB-class at `P = 8` /
  `P = 12`, 3-component YUV-class at `P = 8` → planar `Yuv444P` and at
  `P = 12` → planar `Yuv444P12Le`, and
  4-component CMYK-class at `P = 8` → packed `Cmyk` with the Adobe
  APP14 transform honoured, §J.2.3.1 model with the level-shift-free
  IDCT + direct DC, modulo-2^16 reconstruction). A DCT progression may
  additionally be **terminated by a differential lossless `SOF7`
  (Huffman) or `SOF15` (arithmetic) frame** (§K.7.2 — §J.2.3.2 model,
  add modulo `2^16`, fold to `0..2^P`;
  1/3/4-component, colour class inherited from the DCT first frame).
  Still rejected with
  `Unsupported`: subsampled (`H/V != 1`) hierarchical frames; `P = 12`
  4-component DCT progressions; and the differential DCT arithmetic
  variants (SOF13 / SOF14). The non-hierarchical arithmetic variants
  are all supported: SOF9 (extended sequential) at `P=8`, SOF10
  (progressive) at `P=8` / `P=12`, and SOF11 (lossless) at every
  Annex H precision — see the decoder coverage list above.
- 12-bit 4-component progressive (SOF2 / SOF10 with `Nf = 4,
  P = 12`) — the workspace `PixelFormat` enum has no 12-bit CMYK
  variant. `P = 8` 4-component CMYK / YCCK *is* supported on the
  sequential (SOF0 / SOF1) and progressive (SOF2 / SOF10) scan
  decompositions, with the Adobe APP14 transform flag honoured.
- 4-component lossless above `P = 8` (the workspace `PixelFormat`
  enum has no high-bit-depth CMYK variant — wider precisions are
  rejected with `Unsupported`). `P = 8` 4-component lossless *is*
  supported on both encode and decode with the Adobe APP14 transform
  flag honoured on output.
- Lossless (SOF3 Huffman **and** SOF11 arithmetic) with **non-unit
  sampling factors** is decoded for the three-component YUV-class
  layout: the luma component may be oversampled `1×1` / `2×1` / `2×2` /
  `4×1` with both chroma components at `1×1`, decoding via the T.81
  A.2.3 interleaved-MCU ordering to a planar `Yuv444P` / `Yuv422P` /
  `Yuv420P` / `Yuv411P` frame (each component cropped to its true extent
  per A.2.4). The SOF11 path models each component independently under
  the §H.1.2.3 two-dimensional statistical model over its own padded
  grid, addressing the `L_Context(Da, Db)` conditioning neighbours by
  absolute grid coordinate so the §A.2.3 walk's non-raster sample order
  is handled correctly. The matching encoders are
  `encode_lossless_jpeg_yuv` / `_with_opts` (SOF3) and
  `encode_lossless_arith_jpeg_yuv` / `_with_opts` (SOF11), which accept
  every luma factor pair the §B.2.3 bound admits; layouts without a
  planar pixel format decode to 4:4:4 (see the decoder list). Still
  rejected with `Unsupported`: four-component (CMYK-class) lossless
  subsampling.

## Fuzzing

The `fuzz/` sub-crate runs fourteen cargo-fuzz harnesses against the
public encoder + decoder + RTP surface, executed daily by the
org-wide reusable fuzz workflow (60-minute budget, ~4.5 min/target):

- `metadata_embed` — the metadata marker paths: arbitrary bytes into
  `read_metadata` / `embed_metadata` (fuzz-sized ICC / Exif / XMP
  blobs at the limit bands, a fuzz-chosen `JfifPolicy`) /
  `strip_metadata`, asserting embedded blobs read back, untouched kinds
  are kept, a strip leaves none, the picture decodes to the same
  planes and the registry decoder's frame record equals the record;
  plus fuzz-shaped `APP2 ICC_PROFILE` chunk sequences (sequence
  numbers, counts, zero-length chunks) in front of a valid frame
  through `read_metadata` / `info` / lenient and strict `decode_with`.
  First minutes found the `TEM` length disagreement with the decoder
  (fixed); 10 minutes after: 1.81 M executions, cov 4408, no findings.

- `standalone_api` — the still-image contract surface: arbitrary bytes
  into `probe` / `info` / `decode_with` (1 Mpixel cap; 4096-pixel cap
  or strict mode alternating) / `decode_rgba8`, asserting `info` and
  `decode` agree on layout, geometry, precision and colour, then back
  out through `encode` (lossless, bit-exact round trip) and
  `encode_rgb8`. Found, on its first runs, two header-scan / decoder
  disagreements (truncated-length skipping, differential-SOF
  rejection order) — both fixed so `info` mirrors the decoder exactly.
  Last local 300 s run: 782 k executions, cov 3317, no findings.

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
- `lossless_decode` — wraps fuzz-supplied bytes (≤ 8 KiB) in a minimal
  SOF3 (Huffman) / SOF11 (arithmetic) lossless envelope: full precision
  ladder 2..=16 (incl. the SSSS=16 / `Di = 32768` half-modulus case),
  raw predictor and point-transform nibbles (invalid encodings
  included), DNL `Y = 0` resolution with an `NL = 0` corruption knob,
  DRI and DAC control bits. Contract: never panic.
- `lossless_self_roundtrip` — encoder-in-loop **bit-exact** oracle:
  fuzz-derived pixels through the lossless encoders
  (Huffman/arithmetic × grayscale/3-component × predictor 1..=7 ×
  `Pt` × restart interval) and back through the public decoder;
  reconstruction must equal `(s >> Pt) << Pt` with zero tolerance.
  Found the empty-final-restart-segment decoder bug (see CHANGELOG)
  within its first minute of running.
- `hierarchical_roundtrip` — encoder-in-loop over eight Annex J modes
  (spatial-lossless pyramids, DCT pyramids, DCT terminated by a
  differential lossless stage; Huffman and arithmetic), driving DHP /
  EXP / differential-SOF decode on every iteration. Bit-exact oracle
  on the lossless(-final) modes; decode-success + geometry oracle on
  the lossy DCT modes.
- `t81_roundtrip` — encoder-in-loop over the general T.81 writer:
  sequential / progressive / lossless, 8- and 12-bit (lossless
  2..=16), every §A.1.1 sampling layout, typical vs K.2 optimal
  tables, restart intervals, §B.5 abbreviated pairs (decoded through
  `decode_with` + `DecodeOptions::with_tables`) and JFIF / RGB / CMYK
  signalling.
  Bit-exact oracle on the lossless process (incl. the Adobe-inverted
  CMYK + point-transform composition), decode-success + geometry
  oracle on the DCT processes. First 300 s foreground run: 2.08 M
  executions, no findings.
- `registry_encoder` — the framework encoder's contract surface:
  fuzz-chosen pixel format (declared, alias or undeclared), geometry,
  option bag (`quality` / `process` / `chroma` / `tables` / `restart`
  / `predictor` / `precision` / `abbreviated`) and frame bytes, with
  planes occasionally short or missing. Oracle: no panic, every
  refusal `Unsupported` / `InvalidData`, and an encoded stream decodes
  through the registry decoder to the label stamped on the output
  parameters with that label's plane count. First run: a short plane
  panicked inside the historical 8-bit writers within seconds (fixed:
  `send_frame` checks every plane's extent first); 10 minutes / 7.3 M
  executions after that, no findings.
- `jpeg_self_roundtrip` / `jpeg_progressive_self_roundtrip` —
  encode → decode round-trip with ±2 LSB YUV tolerance.
- `libjpeg_encode_oxideav_decode` / `oxideav_encode_libjpeg_decode` —
  cross-decode against a system JPEG library used as a black-box
  validator, loaded via `dlsym` at runtime (no `*-sys` crate in
  the dep tree). The loader probes the conventional Homebrew install
  paths explicitly so macOS hosts don't run blind.

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

Fourteen small scenarios plus a `full_size` group (1600 × 1200
`encode_rgb8` / `decode` / `decode_rgb8` per chroma layout, serial and
on four workers), each fed by a deterministically-built in-bench
fixture (xorshift32 + low-amplitude triangle-wave gradient — no
committed payload files, no `docs/` reads, no third-party library
calls):

- `baseline_encode/{yuv420_256x256_q75, yuv444_64x64_q75}` — full SOF0
  path: forward DCT, quantise, Huffman run-length encode, marker
  emission; the small 4:4:4 fixture isolates per-call header /
  table-construction overhead from the per-block cost.
- `baseline_decode/{yuv420, yuv444, gray, restart8, rgb24}` — the
  sequential Huffman decode across chroma layouts, the RSTn resync
  path, and the packed-RGB copy-out, all driven through the `Decoder`
  trait so the bench tracks the same code path application callers
  exercise.
- `progressive_decode/yuv420_256x256_q75` — SOF2 multi-scan
  coefficient accumulation + shared render.
- `arith_decode/yuv420_256x256_q75` — SOF9 Annex D Q-coder entropy
  path.
- `lossless_decode/gray_pred{1,4}_256x256` — SOF3 per-sample predictor
  + wide-magnitude Huffman decode.
- `progressive_encode/yuv420_64x64_q75` — SOF2 spectral-selection
  decomposition (7 SOS scans).
- `lossless_encode/gray_pred{1,4}_256x256` — SOF3 grayscale encode;
  pred 1 (Ra) vs pred 4 (Ra + Rb − Rc) A/Bs the predictor-loop cost.

Headline numbers (Apple Silicon dev box, release profile, after the
round-410 hot-path tuning): baseline 4:2:0 encode 256×256 q75 runs
~154 µs / call (≈ 425 Melem/s); the matching decode ~145 µs / call
(≈ 450 Melem/s); progressive decode ~158 µs; arithmetic decode
~236 µs; lossless grayscale decode ~480-495 µs and encode ~360 µs
largely independent of predictor choice (magnitude / Huffman emission
dominates the per-sample cost).

Full-size pictures (`decode` = planar output, `decode_rgb8` = with
the default conversion; Apple Silicon, release, best of 7, quality 85
photograph with ≈ 2 bpp of detail, encoded by `cjpeg`), before → after
this round's decoder work:

| stream | `decode` | `decode_rgb8` |
|---|---|---|
| 1080p baseline 4:2:0 | 11.0 → 5.1–5.9 ms | 17.0 (replicated chroma) → 12.3–13.2 ms (luma-guided) |
| 1080p progressive 4:2:0 | 18.1 → 14.2 ms | 23.9 → 21.4 ms |
| 1080p Motion-JPEG-style 4:2:2, restart per MCU row | 14.2 → 7.4 ms; 2.5 ms with `threads = 8` | 20.1 → 17.2 ms |
| 4K baseline 4:2:0 | 42.7 → 20.0–22.9 ms | 66.0 → 48.7–51.6 ms |
| 4K progressive 4:2:0 | 67.8 → 54.1 ms | 91.0 → 82.7 ms |
| 4K, restart every 5 MCUs | 42.3 → 23.3 ms; 6.0 ms with `threads = 8` | |

The gains come from a factorised, vectorised float IDCT, a 64-bit
bit reader with bulk refills, a 10-bit symbol + magnitude lookahead,
and the fused row-wise chroma reconstruction. Encode times are in
*Encoder quality*.

1600 × 1200 and 4000 × 3000 (12 MP) pictures through the standalone
API, before → after the encoder and conversion work that followed
(same machine class; medians of 9 runs taken while other builds were
running, so read them to about ±10 %; multi-octave noise picture, 1.6 /
1.4 bpp at quality 85, decode streams from a black-box encoder;
`full_size` bench group). Output is byte-identical before and after and
for every worker count.

| case | 1600 × 1200 | 12 MP |
|---|---|---|
| `encode_rgb8` q85 4:2:0, defaults (optimal tables, RD levels) | 33.7 → 19.6 ms; 7.7 ms with `threads = 4` | 203 → 107 ms; 41 ms (4), 29 ms (8) |
| `encode_rgb8` q85 4:2:2, defaults | 39.1 → 20.6 ms; 8.1 ms (4) | 249 → 120 ms; 47 ms (4) |
| `encode_rgb8` q85 4:4:4, defaults | 49.1 → 25.3 ms; 9.7 ms (4) | 291 → 153 ms; 59 ms (4) |
| `encode_rgb8` q85 4:2:0, typical tables + nearest levels | 16.1 → 7.7 ms; 4.2 ms (4) | 109 → 42 ms; 23 ms (4) |
| `encode_rgb8` q85 4:2:0 progressive, defaults | 33.1 → 18.8 ms | 216 → 110 ms |
| `decode` baseline 4:2:0 (planar) | 4.7 → 4.7 ms | 25 → 25 ms |
| `decode_rgb8` baseline 4:2:0 (luma-guided) | 12.5 → 12.5 ms; 7.8 ms (4) | 73 → 73 ms; 43 ms (4) |
| `decode_rgb8` baseline 4:2:2 | 16.2 → 16.2 ms; 10.1 ms (4) | 96 → 96 ms; 58 ms (4) |
| `decode_rgb8` baseline 4:4:4 | 12.5 → 8.1 ms | 73 → 50 ms |
| `decode_rgb8` baseline 4:2:0, restart per MCU row | 11.6 → 11.6 ms; 5.1 ms (4), 4.1 ms (8) | 73 → 73 ms; 33 ms (4), 25 ms (8) |
| `decode` progressive 4:2:0 (planar) | 11.5 → 10.7 ms | 65 → 62 ms |

Where the time went: the RGB → YCbCr conversion now runs in 32-bit
integers with constant divisors (exact, same rounding); quantised
blocks are stored in zig-zag order as 16-bit levels with a non-zero
mask, so the entropy coders touch only the non-zero levels; the Huffman
block coders are compiled once per sink with the pending bits in
registers; the rate-distortion search works on zig-zag inputs with a
pre-scaled rate table. On the decode side the 4:4:4 / replicated
conversion works on row slices, progressive refinement scans visit
non-zero coefficients through a bit mask, and the interpolating
conversions split into row bands under `threads`. What a serial decode
of a subsampled stream spends beyond the planar decode is the
luma-guided reconstruction (see *Chroma reconstruction*);
`ChromaUpsampling::Replicate` is the cheapest conversion.

Every optimisation to these paths must be **bit-transparent**:
`tests/golden.rs` pins FNV-1a-64 hashes of the decoded planes for the
whole `docs/` fixture corpus and of the encoded bytes + decode-back
planes for one call of every public encode family, so any change to
rounding, accumulation order, or bitstream emission fails the suite
instead of silently drifting.

## License

MIT — see [LICENSE](LICENSE).
