# oxideav-mjpeg

Pure-Rust **JPEG / Motion-JPEG** codec and still-image container —
decodes baseline (SOF0), extended-sequential (SOF1) and progressive
(SOF2) 8-bit JPEGs, encodes baseline JPEG using the Annex K "typical"
Huffman tables. YUV 4:4:4 / 4:2:2 / 4:2:0 and grayscale.
Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-container = "0.0"
oxideav-mjpeg = "0.0"
```

## Quick use

A JPEG file is a single SOI..EOI byte stream, so the still-image
container is a pass-through: open the file, pull one packet, decode.
Motion-JPEG streams (inside AVI / MOV / AMV / etc.) reuse the same
codec — each video packet is a full JPEG.

```rust
use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;
use oxideav_core::Frame;

let mut codecs = CodecRegistry::new();
let mut containers = ContainerRegistry::new();
oxideav_mjpeg::register(&mut codecs);
oxideav_mjpeg::register_containers(&mut containers);

let input: Box<dyn oxideav_container::ReadSeek> = Box::new(
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

### Codec / container IDs

- Codec: `"mjpeg"`. Decoder output / encoder input pixel formats:
  `Yuv444P`, `Yuv422P`, `Yuv420P`, plus `Gray8` on the decode side.
- Container: `"jpeg"`, matches `.jpg` / `.jpeg` / `.jpe` / `.jfif` by
  extension and by `FF D8 FF` magic bytes. One frame per file; muxing
  is a pass-through of the codec packet.

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

- SOF0 only, 8-bit Huffman, Annex K tables.
- 4:4:4 / 4:2:2 / 4:2:0 input.
- Optional DRI + `RSTn` emission (off by default; see the Encoder
  section above).

Not supported (decoder returns `Error::Unsupported`):

- Lossless JPEG (SOF3), hierarchical (SOF5+), arithmetic coding
  (SOF9..SOF15).
- 12-bit progressive (SOF2 with `P=12`), 12-bit 4:2:2 / 4:4:4 YUV.
- Progressive 4-component JPEGs.

## License

MIT — see [LICENSE](LICENSE).
