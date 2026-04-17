# oxideav-mjpeg

Pure-Rust JPEG / Motion-JPEG codec for oxideav

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a
100% pure Rust media transcoding and streaming stack. No C libraries, no FFI
wrappers, no `*-sys` crates.

## Features

- Baseline JPEG decode (SOF0) — 8-bit, 4:4:4 / 4:2:2 / 4:2:0 chroma
  subsampling, gray-scale, restart markers.
- **Progressive JPEG decode (SOF2)** — multi-scan spectral selection and
  successive approximation (DC refinement, AC first pass, AC refinement
  with EOB-run).
- Baseline JPEG encode using the Annex K "typical" Huffman tables.
- JPEG still-image container (`.jpg` / `.jpeg`).

## Usage

```toml
[dependencies]
oxideav-mjpeg = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).
