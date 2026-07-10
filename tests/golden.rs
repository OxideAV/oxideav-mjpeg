#![cfg(feature = "registry")]
//! Golden-output pins for performance work on the codec hot paths.
//!
//! Any optimisation of the entropy loops, DCT/IDCT, sample packing, or
//! bit-I/O must be **bit-transparent**: the decoded planes and the encoded
//! byte streams have to stay exactly identical. These tests pin FNV-1a-64
//! hashes of
//!
//! 1. the decoded output planes of every fixture in the
//!    `docs/image/jpeg/fixtures/` corpus (baseline, progressive,
//!    arithmetic, 12-bit, lossless, restart, multi-scan, …), and
//! 2. the encoder byte stream **and** its decode-back planes for one
//!    representative call of every public encode family (baseline,
//!    progressive, spectral-selection + successive-approximation,
//!    arithmetic, lossless Huffman/arithmetic, hierarchical lossless,
//!    hierarchical DCT, CMYK),
//!
//! so a change that perturbs rounding, accumulation order, or emission
//! order anywhere in the pipeline trips a hash mismatch immediately.
//!
//! The corpus half skips (with a note) when `docs/` isn't checked out
//! next to the crate — CI for this repo doesn't carry the fixture tree.
//! The encoder half is fully self-contained (deterministic synthetic
//! frames) and always runs.
//!
//! Re-record after an *intentional* output change with:
//!
//! ```text
//! GOLDEN_RECORD=1 cargo test --test golden -- --nocapture
//! ```
//!
//! and paste the printed rows over the tables below. An intentional
//! output change must be justified in the same commit.

use std::path::PathBuf;

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase, VideoFrame};
use oxideav_mjpeg::registry::make_decoder;

// ---------------------------------------------------------------------------
// FNV-1a 64 (public-domain constants) over structured frame content.
// ---------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(FNV_OFFSET)
    }
    fn update(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        self.0 = h;
    }
    fn update_u64(&mut self, v: u64) {
        self.update(&v.to_le_bytes());
    }
    fn finish(self) -> u64 {
        self.0
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut f = Fnv::new();
    f.update(bytes);
    f.finish()
}

/// Hash a decoded video frame: plane count, then per-plane stride, length,
/// and payload. Captures both geometry and every output sample.
fn hash_video_frame(vf: &VideoFrame) -> u64 {
    let mut f = Fnv::new();
    f.update_u64(vf.planes.len() as u64);
    for p in &vf.planes {
        f.update_u64(p.stride as u64);
        f.update_u64(p.data.len() as u64);
        f.update(&p.data);
    }
    f.finish()
}

fn decode_to_frame(jpeg: &[u8]) -> VideoFrame {
    let params = CodecParameters::video(CodecId::new(oxideav_mjpeg::CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("make_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 1), jpeg.to_vec());
    dec.send_packet(&pkt).expect("send_packet");
    let frame = dec.receive_frame().expect("receive_frame");
    match frame {
        Frame::Video(vf) => vf,
        _ => panic!("expected Frame::Video"),
    }
}

fn record_mode() -> bool {
    std::env::var_os("GOLDEN_RECORD").is_some()
}

// ---------------------------------------------------------------------------
// Deterministic synthetic inputs (gradient + triangle + xorshift jitter —
// spatially correlated so the entropy coders see realistic run lengths).
// ---------------------------------------------------------------------------

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn natural_plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let mut rng = seed;
    for j in 0..h {
        for i in 0..w {
            let base = ((i + j) as i32) & 0xFF;
            let phase = (i as i32) & 31;
            let tri = if phase < 16 { phase } else { 31 - phase };
            let noise = (xorshift32(&mut rng) & 0x07) as i32 - 4;
            out[j * w + i] = (base + tri + noise).clamp(0, 255) as u8;
        }
    }
    out
}

fn natural_yuv_frame(w: u32, h: u32, pix: PixelFormat) -> VideoFrame {
    let (cw, ch): (usize, usize) = match pix {
        PixelFormat::Yuv444P => (w as usize, h as usize),
        PixelFormat::Yuv422P => (w.div_ceil(2) as usize, h as usize),
        PixelFormat::Yuv420P => (w.div_ceil(2) as usize, h.div_ceil(2) as usize),
        _ => panic!("unsupported pixel format in fixture"),
    };
    let y = natural_plane(w as usize, h as usize, 0xA5A5_5A5A);
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32) - (cw as i32) / 2) / 4).clamp(0, 255) as u8;
            cr[j * cw + i] = (128 + ((j as i32) - (ch as i32) / 2) / 4).clamp(0, 255) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y,
            },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

/// Packed RGB24 with three phase-shifted gradients.
fn natural_rgb24(w: usize, h: usize) -> Vec<u8> {
    let r = natural_plane(w, h, 0x1111_2222);
    let g = natural_plane(w, h, 0x3333_4444);
    let b = natural_plane(w, h, 0x5555_6666);
    let mut out = vec![0u8; w * h * 3];
    for k in 0..w * h {
        out[k * 3] = r[k];
        out[k * 3 + 1] = g[k];
        out[k * 3 + 2] = b[k];
    }
    out
}

/// Packed CMYK, four phase-shifted gradients.
fn natural_cmyk(w: usize, h: usize) -> Vec<u8> {
    let c = natural_plane(w, h, 0x0101_0202);
    let m = natural_plane(w, h, 0x0303_0404);
    let y = natural_plane(w, h, 0x0505_0606);
    let k = natural_plane(w, h, 0x0707_0808);
    let mut out = vec![0u8; w * h * 4];
    for i in 0..w * h {
        out[i * 4] = c[i];
        out[i * 4 + 1] = m[i];
        out[i * 4 + 2] = y[i];
        out[i * 4 + 3] = k[i];
    }
    out
}

/// Little-endian 12-bit-in-u16 grayscale ramp with jitter.
fn natural_gray12_le(w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 2];
    let mut rng = 0x9E37_79B9u32;
    for j in 0..h {
        for i in 0..w {
            let base = (((i * 13 + j * 29) as u32) & 0xFFF) as i32;
            let noise = (xorshift32(&mut rng) & 0x1F) as i32 - 16;
            let v = (base + noise).clamp(0, 4095) as u16;
            let o = (j * w + i) * 2;
            out[o] = (v & 0xFF) as u8;
            out[o + 1] = (v >> 8) as u8;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 1. docs corpus decode goldens
// ---------------------------------------------------------------------------

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from("../../docs/image/jpeg/fixtures")
        .join(name)
        .join("input.jpg")
}

/// (fixture dir name, FNV-1a-64 of the decoded frame). Recorded from the
/// decoder state at the start of the r410 performance round; every entry
/// was cross-checked against the ground-truth PPM/PGM tiers in
/// `docs_corpus.rs` at record time.
const CORPUS_GOLDENS: &[(&str, u64)] = &[
    ("arithmetic-coded", 0x3a1bb2ebae03e5e3),
    ("baseline-grayscale-32x32", 0x3a1bb2ebae03e5e3),
    ("baseline-q1-low-quality", 0x9f63e0dace4691a6),
    ("baseline-q100-no-loss", 0xf5aae192e438c1c7),
    ("baseline-rgb-32x32", 0x58070994bf0a7ac0),
    ("baseline-yuv411-32x32", 0xcb567fff08dc7a72),
    ("baseline-yuv420-128x128-q75", 0xf600a08eb2e53296),
    ("baseline-yuv422-32x32", 0xbc13f0003f22505a),
    ("extended-sequential-12bit", 0x6e33fe16e98d4d51),
    ("lossless-1986-mode", 0xcfa3cc771645ac5b),
    ("multi-scan-non-interleaved", 0xf600a08eb2e53296),
    ("progressive-yuv420-128x128", 0xf600a08eb2e53296),
    ("tiny-baseline-1x1", 0x7702fc6d6372a58c),
    ("with-icc-profile-embedded", 0xf600a08eb2e53296),
    ("with-restart-interval-8", 0xf600a08eb2e53296),
    ("without-jfif-marker", 0xf0e8c5f8fb0a31f6),
];

#[test]
fn corpus_decode_outputs_are_golden() {
    let mut checked = 0usize;
    for &(name, want) in CORPUS_GOLDENS {
        let path = fixture_path(name);
        let jpg = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip {name}: {} ({e})", path.display());
                continue;
            }
        };
        let vf = decode_to_frame(&jpg);
        let got = hash_video_frame(&vf);
        if record_mode() {
            println!("    (\"{name}\", {got:#018x}),");
            continue;
        }
        assert_eq!(
            got, want,
            "{name}: decoded output hash changed — decoder is no longer \
             bit-identical for this fixture"
        );
        checked += 1;
    }
    eprintln!(
        "corpus goldens verified: {checked}/{}",
        CORPUS_GOLDENS.len()
    );
}

// ---------------------------------------------------------------------------
// 2. encoder goldens (byte stream + decode-back), self-contained
// ---------------------------------------------------------------------------

/// (case name, FNV of the encoded JPEG bytes, FNV of its decode-back frame).
const ENCODE_GOLDENS: &[(&str, u64, u64)] = &[
    (
        "baseline_yuv420_128x128_q75",
        0x72a63aef8358e8eb,
        0x864ab298a0b5b2d2,
    ),
    (
        "baseline_yuv422_64x64_q90",
        0x2413174c44a0171f,
        0x9ee8968b742c44ae,
    ),
    (
        "baseline_yuv444_64x64_q50",
        0xa40afc212a838331,
        0xb344767f4a9a2d57,
    ),
    (
        "baseline_restart4_yuv420_64x64_q75",
        0x64c69156279fdabe,
        0xb02337044fd1e222,
    ),
    (
        "baseline_gray_64x64_q75",
        0x14feb0b3187f375a,
        0x00c88553a43d8b92,
    ),
    (
        "baseline_rgb24_64x64_q75",
        0x4f077d7971eb7490,
        0xadbaacb18a10e44a,
    ),
    (
        "baseline_cmyk_adobe0_64x64_q75",
        0x15d0d8cd11b98ab4,
        0x6aeba4ca99a080a7,
    ),
    (
        "progressive_yuv420_128x128_q75",
        0x12c23b75bafd7d29,
        0x864ab298a0b5b2d2,
    ),
    (
        "progressive_sa_yuv420_64x64_q75",
        0xbd1608d1b757fa29,
        0xb02337044fd1e222,
    ),
    (
        "arith_yuv420_128x128_q75",
        0x1a32df2072bccf6c,
        0x864ab298a0b5b2d2,
    ),
    (
        "arith_gray_64x64_q75",
        0x25764fe17333ac20,
        0x00c88553a43d8b92,
    ),
    (
        "arith_rgb24_64x64_q75",
        0x0507a91012e698a5,
        0xadbaacb18a10e44a,
    ),
    (
        "lossless_gray8_pred1_128x128",
        0x8ae032015e7f6aef,
        0xfcecae039d5d0877,
    ),
    (
        "lossless_gray8_pred4_128x128",
        0x782033fbfa84e434,
        0xfcecae039d5d0877,
    ),
    (
        "lossless_gray12_pred7_64x64",
        0xfb0716f31afad0fa,
        0xe72d46af3cdb067f,
    ),
    (
        "lossless_rgb8_pred2_64x64",
        0x495d04c59e771ce7,
        0x6ad4973c7c333c48,
    ),
    (
        "lossless_arith_gray8_pred1_64x64",
        0x18f62093a1a494a2,
        0x7df50c02b665df32,
    ),
    (
        "hier_lossless_gray8_pred1_l2_64x64",
        0xe2db9f2b7afbaab0,
        0x7df50c02b665df32,
    ),
    (
        "hier_dct_yuv444_l2_64x64_q75",
        0xfd1356bb4abc9f0d,
        0x924b4b3ccf92818d,
    ),
];

fn encode_case(name: &str) -> Vec<u8> {
    use oxideav_mjpeg::encoder as enc;
    match name {
        "baseline_yuv420_128x128_q75" => {
            let f = natural_yuv_frame(128, 128, PixelFormat::Yuv420P);
            enc::encode_jpeg(&f, 128, 128, PixelFormat::Yuv420P, 75).unwrap()
        }
        "baseline_yuv422_64x64_q90" => {
            let f = natural_yuv_frame(64, 64, PixelFormat::Yuv422P);
            enc::encode_jpeg(&f, 64, 64, PixelFormat::Yuv422P, 90).unwrap()
        }
        "baseline_yuv444_64x64_q50" => {
            let f = natural_yuv_frame(64, 64, PixelFormat::Yuv444P);
            enc::encode_jpeg(&f, 64, 64, PixelFormat::Yuv444P, 50).unwrap()
        }
        "baseline_restart4_yuv420_64x64_q75" => {
            let f = natural_yuv_frame(64, 64, PixelFormat::Yuv420P);
            enc::encode_jpeg_with_opts(&f, 64, 64, PixelFormat::Yuv420P, 75, 4).unwrap()
        }
        "baseline_gray_64x64_q75" => {
            let g = natural_plane(64, 64, 0xDEAD_BEEF);
            enc::encode_jpeg_grayscale(64, 64, &g, 64, 75).unwrap()
        }
        "baseline_rgb24_64x64_q75" => {
            let rgb = natural_rgb24(64, 64);
            enc::encode_jpeg_rgb24(64, 64, &rgb, 64 * 3, 75).unwrap()
        }
        "baseline_cmyk_adobe0_64x64_q75" => {
            let cmyk = natural_cmyk(64, 64);
            enc::encode_jpeg_cmyk(64, 64, &cmyk, 64 * 4, 75, Some(0)).unwrap()
        }
        "progressive_yuv420_128x128_q75" => {
            let f = natural_yuv_frame(128, 128, PixelFormat::Yuv420P);
            enc::encode_jpeg_progressive(&f, 128, 128, PixelFormat::Yuv420P, 75).unwrap()
        }
        "progressive_sa_yuv420_64x64_q75" => {
            let f = natural_yuv_frame(64, 64, PixelFormat::Yuv420P);
            enc::encode_jpeg_progressive_sa(&f, 64, 64, PixelFormat::Yuv420P, 75).unwrap()
        }
        "arith_yuv420_128x128_q75" => {
            let f = natural_yuv_frame(128, 128, PixelFormat::Yuv420P);
            enc::encode_arith_jpeg_yuv(&f, 128, 128, PixelFormat::Yuv420P, 75, 0).unwrap()
        }
        "arith_gray_64x64_q75" => {
            let g = natural_plane(64, 64, 0xDEAD_BEEF);
            enc::encode_arith_jpeg_grayscale(64, 64, &g, 64, 75, 0).unwrap()
        }
        "arith_rgb24_64x64_q75" => {
            let rgb = natural_rgb24(64, 64);
            enc::encode_arith_jpeg_rgb24(64, 64, &rgb, 64 * 3, 75, 0).unwrap()
        }
        "lossless_gray8_pred1_128x128" => {
            let g = natural_plane(128, 128, 0x1234_ABCD);
            enc::encode_lossless_jpeg_grayscale(128, 128, &g, 128, 8, 1).unwrap()
        }
        "lossless_gray8_pred4_128x128" => {
            let g = natural_plane(128, 128, 0x1234_ABCD);
            enc::encode_lossless_jpeg_grayscale(128, 128, &g, 128, 8, 4).unwrap()
        }
        "lossless_gray12_pred7_64x64" => {
            let g = natural_gray12_le(64, 64);
            enc::encode_lossless_jpeg_grayscale(64, 64, &g, 64 * 2, 12, 7).unwrap()
        }
        "lossless_rgb8_pred2_64x64" => {
            let r = natural_plane(64, 64, 0x1111_2222);
            let g = natural_plane(64, 64, 0x3333_4444);
            let b = natural_plane(64, 64, 0x5555_6666);
            enc::encode_lossless_jpeg_rgb(64, 64, [&r, &g, &b], [64, 64, 64], 8, 2).unwrap()
        }
        "lossless_arith_gray8_pred1_64x64" => {
            let g = natural_plane(64, 64, 0x1234_ABCD);
            enc::encode_lossless_arith_jpeg_grayscale(64, 64, &g, 64, 8, 1).unwrap()
        }
        "hier_lossless_gray8_pred1_l2_64x64" => {
            let g = natural_plane(64, 64, 0x1234_ABCD);
            enc::encode_hierarchical_lossless_jpeg_grayscale(64, 64, &g, 64, 8, 1, 2).unwrap()
        }
        "hier_dct_yuv444_l2_64x64_q75" => {
            let y = natural_plane(64, 64, 0xA5A5_5A5A);
            let cb = natural_plane(64, 64, 0x0F0F_F0F0);
            let cr = natural_plane(64, 64, 0xF00D_BEEF);
            enc::encode_hierarchical_dct_jpeg_yuv444(64, 64, [&y, &cb, &cr], [64, 64, 64], 75, 2)
                .unwrap()
        }
        _ => unreachable!("unknown encode golden case {name}"),
    }
}

#[test]
fn encoder_outputs_are_golden() {
    for &(name, want_bytes, want_frame) in ENCODE_GOLDENS {
        let jpeg = encode_case(name);
        let got_bytes = hash_bytes(&jpeg);
        let vf = decode_to_frame(&jpeg);
        let got_frame = hash_video_frame(&vf);
        if record_mode() {
            println!("    (\"{name}\", {got_bytes:#018x}, {got_frame:#018x}),");
            continue;
        }
        assert_eq!(
            got_bytes, want_bytes,
            "{name}: encoded byte stream changed — encoder is no longer \
             bit-identical for this input"
        );
        assert_eq!(
            got_frame, want_frame,
            "{name}: decode-back output changed — decoder is no longer \
             bit-identical for this stream"
        );
    }
}
