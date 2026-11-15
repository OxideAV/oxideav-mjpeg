//! Pinned RGB / RGBA output of every chroma reconstruction and both
//! YCbCr kernels.
//!
//! The conversion paths are free to change how they compute; these
//! hashes hold what they compute. Two corpora:
//!
//! * synthetic planar pictures (noise, ramps, two-level noise and a
//!   noisy gradient) at sizes from 1 × 1 up, in every planar YCbCr
//!   layout, handed to `JpegImage::to_rgb8_with` / `to_rgba8_with`
//!   directly — no encoder or entropy decoder in the way;
//! * every committed `tests/fixtures/**/*.jpg` the default decode
//!   accepts, at full and at half size.
//!
//! Each row of [`SYNTHETIC`] is one layout under one option pair, each
//! row of [`FIXTURES`] one upsampling; a hash is FNV-1a-64 over the
//! bytes of every picture in order. Every thread budget must give the
//! same bytes. Run with `PIN_PRINT=1` to print the tables.

use oxideav_mjpeg::{
    ChromaUpsampling, ConvertOptions, DecodeOptions, DecodeScale, JpegImage, MjpegPixelFormat,
    Plane, YcbcrKernel, decode_with,
};

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn feed(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

const MODES: [(&str, ChromaUpsampling); 4] = [
    ("guided", ChromaUpsampling::LumaGuided),
    ("linear", ChromaUpsampling::Linear),
    ("triangular", ChromaUpsampling::Triangular),
    ("replicate", ChromaUpsampling::Replicate),
];

const KERNELS: [(&str, YcbcrKernel); 2] = [
    ("exact", YcbcrKernel::Exact),
    ("fixed16", YcbcrKernel::FixedPoint16),
];

/// `(name, format, horizontal divisor, vertical divisor, bytes per sample)`.
const LAYOUTS: [(&str, MjpegPixelFormat, usize, usize, usize); 7] = [
    ("yuvj420p", MjpegPixelFormat::YuvJ420P, 2, 2, 1),
    ("yuvj422p", MjpegPixelFormat::YuvJ422P, 2, 1, 1),
    ("yuv411p", MjpegPixelFormat::Yuv411P, 4, 1, 1),
    ("yuvj444p", MjpegPixelFormat::YuvJ444P, 1, 1, 1),
    ("yuv420p12le", MjpegPixelFormat::Yuv420P12Le, 2, 2, 2),
    ("yuv422p12le", MjpegPixelFormat::Yuv422P12Le, 2, 1, 2),
    ("yuv444p12le", MjpegPixelFormat::Yuv444P12Le, 1, 1, 2),
];

const SIZES: [(u32, u32); 26] = [
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
    (3, 3),
    (4, 1),
    (4, 4),
    (5, 1),
    (5, 3),
    (6, 2),
    (7, 9),
    (8, 8),
    (9, 4),
    (15, 17),
    (16, 16),
    (17, 15),
    (31, 33),
    (32, 34),
    (33, 2),
    (2, 33),
    (35, 36),
    (64, 5),
    (65, 66),
    (100, 37),
    (129, 67),
    (255, 3),
];

fn xorshift(s: &mut u32) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s
}

/// The synthetic pictures of one layout, in a fixed order.
fn pictures(format: MjpegPixelFormat, dh: usize, dv: usize, bps: usize) -> Vec<JpegImage> {
    let mut seed = 0x1234_5678u32;
    let max = if bps == 2 { 4095u32 } else { 255 };
    let mut out = Vec::new();
    for (w, h) in SIZES {
        for kind in 0..4u32 {
            let (cw, ch) = ((w as usize).div_ceil(dh), (h as usize).div_ceil(dv));
            let mut plane = |pw: usize, ph: usize, index: u32| -> Plane {
                let mut v = Vec::with_capacity(pw * ph * bps);
                for y in 0..ph {
                    for x in 0..pw {
                        let s = match kind {
                            0 => xorshift(&mut seed) % (max + 1),
                            1 => {
                                ((x as u32 * 7 + y as u32 * 13 + index * 50) * (max / 255))
                                    % (max + 1)
                            }
                            2 => {
                                if xorshift(&mut seed) & 1 == 0 {
                                    0
                                } else {
                                    max
                                }
                            }
                            _ => ((x * 255 / pw) as u32 * (max / 255) + xorshift(&mut seed) % 5)
                                .min(max),
                        };
                        if bps == 2 {
                            v.extend_from_slice(&(s as u16).to_le_bytes());
                        } else {
                            v.push(s as u8);
                        }
                    }
                }
                Plane::new(pw * bps, v)
            };
            let planes = vec![
                plane(w as usize, h as usize, 0),
                plane(cw, ch, 1),
                plane(cw, ch, 2),
            ];
            let mut img = JpegImage::new(w, h, format, planes).expect("synthetic picture");
            if bps == 2 {
                img = img.with_precision(12);
            }
            out.push(img);
        }
    }
    out
}

/// RGB then RGBA bytes of `img`, the same for every thread budget.
fn convert(img: &JpegImage, opts: &ConvertOptions, h: &mut Fnv, what: &str) {
    let rgb = img.to_rgb8_with(opts);
    let rgba = img.to_rgba8_with(opts);
    for threads in [2, 3, 8] {
        let t = opts.with_threads(threads);
        assert!(
            img.to_rgb8_with(&t) == rgb,
            "{what}: RGB on {threads} workers"
        );
        assert!(
            img.to_rgba8_with(&t) == rgba,
            "{what}: RGBA on {threads} workers"
        );
    }
    // RGBA is RGB with opaque alpha.
    let (px, _) = rgba.as_chunks::<4>();
    assert_eq!(px.len() * 3, rgb.len(), "{what}: sizes");
    assert!(
        px.iter()
            .zip(rgb.as_chunks::<3>().0)
            .all(|(a, b)| a[..3] == b[..] && a[3] == 255),
        "{what}: RGBA differs from RGB"
    );
    h.feed(&rgb);
}

const SYNTHETIC: &[(&str, u64)] = &[
    ("yuvj420p guided exact", 0x506ee0bff57d7e51),
    ("yuvj420p guided fixed16", 0x32148540b2a54b98),
    ("yuvj420p linear exact", 0x254364f1d6cdab52),
    ("yuvj420p linear fixed16", 0x0018621753e79c4c),
    ("yuvj420p triangular exact", 0x2953a59796e66803),
    ("yuvj420p triangular fixed16", 0x7dd7b6405b2dc426),
    ("yuvj420p replicate exact", 0x8ba61a746481e33f),
    ("yuvj420p replicate fixed16", 0x6a07f8f0944aa9c6),
    ("yuvj422p guided exact", 0x4e968b0ba8653867),
    ("yuvj422p guided fixed16", 0xec642957bbfcf1f6),
    ("yuvj422p linear exact", 0xe518eb9423e581d3),
    ("yuvj422p linear fixed16", 0x1c491abf85f75e3c),
    ("yuvj422p triangular exact", 0x989326587092fbc8),
    ("yuvj422p triangular fixed16", 0x07f9e55180a33987),
    ("yuvj422p replicate exact", 0x4f86f335bb8d5200),
    ("yuvj422p replicate fixed16", 0x4769329169a276c1),
    ("yuv411p guided exact", 0xfe1edcd886004905),
    ("yuv411p guided fixed16", 0x167e6d947c47e232),
    ("yuv411p linear exact", 0x519a8f6cc7f62436),
    ("yuv411p linear fixed16", 0xb8c32e46d8b3ac78),
    ("yuv411p triangular exact", 0xe48c904190728794),
    ("yuv411p triangular fixed16", 0xeec7f4da449858b3),
    ("yuv411p replicate exact", 0xe48c904190728794),
    ("yuv411p replicate fixed16", 0xeec7f4da449858b3),
    ("yuvj444p guided exact", 0x31531fba7a4fbb74),
    ("yuvj444p guided fixed16", 0xa047e57f3c6c5910),
    ("yuvj444p linear exact", 0x31531fba7a4fbb74),
    ("yuvj444p linear fixed16", 0xa047e57f3c6c5910),
    ("yuvj444p triangular exact", 0x31531fba7a4fbb74),
    ("yuvj444p triangular fixed16", 0xa047e57f3c6c5910),
    ("yuvj444p replicate exact", 0x31531fba7a4fbb74),
    ("yuvj444p replicate fixed16", 0xa047e57f3c6c5910),
    ("yuv420p12le guided exact", 0xe8d6f026d9e145ef),
    ("yuv420p12le guided fixed16", 0xe8d6f026d9e145ef),
    ("yuv420p12le linear exact", 0x3f0da19d96897c6f),
    ("yuv420p12le linear fixed16", 0x3f0da19d96897c6f),
    ("yuv420p12le triangular exact", 0x9ba6ece90a424dca),
    ("yuv420p12le triangular fixed16", 0x9ba6ece90a424dca),
    ("yuv420p12le replicate exact", 0x40e4941763b1bc56),
    ("yuv420p12le replicate fixed16", 0x40e4941763b1bc56),
    ("yuv422p12le guided exact", 0x234ae11dc294b1d6),
    ("yuv422p12le guided fixed16", 0x234ae11dc294b1d6),
    ("yuv422p12le linear exact", 0xe0642fa9da122ecc),
    ("yuv422p12le linear fixed16", 0xe0642fa9da122ecc),
    ("yuv422p12le triangular exact", 0x234fd87da1ffcd3f),
    ("yuv422p12le triangular fixed16", 0x234fd87da1ffcd3f),
    ("yuv422p12le replicate exact", 0x6c9fcf9bd2b3e0b3),
    ("yuv422p12le replicate fixed16", 0x6c9fcf9bd2b3e0b3),
    ("yuv444p12le guided exact", 0xb1d2d62e710c87a0),
    ("yuv444p12le guided fixed16", 0xb1d2d62e710c87a0),
    ("yuv444p12le linear exact", 0xb1d2d62e710c87a0),
    ("yuv444p12le linear fixed16", 0xb1d2d62e710c87a0),
    ("yuv444p12le triangular exact", 0xb1d2d62e710c87a0),
    ("yuv444p12le triangular fixed16", 0xb1d2d62e710c87a0),
    ("yuv444p12le replicate exact", 0xb1d2d62e710c87a0),
    ("yuv444p12le replicate fixed16", 0xb1d2d62e710c87a0),
];

const FIXTURES: &[(&str, u64)] = &[
    ("guided", 0x8768814db777b9fd),
    ("linear", 0x69ba5bad37ebeb14),
    ("triangular", 0x49a8d526ac272be5),
    ("replicate", 0x234078a71cb4a7e9),
];

/// Decoded fixture pictures: the hashes cover this many.
const FIXTURE_PICTURES: usize = 254;

fn check(table: &[(&str, u64)], got: &[(String, u64)], name: &str) {
    if std::env::var_os("PIN_PRINT").is_some() {
        for (k, v) in got {
            println!("    (\"{k}\", 0x{v:016x}),");
        }
        return;
    }
    assert_eq!(table.len(), got.len(), "{name}: row count");
    for ((k, want), (gk, g)) in table.iter().zip(got) {
        assert_eq!(k, gk, "{name}: row order");
        assert_eq!(*want, *g, "{name}: {k} changed (0x{g:016x})");
    }
}

#[test]
fn synthetic_pictures_convert_to_the_pinned_bytes() {
    let mut got = Vec::new();
    for (lname, format, dh, dv, bps) in LAYOUTS {
        let pics = pictures(format, dh, dv, bps);
        for (mname, mode) in MODES {
            for (kname, kernel) in KERNELS {
                let opts = ConvertOptions::new()
                    .with_upsampling(mode)
                    .with_ycbcr(kernel);
                let mut h = Fnv::new();
                for img in &pics {
                    let what = format!("{lname} {}x{} {mname} {kname}", img.width(), img.height());
                    convert(img, &opts, &mut h, &what);
                }
                got.push((format!("{lname} {mname} {kname}"), h.0));
            }
        }
    }
    check(SYNTHETIC, &got, "synthetic");
}

fn jpegs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            jpegs(&p, out);
        } else if p.extension().is_some_and(|x| x == "jpg") {
            out.push(p);
        }
    }
}

#[test]
fn fixture_pictures_convert_to_the_pinned_bytes() {
    let root = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"));
    let mut files = Vec::new();
    jpegs(root, &mut files);
    files.sort();
    let mut images = Vec::new();
    for f in &files {
        let bytes = std::fs::read(f).expect("fixture");
        for scale in [DecodeScale::Full, DecodeScale::Half] {
            if let Ok(img) = decode_with(&bytes, &DecodeOptions::new().with_scale(scale)) {
                images.push((f.strip_prefix(root).expect("under root").to_owned(), img));
            }
        }
    }
    let mut got = Vec::new();
    for (mname, mode) in MODES {
        let opts = ConvertOptions::new().with_upsampling(mode);
        let mut h = Fnv::new();
        for (path, img) in &images {
            convert(img, &opts, &mut h, &format!("{} {mname}", path.display()));
        }
        got.push((mname.to_owned(), h.0));
    }
    if std::env::var_os("PIN_PRINT").is_some() {
        println!("FIXTURE_PICTURES = {}", images.len());
    } else {
        assert_eq!(images.len(), FIXTURE_PICTURES, "decoded fixture pictures");
    }
    check(FIXTURES, &got, "fixtures");
}
