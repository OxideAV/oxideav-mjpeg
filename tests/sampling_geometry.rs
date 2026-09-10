//! T.81 §A.1.1 sampling-factor geometry — black-box oracle.
//!
//! "For each component, sampling factors Hi and Vi are defined … Hi, Vi
//! ∈ 1..=4" (Table B.2), bounded only by the §B.2.3 interleave rule
//! `Σ Hi × Vi ≤ 10`. The decoder used to accept only the luma layouts
//! that map onto a planar `PixelFormat` (1×1 / 2×1 / 2×2 / 4×1 over 1×1
//! chroma) and rejected everything else — 4×2 luma in particular.
//!
//! Fixtures under `tests/fixtures/sampling/` are black-box validator
//! output: `samp_<layout>.jpg` is `cjpeg -quality 90 -sample <layout>`
//! of a 25×19 RGB source (odd dimensions so every layout pads a partial
//! MCU per §A.2.4), `samp_<layout>_y.pgm` is `djpeg -nosmooth -grayscale -pnm`
//! (the Y component, replication-upsampled where it is coded below the
//! frame resolution — the ±1 IDCT-rounding oracle) and
//! `samp_<layout>_rgb.ppm` is `djpeg -nosmooth -pnm` (replication
//! upsampling, the PSNR oracle for the chroma planes). `samp12_4x2.*`
//! is the same at `-precision 12` (16-bit PNM oracles).
//!
//! Layouts: `4x2` (luma 4×2 → 4:4:4 upsampled), `1x2` (luma 1×2 →
//! 4:4:4 upsampled), `2x2,2x1,1x2` (mixed chroma factors) and
//! `1x1,2x2,2x2` (chroma oversampled relative to luma — legal, exotic).

use oxideav_mjpeg::decoder::decode_jpeg;
use oxideav_mjpeg::encoder::{encode_lossless_arith_jpeg_yuv, encode_lossless_jpeg_yuv};

const W: usize = 25;
const H: usize = 19;

struct Fixture {
    name: &'static str,
    jpeg: &'static [u8],
    y: &'static [u8],
    rgb: &'static [u8],
    twelve_bit: bool,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "4x2",
        jpeg: include_bytes!("fixtures/sampling/samp_4x2.jpg"),
        y: include_bytes!("fixtures/sampling/samp_4x2_y.pgm"),
        rgb: include_bytes!("fixtures/sampling/samp_4x2_rgb.ppm"),
        twelve_bit: false,
    },
    Fixture {
        name: "1x2",
        jpeg: include_bytes!("fixtures/sampling/samp_1x2.jpg"),
        y: include_bytes!("fixtures/sampling/samp_1x2_y.pgm"),
        rgb: include_bytes!("fixtures/sampling/samp_1x2_rgb.ppm"),
        twelve_bit: false,
    },
    Fixture {
        name: "2x2,2x1,1x2",
        jpeg: include_bytes!("fixtures/sampling/samp_2x2_2x1_1x2.jpg"),
        y: include_bytes!("fixtures/sampling/samp_2x2_2x1_1x2_y.pgm"),
        rgb: include_bytes!("fixtures/sampling/samp_2x2_2x1_1x2_rgb.ppm"),
        twelve_bit: false,
    },
    Fixture {
        name: "1x1,2x2,2x2",
        jpeg: include_bytes!("fixtures/sampling/samp_1x1_2x2_2x2.jpg"),
        y: include_bytes!("fixtures/sampling/samp_1x1_2x2_2x2_y.pgm"),
        rgb: include_bytes!("fixtures/sampling/samp_1x1_2x2_2x2_rgb.ppm"),
        twelve_bit: false,
    },
    Fixture {
        name: "12-bit 4x2",
        jpeg: include_bytes!("fixtures/sampling/samp12_4x2.jpg"),
        y: include_bytes!("fixtures/sampling/samp12_4x2_y.pgm"),
        rgb: include_bytes!("fixtures/sampling/samp12_4x2_rgb.ppm"),
        twelve_bit: true,
    },
];

/// Parse a binary PNM (`P5` / `P6`) into `(maxval, samples)`; 16-bit
/// files are big-endian per the PNM convention.
fn pnm(data: &[u8]) -> (u32, Vec<u32>) {
    let mut i = 2;
    let mut fields = Vec::new();
    while fields.len() < 3 {
        while data[i].is_ascii_whitespace() {
            i += 1;
        }
        let s = i;
        while !data[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(
            std::str::from_utf8(&data[s..i])
                .unwrap()
                .parse::<u32>()
                .unwrap(),
        );
    }
    i += 1;
    assert_eq!((fields[0], fields[1]), (W as u32, H as u32));
    let maxval = fields[2];
    let body = &data[i..];
    let samples = if maxval > 255 {
        body.chunks(2)
            .map(|c| u32::from(c[0]) << 8 | u32::from(c[1]))
            .collect()
    } else {
        body.iter().map(|&b| u32::from(b)).collect()
    };
    (maxval, samples)
}

/// Read a decoded plane (`stride` bytes per row) as `W × H` samples
/// (8-bit bytes or 16-bit little-endian words).
fn plane(stride: usize, data: &[u8], twelve_bit: bool) -> Vec<u32> {
    let mut out = Vec::with_capacity(W * H);
    for y in 0..H {
        for x in 0..W {
            out.push(if twelve_bit {
                u32::from(data[y * stride + 2 * x]) | u32::from(data[y * stride + 2 * x + 1]) << 8
            } else {
                u32::from(data[y * stride + x])
            });
        }
    }
    out
}

fn psnr(a: &[u32], b: &[u32], peak: f64) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    }
}

#[test]
fn every_legal_sampling_layout_matches_the_validator_luma() {
    for fx in FIXTURES {
        let f = decode_jpeg(fx.jpeg, None).unwrap_or_else(|e| panic!("{}: {e}", fx.name));
        assert_eq!(f.planes.len(), 3, "{}: three planes", fx.name);
        // None of these layouts has a planar pixel format: every plane
        // comes out at full resolution (4:4:4).
        let bps = if fx.twelve_bit { 2 } else { 1 };
        for (pi, p) in f.planes.iter().enumerate() {
            assert_eq!(p.stride, W * bps, "{}: plane {pi} stride", fx.name);
            assert_eq!(p.data.len(), W * bps * H, "{}: plane {pi} size", fx.name);
        }
        let (_, want_y) = pnm(fx.y);
        let got_y = plane(f.planes[0].stride, &f.planes[0].data, fx.twelve_bit);
        let maxd = got_y
            .iter()
            .zip(&want_y)
            .map(|(&a, &b)| (a as i64 - b as i64).abs())
            .max()
            .unwrap();
        let peak = if fx.twelve_bit { 4095.0 } else { 255.0 };
        let db = psnr(&got_y, &want_y, peak);
        eprintln!("{}: Y max|d| = {maxd}, PSNR = {db:.1} dB", fx.name);
        // Both sides replicate samples where a component has to be
        // upsampled (`-nosmooth`): only the IDCT rounding (integer vs
        // float transform) separates them.
        assert!(
            maxd <= 1,
            "{}: Y max|d| = {maxd} vs djpeg -nosmooth -grayscale",
            fx.name
        );
        assert!(db > 50.0, "{}: Y PSNR = {db:.1} dB", fx.name);
    }
}

/// Chroma: convert our 4:4:4 output to RGB (BT.601 full-range, the T.871
/// matrix) and compare against `djpeg -nosmooth`, whose replication
/// upsampling is the same nearest-neighbour rule; only colour-conversion
/// rounding separates the two.
#[test]
fn every_legal_sampling_layout_matches_the_validator_rgb() {
    for fx in FIXTURES {
        let f = decode_jpeg(fx.jpeg, None).unwrap();
        let peak = if fx.twelve_bit { 4095.0 } else { 255.0 };
        let half = if fx.twelve_bit { 2048.0 } else { 128.0 };
        let y = plane(f.planes[0].stride, &f.planes[0].data, fx.twelve_bit);
        let cb = plane(f.planes[1].stride, &f.planes[1].data, fx.twelve_bit);
        let cr = plane(f.planes[2].stride, &f.planes[2].data, fx.twelve_bit);
        let mut rgb = Vec::with_capacity(W * H * 3);
        for i in 0..W * H {
            let (yy, b, r) = (y[i] as f64, cb[i] as f64 - half, cr[i] as f64 - half);
            for v in [
                yy + 1.402 * r,
                yy - 0.344_136 * b - 0.714_136 * r,
                yy + 1.772 * b,
            ] {
                rgb.push(v.round().clamp(0.0, peak) as u32);
            }
        }
        let (maxval, want) = pnm(fx.rgb);
        assert_eq!(maxval as f64, peak);
        let db = psnr(&rgb, &want, peak);
        assert!(
            db > 45.0,
            "{}: RGB PSNR vs djpeg -nosmooth = {db:.1} dB",
            fx.name
        );
    }
}

/// Lossless (SOF3 / SOF11) YUV-class frames with luma factors outside the
/// planar-format set: the §A.2.3 interleaved MCU carries `H × V` luma
/// samples, the decoder replicates chroma onto the frame grid, and the
/// round-trip is bit-exact on every plane.
#[test]
fn lossless_yuv_exotic_luma_factors_roundtrip_bit_exact() {
    let y: Vec<u8> = (0..W * H).map(|i| (i * 7 % 253) as u8).collect();
    for (hf, vf) in [(4u8, 2u8), (1, 2), (2, 4), (3, 1), (1, 3), (2, 3)] {
        let cw = W.div_ceil(hf as usize);
        let ch = H.div_ceil(vf as usize);
        let cb: Vec<u8> = (0..cw * ch).map(|i| (i * 5 % 251) as u8).collect();
        let cr: Vec<u8> = (0..cw * ch).map(|i| (i * 11 % 241) as u8).collect();
        let huff = encode_lossless_jpeg_yuv(W as u32, H as u32, &y, W, &cb, cw, &cr, cw, hf, vf, 4)
            .unwrap();
        let arith =
            encode_lossless_arith_jpeg_yuv(W as u32, H as u32, &y, W, &cb, cw, &cr, cw, hf, vf, 4)
                .unwrap();
        for (tag, jpeg) in [("SOF3", huff), ("SOF11", arith)] {
            let f = decode_jpeg(&jpeg, None).unwrap();
            assert_eq!(f.planes.len(), 3);
            for (pi, p) in f.planes.iter().enumerate() {
                assert_eq!(p.stride, W, "{tag} {hf}x{vf}: plane {pi} is 4:4:4");
            }
            let got_y = plane(f.planes[0].stride, &f.planes[0].data, false);
            assert!(
                got_y.iter().zip(&y).all(|(&a, &b)| a == u32::from(b)),
                "{tag} {hf}x{vf}: Y"
            );
            for (pi, src) in [(1usize, &cb), (2usize, &cr)] {
                let got = plane(f.planes[pi].stride, &f.planes[pi].data, false);
                for yy in 0..H {
                    for xx in 0..W {
                        let want = src[(yy / vf as usize) * cw + xx / hf as usize];
                        assert_eq!(
                            got[yy * W + xx],
                            u32::from(want),
                            "{tag} {hf}x{vf}: plane {pi} at ({xx},{yy})"
                        );
                    }
                }
            }
        }
    }
}
