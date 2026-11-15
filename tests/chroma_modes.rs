//! The chroma reconstructions by their mathematics, and the packed
//! output layouts.
//!
//! Along a 2:1 axis T.871 §9 puts chroma sample `b` between output
//! samples `2b` and `2b + 1`, so the interpolating kernels weight the
//! two nearest samples 3 : 1 — `s(2b + 1) = (3 c[b] + c[b + 1]) / 4`,
//! `s(2b + 2) = (c[b] + 3 c[b + 1]) / 4` — and replicate the first and
//! last sample outwards; replication reads `c[x / 2]`. The expected
//! pixels below are worked out by hand from that and the T.871 §7
//! equations.

use oxideav_mjpeg::{
    ChromaUpsampling, ConvertOptions, JpegImage, MjpegPixelFormat, PackedLayout, Plane, YcbcrKernel,
};

const MODES: [ChromaUpsampling; 4] = [
    ChromaUpsampling::Replicate,
    ChromaUpsampling::Triangular,
    ChromaUpsampling::Linear,
    ChromaUpsampling::LumaGuided,
];

fn image(
    format: MjpegPixelFormat,
    (w, h): (u32, u32),
    (cw, ch): (usize, usize),
    y: impl Fn(usize, usize) -> u8,
    cb: impl Fn(usize, usize) -> u8,
    cr: impl Fn(usize, usize) -> u8,
) -> JpegImage {
    let plane = |pw: usize, ph: usize, f: &dyn Fn(usize, usize) -> u8| {
        let data = (0..ph)
            .flat_map(|j| (0..pw).map(move |i| f(i, j)))
            .collect();
        Plane::new(pw, data)
    };
    let planes = vec![
        plane(w as usize, h as usize, &y),
        plane(cw, ch, &cb),
        plane(cw, ch, &cr),
    ];
    JpegImage::new(w, h, format, planes).expect("test picture")
}

fn rgb(img: &JpegImage, mode: ChromaUpsampling) -> Vec<[u8; 3]> {
    let bytes = img.to_rgb8_with(&ConvertOptions::new().with_upsampling(mode));
    bytes.as_chunks::<3>().0.to_vec()
}

/// T.871 §7 at a fractional chroma value, in `f64` (the cases below
/// stay well away from ties).
fn colour(y: f64, cb: f64, cr: f64) -> [u8; 3] {
    let (b, r) = (cb - 128.0, cr - 128.0);
    let g = y - (0.114 * 1.772 * b + 0.299 * 1.402 * r) / 0.587;
    [y + 1.402 * r, g, y + 1.772 * b].map(|v| (v + 0.5).floor().clamp(0.0, 255.0) as u8)
}

#[test]
fn a_constant_colour_is_the_same_under_every_mode() {
    for (format, cw, ch) in [
        (MjpegPixelFormat::YuvJ420P, 9, 6),
        (MjpegPixelFormat::YuvJ422P, 9, 11),
        (MjpegPixelFormat::Yuv411P, 5, 11),
        (MjpegPixelFormat::YuvJ444P, 17, 11),
    ] {
        for (y, cb, cr) in [
            (100u8, 90u8, 200u8),
            (0, 0, 0),
            (255, 255, 255),
            (17, 128, 128),
        ] {
            let img = image(format, (17, 11), (cw, ch), |_, _| y, |_, _| cb, |_, _| cr);
            let want = colour(y as f64, cb as f64, cr as f64);
            for mode in MODES {
                let got = rgb(&img, mode);
                assert_eq!(got.len(), 17 * 11);
                assert!(
                    got.iter().all(|p| *p == want),
                    "{format} {mode:?} ({y}, {cb}, {cr}): {:?} != {want:?}",
                    got[0]
                );
            }
        }
    }
}

/// Four pixels over two chroma samples `Cb = 100, 201` (4:2:2), `Y = 100`,
/// `Cr = 128`:
///
/// ```text
/// replicate    Cb = 100     100      201      201
/// linear       Cb = 100     125.25   175.75   201
/// triangular   Cb = 100     125      176      201     ((501 + 2) >> 2, (703 + 1) >> 2)
/// ```
///
/// `B = Round(100 + 1.772 (Cb − 128))`, `G = Round(100 − 0.344136 (Cb − 128))`:
/// `Cb = 100` → `−49.616`, `+9.636`; `125.25` → `−4.873`, `+0.946`;
/// `125` → `−5.316`, `+1.032`; `175.75` → `+84.613`, `−16.432`;
/// `176` → `+85.056`, `−16.518`; `201` → `+129.356`, `−25.122`.
#[test]
fn one_axis_hand_computed() {
    let img = image(
        MjpegPixelFormat::YuvJ422P,
        (4, 1),
        (2, 1),
        |_, _| 100,
        |i, _| [100, 201][i],
        |_, _| 128,
    );
    assert_eq!(
        rgb(&img, ChromaUpsampling::Replicate),
        [
            [100, 110, 50],
            [100, 110, 50],
            [100, 75, 229],
            [100, 75, 229]
        ]
    );
    let linear = [
        [100, 110, 50],
        [100, 101, 95],
        [100, 84, 185],
        [100, 75, 229],
    ];
    assert_eq!(rgb(&img, ChromaUpsampling::Linear), linear);
    // Flat luma carries no detail to restore: the guided reconstruction
    // is the linear one.
    assert_eq!(rgb(&img, ChromaUpsampling::LumaGuided), linear);
    // A component two samples wide is replicated by the triangular
    // kernel; three samples interpolate.
    assert_eq!(
        rgb(&img, ChromaUpsampling::Triangular),
        rgb(&img, ChromaUpsampling::Replicate)
    );
    let img = image(
        MjpegPixelFormat::YuvJ422P,
        (6, 1),
        (3, 1),
        |_, _| 100,
        |i, _| [100, 201, 201][i],
        |_, _| 128,
    );
    assert_eq!(
        rgb(&img, ChromaUpsampling::Triangular)[..4],
        [
            [100, 110, 50],
            [100, 101, 95],
            [100, 83, 185],
            [100, 75, 229]
        ]
    );
}

/// The same two values down the rows of a 4:2:0 picture (six columns
/// of constant chroma): rows 0 … 3 carry `Cb = 100, 125.25, 175.75, 201`
/// (linear) and `100, 125, 176, 201` (triangular: `(4 · 501 + 8) >> 4`
/// and `(4 · 703 + 8) >> 4`, with `+ 7` on odd columns giving the same).
#[test]
fn two_axes_hand_computed() {
    let img = image(
        MjpegPixelFormat::YuvJ420P,
        (6, 4),
        (3, 2),
        |_, _| 100,
        |_, j| [100, 201][j],
        |_, _| 128,
    );
    let rows = |mode| -> Vec<[u8; 3]> {
        let px = rgb(&img, mode);
        for row in px.as_chunks::<6>().0 {
            assert!(row.iter().all(|p| *p == row[0]), "{mode:?}: row not flat");
        }
        px.as_chunks::<6>().0.iter().map(|r| r[0]).collect()
    };
    assert_eq!(
        rows(ChromaUpsampling::Replicate),
        [
            [100, 110, 50],
            [100, 110, 50],
            [100, 75, 229],
            [100, 75, 229]
        ]
    );
    let linear = [
        [100, 110, 50],
        [100, 101, 95],
        [100, 84, 185],
        [100, 75, 229],
    ];
    assert_eq!(rows(ChromaUpsampling::Linear), linear);
    assert_eq!(rows(ChromaUpsampling::LumaGuided), linear);
    assert_eq!(
        rows(ChromaUpsampling::Triangular),
        [
            [100, 110, 50],
            [100, 101, 95],
            [100, 83, 185],
            [100, 75, 229]
        ]
    );
}

/// A chroma ramp of 8 per sample is a ramp of 4 per pixel under the
/// linear kernel, `Cb(x) = 4 x + 8` inside the picture, with the first
/// and last pixel on the edge samples; replication gives steps of 8
/// every two pixels. Both axes at once for 4:2:0: `Cb(x, y) = 4 x + 2 y + c`.
#[test]
fn gradients() {
    let (w, h) = (24u32, 16u32);
    let ramp = |i: usize| (10 + 8 * i) as u8;
    let one = image(
        MjpegPixelFormat::YuvJ422P,
        (w, h),
        (12, 16),
        |_, _| 90,
        |i, _| ramp(i),
        |i, _| 255 - ramp(i),
    );
    let lin = rgb(&one, ChromaUpsampling::Linear);
    let rep = rgb(&one, ChromaUpsampling::Replicate);
    let tri = rgb(&one, ChromaUpsampling::Triangular);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let at = y * w as usize + x;
            let cb = match x {
                0 => 10.0,
                23 => 98.0,
                _ => 4.0 * x as f64 + 8.0,
            };
            assert_eq!(lin[at], colour(90.0, cb, 255.0 - cb), "linear ({x}, {y})");
            let cb = ramp(x / 2) as f64;
            assert_eq!(
                rep[at],
                colour(90.0, cb, 255.0 - cb),
                "replicate ({x}, {y})"
            );
            // The ramp lands on whole samples, so the triangular
            // kernel's rounding has nothing to round.
            assert_eq!(tri[at], lin[at], "triangular ({x}, {y})");
        }
    }
    // Luma constant: guided is linear.
    assert_eq!(rgb(&one, ChromaUpsampling::LumaGuided), lin);

    let two = image(
        MjpegPixelFormat::YuvJ420P,
        (w, h),
        (12, 8),
        |_, _| 90,
        |i, j| (20 + 8 * i + 4 * j) as u8,
        |_, _| 128,
    );
    let lin = rgb(&two, ChromaUpsampling::Linear);
    for y in 1..h as usize - 1 {
        for x in 1..w as usize - 1 {
            let cb = 4.0 * x as f64 + 2.0 * y as f64 + 17.0;
            assert_eq!(
                lin[y * w as usize + x],
                colour(90.0, cb, 128.0),
                "({x}, {y})"
            );
        }
    }
    assert_eq!(lin[0], colour(90.0, 20.0, 128.0));
    assert_eq!(rgb(&two, ChromaUpsampling::LumaGuided), lin);
}

/// Two flat colours side by side under textured luma: every mode
/// agrees wherever the 3 × 3 chroma neighbourhood is flat, i.e. away
/// from the boundary column.
#[test]
fn modes_agree_on_flat_areas() {
    for (format, (cw, ch), dh) in [
        (MjpegPixelFormat::YuvJ420P, (16, 8), 2),
        (MjpegPixelFormat::YuvJ422P, (16, 16), 2),
    ] {
        let (w, h) = (32usize, 16usize);
        let img = image(
            format,
            (w as u32, h as u32),
            (cw, ch),
            |x, y| ((x * 37 + y * 91) % 256) as u8,
            |i, _| if i < 8 { 60 } else { 190 },
            |i, _| if i < 8 { 200 } else { 70 },
        );
        let reference = rgb(&img, ChromaUpsampling::Replicate);
        for mode in MODES {
            let got = rgb(&img, mode);
            for y in 0..h {
                for x in 0..w {
                    let c = x / dh;
                    if (6..=9).contains(&c) {
                        continue;
                    }
                    let at = y * w + x;
                    assert_eq!(got[at], reference[at], "{format} {mode:?} ({x}, {y})");
                }
            }
            // And the modes do differ across the boundary.
            if mode != ChromaUpsampling::Replicate {
                assert_ne!(got, reference, "{format} {mode:?}");
            }
        }
    }
}

fn noise(n: usize, seed: &mut u32, max: u32) -> Vec<u32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            *seed % (max + 1)
        })
        .collect()
}

/// Pictures of every native family: the conversions that go through
/// the row kernels and the ones that do not.
fn assorted() -> Vec<JpegImage> {
    let mut seed = 0x2545_F491u32;
    let (w, h) = (37u32, 21u32);
    let bytes = |v: Vec<u32>, wide: bool| -> Vec<u8> {
        if wide {
            v.iter().flat_map(|&s| (s as u16).to_le_bytes()).collect()
        } else {
            v.iter().map(|&s| s as u8).collect()
        }
    };
    let mut out = Vec::new();
    for (format, dh, dv, wide) in [
        (MjpegPixelFormat::YuvJ420P, 2, 2, false),
        (MjpegPixelFormat::YuvJ422P, 2, 1, false),
        (MjpegPixelFormat::Yuv411P, 4, 1, false),
        (MjpegPixelFormat::YuvJ444P, 1, 1, false),
        (MjpegPixelFormat::Yuv420P12Le, 2, 2, true),
    ] {
        let (cw, ch) = ((w as usize).div_ceil(dh), (h as usize).div_ceil(dv));
        let (max, bps) = if wide { (4095, 2) } else { (255, 1) };
        let mut plane = |pw: usize, ph: usize| {
            Plane::new(pw * bps, bytes(noise(pw * ph, &mut seed, max), wide))
        };
        let planes = vec![plane(w as usize, h as usize), plane(cw, ch), plane(cw, ch)];
        let mut img = JpegImage::new(w, h, format, planes).expect("planar picture");
        if wide {
            img = img.with_precision(12);
        }
        out.push(img);
    }
    let n = (w * h) as usize;
    let packed = |format, channels: usize, seed: &mut u32| {
        let data = bytes(noise(n * channels, seed, 255), false);
        JpegImage::new(w, h, format, vec![Plane::new(w as usize * channels, data)])
            .expect("packed picture")
    };
    out.push(packed(MjpegPixelFormat::Gray8, 1, &mut seed));
    out.push(packed(MjpegPixelFormat::Rgb24, 3, &mut seed));
    out.push(packed(MjpegPixelFormat::Cmyk, 4, &mut seed));
    out
}

#[test]
fn packed_layouts_are_the_rgb_bytes_reordered() {
    for img in assorted() {
        for mode in MODES {
            for kernel in [YcbcrKernel::Exact, YcbcrKernel::FixedPoint16] {
                for threads in [1, 3] {
                    let opts = ConvertOptions::new()
                        .with_upsampling(mode)
                        .with_ycbcr(kernel)
                        .with_threads(threads);
                    let what = format!("{} {mode:?} {kernel:?} x{threads}", img.format());
                    let rgb = img.to_rgb8_with(&opts);
                    assert_eq!(img.to_packed_with(PackedLayout::Rgb8, &opts), rgb, "{what}");
                    let px = rgb.as_chunks::<3>().0;
                    let bgr: Vec<u8> = px.iter().flat_map(|p| [p[2], p[1], p[0]]).collect();
                    assert_eq!(img.to_packed_with(PackedLayout::Bgr8, &opts), bgr, "{what}");
                    let rgba: Vec<u8> = px.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect();
                    assert_eq!(
                        img.to_packed_with(PackedLayout::Rgba8, &opts),
                        rgba,
                        "{what}"
                    );
                    assert_eq!(img.to_rgba8_with(&opts), rgba, "{what}");
                    let bgra: Vec<u8> = px.iter().flat_map(|p| [p[2], p[1], p[0], 255]).collect();
                    assert_eq!(
                        img.to_packed_with(PackedLayout::Bgra8, &opts),
                        bgra,
                        "{what}"
                    );
                }
            }
        }
    }
}

#[test]
fn write_packed_fills_rows_at_the_callers_stride() {
    for img in assorted() {
        let (w, h) = (img.width() as usize, img.height() as usize);
        for layout in [
            PackedLayout::Rgb8,
            PackedLayout::Bgr8,
            PackedLayout::Rgba8,
            PackedLayout::Bgra8,
        ] {
            let bpp = layout.bytes_per_pixel();
            for threads in [1, 4] {
                let opts = ConvertOptions::new().with_threads(threads);
                let tight = img.to_packed_with(layout, &opts);
                let stride = w * bpp + 5;
                // The last row may stop at its last pixel.
                let mut dst = vec![0xA5u8; (h - 1) * stride + w * bpp];
                img.write_packed(layout, &mut dst, stride, &opts)
                    .expect("write_packed");
                for (y, row) in dst.chunks(stride).enumerate() {
                    assert_eq!(
                        row[..w * bpp],
                        tight[y * w * bpp..][..w * bpp],
                        "{} {layout:?} row {y}",
                        img.format()
                    );
                    assert!(
                        row[w * bpp..].iter().all(|&b| b == 0xA5),
                        "{} {layout:?}: padding of row {y} written",
                        img.format()
                    );
                }
                // A tight buffer is `to_packed_with`.
                let mut dst = vec![0u8; w * h * bpp];
                img.write_packed(layout, &mut dst, w * bpp, &opts)
                    .expect("tight write_packed");
                assert_eq!(dst, tight);
            }
            let mut short = vec![0u8; w * h * bpp - 1];
            assert!(
                img.write_packed(layout, &mut short, w * bpp, &ConvertOptions::new())
                    .is_err()
            );
            let mut any = vec![0u8; w * h * bpp];
            assert!(
                img.write_packed(layout, &mut any, w * bpp - 1, &ConvertOptions::new())
                    .is_err()
            );
        }
    }
}
