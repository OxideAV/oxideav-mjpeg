//! Reduced-size decode (`DecodeOptions::scale`): geometry, layout and
//! sample values of the 1/2, 1/4 and 1/8 reconstructions.

use oxideav_mjpeg::{
    ChromaSubsampling, DecodeOptions, DecodeScale, EncodeOptions, JpegImage, MjpegError,
    decode_rgb8_with, decode_with, encode_rgb8,
};

/// A smooth picture (low spatial frequencies only, mid-range levels):
/// what a reduced reconstruction and an averaged full one agree on.
fn smooth_picture(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let r = 128.0 + 70.0 * (fx * 3.1).sin() * (fy * 2.3).cos();
            let g = 128.0 + 60.0 * (fx * 1.7 + fy * 2.9).cos();
            let b = 128.0 + 50.0 * (fy * 3.7).sin();
            out.extend([r as u8, g as u8, b as u8]);
        }
    }
    out
}

const SCALES: [DecodeScale; 4] = [
    DecodeScale::Full,
    DecodeScale::Half,
    DecodeScale::Quarter,
    DecodeScale::Eighth,
];

fn decode_at(jpeg: &[u8], scale: DecodeScale) -> JpegImage {
    decode_with(jpeg, &DecodeOptions::new().with_scale(scale))
        .unwrap_or_else(|e| panic!("{scale:?}: {e:?}"))
}

/// Rows of a plane of an image: `(width, height)` of plane `i` follow
/// from its stride and length.
fn plane_dims(img: &JpegImage, i: usize) -> (usize, usize) {
    let p = &img.planes[i];
    (p.stride, p.data.len() / p.stride)
}

/// Mean absolute difference between a reduced plane and the `d × d`
/// cell means of the full plane, over the cells that lie wholly inside
/// the full plane; also the largest single difference.
fn against_cell_means(full: &JpegImage, small: &JpegImage, plane: usize, d: usize) -> (f64, f64) {
    let (fw, fh) = plane_dims(full, plane);
    let (sw, sh) = plane_dims(small, plane);
    let (f, s) = (&full.planes[plane].data, &small.planes[plane].data);
    let (mut sum, mut worst, mut n) = (0f64, 0f64, 0u32);
    for y in 0..sh {
        for x in 0..sw {
            if (x + 1) * d > fw || (y + 1) * d > fh {
                continue;
            }
            let mut acc = 0u32;
            for j in 0..d {
                for i in 0..d {
                    acc += f[(y * d + j) * fw + x * d + i] as u32;
                }
            }
            let mean = acc as f64 / (d * d) as f64;
            let diff = (s[y * sw + x] as f64 - mean).abs();
            sum += diff;
            worst = worst.max(diff);
            n += 1;
        }
    }
    assert!(n > 0, "plane {plane}: no whole cell");
    (sum / n as f64, worst)
}

#[test]
fn reduced_sizes_are_the_rounded_up_quotients() {
    for (w, h) in [
        (64u32, 48u32),
        (97, 61),
        (17, 9),
        (8, 8),
        (1, 1),
        (333, 219),
    ] {
        let rgb = smooth_picture(w, h);
        for chroma in [
            ChromaSubsampling::Yuv444,
            ChromaSubsampling::Yuv422,
            ChromaSubsampling::Yuv420,
            ChromaSubsampling::Yuv411,
            ChromaSubsampling::GrayscaleOnly,
        ] {
            for progressive in [false, true] {
                let opts = EncodeOptions::new()
                    .with_quality(85)
                    .with_chroma(chroma)
                    .with_progressive(progressive);
                let jpeg = encode_rgb8(w, h, &rgb, &opts).unwrap();
                let full = decode_at(&jpeg, DecodeScale::Full);
                for scale in SCALES {
                    let d = scale.denominator();
                    let img = decode_at(&jpeg, scale);
                    let tag = format!("{w}x{h} {chroma:?} progressive {progressive} 1/{d}");
                    assert_eq!(
                        (img.width, img.height),
                        (w.div_ceil(d), h.div_ceil(d)),
                        "{tag}"
                    );
                    assert_eq!(img.format, full.format, "{tag}");
                    assert_eq!(img.planes.len(), full.planes.len(), "{tag}");
                    assert_eq!(
                        plane_dims(&img, 0),
                        (img.width as usize, img.height as usize),
                        "{tag}"
                    );
                    // Converting works at every size.
                    let rgb8 =
                        decode_rgb8_with(&jpeg, &DecodeOptions::new().with_scale(scale)).unwrap();
                    assert_eq!(
                        rgb8.data.len(),
                        (img.width * img.height * 3) as usize,
                        "{tag}"
                    );
                }
            }
        }
    }
}

#[test]
fn full_scale_is_the_plain_decode() {
    let rgb = smooth_picture(120, 88);
    let jpeg = encode_rgb8(120, 88, &rgb, &EncodeOptions::new()).unwrap();
    assert_eq!(
        decode_at(&jpeg, DecodeScale::Full),
        decode_with(&jpeg, &DecodeOptions::new()).unwrap()
    );
    assert_eq!(DecodeOptions::new().scale, DecodeScale::Full);
}

#[test]
fn reduced_samples_follow_the_cell_means_of_the_full_picture() {
    let (w, h) = (256u32, 192u32);
    let rgb = smooth_picture(w, h);
    for chroma in [
        ChromaSubsampling::Yuv444,
        ChromaSubsampling::Yuv420,
        ChromaSubsampling::GrayscaleOnly,
    ] {
        for progressive in [false, true] {
            let opts = EncodeOptions::new()
                .with_quality(92)
                .with_chroma(chroma)
                .with_progressive(progressive);
            let jpeg = encode_rgb8(w, h, &rgb, &opts).unwrap();
            let full = decode_at(&jpeg, DecodeScale::Full);
            for scale in [DecodeScale::Half, DecodeScale::Quarter, DecodeScale::Eighth] {
                let d = scale.denominator() as usize;
                let small = decode_at(&jpeg, scale);
                for plane in 0..full.planes.len() {
                    let (mean, worst) = against_cell_means(&full, &small, plane, d);
                    let tag = format!("{chroma:?} progressive {progressive} 1/{d} plane {plane}");
                    // One sample per block is the block mean (to the
                    // rounding of the full samples); the 2- and 4-point
                    // forms sample the low-pass picture, within a level
                    // or so of the cell means on smooth content.
                    let (mean_cap, worst_cap) = if d == 8 { (0.3, 1.01) } else { (1.0, 4.0) };
                    assert!(mean < mean_cap, "{tag}: mean {mean}");
                    assert!(worst < worst_cap, "{tag}: worst {worst}");
                }
            }
        }
    }
}

#[test]
fn every_entropy_coder_and_scan_layout_reduces_alike() {
    // The reduced picture depends on the coefficients alone: Huffman
    // and arithmetic, interleaved and progressive codings of the same
    // quantised picture give the same samples.
    let (w, h) = (90u32, 70u32);
    let rgb = smooth_picture(w, h);
    let base = EncodeOptions::new()
        .with_quality(80)
        .with_chroma(ChromaSubsampling::Yuv420)
        .with_optimal_tables(false)
        .with_quantisation(oxideav_mjpeg::Quantisation::Nearest);
    let variants = [
        base.clone(),
        base.clone().with_progressive(true),
        base.clone().with_arithmetic(true),
        base.clone().with_arithmetic(true).with_progressive(true),
        base.clone().with_restart_interval(3),
    ];
    for scale in SCALES {
        let reference = decode_at(&encode_rgb8(w, h, &rgb, &variants[0]).unwrap(), scale);
        for (i, opts) in variants.iter().enumerate().skip(1) {
            let jpeg = encode_rgb8(w, h, &rgb, opts).unwrap();
            assert_eq!(decode_at(&jpeg, scale), reference, "variant {i} {scale:?}");
            // And for every worker count.
            for threads in [2, 5] {
                let img = decode_with(
                    &jpeg,
                    &DecodeOptions::new().with_scale(scale).with_threads(threads),
                )
                .unwrap();
                assert_eq!(img, reference, "variant {i} {scale:?} threads {threads}");
            }
        }
    }
}

#[test]
fn processes_without_dct_blocks_of_eight_bits_are_refused() {
    let (w, h) = (24u32, 16u32);
    let rgb = smooth_picture(w, h);
    let lossless = encode_rgb8(w, h, &rgb, &EncodeOptions::new().with_lossless(1)).unwrap();
    let luma: Vec<u16> = (0..w * h).map(|i| (i * 13 % 4096) as u16).collect();
    let twelve = EncodeOptions::new()
        .with_precision(12)
        .encode(w, h, &[&luma])
        .unwrap();
    for (name, jpeg) in [("lossless", &lossless), ("12-bit", &twelve.data)] {
        // Full size decodes; a reduced one is refused, not approximated.
        decode_at(jpeg, DecodeScale::Full);
        for scale in [DecodeScale::Half, DecodeScale::Eighth] {
            let err = decode_with(jpeg, &DecodeOptions::new().with_scale(scale))
                .expect_err("reduced decode must be refused");
            assert!(
                matches!(err, MjpegError::Unsupported(_)),
                "{name} {scale:?}: {err:?}"
            );
        }
    }
}

#[test]
fn scale_for_a_target_size() {
    use DecodeScale::*;
    assert_eq!(DecodeScale::for_target(4000, 3000, 500, 375), Eighth);
    assert_eq!(DecodeScale::for_target(4000, 3000, 501, 375), Quarter);
    assert_eq!(DecodeScale::for_target(4000, 3000, 1000, 750), Quarter);
    assert_eq!(DecodeScale::for_target(4000, 3000, 1001, 750), Half);
    assert_eq!(DecodeScale::for_target(4000, 3000, 2001, 100), Full);
    assert_eq!(DecodeScale::for_target(17, 9, 3, 2), Eighth);
    assert_eq!(DecodeScale::for_target(17, 9, 3, 3), Quarter);
    assert_eq!(Full.denominator(), 1);
    assert_eq!(Eighth.denominator(), 8);
}

fn jpegs_under(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            jpegs_under(&path, out);
        } else if path.extension().is_some_and(|e| e == "jpg" || e == "jpeg") {
            out.push(path);
        }
    }
}

#[test]
fn every_fixture_reduces_or_is_refused() {
    // Every sampling layout, component count and process of the fixture
    // set: a stream that decodes at full size either decodes at each
    // reduced size, to the rounded-up quotient of its dimensions in the
    // same layout, or is refused as unsupported — never an error of
    // another kind, never a panic.
    let mut files = Vec::new();
    jpegs_under(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        &mut files,
    );
    assert!(files.len() > 20, "fixture set missing: {}", files.len());
    let (mut reduced, mut refused) = (0, 0);
    for path in files {
        let jpeg = std::fs::read(&path).unwrap();
        let Ok(full) = decode_with(&jpeg, &DecodeOptions::new()) else {
            continue;
        };
        for scale in [DecodeScale::Half, DecodeScale::Quarter, DecodeScale::Eighth] {
            let d = scale.denominator();
            let opts = DecodeOptions::new().with_scale(scale);
            match decode_with(&jpeg, &opts) {
                Ok(img) => {
                    let tag = format!("{} 1/{d}", path.display());
                    assert_eq!(
                        (img.width, img.height),
                        (full.width.div_ceil(d), full.height.div_ceil(d)),
                        "{tag}"
                    );
                    assert_eq!(img.format, full.format, "{tag}");
                    assert_eq!(img.precision, 8, "{tag}");
                    let rgb = decode_rgb8_with(&jpeg, &opts).unwrap();
                    assert_eq!(
                        rgb.data.len(),
                        (img.width * img.height * 3) as usize,
                        "{tag}"
                    );
                    reduced += 1;
                }
                Err(MjpegError::Unsupported(_)) => refused += 1,
                Err(e) => panic!("{} 1/{d}: {e:?}", path.display()),
            }
        }
    }
    assert!(
        reduced > 60 && refused > 0,
        "{reduced} reduced / {refused} refused"
    );
}
