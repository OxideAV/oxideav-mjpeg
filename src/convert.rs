//! Native layout → packed 8-bit RGB / RGBA.
//!
//! The kernels behind [`JpegImage::to_rgb8`] / [`JpegImage::to_rgba8`].
//! They are deliberately simple, exact and fixed:
//!
//! * **YCbCr → RGB** is the T.871 §7 inverse relationship in its
//!   full-precision form,
//!   `R = Round(Y + 1.402 (CR − 128))`,
//!   `G = Round(Y − (0.114 · 1.772 (CB − 128) + 0.299 · 1.402 (CR − 128)) / 0.587)`,
//!   `B = Round(Y + 1.772 (CB − 128))`, with `Round(x) = ⌊x + 0.5⌋`
//!   and clamping to `0..=255`, evaluated in exact integer arithmetic
//!   (every coefficient scaled to an integer, one rounding at the end).
//!   The same equations serve the 12-bit layouts with `2048` in place of
//!   `128`, the result being rescaled to 8 bits in the same single
//!   rounding step. Full range is assumed for every YCbCr layout: T.871
//!   fixes it for JFIF and T.872 §6.1 extends the T.871 relationship to
//!   every three-component JPEG without an Adobe RGB flag.
//! * **Chroma upsampling** is nearest-neighbour sample replication:
//!   pixel `(x, y)` reads chroma sample `(x / h, y / v)` for divisors
//!   `(h, v)` — i.e. the inverse of the T.81 §A.1.1 component-dimension
//!   expressions. T.871 §9 sites subsampled chroma centred between luma
//!   samples; no interpolation filter is applied (the spec leaves the
//!   reconstruction filter to the decoder). This rule is fixed so a
//!   given stream always yields the same RGB bytes.
//! * **Grayscale** replicates the sample into R, G and B.
//! * **CMYK** (plain ink amounts, `(0, 0, 0, 0)` = white — the decoder
//!   has already undone the Adobe APP14 complement and the YCCK
//!   transform, T.872 §6.5.3 / §7) converts with the multiplicative
//!   convention `R = Round((255 − C) (255 − K) / 255)` and likewise for
//!   M → G, Y → B. T.872 §3.1 leaves ink values device dependent, so
//!   this is a documented convention, not a spec formula; colour-managed
//!   consumers should use the ICC profile instead.
//! * **Deep samples** (10 / 12 / 14 / 16-bit carriers) are rescaled to
//!   8 bits by `Round(v · 255 / (2^P − 1))`, `P` being the frame's
//!   sample precision ([`JpegImage::precision`]), so a lossless 9-bit
//!   frame carried in `Gray16Le` scales by `511`, not `65535`.
//! * **Planar GBR** (`Gbrp*Le`) is re-ordered to R, G, B and rescaled.
//!
//! Alpha is always `255`: JPEG has no alpha channel.

use crate::image::{JpegImage, MjpegPixelFormat as F};

/// Convert `img` to tightly packed 8-bit RGB (`3 × width × height`
/// bytes). See the module docs for the kernels.
pub fn to_rgb8(img: &JpegImage) -> Vec<u8> {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut out = vec![0u8; w * h * 3];
    fill_rgb(img, &mut out, 3);
    out
}

/// Convert `img` to tightly packed 8-bit RGBA with opaque alpha
/// (`4 × width × height` bytes).
pub fn to_rgba8(img: &JpegImage) -> Vec<u8> {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut out = vec![255u8; w * h * 4];
    fill_rgb(img, &mut out, 4);
    out
}

/// Fill the R, G, B bytes of every `bpp`-byte pixel of `out`.
fn fill_rgb(img: &JpegImage, out: &mut [u8], bpp: usize) {
    let w = img.width as usize;
    let h = img.height as usize;
    if w == 0 || h == 0 || img.planes.is_empty() {
        return;
    }
    let maxval = max_value(img);
    match img.format {
        F::Gray8 | F::Gray10Le | F::Gray12Le | F::Gray16Le => {
            let p = &img.planes[0];
            let bps = img.format.bytes_per_sample();
            for y in 0..h {
                for x in 0..w {
                    let v = scale8(sample(p, bps, x, y), maxval);
                    let o = (y * w + x) * bpp;
                    out[o] = v;
                    out[o + 1] = v;
                    out[o + 2] = v;
                }
            }
        }
        F::Rgb24 => {
            let p = &img.planes[0];
            for y in 0..h {
                for x in 0..w {
                    let i = y * p.stride + x * 3;
                    let o = (y * w + x) * bpp;
                    if let Some(px) = p.data.get(i..i + 3) {
                        out[o..o + 3].copy_from_slice(px);
                    }
                }
            }
        }
        F::Rgb48Le => {
            let p = &img.planes[0];
            for y in 0..h {
                for x in 0..w {
                    let o = (y * w + x) * bpp;
                    for c in 0..3 {
                        out[o + c] = scale8(sample(p, 2, x * 3 + c, y), maxval);
                    }
                }
            }
        }
        F::Gbrp10Le | F::Gbrp12Le | F::Gbrp14Le => {
            if img.planes.len() < 3 {
                return;
            }
            let (g, b, r) = (&img.planes[0], &img.planes[1], &img.planes[2]);
            for y in 0..h {
                for x in 0..w {
                    let o = (y * w + x) * bpp;
                    out[o] = scale8(sample(r, 2, x, y), maxval);
                    out[o + 1] = scale8(sample(g, 2, x, y), maxval);
                    out[o + 2] = scale8(sample(b, 2, x, y), maxval);
                }
            }
        }
        F::Cmyk => {
            let p = &img.planes[0];
            for y in 0..h {
                for x in 0..w {
                    let i = y * p.stride + x * 4;
                    let o = (y * w + x) * bpp;
                    if let Some(px) = p.data.get(i..i + 4) {
                        let k = 255 - px[3] as u32;
                        out[o] = cmyk_channel(px[0], k);
                        out[o + 1] = cmyk_channel(px[1], k);
                        out[o + 2] = cmyk_channel(px[2], k);
                    }
                }
            }
        }
        F::Yuv411P
        | F::Yuv420P
        | F::Yuv422P
        | F::Yuv444P
        | F::YuvJ420P
        | F::YuvJ422P
        | F::YuvJ444P
        | F::Yuv420P12Le
        | F::Yuv422P12Le
        | F::Yuv444P12Le => {
            if img.planes.len() < 3 {
                return;
            }
            let bps = img.format.bytes_per_sample();
            let (dh, dv) = img.format.chroma_divisors();
            let (py, pcb, pcr) = (&img.planes[0], &img.planes[1], &img.planes[2]);
            for y in 0..h {
                let cy = y / dv;
                for x in 0..w {
                    let cx = x / dh;
                    let yy = sample(py, bps, x, y);
                    let cb = sample(pcb, bps, cx, cy);
                    let cr = sample(pcr, bps, cx, cy);
                    let (r, g, b) = ycbcr_to_rgb8(yy, cb, cr, maxval);
                    let o = (y * w + x) * bpp;
                    out[o] = r;
                    out[o + 1] = g;
                    out[o + 2] = b;
                }
            }
        }
    }
}

/// `2^P − 1` for the frame's sample precision, clamped to the carrier
/// width so a mislabelled `precision` can never divide by zero or
/// exceed the storage.
fn max_value(img: &JpegImage) -> u32 {
    let carrier_bits = (img.format.bytes_per_sample() * 8) as u8;
    let p = img.precision.clamp(1, carrier_bits);
    (1u32 << p) - 1
}

/// Read sample `index` of row `y` from a 1- or 2-byte-per-sample plane;
/// `0` when the plane is short.
#[inline]
fn sample(p: &crate::image::Plane, bps: usize, index: usize, y: usize) -> u32 {
    let i = y * p.stride + index * bps;
    if bps == 1 {
        p.data.get(i).copied().unwrap_or(0) as u32
    } else {
        match p.data.get(i..i + 2) {
            Some(b) => u16::from_le_bytes([b[0], b[1]]) as u32,
            None => 0,
        }
    }
}

/// `Round(v · 255 / maxval)`, clamped.
#[inline]
fn scale8(v: u32, maxval: u32) -> u8 {
    if maxval == 255 {
        return v.min(255) as u8;
    }
    let v = v.min(maxval) as u64;
    ((v * 255 + (maxval as u64) / 2) / maxval as u64) as u8
}

/// `Round((255 − ink) · (255 − k) / 255)` where `k_inv = 255 − K`.
#[inline]
fn cmyk_channel(ink: u8, k_inv: u32) -> u8 {
    (((255 - ink as u32) * k_inv + 127) / 255) as u8
}

/// T.871 §7 inverse relationship (full-precision form), exact integer
/// arithmetic, one rounding, for samples in `0..=maxval` with the
/// chroma zero point at `(maxval + 1) / 2`; result rescaled to 8 bits
/// in the same rounding step.
///
/// With `D = maxval`, `half = (D + 1) / 2`, `cb' = CB − half`,
/// `cr' = CR − half`:
///
/// ```text
/// R = Round(255 (1000 Y + 1402 cr') / (1000 D))
/// G = Round(255 (587000 Y − 202008 cb' − 419198 cr') / (587000 D))
/// B = Round(255 (1000 Y + 1772 cb') / (1000 D))
/// ```
///
/// (`0.114 · 1.772 = 0.202008`, `0.299 · 1.402 = 0.419198`.) Every
/// intermediate fits comfortably in `i64`.
pub(crate) fn ycbcr_to_rgb8(y: u32, cb: u32, cr: u32, maxval: u32) -> (u8, u8, u8) {
    let d = maxval as i64;
    let half = (d + 1) / 2;
    let y = y as i64;
    let cb = cb as i64 - half;
    let cr = cr as i64 - half;
    let r = round_div(255 * (1000 * y + 1402 * cr), 1000 * d);
    let g = round_div(
        255 * (587_000 * y - 202_008 * cb - 419_198 * cr),
        587_000 * d,
    );
    let b = round_div(255 * (1000 * y + 1772 * cb), 1000 * d);
    (clamp8(r), clamp8(g), clamp8(b))
}

/// `⌊num / den + 0.5⌋` for `den > 0`, exact (floor division).
#[inline]
fn round_div(num: i64, den: i64) -> i64 {
    (2 * num + den).div_euclid(2 * den)
}

#[inline]
fn clamp8(v: i64) -> u8 {
    v.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{JpegImage, Plane};

    /// Reference evaluation of the T.871 §7 equations in `f64`, as
    /// written in the Recommendation (`Round(x) = floor(x + 0.5)`).
    fn reference(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
        let (y, cb, cr) = (y as f64, cb as f64 - 128.0, cr as f64 - 128.0);
        let round = |x: f64| (x + 0.5).floor().clamp(0.0, 255.0) as u8;
        (
            round(y + 1.402 * cr),
            round(y - (0.114 * 1.772 * cb + 0.299 * 1.402 * cr) / 0.587),
            round(y + 1.772 * cb),
        )
    }

    #[test]
    fn ycbcr_kernel_matches_t871_reference_on_a_dense_grid() {
        // Every Y against a chroma grid (step 3 keeps the test fast;
        // the kernel is linear so the grid is representative), plus
        // the extreme corners.
        let mut mismatches = 0;
        for y in 0..=255u32 {
            for cb in (0..=255u32).step_by(3).chain([255]) {
                for cr in (0..=255u32).step_by(3).chain([255]) {
                    let got = ycbcr_to_rgb8(y, cb, cr, 255);
                    let want = reference(y as u8, cb as u8, cr as u8);
                    if got != want {
                        mismatches += 1;
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "integer kernel diverges from the T.871 equations"
        );
    }

    #[test]
    fn ycbcr_primaries() {
        // T.871 §7 forward equations applied to pure red / green / blue
        // / white / black, then inverted: the inverse must land back on
        // the primaries (the forward step rounds, so allow ±1).
        let fwd = |r: f64, g: f64, b: f64| {
            let round = |x: f64| (x + 0.5).floor().clamp(0.0, 255.0) as u32;
            (
                round(0.299 * r + 0.587 * g + 0.114 * b),
                round((-0.299 * r - 0.587 * g + 0.886 * b) / 1.772 + 128.0),
                round((0.701 * r - 0.587 * g - 0.114 * b) / 1.402 + 128.0),
            )
        };
        for (r, g, b) in [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 255),
            (0, 0, 0),
            (128, 128, 128),
            (200, 30, 90),
        ] {
            let (y, cb, cr) = fwd(r as f64, g as f64, b as f64);
            let (rr, gg, bb) = ycbcr_to_rgb8(y, cb, cr, 255);
            assert!((rr as i32 - r).abs() <= 1, "R {r} → {rr}");
            assert!((gg as i32 - g).abs() <= 1, "G {g} → {gg}");
            assert!((bb as i32 - b).abs() <= 1, "B {b} → {bb}");
        }
        assert_eq!(ycbcr_to_rgb8(128, 128, 128, 255), (128, 128, 128));
    }

    #[test]
    fn twelve_bit_ycbcr_rescales() {
        // 12-bit grey (2048, 2048, 2048) → 8-bit 128 (2048·255/4095 =
        // 127.53 → 128); 12-bit white → 255; black → 0.
        assert_eq!(ycbcr_to_rgb8(2048, 2048, 2048, 4095), (128, 128, 128));
        assert_eq!(ycbcr_to_rgb8(4095, 2048, 2048, 4095), (255, 255, 255));
        assert_eq!(ycbcr_to_rgb8(0, 2048, 2048, 4095), (0, 0, 0));
    }

    #[test]
    fn scale8_rounds() {
        assert_eq!(scale8(1023, 1023), 255);
        assert_eq!(scale8(511, 1023), 127); // 511·255/1023 = 127.37
        assert_eq!(scale8(512, 1023), 128); // 127.62
        assert_eq!(scale8(0x0FFF, 0x0FFF), 255);
        assert_eq!(scale8(300, 255), 255);
        assert_eq!(scale8(65535, 65535), 255);
    }

    #[test]
    fn gray_and_rgb_layouts() {
        let g = JpegImage::new(2, 1, F::Gray8, vec![Plane::new(2, vec![7, 200])]).unwrap();
        assert_eq!(to_rgb8(&g), vec![7, 7, 7, 200, 200, 200]);
        assert_eq!(to_rgba8(&g), vec![7, 7, 7, 255, 200, 200, 200, 255]);

        // 10-bit grey in a 16-bit carrier: 1023 → 255, 512 → 128.
        let g10 = JpegImage::new(
            2,
            1,
            F::Gray10Le,
            vec![Plane::new(4, vec![0xFF, 0x03, 0x00, 0x02])],
        )
        .unwrap();
        assert_eq!(to_rgb8(&g10), vec![255, 255, 255, 128, 128, 128]);

        // 9-bit lossless grey carried in Gray16Le: 511 → 255.
        let g9 = JpegImage::new(1, 1, F::Gray16Le, vec![Plane::new(2, vec![0xFF, 0x01])])
            .unwrap()
            .with_precision(9);
        assert_eq!(to_rgb8(&g9), vec![255, 255, 255]);

        // Rgb24 with a padded stride copies the visible bytes.
        let rgb = JpegImage::new(
            1,
            2,
            F::Rgb24,
            vec![Plane::new(4, vec![1, 2, 3, 99, 4, 5, 6, 99])],
        )
        .unwrap();
        assert_eq!(to_rgb8(&rgb), vec![1, 2, 3, 4, 5, 6]);

        // Rgb48Le at P = 16: 0xFFFF → 255, 0x8000 → 128.
        let rgb48 = JpegImage::new(
            1,
            1,
            F::Rgb48Le,
            vec![Plane::new(6, vec![0xFF, 0xFF, 0x00, 0x80, 0x00, 0x00])],
        )
        .unwrap();
        assert_eq!(to_rgb8(&rgb48), vec![255, 128, 0]);

        // Gbrp12Le re-orders planes G, B, R → R, G, B.
        let gbrp = JpegImage::new(
            1,
            1,
            F::Gbrp12Le,
            vec![
                Plane::new(2, vec![0xFF, 0x0F]), // G = 4095
                Plane::new(2, vec![0x00, 0x00]), // B = 0
                Plane::new(2, vec![0x00, 0x08]), // R = 2048
            ],
        )
        .unwrap();
        assert_eq!(to_rgb8(&gbrp), vec![128, 255, 0]);
    }

    #[test]
    fn cmyk_convention() {
        let img = JpegImage::new(
            3,
            1,
            F::Cmyk,
            vec![Plane::new(12, vec![0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 255])],
        )
        .unwrap();
        // white, pure cyan (no red), full black
        assert_eq!(to_rgb8(&img), vec![255, 255, 255, 0, 255, 255, 0, 0, 0]);
        // Half ink, half key: (255 − 128)(255 − 128) / 255 = 63.25 → 63.
        let half =
            JpegImage::new(1, 1, F::Cmyk, vec![Plane::new(4, vec![128, 128, 128, 128])]).unwrap();
        assert_eq!(to_rgb8(&half), vec![63, 63, 63]);
    }

    #[test]
    fn yuv420_upsampling_is_nearest_neighbour() {
        // 3×2 picture, 4:2:0: chroma plane is 2×1. Grey luma with a
        // red-shifted Cr in the first chroma sample and neutral in the
        // second; pixels x = 0, 1 share chroma 0, pixel x = 2 reads
        // chroma 1, both rows read chroma row 0.
        let img = JpegImage::new(
            3,
            2,
            F::Yuv420P,
            vec![
                Plane::new(3, vec![128; 6]),
                Plane::new(2, vec![128, 128]),
                Plane::new(2, vec![200, 128]),
            ],
        )
        .unwrap();
        let rgb = to_rgb8(&img);
        let px = |x: usize, y: usize| &rgb[(y * 3 + x) * 3..(y * 3 + x) * 3 + 3];
        let red = ycbcr_to_rgb8(128, 128, 200, 255);
        assert_eq!(px(0, 0), &[red.0, red.1, red.2]);
        assert_eq!(px(1, 1), &[red.0, red.1, red.2]);
        assert_eq!(px(2, 0), &[128, 128, 128]);
        assert_eq!(px(2, 1), &[128, 128, 128]);
        // The J label converts identically.
        let j = JpegImage::new(3, 2, F::YuvJ420P, img.planes.clone()).unwrap();
        assert_eq!(to_rgb8(&j), rgb);
    }

    #[test]
    fn short_planes_do_not_panic() {
        // `JpegImage::new` refuses these; the kernels still stay
        // defensive for images assembled inside the crate.
        let img = JpegImage::new_unchecked(4, 4, F::Yuv444P, vec![Plane::new(4, vec![0; 4])]);
        assert_eq!(to_rgb8(&img).len(), 48);
        let img = JpegImage::new_unchecked(4, 4, F::Rgb24, vec![Plane::new(12, vec![1; 5])]);
        assert_eq!(to_rgba8(&img).len(), 64);
        let img = JpegImage::new_unchecked(0, 4, F::Gray8, vec![]);
        assert!(to_rgb8(&img).is_empty());
    }
}
