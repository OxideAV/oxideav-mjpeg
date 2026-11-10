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
//! * **Chroma upsampling** (planar subsampled layouts) is selected by
//!   [`ConvertOptions::upsampling`]; the default
//!   [`ChromaUpsampling::LumaGuided`] interpolates at the T.871 §9
//!   centred siting and restores luma-correlated detail
//!   (`crate::jpeg::resample`). The interpolated chroma enters the YCbCr
//!   equations with its fraction (to `1/256` of a sample), so there is
//!   still one rounding. [`ChromaUpsampling::Replicate`] reads chroma
//!   sample `(x / h, y / v)` for divisors `(h, v)`.
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
//!
//! [`to_rgb8_with`] / [`to_rgba8_with`] select other kernels for the
//! planar YCbCr layouts through [`ConvertOptions`], among them triangular
//! chroma upsampling (¾ / ¼ interpolation for the 2:1, 1:2 and 2 × 2
//! ratios) and the 16-bit fixed-point YCbCr kernel
//! ([`ycbcr_to_rgb8_fixed16`]); both are what `djpeg` applies by default.

use crate::image::{
    ChromaUpsampling, ConvertOptions, JpegImage, MjpegPixelFormat as F, YcbcrKernel,
};

/// Convert `img` to tightly packed 8-bit RGB (`3 × width × height`
/// bytes). See the module docs for the kernels.
pub fn to_rgb8(img: &JpegImage) -> Vec<u8> {
    to_rgb8_with(img, &ConvertOptions::new())
}

/// Convert `img` to tightly packed 8-bit RGBA with opaque alpha
/// (`4 × width × height` bytes).
pub fn to_rgba8(img: &JpegImage) -> Vec<u8> {
    to_rgba8_with(img, &ConvertOptions::new())
}

/// [`to_rgb8`] with explicit kernels.
pub fn to_rgb8_with(img: &JpegImage, opts: &ConvertOptions) -> Vec<u8> {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut out = vec![0u8; w * h * 3];
    fill_rgb(img, &mut out, 3, opts);
    out
}

/// [`to_rgba8`] with explicit kernels.
pub fn to_rgba8_with(img: &JpegImage, opts: &ConvertOptions) -> Vec<u8> {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut out = vec![255u8; w * h * 4];
    fill_rgb(img, &mut out, 4, opts);
    out
}

/// Fill the R, G, B bytes of every `bpp`-byte pixel of `out`.
fn fill_rgb(img: &JpegImage, out: &mut [u8], bpp: usize, opts: &ConvertOptions) {
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
            let replicate = opts.upsampling == ChromaUpsampling::Replicate || (dh, dv) == (1, 1);
            if !replicate {
                match opts.upsampling {
                    ChromaUpsampling::Triangular => fill_ycbcr_with(img, out, bpp, opts, maxval),
                    _ => fill_ycbcr_interp(img, out, bpp, opts, maxval),
                }
                return;
            }
            if opts.ycbcr != YcbcrKernel::Exact {
                fill_ycbcr_with(img, out, bpp, opts, maxval);
                return;
            }
            if bps == 1 && maxval == 255 && fill_ycbcr_replicate_8bit(img, out, bpp, opts.threads) {
                return;
            }
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

/// Pixels per step of the row kernels: the per-channel loops of
/// [`rgb_chunk`] run over fixed arrays of this many values.
const CHUNK: usize = 16;

/// `⌊(k · c + 128 000) / 256 000⌋` for a chroma offset `|c| ≤ 2^15` in
/// `1/256` sample units and `k ≤ 1772` — the R and B offsets of
/// [`ycbcr_frac_to_rgb8_8bit`] — in 32-bit arithmetic. `bias` makes the
/// numerator `x` non-negative (`bias · 256 000 ≥ k · 2^15`); then
/// `⌊x / 256 000⌋ = ⌊⌊x / 2^11⌋ / 125⌋`, and `⌊v / 125⌋` is
/// `(v · 33 555) >> 22` for `v < 59 074` (`33 555 · 125 = 2^22 + 71`,
/// so the product overshoots `v / 125` by `71 v / (125 · 2^22)`, less
/// than `1 / 125`), which
/// `v ≤ (1772 · 2^15 + 128 000 + 227 · 256 000) / 2^11 < 56 788` meets.
#[inline(always)]
fn chroma_offset<const K: i32, const BIAS: i32>(c: i32) -> i32 {
    let x = (K * c + 128_000 + BIAS * 256_000) as u32;
    (((x >> 11) * 33_555) >> 22) as i32 - BIAS
}

/// The G offset of [`ycbcr_frac_to_rgb8_8bit`],
/// `Round(−(202 008 cb + 419 198 cr) / 150 272 000)` for chroma offsets
/// `|cb|, |cr| ≤ 2^15` in `1/256` sample units, in 32-bit arithmetic.
///
/// Halving numerator and denominator, the value is `⌊m / 75 136 000⌋`
/// with `m = −101 004 cb − 209 599 cr + 37 568 000`, and
/// `75 136 000 = 2^10 · 73 375`, so it is `⌊u / 73 375⌋` with
/// `u = ⌊m / 2^10⌋`. Writing each coefficient as a multiple of `2^10`
/// plus a remainder (`−101 004 = −99 · 2^10 + 372`, `−209 599 = −205 ·
/// 2^10 + 321`, `37 568 000 = 36 687 · 2^10 + 512`) gives `u` without a
/// 64-bit product: `−99 cb − 205 cr + 36 687 + ⌊(372 cb + 321 cr + 512)
/// / 2^10⌋`. `|u| < 10^7`; adding `137 · 73 375` makes it positive
/// (`u' < 2.01 · 10^7`). The quotient estimate
/// `e = (⌊u' / 2^9⌋ · 29 267) >> 22` (`29 267 = ⌊2^31 / 73 375⌋`) never
/// exceeds `⌊u' / 73 375⌋` and falls short of it by less than one
/// (`< 1/143` from the shift, `< 273 / 29 267` from the multiplier), so
/// one comparison of the remainder completes it.
#[inline(always)]
fn green_offset(cb: i32, cr: i32) -> i32 {
    const BIAS: i32 = 137;
    let low = (372 * cb + 321 * cr + 512) >> 10;
    let u = (low - 99 * cb - 205 * cr + (36_687 + BIAS * 73_375)) as u32;
    let e = ((u >> 9) * 29_267) >> 22;
    let r = u - e * 73_375;
    (e + u32::from(r >= 73_375)) as i32 - BIAS
}

/// [`ycbcr_frac_to_rgb8_8bit`] for [`CHUNK`] pixels: `y` is the luma,
/// `cb` / `cr` the chroma offsets `cb'' = cb256 − 128 · 256`, already
/// limited to `±2^15`. Bit-identical to the scalar kernel, one channel
/// per loop, in 32-bit integers with no division.
#[inline(always)]
fn rgb_chunk(y: &[i32; CHUNK], cb: &[i32; CHUNK], cr: &[i32; CHUNK]) -> [[u8; CHUNK]; 3] {
    let c = |v: i32| v.clamp(0, 255) as u8;
    let mut out = [[0u8; CHUNK]; 3];
    for i in 0..CHUNK {
        out[0][i] = c(y[i] + chroma_offset::<1402, 180>(cr[i]));
    }
    for i in 0..CHUNK {
        out[1][i] = c(y[i] + green_offset(cb[i], cr[i]));
    }
    for i in 0..CHUNK {
        out[2][i] = c(y[i] + chroma_offset::<1772, 227>(cb[i]));
    }
    out
}

/// Interleave the three channels of [`rgb_chunk`] into `N`-byte pixels.
#[inline(always)]
fn put_chunk<const N: usize>(px: &mut [[u8; N]; CHUNK], rgb: &[[u8; CHUNK]; 3]) {
    for (i, p) in px.iter_mut().enumerate() {
        p[0] = rgb[0][i];
        p[1] = rgb[1][i];
        p[2] = rgb[2][i];
    }
}

/// One row of 8-bit `Y`, `Cb`, `Cr` (equal lengths) into `N`-byte
/// pixels with the exact kernel (`N = 4` leaves alpha as it is).
#[inline(never)]
fn row_ycbcr_8bit<const N: usize>(dst: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8]) {
    let (px, _) = dst.as_chunks_mut::<N>();
    let n = px.len().min(y.len()).min(cb.len()).min(cr.len());
    let (px, y, cb, cr) = (&mut px[..n], &y[..n], &cb[..n], &cr[..n]);
    let (yc, yt) = y.as_chunks::<CHUNK>();
    let (bc, bt) = cb.as_chunks::<CHUNK>();
    let (rc, rt) = cr.as_chunks::<CHUNK>();
    let (groups, tail) = px.as_chunks_mut::<CHUNK>();
    for (((p, y), b), r) in groups.iter_mut().zip(yc).zip(bc).zip(rc) {
        // The fractional kernel at a whole chroma sample: the exact
        // rounding of the same T.871 §7 value as `ycbcr_to_rgb8`.
        let yy: [i32; CHUNK] = std::array::from_fn(|i| y[i] as i32);
        let bb: [i32; CHUNK] = std::array::from_fn(|i| (b[i] as i32 - 128) << 8);
        let rr: [i32; CHUNK] = std::array::from_fn(|i| (r[i] as i32 - 128) << 8);
        put_chunk(p, &rgb_chunk(&yy, &bb, &rr));
    }
    for (((p, &yy), &b), &r) in tail.iter_mut().zip(yt).zip(bt).zip(rt) {
        let rgb = ycbcr_frac_to_rgb8_8bit(yy as i32, (b as i32) << 8, (r as i32) << 8);
        p[..3].copy_from_slice(&rgb);
    }
}

/// One row of 8-bit-range `Y` with fractional `Cb`, `Cr` (`f32`, equal
/// lengths) into `N`-byte pixels: the chroma rounded to `1/256` of a
/// sample and handed to the exact kernel, as
/// `ycbcr_frac_to_rgb8_8bit(y as i32, round(cb · 256), round(cr · 256))`.
#[inline(never)]
fn row_ycbcr_frac_8bit<const N: usize>(dst: &mut [u8], y: &[f32], cb: &[f32], cr: &[f32]) {
    let (px, _) = dst.as_chunks_mut::<N>();
    let n = px.len().min(y.len()).min(cb.len()).min(cr.len());
    let (px, y, cb, cr) = (&mut px[..n], &y[..n], &cb[..n], &cr[..n]);
    let (yc, yt) = y.as_chunks::<CHUNK>();
    let (bc, bt) = cb.as_chunks::<CHUNK>();
    let (rc, rt) = cr.as_chunks::<CHUNK>();
    use crate::jpeg::fround::round_clamped;
    // `(round(v · 256) − 128 · 256).clamp(−2^15, 2^15)`: limiting the
    // scaled value to `0..=2^16` before rounding limits the offset the
    // same way. The luma is a whole sample value (`0..=65 535`).
    let offset = |v: f32| round_clamped(v * 256.0, 65_536.0) - 128 * 256;
    let (groups, tail) = px.as_chunks_mut::<CHUNK>();
    for (((p, y), b), r) in groups.iter_mut().zip(yc).zip(bc).zip(rc) {
        let yy: [i32; CHUNK] = std::array::from_fn(|i| round_clamped(y[i], 65_535.0));
        let bb: [i32; CHUNK] = std::array::from_fn(|i| offset(b[i]));
        let rr: [i32; CHUNK] = std::array::from_fn(|i| offset(r[i]));
        put_chunk(p, &rgb_chunk(&yy, &bb, &rr));
    }
    for (((p, &yy), &b), &r) in tail.iter_mut().zip(yt).zip(bt).zip(rt) {
        let q = |v: f32| (v * 256.0).round() as i32;
        let rgb = ycbcr_frac_to_rgb8_8bit(yy as i32, q(b), q(r));
        p[..3].copy_from_slice(&rgb);
    }
}

/// The 8-bit planar YCbCr layouts with replicated chroma and the exact
/// kernel, row by row on slices (and in `threads` bands of rows).
/// Returns `false`, leaving `out` alone, when a plane is shorter than
/// its geometry (the per-sample path then reads what is there).
fn fill_ycbcr_replicate_8bit(img: &JpegImage, out: &mut [u8], bpp: usize, threads: usize) -> bool {
    let (w, h) = (img.width as usize, img.height as usize);
    let (dh, dv) = img.format.chroma_divisors();
    let (cw, ch) = (w.div_ceil(dh), h.div_ceil(dv));
    let (py, pcb, pcr) = (&img.planes[0], &img.planes[1], &img.planes[2]);
    let fits = |p: &crate::image::Plane, pw: usize, ph: usize| {
        p.stride >= pw && p.data.len() >= (ph - 1) * p.stride + pw
    };
    if !(fits(py, w, h) && fits(pcb, cw, ch) && fits(pcr, cw, ch)) || out.len() < w * h * bpp {
        return false;
    }
    let convert = |y0: usize, band: &mut [u8]| {
        // Chroma rows widened to the luma grid (replication).
        let mut wide = [vec![0u8; cw * dh], vec![0u8; cw * dh]];
        for (i, dst) in band.chunks_exact_mut(w * bpp).enumerate() {
            let y = y0 + i;
            let lrow = &py.data[y * py.stride..][..w];
            let cy = y / dv;
            let (cb_row, cr_row) = (
                &pcb.data[cy * pcb.stride..][..cw],
                &pcr.data[cy * pcr.stride..][..cw],
            );
            let (cb, cr): (&[u8], &[u8]) = if dh == 1 {
                (cb_row, cr_row)
            } else {
                for (src, buf) in [cb_row, cr_row].into_iter().zip(wide.iter_mut()) {
                    // Fixed-size cells for the common ratios, so the
                    // widening is a plain loop.
                    match dh {
                        2 => {
                            for (cell, &v) in buf.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                                *cell = [v; 2];
                            }
                        }
                        4 => {
                            for (cell, &v) in buf.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                                *cell = [v; 4];
                            }
                        }
                        _ => {
                            for (cell, &v) in buf.chunks_exact_mut(dh).zip(src) {
                                cell.fill(v);
                            }
                        }
                    }
                }
                (&wide[0][..w], &wide[1][..w])
            };
            if bpp == 3 {
                row_ycbcr_8bit::<3>(dst, lrow, cb, cr);
            } else {
                row_ycbcr_8bit::<4>(dst, lrow, cb, cr);
            }
        }
    };
    let out = &mut out[..w * h * bpp];
    let workers = threads.clamp(1, h);
    if workers <= 1 {
        convert(0, out);
    } else {
        let rows_per = h.div_ceil(workers);
        std::thread::scope(|sc| {
            let convert = &convert;
            for (i, band) in out.chunks_mut(rows_per * w * bpp).enumerate() {
                sc.spawn(move || convert(i * rows_per, band));
            }
        });
    }
    true
}

/// [`fill_rgb`] for the planar YCbCr layouts under non-default
/// [`ConvertOptions`]: chroma brought to the luma grid by the selected
/// upsampling (falling back to replication where the triangular kernel
/// does not apply), then the selected YCbCr kernel.
fn fill_ycbcr_with(
    img: &JpegImage,
    out: &mut [u8],
    bpp: usize,
    opts: &ConvertOptions,
    maxval: u32,
) {
    let w = img.width as usize;
    let h = img.height as usize;
    let bps = img.format.bytes_per_sample();
    let (dh, dv) = img.format.chroma_divisors();
    let (cw, ch) = (w.div_ceil(dh), h.div_ceil(dv));
    let full_chroma = |p: &crate::image::Plane| -> Vec<u32> {
        let get = |x: usize, y: usize| sample(p, bps, x, y);
        let tri = match opts.upsampling {
            ChromaUpsampling::Triangular => {
                let comp = crate::jpeg::upsample::Component {
                    width: cw,
                    height: ch,
                    get: &get,
                };
                crate::jpeg::upsample::triangular(&comp, dh, dv, w, h)
            }
            _ => None,
        };
        tri.unwrap_or_else(|| {
            let mut o = vec![0u32; w * h];
            for y in 0..h {
                for x in 0..w {
                    o[y * w + x] = get(x / dh, y / dv);
                }
            }
            o
        })
    };
    let (cb, cr) = (full_chroma(&img.planes[1]), full_chroma(&img.planes[2]));
    let fixed = opts.ycbcr == YcbcrKernel::FixedPoint16 && maxval == 255;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let yy = sample(&img.planes[0], bps, x, y);
            let (r, g, b) = if fixed {
                ycbcr_to_rgb8_fixed16(yy, cb[i], cr[i])
            } else {
                ycbcr_to_rgb8(yy, cb[i], cr[i], maxval)
            };
            let o = i * bpp;
            out[o] = r;
            out[o + 1] = g;
            out[o + 2] = b;
        }
    }
}

/// The planar YCbCr layouts under [`ChromaUpsampling::Linear`] /
/// [`ChromaUpsampling::LumaGuided`]: chroma reconstructed row by row in
/// `f32` (`crate::jpeg::resample`) and handed to the YCbCr kernel with
/// its fraction (to `1/256` of a sample), so the result is rounded once.
fn fill_ycbcr_interp(
    img: &JpegImage,
    out: &mut [u8],
    bpp: usize,
    opts: &ConvertOptions,
    maxval: u32,
) {
    use crate::jpeg::resample::{GuideLuma, Guided, Linear};
    let w = img.width as usize;
    let h = img.height as usize;
    let bps = img.format.bytes_per_sample();
    let (dh, dv) = img.format.chroma_divisors();
    let (cw, ch) = (w.div_ceil(dh), h.div_ceil(dv));
    let read_row = |p: &crate::image::Plane, y: usize, row: &mut [f32]| {
        let start = y * p.stride;
        match (bps, p.data.get(start..start + row.len() * bps)) {
            (1, Some(src)) => {
                for (d, &s) in row.iter_mut().zip(src) {
                    *d = s as f32;
                }
            }
            (_, Some(src)) => {
                for (d, s) in row.iter_mut().zip(src.as_chunks::<2>().0) {
                    *d = u16::from_le_bytes([s[0], s[1]]) as f32;
                }
            }
            _ => {
                for (x, d) in row.iter_mut().enumerate() {
                    *d = sample(p, bps, x, y) as f32;
                }
            }
        }
    };
    let plane = |p: &crate::image::Plane, pw: usize, ph: usize| -> Vec<f32> {
        let mut v = vec![0f32; pw * ph];
        for (y, row) in v.chunks_exact_mut(pw).enumerate() {
            read_row(p, y, row);
        }
        v
    };
    enum Recon {
        Lin(Linear),
        Guide(Guided),
    }
    // The luma statistics of the guided reconstruction serve both chroma
    // components.
    let guide = (opts.upsampling == ChromaUpsampling::LumaGuided).then(|| {
        let py = &img.planes[0];
        let whole = py.stride >= w && py.data.len() >= (h - 1) * py.stride + w;
        if bps == 1 && maxval == 255 && whole {
            GuideLuma::from_u8(&py.data, py.stride, cw, ch, dh, dv, w, h)
        } else {
            let mut luma = |y: usize, row: &mut [f32]| read_row(py, y, row);
            GuideLuma::new(&mut luma, cw, ch, dh, dv, w, h, maxval)
        }
    });
    let make = |p: &crate::image::Plane| {
        let c = plane(p, cw, ch);
        match &guide {
            Some(g) => Recon::Guide(Guided::new(g, c)),
            None => Recon::Lin(Linear::new(c, cw, ch, (1, dh), (1, dv), w, h)),
        }
    };
    let workers = opts.threads.clamp(1, h);
    // The two chroma planes are prepared independently.
    let recon = if workers > 1 {
        std::thread::scope(|sc| {
            let cr = sc.spawn(|| make(&img.planes[2]));
            let cb = make(&img.planes[1]);
            [cb, cr.join().expect("chroma worker")]
        })
    } else {
        [make(&img.planes[1]), make(&img.planes[2])]
    };
    let fixed = opts.ycbcr == YcbcrKernel::FixedPoint16 && maxval == 255;
    // Rows `y0..` of the picture into `band`: every row is a function of
    // the shared reconstruction state and its own luma row, so bands can
    // be converted by separate workers.
    let convert = |y0: usize, band: &mut [u8]| {
        let mut rows = [vec![0f32; w], vec![0f32; w]];
        let mut lrow = vec![0f32; w];
        let mut lin = [Vec::new(), Vec::new()];
        let mut gui: [[Vec<f32>; 2]; 2] = Default::default();
        for (i, r) in recon.iter().enumerate() {
            match (r, &guide) {
                (Recon::Guide(g), Some(gl)) => gui[i] = g.scratch(gl),
                (Recon::Lin(l), _) => lin[i] = l.scratch(),
                _ => {}
            }
        }
        let mut ycol = guide.as_ref().map(GuideLuma::scratch).unwrap_or_default();
        let mut detail = vec![0f32; if guide.is_some() { w } else { 0 }];
        for (i, dst) in band.chunks_exact_mut(w * bpp).enumerate() {
            let y = y0 + i;
            read_row(&img.planes[0], y, &mut lrow);
            if let Some(gl) = &guide {
                gl.detail_row(&mut ycol, y, &lrow, &mut detail);
            }
            for (i, (r, row)) in recon.iter().zip(rows.iter_mut()).enumerate() {
                match (r, &guide) {
                    (Recon::Guide(g), Some(gl)) => g.row_with(gl, &mut gui[i], y, &detail, row),
                    (Recon::Lin(l), _) => l.row_with(&mut lin[i], y, row),
                    _ => {}
                }
            }
            let (l, cb, cr) = (&lrow[..], &rows[0][..], &rows[1][..]);
            if fixed {
                let k = |yy: f32, cb: f32, cr: f32| {
                    let q = |v: f32| v.round().clamp(0.0, 255.0) as u32;
                    let (r, g, b) = ycbcr_to_rgb8_fixed16(yy as u32, q(cb), q(cr));
                    [r, g, b]
                };
                store_row_any(bpp, dst, l, cb, cr, k);
            } else if maxval == 255 {
                if bpp == 3 {
                    row_ycbcr_frac_8bit::<3>(dst, l, cb, cr);
                } else {
                    row_ycbcr_frac_8bit::<4>(dst, l, cb, cr);
                }
            } else {
                let k = |yy: f32, cb: f32, cr: f32| {
                    let q = |v: f32| (v * 256.0).round() as i64;
                    let (r, g, b) = ycbcr_frac_to_rgb8(yy as u32, q(cb), q(cr), maxval);
                    [r, g, b]
                };
                store_row_any(bpp, dst, l, cb, cr, k);
            }
        }
    };
    let n = (w * bpp * h).min(out.len());
    let out = &mut out[..n];
    if workers <= 1 {
        convert(0, out);
    } else {
        let rows_per = h.div_ceil(workers);
        std::thread::scope(|sc| {
            let convert = &convert;
            for (i, band) in out.chunks_mut(rows_per * w * bpp).enumerate() {
                sc.spawn(move || convert(i * rows_per, band));
            }
        });
    }
}

/// [`store_row`] for a runtime pixel size (`3` or `4`).
#[inline]
fn store_row_any(
    bpp: usize,
    dst: &mut [u8],
    y: &[f32],
    cb: &[f32],
    cr: &[f32],
    kernel: impl Fn(f32, f32, f32) -> [u8; 3],
) {
    if bpp == 3 {
        store_row::<3>(dst, y, cb, cr, kernel)
    } else {
        store_row::<4>(dst, y, cb, cr, kernel)
    }
}

/// Convert one row of `Y`, `Cb`, `Cr` into `N`-byte pixels (`N = 3` RGB,
/// `N = 4` RGBA, whose alpha is left as it is).
#[inline]
fn store_row<const N: usize>(
    dst: &mut [u8],
    y: &[f32],
    cb: &[f32],
    cr: &[f32],
    kernel: impl Fn(f32, f32, f32) -> [u8; 3],
) {
    let (px, _) = dst.as_chunks_mut::<N>();
    for (((p, &yy), &b), &r) in px.iter_mut().zip(y).zip(cb).zip(cr) {
        let [r, g, b] = kernel(yy, b, r);
        p[0] = r;
        p[1] = g;
        p[2] = b;
    }
}

/// [`ycbcr_frac_to_rgb8`] at `D = 255`: `Y` is an integer, so
/// `Round(Y + t) = Y + Round(t)` and only the chroma offsets are divided
/// (R and B as unsigned divisions by a constant after a bias that makes
/// the numerator non-negative, G in exact `f64`). Bit-identical to the
/// general form.
#[inline]
pub(crate) fn ycbcr_frac_to_rgb8_8bit(y: i32, cb256: i32, cr256: i32) -> [u8; 3] {
    // cb'' / cr'' are within ±2^15 (chroma clamped by the reconstruction
    // to the sample range, then scaled by 256).
    let cb = (cb256 - 128 * 256).clamp(-(1 << 15), 1 << 15);
    let cr = (cr256 - 128 * 256).clamp(-(1 << 15), 1 << 15);
    // Round(n / d) = ⌊(n + d/2 + K·d) / d⌋ − K with K·d > |n|; the R and
    // B numerators stay below 2^31, G needs 64 bits.
    const K: i32 = 256;
    let rr = ((1402 * cr + 128_000 + K * 256_000) as u32 / 256_000) as i32 - K;
    let bb = ((1772 * cb + 128_000 + K * 256_000) as u32 / 256_000) as i32 - K;
    // G in f64: the numerator (|n| < 2^35) and the denominator are exact,
    // the quotient is correctly rounded, and a quotient that is not a
    // half-integer lies at least 1 / (2 · 150 272 000) from one — far
    // beyond f64's resolution at |q| < 2^8 — so ⌊q + ½⌋ is exact.
    let n = -202_008.0 * cb as f64 - 419_198.0 * cr as f64;
    let gg = (n / 150_272_000.0 + 0.5).floor() as i32;
    let c = |v: i32| v.clamp(0, 255) as u8;
    [c(y + rr), c(y + gg), c(y + bb)]
}

/// [`ycbcr_to_rgb8`] with the chroma given in `1/256` sample units
/// (`cb256 = round(CB · 256)`), exact, one rounding:
///
/// ```text
/// R = Round(255 (256000 Y + 1402 cr'') / (256000 D))
/// G = Round(255 (150272000 Y − 202008 cb'' − 419198 cr'') / (150272000 D))
/// B = Round(255 (256000 Y + 1772 cb'') / (256000 D))
/// ```
///
/// with `cb'' = cb256 − 256 · half`. At `D = 255` the outer `255`
/// cancels and the denominators are constants.
#[inline]
pub(crate) fn ycbcr_frac_to_rgb8(y: u32, cb256: i64, cr256: i64, maxval: u32) -> (u8, u8, u8) {
    let y = y as i64;
    let d = maxval as i64;
    let half = (d + 1) / 2;
    let cb = cb256 - half * 256;
    let cr = cr256 - half * 256;
    let r = round_div(255 * (256_000 * y + 1402 * cr), 256_000 * d);
    let g = round_div(
        255 * (150_272_000 * y - 202_008 * cb - 419_198 * cr),
        150_272_000 * d,
    );
    let b = round_div(255 * (256_000 * y + 1772 * cb), 256_000 * d);
    (clamp8(r), clamp8(g), clamp8(b))
}

/// `round(1.402 · 2^16)` — the T.871 §7 Cr → R coefficient.
const CR_TO_R: i64 = 91_881;
/// Cb → G (magnitude; it is subtracted): `2 · 11 277`, the T.871 §7
/// coefficient `0.114 · 1.772 / 0.587 = 0.344136…` held to 15-bit
/// resolution (`11 277 / 2^15 = 0.344146`). Measured, not derived: over
/// every `(Cb, Cr)` pair, `djpeg`'s G offset is reproduced by exactly one
/// constant pair within ±4 of the nominal values at each scale 2^15 …
/// 2^18 (this one, scaled) and by none at 2^12 … 2^14 or 2^19 … 2^20;
/// the 16-bit-nearest full-precision pair `(22 553, 46 802)` misses 59
/// pairs (`notes/reference-compat.md`).
const CB_TO_G: i64 = 22_554;
/// Cr → G (magnitude; subtracted): `2 · 23 401`, `0.714136…` at the
/// same 15-bit resolution (`0.714142`); measured with [`CB_TO_G`].
const CR_TO_G: i64 = 46_802;
/// `round(1.772 · 2^16)` — the T.871 §7 Cb → B coefficient.
const CB_TO_B: i64 = 116_130;
/// Half of the `2^16` scale: round-half-up bias of every chroma product.
const HALF_16: i64 = 1 << 15;

/// T.871 §7 inverse relationship in 16-bit fixed point for 8-bit samples:
/// each chroma contribution `(k · (C − 128) + 2^15) >> 16` (the two G
/// products summed before the shift) is added to `Y`, then clamped to
/// `0..=255`. Bit-exact with `djpeg`'s default colour conversion,
/// measured black-box over every `(Cb, Cr)` pair at `Y = 0` and
/// `Y = 255` (the chroma offsets do not depend on `Y`) and on a 4:4:4
/// noise image under `djpeg -nosmooth`.
pub(crate) fn ycbcr_to_rgb8_fixed16(y: u32, cb: u32, cr: u32) -> (u8, u8, u8) {
    let y = y as i64;
    let cb = cb as i64 - 128;
    let cr = cr as i64 - 128;
    let r = y + ((CR_TO_R * cr + HALF_16) >> 16);
    let g = y + ((-CB_TO_G * cb - CR_TO_G * cr + HALF_16) >> 16);
    let b = y + ((CB_TO_B * cb + HALF_16) >> 16);
    (clamp8(r), clamp8(g), clamp8(b))
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
    fn yuv420_replicate_upsampling_is_nearest_neighbour() {
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
        let rep = ConvertOptions::new().with_upsampling(ChromaUpsampling::Replicate);
        let rgb = to_rgb8_with(&img, &rep);
        let px = |x: usize, y: usize| &rgb[(y * 3 + x) * 3..(y * 3 + x) * 3 + 3];
        let red = ycbcr_to_rgb8(128, 128, 200, 255);
        assert_eq!(px(0, 0), &[red.0, red.1, red.2]);
        assert_eq!(px(1, 1), &[red.0, red.1, red.2]);
        assert_eq!(px(2, 0), &[128, 128, 128]);
        assert_eq!(px(2, 1), &[128, 128, 128]);
        // The J label converts identically.
        let j = JpegImage::new(3, 2, F::YuvJ420P, img.planes.clone()).unwrap();
        assert_eq!(to_rgb8_with(&j, &rep), rgb);
    }

    #[test]
    fn default_upsampling_interpolates_with_centred_siting() {
        // 4 × 1 picture, 4:2:2, flat luma: chroma samples 100 and 200
        // sit at x = 0.5 and x = 2.5. Linear interpolation gives 100,
        // 125, 175, 200 (edges replicated); flat luma leaves the
        // luma-guided default on the linear estimate.
        let img = JpegImage::new(
            4,
            1,
            F::YuvJ422P,
            vec![
                Plane::new(4, vec![128; 4]),
                Plane::new(2, vec![100, 200]),
                Plane::new(2, vec![128, 128]),
            ],
        )
        .unwrap();
        let want: Vec<u8> = [100.0f32, 125.0, 175.0, 200.0]
            .iter()
            .flat_map(|&cb| {
                let (r, g, b) = ycbcr_frac_to_rgb8(128, (cb * 256.0) as i64, 128 * 256, 255);
                [r, g, b]
            })
            .collect();
        assert_eq!(to_rgb8(&img), want);
        let lin = ConvertOptions::new().with_upsampling(ChromaUpsampling::Linear);
        assert_eq!(to_rgb8_with(&img, &lin), want);
    }

    /// The row converter of the replicating path uses the fractional
    /// 8-bit kernel at whole chroma samples; it must be the integer
    /// kernel for every `(Cb, Cr)` pair (each over a spread of `Y`).
    #[test]
    fn whole_sample_fast_kernel_is_the_integer_kernel() {
        for cb in 0..=255u32 {
            for cr in 0..=255u32 {
                for y in [0u32, 1, 2, 17, 63, 64, 127, 128, 129, 200, 253, 254, 255] {
                    let (r, g, b) = ycbcr_to_rgb8(y, cb, cr, 255);
                    assert_eq!(
                        ycbcr_frac_to_rgb8_8bit(y as i32, (cb as i32) << 8, (cr as i32) << 8),
                        [r, g, b],
                        "y {y} cb {cb} cr {cr}"
                    );
                }
            }
        }
    }

    /// `rgb_chunk` against the scalar kernel for one `(cb'', cr'')`
    /// pair at a spread of luma values.
    fn assert_chunk(cb: i32, cr: i32) {
        let y: [i32; CHUNK] = std::array::from_fn(|i| [0, 1, 127, 128, 254, 255, 90, 200][i % 8]);
        let got = rgb_chunk(&y, &[cb; CHUNK], &[cr; CHUNK]);
        for i in 0..8 {
            let want = ycbcr_frac_to_rgb8_8bit(y[i], cb + 128 * 256, cr + 128 * 256);
            assert_eq!([got[0][i], got[1][i], got[2][i]], want, "cb {cb} cr {cr}");
        }
    }

    #[test]
    fn chunk_kernel_is_the_scalar_kernel() {
        const LIM: i32 = 1 << 15;
        // A coprime-stride lattice over the whole ±2^15 square, the
        // square's border, and every whole-sample pair.
        let (mut cb, mut cr) = (-LIM, -LIM);
        for _ in 0..600_000 {
            assert_chunk(cb, cr);
            cb += 7919;
            if cb > LIM {
                cb -= 2 * LIM + 1;
            }
            cr += 104_729 % (2 * LIM + 1);
            if cr > LIM {
                cr -= 2 * LIM + 1;
            }
        }
        for v in (-LIM..=LIM).step_by(3) {
            for edge in [-LIM, LIM] {
                assert_chunk(v, edge);
                assert_chunk(edge, v);
            }
        }
        for cb in -128..128 {
            for cr in -128..128 {
                assert_chunk(cb << 8, cr << 8);
            }
        }
        // The G quotients that are exact half-integers:
        // 101004 cb'' + 209599 cr'' = 37568000 (2k + 1).
        let mut ties = 0;
        for k in -136i64..136 {
            let target = 37_568_000 * (2 * k + 1);
            for cb in -LIM as i64..=LIM as i64 {
                let rest = target - 101_004 * cb;
                if rest % 209_599 == 0 && (rest / 209_599).abs() <= LIM as i64 {
                    let cr = rest / 209_599;
                    assert_chunk(cb as i32, cr as i32);
                    assert_chunk(-cb as i32, -cr as i32);
                    ties += 1;
                }
            }
        }
        assert!(ties > 0, "no exact tie exercised");
    }

    #[test]
    fn green_offset_is_the_rounded_quotient() {
        const LIM: i64 = 1 << 15;
        let exact = |cb: i64, cr: i64| {
            let n = -202_008 * cb - 419_198 * cr;
            (2 * n + 150_272_000).div_euclid(2 * 150_272_000) as i32
        };
        // Every pair on a lattice of coprime strides (the full square is
        // 2^34 pairs), the borders, and the quotients next to a step.
        for cb in (-LIM..=LIM).step_by(61) {
            for cr in (-LIM..=LIM).step_by(67) {
                assert_eq!(
                    green_offset(cb as i32, cr as i32),
                    exact(cb, cr),
                    "{cb} {cr}"
                );
            }
        }
        for v in -LIM..=LIM {
            for e in [-LIM, 0, LIM] {
                assert_eq!(green_offset(v as i32, e as i32), exact(v, e));
                assert_eq!(green_offset(e as i32, v as i32), exact(e, v));
            }
        }
        // For each cb, the cr values around every step of the quotient.
        for cb in (-LIM..=LIM).step_by(257) {
            for k in -136i64..=136 {
                // 419198 cr ≈ −202008 cb − (2k + 1) · 75136000
                let cr = (-202_008 * cb - (2 * k + 1) * 75_136_000) / 419_198;
                for cr in cr - 2..=cr + 2 {
                    if cr.abs() <= LIM {
                        assert_eq!(green_offset(cb as i32, cr as i32), exact(cb, cr));
                    }
                }
            }
        }
    }

    #[test]
    fn chroma_offsets_are_the_rounded_quotients() {
        for c in -(1 << 15)..=(1 << 15) {
            let exact = |k: i64| (k * c as i64 + 128_000).div_euclid(256_000) as i32;
            assert_eq!(chroma_offset::<1402, 180>(c), exact(1402), "R at {c}");
            assert_eq!(chroma_offset::<1772, 227>(c), exact(1772), "B at {c}");
        }
        for v in 0..59_074u32 {
            assert_eq!((v * 33_555) >> 22, v / 125);
        }
    }

    #[test]
    fn row_kernels_match_the_scalar_kernel() {
        let mut seed = 0x9E37_79B9u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for n in [1usize, 15, 16, 17, 40, 257] {
            let y: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let cb: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let cr: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut got = vec![7u8; n * 4];
            row_ycbcr_8bit::<4>(&mut got, &y, &cb, &cr);
            for i in 0..n {
                let (r, g, b) = ycbcr_to_rgb8(y[i] as u32, cb[i] as u32, cr[i] as u32, 255);
                assert_eq!(got[i * 4..i * 4 + 4], [r, g, b, 7]);
            }
            let yf: Vec<f32> = y.iter().map(|&v| v as f32).collect();
            let frac = |v: &[u8], next: &mut dyn FnMut() -> u32| -> Vec<f32> {
                v.iter()
                    .map(|&v| (v as f32 + (next() % 4096) as f32 / 4096.0).min(255.0))
                    .collect()
            };
            let (bf, rf) = (frac(&cb, &mut next), frac(&cr, &mut next));
            let mut got = vec![0u8; n * 3];
            row_ycbcr_frac_8bit::<3>(&mut got, &yf, &bf, &rf);
            for i in 0..n {
                let q = |v: f32| (v * 256.0).round() as i32;
                let want = ycbcr_frac_to_rgb8_8bit(y[i] as i32, q(bf[i]), q(rf[i]));
                assert_eq!(got[i * 3..i * 3 + 3], want);
            }
        }
    }

    #[test]
    fn fractional_chroma_kernel_agrees_with_the_integer_one() {
        for y in (0..=255u32).step_by(5) {
            for cb in (0..=255u32).step_by(7) {
                for cr in (0..=255u32).step_by(11) {
                    let f = |v: u32| v as i64 * 256;
                    assert_eq!(
                        ycbcr_frac_to_rgb8(y, f(cb), f(cr), 255),
                        ycbcr_to_rgb8(y, cb, cr, 255)
                    );
                    // The 8-bit fast form, at every fraction step.
                    for frac in [0i64, 1, 77, 128, 255] {
                        let (a, b) = (f(cb) + frac, f(cr) + 255 - frac);
                        let (r, g, bb) = ycbcr_frac_to_rgb8(y, a, b, 255);
                        assert_eq!(
                            ycbcr_frac_to_rgb8_8bit(y as i32, a as i32, b as i32),
                            [r, g, bb]
                        );
                    }
                    let g = |v: u32| v as i64 * 16 * 256;
                    assert_eq!(
                        ycbcr_frac_to_rgb8(y * 16, g(cb), g(cr), 4095),
                        ycbcr_to_rgb8(y * 16, cb * 16, cr * 16, 4095)
                    );
                }
            }
        }
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
