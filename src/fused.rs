//! 8-bit planar YCbCr → packed pixels in one pass per row.
//!
//! [`fill`] converts the 8-bit planar YCbCr layouts (4:4:4, 4:2:2,
//! 4:2:0, 4:1:1) straight into the caller's pixel layout: every output
//! row is produced from the chroma rows it depends on and its luma row,
//! through row-sized scratch only — no full-resolution chroma plane is
//! ever built — and stored as `R G B`, `B G R`, `R G B A` or `B G R A`
//! bytes. The bytes are those of the plane-at-a-time definitions in
//! `crate::convert`; what changes is the arithmetic that reaches them.
//!
//! # Chroma at an output pixel
//!
//! T.871 §9 sites a 2:1 subsampled chroma sample at the centre of the
//! two luma samples it covers, so along such an axis output `0` lies a
//! quarter sample outside chroma sample `0`, and outputs `2b + 1`,
//! `2b + 2` lie a quarter and three quarters of the way from sample `b`
//! to sample `b + 1`:
//!
//! ```text
//! s(0)      = c[0]
//! s(2b + 1) = (3 c[b] + c[b + 1]) / 4
//! s(2b + 2) = (c[b] + 3 c[b + 1]) / 4
//! s(2n − 1) = c[n − 1]                  (n samples, even output length)
//! ```
//!
//! [`widen_3_1`] forms `4 s` in integers. With both axes subsampled
//! (4:2:0) the same rule is applied to the rows first, so a pixel's
//! chroma is `(9 a + 3 b + 3 c + d) / 16` of its four nearest samples.
//!
//! * `Replicate` reads `c[x / 2]` instead: the colour offsets are a
//!   function of the chroma sample alone, so they are computed once per
//!   chroma sample and added to the two (4:2:2) or four (4:2:0) luma
//!   samples it covers.
//! * `Triangular` rounds the weighted sum to a whole sample — `+ 8` on
//!   even and `+ 7` on odd output columns before `>> 4` for 4:2:0,
//!   `+ 1` / `+ 2` before `>> 2` for one axis — and converts that.
//! * `Linear` keeps the sum as it is, in sixteenths of a sample, and
//!   converts that ([`frac16_offsets`]); at the 4:1 ratio of 4:1:1 the
//!   weights are eighths ([`widen_4_to_1`]).
//! * `LumaGuided` streams its rows from `crate::jpeg::resample` in
//!   blocks of sixteen pixels that go from the reconstruction to the
//!   colour equations without touching memory.
//!
//! # Colour offsets
//!
//! With `Y` an integer, `Round(Y + t) = Y + Round(t)`: each channel is
//! the luma sample plus a rounded chroma offset, clamped to `0..=255`.
//! For whole chroma samples `cb, cr ∈ −128..=127` (the value minus 128)
//! the T.871 §7 offsets
//!
//! ```text
//! Round(1.402 cr)   Round(−(202 008 cb + 419 198 cr) / 587 000)   Round(1.772 cb)
//! ```
//!
//! equal `(m · c + bias) >> s` in 32-bit integers for the constants of
//! [`EXACT`] — over all 256 values of one chroma sample, and all 65 536
//! pairs for G (`tests::whole_sample_offsets_are_the_rounded_quotients`
//! checks every one). The `2^16` fixed-point kernel has the same shape
//! with its own constants ([`FIXED16`]).

use crate::convert::{chroma_offset, green_offset};
use crate::image::{ChromaUpsampling, ConvertOptions, JpegImage, Plane, YcbcrKernel};
use crate::jpeg::fround::round_in_range;
use crate::jpeg::resample::{GuidedGeom, GuidedRows, GuidedSource};

/// One channel's offset from a whole chroma sample `c` (the value minus
/// 128): `(m · c + bias) >> shift`.
#[derive(Clone, Copy)]
struct Scale {
    m: i32,
    bias: i32,
    shift: u32,
}

impl Scale {
    /// The same form on the raw sample `v = c + 128`.
    const fn on_sample(self) -> Self {
        Self {
            m: self.m,
            bias: self.bias - 128 * self.m,
            shift: self.shift,
        }
    }
}

/// The three offsets of a YCbCr kernel as multiply-shift forms; G is
/// `(bias − mb · cb − mr · cr) >> shift`.
struct Coefficients {
    r: Scale,
    b: Scale,
    g_cb: i32,
    g_cr: i32,
    g_bias: i32,
    g_shift: u32,
}

/// T.871 §7, one rounding (`YcbcrKernel::Exact`) on whole chroma
/// samples. `1.402 · 2^15 = 45 940.7`, `1.772 · 2^15 = 58 064.9`, and
/// the two G coefficients at `2^23`; the biases are one half plus the
/// margin that keeps every quotient on the right side of its integer.
const EXACT: Coefficients = Coefficients {
    r: Scale {
        m: 45_941,
        bias: 16_384,
        shift: 15,
    },
    b: Scale {
        m: 58_065,
        bias: 16_400,
        shift: 15,
    },
    g_cb: 2_886_825,
    g_cr: 5_990_610,
    g_bias: 4_194_307,
    g_shift: 23,
};

/// `YcbcrKernel::FixedPoint16`: every coefficient at `2^16`, each
/// product rounded half-up, the two G products summed first.
const FIXED16: Coefficients = Coefficients {
    r: Scale {
        m: 91_881,
        bias: 1 << 15,
        shift: 16,
    },
    b: Scale {
        m: 116_130,
        bias: 1 << 15,
        shift: 16,
    },
    g_cb: 22_554,
    g_cr: 46_802,
    g_bias: 1 << 15,
    g_shift: 16,
};

const fn coefficients<const FIXED: bool>() -> &'static Coefficients {
    if FIXED { &FIXED16 } else { &EXACT }
}

/// The R, G, B offsets of the raw chroma samples `cb`, `cr` (`0..=255`).
#[inline(always)]
fn sample_offsets<const FIXED: bool>(cb: i32, cr: i32) -> [i32; 3] {
    let k = coefficients::<FIXED>();
    let (r, b) = (k.r.on_sample(), k.b.on_sample());
    let g_bias = k.g_bias + 128 * (k.g_cb + k.g_cr);
    [
        (cr * r.m + r.bias) >> r.shift,
        (g_bias - cb * k.g_cb - cr * k.g_cr) >> k.g_shift,
        (cb * b.m + b.bias) >> b.shift,
    ]
}

/// The R, G and B rows of one picture row, filled channel by channel
/// and then interleaved into the pixel layout ([`store`]).
struct Channels {
    r: Vec<u8>,
    g: Vec<u8>,
    b: Vec<u8>,
}

impl Channels {
    fn new(w: usize) -> Self {
        Self {
            r: vec![0; w],
            g: vec![0; w],
            b: vec![0; w],
        }
    }
}

/// `y + offset`, clamped to a sample.
#[inline(always)]
fn add_offset(y: u8, offset: i32) -> u8 {
    (y as i16 + offset as i16).clamp(0, 255) as u8
}

/// One row of whole-sample `Y`, `Cb`, `Cr` (chroma `0..=255` in any
/// integer type), one channel per pass.
#[inline(never)]
fn samples_to_channels<const FIXED: bool, C: Copy + Into<i32>>(
    y: &[u8],
    cb: &[C],
    cr: &[C],
    out: &mut Channels,
) {
    let n = y.len().min(cb.len()).min(cr.len()).min(out.r.len());
    let (y, cb, cr) = (&y[..n], &cb[..n], &cr[..n]);
    let (r, g, b) = (&mut out.r[..n], &mut out.g[..n], &mut out.b[..n]);
    let k = coefficients::<FIXED>();
    let (kr, kb) = (k.r.on_sample(), k.b.on_sample());
    let g_bias = k.g_bias + 128 * (k.g_cb + k.g_cr);
    for ((o, &y), &c) in r.iter_mut().zip(y).zip(cr) {
        *o = add_offset(y, (c.into() * kr.m + kr.bias) >> kr.shift);
    }
    for (((o, &y), &cb), &cr) in g.iter_mut().zip(y).zip(cb).zip(cr) {
        *o = add_offset(
            y,
            (g_bias - cb.into() * k.g_cb - cr.into() * k.g_cr) >> k.g_shift,
        );
    }
    for ((o, &y), &c) in b.iter_mut().zip(y).zip(cb) {
        *o = add_offset(y, (c.into() * kb.m + kb.bias) >> kb.shift);
    }
}

/// Chroma in sixteenths of a sample: `h = 16 · C`, `0..=4080`, so the
/// offsets are those of `crate::convert::ycbcr_frac_to_rgb8_8bit` at
/// `C · 256 = 16 h`. With `n = h − 2048`:
///
/// ```text
/// R: Round(701 n / 8000)  = (367 526 n + 2^21) >> 22
/// B: Round(443 n / 4000)  = (464 519 n + 2^21 + 524) >> 22
/// G: Round(−(101 004 nb + 209 599 nr) / 4 696 000)
/// ```
///
/// R and B hold for every `n` (`tests::sixteenth_offsets_…`). G is an
/// exact quotient: the numerator plus half the divisor plus `137`
/// divisors is positive and below `2^31`; `4 696 000 = 2^6 · 73 375`,
/// so with `u = ⌊numerator / 2^6⌋ < 2.01 · 10^7` the quotient is
/// `⌊u / 73 375⌋`, estimated from below by `((u >> 9) · 29 267) >> 22`
/// (`29 267 = ⌊2^31 / 73 375⌋`; short by less than one) and completed
/// by one comparison of the remainder.
#[inline(always)]
fn frac16_offsets(hb: i32, hr: i32) -> [i32; 3] {
    const R_M: i32 = 367_526;
    const B_M: i32 = 464_519;
    const BIAS: u32 = 137;
    // 2 348 000 + 137 · 4 696 000 + 2048 · (101 004 + 209 599).
    const G_TOP: u32 = 2_348_000 + BIAS * 4_696_000 + 2048 * 310_603;
    let r = (hr * R_M + ((1 << 21) - 2048 * R_M)) >> 22;
    let b = (hb * B_M + ((1 << 21) + 524 - 2048 * B_M)) >> 22;
    let u = (G_TOP - 101_004 * hb as u32 - 209_599 * hr as u32) >> 6;
    let e = ((u >> 9) * 29_267) >> 22;
    let rem = u - e * 73_375;
    let q = e + u32::from(rem >= 73_375);
    [r, q as i32 - BIAS as i32, b]
}

/// One row of `Y` with chroma in sixteenths of a sample, one channel
/// per pass ([`frac16_offsets`] taken apart).
#[inline(never)]
fn frac16_to_channels(y: &[u8], hb: &[u16], hr: &[u16], out: &mut Channels) {
    let n = y.len().min(hb.len()).min(hr.len()).min(out.r.len());
    let (y, hb, hr) = (&y[..n], &hb[..n], &hr[..n]);
    let (r, g, b) = (&mut out.r[..n], &mut out.g[..n], &mut out.b[..n]);
    for ((o, &y), &h) in r.iter_mut().zip(y).zip(hr) {
        *o = add_offset(y, frac16_offsets(0, h as i32)[0]);
    }
    for (((o, &y), &hb), &hr) in g.iter_mut().zip(y).zip(hb).zip(hr) {
        *o = add_offset(y, frac16_offsets(hb as i32, hr as i32)[1]);
    }
    for ((o, &y), &h) in b.iter_mut().zip(y).zip(hb) {
        *o = add_offset(y, frac16_offsets(h as i32, 0)[2]);
    }
}

/// One row of `Y` with the chroma offsets of the exact kernel in
/// `1/256` of a sample (`Round(256 C) − 128 · 256`), one channel per
/// pass: the arithmetic of `crate::convert::ycbcr_frac_to_rgb8_8bit`.
#[inline(never)]
fn frac256_to_channels(y: &[u8], cb: &[i16], cr: &[i16], out: &mut Channels) {
    let n = y.len().min(cb.len()).min(cr.len()).min(out.r.len());
    let (y, cb, cr) = (&y[..n], &cb[..n], &cr[..n]);
    let (r, g, b) = (&mut out.r[..n], &mut out.g[..n], &mut out.b[..n]);
    for ((o, &y), &c) in r.iter_mut().zip(y).zip(cr) {
        *o = add_offset(y, chroma_offset::<1402, 180>(c as i32));
    }
    for (((o, &y), &cb), &cr) in g.iter_mut().zip(y).zip(cb).zip(cr) {
        *o = add_offset(y, green_offset(cb as i32, cr as i32));
    }
    for ((o, &y), &c) in b.iter_mut().zip(y).zip(cb) {
        *o = add_offset(y, chroma_offset::<1772, 227>(c as i32));
    }
}

/// The three offsets of every sample of one chroma row.
#[inline(never)]
fn offsets_row<const FIXED: bool>(cb: &[u8], cr: &[u8], off: &mut [Vec<i16>; 3]) {
    let [r, g, b] = off;
    let n = cb.len().min(cr.len()).min(r.len());
    let (cb, cr) = (&cb[..n], &cr[..n]);
    let (r, g, b) = (&mut r[..n], &mut g[..n], &mut b[..n]);
    for (o, &c) in r.iter_mut().zip(cr) {
        *o = sample_offsets::<FIXED>(0, c as i32)[0] as i16;
    }
    for ((o, &cb), &cr) in g.iter_mut().zip(cb).zip(cr) {
        *o = sample_offsets::<FIXED>(cb as i32, cr as i32)[1] as i16;
    }
    for (o, &c) in b.iter_mut().zip(cb) {
        *o = sample_offsets::<FIXED>(c as i32, 0)[2] as i16;
    }
}

/// One luma row plus the offsets of its chroma row, each offset serving
/// `SPAN` neighbouring pixels.
#[inline(never)]
fn offsets_to_channels<const SPAN: usize>(y: &[u8], off: &[Vec<i16>; 3], out: &mut Channels) {
    let n = y.len().min(out.r.len());
    let y = &y[..n];
    let rows = [&mut out.r[..n], &mut out.g[..n], &mut out.b[..n]];
    for (row, off) in rows.into_iter().zip(off) {
        let (cells, tail) = row.as_chunks_mut::<SPAN>();
        let (yc, yt) = y.as_chunks::<SPAN>();
        for ((cell, y), &o) in cells.iter_mut().zip(yc).zip(off) {
            for (d, &y) in cell.iter_mut().zip(y) {
                *d = (y as i16 + o).clamp(0, 255) as u8;
            }
        }
        if let Some(&o) = off.get(cells.len()) {
            for (d, &y) in tail.iter_mut().zip(yt) {
                *d = (y as i16 + o).clamp(0, 255) as u8;
            }
        }
    }
}

/// Interleave one row of channels into pixels: `N` bytes each, `B G R`
/// order when `BGR`, the fourth byte (if any) opaque.
#[inline(never)]
fn store<const N: usize, const BGR: bool>(dst: &mut [u8], ch: &Channels) {
    let (px, _) = dst.as_chunks_mut::<N>();
    let n = px.len().min(ch.r.len());
    let (first, last) = if BGR { (&ch.b, &ch.r) } else { (&ch.r, &ch.b) };
    let (first, g, last) = (&first[..n], &ch.g[..n], &last[..n]);
    for (((p, &a), &g), &c) in px[..n].iter_mut().zip(first).zip(g).zip(last) {
        p[0] = a;
        p[1] = g;
        p[2] = c;
        if N == 4 {
            p[3] = 255;
        }
    }
}

/// `4 s(x)` of the module docs for every output `x` of a 2:1 axis:
/// `v` holds the chroma samples (or their vertical sums), `out` gets
/// `2 · v.len()` or `2 · v.len() − 1` values, each passed through
/// `even` / `odd` by the parity of `x`.
#[inline(never)]
fn widen_3_1(v: &[u16], out: &mut [u16], even: impl Fn(u16) -> u16, odd: impl Fn(u16) -> u16) {
    let last = v.len() - 1;
    let w = out.len();
    debug_assert!(w == 2 * last + 1 || w == 2 * last + 2);
    out[0] = even(4 * v[0]);
    let (pairs, _) = out[1..2 * last + 1].as_chunks_mut::<2>();
    for ((o, &a), &b) in pairs.iter_mut().zip(&v[..last]).zip(&v[1..]) {
        o[0] = odd(3 * a + b);
        o[1] = even(a + 3 * b);
    }
    if w == 2 * last + 2 {
        out[w - 1] = odd(4 * v[last]);
    }
}

/// `16 s(x)` for every output `x` of a 4:1 axis. Chroma sample `b`
/// lies between outputs `4b + 1` and `4b + 2`, so the four outputs
/// from `4b + 2` on lie one, three, five and seven eighths of the way
/// to sample `b + 1`; the two outputs before the first sample and
/// after the last replicate it. `out` holds `4 · v.len()` values (the
/// picture may use up to three fewer).
#[inline(never)]
fn widen_4_to_1(v: &[u16], out: &mut [u16]) {
    let last = v.len() - 1;
    debug_assert_eq!(out.len(), 4 * v.len());
    out[..2].fill(16 * v[0]);
    let (cells, _) = out[2..4 * last + 2].as_chunks_mut::<4>();
    for ((o, &a), &b) in cells.iter_mut().zip(&v[..last]).zip(&v[1..]) {
        *o = [
            2 * (7 * a + b),
            2 * (5 * a + 3 * b),
            2 * (3 * a + 5 * b),
            2 * (a + 7 * b),
        ];
    }
    out[4 * last + 2..].fill(16 * v[last]);
}

/// The two chroma rows output row `y` of a 2:1 vertical axis reads and
/// their weights (`4 s(y)` of the module docs): `(i0, i1, w0, w1)`.
fn vertical_taps(y: usize, ch: usize) -> (usize, usize, u16, u16) {
    let last = ch - 1;
    if y == 0 {
        return (0, 0, 4, 0);
    }
    let j = (y - 1) / 2;
    if j >= last {
        (last, last, 4, 0)
    } else if y % 2 == 1 {
        (j, j + 1, 3, 1)
    } else {
        (j, j + 1, 1, 3)
    }
}

/// The planes of a well-formed 8-bit planar YCbCr picture.
struct Planes<'a> {
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    dh: usize,
    dv: usize,
    y: &'a Plane,
    cb: &'a Plane,
    cr: &'a Plane,
}

impl<'a> Planes<'a> {
    fn luma(&self, y: usize) -> &'a [u8] {
        &self.y.data[y * self.y.stride..][..self.w]
    }

    fn chroma(&self, j: usize) -> [&'a [u8]; 2] {
        [
            &self.cb.data[j * self.cb.stride..][..self.cw],
            &self.cr.data[j * self.cr.stride..][..self.cw],
        ]
    }

    /// `4 s(y)` along the vertical axis for both components (the
    /// samples themselves times `unit` when the axis is not subsampled).
    fn vertical(&self, y: usize, unit: u16, v: &mut [Vec<u16>; 2]) {
        if self.dv == 1 {
            for (v, row) in v.iter_mut().zip(self.chroma(y)) {
                for (d, &s) in v.iter_mut().zip(row) {
                    *d = unit * s as u16;
                }
            }
            return;
        }
        let (i0, i1, w0, w1) = vertical_taps(y, self.ch);
        let (r0, r1) = (self.chroma(i0), self.chroma(i1));
        for ((v, a), b) in v.iter_mut().zip(r0).zip(r1) {
            for ((d, &a), &b) in v.iter_mut().zip(a).zip(b) {
                *d = w0 * a as u16 + w1 * b as u16;
            }
        }
    }
}

/// Rows `y0 ..` of the picture into `band` (rows `stride` bytes apart).
type Band<'a> = (usize, &'a mut [u8]);

/// 4:4:4: the plane rows as they are.
fn band_direct<const N: usize, const BGR: bool, const FIXED: bool>(
    p: &Planes<'_>,
    stride: usize,
    (y0, band): Band<'_>,
) {
    let mut ch = Channels::new(p.w);
    for (i, dst) in band.chunks_mut(stride).enumerate() {
        let y = y0 + i;
        let [cb, cr] = p.chroma(y);
        samples_to_channels::<FIXED, u8>(p.luma(y), cb, cr, &mut ch);
        store::<N, BGR>(&mut dst[..p.w * N], &ch);
    }
}

/// Replicated chroma: offsets per chroma sample, formed once per chroma
/// row and added to every luma row under it.
fn band_replicate<const N: usize, const BGR: bool, const FIXED: bool>(
    p: &Planes<'_>,
    stride: usize,
    (y0, band): Band<'_>,
) {
    let mut ch = Channels::new(p.w);
    let mut off: [Vec<i16>; 3] = std::array::from_fn(|_| vec![0i16; p.cw]);
    let mut have = usize::MAX;
    for (i, dst) in band.chunks_mut(stride).enumerate() {
        let y = y0 + i;
        let j = y / p.dv;
        if j != have {
            let [cb, cr] = p.chroma(j);
            offsets_row::<FIXED>(cb, cr, &mut off);
            have = j;
        }
        if p.dh == 2 {
            offsets_to_channels::<2>(p.luma(y), &off, &mut ch);
        } else {
            offsets_to_channels::<4>(p.luma(y), &off, &mut ch);
        }
        store::<N, BGR>(&mut dst[..p.w * N], &ch);
    }
}

/// Interpolated chroma rounded to whole samples: `Triangular` (`tri`),
/// or `Linear` ahead of the fixed-point kernel, which takes whole
/// samples (`Round(h / 16)`).
fn band_rounded<const N: usize, const BGR: bool, const FIXED: bool>(
    p: &Planes<'_>,
    stride: usize,
    (y0, band): Band<'_>,
    tri: bool,
) {
    let mut ch = Channels::new(p.w);
    let mut v = [vec![0u16; p.cw], vec![0u16; p.cw]];
    let n = if p.dh == 4 { 4 * p.cw } else { p.w };
    let mut wide = [vec![0u16; n], vec![0u16; n]];
    // Triangular on one axis works in quarters, everything else in
    // sixteenths.
    let quarters = tri && p.dv == 1;
    for (i, dst) in band.chunks_mut(stride).enumerate() {
        let y = y0 + i;
        if p.dh == 4 {
            p.vertical(y, 1, &mut v);
            for (v, wide) in v.iter().zip(wide.iter_mut()) {
                widen_4_to_1(v, wide);
                for h in wide.iter_mut() {
                    *h = (*h + 8) >> 4;
                }
            }
        } else {
            p.vertical(y, if quarters { 1 } else { 4 }, &mut v);
            for (v, wide) in v.iter().zip(wide.iter_mut()) {
                match (tri, quarters) {
                    (true, true) => widen_3_1(v, wide, |h| (h + 1) >> 2, |h| (h + 2) >> 2),
                    (true, false) => widen_3_1(v, wide, |h| (h + 8) >> 4, |h| (h + 7) >> 4),
                    _ => widen_3_1(v, wide, |h| (h + 8) >> 4, |h| (h + 8) >> 4),
                }
            }
        }
        samples_to_channels::<FIXED, u16>(p.luma(y), &wide[0], &wide[1], &mut ch);
        store::<N, BGR>(&mut dst[..p.w * N], &ch);
    }
}

/// `Linear` under the exact kernel: chroma in sixteenths, one rounding.
fn band_linear<const N: usize, const BGR: bool>(
    p: &Planes<'_>,
    stride: usize,
    (y0, band): Band<'_>,
) {
    let mut ch = Channels::new(p.w);
    let mut v = [vec![0u16; p.cw], vec![0u16; p.cw]];
    let n = if p.dh == 4 { 4 * p.cw } else { p.w };
    let mut wide = [vec![0u16; n], vec![0u16; n]];
    for (i, dst) in band.chunks_mut(stride).enumerate() {
        let y = y0 + i;
        if p.dh == 4 {
            p.vertical(y, 1, &mut v);
            for (v, wide) in v.iter().zip(wide.iter_mut()) {
                widen_4_to_1(v, wide);
            }
        } else {
            p.vertical(y, 4, &mut v);
            for (v, wide) in v.iter().zip(wide.iter_mut()) {
                widen_3_1(v, wide, |h| h, |h| h);
            }
        }
        frac16_to_channels(p.luma(y), &wide[0], &wide[1], &mut ch);
        store::<N, BGR>(&mut dst[..p.w * N], &ch);
    }
}

/// `LumaGuided` at a 2:1 horizontal ratio: the reconstruction hands
/// each row over as the kernel's chroma offsets (exact kernel, to
/// `1/256` of a sample) or as whole samples (fixed-point kernel).
fn band_guided<const N: usize, const BGR: bool, const FIXED: bool>(
    p: &Planes<'_>,
    stride: usize,
    (y0, band): Band<'_>,
) {
    let geom = GuidedGeom::new(p.cw, p.ch, p.dh, p.dv, p.w, p.h, 255);
    let read = |plane: &Plane, j: usize, row: &mut [f32]| {
        for (d, &s) in row.iter_mut().zip(&plane.data[j * plane.stride..]) {
            *d = s as f32;
        }
    };
    let luma_row = |y: usize, row: &mut [f32]| read(p.y, y, row);
    let chroma_row = |k: usize, j: usize, row: &mut [f32]| read([p.cb, p.cr][k], j, row);
    let source = GuidedSource {
        luma: &luma_row,
        luma_u8: Some((&p.y.data[..], p.y.stride)),
        chroma: &chroma_row,
    };
    let mut rows = GuidedRows::new(&geom, 2);
    rows.seek(&geom, y0);
    // Every reconstructed value is clamped to the range of real chroma
    // samples around it, so it lies in `0..=255` and needs no limiting
    // on its way to an integer.
    let mut ch = Channels::new(p.w);
    let mut wide = [vec![0i16; p.w], vec![0i16; p.w]];
    for (i, dst) in band.chunks_mut(stride).enumerate() {
        let y = y0 + i;
        let luma = p.luma(y);
        let [cb, cr] = &mut wide;
        if FIXED {
            let whole = |v: f32| round_in_range(v) as i16;
            rows.row_2to1(&geom, &source, y, luma, [cb, cr], whole);
            samples_to_channels::<true, i16>(luma, &wide[0], &wide[1], &mut ch);
        } else {
            let offset = |v: f32| (round_in_range(v * 256.0) - 128 * 256) as i16;
            rows.row_2to1(&geom, &source, y, luma, [cb, cr], offset);
            frac256_to_channels(luma, &wide[0], &wide[1], &mut ch);
        }
        store::<N, BGR>(&mut dst[..p.w * N], &ch);
    }
}

/// How the chroma of a layout reaches the colour equations.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Plan {
    Direct,
    Replicate,
    Triangular,
    Linear,
    Guided,
}

fn fill_as<const N: usize, const BGR: bool>(
    p: &Planes<'_>,
    out: &mut [u8],
    stride: usize,
    opts: &ConvertOptions,
) -> bool {
    let plan = match (opts.upsampling, p.dh, p.dv) {
        (_, 1, 1) => Plan::Direct,
        (ChromaUpsampling::Replicate, 2 | 4, _) => Plan::Replicate,
        // The triangular kernel covers the 2:1 ratios of components
        // more than two samples wide; the rest is replicated.
        (ChromaUpsampling::Triangular, 2, _) if p.cw > 2 => Plan::Triangular,
        (ChromaUpsampling::Triangular, 2 | 4, _) => Plan::Replicate,
        (ChromaUpsampling::Linear, 2, _) | (ChromaUpsampling::Linear, 4, 1) => Plan::Linear,
        (ChromaUpsampling::LumaGuided, 2, _) => Plan::Guided,
        _ => return false,
    };
    let fixed = opts.ycbcr == YcbcrKernel::FixedPoint16;
    let run = |band: Band<'_>| match (plan, fixed) {
        (Plan::Direct, false) => band_direct::<N, BGR, false>(p, stride, band),
        (Plan::Direct, true) => band_direct::<N, BGR, true>(p, stride, band),
        (Plan::Replicate, false) => band_replicate::<N, BGR, false>(p, stride, band),
        (Plan::Replicate, true) => band_replicate::<N, BGR, true>(p, stride, band),
        (Plan::Triangular, false) => band_rounded::<N, BGR, false>(p, stride, band, true),
        (Plan::Triangular, true) => band_rounded::<N, BGR, true>(p, stride, band, true),
        (Plan::Linear, false) => band_linear::<N, BGR>(p, stride, band),
        (Plan::Linear, true) => band_rounded::<N, BGR, true>(p, stride, band, false),
        (Plan::Guided, false) => band_guided::<N, BGR, false>(p, stride, band),
        (Plan::Guided, true) => band_guided::<N, BGR, true>(p, stride, band),
    };
    let workers = opts.threads.clamp(1, p.h);
    if workers <= 1 {
        run((0, out));
    } else {
        let rows_per = p.h.div_ceil(workers);
        std::thread::scope(|sc| {
            let run = &run;
            for (i, band) in out.chunks_mut(rows_per * stride).enumerate() {
                sc.spawn(move || run((i * rows_per, band)));
            }
        });
    }
    true
}

/// Convert `img` into `out` — rows `stride` bytes apart, `n` bytes per
/// pixel (`3`, or `4` with opaque alpha), `B G R` order when `bgr` — if
/// it is an 8-bit planar YCbCr picture with whole planes at a ratio
/// this module covers. Returns `false`, leaving `out` alone, otherwise.
pub(crate) fn fill(
    img: &JpegImage,
    out: &mut [u8],
    stride: usize,
    n: usize,
    bgr: bool,
    opts: &ConvertOptions,
) -> bool {
    let (w, h) = (img.width as usize, img.height as usize);
    if !img.format.is_yuv()
        || img.format.bytes_per_sample() != 1
        || img.precision < 8
        || img.planes.len() < 3
        || w == 0
        || h == 0
    {
        return false;
    }
    let (dh, dv) = img.format.chroma_divisors();
    let (cw, ch) = (w.div_ceil(dh), h.div_ceil(dv));
    let whole = |p: &Plane, pw: usize, ph: usize| {
        p.stride >= pw && p.data.len() >= (ph - 1) * p.stride + pw
    };
    let (y, cb, cr) = (&img.planes[0], &img.planes[1], &img.planes[2]);
    if !(whole(y, w, h) && whole(cb, cw, ch) && whole(cr, cw, ch))
        || stride < w * n
        || out.len() < (h - 1) * stride + w * n
    {
        return false;
    }
    let p = Planes {
        w,
        h,
        cw,
        ch,
        dh,
        dv,
        y,
        cb,
        cr,
    };
    let out = &mut out[..(h - 1) * stride + w * n];
    match (n, bgr) {
        (3, false) => fill_as::<3, false>(&p, out, stride, opts),
        (3, true) => fill_as::<3, true>(&p, out, stride, opts),
        (4, false) => fill_as::<4, false>(&p, out, stride, opts),
        (4, true) => fill_as::<4, true>(&p, out, stride, opts),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::{ycbcr_frac_to_rgb8_8bit, ycbcr_to_rgb8, ycbcr_to_rgb8_fixed16};

    #[test]
    fn whole_sample_offsets_are_the_rounded_quotients() {
        let floor = |n: i64, d: i64| n.div_euclid(d);
        for cb in 0..=255i64 {
            for cr in 0..=255i64 {
                let (b, r) = (cb - 128, cr - 128);
                let exact = sample_offsets::<false>(cb as i32, cr as i32).map(i64::from);
                let want = [
                    floor(1402 * r + 500, 1000),
                    floor(-202_008 * b - 419_198 * r + 293_500, 587_000),
                    floor(1772 * b + 500, 1000),
                ];
                assert_eq!(exact, want, "exact {cb} {cr}");
                let fixed = sample_offsets::<true>(cb as i32, cr as i32).map(i64::from);
                let want = [
                    (91_881 * r + 32_768) >> 16,
                    (-22_554 * b - 46_802 * r + 32_768) >> 16,
                    (116_130 * b + 32_768) >> 16,
                ];
                assert_eq!(fixed, want, "fixed {cb} {cr}");
                // And through the scalar kernels, clamps included.
                for y in [0u32, 77, 255] {
                    let (cb, cr) = (cb as u32, cr as u32);
                    let (r, g, b) = ycbcr_to_rgb8(y, cb, cr, 255);
                    let got = sample_offsets::<false>(cb as i32, cr as i32);
                    assert_eq!(got.map(|o| add_offset(y as u8, o)), [r, g, b]);
                    let (r, g, b) = ycbcr_to_rgb8_fixed16(y, cb, cr);
                    let got = sample_offsets::<true>(cb as i32, cr as i32);
                    assert_eq!(got.map(|o| add_offset(y as u8, o)), [r, g, b]);
                }
            }
        }
    }

    #[test]
    fn sixteenth_offsets_are_the_rounded_quotients() {
        // Every chroma value for R and B, every pair for G.
        let floor = |n: i64, d: i64| n.div_euclid(d);
        for hb in 0..=4080i64 {
            let nb = hb - 2048;
            let got = frac16_offsets(hb as i32, hb as i32);
            assert_eq!(got[0] as i64, floor(701 * nb + 4000, 8000), "R {hb}");
            assert_eq!(got[2] as i64, floor(443 * nb + 2000, 4000), "B {hb}");
            for hr in 0..=4080i64 {
                let n = -(101_004 * nb + 209_599 * (hr - 2048));
                let got = frac16_offsets(hb as i32, hr as i32)[1] as i64;
                assert_eq!(got, floor(n + 2_348_000, 4_696_000), "G {hb} {hr}");
            }
        }
    }

    #[test]
    fn sixteenth_kernel_is_the_fractional_kernel() {
        let mut s = 0x9E37_79B9u32;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for _ in 0..200_000 {
            let (y, hb, hr) = (next() % 256, next() % 4081, next() % 4081);
            let want = ycbcr_frac_to_rgb8_8bit(y as i32, 16 * hb as i32, 16 * hr as i32);
            let off = frac16_offsets(hb as i32, hr as i32);
            assert_eq!(off.map(|o| add_offset(y as u8, o)), want, "{y} {hb} {hr}");
        }
    }

    #[test]
    fn widening_weights_three_to_one_with_replicated_ends() {
        let v = [10u16, 20, 60];
        let mut odd = [0u16; 5];
        widen_3_1(&v, &mut odd, |h| h, |h| h);
        assert_eq!(odd, [40, 50, 70, 120, 200]);
        let mut even = [0u16; 6];
        widen_3_1(&v, &mut even, |h| h, |h| h + 1000);
        assert_eq!(even, [40, 1050, 70, 1120, 200, 1240]);
        let mut one = [0u16; 2];
        widen_3_1(&[7], &mut one, |h| h, |h| h);
        assert_eq!(one, [28, 28]);
    }

    #[test]
    fn four_to_one_widening_weights_in_eighths() {
        // Between 8 and 24: 10, 14, 18, 22 — one, three, five and seven
        // eighths of the way — in sixteenths.
        let mut out = [0u16; 8];
        widen_4_to_1(&[8, 24], &mut out);
        assert_eq!(out, [128, 128, 160, 224, 288, 352, 384, 384]);
        let mut one = [0u16; 4];
        widen_4_to_1(&[5], &mut one);
        assert_eq!(one, [80; 4]);
    }

    #[test]
    fn vertical_taps_follow_centred_siting() {
        // Five rows over three chroma rows, six rows over three.
        let rows =
            |h: usize, ch: usize| -> Vec<_> { (0..h).map(|y| vertical_taps(y, ch)).collect() };
        assert_eq!(
            rows(5, 3),
            [
                (0, 0, 4, 0),
                (0, 1, 3, 1),
                (0, 1, 1, 3),
                (1, 2, 3, 1),
                (1, 2, 1, 3)
            ]
        );
        assert_eq!(rows(6, 3)[5], (2, 2, 4, 0));
        assert_eq!(rows(2, 1), [(0, 0, 4, 0), (0, 0, 4, 0)]);
    }
}
