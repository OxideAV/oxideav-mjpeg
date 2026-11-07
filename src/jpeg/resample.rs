//! Default-quality chroma reconstruction: centred-siting linear
//! interpolation for any sampling ratio, and the luma-guided
//! reconstruction built on it.
//!
//! **Siting.** T.871 §9 places a subsampled chroma sample at the centre
//! of the luma samples it covers, and T.81 §A.1.1 relates the component
//! grids by `xc = ⌈X · Hc / Hmax⌉`. Output sample `x` (centre `x + ½` in
//! luma units) therefore sits at chroma coordinate
//! `u = (x + ½) · Hc / Hmax − ½`, for every ratio — 2:1, 4:1, 2:1
//! vertical (4:4:0), and the non-power-of-two ratios mixed layouts
//! produce (3:1, 3:2 …). Off the ends of the component the nearest real
//! sample is replicated.
//!
//! **Linear** ([`Linear`]) reconstructs each subsampled axis by linear
//! interpolation between the two chroma samples bracketing `u`
//! (separable, vertical first), in `f32`, with no intermediate rounding.
//!
//! **Luma-guided** ([`Guided`]) adds to the linear estimate the luma
//! detail the subsampling removed, scaled by the local chroma-on-luma
//! regression slope. Per chroma sample, with `Ŷ` the luma averaged over
//! the cell the sample covers (the encoder-side two-tap / box filter of
//! T.871 §9 NOTE 1) and the 3 × 3 chroma-grid neighbourhood `N`:
//!
//! ```text
//! a = cov_N(Ŷ, C) / (var_N(Ŷ) + ε)
//! C(x, y) = lin(C) + lin(a) · (Y(x, y) − lin(Ŷ))
//! ```
//!
//! clamped to the `[min, max]` of `C` over `N` of the chroma sample the
//! output pixel lies in, so it never overshoots the neighbourhood.
//! Where chroma does not follow luma `a ≈ 0` and the result is the
//! linear interpolation; `ε` (`16` in 8-bit units, scaled by the square
//! of the sample range at other precisions) keeps flat or noisy regions
//! from amplifying luma noise into chroma.
//!
//! Both produce unrounded `f32` chroma so the colour conversion can
//! round once.

/// Interpolation tap pair along one axis: output `o` reads
/// `s[i0] + f · (s[i1] − s[i0])`.
#[derive(Clone, Copy, Debug)]
struct Tap {
    i0: u32,
    i1: u32,
    f: f32,
}

/// Taps for `out` output samples over `len` chroma samples at
/// `num / den` chroma samples per output sample (`Hc / Hmax`).
fn taps(out: usize, len: usize, num: usize, den: usize) -> Vec<Tap> {
    let last = len.saturating_sub(1);
    (0..out)
        .map(|o| {
            if num == den {
                let i = o.min(last) as u32;
                return Tap {
                    i0: i,
                    i1: i,
                    f: 0.0,
                };
            }
            // u = (o + ½)·num/den − ½ = ((2o + 1)·num − den) / (2·den)
            let n = (2 * o + 1) as i64 * num as i64 - den as i64;
            let d = 2 * den as i64;
            let fl = n.div_euclid(d);
            let f = (n.rem_euclid(d)) as f32 / d as f32;
            if fl < 0 {
                Tap {
                    i0: 0,
                    i1: 0,
                    f: 0.0,
                }
            } else if fl as usize >= last {
                Tap {
                    i0: last as u32,
                    i1: last as u32,
                    f: 0.0,
                }
            } else {
                Tap {
                    i0: fl as u32,
                    i1: fl as u32 + 1,
                    f,
                }
            }
        })
        .collect()
}

#[inline]
fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + f * (b - a)
}

/// Centred-siting linear interpolation of one `cw × ch` component to
/// `w × h` (module docs). `hc / hmax` and `vc / vmax` are the
/// component's sampling factors over the frame maxima.
#[derive(Debug)]
pub(crate) struct Linear {
    cw: usize,
    c: Vec<f32>,
    tx: Vec<Tap>,
    ty: Vec<Tap>,
    col: Vec<f32>,
}

impl Linear {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        c: Vec<f32>,
        cw: usize,
        ch: usize,
        (hc, hmax): (usize, usize),
        (vc, vmax): (usize, usize),
        w: usize,
        h: usize,
    ) -> Self {
        debug_assert_eq!(c.len(), cw * ch);
        Self {
            cw,
            c,
            tx: taps(w, cw, hc, hmax),
            ty: taps(h, ch, vc, vmax),
            col: vec![0.0; cw],
        }
    }

    /// Output row `y` into `out` (`w` samples).
    pub(crate) fn row(&mut self, y: usize, out: &mut [f32]) {
        let t = self.ty[y];
        let cw = self.cw;
        let r0 = &self.c[t.i0 as usize * cw..][..cw];
        let r1 = &self.c[t.i1 as usize * cw..][..cw];
        for ((d, &a), &b) in self.col.iter_mut().zip(r0).zip(r1) {
            *d = lerp(a, b, t.f);
        }
        for (o, tx) in out.iter_mut().zip(&self.tx) {
            *o = lerp(self.col[tx.i0 as usize], self.col[tx.i1 as usize], tx.f);
        }
    }
}

/// `ε` of the luma-guided reconstruction in 8-bit sample units².
const GUIDED_EPS_8BIT: f32 = 16.0;

/// Luma-guided reconstruction of one component at an integer ratio
/// `rh × rv` (module docs).
#[derive(Debug)]
pub(crate) struct Guided {
    cw: usize,
    rh: usize,
    rv: usize,
    c: Vec<f32>,
    yd: Vec<f32>,
    a: Vec<f32>,
    lo: Vec<f32>,
    hi: Vec<f32>,
    tx: Vec<Tap>,
    ty: Vec<Tap>,
    col_c: Vec<f32>,
    col_y: Vec<f32>,
    col_a: Vec<f32>,
}

/// 3 × 3 box mean with replicated edges of a `w × h` grid.
fn box3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut t = vec![0f32; w * h];
    for (row, dst) in src.chunks_exact(w).zip(t.chunks_exact_mut(w)) {
        for x in 0..w {
            let l = row[x.saturating_sub(1)];
            let r = row[(x + 1).min(w - 1)];
            dst[x] = l + row[x] + r;
        }
    }
    let mut o = vec![0f32; w * h];
    for y in 0..h {
        let (u, m, d) = (
            &t[y.saturating_sub(1) * w..][..w],
            &t[y * w..][..w],
            &t[(y + 1).min(h - 1) * w..][..w],
        );
        for (x, v) in o[y * w..][..w].iter_mut().enumerate() {
            *v = (u[x] + m[x] + d[x]) * (1.0 / 9.0);
        }
    }
    o
}

impl Guided {
    /// `luma` is the full-resolution guide (`w × h`), `c` the `cw × ch`
    /// component (`cw = ⌈w / rh⌉`, `ch = ⌈h / rv⌉`), `maxval` the
    /// sample range `2^P − 1`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        luma: &[f32],
        c: Vec<f32>,
        cw: usize,
        ch: usize,
        rh: usize,
        rv: usize,
        w: usize,
        h: usize,
        maxval: u32,
    ) -> Self {
        debug_assert_eq!(luma.len(), w * h);
        debug_assert_eq!(c.len(), cw * ch);
        // Ŷ: luma averaged over each chroma cell's real samples.
        let mut yd = vec![0f32; cw * ch];
        let mut cnt = vec![0u32; cw * ch];
        for (y, row) in luma.chunks_exact(w).enumerate() {
            let base = (y / rv).min(ch - 1) * cw;
            for (x, &v) in row.iter().enumerate() {
                let i = base + (x / rh).min(cw - 1);
                yd[i] += v;
                cnt[i] += 1;
            }
        }
        for (v, &n) in yd.iter_mut().zip(&cnt) {
            *v /= n.max(1) as f32;
        }
        let scale = maxval as f32 / 255.0;
        let eps = GUIDED_EPS_8BIT * scale * scale;
        let yy: Vec<f32> = yd.iter().map(|v| v * v).collect();
        let yc: Vec<f32> = yd.iter().zip(&c).map(|(a, b)| a * b).collect();
        let (my, mc, myy, myc) = (
            box3(&yd, cw, ch),
            box3(&c, cw, ch),
            box3(&yy, cw, ch),
            box3(&yc, cw, ch),
        );
        let a: Vec<f32> = (0..cw * ch)
            .map(|i| {
                let var = (myy[i] - my[i] * my[i]).max(0.0);
                (myc[i] - my[i] * mc[i]) / (var + eps)
            })
            .collect();
        // Neighbourhood bounds.
        let mut lo = vec![0f32; cw * ch];
        let mut hi = vec![0f32; cw * ch];
        for y in 0..ch {
            for x in 0..cw {
                let (mut l, mut u) = (f32::INFINITY, f32::NEG_INFINITY);
                for yy in y.saturating_sub(1)..=(y + 1).min(ch - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(cw - 1) {
                        let v = c[yy * cw + xx];
                        l = l.min(v);
                        u = u.max(v);
                    }
                }
                lo[y * cw + x] = l;
                hi[y * cw + x] = u;
            }
        }
        Self {
            cw,
            rh,
            rv,
            c,
            yd,
            a,
            lo,
            hi,
            tx: taps(w, cw, 1, rh),
            ty: taps(h, ch, 1, rv),
            col_c: vec![0.0; cw],
            col_y: vec![0.0; cw],
            col_a: vec![0.0; cw],
        }
    }

    /// Output row `y` into `out`, guided by the matching luma row.
    pub(crate) fn row(&mut self, y: usize, luma: &[f32], out: &mut [f32]) {
        let t = self.ty[y];
        let cw = self.cw;
        let (o0, o1) = (t.i0 as usize * cw, t.i1 as usize * cw);
        for i in 0..cw {
            self.col_c[i] = lerp(self.c[o0 + i], self.c[o1 + i], t.f);
            self.col_y[i] = lerp(self.yd[o0 + i], self.yd[o1 + i], t.f);
            self.col_a[i] = lerp(self.a[o0 + i], self.a[o1 + i], t.f);
        }
        let cy = (y / self.rv).min(self.c.len() / cw - 1) * cw;
        let (lo, hi) = (&self.lo[cy..][..cw], &self.hi[cy..][..cw]);
        for (x, ((o, tx), &l)) in out.iter_mut().zip(&self.tx).zip(luma).enumerate() {
            let (i0, i1) = (tx.i0 as usize, tx.i1 as usize);
            let cu = lerp(self.col_c[i0], self.col_c[i1], tx.f);
            let yu = lerp(self.col_y[i0], self.col_y[i1], tx.f);
            let au = lerp(self.col_a[i0], self.col_a[i1], tx.f);
            let cx = (x / self.rh).min(cw - 1);
            *o = (cu + au * (l - yu)).clamp(lo[cx], hi[cx]);
        }
    }
}

/// Whole-component reconstruction rounded to integer samples, for the
/// decoder's resample-to-4:4:4 layouts: `get(x, y)` reads the `cw × ch`
/// component, `guide` optionally the full-resolution luma and the
/// integer ratio `(rh, rv)` (luma-guided), otherwise linear at
/// `hc / hmax`, `vc / vmax`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn reconstruct(
    get: &dyn Fn(usize, usize) -> u32,
    cw: usize,
    ch: usize,
    hr: (usize, usize),
    vr: (usize, usize),
    w: usize,
    h: usize,
    maxval: u32,
    guide: Option<(&dyn Fn(usize, usize) -> u32, (usize, usize))>,
) -> Vec<u32> {
    let grid = |g: &dyn Fn(usize, usize) -> u32, gw: usize, gh: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(gw * gh);
        for y in 0..gh {
            v.extend((0..gw).map(|x| g(x, y) as f32));
        }
        v
    };
    let c = grid(get, cw, ch);
    let mut out = vec![0u32; w * h];
    let mut row = vec![0f32; w];
    let store = |row: &[f32], dst: &mut [u32]| {
        for (d, &v) in dst.iter_mut().zip(row) {
            *d = v.round().clamp(0.0, maxval as f32) as u32;
        }
    };
    match guide {
        Some((gy, (rh, rv))) => {
            let luma = grid(gy, w, h);
            let mut g = Guided::new(&luma, c, cw, ch, rh, rv, w, h, maxval);
            for (y, dst) in out.chunks_exact_mut(w).enumerate() {
                g.row(y, &luma[y * w..][..w], &mut row);
                store(&row, dst);
            }
        }
        None => {
            let mut l = Linear::new(c, cw, ch, hr, vr, w, h);
            for (y, dst) in out.chunks_exact_mut(w).enumerate() {
                l.row(y, &mut row);
                store(&row, dst);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_follow_centred_siting() {
        // 2:1 — outputs sit at u = -0.25, 0.25, 0.75, 1.25, …
        let t = taps(6, 3, 1, 2);
        assert_eq!((t[0].i0, t[0].i1, t[0].f), (0, 0, 0.0));
        assert_eq!((t[1].i0, t[1].i1, t[1].f), (0, 1, 0.25));
        assert_eq!((t[2].i0, t[2].i1, t[2].f), (0, 1, 0.75));
        assert_eq!((t[5].i0, t[5].i1), (2, 2));
        // 4:1 — u = (x + ½)/4 − ½: x = 2 → 0.125.
        let t = taps(8, 2, 1, 4);
        assert_eq!((t[2].i0, t[2].i1, t[2].f), (0, 1, 0.125));
        // 2:3 (two chroma samples per three output samples).
        let t = taps(3, 2, 2, 3);
        // u = (x + ½)·2/3 − ½ : −1/6, 1/2, 7/6
        assert_eq!((t[1].i0, t[1].i1, t[1].f), (0, 1, 0.5));
        assert_eq!((t[2].i0, t[2].i1), (1, 1));
    }

    #[test]
    fn linear_ramps_and_flat_fields() {
        let c = vec![0.0, 100.0, 200.0];
        let mut l = Linear::new(c, 3, 1, (1, 2), (1, 1), 6, 1);
        let mut o = vec![0f32; 6];
        l.row(0, &mut o);
        assert_eq!(o, vec![0.0, 25.0, 75.0, 125.0, 175.0, 200.0]);
        let mut l = Linear::new(vec![77.0; 12], 4, 3, (1, 2), (1, 2), 7, 5);
        for y in 0..5 {
            l.row(y, &mut o[..]);
        }
        assert!(o.iter().take(6).all(|&v| v == 77.0));
    }

    #[test]
    fn guided_follows_a_luma_edge_and_stays_in_bounds() {
        // An 8 × 2 picture whose luma and chroma share a step at x = 3
        // (mid-way through chroma cell 1): the guided result keeps the
        // edge sharp, the linear one smears it.
        let (w, h) = (8, 2);
        let luma: Vec<f32> = (0..w * h)
            .map(|i| if i % w < 3 { 50.0 } else { 200.0 })
            .collect();
        let full_c: Vec<f32> = (0..w).map(|x| if x < 3 { 100.0 } else { 160.0 }).collect();
        // 2:1 box downsampling of the chroma.
        let c: Vec<f32> = (0..4)
            .map(|i| (full_c[2 * i] + full_c[2 * i + 1]) / 2.0)
            .collect();
        let c2 = [c.clone(), c.clone()].concat();
        let mut g = Guided::new(&luma, c2.clone(), 4, 2, 2, 1, w, h, 255);
        let mut out = vec![0f32; w];
        g.row(0, &luma[..w], &mut out);
        let mut l = Linear::new(c2, 4, 2, (1, 2), (1, 1), w, h);
        let mut lin = vec![0f32; w];
        l.row(0, &mut lin);
        let err = |v: &[f32]| {
            v.iter()
                .zip(&full_c)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
        };
        assert!(err(&out) < err(&lin) / 2.0, "{out:?} vs {lin:?}");
        assert!(out.iter().all(|&v| (100.0..=160.0).contains(&v)));
    }

    #[test]
    fn guided_flat_chroma_stays_flat() {
        let (w, h) = (9, 7);
        let luma: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 255) as f32).collect();
        let (cw, ch) = (5, 4);
        let mut g = Guided::new(&luma, vec![90.0; cw * ch], cw, ch, 2, 2, w, h, 255);
        let mut out = vec![0f32; w];
        for y in 0..h {
            g.row(y, &luma[y * w..][..w], &mut out);
            assert!(out.iter().all(|&v| v == 90.0));
        }
    }
}
