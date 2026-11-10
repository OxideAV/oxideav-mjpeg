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
        let mut col = std::mem::take(&mut self.col);
        self.row_with(&mut col, y, out);
        self.col = col;
    }

    /// Scratch for [`Linear::row_with`].
    pub(crate) fn scratch(&self) -> Vec<f32> {
        vec![0.0; self.cw]
    }

    /// [`Linear::row`] on shared state: `col` is a caller-owned
    /// [`Linear::scratch`], so several workers can produce rows at once.
    pub(crate) fn row_with(&self, col: &mut [f32], y: usize, out: &mut [f32]) {
        let t = self.ty[y];
        let cw = self.cw;
        let r0 = &self.c[t.i0 as usize * cw..][..cw];
        let r1 = &self.c[t.i1 as usize * cw..][..cw];
        for ((d, &a), &b) in col.iter_mut().zip(r0).zip(r1) {
            *d = lerp(a, b, t.f);
        }
        for (o, tx) in out.iter_mut().zip(&self.tx) {
            *o = lerp(col[tx.i0 as usize], col[tx.i1 as usize], tx.f);
        }
    }
}

/// `ε` of the luma-guided reconstruction in 8-bit sample units².
const GUIDED_EPS_8BIT: f32 = 16.0;

use crate::jpeg::fround::{fmax, fmin};

/// Horizontal phase `k` of an integer ratio `r`: output `i · r + k`
/// interpolates chroma samples `i − 1, i` (`left`) or `i, i + 1` at
/// fraction `f` — the [`taps`] of that ratio, cell by cell.
#[derive(Clone, Copy, Debug)]
struct Phase {
    left: bool,
    f: f32,
}

fn phases(r: usize) -> Vec<Phase> {
    (0..r)
        .map(|k| {
            // u − i = (2k + 1 − r) / 2r
            let n = 2 * k as i64 + 1 - r as i64;
            let d = 2 * r as i64;
            if n < 0 {
                Phase {
                    left: true,
                    f: (n + d) as f32 / d as f32,
                }
            } else {
                Phase {
                    left: false,
                    f: n as f32 / d as f32,
                }
            }
        })
        .collect()
}

/// 3-tap horizontal pass of a separable neighbourhood filter with
/// replicated edges: `dst[x] = op(op(src[x − 1], src[x]), src[x + 1])`.
/// The interior runs over three shifted slices, so it vectorises.
#[inline(always)]
fn three_tap(src: &[f32], dst: &mut [f32], op: impl Fn(f32, f32) -> f32) {
    let w = src.len();
    let dst = &mut dst[..w];
    if w == 1 {
        dst[0] = op(op(src[0], src[0]), src[0]);
        return;
    }
    dst[0] = op(op(src[0], src[0]), src[1]);
    let centre_right = src[1..w - 1].iter().zip(&src[2..]);
    let left = src[..w - 2].iter();
    for ((d, &l), (&m, &r)) in dst[1..w - 1].iter_mut().zip(left).zip(centre_right) {
        *d = op(op(l, m), r);
    }
    dst[w - 1] = op(op(src[w - 2], src[w - 1]), src[w - 1]);
}

/// The horizontally filtered rows `j − 1`, `j`, `j + 1` of a plane that
/// is filtered top to bottom: three row buffers, row `r` in slot
/// `r mod 3`, so the vertical pass reads rows that are still in cache
/// instead of a second whole-plane array.
struct RowRing {
    rows: [Vec<f32>; 3],
}

impl RowRing {
    fn new(w: usize) -> Self {
        Self {
            rows: std::array::from_fn(|_| vec![0f32; w]),
        }
    }

    fn slot(&mut self, r: usize) -> &mut [f32] {
        &mut self.rows[r % 3]
    }

    /// Rows above, at and below `j` of an `n`-row plane (replicated at
    /// the first and last row).
    fn window(&self, j: usize, n: usize) -> [&[f32]; 3] {
        [
            &self.rows[j.saturating_sub(1) % 3],
            &self.rows[j % 3],
            &self.rows[(j + 1).min(n - 1) % 3],
        ]
    }
}

/// The luma side of the luma-guided reconstruction (module docs),
/// common to every component subsampled at the same integer ratio
/// `rh × rv`: `Ŷ`, its 3 × 3 box mean and `var(Ŷ) + ε`.
#[derive(Debug)]
pub(crate) struct GuideLuma {
    cw: usize,
    ch: usize,
    rv: usize,
    w: usize,
    /// `Ŷ`: luma averaged over each chroma cell's real samples.
    yd: Vec<f32>,
    /// Box mean of `Ŷ` over each 3 × 3 neighbourhood.
    my: Vec<f32>,
    /// `max(mean(Ŷ²) − mean(Ŷ)², 0) + ε`.
    den: Vec<f32>,
    px: Vec<Phase>,
    ty: Vec<Tap>,
}

impl GuideLuma {
    /// `luma(y, row)` writes luma row `y` of the full-resolution guide
    /// (`w` samples) into `row`; the components it guides are `cw × ch`
    /// (`cw = ⌈w / rh⌉`, `ch = ⌈h / rv⌉`), `maxval` the sample range
    /// `2^P − 1`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        luma: &mut dyn FnMut(usize, &mut [f32]),
        cw: usize,
        ch: usize,
        rh: usize,
        rv: usize,
        w: usize,
        h: usize,
        maxval: u32,
    ) -> Self {
        let mut yd = vec![0f32; cw * ch];
        let mut row = vec![0f32; w];
        for y in 0..h {
            luma(y, &mut row);
            let dst = &mut yd[(y / rv).min(ch - 1) * cw..][..cw];
            if rh == 2 {
                let (pairs, tail) = row.as_chunks::<2>();
                for (d, p) in dst.iter_mut().zip(pairs) {
                    *d = (*d + p[0]) + p[1];
                }
                if let [t] = tail {
                    dst[cw - 1] += t;
                }
            } else {
                for (d, cell) in dst.iter_mut().zip(row.chunks(rh)) {
                    for &v in cell {
                        *d += v;
                    }
                }
            }
        }
        for (j, r) in yd.chunks_exact_mut(cw).enumerate() {
            let ny = rv.min(h - j * rv);
            for (i, v) in r.iter_mut().enumerate() {
                let nx = rh.min(w - i * rh);
                *v /= (nx * ny) as f32;
            }
        }
        Self::from_cell_means(yd, cw, ch, rh, rv, w, h, maxval)
    }

    /// [`GuideLuma::new`] over an 8-bit plane (`stride ≥ w`, at least
    /// `h` rows; `cw = ⌈w / rh⌉`, `ch = ⌈h / rv⌉`, `rh · rv ≤ 257`): the
    /// cell sums are accumulated as integers. A cell sum is below 2^24,
    /// so the `f32` accumulation of [`GuideLuma::new`] is exact too and
    /// both divide the same value.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_u8(
        data: &[u8],
        stride: usize,
        cw: usize,
        ch: usize,
        rh: usize,
        rv: usize,
        w: usize,
        h: usize,
    ) -> Self {
        debug_assert!(cw == w.div_ceil(rh) && ch == h.div_ceil(rv) && rh * rv <= 257);
        let mut yd = vec![0f32; cw * ch];
        let mut acc = vec![0u16; cw];
        for (j, dst) in yd.chunks_exact_mut(cw).enumerate() {
            acc.fill(0);
            let (y0, y1) = (j * rv, (j * rv + rv).min(h));
            for y in y0..y1 {
                let row = &data[y * stride..][..w];
                if rh == 2 {
                    let (pairs, tail) = row.as_chunks::<2>();
                    for (a, p) in acc.iter_mut().zip(pairs) {
                        *a += p[0] as u16 + p[1] as u16;
                    }
                    if let [t] = tail {
                        acc[cw - 1] += *t as u16;
                    }
                } else {
                    for (a, cell) in acc.iter_mut().zip(row.chunks(rh)) {
                        for &v in cell {
                            *a += v as u16;
                        }
                    }
                }
            }
            let ny = y1 - y0;
            let full = (rh * ny) as f32;
            for (v, &a) in dst[..cw - 1].iter_mut().zip(&acc) {
                *v = a as f32 / full;
            }
            let nx = w - (cw - 1) * rh;
            dst[cw - 1] = acc[cw - 1] as f32 / (nx * ny) as f32;
        }
        Self::from_cell_means(yd, cw, ch, rh, rv, w, h, 255)
    }

    /// The neighbourhood statistics of `Ŷ`: the box means of `Ŷ` and
    /// `Ŷ²`, each `((u + m) + d) / 9` of the horizontal sums
    /// `((l + m) + r)` (replicated edges).
    #[allow(clippy::too_many_arguments)]
    fn from_cell_means(
        yd: Vec<f32>,
        cw: usize,
        ch: usize,
        rh: usize,
        rv: usize,
        w: usize,
        h: usize,
        maxval: u32,
    ) -> Self {
        let scale = maxval as f32 / 255.0;
        let eps = GUIDED_EPS_8BIT * scale * scale;
        let mut my = vec![0f32; cw * ch];
        let mut den = vec![0f32; cw * ch];
        let (mut sy, mut syy) = (RowRing::new(cw), RowRing::new(cw));
        let mut sq = vec![0f32; cw];
        let mut filled = 0;
        let out = my.chunks_exact_mut(cw).zip(den.chunks_exact_mut(cw));
        for (j, (my, den)) in out.enumerate() {
            while filled <= (j + 1).min(ch - 1) {
                let row = &yd[filled * cw..][..cw];
                three_tap(row, sy.slot(filled), |a, b| a + b);
                for (s, &v) in sq.iter_mut().zip(row) {
                    *s = v * v;
                }
                three_tap(&sq, syy.slot(filled), |a, b| a + b);
                filled += 1;
            }
            let [u, m, d] = sy.window(j, ch);
            let [uu, mm, dd] = syy.window(j, ch);
            let sums = u.iter().zip(m).zip(d).zip(uu.iter().zip(mm).zip(dd));
            for ((my, den), (((&u, &m), &d), ((&uu, &mm), &dd))) in
                my.iter_mut().zip(den.iter_mut()).zip(sums)
            {
                let y = ((u + m) + d) * (1.0 / 9.0);
                let yy = ((uu + mm) + dd) * (1.0 / 9.0);
                *my = y;
                *den = fmax(yy - y * y, 0.0) + eps;
            }
        }
        Self {
            cw,
            ch,
            rv,
            w,
            yd,
            my,
            den,
            px: phases(rh),
            ty: taps(h, ch, 1, rv),
        }
    }

    /// Scratch for [`GuideLuma::detail_row`].
    pub(crate) fn scratch(&self) -> Vec<f32> {
        vec![0.0; self.cw]
    }

    /// The luma detail of output row `y`, `d[x] = Y(x, y) − lin(Ŷ)`,
    /// from the matching luma row. `col` is a caller-owned
    /// [`GuideLuma::scratch`].
    pub(crate) fn detail_row(&self, col: &mut [f32], y: usize, luma: &[f32], d: &mut [f32]) {
        let t = self.ty[y];
        let cw = self.cw;
        let (y0, y1) = (
            &self.yd[t.i0 as usize * cw..][..cw],
            &self.yd[t.i1 as usize * cw..][..cw],
        );
        let col = &mut col[..cw];
        for ((c, &a), &b) in col.iter_mut().zip(y0).zip(y1) {
            *c = lerp(a, b, t.f);
        }
        let cy = &*col;
        let rh = self.px.len();
        let last = cw - 1;
        let d = &mut d[..self.w];
        let luma = &luma[..self.w];
        let cell = |i: usize, d: &mut [f32]| {
            let (l, r) = (i.saturating_sub(1), (i + 1).min(last));
            let end = (i * rh + rh).min(d.len());
            for (x, p) in (i * rh..end).zip(&self.px) {
                let (i0, i1) = if p.left { (l, i) } else { (i, r) };
                d[x] = luma[x] - lerp(cy[i0], cy[i1], p.f);
            }
        };
        if rh == 2 && cw >= 2 {
            // 2:1: the outputs `2b + 1` (¼ of the way from sample `b` to
            // `b + 1`) and `2b + 2` (¾ of the way) on either side of
            // each cell boundary interpolate the same pair, so the loop
            // runs boundary by boundary over straight slices; the first
            // and last output replicate an edge sample (general form).
            let (fl, fr) = (self.px[0].f, self.px[1].f);
            let body = d[1..2 * last + 1].as_chunks_mut::<2>().0.iter_mut();
            let lbody = luma[1..2 * last + 1].as_chunks::<2>().0.iter();
            for ((o, l), (&y0, &y1)) in body.zip(lbody).zip(cy[..last].iter().zip(&cy[1..])) {
                o[0] = l[0] - lerp(y0, y1, fr);
                o[1] = l[1] - lerp(y0, y1, fl);
            }
            d[0] = luma[0] - lerp(cy[0], cy[0], fl);
            if let (Some(o), Some(&l)) = (d.get_mut(2 * last + 1), luma.get(2 * last + 1)) {
                *o = l - lerp(cy[last], cy[last], fr);
            }
            return;
        }
        for i in 0..cw {
            cell(i, d);
        }
    }
}

/// Luma-guided reconstruction of one component (module docs) on a
/// shared [`GuideLuma`].
#[derive(Debug)]
pub(crate) struct Guided {
    c: Vec<f32>,
    a: Vec<f32>,
    lo: Vec<f32>,
    hi: Vec<f32>,
}

impl Guided {
    /// `c` is the `cw × ch` component `g` was built for. The regression
    /// slope `a = cov(Ŷ, C) / (var(Ŷ) + ε)` over every 3 × 3
    /// neighbourhood (replicated edges) comes from the box means of `C`
    /// and `Ŷ · C`, formed like those of [`GuideLuma`]; `lo` / `hi` are
    /// the neighbourhood bounds of `C`.
    pub(crate) fn new(g: &GuideLuma, c: Vec<f32>) -> Self {
        let (cw, ch) = (g.cw, g.ch);
        debug_assert_eq!(c.len(), cw * ch);
        let n = cw * ch;
        let (mut a, mut lo, mut hi) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        let mut rings: [RowRing; 4] = std::array::from_fn(|_| RowRing::new(cw));
        let mut prod = vec![0f32; cw];
        let mut filled = 0;
        for j in 0..ch {
            while filled <= (j + 1).min(ch - 1) {
                let row = &c[filled * cw..][..cw];
                let yrow = &g.yd[filled * cw..][..cw];
                let [sc, syc, slo, shi] = &mut rings;
                three_tap(row, sc.slot(filled), |a, b| a + b);
                for ((p, &y), &c) in prod.iter_mut().zip(yrow).zip(row) {
                    *p = y * c;
                }
                three_tap(&prod, syc.slot(filled), |a, b| a + b);
                three_tap(row, slo.slot(filled), fmin);
                three_tap(row, shi.slot(filled), fmax);
                filled += 1;
            }
            let o = j * cw;
            let [u, m, d] = rings[0].window(j, ch);
            let [uy, my_, dy] = rings[1].window(j, ch);
            let (gm, gd) = (&g.my[o..][..cw], &g.den[o..][..cw]);
            let sums = u.iter().zip(m).zip(d).zip(uy.iter().zip(my_).zip(dy));
            for ((a, (&my, &den)), (((&u, &m), &d), ((&uy, &myc), &dy))) in
                a[o..][..cw].iter_mut().zip(gm.iter().zip(gd)).zip(sums)
            {
                let mc = ((u + m) + d) * (1.0 / 9.0);
                let myc = ((uy + myc) + dy) * (1.0 / 9.0);
                *a = (myc - my * mc) / den;
            }
            let [u, m, d] = rings[2].window(j, ch);
            for (((v, &u), &m), &d) in lo[o..][..cw].iter_mut().zip(u).zip(m).zip(d) {
                *v = fmin(fmin(u, m), d);
            }
            let [u, m, d] = rings[3].window(j, ch);
            for (((v, &u), &m), &d) in hi[o..][..cw].iter_mut().zip(u).zip(m).zip(d) {
                *v = fmax(fmax(u, m), d);
            }
        }
        Self { c, a, lo, hi }
    }

    /// Scratch for [`Guided::row_with`].
    pub(crate) fn scratch(&self, g: &GuideLuma) -> [Vec<f32>; 2] {
        std::array::from_fn(|_| vec![0.0; g.cw])
    }

    /// Output row `y` into `out`: `lin(C) + lin(a) · d`, clamped to the
    /// neighbourhood bounds, with `d` the row's [`GuideLuma::detail_row`].
    /// `cols` is a caller-owned [`Guided::scratch`], so several workers
    /// can produce rows at once.
    pub(crate) fn row_with(
        &self,
        g: &GuideLuma,
        cols: &mut [Vec<f32>; 2],
        y: usize,
        d: &[f32],
        out: &mut [f32],
    ) {
        let t = g.ty[y];
        let cw = g.cw;
        let (o0, o1) = (t.i0 as usize * cw, t.i1 as usize * cw);
        let (c0, c1) = (&self.c[o0..][..cw], &self.c[o1..][..cw]);
        let (a0, a1) = (&self.a[o0..][..cw], &self.a[o1..][..cw]);
        {
            let [col_c, col_a] = &mut *cols;
            let rows = c0.iter().zip(c1).zip(a0.iter().zip(a1));
            for ((cc, ca), ((&c0, &c1), (&a0, &a1))) in
                col_c[..cw].iter_mut().zip(col_a[..cw].iter_mut()).zip(rows)
            {
                *cc = lerp(c0, c1, t.f);
                *ca = lerp(a0, a1, t.f);
            }
        }
        let cy = (y / g.rv).min(g.ch - 1) * cw;
        let (lo, hi) = (&self.lo[cy..][..cw], &self.hi[cy..][..cw]);
        let (cc, ca) = (&cols[0][..cw], &cols[1][..cw]);
        let rh = g.px.len();
        let last = cw - 1;
        let out = &mut out[..g.w];
        let d = &d[..g.w];
        let cell = |i: usize, out: &mut [f32]| {
            let (l, r) = (i.saturating_sub(1), (i + 1).min(last));
            let end = (i * rh + rh).min(out.len());
            for (x, p) in (i * rh..end).zip(&g.px) {
                let (i0, i1) = if p.left { (l, i) } else { (i, r) };
                let cu = lerp(cc[i0], cc[i1], p.f);
                let au = lerp(ca[i0], ca[i1], p.f);
                out[x] = fmin(fmax(cu + au * d[x], lo[i]), hi[i]);
            }
        };
        if rh == 2 && cw >= 2 {
            // As in `GuideLuma::detail_row`: boundary by boundary, each
            // output clamped to the bounds of the cell it lies in.
            let (fl, fr) = (g.px[0].f, g.px[1].f);
            let body = out[1..2 * last + 1].as_chunks_mut::<2>().0.iter_mut();
            let dbody = d[1..2 * last + 1].as_chunks::<2>().0.iter();
            let cs = cc[..last].iter().zip(&cc[1..]);
            let as_ = ca[..last].iter().zip(&ca[1..]);
            let los = lo[..last].iter().zip(&lo[1..]);
            let his = hi[..last].iter().zip(&hi[1..]);
            for (((((o, d), (&c0, &c1)), (&a0, &a1)), (&lo0, &lo1)), (&hi0, &hi1)) in
                body.zip(dbody).zip(cs).zip(as_).zip(los).zip(his)
            {
                o[0] = fmin(fmax(lerp(c0, c1, fr) + lerp(a0, a1, fr) * d[0], lo0), hi0);
                o[1] = fmin(fmax(lerp(c0, c1, fl) + lerp(a0, a1, fl) * d[1], lo1), hi1);
            }
            cell(0, &mut out[..1]);
            if out.len() == 2 * last + 2 {
                cell(last, out);
            }
            return;
        }
        for i in 0..cw {
            cell(i, out);
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
            let mut fill = |y: usize, r: &mut [f32]| r.copy_from_slice(&luma[y * w..][..w]);
            let gl = GuideLuma::new(&mut fill, cw, ch, rh, rv, w, h, maxval);
            let g = Guided::new(&gl, c);
            let (mut col, mut cols) = (gl.scratch(), g.scratch(&gl));
            let mut detail = vec![0f32; w];
            for (y, dst) in out.chunks_exact_mut(w).enumerate() {
                gl.detail_row(&mut col, y, &luma[y * w..][..w], &mut detail);
                g.row_with(&gl, &mut cols, y, &detail, &mut row);
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

    /// The one-pass-per-plane form of the luma-guided reconstruction
    /// (whole-plane neighbourhood passes, luma statistics recomputed for
    /// every component), kept as the oracle of
    /// `guided_matches_the_plane_at_a_time_oracle`.
    mod oracle {
        use super::super::{GUIDED_EPS_8BIT, Phase, Tap, lerp, phases, taps};

        /// Luma-guided reconstruction of one component at an integer ratio
        /// `rh × rv` (module docs).
        #[derive(Debug)]
        pub(super) struct Guided {
            cw: usize,
            ch: usize,
            rv: usize,
            w: usize,
            c: Vec<f32>,
            yd: Vec<f32>,
            a: Vec<f32>,
            lo: Vec<f32>,
            hi: Vec<f32>,
            px: Vec<Phase>,
            ty: Vec<Tap>,
            col_c: Vec<f32>,
            col_y: Vec<f32>,
            col_a: Vec<f32>,
        }

        /// 3-tap horizontal pass of a separable neighbourhood filter with
        /// replicated edges: `dst[x] = op(op(row[x − 1], row[x]), row[x + 1])`.
        #[inline]
        fn three_tap(row: &[f32], dst: &mut [f32], op: impl Fn(f32, f32) -> f32) {
            let w = row.len();
            if w == 1 {
                dst[0] = op(op(row[0], row[0]), row[0]);
                return;
            }
            dst[0] = op(op(row[0], row[0]), row[1]);
            for x in 1..w - 1 {
                dst[x] = op(op(row[x - 1], row[x]), row[x + 1]);
            }
            dst[w - 1] = op(op(row[w - 2], row[w - 1]), row[w - 1]);
        }

        /// Separable 3 × 3 neighbourhood filter with replicated edges; `fin`
        /// maps the vertical combination.
        fn neighbourhood(
            src: &[f32],
            w: usize,
            h: usize,
            op: impl Fn(f32, f32) -> f32 + Copy,
            fin: impl Fn(f32) -> f32,
        ) -> Vec<f32> {
            let mut t = vec![0f32; w * h];
            for (row, dst) in src.chunks_exact(w).zip(t.chunks_exact_mut(w)) {
                three_tap(row, dst, op);
            }
            let mut o = vec![0f32; w * h];
            for y in 0..h {
                let (u, m, d) = (
                    &t[y.saturating_sub(1) * w..][..w],
                    &t[y * w..][..w],
                    &t[(y + 1).min(h - 1) * w..][..w],
                );
                for (((v, &a), &b), &c) in o[y * w..][..w].iter_mut().zip(u).zip(m).zip(d) {
                    *v = fin(op(op(a, b), c));
                }
            }
            o
        }

        /// The regression slope `a = cov(Ŷ, C) / (var(Ŷ) + ε)` over every 3 × 3
        /// neighbourhood (replicated edges): the four box means of `Ŷ`, `C`,
        /// `Ŷ²` and `Ŷ·C` in one horizontal and one vertical pass, each mean
        /// `((u + m) + d) / 9` of the horizontal sums `((l + m) + r)`.
        fn regression(yd: &[f32], c: &[f32], w: usize, h: usize, eps: f32) -> Vec<f32> {
            let n = w * h;
            let mut t = vec![[0f32; 4]; n];
            let at = |i: usize| {
                let (y, c) = (yd[i], c[i]);
                [y, c, y * y, y * c]
            };
            let sum3 = |a: [f32; 4], b: [f32; 4], c: [f32; 4]| -> [f32; 4] {
                std::array::from_fn(|k| (a[k] + b[k]) + c[k])
            };
            for y in 0..h {
                let o = y * w;
                for x in 0..w {
                    let (l, r) = (x.saturating_sub(1), (x + 1).min(w - 1));
                    t[o + x] = sum3(at(o + l), at(o + x), at(o + r));
                }
            }
            let mut a = vec![0f32; n];
            for y in 0..h {
                let (u, m, d) = (
                    &t[y.saturating_sub(1) * w..][..w],
                    &t[y * w..][..w],
                    &t[(y + 1).min(h - 1) * w..][..w],
                );
                for (((v, &p), &q), &r) in a[y * w..][..w].iter_mut().zip(u).zip(m).zip(d) {
                    let [my, mc, myy, myc] = sum3(p, q, r).map(|s| s * (1.0 / 9.0));
                    let var = (myy - my * my).max(0.0);
                    *v = (myc - my * mc) / (var + eps);
                }
            }
            a
        }

        impl Guided {
            /// The outputs of chroma cell `i` of the current row (general path).
            #[inline]
            fn cell(
                &self,
                [cc, cy, ca]: [&[f32]; 3],
                i: usize,
                out: &mut [f32],
                luma: &[f32],
                lo: f32,
                hi: f32,
            ) {
                let rh = self.px.len();
                let last = self.cw - 1;
                let (l, r) = (i.saturating_sub(1), (i + 1).min(last));
                let end = (i * rh + rh).min(out.len());
                for (x, p) in (i * rh..end).zip(&self.px) {
                    let (i0, i1) = if p.left { (l, i) } else { (i, r) };
                    let cu = lerp(cc[i0], cc[i1], p.f);
                    let yu = lerp(cy[i0], cy[i1], p.f);
                    let au = lerp(ca[i0], ca[i1], p.f);
                    out[x] = (cu + au * (luma[x] - yu)).max(lo).min(hi);
                }
            }

            /// `luma(y, row)` writes luma row `y` of the full-resolution guide
            /// (`w` samples) into `row`; `c` is the `cw × ch` component
            /// (`cw = ⌈w / rh⌉`, `ch = ⌈h / rv⌉`), `maxval` the sample range
            /// `2^P − 1`.
            #[allow(clippy::too_many_arguments)]
            pub(super) fn new(
                luma: &mut dyn FnMut(usize, &mut [f32]),
                c: Vec<f32>,
                cw: usize,
                ch: usize,
                rh: usize,
                rv: usize,
                w: usize,
                h: usize,
                maxval: u32,
            ) -> Self {
                debug_assert_eq!(c.len(), cw * ch);
                // Ŷ: luma averaged over each chroma cell's real samples.
                let mut yd = vec![0f32; cw * ch];
                let mut row = vec![0f32; w];
                for y in 0..h {
                    luma(y, &mut row);
                    let dst = &mut yd[(y / rv).min(ch - 1) * cw..][..cw];
                    if rh == 2 {
                        let (pairs, tail) = row.as_chunks::<2>();
                        for (d, p) in dst.iter_mut().zip(pairs) {
                            *d = (*d + p[0]) + p[1];
                        }
                        if let [t] = tail {
                            dst[cw - 1] += t;
                        }
                    } else {
                        for (d, cell) in dst.iter_mut().zip(row.chunks(rh)) {
                            for &v in cell {
                                *d += v;
                            }
                        }
                    }
                }
                for (j, r) in yd.chunks_exact_mut(cw).enumerate() {
                    let ny = rv.min(h - j * rv);
                    for (i, v) in r.iter_mut().enumerate() {
                        let nx = rh.min(w - i * rh);
                        *v /= (nx * ny) as f32;
                    }
                }
                let scale = maxval as f32 / 255.0;
                let eps = GUIDED_EPS_8BIT * scale * scale;
                let a = regression(&yd, &c, cw, ch, eps);
                // Neighbourhood bounds.
                let lo = neighbourhood(&c, cw, ch, f32::min, |v| v);
                let hi = neighbourhood(&c, cw, ch, f32::max, |v| v);
                Self {
                    cw,
                    ch,
                    rv,
                    w,
                    c,
                    yd,
                    a,
                    lo,
                    hi,
                    px: phases(rh),
                    ty: taps(h, ch, 1, rv),
                    col_c: vec![0.0; cw],
                    col_y: vec![0.0; cw],
                    col_a: vec![0.0; cw],
                }
            }

            /// Output row `y` into `out`, guided by the matching luma row.
            pub(super) fn row(&mut self, y: usize, luma: &[f32], out: &mut [f32]) {
                let mut cols = [
                    std::mem::take(&mut self.col_c),
                    std::mem::take(&mut self.col_y),
                    std::mem::take(&mut self.col_a),
                ];
                self.row_with(&mut cols, y, luma, out);
                let [c, yy, a] = cols;
                (self.col_c, self.col_y, self.col_a) = (c, yy, a);
            }

            /// [`Guided::row`] on shared state: `cols` is a caller-owned
            /// [`Guided::scratch`], so several workers can produce rows at once.
            pub(super) fn row_with(
                &self,
                cols: &mut [Vec<f32>; 3],
                y: usize,
                luma: &[f32],
                out: &mut [f32],
            ) {
                let t = self.ty[y];
                let cw = self.cw;
                let (o0, o1) = (t.i0 as usize * cw, t.i1 as usize * cw);
                let (c0, c1) = (&self.c[o0..][..cw], &self.c[o1..][..cw]);
                let (y0, y1) = (&self.yd[o0..][..cw], &self.yd[o1..][..cw]);
                let (a0, a1) = (&self.a[o0..][..cw], &self.a[o1..][..cw]);
                {
                    let [col_c, col_y, col_a] = &mut *cols;
                    for i in 0..cw {
                        col_c[i] = lerp(c0[i], c1[i], t.f);
                        col_y[i] = lerp(y0[i], y1[i], t.f);
                        col_a[i] = lerp(a0[i], a1[i], t.f);
                    }
                }
                let cy = (y / self.rv).min(self.ch - 1) * cw;
                let (lo, hi) = (&self.lo[cy..][..cw], &self.hi[cy..][..cw]);
                let (cc, cy_, ca) = (&cols[0][..cw], &cols[1][..cw], &cols[2][..cw]);
                let rh = self.px.len();
                let last = cw - 1;
                let out = &mut out[..self.w];
                let luma = &luma[..self.w];
                if rh == 2 && cw >= 3 {
                    // The two phases of 2:1 (¾ toward the left neighbour, ¼
                    // toward the right) over the interior cells, as straight
                    // slices so the loop vectorises; the edge cells take the
                    // general path below. Same arithmetic as the general path.
                    let px = |c: &[f32], i: usize| (c[i - 1], c[i], c[i + 1]);
                    let (fl, fr) = (self.px[0].f, self.px[1].f);
                    let body = out[2..2 * last].as_chunks_mut::<2>().0.iter_mut();
                    let lbody = luma[2..2 * last].as_chunks::<2>().0.iter();
                    for (k, (o, l)) in body.zip(lbody).enumerate() {
                        let i = k + 1;
                        let (cl, cm, cr) = px(cc, i);
                        let (yl, ym, yr) = px(cy_, i);
                        let (al, am, ar) = px(ca, i);
                        let (lo, hi) = (lo[i], hi[i]);
                        o[0] = (lerp(cl, cm, fl) + lerp(al, am, fl) * (l[0] - lerp(yl, ym, fl)))
                            .max(lo)
                            .min(hi);
                        o[1] = (lerp(cm, cr, fr) + lerp(am, ar, fr) * (l[1] - lerp(ym, yr, fr)))
                            .max(lo)
                            .min(hi);
                    }
                    for i in [0, last] {
                        self.cell([cc, cy_, ca], i, out, luma, lo[i], hi[i]);
                    }
                    return;
                }
                for (i, (dst, lrow)) in out.chunks_mut(rh).zip(luma.chunks(rh)).enumerate() {
                    let (l, r) = (i.saturating_sub(1), (i + 1).min(last));
                    let (lo, hi) = (lo[i], hi[i]);
                    for ((o, &yv), p) in dst.iter_mut().zip(lrow).zip(&self.px) {
                        let (i0, i1) = if p.left { (l, i) } else { (i, r) };
                        let cu = lerp(cc[i0], cc[i1], p.f);
                        let yu = lerp(cy_[i0], cy_[i1], p.f);
                        let au = lerp(ca[i0], ca[i1], p.f);
                        *o = (cu + au * (yv - yu)).max(lo).min(hi);
                    }
                }
            }
        }
    }

    /// Every output row of the guided reconstruction of `c` (`cw × ch`,
    /// ratio `rh × rv`) under `luma` (`w × h`).
    fn guided(
        luma: &[f32],
        c: Vec<f32>,
        (cw, ch): (usize, usize),
        (rh, rv): (usize, usize),
        (w, h): (usize, usize),
        maxval: u32,
    ) -> Vec<f32> {
        let mut fill = |y: usize, r: &mut [f32]| r.copy_from_slice(&luma[y * w..][..w]);
        let gl = GuideLuma::new(&mut fill, cw, ch, rh, rv, w, h, maxval);
        let g = Guided::new(&gl, c);
        let (mut col, mut cols) = (gl.scratch(), g.scratch(&gl));
        let mut detail = vec![0f32; w];
        let mut out = vec![0f32; w * h];
        for (y, row) in out.chunks_exact_mut(w).enumerate() {
            gl.detail_row(&mut col, y, &luma[y * w..][..w], &mut detail);
            g.row_with(&gl, &mut cols, y, &detail, row);
        }
        out
    }

    #[test]
    fn guided_matches_the_plane_at_a_time_oracle() {
        let mut seed = 0x2545_F491u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let ratios = [
            (2, 2),
            (2, 1),
            (1, 2),
            (4, 1),
            (4, 2),
            (3, 1),
            (1, 3),
            (4, 4),
        ];
        for case in 0..400 {
            let (rh, rv) = ratios[case % ratios.len()];
            let (w, h) = (1 + next() as usize % 41, 1 + next() as usize % 23);
            let (cw, ch) = (w.div_ceil(rh), h.div_ceil(rv));
            let maxval = if case % 5 == 4 { 4095 } else { 255 };
            // Smooth, noisy and saturated fields.
            let style = next() % 3;
            let mut field = |n: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| match style {
                        0 => (next() % (maxval + 1)) as f32,
                        1 => ((i as u32 * 3 + next() % 7) % (maxval + 1)) as f32,
                        _ => [0.0, maxval as f32][(next() % 2) as usize],
                    })
                    .collect()
            };
            let luma = field(w * h);
            let c = field(cw * ch);
            let got = guided(&luma, c.clone(), (cw, ch), (rh, rv), (w, h), maxval);
            let mut fill = |y: usize, r: &mut [f32]| r.copy_from_slice(&luma[y * w..][..w]);
            let mut o = oracle::Guided::new(&mut fill, c.clone(), cw, ch, rh, rv, w, h, maxval);
            let mut row = vec![0f32; w];
            for y in 0..h {
                o.row(y, &luma[y * w..][..w], &mut row);
                let same = row
                    .iter()
                    .zip(&got[y * w..][..w])
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "case {case}: {w}x{h} at {rh}x{rv}, row {y}");
            }
            // The 8-bit plane constructor sees the same cell means.
            if maxval == 255 {
                let bytes: Vec<u8> = luma.iter().map(|&v| v as u8).collect();
                let a = GuideLuma::from_u8(&bytes, w, cw, ch, rh, rv, w, h);
                let b = GuideLuma::new(&mut fill, cw, ch, rh, rv, w, h, 255);
                assert_eq!(a.yd, b.yd, "case {case}");
                assert_eq!(a.den, b.den, "case {case}");
            }
        }
    }

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
        let out = guided(&luma, c2.clone(), (4, 2), (2, 1), (w, h), 255)[..w].to_vec();
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
        let out = guided(&luma, vec![90.0; cw * ch], (cw, ch), (2, 2), (w, h), 255);
        assert!(out.iter().all(|&v| v == 90.0));
    }
}
