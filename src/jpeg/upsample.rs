//! Triangular ("linear") chroma upsampling.
//!
//! T.871 §9 sites a subsampled sample at the centre of the luma samples it
//! covers: for a 2:1 axis, chroma sample `i` sits half-way between output
//! samples `2i` and `2i + 1`. Output sample `2i` therefore lies a quarter
//! sample left of chroma sample `i` and three quarters right of `i − 1`, so
//! linear interpolation gives `¾ c[i] + ¼ c[i − 1]`, and output `2i + 1`
//! gives `¾ c[i] + ¼ c[i + 1]`. Off the ends of the component (beyond its
//! §A.1.1 extent `⌈X · Hc / Hmax⌉`) the nearest real sample is replicated.
//!
//! A 2 × 2 ratio applies the same weights along both axes as a separable
//! filter: the vertical pass forms `3 · near + far` without rounding, the
//! horizontal pass weights those sums `3 : 1` and divides by 16 once.
//!
//! The integer rounding, observed black-box against `djpeg`'s default
//! output (libjpeg-turbo 3.2.0) on a corpus of every size 1–11 px and
//! larger frames, alternates between the two output phases so the
//! rounding error does not drift: one-axis ratios add `1` for even and
//! `2` for odd outputs before dividing by 4; the 2 × 2 ratio adds `8` for
//! even and `7` for odd output columns before dividing by 16. A
//! horizontally subsampled component (2:1 or 2 × 2) whose coded width is
//! at most two samples is replicated instead, on both axes (also
//! observed: `djpeg` does not interpolate such narrow components); the
//! vertical-only 1:2 ratio interpolates at every width.

/// One subsampled component: `width × height` real samples read through
/// `get(x, y)`.
pub(crate) struct Component<'a> {
    pub width: usize,
    pub height: usize,
    pub get: &'a dyn Fn(usize, usize) -> u32,
}

/// Bias pair for one-axis 2:1 interpolation (`/ 4`), by output phase.
const ONE_AXIS_BIAS: [u32; 2] = [1, 2];
/// Bias pair for the 2 × 2 ratio (`/ 16`), by output column phase.
const TWO_AXIS_BIAS: [u32; 2] = [8, 7];

/// Resample `c` to `out_w × out_h` with ratio `(rh, rv)` (output samples
/// per chroma sample along each axis). Returns `None` when the ratio has
/// no triangular kernel (anything but 2:1, 1:2 and 2 × 2) or a
/// horizontally subsampled component is too narrow — the caller
/// replicates instead.
pub(crate) fn triangular(
    c: &Component<'_>,
    rh: usize,
    rv: usize,
    out_w: usize,
    out_h: usize,
) -> Option<Vec<u32>> {
    if !matches!((rh, rv), (2, 1) | (1, 2) | (2, 2)) || c.width == 0 || c.height == 0 {
        return None;
    }
    if rh == 2 && c.width <= 2 {
        return None;
    }
    let at = |x: usize, y: usize| (c.get)(x.min(c.width - 1), y.min(c.height - 1));
    // Neighbour index along a 2:1 axis for output position `p`: the chroma
    // sample on the far side of the output sample.
    let far = |p: usize, n: usize| {
        if p.is_multiple_of(2) {
            n.saturating_sub(1)
        } else {
            n + 1
        }
    };
    let mut out = vec![0u32; out_w * out_h];
    for y in 0..out_h {
        let cy = y / rv;
        for x in 0..out_w {
            let cx = x / rh;
            out[y * out_w + x] = match (rh, rv) {
                (2, 1) => (3 * at(cx, cy) + at(far(x, cx), cy) + ONE_AXIS_BIAS[x % 2]) >> 2,
                (1, 2) => (3 * at(cx, cy) + at(cx, far(y, cy)) + ONE_AXIS_BIAS[y % 2]) >> 2,
                _ => {
                    let fy = far(y, cy);
                    let column = |xx: usize| 3 * at(xx, cy) + at(xx, fy);
                    (3 * column(cx) + column(far(x, cx)) + TWO_AXIS_BIAS[x % 2]) >> 4
                }
            };
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_weights_and_edges() {
        let v = [0u32, 100, 200];
        let get = |x: usize, _: usize| v[x];
        let c = Component {
            width: 3,
            height: 1,
            get: &get,
        };
        let o = triangular(&c, 2, 1, 6, 1).unwrap();
        // edge, ¾·0+¼·100, ¼·0+¾·100, ¾·100+¼·200, ¼·100+¾·200, edge
        assert_eq!(o, vec![0, 25, 75, 125, 175, 200]);
    }

    #[test]
    fn narrow_components_are_left_to_replication() {
        let get = |_: usize, _: usize| 7u32;
        let c = Component {
            width: 2,
            height: 4,
            get: &get,
        };
        assert!(triangular(&c, 2, 2, 4, 8).is_none());
        assert!(triangular(&c, 2, 1, 4, 4).is_none());
        assert!(triangular(&c, 4, 1, 8, 4).is_none());
        // Vertical-only interpolation does not care about the width.
        assert!(triangular(&c, 1, 2, 2, 8).is_some());
    }

    #[test]
    fn flat_input_stays_flat() {
        let get = |_: usize, _: usize| 133u32;
        let c = Component {
            width: 5,
            height: 4,
            get: &get,
        };
        for (rh, rv) in [(2, 1), (1, 2), (2, 2)] {
            let o = triangular(&c, rh, rv, 5 * rh - 1, 4 * rv - 1).unwrap();
            assert!(o.iter().all(|&v| v == 133), "{rh}x{rv}");
        }
    }
}
