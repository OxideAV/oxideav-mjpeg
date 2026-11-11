//! Opt-in block-edge smoothing for decoded DCT components.
//!
//! Coarse quantisation leaves steps at the 8 × 8 block boundaries (the
//! §A.3.3 blocks are reconstructed independently). Across every block
//! edge of a component, the two samples on each side `p1 p0 | q0 q1` are
//! examined. When the step `d = q0 − p0` is smaller than the quantiser
//! step `Q` of the first AC coefficient along that axis (`Q(0, 1)` for
//! vertical edges, `Q(1, 0)` for horizontal ones), and both sides are
//! flat (`|p1 − p0|`, `|q1 − q0| < Q / 2`), the step is spread:
//! `p0 += d/4`, `q0 −= d/4`, `p1 += d/8`, `q1 −= d/8`. A step at least
//! one quantiser step high is real image content and is left alone.
//! Vertical edges go first, then horizontal ones on the result, both in
//! `f32` with one rounding at the end.

/// Smooth the block edges of one `w × h` component held in `buf` (row
/// pitch `stride`, block grid at multiples of 8). `qh` / `qv`: the
/// quantiser steps of the first horizontal / vertical AC coefficient.
pub(crate) fn deblock_u8(buf: &mut [u8], stride: usize, w: usize, h: usize, qh: f32, qv: f32) {
    // Nothing to smooth without a block edge inside the component (and
    // a frame header can declare no samples per line at all).
    if w == 0 || h == 0 || (w < 10 && h < 10) {
        return;
    }
    let mut f = vec![0f32; w * h];
    for (y, row) in f.chunks_exact_mut(w).enumerate() {
        for (d, &s) in row.iter_mut().zip(&buf[y * stride..y * stride + w]) {
            *d = s as f32;
        }
    }
    let edge = |p1: f32, p0: f32, q0: f32, q1: f32, t: f32| -> Option<[f32; 4]> {
        let d = q0 - p0;
        if d.abs() >= t || (p1 - p0).abs() >= t * 0.5 || (q1 - q0).abs() >= t * 0.5 {
            return None;
        }
        let k = d * 0.25;
        Some([p1 + k * 0.5, p0 + k, q0 - k, q1 - k * 0.5])
    };
    // Vertical edges (between columns 8m − 1 and 8m).
    let mut g = f.clone();
    for (src, dst) in f.chunks_exact(w).zip(g.chunks_exact_mut(w)) {
        let mut x = 8;
        while x + 1 < w {
            if let Some(o) = edge(src[x - 2], src[x - 1], src[x], src[x + 1], qh) {
                dst[x - 2..x + 2].copy_from_slice(&o);
            }
            x += 8;
        }
    }
    // Horizontal edges, on the vertically filtered samples.
    let src = g.clone();
    let mut y = 8;
    while y + 1 < h {
        for x in 0..w {
            let at = |r: usize| src[r * w + x];
            if let Some(o) = edge(at(y - 2), at(y - 1), at(y), at(y + 1), qv) {
                for (i, v) in o.into_iter().enumerate() {
                    g[(y - 2 + i) * w + x] = v;
                }
            }
        }
        y += 8;
    }
    for (y, row) in g.chunks_exact(w).enumerate() {
        for (d, &s) in buf[y * stride..y * stride + w].iter_mut().zip(row) {
            *d = crate::jpeg::fround::round_clamped(s, 255.0) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_steps_are_spread_and_large_ones_kept() {
        // 16 × 1: two flat blocks 100 | 104 (step below Q = 10) …
        // 100.5, 101, 103, 103.5, rounded half away from zero.
        let mut a: Vec<u8> = (0..16).map(|x| if x < 8 { 100 } else { 104 }).collect();
        deblock_u8(&mut a, 16, 16, 1, 10.0, 10.0);
        assert_eq!(&a[6..10], &[101, 101, 103, 104]);
        // … and a real edge 100 | 140 stays.
        let mut b: Vec<u8> = (0..16).map(|x| if x < 8 { 100 } else { 140 }).collect();
        let orig = b.clone();
        deblock_u8(&mut b, 16, 16, 1, 10.0, 10.0);
        assert_eq!(b, orig);
    }

    #[test]
    fn flat_input_is_untouched() {
        let mut a = vec![77u8; 24 * 24];
        deblock_u8(&mut a, 24, 24, 24, 16.0, 16.0);
        assert!(a.iter().all(|&v| v == 77));
    }
}
