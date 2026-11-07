//! Rate-distortion optimised quantisation of the AC coefficients of one
//! 8 × 8 block.
//!
//! T.81 §A.3.4 rounds every coefficient to the nearest multiple of its
//! quantiser step, which minimises the error of each coefficient on its
//! own but ignores what the levels cost to code. Here the AC levels of a
//! block are chosen together to minimise `D + λ · R`:
//!
//! * `D` is the squared error `Σ ((S/Q − Sq) · Q)²` — the pixel-domain
//!   squared error, the §A.3.3 transform being orthonormal;
//! * `R` is the bit count of the §F.1.2.2 sequential Huffman coding of
//!   the levels with the table in force: `RRRRSSSS` code lengths plus the
//!   `SSSS` magnitude bits, `ZRL` codes for runs past 15, and the `EOB`
//!   code when the block ends early.
//!
//! Each coefficient may keep its rounded level, take the level one step
//! nearer zero, or become zero. A dynamic programme over the positions
//! that can carry a non-zero level (indexed by the last non-zero
//! position) finds the exact minimum over those choices. `λ` is
//! [`LAMBDA_SCALE`] times the square of the mean AC quantiser step of
//! the component's table. The DC level keeps its §A.3.4 rounding (its
//! difference coding links neighbouring blocks).

use crate::jpeg::zigzag::ZIGZAG;

/// `λ / mean(Q_ac)²`, chosen by sweeping the rate-PSNR curve (README,
/// *Encoder quality*).
pub(crate) const LAMBDA_SCALE: f32 = 0.0085;

/// Code lengths of an AC table, indexed by the `RRRRSSSS` symbol (`0` =
/// no code).
#[derive(Clone, Debug)]
pub(crate) struct AcRate {
    /// Bits per `RRRRSSSS` symbol including its `SSSS` magnitude bits
    /// (`1e9` for a symbol the table cannot code, so it is never
    /// chosen).
    bits: [f32; 256],
    min_len: f32,
}

impl AcRate {
    /// From a §C.2 `BITS` / `HUFFVAL` pair.
    pub(crate) fn new(bits: &[u8; 16], vals: &[u8]) -> Self {
        let mut len = [0u8; 256];
        let mut i = 0usize;
        for (l, &n) in bits.iter().enumerate() {
            for _ in 0..n {
                if let Some(&v) = vals.get(i) {
                    len[v as usize] = l as u8 + 1;
                }
                i += 1;
            }
        }
        let bits = std::array::from_fn(|rs| match len[rs] {
            0 => 1.0e9,
            l => (l as u32 + (rs as u32 & 0x0F)) as f32,
        });
        let min_len = len.iter().filter(|&&l| l > 0).min().copied().unwrap_or(1) as f32;
        Self { bits, min_len }
    }

    /// Bits of symbol `rs` and its magnitude bits.
    #[inline(always)]
    fn bits(&self, rs: usize) -> f32 {
        self.bits[rs]
    }
}

/// Magnitude category `SSSS` of a non-zero level.
#[inline]
fn category(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

/// Choose the AC levels of `out` (natural order; `out[0]` is left as
/// is) from the unquantised `ratio[k] = S_k / Q_k` (natural order), the
/// steps `q` and the rate model, minimising `D + λ · R`.
pub(crate) fn quantise_ac(
    ratio: &[f32; 64],
    q: &[u16; 64],
    rate: &AcRate,
    lambda: f32,
    out: &mut [i32; 64],
) {
    // Zigzag positions 1..=63: error of coding zero, and the candidate
    // non-zero levels with their errors.
    let mut zero = [0f32; 64];
    let mut cand = [[(0i32, 0f32); 2]; 64];
    let mut ncand = [0u8; 64];
    let mut live = [0u8; 64];
    let mut nlive = 0usize;
    for k in 1..64 {
        let n = ZIGZAG[k];
        let r = ratio[n];
        let step = q[n] as f32;
        zero[k] = (r * step) * (r * step);
        let l = r.round() as i32;
        if l != 0 {
            let e = (r - l as f32) * step;
            cand[k][0] = (l, e * e);
            ncand[k] = 1;
            if l.abs() >= 2 {
                let l2 = l - l.signum();
                let e2 = (r - l2 as f32) * step;
                cand[k][1] = (l2, e2 * e2);
                ncand[k] = 2;
            }
            live[nlive] = k as u8;
            nlive += 1;
        }
        out[n] = 0;
    }
    if nlive == 0 {
        return;
    }
    // Prefix sums of the zero-coding error (zigzag order).
    let mut zsum = [0f32; 65];
    for k in 1..64 {
        zsum[k + 1] = zsum[k] + zero[k];
    }
    let zrange = |a: usize, b: usize| if b > a { zsum[b] - zsum[a] } else { 0.0 };
    let zrl = rate.bits(0xF0);
    // Every non-zero level costs at least the shortest code plus one
    // magnitude bit: a lower bound for pruning predecessors.
    let min_rate = lambda * (rate.min_len + 1.0);
    // best[i]: least cost of positions 1..=live[i] with live[i] the last
    // non-zero; from[i]: predecessor (usize::MAX = none), pick[i]: level.
    // With `a[j] = best[j] − zsum[live[j] + 1]` (and `0` for "no
    // predecessor"), predecessor `j` costs `a[j] + zsum[k] + λ · R` at
    // `k`; `pm[j] = min(0, a[0..=j])` bounds every predecessor at or
    // before `j`, so the scan stops once that bound cannot win.
    let mut best = [0f32; 64];
    let mut from = [usize::MAX; 64];
    let mut pick = [0i32; 64];
    let mut pm = [0f32; 64];
    for i in 0..nlive {
        let k = live[i] as usize;
        let mut b = f32::INFINITY;
        let (mut bf, mut bl) = (usize::MAX, 0i32);
        let cands = &cand[k][..ncand[k] as usize];
        let mut consider = |p: usize, base: f32, j: usize, b: &mut f32| {
            let run = k - p - 1;
            let pre = base + zrange(p + 1, k) + lambda * zrl * (run / 16) as f32;
            for &(l, d) in cands {
                let rs = ((run % 16) << 4) | category(l) as usize;
                let c = pre + d + lambda * rate.bits(rs);
                if c < *b {
                    *b = c;
                    bf = j;
                    bl = l;
                }
            }
        };
        for j in (0..i).rev() {
            if pm[j] + zsum[k] + min_rate >= b {
                break;
            }
            consider(live[j] as usize, best[j], j, &mut b);
        }
        // No predecessor (bounded by `pm`'s 0 term, so always tried).
        if zsum[k] - zsum[1] + min_rate < b {
            consider(0, 0.0, usize::MAX, &mut b);
        }
        best[i] = b;
        from[i] = bf;
        pick[i] = bl;
        let a = b - zsum[k + 1];
        pm[i] = if i == 0 { a.min(0.0) } else { pm[i - 1].min(a) };
    }
    // End: all zero, or last non-zero at some live position, then EOB
    // unless that position is 63.
    let eob = lambda * rate.bits(0x00);
    let mut end_cost = zrange(1, 64) + eob;
    let mut end = usize::MAX;
    for i in 0..nlive {
        let k = live[i] as usize;
        let c = best[i] + zrange(k + 1, 64) + if k < 63 { eob } else { 0.0 };
        if c < end_cost {
            end_cost = c;
            end = i;
        }
    }
    let mut i = end;
    while i != usize::MAX {
        out[ZIGZAG[live[i] as usize]] = pick[i];
        i = from[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::huffman::{STD_AC_LUMA_BITS, STD_AC_LUMA_VALS};

    fn rate() -> AcRate {
        AcRate::new(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS)
    }

    #[test]
    fn zero_lambda_is_plain_rounding() {
        let mut ratio = [0f32; 64];
        for (k, r) in ratio.iter_mut().enumerate() {
            *r = ((k as f32 * 1.37).sin() * 6.0) + 0.1;
        }
        let q = [10u16; 64];
        let mut out = [0i32; 64];
        quantise_ac(&ratio, &q, &rate(), 0.0, &mut out);
        for k in 1..64 {
            assert_eq!(out[k], ratio[k].round() as i32, "{k}");
        }
    }

    #[test]
    fn isolated_small_level_far_out_is_dropped() {
        // One ±1 level at the end of the block costs a long run code; a
        // moderate λ drops it, a large level stays.
        let mut ratio = [0f32; 64];
        ratio[ZIGZAG[1]] = 7.2;
        ratio[ZIGZAG[60]] = 0.6;
        let q = [16u16; 64];
        let mut out = [0i32; 64];
        quantise_ac(&ratio, &q, &rate(), 0.06 * 256.0, &mut out);
        assert_eq!(out[ZIGZAG[1]], 7);
        assert_eq!(out[ZIGZAG[60]], 0);
    }

    #[test]
    fn levels_never_grow() {
        let mut seed = 7u32;
        for _ in 0..500 {
            let mut ratio = [0f32; 64];
            for r in ratio.iter_mut() {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *r = (seed % 4001) as f32 / 200.0 - 10.0;
            }
            let q = [8u16; 64];
            let mut out = [0i32; 64];
            quantise_ac(&ratio, &q, &rate(), 20.0, &mut out);
            for k in 1..64 {
                let l = ratio[k].round() as i32;
                assert!(out[k] == 0 || out[k].signum() == l.signum());
                assert!(out[k].abs() <= l.abs());
            }
        }
    }
}
