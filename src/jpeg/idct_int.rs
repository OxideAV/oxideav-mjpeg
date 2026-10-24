//! Fixed-point ("accurate integer") 8×8 inverse DCT.
//!
//! # Derivation (T.81 §A.3.3)
//!
//! The IDCT is separable: `s(y, x) = Σ_v Σ_u ¼ C(u) C(v) S(v, u)
//! cos((2x+1)uπ/16) cos((2y+1)vπ/16)`, i.e. a 1-D transform
//! `t(x) = ½ Σ_u C(u) S(u) cos((2x+1)uπ/16)` along columns, then along
//! rows. Each 1-D pass here computes `√8 · t(x)`, which removes every
//! irrational factor from the `u = 0` and `u = 4` terms:
//!
//! ```text
//! √8 · t(x) = Σ_u K(u, x) · S(u),   K(u, x) = √2 · C(u) · cos((2x+1)uπ/16)
//! K(0, x) = 1,   K(4, x) = ±1,   K(u, x) = ±k_n  with  k_n = √2 · cos(nπ/16)
//! ```
//!
//! where `n` is `u(2x+1)` folded into `0..=8` (`cos(aπ/16)` for
//! `a = u(2x+1) mod 32`, using `cos(π − θ) = −cos θ` and
//! `cos(2π − θ) = cos θ`). The two passes together carry a factor
//! `√8 · √8 = 8 = 2^3`, removed by the final rounding shift.
//!
//! **Even half** (`u = 0, 2, 4, 6`), for `x = 0..=3`:
//!
//! ```text
//! x = 0:  S0 + S4 + (k2 S2 + k6 S6)       x = 3:  S0 + S4 − (k2 S2 + k6 S6)
//! x = 1:  S0 − S4 + (k6 S2 − k2 S6)       x = 2:  S0 − S4 − (k6 S2 − k2 S6)
//! ```
//!
//! The (S2, S6) rotation is formed with three products from the identity
//! `k2 S2 + k6 S6 = k6 (S2 + S6) + (k2 − k6) S2` and
//! `k6 S2 − k2 S6 = k6 (S2 + S6) − (k2 + k6) S6`.
//!
//! **Odd half** (`u = 1, 3, 5, 7`), for `x = 0..=3` (unfolding the angles
//! as above):
//!
//! ```text
//! o0 = k1 S1 + k3 S3 + k5 S5 + k7 S7
//! o1 = k3 S1 − k7 S3 − k1 S5 − k5 S7
//! o2 = k5 S1 − k1 S3 + k7 S5 + k3 S7
//! o3 = k7 S1 − k5 S3 + k3 S5 − k1 S7
//! ```
//!
//! `k3` appears exactly once in every row, so it is factored out as one
//! shared product `w = k3 (S1 + S3 + S5 + S7)`. Subtracting `w` leaves, in
//! each row, two off-diagonal coefficients that pair up across rows on
//! the four sums `S1 + S7`, `S3 + S5`, `S3 + S7`, `S1 + S5`:
//!
//! ```text
//! o0 − w: (k5 − k3) on S1 + S5 ,  (k7 − k3) on S1 + S7
//! o1 − w: −(k1 + k3) on S3 + S5 , −(k3 + k5) on S3 + S7
//! o2 − w: −(k1 + k3) on S3 + S5 ,  (k5 − k3) on S1 + S5
//! o3 − w:  (k7 − k3) on S1 + S7 , −(k3 + k5) on S3 + S7
//! ```
//!
//! and each row's diagonal input keeps a residual coefficient equal to its
//! target minus everything already assigned to it:
//! `k1 + k3 − k5 − k7` (S1 in o0), `k1 + k3 + k5 − k7` (S3 in o1),
//! `k1 + k3 − k5 + k7` (S5 in o2), `−k1 + k3 + k5 − k7` (S7 in o3).
//! Nine products in all. Outputs: `t(x) = e(x) + o(x)`,
//! `t(7 − x) = e(x) − o(x)`.
//!
//! # Fixed point
//!
//! Every constant is `round(2^13 · value)`; the `S0 ± S4` term is shifted
//! by 13 bits (exact). The first pass (columns) is shifted down by `2^11`
//! with round-half-up, keeping 2 extra fraction bits between passes; the
//! second pass (rows) by `2^(13 + 2 + 3) = 2^18`, then `+ 128` level shift
//! (§A.3.1) and the range limit of [`limit_sample`] (a clamp to `0..=255`
//! for any overshoot below 896 levels). Each pass is exact integer
//! arithmetic between the two roundings, so the result depends only on
//! the rounded constants above and the pass order — and with this choice
//! matches `djpeg`'s default output (libjpeg-turbo 3.2.0, observed
//! black-box) on every 8-bit stream of the comparison corpus.

/// `round(2^13 · √2 · cos(6π/16))`.
const K6: i64 = 4433;
/// `round(2^13 · √2 · (cos(2π/16) − cos(6π/16)))`.
const K2_MINUS_K6: i64 = 6270;
/// `round(2^13 · √2 · (cos(2π/16) + cos(6π/16)))`.
const K2_PLUS_K6: i64 = 15137;
/// `round(2^13 · √2 · cos(3π/16))`.
const K3: i64 = 9633;
/// `round(2^13 · √2 · (cos(3π/16) + cos(5π/16)))`.
const K3_PLUS_K5: i64 = 16069;
/// `round(2^13 · √2 · (cos(3π/16) − cos(5π/16)))`.
const K3_MINUS_K5: i64 = 3196;
/// `round(2^13 · √2 · (cos(3π/16) − cos(7π/16)))`.
const K3_MINUS_K7: i64 = 7373;
/// `round(2^13 · √2 · (cos(π/16) + cos(3π/16)))`.
const K1_PLUS_K3: i64 = 20995;
/// `round(2^13 · √2 · (cos(π/16) + cos(3π/16) − cos(5π/16) − cos(7π/16)))`.
const DIAG_S1: i64 = 12299;
/// `round(2^13 · √2 · (cos(π/16) + cos(3π/16) + cos(5π/16) − cos(7π/16)))`.
const DIAG_S3: i64 = 25172;
/// `round(2^13 · √2 · (cos(π/16) + cos(3π/16) − cos(5π/16) + cos(7π/16)))`.
const DIAG_S5: i64 = 16819;
/// `round(2^13 · √2 · (−cos(π/16) + cos(3π/16) + cos(5π/16) − cos(7π/16)))`.
const DIAG_S7: i64 = 2446;

/// Fraction bits of the constants.
const CONST_FRACTION: u32 = 13;
/// Extra fraction bits kept between the column and row passes.
const MID_FRACTION: u32 = 2;
/// Rounding shift after the column pass.
const COLUMN_SHIFT: u32 = CONST_FRACTION - MID_FRACTION;
/// Rounding shift after the row pass: constants, mid fraction, and the `2^3`
/// of the two `√8` pass gains.
const ROW_SHIFT: u32 = CONST_FRACTION + MID_FRACTION + 3;

/// One 1-D pass: `√8 · t(x) · 2^13` for the eight frequency inputs `f`
/// (see the module docs for the factorisation).
#[inline]
fn transform_1d(f: [i64; 8]) -> [i64; 8] {
    // Even half.
    let dc_plus_mid = (f[0] + f[4]) << CONST_FRACTION;
    let dc_minus_mid = (f[0] - f[4]) << CONST_FRACTION;
    let rot_shared = K6 * (f[2] + f[6]);
    let rot_cos = rot_shared + K2_MINUS_K6 * f[2]; // k2 S2 + k6 S6
    let rot_sin = rot_shared - K2_PLUS_K6 * f[6]; // k6 S2 − k2 S6
    let even = [
        dc_plus_mid + rot_cos,
        dc_minus_mid + rot_sin,
        dc_minus_mid - rot_sin,
        dc_plus_mid - rot_cos,
    ];

    // Odd half.
    let all_odd = K3 * (f[1] + f[3] + f[5] + f[7]);
    let on_1_5 = -K3_MINUS_K5 * (f[1] + f[5]);
    let on_1_7 = -K3_MINUS_K7 * (f[1] + f[7]);
    let on_3_5 = -K1_PLUS_K3 * (f[3] + f[5]);
    let on_3_7 = -K3_PLUS_K5 * (f[3] + f[7]);
    let odd = [
        DIAG_S1 * f[1] + on_1_5 + on_1_7 + all_odd,
        DIAG_S3 * f[3] + on_3_5 + on_3_7 + all_odd,
        DIAG_S5 * f[5] + on_3_5 + on_1_5 + all_odd,
        DIAG_S7 * f[7] + on_1_7 + on_3_7 + all_odd,
    ];

    let mut out = [0i64; 8];
    for x in 0..4 {
        out[x] = even[x] + odd[x];
        out[7 - x] = even[x] - odd[x];
    }
    out
}

/// `⌊v / 2^shift + ½⌋` (arithmetic shift after a half bias).
#[inline]
fn round_shift(v: i64, shift: u32) -> i64 {
    (v + (1i64 << (shift - 1))) >> shift
}

/// Map a level-shifted reconstructed value to an 8-bit sample. In range
/// it is the value itself; out of range it follows the periodic
/// saturation observed black-box in `djpeg` (libjpeg-turbo 3.2.0, DC
/// sweeps over `−1919..=2175`): with `u = v mod 2048`, `u < 256` is the
/// sample `u`, `256 ≤ u < 1152` saturates to 255 and `u ≥ 1152` to 0. Every
/// value a conforming stream can reach while the overshoot stays within
/// `−896..1152` therefore clamps; only gross overshoot wraps.
#[inline]
fn limit_sample(v: i64) -> u8 {
    let u = v.rem_euclid(2048);
    if u < 256 {
        u as u8
    } else if u < 1152 {
        255
    } else {
        0
    }
}

/// Dequantise `coef` (natural order) by `quant`, inverse-transform it and
/// store the level-shifted, clamped 8-bit samples at `(x0, y0)` of `out`
/// (row pitch `stride`).
pub(crate) fn idct8x8_store_u8(
    coef: &[i32; 64],
    quant: &[u16; 64],
    out: &mut [u8],
    stride: usize,
    x0: usize,
    y0: usize,
) {
    // Column pass: column u of the coefficient block → column u of `mid`.
    let mut mid = [0i64; 64];
    for u in 0..8 {
        let f: [i64; 8] = std::array::from_fn(|v| coef[v * 8 + u] as i64 * quant[v * 8 + u] as i64);
        let t = transform_1d(f);
        for y in 0..8 {
            mid[y * 8 + u] = round_shift(t[y], COLUMN_SHIFT);
        }
    }
    // Row pass.
    for y in 0..8 {
        let f: [i64; 8] = std::array::from_fn(|u| mid[y * 8 + u]);
        let t = transform_1d(f);
        let row = &mut out[(y0 + y) * stride + x0..][..8];
        for (o, &v) in row.iter_mut().zip(t.iter()) {
            *o = limit_sample(round_shift(v, ROW_SHIFT) + 128);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, SQRT_2};

    fn k(n: u32) -> f64 {
        SQRT_2 * (n as f64 * PI / 16.0).cos()
    }

    fn r(v: f64) -> i64 {
        (v * 8192.0).round() as i64
    }

    #[test]
    fn constants_are_the_rounded_cosine_combinations() {
        assert_eq!(K6, r(k(6)));
        assert_eq!(K2_MINUS_K6, r(k(2) - k(6)));
        assert_eq!(K2_PLUS_K6, r(k(2) + k(6)));
        assert_eq!(K3, r(k(3)));
        assert_eq!(K3_PLUS_K5, r(k(3) + k(5)));
        assert_eq!(K3_MINUS_K5, r(k(3) - k(5)));
        assert_eq!(K3_MINUS_K7, r(k(3) - k(7)));
        assert_eq!(K1_PLUS_K3, r(k(1) + k(3)));
        assert_eq!(DIAG_S1, r(k(1) + k(3) - k(5) - k(7)));
        assert_eq!(DIAG_S3, r(k(1) + k(3) + k(5) - k(7)));
        assert_eq!(DIAG_S5, r(k(1) + k(3) - k(5) + k(7)));
        assert_eq!(DIAG_S7, r(-k(1) + k(3) + k(5) - k(7)));
    }

    /// Each impulse response of the factorised pass is within one unit
    /// (per accumulated rounded constant) of `2^13 · √2 C(u) cos(...)`.
    #[test]
    fn factorisation_matches_the_cosine_matrix() {
        for u in 0..8 {
            let mut f = [0i64; 8];
            f[u] = 1;
            let t = transform_1d(f);
            for (x, &got) in t.iter().enumerate() {
                let cu = if u == 0 { 1.0 / SQRT_2 } else { 1.0 };
                let want = 8192.0 * SQRT_2 * cu * (((2 * x + 1) * u) as f64 * PI / 16.0).cos();
                assert!(
                    (got as f64 - want).abs() <= 2.5,
                    "u={u} x={x}: {got} vs {want:.2}"
                );
            }
        }
    }

    #[test]
    fn out_of_range_saturates_then_wraps() {
        assert_eq!(limit_sample(0), 0);
        assert_eq!(limit_sample(255), 255);
        assert_eq!(limit_sample(256), 255);
        assert_eq!(limit_sample(1151), 255);
        assert_eq!(limit_sample(1152), 0);
        assert_eq!(limit_sample(2048 + 7), 7);
        assert_eq!(limit_sample(-1), 0);
        assert_eq!(limit_sample(-896), 0);
        assert_eq!(limit_sample(-897), 255);
        assert_eq!(limit_sample(-1919), 129);
    }

    #[test]
    fn dc_only_block_is_flat() {
        let mut coef = [0i32; 64];
        coef[0] = 80; // DC 80 · q 1 → sample offset 80 / 8 = 10
        let mut out = [0u8; 64];
        idct8x8_store_u8(&coef, &[1; 64], &mut out, 8, 0, 0);
        assert!(out.iter().all(|&v| v == 138), "{out:?}");
    }

    /// Against the f64 evaluation of the §A.3.3 equation: never more than
    /// one level apart on random blocks.
    #[test]
    fn close_to_the_ideal_idct() {
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for _ in 0..200 {
            let mut coef = [0i32; 64];
            for c in coef.iter_mut() {
                *c = (next() % 61) as i32 - 30;
            }
            coef[0] = (next() % 801) as i32 - 400;
            let mut out = [0u8; 64];
            idct8x8_store_u8(&coef, &[1; 64], &mut out, 8, 0, 0);
            for y in 0..8 {
                for x in 0..8 {
                    let mut s = 0.0;
                    for v in 0..8 {
                        for u in 0..8 {
                            let cu = if u == 0 { 1.0 / SQRT_2 } else { 1.0 };
                            let cv = if v == 0 { 1.0 / SQRT_2 } else { 1.0 };
                            s += 0.25
                                * cu
                                * cv
                                * coef[v * 8 + u] as f64
                                * (((2 * x + 1) * u) as f64 * PI / 16.0).cos()
                                * (((2 * y + 1) * v) as f64 * PI / 16.0).cos();
                        }
                    }
                    let want = (s + 128.0).round().clamp(0.0, 255.0);
                    let got = out[y * 8 + x] as f64;
                    assert!((got - want).abs() <= 1.0, "({x},{y}) {got} vs {want}");
                }
            }
        }
    }
}
