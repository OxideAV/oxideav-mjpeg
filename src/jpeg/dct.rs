//! 8×8 forward and inverse Discrete Cosine Transform (T.81 §A.3.3).
//!
//! The inverse transform is separable and factorised by the symmetry of
//! the DCT basis, `t[k][7 − n] = (−1)^k t[k][n]`: each 1-D pass splits
//! into an even half (coefficients 0, 2, 4, 6, itself split once more)
//! and an odd half (1, 3, 5, 7), 22 multiplications instead of 64. Both
//! passes run eight independent 1-D transforms side by side, one per
//! `f32` lane, so they vectorise. Accuracy: against the `f64` evaluation
//! of the §A.3.3 definition, rounded, it differs on 13 of 2 073 600
//! samples of a 1080p quality-85 frame (the previous direct
//! cosine-matrix product: 17; the accurate-integer kernel: 39 162).
//!
//! The forward transform is the direct cosine-matrix product.

use std::f32::consts::PI;
use std::sync::OnceLock;

/// `t[k][n] = C(k)/2 * cos((2n+1)kπ/16)`, stored row-major. With
/// `C(0) = 1/√2` and `C(k>0) = 1`. Applying this matrix along each axis
/// implements the standard JPEG FDCT/IDCT normalisation (factor of 1/4
/// after two 1-D passes, with the `C(u)C(v)` factor absorbed).
fn cos_table() -> &'static [[f32; 8]; 8] {
    static T: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0.0f32; 8]; 8];
        for k in 0..8 {
            let c_k = if k == 0 {
                (1.0_f32 / 2.0_f32).sqrt()
            } else {
                1.0
            };
            for n in 0..8 {
                t[k][n] = 0.5 * c_k * ((2 * n + 1) as f32 * k as f32 * PI / 16.0).cos();
            }
        }
        t
    })
}

/// `½ · cos(kπ/16)`, `k = 1..=7` (index 0 unused), and `½ · C(0) = 1/(2√2)`.
const C1: f32 = 0.490_392_64;
const C2: f32 = 0.461_939_77;
const C3: f32 = 0.415_734_8;
const C4: f32 = 0.353_553_38;
const C5: f32 = 0.277_785_12;
const C6: f32 = 0.191_341_72;
const C7: f32 = 0.097_545_16;

type Lanes = [[f32; 8]; 8];

/// Eight 1-D inverse transforms side by side: `x[n][j] = Σk t[k][n] ·
/// v[k][j]` for every lane `j`, by the even / odd factorisation.
#[inline(always)]
fn idct_lanes(v: &Lanes) -> Lanes {
    let mut x = [[0f32; 8]; 8];
    for j in 0..8 {
        let (v0, v1, v2, v3) = (v[0][j], v[1][j], v[2][j], v[3][j]);
        let (v4, v5, v6, v7) = (v[4][j], v[5][j], v[6][j], v[7][j]);
        // Even-even: k = 0, 4 (both ½·C(0) up to sign).
        let ee0 = C4 * (v0 + v4);
        let ee1 = C4 * (v0 - v4);
        // Even-odd: k = 2, 6.
        let eo0 = C2 * v2 + C6 * v6;
        let eo1 = C6 * v2 - C2 * v6;
        let e0 = ee0 + eo0;
        let e3 = ee0 - eo0;
        let e1 = ee1 + eo1;
        let e2 = ee1 - eo1;
        // Odd: k = 1, 3, 5, 7.
        let o0 = C1 * v1 + C3 * v3 + C5 * v5 + C7 * v7;
        let o1 = C3 * v1 - C7 * v3 - C1 * v5 - C5 * v7;
        let o2 = C5 * v1 - C1 * v3 + C7 * v5 + C3 * v7;
        let o3 = C7 * v1 - C5 * v3 + C3 * v5 - C1 * v7;
        x[0][j] = e0 + o0;
        x[7][j] = e0 - o0;
        x[1][j] = e1 + o1;
        x[6][j] = e1 - o1;
        x[2][j] = e2 + o2;
        x[5][j] = e2 - o2;
        x[3][j] = e3 + o3;
        x[4][j] = e3 - o3;
    }
    x
}

/// [`idct_lanes`] when rows 4..=7 of `v` are zero: the same sums with
/// the terms of `v4 … v7` left out. A left-out term is a product with
/// zero added to (or subtracted from) a partial sum, which leaves that
/// sum as it is, so every lane gets the value [`idct_lanes`] gives it
/// (the sign of a zero aside, and a zero's sign never reaches a sample:
/// the level shift adds 128).
#[inline(always)]
fn idct_lanes_low(v: &Lanes) -> Lanes {
    let mut x = [[0f32; 8]; 8];
    for j in 0..8 {
        let (v0, v1, v2, v3) = (v[0][j], v[1][j], v[2][j], v[3][j]);
        // Even-even: k = 0 alone, so both halves are `C4 · v0`.
        let ee = C4 * v0;
        // Even-odd: k = 2.
        let eo0 = C2 * v2;
        let eo1 = C6 * v2;
        let e0 = ee + eo0;
        let e3 = ee - eo0;
        let e1 = ee + eo1;
        let e2 = ee - eo1;
        // Odd: k = 1, 3.
        let o0 = C1 * v1 + C3 * v3;
        let o1 = C3 * v1 - C7 * v3;
        let o2 = C5 * v1 - C1 * v3;
        let o3 = C7 * v1 - C5 * v3;
        x[0][j] = e0 + o0;
        x[7][j] = e0 - o0;
        x[1][j] = e1 + o1;
        x[6][j] = e1 - o1;
        x[2][j] = e2 + o2;
        x[5][j] = e2 - o2;
        x[3][j] = e3 + o3;
        x[4][j] = e3 - o3;
    }
    x
}

#[inline(always)]
fn transpose(m: &Lanes) -> Lanes {
    std::array::from_fn(|r| std::array::from_fn(|c| m[c][r]))
}

/// Inverse DCT of an 8×8 block (natural order, already dequantised),
/// in place: `s(y, x) = Σv Σu t[v][y] · t[u][x] · S(v, u)` — the
/// §A.3.3 IDCT (`¼ C(u) C(v)` with the `½ C(k)` folded into each axis).
///
/// A block whose AC coefficients are all zero is the constant
/// `C(0)² · S(0, 0) / 4` (`= S(0, 0) / 8`).
pub fn idct8x8(block: &mut [f32; 64]) {
    if block[1..].iter().all(|&c| c == 0.0) {
        let v = block[0] * 0.125;
        block.fill(v);
        return;
    }
    // Rows of the input are vertical frequencies `v`, columns `u`. Pass
    // one transforms down the columns (lanes = `u`): `c[y][u]`.
    let rows: Lanes = std::array::from_fn(|r| block[r * 8..r * 8 + 8].try_into().unwrap());
    let c = idct_lanes(&rows);
    // Pass two along the rows: lanes = `y` after a transpose.
    let out = transpose(&idct_lanes(&transpose(&c)));
    for (dst, row) in block.as_chunks_mut::<8>().0.iter_mut().zip(&out) {
        dst.copy_from_slice(row);
    }
}

/// Where a block's non-zero coefficients are: bit `8 + v` is set when
/// row `v` (vertical frequency) holds one, bit `u` when column `u`
/// does. A coefficient at natural index `i` contributes
/// [`extent_bit`]`(i)`; [`FULL_EXTENT`] claims nothing.
pub(crate) type Extent = u16;

/// The [`Extent`] of a block about which nothing is known.
pub(crate) const FULL_EXTENT: Extent = 0xFFFF;

/// The [`Extent`] bits of the coefficient at natural index `i`.
#[inline(always)]
pub(crate) const fn extent_bit(i: usize) -> Extent {
    (1 << (8 + (i >> 3))) | (1 << (i & 7))
}

/// The [`Extent`] of a block whose last non-zero coefficient is at
/// zig-zag position `k` (Figure A.6): the bits of positions `0..=k`.
pub(crate) const ZIGZAG_EXTENT_UP_TO: [Extent; 64] = {
    let mut t = [0; 64];
    let mut acc = 0;
    let mut k = 0;
    while k < 64 {
        acc |= extent_bit(crate::jpeg::zigzag::ZIGZAG[k]);
        t[k] = acc;
        k += 1;
    }
    t
};

/// The [`Extent`] of a natural-order block, by inspection.
#[inline]
pub(crate) fn extent_of(block: &[i32; 64]) -> Extent {
    let mut cols = [0i32; 8];
    let mut ext = 0;
    for (v, row) in block.as_chunks::<8>().0.iter().enumerate() {
        let mut any = 0;
        for (c, &x) in cols.iter_mut().zip(row) {
            *c |= x;
            any |= x;
        }
        ext |= Extent::from(any != 0) << (8 + v);
    }
    for (u, &c) in cols.iter().enumerate() {
        ext |= Extent::from(c != 0) << u;
    }
    ext
}

/// [`idct8x8`] of a block whose non-zero coefficients lie within
/// `extent`, with the work its zero rows and columns do not need left
/// out; every sample is the value [`idct8x8`] produces. `extent` may
/// claim more rows and columns than hold a coefficient, with one
/// exception: a block whose AC coefficients are all zero must come with
/// no bit beyond [`extent_bit`]`(0)`, because [`idct8x8`] gives such a
/// block `S(0, 0) / 8` and the two passes give it `C4 · (C4 · S(0, 0))`,
/// which is not always the same `f32`.
///
/// * Only the DC coefficient: the constant `S(0, 0) / 8`.
/// * No coefficient in rows 4..=7: the column pass leaves those terms
///   out ([`idct_lanes_low`]); none in columns 4..=7: the row pass does
///   (its inputs in those columns are the column pass's zeros).
#[inline]
pub(crate) fn idct8x8_within(block: &mut [f32; 64], extent: Extent) {
    if extent & !extent_bit(0) == 0 {
        let v = block[0] * 0.125;
        block.fill(v);
        return;
    }
    let rows: Lanes = std::array::from_fn(|r| block[r * 8..r * 8 + 8].try_into().unwrap());
    let c = if extent & 0xF000 == 0 {
        idct_lanes_low(&rows)
    } else {
        idct_lanes(&rows)
    };
    let ct = transpose(&c);
    let o = if extent & 0x00F0 == 0 {
        idct_lanes_low(&ct)
    } else {
        idct_lanes(&ct)
    };
    let out = transpose(&o);
    for (dst, row) in block.as_chunks_mut::<8>().0.iter_mut().zip(&out) {
        dst.copy_from_slice(row);
    }
}

/// Transpose of [`cos_table`]: `tt[n][k] = t[k][n]`. Gives the forward
/// transform's row pass a contiguous 8-lane row per input sample; the
/// values are the very same f32s, only the storage order differs.
fn cos_table_t() -> &'static [[f32; 8]; 8] {
    static TT: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    TT.get_or_init(|| {
        let t = cos_table();
        let mut tt = [[0.0f32; 8]; 8];
        for k in 0..8 {
            for n in 0..8 {
                tt[n][k] = t[k][n];
            }
        }
        tt
    })
}

/// Forward DCT of an 8×8 block in natural order, in-place. Caller should
/// subtract 128 from sample values before calling so input is centred on 0.
///
/// Both passes run their reduction with the summation index `n` outermost
/// over a whole-block accumulator: for every output the terms still land
/// in ascending-n order starting from +0.0 — the exact accumulation
/// sequence of the original scalar reduction, so results are bit-identical
/// — while each n-step is 64 independent elementwise multiply-adds. The
/// full 8×8 accumulator gives the compiler 16 independent vector chains
/// (throughput-bound) instead of one serial reduction per output
/// (latency-bound); the row pass reads the transposed table so every
/// per-step row is contiguous.
pub fn fdct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let tt = cos_table_t();

    // Row-wise 1-D forward: tmp[y][k] = Σn t[k][n] * block[y][n], with n
    // outermost and tt[n][k] = t[k][n] providing the contiguous 8-lane row.
    let mut tmp = [0.0f32; 64];
    for n in 0..8 {
        let tn = &tt[n];
        for y in 0..8 {
            let c = block[y * 8 + n];
            for k in 0..8 {
                tmp[y * 8 + k] += tn[k] * c;
            }
        }
    }
    // Column-wise: block[k][x] = Σn t[k][n] * tmp[n][x], with n outermost;
    // `tmp[n][·]` rows are contiguous.
    let mut out = [0.0f32; 64];
    for n in 0..8 {
        let tr = &tmp[n * 8..n * 8 + 8];
        for (k, tk) in t.iter().enumerate() {
            let c = tk[n];
            for x in 0..8 {
                out[k * 8 + x] += c * tr[x];
            }
        }
    }
    *block = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_of_dct_is_identity() {
        let mut block = [0.0f32; 64];
        for i in 0..64 {
            block[i] = ((i * 7) % 255) as f32 - 128.0;
        }
        let original = block;
        fdct8x8(&mut block);
        idct8x8(&mut block);
        for i in 0..64 {
            assert!(
                (block[i] - original[i]).abs() < 1e-2,
                "mismatch at {i}: got {}, want {}",
                block[i],
                original[i]
            );
        }
    }

    #[test]
    fn factorised_idct_matches_the_definition() {
        // Pseudo-random coefficient blocks against the f64 §A.3.3 sum.
        let mut seed = 0x1234_5678u32;
        for _ in 0..200 {
            let mut block = [0f32; 64];
            for c in block.iter_mut() {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *c = (seed % 2001) as f32 - 1000.0;
            }
            let src = block;
            idct8x8(&mut block);
            for y in 0..8 {
                for x in 0..8 {
                    let mut s = 0f64;
                    for v in 0..8 {
                        for u in 0..8 {
                            let cu = if u == 0 { 0.5f64.sqrt() } else { 1.0 };
                            let cv = if v == 0 { 0.5f64.sqrt() } else { 1.0 };
                            s += cu
                                * cv
                                * src[v * 8 + u] as f64
                                * (((2 * x + 1) * u) as f64 * std::f64::consts::PI / 16.0).cos()
                                * (((2 * y + 1) * v) as f64 * std::f64::consts::PI / 16.0).cos();
                        }
                    }
                    let d = (block[y * 8 + x] as f64 - s / 4.0).abs();
                    assert!(d < 2e-3, "({x},{y}): {} vs {}", block[y * 8 + x], s / 4.0);
                }
            }
        }
    }

    #[test]
    fn bounded_idct_is_the_full_idct() {
        // Random blocks confined to every extent rectangle: the pruned
        // passes give the samples of the full transform (compared as
        // values: a zero's sign may differ).
        let mut seed = 0x0BAD_5EEDu32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for rows in 1..=8usize {
            for cols in 1..=8usize {
                for round in 0..40 {
                    let mut coef = [0i32; 64];
                    for v in 0..rows {
                        for u in 0..cols {
                            // Sparse and dense fills, small and large values.
                            if round % 4 != 0 || next() % 3 == 0 {
                                coef[v * 8 + u] = (next() % 4001) as i32 - 2000;
                            }
                        }
                    }
                    let ext = extent_of(&coef);
                    assert_eq!(u32::from(ext) >> (8 + rows), 0);
                    assert_eq!((ext & 0xFF) >> cols, 0);
                    let mut full: [f32; 64] = std::array::from_fn(|i| coef[i] as f32 * 3.0);
                    let mut within = full;
                    idct8x8(&mut full);
                    idct8x8_within(&mut within, ext);
                    assert_eq!(full, within, "{rows} x {cols}, round {round}");
                    // A looser claim is as good, as long as a block with
                    // only its DC coefficient is declared as one.
                    if ext & !extent_bit(0) != 0 {
                        let mut loose: [f32; 64] = std::array::from_fn(|i| coef[i] as f32 * 3.0);
                        idct8x8_within(&mut loose, FULL_EXTENT);
                        assert_eq!(full, loose);
                    }
                }
            }
        }
    }

    #[test]
    fn extent_bits_follow_the_zigzag() {
        let mut b = [0i32; 64];
        assert_eq!(extent_of(&b), 0);
        for k in 0..64 {
            b[crate::jpeg::zigzag::ZIGZAG[k]] = -1;
            assert_eq!(extent_of(&b), ZIGZAG_EXTENT_UP_TO[k]);
            let mut one = [0i32; 64];
            one[crate::jpeg::zigzag::ZIGZAG[k]] = 7;
            assert_eq!(extent_of(&one) & !ZIGZAG_EXTENT_UP_TO[k], 0);
        }
        assert_eq!(ZIGZAG_EXTENT_UP_TO[0], extent_bit(0));
        // Positions 0..=9 are the 4 × 4 low-frequency corner's diagonal
        // half; position 10 is the first in row 4.
        assert_eq!(ZIGZAG_EXTENT_UP_TO[9], 0x0F0F);
        assert_eq!(ZIGZAG_EXTENT_UP_TO[10], 0x1F0F);
        assert_eq!(ZIGZAG_EXTENT_UP_TO[63], FULL_EXTENT);
    }

    #[test]
    fn dc_of_constant_block() {
        // All samples = 100. With our normalisation, DC = 100 * 8.
        let mut block = [100.0f32; 64];
        fdct8x8(&mut block);
        assert!((block[0] - 800.0).abs() < 1e-2, "DC = {}", block[0]);
        for i in 1..64 {
            assert!(block[i].abs() < 1e-2, "AC[{i}] = {}", block[i]);
        }
    }
}
