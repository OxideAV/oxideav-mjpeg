//! 8×8 forward and inverse Discrete Cosine Transform.
//!
//! Straight textbook implementation over f32. Not the fastest algorithm out
//! there — AAN or the Loeffler factorisation would be faster — but it is
//! short, correct, and pure-Rust. Suitable for small-scale MJPEG workloads;
//! optimisation can come later.

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

/// Inverse DCT of an 8×8 block (natural order, already dequantised). In-place.
///
/// Two short-circuits keep the output **bit-identical** to the full
/// two-pass matrix multiply while skipping arithmetic that provably
/// contributes nothing:
///
/// * **DC-only.** When every AC coefficient is zero, the full transform
///   collapses to a constant block. The row pass leaves only `tmp[0][x] =
///   t[0][x] * dc` non-zero, and the column pass yields `out[m][x] =
///   t[0][m] * tmp[0][x]`. Because `cos(0) = 1` exactly, every `t[0][·]`
///   is the *same* f32 (`t[0][0]`), so `t[0][0] * (t[0][0] * dc)` reproduces
///   the two-pass multiply order exactly — same operations, same rounding.
/// * **Per-row AC-zero skip in the row pass.** A row whose 8 inputs are all
///   zero produces 8 zero outputs; a row whose only non-zero input is its DC
///   (`block[y*8]`) produces `tmp[y][n] = t[0][n] * dc` — again the literal
///   single-term form of the inner sum (the other 7 terms are `t[k][n] * 0.0
///   = 0.0`, and adding `0.0` to a running f32 sum is the identity, so the
///   accumulated result is bit-identical to the full loop).
pub fn idct8x8(block: &mut [f32; 64]) {
    let t = cos_table();

    // DC-only fast path: scan the 63 AC slots; bail to constant fill if all
    // are zero. The constant equals the two-pass result exactly (see above).
    if block[1..].iter().all(|&c| c == 0.0) {
        let dc = block[0];
        let v = t[0][0] * (t[0][0] * dc);
        block.fill(v);
        return;
    }

    let mut tmp = [0.0f32; 64];

    // Row-wise 1-D inverse: for each row y, tmp[y][n] = Σk t[k][n] * block[y][k]
    for y in 0..8 {
        let row = &block[y * 8..y * 8 + 8];
        // Adding `t[k][n] * 0.0` (= 0.0) to a running f32 sum is the exact
        // identity, so dropping all-zero AC terms — or an entirely zero row —
        // leaves every `tmp[y][n]` bit-identical to the full inner loop.
        if row[1..].iter().all(|&c| c == 0.0) {
            let dc = row[0];
            if dc == 0.0 {
                for n in 0..8 {
                    tmp[y * 8 + n] = 0.0;
                }
            } else {
                for n in 0..8 {
                    tmp[y * 8 + n] = t[0][n] * dc;
                }
            }
            continue;
        }
        for n in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += t[k][n] * row[k];
            }
            tmp[y * 8 + n] = s;
        }
    }
    // Column-wise 1-D inverse.
    for x in 0..8 {
        for m in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += t[k][m] * tmp[k * 8 + x];
            }
            block[m * 8 + x] = s;
        }
    }
}

/// Accurate integer inverse DCT of one dequantised 8×8 block
/// (natural order), the [`crate::IdctMethod::Integer`] kernel.
///
/// Evaluates the T.81 §A.3.3 IDCT in fixed point: the 2-D transform is
/// separated into two 1-D passes, each 1-D pass is the even / odd
/// factorisation of the 8-point IDCT (the even half is a 4-point IDCT
/// on `S0 S2 S4 S6`, the odd half a rotation network on `S1 S3 S5 S7`;
/// outputs `n` and `7 − n` are the sum and difference of the two
/// halves), and every irrational `√2 · cos(kπ/16)` product is a
/// 13-bit fixed-point constant (`round(c · 2^13)`). The first pass
/// runs down the columns and keeps two extra fraction bits
/// (`CONST_BITS − PASS1_BITS = 11` bits are shifted out, with
/// round-half-up); the second pass runs along the rows and shifts out
/// the remaining `CONST_BITS + PASS1_BITS + 3 = 18` bits, the `+ 3`
/// being the `1/8` of the two-pass normalisation (`¼ · C_u C_v` with
/// `C_0 = 1/√2` written as `w_0 = 1`, `w_k = √2`, i.e. `w_u w_v / 8`).
/// Arithmetic is `i64`, so no product can overflow for any T.81
/// coefficient × quantiser pair.
///
/// The result is the signed sample block before the §F.2.1.5 level
/// shift; the caller adds `128` and clamps. Pass order, constant
/// width and the two rounding biases were fixed by black-box bit-exact
/// comparison against the reference decoder's default integer IDCT
/// (see `tests/libjpeg_compat.rs`); the float kernel [`idct8x8`]
/// agrees with it to within ±1 on every block.
pub fn idct8x8_int(block: &[i32; 64]) -> [i32; 64] {
    const CONST_BITS: u32 = 13;
    const PASS1_BITS: u32 = 2;
    // round(c · 2^13) for the twelve constants of the factorisation.
    const FIX_0_298631336: i64 = 2446;
    const FIX_0_390180644: i64 = 3196;
    const FIX_0_541196100: i64 = 4433;
    const FIX_0_765366865: i64 = 6270;
    const FIX_0_899976223: i64 = 7373;
    const FIX_1_175875602: i64 = 9633;
    const FIX_1_501321110: i64 = 12299;
    const FIX_1_847759065: i64 = 15137;
    const FIX_1_961570560: i64 = 16069;
    const FIX_2_053119869: i64 = 16819;
    const FIX_2_562915447: i64 = 20995;
    const FIX_3_072711026: i64 = 25172;

    /// One 1-D pass on eight inputs; `shift` bits are rounded out of
    /// the result (round half up).
    #[inline(always)]
    fn pass(x: [i64; 8], shift: u32) -> [i64; 8] {
        // Even half: 4-point IDCT of x0 x2 x4 x6.
        let z2 = x[2];
        let z3 = x[6];
        let z1 = (z2 + z3) * FIX_0_541196100;
        let t2 = z1 - z3 * FIX_1_847759065;
        let t3 = z1 + z2 * FIX_0_765366865;
        let t0 = (x[0] + x[4]) << CONST_BITS;
        let t1 = (x[0] - x[4]) << CONST_BITS;
        let e0 = t0 + t3;
        let e3 = t0 - t3;
        let e1 = t1 + t2;
        let e2 = t1 - t2;
        // Odd half: x7 x5 x3 x1.
        let (o0, o1, o2, o3) = (x[7], x[5], x[3], x[1]);
        let z1 = o0 + o3;
        let z2 = o1 + o2;
        let z3 = o0 + o2;
        let z4 = o1 + o3;
        let z5 = (z3 + z4) * FIX_1_175875602;
        let o0 = o0 * FIX_0_298631336;
        let o1 = o1 * FIX_2_053119869;
        let o2 = o2 * FIX_3_072711026;
        let o3 = o3 * FIX_1_501321110;
        let z1 = -z1 * FIX_0_899976223;
        let z2 = -z2 * FIX_2_562915447;
        let z3 = -z3 * FIX_1_961570560 + z5;
        let z4 = -z4 * FIX_0_390180644 + z5;
        let o0 = o0 + z1 + z3;
        let o1 = o1 + z2 + z4;
        let o2 = o2 + z2 + z3;
        let o3 = o3 + z1 + z4;
        let half = 1i64 << (shift - 1);
        let d = |v: i64| (v + half) >> shift;
        [
            d(e0 + o3),
            d(e1 + o2),
            d(e2 + o1),
            d(e3 + o0),
            d(e3 - o0),
            d(e2 - o1),
            d(e1 - o2),
            d(e0 - o3),
        ]
    }

    // Pass 1: columns (vertical frequency v → rows), keeping PASS1_BITS.
    let mut ws = [0i64; 64];
    for col in 0..8 {
        let x: [i64; 8] = std::array::from_fn(|v| block[v * 8 + col] as i64);
        let y = pass(x, CONST_BITS - PASS1_BITS);
        for (r, v) in y.into_iter().enumerate() {
            ws[r * 8 + col] = v;
        }
    }
    // Pass 2: rows, shifting out the rest plus the 1/8 normalisation.
    let mut out = [0i32; 64];
    for row in 0..8 {
        let x: [i64; 8] = std::array::from_fn(|u| ws[row * 8 + u]);
        let y = pass(x, CONST_BITS + PASS1_BITS + 3);
        for (c, v) in y.into_iter().enumerate() {
            out[row * 8 + c] = v as i32;
        }
    }
    out
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
    fn integer_idct_tracks_the_float_kernel() {
        // Pseudo-random dequantised blocks: the 13-bit fixed-point
        // kernel is within ±1 of the f32 reference on every sample,
        // and a DC-only block reproduces `round(DC / 8)` exactly.
        let mut seed = 0x9E37_79B9u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for _ in 0..500 {
            let mut blk = [0i32; 64];
            for (k, b) in blk.iter_mut().enumerate() {
                let range: u32 = if k == 0 { 2048 } else { 300 >> (k / 16) };
                *b = (rnd() % (2 * range + 1)) as i32 - range as i32;
            }
            let int = idct8x8_int(&blk);
            let mut flt: [f32; 64] = std::array::from_fn(|k| blk[k] as f32);
            idct8x8(&mut flt);
            for k in 0..64 {
                assert!(
                    (int[k] as f32 - flt[k]).abs() <= 1.0 + 1e-3,
                    "sample {k}: int {} vs float {}",
                    int[k],
                    flt[k]
                );
            }
        }
        for dc in [-1024, -37, -8, -7, -1, 0, 1, 3, 4, 5, 12, 1016, 2047] {
            let mut blk = [0i32; 64];
            blk[0] = dc;
            let out = idct8x8_int(&blk);
            // Exactly `(dc · 2^15 + 2^17) >> 18 = ⌊dc / 8 + ½⌋`.
            let want = ((dc as i64) * 4 * 8192 + (1 << 17)) >> 18;
            assert!(
                out.iter().all(|&v| v as i64 == want),
                "DC {dc}: got {:?}, want {want}",
                &out[..8]
            );
        }
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
