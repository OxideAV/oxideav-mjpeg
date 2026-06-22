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

/// Forward DCT of an 8×8 block in natural order, in-place. Caller should
/// subtract 128 from sample values before calling so input is centred on 0.
pub fn fdct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];

    // Row-wise 1-D forward: for each row y, tmp[y][k] = Σn t[k][n] * block[y][n]
    for y in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += t[k][n] * block[y * 8 + n];
            }
            tmp[y * 8 + k] = s;
        }
    }
    // Column-wise.
    for x in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += t[k][n] * tmp[n * 8 + x];
            }
            block[k * 8 + x] = s;
        }
    }
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
