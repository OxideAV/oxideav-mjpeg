//! Float → integer sample rounding without a library call.
//!
//! The sample paths round a non-negative `f32` half away from zero
//! (`f32::round`) and store it as an integer. Where the target's baseline
//! has a rounding instruction that is exactly `x.round() as i32`; where
//! it has none, `round` is a library call per sample and a float →
//! integer `as` cast carries saturation checks, and neither vectorises.
//! [`round_clamped`] gives the same integers from additions and one
//! comparison on those targets.

/// `2^23`: an `f32` at or above it has a unit in the last place of 1.
const INT_GRID: f32 = 8_388_608.0;

/// `x` limited to `0.0..=max` (negative values and the quiet `NaN`
/// arithmetic produces → `0`), then rounded half away from zero.
/// `max ≤ 2^22`.
///
/// For any `x` with `0 ≤ x ≤ max` this is `x.round() as i32`; with
/// `max = 255` it is `x.clamp(0.0, 255.0).round() as u8` of every `x`.
#[inline(always)]
pub(crate) fn round_clamped(x: f32, max: f32) -> i32 {
    if cfg!(target_arch = "aarch64") {
        x.max(0.0).min(max).round() as i32
    } else {
        round_clamped_by_grid(x, max)
    }
}

/// [`round_clamped`] in additions. With `c` the limited value, `c + 2^23`
/// lies in `[2^23, 2^24)`, where consecutive floats are 1 apart: the
/// addition rounds `c` to the nearest integer (ties to even) and the low
/// 23 bits of the sum's representation are that integer `n`. `c − n` is
/// exact (`|c − n| ≤ ½`), and equals `½` exactly when `c` was a tie that
/// went down to the even neighbour — the one case where rounding half
/// away from zero is `n + 1`.
#[inline(always)]
pub(crate) fn round_clamped_by_grid(x: f32, max: f32) -> i32 {
    debug_assert!(max <= 4_194_304.0);
    let c = if x > 0.0 { x } else { 0.0 };
    let c = if c < max { c } else { max };
    let t = c + INT_GRID;
    let n = (t.to_bits() & 0x007F_FFFF) as i32;
    n + i32::from(c - (t - INT_GRID) == 0.5)
}

/// The smaller of two numbers. `f32::min` also defines the result for
/// a `NaN` operand, which on targets whose minimum instruction has a
/// different `NaN` rule costs a comparison and a blend per call; the
/// reconstruction's operands are never `NaN`, so a plain selection is
/// the same value there.
#[inline(always)]
pub(crate) fn fmin(a: f32, b: f32) -> f32 {
    if cfg!(target_arch = "aarch64") {
        a.min(b)
    } else if a < b {
        a
    } else {
        b
    }
}

/// The larger of two numbers (see [`fmin`]).
#[inline(always)]
pub(crate) fn fmax(a: f32, b: f32) -> f32 {
    if cfg!(target_arch = "aarch64") {
        a.max(b)
    } else if a > b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(x: f32, max: f32) {
        // A signalling NaN is not an arithmetic result; how a selection
        // treats one is the code generator's choice.
        if x.is_nan() && x.to_bits() & 0x0040_0000 == 0 {
            return;
        }
        let want = if x.is_nan() {
            0
        } else {
            x.clamp(0.0, max).round() as i32
        };
        assert_eq!(round_clamped_by_grid(x, max), want, "{x:e} (grid)");
        assert_eq!(round_clamped(x, max), want, "{x:e}");
    }

    #[test]
    fn grid_rounding_is_round_half_away() {
        // Every tie and its two neighbouring floats up to 2^16, every
        // integer, the floats around ½, and a stride over all bit
        // patterns (negative, huge, infinite and NaN inputs included).
        for max in [255.0, 4095.0, 65_535.0, 65_536.0, 4_194_304.0] {
            for k in 0..=65_536u32 {
                let h = k as f32 + 0.5;
                for x in [
                    h,
                    f32::from_bits(h.to_bits() - 1),
                    f32::from_bits(h.to_bits() + 1),
                    k as f32,
                ] {
                    check(x, max);
                    check(-x, max);
                }
            }
            for bits in (0..=u32::MAX).step_by(3571) {
                check(f32::from_bits(bits), max);
            }
            for x in [
                0.0,
                -0.0,
                0.49999997,
                0.5,
                0.50000006,
                1.0e9,
                f32::INFINITY,
                f32::NAN,
            ] {
                check(x, max);
                check(-x, max);
            }
        }
    }

    #[test]
    fn selections_are_min_and_max() {
        let v = [
            0.0f32, 0.25, 1.0, 127.5, 255.0, 4095.0, -3.5, 1.0e-30, 6.0e7,
        ];
        for &a in &v {
            for &b in &v {
                assert_eq!(fmin(a, b), a.min(b));
                assert_eq!(fmax(a, b), a.max(b));
            }
        }
    }

    #[test]
    fn every_sample_value_in_the_eight_bit_range() {
        // All floats in 0..=255.5 that a level-shifted sample can be:
        // multiples of 2^-16 (the spacing of an `f32` in 128..256).
        for i in 0..=(255 * 65_536 + 32_768u32) {
            let x = i as f32 / 65_536.0;
            assert_eq!(
                round_clamped_by_grid(x, 255.0),
                x.clamp(0.0, 255.0).round() as i32
            );
        }
    }
}
