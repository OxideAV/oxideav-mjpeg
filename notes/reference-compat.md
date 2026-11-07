# Reference-compatible decode (#19): fitted parameters

`DecodeOptions::with_reference_compat(true)` (registry: `reference_compat
= true`) reproduces the default output of `djpeg` sample for sample. The
arithmetic of each stage below was measured against `djpeg` 3.2.0 as a
black box: crafted or generated streams in, `djpeg -pnm` (also
`-nosmooth`, `-grayscale`) out, compared with candidate integer models.
The candidates are the T.81 §A.3.3 IDCT, the T.871 §7 inverse colour
equations and the T.871 §9 chroma siting; only the precision, rounding
and edge rules were searched.

The streams for the IDCT and colour experiments were written by a small
coefficient-level JPEG writer (Annex K typical Huffman tables from
`docs/image/jpeg/tables/`, chosen quantisation tables, chosen quantised
coefficients), so the input to each stage is known exactly.

## Inverse DCT (`IdctMethod::AccurateInteger`, `src/jpeg/idct_int.rs`)

Model: the separable §A.3.3 transform as two 1-D passes (columns, then
rows). Each pass computes `√8 · t(x)` using the factorisation documented
in `src/jpeg/idct_int.rs` (3 products for the even rotation, 9 for the odd
half), with constants `round(2^P · value)`. The column-pass output is
shifted down to keep `M` fraction bits, the row pass by `P + M + 3`, both
`⌊v / 2^s + ½⌋`, then the `+128` level shift and the range limit.

Measured on one quant-1 grey stream holding 1 408 impulse blocks (every
coefficient position, amplitudes ±1, 2, 3, 5, 8, 13, 40, 100, 255, 511,
1000) and one holding 2 048 random blocks (30 % non-zero AC in −60…60,
DC in −600…600). Mismatching blocks `(impulse, random)`:

| P \ M | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 11 | 300, 2037 | 156, 1970 | 28, 641 | 76, 1594 | 124, 1483 |
| 12 | 296, 2038 | 168, 1955 | 56, 659 | 102, 1582 | 154, 1473 |
| **13** | 288, 2040 | 148, 1960 | **0, 0** | 68, 1575 | 116, 1467 |
| 14 | 300, 2039 | 148, 1960 | 28, 308 | 68, 1586 | 124, 1461 |
| 15 | 300, 2039 | 148, 1959 | 28, 300 | 76, 1584 | 124, 1478 |

- Fitted: **`P = 13` constant fraction bits, `M = 2` fraction bits kept
  between the passes, round-half-up at both shifts.**
- Truncating shifts (no half bias) miss 1 377 of the 1 408 impulses at
  every `(P, M)`. One rounded constant per matrix entry (no factorisation)
  misses at least 272 impulses: the impulse responses pin which rounded
  constants multiply which sums.
- 12-bit frames: `M = 1` (`M = 2` differs from `djpeg -grayscale`'s
  16-bit luma; `M = 1` matches it exactly), level shift 2048.
- Range limit, measured with flat DC-only blocks
  (`tests/fixtures/libjpeg/idct_dc_sweep.jpg`, `Q00 = 16`, levels
  −1792…2304). With `u = v mod 2048`, `u < 256` gives `u`,
  `256 ≤ u < 1152` gives 255, and `u ≥ 1152` gives 0. So every overshoot
  inside −896…1151 clamps, and only gross overshoot wraps. For 12-bit:
  `u = v mod 16384`, with cut points 4096 and 10240.

## YCbCr → RGB (`YcbcrKernel::FixedPoint16`, `src/convert.rs`)

Model: `R = Y + ⌊(kR·(Cr−128) + h) / 2^S⌋`, the same form for B, and for G
the two products summed before one shift, then a clamp to 0…255.

Experiment: a crafted 4:4:4 stream of 256 × 256 flat quant-1 blocks, one
per `(Cb, Cr)` pair, at `Y = 0` and again at `Y = 255`. DC-only blocks
reconstruct exactly. The chroma offset does not depend on `Y`, so every
offset is seen unclamped at one of the two levels. Mismatches over the
observed offsets (255 R, 255 B, 65 179 G), with the T.871 coefficients
rounded at scale `2^S`:

| S | 10 | 12 | 14 | **15** | 16 | 17 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|
| R | 2 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| B | 6 | 2 | 2 | **0** | 0 | 0 | 2 | 0 |
| G | 1916 | 394 | 161 | **0** | 59 | 39 | 51 | 43 |

A search of every G constant pair within ±4 of nominal finds exactly one
zero-mismatch pair at each of `2^15…2^18`: `(11 277, 23 401) · 2^(S−15)`,
which is `0.344146 / 0.714142`. It finds none at `2^12…2^14` or
`2^19…2^20`. Product-by-product rounding of G misses 15 918 samples on a
4:4:4 noise image (`djpeg -nosmooth`); truncation misses 97 000.

- Fitted: **scale `2^16`, `kR = 91 881`, `kB = 116 130` (R and B accept
  ranges, and these are the nominal values), `kG = (22 554, 46 802)`,
  half bias `2^15`, G products summed before the shift.**
- `tests/libjpeg_compat.rs::ycbcr_kernel_matches_djpeg_over_every_chroma_pair`
  pins all 2 × 65 536 outputs against `djpeg`'s hashes.

## Chroma upsampling (`ChromaUpsampling::Triangular`, `src/jpeg/upsample.rs`)

Model: T.871 §9 centred siting. Along a 2:1 axis, output `2i` is
`¾ c[i] + ¼ c[i−1]` and `2i+1` is `¾ c[i] + ¼ c[i+1]`, with edge samples
replicated. The 2 × 2 ratio is separable: vertical `3·near + far`
unrounded, then horizontal `3 : 1`, divided by 16 once. Searched: the
rounding bias per output phase, and the narrowest component that is
still interpolated.

Experiment: `cjpeg -quality 100` noise images of 61×47, 64×64, 5×3, 3×7,
4×4, 2×6 and 6×2 at `-sample 2x1 / 1x2 / 2x2`. Native chroma planes come
from the fitted IDCT, colour from the fitted kernel. Mismatching samples
against `djpeg -pnm`, best candidates:

| ratio | biases (even, odd) | narrow-width rule | mismatches |
|---|---|---|---|
| 2:1 | **(1, 2) / 4** | **width ≤ 2 replicated** | **0** |
| 2:1 | (1, 2) / 4 | width ≤ 3 replicated | 57 |
| 2:1 | (1, 2) / 4 | never replicated | 64 |
| 2:1 | (1, 1) / 4 | width ≤ 2 replicated | 2 522 |
| 1:2 | **(1, 2) / 4** (by output row) | none | **0** |
| 1:2 | (2, 2) / 4 | none | 2 503 |
| 2 × 2 | **(8, 7) / 16** (by output column) | **width ≤ 2 replicated** | **0** |
| 2 × 2 | (8, 7) / 16 | width ≤ 3 replicated | 64 |
| 2 × 2 | (8, 8) / 16 | width ≤ 2 replicated | 742 |

- Fitted: **one-axis bias 1 for even and 2 for odd outputs (`/ 4`); 2 × 2
  bias 8 for even and 7 for odd output columns (`/ 16`); a horizontally
  subsampled component at most 2 samples wide is replicated on both
  axes.**
- Other ratios (4:1:1, 4 × 2 luma, mixed factors) are replicated, which
  is bit-exact on every such stream of the corpus.

## Verification corpus (all bit-exact, 0 differing bytes)

| source | streams |
|---|---|
| `cjpeg`: 52 images (1×1 … 101×67; noise, gradients with edges, saturated primaries, chroma impulses) × 21 settings (4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, 4×2, two mixed-factor layouts, q10, q100, restart `1B` / 3, progressive, arithmetic, arithmetic progressive, grey, grey progressive, `-optimize -dct float`, `-rgb`) | 1 092 |
| this crate's encoder: the same 52 images × 4:4:4 / 4:2:2 / 4:2:0 / 4:1:1 / grey × sequential q85, progressive q60, restart 1 q95, q100, optimal tables q30 | 1 300 |
| `ffmpeg -pix_fmt yuvj420p / yuvj422p / yuvj444p / gray`, `-q:v 2 / 10` | 208 |
| `docs/image/jpeg/fixtures/*/input.jpg` (8-bit) | 15 |
| crafted: IDCT impulses, random blocks, DC sweep, the full `(Cb, Cr)` grid | 6 |

Committed subset: `tests/fixtures/libjpeg/` (87 streams; `oracle.txt`
holds FNV-1a-64 hashes of `djpeg -pnm`'s sample bytes).

## Residual: incomplete progressive streams

`djpeg` changes the output of a progressive stream that never fully codes
its five lowest AC coefficients (zig-zag 1…5): they are not sent, or
their last approximation stays at `Al > 0`. This compat mode does not
model that change. It reconstructs from the coefficients as coded, and
these streams differ. Measured on a 64×48 grey image (`cjpeg -scans`),
as differing samples out of 3 072:

| scans after a full DC scan | differing |
|---|---|
| AC 1…1 only | 2 785 |
| AC 1…2 only | 1 819 |
| AC 1…3 only | 1 394 |
| AC 1…5 or 1…9 only | 0 |
| AC 6…63 only | 2 788 |
| AC 1…63 at `Al = 1`, never refined | 1 566 |
| AC 1…63 at `Al = 2`, refined to `Al = 1` only | 1 566 |
| AC 1…63 refined to `Al = 0` | 0 |
| DC at `Al = 1` never refined, AC 1…63 complete | 0 |

A DC-only colour stream differs in 3 138 to 19 580 bytes of RGB output
(37×29 to 101×67). The change spreads from neighbouring blocks: a single non-zero DC
alters samples two blocks away. Every complete progressive stream of the
corpus is exact, including `cjpeg`'s default script and this crate's
encoder.
