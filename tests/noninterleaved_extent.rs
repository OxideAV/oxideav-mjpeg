//! T.81 §A.2.2 / §A.2.4 — non-interleaved scans cover the component's
//! own block extent, not the MCU-padded grid.
//!
//! "If the component is to be interleaved, the encoding process shall
//! also extend the number of samples by one or more additional blocks,
//! if necessary, so that the number of blocks is an integer multiple of
//! Hi" (A.2.4) — the extension to a multiple of `Hi` / `Vi` blocks is
//! conditional on interleaving. A scan with `Ns = 1` therefore walks
//! `ceil(xi / 8) × ceil(yi / 8)` data units (A.2.2 raster order). The
//! decoder and the legacy progressive encoder walked `mcus_x × Hi`
//! blocks instead, which only differs when `ceil(xi / 8) < mcus_x × Hi`
//! — e.g. a 37-pixel-wide 4:2:2 frame (luma 5 true blocks, 6 padded).
//!
//! Fixtures under `tests/fixtures/noninterleaved/` are black-box
//! validator output on a 37×29 RGB source: `prog37_<s>.jpg` is `cjpeg
//! -quality 90 -progressive -sample <s>`, `seqni37_2x1.jpg` is a
//! sequential frame with three non-interleaved scans (`-scans` script
//! `0; 1; 2;`), and `*_y.pgm` is `djpeg -nosmooth -grayscale -pnm` (the
//! luma plane, the ±1 IDCT-rounding oracle).

use oxideav_mjpeg::decoder::decode_jpeg;

const W: usize = 37;
const H: usize = 29;

const FIXTURES: &[(&str, &[u8], &[u8])] = &[
    (
        "progressive 2x1",
        include_bytes!("fixtures/noninterleaved/prog37_2x1.jpg"),
        include_bytes!("fixtures/noninterleaved/prog37_2x1_y.pgm"),
    ),
    (
        "progressive 2x2",
        include_bytes!("fixtures/noninterleaved/prog37_2x2.jpg"),
        include_bytes!("fixtures/noninterleaved/prog37_2x2_y.pgm"),
    ),
    (
        "sequential non-interleaved 2x1",
        include_bytes!("fixtures/noninterleaved/seqni37_2x1.jpg"),
        include_bytes!("fixtures/noninterleaved/seqni37_2x1_y.pgm"),
    ),
];

fn pgm(data: &[u8]) -> Vec<u8> {
    let header = format!("P5\n{W} {H}\n255\n");
    assert!(data.starts_with(header.as_bytes()));
    data[header.len()..].to_vec()
}

#[test]
fn non_interleaved_scans_of_odd_extent_frames_match_the_validator() {
    for &(name, jpeg, y) in FIXTURES {
        let f = decode_jpeg(jpeg, None).unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = pgm(y);
        let pl = &f.planes[0];
        let mut maxd = 0i32;
        for yy in 0..H {
            for xx in 0..W {
                let d =
                    (i32::from(pl.data[yy * pl.stride + xx]) - i32::from(want[yy * W + xx])).abs();
                maxd = maxd.max(d);
            }
        }
        assert!(maxd <= 1, "{name}: luma max|d| = {maxd} vs djpeg");
    }
}
