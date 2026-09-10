//! T.81 §H.1.2.1 restart-interval prediction rule — black-box oracle.
//!
//! "The one-dimensional horizontal predictor (prediction sample Ra) is
//! used for the first line of samples at the start of the scan **and at
//! the beginning of each restart interval**. The selected predictor is
//! used for all other lines. The sample from the line above (Rb) is used
//! at the start of each line, except for the first line. At the beginning
//! of the first line and at the beginning of each restart interval the
//! prediction value of 2^(P − 1) is used."
//!
//! The fixtures under `tests/fixtures/lossless_restart/` are black-box
//! validator output: `src.pgm` is a 37×23 8-bit gradient-plus-noise image
//! and every `ll_p<N>_r<R>.jpg` is `cjpeg -lossless <N> -restart <R>
//! src.pgm` (one restart interval = `R` MCU rows = `R × 37` samples,
//! Table B.7's `n × MCUR`). With `-restart 1` every line is the first line
//! of its interval, so predictors 1 / 4 / 7 must all degenerate to `Ra`
//! and the three streams are byte-identical apart from the `Ss` byte —
//! a direct transcription of the clause above. `ll_p4_r2` (two rows per
//! interval) exercises the 2-D predictor on the second line of each
//! interval while the first line still uses `Ra`.
//!
//! The encode-side tests push our own `RSTn`-bearing streams through
//! `djpeg` when it is installed (skipped otherwise) and always check the
//! predictor-independence property, which needs no external tool.

use std::io::Write;
use std::process::Command;

use oxideav_mjpeg::decoder::decode_jpeg;
use oxideav_mjpeg::encoder::{
    encode_lossless_jpeg_cmyk_with_opts, encode_lossless_jpeg_grayscale_with_opts,
    encode_lossless_jpeg_rgb_with_opts, encode_lossless_jpeg_yuv_with_opts,
};

const SRC_PGM: &[u8] = include_bytes!("fixtures/lossless_restart/src.pgm");
const P1_R1: &[u8] = include_bytes!("fixtures/lossless_restart/ll_p1_r1.jpg");
const P4_R1: &[u8] = include_bytes!("fixtures/lossless_restart/ll_p4_r1.jpg");
const P7_R1: &[u8] = include_bytes!("fixtures/lossless_restart/ll_p7_r1.jpg");
const P4_R2: &[u8] = include_bytes!("fixtures/lossless_restart/ll_p4_r2.jpg");

const W: usize = 37;
const H: usize = 23;

/// Strip the `P5\n37 23\n255\n` header and return the raw samples.
fn src_pixels() -> &'static [u8] {
    let header = b"P5\n37 23\n255\n";
    assert!(SRC_PGM.starts_with(header));
    &SRC_PGM[header.len()..]
}

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("-version")
        .output()
        .map(|o| o.status.success() || !o.stderr.is_empty())
        .unwrap_or(false)
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("oxideav_mjpeg_h121");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Run `djpeg -pnm` on `jpeg` and return the raw PNM sample payload
/// (header stripped). `None` when djpeg is not installed.
fn djpeg_pnm(jpeg: &[u8], tag: &str) -> Option<Vec<u8>> {
    if !have("djpeg") {
        eprintln!("djpeg not available — skipping black-box check");
        return None;
    }
    let inp = scratch(&format!("{tag}.jpg"));
    let outp = scratch(&format!("{tag}.pnm"));
    std::fs::File::create(&inp)
        .unwrap()
        .write_all(jpeg)
        .unwrap();
    let st = Command::new("djpeg")
        .args(["-pnm", "-outfile"])
        .arg(&outp)
        .arg(&inp)
        .status()
        .expect("spawn djpeg");
    assert!(st.success(), "djpeg rejected our {tag} stream");
    let pnm = std::fs::read(&outp).unwrap();
    // P5/P6 header = three whitespace-separated fields after the magic.
    let mut fields = 0;
    let mut i = 2;
    while fields < 3 {
        while pnm[i].is_ascii_whitespace() {
            i += 1;
        }
        while !pnm[i].is_ascii_whitespace() {
            i += 1;
        }
        fields += 1;
    }
    Some(pnm[i + 1..].to_vec())
}

/// Decode `jpeg` with ImageMagick into raw 8-bit samples of colourspace
/// `space` (`"cmyk"` / `"rgb"` / `"gray"`). `None` when magick is absent.
fn magick_raw(jpeg: &[u8], space: &str, tag: &str) -> Option<Vec<u8>> {
    if !have("magick") {
        eprintln!("magick not available — skipping black-box check");
        return None;
    }
    let inp = scratch(&format!("{tag}.jpg"));
    let outp = scratch(&format!("{tag}.raw"));
    std::fs::File::create(&inp)
        .unwrap()
        .write_all(jpeg)
        .unwrap();
    let st = Command::new("magick")
        .arg(&inp)
        .args(["-depth", "8"])
        .arg(format!("{space}:{}", outp.display()))
        .status()
        .expect("spawn magick");
    assert!(st.success(), "magick rejected our {tag} stream");
    Some(std::fs::read(&outp).unwrap())
}

fn decode_gray(jpeg: &[u8]) -> Vec<u8> {
    let f = decode_jpeg(jpeg, None).expect("decode");
    assert_eq!(f.planes.len(), 1);
    let p = &f.planes[0];
    let mut out = Vec::with_capacity(W * H);
    for y in 0..H {
        out.extend_from_slice(&p.data[y * p.stride..y * p.stride + W]);
    }
    out
}

fn entropy_segment(jpeg: &[u8]) -> &[u8] {
    let sos = jpeg.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
    let len = u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]) as usize;
    &jpeg[sos + 2 + len..jpeg.len() - 2]
}

#[test]
fn validator_streams_with_one_row_intervals_are_predictor_independent() {
    // The fixture property the test relies on (Ra on every line).
    assert_eq!(entropy_segment(P1_R1), entropy_segment(P4_R1));
    assert_eq!(entropy_segment(P1_R1), entropy_segment(P7_R1));
    assert!(P4_R1.windows(4).any(|w| w == [0xFF, 0xDD, 0x00, 0x04]));
}

#[test]
fn decoder_applies_the_first_line_rule_at_every_restart() {
    let want = src_pixels();
    for (name, jpeg) in [
        ("p1_r1", P1_R1),
        ("p4_r1", P4_R1),
        ("p7_r1", P7_R1),
        ("p4_r2", P4_R2),
    ] {
        let got = decode_gray(jpeg);
        assert!(
            got == want,
            "fixture {name}: decoded samples diverge from the source"
        );
    }
}

/// Our encoder with one-row intervals: the entropy segment must not
/// depend on the predictor (every line is a first line → `Ra`).
#[test]
fn encoder_one_row_intervals_are_predictor_independent() {
    let src = src_pixels();
    let base =
        encode_lossless_jpeg_grayscale_with_opts(W as u32, H as u32, src, W, 8, 1, W as u16, 0)
            .unwrap();
    for p in 2..=7u8 {
        let s =
            encode_lossless_jpeg_grayscale_with_opts(W as u32, H as u32, src, W, 8, p, W as u16, 0)
                .unwrap();
        assert_eq!(
            entropy_segment(&base),
            entropy_segment(&s),
            "predictor {p} changed a one-row-interval stream"
        );
        assert_eq!(decode_gray(&s), src);
    }
}

/// Our restart-bearing grayscale streams decode byte-exact in `djpeg`
/// for every predictor and for one- and two-row intervals.
#[test]
fn encoder_grayscale_restart_streams_decode_byte_exact_in_djpeg() {
    let src = src_pixels();
    for p in 1..=7u8 {
        for rows in [1u16, 2, 5] {
            let ri = rows * W as u16;
            let s =
                encode_lossless_jpeg_grayscale_with_opts(W as u32, H as u32, src, W, 8, p, ri, 0)
                    .unwrap();
            assert_eq!(decode_gray(&s), src, "self-roundtrip p{p} r{rows}");
            if let Some(got) = djpeg_pnm(&s, &format!("gray_p{p}_r{rows}")) {
                assert!(got == src, "djpeg diverges on our gray p{p} r{rows} stream");
            }
        }
    }
}

/// Three-component (RGB-class) and four-component (CMYK-class) streams
/// with restart intervals: self round-trip always, `djpeg` when present
/// (RGB via the component-id triple; CMYK needs no APP14).
#[test]
fn encoder_multi_component_restart_streams_decode_byte_exact() {
    let src = src_pixels();
    let r: Vec<u8> = src.to_vec();
    let g: Vec<u8> = src.iter().map(|&v| v.wrapping_mul(3)).collect();
    let b: Vec<u8> = src.iter().rev().copied().collect();
    let k: Vec<u8> = src.iter().map(|&v| v ^ 0x5A).collect();
    for p in [1u8, 4, 6, 7] {
        for rows in [1u16, 3] {
            let ri = rows * W as u16;
            let rgb = encode_lossless_jpeg_rgb_with_opts(
                W as u32,
                H as u32,
                [&r, &g, &b],
                [W, W, W],
                8,
                p,
                ri,
                0,
            )
            .unwrap();
            let f = decode_jpeg(&rgb, None).unwrap();
            let pl = &f.planes[0];
            let mut packed = Vec::with_capacity(W * H * 3);
            for y in 0..H {
                packed.extend_from_slice(&pl.data[y * pl.stride..y * pl.stride + W * 3]);
            }
            let mut want = Vec::with_capacity(W * H * 3);
            for i in 0..W * H {
                want.extend_from_slice(&[r[i], g[i], b[i]]);
            }
            assert!(packed == want, "rgb self-roundtrip p{p} r{rows}");
            if let Some(got) = djpeg_pnm(&rgb, &format!("rgb_p{p}_r{rows}")) {
                assert!(got == want, "djpeg diverges on our rgb p{p} r{rows} stream");
            }

            let cmyk = encode_lossless_jpeg_cmyk_with_opts(
                W as u32,
                H as u32,
                [&r, &g, &b, &k],
                [W, W, W, W],
                p,
                None,
                ri,
                0,
            )
            .unwrap();
            let f = decode_jpeg(&cmyk, None).unwrap();
            let pl = &f.planes[0];
            let mut want4 = Vec::with_capacity(W * H * 4);
            for y in 0..H {
                for x in 0..W {
                    let i = y * W + x;
                    let o = y * pl.stride + x * 4;
                    assert_eq!(
                        &pl.data[o..o + 4],
                        &[r[i], g[i], b[i], k[i]],
                        "cmyk self-roundtrip p{p} r{rows} at ({x},{y})"
                    );
                    want4.extend_from_slice(&[r[i], g[i], b[i], k[i]]);
                }
            }
            if let Some(got) = magick_raw(&cmyk, "cmyk", &format!("cmyk_p{p}_r{rows}")) {
                // The ink polarity of an APP14-less four-component stream
                // is a reader convention (this crate: "regular", C = 0 is
                // no ink; the validator: Adobe-inverted), so accept either
                // polarity — the entropy-coded samples must match exactly.
                let inverted: Vec<u8> = want4.iter().map(|&v| 255 - v).collect();
                assert!(
                    got == want4 || got == inverted,
                    "magick diverges on our cmyk p{p} r{rows} stream"
                );
            }
        }
    }
    // Luma-oversampled YUV-class lossless (§A.2.3 interleaved MCUs, 2×2
    // luma): one interval = one MCU row = ceil(W/2) MCUs.
    let cw = W.div_ceil(2);
    let ch = H.div_ceil(2);
    let cb: Vec<u8> = (0..cw * ch).map(|i| (i * 5 % 251) as u8).collect();
    let cr: Vec<u8> = (0..cw * ch).map(|i| (i * 11 % 241) as u8).collect();
    for p in [4u8, 7] {
        let ri = cw as u16;
        let s = encode_lossless_jpeg_yuv_with_opts(
            W as u32, H as u32, src, W, &cb, cw, &cr, cw, 2, 2, p, ri, 0,
        )
        .unwrap();
        let f = decode_jpeg(&s, None).unwrap();
        assert_eq!(f.planes.len(), 3);
        for y in 0..H {
            let pl = &f.planes[0];
            assert_eq!(
                &pl.data[y * pl.stride..y * pl.stride + W],
                &src[y * W..y * W + W]
            );
        }
        for (pi, plane) in [(1usize, &cb), (2usize, &cr)] {
            let pl = &f.planes[pi];
            for y in 0..ch {
                assert_eq!(
                    &pl.data[y * pl.stride..y * pl.stride + cw],
                    &plane[y * cw..y * cw + cw]
                );
            }
        }
    }
}
