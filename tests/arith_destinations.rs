//! Arithmetic-coding statistics areas follow the table destinations.
//!
//! T.81 §B.2.3: "In the latter case [arithmetic coding] the entropy coding
//! table destination selector specifies both an arithmetic coding
//! conditioning table destination and an associated statistics area." Scan
//! components that name the same `Td` / `Ta` therefore share one area — in
//! the usual YCbCr layout `Cb` and `Cr` both select destination 1. The
//! decoder and the in-tree arithmetic encoders used to keep one area per
//! component, which decoded `cjpeg -arithmetic` colour JPEGs to noise and
//! produced colour streams that `djpeg` could not read.
//!
//! Fixtures under `tests/fixtures/arith/` are black-box validator output
//! (libjpeg-turbo 3.2.0): `huff_<layout>.jpg` is `cjpeg -quality 80
//! -sample <layout>` of a 37×29 `ffmpeg` `testsrc2` frame (`-grayscale`
//! for `gray`, `-progressive -restart 1` for `prog_rst`) and
//! `arith_<layout>.jpg` is `jpegtran -arithmetic` of it (`-progressive
//! -restart 1` again for `prog_rst`), a lossless transcode: both files
//! carry the same quantised coefficients, so a correct decoder produces
//! the same samples from each.
//!
//! `ours_<layout>_arith.jpg` is `jpegtran -arithmetic` of this crate's
//! own Huffman encode of the synthetic frame built below (regenerate with
//! `OXIDEAV_MJPEG_DUMP=<dir> cargo test --test arith_destinations`, which
//! writes the Huffman streams as `<dir>/ours_<layout>_huff.jpg` and our
//! arithmetic streams as `<dir>/ours_<layout>_arith_ours.jpg`). Our
//! arithmetic encoder, coding the same coefficients, must emit the very
//! same entropy-coded segment.

use oxideav_mjpeg::decode;

macro_rules! fx {
    ($name:literal) => {
        include_bytes!(concat!("fixtures/arith/", $name))
    };
}

const PAIRS: &[(&str, &[u8], &[u8])] = &[
    ("2x2", fx!("huff_2x2.jpg"), fx!("arith_2x2.jpg")),
    ("2x1", fx!("huff_2x1.jpg"), fx!("arith_2x1.jpg")),
    ("1x1", fx!("huff_1x1.jpg"), fx!("arith_1x1.jpg")),
    ("gray", fx!("huff_gray.jpg"), fx!("arith_gray.jpg")),
    (
        "prog_rst",
        fx!("huff_prog_rst.jpg"),
        fx!("arith_prog_rst.jpg"),
    ),
];

#[test]
fn arithmetic_transcodes_decode_to_the_huffman_samples() {
    for &(name, huff, arith) in PAIRS {
        let h = decode(huff).unwrap_or_else(|e| panic!("{name}: huffman decode: {e}"));
        let a = decode(arith).unwrap_or_else(|e| panic!("{name}: arithmetic decode: {e}"));
        assert_eq!((a.width, a.height), (37, 29), "{name}: geometry");
        assert_eq!(a.format, h.format, "{name}: layout");
        assert_eq!(a.planes, h.planes, "{name}: samples differ");
    }
}

/// The entropy-coded segments of a JPEG stream, in order (restart markers
/// kept inside their segment).
#[cfg(feature = "registry")]
fn scan_segments(jpeg: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        assert_eq!(jpeg[i], 0xFF, "marker expected at {i}");
        let m = jpeg[i + 1];
        if m == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        if m != 0xDA {
            i += 2 + len;
            continue;
        }
        let start = i + 2 + len;
        let mut k = start;
        while !(jpeg[k] == 0xFF && jpeg[k + 1] != 0 && !(0xD0..=0xD7).contains(&jpeg[k + 1])) {
            k += 1;
        }
        out.push(jpeg[start..k].to_vec());
        i = k;
    }
    out
}

#[cfg(feature = "registry")]
mod encoder {
    use super::scan_segments;
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{PixelFormat, VideoFrame};
    use oxideav_mjpeg::encoder as enc;

    // 41 × 29: partial blocks on both edges, but no 4:2:0 MCU carries a
    // block lying wholly outside the component (§A.2.4 dummy blocks) —
    // their content is the encoder's choice, so a byte comparison with
    // another encoder is only meaningful without them.
    const W: usize = 41;
    const H: usize = 29;

    fn plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..w * h)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let (x, y) = (i % w, i / w);
                ((x * 5 + y * 3) as u32 + (s & 15)) as u8
            })
            .collect()
    }

    fn frame(pix: PixelFormat) -> VideoFrame {
        let (cw, ch) = match pix {
            PixelFormat::Yuv420P => (W.div_ceil(2), H.div_ceil(2)),
            _ => (W, H),
        };
        VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: W,
                    data: plane(W, H, 1),
                },
                VideoPlane {
                    stride: cw,
                    data: plane(cw, ch, 2),
                },
                VideoPlane {
                    stride: cw,
                    data: plane(cw, ch, 3),
                },
            ],
        }
    }

    fn dump(name: &str, bytes: &[u8]) {
        if let Some(dir) = std::env::var_os("OXIDEAV_MJPEG_DUMP") {
            let p = std::path::Path::new(&dir).join(format!("ours_{name}.jpg"));
            std::fs::write(p, bytes).unwrap();
        }
    }

    /// (layout, our Huffman encode, our arithmetic encode, validator
    /// arithmetic transcode of the Huffman encode).
    type Case = (&'static str, Vec<u8>, Vec<u8>, &'static [u8]);

    #[test]
    fn arithmetic_encoders_match_the_validator_transcode() {
        let cases: [Case; 3] = {
            let f420 = frame(PixelFormat::Yuv420P);
            let f444 = frame(PixelFormat::Yuv444P);
            let g = plane(W, H, 1);
            let (w, h) = (W as u32, H as u32);
            [
                (
                    "420",
                    enc::encode_jpeg_with_opts(&f420, w, h, PixelFormat::Yuv420P, 85, 0).unwrap(),
                    enc::encode_arith_jpeg_yuv(&f420, w, h, PixelFormat::Yuv420P, 85, 0).unwrap(),
                    include_bytes!("fixtures/arith/ours_420_arith.jpg"),
                ),
                (
                    "444",
                    enc::encode_jpeg_with_opts(&f444, w, h, PixelFormat::Yuv444P, 85, 0).unwrap(),
                    enc::encode_arith_jpeg_yuv(&f444, w, h, PixelFormat::Yuv444P, 85, 0).unwrap(),
                    include_bytes!("fixtures/arith/ours_444_arith.jpg"),
                ),
                (
                    "gray",
                    enc::encode_jpeg_grayscale(w, h, &g, W, 85).unwrap(),
                    enc::encode_arith_jpeg_grayscale(w, h, &g, W, 85, 0).unwrap(),
                    include_bytes!("fixtures/arith/ours_gray_arith.jpg"),
                ),
            ]
        };
        for (name, huff, arith, _) in &cases {
            dump(&format!("{name}_huff"), huff);
            dump(&format!("{name}_arith_ours"), arith);
        }
        for (name, _, arith, reference) in cases {
            assert_eq!(
                scan_segments(&arith),
                scan_segments(reference),
                "{name}: arithmetic scan differs from `jpegtran -arithmetic` of the Huffman encode"
            );
        }
    }
}
