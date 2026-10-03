//! The still-image contract surface over the `docs/image/jpeg/fixtures/`
//! corpus, feature-independent: `probe` / `info` / `decode` /
//! `decode_rgb8` / `decode_rgba8` agree with each other and with the
//! corpus' expected output. Runs with and without `registry`; with it,
//! the framework decoder is pinned to produce the very same planes (one
//! implementation behind both doors).

use std::fs;
use std::path::PathBuf;

use oxideav_mjpeg::{ColorInfo, DecodeOptions, MjpegPixelFormat as F};

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/image/jpeg/fixtures").join(name)
}

fn fixture(name: &str) -> Option<(Vec<u8>, Vec<u8>)> {
    let dir = fixture_dir(name);
    let jpg = fs::read(dir.join("input.jpg")).ok()?;
    let ppm = fs::read(dir.join("expected.ppm"))
        .or_else(|_| fs::read(dir.join("expected.pgm")))
        .ok()?;
    Some((jpg, ppm))
}

/// Minimal P5 / P6 reader: `(channels, width, height, maxval, samples)`.
fn parse_pnm(bytes: &[u8]) -> (usize, u32, u32, u32, Vec<u32>) {
    assert_eq!(bytes[0], b'P');
    let channels = match bytes[1] {
        b'5' => 1,
        b'6' => 3,
        m => panic!("P{}", m as char),
    };
    let mut i = 2;
    let mut toks = Vec::new();
    while toks.len() < 3 {
        while matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
        }
        if bytes[i] == b'#' {
            while bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while !matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r' | b'#') {
            i += 1;
        }
        toks.push(
            std::str::from_utf8(&bytes[s..i])
                .unwrap()
                .parse::<u32>()
                .unwrap(),
        );
    }
    i += 1;
    let (w, h, maxval) = (toks[0], toks[1], toks[2]);
    let payload = &bytes[i..];
    let samples: Vec<u32> = if maxval > 255 {
        payload
            .chunks_exact(2)
            .map(|b| u32::from(b[0]) << 8 | u32::from(b[1]))
            .collect()
    } else {
        payload.iter().map(|&b| u32::from(b)).collect()
    };
    assert_eq!(samples.len(), (w * h) as usize * channels);
    (channels, w, h, maxval, samples)
}

/// Fixture name → the layout `decode` must report, whether `info` must
/// flag JFIF, and the colour description.
const EXPECTED: &[(&str, F, bool, ColorInfo)] = &[
    ("tiny-baseline-1x1", F::Gray8, true, ColorInfo::gray()),
    (
        "baseline-grayscale-32x32",
        F::Gray8,
        true,
        ColorInfo::gray(),
    ),
    ("baseline-rgb-32x32", F::Rgb24, false, ColorInfo::srgb()),
    (
        "baseline-yuv422-32x32",
        F::YuvJ422P,
        true,
        ColorInfo::jfif_ycbcr(),
    ),
    (
        "baseline-yuv411-32x32",
        F::Yuv411P,
        true,
        ColorInfo::jfif_ycbcr(),
    ),
    (
        "baseline-yuv420-128x128-q75",
        F::YuvJ420P,
        true,
        ColorInfo::jfif_ycbcr(),
    ),
    (
        "baseline-q1-low-quality",
        F::YuvJ420P,
        true,
        ColorInfo::jfif_ycbcr(),
    ),
    (
        "baseline-q100-no-loss",
        F::YuvJ420P,
        true,
        ColorInfo::jfif_ycbcr(),
    ),
    (
        "progressive-yuv420-128x128",
        F::YuvJ420P,
        true,
        ColorInfo::jfif_ycbcr(),
    ),
    (
        "multi-scan-non-interleaved",
        F::YuvJ420P,
        true,
        ColorInfo::jfif_ycbcr(),
    ),
    (
        "extended-sequential-12bit",
        F::Gray12Le,
        true,
        ColorInfo::gray(),
    ),
    ("lossless-1986-mode", F::Gray8, true, ColorInfo::gray()),
    ("arithmetic-coded", F::Gray8, true, ColorInfo::gray()),
    (
        "with-restart-interval-8",
        F::YuvJ420P,
        true,
        ColorInfo::jfif_ycbcr(),
    ),
    (
        "with-icc-profile-embedded",
        F::YuvJ420P,
        true,
        ColorInfo::jfif_ycbcr(),
    ),
    (
        "without-jfif-marker",
        F::Yuv420P,
        false,
        ColorInfo::jfif_ycbcr(),
    ),
];

#[test]
fn probe_info_decode_agree_over_the_corpus() {
    let mut seen = 0;
    for (name, format, jfif, color) in EXPECTED {
        let Some((jpg, ppm)) = fixture(name) else {
            continue;
        };
        seen += 1;
        assert!(oxideav_mjpeg::probe(&jpg), "{name}: probe");
        let info = oxideav_mjpeg::info(&jpg).unwrap_or_else(|e| panic!("{name}: info: {e}"));
        let img = oxideav_mjpeg::decode(&jpg).unwrap_or_else(|e| panic!("{name}: decode: {e}"));
        let (channels, w, h, maxval, _) = parse_pnm(&ppm);
        assert_eq!((info.width, info.height), (w, h), "{name}: info dims");
        assert_eq!((img.width(), img.height()), (w, h), "{name}: image dims");
        assert_eq!(info.format, *format, "{name}: info format");
        assert_eq!(img.format(), *format, "{name}: decode format");
        assert_eq!(info.has_jfif, *jfif, "{name}: jfif");
        assert_eq!(info.color, *color, "{name}: info colour");
        assert_eq!(img.color, *color, "{name}: image colour");
        assert_eq!(info.frames, 1);
        assert!(!info.has_alpha);
        assert_eq!(
            info.components as usize,
            if channels == 1 { 1 } else { 3 },
            "{name}"
        );
        assert_eq!(info.precision, if maxval > 255 { 12 } else { 8 }, "{name}");
        assert_eq!(img.precision, info.precision, "{name}");
        assert_eq!(img.planes.len(), format.plane_count(), "{name}: planes");
        for (i, p) in img.planes.iter().enumerate() {
            let (_, ph) = format.plane_dimensions(w, h, i);
            assert_eq!(
                p.stride,
                format.tight_stride(w, h, i),
                "{name}: plane {i} stride"
            );
            assert_eq!(p.data.len(), p.stride * ph, "{name}: plane {i} size");
        }
        assert_eq!(
            info.has_icc,
            *name == "with-icc-profile-embedded",
            "{name}: has_icc"
        );
        assert_eq!(img.metadata.icc.is_some(), info.has_icc, "{name}: icc blob");
        if info.has_icc {
            let icc = img.metadata.icc.as_ref().unwrap();
            // ICC.1 header: profile size (big-endian) in bytes 0..4
            // equals the reassembled length; `acsp` signature at 36..40.
            let declared = u32::from_be_bytes([icc[0], icc[1], icc[2], icc[3]]) as usize;
            assert_eq!(declared, icc.len(), "{name}: ICC size field");
            assert_eq!(&icc[36..40], b"acsp", "{name}: ICC signature");
        }
        assert!(!info.has_exif && !info.has_xmp, "{name}");
        assert_eq!(
            info.progressive,
            matches!(
                *name,
                "progressive-yuv420-128x128" | "multi-scan-non-interleaved"
            ),
            "{name}"
        );
        assert_eq!(info.lossless, name.starts_with("lossless"), "{name}");
        assert_eq!(info.arithmetic, name.starts_with("arithmetic"), "{name}");
        assert!(!info.hierarchical, "{name}");

        // The one-call paths agree with the two-step path.
        let rgb = oxideav_mjpeg::decode_rgb8(&jpg).unwrap();
        let rgba = oxideav_mjpeg::decode_rgba8(&jpg).unwrap();
        assert_eq!(rgb.data, img.to_rgb8(), "{name}: decode_rgb8");
        assert_eq!(rgba.data, img.to_rgba8(), "{name}: decode_rgba8");
        assert_eq!(rgb.data.len(), (w * h * 3) as usize);
        assert_eq!(rgba.data.len(), (w * h * 4) as usize);
        assert!(rgba.data.iter().skip(3).step_by(4).all(|&a| a == 255));
        for (px3, px4) in rgb.data.chunks(3).zip(rgba.data.chunks(4)) {
            assert_eq!(px3, &px4[..3]);
        }

        // Strict mode accepts the conformant corpus; decode_with limits
        // cut in before decoding.
        assert!(
            oxideav_mjpeg::decode_with(&jpg, &DecodeOptions::new().with_strict(true)).is_ok(),
            "{name}: strict"
        );
        assert!(
            matches!(
                oxideav_mjpeg::decode_with(&jpg, &DecodeOptions::new().with_max_pixels(1)),
                Err(oxideav_mjpeg::Error::LimitExceeded(_))
            ) || w * h <= 1
        );
    }
    assert!(
        seen >= 10,
        "docs/image/jpeg/fixtures not found ({seen} seen)"
    );
}

#[test]
fn to_rgb8_tracks_the_corpus_reference_output() {
    // The corpus `expected.ppm` files were produced by an independent
    // decoder with its own IDCT rounding and chroma interpolation. Our
    // exact T.871 kernel + nearest-neighbour upsampling reproduces the
    // grayscale and RGB-coded fixtures to within IDCT rounding and the
    // subsampled YCbCr ones to within interpolation slack.
    for (name, format, _, _) in EXPECTED {
        let Some((jpg, ppm)) = fixture(name) else {
            continue;
        };
        let img = oxideav_mjpeg::decode(&jpg).unwrap();
        let (channels, w, h, maxval, want) = parse_pnm(&ppm);
        let got = img.to_rgb8();
        let mut worst = 0i64;
        let mut sum = 0i64;
        for (i, px) in got.chunks(3).enumerate() {
            for c in 0..3 {
                let expect = if channels == 1 {
                    want[i]
                } else {
                    want[i * 3 + c]
                };
                // Netpbm 16-bit → 8-bit by the same Round(v·255/maxval).
                let expect8 = ((expect as u64 * 255 + (maxval as u64) / 2) / maxval as u64) as i64;
                let d = (px[c] as i64 - expect8).abs();
                worst = worst.max(d);
                sum += d;
            }
        }
        let mean = sum as f64 / (w as f64 * h as f64 * 3.0);
        // Grayscale / RGB-coded: exact (the corpus IDCT rounds like
        // ours). Subsampled YCbCr: the reference interpolates chroma,
        // we replicate, so individual pixels at chroma edges differ
        // (badly so on the q = 1 fixture's giant blocks) while the mean
        // stays small.
        let (max_worst, max_mean) = match format {
            F::Gray8 | F::Gray12Le | F::Rgb24 => (0, 0.0),
            _ => (128, 3.0),
        };
        eprintln!("{name}: worst {worst} mean {mean:.3}");
        assert!(
            worst <= max_worst && mean <= max_mean,
            "{name}: worst {worst} mean {mean:.3} (limits {max_worst} / {max_mean})"
        );
    }
}

#[cfg(feature = "registry")]
#[test]
fn registry_decoder_hands_out_the_same_planes() {
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};
    for (name, format, _, _) in EXPECTED {
        let Some((jpg, _)) = fixture(name) else {
            continue;
        };
        let img = oxideav_mjpeg::decode(&jpg).unwrap();
        let params = CodecParameters::video(CodecId::new(oxideav_mjpeg::CODEC_ID_STR));
        let mut dec = oxideav_mjpeg::registry::make_decoder(&params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), jpg.clone()))
            .unwrap();
        let Frame::Video(vf) = dec.receive_frame().unwrap() else {
            panic!("{name}: not a video frame");
        };
        assert_eq!(vf.planes.len(), img.planes.len(), "{name}");
        for (a, b) in vf.planes.iter().zip(&img.planes) {
            assert_eq!(a.stride, b.stride, "{name}");
            assert_eq!(a.data, b.data, "{name}");
        }
        // The pixel-format enums map 1:1 by name.
        let core: PixelFormat = (*format).into();
        assert_eq!(format!("{core:?}"), format!("{format:?}"), "{name}");
        let back = oxideav_mjpeg::MjpegPixelFormat::try_from(core).unwrap();
        assert_eq!(back, *format);
        let as_frame = oxideav_core::VideoFrame::from(img.clone());
        assert_eq!(as_frame.planes.len(), img.planes.len());
        let sig = oxideav_core::ColorSignal::from(img.color);
        assert_eq!(sig.matrix.code_point(), img.color.matrix);
        assert_eq!(ColorInfo::from(sig), img.color);
    }
}
