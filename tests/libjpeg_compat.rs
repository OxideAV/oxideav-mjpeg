//! `DecodeOptions::libjpeg_compatible()` — bit-exact against the
//! reference decoder's defaults.
//!
//! Three oracles, none of which reads any reference-implementation
//! source:
//!
//! 1. `tests/fixtures/libjpeg_compat/*.jpg` with a sibling `.ppm` /
//!    `.pgm`: streams from `cjpeg` / `ffmpeg` (4:4:4, 4:2:2, 4:2:0,
//!    4:1:1, 4:4:0, grey, progressive, restart intervals, arithmetic
//!    coding, odd sizes, 2- and 3-wide chroma planes, and ffmpeg's
//!    `Y 2×2 / C 1×2` 4:2:2 signalling) next to `djpeg`'s own output.
//! 2. The `docs/image/jpeg/fixtures/` corpus, whose `expected.ppm` is
//!    the reference decoder's output (skipped when the umbrella `docs/`
//!    tree is not checked out).
//! 3. libturbojpeg loaded at run time through `dlsym` (skipped when it
//!    is not installed): synthetic pictures go through `tjCompress2`
//!    at every subsampling and come back through `tjDecompress2` with
//!    its default flags (accurate IDCT, fancy upsampling), which must
//!    equal our compat decode byte for byte.
//!
//! The default options stay what they were: `decode_rgb8` equals
//! `decode_rgb8_with(&DecodeOptions::new())`, and the two IDCT kernels
//! never differ by more than one level on any sample.

#![allow(unsafe_code)]

use oxideav_mjpeg::{
    ChromaUpsampling, ConvertOptions, DecodeOptions, IdctMethod, MjpegPixelFormat, YcbcrKernel,
    decode_rgb8, decode_rgb8_with, decode_rgba8_with, decode_with,
};
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Netpbm (P5 / P6, maxval 255) reader.
// ---------------------------------------------------------------------------

struct Pnm {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u8>,
}

fn parse_pnm(bytes: &[u8]) -> Pnm {
    let mut i = 0;
    let mut tokens: Vec<String> = Vec::new();
    while tokens.len() < 4 {
        while bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if bytes[i] == b'#' {
            while bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        tokens.push(std::str::from_utf8(&bytes[s..i]).unwrap().to_string());
    }
    i += 1; // the single whitespace after maxval
    let channels = match tokens[0].as_str() {
        "P6" => 3,
        "P5" => 1,
        other => panic!("unsupported PNM magic {other}"),
    };
    assert_eq!(tokens[3], "255", "8-bit PNM expected");
    let width: usize = tokens[1].parse().unwrap();
    let height: usize = tokens[2].parse().unwrap();
    let data = bytes[i..].to_vec();
    assert_eq!(data.len(), width * height * channels, "PNM payload length");
    Pnm {
        width,
        height,
        channels,
        data,
    }
}

/// Decode `jpeg` in compat mode into the sample layout of `reference`
/// (packed RGB for a P6, the grey plane for a P5) and return it.
fn compat_decode_like(jpeg: &[u8], reference: &Pnm) -> Vec<u8> {
    let opts = DecodeOptions::libjpeg_compatible();
    if reference.channels == 3 {
        let img = decode_rgb8_with(jpeg, &opts).expect("decode_rgb8_with");
        assert_eq!(
            (img.width as usize, img.height as usize),
            (reference.width, reference.height)
        );
        img.data
    } else {
        let img = decode_with(jpeg, &opts).expect("decode_with");
        assert_eq!(img.format, MjpegPixelFormat::Gray8);
        assert_eq!(
            (img.width as usize, img.height as usize),
            (reference.width, reference.height)
        );
        let p = &img.planes[0];
        let mut out = Vec::with_capacity(reference.width * reference.height);
        for y in 0..reference.height {
            out.extend_from_slice(&p.data[y * p.stride..y * p.stride + reference.width]);
        }
        out
    }
}

/// Count the differing pixels and the largest per-sample delta.
fn diff_stats(a: &[u8], b: &[u8], channels: usize) -> (usize, i32) {
    assert_eq!(a.len(), b.len());
    let mut pixels = 0;
    let mut max = 0i32;
    for (pa, pb) in a.chunks(channels).zip(b.chunks(channels)) {
        let mut d = false;
        for (x, y) in pa.iter().zip(pb) {
            let delta = (*x as i32 - *y as i32).abs();
            if delta != 0 {
                d = true;
                max = max.max(delta);
            }
        }
        if d {
            pixels += 1;
        }
    }
    (pixels, max)
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/libjpeg_compat")
}

// ---------------------------------------------------------------------------
// 1. Committed cjpeg / ffmpeg streams next to djpeg's output.
// ---------------------------------------------------------------------------

#[test]
fn committed_streams_are_bit_exact_against_djpeg() {
    let mut entries: Vec<PathBuf> = fs::read_dir(fixture_dir())
        .expect("tests/fixtures/libjpeg_compat")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "jpg"))
        .collect();
    entries.sort();
    assert!(entries.len() >= 15, "fixture set shrank: {}", entries.len());
    for jpg_path in entries {
        let name = jpg_path.file_name().unwrap().to_string_lossy().to_string();
        let jpeg = fs::read(&jpg_path).unwrap();
        let ppm_path = jpg_path.with_extension("ppm");
        let reference =
            parse_pnm(&fs::read(&ppm_path).unwrap_or_else(|e| {
                panic!("{name}: missing reference {}: {e}", ppm_path.display())
            }));
        let ours = compat_decode_like(&jpeg, &reference);
        let (pixels, max) = diff_stats(&ours, &reference.data, reference.channels);
        assert_eq!(
            (pixels, max),
            (0, 0),
            "{name}: {pixels} of {} pixels differ from djpeg (max delta {max})",
            reference.width * reference.height
        );
    }
}

/// ffmpeg signals 4:2:2 as `Y 2×2 / C 1×2`; the layout is decided by the
/// sampling *ratio*, so those planes come out as native 4:2:2 (not
/// nearest-upsampled 4:4:4) and reach the triangle filter.
#[test]
fn ffmpeg_style_422_is_a_native_422_layout() {
    let jpeg = fs::read(fixture_dir().join("ffmpeg-yuvj422p-33x17.jpg")).unwrap();
    let img = decode_with(&jpeg, &DecodeOptions::new()).unwrap();
    assert_eq!(img.format, MjpegPixelFormat::YuvJ422P);
    assert_eq!(img.planes[1].data.len(), 17 * 17);
    assert_eq!(img.planes[2].stride, 17);
}

// ---------------------------------------------------------------------------
// 2. The docs/ corpus (optional).
// ---------------------------------------------------------------------------

#[test]
fn docs_corpus_lossy_fixtures_are_bit_exact_in_compat_mode() {
    // Every 8-bit lossy fixture; the 12-bit and lossless ones are
    // outside the compat contract (and already exact in docs_corpus.rs).
    const NAMES: &[&str] = &[
        "tiny-baseline-1x1",
        "baseline-grayscale-32x32",
        "baseline-rgb-32x32",
        "baseline-yuv422-32x32",
        "baseline-yuv411-32x32",
        "baseline-yuv420-128x128-q75",
        "baseline-q1-low-quality",
        "baseline-q100-no-loss",
        "progressive-yuv420-128x128",
        "multi-scan-non-interleaved",
        "arithmetic-coded",
        "with-restart-interval-8",
        "with-icc-profile-embedded",
        "without-jfif-marker",
    ];
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/image/jpeg/fixtures");
    if !root.is_dir() {
        eprintln!(
            "docs corpus not checked out at {}; skipping",
            root.display()
        );
        return;
    }
    let mut checked = 0;
    for name in NAMES {
        let dir = root.join(name);
        let Ok(jpeg) = fs::read(dir.join("input.jpg")) else {
            eprintln!("skip {name}: no input.jpg");
            continue;
        };
        let reference = ["expected.ppm", "expected.pgm"]
            .iter()
            .find_map(|f| fs::read(dir.join(f)).ok())
            .map(|b| parse_pnm(&b));
        let Some(reference) = reference else {
            eprintln!("skip {name}: no expected.p?m");
            continue;
        };
        let ours = compat_decode_like(&jpeg, &reference);
        let (pixels, max) = diff_stats(&ours, &reference.data, reference.channels);
        assert_eq!(
            (pixels, max),
            (0, 0),
            "{name}: {pixels} pixels differ from the corpus reference (max delta {max})"
        );
        checked += 1;
    }
    assert!(checked > 0, "no docs fixture was checked");
}

// ---------------------------------------------------------------------------
// Default behaviour is unchanged; the knobs are what they say.
// ---------------------------------------------------------------------------

#[test]
fn defaults_are_unchanged_and_knobs_are_independent() {
    let jpeg = fs::read(fixture_dir().join("cjpeg-noise-17x9-420-q75.jpg")).unwrap();
    let dflt = DecodeOptions::new();
    assert_eq!(dflt.idct, IdctMethod::Float);
    assert_eq!(dflt.convert, ConvertOptions::new());
    assert_eq!(dflt.convert.upsampling, ChromaUpsampling::Nearest);
    assert_eq!(dflt.convert.ycbcr, YcbcrKernel::Exact);
    assert_eq!(
        decode_rgb8(&jpeg).unwrap().data,
        decode_rgb8_with(&jpeg, &dflt).unwrap().data
    );

    let compat = DecodeOptions::libjpeg_compatible();
    assert_eq!(compat.idct, IdctMethod::Integer);
    assert_eq!(compat.convert, ConvertOptions::libjpeg_compatible());
    assert_eq!(compat, DecodeOptions::new().with_libjpeg_compat());
    assert_eq!(compat.max_pixels, dflt.max_pixels, "limits untouched");

    // The two IDCT kernels agree to within one level on every sample.
    let flt = decode_with(&jpeg, &dflt).unwrap();
    let int = decode_with(&jpeg, &DecodeOptions::new().with_idct(IdctMethod::Integer)).unwrap();
    for (pf, pi) in flt.planes.iter().zip(&int.planes) {
        assert_eq!(pf.data.len(), pi.data.len());
        assert!(
            pf.data
                .iter()
                .zip(&pi.data)
                .all(|(a, b)| (*a as i32 - *b as i32).abs() <= 1),
            "float and integer IDCT differ by more than 1"
        );
    }

    // The conversion knobs act on the same planes.
    let nearest_exact = flt.to_rgb8();
    let nearest_fixed =
        flt.to_rgb8_with(&ConvertOptions::new().with_ycbcr(YcbcrKernel::FixedPoint));
    let tri_exact =
        flt.to_rgb8_with(&ConvertOptions::new().with_upsampling(ChromaUpsampling::Triangle));
    assert_eq!(nearest_exact, flt.to_rgb8_with(&ConvertOptions::new()));
    assert_ne!(
        nearest_exact, tri_exact,
        "triangle filter changes 4:2:0 output"
    );
    assert!(
        diff_stats(&nearest_exact, &nearest_fixed, 3).1 <= 1,
        "the two YCbCr kernels differ by at most one level"
    );
    // RGBA carries the same bytes plus opaque alpha.
    let rgba = decode_rgba8_with(&jpeg, &compat).unwrap().data;
    let rgb = decode_rgb8_with(&jpeg, &compat).unwrap().data;
    assert_eq!(rgba.len(), rgb.len() / 3 * 4);
    for (px4, px3) in rgba.chunks(4).zip(rgb.chunks(3)) {
        assert_eq!(&px4[..3], px3);
        assert_eq!(px4[3], 255);
    }
}

// ---------------------------------------------------------------------------
// 3. libturbojpeg through dlsym.
// ---------------------------------------------------------------------------

/// Deterministic test pictures: xorshift noise plus a smooth gradient so
/// every chroma filter tap matters.
fn synth_rgb(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s & 0x3F) as i32 - 32;
            let o = (y * w + x) * 3;
            let gx = ((x * 255) / w.max(1)) as i32;
            let gy = ((y * 255) / h.max(1)) as i32;
            let gd = (((x + y) * 127) / (w + h).max(1) + 64) as i32;
            out[o] = (gx + n).clamp(0, 255) as u8;
            out[o + 1] = (gy - n).clamp(0, 255) as u8;
            out[o + 2] = (gd + n / 2).clamp(0, 255) as u8;
        }
    }
    out
}

#[test]
fn libturbojpeg_default_decode_matches_compat_mode() {
    if !libjpeg::available() {
        eprintln!(
            "libjpeg_compat: libturbojpeg not found; skipping. Install with \
             `brew install jpeg-turbo` (macOS) or `apt install libturbojpeg0-dev`."
        );
        return;
    }
    let sizes = [
        (1, 1),
        (2, 9),
        (3, 5),
        (5, 3),
        (8, 8),
        (17, 9),
        (33, 17),
        (64, 64),
        (127, 61),
    ];
    let samplings = [
        ("444", libjpeg::TJSAMP_444),
        ("422", libjpeg::TJSAMP_422),
        ("420", libjpeg::TJSAMP_420),
        ("gray", libjpeg::TJSAMP_GRAY),
        ("440", libjpeg::TJSAMP_440),
        ("411", libjpeg::TJSAMP_411),
    ];
    let compat = DecodeOptions::libjpeg_compatible();
    let mut cases = 0;
    for (i, &(w, h)) in sizes.iter().enumerate() {
        let rgb = synth_rgb(w, h, 0x1234_5678 ^ (i as u32 * 0x9E37_79B9));
        for &(sname, samp) in &samplings {
            for &q in &[30u8, 75, 95] {
                let Some(jpeg) = libjpeg::encode_rgb(&rgb, w as u32, h as u32, samp, q) else {
                    panic!("{w}x{h} {sname} q{q}: tjCompress2 failed");
                };
                let oracle = libjpeg::decode_to_rgb(&jpeg).expect("tjDecompress2");
                assert_eq!((oracle.width, oracle.height), (w as u32, h as u32));
                let ours = decode_rgb8_with(&jpeg, &compat).unwrap();
                let (pixels, max) = diff_stats(&ours.data, &oracle.rgb, 3);
                assert_eq!(
                    (pixels, max),
                    (0, 0),
                    "{w}x{h} {sname} q{q}: {pixels} of {} pixels differ from libturbojpeg \
                     (max delta {max})",
                    w * h
                );
                cases += 1;
            }
        }
    }
    assert_eq!(cases, sizes.len() * samplings.len() * 3);
}

/// Our arithmetic-coded streams decode in libturbojpeg to exactly what
/// our compat decoder sees (the shared statistics areas of T.81 §B.2.3
/// on both sides); the RGB stream carries no JFIF APP0, so the
/// reference keeps the R/G/B interpretation.
#[cfg(feature = "registry")]
#[test]
fn our_arithmetic_streams_decode_identically_in_libturbojpeg() {
    if !libjpeg::available() {
        eprintln!("libjpeg_compat: libturbojpeg not found; skipping.");
        return;
    }
    use oxideav_mjpeg::encoder as enc;
    let (w, h) = (37usize, 23usize);
    let rgb = synth_rgb(w, h, 0xC0FF_EE11);
    let grey: Vec<u8> = rgb
        .chunks(3)
        .map(|p| ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000) as u8)
        .collect();
    // 4:2:0 planes from the standalone encoder's own colour path.
    let yuv = decode_with(
        &oxideav_mjpeg::encode_rgb8(
            w as u32,
            h as u32,
            &rgb,
            &oxideav_mjpeg::EncodeOptions::new(),
        )
        .unwrap(),
        &DecodeOptions::new(),
    )
    .unwrap();
    assert_eq!(yuv.format, MjpegPixelFormat::YuvJ420P);
    let frame: oxideav_core::VideoFrame = yuv.clone().into();
    let streams: Vec<(&str, Vec<u8>)> = vec![
        (
            "arith yuv420",
            enc::encode_arith_jpeg_yuv(
                &frame,
                w as u32,
                h as u32,
                oxideav_core::PixelFormat::Yuv420P,
                80,
                0,
            )
            .unwrap(),
        ),
        (
            "arith yuv420 rst",
            enc::encode_arith_jpeg_yuv(
                &frame,
                w as u32,
                h as u32,
                oxideav_core::PixelFormat::Yuv420P,
                80,
                2,
            )
            .unwrap(),
        ),
        (
            "arith rgb24",
            enc::encode_arith_jpeg_rgb24(w as u32, h as u32, &rgb, w * 3, 80, 0).unwrap(),
        ),
        (
            "arith rgb24 rst",
            enc::encode_arith_jpeg_rgb24(w as u32, h as u32, &rgb, w * 3, 80, 3).unwrap(),
        ),
        (
            "arith grey",
            enc::encode_arith_jpeg_grayscale(w as u32, h as u32, &grey, w, 80, 0).unwrap(),
        ),
        (
            "baseline rgb24",
            enc::encode_jpeg_rgb24(w as u32, h as u32, &rgb, w * 3, 80).unwrap(),
        ),
    ];
    let compat = DecodeOptions::libjpeg_compatible();
    for (name, jpeg) in streams {
        let oracle =
            libjpeg::decode_to_rgb(&jpeg).unwrap_or_else(|| panic!("{name}: tjDecompress2"));
        let ours = decode_rgb8_with(&jpeg, &compat).unwrap();
        let (pixels, max) = diff_stats(&ours.data, &oracle.rgb, 3);
        assert_eq!(
            (pixels, max),
            (0, 0),
            "{name}: {pixels} pixels differ (max {max})"
        );
        // And the picture is the one we encoded (an RGB stream read as
        // YCbCr would be ~8 dB, a mis-adapted Q-coder pure noise).
        let (_, max_src) = diff_stats(
            &oracle.rgb,
            if name.contains("grey") {
                &oracle.rgb
            } else {
                &rgb
            },
            3,
        );
        assert!(
            max_src <= 48,
            "{name}: reference decode is {max_src} levels off the source"
        );
    }
}

// ---------------------------------------------------------------------------
// libturbojpeg shim (dlsym; see tests/external_roundtrip.rs).
// ---------------------------------------------------------------------------

mod libjpeg {
    use dlsym::Library;
    use std::os::raw::{c_int, c_uchar, c_ulong};
    use std::sync::OnceLock;

    const CANDIDATES: &[&str] = &[
        "libturbojpeg.dylib",
        "libturbojpeg.0.dylib",
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "turbojpeg.dll",
    ];

    const TJPF_RGB: c_int = 0;
    pub const TJSAMP_444: c_int = 0;
    pub const TJSAMP_422: c_int = 1;
    pub const TJSAMP_420: c_int = 2;
    pub const TJSAMP_GRAY: c_int = 3;
    pub const TJSAMP_440: c_int = 4;
    pub const TJSAMP_411: c_int = 5;

    type TjHandle = *mut std::ffi::c_void;
    type InitFn = unsafe extern "C" fn() -> TjHandle;
    type DestroyFn = unsafe extern "C" fn(TjHandle) -> c_int;
    type FreeFn = unsafe extern "C" fn(*mut c_uchar) -> c_int;

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: libturbojpeg is well-behaved at load time.
                if let Ok(l) = unsafe { Library::open(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    pub fn available() -> bool {
        lib().is_some()
    }

    /// `tjCompress2` with the default flags.
    pub fn encode_rgb(
        rgb: &[u8],
        width: u32,
        height: u32,
        samp: c_int,
        quality: u8,
    ) -> Option<Vec<u8>> {
        type CompressFn = unsafe extern "C" fn(
            TjHandle,
            *const c_uchar,
            c_int,
            c_int,
            c_int,
            c_int,
            *mut *mut c_uchar,
            *mut c_ulong,
            c_int,
            c_int,
            c_int,
        ) -> c_int;
        let l = lib()?;
        let stride = (width as usize).checked_mul(3)?;
        if rgb.len() < stride.checked_mul(height as usize)? {
            return None;
        }
        unsafe {
            let init: InitFn = l.get::<InitFn>(c"tjInitCompress").ok()?.into_inner();
            let destroy: DestroyFn = l.get::<DestroyFn>(c"tjDestroy").ok()?.into_inner();
            let free: FreeFn = l.get::<FreeFn>(c"tjFree").ok()?.into_inner();
            let compress: CompressFn = l.get::<CompressFn>(c"tjCompress2").ok()?.into_inner();
            let handle = init();
            if handle.is_null() {
                return None;
            }
            let mut buf: *mut c_uchar = std::ptr::null_mut();
            let mut size: c_ulong = 0;
            let rc = compress(
                handle,
                rgb.as_ptr(),
                width as c_int,
                stride as c_int,
                height as c_int,
                TJPF_RGB,
                &mut buf,
                &mut size,
                samp,
                quality as c_int,
                0,
            );
            let result = if rc == 0 && !buf.is_null() && size > 0 {
                Some(std::slice::from_raw_parts(buf, size as usize).to_vec())
            } else {
                None
            };
            if !buf.is_null() {
                let _ = free(buf);
            }
            let _ = destroy(handle);
            result
        }
    }

    pub struct DecodedRgb {
        pub width: u32,
        pub height: u32,
        pub rgb: Vec<u8>,
    }

    /// `tjDecompressHeader2` + `tjDecompress2` with the **default** flags
    /// (no `TJFLAG_FASTDCT`, no `TJFLAG_FASTUPSAMPLE`): the accurate
    /// IDCT and fancy upsampling `djpeg` uses by default.
    pub fn decode_to_rgb(data: &[u8]) -> Option<DecodedRgb> {
        type HeaderFn = unsafe extern "C" fn(
            TjHandle,
            *const c_uchar,
            c_ulong,
            *mut c_int,
            *mut c_int,
            *mut c_int,
        ) -> c_int;
        type DecompressFn = unsafe extern "C" fn(
            TjHandle,
            *const c_uchar,
            c_ulong,
            *mut c_uchar,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
        ) -> c_int;
        let l = lib()?;
        unsafe {
            let init: InitFn = l.get::<InitFn>(c"tjInitDecompress").ok()?.into_inner();
            let destroy: DestroyFn = l.get::<DestroyFn>(c"tjDestroy").ok()?.into_inner();
            let header: HeaderFn = l.get::<HeaderFn>(c"tjDecompressHeader2").ok()?.into_inner();
            let decompress: DecompressFn =
                l.get::<DecompressFn>(c"tjDecompress2").ok()?.into_inner();
            let handle = init();
            if handle.is_null() {
                return None;
            }
            let (mut w, mut h, mut subsamp) = (0 as c_int, 0 as c_int, 0 as c_int);
            let rc = header(
                handle,
                data.as_ptr(),
                data.len() as c_ulong,
                &mut w,
                &mut h,
                &mut subsamp,
            );
            if rc != 0 || w <= 0 || h <= 0 {
                let _ = destroy(handle);
                return None;
            }
            let stride = (w as usize).checked_mul(3)?;
            let mut buf = vec![0u8; stride.checked_mul(h as usize)?];
            let rc = decompress(
                handle,
                data.as_ptr(),
                data.len() as c_ulong,
                buf.as_mut_ptr(),
                w,
                stride as c_int,
                h,
                TJPF_RGB,
                0,
            );
            let _ = destroy(handle);
            if rc != 0 {
                return None;
            }
            Some(DecodedRgb {
                width: w as u32,
                height: h as u32,
                rgb: buf,
            })
        }
    }
}
