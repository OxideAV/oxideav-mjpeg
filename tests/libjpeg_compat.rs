//! `DecodeOptions::libjpeg_compatible()` reproduces `djpeg`'s default
//! output bit-exactly.
//!
//! # Committed oracle
//!
//! `tests/fixtures/libjpeg/*.jpg` are black-box validator output
//! (libjpeg-turbo 3.2.0 `cjpeg`, `jpegtran -arithmetic`, `ffmpeg`): 4:4:4,
//! 4:2:2, 4:2:0, 4:4:0, 4:1:1, 4×2 luma, mixed chroma factors, ffmpeg's
//! `Y 2×2 / C 1×2` 4:2:2, grey, progressive, restart intervals, arithmetic
//! coding, quality 10 and 100, and frames of 1×1 to 9×9 pixels.
//! `oracle.txt` lists, per file, the FNV-1a-64 hash of the sample bytes
//! `djpeg -pnm <file>` writes (default options: accurate-integer IDCT,
//! interpolating chroma upsampling). Regenerate a line with
//! `djpeg -pnm f.jpg | tail -c <w·h·channels> | <fnv1a64>`.
//!
//! # Live oracle
//!
//! When libturbojpeg is installed it is loaded through `dlsym` (the same
//! black-box harness as `tests/external_roundtrip.rs`) and every fixture
//! plus a set of streams written by this crate's own encoder (every
//! sampling layout, odd sizes, progressive / restart) is decoded by
//! `tjDecompress2` (accurate DCT, default upsampling) and compared byte for
//! byte; the test skips that half silently when the library is missing.

use oxideav_mjpeg::{
    ChromaUpsampling, ConvertOptions, DecodeOptions, EncodeOptions, IdctMethod, JpegImage,
    MjpegPixelFormat, Plane, YcbcrKernel, decode, decode_rgb8, decode_rgb8_with, decode_with,
    encode_rgb8,
};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/libjpeg");

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

struct OracleRow {
    name: String,
    width: u32,
    height: u32,
    channels: usize,
    hash: u64,
}

fn oracle() -> Vec<OracleRow> {
    let text = std::fs::read_to_string(format!("{DIR}/oracle.txt")).unwrap();
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            OracleRow {
                name: f[0].to_string(),
                width: f[1].parse().unwrap(),
                height: f[2].parse().unwrap(),
                channels: f[3].parse().unwrap(),
                hash: u64::from_str_radix(f[4], 16).unwrap(),
            }
        })
        .collect()
}

/// What `djpeg -pnm` writes for this stream under `opts`: the grey plane
/// for one-component frames, packed RGB otherwise.
fn djpeg_like(jpeg: &[u8], opts: &DecodeOptions) -> (u32, u32, Vec<u8>) {
    let img = decode_with(jpeg, opts).unwrap();
    if img.format == MjpegPixelFormat::Gray8 {
        let p = &img.planes[0];
        let w = img.width as usize;
        let mut out = Vec::with_capacity(w * img.height as usize);
        for y in 0..img.height as usize {
            out.extend_from_slice(&p.data[y * p.stride..y * p.stride + w]);
        }
        (img.width, img.height, out)
    } else {
        let rgb = decode_rgb8_with(jpeg, opts).unwrap();
        (rgb.width, rgb.height, rgb.data)
    }
}

#[test]
fn compatible_mode_matches_the_committed_djpeg_oracle() {
    let rows = oracle();
    assert!(rows.len() >= 20, "oracle has {} rows", rows.len());
    let opts = DecodeOptions::libjpeg_compatible();
    let mut mismatched = Vec::new();
    for row in &rows {
        let jpeg = std::fs::read(format!("{DIR}/{}", row.name)).unwrap();
        let (w, h, px) = djpeg_like(&jpeg, &opts);
        assert_eq!((w, h), (row.width, row.height), "{}", row.name);
        assert_eq!(px.len(), (w * h) as usize * row.channels, "{}", row.name);
        if fnv1a64(&px) != row.hash {
            mismatched.push(row.name.clone());
        }
    }
    assert!(mismatched.is_empty(), "differs from djpeg: {mismatched:?}");
}

#[test]
fn defaults_are_unchanged() {
    for row in oracle() {
        let jpeg = std::fs::read(format!("{DIR}/{}", row.name)).unwrap();
        let plain = decode_rgb8(&jpeg).unwrap();
        let with = decode_rgb8_with(&jpeg, &DecodeOptions::new()).unwrap();
        assert_eq!(plain, with, "{}", row.name);
        let img = decode(&jpeg).unwrap();
        assert_eq!(img.to_rgb8(), img.to_rgb8_with(&ConvertOptions::new()));
        assert_eq!(img.to_rgba8(), img.to_rgba8_with(&ConvertOptions::new()));
    }
    assert_eq!(DecodeOptions::new().idct, IdctMethod::Float);
    assert_eq!(DecodeOptions::new().upsampling, ChromaUpsampling::Replicate);
    assert_eq!(DecodeOptions::new().ycbcr, YcbcrKernel::Exact);
}

/// Each knob is independent and each one matters somewhere in the corpus.
#[test]
fn the_three_knobs_are_independent() {
    let read = |n: &str| std::fs::read(format!("{DIR}/{n}")).unwrap();
    let base = DecodeOptions::new();
    // IDCT alone: visible on grey (no conversion at all).
    let g = read("gray.jpg");
    let float = djpeg_like(&g, &base).2;
    let int = djpeg_like(&g, &base.clone().with_idct(IdctMethod::AccurateInteger)).2;
    assert_ne!(float, int);
    // Upsampling alone changes 4:2:0 but not 4:4:4.
    let tri = base.clone().with_upsampling(ChromaUpsampling::Triangular);
    let c420 = read("c420.jpg");
    assert_ne!(djpeg_like(&c420, &base).2, djpeg_like(&c420, &tri).2);
    let c444 = read("c444.jpg");
    assert_eq!(djpeg_like(&c444, &base).2, djpeg_like(&c444, &tri).2);
    // The YCbCr kernel alone never touches grey, and over every (Y, Cb)
    // pair (Cr varying too) it differs from the exact kernel by at most
    // one level — but it does differ.
    let fixed = base.clone().with_ycbcr(YcbcrKernel::FixedPoint16);
    assert_eq!(djpeg_like(&g, &base).2, djpeg_like(&g, &fixed).2);
    let plane = |f: &dyn Fn(usize, usize) -> u8| {
        let d: Vec<u8> = (0..256 * 256).map(|i| f(i % 256, i / 256)).collect();
        Plane::new(256, d)
    };
    let grid = JpegImage::new(
        256,
        256,
        MjpegPixelFormat::YuvJ444P,
        vec![
            plane(&|x, _| x as u8),
            plane(&|_, y| y as u8),
            plane(&|x, y| ((x * 7 + y * 3) % 256) as u8),
        ],
    )
    .unwrap();
    let exact = grid.to_rgb8();
    let fixed16 = grid.to_rgb8_with(&ConvertOptions::new().with_ycbcr(YcbcrKernel::FixedPoint16));
    assert_ne!(exact, fixed16);
    assert!(exact.iter().zip(&fixed16).all(|(a, b)| a.abs_diff(*b) <= 1));
    // The preset is exactly the three knobs together.
    let all = base
        .with_idct(IdctMethod::AccurateInteger)
        .with_upsampling(ChromaUpsampling::Triangular)
        .with_ycbcr(YcbcrKernel::FixedPoint16);
    assert_eq!(all, DecodeOptions::libjpeg_compatible());
    assert_eq!(
        DecodeOptions::libjpeg_compatible().convert_options(),
        ConvertOptions::libjpeg_compatible()
    );
}

/// A smooth RGB test card with hard edges (sharp chroma transitions).
fn card(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let edge = (x / 5 + y / 3) % 2 == 0;
            v.push(((x * 255) / w.max(1)) as u8);
            v.push(if edge {
                230
            } else {
                ((y * 255) / h.max(1)) as u8
            });
            v.push(((x * 7 + y * 13) % 256) as u8);
        }
    }
    v
}

/// Streams from this crate's encoder covering every layout.
fn own_streams() -> Vec<(String, Vec<u8>)> {
    let samplings: [&[(u8, u8)]; 7] = [
        &[(1, 1), (1, 1), (1, 1)],
        &[(2, 1), (1, 1), (1, 1)],
        &[(2, 2), (1, 1), (1, 1)],
        &[(1, 2), (1, 1), (1, 1)],
        &[(4, 1), (1, 1), (1, 1)],
        &[(2, 2), (1, 2), (1, 2)],
        &[(2, 2), (2, 1), (1, 2)],
    ];
    let mut out = Vec::new();
    for (w, h) in [(1, 1), (3, 5), (2, 9), (37, 29), (64, 48)] {
        let rgb = card(w, h);
        for s in samplings {
            for (tag, opts) in [
                ("seq", EncodeOptions::new().with_quality(85)),
                (
                    "prog",
                    EncodeOptions::new().with_quality(60).with_progressive(true),
                ),
                (
                    "rst",
                    EncodeOptions::new()
                        .with_quality(95)
                        .with_restart_interval(1),
                ),
            ] {
                let opts = opts.with_sampling(s.to_vec());
                if let Ok(j) = encode_rgb8(w, h, &rgb, &opts) {
                    out.push((format!("{w}x{h} {s:?} {tag}"), j));
                }
            }
        }
    }
    out
}

#[test]
fn compatible_mode_matches_libturbojpeg_live() {
    if !turbo::available() {
        eprintln!("libjpeg_compat: libturbojpeg not found; live comparison skipped");
        return;
    }
    let mut cases: Vec<(String, Vec<u8>)> = oracle()
        .into_iter()
        .map(|r| {
            (
                r.name.clone(),
                std::fs::read(format!("{DIR}/{}", r.name)).unwrap(),
            )
        })
        .collect();
    cases.extend(own_streams());
    assert!(cases.len() > 100);
    let opts = DecodeOptions::libjpeg_compatible();
    let mut bad = Vec::new();
    for (name, jpeg) in &cases {
        let (w, h, ours) = djpeg_like(jpeg, &opts);
        let grey = ours.len() == (w * h) as usize;
        let theirs = turbo::decode(jpeg, grey).unwrap_or_else(|| panic!("{name}: tjDecompress2"));
        if theirs != ours {
            let n = theirs.iter().zip(&ours).filter(|(a, b)| a != b).count();
            bad.push(format!("{name}: {n} bytes"));
        }
    }
    assert!(bad.is_empty(), "differs from libturbojpeg: {bad:#?}");
}

#[allow(unsafe_code)]
mod turbo {
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
    /// Pixel formats and flags of the TurboJPEG API (`TJPF_RGB`,
    /// `TJPF_GRAY`, `TJFLAG_ACCURATEDCT`).
    const PF_RGB: c_int = 0;
    const PF_GRAY: c_int = 6;
    const FLAG_ACCURATE_DCT: c_int = 4096;

    type Handle = *mut std::ffi::c_void;

    // The library lives in a never-closed `'static`, so the detached
    // function pointers below cannot dangle.
    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            CANDIDATES
                .iter()
                // SAFETY: loading a system library may run its initialisers;
                // acceptable for black-box test tooling.
                .find_map(|n| unsafe { Library::open(n) }.ok())
        })
        .as_ref()
    }

    pub fn available() -> bool {
        lib().is_some()
    }

    /// Decode with `tjDecompress2` to packed RGB (or grey).
    pub fn decode(data: &[u8], grey: bool) -> Option<Vec<u8>> {
        type InitFn = unsafe extern "C" fn() -> Handle;
        type DestroyFn = unsafe extern "C" fn(Handle) -> c_int;
        type HeaderFn = unsafe extern "C" fn(
            Handle,
            *const c_uchar,
            c_ulong,
            *mut c_int,
            *mut c_int,
            *mut c_int,
        ) -> c_int;
        type DecompressFn = unsafe extern "C" fn(
            Handle,
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
        // SAFETY: the symbols have the documented TurboJPEG signatures;
        // every buffer passed is sized for the reported geometry.
        unsafe {
            let init: InitFn = l.get::<InitFn>(c"tjInitDecompress").ok()?.into_inner();
            let destroy: DestroyFn = l.get::<DestroyFn>(c"tjDestroy").ok()?.into_inner();
            let header: HeaderFn = l.get::<HeaderFn>(c"tjDecompressHeader2").ok()?.into_inner();
            let decompress: DecompressFn =
                l.get::<DecompressFn>(c"tjDecompress2").ok()?.into_inner();
            let h = init();
            if h.is_null() {
                return None;
            }
            let (mut w, mut ht, mut ss) = (0, 0, 0);
            if header(
                h,
                data.as_ptr(),
                data.len() as c_ulong,
                &mut w,
                &mut ht,
                &mut ss,
            ) != 0
            {
                destroy(h);
                return None;
            }
            let bpp = if grey { 1 } else { 3 };
            let pitch = w as usize * bpp;
            let mut buf = vec![0u8; pitch * ht as usize];
            let rc = decompress(
                h,
                data.as_ptr(),
                data.len() as c_ulong,
                buf.as_mut_ptr(),
                w,
                pitch as c_int,
                ht,
                if grey { PF_GRAY } else { PF_RGB },
                FLAG_ACCURATE_DCT,
            );
            destroy(h);
            (rc == 0).then_some(buf)
        }
    }
}
