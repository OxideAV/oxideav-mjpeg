//! The registry encoder contract, cell by cell: every layout the
//! encoder declares through `CodecCapabilities::accepted_pixel_formats`
//! × representative sizes (1×1, odd, MCU-unaligned, MCU-aligned) ×
//! the option sets that apply to it encodes through the registry,
//! decodes through the registry to the label the encoder stamped on its
//! output parameters (which is also what the `jpeg` demuxer declares),
//! and reconstructs the picture (bit-exact for lossless, above a PSNR
//! floor for the DCT processes). When the black-box reference decoder
//! binary is on `PATH` every stream is also decoded there and must
//! come back with the declared geometry and sample precision.
//!
//! Needs the default `registry` feature.
#![cfg(feature = "registry")]

use std::io::Write;
use std::process::{Command, Stdio};

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, Frame, Packet, PixelFormat, RuntimeContext, TimeBase,
    VideoFrame,
};
use oxideav_mjpeg::registry::{
    accepted_pixel_formats, coded_pixel_format, make_decoder, make_encoder, ACCEPTED_PIXEL_FORMATS,
};
use oxideav_mjpeg::{ChromaSubsampling, ColorSignalling, EncodeOptions, JpegImage, JpegProcess};
use PixelFormat as P;

/// Representative geometries: a single pixel, an odd pair, an
/// MCU-unaligned size with odd chroma extents, and an aligned one.
const SIZES: &[(u32, u32)] = &[(1, 1), (2, 3), (17, 11), (33, 18), (64, 48)];

/// The range-agnostic aliases the encoder accepts without declaring.
const ALIASES: &[PixelFormat] = &[P::Yuv444P, P::Yuv422P, P::Yuv420P];

// ---------------------------------------------------------------------------
// Layout facts (mirrors the contract table; the test, not the crate,
// owns this copy so a drift shows up as a failure).
// ---------------------------------------------------------------------------

/// Bits per sample, bytes per sample, plane count (1 = packed), packed
/// channels, chroma divisors.
fn facts(p: PixelFormat) -> (u32, usize, usize, usize, (usize, usize)) {
    match p {
        P::Gray8 => (8, 1, 1, 1, (1, 1)),
        P::Gray10Le => (10, 2, 1, 1, (1, 1)),
        P::Gray12Le => (12, 2, 1, 1, (1, 1)),
        P::Gray16Le => (16, 2, 1, 1, (1, 1)),
        P::YuvJ444P | P::Yuv444P => (8, 1, 3, 1, (1, 1)),
        P::YuvJ422P | P::Yuv422P => (8, 1, 3, 1, (2, 1)),
        P::YuvJ420P | P::Yuv420P => (8, 1, 3, 1, (2, 2)),
        P::Yuv411P => (8, 1, 3, 1, (4, 1)),
        P::Yuv440P => (8, 1, 3, 1, (1, 2)),
        P::Yuv444P12Le => (12, 2, 3, 1, (1, 1)),
        P::Yuv422P12Le => (12, 2, 3, 1, (2, 1)),
        P::Yuv420P12Le => (12, 2, 3, 1, (2, 2)),
        P::Rgb24 => (8, 1, 1, 3, (1, 1)),
        P::Rgb48Le => (16, 2, 1, 3, (1, 1)),
        P::Gbrp10Le => (10, 2, 3, 1, (1, 1)),
        P::Gbrp12Le => (12, 2, 3, 1, (1, 1)),
        P::Gbrp14Le => (14, 2, 3, 1, (1, 1)),
        P::Cmyk | P::CmykInverted => (8, 1, 1, 4, (1, 1)),
        other => panic!("{other:?} is not in the contract"),
    }
}

fn is_lossless_only(p: PixelFormat) -> bool {
    matches!(
        p,
        P::Gray10Le | P::Gray16Le | P::Rgb48Le | P::Gbrp10Le | P::Gbrp12Le | P::Gbrp14Le
    )
}

/// The decoder labels a lossless stream of this layout with the same
/// label (4:4:4 YCbCr and 12-bit subsampled YCbCr are refused on the
/// lossless process; see `set_encode_options`).
fn lossless_labelled(p: PixelFormat) -> bool {
    !matches!(
        p,
        P::YuvJ444P | P::Yuv444P | P::Yuv444P12Le | P::Yuv422P12Le | P::Yuv420P12Le
    )
}

// ---------------------------------------------------------------------------
// Synthetic pictures: smooth gradients (no wrap-around edges) so the
// DCT processes reconstruct them well above the PSNR floor.
// ---------------------------------------------------------------------------

/// Sample `c` of a `w × h` picture at `(x, y)` for a `bits`-bit range.
fn sample(c: usize, x: usize, y: usize, w: usize, h: usize, bits: u32) -> u16 {
    let max = ((1u32 << bits) - 1) as f64;
    let fx = x as f64 / w.max(2) as f64;
    let fy = y as f64 / h.max(2) as f64;
    let v = match c {
        0 => 0.15 + 0.6 * fx + 0.2 * fy,
        1 => 0.6 - 0.3 * fx + 0.1 * fy,
        2 => 0.3 + 0.2 * fx + 0.4 * fy,
        _ => 0.5 - 0.2 * fx - 0.2 * fy,
    };
    ((v.clamp(0.0, 1.0) * max).round() as u32).min(max as u32) as u16
}

/// Tightly packed source planes as `u16` samples per component, at the
/// A.1.1 extents the layout implies.
fn source_planes(p: PixelFormat, w: u32, h: u32) -> Vec<Vec<u16>> {
    let (bits, _, planes, channels, (dh, dv)) = facts(p);
    let (w, h) = (w as usize, h as usize);
    let nf = if planes == 1 { channels } else { planes };
    (0..nf)
        .map(|c| {
            let (cw, ch) = if c == 0 || planes == 1 {
                (w, h)
            } else {
                (w.div_ceil(dh), h.div_ceil(dv))
            };
            let mut v = Vec::with_capacity(cw * ch);
            for y in 0..ch {
                for x in 0..cw {
                    v.push(sample(c, x, y, cw, ch, bits));
                }
            }
            v
        })
        .collect()
}

/// A frame in layout `p` carrying [`source_planes`].
fn frame(p: PixelFormat, w: u32, h: u32) -> VideoFrame {
    let (_, bps, planes, channels, _) = facts(p);
    let src = source_planes(p, w, h);
    let put = |out: &mut Vec<u8>, s: u16| {
        if bps == 2 {
            out.extend_from_slice(&s.to_le_bytes());
        } else {
            out.push(s as u8);
        }
    };
    let vplanes = if planes == 1 {
        let n = src[0].len();
        let mut data = Vec::with_capacity(n * channels * bps);
        for i in 0..n {
            for plane in &src {
                put(&mut data, plane[i]);
            }
        }
        vec![VideoPlane {
            stride: w as usize * channels * bps,
            data,
        }]
    } else {
        let (w, h) = (w as usize, h as usize);
        let (_, _, _, _, (dh, dv)) = facts(p);
        src.iter()
            .enumerate()
            .map(|(c, plane)| {
                let cw = if c == 0 { w } else { w.div_ceil(dh) };
                let _ch = if c == 0 { h } else { h.div_ceil(dv) };
                let mut data = Vec::with_capacity(plane.len() * bps);
                for &s in plane {
                    put(&mut data, s);
                }
                VideoPlane {
                    stride: cw * bps,
                    data,
                }
            })
            .collect()
    };
    VideoFrame {
        pts: Some(7),
        planes: vplanes,
    }
}

// ---------------------------------------------------------------------------
// Registry plumbing
// ---------------------------------------------------------------------------

fn params(p: PixelFormat, w: u32, h: u32, opts: &[(&str, &str)]) -> CodecParameters {
    let mut cp = CodecParameters::video(CodecId::new("mjpeg"));
    cp.width = Some(w);
    cp.height = Some(h);
    cp.pixel_format = Some(p);
    let mut bag = CodecOptions::new();
    for (k, v) in opts {
        bag.insert(*k, *v);
    }
    cp.options = bag;
    cp
}

/// Encode one frame: `(bytes, output parameters)`.
fn encode(
    cp: &CodecParameters,
    f: &VideoFrame,
) -> oxideav_core::Result<(Vec<u8>, CodecParameters)> {
    let mut enc = make_encoder(cp)?;
    enc.send_frame(&Frame::Video(f.clone()))?;
    let pkt = enc.receive_packet()?;
    Ok((pkt.data, enc.output_params().clone()))
}

/// Decode through the registry decoder (the frame) and the `jpeg`
/// demuxer (the stream label).
fn decode(bytes: &[u8]) -> (VideoFrame, CodecParameters) {
    let mut ctx = RuntimeContext::new();
    oxideav_mjpeg::register(&mut ctx);
    let input: Box<dyn oxideav_core::ReadSeek> = Box::new(std::io::Cursor::new(bytes.to_vec()));
    let dmx = ctx
        .containers
        .open_demuxer("jpeg", input, &ctx.codecs)
        .expect("jpeg demuxer");
    let stream = dmx.streams()[0].params.clone();
    let mut dec = make_decoder(&stream).unwrap();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), bytes.to_vec()))
        .unwrap();
    match dec.receive_frame().unwrap() {
        Frame::Video(v) => (v, stream),
        _ => panic!("video frame"),
    }
}

/// Samples of plane `i` of a decoded frame in layout `p`, tightly
/// packed, `channels` interleaved (for the packed layouts the plane
/// holds every channel).
fn plane_samples(v: &VideoFrame, p: PixelFormat, i: usize, w: u32, h: u32) -> Vec<u16> {
    let (_, bps, planes, channels, (dh, dv)) = facts(p);
    let (w, h) = (w as usize, h as usize);
    let (cw, ch) = if i == 0 || planes == 1 {
        (w, h)
    } else {
        (w.div_ceil(dh), h.div_ceil(dv))
    };
    let pl = &v.planes[i];
    let row = cw * channels;
    let mut out = Vec::with_capacity(row * ch);
    for y in 0..ch {
        for x in 0..row {
            let o = y * pl.stride + x * bps;
            out.push(if bps == 2 {
                u16::from_le_bytes([pl.data[o], pl.data[o + 1]])
            } else {
                u16::from(pl.data[o])
            });
        }
    }
    out
}

fn psnr(a: &[u16], b: &[u16], peak: u32) -> f64 {
    assert_eq!(a.len(), b.len(), "sample count");
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        / a.len().max(1) as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * ((f64::from(peak) * f64::from(peak)) / mse).log10()
    }
}

// ---------------------------------------------------------------------------
// The black-box reference decoder (optional, local)
// ---------------------------------------------------------------------------

/// `(width, height, channels, maxval)` the black-box reference decoder
/// reports for `bytes`, `None` when the binary is not installed. A
/// stream it rejects is a failure: the registry encoder's output must
/// decode everywhere, and the test runs this gate wherever the tool
/// exists. Four-component streams go through the black-box image
/// identifier instead (the reference decoder has no CMYK output
/// format): it reports the component count as the colour space.
fn black_box(bytes: &[u8], cmyk: bool, tag: &str) -> Option<(u32, u32, u32, u32)> {
    if cmyk {
        let mut child = Command::new("magick")
            .args(["identify", "-format", "%w %h %[jpeg:colorspace] %z", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .ok()?;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(bytes)
            .expect("feed the identifier");
        let out = child.wait_with_output().expect("identifier");
        assert!(
            out.status.success(),
            "{tag}: the black-box identifier rejected the stream: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8_lossy(&out.stdout);
        let v: Vec<u32> = text
            .split_ascii_whitespace()
            .take(4)
            .map(|t| t.parse().unwrap())
            .collect();
        assert_eq!(v.len(), 4, "identifier output {text:?}");
        // `jpeg:colorspace` 4 = CMYK; `%z` is the depth in bits.
        return Some((v[0], v[1], v[2], (1 << v[3]) - 1));
    }
    let run = |grayscale: bool| {
        let mut cmd = Command::new("djpeg");
        cmd.arg("-pnm");
        if grayscale {
            cmd.arg("-grayscale");
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .ok()?;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(bytes)
            .expect("feed the reference decoder");
        Some(child.wait_with_output().expect("reference decoder"))
    };
    let mut out = run(false)?;
    let mut channels_known = true;
    if !out.status.success()
        && String::from_utf8_lossy(&out.stderr).contains("Unsupported color conversion")
    {
        // The reference decoder has no colour conversion for a
        // lossless subsampled YCbCr scan (T.81 Annex H admits the
        // sampling factors; the crate's own decoder shapes them and
        // `tests/lossless_roundtrip.rs` pins the round trip). Its
        // luma-only output, when it has one, still proves the stream
        // parses with the declared geometry and precision; otherwise
        // the cell is outside what this validator covers.
        out = run(true)?;
        channels_known = false;
        if !out.status.success()
            && String::from_utf8_lossy(&out.stderr).contains("Unsupported color conversion")
        {
            return None;
        }
    }
    assert!(
        out.status.success(),
        "{tag}: the black-box reference decoder rejected the stream: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let head = &out.stdout[..out.stdout.len().min(64)];
    let text = String::from_utf8_lossy(head);
    let mut it = text.split_ascii_whitespace();
    let magic = it.next().unwrap_or("");
    let channels = match magic {
        _ if !channels_known => 0,
        "P5" => 1,
        "P6" => 3,
        other => panic!("unexpected PNM magic {other:?}"),
    };
    let w: u32 = it.next().unwrap().parse().unwrap();
    let h: u32 = it.next().unwrap().parse().unwrap();
    let maxval: u32 = it.next().unwrap().parse().unwrap();
    Some((w, h, channels, maxval))
}

// ---------------------------------------------------------------------------
// The matrix
// ---------------------------------------------------------------------------

/// One option set of the matrix.
struct Cell {
    name: &'static str,
    opts: Vec<(&'static str, &'static str)>,
    lossless: bool,
    /// PSNR floor in dB for the DCT processes (tiny pictures at the
    /// default quality sit lowest).
    floor: f64,
    /// Compare the decoded picture with the source (`false` for the
    /// YCCK cell: the decoder converts `Y Cb Cr K` to ink amounts, a
    /// different picture by design).
    compare: bool,
}

fn cells_for(p: PixelFormat) -> Vec<Cell> {
    let mut v = vec![Cell {
        name: "default",
        opts: vec![],
        lossless: is_lossless_only(p),
        floor: 27.0,
        compare: true,
    }];
    if !is_lossless_only(p) {
        v.push(Cell {
            name: "sequential q90 optimal restart",
            opts: vec![
                ("quality", "90"),
                ("tables", "optimal"),
                ("process", "sequential"),
                ("restart", "2"),
            ],
            lossless: false,
            floor: 30.0,
            compare: true,
        });
        v.push(Cell {
            name: "progressive q90",
            opts: vec![("quality", "90"), ("process", "progressive")],
            lossless: false,
            floor: 30.0,
            compare: true,
        });
    }
    if matches!(p, P::Cmyk | P::CmykInverted) {
        v.push(Cell {
            name: "adobe cmyk q90",
            opts: vec![("quality", "90"), ("adobe_transform", "cmyk")],
            lossless: false,
            floor: 30.0,
            compare: true,
        });
        v.push(Cell {
            name: "adobe cmyk lossless",
            opts: vec![("process", "lossless"), ("adobe_transform", "cmyk")],
            lossless: true,
            floor: f64::INFINITY,
            compare: true,
        });
        v.push(Cell {
            name: "ycck q90",
            opts: vec![("quality", "90"), ("adobe_transform", "ycck")],
            lossless: false,
            floor: 30.0,
            compare: false,
        });
    }
    if lossless_labelled(p) {
        v.push(Cell {
            name: "lossless predictor 4",
            opts: vec![("process", "lossless"), ("predictor", "4")],
            lossless: true,
            floor: f64::INFINITY,
            compare: true,
        });
    }
    v
}

/// Expected decoded samples of plane `i` for source layout `p`.
fn expected_plane(p: PixelFormat, src: &[Vec<u16>], i: usize, w: u32, h: u32) -> Vec<u16> {
    let (_, _, planes, channels, (dh, dv)) = facts(p);
    if planes == 1 {
        // Packed: interleave the channels (complemented ink for
        // `CmykInverted`: the decoder hands back plain ink amounts).
        let n = src[0].len();
        let mut out = Vec::with_capacity(n * channels);
        for j in 0..n {
            for plane in src {
                out.push(if p == P::CmykInverted {
                    255 - plane[j]
                } else {
                    plane[j]
                });
            }
        }
        out
    } else if p == P::Yuv440P && i > 0 {
        // The decoder replicates 4:4:0 chroma onto the frame grid.
        let (w, h) = (w as usize, h as usize);
        let cw = w.div_ceil(dh);
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                out.push(src[i][(y / dv) * cw + x / dh]);
            }
        }
        out
    } else {
        src[i].clone()
    }
}

#[test]
fn every_declared_layout_encodes_decodes_and_labels_truthfully() {
    let mut cells = 0usize;
    let mut black_boxed = 0usize;
    for &p in ACCEPTED_PIXEL_FORMATS.iter().chain(ALIASES) {
        let coded = coded_pixel_format(p).unwrap_or_else(|| panic!("{p:?} not accepted"));
        let (bits, _, _, _, _) = facts(p);
        for &(w, h) in SIZES {
            let f = frame(p, w, h);
            let src = source_planes(p, w, h);
            for cell in cells_for(p) {
                let tag = format!("{p:?} {w}x{h} [{}]", cell.name);
                let cp = params(p, w, h, &cell.opts);
                let (bytes, out) = encode(&cp, &f).unwrap_or_else(|e| panic!("{tag}: {e}"));
                assert!(
                    bytes.starts_with(&[0xFF, 0xD8]) && bytes.ends_with(&[0xFF, 0xD9]),
                    "{tag}: interchange stream"
                );
                // The default cell is what `coded_pixel_format` promises.
                let label = out
                    .pixel_format
                    .unwrap_or_else(|| panic!("{tag}: unlabelled"));
                if cell.name == "default" {
                    assert_eq!(label, coded, "{tag}: coded_pixel_format");
                }
                // Decoder and demuxer agree with the encoder's label.
                let (v, stream) = decode(&bytes);
                assert_eq!(stream.pixel_format, Some(label), "{tag}: demuxer label");
                assert_eq!(
                    stream.color_signal, out.color_signal,
                    "{tag}: colour signal"
                );
                assert_eq!((stream.width, stream.height), (Some(w), Some(h)), "{tag}");
                let (_, _, dplanes, _, _) = facts(label);
                assert_eq!(v.planes.len(), dplanes, "{tag}: decoded plane count");
                // Picture: exact or above the floor, every plane.
                let peak = (1u32 << bits) - 1;
                for i in 0..dplanes {
                    if !cell.compare {
                        break;
                    }
                    let got = plane_samples(&v, label, i, w, h);
                    let want = expected_plane(p, &src, i, w, h);
                    if cell.lossless {
                        assert_eq!(got, want, "{tag}: plane {i} bit-exact");
                    } else {
                        let db = psnr(&got, &want, peak);
                        assert!(db >= cell.floor, "{tag}: plane {i} PSNR {db:.1} dB");
                    }
                }
                // The black-box reference decoder sees the declared
                // geometry and precision.
                if let Some((bw, bh, channels, maxval)) = black_box(&bytes, label == P::Cmyk, &tag)
                {
                    assert_eq!((bw, bh), (w, h), "{tag}: black-box geometry");
                    let want_channels = match label {
                        P::Cmyk => 4,
                        l if facts(l).3 == 1 && facts(l).2 == 1 => 1,
                        _ => 3,
                    };
                    if channels != 0 {
                        assert_eq!(channels, want_channels, "{tag}: black-box channels");
                    }
                    let precision = if cell.lossless || bits == 12 { bits } else { 8 };
                    assert_eq!(maxval, (1 << precision) - 1, "{tag}: black-box maxval");
                    black_boxed += 1;
                }
                cells += 1;
            }
        }
    }
    eprintln!("registry encoder matrix: {cells} cells proved, {black_boxed} through the black-box decoder");
    assert!(cells >= 19 * SIZES.len());
}

#[test]
fn declaration_matches_the_registry_and_names_every_layout_once() {
    let mut ctx = RuntimeContext::new();
    oxideav_mjpeg::register(&mut ctx);
    let impls = ctx.codecs.implementations(&CodecId::new("mjpeg"));
    let enc = impls
        .iter()
        .find(|i| i.make_encoder.is_some())
        .expect("mjpeg encoder registered");
    assert_eq!(enc.caps.accepted_pixel_formats, ACCEPTED_PIXEL_FORMATS);
    assert_eq!(accepted_pixel_formats(), ACCEPTED_PIXEL_FORMATS);
    assert!(enc.caps.lossy && enc.caps.lossless);
    let mut seen = std::collections::HashSet::new();
    for &p in ACCEPTED_PIXEL_FORMATS {
        assert!(seen.insert(p), "{p:?} declared twice");
        assert!(coded_pixel_format(p).is_some(), "{p:?}");
        assert!(!p.has_alpha(), "{p:?}: JPEG has no alpha");
    }
    for &p in ALIASES {
        assert!(
            !ACCEPTED_PIXEL_FORMATS.contains(&p),
            "{p:?} is an alias, not declared"
        );
        assert!(coded_pixel_format(p).is_some(), "{p:?} is still accepted");
    }
    assert_eq!(coded_pixel_format(P::Yuv420P), Some(P::YuvJ420P));
    assert_eq!(coded_pixel_format(P::Yuv440P), Some(P::YuvJ444P));
    assert_eq!(coded_pixel_format(P::CmykInverted), Some(P::Cmyk));
    assert_eq!(coded_pixel_format(P::Rgb48Le), Some(P::Rgb48Le));
    assert_eq!(coded_pixel_format(P::Rgba), None);
}

#[test]
fn undeclared_layouts_are_refused_by_type_naming_the_layout() {
    for p in [
        P::Rgba,
        P::Bgra,
        P::Argb,
        P::Rgba64Le,
        P::Ya8,
        P::Ya16Le,
        P::Yuva420P,
        P::Yuva444P12Le,
        P::Gbrap8,
    ] {
        let err = make_encoder(&params(p, 8, 8, &[]))
            .err()
            .unwrap_or_else(|| panic!("{p:?}"));
        match err {
            oxideav_core::Error::Unsupported(m) => {
                assert!(m.contains(&format!("{p:?}")), "{p:?}: {m}");
                assert!(m.contains("alpha"), "{p:?}: {m}");
            }
            other => panic!("{p:?}: expected Unsupported, got {other:?}"),
        }
    }
    for p in [
        P::Bgr24,
        P::Pal8,
        P::Nv12,
        P::Yuyv422,
        P::MonoBlack,
        P::Yuv420P10Le,
        P::Yuv444P16Le,
        P::Gbrp8,
        P::Gbrp16Le,
        P::GrayF32Le,
    ] {
        let err = make_encoder(&params(p, 8, 8, &[]))
            .err()
            .unwrap_or_else(|| panic!("{p:?}"));
        match err {
            oxideav_core::Error::Unsupported(m) => {
                assert!(m.contains(&format!("{p:?}")), "{p:?}: {m}");
                assert!(m.contains("accepted"), "{p:?}: {m}");
            }
            other => panic!("{p:?}: expected Unsupported, got {other:?}"),
        }
    }
    // A frame with too few planes is a caller error, not a panic.
    let mut enc = make_encoder(&params(P::YuvJ420P, 8, 8, &[])).unwrap();
    let err = enc
        .send_frame(&Frame::Video(VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![0; 64],
            }],
        }))
        .unwrap_err();
    assert!(matches!(err, oxideav_core::Error::InvalidData(ref m) if m.contains("YuvJ420P")));
    // So is a plane shorter than its rows, on every path (the
    // historical 8-bit writers indexed past a short plane before).
    for (p, planes, short) in [
        (P::YuvJ420P, 3, 2),
        (P::Gray8, 1, 0),
        (P::Rgb24, 1, 0),
        (P::Cmyk, 1, 0),
        (P::Gray16Le, 1, 0),
        (P::Yuv444P12Le, 3, 1),
        (P::Rgb48Le, 1, 0),
    ] {
        let mut f = frame(p, 9, 7);
        assert_eq!(f.planes.len(), planes);
        f.planes[short].data.pop();
        let mut enc = make_encoder(&params(p, 9, 7, &[])).unwrap();
        let err = enc.send_frame(&Frame::Video(f)).unwrap_err();
        assert!(
            matches!(err, oxideav_core::Error::InvalidData(ref m) if m.contains(&format!("{p:?}")) && m.contains("plane")),
            "{p:?}: {err:?}"
        );
    }
}

#[test]
fn combinations_the_decoder_cannot_label_are_refused_by_type() {
    let unsupported = |p: PixelFormat, opts: &[(&str, &str)], needle: &str| {
        let err = make_encoder(&params(p, 16, 16, opts))
            .err()
            .unwrap_or_else(|| panic!("{p:?} {opts:?} should be refused"));
        match err {
            oxideav_core::Error::Unsupported(m) => {
                assert!(m.contains(&format!("{p:?}")), "{p:?} {opts:?}: {m}");
                assert!(m.contains(needle), "{p:?} {opts:?}: {m}");
            }
            other => panic!("{p:?} {opts:?}: expected Unsupported, got {other:?}"),
        }
    };
    // Lossless 4:4:4 YCbCr decodes as the RGB-class carrier.
    unsupported(P::YuvJ444P, &[("process", "lossless")], "lossless 4:4:4");
    unsupported(P::Yuv444P12Le, &[("process", "lossless")], "lossless 4:4:4");
    // Subsampled lossless YCbCr is shaped at 8 bits only.
    unsupported(P::Yuv420P12Le, &[("process", "lossless")], "P = 8");
    // The DCT processes carry 8 or 12 bits.
    for p in [
        P::Gray10Le,
        P::Gray16Le,
        P::Rgb48Le,
        P::Gbrp10Le,
        P::Gbrp12Le,
        P::Gbrp14Le,
    ] {
        unsupported(p, &[("process", "sequential")], "lossless-only");
        unsupported(p, &[("process", "progressive")], "lossless-only");
    }
    // `chroma` applies to Rgb24 (any) and to YCbCr / grey (`gray` only).
    unsupported(P::Gray8, &[("chroma", "yuv420")], "chroma");
    unsupported(P::YuvJ420P, &[("chroma", "yuv444")], "chroma");
    unsupported(P::Cmyk, &[("chroma", "gray")], "chroma");
    unsupported(P::Gbrp12Le, &[("chroma", "yuv444")], "chroma");
    // Precision must match the layout except on the 16-bit carriers.
    assert!(make_encoder(&params(P::Gray12Le, 16, 16, &[("precision", "8")])).is_err());
    assert!(make_encoder(&params(P::Gray16Le, 16, 16, &[("precision", "14")])).is_ok());
    assert!(make_encoder(&params(P::Rgb48Le, 16, 16, &[("precision", "12")])).is_err());
    assert!(make_encoder(&params(P::Rgb48Le, 16, 16, &[("precision", "15")])).is_ok());
}

#[test]
fn chroma_option_converts_rgb_and_keeps_luma_of_ycbcr() {
    let (w, h) = (33u32, 18u32);
    let rgb = frame(P::Rgb24, w, h);
    let src = source_planes(P::Rgb24, w, h);
    for (chroma, label) in [
        ("yuv444", P::YuvJ444P),
        ("yuv422", P::YuvJ422P),
        ("yuv420", P::YuvJ420P),
        ("yuv411", P::Yuv411P),
        ("gray", P::Gray8),
    ] {
        let cp = params(P::Rgb24, w, h, &[("chroma", chroma), ("quality", "92")]);
        let (bytes, out) = encode(&cp, &rgb).unwrap();
        assert_eq!(out.pixel_format, Some(label), "{chroma}");
        let (v, stream) = decode(&bytes);
        assert_eq!(stream.pixel_format, Some(label), "{chroma}");
        assert_eq!(v.planes.len(), facts(label).2, "{chroma}");
        // Back to RGB through the standalone decoder: the picture
        // survives the T.871 round trip above the floor.
        let img = oxideav_mjpeg::decode(&bytes).unwrap();
        let back = img.to_rgb8();
        let got: Vec<u16> = back.iter().map(|&b| u16::from(b)).collect();
        let want = expected_plane(P::Rgb24, &src, 0, w, h);
        if label == P::Gray8 {
            // Grey: every channel is the luma; compare against Y.
            let luma: Vec<u16> = want
                .chunks(3)
                .map(|px| {
                    let (r, g, b) = (f64::from(px[0]), f64::from(px[1]), f64::from(px[2]));
                    (0.299 * r + 0.587 * g + 0.114 * b).round() as u16
                })
                .collect();
            let y = plane_samples(&v, P::Gray8, 0, w, h);
            let db = psnr(&y, &luma, 255);
            assert!(db >= 35.0, "{chroma}: luma PSNR {db:.1} dB");
        } else {
            let db = psnr(&got, &want, 255);
            assert!(db >= 30.0, "{chroma}: RGB PSNR {db:.1} dB");
        }
        // Byte identity with the standalone RGB entry point.
        let c = match chroma {
            "yuv444" => ChromaSubsampling::Yuv444,
            "yuv422" => ChromaSubsampling::Yuv422,
            "yuv420" => ChromaSubsampling::Yuv420,
            "yuv411" => ChromaSubsampling::Yuv411,
            _ => ChromaSubsampling::GrayscaleOnly,
        };
        let o = EncodeOptions::new().with_quality(92).with_chroma(c);
        let standalone = oxideav_mjpeg::encode_rgb8(w, h, &rgb.planes[0].data, &o).unwrap();
        assert_eq!(bytes, standalone, "{chroma}: registry == encode_rgb8");
    }
    // A YCbCr input keeps its luma only.
    let ycc = frame(P::YuvJ420P, w, h);
    let cp = params(P::YuvJ420P, w, h, &[("chroma", "gray")]);
    let (bytes, out) = encode(&cp, &ycc).unwrap();
    assert_eq!(out.pixel_format, Some(P::Gray8));
    let (v, _) = decode(&bytes);
    let y = plane_samples(&v, P::Gray8, 0, w, h);
    let want = source_planes(P::YuvJ420P, w, h)[0].clone();
    assert!(psnr(&y, &want, 255) >= 30.0);
}

#[test]
fn registry_options_and_the_standalone_contract_write_the_same_bytes() {
    let (w, h) = (17u32, 11u32);
    let mut pinned = 0usize;
    for &p in ACCEPTED_PIXEL_FORMATS {
        // `Yuv440P` / `CmykInverted` have no `JpegImage` layout: they
        // reach the writer through the registry only.
        if matches!(p, P::Yuv440P | P::CmykInverted) {
            continue;
        }
        let f = frame(p, w, h);
        for cell in cells_for(p) {
            let cp = params(p, w, h, &cell.opts);
            let (via_registry, _) = encode(&cp, &f).unwrap();
            // The same request as typed options on the same image.
            let img = JpegImage::from_video_frame(&f, &cp).unwrap();
            let mut o = EncodeOptions::new();
            for (k, v) in &cell.opts {
                match *k {
                    "quality" => o = o.with_quality(v.parse().unwrap()),
                    "tables" => o = o.with_optimal_tables(*v == "optimal"),
                    "process" => {
                        o = match *v {
                            "progressive" => o.with_process(JpegProcess::Progressive),
                            "lossless" => o.with_process(JpegProcess::Lossless {
                                predictor: 1,
                                point_transform: 0,
                            }),
                            _ => o.with_process(JpegProcess::Sequential),
                        }
                    }
                    "predictor" => {
                        o = o.with_process(JpegProcess::Lossless {
                            predictor: v.parse().unwrap(),
                            point_transform: 0,
                        })
                    }
                    "restart" => o = o.with_restart_interval(v.parse().unwrap()),
                    "adobe_transform" => {
                        o = o.with_signalling(ColorSignalling::Cmyk {
                            adobe_transform: match *v {
                                "cmyk" => Some(0),
                                "ycck" => Some(2),
                                _ => None,
                            },
                        })
                    }
                    other => panic!("{other}"),
                }
            }
            if cell.opts.is_empty() && is_lossless_only(p) {
                o = o.with_lossless(1);
            }
            if cell.opts.is_empty()
                && matches!(
                    p,
                    P::Gray8 | P::YuvJ444P | P::YuvJ422P | P::YuvJ420P | P::Rgb24 | P::Cmyk
                )
            {
                // The empty bag keeps the historical per-format stream
                // where one exists (segment order pinned by
                // `tests/golden.rs`); the contract comparison is for the
                // option-driven writer and the layouts it alone serves.
                continue;
            }
            let via_contract = oxideav_mjpeg::encode(&img, &o).unwrap();
            assert_eq!(
                via_registry, via_contract,
                "{p:?} [{}]: registry bytes == encode(&JpegImage, &EncodeOptions)",
                cell.name
            );
            pinned += 1;
        }
    }
    eprintln!("registry/contract byte identity: {pinned} requests pinned");
    assert!(pinned >= 30);
}

#[test]
fn abbreviated_streams_for_every_declared_layout_decode_with_extradata() {
    let (w, h) = (17u32, 11u32);
    let mut cells = 0usize;
    for &p in ACCEPTED_PIXEL_FORMATS {
        let f = frame(p, w, h);
        let src = source_planes(p, w, h);
        let (bits, _, _, _, _) = facts(p);
        let cp = params(p, w, h, &[("abbreviated", "true")]);
        let mut enc = make_encoder(&cp).unwrap_or_else(|e| panic!("{p:?}: {e}"));
        let mut streams = Vec::new();
        for _ in 0..2 {
            enc.send_frame(&Frame::Video(f.clone()))
                .unwrap_or_else(|e| panic!("{p:?}: {e}"));
            streams.push(enc.receive_packet().unwrap().data);
        }
        let out = enc.output_params().clone();
        let tables = out.extradata.clone();
        assert!(
            tables.starts_with(&[0xFF, 0xD8]) && tables.ends_with(&[0xFF, 0xD9]),
            "{p:?}: tables stream"
        );
        assert_eq!(
            streams[0], streams[1],
            "{p:?}: shared tables → identical frames"
        );
        let label = out.pixel_format.unwrap();
        for data in &streams {
            assert!(
                !data.windows(2).any(|m| m == [0xFF, 0xC4]),
                "{p:?}: frame carries DHT"
            );
            let mut dp = CodecParameters::video(CodecId::new("mjpeg"));
            dp.extradata = tables.clone();
            let mut dec = make_decoder(&dp).unwrap();
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data.clone()))
                .unwrap();
            let Frame::Video(v) = dec.receive_frame().unwrap_or_else(|e| panic!("{p:?}: {e}"))
            else {
                panic!("video")
            };
            let (_, _, dplanes, _, _) = facts(label);
            assert_eq!(v.planes.len(), dplanes, "{p:?}");
            for i in 0..dplanes {
                let got = plane_samples(&v, label, i, w, h);
                let want = expected_plane(p, &src, i, w, h);
                if is_lossless_only(p) {
                    assert_eq!(got, want, "{p:?}: plane {i}");
                } else {
                    let db = psnr(&got, &want, (1u32 << bits) - 1);
                    assert!(db >= 27.0, "{p:?}: plane {i} PSNR {db:.1} dB");
                }
            }
            cells += 1;
        }
    }
    eprintln!("abbreviated: {cells} frames decoded through extradata tables");
    assert_eq!(cells, 2 * ACCEPTED_PIXEL_FORMATS.len());
}

#[test]
fn sixteen_bit_carriers_take_any_lossless_precision_and_label_the_result() {
    let (w, h) = (9u32, 7u32);
    // (input, precision option, label the decoder reports)
    for (p, precision, label) in [
        (P::Gray16Le, "2", P::Gray16Le),
        (P::Gray16Le, "9", P::Gray16Le),
        (P::Gray16Le, "10", P::Gray10Le),
        (P::Gray16Le, "12", P::Gray12Le),
        (P::Gray16Le, "14", P::Gray16Le),
        (P::Rgb48Le, "9", P::Rgb48Le),
        (P::Rgb48Le, "13", P::Rgb48Le),
        (P::Rgb48Le, "16", P::Rgb48Le),
    ] {
        let bits: u32 = precision.parse().unwrap();
        // Samples in the low `precision` bits of each 16-bit word.
        let mut f = frame(p, w, h);
        let mask = ((1u32 << bits) - 1) as u16;
        for pl in &mut f.planes {
            for px in pl.data.chunks_mut(2) {
                let v = u16::from_le_bytes([px[0], px[1]]) & mask;
                px.copy_from_slice(&v.to_le_bytes());
            }
        }
        let cp = params(p, w, h, &[("precision", precision)]);
        let (bytes, out) = encode(&cp, &f).unwrap_or_else(|e| panic!("{p:?} P={precision}: {e}"));
        assert_eq!(out.pixel_format, Some(label), "{p:?} P={precision}");
        let (v, stream) = decode(&bytes);
        assert_eq!(
            stream.pixel_format,
            Some(label),
            "{p:?} P={precision}: demuxer"
        );
        let img = oxideav_mjpeg::decode(&bytes).unwrap();
        assert_eq!(
            u32::from(img.precision),
            bits,
            "{p:?} P={precision}: precision"
        );
        let (_, _, dplanes, _, _) = facts(label);
        for i in 0..dplanes {
            let got = plane_samples(&v, label, i, w, h);
            let want = plane_samples(&f, p, i, w, h);
            assert_eq!(got, want, "{p:?} P={precision}: plane {i} bit-exact");
        }
    }
}
