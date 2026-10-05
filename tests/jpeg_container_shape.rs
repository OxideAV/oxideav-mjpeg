//! The `jpeg` / `mjpeg-raw` demuxers declare on their stream exactly
//! what the `mjpeg` decoder emits — `pixel_format` (the native layout
//! of the standalone `decode`) and the colour signal — for every JPEG
//! process and colour case, and the registry encoder publishes its
//! option schema so a generic caller can forward `quality`.
#![cfg(feature = "registry")]

use std::io::Cursor;
use std::path::{Path, PathBuf};

use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, ColorSignal, Frame, PixelFormat, RuntimeContext,
    VideoFrame, VideoPlane,
};
use oxideav_mjpeg::{decode, JpegImage};

fn context() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    oxideav_mjpeg::register(&mut ctx);
    ctx
}

/// Every `.jpg` under `tests/fixtures`, sorted.
fn fixtures() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().and_then(|e| e.to_str()) == Some("jpg")
                && p.metadata().map(|m| m.len() > 0).unwrap_or(false)
            {
                out.push(p);
            }
        }
    }
    let mut v = Vec::new();
    walk(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        &mut v,
    );
    v.sort();
    assert!(v.len() >= 12, "fixture count: {}", v.len());
    v
}

/// Open `bytes` as `container`, read every packet of stream 0 through
/// the registry decoder: the stream parameters and the frames.
fn demux_decode(
    ctx: &RuntimeContext,
    container: &str,
    bytes: Vec<u8>,
) -> (CodecParameters, Vec<VideoFrame>) {
    let mut d = ctx
        .containers
        .open_demuxer(container, Box::new(Cursor::new(bytes)), &ctx.codecs)
        .unwrap();
    let params = d.streams()[0].params.clone();
    let mut dec = ctx.codecs.first_decoder(&params).unwrap();
    let mut frames = Vec::new();
    while let Ok(p) = d.next_packet() {
        dec.send_packet(&p).unwrap();
        match dec.receive_frame().unwrap() {
            Frame::Video(v) => frames.push(v),
            other => panic!("expected a video frame, got {other:?}"),
        }
    }
    (params, frames)
}

/// The stream describes the decoder's output for `bytes`: layout,
/// geometry and colour as `decode` reports them, the frame's plane
/// count fits the layout, and rebuilding the image from the frame plus
/// the stream parameters gives the standalone decode byte for byte.
fn assert_declares_what_it_decodes(ctx: &RuntimeContext, bytes: &[u8], tag: &str) {
    let want = decode(bytes).unwrap_or_else(|e| panic!("{tag}: decode: {e}"));
    let (params, frames) = demux_decode(ctx, "jpeg", bytes.to_vec());
    assert_eq!(frames.len(), 1, "{tag}");
    assert_eq!(
        params.pixel_format,
        Some(PixelFormat::from(want.format)),
        "{tag}: declared layout"
    );
    assert_eq!(
        (params.width, params.height),
        (Some(want.width), Some(want.height)),
        "{tag}: geometry"
    );
    assert_eq!(
        params.color_signal,
        ColorSignal::from(want.color),
        "{tag}: colour signal"
    );
    let pf = params.pixel_format.unwrap();
    assert_eq!(
        frames[0].image_planes().len(),
        pf.plane_count(),
        "{tag}: plane count for {pf:?}"
    );
    let back = JpegImage::from_video_frame(&frames[0], &params)
        .unwrap_or_else(|e| panic!("{tag}: from_video_frame: {e}"));
    assert_eq!(back.format, want.format, "{tag}");
    assert_eq!(back.planes, want.planes, "{tag}: planes byte for byte");
    assert_eq!(back.color, want.color, "{tag}");
}

#[test]
fn every_fixture_is_declared_as_the_decoder_emits_it() {
    let ctx = context();
    let mut layouts = std::collections::BTreeSet::new();
    for path in fixtures() {
        let bytes = std::fs::read(&path).unwrap();
        let tag = path.file_name().unwrap().to_string_lossy().into_owned();
        assert_declares_what_it_decodes(&ctx, &bytes, &tag);
        layouts.insert(format!("{:?}", decode(&bytes).unwrap().format));
    }
    // The corpus spans subsampled YCbCr (native and promoted), gray and
    // lossless shapes.
    assert!(layouts.len() >= 3, "fixture layouts: {layouts:?}");
}

/// A `width × height` frame of `pf` with every sample varying: packed
/// layouts in one plane, planar ones with the chroma geometry of the
/// layout, 16-bit containers little-endian under `max`.
fn frame(pf: PixelFormat, w: u32, h: u32) -> VideoFrame {
    let (planes, bytes_per_sample, max): (Vec<(u32, u32, usize)>, usize, u32) = match pf {
        PixelFormat::Gray8 => (vec![(w, h, 1)], 1, 255),
        PixelFormat::Gray12Le => (vec![(w, h, 1)], 2, 4095),
        PixelFormat::Gray16Le => (vec![(w, h, 1)], 2, 65535),
        PixelFormat::Rgb24 => (vec![(w, h, 3)], 1, 255),
        PixelFormat::Cmyk => (vec![(w, h, 4)], 1, 255),
        PixelFormat::Yuv444P => (vec![(w, h, 1), (w, h, 1), (w, h, 1)], 1, 255),
        PixelFormat::Yuv422P => (vec![(w, h, 1), (w / 2, h, 1), (w / 2, h, 1)], 1, 255),
        PixelFormat::Yuv420P => (
            vec![(w, h, 1), (w / 2, h / 2, 1), (w / 2, h / 2, 1)],
            1,
            255,
        ),
        other => panic!("no builder for {other:?}"),
    };
    let planes = planes
        .into_iter()
        .enumerate()
        .map(|(p, (pw, ph, ch))| {
            let stride = pw as usize * ch * bytes_per_sample;
            let mut data = vec![0u8; stride * ph as usize];
            for y in 0..ph {
                for x in 0..pw {
                    for c in 0..ch {
                        let v = (x * 37 + y * 11 + (p + c) as u32 * 61 + 7) % (max + 1);
                        let at = y as usize * stride + (x as usize * ch + c) * bytes_per_sample;
                        if bytes_per_sample == 1 {
                            data[at] = v as u8;
                        } else {
                            data[at..at + 2].copy_from_slice(&(v as u16).to_le_bytes());
                        }
                    }
                }
            }
            VideoPlane { stride, data }
        })
        .collect();
    VideoFrame {
        pts: Some(0),
        planes,
    }
}

fn encode_one(ctx: &RuntimeContext, pf: PixelFormat, opts: &[(&str, &str)]) -> Vec<u8> {
    let (w, h) = (16u32, 8u32);
    let mut p = CodecParameters::video(CodecId::new("mjpeg"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    let mut bag = CodecOptions::new();
    for (k, v) in opts {
        bag.insert(*k, *v);
    }
    p.options = bag;
    let mut enc = ctx
        .codecs
        .first_encoder(&p)
        .unwrap_or_else(|e| panic!("{pf:?} {opts:?}: encoder: {e}"));
    enc.send_frame(&Frame::Video(frame(pf, w, h)))
        .unwrap_or_else(|e| panic!("{pf:?} {opts:?}: send: {e}"));
    enc.flush().unwrap();
    enc.receive_packet().unwrap().data
}

/// Every process / colour case the encoder can write — sequential,
/// progressive and lossless; gray at 8 / 12 / 16 bits; YCbCr 4:4:4 /
/// 4:2:2 / 4:2:0 (12-bit YCbCr is decode-only: the `samp12_4x2.jpg`
/// fixture covers its label); RGB (Adobe `transform = 0`) packed
/// (the lossless `Gbrp*Le` / `Rgb48Le` carriers are decode-only);
/// CMYK — demuxes with the decoder's own label.
#[test]
fn every_process_and_colour_case_is_declared_as_the_decoder_emits_it() {
    let ctx = context();
    type Case = (PixelFormat, Vec<(&'static str, &'static str)>, PixelFormat);
    let cases: Vec<Case> = vec![
        (PixelFormat::Gray8, vec![], PixelFormat::Gray8),
        (
            PixelFormat::Gray8,
            vec![("process", "progressive")],
            PixelFormat::Gray8,
        ),
        (
            PixelFormat::Gray8,
            vec![("process", "lossless")],
            PixelFormat::Gray8,
        ),
        (
            PixelFormat::Gray12Le,
            vec![("precision", "12")],
            PixelFormat::Gray12Le,
        ),
        (
            PixelFormat::Gray16Le,
            vec![("process", "lossless"), ("precision", "16")],
            PixelFormat::Gray16Le,
        ),
        (PixelFormat::Yuv420P, vec![], PixelFormat::YuvJ420P),
        (PixelFormat::Yuv422P, vec![], PixelFormat::YuvJ422P),
        (PixelFormat::Yuv444P, vec![], PixelFormat::YuvJ444P),
        (
            PixelFormat::Yuv420P,
            vec![("process", "progressive")],
            PixelFormat::YuvJ420P,
        ),
        (PixelFormat::Rgb24, vec![], PixelFormat::Rgb24),
        (
            PixelFormat::Rgb24,
            vec![("process", "progressive")],
            PixelFormat::Rgb24,
        ),
        (
            PixelFormat::Rgb24,
            vec![("process", "lossless")],
            PixelFormat::Rgb24,
        ),
        (PixelFormat::Cmyk, vec![], PixelFormat::Cmyk),
    ];
    for (pf, opts, declared) in cases {
        let tag = format!("{pf:?} {opts:?}");
        let bytes = encode_one(&ctx, pf, &opts);
        assert_declares_what_it_decodes(&ctx, &bytes, &tag);
        let (params, _) = demux_decode(&ctx, "jpeg", bytes.clone());
        assert_eq!(
            params.pixel_format,
            Some(declared),
            "{tag}: the expected label"
        );
        // The raw Motion-JPEG container labels its stream the same way
        // (from the first frame).
        let mut two = bytes.clone();
        two.extend_from_slice(&bytes);
        let (params, frames) = demux_decode(&ctx, "mjpeg-raw", two);
        assert_eq!(
            params.pixel_format,
            Some(declared),
            "{tag}: mjpeg-raw label"
        );
        assert_eq!(frames.len(), 2, "{tag}: mjpeg-raw frames");
        assert_eq!(
            frames[0].image_planes().len(),
            declared.plane_count(),
            "{tag}"
        );
    }
}

/// The registry entry declares the encoder's option schema, so
/// `encoder_options_schema` lets a generic caller discover and forward
/// `quality`; two qualities give two different decodable files.
#[test]
fn encoder_schema_is_published_and_quality_reaches_the_encoder() {
    let ctx = context();
    let schema = ctx
        .codecs
        .encoder_options_schema(&CodecId::new("mjpeg"))
        .expect("mjpeg declares its encoder options");
    let names: Vec<&str> = schema.iter().map(|f| f.name).collect();
    for n in ["quality", "tables", "process", "precision", "sampling"] {
        assert!(names.contains(&n), "schema has {n}: {names:?}");
    }
    let lo = encode_one(&ctx, PixelFormat::Yuv420P, &[("quality", "10")]);
    let hi = encode_one(&ctx, PixelFormat::Yuv420P, &[("quality", "95")]);
    assert_ne!(lo, hi, "quality changes the output");
    assert!(
        lo.len() < hi.len(),
        "q10 ({}) smaller than q95 ({})",
        lo.len(),
        hi.len()
    );
    assert!(decode(&lo).is_ok() && decode(&hi).is_ok());
    // An unknown option is refused, as the schema promises.
    let mut p = CodecParameters::video(CodecId::new("mjpeg"));
    p.width = Some(16);
    p.height = Some(8);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p.options = CodecOptions::new().set("nosuchoption", "1");
    assert!(ctx.codecs.first_encoder(&p).is_err());
}
