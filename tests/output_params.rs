//! `Decoder::output_params` / `output_params_generation` (oxideav-core
//! 0.1.42): the registry decoder states the decoded pictures' size,
//! layout and colour signal at the stream level — the place an AVI
//! `MJPG` stream (whose `BITMAPINFOHEADER` states no sampling) gets
//! its pixel format from — as soon as the first packet is sent, and
//! bumps the generation exactly at the first frame of a new shape.
//!
//! The `*.mjpeg` fixtures come from black-box ffmpeg
//! (`tests/fixtures/output_params/gen.sh`, which also prints the
//! `ffprobe -show_streams` values of the AVI they were copied from:
//! `yuvj420p` 70×46, `yuvj422p` 48×30, `yuvj444p` 33×17, all
//! `color_range=pc color_space=bt470bg`). The layouts ffmpeg's encoder
//! does not write are encoded in-process with this crate's encoder;
//! `OXIDEAV_MJPEG_OUTPUT_PARAMS_DUMP=<dir>` writes them out as `.jpg`
//! for the same ffprobe check (verified: every stream at the encoded
//! size with the same sampling — `gray` for Gray8, `yuvj411p`, the
//! 12-bit / 16-bit grey and 12-bit 4:2:0 / 4:4:4 DCT streams in
//! ffprobe's 16-bit carriers `gray16le` / `yuv420p16le` /
//! `yuv444p16le`, the RGB-coded stream as `gbrp`, the lossless 10-bit
//! RGB stream as `bgr48le`, the four-component stream as `yuva444p`).
#![cfg(feature = "registry")]

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, ColorRange, Decoder, Error, Frame, MatrixCoefficients, Packet,
    PixelFormat, TimeBase, VideoFrame,
};
use oxideav_mjpeg::registry::{make_decoder, make_encoder};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/output_params");

/// The SOI..EOI interchange streams of a raw Motion-JPEG file (EOI
/// cannot occur inside entropy-coded data: `FF` is always stuffed).
fn frames(name: &str) -> Vec<Vec<u8>> {
    let data = std::fs::read(format!("{FIX}/{name}")).expect("fixture readable");
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xD9 {
            out.push(data[start..i + 2].to_vec());
            start = i + 2;
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

/// Per plane: (horizontal divisor, vertical divisor, bytes per pixel).
fn layout(pf: PixelFormat) -> Vec<(u32, u32, usize)> {
    use PixelFormat::*;
    let yuv = |dx, dy, b| vec![(1, 1, b), (dx, dy, b), (dx, dy, b)];
    match pf {
        Gray8 => vec![(1, 1, 1)],
        Gray10Le | Gray12Le | Gray16Le => vec![(1, 1, 2)],
        Rgb24 => vec![(1, 1, 3)],
        Rgb48Le => vec![(1, 1, 6)],
        Cmyk => vec![(1, 1, 4)],
        YuvJ420P | Yuv420P => yuv(2, 2, 1),
        YuvJ422P | Yuv422P => yuv(2, 1, 1),
        YuvJ444P | Yuv444P => yuv(1, 1, 1),
        Yuv411P => yuv(4, 1, 1),
        Yuv420P12Le => yuv(2, 2, 2),
        Yuv444P12Le => yuv(1, 1, 2),
        Gbrp10Le | Gbrp12Le => yuv(1, 1, 2),
        other => panic!("unexpected layout {other:?}"),
    }
}

fn plane_dims(pf: PixelFormat, w: u32, h: u32) -> Vec<(u32, u32, usize)> {
    layout(pf)
        .into_iter()
        .map(|(dx, dy, b)| (w.div_ceil(dx), h.div_ceil(dy), b))
        .collect()
}

/// The emitted frame's planes are the truth: they must have exactly
/// the shape `params` states.
fn assert_frame_matches(vf: &VideoFrame, p: &CodecParameters) {
    let dims = plane_dims(p.pixel_format.unwrap(), p.width.unwrap(), p.height.unwrap());
    assert_eq!(vf.planes.len(), dims.len(), "plane count for {p:?}");
    for (plane, &(pw, ph, b)) in vf.planes.iter().zip(&dims) {
        assert!(plane.stride >= pw as usize * b, "stride for {p:?}");
        assert_eq!(
            plane.data.len(),
            plane.stride * ph as usize,
            "rows for {p:?}"
        );
    }
}

fn decoder() -> Box<dyn Decoder> {
    make_decoder(&CodecParameters::video(CodecId::new("mjpeg"))).expect("decoder")
}

/// Send each frame, read the description before receiving (the
/// header announcement / the previous frame), receive, read it again.
/// Returns `(generation, params)` per returned frame.
fn decode_all(dec: &mut dyn Decoder, packets: &[Vec<u8>]) -> Vec<(u64, CodecParameters)> {
    let mut out: Vec<(u64, CodecParameters)> = Vec::new();
    for (i, data) in packets.iter().enumerate() {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), data.clone()).with_pts(i as i64))
            .expect("packet");
        let before = dec.output_params().expect("known once a packet is sent");
        let g_before = dec.output_params_generation();
        if let Some((g, prev)) = out.last() {
            // A sent-but-not-returned frame does not move the answer.
            assert_eq!(g_before, *g);
            assert_eq!(format!("{before:?}"), format!("{prev:?}"));
        }
        let Frame::Video(vf) = dec.receive_frame().expect("frame") else {
            panic!("video only");
        };
        let p = dec.output_params().expect("described after a frame");
        assert_frame_matches(&vf, &p);
        assert_eq!(dec.output_pixel_format(), p.pixel_format);
        if out.is_empty() {
            // The announcement from the header is the first frame's
            // description.
            assert_eq!(format!("{before:?}"), format!("{p:?}"));
        }
        out.push((dec.output_params_generation(), p));
    }
    dec.flush().unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    out
}

#[test]
fn ffmpeg_fixtures_match_ffprobe() {
    for (name, w, h, pf) in [
        ("yuvj420p_70x46.mjpeg", 70, 46, PixelFormat::YuvJ420P),
        ("yuvj422p_48x30.mjpeg", 48, 30, PixelFormat::YuvJ422P),
        ("yuvj444p_33x17.mjpeg", 33, 17, PixelFormat::YuvJ444P),
    ] {
        let packets = frames(name);
        assert_eq!(packets.len(), 2, "{name}");
        let mut dec = decoder();
        assert!(
            dec.output_params().is_none(),
            "nothing known before a packet"
        );
        let seen = decode_all(dec.as_mut(), &packets);
        for (g, p) in &seen {
            assert_eq!(*g, 0, "{name}: fixed shape");
            assert_eq!(
                (p.width, p.height, p.pixel_format),
                (Some(w), Some(h), Some(pf))
            );
            // ffprobe: color_range=pc color_space=bt470bg (T.871 JFIF:
            // full-range BT.601 YCbCr).
            assert_eq!(p.color_signal.range, ColorRange::Full, "{name}");
            assert_eq!(
                p.color_signal.matrix,
                MatrixCoefficients::BT470_SYSTEM_BG,
                "{name}"
            );
        }
    }
}

/// One frame of `pf` from this crate's registry encoder.
fn own_encode(pf: PixelFormat, w: u32, h: u32) -> Vec<u8> {
    let mut p = CodecParameters::video(CodecId::new("mjpeg"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    let mut enc = make_encoder(&p).expect("encoder");
    let planes = plane_dims(pf, w, h)
        .into_iter()
        .enumerate()
        .map(|(pi, (pw, ph, b))| {
            let mut data = Vec::new();
            for k in 0..(pw * ph) as usize * (b / if b == 2 || b == 6 { 2 } else { 1 }) {
                let v = ((k * (3 + pi)) % 200) as u16 + 16;
                if matches!(b, 2 | 6) {
                    data.extend_from_slice(&(v << 2).to_le_bytes());
                } else {
                    data.push(v as u8);
                }
            }
            VideoPlane {
                stride: pw as usize * b,
                data,
            }
        })
        .collect();
    enc.send_frame(&Frame::Video(VideoFrame {
        pts: Some(0),
        planes,
    }))
    .expect("frame");
    enc.flush().unwrap();
    enc.receive_packet().expect("packet").data
}

#[test]
fn own_encoder_layouts_are_announced_then_described() {
    let dump = std::env::var("OXIDEAV_MJPEG_OUTPUT_PARAMS_DUMP").ok();
    for (pf, w, h) in [
        (PixelFormat::Gray8, 21, 13),
        (PixelFormat::Gray12Le, 16, 9),
        (PixelFormat::Gray16Le, 8, 8),
        (PixelFormat::Yuv420P12Le, 18, 10),
        (PixelFormat::Yuv444P12Le, 9, 7),
        (PixelFormat::Yuv411P, 36, 8),
        (PixelFormat::Rgb24, 11, 5),
        (PixelFormat::Cmyk, 10, 6),
        (PixelFormat::Gbrp10Le, 7, 7),
    ] {
        let jpg = own_encode(pf, w, h);
        if let Some(dir) = &dump {
            std::fs::write(format!("{dir}/own_{pf:?}_{w}x{h}.jpg"), &jpg).unwrap();
        }
        let mut dec = decoder();
        let seen = decode_all(dec.as_mut(), &[jpg]);
        let (g, p) = &seen[0];
        assert_eq!(*g, 0);
        assert_eq!(
            (p.width, p.height, p.pixel_format),
            (Some(w), Some(h), Some(pf))
        );
    }
}

/// A Motion-JPEG stream whose frames change size and sampling: the
/// generation bumps exactly at the first frame of each new shape (a
/// sent-but-not-received packet never moves the answer), and a return
/// to an earlier shape is a change too.
#[test]
fn mid_stream_changes_bump_the_generation_at_the_first_new_frame() {
    let a = frames("yuvj420p_70x46.mjpeg");
    let b = frames("yuvj422p_48x30.mjpeg");
    let c = frames("yuvj444p_33x17.mjpeg");
    let g = vec![own_encode(PixelFormat::Gray8, 70, 46)];
    let packets: Vec<Vec<u8>> = a
        .iter()
        .chain(&b)
        .chain(&c)
        .chain(&g)
        .chain(&a)
        .cloned()
        .collect();
    let mut dec = decoder();
    let seen = decode_all(dec.as_mut(), &packets);
    let got: Vec<_> = seen
        .iter()
        .map(|(g, p)| {
            (
                *g,
                p.width.unwrap(),
                p.height.unwrap(),
                p.pixel_format.unwrap(),
            )
        })
        .collect();
    use PixelFormat::*;
    assert_eq!(
        got,
        [
            (0, 70, 46, YuvJ420P),
            (0, 70, 46, YuvJ420P),
            (1, 48, 30, YuvJ422P),
            (1, 48, 30, YuvJ422P),
            (2, 33, 17, YuvJ444P),
            (2, 33, 17, YuvJ444P),
            (3, 70, 46, Gray8),
            (4, 70, 46, YuvJ420P),
            (4, 70, 46, YuvJ420P),
        ]
    );
}
