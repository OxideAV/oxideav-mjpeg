#![no_main]

//! Robustness target for the registry encoder's contract surface
//! (`make_encoder` / `send_frame` / `receive_packet`): fuzz-chosen
//! pixel format (declared, alias or undeclared), geometry, option bag
//! and frame bytes — with planes occasionally too short or too few.
//!
//! The bar: no panic; every refusal is `Error::Unsupported` or
//! `Error::InvalidData`; and when a frame does encode, the stream
//! decodes through the registry decoder to the label the encoder
//! stamped on its output parameters, with the plane count that label
//! implies (the symmetric check of the contract).

use PixelFormat as P;
use libfuzzer_sys::fuzz_target;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, Error, Frame, Packet, PixelFormat, TimeBase, VideoFrame,
};
use oxideav_mjpeg::registry::{coded_pixel_format, make_decoder, make_encoder};

/// Declared layouts, the range-agnostic aliases, and layouts the
/// encoder must refuse by type.
const FORMATS: &[PixelFormat] = &[
    P::YuvJ444P,
    P::YuvJ422P,
    P::YuvJ420P,
    P::Yuv411P,
    P::Yuv440P,
    P::Gray8,
    P::Gray10Le,
    P::Gray12Le,
    P::Gray16Le,
    P::Yuv444P12Le,
    P::Yuv422P12Le,
    P::Yuv420P12Le,
    P::Rgb24,
    P::Rgb48Le,
    P::Gbrp10Le,
    P::Gbrp12Le,
    P::Gbrp14Le,
    P::Cmyk,
    P::CmykInverted,
    P::Yuv444P,
    P::Yuv422P,
    P::Yuv420P,
    P::Rgba,
    P::Bgr24,
    P::Ya8,
    P::Ya16Le,
    P::Yuva420P,
    P::Yuv420P10Le,
    P::Yuv444P16Le,
    P::Pal8,
    P::Nv12,
    P::Gbrp8,
    P::GrayF32Le,
    P::MonoBlack,
];

/// Bytes per sample, plane count (1 = packed), packed channels, chroma
/// divisors of an accepted layout.
fn facts(p: PixelFormat) -> Option<(usize, usize, usize, (usize, usize))> {
    Some(match p {
        P::Gray8 => (1, 1, 1, (1, 1)),
        P::Gray10Le | P::Gray12Le | P::Gray16Le => (2, 1, 1, (1, 1)),
        P::YuvJ444P | P::Yuv444P => (1, 3, 1, (1, 1)),
        P::YuvJ422P | P::Yuv422P => (1, 3, 1, (2, 1)),
        P::YuvJ420P | P::Yuv420P => (1, 3, 1, (2, 2)),
        P::Yuv411P => (1, 3, 1, (4, 1)),
        P::Yuv440P => (1, 3, 1, (1, 2)),
        P::Yuv444P12Le => (2, 3, 1, (1, 1)),
        P::Yuv422P12Le => (2, 3, 1, (2, 1)),
        P::Yuv420P12Le => (2, 3, 1, (2, 2)),
        P::Rgb24 => (1, 1, 3, (1, 1)),
        P::Rgb48Le => (2, 1, 3, (1, 1)),
        P::Gbrp10Le | P::Gbrp12Le | P::Gbrp14Le => (2, 3, 1, (1, 1)),
        P::Cmyk | P::CmykInverted => (1, 1, 4, (1, 1)),
        _ => return None,
    })
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let pix = FORMATS[data[0] as usize % FORMATS.len()];
    let w = u32::from(data[1] % 24) + 1;
    let h = u32::from(data[2] % 24) + 1;
    let (sel, sel2) = (data[3], data[4]);
    let mut p = CodecParameters::video(CodecId::new("mjpeg"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pix);
    let mut bag = CodecOptions::new();
    if sel & 1 != 0 {
        bag.insert(
            "quality",
            [1u8, 50, 75, 90, 100][sel2 as usize % 5].to_string(),
        );
    }
    if sel & 2 != 0 {
        bag.insert(
            "process",
            ["auto", "sequential", "progressive", "lossless"][(sel2 >> 1) as usize % 4],
        );
    }
    if sel & 4 != 0 {
        bag.insert(
            "chroma",
            ["yuv444", "yuv422", "yuv420", "yuv411", "gray"][(sel2 >> 3) as usize % 5],
        );
    }
    if sel & 8 != 0 {
        bag.insert("tables", "optimal");
    }
    if sel & 16 != 0 {
        bag.insert("restart", (sel2 % 7).to_string());
    }
    if sel & 32 != 0 {
        bag.insert("predictor", (sel2 % 9).to_string());
    }
    if sel & 64 != 0 {
        bag.insert(
            "precision",
            [0u8, 8, 10, 12, 14, 16][(sel2 >> 2) as usize % 6].to_string(),
        );
    }
    if sel & 128 != 0 {
        bag.insert("abbreviated", "true");
    }
    p.options = bag;

    let mut enc = match make_encoder(&p) {
        Ok(e) => e,
        Err(e @ Error::Unsupported(_)) | Err(e @ Error::InvalidData(_)) => {
            // The declaration and the layout table agree, and an
            // accepted layout never fails without options.
            assert_eq!(
                coded_pixel_format(pix).is_some(),
                facts(pix).is_some(),
                "{pix:?}"
            );
            assert!(
                !p.options.is_empty() || facts(pix).is_none(),
                "{pix:?}: accepted layout refused without options: {e}"
            );
            return;
        }
        Err(e) => panic!("{pix:?}: unexpected error kind {e:?}"),
    };
    let Some((bps, planes, channels, (dh, dv))) = facts(pix) else {
        panic!("{pix:?}: undeclared layout accepted");
    };
    let label = enc
        .output_params()
        .pixel_format
        .expect("the encoder labels its output");

    // Frame bytes from the input, cycled; the shape knob occasionally
    // shortens a plane or drops one.
    let body = &data[5..];
    let shape_knob = data[5] & 0x0F;
    let (w_, h_) = (w as usize, h as usize);
    let mut vplanes = Vec::with_capacity(planes);
    for c in 0..planes {
        let (cw, ch) = if c == 0 || planes == 1 {
            (w_, h_)
        } else {
            (w_.div_ceil(dh), h_.div_ceil(dv))
        };
        let stride = cw * channels * bps + if shape_knob == 1 { 3 } else { 0 };
        let mut len = stride * ch;
        if shape_knob == 2 {
            len = len.saturating_sub(1 + (data[6] as usize % 64));
        }
        let plane: Vec<u8> = (0..len).map(|i| body[i % body.len()]).collect();
        vplanes.push(VideoPlane {
            stride,
            data: plane,
        });
    }
    if shape_knob == 3 {
        vplanes.pop();
    }
    let frame = VideoFrame {
        pts: Some(i64::from(data[7])),
        planes: vplanes,
    };
    match enc.send_frame(&Frame::Video(frame)) {
        Ok(()) => {}
        Err(Error::Unsupported(_)) | Err(Error::InvalidData(_)) => return,
        Err(e) => panic!("{pix:?} {w}x{h}: unexpected error kind {e:?}"),
    }
    let pkt = enc.receive_packet().expect("one packet per frame");
    let bytes = pkt.data;
    assert!(bytes.starts_with(&[0xFF, 0xD8]), "{pix:?}: SOI");

    // Symmetric check: the stream decodes to the stamped label.
    let mut dp = CodecParameters::video(CodecId::new("mjpeg"));
    dp.extradata = enc.output_params().extradata.clone();
    let mut dec = make_decoder(&dp).expect("decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), bytes.clone()))
        .expect("send");
    let Frame::Video(v) = dec
        .receive_frame()
        .unwrap_or_else(|e| panic!("{pix:?} {w}x{h} {:?}: own stream refused: {e}", p.options))
    else {
        panic!("video frame");
    };
    let Some((_, dplanes, _, _)) = facts(label) else {
        panic!("{pix:?}: output labelled {label:?}, not a layout the decoder emits");
    };
    assert_eq!(v.planes.len(), dplanes, "{pix:?} → {label:?}: plane count");
    if dp.extradata.is_empty() {
        let info = oxideav_mjpeg::info(&bytes).expect("info");
        assert_eq!(
            PixelFormat::from(info.format),
            label,
            "{pix:?} {:?}: stamped label vs stream",
            p.options
        );
        assert_eq!((info.width, info.height), (w, h));
    }
});
