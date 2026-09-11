//! Registry-side surface: needs the default `registry` feature.
#![cfg(feature = "registry")]

//! Registry encoder options ↔ `JpegEncodeOptions` parity, and the
//! §B.5 abbreviated pair through the registry (`extradata` tables).

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, Encoder, Frame, Packet, PixelFormat, TimeBase,
    VideoFrame,
};
use oxideav_mjpeg::decoder::{decode_jpeg, decode_jpeg_with_tables};
use oxideav_mjpeg::encoder::MjpegEncoder;
use oxideav_mjpeg::registry::{make_decoder, make_encoder};
use oxideav_mjpeg::t81::{HuffmanTables, JpegEncodeOptions, JpegProcess};

const W: u32 = 41;
const H: u32 = 23;

fn gray8() -> VideoFrame {
    let data: Vec<u8> = (0..(W * H) as usize)
        .map(|i| ((i % W as usize) * 5 + (i / W as usize) * 9 + (i * 7 % 13)) as u8)
        .collect();
    VideoFrame {
        pts: Some(3),
        planes: vec![VideoPlane {
            stride: W as usize,
            data,
        }],
    }
}

fn gray12() -> VideoFrame {
    let mut data = Vec::with_capacity((W * H * 2) as usize);
    for i in 0..(W * H) as usize {
        let v = ((i * 37) % 4096) as u16;
        data.extend_from_slice(&v.to_le_bytes());
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: W as usize * 2,
            data,
        }],
    }
}

fn yuv420() -> VideoFrame {
    let (cw, ch) = ((W as usize).div_ceil(2), (H as usize).div_ceil(2));
    let y: Vec<u8> = (0..(W * H) as usize).map(|i| (i * 3 % 251) as u8).collect();
    let cb: Vec<u8> = (0..cw * ch).map(|i| (i * 5 % 241) as u8).collect();
    let cr: Vec<u8> = (0..cw * ch).map(|i| (i * 11 % 239) as u8).collect();
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane {
                stride: W as usize,
                data: y,
            },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

fn params(pix: PixelFormat, opts: &[(&str, &str)]) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("mjpeg"));
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(pix);
    let mut bag = CodecOptions::new();
    for (k, v) in opts {
        bag.insert(*k, *v);
    }
    p.options = bag;
    p
}

fn encode_one(p: &CodecParameters, f: &VideoFrame) -> (Vec<u8>, Vec<u8>) {
    let mut enc = make_encoder(p).expect("make_encoder");
    enc.send_frame(&Frame::Video(f.clone()))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("packet");
    (pkt.data, enc.output_params().extradata.clone())
}

fn decode_with(extradata: &[u8], data: &[u8]) -> VideoFrame {
    let mut p = CodecParameters::video(CodecId::new("mjpeg"));
    p.extradata = extradata.to_vec();
    let mut dec = make_decoder(&p).unwrap();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), data.to_vec()))
        .unwrap();
    match dec.receive_frame().unwrap() {
        Frame::Video(v) => v,
        _ => panic!("video frame"),
    }
}

#[test]
fn lossless_options_round_trip_bit_exact_through_the_registry() {
    let src = gray8();
    let p = params(
        PixelFormat::Gray8,
        &[
            ("process", "lossless"),
            ("predictor", "6"),
            ("restart", "82"),
        ],
    );
    let (data, extra) = encode_one(&p, &src);
    assert!(extra.is_empty());
    assert!(data.windows(2).any(|m| m == [0xFF, 0xC3]));
    assert!(data.windows(2).any(|m| m == [0xFF, 0xDD]));
    let got = decode_with(&[], &data);
    assert_eq!(
        got.planes[0].data[..(W * H) as usize],
        src.planes[0].data[..]
    );
}

#[test]
fn abbreviated_frames_decode_with_extradata_tables() {
    for (pix, frame, opts) in [
        (
            PixelFormat::Gray8,
            gray8(),
            vec![("process", "lossless"), ("abbreviated", "true")],
        ),
        (
            PixelFormat::Yuv420P,
            yuv420(),
            vec![
                ("tables", "optimal"),
                ("abbreviated", "true"),
                ("restart", "2"),
            ],
        ),
        (
            PixelFormat::Yuv420P,
            yuv420(),
            vec![("process", "progressive"), ("abbreviated", "true")],
        ),
        (
            PixelFormat::Gray12Le,
            gray12(),
            vec![("abbreviated", "true"), ("quality", "95")],
        ),
    ] {
        let p = params(pix, &opts);
        let mut enc = make_encoder(&p).unwrap();
        let typical_tables = enc.output_params().extradata.clone();
        enc.send_frame(&Frame::Video(frame.clone())).unwrap();
        let a = enc.receive_packet().unwrap().data;
        enc.send_frame(&Frame::Video(frame.clone())).unwrap();
        let b = enc.receive_packet().unwrap().data;
        let tables = enc.output_params().extradata.clone();
        assert!(
            tables.starts_with(&[0xFF, 0xD8]) && tables.ends_with(&[0xFF, 0xD9]),
            "{pix:?}"
        );
        if !typical_tables.is_empty() {
            // Typical tables are known before the first frame.
            assert_eq!(typical_tables, tables, "{pix:?}");
        }
        for data in [&a, &b] {
            assert!(
                !data.windows(2).any(|m| m == [0xFF, 0xC4]),
                "{pix:?}: frame carries DHT"
            );
            assert!(
                decode_jpeg(data, None).is_err(),
                "{pix:?}: table-less frame must not decode alone"
            );
            let via_fn = decode_jpeg_with_tables(&tables, data, None).unwrap();
            let via_reg = decode_with(&tables, data);
            assert_eq!(via_fn.planes.len(), via_reg.planes.len());
            assert_eq!(via_fn.planes[0].data, via_reg.planes[0].data, "{pix:?}");
        }
        // Both frames share the tables, so the streams are identical.
        assert_eq!(a, b, "{pix:?}");
        // Lossless: bit-exact.
        if pix == PixelFormat::Gray8 {
            let got = decode_with(&tables, &a);
            assert_eq!(
                got.planes[0].data[..(W * H) as usize],
                frame.planes[0].data[..]
            );
        }
    }
}

#[test]
fn twelve_bit_input_takes_sof1_and_precision_must_match() {
    let (data, _) = encode_one(
        &params(PixelFormat::Gray12Le, &[("tables", "optimal")]),
        &gray12(),
    );
    assert!(data.windows(2).any(|m| m == [0xFF, 0xC1]));
    let got = decode_with(&[], &data);
    assert_eq!(got.planes[0].stride, W as usize * 2);
    assert!(make_encoder(&params(PixelFormat::Gray12Le, &[("precision", "8")])).is_err());
    assert!(make_encoder(&params(PixelFormat::Gray8, &[("precision", "12")])).is_err());
}

#[test]
fn option_bag_is_strict_and_sampling_must_match_the_pixel_format() {
    assert!(make_encoder(&params(PixelFormat::Gray8, &[("bogus", "1")])).is_err());
    assert!(make_encoder(&params(PixelFormat::Gray8, &[("quality", "0")])).is_err());
    assert!(make_encoder(&params(
        PixelFormat::Gray8,
        &[("process", "lossless"), ("predictor", "9")]
    ))
    .is_err());
    assert!(make_encoder(&params(
        PixelFormat::Yuv420P,
        &[("sampling", "4x2,1x1,1x1")]
    ))
    .is_err());
    let (data, _) = encode_one(
        &params(PixelFormat::Yuv420P, &[("sampling", "2x2,1x1,1x1")]),
        &yuv420(),
    );
    let info = oxideav_mjpeg::inspect_jpeg(&data).unwrap();
    assert_eq!(info.components.len(), 3);
    assert_eq!(
        (info.components[0].h_sampling, info.components[0].v_sampling),
        (2, 2)
    );
    let got = decode_with(&[], &data);
    assert_eq!(got.planes.len(), 3);
    assert_eq!(got.planes[1].stride, (W as usize).div_ceil(2));
}

#[test]
fn typed_options_on_the_concrete_encoder_match_the_string_bag() {
    let src = yuv420();
    let (via_bag, _) = encode_one(
        &params(
            PixelFormat::Yuv420P,
            &[
                ("process", "progressive"),
                ("tables", "optimal"),
                ("restart", "3"),
            ],
        ),
        &src,
    );
    let mut enc = MjpegEncoder::from_params(&params(PixelFormat::Yuv420P, &[])).unwrap();
    enc.set_encode_options(JpegEncodeOptions {
        process: JpegProcess::Progressive,
        tables: HuffmanTables::Optimal,
        restart_interval: 3,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(enc.encode_options().map(|o| o.precision), Some(8));
    assert_eq!(
        enc.encode_options().map(|o| o.sampling.clone()),
        Some(vec![(2, 2), (1, 1), (1, 1)])
    );
    enc.send_frame(&Frame::Video(src)).unwrap();
    let via_typed = enc.receive_packet().unwrap().data;
    assert_eq!(via_bag, via_typed);
    assert!(via_typed.windows(2).any(|m| m == [0xFF, 0xC2]));
}
