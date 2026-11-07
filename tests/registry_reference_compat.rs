//! Registry-side surface: needs the default `registry` feature.
#![cfg(feature = "registry")]

//! The registry decoder's option bag (`MjpegDecoderOptions`, #19): the
//! opt-in reference-compatible decode and its two plane-level knobs
//! reach the decoder through `CodecParameters::options`, an empty bag is
//! the default decode, bad values are refused and other schemas' keys
//! are ignored.

use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, CodecRegistry, Frame, Packet, TimeBase,
};
use oxideav_mjpeg::registry::{make_decoder, register_codecs};
use oxideav_mjpeg::{
    ChromaUpsampling, DecodeOptions, IdctMethod, MjpegDecoderOptions, decode_with,
};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/libjpeg");

fn read(name: &str) -> Vec<u8> {
    std::fs::read(format!("{DIR}/{name}")).unwrap()
}

fn params(opts: &[(&str, &str)]) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("mjpeg"));
    let mut bag = CodecOptions::new();
    for (k, v) in opts {
        bag.insert(*k, *v);
    }
    p.options = bag;
    p
}

/// The registry frame's planes, tightly packed.
fn registry_planes(opts: &[(&str, &str)], jpeg: &[u8]) -> Vec<Vec<u8>> {
    let mut dec = make_decoder(&params(opts)).unwrap();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), jpeg.to_vec()))
        .unwrap();
    match dec.receive_frame().unwrap() {
        Frame::Video(v) => v.planes.into_iter().map(|p| p.data).collect(),
        _ => panic!("video frame"),
    }
}

fn standalone_planes(opts: &DecodeOptions, jpeg: &[u8]) -> Vec<Vec<u8>> {
    decode_with(jpeg, opts)
        .unwrap()
        .planes
        .into_iter()
        .map(|p| p.data)
        .collect()
}

#[test]
fn option_bag_reaches_the_decoder() {
    // 4:4:0 is resampled to 4:4:4 by the decoder, so both the IDCT and
    // the upsampling knob show in the planes; grey shows the IDCT alone.
    let c440 = read("c440.jpg");
    let gray = read("gray.jpg");
    let compat = DecodeOptions::new().with_reference_compat(true);
    assert_eq!(
        registry_planes(&[("reference_compat", "true")], &c440),
        standalone_planes(&compat, &c440)
    );
    assert_ne!(
        registry_planes(&[("reference_compat", "true")], &c440),
        registry_planes(&[], &c440)
    );
    assert_eq!(
        registry_planes(&[], &c440),
        standalone_planes(&DecodeOptions::new(), &c440)
    );
    assert_eq!(
        registry_planes(&[("idct", "integer")], &gray),
        standalone_planes(
            &DecodeOptions::new().with_idct(IdctMethod::AccurateInteger),
            &gray
        )
    );
    assert_eq!(
        registry_planes(&[("upsampling", "triangular")], &c440),
        standalone_planes(
            &DecodeOptions::new().with_upsampling(ChromaUpsampling::Triangular),
            &c440
        )
    );
    // A per-knob key overrides the switch.
    assert_eq!(
        registry_planes(&[("reference_compat", "true"), ("idct", "float")], &gray),
        registry_planes(&[], &gray)
    );
}

#[test]
fn bag_values_are_checked() {
    assert!(make_decoder(&params(&[("idct", "fast")])).is_err());
    assert!(make_decoder(&params(&[("upsampling", "bicubic")])).is_err());
    assert!(make_decoder(&params(&[("reference_compat", "maybe")])).is_err());
    // Keys of other schemas (an encoder's output parameters carry its
    // bag) are not the decoder's business.
    assert!(make_decoder(&params(&[("quality", "90"), ("idct", "integer")])).is_ok());
    assert!(make_decoder(&params(&[])).is_ok());
}

#[test]
fn typed_options_and_schema() {
    let o = MjpegDecoderOptions::new().with_reference_compat(true);
    assert_eq!(
        o.to_decode_options().unwrap(),
        DecodeOptions::new().with_reference_compat(true)
    );
    let o = MjpegDecoderOptions::new()
        .with_idct("integer")
        .with_upsampling("replicate");
    assert_eq!(
        o.to_decode_options().unwrap(),
        DecodeOptions::new().with_idct(IdctMethod::AccurateInteger)
    );
    assert_eq!(
        MjpegDecoderOptions::default().to_decode_options().unwrap(),
        DecodeOptions::new()
    );
    let mut reg = CodecRegistry::new();
    register_codecs(&mut reg);
    let schema = reg
        .decoder_options_schema(&CodecId::new("mjpeg"))
        .expect("decoder schema");
    let names: Vec<&str> = schema.iter().map(|f| f.name).collect();
    assert_eq!(names, ["reference_compat", "idct", "upsampling"]);
}
