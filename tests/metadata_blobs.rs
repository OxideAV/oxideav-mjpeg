//! Metadata through the registry (core blobs): the `jpeg` demuxer
//! declares the picture's ICC / Exif / XMP on the stream, the decoder
//! puts them on the frame record, the encoder embeds the stream blobs
//! with a frame's own blobs overriding per kind, the marker order and
//! the JFIF / Exif slot rule hold on every path, the carriage limits
//! are typed refusals, and the standalone `Metadata` record agrees
//! with the blobs byte for byte.
#![cfg(feature = "registry")]

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    BlobKind, CodecId, CodecOptions, CodecParameters, Encoder, Frame, MetadataBlob, Packet,
    PixelFormat, RuntimeContext, StreamInfo, TimeBase, VideoFrame,
};
use oxideav_mjpeg::encoder::MjpegEncoder;
use oxideav_mjpeg::metadata::{EXIF_MAX_BYTES, ICC_CHUNK_DATA, ICC_MAX_BYTES, XMP_MAX_BYTES};
use oxideav_mjpeg::registry::{make_decoder, make_encoder};
use oxideav_mjpeg::{JfifPolicy, Metadata};

const W: u32 = 37;
const H: u32 = 21;

fn icc_profile(n: usize) -> Vec<u8> {
    // Any bytes: the carriage is opaque. The header-shaped prefix only
    // makes the blob look like a profile to a human reading a dump.
    let mut p = vec![0u8; n];
    if n >= 40 {
        p[..4].copy_from_slice(&(n as u32).to_be_bytes());
        p[36..40].copy_from_slice(b"acsp");
    }
    for (i, b) in p.iter_mut().enumerate().skip(40) {
        *b = (i * 7 % 251) as u8;
    }
    p
}

fn exif_block() -> Vec<u8> {
    // A TIFF header plus an empty 0th IFD: `II*\0`, offset 8, zero entries, no next IFD.
    b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec()
}

fn xmp_packet(tag: &str) -> Vec<u8> {
    format!("<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description rdf:about=\"\" tag=\"{tag}\"/></rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>").into_bytes()
}

fn blobs(icc: Option<&[u8]>, exif: Option<&[u8]>, xmp: Option<&[u8]>) -> Vec<MetadataBlob> {
    let mut v = Vec::new();
    if let Some(b) = icc {
        v.push(MetadataBlob::new(BlobKind::ICC, b.to_vec()));
    }
    if let Some(b) = exif {
        v.push(MetadataBlob::new(BlobKind::EXIF, b.to_vec()));
    }
    if let Some(b) = xmp {
        v.push(MetadataBlob::new(BlobKind::XMP, b.to_vec()).with_mime("application/rdf+xml"));
    }
    v
}

fn gray8(seed: u8) -> VideoFrame {
    let data: Vec<u8> = (0..(W * H) as usize)
        .map(|i| ((i % W as usize) * 5 + (i / W as usize) * 9 + usize::from(seed)) as u8)
        .collect();
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: W as usize,
            data,
        }],
    }
}

fn yuv420() -> VideoFrame {
    let (cw, ch) = ((W as usize).div_ceil(2), (H as usize).div_ceil(2));
    let plane = |n: usize, k: usize, stride: usize| VideoPlane {
        stride,
        data: (0..n).map(|i| (i * k % 251) as u8).collect(),
    };
    VideoFrame {
        pts: Some(0),
        planes: vec![
            plane((W * H) as usize, 3, W as usize),
            plane(cw * ch, 5, cw),
            plane(cw * ch, 11, cw),
        ],
    }
}

fn rgb24() -> VideoFrame {
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: W as usize * 3,
            data: (0..(W * H * 3) as usize)
                .map(|i| (i * 13 % 256) as u8)
                .collect(),
        }],
    }
}

fn cmyk() -> VideoFrame {
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: W as usize * 4,
            data: (0..(W * H * 4) as usize)
                .map(|i| (i * 17 % 256) as u8)
                .collect(),
        }],
    }
}

fn params(pix: PixelFormat, opts: &[(&str, &str)], blobs: Vec<MetadataBlob>) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("mjpeg"));
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(pix);
    let mut bag = CodecOptions::new();
    for (k, v) in opts {
        bag.insert(*k, *v);
    }
    p.options = bag;
    p.blobs = blobs;
    p
}

fn encode_one(cp: &CodecParameters, f: &VideoFrame) -> oxideav_core::Result<Vec<u8>> {
    let mut enc = make_encoder(cp)?;
    enc.send_frame(&Frame::Video(f.clone()))?;
    Ok(enc.receive_packet()?.data)
}

/// Open the bytes through the `jpeg` demuxer: the stream parameters
/// and the one packet.
fn demux(bytes: &[u8]) -> (CodecParameters, Packet) {
    let mut ctx = RuntimeContext::new();
    oxideav_mjpeg::register(&mut ctx);
    let input: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(bytes.to_vec()));
    let mut dmx = ctx
        .containers
        .open_demuxer("jpeg", input, &ctx.codecs)
        .expect("jpeg demuxer");
    let stream = dmx.streams()[0].params.clone();
    let pkt = dmx.next_packet().expect("one packet");
    (stream, pkt)
}

fn decode(stream: &CodecParameters, pkt: &Packet) -> VideoFrame {
    let mut dec = make_decoder(stream).unwrap();
    dec.send_packet(pkt).unwrap();
    match dec.receive_frame().unwrap() {
        Frame::Video(v) => v,
        _ => panic!("video frame"),
    }
}

/// The marker bytes of the leading `APPn` / `COM` segments, in order,
/// with the first bytes of each payload.
fn leading_segments(jpeg: &[u8]) -> Vec<(u8, Vec<u8>)> {
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    let mut out = Vec::new();
    let mut pos = 2;
    while pos + 4 <= jpeg.len() && jpeg[pos] == 0xFF && (0xE0..=0xEF).contains(&jpeg[pos + 1]) {
        let len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        let payload = &jpeg[pos + 4..pos + 2 + len];
        out.push((jpeg[pos + 1], payload[..payload.len().min(12)].to_vec()));
        pos += 2 + len;
    }
    out
}

fn markers(jpeg: &[u8]) -> Vec<u8> {
    leading_segments(jpeg).into_iter().map(|(m, _)| m).collect()
}

fn is_exif(seg: &(u8, Vec<u8>)) -> bool {
    seg.0 == 0xE1 && seg.1.starts_with(b"Exif\0\0")
}
fn is_jfif(seg: &(u8, Vec<u8>)) -> bool {
    seg.0 == 0xE0 && seg.1.starts_with(b"JFIF\0")
}
fn is_xmp(seg: &(u8, Vec<u8>)) -> bool {
    seg.0 == 0xE1 && seg.1.starts_with(b"http://ns.ad")
}
fn is_icc(seg: &(u8, Vec<u8>)) -> bool {
    seg.0 == 0xE2 && seg.1.starts_with(b"ICC_PROFILE\0")
}

fn corpus(name: &str) -> Option<Vec<u8>> {
    let path = format!("../../docs/image/jpeg/fixtures/{name}/input.jpg");
    match std::fs::read(&path) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skip {name}: {path} ({e})");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Read side
// ---------------------------------------------------------------------------

#[test]
fn demuxer_declares_the_profile_the_decoder_puts_on_the_frame() {
    let Some(jpg) = corpus("with-icc-profile-embedded") else {
        return;
    };
    let (stream, pkt) = demux(&jpg);
    let icc = stream.blob(&BlobKind::ICC).expect("stream ICC blob");
    assert_eq!(
        icc.data.len(),
        2576,
        "the corpus notes give one 2576-byte chunk"
    );
    assert!(stream.blob(&BlobKind::EXIF).is_none());
    assert!(stream.blob(&BlobKind::XMP).is_none());
    // One source of truth: the contract record says the same bytes.
    let img = oxideav_mjpeg::decode(&jpg).unwrap();
    assert_eq!(img.metadata.icc.as_deref(), Some(&icc.data[..]));
    assert_eq!(img.metadata.to_blobs(), stream.blobs);
    assert_eq!(oxideav_mjpeg::read_metadata(&jpg).unwrap(), img.metadata);
    // A still's frame blobs equal its stream blobs; the image planes
    // are untouched by the record.
    let vf = decode(&stream, &pkt);
    assert_eq!(vf.blobs(), stream.blobs);
    assert_eq!(vf.image_planes().len(), 3);
    assert_eq!(Metadata::from_blobs(&vf.blobs()), img.metadata);
}

#[test]
fn fixtures_without_metadata_declare_no_blobs() {
    for name in [
        "baseline-yuv420-128x128-q75",
        "without-jfif-marker",
        "baseline-rgb-32x32",
    ] {
        let Some(jpg) = corpus(name) else {
            continue;
        };
        let (stream, pkt) = demux(&jpg);
        assert!(stream.blobs.is_empty(), "{name}");
        let vf = decode(&stream, &pkt);
        assert!(vf.blobs().is_empty(), "{name}");
        assert_eq!(
            vf.planes.len(),
            vf.image_planes().len(),
            "{name}: no record attached"
        );
    }
}

#[test]
fn mjpeg_raw_demuxer_declares_the_first_frames_blobs() {
    let a = encode_one(
        &params(
            PixelFormat::Gray8,
            &[],
            blobs(None, None, Some(&xmp_packet("first"))),
        ),
        &gray8(1),
    )
    .unwrap();
    let b = encode_one(&params(PixelFormat::Gray8, &[], Vec::new()), &gray8(2)).unwrap();
    let mut raw = a.clone();
    raw.extend_from_slice(&b);
    let mut ctx = RuntimeContext::new();
    oxideav_mjpeg::register(&mut ctx);
    let input: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(raw));
    let dmx = ctx
        .containers
        .open_demuxer("mjpeg-raw", input, &ctx.codecs)
        .expect("mjpeg-raw demuxer");
    let xmp = dmx.streams()[0]
        .params
        .blob(&BlobKind::XMP)
        .expect("stream XMP");
    assert_eq!(xmp.data, xmp_packet("first"));
}

// ---------------------------------------------------------------------------
// Write side: the chain demux → decode → encode → demux
// ---------------------------------------------------------------------------

#[test]
fn stream_blobs_survive_the_registry_chain() {
    let icc = icc_profile(3000);
    let exif = exif_block();
    let xmp = xmp_packet("chain");
    let cp = params(
        PixelFormat::YuvJ420P,
        &[],
        blobs(Some(&icc), Some(&exif), Some(&xmp)),
    );
    let bytes = encode_one(&cp, &yuv420()).unwrap();
    // Order: Exif takes the slot after SOI, no JFIF, XMP, ICC.
    let segs = leading_segments(&bytes);
    assert!(is_exif(&segs[0]), "{segs:?}");
    assert!(is_xmp(&segs[1]), "{segs:?}");
    assert!(is_icc(&segs[2]), "{segs:?}");
    assert!(
        !segs.iter().any(is_jfif),
        "Exif takes the JFIF slot: {segs:?}"
    );

    let (stream, pkt) = demux(&bytes);
    assert_eq!(stream.blob(&BlobKind::ICC).unwrap().data, icc);
    assert_eq!(stream.blob(&BlobKind::EXIF).unwrap().data, exif);
    let got_xmp = stream.blob(&BlobKind::XMP).unwrap();
    assert_eq!(got_xmp.data, xmp);
    assert_eq!(got_xmp.mime.as_deref(), Some("application/rdf+xml"));
    assert_eq!(stream.pixel_format, Some(PixelFormat::YuvJ420P));
    let vf = decode(&stream, &pkt);
    assert_eq!(vf.blobs(), stream.blobs);

    // Re-encode what the chain produced: blobs out == blobs in.
    let mut cp2 = stream.clone();
    cp2.options = CodecOptions::new();
    let again = encode_one(
        &cp2,
        &VideoFrame {
            pts: Some(0),
            planes: vf.image_planes().to_vec(),
        },
    )
    .unwrap();
    let (stream2, _) = demux(&again);
    assert_eq!(stream2.blobs, stream.blobs);

    // The contract path writes the same segments for the same request.
    let img = oxideav_mjpeg::decode(&bytes).unwrap();
    assert_eq!(img.metadata, Metadata::from_blobs(&stream.blobs));
    let enc = oxideav_mjpeg::encode(&img, &oxideav_mjpeg::EncodeOptions::new()).unwrap();
    assert_eq!(oxideav_mjpeg::read_metadata(&enc).unwrap(), img.metadata);
}

#[test]
fn every_path_embeds_the_same_segments() {
    let icc = icc_profile(500);
    let exif = exif_block();
    let xmp = xmp_packet("paths");
    let all = blobs(Some(&icc), Some(&exif), Some(&xmp));
    // (layout, frame, options): the historical per-layout encoders
    // (empty bag), the general writer (any key), the arithmetic and
    // progressive knobs, abbreviated frames.
    type Cell<'a> = (PixelFormat, VideoFrame, Vec<(&'a str, &'a str)>);
    let cells: Vec<Cell<'_>> = vec![
        (PixelFormat::Gray8, gray8(0), vec![]),
        (PixelFormat::Gray8, gray8(0), vec![("quality", "80")]),
        (PixelFormat::Gray8, gray8(0), vec![("process", "lossless")]),
        (PixelFormat::Gray8, gray8(0), vec![("abbreviated", "true")]),
        (PixelFormat::YuvJ420P, yuv420(), vec![]),
        (
            PixelFormat::YuvJ420P,
            yuv420(),
            vec![("process", "progressive")],
        ),
        (PixelFormat::Rgb24, rgb24(), vec![]),
        (PixelFormat::Rgb24, rgb24(), vec![("chroma", "yuv422")]),
        (PixelFormat::Cmyk, cmyk(), vec![]),
        (PixelFormat::Cmyk, cmyk(), vec![("adobe_transform", "cmyk")]),
    ];
    for (pix, frame, opts) in &cells {
        let cp = params(*pix, opts, all.clone());
        let bytes = encode_one(&cp, frame).unwrap();
        let segs = leading_segments(&bytes);
        let kinds: Vec<&str> = segs
            .iter()
            .map(|s| {
                if is_exif(s) {
                    "exif"
                } else if is_jfif(s) {
                    "jfif"
                } else if is_xmp(s) {
                    "xmp"
                } else if is_icc(s) {
                    "icc"
                } else if s.0 == 0xEE {
                    "adobe"
                } else {
                    "other"
                }
            })
            .collect();
        assert!(
            kinds.starts_with(&["exif", "xmp", "icc"]),
            "{pix:?} {opts:?}: {kinds:?}"
        );
        assert!(!kinds.contains(&"jfif"), "{pix:?} {opts:?}: {kinds:?}");
        let m = oxideav_mjpeg::read_metadata(&bytes).unwrap();
        assert_eq!(m.icc.as_deref(), Some(&icc[..]), "{pix:?} {opts:?}");
        assert_eq!(m.exif.as_deref(), Some(&exif[..]), "{pix:?} {opts:?}");
        assert_eq!(m.xmp.as_deref(), Some(&xmp[..]), "{pix:?} {opts:?}");
        if opts.iter().any(|(k, _)| *k == "abbreviated") {
            // The table-less frame still carries the segments; the
            // decoder needs the tables from `extradata`.
            let mut enc = make_encoder(&cp).unwrap();
            enc.send_frame(&Frame::Video(frame.clone())).unwrap();
            let pkt = enc.receive_packet().unwrap();
            let vf = decode(enc.output_params(), &pkt);
            assert_eq!(Metadata::from_blobs(&vf.blobs()), m);
        } else {
            let (stream, pkt) = demux(&bytes);
            assert_eq!(Metadata::from_blobs(&stream.blobs), m, "{pix:?} {opts:?}");
            let vf = decode(&stream, &pkt);
            assert_eq!(vf.blobs(), stream.blobs, "{pix:?} {opts:?}");
        }
    }
}

#[test]
fn arithmetic_knob_path_embeds_too() {
    let cp = params(
        PixelFormat::Gray8,
        &[],
        blobs(None, None, Some(&xmp_packet("arith"))),
    );
    let mut enc = MjpegEncoder::from_params(&cp).unwrap();
    enc.set_arithmetic(true);
    enc.send_frame(&Frame::Video(gray8(0))).unwrap();
    let bytes = enc.receive_packet().unwrap().data;
    assert!(bytes.windows(2).any(|w| w == [0xFF, 0xC9]), "SOF9 frame");
    let m = oxideav_mjpeg::read_metadata(&bytes).unwrap();
    assert_eq!(m.xmp, Some(xmp_packet("arith")));
    assert_eq!(markers(&bytes)[..2], [0xE0, 0xE1]);
}

#[test]
fn frames_without_blobs_are_byte_identical_to_before() {
    // No metadata, default policy: the historical paths and the general
    // writer produce exactly what they produced without the carriage.
    for (pix, frame, opts) in [
        (PixelFormat::Gray8, gray8(0), vec![]),
        (PixelFormat::YuvJ420P, yuv420(), vec![("quality", "60")]),
        (PixelFormat::Rgb24, rgb24(), vec![]),
    ] {
        let bytes = encode_one(&params(pix, &opts, Vec::new()), &frame).unwrap();
        let cp = params(pix, &opts, Vec::new());
        let mut enc = make_encoder(&cp).unwrap();
        enc.send_frame(&Frame::Video(frame.clone())).unwrap();
        assert_eq!(enc.receive_packet().unwrap().data, bytes);
        let img = oxideav_mjpeg::decode(&bytes).unwrap();
        assert!(img.metadata.is_empty());
        let (stream, _) = demux(&bytes);
        assert!(stream.blobs.is_empty());
        let segs = leading_segments(&bytes);
        assert!(
            !segs.iter().any(|s| is_exif(s) || is_xmp(s) || is_icc(s)),
            "{pix:?}: {segs:?}"
        );
    }
}

#[test]
fn an_exif_block_holding_a_thumbnail_does_not_end_the_file_early() {
    // An Exif 1st IFD thumbnail is a complete `SOI … EOI` inside APP1;
    // the demuxer must skip the segment whole, not stop at the inner
    // EOI. The ICC chunk below also carries the two marker bytes.
    let mut exif = exif_block();
    exif.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x02, 0xFF, 0xD9]);
    let mut icc = icc_profile(100);
    icc[50..52].copy_from_slice(&[0xFF, 0xD9]);
    let cp = params(
        PixelFormat::YuvJ420P,
        &[],
        blobs(Some(&icc), Some(&exif), None),
    );
    let bytes = encode_one(&cp, &yuv420()).unwrap();
    let (stream, pkt) = demux(&bytes);
    assert_eq!(pkt.data, bytes, "the whole file is the packet");
    assert_eq!(stream.blob(&BlobKind::EXIF).unwrap().data, exif);
    assert_eq!(stream.blob(&BlobKind::ICC).unwrap().data, icc);
    let vf = decode(&stream, &pkt);
    assert_eq!(vf.image_planes().len(), 3);
    assert_eq!(vf.blobs(), stream.blobs);
}

// ---------------------------------------------------------------------------
// Precedence: frame blobs over stream blobs, per kind
// ---------------------------------------------------------------------------

#[test]
fn frame_blobs_override_stream_blobs_of_the_same_kind() {
    let stream_icc = icc_profile(300);
    let frame_icc = icc_profile(700);
    let xmp = xmp_packet("stream");
    let cp = params(
        PixelFormat::YuvJ420P,
        &[],
        blobs(Some(&stream_icc), None, Some(&xmp)),
    );
    let mut enc = make_encoder(&cp).unwrap();
    assert_eq!(
        enc.output_params().blobs,
        cp.blobs,
        "output params hand the muxer the stream blobs"
    );

    // Frame 1: its own ICC (wins) and Exif (adds); the stream XMP stays.
    let exif = exif_block();
    let f1 = yuv420().with_blobs(blobs(Some(&frame_icc), Some(&exif), None));
    enc.send_frame(&Frame::Video(f1)).unwrap();
    let m1 = oxideav_mjpeg::read_metadata(&enc.receive_packet().unwrap().data).unwrap();
    assert_eq!(m1.icc, Some(frame_icc.clone()));
    assert_eq!(m1.exif, Some(exif.clone()));
    assert_eq!(m1.xmp, Some(xmp.clone()));

    // Frame 2: no record → the stream blobs alone.
    enc.send_frame(&Frame::Video(yuv420())).unwrap();
    let m2 = oxideav_mjpeg::read_metadata(&enc.receive_packet().unwrap().data).unwrap();
    assert_eq!(m2.icc, Some(stream_icc.clone()));
    assert_eq!(m2.exif, None);
    assert_eq!(m2.xmp, Some(xmp.clone()));

    // Frame 3: a frame XMP of its own; blobs of other kinds are ignored.
    let f3 = yuv420().with_blobs(vec![
        MetadataBlob::new(BlobKind::XMP, xmp_packet("frame")),
        MetadataBlob::new(BlobKind::IPTC, vec![0x1C, 0x02, 0x00]),
        MetadataBlob::new(BlobKind::custom("png-text"), vec![1]),
    ]);
    enc.send_frame(&Frame::Video(f3)).unwrap();
    let m3 = oxideav_mjpeg::read_metadata(&enc.receive_packet().unwrap().data).unwrap();
    assert_eq!(m3.xmp, Some(xmp_packet("frame")));
    assert_eq!(m3.icc, Some(stream_icc));
}

#[test]
fn options_metadata_sits_under_the_stream_record() {
    // `set_encode_options` metadata is the lowest layer; the stream
    // record from `params.blobs` sits over it; `set_stream_metadata`
    // replaces the latter.
    let cp = params(
        PixelFormat::Gray8,
        &[],
        blobs(None, None, Some(&xmp_packet("stream"))),
    );
    let mut enc = MjpegEncoder::from_params(&cp).unwrap();
    assert_eq!(enc.stream_metadata().xmp, Some(xmp_packet("stream")));
    enc.set_encode_options(
        oxideav_mjpeg::EncodeOptions::new().with_metadata(
            Metadata::new()
                .with_xmp(xmp_packet("options"))
                .with_icc(icc_profile(64)),
        ),
    )
    .unwrap();
    enc.send_frame(&Frame::Video(gray8(0))).unwrap();
    let m = oxideav_mjpeg::read_metadata(&enc.receive_packet().unwrap().data).unwrap();
    assert_eq!(m.xmp, Some(xmp_packet("stream")));
    assert_eq!(m.icc, Some(icc_profile(64)));

    enc.set_stream_metadata(Metadata::new().with_exif(exif_block()));
    enc.send_frame(&Frame::Video(gray8(0))).unwrap();
    let m = oxideav_mjpeg::read_metadata(&enc.receive_packet().unwrap().data).unwrap();
    assert_eq!(m.xmp, Some(xmp_packet("options")));
    assert_eq!(m.exif, Some(exif_block()));
    assert_eq!(m.icc, Some(icc_profile(64)));
}

#[test]
fn motion_jpeg_frames_carry_their_own_blobs() {
    let cp = params(PixelFormat::Gray8, &[], Vec::new());
    let mut enc = make_encoder(&cp).unwrap();
    let mut packets = Vec::new();
    for i in 0..3u8 {
        let f = gray8(i).with_blobs(blobs(None, None, Some(&xmp_packet(&format!("frame-{i}")))));
        enc.send_frame(&Frame::Video(f)).unwrap();
        packets.push(enc.receive_packet().unwrap());
    }
    let mut dec = make_decoder(enc.output_params()).unwrap();
    for (i, pkt) in packets.iter().enumerate() {
        dec.send_packet(pkt).unwrap();
        let Frame::Video(vf) = dec.receive_frame().unwrap() else {
            panic!("video");
        };
        let xmp = vf.blob(&BlobKind::XMP).expect("frame XMP");
        assert_eq!(xmp.data, xmp_packet(&format!("frame-{i}")));
        assert_eq!(vf.image_planes().len(), 1);
    }
}

// ---------------------------------------------------------------------------
// The JFIF / Exif slot rule
// ---------------------------------------------------------------------------

#[test]
fn jfif_option_places_the_app0_segment() {
    let exif = exif_block();
    let with_exif = blobs(None, Some(&exif), None);
    let f = yuv420();
    // auto, Exif: Exif first, no JFIF.
    let bytes = encode_one(
        &params(
            PixelFormat::YuvJ420P,
            &[("jfif", "auto")],
            with_exif.clone(),
        ),
        &f,
    )
    .unwrap();
    let segs = leading_segments(&bytes);
    assert!(is_exif(&segs[0]) && !segs.iter().any(is_jfif), "{segs:?}");
    // auto, no Exif: JFIF first.
    let bytes = encode_one(
        &params(PixelFormat::YuvJ420P, &[("jfif", "auto")], Vec::new()),
        &f,
    )
    .unwrap();
    assert!(is_jfif(&leading_segments(&bytes)[0]));
    // always, Exif: Exif then JFIF.
    let bytes = encode_one(
        &params(
            PixelFormat::YuvJ420P,
            &[("jfif", "always")],
            with_exif.clone(),
        ),
        &f,
    )
    .unwrap();
    let segs = leading_segments(&bytes);
    assert!(is_exif(&segs[0]) && is_jfif(&segs[1]), "{segs:?}");
    // never: no JFIF, with or without Exif; the picture still decodes
    // as full-range YCbCr.
    for b in [with_exif.clone(), Vec::new()] {
        let bytes =
            encode_one(&params(PixelFormat::YuvJ420P, &[("jfif", "never")], b), &f).unwrap();
        assert!(!leading_segments(&bytes).iter().any(is_jfif));
        let (stream, pkt) = demux(&bytes);
        assert_eq!(stream.pixel_format, Some(PixelFormat::YuvJ420P));
        let vf = decode(&stream, &pkt);
        assert_eq!(vf.image_planes().len(), 3);
        let info = oxideav_mjpeg::info(&bytes).unwrap();
        assert!(!info.has_jfif);
    }
    // The historical path honours the knob too.
    let mut enc = MjpegEncoder::from_params(&params(PixelFormat::Gray8, &[], Vec::new())).unwrap();
    enc.set_jfif_policy(JfifPolicy::Never);
    assert_eq!(enc.jfif_policy(), JfifPolicy::Never);
    enc.send_frame(&Frame::Video(gray8(0))).unwrap();
    let bytes = enc.receive_packet().unwrap().data;
    assert!(markers(&bytes).is_empty(), "{:?}", markers(&bytes));
    // Every policy yields the same picture.
    let reference = oxideav_mjpeg::decode(
        &encode_one(&params(PixelFormat::Gray8, &[], Vec::new()), &gray8(0)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        oxideav_mjpeg::decode(&bytes).unwrap().planes,
        reference.planes
    );
    // A bad spelling is a typed refusal.
    assert!(
        make_encoder(&params(
            PixelFormat::Gray8,
            &[("jfif", "maybe")],
            Vec::new()
        ))
        .is_err()
    );
}

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

#[test]
fn carriage_limits_are_typed_refusals_on_every_path() {
    let f = gray8(0);
    let too_big = [
        blobs(None, Some(&vec![0u8; EXIF_MAX_BYTES + 1]), None),
        blobs(None, None, Some(&vec![b' '; XMP_MAX_BYTES + 1])),
        blobs(Some(&vec![0u8; ICC_MAX_BYTES + 1]), None, None),
    ];
    for b in &too_big {
        for opts in [vec![], vec![("quality", "50")]] {
            let mut enc = make_encoder(&params(PixelFormat::Gray8, &opts, b.clone())).unwrap();
            let err = enc.send_frame(&Frame::Video(f.clone())).unwrap_err();
            assert!(matches!(err, oxideav_core::Error::Unsupported(_)), "{err}");
        }
    }
    // At the limit, every blob goes through and comes back.
    let at_limit = blobs(
        Some(&icc_profile(ICC_CHUNK_DATA * 3 + 1)),
        Some(&vec![0u8; EXIF_MAX_BYTES]),
        Some(&vec![b' '; XMP_MAX_BYTES]),
    );
    let bytes = encode_one(&params(PixelFormat::Gray8, &[], at_limit.clone()), &f).unwrap();
    let segs = leading_segments(&bytes);
    assert_eq!(
        segs.iter().filter(|s| is_icc(s)).count(),
        4,
        "four ICC chunks"
    );
    let (stream, pkt) = demux(&bytes);
    assert_eq!(stream.blobs, at_limit);
    assert_eq!(decode(&stream, &pkt).blobs(), at_limit);
}

#[test]
fn the_largest_profile_the_carriage_admits_round_trips() {
    // 255 chunks of 65 519 bytes (ICC.1 Annex B.4's 16 707 345 figure).
    let icc = icc_profile(ICC_MAX_BYTES);
    let bytes = encode_one(
        &params(PixelFormat::Gray8, &[], blobs(Some(&icc), None, None)),
        &gray8(0),
    )
    .unwrap();
    let m = oxideav_mjpeg::read_metadata(&bytes).unwrap();
    assert_eq!(m.icc.as_deref(), Some(&icc[..]));
    let (stream, _) = demux(&bytes);
    assert_eq!(
        stream.blob(&BlobKind::ICC).unwrap().data.len(),
        ICC_MAX_BYTES
    );
}

// ---------------------------------------------------------------------------
// Muxer pass-through
// ---------------------------------------------------------------------------

/// A `Cursor` whose buffer the test keeps a handle on.
struct Shared(Arc<Mutex<Vec<u8>>>, u64);

impl Write for Shared {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut v = self.0.lock().unwrap();
        let at = self.1 as usize;
        if v.len() < at + buf.len() {
            v.resize(at + buf.len(), 0);
        }
        v[at..at + buf.len()].copy_from_slice(buf);
        self.1 += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Seek for Shared {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let len = self.0.lock().unwrap().len() as i64;
        let new = match pos {
            SeekFrom::Start(s) => s as i64,
            SeekFrom::End(e) => len + e,
            SeekFrom::Current(c) => self.1 as i64 + c,
        };
        self.1 = new.max(0) as u64;
        Ok(self.1)
    }
}

#[test]
fn muxed_file_declares_the_blobs_again() {
    let icc = icc_profile(2000);
    let cp = params(PixelFormat::YuvJ420P, &[], blobs(Some(&icc), None, None));
    let mut enc = make_encoder(&cp).unwrap();
    enc.send_frame(&Frame::Video(yuv420())).unwrap();
    let pkt = enc.receive_packet().unwrap();
    let buf = Arc::new(Mutex::new(Vec::new()));
    let mut ctx = RuntimeContext::new();
    oxideav_mjpeg::register(&mut ctx);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1),
        duration: Some(1),
        start_time: Some(0),
        params: enc.output_params().clone(),
    };
    let mut mux = ctx
        .containers
        .open_muxer(
            "jpeg",
            Box::new(Shared(buf.clone(), 0)),
            std::slice::from_ref(&stream),
        )
        .unwrap();
    mux.write_header().unwrap();
    mux.write_packet(&pkt).unwrap();
    mux.write_trailer().unwrap();
    let file = buf.lock().unwrap().clone();
    let (stream, _) = demux(&file);
    assert_eq!(stream.blob(&BlobKind::ICC).unwrap().data, icc);
}
