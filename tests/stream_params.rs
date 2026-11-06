//! The in-band `Frame::StreamParams` event (oxideav-core 0.1.43): the
//! registry decoder returns one event immediately before the first
//! frame of a new output shape — each Motion-JPEG frame is a complete
//! interchange stream whose frame header (T.81 §B.2.2: `X`, `Y`, the
//! component count and the sampling factors) may change size and
//! layout — with `output_params()` / `output_params_generation()`
//! already moved and the payload equal to `output_params()`; never for
//! the first shape, never for a constant stream. The encoder accepts the
//! event and re-configures for the next frame (MJPEG is intra-only).
//!
//! `fixtures/stream_params/splice.mjpeg` is black-box ffmpeg output
//! (`gen.sh`: four `-c:v mjpeg` segments spliced with the concat
//! demuxer; the script prints the per-frame ffprobe view pinned in
//! [`SPLICE`]). The other streams are hand-assembled from this crate's
//! encoder.
#![cfg(feature = "registry")]

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Decoder, Encoder, Error, Frame, Packet, PixelFormat, TimeBase,
    VideoFrame,
};
use oxideav_mjpeg::registry::{MjpegEncoder, make_decoder, make_encoder};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// ffprobe `-show_frames` of the splice: (width, height, pix_fmt).
const SPLICE: [(u32, u32, PixelFormat); 8] = [
    (40, 24, PixelFormat::YuvJ420P),
    (40, 24, PixelFormat::YuvJ420P),
    (64, 32, PixelFormat::YuvJ422P),
    (64, 32, PixelFormat::YuvJ422P),
    (40, 24, PixelFormat::YuvJ444P),
    (40, 24, PixelFormat::YuvJ444P),
    (40, 24, PixelFormat::YuvJ420P),
    (40, 24, PixelFormat::YuvJ420P),
];

/// The SOI..EOI interchange streams of a raw Motion-JPEG file (EOI
/// cannot occur inside entropy-coded data: `FF` is always stuffed).
fn frames(path: &str) -> Vec<Vec<u8>> {
    let data = std::fs::read(format!("{FIX}/{path}")).expect("fixture readable");
    let mut out = Vec::new();
    let (mut start, mut i) = (0, 0);
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

/// Per plane (width, height, bytes per sample-group) of `pf`.
fn plane_dims(pf: PixelFormat, w: u32, h: u32) -> Vec<(u32, u32, usize)> {
    use PixelFormat::*;
    let yuv = |dx: u32, dy: u32| {
        vec![
            (w, h, 1),
            (w.div_ceil(dx), h.div_ceil(dy), 1),
            (w.div_ceil(dx), h.div_ceil(dy), 1),
        ]
    };
    match pf {
        Gray8 => vec![(w, h, 1)],
        Rgb24 => vec![(w, h, 3)],
        YuvJ444P => yuv(1, 1),
        YuvJ422P => yuv(2, 1),
        YuvJ420P => yuv(2, 2),
        other => panic!("no test layout for {other:?}"),
    }
}

/// A synthetic frame of `pf` (`seed` varies the content).
fn picture(pf: PixelFormat, w: u32, h: u32, seed: usize) -> VideoFrame {
    let planes = plane_dims(pf, w, h)
        .into_iter()
        .enumerate()
        .map(|(pi, (pw, ph, b))| VideoPlane {
            stride: pw as usize * b,
            data: (0..(pw * ph) as usize * b)
                .map(|k| ((k * (3 + pi) + seed * 17) % 200 + 16) as u8)
                .collect(),
        })
        .collect();
    VideoFrame {
        pts: Some(seed as i64),
        planes,
    }
}

fn params(pf: PixelFormat, w: u32, h: u32) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("mjpeg"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    p
}

/// One frame of `pf` from this crate's registry encoder.
fn own_encode(pf: PixelFormat, w: u32, h: u32, seed: usize) -> Vec<u8> {
    let mut enc = make_encoder(&params(pf, w, h)).expect("encoder");
    enc.send_frame(&Frame::Video(picture(pf, w, h, seed)))
        .expect("frame");
    enc.receive_packet().expect("packet").data
}

fn decoder() -> Box<dyn Decoder> {
    make_decoder(&CodecParameters::video(CodecId::new("mjpeg"))).expect("decoder")
}

fn packet(data: &[u8], pts: i64) -> Packet {
    Packet::new(0, TimeBase::new(1, 25), data.to_vec()).with_pts(pts)
}

/// What a consumer sees, item by item.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Seen {
    /// A picture: (pts, width, height, pixel format) per
    /// `output_params()` right after it, and the generation then.
    Video(i64, u32, u32, PixelFormat, u64),
    /// An event: its payload's (width, height, pixel format) and the
    /// generation when it is returned.
    Event(u32, u32, PixelFormat, u64),
}

fn shape(p: &CodecParameters) -> (u32, u32, PixelFormat) {
    (
        p.width.expect("width"),
        p.height.expect("height"),
        p.pixel_format.expect("pixel format"),
    )
}

/// Check one returned item against the contract and record it.
fn note(dec: &dyn Decoder, f: Frame, out: &mut Vec<Seen>) {
    let now = dec.output_params().expect("described");
    let g = dec.output_params_generation();
    match f {
        Frame::Video(v) => {
            let (w, h, pf) = shape(&now);
            let dims = plane_dims(pf, w, h);
            assert_eq!(v.image_planes().len(), dims.len(), "planes of {pf:?}");
            for (pl, &(pw, ph, b)) in v.image_planes().iter().zip(&dims) {
                assert!(pl.stride >= pw as usize * b);
                assert_eq!(pl.data.len(), pl.stride * ph as usize, "{pf:?} {w}x{h}");
            }
            if let Some(Seen::Video(.., pg)) = out.last() {
                assert_eq!(*pg, g, "only an event moves the generation");
            }
            out.push(Seen::Video(v.pts.expect("pts"), w, h, pf, g));
        }
        Frame::StreamParams(p) => {
            assert_eq!(
                format!("{p:?}"),
                format!("{now:?}"),
                "payload == output_params()"
            );
            let (w, h, pf) = shape(&p);
            let (prev_shape, prev_g) = match out.last() {
                Some(&Seen::Video(_, w, h, pf, g)) | Some(&Seen::Event(w, h, pf, g)) => {
                    ((w, h, pf), g)
                }
                None => panic!("no event before the first shape"),
            };
            assert_eq!(g, prev_g.wrapping_add(1), "one bump per event");
            // Two events in a row only when a reset dropped the frame
            // behind the first: each one is a change of its own.
            assert_ne!((w, h, pf), prev_shape, "at most one event per change");
            out.push(Seen::Event(w, h, pf, g));
        }
        other => panic!("unexpected {other:?}"),
    }
}

/// Send each packet and pull everything it yields.
fn run(dec: &mut dyn Decoder, packets: &[Vec<u8>], out: &mut Vec<Seen>) {
    for (i, data) in packets.iter().enumerate() {
        dec.send_packet(&packet(data, i as i64)).expect("packet");
        loop {
            match dec.receive_frame() {
                Ok(f) => note(dec, f, out),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("decode: {e:?}"),
            }
        }
    }
}

fn decode(packets: &[Vec<u8>]) -> Vec<Seen> {
    let mut dec = decoder();
    let mut out = Vec::new();
    run(dec.as_mut(), packets, &mut out);
    dec.flush().unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    out
}

#[test]
fn ffmpeg_splice_emits_one_event_per_change() {
    use PixelFormat::*;
    let packets = frames("stream_params/splice.mjpeg");
    assert_eq!(packets.len(), SPLICE.len());
    let seen = decode(&packets);
    assert_eq!(
        seen,
        [
            Seen::Video(0, 40, 24, YuvJ420P, 0),
            Seen::Video(1, 40, 24, YuvJ420P, 0),
            Seen::Event(64, 32, YuvJ422P, 1),
            Seen::Video(2, 64, 32, YuvJ422P, 1),
            Seen::Video(3, 64, 32, YuvJ422P, 1),
            // Same size, other sampling: a change.
            Seen::Event(40, 24, YuvJ444P, 2),
            Seen::Video(4, 40, 24, YuvJ444P, 2),
            Seen::Video(5, 40, 24, YuvJ444P, 2),
            // Back to an earlier shape: a change too.
            Seen::Event(40, 24, YuvJ420P, 3),
            Seen::Video(6, 40, 24, YuvJ420P, 3),
            Seen::Video(7, 40, 24, YuvJ420P, 3),
        ]
    );
    // The pictures are what ffprobe reports for the splice.
    let pictures: Vec<_> = seen
        .iter()
        .filter_map(|s| match *s {
            Seen::Video(_, w, h, pf, _) => Some((w, h, pf)),
            _ => None,
        })
        .collect();
    assert_eq!(pictures, SPLICE);
}

#[test]
fn constant_streams_emit_no_event() {
    for name in [
        "output_params/yuvj420p_70x46.mjpeg",
        "output_params/yuvj422p_48x30.mjpeg",
        "output_params/yuvj444p_33x17.mjpeg",
    ] {
        let seen = decode(&frames(name));
        assert_eq!(seen.len(), 2, "{name}");
        assert!(
            seen.iter().all(|s| matches!(s, Seen::Video(.., 0))),
            "{name}: {seen:?}"
        );
    }
    // A single first frame of any shape is announced up front, not by
    // an event.
    let seen = decode(&[own_encode(PixelFormat::Gray8, 13, 9, 0)]);
    assert_eq!(seen, [Seen::Video(0, 13, 9, PixelFormat::Gray8, 0)]);
}

#[test]
fn hand_assembled_layout_changes() {
    use PixelFormat::*;
    // Size held, component count / sampling / colour coding changing.
    let packets = [
        own_encode(Gray8, 32, 16, 0),
        own_encode(Gray8, 32, 16, 1),
        own_encode(YuvJ420P, 32, 16, 2),
        own_encode(YuvJ422P, 32, 16, 3),
        own_encode(Rgb24, 32, 16, 4),
        own_encode(Rgb24, 32, 16, 5),
        // Size changing, layout held.
        own_encode(Rgb24, 17, 33, 6),
    ];
    assert_eq!(
        decode(&packets),
        [
            Seen::Video(0, 32, 16, Gray8, 0),
            Seen::Video(1, 32, 16, Gray8, 0),
            Seen::Event(32, 16, YuvJ420P, 1),
            Seen::Video(2, 32, 16, YuvJ420P, 1),
            Seen::Event(32, 16, YuvJ422P, 2),
            Seen::Video(3, 32, 16, YuvJ422P, 2),
            Seen::Event(32, 16, Rgb24, 3),
            Seen::Video(4, 32, 16, Rgb24, 3),
            Seen::Video(5, 32, 16, Rgb24, 3),
            Seen::Event(17, 33, Rgb24, 4),
            Seen::Video(6, 17, 33, Rgb24, 4),
        ]
    );
}

/// An interlaced field pair in one packet (two interchange streams of
/// the same frame header, RFC 2435 §4.1 type-specific 1 then 2) is one
/// picture of the field's shape: no event between the fields, none
/// between pairs, one when the field shape changes.
#[test]
fn field_pairs_count_as_one_picture_shape() {
    use PixelFormat::*;
    let pair = |w, h, seed| {
        let mut p = own_encode(YuvJ422P, w, h, seed);
        p.extend(own_encode(YuvJ422P, w, h, seed + 1));
        p
    };
    let packets = [
        pair(48, 16, 0),
        pair(48, 16, 2),
        pair(48, 16, 4),
        pair(64, 24, 6),
    ];
    assert_eq!(
        decode(&packets),
        [
            Seen::Video(0, 48, 16, YuvJ422P, 0),
            Seen::Video(1, 48, 16, YuvJ422P, 0),
            Seen::Video(2, 48, 16, YuvJ422P, 0),
            Seen::Event(64, 24, YuvJ422P, 1),
            Seen::Video(3, 64, 24, YuvJ422P, 1),
        ]
    );
}

/// The event is held in front of its frame: a packet sent before the
/// frame is pulled is refused, as for any undelivered frame.
#[test]
fn held_frame_blocks_the_next_packet() {
    let mut dec = decoder();
    dec.send_packet(&packet(&own_encode(PixelFormat::Gray8, 8, 8, 0), 0))
        .unwrap();
    assert!(!dec.receive_frame().unwrap().is_stream_params());
    dec.send_packet(&packet(&own_encode(PixelFormat::Gray8, 16, 8, 1), 1))
        .unwrap();
    assert!(dec.receive_frame().unwrap().is_stream_params());
    assert!(
        dec.send_packet(&packet(&own_encode(PixelFormat::Gray8, 16, 8, 2), 2))
            .is_err(),
        "the frame behind the event is still undelivered"
    );
    assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
    assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
}

#[test]
fn reset_and_flush_follow_the_same_rule() {
    use PixelFormat::*;
    let a = own_encode(YuvJ420P, 32, 16, 0);
    let b = own_encode(YuvJ444P, 24, 24, 1);
    let mut dec = decoder();
    let mut seen = Vec::new();
    run(dec.as_mut(), std::slice::from_ref(&a), &mut seen);

    // reset() with a packet sent but not decoded: dropped, the shape of
    // the last returned frame stays published, decoding resumes.
    dec.send_packet(&packet(&b, 1)).unwrap();
    dec.reset().unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    assert_eq!(shape(&dec.output_params().unwrap()), (32, 16, YuvJ420P));
    assert_eq!(dec.output_params_generation(), 0);
    // Same shape after the seek: no event.
    run(dec.as_mut(), std::slice::from_ref(&a), &mut seen);

    // reset() after the event, before its frame: the frame is dropped,
    // the new shape stays published, so the old shape after the seek is
    // a change of its own.
    dec.send_packet(&packet(&b, 2)).unwrap();
    let ev = dec.receive_frame().unwrap();
    note(dec.as_ref(), ev, &mut seen);
    dec.reset().unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    run(dec.as_mut(), std::slice::from_ref(&a), &mut seen);

    // Draining after flush(): event, frame, end of stream.
    dec.send_packet(&packet(&b, 3)).unwrap();
    dec.flush().unwrap();
    for _ in 0..2 {
        let f = dec.receive_frame().unwrap();
        note(dec.as_ref(), f, &mut seen);
    }
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    // reset() leaves end of stream.
    dec.reset().unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));

    let mut pts = 0;
    let seen: Vec<Seen> = seen
        .into_iter()
        .map(|s| match s {
            Seen::Video(_, w, h, pf, g) => {
                pts += 1;
                Seen::Video(pts - 1, w, h, pf, g)
            }
            e => e,
        })
        .collect();
    assert_eq!(
        seen,
        [
            Seen::Video(0, 32, 16, YuvJ420P, 0),
            Seen::Video(1, 32, 16, YuvJ420P, 0),
            Seen::Event(24, 24, YuvJ444P, 1),
            Seen::Event(32, 16, YuvJ420P, 2),
            Seen::Video(2, 32, 16, YuvJ420P, 2),
            Seen::Event(24, 24, YuvJ444P, 3),
            Seen::Video(3, 24, 24, YuvJ444P, 3),
        ]
    );
}

#[test]
fn receive_arena_frame_skips_the_event() {
    let packets = frames("stream_params/splice.mjpeg");
    let mut dec = decoder();
    let mut got = Vec::new();
    for (i, data) in packets.iter().enumerate() {
        dec.send_packet(&packet(data, i as i64)).unwrap();
        let f = dec.receive_arena_frame().expect("arena frame");
        // Luma plane: `stride × height` of the new shape.
        let h = dec.output_params().unwrap().height.unwrap();
        assert_eq!(f.header().height, h);
        assert_eq!(f.header().presentation_timestamp, Some(i as i64));
        got.push((h, dec.output_params_generation()));
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    }
    assert_eq!(
        got,
        [
            (24, 0),
            (24, 0),
            (32, 1),
            (32, 1),
            (24, 2),
            (24, 2),
            (24, 3),
            (24, 3)
        ]
    );
}

/// The standalone still-image API knows nothing of events, and a
/// still through the registry decoder is one picture.
#[test]
fn still_image_decode_is_unaffected() {
    let jpg = own_encode(PixelFormat::YuvJ420P, 21, 11, 0);
    let img = oxideav_mjpeg::decode(&jpg).expect("standalone decode");
    assert_eq!((img.width, img.height), (21, 11));
    assert_eq!(
        decode(&[jpg]),
        [Seen::Video(0, 21, 11, PixelFormat::YuvJ420P, 0)]
    );
}

// ---- encoder ------------------------------------------------------------

fn event(pf: PixelFormat, w: u32, h: u32) -> Frame {
    Frame::StreamParams(Box::new(params(pf, w, h)))
}

/// Decode one packet to its (width, height, pixel format).
fn decoded_shape(data: &[u8]) -> (u32, u32, PixelFormat) {
    let mut dec = decoder();
    dec.send_packet(&packet(data, 0)).unwrap();
    assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
    shape(&dec.output_params().unwrap())
}

#[test]
fn encoder_reconfigures_on_the_event() {
    use PixelFormat::*;
    let mut enc = make_encoder(&params(YuvJ420P, 32, 16)).unwrap();
    enc.send_frame(&Frame::Video(picture(YuvJ420P, 32, 16, 0)))
        .unwrap();
    // Size and layout change from the next frame on.
    for (pf, w, h, coded) in [
        (YuvJ422P, 48, 24, YuvJ422P),
        (Gray8, 48, 24, Gray8),
        (Rgb24, 20, 10, Rgb24),
        (YuvJ444P, 20, 10, YuvJ444P),
    ] {
        enc.send_frame(&event(pf, w, h)).unwrap();
        assert!(enc.receive_packet().is_ok());
        assert!(
            matches!(enc.receive_packet(), Err(Error::NeedMore)),
            "the event itself yields no packet"
        );
        assert_eq!(shape(enc.output_params()), (w, h, coded));
        enc.send_frame(&Frame::Video(picture(pf, w, h, 1))).unwrap();
        // A frame of the old shape no longer fits.
        assert!(
            enc.send_frame(&Frame::Video(picture(YuvJ420P, 4, 2, 1)))
                .is_err()
        );
    }
    let last = enc.receive_packet().unwrap();
    assert_eq!(decoded_shape(&last.data), (20, 10, YuvJ444P));
    // An event repeating the current shape changes nothing.
    enc.send_frame(&event(YuvJ444P, 20, 10)).unwrap();
    assert_eq!(shape(enc.output_params()), (20, 10, YuvJ444P));
}

#[test]
fn encoder_reconfigure_keeps_the_knobs_and_the_options() {
    use PixelFormat::*;
    // An option bag: re-derived for the new layout.
    let mut p = params(YuvJ420P, 16, 16);
    p.options.insert("process", "progressive");
    let mut enc = make_encoder(&p).unwrap();
    enc.send_frame(&event(Gray8, 24, 8)).unwrap();
    enc.send_frame(&Frame::Video(picture(Gray8, 24, 8, 0)))
        .unwrap();
    let data = enc.receive_packet().unwrap().data;
    assert!(data.windows(2).any(|w| w == [0xFF, 0xC2]), "still SOF2");
    assert_eq!(decoded_shape(&data), (24, 8, Gray8));

    // A knob: kept across the change.
    let mut enc = MjpegEncoder::from_params(&params(Gray8, 16, 16)).unwrap();
    enc.set_progressive(true);
    enc.send_frame(&event(YuvJ422P, 32, 8)).unwrap();
    enc.send_frame(&Frame::Video(picture(YuvJ422P, 32, 8, 0)))
        .unwrap();
    let data = enc.receive_packet().unwrap().data;
    assert!(data.windows(2).any(|w| w == [0xFF, 0xC2]), "still SOF2");
    assert_eq!(decoded_shape(&data), (32, 8, YuvJ422P));
}

#[test]
fn encoder_refuses_what_it_cannot_reconfigure() {
    use PixelFormat::*;
    // Abbreviated streams: the shared tables are a stream-level fact.
    let mut p = params(YuvJ420P, 16, 16);
    p.options.insert("abbreviated", "true");
    let mut enc = make_encoder(&p).unwrap();
    let before = format!("{:?}", enc.output_params());
    assert!(matches!(
        enc.send_frame(&event(YuvJ420P, 32, 16)),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(format!("{:?}", enc.output_params()), before);

    // A layout the encoder does not take: refused, nothing changed.
    let mut enc = make_encoder(&params(YuvJ420P, 16, 16)).unwrap();
    let before = format!("{:?}", enc.output_params());
    assert!(enc.send_frame(&event(Rgba, 16, 16)).is_err());
    assert!(enc.send_frame(&event(YuvJ420P, 0, 16)).is_err());
    assert_eq!(format!("{:?}", enc.output_params()), before);
    enc.send_frame(&Frame::Video(picture(YuvJ420P, 16, 16, 0)))
        .unwrap();
    let data = enc.receive_packet().unwrap().data;
    assert_eq!(decoded_shape(&data), (16, 16, YuvJ420P));
}
