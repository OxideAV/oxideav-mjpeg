#![cfg(feature = "registry")]
//! Integration tests for the raw Motion-JPEG container demuxer's
//! [`Demuxer::seek_to`] implementation.
//!
//! Fixture: a tiny `.mjpeg` file generated on-the-fly with
//! `ffmpeg -f lavfi -i testsrc=rate=5:duration=2:size=160x120
//!   -c:v mjpeg -f mjpeg <out>` — yielding 10 frames of 160×120
//! Motion-JPEG. The container's default frame rate is 25 fps, so
//! pts = frame index (0..=9) in a `1/25` time-base.
//!
//! When ffmpeg is unavailable on the host (which happens in some
//! sandboxed CI runners), the test gracefully skips with an
//! `eprintln!` rather than failing — matching the convention used
//! by other ffmpeg-driven tests in this crate (see
//! `tests/decode_real_jpeg.rs`).

use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

use oxideav_core::{ContainerRegistry, NullCodecResolver, ReadSeek};

/// Generate the 10-frame raw MJPEG fixture under `OUT_DIR/testsrc_10frames.mjpeg`
/// and return its absolute path. Returns `None` when ffmpeg is missing.
fn generate_fixture() -> Option<PathBuf> {
    let out = std::env::temp_dir().join("oxideav_mjpeg_seek_testsrc_10frames.mjpeg");
    // Always regenerate so a stale half-written file from a previous
    // run can't make the seek tests flaky.
    let _ = std::fs::remove_file(&out);
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=rate=5:duration=2:size=160x120",
            "-c:v",
            "mjpeg",
            "-f",
            "mjpeg",
        ])
        .arg(&out)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    Some(out)
}

fn open_demuxer(bytes: Vec<u8>) -> Box<dyn oxideav_core::Demuxer> {
    let mut reg = ContainerRegistry::new();
    oxideav_mjpeg::register_containers(&mut reg);
    let input: Box<dyn ReadSeek> = Box::new(Cursor::new(bytes));
    reg.open_demuxer("mjpeg-raw", input, &NullCodecResolver)
        .expect("open mjpeg-raw demuxer")
}

fn read_all_pts(d: &mut dyn oxideav_core::Demuxer) -> Vec<i64> {
    let mut pts = Vec::new();
    loop {
        match d.next_packet() {
            Ok(pkt) => pts.push(pkt.pts.expect("pts set on every packet")),
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("unexpected demuxer error: {e}"),
        }
    }
    pts
}

#[test]
fn fixture_has_ten_frames() {
    let Some(path) = generate_fixture() else {
        eprintln!("ffmpeg not available — skipping");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let mut d = open_demuxer(bytes);
    let pts = read_all_pts(d.as_mut());
    assert_eq!(pts.len(), 10, "expected 10 frames, got {}", pts.len());
    assert_eq!(pts, (0..10).collect::<Vec<i64>>());
}

#[test]
fn seek_to_zero_resets_to_first_frame() {
    let Some(path) = generate_fixture() else {
        eprintln!("ffmpeg not available — skipping");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let mut d = open_demuxer(bytes);

    // Drain a couple of packets.
    let p0 = d.next_packet().expect("frame 0");
    assert_eq!(p0.pts, Some(0));
    let p1 = d.next_packet().expect("frame 1");
    assert_eq!(p1.pts, Some(1));

    let landed = d.seek_to(0, 0).expect("seek to 0");
    assert_eq!(landed, 0);
    let after = d.next_packet().expect("post-seek frame");
    assert_eq!(after.pts, Some(0), "first packet after seek_to(0) is pts=0");
}

#[test]
fn seek_to_pts_3_lands_at_fourth_frame() {
    let Some(path) = generate_fixture() else {
        eprintln!("ffmpeg not available — skipping");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let mut d = open_demuxer(bytes);

    let landed = d.seek_to(0, 3).expect("seek to 3");
    assert_eq!(landed, 3);
    let pkt = d.next_packet().expect("packet after seek");
    assert_eq!(
        pkt.pts,
        Some(3),
        "next_packet after seek_to(3) should emit pts=3"
    );

    // And the rest of the stream advances cleanly from there.
    let rest = read_all_pts(d.as_mut());
    assert_eq!(rest, vec![4, 5, 6, 7, 8, 9]);
}

#[test]
fn seek_past_end_clamps() {
    let Some(path) = generate_fixture() else {
        eprintln!("ffmpeg not available — skipping");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let mut d = open_demuxer(bytes);

    // Past-end target: 1000 ≫ 10 frames.
    let landed = d.seek_to(0, 1000).expect("seek past end");
    // Spec: clamps to last frame; landed pts is the last frame we
    // actually visited (frame index 9 here).
    assert!(
        landed <= 9,
        "expected clamp to <= 9 frames, got pts={landed}"
    );
    // And the next read returns EOF.
    let err = d
        .next_packet()
        .expect_err("EOF expected after seek-past-end");
    assert!(matches!(err, oxideav_core::Error::Eof));
}

#[test]
fn frame_index_skips_byte_stuffed_ff_d8_inside_jpeg_data() {
    // Hand-crafted MJPEG stream that exercises the two ways a naive
    // SOI scanner could false-match inside compressed JPEG data:
    //
    // (a) `FF 00 D8 D8`: 0xFF 0x00 is a stuffed literal 0xFF inside
    //     entropy data; the bytes 0xD8 0xD8 that follow are literal
    //     scan data (not an SOI). A spec-blind `memmem("\xFF\xD8")`
    //     scanner would mis-detect an SOI here. Frame_a embeds this
    //     pattern inside its SOS scan payload.
    //
    // (b) An APP1 segment that carries an embedded JPEG payload
    //     (the EXIF "thumbnail" case in real cameras): inside the
    //     APP1 length-prefixed body sits a full `FF D8 .. FF D9`
    //     byte sequence. A naive scanner WOULD see that as a frame
    //     boundary. The proper marker-aware walker stays inside the
    //     APP1 segment's declared length.
    //
    // Frame_b carries a valid SOF0 so the demuxer can probe canvas
    // size at open time. Frame_a deliberately omits SOF (it's a
    // contrived frame).
    let frame_a: &[u8] = &[
        0xFF, 0xD8, // SOI
        0xFF, 0xE0, 0x00, 0x02, // APP0 with length=2 (empty)
        // APP1 segment whose body contains a literal embedded JPEG.
        // length = 2 (length field) + 4 (embedded "JPEG" bytes) = 6
        // → 0x0006.
        0xFF, 0xE1, 0x00, 0x06, 0xFF, 0xD8, 0xFF, 0xD9, 0xFF, 0xDA, 0x00,
        0x02, // SOS with length=2
        0xFF, 0x00, 0xD8, 0xD8, // (a) stuffed FF + literal D8 D8
        0xFF, 0xD9, // EOI
    ];
    let frame_b: &[u8] = &[
        0xFF, 0xD8, // SOI
        0xFF, 0xE0, 0x00, 0x02, // APP0 with length=2 (empty)
        // Mini-SOF0 so the demuxer can probe canvas size:
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF,
        0xD9, // EOI
    ];
    // Build the stream as frame_b + frame_a + frame_b + frame_a so the
    // first frame carries a valid SOF (needed by the demuxer at open
    // time) and subsequent frames exercise the stuffing-aware scanner.
    let mut stream = Vec::new();
    stream.extend_from_slice(frame_b);
    stream.extend_from_slice(frame_a);
    stream.extend_from_slice(frame_b);
    stream.extend_from_slice(frame_a);

    let mut d = open_demuxer(stream);
    let pts = read_all_pts(d.as_mut());
    // A naive `memmem("FF D8")` scanner would emit 6+ frames here
    // (two extras for each embedded APP1 thumbnail). Our walker
    // stays at 4 — one per real top-level frame.
    assert_eq!(
        pts,
        vec![0, 1, 2, 3],
        "scanner should treat embedded FF D8 inside APP1 / stuffed entropy as literal bytes"
    );
}

#[test]
fn seek_then_full_drain_matches_baseline() {
    let Some(path) = generate_fixture() else {
        eprintln!("ffmpeg not available — skipping");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");

    // Baseline: read all packets and collect their bodies.
    let mut d0 = open_demuxer(bytes.clone());
    let mut baseline_bodies: Vec<(i64, Vec<u8>)> = Vec::new();
    while let Ok(p) = d0.next_packet() {
        baseline_bodies.push((p.pts.unwrap(), p.data));
    }

    // Now seek to pts=5 and verify the bytes match baseline frames 5..=9.
    let mut d1 = open_demuxer(bytes);
    let landed = d1.seek_to(0, 5).expect("seek to 5");
    assert_eq!(landed, 5);
    let mut after: Vec<(i64, Vec<u8>)> = Vec::new();
    while let Ok(p) = d1.next_packet() {
        after.push((p.pts.unwrap(), p.data));
    }
    assert_eq!(
        after,
        baseline_bodies[5..],
        "seek-then-drain bodies must equal baseline tail"
    );
}
