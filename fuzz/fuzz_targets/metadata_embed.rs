#![no_main]

//! Robustness + oracle target for the metadata marker paths
//! (`oxideav_mjpeg::metadata`): the leading-segment splitter behind
//! `embed_metadata` / `strip_metadata`, the APP2 ICC chunk reassembly
//! behind `read_metadata` (hostile sequence numbers, repeated chunks,
//! disagreeing counts, zero-length chunks), and the registry decoder's
//! blob record.
//!
//! Two shapes per iteration, both derived from the fuzz bytes:
//!
//! 1. **Arbitrary stream.** The tail of the input is treated as a JPEG
//!    and fed to `read_metadata`, then `embed_metadata` with a
//!    fuzz-chosen `JfifPolicy` and fuzz-sized blobs, then
//!    `strip_metadata`. When the stream has a frame header (`read` is
//!    `Ok`) the embedded blobs must read back exactly, the strip must
//!    leave none of the three kinds, and the picture (when it decodes
//!    under a pixel cap) must be the same planes before and after.
//! 2. **Synthetic ICC chunk sequence.** Fuzz-chosen `(seq, total,
//!    len)` triples become `APP2 ICC_PROFILE` segments in front of a
//!    small valid grey frame; `read_metadata`, `info`, lenient and
//!    strict `decode_with` must return (`Ok` or `Err`), never panic,
//!    and whatever `read` reports must re-embed and read back.
//!
//! The bar: no panic; `embed_metadata` fails only with `InvalidData`
//! (malformed leading segment) or `Unsupported` (a blob over its
//! limit).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_mjpeg::metadata::{EXIF_MAX_BYTES, ICC_IDENTIFIER, XMP_MAX_BYTES};
use oxideav_mjpeg::{
    DecodeOptions, EncodeOptions, JfifPolicy, JpegImage, Metadata, MjpegError, PixelFormat, Plane,
    embed_metadata, read_metadata, strip_metadata,
};

const MAX_INPUT_LEN: usize = 64 * 1024;

/// A blob of `len` bytes filled from a short pattern (any bytes carry).
fn blob(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
        .collect()
}

/// Fuzz-shaped metadata: which kinds are set and how long each is,
/// sizes spanning the interesting bands (empty, one segment, the
/// chunk boundary, just over a limit).
fn meta_from(ctl: &[u8]) -> Metadata {
    let pick = |i: usize| ctl.get(i).copied().unwrap_or(0);
    let size = |b: u8| -> usize {
        match b % 8 {
            0 => 0,
            1 => 1,
            2 => usize::from(b) * 3,
            3 => 65_519,
            4 => 65_520,
            5 => EXIF_MAX_BYTES,
            6 => XMP_MAX_BYTES + 1,
            _ => usize::from(b) * 300,
        }
    };
    let mut m = Metadata::new();
    if pick(0) & 1 != 0 {
        m.icc = Some(blob(size(pick(1)), 1));
    }
    if pick(0) & 2 != 0 {
        m.exif = Some(blob(size(pick(2)), 2));
    }
    if pick(0) & 4 != 0 {
        m.xmp = Some(blob(size(pick(3)), 3));
    }
    m
}

fn policy(b: u8) -> JfifPolicy {
    match b % 3 {
        0 => JfifPolicy::Auto,
        1 => JfifPolicy::Always,
        _ => JfifPolicy::Never,
    }
}

fn cap() -> DecodeOptions {
    DecodeOptions::new().with_max_pixels(Some(1 << 16))
}

/// Shape 1: arbitrary bytes as the stream.
fn arbitrary_stream(ctl: &[u8], jpeg: &[u8]) {
    let before = read_metadata(jpeg);
    let meta = meta_from(ctl);
    let pol = policy(ctl.first().copied().unwrap_or(0) >> 3);
    let embedded = match embed_metadata(jpeg, &meta, pol) {
        Ok(out) => out,
        Err(MjpegError::InvalidData(_)) | Err(MjpegError::Unsupported(_)) => return,
        Err(e) => panic!("embed_metadata: unexpected error kind {e}"),
    };
    let Ok(before) = before else {
        // No frame header: the rewrite is still well formed as far as
        // its own splitter is concerned.
        let _ = strip_metadata(&embedded).expect("strip after embed");
        return;
    };
    let after = read_metadata(&embedded).expect("read after embed");
    for (name, want, got) in [
        ("icc", &meta.icc, &after.icc),
        ("exif", &meta.exif, &after.exif),
        ("xmp", &meta.xmp, &after.xmp),
    ] {
        if want.is_some() {
            assert_eq!(got, want, "{name}: embedded blob reads back");
        }
    }
    // Kinds `meta` left alone keep what the stream had.
    if meta.exif.is_none() {
        assert_eq!(after.exif, before.exif, "exif kept");
    }
    if meta.xmp.is_none() {
        assert_eq!(after.xmp, before.xmp, "xmp kept");
    }
    let stripped = strip_metadata(&embedded).expect("strip");
    let none = read_metadata(&stripped).expect("read after strip");
    assert!(none.icc.is_none() && none.exif.is_none() && none.xmp.is_none());

    // The picture is untouched by the rewrite.
    if let Ok(orig) = oxideav_mjpeg::decode_with(jpeg, &cap()) {
        let re = oxideav_mjpeg::decode_with(&embedded, &cap()).expect("decode after embed");
        assert_eq!(re.planes, orig.planes, "planes unchanged by embed");
        assert_eq!(re.format, orig.format);
        // The registry decoder's frame record is the same metadata.
        let params = CodecParameters::video(CodecId::new(oxideav_mjpeg::CODEC_ID_STR));
        let mut dec = oxideav_mjpeg::registry::make_decoder(&params).expect("decoder");
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), embedded.clone()))
            .expect("send");
        if let Ok(Frame::Video(vf)) = dec.receive_frame() {
            assert_eq!(
                Metadata::from_blobs(&vf.blobs()),
                after,
                "frame blobs == metadata"
            );
        }
    }
}

/// Shape 2: a fuzz-chosen ICC chunk sequence in front of a valid frame.
fn chunk_sequence(ctl: &[u8]) {
    let img = JpegImage::new(
        4,
        4,
        PixelFormat::Gray8,
        vec![Plane::new(4, (0..16u8).collect())],
    )
    .unwrap();
    let base = oxideav_mjpeg::encode(&img, &EncodeOptions::new()).unwrap();
    let mut jpeg = vec![0xFF, 0xD8];
    for t in ctl.chunks(3).take(12) {
        let (seq, total) = (t[0], t.get(1).copied().unwrap_or(1));
        let len = usize::from(t.get(2).copied().unwrap_or(0)) * 7;
        let lp = 2 + ICC_IDENTIFIER.len() + 2 + len;
        jpeg.extend_from_slice(&[0xFF, 0xE2, (lp >> 8) as u8, lp as u8]);
        jpeg.extend_from_slice(ICC_IDENTIFIER);
        jpeg.extend_from_slice(&[seq, total]);
        jpeg.extend(std::iter::repeat_n(seq, len));
    }
    jpeg.extend_from_slice(&base[2..]);
    let m = read_metadata(&jpeg).expect("valid frame behind the chunks");
    let info = oxideav_mjpeg::info(&jpeg).expect("info");
    assert_eq!(info.has_icc, !ctl.is_empty());
    let lenient = oxideav_mjpeg::decode(&jpeg).expect("lenient decode");
    assert_eq!(lenient.metadata.icc, m.icc, "decode agrees with read");
    let strict = oxideav_mjpeg::decode_with(&jpeg, &DecodeOptions::new().with_strict(true));
    if m.icc.is_none() && !ctl.is_empty() {
        assert!(strict.is_err(), "strict mode rejects an unusable sequence");
    }
    // What was read re-embeds and reads back; a `None` profile leaves
    // the hostile chunks in place (they are the stream's, not ours).
    let again = embed_metadata(&jpeg, &m, JfifPolicy::Auto).expect("re-embed");
    assert_eq!(read_metadata(&again).expect("read").icc, m.icc);
    let stripped = strip_metadata(&jpeg).expect("strip");
    assert!(!oxideav_mjpeg::info(&stripped).expect("info").has_icc);
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_LEN || data.len() < 2 {
        return;
    }
    let (ctl, jpeg) = data.split_at(data.len().min(8));
    arbitrary_stream(ctl, jpeg);
    chunk_sequence(&data[..data.len().min(36)]);
});
