//! `COM` (comment) segments, T.81 §B.2.4.5 (#21): read into
//! [`Metadata::comments`], counted by [`info`], written by every encoder
//! that takes [`EncodeOptions::metadata`], replaced by [`embed_metadata`]
//! and removed by [`strip_metadata`].
//!
//! `fixtures/comments/two_com.jpg` is black-box validator output:
//! `cjpeg -quality 50` of a 16×12 frame piped through `wrjpgcom -comment
//! "first comment"` and `wrjpgcom -comment "second: ü UTF-8"` (two `COM`
//! segments, between the `DQT` segments and `SOF0`). When `rdjpgcom` is
//! installed the comments this crate writes are read back through it.

use oxideav_mjpeg::{
    EncodeOptions, JfifPolicy, Metadata, decode, embed_metadata, encode, encode_rgb8, info,
    read_metadata, strip_metadata,
};

const TWO_COM: &[u8] = include_bytes!("fixtures/comments/two_com.jpg");

/// Marker codes of the header segments, up to and including the first
/// `SOS`.
fn header_markers(jpeg: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        let m = jpeg[i + 1];
        out.push(m);
        if m == 0xDA {
            break;
        }
        i += 2 + u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
    }
    out
}

#[test]
fn comments_are_read_in_stream_order() {
    let want: Vec<Vec<u8>> = vec![
        b"first comment".to_vec(),
        "second: ü UTF-8".as_bytes().to_vec(),
    ];
    assert_eq!(read_metadata(TWO_COM).unwrap().comments, want);
    assert_eq!(info(TWO_COM).unwrap().comments, 2);
    let img = decode(TWO_COM).unwrap();
    assert_eq!(img.metadata.comments, want);
    // A stream without COM reports none.
    let rgb = vec![128u8; 16 * 12 * 3];
    let plain = encode_rgb8(16, 12, &rgb, &EncodeOptions::default()).unwrap();
    assert!(read_metadata(&plain).unwrap().comments.is_empty());
    assert_eq!(info(&plain).unwrap().comments, 0);
}

#[test]
fn comments_are_written_after_the_app_segments() {
    let rgb: Vec<u8> = (0..16 * 12 * 3).map(|i| (i * 7) as u8).collect();
    let mut opts = EncodeOptions::default();
    opts.metadata = Metadata::new()
        .with_comment("hello")
        .with_comment(vec![0u8, 1, 2, 0xFF])
        .with_comment("");
    let out = encode_rgb8(16, 12, &rgb, &opts).unwrap();
    assert_eq!(
        header_markers(&out),
        [
            0xE0, 0xFE, 0xFE, 0xFE, 0xDB, 0xDB, 0xC0, 0xC4, 0xC4, 0xC4, 0xC4, 0xDA
        ]
    );
    let back = read_metadata(&out).unwrap();
    assert_eq!(back.comments, opts.metadata.comments);
    // With an ICC profile the COM segments still follow every APPn.
    opts.metadata = opts.metadata.clone().with_icc(vec![9; 40]);
    let out = encode_rgb8(16, 12, &rgb, &opts).unwrap();
    assert_eq!(&header_markers(&out)[..5], [0xE0, 0xE2, 0xFE, 0xFE, 0xFE]);
    decode(&out).unwrap();
}

#[test]
fn a_decoded_image_keeps_its_comments_through_encode() {
    let img = decode(TWO_COM).unwrap();
    let out = encode(&img, &EncodeOptions::default()).unwrap();
    assert_eq!(read_metadata(&out).unwrap().comments, img.metadata.comments);
    // The option wins over the image's own list.
    let mut opts = EncodeOptions::default();
    opts.metadata = Metadata::new().with_comment("replaced");
    let out = encode(&img, &opts).unwrap();
    assert_eq!(
        read_metadata(&out).unwrap().comments,
        vec![b"replaced".to_vec()]
    );
}

#[test]
fn embed_replaces_and_strip_removes_comments() {
    // Other metadata leaves the existing COM segments where they were.
    let with_xmp = embed_metadata(
        TWO_COM,
        &Metadata::new().with_xmp(b"<x/>".to_vec()),
        JfifPolicy::Auto,
    )
    .unwrap();
    assert_eq!(
        header_markers(&with_xmp),
        [
            0xE0, 0xE1, 0xDB, 0xDB, 0xFE, 0xFE, 0xC0, 0xC4, 0xC4, 0xC4, 0xC4, 0xDA
        ]
    );
    // A comment list replaces them, ahead of the tables.
    let replaced = embed_metadata(
        TWO_COM,
        &Metadata::new().with_comment("only"),
        JfifPolicy::Auto,
    )
    .unwrap();
    assert_eq!(
        header_markers(&replaced),
        [0xE0, 0xFE, 0xDB, 0xDB, 0xC0, 0xC4, 0xC4, 0xC4, 0xC4, 0xDA]
    );
    assert_eq!(
        read_metadata(&replaced).unwrap().comments,
        vec![b"only".to_vec()]
    );
    assert_eq!(
        decode(&replaced).unwrap().planes,
        decode(TWO_COM).unwrap().planes
    );
    let stripped = strip_metadata(TWO_COM).unwrap();
    assert!(read_metadata(&stripped).unwrap().comments.is_empty());
    assert_eq!(
        decode(&stripped).unwrap().planes,
        decode(TWO_COM).unwrap().planes
    );
}

#[test]
fn an_oversized_comment_is_refused() {
    let rgb = vec![0u8; 8 * 8 * 3];
    let mut opts = EncodeOptions::default();
    opts.metadata = Metadata::new().with_comment(vec![b'a'; 65_533]);
    assert!(encode_rgb8(8, 8, &rgb, &opts).is_ok());
    opts.metadata = Metadata::new().with_comment(vec![b'a'; 65_534]);
    assert!(encode_rgb8(8, 8, &rgb, &opts).is_err());
}

/// Black-box: `rdjpgcom` reads back what this crate writes.
#[test]
fn rdjpgcom_reads_our_comments() {
    let rgb: Vec<u8> = (0..16 * 12 * 3).map(|i| (i * 3) as u8).collect();
    let mut opts = EncodeOptions::default();
    opts.metadata = Metadata::new()
        .with_comment("written by oxideav")
        .with_comment("line two");
    let out = encode_rgb8(16, 12, &rgb, &opts).unwrap();
    let dir = std::env::temp_dir().join(format!("oxideav-mjpeg-com-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("c.jpg");
    std::fs::write(&path, &out).unwrap();
    let Ok(run) = std::process::Command::new("rdjpgcom").arg(&path).output() else {
        return; // validator not installed
    };
    let _ = std::fs::remove_dir_all(&dir);
    assert!(run.status.success());
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "written by oxideav\nline two\n"
    );
}

/// Fuzz regression (`metadata_embed`, crash `cce9333d`): a `COM` segment
/// has no framework blob form, so the registry decoder's frame record is
/// the ICC / Exif / XMP part of [`read_metadata`] and nothing else. The
/// harness used to expect the two records to be equal, comments included.
#[cfg(feature = "registry")]
#[test]
fn registry_frame_blobs_leave_comments_out() {
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

    let meta = Metadata::new()
        .with_exif(vec![0x4D, 0x4D, 0x00, 0x2A, 0, 0, 0, 8, 0, 0])
        .with_comment(b"kept in the stream".to_vec());
    let jpeg = embed_metadata(TWO_COM, &meta, JfifPolicy::Auto).expect("embed");
    let read = read_metadata(&jpeg).expect("read");
    assert_eq!(read.comments, vec![b"kept in the stream".to_vec()]);
    assert!(read.exif.is_some());

    let params = CodecParameters::video(CodecId::new(oxideav_mjpeg::CODEC_ID_STR));
    let mut dec = oxideav_mjpeg::registry::make_decoder(&params).expect("decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), jpeg))
        .expect("send");
    let Ok(Frame::Video(vf)) = dec.receive_frame() else {
        panic!("no frame");
    };
    let from_frame = Metadata::from_blobs(&vf.blobs());
    assert!(from_frame.comments.is_empty());
    let mut carried = read.clone();
    carried.comments.clear();
    assert_eq!(from_frame, carried);
}
