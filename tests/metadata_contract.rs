//! Metadata on the standalone contract surface (no `oxideav-core`):
//! `embed_metadata` / `strip_metadata` / `read_metadata`, the
//! `EncodeOptions::jfif` slot rule, and the hostile ICC chunk sequences
//! the header scan must survive. One source of truth: whatever
//! `read_metadata` reports is what `decode` reports and what the
//! registry turns into blobs.

use oxideav_mjpeg::metadata::{ICC_CHUNK_DATA, ICC_IDENTIFIER};
use oxideav_mjpeg::{
    embed_metadata, read_metadata, strip_metadata, DecodeOptions, EncodeOptions, JfifPolicy,
    JpegImage, Metadata, PixelFormat, Plane,
};

fn gray(w: u32, h: u32) -> JpegImage {
    let data: Vec<u8> = (0..(w * h) as usize).map(|i| (i * 7 % 256) as u8).collect();
    JpegImage::new(w, h, PixelFormat::Gray8, vec![Plane::new(w as usize, data)]).unwrap()
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

/// `(marker, first payload bytes)` of the leading APPn segments.
fn leading(jpeg: &[u8]) -> Vec<(u8, Vec<u8>)> {
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

fn exif() -> Vec<u8> {
    b"MM\0*\0\0\0\x08\0\0\0\0\0\0".to_vec()
}

#[test]
fn encode_embeds_in_the_documented_order_and_read_agrees_with_decode() {
    let meta = Metadata::new()
        .with_icc(vec![3; 2 * ICC_CHUNK_DATA + 5])
        .with_exif(exif())
        .with_xmp(b"<x:xmpmeta/>".to_vec());
    let img = gray(17, 9).with_metadata(meta.clone());
    let bytes = oxideav_mjpeg::encode(&img, &EncodeOptions::new()).unwrap();
    let segs = leading(&bytes);
    assert!(segs[0].1.starts_with(b"Exif\0\0"), "{segs:?}");
    assert!(segs[1].1.starts_with(b"http://ns.ad"), "{segs:?}");
    assert_eq!(segs[2].0, 0xE2);
    assert_eq!(segs[3].0, 0xE2);
    assert_eq!(segs[4].0, 0xE2);
    assert_eq!(segs.len(), 5, "no JFIF with Exif: {segs:?}");
    assert_eq!(read_metadata(&bytes).unwrap(), meta);
    assert_eq!(oxideav_mjpeg::decode(&bytes).unwrap().metadata, meta);
    let info = oxideav_mjpeg::info(&bytes).unwrap();
    assert!(info.has_icc && info.has_exif && info.has_xmp && !info.has_jfif);

    // Without Exif the JFIF segment leads; `jfif = Never` drops it.
    let no_exif = Metadata::new().with_xmp(b"<x/>".to_vec());
    let bytes = oxideav_mjpeg::encode(
        &gray(17, 9).with_metadata(no_exif.clone()),
        &EncodeOptions::new(),
    )
    .unwrap();
    let segs = leading(&bytes);
    assert!(segs[0].1.starts_with(b"JFIF\0") && segs[1].1.starts_with(b"http://ns.ad"));
    let bytes = oxideav_mjpeg::encode(
        &gray(17, 9).with_metadata(no_exif.clone()),
        &EncodeOptions::new().with_jfif(JfifPolicy::Never),
    )
    .unwrap();
    assert_eq!(leading(&bytes).len(), 1);
    assert_eq!(read_metadata(&bytes).unwrap(), no_exif);
    // `Always` keeps JFIF after Exif.
    let bytes = oxideav_mjpeg::encode(
        &gray(17, 9).with_metadata(Metadata::new().with_exif(exif())),
        &EncodeOptions::new().with_jfif(JfifPolicy::Always),
    )
    .unwrap();
    let segs = leading(&bytes);
    assert!(
        segs[0].1.starts_with(b"Exif\0\0") && segs[1].1.starts_with(b"JFIF\0"),
        "{segs:?}"
    );
    assert!(oxideav_mjpeg::info(&bytes).unwrap().has_jfif);
}

#[test]
fn embed_on_the_corpus_profile_fixture() {
    let Some(jpg) = corpus("with-icc-profile-embedded") else {
        return;
    };
    let before = read_metadata(&jpg).unwrap();
    assert_eq!(before.icc.as_ref().map(Vec::len), Some(2576));
    let segs = leading(&jpg);
    assert!(segs[0].1.starts_with(b"JFIF\0") && segs[1].1.starts_with(ICC_IDENTIFIER));

    // Add Exif: it takes the slot, JFIF goes, the profile stays, the
    // picture is untouched.
    let with_exif =
        embed_metadata(&jpg, &Metadata::new().with_exif(exif()), JfifPolicy::Auto).unwrap();
    let segs = leading(&with_exif);
    assert!(segs[0].1.starts_with(b"Exif\0\0"), "{segs:?}");
    assert!(segs[1].1.starts_with(ICC_IDENTIFIER), "{segs:?}");
    assert_eq!(segs.len(), 2);
    let m = read_metadata(&with_exif).unwrap();
    assert_eq!(m.icc, before.icc);
    assert_eq!(m.exif, Some(exif()));
    assert_eq!(
        oxideav_mjpeg::decode(&with_exif).unwrap().planes,
        oxideav_mjpeg::decode(&jpg).unwrap().planes
    );

    // Strip: colour signalling stays, metadata goes.
    let bare = strip_metadata(&with_exif).unwrap();
    assert!(leading(&bare).is_empty());
    assert!(read_metadata(&bare).unwrap().is_empty());
    let back = embed_metadata(&bare, &before, JfifPolicy::Auto).unwrap();
    assert_eq!(read_metadata(&back).unwrap(), before);
}

/// One `APP2 ICC_PROFILE` chunk: `(sequence number, chunk count, data)`.
type Chunk<'a> = (u8, u8, &'a [u8]);
/// A hostile-sequence case: name, chunks, the profile `read_metadata`
/// must report (`None` = unusable sequence).
type Case<'a> = (&'a str, &'a [Chunk<'a>], Option<Vec<u8>>);

/// Build `SOI, [chunks…], <gray frame without its own metadata>`.
fn with_icc_chunks(chunks: &[Chunk<'_>]) -> Vec<u8> {
    let base = oxideav_mjpeg::encode(&gray(8, 8), &EncodeOptions::new()).unwrap();
    let mut out = vec![0xFF, 0xD8];
    for (seq, total, data) in chunks {
        let len = 2 + ICC_IDENTIFIER.len() + 2 + data.len();
        out.extend_from_slice(&[0xFF, 0xE2, (len >> 8) as u8, len as u8]);
        out.extend_from_slice(ICC_IDENTIFIER);
        out.extend_from_slice(&[*seq, *total]);
        out.extend_from_slice(data);
    }
    out.extend_from_slice(&base[2..]);
    out
}

#[test]
fn hostile_icc_sequences_never_panic_and_strict_mode_names_them() {
    let a = [1u8; 5];
    let b = [2u8; 5];
    let cases: &[Case<'_>] = &[
        (
            "in order",
            &[(1, 2, &a), (2, 2, &b)],
            Some(vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        ),
        (
            "reversed",
            &[(2, 2, &b), (1, 2, &a)],
            Some(vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        ),
        ("duplicate chunk", &[(1, 2, &a), (1, 2, &b)], None),
        ("missing chunk", &[(1, 2, &a)], None),
        ("count disagrees", &[(1, 2, &a), (2, 3, &b)], None),
        ("sequence 0", &[(0, 1, &a)], None),
        ("sequence past count", &[(3, 2, &a)], None),
        ("count 0", &[(1, 0, &a)], None),
        (
            "zero-length chunks",
            &[(1, 2, &[]), (2, 2, &[])],
            Some(Vec::new()),
        ),
        ("one empty chunk", &[(1, 1, &[])], Some(Vec::new())),
        ("surplus chunk", &[(1, 1, &a), (2, 1, &b)], None),
    ];
    for (name, chunks, want) in cases {
        let jpeg = with_icc_chunks(chunks);
        let m = read_metadata(&jpeg).unwrap();
        assert_eq!(&m.icc, want, "{name}");
        let img = oxideav_mjpeg::decode(&jpeg).unwrap();
        assert_eq!(&img.metadata.icc, want, "{name}");
        assert!(oxideav_mjpeg::info(&jpeg).unwrap().has_icc, "{name}");
        let strict = oxideav_mjpeg::decode_with(&jpeg, &DecodeOptions::new().with_strict(true));
        assert_eq!(strict.is_ok(), want.is_some(), "{name}: strict mode");
        // Re-embedding what was read (or nothing) is always well formed.
        let again = embed_metadata(&jpeg, &m, JfifPolicy::Auto).unwrap();
        assert_eq!(read_metadata(&again).unwrap().icc, m.icc, "{name}");
    }
}

#[test]
fn duplicate_exif_and_xmp_segments_keep_the_first() {
    let base = oxideav_mjpeg::encode(&gray(8, 8), &EncodeOptions::new()).unwrap();
    let mut jpeg = vec![0xFF, 0xD8];
    for (id, body) in [
        (b"Exif\0\0".to_vec(), b"first"),
        (b"Exif\0\0".to_vec(), b"other"),
    ] {
        let len = 2 + id.len() + body.len();
        jpeg.extend_from_slice(&[0xFF, 0xE1, 0, len as u8]);
        jpeg.extend_from_slice(&id);
        jpeg.extend_from_slice(body);
    }
    jpeg.extend_from_slice(&base[2..]);
    assert_eq!(
        read_metadata(&jpeg).unwrap().exif.as_deref(),
        Some(&b"first"[..])
    );
    // Embedding a new Exif block replaces both.
    let out = embed_metadata(&jpeg, &Metadata::new().with_exif(exif()), JfifPolicy::Auto).unwrap();
    assert_eq!(leading(&out).iter().filter(|s| s.0 == 0xE1).count(), 1);
    assert_eq!(read_metadata(&out).unwrap().exif, Some(exif()));
    // Embedding nothing keeps both, first still wins.
    let kept = embed_metadata(&jpeg, &Metadata::new(), JfifPolicy::Auto).unwrap();
    assert_eq!(leading(&kept).iter().filter(|s| s.0 == 0xE1).count(), 2);
    assert_eq!(
        read_metadata(&kept).unwrap().exif.as_deref(),
        Some(&b"first"[..])
    );
}

#[test]
fn overrides_compose_per_field() {
    let base = Metadata::new().with_icc(vec![1]).with_xmp(b"a".to_vec());
    let over = Metadata::new().with_xmp(b"b".to_vec()).with_exif(exif());
    let m = base.with_overrides(&over);
    assert_eq!(m.icc, Some(vec![1]));
    assert_eq!(m.xmp, Some(b"b".to_vec()));
    assert_eq!(m.exif, Some(exif()));
    assert_eq!(base.with_overrides(&Metadata::new()), base);
    // The option layer of `encode` is the same rule.
    let img = gray(8, 8).with_metadata(base.clone());
    let bytes =
        oxideav_mjpeg::encode(&img, &EncodeOptions::new().with_metadata(over.clone())).unwrap();
    assert_eq!(read_metadata(&bytes).unwrap(), m);
}

#[test]
fn read_metadata_needs_a_frame_header() {
    assert!(read_metadata(&[]).is_err());
    assert!(read_metadata(&[0xFF, 0xD8]).is_err());
    assert!(read_metadata(&[0xFF, 0xD8, 0xFF, 0xD9]).is_err());
    let mut hdr = vec![0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x08];
    hdr.extend_from_slice(b"Exif\0\0");
    assert!(read_metadata(&hdr).is_err(), "no SOF");
}
