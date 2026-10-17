//! Metadata marker segments — the one place that knows how ICC, Exif
//! and XMP travel in a JPEG stream and in which order the segments go.
//!
//! Every writer in this crate (the general T.81 writer, the historical
//! per-layout encoders behind the registry, the registry encoder's
//! blob carriage) finishes its stream with [`embed_metadata`], and
//! every reader (the root `info` / `decode`, the `jpeg` demuxer, the
//! registry decoder) reads the same segments through the header scan
//! behind [`read_metadata`] — so the standalone [`Metadata`] record and
//! the framework blobs are two views of one set of bytes.
//!
//! # Carriage (payload after the identifier)
//!
//! | Blob | Segment | Identifier | Payload | Limit |
//! |---|---|---|---|---|
//! | `icc` | `APP2`, one segment per chunk | `"ICC_PROFILE\0"`, chunk number (from 1), chunk count | the profile bytes, split in order | [`ICC_CHUNK_DATA`] (65 519) bytes per chunk, [`ICC_MAX_CHUNKS`] (255) chunks, so [`ICC_MAX_BYTES`] (16 707 345) bytes — ICC.1 Annex B.4 |
//! | `exif` | `APP1` | `"Exif\0\0"` | the TIFF-structured attribute block from its byte-order mark | [`EXIF_MAX_BYTES`] (65 527) — one segment, Exif §4.7.2 |
//! | `xmp` | `APP1` | `"http://ns.adobe.com/xap/1.0/\0"` | the XMP packet (UTF-8 RDF/XML) | [`XMP_MAX_BYTES`] (65 502) — the standard packet; no extended-XMP continuation is written |
//!
//! JFIF (`APP0 "JFIF\0"`) and the Adobe `APP14` segment are not
//! metadata: they signal the colour model and the decoder interprets
//! them (see [`crate::t81::ColorSignalling`]).
//!
//! # Order (T.81 §B.2.4.6 lets APPn segments sit anywhere before the
//! frame header; the two carriage specs each claim the first slot)
//!
//! * T.871 §6.3: the JFIF `APP0` *shall immediately follow* `SOI`.
//! * Exif §4.5.4 / §4.7.2.1: the Exif `APP1` *shall be recorded
//!   immediately after* `SOI`; `APPn` other than `APP1` / `APP2` /
//!   `APP11` are not used by Exif, though §4.5.5 Table 2 lets a writer
//!   record further `APPn` (`n` from 0) after the Exif segments.
//!
//! A stream cannot satisfy both, so [`JfifPolicy`] decides:
//!
//! | Policy | Without Exif | With Exif |
//! |---|---|---|
//! | `Auto` (default) | `SOI, APP0 JFIF, …` — a JFIF file | `SOI, APP1 Exif, …` — an Exif file; no `APP0` is written (T.871's slot is taken, and Exif §4.5.4 does not use `APP0`) |
//! | `Always` | as `Auto` | `SOI, APP1 Exif, APP0 JFIF, …` — Exif-conformant placement; the JFIF segment is where Exif Table 2 row 6 allows other `APPn`, not where T.871 wants it |
//! | `Never` | no `APP0` | `SOI, APP1 Exif, …` |
//!
//! After the slot-0 segment: `APP1` XMP, the `APP2` ICC chunks in
//! sequence order, then every other leading segment the stream already
//! had (`APP14` Adobe, `COM`, foreign `APPn`) in its original order,
//! then the tables and the frame. The decoder of this crate reads the
//! colour-model segments wherever they sit before the first `SOS`, so
//! every policy decodes to the same picture; the policy only changes
//! which outside reader recognises the file by its first segment.
//!
//! Everything here builds without `oxideav-core`.

use crate::error::{MjpegError as Error, Result};
use crate::image::Metadata;
use crate::jpeg::markers;

/// `APP1` Exif identifier.
pub const EXIF_IDENTIFIER: &[u8] = b"Exif\0\0";
/// `APP1` XMP identifier (`"http://ns.adobe.com/xap/1.0/\0"`).
pub const XMP_IDENTIFIER: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
/// `APP2` ICC identifier (`"ICC_PROFILE\0"`).
pub const ICC_IDENTIFIER: &[u8] = b"ICC_PROFILE\0";
/// `APP0` JFIF identifier.
const JFIF_IDENTIFIER: &[u8] = b"JFIF\0";
/// `APP0` JFIF-extension identifier (T.871 §6.4: must immediately
/// follow the JFIF segment, so it goes wherever that one goes).
const JFXX_IDENTIFIER: &[u8] = b"JFXX\0";

/// Largest APPn payload after the two length bytes (T.81 §B.1.1.4:
/// `Lp ≤ 65535`, the length field counted).
pub const MAX_APP_PAYLOAD: usize = 65533;
/// `APP2 "ICC_PROFILE\0"` chunk header: 12-byte identifier + chunk
/// number + chunk count (ICC.1 Annex B.4).
const ICC_CHUNK_HEADER: usize = 14;
/// Profile bytes per `APP2` chunk.
pub const ICC_CHUNK_DATA: usize = MAX_APP_PAYLOAD - ICC_CHUNK_HEADER;
/// Chunks an ICC profile may span (one-byte chunk count, ICC.1 Annex
/// B.4).
pub const ICC_MAX_CHUNKS: usize = 255;
/// Largest embeddable ICC profile: `255 × 65 519` = 16 707 345 bytes
/// (the figure ICC.1 Annex B.4 states).
pub const ICC_MAX_BYTES: usize = ICC_MAX_CHUNKS * ICC_CHUNK_DATA;
/// Largest Exif block: one `APP1` segment minus the six-byte identifier
/// (Exif §4.7.2: the segment, identifier included, is bounded by the
/// T.81 segment length).
pub const EXIF_MAX_BYTES: usize = MAX_APP_PAYLOAD - EXIF_IDENTIFIER.len();
/// Largest XMP packet written: the standard-packet bound of 65 502
/// bytes (a larger packet needs the extended-XMP continuation
/// segments, which this crate neither writes nor reassembles; the
/// bound is two bytes under what the segment length alone allows).
pub const XMP_MAX_BYTES: usize = 65502;

/// How a JFIF `APP0` and an Exif `APP1` share the first slot after
/// `SOI` (see the [module docs](self)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum JfifPolicy {
    /// JFIF first when there is no Exif block; an Exif block takes the
    /// slot and the JFIF segment is dropped.
    #[default]
    Auto,
    /// Keep the JFIF segment even with Exif, placed right after the
    /// Exif `APP1`.
    Always,
    /// Never write (or keep) a JFIF segment.
    Never,
}

impl JfifPolicy {
    /// Parse the registry option spelling (`auto` / `always` / `never`;
    /// empty = `Auto`).
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "" | "auto" => JfifPolicy::Auto,
            "always" => JfifPolicy::Always,
            "never" => JfifPolicy::Never,
            other => {
                return Err(Error::invalid(format!(
                    "JPEG: jfif policy '{other}' (auto / always / never)"
                )))
            }
        })
    }

    /// The registry option spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            JfifPolicy::Auto => "auto",
            JfifPolicy::Always => "always",
            JfifPolicy::Never => "never",
        }
    }
}

/// What a leading marker segment carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Exif,
    Xmp,
    Icc,
    Jfif,
    /// `APP0 "JFXX\0"` — travels with the JFIF segment.
    Jfxx,
    Other,
}

/// One leading segment (`FF marker Lp payload`), classified.
struct Segment<'a> {
    kind: Kind,
    bytes: &'a [u8],
}

fn classify(marker: u8, payload: &[u8]) -> Kind {
    let starts = |id: &[u8]| payload.len() >= id.len() && &payload[..id.len()] == id;
    match marker {
        markers::APP0 if starts(JFIF_IDENTIFIER) => Kind::Jfif,
        markers::APP0 if starts(JFXX_IDENTIFIER) => Kind::Jfxx,
        markers::APP1 if starts(EXIF_IDENTIFIER) => Kind::Exif,
        markers::APP1 if starts(XMP_IDENTIFIER) => Kind::Xmp,
        markers::APP2 if starts(ICC_IDENTIFIER) => Kind::Icc,
        _ => Kind::Other,
    }
}

/// Split `jpeg` into its leading `APPn` / `COM` segments and the rest
/// (from the first marker that is neither). `SOI` must be at offset 0;
/// a truncated segment length is an error.
fn split_leading(jpeg: &[u8]) -> Result<(Vec<Segment<'_>>, &[u8])> {
    if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != markers::SOI {
        return Err(Error::invalid("JPEG: missing SOI"));
    }
    let mut segs = Vec::new();
    let mut pos = 2;
    loop {
        // T.81 §B.1.1.2: any number of `FF` fill bytes may precede a
        // marker; they belong to no segment and are not carried over.
        let start = pos;
        while pos < jpeg.len() && jpeg[pos] == 0xFF {
            pos += 1;
        }
        if pos == start || pos >= jpeg.len() {
            // Not at a marker (stray byte, or the stream ends here):
            // everything from here on is "the rest" and is left to
            // the decoder's lenient walker.
            return Ok((segs, &jpeg[start..]));
        }
        let marker = jpeg[pos];
        if !(markers::is_app(marker) || marker == markers::COM) {
            return Ok((segs, &jpeg[start..]));
        }
        let body = pos + 1;
        if body + 2 > jpeg.len() {
            return Err(Error::invalid("JPEG: marker segment: truncated length"));
        }
        let len = u16::from_be_bytes([jpeg[body], jpeg[body + 1]]) as usize;
        if len < 2 {
            return Err(Error::invalid("JPEG: marker segment: length < 2"));
        }
        if body + len > jpeg.len() {
            return Err(Error::invalid("JPEG: marker segment: payload truncated"));
        }
        let payload = &jpeg[body + 2..body + len];
        segs.push(Segment {
            kind: classify(marker, payload),
            bytes: &jpeg[pos - 1..body + len],
        });
        pos = body + len;
    }
}

/// Append one APPn segment (`0xFF marker, Lp, payload`).
fn write_app_segment(out: &mut Vec<u8>, marker: u8, parts: &[&[u8]]) -> Result<()> {
    let len: usize = parts.iter().map(|p| p.len()).sum();
    if len > MAX_APP_PAYLOAD {
        return Err(Error::unsupported(format!(
            "JPEG encode: APP{} payload of {len} bytes exceeds the {MAX_APP_PAYLOAD}-byte segment limit",
            marker - markers::APP0
        )));
    }
    out.extend_from_slice(&[0xFF, marker, ((len + 2) >> 8) as u8, (len + 2) as u8]);
    for p in parts {
        out.extend_from_slice(p);
    }
    Ok(())
}

/// Serialise the `APP1 "Exif\0\0"` segment.
fn write_exif(out: &mut Vec<u8>, exif: &[u8]) -> Result<()> {
    if exif.len() > EXIF_MAX_BYTES {
        return Err(Error::unsupported(format!(
            "JPEG encode: Exif block of {} bytes exceeds the single APP1 segment ({EXIF_MAX_BYTES} bytes, Exif §4.7.2)",
            exif.len()
        )));
    }
    write_app_segment(out, markers::APP1, &[EXIF_IDENTIFIER, exif])
}

/// Serialise the `APP1` XMP standard packet segment.
fn write_xmp(out: &mut Vec<u8>, xmp: &[u8]) -> Result<()> {
    if xmp.len() > XMP_MAX_BYTES {
        return Err(Error::unsupported(format!(
            "JPEG encode: XMP packet of {} bytes exceeds the standard-packet APP1 bound ({XMP_MAX_BYTES} bytes); extended-XMP continuation segments are not written",
            xmp.len()
        )));
    }
    write_app_segment(out, markers::APP1, &[XMP_IDENTIFIER, xmp])
}

/// Serialise the `APP2 "ICC_PROFILE\0"` chunk sequence: chunks of at
/// most [`ICC_CHUNK_DATA`] profile bytes, numbered from 1, every chunk
/// carrying the total count (ICC.1 Annex B.4). An empty profile is
/// one chunk with no profile bytes.
fn write_icc(out: &mut Vec<u8>, icc: &[u8]) -> Result<()> {
    let chunks = icc.len().div_ceil(ICC_CHUNK_DATA).max(1);
    if chunks > ICC_MAX_CHUNKS {
        return Err(Error::unsupported(format!(
            "JPEG encode: ICC profile of {} bytes needs {chunks} APP2 chunks ({ICC_MAX_CHUNKS} maximum, {ICC_MAX_BYTES} bytes — ICC.1 Annex B.4)",
            icc.len()
        )));
    }
    let total = chunks as u8;
    if icc.is_empty() {
        return write_app_segment(out, markers::APP2, &[ICC_IDENTIFIER, &[1, total]]);
    }
    for (i, chunk) in icc.chunks(ICC_CHUNK_DATA).enumerate() {
        let seq = (i + 1) as u8;
        write_app_segment(out, markers::APP2, &[ICC_IDENTIFIER, &[seq, total], chunk])?;
    }
    Ok(())
}

/// Rewrite the leading marker segments of `jpeg` so that the blobs set
/// in `meta` are embedded (replacing any segment of the same kind the
/// stream already carried; kinds `meta` leaves `None` keep what the
/// stream had) in the order the [module docs](self) give, with the
/// JFIF `APP0` placed by `jfif`. Tables, frame and scans are copied
/// untouched, so the picture is unchanged.
///
/// Errors: no `SOI` at offset 0 or a truncated leading segment
/// (`InvalidData`); a blob over its carriage limit (`Unsupported`,
/// naming the limit: [`EXIF_MAX_BYTES`], [`XMP_MAX_BYTES`],
/// [`ICC_MAX_CHUNKS`]).
pub fn embed_metadata(jpeg: &[u8], meta: &Metadata, jfif: JfifPolicy) -> Result<Vec<u8>> {
    let (segs, rest) = split_leading(jpeg)?;
    let has_exif = meta.exif.is_some() || segs.iter().any(|s| s.kind == Kind::Exif);
    let keep_jfif = match jfif {
        JfifPolicy::Auto => !has_exif,
        JfifPolicy::Always => true,
        JfifPolicy::Never => false,
    };

    let mut out = Vec::with_capacity(jpeg.len() + meta_len(meta));
    out.extend_from_slice(&[0xFF, markers::SOI]);
    let existing = |kind: Kind| segs.iter().filter(move |s| s.kind == kind);

    // Slot 0: Exif (Exif §4.7.2.1), else JFIF (T.871 §6.3).
    match &meta.exif {
        Some(exif) => write_exif(&mut out, exif)?,
        None => existing(Kind::Exif).for_each(|s| out.extend_from_slice(s.bytes)),
    }
    if keep_jfif {
        for s in existing(Kind::Jfif).chain(existing(Kind::Jfxx)) {
            out.extend_from_slice(s.bytes);
        }
    }
    match &meta.xmp {
        Some(xmp) => write_xmp(&mut out, xmp)?,
        None => existing(Kind::Xmp).for_each(|s| out.extend_from_slice(s.bytes)),
    }
    match &meta.icc {
        Some(icc) => write_icc(&mut out, icc)?,
        None => existing(Kind::Icc).for_each(|s| out.extend_from_slice(s.bytes)),
    }
    for s in existing(Kind::Other) {
        out.extend_from_slice(s.bytes);
    }
    out.extend_from_slice(rest);
    Ok(out)
}

/// Bytes the serialised form of `meta` adds, for the pre-allocation.
fn meta_len(meta: &Metadata) -> usize {
    let seg = |b: &Option<Vec<u8>>, id: usize| b.as_ref().map_or(0, |b| b.len() + id + 4);
    seg(&meta.exif, EXIF_IDENTIFIER.len())
        + seg(&meta.xmp, XMP_IDENTIFIER.len())
        + meta.icc.as_ref().map_or(0, |icc| {
            icc.len() + icc.len().div_ceil(ICC_CHUNK_DATA).max(1) * (ICC_CHUNK_HEADER + 4)
        })
}

/// Remove every ICC / Exif / XMP segment from the leading marker
/// segments of `jpeg` (colour-model segments and the picture are
/// kept). The inverse of [`embed_metadata`] for the three carried
/// kinds.
pub fn strip_metadata(jpeg: &[u8]) -> Result<Vec<u8>> {
    let (segs, rest) = split_leading(jpeg)?;
    let mut out = Vec::with_capacity(jpeg.len());
    out.extend_from_slice(&[0xFF, markers::SOI]);
    for s in segs
        .iter()
        .filter(|s| !matches!(s.kind, Kind::Exif | Kind::Xmp | Kind::Icc))
    {
        out.extend_from_slice(s.bytes);
    }
    out.extend_from_slice(rest);
    Ok(out)
}

/// The metadata the decoder reports for `jpeg`
/// ([`JpegImage::metadata`](crate::JpegImage::metadata)), from the
/// marker segments before the first `SOS` and nothing else: the ICC
/// chunk sequence reassembled in order (`None` when a chunk is
/// missing, repeated or disagrees on the count), the first Exif and
/// the first XMP standard packet. Lenient like `decode`; a stream
/// without a frame header is an error.
pub fn read_metadata(jpeg: &[u8]) -> Result<Metadata> {
    crate::api::header_metadata(jpeg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SOI, APP0 JFIF, APP14 Adobe, DQT-ish, EOI` — a header skeleton
    /// (never decoded here).
    fn skeleton() -> Vec<u8> {
        let mut v = vec![0xFF, markers::SOI];
        write_app_segment(
            &mut v,
            markers::APP0,
            &[b"JFIF\0", &[1, 1, 0, 0, 1, 0, 1, 0, 0]],
        )
        .unwrap();
        write_app_segment(
            &mut v,
            markers::APP14,
            &[b"Adobe", &[0, 100, 0, 0, 0, 0, 1]],
        )
        .unwrap();
        v.extend_from_slice(&[0xFF, markers::DQT, 0, 3, 0]);
        v.extend_from_slice(&[0xFF, markers::EOI]);
        v
    }

    fn markers_of(jpeg: &[u8]) -> Vec<u8> {
        let (segs, _) = split_leading(jpeg).unwrap();
        segs.iter().map(|s| s.bytes[1]).collect()
    }

    #[test]
    fn jfif_policy_round_trips_its_spelling() {
        for p in [JfifPolicy::Auto, JfifPolicy::Always, JfifPolicy::Never] {
            assert_eq!(JfifPolicy::parse(p.as_str()).unwrap(), p);
        }
        assert_eq!(JfifPolicy::parse("").unwrap(), JfifPolicy::Auto);
        assert!(JfifPolicy::parse("sometimes").is_err());
    }

    #[test]
    fn order_without_exif_keeps_jfif_first() {
        let meta = Metadata::new()
            .with_xmp(b"<x/>".to_vec())
            .with_icc(vec![7; 10]);
        let out = embed_metadata(&skeleton(), &meta, JfifPolicy::Auto).unwrap();
        assert_eq!(
            markers_of(&out),
            [markers::APP0, markers::APP1, markers::APP2, markers::APP14]
        );
        assert!(out.ends_with(&[0xFF, markers::DQT, 0, 3, 0, 0xFF, markers::EOI]));
    }

    #[test]
    fn exif_takes_slot_zero_and_jfif_follows_the_policy() {
        let meta = Metadata::new()
            .with_exif(b"II*\0\x08\0\0\0".to_vec())
            .with_xmp(b"<x/>".to_vec())
            .with_icc(vec![7; 10]);
        let auto = embed_metadata(&skeleton(), &meta, JfifPolicy::Auto).unwrap();
        assert_eq!(
            markers_of(&auto),
            [markers::APP1, markers::APP1, markers::APP2, markers::APP14]
        );
        assert_eq!(&auto[6..12], EXIF_IDENTIFIER);
        let always = embed_metadata(&skeleton(), &meta, JfifPolicy::Always).unwrap();
        assert_eq!(
            markers_of(&always),
            [
                markers::APP1,
                markers::APP0,
                markers::APP1,
                markers::APP2,
                markers::APP14
            ]
        );
        let never = embed_metadata(&skeleton(), &Metadata::new(), JfifPolicy::Never).unwrap();
        assert_eq!(markers_of(&never), [markers::APP14]);
    }

    #[test]
    fn embedding_replaces_same_kind_and_keeps_the_others() {
        let first = embed_metadata(
            &skeleton(),
            &Metadata::new().with_icc(vec![1; 5]).with_xmp(b"a".to_vec()),
            JfifPolicy::Auto,
        )
        .unwrap();
        let second = embed_metadata(
            &first,
            &Metadata::new().with_icc(vec![2; 6]),
            JfifPolicy::Auto,
        )
        .unwrap();
        let m = read_metadata_lenient(&second);
        assert_eq!(m.icc.as_deref(), Some(&[2u8; 6][..]));
        assert_eq!(m.xmp.as_deref(), Some(&b"a"[..]));
        let stripped = strip_metadata(&second).unwrap();
        assert_eq!(markers_of(&stripped), [markers::APP0, markers::APP14]);
    }

    /// The header scan needs a frame header; the skeleton has none, so
    /// re-read through the splitter instead.
    fn read_metadata_lenient(jpeg: &[u8]) -> Metadata {
        let (segs, _) = split_leading(jpeg).unwrap();
        let mut m = Metadata::new();
        let mut icc = Vec::new();
        for s in &segs {
            let payload = &s.bytes[4..];
            match s.kind {
                Kind::Exif => m.exif = Some(payload[EXIF_IDENTIFIER.len()..].to_vec()),
                Kind::Xmp => m.xmp = Some(payload[XMP_IDENTIFIER.len()..].to_vec()),
                Kind::Icc => icc.extend_from_slice(&payload[ICC_CHUNK_HEADER..]),
                _ => {}
            }
        }
        if segs.iter().any(|s| s.kind == Kind::Icc) {
            m.icc = Some(icc);
        }
        m
    }

    #[test]
    fn icc_chunking_at_the_limits() {
        let two = vec![9u8; ICC_CHUNK_DATA + 1];
        let out = embed_metadata(
            &skeleton(),
            &Metadata::new().with_icc(two.clone()),
            JfifPolicy::Auto,
        )
        .unwrap();
        let (segs, _) = split_leading(&out).unwrap();
        let icc: Vec<_> = segs.iter().filter(|s| s.kind == Kind::Icc).collect();
        assert_eq!(icc.len(), 2);
        assert_eq!(&icc[0].bytes[16..18], &[1, 2]);
        assert_eq!(&icc[1].bytes[16..18], &[2, 2]);
        assert_eq!(icc[1].bytes.len(), 4 + ICC_CHUNK_HEADER + 1);
        assert_eq!(read_metadata_lenient(&out).icc, Some(two));

        let empty = embed_metadata(
            &skeleton(),
            &Metadata::new().with_icc(Vec::new()),
            JfifPolicy::Auto,
        )
        .unwrap();
        assert_eq!(read_metadata_lenient(&empty).icc, Some(Vec::new()));

        let too_big = vec![0u8; ICC_MAX_BYTES + 1];
        let err = embed_metadata(
            &skeleton(),
            &Metadata::new().with_icc(too_big),
            JfifPolicy::Auto,
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(ref s) if s.contains("255")),
            "{err}"
        );
    }

    #[test]
    fn exif_and_xmp_limits_are_typed_refusals() {
        let exif = vec![0u8; EXIF_MAX_BYTES];
        assert!(embed_metadata(
            &skeleton(),
            &Metadata::new().with_exif(exif),
            JfifPolicy::Auto
        )
        .is_ok());
        let err = embed_metadata(
            &skeleton(),
            &Metadata::new().with_exif(vec![0u8; EXIF_MAX_BYTES + 1]),
            JfifPolicy::Auto,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err}");
        let xmp = vec![b' '; XMP_MAX_BYTES];
        assert!(embed_metadata(
            &skeleton(),
            &Metadata::new().with_xmp(xmp),
            JfifPolicy::Auto
        )
        .is_ok());
        let err = embed_metadata(
            &skeleton(),
            &Metadata::new().with_xmp(vec![b' '; XMP_MAX_BYTES + 1]),
            JfifPolicy::Auto,
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(ref s) if s.contains("65502")),
            "{err}"
        );
    }

    #[test]
    fn hostile_leading_segments_are_errors_not_panics() {
        let meta = Metadata::new().with_xmp(b"x".to_vec());
        assert!(embed_metadata(&[], &meta, JfifPolicy::Auto).is_err());
        assert!(embed_metadata(&[0xFF, 0xD8, 0xFF, 0xE1], &meta, JfifPolicy::Auto).is_err());
        assert!(embed_metadata(
            &[0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x01],
            &meta,
            JfifPolicy::Auto
        )
        .is_err());
        assert!(embed_metadata(
            &[0xFF, 0xD8, 0xFF, 0xE1, 0xFF, 0xFF],
            &meta,
            JfifPolicy::Auto
        )
        .is_err());
        // A bare SOI, fill bytes, or a non-APP marker right away: the
        // rest is copied as-is.
        let out = embed_metadata(&[0xFF, 0xD8], &meta, JfifPolicy::Auto).unwrap();
        assert_eq!(markers_of(&out), [markers::APP1]);
        let out = embed_metadata(
            &[0xFF, 0xD8, 0xFF, 0xFF, 0xFF, 0xD9],
            &meta,
            JfifPolicy::Auto,
        )
        .unwrap();
        assert!(out.ends_with(&[0xFF, 0xFF, 0xFF, 0xD9]));
        let out = embed_metadata(&[0xFF, 0xD8, 0x00, 0x12], &meta, JfifPolicy::Auto).unwrap();
        assert!(out.ends_with(&[0x00, 0x12]));
    }
}
