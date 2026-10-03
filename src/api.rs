//! The root still-image vocabulary: `probe` / `info` / `decode*` /
//! `encode*` — the workspace image-crate API contract, implemented over
//! the same decoder and T.81 writer the Motion-JPEG video path uses.
//!
//! Every function here builds and runs with `default-features = false`
//! (no `oxideav-core`). The `registry` adapter calls these same
//! functions, so there is one decoder and one writer.

use std::io::{Read, Write};

use crate::error::{MjpegError as Error, Result};
use crate::image::{
    ColorInfo, DecodeOptions, ImageInfo, JpegImage, Metadata, MjpegPixelFormat as PixelFormat,
    Plane, RgbImage, RgbaImage,
};
use crate::jpeg::inspect::{ChromaSubsampling, SofKind};
use crate::jpeg::markers;
use crate::jpeg::parser::{parse_sof, MarkerWalker, SofInfo};
use crate::t81::{ColorSignalling, EncodeOptions, JpegProcess, ICC_IDENTIFIER, XMP_IDENTIFIER};

// ---------------------------------------------------------------------------
// probe / info
// ---------------------------------------------------------------------------

/// Signature sniff: `true` when `bytes` starts with `SOI` (`FF D8`)
/// followed by another marker prefix (`FF` and a marker byte that is
/// neither a stuffed zero nor a fill byte). No allocation, never
/// panics, `false` on short input.
pub fn probe(bytes: &[u8]) -> bool {
    bytes.len() >= 4
        && bytes[0] == 0xFF
        && bytes[1] == markers::SOI
        && bytes[2] == 0xFF
        && bytes[3] != 0x00
        && bytes[3] != 0xFF
}

/// Header-only inspection: dimensions, the layout [`decode`] will
/// produce, sample precision, the coding process family, the colour
/// description and which metadata segments are present. Walks the
/// marker segments up to the first `SOS` (and, for a `Y = 0` frame
/// header, to the `DNL` that follows the first scan); the entropy-coded
/// data is never decoded.
pub fn info(bytes: &[u8]) -> Result<ImageInfo> {
    let hdr = scan_header(bytes, false)?;
    let shape = infer_shape(&hdr)?;
    if hdr.no_scan {
        return Err(Error::invalid("JPEG: stream ends before the first SOS"));
    }
    Ok(ImageInfo {
        width: hdr.frame.width as u32,
        height: hdr.height,
        format: shape.format,
        frames: 1,
        has_alpha: false,
        color: shape.color,
        has_icc: hdr.icc_seen,
        has_exif: hdr.exif.is_some(),
        has_xmp: hdr.xmp.is_some(),
        precision: hdr.frame.precision,
        components: hdr.frame.components.len() as u8,
        progressive: matches!(hdr.kind, SofKind::Progressive | SofKind::ProgressiveArith),
        lossless: matches!(hdr.kind, SofKind::Lossless | SofKind::LosslessArith),
        arithmetic: hdr.kind.is_arithmetic(),
        hierarchical: hdr.hierarchical,
        has_jfif: hdr.jfif,
        has_adobe: hdr.adobe.is_some(),
    })
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

/// Decode one JPEG interchange stream into its native layout with the
/// default [`DecodeOptions`]. Colour and metadata are filled from the
/// JFIF / Adobe / ICC / Exif / XMP segments.
pub fn decode(bytes: &[u8]) -> Result<JpegImage> {
    decode_with(bytes, &DecodeOptions::default())
}

/// [`decode`] with explicit limits, strictness and an optional §B.5
/// tables stream. Limits are checked against the header before any
/// sample buffer is allocated.
pub fn decode_with(bytes: &[u8], opts: &DecodeOptions) -> Result<JpegImage> {
    if bytes.len() > opts.max_bytes {
        return Err(Error::limit(format!(
            "JPEG: input of {} bytes exceeds max_bytes = {}",
            bytes.len(),
            opts.max_bytes
        )));
    }
    let hdr = scan_header(bytes, opts.strict)?;
    // Reject the layouts the decoder cannot shape before decoding.
    infer_shape(&hdr)?;
    let (w, h) = (hdr.frame.width as u32, hdr.height);
    if w > opts.max_width || h > opts.max_height {
        return Err(Error::limit(format!(
            "JPEG: {w}×{h} exceeds max_width × max_height = {}×{}",
            opts.max_width, opts.max_height
        )));
    }
    if (w as u64) * (h as u64) > opts.max_pixels {
        return Err(Error::limit(format!(
            "JPEG: {w}×{h} = {} pixels exceeds max_pixels = {}",
            (w as u64) * (h as u64),
            opts.max_pixels
        )));
    }
    if opts.strict {
        strict_structure_check(bytes)?;
    }
    let mut img = match &opts.tables {
        Some(t) => crate::decoder::decode_planes_with_tables(t, bytes)?,
        None => crate::decoder::decode_planes(bytes)?,
    };
    // Full-range labelling where JFIF says so (T.871 §7).
    if hdr.jfif {
        img.format = img.format.full_range_label();
    }
    img.color = color_for(img.format);
    img.metadata = hdr.metadata();
    Ok(img)
}

/// One-call raw path: decode and convert to tightly packed 8-bit RGB.
pub fn decode_rgb8(bytes: &[u8]) -> Result<RgbImage> {
    let img = decode(bytes)?;
    Ok(RgbImage::new(img.width, img.height, img.to_rgb8()))
}

/// One-call raw path: decode and convert to tightly packed 8-bit RGBA
/// (alpha opaque — JPEG has none).
pub fn decode_rgba8(bytes: &[u8]) -> Result<RgbaImage> {
    let img = decode(bytes)?;
    Ok(RgbaImage::new(img.width, img.height, img.to_rgba8()))
}

/// Read `r` to its end and [`decode`] the bytes.
pub fn decode_from<R: Read>(mut r: R) -> Result<JpegImage> {
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    decode(&bytes)
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

/// Write `image` as given — its own layout, precision and sampling —
/// as one JPEG interchange stream. Layouts the format cannot carry
/// under `opts.process` are refused with [`Error::Unsupported`]; nothing
/// is converted silently. The image's [`JpegImage::metadata`] is
/// embedded (merged with `opts.metadata`, the option winning per blob).
///
/// Layout → stream:
///
/// | Layout | Components | Signalling (`Auto`) | Processes |
/// |---|---|---|---|
/// | `Gray8` | 1 | JFIF | sequential / progressive / lossless |
/// | `Gray10Le` / `Gray16Le` | 1 | JFIF | lossless only (`P = 10` / `precision`) |
/// | `Gray12Le` | 1 | JFIF | sequential / progressive / lossless at `P = 12` |
/// | `Yuv4xxP`, `YuvJ4xxP` | 3, subsampled | JFIF | sequential / progressive / lossless (`P = 8`) |
/// | `Yuv4xxP12Le` | 3, subsampled | JFIF | sequential / progressive / lossless (`P = 12`) |
/// | `Rgb24` | 3 | Adobe APP14 `transform = 0`, ids `R G B` | sequential / progressive / lossless |
/// | `Rgb48Le` | 3 | Adobe RGB | lossless only (`P = precision`) |
/// | `Gbrp10Le` / `Gbrp12Le` / `Gbrp14Le` | 3 (G, B, R order kept, ids 1..3) | none | lossless only |
/// | `Cmyk` | 4 | none (plain CMYK) | sequential / progressive / lossless |
pub fn encode(image: &JpegImage, opts: &EncodeOptions) -> Result<Vec<u8>> {
    let prepared = prepare_planes(image, opts)?;
    let refs: Vec<&[u16]> = prepared.planes.iter().map(|p| p.as_slice()).collect();
    let out = prepared
        .opts
        .encode(image.width, image.height, &refs)
        .map(|e| e.data)?;
    Ok(out)
}

/// One-call raw path: tightly packed 8-bit RGB (`3 × width` bytes per
/// row) → JPEG. The RGB is converted to full-range YCbCr with the
/// T.871 §7 forward equations, the chroma is box-filtered down to
/// [`EncodeOptions::chroma`] (4:2:0 by default; T.871 §9 NOTE 1's
/// two-tap `(1/2, 1/2)` filter per axis) and a JFIF APP0 is written.
/// With [`ColorSignalling::Rgb`] selected the RGB is written as-is
/// (Adobe APP14 `transform = 0`, 4:4:4).
pub fn encode_rgb8(width: u32, height: u32, rgb: &[u8], opts: &EncodeOptions) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize) * 3;
    if rgb.len() < expected {
        return Err(Error::invalid(format!(
            "JPEG encode: {width}×{height} RGB needs {expected} bytes, got {}",
            rgb.len()
        )));
    }
    let img = rgb8_to_native(width, height, rgb, opts)?;
    encode(&img, opts)
}

/// One-call raw path: tightly packed 8-bit RGBA → JPEG. **The alpha
/// channel is dropped**: JPEG has no alpha mechanism, so the three
/// colour bytes of every pixel are encoded as by [`encode_rgb8`] and
/// the fourth is discarded. Callers that need alpha preserved must
/// choose another format.
pub fn encode_rgba8(width: u32, height: u32, rgba: &[u8], opts: &EncodeOptions) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize) * 4;
    if rgba.len() < expected {
        return Err(Error::invalid(format!(
            "JPEG encode: {width}×{height} RGBA needs {expected} bytes, got {}",
            rgba.len()
        )));
    }
    let rgb: Vec<u8> = rgba[..expected]
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    encode_rgb8(width, height, &rgb, opts)
}

/// [`encode`] into a writer.
pub fn encode_to<W: Write>(image: &JpegImage, opts: &EncodeOptions, mut w: W) -> Result<()> {
    let bytes = encode(image, opts)?;
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Header scan
// ---------------------------------------------------------------------------

/// What the marker segments before the first `SOS` say.
struct Header {
    /// The frame header the output geometry follows: the `DHP` for a
    /// hierarchical sequence, else the first `SOFn`.
    frame: SofInfo,
    /// `frame.height`, or the `DNL` line count when the header coded
    /// `Y = 0`.
    height: u32,
    /// Classification of the first `SOFn` marker.
    kind: SofKind,
    first_sof_marker: u8,
    hierarchical: bool,
    /// The stream ended (or hit `EOI`) after the frame header but
    /// before any `SOS`. The shape is still inferable — so an
    /// unsupported layout is reported as such, as the decoder would —
    /// but there is nothing to decode.
    no_scan: bool,
    jfif: bool,
    /// Adobe APP14 transform flag.
    adobe: Option<u8>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
    icc_seen: bool,
    /// Reassembled profile (`None` when absent or incomplete).
    icc: Option<Vec<u8>>,
}

impl Header {
    fn metadata(&self) -> Metadata {
        Metadata {
            icc: self.icc.clone(),
            exif: self.exif.clone(),
            xmp: self.xmp.clone(),
            gamma: None,
        }
    }
}

/// Walk the marker segments up to the first `SOS`. In strict mode a
/// malformed JFIF / Adobe / ICC segment is an error; otherwise it is
/// ignored (the stream still decodes).
fn scan_header(bytes: &[u8], strict: bool) -> Result<Header> {
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != markers::SOI {
        return Err(Error::invalid("JPEG: missing SOI"));
    }
    let mut walker = MarkerWalker::new(&bytes[2..]);
    let mut sof: Option<(SofInfo, u8)> = None;
    let mut dhp: Option<SofInfo> = None;
    let mut jfif = false;
    let mut adobe: Option<u8> = None;
    let mut exif: Option<Vec<u8>> = None;
    let mut xmp: Option<Vec<u8>> = None;
    let mut icc_total: Option<u8> = None;
    let mut icc_chunks: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut icc_seen = false;
    let mut icc_bad = false;
    let mut no_scan = false;

    loop {
        let Some(marker) = walker.next_marker()? else {
            if sof.is_none() {
                return Err(Error::invalid("JPEG: stream ends before the first SOS"));
            }
            no_scan = true;
            break;
        };
        match marker {
            markers::SOI | markers::TEM => continue,
            m if markers::is_rst(m) => continue,
            markers::EOI => {
                if sof.is_none() {
                    return Err(Error::invalid("JPEG: EOI before the first SOS"));
                }
                no_scan = true;
                break;
            }
            markers::SOS => break,
            markers::DHP => {
                let p = walker.read_segment_payload()?;
                if dhp.is_none() && sof.is_none() {
                    dhp = Some(parse_sof(p)?);
                }
            }
            m if markers::is_sof(m) => {
                if dhp.is_none() && matches!(m, 0xC5..=0xC7 | 0xCD..=0xCF) {
                    // A differential frame outside a hierarchical
                    // sequence — rejected before its header is read,
                    // as the decoder does.
                    return Err(Error::unsupported(format!(
                        "JPEG: differential frame SOF{} without a preceding DHP (a differential frame is only meaningful inside a hierarchical sequence)",
                        m - 0xC0
                    )));
                }
                let p = walker.read_segment_payload()?;
                if sof.is_none() {
                    sof = Some((parse_sof(p)?, m));
                }
            }
            markers::APP0 => {
                let p = walker.read_segment_payload()?;
                if p.len() >= 5 && &p[..5] == b"JFIF\0" {
                    jfif = true;
                    if strict {
                        crate::jpeg::inspect::parse_jfif_app0(p)?;
                    }
                }
            }
            markers::APP1 => {
                let p = walker.read_segment_payload()?;
                if p.len() >= 6 && &p[..6] == b"Exif\0\0" {
                    if exif.is_none() {
                        exif = Some(p[6..].to_vec());
                    }
                } else if p.len() >= XMP_IDENTIFIER.len()
                    && &p[..XMP_IDENTIFIER.len()] == XMP_IDENTIFIER
                    && xmp.is_none()
                {
                    xmp = Some(p[XMP_IDENTIFIER.len()..].to_vec());
                }
            }
            markers::APP2 => {
                let p = walker.read_segment_payload()?;
                if p.len() >= ICC_IDENTIFIER.len() && &p[..ICC_IDENTIFIER.len()] == ICC_IDENTIFIER {
                    icc_seen = true;
                    match crate::jpeg::inspect::parse_icc_profile_app2(p) {
                        Ok(chunk) => {
                            let total = *icc_total.get_or_insert(chunk.total);
                            if chunk.total != total || chunk.seq_no == 0 || chunk.seq_no > total {
                                icc_bad = true;
                            } else {
                                icc_chunks.push((chunk.seq_no, chunk.profile_bytes.to_vec()));
                            }
                        }
                        Err(e) => {
                            if strict {
                                return Err(e);
                            }
                            icc_bad = true;
                        }
                    }
                }
            }
            markers::APP14 => {
                let p = walker.read_segment_payload()?;
                if p.len() >= 12 && &p[..5] == b"Adobe" {
                    if adobe.is_none() {
                        adobe = Some(p[11]);
                    }
                    if strict {
                        crate::jpeg::inspect::parse_adobe_app14(p)?;
                    }
                }
            }
            _ => {
                let _ = walker.read_segment_payload()?;
            }
        }
    }

    let (first, first_sof_marker) = sof.ok_or_else(|| Error::invalid("JPEG: SOS before SOF"))?;
    let kind = SofKind::from_marker(first_sof_marker)
        .ok_or_else(|| Error::invalid("JPEG: SOF marker not classifiable"))?;
    let hierarchical = dhp.is_some();
    let frame = dhp.unwrap_or(first);
    let height = if frame.height == 0 && !no_scan {
        match crate::decoder::resolve_dnl_height(&bytes[2..])? {
            Some(nl) => nl as u32,
            None => return Err(Error::invalid("JPEG: SOF Y = 0 and no DNL segment")),
        }
    } else {
        frame.height as u32
    };

    // Reassemble the ICC profile in sequence order; every chunk 1..=N
    // exactly once, else the profile is unusable.
    let icc = if icc_seen && !icc_bad {
        let total = icc_total.unwrap_or(0) as usize;
        icc_chunks.sort_by_key(|(seq, _)| *seq);
        let complete = icc_chunks.len() == total
            && icc_chunks
                .iter()
                .enumerate()
                .all(|(i, (seq, _))| *seq as usize == i + 1);
        if complete {
            Some(icc_chunks.into_iter().flat_map(|(_, d)| d).collect())
        } else {
            if strict {
                return Err(Error::invalid(
                    "JPEG: APP2 ICC_PROFILE chunk sequence is incomplete",
                ));
            }
            None
        }
    } else {
        if icc_bad && strict {
            return Err(Error::invalid(
                "JPEG: APP2 ICC_PROFILE chunk sequence is inconsistent",
            ));
        }
        None
    };

    Ok(Header {
        frame,
        height,
        kind,
        first_sof_marker,
        hierarchical,
        no_scan,
        jfif,
        adobe,
        exif,
        xmp,
        icc_seen,
        icc,
    })
}

/// Strict mode: the stream must be `SOI`, segments and scans, `EOI`,
/// with nothing after the `EOI`.
fn strict_structure_check(bytes: &[u8]) -> Result<()> {
    let mut walker = MarkerWalker::new(&bytes[2..]);
    loop {
        let Some(marker) = walker.next_marker()? else {
            return Err(Error::invalid("JPEG (strict): no EOI"));
        };
        match marker {
            markers::SOI | markers::TEM => {}
            m if markers::is_rst(m) => {}
            markers::EOI => {
                if walker.pos != bytes.len() - 2 {
                    return Err(Error::invalid(format!(
                        "JPEG (strict): {} trailing byte(s) after EOI",
                        bytes.len() - 2 - walker.pos
                    )));
                }
                return Ok(());
            }
            markers::SOS => {
                let _ = walker.read_segment_payload()?;
                let _ = walker.read_scan_data()?;
            }
            _ => {
                let _ = walker.read_segment_payload()?;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Output-shape inference (mirrors the decoder's shaping rules)
// ---------------------------------------------------------------------------

struct Shape {
    format: PixelFormat,
    color: ColorInfo,
}

/// `true` when a three-component frame is RGB-coded: Adobe APP14
/// `transform = 0`, or component identifiers `'R' 'G' 'B'`.
fn rgb_class(frame: &SofInfo, adobe: Option<u8>) -> bool {
    frame.components.len() == 3
        && (adobe == Some(0)
            || (frame.components[0].id == b'R'
                && frame.components[1].id == b'G'
                && frame.components[2].id == b'B'))
}

/// The planar layout of a three-component YCbCr frame (T.81 §A.1.1):
/// a native subsampled format when chroma is `1×1` under a
/// maximal-factor luma, `4:4:4` for every other legal combination.
fn yuv_layout(frame: &SofInfo, twelve_bit: bool) -> PixelFormat {
    let y = frame.components[0];
    let cb = frame.components[1];
    let cr = frame.components[2];
    let h_max = frame
        .components
        .iter()
        .map(|c| c.h_factor)
        .max()
        .unwrap_or(1);
    let v_max = frame
        .components
        .iter()
        .map(|c| c.v_factor)
        .max()
        .unwrap_or(1);
    let chroma_unit = cb.h_factor == 1 && cb.v_factor == 1 && cr.h_factor == 1 && cr.v_factor == 1;
    let luma_max = y.h_factor == h_max && y.v_factor == v_max;
    if chroma_unit && luma_max {
        match (y.h_factor, y.v_factor, twelve_bit) {
            (1, 1, false) => return PixelFormat::Yuv444P,
            (2, 1, false) => return PixelFormat::Yuv422P,
            (2, 2, false) => return PixelFormat::Yuv420P,
            (4, 1, false) => return PixelFormat::Yuv411P,
            (1, 1, true) => return PixelFormat::Yuv444P12Le,
            (2, 1, true) => return PixelFormat::Yuv422P12Le,
            (2, 2, true) => return PixelFormat::Yuv420P12Le,
            _ => {}
        }
    }
    if twelve_bit {
        PixelFormat::Yuv444P12Le
    } else {
        PixelFormat::Yuv444P
    }
}

/// Grayscale carrier for precision `p`.
fn gray_format(p: u8) -> PixelFormat {
    match p {
        8 => PixelFormat::Gray8,
        10 => PixelFormat::Gray10Le,
        12 => PixelFormat::Gray12Le,
        _ => PixelFormat::Gray16Le,
    }
}

/// Three-component lossless carrier for precision `p` (RGB-class
/// shaping: packed at 8 bits, planar GBR at 10 / 12 / 14, `Rgb48Le`
/// elsewhere).
fn rgb_lossless_format(p: u8) -> PixelFormat {
    match p {
        8 => PixelFormat::Rgb24,
        10 => PixelFormat::Gbrp10Le,
        12 => PixelFormat::Gbrp12Le,
        14 => PixelFormat::Gbrp14Le,
        _ => PixelFormat::Rgb48Le,
    }
}

/// The layout the decoder produces for `hdr`, or the `Unsupported`
/// error it would raise. Mirrors `decoder.rs`'s shaping decisions
/// (`decode_scan`, `render_from_coefs*`, `shape_lossless_*`,
/// `shape_hierarchical_frame`); the `info == decode` test over the
/// fixture corpus pins the agreement.
fn infer_shape(hdr: &Header) -> Result<Shape> {
    let frame = &hdr.frame;
    let nc = frame.components.len();
    let p = frame.precision;
    if nc == 0 {
        return Err(Error::invalid("SOF: no components"));
    }
    let lossless = matches!(hdr.kind, SofKind::Lossless | SofKind::LosslessArith);
    if !hdr.hierarchical
        && matches!(
            hdr.kind,
            SofKind::HierarchicalDct | SofKind::HierarchicalArith
        )
    {
        return Err(Error::unsupported(format!(
            "JPEG: differential frame SOF{} without a preceding DHP (a differential frame is only meaningful inside a hierarchical sequence)",
            hdr.first_sof_marker - 0xC0
        )));
    }
    let h_max = frame
        .components
        .iter()
        .map(|c| c.h_factor)
        .max()
        .unwrap_or(1);
    let v_max = frame
        .components
        .iter()
        .map(|c| c.v_factor)
        .max()
        .unwrap_or(1);
    let subsampled = nc > 1 && (h_max != 1 || v_max != 1);
    let is_rgb = rgb_class(frame, hdr.adobe);

    let format = if hdr.hierarchical {
        // Annex J: geometry from the DHP; the first frame's process
        // class decides between YCbCr 4:4:4 (DCT, non-RGB) and the
        // lossless shaping rules.
        let first_is_dct = !matches!(hdr.kind, SofKind::Lossless | SofKind::LosslessArith);
        match nc {
            1 => gray_format(p),
            3 if first_is_dct && !is_rgb => {
                if p == 12 {
                    PixelFormat::Yuv444P12Le
                } else {
                    PixelFormat::Yuv444P
                }
            }
            3 => rgb_lossless_format(p),
            4 => PixelFormat::Cmyk,
            _ => return Err(Error::unsupported(format!("{nc}-component JPEG"))),
        }
    } else if lossless {
        if subsampled {
            yuv_layout(frame, false)
        } else {
            match nc {
                1 => gray_format(p),
                3 => rgb_lossless_format(p),
                4 => PixelFormat::Cmyk,
                _ => return Err(Error::unsupported(format!("{nc}-component JPEG"))),
            }
        }
    } else {
        match (nc, p) {
            (1, 8) => PixelFormat::Gray8,
            (1, 12) => PixelFormat::Gray12Le,
            (3, 8) if is_rgb => PixelFormat::Rgb24,
            (3, 8) => yuv_layout(frame, false),
            (3, 12) => yuv_layout(frame, true),
            (4, 8) => PixelFormat::Cmyk,
            (4, _) => {
                return Err(Error::unsupported(format!(
                    "12-bit: {nc}-component JPEGs not supported"
                )))
            }
            (2, _) => return Err(Error::unsupported("2-component JPEG")),
            (1 | 3, _) => return Err(Error::unsupported(format!("precision {p}"))),
            _ => return Err(Error::unsupported(format!("{nc}-component JPEG"))),
        }
    };
    let format = if hdr.jfif {
        format.full_range_label()
    } else {
        format
    };
    Ok(Shape {
        format,
        color: color_for(format),
    })
}

/// Colour description for a decoded layout (see [`ColorInfo`]).
fn color_for(format: PixelFormat) -> ColorInfo {
    if format.is_gray() {
        ColorInfo::gray()
    } else if format == PixelFormat::Cmyk {
        ColorInfo::cmyk()
    } else if format.is_rgb() {
        ColorInfo::srgb()
    } else {
        // Three-component YCbCr: full range per T.871 / T.872 §6.1,
        // whether or not a JFIF segment is present.
        ColorInfo::jfif_ycbcr()
    }
}

// ---------------------------------------------------------------------------
// Encode helpers
// ---------------------------------------------------------------------------

struct PreparedPlanes {
    planes: Vec<Vec<u16>>,
    opts: EncodeOptions,
}

/// Read every sample of plane `p` as `u16` into a tightly packed
/// `w × h` vector (`channels` interleaved channels, channel `c`).
fn gather(
    p: &Plane,
    bps: usize,
    channels: usize,
    c: usize,
    w: usize,
    h: usize,
) -> Result<Vec<u16>> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = y * p.stride;
        for x in 0..w {
            let i = row + (x * channels + c) * bps;
            let v = if bps == 1 {
                *p.data
                    .get(i)
                    .ok_or_else(|| Error::invalid("JPEG encode: plane shorter than its geometry"))?
                    as u16
            } else {
                let b = p.data.get(i..i + 2).ok_or_else(|| {
                    Error::invalid("JPEG encode: plane shorter than its geometry")
                })?;
                u16::from_le_bytes([b[0], b[1]])
            };
            out.push(v);
        }
    }
    Ok(out)
}

/// Split a [`JpegImage`] into per-component sample planes and the
/// writer options its layout implies.
fn prepare_planes(image: &JpegImage, opts: &EncodeOptions) -> Result<PreparedPlanes> {
    let f = image.format;
    if image.planes.len() != f.plane_count() {
        return Err(Error::invalid(format!(
            "JPEG encode: {} plane(s) do not fit {f} ({} expected)",
            image.planes.len(),
            f.plane_count()
        )));
    }
    let (w, h) = (image.width as usize, image.height as usize);
    let bps = f.bytes_per_sample();
    let carrier_bits = (bps * 8) as u8;
    let precision = if f.nominal_bits() == carrier_bits {
        image.precision.clamp(1, carrier_bits)
    } else {
        f.nominal_bits()
    };
    let lossless = matches!(opts.process, JpegProcess::Lossless { .. });

    let mut o = opts.clone();
    o.precision = precision;
    // `metadata` from the image, overridden blob-by-blob by the option.
    let mut meta = image.metadata.clone();
    if opts.metadata.icc.is_some() {
        meta.icc = opts.metadata.icc.clone();
    }
    if opts.metadata.exif.is_some() {
        meta.exif = opts.metadata.exif.clone();
    }
    if opts.metadata.xmp.is_some() {
        meta.xmp = opts.metadata.xmp.clone();
    }
    o.metadata = meta;

    let auto = opts.signalling == ColorSignalling::Auto;
    let planes: Vec<Vec<u16>>;
    let sampling: Vec<(u8, u8)>;
    match f {
        PixelFormat::Gray8
        | PixelFormat::Gray10Le
        | PixelFormat::Gray12Le
        | PixelFormat::Gray16Le => {
            planes = vec![gather(&image.planes[0], bps, 1, 0, w, h)?];
            sampling = vec![(1, 1)];
        }
        PixelFormat::Rgb24 | PixelFormat::Rgb48Le => {
            let p = &image.planes[0];
            planes = (0..3)
                .map(|c| gather(p, bps, 3, c, w, h))
                .collect::<Result<_>>()?;
            sampling = vec![(1, 1); 3];
            if auto {
                o.signalling = ColorSignalling::Rgb;
            }
        }
        PixelFormat::Gbrp10Le | PixelFormat::Gbrp12Le | PixelFormat::Gbrp14Le => {
            if !lossless {
                return Err(Error::unsupported(format!(
                    "JPEG encode: {f} is lossless-only (the DCT processes have no RGB-class signalling at this precision)"
                )));
            }
            planes = image
                .planes
                .iter()
                .map(|p| gather(p, 2, 1, 0, w, h))
                .collect::<Result<_>>()?;
            sampling = vec![(1, 1); 3];
            if auto {
                o.signalling = ColorSignalling::None;
            }
        }
        PixelFormat::Cmyk => {
            let p = &image.planes[0];
            planes = (0..4)
                .map(|c| gather(p, 1, 4, c, w, h))
                .collect::<Result<_>>()?;
            sampling = vec![(1, 1); 4];
        }
        _ => {
            // Planar YCbCr: luma at (h, v) = the divisors, chroma 1×1.
            let (dh, dv) = f.chroma_divisors();
            let mut v = Vec::with_capacity(3);
            for (i, p) in image.planes.iter().enumerate() {
                let (pw, ph) = f.plane_dimensions(image.width, image.height, i);
                v.push(gather(p, bps, 1, 0, pw, ph)?);
            }
            planes = v;
            sampling = vec![(dh as u8, dv as u8), (1, 1), (1, 1)];
            if auto {
                o.signalling = ColorSignalling::Jfif;
            }
        }
    }
    if !opts.sampling.is_empty() && opts.sampling != sampling {
        return Err(Error::unsupported(format!(
            "JPEG encode: options ask for sampling {:?} but the {f} image is {:?}",
            opts.sampling, sampling
        )));
    }
    o.sampling = sampling;
    Ok(PreparedPlanes { planes, opts: o })
}

/// Convert packed RGB to the native layout [`encode_rgb8`] writes:
/// RGB-coded 4:4:4 when `ColorSignalling::Rgb` is requested, otherwise
/// full-range YCbCr at `opts.chroma` (T.871 §7 forward equations,
/// two-tap box filter per subsampled axis).
fn rgb8_to_native(width: u32, height: u32, rgb: &[u8], opts: &EncodeOptions) -> Result<JpegImage> {
    let (w, h) = (width as usize, height as usize);
    if opts.signalling == ColorSignalling::Rgb {
        return Ok(JpegImage::from_rgb8(
            width,
            height,
            rgb[..w * h * 3].to_vec(),
        ));
    }
    let (format, dh, dv) = match opts.chroma {
        ChromaSubsampling::Yuv444 => (PixelFormat::YuvJ444P, 1, 1),
        ChromaSubsampling::Yuv422 => (PixelFormat::YuvJ422P, 2, 1),
        ChromaSubsampling::Yuv420 => (PixelFormat::YuvJ420P, 2, 2),
        ChromaSubsampling::Yuv411 => (PixelFormat::Yuv411P, 4, 1),
        ChromaSubsampling::GrayscaleOnly => (PixelFormat::Gray8, 1, 1),
        ChromaSubsampling::Custom => {
            return Err(Error::unsupported(
                "JPEG encode: `chroma = Custom` has no layout; use `sampling` with the plane-level writer",
            ))
        }
    };
    // Forward T.871 §7 (full-precision form), exact integer rounding:
    //   Y  = Round(0.299 R + 0.587 G + 0.114 B)
    //   CB = Round((−0.299 R − 0.587 G + 0.886 B) / 1.772 + 128)
    //   CR = Round((0.701 R − 0.587 G − 0.114 B) / 1.402 + 128)
    let mut yp = vec![0u8; w * h];
    let (cw, ch) = (w.div_ceil(dh), h.div_ceil(dv));
    let mut cb_sum = vec![0u32; cw * ch];
    let mut cr_sum = vec![0u32; cw * ch];
    let mut cnt = vec![0u32; cw * ch];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (r, g, b) = (rgb[i] as i64, rgb[i + 1] as i64, rgb[i + 2] as i64);
            let yy = round_div(299 * r + 587 * g + 114 * b, 1000);
            yp[y * w + x] = yy.clamp(0, 255) as u8;
            if format != PixelFormat::Gray8 {
                let cb = round_div(-299 * r - 587 * g + 886 * b + 128 * 1772, 1772);
                let cr = round_div(701 * r - 587 * g - 114 * b + 128 * 1402, 1402);
                let ci = (y / dv) * cw + x / dh;
                cb_sum[ci] += cb.clamp(0, 255) as u32;
                cr_sum[ci] += cr.clamp(0, 255) as u32;
                cnt[ci] += 1;
            }
        }
    }
    let mut planes = vec![Plane::new(w, yp)];
    if format != PixelFormat::Gray8 {
        let avg = |sum: &[u32]| -> Vec<u8> {
            sum.iter()
                .zip(&cnt)
                .map(|(&s, &n)| ((s + n / 2) / n.max(1)) as u8)
                .collect()
        };
        planes.push(Plane::new(cw, avg(&cb_sum)));
        planes.push(Plane::new(cw, avg(&cr_sum)));
    }
    let color = if format == PixelFormat::Gray8 {
        ColorInfo::gray()
    } else {
        ColorInfo::jfif_ycbcr()
    };
    Ok(JpegImage::new(width, height, format, planes).with_color(color))
}

/// `⌊num / den + 0.5⌋`, exact.
#[inline]
fn round_div(num: i64, den: i64) -> i64 {
    (2 * num + den).div_euclid(2 * den)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::MjpegPixelFormat as F;

    fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                v.push((x * 255 / w.max(1)) as u8);
                v.push((y * 255 / h.max(1)) as u8);
                v.push(((x + y) * 7 % 256) as u8);
            }
        }
        v
    }

    #[test]
    fn probe_is_a_pure_sniff() {
        assert!(!probe(&[]));
        assert!(!probe(&[0xFF, 0xD8]));
        assert!(!probe(&[0xFF, 0xD8, 0xFF]));
        assert!(probe(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(probe(&[0xFF, 0xD8, 0xFF, 0xDB]));
        assert!(!probe(&[0xFF, 0xD8, 0xFF, 0x00]));
        assert!(!probe(&[0xFF, 0xD8, 0x00, 0xE0]));
        assert!(!probe(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn rgb8_round_trip_through_every_chroma_layout() {
        let (w, h) = (19u32, 11u32);
        let rgb = gradient_rgb(w, h);
        for (chroma, fmt) in [
            (ChromaSubsampling::Yuv420, F::YuvJ420P),
            (ChromaSubsampling::Yuv422, F::YuvJ422P),
            (ChromaSubsampling::Yuv444, F::YuvJ444P),
            (ChromaSubsampling::Yuv411, F::Yuv411P),
            (ChromaSubsampling::GrayscaleOnly, F::Gray8),
        ] {
            let opts = EncodeOptions::new().with_quality(95).with_chroma(chroma);
            let jpeg = encode_rgb8(w, h, &rgb, &opts).unwrap();
            assert!(probe(&jpeg));
            let info = info(&jpeg).unwrap();
            assert_eq!((info.width, info.height), (w, h));
            assert_eq!(info.format, fmt, "{chroma:?}");
            assert!(info.has_jfif);
            let img = decode(&jpeg).unwrap();
            assert_eq!(img.format, fmt);
            assert_eq!(img.color, info.color);
            let back = img.to_rgb8();
            assert_eq!(back.len(), rgb.len());
            if fmt != F::Gray8 {
                // Lossy, high quality: every channel within a modest band.
                let worst = rgb
                    .iter()
                    .zip(&back)
                    .map(|(a, b)| (*a as i32 - *b as i32).abs())
                    .max()
                    .unwrap();
                assert!(worst <= 40, "{chroma:?}: worst channel error {worst}");
            }
            let rgba = decode_rgba8(&jpeg).unwrap();
            assert_eq!(rgba.data.len(), (w * h * 4) as usize);
            assert!(rgba.data.iter().skip(3).step_by(4).all(|&a| a == 255));
        }
    }

    #[test]
    fn rgba8_drops_alpha_and_rgb_signalling_writes_rgb() {
        let (w, h) = (8u32, 8u32);
        let rgb = gradient_rgb(w, h);
        let rgba: Vec<u8> = rgb.chunks(3).flat_map(|p| [p[0], p[1], p[2], 7]).collect();
        let opts = EncodeOptions::new().with_signalling(ColorSignalling::Rgb);
        let a = encode_rgba8(w, h, &rgba, &opts).unwrap();
        let b = encode_rgb8(w, h, &rgb, &opts).unwrap();
        assert_eq!(a, b);
        let img = decode(&a).unwrap();
        assert_eq!(img.format, F::Rgb24);
        assert_eq!(img.color, ColorInfo::srgb());
        assert!(info(&a).unwrap().has_adobe);
    }

    #[test]
    fn lossless_round_trips_planes_and_metadata() {
        let (w, h) = (13u32, 7u32);
        let icc = vec![0x11u8; 300];
        let exif = b"II*\0\x08\0\0\0\0\0".to_vec();
        let xmp = b"<x:xmpmeta/>".to_vec();
        let meta = Metadata::new()
            .with_icc(icc.clone())
            .with_exif(exif.clone())
            .with_xmp(xmp.clone());
        let lossless = EncodeOptions::new().with_lossless(1);

        // Gray8.
        let g = JpegImage::new(
            w,
            h,
            F::Gray8,
            vec![Plane::new(
                w as usize,
                gradient_rgb(w, h)[..(w * h) as usize].to_vec(),
            )],
        )
        .with_metadata(meta.clone());
        let bytes = encode(&g, &lossless).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.planes, g.planes);
        assert_eq!(back.metadata, g.metadata);
        assert_eq!(back.format, F::Gray8);
        let i = info(&bytes).unwrap();
        assert!(i.has_icc && i.has_exif && i.has_xmp && i.lossless);

        // Rgb24 (RGB-coded).
        let rgb = JpegImage::from_rgb8(w, h, gradient_rgb(w, h));
        let bytes = encode(&rgb, &lossless).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.format, F::Rgb24);
        assert_eq!(back.planes, rgb.planes);
        assert_eq!(back.to_rgb8(), rgb.planes[0].data);

        // Gray16Le at P = 16 and Gray12Le at P = 12.
        let deep: Vec<u8> = (0..w * h)
            .flat_map(|i| ((i * 2749) as u16).to_le_bytes())
            .collect();
        let g16 = JpegImage::new(
            w,
            h,
            F::Gray16Le,
            vec![Plane::new(w as usize * 2, deep.clone())],
        );
        let bytes = encode(&g16, &lossless).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.format, F::Gray16Le);
        assert_eq!(back.precision, 16);
        assert_eq!(back.planes, g16.planes);

        let twelve: Vec<u8> = (0..w * h)
            .flat_map(|i| ((i * 173 % 4096) as u16).to_le_bytes())
            .collect();
        let g12 = JpegImage::new(w, h, F::Gray12Le, vec![Plane::new(w as usize * 2, twelve)]);
        let bytes = encode(&g12, &lossless).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.format, F::Gray12Le);
        assert_eq!(back.planes, g12.planes);

        // Gbrp12Le keeps plane order.
        let gb = JpegImage::new(
            w,
            h,
            F::Gbrp12Le,
            (0..3)
                .map(|c| {
                    Plane::new(
                        w as usize * 2,
                        (0..w * h)
                            .flat_map(|i| (((i + c) * 911 % 4096) as u16).to_le_bytes())
                            .collect(),
                    )
                })
                .collect(),
        );
        let bytes = encode(&gb, &lossless).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.format, F::Gbrp12Le);
        assert_eq!(back.planes, gb.planes);
        assert!(matches!(
            encode(&gb, &EncodeOptions::new()),
            Err(Error::Unsupported(_))
        ));

        // Yuv420P (lossless, JFIF) comes back as YuvJ420P with the
        // same planes.
        let yuv = JpegImage::new(
            w,
            h,
            F::Yuv420P,
            vec![
                Plane::new(w as usize, vec![200; (w * h) as usize]),
                Plane::new(7, (0..7 * 4).map(|i| (i * 9) as u8).collect()),
                Plane::new(7, (0..7 * 4).map(|i| (255 - i * 9) as u8).collect()),
            ],
        );
        let bytes = encode(&yuv, &lossless).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.format, F::YuvJ420P);
        assert_eq!(back.planes, yuv.planes);
        assert_eq!(info(&bytes).unwrap().format, F::YuvJ420P);

        // Cmyk.
        let cmyk = JpegImage::new(
            w,
            h,
            F::Cmyk,
            vec![Plane::new(
                w as usize * 4,
                (0..w * h * 4).map(|i| (i * 31) as u8).collect(),
            )],
        );
        let bytes = encode(&cmyk, &lossless).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.format, F::Cmyk);
        assert_eq!(back.planes, cmyk.planes);
        assert_eq!(back.color, ColorInfo::cmyk());
    }

    #[test]
    fn encode_refuses_mismatched_planes_and_sampling() {
        let img = JpegImage::new(4, 4, F::Yuv420P, vec![Plane::new(4, vec![0; 16])]);
        assert!(matches!(
            encode(&img, &EncodeOptions::new()),
            Err(Error::InvalidData(_))
        ));
        let img = JpegImage::from_rgb8(4, 4, vec![0; 48]);
        let opts = EncodeOptions::new().with_sampling(vec![(2, 2), (1, 1), (1, 1)]);
        assert!(matches!(encode(&img, &opts), Err(Error::Unsupported(_))));
        let short = JpegImage::new(4, 4, F::Gray8, vec![Plane::new(4, vec![0; 3])]);
        assert!(matches!(
            encode(&short, &EncodeOptions::new()),
            Err(Error::InvalidData(_))
        ));
        assert!(encode_rgb8(4, 4, &[0; 47], &EncodeOptions::new()).is_err());
        assert!(encode_rgba8(4, 4, &[0; 63], &EncodeOptions::new()).is_err());
    }

    #[test]
    fn decode_options_limits_and_strict() {
        let rgb = gradient_rgb(16, 8);
        let jpeg = encode_rgb8(16, 8, &rgb, &EncodeOptions::new()).unwrap();
        let lim = |o: DecodeOptions| decode_with(&jpeg, &o);
        assert!(matches!(
            lim(DecodeOptions::new().with_max_width(15)),
            Err(Error::LimitExceeded(_))
        ));
        assert!(matches!(
            lim(DecodeOptions::new().with_max_height(7)),
            Err(Error::LimitExceeded(_))
        ));
        assert!(matches!(
            lim(DecodeOptions::new().with_max_pixels(127)),
            Err(Error::LimitExceeded(_))
        ));
        assert!(matches!(
            lim(DecodeOptions::new().with_max_bytes(jpeg.len() - 1)),
            Err(Error::LimitExceeded(_))
        ));
        assert!(lim(DecodeOptions::new().with_max_pixels(128)).is_ok());
        assert!(lim(DecodeOptions::new().with_strict(true)).is_ok());
        let mut trailing = jpeg.clone();
        trailing.extend_from_slice(&[0, 0, 0]);
        assert!(decode(&trailing).is_ok());
        assert!(matches!(
            decode_with(&trailing, &DecodeOptions::new().with_strict(true)),
            Err(Error::InvalidData(_))
        ));
        // Progressive output decodes too and reports as such.
        let prog = encode_rgb8(16, 8, &rgb, &EncodeOptions::new().with_progressive(true)).unwrap();
        assert!(info(&prog).unwrap().progressive);
        assert_eq!(decode(&prog).unwrap().format, F::YuvJ420P);
    }

    #[test]
    fn decode_from_and_encode_to_stream() {
        let rgb = gradient_rgb(5, 5);
        let img = JpegImage::from_rgb8(5, 5, rgb);
        let mut out = Vec::new();
        encode_to(&img, &EncodeOptions::new(), &mut out).unwrap();
        let back = decode_from(std::io::Cursor::new(&out)).unwrap();
        assert_eq!(back.format, F::Rgb24);
        assert_eq!((back.width, back.height), (5, 5));
        let e = decode_from(&b"not a jpeg"[..]).unwrap_err();
        assert!(matches!(e, Error::InvalidData(_)));
    }

    #[test]
    fn abbreviated_tables_through_decode_options() {
        let img = JpegImage::from_rgb8(8, 8, gradient_rgb(8, 8));
        let opts = EncodeOptions::new().with_abbreviated(true).with_lossless(1);
        let prepared = prepare_planes(&img, &opts).unwrap();
        let refs: Vec<&[u16]> = prepared.planes.iter().map(|p| p.as_slice()).collect();
        let enc = prepared.opts.encode(8, 8, &refs).unwrap();
        let tables = enc.tables.expect("abbreviated pair");
        assert!(decode(&enc.data).is_err());
        let back = decode_with(&enc.data, &DecodeOptions::new().with_tables(tables)).unwrap();
        assert_eq!(back.format, F::Rgb24);
        assert_eq!(back.planes, img.planes);
    }

    #[test]
    fn hostile_inputs_error_not_panic() {
        for bad in [
            &[][..],
            &[0xFF][..],
            &[0xFF, 0xD8][..],
            &[0xFF, 0xD8, 0xFF, 0xD9][..],
            &[0xFF, 0xD8, 0xFF, 0xDA, 0x00, 0x02][..],
            &[0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x05, 8, 0, 1][..],
            &[
                0xFF, 0xD8, 0xFF, 0xE2, 0x00, 0x10, b'I', b'C', b'C', b'_', b'P', b'R', b'O', b'F',
                b'I', b'L', b'E', 0, 1,
            ][..],
        ] {
            assert!(info(bad).is_err());
            assert!(decode(bad).is_err());
            assert!(decode_rgb8(bad).is_err());
        }
    }
}
