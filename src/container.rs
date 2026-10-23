//! Still-image JPEG container (`.jpg` / `.jpeg`).
//!
//! A JPEG file is simply a single SOI..EOI byte stream — there is no
//! container wrapping around the codec payload. Registering this as a
//! container lets the oxideav pipeline treat a standalone `.jpg` as a
//! one-frame "video" whose stream uses the existing `mjpeg` codec id
//! (Motion JPEG = concatenated stills, so a still is the N=1 case).
//!
//! * Demuxer: reads the file, walks its marker segments to discover the
//!   canvas dimensions and the SOI..EOI bounds (robust against trailing
//!   garbage like some cameras append), and emits the whole frame as
//!   one packet tagged `codec_id = "mjpeg"`. The stream's
//!   `pixel_format` and `color_signal` are exactly what the `mjpeg`
//!   decoder emits for that file — the same header rules the standalone
//!   `info` / `decode` apply (packed `Rgb24` for an Adobe `transform =
//!   0` / `'R' 'G' 'B'` three-component frame, `YuvJ*` full-range
//!   planar YCbCr with the sYCC signal, `Gray8` / `Gray12Le` /
//!   `Gray16Le`, `Cmyk`, the lossless `Gbrp*` / `Rgb48Le` carriers) —
//!   so a frame read through the registry can be labelled from the
//!   stream. A layout the decoder refuses leaves `pixel_format` unset.
//! * Muxer: writes the encoded packet's bytes verbatim. JPEG has no
//!   container wrapping, so this is a pass-through.
//! * Probe: three-byte magic `FF D8 FF` at offset 0 → score 100.
//!
//! Caveats / notes:
//!
//! - Exif / XMP / ICC data lives in APP1 / APP2 segments; those are
//!   preserved as part of the packet payload because we copy the full
//!   SOI..EOI span, and the primary picture's blobs (the ICC profile
//!   reassembled from its chunk sequence, the Exif block, the XMP
//!   packet — [`crate::metadata`]) are declared on the stream's
//!   `CodecParameters::blobs`, so a consumer that never decodes still
//!   sees them. The decoder puts the same blobs on the frame record.
//! - We accept SOF0/1/2/3 for dimension discovery so that probing /
//!   metadata works even if the underlying decoder can't yet decode the
//!   bitstream (progressive JPEGs still round-trip at the container
//!   level).
//! - Trailing bytes past EOI are dropped. Leading bytes before SOI are
//!   not allowed (that would be a different file format).

use std::io::{Read, SeekFrom, Write};

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, Error, MediaType, Packet, PixelFormat, Result,
    StreamInfo, TimeBase,
};
use oxideav_core::{ContainerRegistry, Demuxer, Muxer, ProbeData, ReadSeek, WriteSeek};

use crate::jpeg::markers::{self, EOI, SOI};
use crate::jpeg::parser::{SofInfo, parse_sof};

/// Register the still-image JPEG container under the name `jpeg`.
pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("jpeg", open_demuxer);
    reg.register_muxer("jpeg", open_muxer);
    reg.register_extension("jpg", "jpeg");
    reg.register_extension("jpeg", "jpeg");
    reg.register_extension("jpe", "jpeg");
    reg.register_extension("jfif", "jpeg");
    reg.register_probe("jpeg", probe);
}

/// Probe: `FF D8 FF` at offset 0 is an unambiguous JPEG signature.
pub fn probe(p: &ProbeData) -> u8 {
    if p.buf.len() >= 3 && p.buf[0] == 0xFF && p.buf[1] == 0xD8 && p.buf[2] == 0xFF {
        100
    } else {
        0
    }
}

// --- Demuxer ---------------------------------------------------------------

fn open_demuxer(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> Result<Box<dyn Demuxer>> {
    input.seek(SeekFrom::Start(0))?;
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    drop(input);

    // Validate SOI at offset 0 and locate the EOI that terminates this JPEG,
    // trimming any trailing garbage some cameras append (MPF thumbnails, etc.).
    let (start, end) = find_soi_eoi(&buf)?;
    if start != 0 {
        // We insist the file *starts* with SOI — a JPEG with leading junk
        // is probably a different container (or a TIFF-with-embedded-JPEG).
        return Err(Error::invalid("JPEG: SOI not at offset 0"));
    }
    let data = buf[start..end].to_vec();

    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    declare_shape(&mut params, &data)?;

    let time_base = TimeBase::new(1, 1);
    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(1),
        start_time: Some(0),
        params,
    };

    let mut pkt = Packet::new(0, time_base, data);
    pkt.pts = Some(0);
    pkt.dts = Some(0);
    pkt.duration = Some(1);
    pkt.flags.keyframe = true;

    Ok(Box::new(JpegDemuxer {
        stream,
        pending: Some(pkt),
    }))
}

/// Return `(start, end_exclusive)` of the first complete SOI..EOI JPEG in
/// `buf`. The returned span includes the SOI and EOI markers themselves.
///
/// Walks the stream the way the decoder does: every length-prefixed
/// segment is skipped as a whole (an `APPn` payload may hold anything,
/// including a complete JPEG — an Exif block's thumbnail is `SOI … EOI`
/// inside `APP1` — or the byte pair of a marker inside an ICC chunk),
/// and only entropy-coded data after `SOS` is scanned byte by byte
/// (stuffed `FF 00` and `RSTn` belong to the scan). A segment whose
/// length runs past the buffer is scanned byte by byte from its marker
/// instead, so a damaged header still finds its `EOI` leniently.
fn find_soi_eoi(buf: &[u8]) -> Result<(usize, usize)> {
    if buf.len() < 4 {
        return Err(Error::invalid("JPEG: file too short"));
    }
    if buf[0] != 0xFF || buf[1] != SOI {
        return Err(Error::invalid("JPEG: missing SOI"));
    }
    let mut i = 2;
    while i + 1 < buf.len() {
        if buf[i] != 0xFF {
            i += 1;
            continue;
        }
        // Collapse any run of 0xFF fill bytes.
        let mut j = i + 1;
        while j < buf.len() && buf[j] == 0xFF {
            j += 1;
        }
        if j >= buf.len() {
            break;
        }
        let m = buf[j];
        if m == 0x00 || markers::is_rst(m) || m == SOI {
            // Stuffed zero, restart marker, or a repeated SOI: no
            // segment follows.
            i = j + 1;
            continue;
        }
        if m == EOI {
            return Ok((0, j + 1));
        }
        // Length-prefixed segment (T.81 Table B.1: every marker other
        // than SOI / EOI / RSTn / TEM (`0x01`) carries `Lp`). Skip its payload
        // whole when the length fits; otherwise keep scanning from the
        // marker byte.
        let lp = if j + 2 < buf.len() {
            u16::from_be_bytes([buf[j + 1], buf[j + 2]]) as usize
        } else {
            0
        };
        if m == 0x01 || lp < 2 || j + 1 + lp > buf.len() {
            i = j + 1;
            continue;
        }
        i = j + 1 + lp;
        // After SOS the entropy-coded data is scanned byte by byte by
        // the loop itself (the next iteration lands on its bytes).
    }
    Err(Error::invalid("JPEG: no EOI marker found"))
}

/// Walk the marker segments until we find any SOF marker and parse it.
/// Returns the canvas size and component layout so the demuxer can report
/// width/height on its stream.
fn scan_for_sof(data: &[u8]) -> Result<SofInfo> {
    // Skip past SOI.
    let body = &data[2..];
    let mut walker = crate::jpeg::parser::MarkerWalker::new(body);
    loop {
        let Some(marker) = walker.next_marker()? else {
            return Err(Error::invalid("JPEG: SOF not found before EOF"));
        };
        if markers::is_sof(marker) {
            let payload = walker.read_segment_payload()?;
            return Ok(parse_sof(payload)?);
        }
        // SOI / EOI / RST have no length field. Everything else does.
        if marker == SOI || marker == EOI || markers::is_rst(marker) {
            continue;
        }
        // Length-prefixed segment — consume and ignore.
        let _ = walker.read_segment_payload()?;
    }
}

/// Fill `params` with what the `mjpeg` decoder emits for the JPEG in
/// `data`: geometry, the native `pixel_format` and the colour signal,
/// from the header rules the standalone `decode` applies
/// ([`crate::api::stream_shape`]). A layout the decoder refuses still
/// opens (geometry from the SOF) with `pixel_format` left unset, so a
/// consumer sees "unknown" rather than a guess; a header the decoder
/// cannot parse at all is an error.
pub(crate) fn declare_shape(params: &mut CodecParameters, data: &[u8]) -> Result<()> {
    // The picture's metadata segments, read as the decoder reads them
    // (lenient: a damaged ICC sequence is simply absent).
    params.blobs = crate::metadata::read_metadata(data)
        .map(|m| m.to_blobs())
        .unwrap_or_default();
    match crate::api::stream_shape(data) {
        Ok((width, height, format, color)) => {
            params.width = Some(width);
            params.height = Some(height);
            params.pixel_format = Some(PixelFormat::from(format));
            params.color_signal = oxideav_core::ColorSignal::from(color);
        }
        Err(crate::MjpegError::Unsupported(_)) => {
            let sof = scan_for_sof(data)?;
            params.width = Some(u32::from(sof.width));
            params.height = Some(u32::from(sof.height));
            params.pixel_format = None;
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}

struct JpegDemuxer {
    stream: StreamInfo,
    pending: Option<Packet>,
}

impl Demuxer for JpegDemuxer {
    fn format_name(&self) -> &str {
        "jpeg"
    }

    fn streams(&self) -> &[StreamInfo] {
        std::slice::from_ref(&self.stream)
    }

    fn next_packet(&mut self) -> Result<Packet> {
        self.pending.take().ok_or(Error::Eof)
    }
}

// --- Muxer -----------------------------------------------------------------

fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::invalid(
            "JPEG container holds exactly one video stream",
        ));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Video {
        return Err(Error::invalid("JPEG container: stream must be video"));
    }
    if s.params.codec_id.as_str() != crate::CODEC_ID_STR {
        return Err(Error::invalid(format!(
            "JPEG container requires codec_id={} (got {})",
            crate::CODEC_ID_STR,
            s.params.codec_id
        )));
    }
    Ok(Box::new(JpegMuxer {
        output,
        header_written: false,
        wrote_packet: false,
        trailer_written: false,
    }))
}

struct JpegMuxer {
    output: Box<dyn WriteSeek>,
    header_written: bool,
    wrote_packet: bool,
    trailer_written: bool,
}

impl Muxer for JpegMuxer {
    fn format_name(&self) -> &str {
        "jpeg"
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("JPEG muxer: write_header called twice"));
        }
        // No container wrapping — nothing to emit here.
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("JPEG muxer: write_header not called"));
        }
        if self.wrote_packet {
            return Err(Error::invalid(
                "JPEG container is single-frame: exactly one packet per file",
            ));
        }
        self.output.write_all(&packet.data)?;
        self.wrote_packet = true;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_accepts_soi_ff() {
        let p = ProbeData {
            buf: &[0xFF, 0xD8, 0xFF, 0xE0],
            ext: None,
        };
        assert_eq!(probe(&p), 100);
    }

    #[test]
    fn probe_rejects_short_buffer() {
        let p = ProbeData {
            buf: &[0xFF, 0xD8],
            ext: None,
        };
        assert_eq!(probe(&p), 0);
    }

    #[test]
    fn probe_rejects_non_jpeg() {
        let p = ProbeData {
            buf: &[0x00, 0x00, 0x00, 0x00],
            ext: None,
        };
        assert_eq!(probe(&p), 0);
    }

    #[test]
    fn find_soi_eoi_trims_trailing_garbage() {
        // Minimal SOI..EOI with 4 bytes of trailing garbage.
        let jpeg = [0xFF, 0xD8, 0xFF, 0xD9, 0x00, 0x11, 0x22, 0x33];
        let (s, e) = find_soi_eoi(&jpeg).expect("span");
        assert_eq!((s, e), (0, 4));
    }

    #[test]
    fn find_soi_eoi_skips_segment_payloads_whole() {
        // APP1 whose payload is a complete JPEG (an Exif thumbnail):
        // the inner EOI must not end the outer file.
        let inner = [0xFF, 0xD8, 0xFF, 0xD9];
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x06];
        jpeg.extend_from_slice(&inner);
        jpeg.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x04, 0xFF, 0xD9]); // DQT payload = FF D9
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        let (s, e) = find_soi_eoi(&jpeg).expect("span");
        assert_eq!((s, e), (0, jpeg.len()));
        // A segment length past the end falls back to the byte scan.
        let mut cut = vec![0xFF, 0xD8, 0xFF, 0xE1, 0xFF, 0xFF, 0x01, 0xFF, 0xD9];
        assert_eq!(find_soi_eoi(&cut).unwrap(), (0, cut.len()));
        cut.pop();
        assert!(find_soi_eoi(&cut).is_err());
    }

    #[test]
    fn find_soi_eoi_rejects_missing_soi() {
        let not_jpeg = [0x00, 0x01, 0x02, 0x03];
        assert!(find_soi_eoi(&not_jpeg).is_err());
    }
}
