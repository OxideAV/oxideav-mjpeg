//! RTP/JPEG depacketization (RFC 2435).
//!
//! Motion-JPEG carried over RTP omits the JPEG frame and scan headers
//! from the wire: most of the table-specification data rarely changes
//! frame to frame, so RTP/JPEG transmits the entropy-coded scan in
//! *abbreviated* form and carries the information that would otherwise
//! live in the frame/scan headers inside a small fixed-layout RTP/JPEG
//! header (RFC 2435 §3.1). This module parses that header, reassembles
//! the fragmented scan payload, and reconstructs the absent SOI / DQT /
//! [DRI] / SOF0 / DHT / SOS / EOI marker segments so the resulting bytes
//! form a complete JPEG interchange stream the rest of this crate's
//! decoder can consume directly.
//!
//! The depacketizer covers the well-known fixed type mappings the RFC
//! defines: types 0/64 (4:2:2-class, `H=2 V=1` luma) and types 1/65
//! (4:2:0-class, `H=2 V=2` luma), each with three YUV components and a
//! single interleaved scan (§4.1). Types in the 64..=127 range carry an
//! additional Restart Marker header (§3.1.7) that supplies the DRI
//! value. Q values 1..=99 select tables computed with the Independent
//! JPEG Group scale formula over Annex K.1 / K.2 (§4.2); Q values
//! 128..=255 carry the quantization tables in-band via a Quantization
//! Table header (§3.1.8).
//!
//! What is *not* the depacketizer's job: RTP transport itself. Callers
//! strip the 12-byte RTP fixed header (and any CSRC / extension words),
//! sort packets by sequence number, and feed the RTP/JPEG payload of
//! each packet — i.e. the bytes that immediately follow the RTP header —
//! to [`JpegDepacketizer::push`]. The depacketizer keys reassembly on
//! the in-band Fragment Offset field, so misordered delivery within a
//! single frame is tolerated as long as the RTP marker-bit packet
//! (the last fragment) arrives.

use crate::error::{MjpegError as Error, Result};
use crate::jpeg::huffman::{
    STD_AC_CHROMA_BITS, STD_AC_CHROMA_VALS, STD_AC_LUMA_BITS, STD_AC_LUMA_VALS, STD_DC_CHROMA_BITS,
    STD_DC_CHROMA_VALS, STD_DC_LUMA_BITS, STD_DC_LUMA_VALS,
};
use crate::jpeg::markers;
use crate::jpeg::quant::{scale_for_quality, DEFAULT_CHROMA_Q50, DEFAULT_LUMA_Q50};
use crate::jpeg::zigzag::ZIGZAG;

/// Length in bytes of the RFC 2435 §3.1 main JPEG header.
const MAIN_HDR_LEN: usize = 8;
/// Length in bytes of the RFC 2435 §3.1.7 Restart Marker header.
const RST_HDR_LEN: usize = 4;
/// Length in bytes of the RFC 2435 §3.1.8 Quantization Table header
/// (before the table data itself).
const QTBL_HDR_LEN: usize = 4;
/// The `0x40` bit in the Type field marking restart-marker presence
/// (RFC 2435 §3.1.3 / Appendix C `RTP_JPEG_RESTART`).
const TYPE_RESTART_BIT: u8 = 0x40;

/// Parsed RFC 2435 §3.1 main JPEG header (the first 8 bytes of every
/// RTP/JPEG payload).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MainHeader {
    /// Type-specific field (§3.1.1). For types 0/1 the low values carry
    /// the interlace mode (0 = progressive frame, 1/2 = odd/even field,
    /// 3 = single field shown full-frame).
    pub type_specific: u8,
    /// Byte offset of this fragment's payload within the whole frame
    /// (§3.1.2, 24-bit network order).
    pub fragment_offset: u32,
    /// Type field (§3.1.3). The `0x40` bit means restart markers are
    /// present and a Restart Marker header follows.
    pub typ: u8,
    /// Q field (§3.1.4). 1..=99 = IJG-scaled tables; 128..=255 = in-band
    /// Quantization Table header present (first fragment only).
    pub q: u8,
    /// Frame width in pixels (the wire field is in 8-pixel units, §3.1.5).
    pub width: u16,
    /// Frame height in pixels (the wire field is in 8-pixel units, §3.1.6).
    pub height: u16,
}

impl MainHeader {
    /// True if this type carries a Restart Marker header (types 64..=127).
    pub fn has_restart(&self) -> bool {
        self.typ & TYPE_RESTART_BIT != 0
    }

    /// The base type with the restart bit stripped (0 or 1 for the
    /// well-known fixed mappings).
    pub fn base_type(&self) -> u8 {
        self.typ & !TYPE_RESTART_BIT
    }
}

/// Parsed RFC 2435 §3.1.7 Restart Marker header (present for types
/// 64..=127, immediately after the main header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestartHeader {
    /// MCUs between restart markers — identical to the DRI marker value.
    /// MUST NOT be zero per the RFC.
    pub restart_interval: u16,
    /// First bit: this packet starts a reassembly chunk.
    pub first: bool,
    /// Last bit: this packet ends a reassembly chunk.
    pub last: bool,
    /// Restart count (14-bit): position of the first restart interval in
    /// this chunk, or `0x3FFF` when the whole frame must be reassembled
    /// before decoding.
    pub count: u16,
}

/// Result of feeding one RTP/JPEG payload into the depacketizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Progress {
    /// More fragments are required before a full frame is available.
    NeedMore,
    /// A complete JPEG interchange stream was reconstructed (the marker
    /// bit was supplied on this packet).
    Frame(Vec<u8>),
}

/// Parse the §3.1 main JPEG header from the front of an RTP/JPEG payload.
///
/// `payload` is the RTP packet body with the 12-byte RTP fixed header
/// (and any CSRC / extension words) already stripped by the caller.
pub fn parse_main_header(payload: &[u8]) -> Result<MainHeader> {
    if payload.len() < MAIN_HDR_LEN {
        return Err(Error::invalid("RTP/JPEG: payload shorter than main header"));
    }
    let fragment_offset =
        ((payload[1] as u32) << 16) | ((payload[2] as u32) << 8) | (payload[3] as u32);
    Ok(MainHeader {
        type_specific: payload[0],
        fragment_offset,
        typ: payload[4],
        q: payload[5],
        // Wire fields are in 8-pixel units; convert to pixels.
        width: (payload[6] as u16) * 8,
        height: (payload[7] as u16) * 8,
    })
}

/// Parse the §3.1.7 Restart Marker header from `bytes` (the 4 bytes
/// immediately following the main header for types 64..=127).
pub fn parse_restart_header(bytes: &[u8]) -> Result<RestartHeader> {
    if bytes.len() < RST_HDR_LEN {
        return Err(Error::invalid("RTP/JPEG: truncated restart-marker header"));
    }
    let restart_interval = u16::from_be_bytes([bytes[0], bytes[1]]);
    if restart_interval == 0 {
        // §3.1.7: "This value MUST NOT be zero."
        return Err(Error::invalid("RTP/JPEG: zero restart interval"));
    }
    let fl_count = u16::from_be_bytes([bytes[2], bytes[3]]);
    Ok(RestartHeader {
        restart_interval,
        first: fl_count & 0x8000 != 0,
        last: fl_count & 0x4000 != 0,
        count: fl_count & 0x3FFF,
    })
}

/// Two reconstructed 8-bit quantization tables (luma id 0, chroma id 1),
/// in zigzag order exactly as a DQT segment would carry them.
struct QuantPair {
    luma: [u8; 64],
    chroma: [u8; 64],
}

/// Compute the §4.2 IJG-scaled luma + chroma tables for a Q value
/// 1..=99, returned in zigzag order.
fn tables_from_q(q: u8) -> QuantPair {
    // `scale_for_quality` applies exactly the IJG scale-factor formula
    // (S = 5000/Q for Q<50, 200-2Q otherwise) over the Annex K.1 / K.2
    // base tables and saturates to 8 bits — the same computation RFC
    // 2435 §4.2 / Appendix A specifies. Tables are produced in natural
    // (row-major) order; reorder to zigzag for DQT carriage.
    let luma_nat = scale_for_quality(&DEFAULT_LUMA_Q50, q);
    let chroma_nat = scale_for_quality(&DEFAULT_CHROMA_Q50, q);
    let mut luma = [0u8; 64];
    let mut chroma = [0u8; 64];
    for k in 0..64 {
        luma[k] = luma_nat[ZIGZAG[k]].min(255) as u8;
        chroma[k] = chroma_nat[ZIGZAG[k]].min(255) as u8;
    }
    QuantPair { luma, chroma }
}

/// Write a DQT segment carrying one 8-bit (Pq=0) table already in zigzag
/// order.
fn write_dqt8(out: &mut Vec<u8>, table_id: u8, zigzag_vals: &[u8; 64]) {
    out.push(0xFF);
    out.push(markers::DQT);
    // length = 2 (length field) + 1 (Pq|Tq) + 64 (table) = 67.
    out.push(0x00);
    out.push(67);
    out.push(table_id & 0x0F); // Pq=0, Tq=table_id
    out.extend_from_slice(zigzag_vals);
}

/// Write a SOF0 segment for the well-known three-component YUV layout,
/// with the luma sampling factors derived from the base type.
fn write_sof0(out: &mut Vec<u8>, width: u16, height: u16, luma_h: u8, luma_v: u8) {
    out.push(0xFF);
    out.push(markers::SOF0);
    out.push(0x00);
    out.push(17); // length: 2 + 1(prec) + 4(dims) + 1(nc) + 3*3
    out.push(8); // precision
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.push(3); // components
                 // Y: id 1, sampling luma_h x luma_v, quant table 0.
    out.push(1);
    out.push((luma_h << 4) | luma_v);
    out.push(0);
    // Cb: id 2, 1x1, quant table 1.
    out.push(2);
    out.push(0x11);
    out.push(1);
    // Cr: id 3, 1x1, quant table 1.
    out.push(3);
    out.push(0x11);
    out.push(1);
}

/// Write a DHT segment.
fn write_dht(out: &mut Vec<u8>, class: u8, id: u8, bits: &[u8; 16], vals: &[u8]) {
    out.push(0xFF);
    out.push(markers::DHT);
    let len = 2 + 1 + 16 + vals.len();
    out.extend_from_slice(&(len as u16).to_be_bytes());
    out.push(((class & 0x01) << 4) | (id & 0x0F));
    out.extend_from_slice(bits);
    out.extend_from_slice(vals);
}

/// Write a DRI segment.
fn write_dri(out: &mut Vec<u8>, interval: u16) {
    out.push(0xFF);
    out.push(markers::DRI);
    out.push(0x00);
    out.push(0x04);
    out.extend_from_slice(&interval.to_be_bytes());
}

/// Write the SOS segment for the well-known three-component interleaved
/// scan: Y→(DC0,AC0), Cb→(DC1,AC1), Cr→(DC1,AC1), Ss=0 Se=63 Ah=Al=0.
fn write_sos(out: &mut Vec<u8>) {
    out.push(0xFF);
    out.push(markers::SOS);
    out.push(0x00);
    out.push(12); // length
    out.push(3); // components
    out.push(1);
    out.push(0x00); // Y: DC0 AC0
    out.push(2);
    out.push(0x11); // Cb: DC1 AC1
    out.push(3);
    out.push(0x11); // Cr: DC1 AC1
    out.push(0); // Ss
    out.push(63); // Se
    out.push(0); // Ah|Al
}

/// Build the full set of marker segments that precede the entropy-coded
/// scan, for the well-known §4.1 type mappings.
///
/// `quant` carries the two 8-bit tables (zigzag order). `dri` is the
/// restart interval (0 = no DRI segment). `base_type` selects the luma
/// sampling factors: 0 → `H=2 V=1` (4:2:2), 1 → `H=2 V=2` (4:2:0).
fn build_headers(
    width: u16,
    height: u16,
    base_type: u8,
    quant: &QuantPair,
    dri: u16,
) -> Result<Vec<u8>> {
    let (luma_h, luma_v) = match base_type {
        0 => (2u8, 1u8),
        1 => (2u8, 2u8),
        other => {
            return Err(Error::unsupported(format!(
                "RTP/JPEG: type {other} is not a well-known fixed mapping (only 0/1 + 64/65)"
            )));
        }
    };

    let mut out = Vec::with_capacity(700);
    // SOI.
    out.push(0xFF);
    out.push(markers::SOI);
    // Two DQT tables (luma id 0, chroma id 1).
    write_dqt8(&mut out, 0, &quant.luma);
    write_dqt8(&mut out, 1, &quant.chroma);
    // SOF0.
    write_sof0(&mut out, width, height, luma_h, luma_v);
    // The four Annex K typical Huffman tables (RFC 2435 Appendix B uses
    // these same Annex K.3 tables).
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_dht(&mut out, 0, 1, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    write_dht(&mut out, 1, 1, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);
    // DRI before SOS when restart markers are present.
    if dri != 0 {
        write_dri(&mut out, dri);
    }
    // SOS.
    write_sos(&mut out);
    Ok(out)
}

/// State accumulated across the fragments of one RTP/JPEG frame.
struct FrameState {
    /// Frame parameters captured from the first-seen fragment; all
    /// fragments of the same frame MUST agree on these (RFC 2435 §3.1).
    main: MainHeader,
    /// Restart interval (0 = none), captured from the §3.1.7 header.
    dri: u16,
    /// In-band quantization tables, present when Q >= 128.
    quant: Option<QuantPair>,
    /// Reassembled scan bytes, indexed by fragment offset.
    scan: Vec<u8>,
    /// Highest `offset + len` written so far (for gap detection).
    scan_len: usize,
}

/// Streaming RFC 2435 depacketizer. Feed it the RTP/JPEG payload of each
/// packet (RTP header already stripped) plus the RTP marker bit; it emits
/// a reconstructed JPEG interchange stream once the marker-bit fragment
/// has been supplied.
///
/// ```
/// use oxideav_mjpeg::rtp::{JpegDepacketizer, Progress};
/// let mut dp = JpegDepacketizer::new();
/// // `payload` is one RTP packet body with the 12-byte RTP header removed;
/// // `marker` is the RTP marker bit (set on the final fragment of a frame).
/// # let payload: &[u8] = &[];
/// # let marker = false;
/// match dp.push(payload, marker) {
///     Ok(Progress::NeedMore) => { /* await further fragments */ }
///     Ok(Progress::Frame(jpeg)) => { /* `jpeg` is a complete SOI..EOI stream */ let _ = jpeg; }
///     Err(_) => { /* malformed packet */ }
/// }
/// ```
#[derive(Default)]
pub struct JpegDepacketizer {
    state: Option<FrameState>,
}

impl JpegDepacketizer {
    /// Create an empty depacketizer.
    pub fn new() -> Self {
        Self { state: None }
    }

    /// Discard any partially-reassembled frame (e.g. after a detected
    /// packet loss the caller cannot recover from).
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Feed one RTP/JPEG payload. `marker` is the RTP marker bit, set on
    /// the last packet of a frame.
    ///
    /// Returns [`Progress::Frame`] with a complete JPEG interchange
    /// stream when the marker bit closes a frame, or
    /// [`Progress::NeedMore`] when more fragments are required.
    pub fn push(&mut self, payload: &[u8], marker: bool) -> Result<Progress> {
        let main = parse_main_header(payload)?;
        let mut cursor = MAIN_HDR_LEN;

        // Optional Restart Marker header (types 64..=127).
        let dri = if main.has_restart() {
            let rst = parse_restart_header(&payload[cursor..])?;
            cursor += RST_HDR_LEN;
            rst.restart_interval
        } else {
            0
        };

        // Optional Quantization Table header (Q >= 128, first fragment).
        let inband_quant = if main.q >= 128 && main.fragment_offset == 0 {
            let (qp, consumed) = parse_qtable_header(&payload[cursor..], main.q)?;
            cursor += consumed;
            qp
        } else {
            None
        };

        // The remainder of the payload is entropy-coded scan data.
        let scan = &payload[cursor..];

        // Begin a new frame on a zero-offset fragment, or when no frame
        // is in progress.
        if main.fragment_offset == 0 || self.state.is_none() {
            // A non-zero offset with no in-progress frame means we joined
            // mid-frame (e.g. the zero-offset packet was lost). Without
            // the frame's header parameters we cannot reconstruct it.
            if main.fragment_offset != 0 && self.state.is_none() {
                return Err(Error::invalid(
                    "RTP/JPEG: first fragment has non-zero offset (lost frame start)",
                ));
            }
            self.state = Some(FrameState {
                main,
                dri,
                quant: inband_quant,
                scan: Vec::new(),
                scan_len: 0,
            });
        }

        let st = self.state.as_mut().expect("frame state initialised above");

        // §3.1: all header fields except Fragment Offset MUST match
        // across the fragments of one frame.
        if st.main.typ != main.typ
            || st.main.q != main.q
            || st.main.width != main.width
            || st.main.height != main.height
        {
            return Err(Error::invalid(
                "RTP/JPEG: fragment header disagrees with frame start",
            ));
        }

        // Place this fragment's scan bytes at its offset.
        let off = main.fragment_offset as usize;
        let end = off + scan.len();
        if end > st.scan.len() {
            st.scan.resize(end, 0);
        }
        st.scan[off..end].copy_from_slice(scan);
        st.scan_len = st.scan_len.max(end);

        if !marker {
            return Ok(Progress::NeedMore);
        }

        // Marker bit set: the frame is complete. Reconstruct it.
        let st = self.state.take().expect("frame state present at marker");
        let jpeg = self.assemble(st)?;
        Ok(Progress::Frame(jpeg))
    }

    /// Build the full JPEG interchange stream from a completed frame.
    fn assemble(&self, st: FrameState) -> Result<Vec<u8>> {
        let quant = match st.quant {
            Some(q) => q,
            None => {
                if st.main.q >= 128 {
                    // Q >= 128 with no in-band tables means out-of-band
                    // table negotiation, which this depacketizer does not
                    // support — there is no JPEG to build without tables.
                    return Err(Error::unsupported(
                        "RTP/JPEG: Q >= 128 without an in-band quantization table",
                    ));
                }
                if st.main.q == 0 {
                    return Err(Error::invalid("RTP/JPEG: Q = 0 is reserved"));
                }
                tables_from_q(st.main.q)
            }
        };

        let mut out = build_headers(
            st.main.width,
            st.main.height,
            st.main.base_type(),
            &quant,
            st.dri,
        )?;
        // Entropy-coded scan (already byte-stuffed on the wire per
        // §3.1.9), then the EOI terminator the wire stream omits.
        out.extend_from_slice(&st.scan[..st.scan_len]);
        out.push(0xFF);
        out.push(markers::EOI);
        Ok(out)
    }
}

/// Parse the §3.1.8 Quantization Table header plus its table data.
///
/// Returns the reconstructed luma/chroma pair (when `length > 0`) and the
/// number of bytes consumed (header + data). For the well-known types
/// 0/1 the data carries exactly two tables (one luma, one chroma).
fn parse_qtable_header(bytes: &[u8], q: u8) -> Result<(Option<QuantPair>, usize)> {
    if bytes.len() < QTBL_HDR_LEN {
        return Err(Error::invalid(
            "RTP/JPEG: truncated quantization-table header",
        ));
    }
    let precision = bytes[1];
    let length = u16::from_be_bytes([bytes[2], bytes[3]]) as usize;
    if length == 0 {
        // §4.2: a zero-length table header means the tables were sent in
        // an earlier frame (in-band) or out of band. Q == 255 with
        // length 0 is explicitly forbidden by §3.1.8.
        if q == 255 {
            return Err(Error::invalid("RTP/JPEG: Q = 255 with zero table length"));
        }
        return Ok((None, QTBL_HDR_LEN));
    }
    let data = &bytes[QTBL_HDR_LEN..];
    if length > data.len() {
        // §3.1.8: "If the Length field ... is larger than the remaining
        // number of bytes, the packet MUST be discarded."
        return Err(Error::invalid(
            "RTP/JPEG: quantization-table length exceeds payload",
        ));
    }

    // For types 0/1 there are two tables. The Precision field's
    // rightmost bit (bit 0) describes the first table, the next bit the
    // second; a set bit means 16-bit coefficients (128 bytes), clear
    // means 8-bit (64 bytes). We reconstruct the two tables that the
    // SOF0/SOS we emit reference (ids 0 and 1).
    let mut offset = 0usize;
    let mut read_table = |table_idx: usize| -> Result<[u8; 64]> {
        let is16 = precision & (1 << table_idx) != 0;
        let need = if is16 { 128 } else { 64 };
        if offset + need > length {
            return Err(Error::invalid(
                "RTP/JPEG: quantization-table data truncated",
            ));
        }
        let mut out = [0u8; 64];
        if is16 {
            // 16-bit coefficients, network order; saturate to the 8-bit
            // DQT we emit (the well-known types use 8-bit tables).
            for k in 0..64 {
                let v = u16::from_be_bytes([data[offset + k * 2], data[offset + k * 2 + 1]]);
                out[k] = v.min(255) as u8;
            }
        } else {
            out.copy_from_slice(&data[offset..offset + 64]);
        }
        offset += need;
        Ok(out)
    };

    let luma = read_table(0)?;
    let chroma = read_table(1)?;
    Ok((Some(QuantPair { luma, chroma }), QTBL_HDR_LEN + length))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal RTP/JPEG payload (main header + scan) for type 1,
    /// Q = 50, given width/height in pixels and an offset.
    fn payload_type1(width: u16, height: u16, q: u8, offset: u32, scan: &[u8]) -> Vec<u8> {
        let mut p = vec![
            0,                    // type-specific
            (offset >> 16) as u8, // fragment offset (24-bit, network order)
            (offset >> 8) as u8,
            offset as u8,
            1, // type 1 (4:2:0)
            q,
            (width / 8) as u8,  // width in 8-pixel units
            (height / 8) as u8, // height in 8-pixel units
        ];
        p.extend_from_slice(scan);
        p
    }

    #[test]
    fn parses_main_header_fields() {
        let p = payload_type1(320, 240, 50, 0, &[0xAA, 0xBB]);
        let h = parse_main_header(&p).unwrap();
        assert_eq!(h.fragment_offset, 0);
        assert_eq!(h.typ, 1);
        assert_eq!(h.q, 50);
        assert_eq!(h.width, 320);
        assert_eq!(h.height, 240);
        assert!(!h.has_restart());
        assert_eq!(h.base_type(), 1);
    }

    #[test]
    fn width_height_are_eight_pixel_units() {
        // 40 * 8 = 320, 30 * 8 = 240 (the RFC §3.1.5 example).
        let p = payload_type1(320, 240, 1, 0, &[]);
        let h = parse_main_header(&p).unwrap();
        assert_eq!(h.width, 320);
        assert_eq!(h.height, 240);
    }

    #[test]
    fn rejects_short_payload() {
        assert!(parse_main_header(&[0, 1, 2, 3]).is_err());
    }

    #[test]
    fn single_packet_frame_reconstructs_valid_jpeg() {
        let scan = vec![0x12, 0x34, 0x56, 0x78];
        let p = payload_type1(64, 64, 50, 0, &scan);
        let mut dp = JpegDepacketizer::new();
        let prog = dp.push(&p, true).unwrap();
        let jpeg = match prog {
            Progress::Frame(j) => j,
            Progress::NeedMore => panic!("expected a complete frame"),
        };
        // SOI ... EOI bookends.
        assert_eq!(&jpeg[0..2], &[0xFF, markers::SOI]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, markers::EOI]);
        // Scan bytes appear immediately before EOI.
        let scan_start = jpeg.len() - 2 - scan.len();
        assert_eq!(&jpeg[scan_start..jpeg.len() - 2], &scan[..]);
        // Two DQT, one SOF0, four DHT, one SOS marker present.
        let count = |m: u8| {
            jpeg.windows(2)
                .filter(|w| w[0] == 0xFF && w[1] == m)
                .count()
        };
        assert_eq!(count(markers::DQT), 2);
        assert_eq!(count(markers::SOF0), 1);
        assert_eq!(count(markers::DHT), 4);
        // No DRI for a non-restart type.
        assert_eq!(count(markers::DRI), 0);
    }

    #[test]
    fn type1_emits_420_sampling() {
        let p = payload_type1(64, 64, 50, 0, &[0u8; 8]);
        let mut dp = JpegDepacketizer::new();
        let jpeg = match dp.push(&p, true).unwrap() {
            Progress::Frame(j) => j,
            _ => panic!(),
        };
        // Find SOF0 and read the first component's sampling byte.
        let sof = jpeg
            .windows(2)
            .position(|w| w[0] == 0xFF && w[1] == markers::SOF0)
            .unwrap();
        // SOF0 payload: marker(2) len(2) prec(1) h(2) w(2) nc(1) then comp.
        let comp0_samp = jpeg[sof + 2 + 2 + 1 + 2 + 2 + 1 + 1];
        assert_eq!(comp0_samp, 0x22); // H=2 V=2
    }

    #[test]
    fn type0_emits_422_sampling() {
        let mut p = payload_type1(64, 64, 50, 0, &[0u8; 8]);
        p[4] = 0; // switch to type 0
        let mut dp = JpegDepacketizer::new();
        let jpeg = match dp.push(&p, true).unwrap() {
            Progress::Frame(j) => j,
            _ => panic!(),
        };
        let sof = jpeg
            .windows(2)
            .position(|w| w[0] == 0xFF && w[1] == markers::SOF0)
            .unwrap();
        let comp0_samp = jpeg[sof + 2 + 2 + 1 + 2 + 2 + 1 + 1];
        assert_eq!(comp0_samp, 0x21); // H=2 V=1
    }

    #[test]
    fn multi_fragment_reassembly_in_order() {
        let first = payload_type1(64, 64, 50, 0, &[1, 2, 3, 4]);
        let second = payload_type1(64, 64, 50, 4, &[5, 6, 7, 8]);
        let mut dp = JpegDepacketizer::new();
        assert_eq!(dp.push(&first, false).unwrap(), Progress::NeedMore);
        let jpeg = match dp.push(&second, true).unwrap() {
            Progress::Frame(j) => j,
            _ => panic!(),
        };
        let scan_start = jpeg.len() - 2 - 8;
        assert_eq!(&jpeg[scan_start..jpeg.len() - 2], &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn multi_fragment_reassembly_out_of_order() {
        // Second fragment arrives before the (zero-offset) first.
        let first = payload_type1(64, 64, 50, 0, &[1, 2, 3, 4]);
        let second = payload_type1(64, 64, 50, 4, &[5, 6, 7, 8]);
        let mut dp = JpegDepacketizer::new();
        assert_eq!(dp.push(&first, false).unwrap(), Progress::NeedMore);
        // Out-of-order within the frame is fine because we key on offset;
        // the marker bit just has to land on the last packet pushed.
        let jpeg = match dp.push(&second, true).unwrap() {
            Progress::Frame(j) => j,
            _ => panic!(),
        };
        let scan_start = jpeg.len() - 2 - 8;
        assert_eq!(&jpeg[scan_start..jpeg.len() - 2], &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn mid_frame_join_without_start_is_error() {
        let mid = payload_type1(64, 64, 50, 16, &[9, 9]);
        let mut dp = JpegDepacketizer::new();
        assert!(dp.push(&mid, false).is_err());
    }

    #[test]
    fn restart_type_emits_dri_and_consumes_header() {
        // Type 64 = 0 | restart bit. Restart Marker header follows main.
        let mut p = Vec::new();
        p.extend_from_slice(&[0, 0, 0, 0]); // tspec + offset 0
        p.push(64); // type 64
        p.push(50); // Q
        p.push(8); // width 64
        p.push(8); // height 64
                   // Restart Marker header: DRI=4, F=L=1, count=0x3FFF.
        p.extend_from_slice(&4u16.to_be_bytes());
        p.extend_from_slice(&0xFFFFu16.to_be_bytes());
        p.extend_from_slice(&[0xAA, 0xBB]); // scan
        let mut dp = JpegDepacketizer::new();
        let jpeg = match dp.push(&p, true).unwrap() {
            Progress::Frame(j) => j,
            _ => panic!(),
        };
        let dri_pos = jpeg
            .windows(2)
            .position(|w| w[0] == 0xFF && w[1] == markers::DRI)
            .expect("DRI segment present for restart type");
        // DRI payload: marker(2) len(2)=4 interval(2).
        let interval = u16::from_be_bytes([jpeg[dri_pos + 4], jpeg[dri_pos + 5]]);
        assert_eq!(interval, 4);
        // Scan bytes preserved.
        assert_eq!(&jpeg[jpeg.len() - 4..jpeg.len() - 2], &[0xAA, 0xBB]);
    }

    #[test]
    fn zero_restart_interval_rejected() {
        let mut p = Vec::new();
        p.extend_from_slice(&[0, 0, 0, 0]);
        p.push(64);
        p.push(50);
        p.push(8);
        p.push(8);
        p.extend_from_slice(&0u16.to_be_bytes()); // DRI = 0 (illegal)
        p.extend_from_slice(&0u16.to_be_bytes());
        let mut dp = JpegDepacketizer::new();
        assert!(dp.push(&p, true).is_err());
    }

    #[test]
    fn inband_quant_tables_used() {
        // Q = 200 (>=128) with in-band 8-bit tables.
        let mut p = Vec::new();
        p.extend_from_slice(&[0, 0, 0, 0]); // tspec + offset 0
        p.push(1); // type 1
        p.push(200); // Q >= 128
        p.push(8);
        p.push(8);
        // Quant Table header: mbz=0, precision=0 (both 8-bit), length=128.
        p.push(0);
        p.push(0);
        p.extend_from_slice(&128u16.to_be_bytes());
        // Two distinguishable 64-byte tables.
        let luma: Vec<u8> = (0..64).map(|i| (i as u8) + 1).collect();
        let chroma: Vec<u8> = (0..64).map(|i| 200 - i as u8).collect();
        p.extend_from_slice(&luma);
        p.extend_from_slice(&chroma);
        p.extend_from_slice(&[0xCC, 0xDD]); // scan

        let mut dp = JpegDepacketizer::new();
        let jpeg = match dp.push(&p, true).unwrap() {
            Progress::Frame(j) => j,
            _ => panic!(),
        };
        // First DQT carries `luma` verbatim in zigzag order. The header
        // tables are already zigzag-ordered, so they pass through.
        let dqt0 = jpeg
            .windows(2)
            .position(|w| w[0] == 0xFF && w[1] == markers::DQT)
            .unwrap();
        // DQT: marker(2) len(2) pqtq(1) then 64 bytes.
        let table_start = dqt0 + 5;
        assert_eq!(&jpeg[table_start..table_start + 64], &luma[..]);
    }

    #[test]
    fn q255_zero_length_rejected() {
        let mut p = Vec::new();
        p.extend_from_slice(&[0, 0, 0, 0]);
        p.push(1);
        p.push(255); // Q = 255
        p.push(8);
        p.push(8);
        p.push(0);
        p.push(0);
        p.extend_from_slice(&0u16.to_be_bytes()); // length 0 — illegal for Q=255
        let mut dp = JpegDepacketizer::new();
        assert!(dp.push(&p, true).is_err());
    }

    #[test]
    fn tables_from_q_matches_ijg_formula() {
        // Q = 50 should reproduce the K.1 / K.2 base tables (scale = 100,
        // so v = (base*100 + 50)/100 = base) saturated to 8 bits.
        let qp = tables_from_q(50);
        // Zigzag position 0 is natural index 0 = the DC term.
        assert_eq!(qp.luma[0], DEFAULT_LUMA_Q50[0].min(255) as u8);
        assert_eq!(qp.chroma[0], DEFAULT_CHROMA_Q50[0].min(255) as u8);
    }

    #[test]
    fn fragment_header_mismatch_rejected() {
        let first = payload_type1(64, 64, 50, 0, &[1, 2]);
        // Second fragment claims a different Q.
        let second = payload_type1(64, 64, 60, 2, &[3, 4]);
        let mut dp = JpegDepacketizer::new();
        assert_eq!(dp.push(&first, false).unwrap(), Progress::NeedMore);
        assert!(dp.push(&second, true).is_err());
    }

    // ---- End-to-end: encode → strip to RTP payload → depacketize →
    // decode. Proves a reconstructed stream is genuinely decodable, not
    // just structurally plausible. Gated on `registry` because the
    // encoder/decoder entry points operate on `oxideav_core` frame
    // types in that build. ----

    #[cfg(feature = "registry")]
    use crate::decoder::decode_jpeg;
    #[cfg(feature = "registry")]
    use crate::encoder::encode_jpeg;
    #[cfg(feature = "registry")]
    use oxideav_core::frame::VideoPlane;
    #[cfg(feature = "registry")]
    use oxideav_core::{PixelFormat, VideoFrame};

    /// Find the byte span `[start, end)` of the entropy-coded scan in a
    /// JPEG: everything after the SOS segment's payload up to the final
    /// EOI marker.
    #[cfg(feature = "registry")]
    fn scan_span(jpeg: &[u8]) -> (usize, usize) {
        let sos = jpeg
            .windows(2)
            .position(|w| w[0] == 0xFF && w[1] == markers::SOS)
            .expect("SOS present");
        let sos_len = u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]) as usize;
        let scan_start = sos + 2 + sos_len;
        // EOI is the final 0xFF 0xD9.
        let eoi = jpeg.len() - 2;
        assert_eq!(&jpeg[eoi..], &[0xFF, markers::EOI]);
        (scan_start, eoi)
    }

    /// Extract the 64-byte (8-bit, Pq=0) zigzag-order table from the n-th
    /// DQT segment of a JPEG.
    #[cfg(feature = "registry")]
    fn dqt_table(jpeg: &[u8], which: usize) -> [u8; 64] {
        let mut found = 0;
        let mut pos = 2usize;
        while pos + 4 < jpeg.len() {
            if jpeg[pos] == 0xFF && jpeg[pos + 1] == markers::DQT {
                if found == which {
                    let table_start = pos + 5; // marker(2) len(2) pqtq(1)
                    let mut t = [0u8; 64];
                    t.copy_from_slice(&jpeg[table_start..table_start + 64]);
                    return t;
                }
                found += 1;
                let len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
                pos += 2 + len;
            } else {
                pos += 1;
            }
        }
        panic!("DQT #{which} not found");
    }

    /// Build a flat 4:2:0 test frame with a smooth gradient.
    #[cfg(feature = "registry")]
    fn make_420_frame(w: usize, h: usize) -> VideoFrame {
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let y: Vec<u8> = (0..w * h).map(|i| ((i * 3) & 0xFF) as u8).collect();
        let cb: Vec<u8> = (0..cw * ch).map(|i| ((i * 5 + 40) & 0xFF) as u8).collect();
        let cr: Vec<u8> = (0..cw * ch).map(|i| ((i * 7 + 80) & 0xFF) as u8).collect();
        VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane { stride: w, data: y },
                VideoPlane {
                    stride: cw,
                    data: cb,
                },
                VideoPlane {
                    stride: cw,
                    data: cr,
                },
            ],
        }
    }

    #[cfg(feature = "registry")]
    #[test]
    fn end_to_end_inband_qtables_decodes() {
        // 64×64 keeps the scan small enough to fit in one fragment but
        // exercises a full multi-MCU 4:2:0 image.
        let (w, h) = (64usize, 64usize);
        let frame = make_420_frame(w, h);
        let jpeg =
            encode_jpeg(&frame, w as u32, h as u32, PixelFormat::Yuv420P, 75).expect("encode");

        // Pull the two real DQT tables and the entropy scan out of the
        // encoded stream.
        let luma = dqt_table(&jpeg, 0);
        let chroma = dqt_table(&jpeg, 1);
        let (s0, s1) = scan_span(&jpeg);
        let scan = &jpeg[s0..s1];

        // Build a type-1 RTP/JPEG payload with in-band Q tables (Q=200).
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0, 0, 0, 0]); // tspec + offset 0
        payload.push(1); // type 1 (4:2:0)
        payload.push(200); // Q >= 128 → in-band tables
        payload.push((w / 8) as u8);
        payload.push((h / 8) as u8);
        // Quantization Table header: mbz, precision=0, length=128.
        payload.push(0);
        payload.push(0);
        payload.extend_from_slice(&128u16.to_be_bytes());
        payload.extend_from_slice(&luma);
        payload.extend_from_slice(&chroma);
        payload.extend_from_slice(scan);

        let mut dp = JpegDepacketizer::new();
        let rebuilt = match dp.push(&payload, true).unwrap() {
            Progress::Frame(j) => j,
            _ => panic!("expected complete frame"),
        };

        // The reconstructed stream must decode without error and yield a
        // 3-plane 4:2:0 frame of the right dimensions.
        let decoded = decode_jpeg(&rebuilt, None).expect("decode reconstructed RTP/JPEG");
        assert_eq!(decoded.planes.len(), 3);
        assert_eq!(decoded.planes[0].data.len(), w * h);
        assert_eq!(decoded.planes[1].data.len(), w.div_ceil(2) * h.div_ceil(2));
    }

    #[cfg(feature = "registry")]
    #[test]
    fn end_to_end_q_field_tables_decodes() {
        // Same as above but rely on the Q field (1..=99) to regenerate
        // the tables, exercising the IJG-scale reconstruction path.
        // Encode at quality 50 so the regenerated tables match the
        // scan's quantization exactly.
        let (w, h) = (64usize, 48usize);
        let frame = make_420_frame(w, h);
        let jpeg =
            encode_jpeg(&frame, w as u32, h as u32, PixelFormat::Yuv420P, 50).expect("encode");
        let (s0, s1) = scan_span(&jpeg);
        let scan = &jpeg[s0..s1];

        let payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&[0, 0, 0, 0]);
            p.push(1); // type 1
            p.push(50); // Q = 50 → IJG-scaled tables
            p.push((w / 8) as u8);
            p.push((h / 8) as u8);
            p.extend_from_slice(scan);
            p
        };

        let mut dp = JpegDepacketizer::new();
        let rebuilt = match dp.push(&payload, true).unwrap() {
            Progress::Frame(j) => j,
            _ => panic!(),
        };
        let decoded = decode_jpeg(&rebuilt, None).expect("decode Q-field RTP/JPEG");
        assert_eq!(decoded.planes.len(), 3);
        assert_eq!(decoded.planes[0].data.len(), w * h);
    }
}
