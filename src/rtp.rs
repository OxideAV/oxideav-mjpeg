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
//! Table header (§3.1.8). A *static* Q value (128..=254) may carry its
//! tables only once and then omit them (`Length = 0`) on later frames; the
//! depacketizer caches the tables per Q value across [`JpegDepacketizer`]
//! frames and reuses them when a later frame's header is empty (§4.2).
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
#[derive(Clone, Copy)]
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
    /// Most-recent in-band quantization tables for a *static* Q value
    /// (128..=254), with the Q value they belong to. §4.2: in-band tables
    /// for a static Q "need not be sent with every frame"; later frames
    /// carry a Quantization Table header with `Length = 0` and the receiver
    /// reuses the tables it cached the first time it saw that Q. Q = 255 is
    /// dynamic and never populates this cache (the spec forbids depending on
    /// a previous frame's tables for Q = 255).
    cached_tables: Option<(u8, QuantPair)>,
}

impl JpegDepacketizer {
    /// Create an empty depacketizer.
    pub fn new() -> Self {
        Self {
            state: None,
            cached_tables: None,
        }
    }

    /// Discard any partially-reassembled frame (e.g. after a detected
    /// packet loss the caller cannot recover from).
    ///
    /// This drops only the in-progress reassembly buffer; the cached static
    /// quantization tables (§4.2) are retained so frames that follow can
    /// still decode. Use [`JpegDepacketizer::new`] for a fully fresh state.
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
            // §4.2: a static Q (128..=254) carrying tables in-band caches
            // them so later frames may omit them (Length = 0). A new static
            // Q replaces any previously cached pair; Q = 255 (dynamic) never
            // touches the cache.
            if let Some(qp) = qp {
                if (128..=254).contains(&main.q) {
                    self.cached_tables = Some((main.q, qp));
                }
            }
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
                    // No in-band tables this frame. For a static Q
                    // (128..=254) the tables may have arrived on an earlier
                    // frame (§4.2 "need not be sent with every frame"); reuse
                    // the cached pair if it belongs to this exact Q value.
                    // Q = 255 is dynamic and the cache is never consulted —
                    // a Q = 255 / Length = 0 packet is already rejected during
                    // header parsing, and a genuinely table-less Q = 255 frame
                    // is undecodable.
                    match self.cached_tables {
                        Some((cached_q, qp)) if cached_q == st.main.q => qp,
                        _ => {
                            return Err(Error::unsupported(
                                "RTP/JPEG: Q >= 128 without in-band tables and none cached \
                                 for this Q (out-of-band negotiation unsupported)",
                            ));
                        }
                    }
                } else if st.main.q == 0 {
                    return Err(Error::invalid("RTP/JPEG: Q = 0 is reserved"));
                } else {
                    tables_from_q(st.main.q)
                }
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

// ============================================================================
// Packetization (encode side) — the inverse of the depacketizer above.
// ============================================================================

/// One RTP/JPEG payload produced by the packetizer.
///
/// This is the RTP *payload* only — the bytes the caller places immediately
/// after the 12-byte RTP fixed header (and any CSRC / extension words). The
/// caller owns the RTP transport: it assigns the sequence number, the 90 kHz
/// timestamp (identical across all fragments of one frame, §3), and sets the
/// RTP marker bit when [`JpegPacket::marker`] is true.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JpegPacket {
    /// The RTP/JPEG payload bytes: main JPEG header + optional Restart Marker
    /// header + optional Quantization Table header (first fragment only) +
    /// a slice of the entropy-coded scan.
    pub payload: Vec<u8>,
    /// True on the last fragment of the frame — the caller MUST set the RTP
    /// marker bit on the packet carrying this payload (§3).
    pub marker: bool,
}

/// How the packetizer carries the frame's quantization tables on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QMode {
    /// Carry a Q value 1..=99 in the Q field; the receiver regenerates the
    /// tables with the §4.2 IJG scale formula. Use this only when the JPEG's
    /// DQT tables actually equal the IJG-scaled Annex K tables for this Q —
    /// otherwise the receiver's tables will not match the scan and the image
    /// will be wrong. Suitable for streams this crate's encoder produced at
    /// the same quality factor.
    Quality(u8),
    /// Carry the JPEG's own two DQT tables in-band via a §3.1.8 Quantization
    /// Table header on the first fragment, with the Q field set to `q`
    /// (128..=255). `q = 255` means "tables may change every frame"; 128..=254
    /// means "static mapping". This is the safe default for an arbitrary JPEG
    /// whose tables are not a known IJG quality.
    InBand(u8),
}

/// Parsed pieces of a complete baseline JPEG that the packetizer needs to
/// build the RTP/JPEG main header and (optionally) the in-band tables.
struct JpegParts {
    width: u16,
    height: u16,
    /// Base type 0 (4:2:2, luma `H=2 V=1`) or 1 (4:2:0, luma `H=2 V=2`).
    base_type: u8,
    /// Luma quantization table (id 0) in zigzag order, as carried in DQT.
    qt_luma: [u8; 64],
    /// Chroma quantization table (id 1) in zigzag order.
    qt_chroma: [u8; 64],
    /// DRI value (0 = no restart markers).
    dri: u16,
    /// Byte span `[start, end)` of the entropy-coded scan: everything after
    /// the SOS segment up to (and excluding) the final EOI marker.
    scan_start: usize,
    scan_end: usize,
}

/// Read a big-endian u16 segment length at `pos` (the two bytes following a
/// marker), returning the length value.
fn seg_len(jpeg: &[u8], pos: usize) -> Result<usize> {
    if pos + 2 > jpeg.len() {
        return Err(Error::invalid(
            "RTP/JPEG packetize: truncated segment length",
        ));
    }
    Ok(u16::from_be_bytes([jpeg[pos], jpeg[pos + 1]]) as usize)
}

/// Parse the subset of a complete JPEG interchange stream that RTP/JPEG can
/// carry: a baseline (SOF0/SOF1) three-component YUV image with luma `H=2`
/// and unit-sampled chroma, exactly the well-known §4.1 type-0/1 layout.
///
/// Anything outside that envelope (progressive, lossless, grayscale, CMYK,
/// non-2:x luma sampling, 16-bit DQT) returns `Unsupported` — RTP/JPEG has no
/// way to signal it with the well-known types.
fn parse_jpeg(jpeg: &[u8]) -> Result<JpegParts> {
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != markers::SOI {
        return Err(Error::invalid("RTP/JPEG packetize: missing SOI"));
    }
    let mut pos = 2usize;
    let mut width = 0u16;
    let mut height = 0u16;
    let mut base_type: Option<u8> = None;
    let mut dqt: [Option<[u8; 64]>; 4] = [None, None, None, None];
    let mut dri = 0u16;

    loop {
        // Markers may be preceded by fill 0xFF bytes (§B.1.1.2).
        while pos < jpeg.len() && jpeg[pos] == 0xFF && pos + 1 < jpeg.len() && jpeg[pos + 1] == 0xFF
        {
            pos += 1;
        }
        if pos + 1 >= jpeg.len() {
            return Err(Error::invalid("RTP/JPEG packetize: ran off end before SOS"));
        }
        if jpeg[pos] != 0xFF {
            return Err(Error::invalid("RTP/JPEG packetize: expected marker"));
        }
        let marker = jpeg[pos + 1];
        pos += 2;

        match marker {
            markers::SOF0 | markers::SOF1 => {
                let len = seg_len(jpeg, pos)?;
                // T.81 §B.2.2: SOF length excludes the marker itself; minimum
                // is 8 + 3*Nf. A 3-component frame therefore needs at least
                // 17 bytes inside the segment. Bound `len` first so the
                // `len - 2` payload arithmetic cannot underflow when an
                // adversarial header carries a tiny length.
                if len < 8 {
                    return Err(Error::invalid("RTP/JPEG packetize: truncated SOF"));
                }
                let body = pos + 2;
                if body + (len - 2) > jpeg.len() {
                    return Err(Error::invalid("RTP/JPEG packetize: truncated SOF"));
                }
                let precision = jpeg[body];
                if precision != 8 {
                    return Err(Error::unsupported(
                        "RTP/JPEG packetize: only 8-bit precision is carryable",
                    ));
                }
                height = u16::from_be_bytes([jpeg[body + 1], jpeg[body + 2]]);
                width = u16::from_be_bytes([jpeg[body + 3], jpeg[body + 4]]);
                let nc = jpeg[body + 5];
                if nc != 3 {
                    return Err(Error::unsupported(
                        "RTP/JPEG packetize: only three-component YUV is carryable",
                    ));
                }
                // Component records: id(1) H|V(1) Tq(1), three of them.
                // The segment must therefore carry 8 + 3*3 = 17 bytes after
                // its 2-byte length field; refuse anything shorter before
                // indexing into the component records.
                if len < 8 + 3 * (nc as usize) {
                    return Err(Error::invalid(
                        "RTP/JPEG packetize: truncated SOF components",
                    ));
                }
                let c0 = body + 6;
                let y_samp = jpeg[c0 + 1];
                let cb_samp = jpeg[c0 + 3 + 1];
                let cr_samp = jpeg[c0 + 6 + 1];
                let (yh, yv) = (y_samp >> 4, y_samp & 0x0F);
                // Chroma must be unit-sampled per the well-known mappings.
                if cb_samp != 0x11 || cr_samp != 0x11 {
                    return Err(Error::unsupported(
                        "RTP/JPEG packetize: chroma must be 1x1 sampled (type 0/1)",
                    ));
                }
                base_type = Some(match (yh, yv) {
                    (2, 1) => 0,
                    (2, 2) => 1,
                    _ => return Err(Error::unsupported(
                        "RTP/JPEG packetize: luma sampling must be 2x1 (type 0) or 2x2 (type 1)",
                    )),
                });
                pos = body + len - 2;
            }
            markers::DQT => {
                let len = seg_len(jpeg, pos)?;
                if len < 2 {
                    return Err(Error::invalid("RTP/JPEG packetize: truncated DQT"));
                }
                let body = pos + 2;
                let end = body + (len - 2);
                if end > jpeg.len() {
                    return Err(Error::invalid("RTP/JPEG packetize: truncated DQT"));
                }
                let mut i = body;
                while i < end {
                    let pq_tq = jpeg[i];
                    let pq = pq_tq >> 4;
                    let tq = (pq_tq & 0x0F) as usize;
                    i += 1;
                    if pq != 0 {
                        return Err(Error::unsupported(
                            "RTP/JPEG packetize: 16-bit DQT not carryable in 8-bit type 0/1",
                        ));
                    }
                    if tq >= 4 || i + 64 > end {
                        return Err(Error::invalid("RTP/JPEG packetize: bad DQT entry"));
                    }
                    let mut t = [0u8; 64];
                    t.copy_from_slice(&jpeg[i..i + 64]);
                    dqt[tq] = Some(t);
                    i += 64;
                }
                pos = end;
            }
            markers::DRI => {
                let len = seg_len(jpeg, pos)?;
                if len != 4 || pos + 4 > jpeg.len() {
                    return Err(Error::invalid("RTP/JPEG packetize: bad DRI"));
                }
                dri = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]);
                pos += len;
            }
            markers::SOS => {
                let len = seg_len(jpeg, pos)?;
                if len < 2 || pos + len > jpeg.len() {
                    return Err(Error::invalid("RTP/JPEG packetize: truncated SOS"));
                }
                let scan_start = pos + len;
                // Find the EOI that terminates the entropy-coded scan. Skip
                // 0x00 stuffing and RSTn markers inside the scan.
                let mut s = scan_start;
                let scan_end = loop {
                    if s + 1 >= jpeg.len() {
                        return Err(Error::invalid("RTP/JPEG packetize: scan has no EOI"));
                    }
                    if jpeg[s] == 0xFF {
                        let m = jpeg[s + 1];
                        if m == markers::EOI {
                            break s;
                        }
                        // A second 0xFF is a fill byte (§B.1.1.2): advance one
                        // and re-examine, so 0xFF 0xFF … D9 still finds EOI.
                        if m == 0xFF {
                            s += 1;
                            continue;
                        }
                        // 0xFF00 stuffing and RSTn stay part of the scan.
                        if m == 0x00 || markers::is_rst(m) {
                            s += 2;
                            continue;
                        }
                        // Any other marker terminates the scan (shouldn't
                        // happen for the single-scan baseline images we carry).
                        break s;
                    }
                    s += 1;
                };
                let base_type = base_type
                    .ok_or_else(|| Error::invalid("RTP/JPEG packetize: SOS before SOF"))?;
                let qt_luma = dqt[0]
                    .ok_or_else(|| Error::invalid("RTP/JPEG packetize: missing luma DQT (id 0)"))?;
                let qt_chroma = dqt[1].ok_or_else(|| {
                    Error::invalid("RTP/JPEG packetize: missing chroma DQT (id 1)")
                })?;
                return Ok(JpegParts {
                    width,
                    height,
                    base_type,
                    qt_luma,
                    qt_chroma,
                    dri,
                    scan_start,
                    scan_end,
                });
            }
            markers::SOF2 | markers::SOF3 | markers::SOF9 => {
                return Err(Error::unsupported(
                    "RTP/JPEG packetize: only baseline SOF0/SOF1 is carryable",
                ));
            }
            markers::EOI => {
                return Err(Error::invalid("RTP/JPEG packetize: EOI before SOS"));
            }
            // Skip every other length-prefixed segment (APPn, COM, DHT, …):
            // RTP/JPEG infers the Huffman tables from the type, and the
            // JFIF/APP/COM metadata is not carried on the wire.
            _ if markers::is_rst(marker) => { /* stray RSTn outside scan — skip */ }
            _ => {
                let len = seg_len(jpeg, pos)?;
                // A well-formed length-prefixed segment carries its own
                // length-field bytes inside `len`, so `len < 2` is malformed
                // (and would also fail to advance `pos`, producing an
                // infinite loop on adversarial input).
                if len < 2 || pos + len > jpeg.len() {
                    return Err(Error::invalid(
                        "RTP/JPEG packetize: truncated/oversized segment",
                    ));
                }
                pos += len;
            }
        }
    }
}

/// Validate that `width`/`height` fit the §3.1.5–3.1.6 8-pixel-unit fields
/// (max 2040) and return them divided by 8.
fn dim_units(width: u16, height: u16) -> Result<(u8, u8)> {
    if width == 0 || height == 0 || width > 2040 || height > 2040 {
        return Err(Error::unsupported(
            "RTP/JPEG packetize: dimensions must be 8..=2040 px (8-pixel-unit wire field)",
        ));
    }
    // The wire field counts 8-pixel units; round up so a non-multiple-of-8
    // image still spans its full MCU grid (the receiver clips to width/height
    // from the reconstructed SOF0, which carries the exact pixel size).
    Ok((width.div_ceil(8) as u8, height.div_ceil(8) as u8))
}

/// Packetize a complete baseline JPEG interchange stream into RFC 2435
/// RTP/JPEG payloads — the inverse of [`JpegDepacketizer`].
///
/// `jpeg` is one SOI..EOI baseline (SOF0/SOF1) three-component YUV image with
/// luma sampling `2x1` (→ type 0, 4:2:2) or `2x2` (→ type 1, 4:2:0) and
/// unit-sampled chroma. `max_payload` is the largest RTP/JPEG payload the
/// caller's MTU allows (header bytes included; e.g. 1400 for a typical
/// Ethernet path). `qmode` selects how the quantization tables travel.
///
/// The returned `Vec` holds the fragments in order: the first has fragment
/// offset 0 (and carries the in-band Quantization Table header when
/// `qmode` is [`QMode::InBand`]); the last has `marker == true`. The scan is
/// fragmented on arbitrary byte boundaries unless the image uses restart
/// markers, in which case fragments are kept whole (single packet) because
/// this entry point does not split a restart-aligned scan across packets.
///
/// Feeding the produced payloads (with their `marker` flags) back into a
/// [`JpegDepacketizer`] reconstructs a JPEG that decodes to the same image.
pub fn packetize(jpeg: &[u8], max_payload: usize, qmode: QMode) -> Result<Vec<JpegPacket>> {
    let parts = parse_jpeg(jpeg)?;
    let (w_units, h_units) = dim_units(parts.width, parts.height)?;

    // Resolve the Q field and any in-band Quantization Table header bytes.
    let (q_field, qtable_hdr): (u8, Vec<u8>) = match qmode {
        QMode::Quality(q) => {
            if !(1..=99).contains(&q) {
                return Err(Error::invalid(
                    "RTP/JPEG packetize: QMode::Quality must be 1..=99",
                ));
            }
            (q, Vec::new())
        }
        QMode::InBand(q) => {
            if q < 128 {
                return Err(Error::invalid(
                    "RTP/JPEG packetize: QMode::InBand Q must be 128..=255",
                ));
            }
            // §3.1.8 Quantization Table header: MBZ(1) Precision(1) Length(2)
            // then two 8-bit tables (luma id 0, chroma id 1) in zigzag order.
            let mut h = Vec::with_capacity(QTBL_HDR_LEN + 128);
            h.push(0); // MBZ
            h.push(0); // Precision: both tables 8-bit
            h.extend_from_slice(&128u16.to_be_bytes()); // Length = two 64-byte tables
            h.extend_from_slice(&parts.qt_luma);
            h.extend_from_slice(&parts.qt_chroma);
            (q, h)
        }
    };

    let typ = if parts.dri != 0 {
        parts.base_type | TYPE_RESTART_BIT
    } else {
        parts.base_type
    };

    // Per-fragment header size: main(8) + restart(4 if any) + qtable header
    // (first fragment only). Reserve room so at least one scan byte fits.
    let rst_len = if parts.dri != 0 { RST_HDR_LEN } else { 0 };
    let first_hdr = MAIN_HDR_LEN + rst_len + qtable_hdr.len();
    let cont_hdr = MAIN_HDR_LEN + rst_len;
    if max_payload <= first_hdr || max_payload <= cont_hdr {
        return Err(Error::invalid(
            "RTP/JPEG packetize: max_payload too small for the headers",
        ));
    }

    let scan = &jpeg[parts.scan_start..parts.scan_end];
    let mut packets = Vec::new();
    let mut offset = 0usize;
    let mut first = true;

    loop {
        let hdr_room = if first { first_hdr } else { cont_hdr };
        let chunk = (max_payload - hdr_room).min(scan.len() - offset);
        let is_last = offset + chunk >= scan.len();

        let mut payload = Vec::with_capacity(hdr_room + chunk);
        // Main JPEG header (§3.1).
        payload.push(0); // Type-specific (progressive frame).
        payload.push((offset >> 16) as u8);
        payload.push((offset >> 8) as u8);
        payload.push(offset as u8);
        payload.push(typ);
        payload.push(q_field);
        payload.push(w_units);
        payload.push(h_units);
        // Restart Marker header (§3.1.7) when restart markers are present.
        if parts.dri != 0 {
            payload.extend_from_slice(&parts.dri.to_be_bytes());
            // Whole frame reassembled before decode: F=L=1, count=0x3FFF.
            payload.extend_from_slice(&0xFFFFu16.to_be_bytes());
        }
        // Quantization Table header on the first fragment only (§3.1.4).
        if first {
            payload.extend_from_slice(&qtable_hdr);
        }
        payload.extend_from_slice(&scan[offset..offset + chunk]);

        packets.push(JpegPacket {
            payload,
            marker: is_last,
        });

        offset += chunk;
        first = false;
        if is_last {
            break;
        }
    }

    Ok(packets)
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

    /// Build a type-1 RTP/JPEG single-packet payload for a Q >= 128 value.
    /// When `tables` is `Some((luma, chroma))` an in-band Quantization Table
    /// header carrying both 8-bit tables is emitted; when `None` a header
    /// with `Length = 0` (no tables this frame) is emitted instead.
    fn payload_q_inband(q: u8, tables: Option<(&[u8; 64], &[u8; 64])>, scan: &[u8]) -> Vec<u8> {
        let mut p = vec![0, 0, 0, 0, 1, q, 8, 8]; // tspec+off0, type 1, Q, 64x64
        p.push(0); // MBZ
        p.push(0); // Precision: both 8-bit
        match tables {
            Some((luma, chroma)) => {
                p.extend_from_slice(&128u16.to_be_bytes()); // Length = 2*64
                p.extend_from_slice(luma);
                p.extend_from_slice(chroma);
            }
            None => p.extend_from_slice(&0u16.to_be_bytes()), // Length = 0
        }
        p.extend_from_slice(scan);
        p
    }

    /// Read the n-th DQT segment's 64-byte (8-bit) table out of a JPEG built
    /// by the depacketizer.
    fn dqt_of(jpeg: &[u8], which: usize) -> [u8; 64] {
        let mut found = 0;
        let mut pos = 2usize;
        while pos + 4 < jpeg.len() {
            if jpeg[pos] == 0xFF && jpeg[pos + 1] == markers::DQT {
                if found == which {
                    let mut t = [0u8; 64];
                    t.copy_from_slice(&jpeg[pos + 5..pos + 5 + 64]);
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

    #[test]
    fn static_q_tables_cached_then_reused_when_omitted() {
        // §4.2: in-band tables for a static Q (128..=254) "need not be sent
        // with every frame". Frame 1 carries them; frame 2 sends Length = 0
        // and must reuse the cached pair.
        let luma: [u8; 64] = std::array::from_fn(|i| (i as u8) + 1);
        let chroma: [u8; 64] = std::array::from_fn(|i| 200 - i as u8);

        let mut dp = JpegDepacketizer::new();
        // Frame 1: tables present.
        let p1 = payload_q_inband(200, Some((&luma, &chroma)), &[0x01, 0x02]);
        let j1 = match dp.push(&p1, true).unwrap() {
            Progress::Frame(j) => j,
            _ => panic!("frame 1"),
        };
        assert_eq!(dqt_of(&j1, 0), luma);
        assert_eq!(dqt_of(&j1, 1), chroma);

        // Frame 2: same static Q, Length = 0 — the cached tables reappear.
        let p2 = payload_q_inband(200, None, &[0x03, 0x04]);
        let j2 = match dp.push(&p2, true).unwrap() {
            Progress::Frame(j) => j,
            _ => panic!("frame 2"),
        };
        assert_eq!(dqt_of(&j2, 0), luma, "frame 2 reuses cached luma table");
        assert_eq!(dqt_of(&j2, 1), chroma, "frame 2 reuses cached chroma table");
        // Frame 2's scan is its own, not frame 1's.
        assert_eq!(&j2[j2.len() - 4..j2.len() - 2], &[0x03, 0x04]);
    }

    #[test]
    fn static_q_length_zero_without_prior_tables_errors() {
        // A receiver that joins after the table-bearing frame has no cache;
        // §4.2 acknowledges it "will be unable to properly decode frames
        // from the time they start up until they receive the tables".
        let mut dp = JpegDepacketizer::new();
        let p = payload_q_inband(200, None, &[0xAA, 0xBB]);
        assert!(dp.push(&p, true).is_err());
    }

    #[test]
    fn cache_is_keyed_on_the_exact_q_value() {
        // Tables cached for Q = 200 must not satisfy a Length = 0 frame that
        // advertises a different static Q (each static Q maps to its own
        // table set, §3.1.8 / §4.2).
        let luma: [u8; 64] = std::array::from_fn(|i| (i as u8) + 1);
        let chroma: [u8; 64] = std::array::from_fn(|i| 200 - i as u8);
        let mut dp = JpegDepacketizer::new();
        dp.push(
            &payload_q_inband(200, Some((&luma, &chroma)), &[0x01]),
            true,
        )
        .unwrap();
        // Different static Q, no tables, no cache for *this* Q → error.
        let other = payload_q_inband(201, None, &[0x02]);
        assert!(dp.push(&other, true).is_err());
    }

    #[test]
    fn q255_does_not_populate_the_static_cache() {
        // Q = 255 is dynamic: its tables MUST NOT be cached for reuse. After
        // a Q = 255 frame, a static-Q Length = 0 frame still has no cache.
        let luma: [u8; 64] = std::array::from_fn(|i| (i as u8) + 1);
        let chroma: [u8; 64] = std::array::from_fn(|i| 200 - i as u8);
        let mut dp = JpegDepacketizer::new();
        // A Q = 255 frame with tables decodes, but caches nothing.
        dp.push(
            &payload_q_inband(255, Some((&luma, &chroma)), &[0x01]),
            true,
        )
        .unwrap();
        // A later static Q = 200 with Length = 0 finds an empty cache.
        let p = payload_q_inband(200, None, &[0x02]);
        assert!(dp.push(&p, true).is_err());
    }

    #[test]
    fn reset_keeps_the_table_cache() {
        // reset() drops only the in-progress reassembly buffer; the static
        // table cache survives so subsequent frames still decode.
        let luma: [u8; 64] = std::array::from_fn(|i| (i as u8) + 1);
        let chroma: [u8; 64] = std::array::from_fn(|i| 200 - i as u8);
        let mut dp = JpegDepacketizer::new();
        // Cache tables via a complete frame.
        dp.push(
            &payload_q_inband(200, Some((&luma, &chroma)), &[0x01]),
            true,
        )
        .unwrap();
        // Begin a second frame, then abandon its partial reassembly.
        assert_eq!(
            dp.push(&payload_q_inband(200, None, &[0x02]), false)
                .unwrap(),
            Progress::NeedMore
        );
        dp.reset();
        // A fresh Length = 0 frame still reuses the surviving cache.
        let j = match dp
            .push(&payload_q_inband(200, None, &[0x03]), true)
            .unwrap()
        {
            Progress::Frame(j) => j,
            _ => panic!(),
        };
        assert_eq!(dqt_of(&j, 0), luma);
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

    // ---- Packetizer (encode side) ----

    /// Build a complete baseline JPEG by hand: SOI, two DQT (ids 0/1), SOF0
    /// (3-component, given luma sampling, chroma 1x1), optional DRI, SOS, a
    /// short stuffed scan, EOI. This avoids depending on the real encoder for
    /// the non-registry tests.
    fn handmade_jpeg(width: u16, height: u16, luma_samp: u8, dri: u16, scan: &[u8]) -> Vec<u8> {
        let mut j = vec![0xFF, markers::SOI];
        // DQT id 0 (luma): 1..=64.
        j.extend_from_slice(&[0xFF, markers::DQT, 0x00, 67, 0x00]);
        j.extend((0..64).map(|i| (i as u8) + 1));
        // DQT id 1 (chroma): 64..=1.
        j.extend_from_slice(&[0xFF, markers::DQT, 0x00, 67, 0x01]);
        j.extend((0..64).map(|i| 64 - i as u8));
        // SOF0.
        j.extend_from_slice(&[0xFF, markers::SOF0, 0x00, 17, 8]);
        j.extend_from_slice(&height.to_be_bytes());
        j.extend_from_slice(&width.to_be_bytes());
        j.push(3);
        j.extend_from_slice(&[1, luma_samp, 0]); // Y
        j.extend_from_slice(&[2, 0x11, 1]); // Cb
        j.extend_from_slice(&[3, 0x11, 1]); // Cr
        if dri != 0 {
            j.extend_from_slice(&[0xFF, markers::DRI, 0x00, 0x04]);
            j.extend_from_slice(&dri.to_be_bytes());
        }
        // SOS: 3 components, Ss=0 Se=63 Ah|Al=0.
        j.extend_from_slice(&[
            0xFF,
            markers::SOS,
            0x00,
            12,
            3,
            1,
            0x00,
            2,
            0x11,
            3,
            0x11,
            0,
            63,
            0,
        ]);
        j.extend_from_slice(scan);
        j.extend_from_slice(&[0xFF, markers::EOI]);
        j
    }

    #[test]
    fn packetize_single_fragment_inband() {
        let scan = vec![0x11, 0x22, 0x33, 0x44];
        let jpeg = handmade_jpeg(64, 64, 0x22, 0, &scan); // 4:2:0 → type 1
        let pkts = packetize(&jpeg, 1400, QMode::InBand(255)).unwrap();
        assert_eq!(pkts.len(), 1);
        assert!(pkts[0].marker);
        let p = &pkts[0].payload;
        let h = parse_main_header(p).unwrap();
        assert_eq!(h.fragment_offset, 0);
        assert_eq!(h.typ, 1);
        assert_eq!(h.q, 255);
        assert_eq!(h.width, 64);
        assert_eq!(h.height, 64);
        assert!(!h.has_restart());
        // Quantization Table header (MBZ, Precision=0, Length=128) + two tables.
        let qh = &p[MAIN_HDR_LEN..];
        assert_eq!(u16::from_be_bytes([qh[2], qh[3]]), 128);
        assert_eq!(&qh[QTBL_HDR_LEN..QTBL_HDR_LEN + 64][..4], &[1, 2, 3, 4]);
        // Scan trails after the table header.
        assert_eq!(&p[p.len() - 4..], &scan[..]);
    }

    #[test]
    fn packetize_type0_from_422_sampling() {
        let jpeg = handmade_jpeg(64, 64, 0x21, 0, &[0xAB; 8]); // 4:2:2 → type 0
        let pkts = packetize(&jpeg, 1400, QMode::Quality(50)).unwrap();
        assert_eq!(pkts.len(), 1);
        let h = parse_main_header(&pkts[0].payload).unwrap();
        assert_eq!(h.typ, 0);
        assert_eq!(h.q, 50);
        // Q-field mode emits no in-band table header; scan starts at byte 8.
        assert_eq!(&pkts[0].payload[MAIN_HDR_LEN..], &[0xAB; 8]);
    }

    #[test]
    fn packetize_restart_sets_type_bit_and_header() {
        let jpeg = handmade_jpeg(64, 64, 0x22, 7, &[0x5A; 6]);
        let pkts = packetize(&jpeg, 1400, QMode::Quality(60)).unwrap();
        let p = &pkts[0].payload;
        let h = parse_main_header(p).unwrap();
        assert_eq!(h.typ, 1 | TYPE_RESTART_BIT); // type 65
        assert!(h.has_restart());
        let rh = parse_restart_header(&p[MAIN_HDR_LEN..]).unwrap();
        assert_eq!(rh.restart_interval, 7);
        assert!(rh.first && rh.last);
        assert_eq!(rh.count, 0x3FFF);
    }

    #[test]
    fn packetize_fragments_long_scan() {
        // 100-byte scan, max_payload 8(hdr)+10(scan)=18 → 10 scan bytes/frag.
        let scan: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let jpeg = handmade_jpeg(64, 64, 0x22, 0, &scan);
        let pkts = packetize(&jpeg, MAIN_HDR_LEN + 10, QMode::Quality(50)).unwrap();
        assert_eq!(pkts.len(), 10);
        // Offsets are contiguous and only the last carries the marker.
        let mut expect_off = 0u32;
        for (i, pk) in pkts.iter().enumerate() {
            let h = parse_main_header(&pk.payload).unwrap();
            assert_eq!(h.fragment_offset, expect_off);
            expect_off += (pk.payload.len() - MAIN_HDR_LEN) as u32;
            assert_eq!(pk.marker, i == pkts.len() - 1);
        }
        assert_eq!(expect_off, 100);
    }

    #[test]
    fn packetize_rejects_progressive() {
        // SOF2 in place of SOF0.
        let mut jpeg = handmade_jpeg(64, 64, 0x22, 0, &[0u8; 4]);
        let sof = jpeg
            .windows(2)
            .position(|w| w[0] == 0xFF && w[1] == markers::SOF0)
            .unwrap();
        jpeg[sof + 1] = markers::SOF2;
        assert!(packetize(&jpeg, 1400, QMode::Quality(50)).is_err());
    }

    #[test]
    fn packetize_rejects_unsupported_qmode_ranges() {
        let jpeg = handmade_jpeg(64, 64, 0x22, 0, &[0u8; 4]);
        assert!(packetize(&jpeg, 1400, QMode::Quality(0)).is_err());
        assert!(packetize(&jpeg, 1400, QMode::Quality(100)).is_err());
        assert!(packetize(&jpeg, 1400, QMode::InBand(127)).is_err());
    }

    #[test]
    fn packetize_rejects_oversize_dimensions() {
        let jpeg = handmade_jpeg(2048, 64, 0x22, 0, &[0u8; 4]);
        assert!(packetize(&jpeg, 1400, QMode::Quality(50)).is_err());
    }

    #[test]
    fn packetize_then_depacketize_roundtrips_scan() {
        // Structural round trip without the codec: the scan bytes that go in
        // come back out, and the reconstructed type/dims match.
        // A real entropy-coded scan never carries a bare 0xFF (the encoder
        // byte-stuffs them), so keep the synthetic scan below 0xFF.
        let scan: Vec<u8> = (0..200).map(|i| (i % 0xF0) as u8).collect();
        let jpeg = handmade_jpeg(128, 96, 0x22, 0, &scan);
        // Quant header is 132 B; size the MTU to fit it plus a small scan
        // chunk so the 200-byte scan still spans more than one fragment.
        let pkts = packetize(&jpeg, MAIN_HDR_LEN + 132 + 40, QMode::InBand(200)).unwrap();
        assert!(pkts.len() > 1);
        let mut dp = JpegDepacketizer::new();
        let mut rebuilt = None;
        for pk in &pkts {
            match dp.push(&pk.payload, pk.marker).unwrap() {
                Progress::NeedMore => {}
                Progress::Frame(j) => rebuilt = Some(j),
            }
        }
        let rebuilt = rebuilt.expect("frame reassembled");
        // SOI..EOI bookends and the scan survives intact.
        assert_eq!(&rebuilt[0..2], &[0xFF, markers::SOI]);
        assert_eq!(&rebuilt[rebuilt.len() - 2..], &[0xFF, markers::EOI]);
        let scan_start = rebuilt.len() - 2 - scan.len();
        assert_eq!(&rebuilt[scan_start..rebuilt.len() - 2], &scan[..]);
        // 4:2:0 sampling preserved through the round trip.
        let sof = rebuilt
            .windows(2)
            .position(|w| w[0] == 0xFF && w[1] == markers::SOF0)
            .unwrap();
        assert_eq!(rebuilt[sof + 2 + 2 + 1 + 2 + 2 + 1 + 1], 0x22);
    }

    // ---- End-to-end: encode → packetize → depacketize → decode. Proves a
    // packetized stream really decodes back to the source image. ----

    #[cfg(feature = "registry")]
    #[test]
    fn end_to_end_packetize_inband_decodes() {
        let (w, h) = (96usize, 64usize);
        let frame = make_420_frame(w, h);
        let jpeg =
            encode_jpeg(&frame, w as u32, h as u32, PixelFormat::Yuv420P, 75).expect("encode");

        // Fragment small enough to force several packets out of the scan.
        let pkts = packetize(&jpeg, MAIN_HDR_LEN + 200, QMode::InBand(255)).expect("packetize");
        assert!(pkts.len() > 1, "scan should span multiple fragments");
        assert!(pkts.last().unwrap().marker);

        let mut dp = JpegDepacketizer::new();
        let mut rebuilt = None;
        for pk in &pkts {
            if let Progress::Frame(j) = dp.push(&pk.payload, pk.marker).unwrap() {
                rebuilt = Some(j);
            }
        }
        let rebuilt = rebuilt.expect("frame reassembled");
        let decoded = decode_jpeg(&rebuilt, None).expect("decode packetized RTP/JPEG");
        assert_eq!(decoded.planes.len(), 3);
        assert_eq!(decoded.planes[0].data.len(), w * h);
        assert_eq!(decoded.planes[1].data.len(), w.div_ceil(2) * h.div_ceil(2));
    }

    #[cfg(feature = "registry")]
    #[test]
    fn end_to_end_packetize_q_field_decodes() {
        // Encode at quality 50 so the depacketizer's IJG-regenerated tables
        // match the scan's quantization exactly.
        let (w, h) = (64usize, 64usize);
        let frame = make_420_frame(w, h);
        let jpeg =
            encode_jpeg(&frame, w as u32, h as u32, PixelFormat::Yuv420P, 50).expect("encode");
        let pkts = packetize(&jpeg, 1400, QMode::Quality(50)).expect("packetize");

        let mut dp = JpegDepacketizer::new();
        let mut rebuilt = None;
        for pk in &pkts {
            if let Progress::Frame(j) = dp.push(&pk.payload, pk.marker).unwrap() {
                rebuilt = Some(j);
            }
        }
        let decoded = decode_jpeg(&rebuilt.unwrap(), None).expect("decode");
        assert_eq!(decoded.planes.len(), 3);
        assert_eq!(decoded.planes[0].data.len(), w * h);
    }

    // -------------------------------------------------------------------
    // `parse_jpeg` (packetize encode-side parser) panic-surface regression
    // tests. The packetizer accepts an arbitrary external JPEG; every
    // length-prefixed segment must validate before indexing.
    // -------------------------------------------------------------------

    /// Build a stub `SOI ... EOI` byte stream with a single SOF0 segment
    /// whose declared length is `sof_len`. Used to drive the length /
    /// component-records bounds checks below.
    fn jpeg_with_truncated_sof(sof_len: u16) -> Vec<u8> {
        let mut j = vec![0xFF, markers::SOI, 0xFF, markers::SOF0];
        j.extend_from_slice(&sof_len.to_be_bytes());
        // No body bytes — declared length lies about what follows.
        j.push(0xFF);
        j.push(markers::EOI);
        j
    }

    #[test]
    fn packetize_rejects_sof_with_underflowing_length() {
        // `len = 0` would underflow `len - 2` in the SOF arm.
        let j = jpeg_with_truncated_sof(0);
        let err = packetize(&j, 1400, QMode::Quality(50)).unwrap_err();
        assert!(format!("{err}").contains("truncated SOF"));
    }

    #[test]
    fn packetize_rejects_sof_with_undersized_component_records() {
        // A 3-component SOF needs 8 + 3*3 = 17 bytes. Declare 12 (precision +
        // height + width + Nf + only a half-component) and watch the parser
        // refuse rather than read past the segment.
        let mut j = vec![0xFF, markers::SOI, 0xFF, markers::SOF0];
        j.extend_from_slice(&12u16.to_be_bytes());
        j.push(8); // precision
        j.extend_from_slice(&16u16.to_be_bytes()); // height
        j.extend_from_slice(&16u16.to_be_bytes()); // width
        j.push(3); // Nf = 3 — but we only carry 4 more bytes, not 9.
        j.extend_from_slice(&[0, 0, 0, 0]);
        j.push(0xFF);
        j.push(markers::EOI);
        let err = packetize(&j, 1400, QMode::Quality(50)).unwrap_err();
        assert!(format!("{err}").contains("truncated SOF components"));
    }

    #[test]
    fn packetize_rejects_dqt_with_underflowing_length() {
        let mut j = vec![0xFF, markers::SOI, 0xFF, markers::DQT];
        j.extend_from_slice(&0u16.to_be_bytes()); // len = 0 → underflow trap
        j.push(0xFF);
        j.push(markers::EOI);
        let err = packetize(&j, 1400, QMode::Quality(50)).unwrap_err();
        assert!(format!("{err}").contains("truncated DQT"));
    }

    #[test]
    fn packetize_rejects_sos_with_underflowing_length() {
        // SOF0 (minimum-valid 3-comp at 16x16, qt 0/1/1) then SOS with len=0.
        let mut j = vec![0xFF, markers::SOI];
        // SOF0
        j.push(0xFF);
        j.push(markers::SOF0);
        j.extend_from_slice(&17u16.to_be_bytes());
        j.push(8); // precision
        j.extend_from_slice(&16u16.to_be_bytes());
        j.extend_from_slice(&16u16.to_be_bytes());
        j.push(3);
        // Components 1/2/3 with 4:2:0 sampling. Quant table ids 0/1/1.
        j.extend_from_slice(&[1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
        // SOS with len = 1 (underflow trap on len - 2).
        j.push(0xFF);
        j.push(markers::SOS);
        j.extend_from_slice(&1u16.to_be_bytes());
        j.push(0xFF);
        j.push(markers::EOI);
        let err = packetize(&j, 1400, QMode::Quality(50)).unwrap_err();
        assert!(format!("{err}").contains("truncated SOS"));
    }

    #[test]
    fn packetize_rejects_generic_segment_with_underflowing_length() {
        // APP0 (catch-all branch) with len = 0 would loop forever in
        // `pos += len`. The guard turns it into a clean error.
        let mut j = vec![0xFF, markers::SOI];
        j.push(0xFF);
        j.push(0xE0); // APP0 — not specially handled, falls into the `_` arm.
        j.extend_from_slice(&1u16.to_be_bytes()); // len = 1, too short.
        j.push(0xFF);
        j.push(markers::EOI);
        let err = packetize(&j, 1400, QMode::Quality(50)).unwrap_err();
        assert!(format!("{err}").contains("truncated/oversized segment"));
    }
}
