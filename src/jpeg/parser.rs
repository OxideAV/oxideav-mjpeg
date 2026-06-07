//! JPEG marker-segment walker.
//!
//! Advances through a JPEG byte stream, returning each marker segment plus
//! the entropy-coded scan data between SOS and the next marker.

use crate::error::{MjpegError as Error, Result};

use super::markers;

/// One parsed marker segment.
#[derive(Debug)]
pub struct MarkerSegment<'a> {
    /// Marker byte (without the leading 0xFF).
    pub marker: u8,
    /// Payload bytes (excludes the 2-byte length and the marker itself).
    pub payload: &'a [u8],
}

/// Start-of-frame (SOF0) descriptor.
#[derive(Clone, Debug)]
pub struct SofInfo {
    pub precision: u8,
    pub height: u16,
    pub width: u16,
    pub components: Vec<SofComponent>,
}

#[derive(Clone, Copy, Debug)]
pub struct SofComponent {
    pub id: u8,
    pub h_factor: u8,
    pub v_factor: u8,
    pub qt_id: u8,
}

pub fn parse_sof(payload: &[u8]) -> Result<SofInfo> {
    if payload.len() < 6 {
        return Err(Error::invalid("SOF: too short"));
    }
    let precision = payload[0];
    let height = u16::from_be_bytes([payload[1], payload[2]]);
    let width = u16::from_be_bytes([payload[3], payload[4]]);
    let nf = payload[5] as usize;
    if payload.len() < 6 + nf * 3 {
        return Err(Error::invalid("SOF: component list truncated"));
    }
    let mut components = Vec::with_capacity(nf);
    for i in 0..nf {
        let off = 6 + i * 3;
        let id = payload[off];
        let hv = payload[off + 1];
        let qt = payload[off + 2];
        components.push(SofComponent {
            id,
            h_factor: hv >> 4,
            v_factor: hv & 0x0F,
            qt_id: qt,
        });
    }
    Ok(SofInfo {
        precision,
        height,
        width,
        components,
    })
}

/// Scan-header (SOS) info.
#[derive(Clone, Debug)]
pub struct SosInfo {
    pub components: Vec<SosComponent>,
    pub ss: u8,
    pub se: u8,
    pub ah: u8,
    pub al: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct SosComponent {
    pub id: u8,
    pub dc_table: u8,
    pub ac_table: u8,
}

pub fn parse_sos(payload: &[u8]) -> Result<SosInfo> {
    if payload.is_empty() {
        return Err(Error::invalid("SOS: empty"));
    }
    let ns = payload[0] as usize;
    if payload.len() < 1 + ns * 2 + 3 {
        return Err(Error::invalid("SOS: truncated"));
    }
    let mut comps = Vec::with_capacity(ns);
    for i in 0..ns {
        let off = 1 + i * 2;
        let id = payload[off];
        let ta_td = payload[off + 1];
        comps.push(SosComponent {
            id,
            dc_table: ta_td >> 4,
            ac_table: ta_td & 0x0F,
        });
    }
    let off = 1 + ns * 2;
    let ss = payload[off];
    let se = payload[off + 1];
    let ah_al = payload[off + 2];
    Ok(SosInfo {
        components: comps,
        ss,
        se,
        ah: ah_al >> 4,
        al: ah_al & 0x0F,
    })
}

/// One arithmetic-conditioning entry decoded from a DAC marker (B.2.4.3).
///
/// `tc=0` is a DC (or lossless) table; `cs` packs the lower-bound `L` in the
/// low nibble and the upper-bound `U` in the high nibble (so `L = cs & 0x0F`,
/// `U = cs >> 4`). `tc=1` is an AC table; `cs` is `Kx` in the range 1..=63.
#[derive(Clone, Copy, Debug)]
pub struct DacEntry {
    pub tc: u8,
    pub tb: u8,
    pub cs: u8,
}

/// Parse a DAC segment payload. Returns the list of entries; up to four DC
/// destinations and four AC destinations may be specified per segment.
pub fn parse_dac(payload: &[u8]) -> Result<Vec<DacEntry>> {
    if payload.len() % 2 != 0 {
        return Err(Error::invalid("DAC: payload length must be even"));
    }
    let n = payload.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let tc_tb = payload[i * 2];
        let cs = payload[i * 2 + 1];
        let tc = tc_tb >> 4;
        let tb = tc_tb & 0x0F;
        if tc > 1 {
            return Err(Error::invalid("DAC: Tc must be 0 or 1"));
        }
        if tb > 3 {
            return Err(Error::invalid("DAC: Tb must be 0..=3"));
        }
        if tc == 1 && !(1..=63).contains(&cs) {
            return Err(Error::invalid("DAC: AC Kx must be 1..=63"));
        }
        out.push(DacEntry { tc, tb, cs });
    }
    Ok(out)
}

/// DRI payload is a 16-bit big-endian restart interval count (in MCUs).
pub fn parse_dri(payload: &[u8]) -> Result<u16> {
    if payload.len() < 2 {
        return Err(Error::invalid("DRI: too short"));
    }
    Ok(u16::from_be_bytes([payload[0], payload[1]]))
}

/// JFIF APP0 density unit selector (T.871 §10.1, "units" byte).
///
/// The byte chooses how `Hdensity` / `Vdensity` are interpreted:
///
/// * `0x00` — units unspecified: the densities express only the pixel
///   aspect ratio (width : height = Hdensity : Vdensity).
/// * `0x01` — dots per inch.
/// * `0x02` — dots per cm.
///
/// The spec does not allocate any other value; the parser accepts only
/// these three and rejects everything else as malformed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JfifUnits {
    /// `units = 0`: H/V densities encode the pixel aspect ratio only.
    AspectRatio,
    /// `units = 1`: H/V densities are dots per inch.
    DotsPerInch,
    /// `units = 2`: H/V densities are dots per centimetre.
    DotsPerCm,
}

/// Typed view of a JFIF APP0 marker segment (T.871 §10.1).
///
/// Mirrors the on-wire field layout one-to-one. The parser does not
/// dereference the thumbnail payload — callers who want the pixels read
/// `thumbnail_width` / `thumbnail_height` and slice into the original
/// payload themselves. Keeping the typed view lifetime-free (no `'a`
/// reference into the payload) is intentional: the typed parser is
/// often called for the `inspect_jpeg` side-band and the caller's
/// payload buffer may not outlive the inspector's frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JfifApp0 {
    /// JFIF major version byte. T.871 §10.1 mandates `1` for any file
    /// conforming to the recommendation; values from the wild include
    /// `1` (v1.00 / v1.01 / v1.02) and the encoder-emitted `2` from the
    /// occasional non-conforming source. The parser reports the byte
    /// verbatim — clients that want to refuse anything but v1.02
    /// inspect `version_major == 1 && version_minor == 2` themselves.
    pub version_major: u8,
    /// JFIF minor version byte. T.871 §10.1 mandates `0x02` for
    /// conforming files; older v1.00 / v1.01 streams report `0` / `1`.
    pub version_minor: u8,
    /// Density-unit selector (T.871 §10.1, "units" byte).
    pub units: JfifUnits,
    /// Horizontal pixel density (must be non-zero per the spec, but the
    /// parser does not enforce — it reports the wire value so callers
    /// can detect malformed encoders).
    pub h_density: u16,
    /// Vertical pixel density (must be non-zero per the spec; reported
    /// verbatim, see `h_density`).
    pub v_density: u16,
    /// Thumbnail horizontal pixel count `HthumbnailA` (T.871 §10.1).
    /// Zero indicates no embedded thumbnail.
    pub thumbnail_width: u8,
    /// Thumbnail vertical pixel count `VthumbnailA`. Zero indicates no
    /// embedded thumbnail.
    pub thumbnail_height: u8,
}

impl JfifApp0 {
    /// True when both thumbnail dimensions are non-zero — i.e. the APP0
    /// payload carries `3 * w * h` bytes of packed RGB after the
    /// fixed-length header per T.871 §10.1.
    pub fn has_thumbnail(self) -> bool {
        self.thumbnail_width != 0 && self.thumbnail_height != 0
    }

    /// Byte size of the embedded thumbnail RGB block (`3 * w * h`).
    /// Returns `0` when `has_thumbnail()` is false.
    pub fn thumbnail_byte_len(self) -> usize {
        3 * (self.thumbnail_width as usize) * (self.thumbnail_height as usize)
    }
}

/// JFIF identifier from the start of an APP0 payload, T.871 §10.1.
/// Five bytes: `"JFIF\0"`.
pub(crate) const JFIF_MAGIC: &[u8; 5] = b"JFIF\0";

/// Parse a JFIF APP0 marker payload per T.871 §10.1.
///
/// The `payload` slice is the APP0 segment body **with the leading 0xFF
/// 0xE0 marker and the 2-byte length already stripped** — i.e. the
/// `MarkerWalker::read_segment_payload` return value.
///
/// Layout (16 fixed bytes + optional `3 * HthumbnailA * VthumbnailA`
/// thumbnail body):
///
/// ```text
///   offset  field           bytes
///        0  identifier      5    ("JFIF\0")
///        5  version_major   1
///        6  version_minor   1
///        7  units           1
///        8  Hdensity        2    big-endian
///       10  Vdensity        2    big-endian
///       12  HthumbnailA     1
///       13  VthumbnailA     1
///       14  thumbnail RGB   3 * HthumbnailA * VthumbnailA
/// ```
///
/// Note that the APP0 payload as delivered by `MarkerWalker` is 2 bytes
/// shorter than T.871's `Lp` byte count (which includes the length
/// field itself). The fixed header is therefore 14 bytes of payload
/// here, matching `Lp = 16 + 3*k - 2`.
///
/// Errors:
/// * `Invalid` if the payload is shorter than the 14-byte fixed header.
/// * `Invalid` if the first 5 bytes are not literally `JFIF\0` (the
///   caller must filter for that magic before delegating here; an APP0
///   carrying e.g. the `JFXX\0` JFIF-extension identifier is rejected
///   by this entry point on purpose — extension-marker parsing belongs
///   in a separate routine).
/// * `Invalid` if the `units` byte is not `0`, `1`, or `2`.
/// * `Invalid` if the thumbnail dimensions claim more bytes than the
///   payload supplies (`14 + 3 * w * h > payload.len()`).
pub fn parse_jfif_app0(payload: &[u8]) -> Result<JfifApp0> {
    if payload.len() < 14 {
        return Err(Error::invalid("JFIF APP0: too short"));
    }
    if &payload[..5] != JFIF_MAGIC {
        return Err(Error::invalid("JFIF APP0: identifier is not JFIF\\0"));
    }
    let version_major = payload[5];
    let version_minor = payload[6];
    let units = match payload[7] {
        0 => JfifUnits::AspectRatio,
        1 => JfifUnits::DotsPerInch,
        2 => JfifUnits::DotsPerCm,
        _ => return Err(Error::invalid("JFIF APP0: invalid units byte")),
    };
    let h_density = u16::from_be_bytes([payload[8], payload[9]]);
    let v_density = u16::from_be_bytes([payload[10], payload[11]]);
    let thumbnail_width = payload[12];
    let thumbnail_height = payload[13];

    let thumb_len = 3usize * (thumbnail_width as usize) * (thumbnail_height as usize);
    if payload.len() < 14 + thumb_len {
        return Err(Error::invalid("JFIF APP0: thumbnail body truncated"));
    }

    Ok(JfifApp0 {
        version_major,
        version_minor,
        units,
        h_density,
        v_density,
        thumbnail_width,
        thumbnail_height,
    })
}

/// Walker over JPEG markers.
///
/// SOI/EOI/RST* have no payload or length field. Every other marker from
/// 0xC0..0xFE (excluding SOI/EOI/RST*) is followed by a big-endian 16-bit
/// length (inclusive of those two bytes themselves) and a payload.
pub struct MarkerWalker<'a> {
    buf: &'a [u8],
    pub pos: usize,
}

impl<'a> MarkerWalker<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Seek forward to the next `0xFF` marker byte, consuming fill bytes.
    /// Returns the marker byte (without 0xFF) or None at EOF.
    pub fn next_marker(&mut self) -> Result<Option<u8>> {
        while self.pos < self.buf.len() {
            if self.buf[self.pos] != 0xFF {
                self.pos += 1;
                continue;
            }
            // Skip a run of 0xFF fill bytes.
            while self.pos < self.buf.len() && self.buf[self.pos] == 0xFF {
                self.pos += 1;
            }
            if self.pos >= self.buf.len() {
                return Ok(None);
            }
            let m = self.buf[self.pos];
            self.pos += 1;
            if m == 0x00 {
                // Stuffed zero — not a real marker. Keep looking.
                continue;
            }
            return Ok(Some(m));
        }
        Ok(None)
    }

    /// Read the length-prefixed payload of the marker we most recently
    /// returned from `next_marker`.
    pub fn read_segment_payload(&mut self) -> Result<&'a [u8]> {
        if self.pos + 2 > self.buf.len() {
            return Err(Error::invalid("marker segment: truncated length"));
        }
        let len = u16::from_be_bytes([self.buf[self.pos], self.buf[self.pos + 1]]) as usize;
        if len < 2 {
            return Err(Error::invalid("marker segment: length < 2"));
        }
        if self.pos + len > self.buf.len() {
            return Err(Error::invalid("marker segment: payload truncated"));
        }
        let p = &self.buf[self.pos + 2..self.pos + len];
        self.pos += len;
        Ok(p)
    }

    /// Consume the entropy-coded scan immediately after an SOS segment. Ends
    /// at the byte just before the next real marker (excluding 0xFF00 stuffs
    /// and RST* markers, which belong to the scan).
    pub fn read_scan_data(&mut self) -> Result<&'a [u8]> {
        let start = self.pos;
        while self.pos < self.buf.len() {
            if self.buf[self.pos] == 0xFF {
                // Lookahead.
                if self.pos + 1 >= self.buf.len() {
                    return Err(Error::invalid("scan: truncated at 0xFF"));
                }
                let nxt = self.buf[self.pos + 1];
                if nxt == 0x00 {
                    // 0xFF 0x00 → literal 0xFF, part of scan.
                    self.pos += 2;
                    continue;
                }
                if markers::is_rst(nxt) {
                    // Restart markers stay inside the scan.
                    self.pos += 2;
                    continue;
                }
                // Real marker. End of scan.
                return Ok(&self.buf[start..self.pos]);
            }
            self.pos += 1;
        }
        Ok(&self.buf[start..self.pos])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The minimum-size JFIF APP0 payload — five-byte identifier, two
    /// version bytes, units, four density bytes, two thumbnail-count
    /// bytes, no embedded thumbnail. The raw byte layout follows
    /// T.871 §10.1 verbatim.
    fn jfif_app0_no_thumb(ver_major: u8, ver_minor: u8, units: u8, hd: u16, vd: u16) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(b"JFIF\0");
        p.push(ver_major);
        p.push(ver_minor);
        p.push(units);
        p.extend_from_slice(&hd.to_be_bytes());
        p.extend_from_slice(&vd.to_be_bytes());
        p.push(0);
        p.push(0);
        p
    }

    #[test]
    fn jfif_app0_v102_dpi_72() {
        // The canonical v1.02-dpi-72 APP0 every JFIF encoder emits.
        let p = jfif_app0_no_thumb(1, 2, 1, 72, 72);
        let info = parse_jfif_app0(&p).expect("parse JFIF v1.02 72dpi");
        assert_eq!(info.version_major, 1);
        assert_eq!(info.version_minor, 2);
        assert_eq!(info.units, JfifUnits::DotsPerInch);
        assert_eq!(info.h_density, 72);
        assert_eq!(info.v_density, 72);
        assert_eq!(info.thumbnail_width, 0);
        assert_eq!(info.thumbnail_height, 0);
        assert!(!info.has_thumbnail());
        assert_eq!(info.thumbnail_byte_len(), 0);
    }

    #[test]
    fn jfif_app0_aspect_ratio_only() {
        // units = 0 — the densities are a pixel-aspect ratio, not a
        // physical resolution.
        let p = jfif_app0_no_thumb(1, 1, 0, 1, 1);
        let info = parse_jfif_app0(&p).expect("parse JFIF aspect-ratio");
        assert_eq!(info.units, JfifUnits::AspectRatio);
        assert_eq!(info.h_density, 1);
        assert_eq!(info.v_density, 1);
    }

    #[test]
    fn jfif_app0_dpcm() {
        // units = 2 — dots per centimetre.
        let p = jfif_app0_no_thumb(1, 2, 2, 300, 300);
        let info = parse_jfif_app0(&p).expect("parse JFIF dpcm");
        assert_eq!(info.units, JfifUnits::DotsPerCm);
    }

    #[test]
    fn jfif_app0_rejects_short_payload() {
        // 13 bytes — one byte short of the 14-byte fixed header.
        let p = vec![0u8; 13];
        assert!(parse_jfif_app0(&p).is_err());
    }

    #[test]
    fn jfif_app0_rejects_wrong_identifier() {
        let mut p = jfif_app0_no_thumb(1, 2, 1, 72, 72);
        // Flip the magic to "JFXX\0" — that's the extension marker.
        p[..5].copy_from_slice(b"JFXX\0");
        assert!(parse_jfif_app0(&p).is_err());
    }

    #[test]
    fn jfif_app0_rejects_unknown_units() {
        let mut p = jfif_app0_no_thumb(1, 2, 0, 1, 1);
        p[7] = 3; // not in {0, 1, 2}
        assert!(parse_jfif_app0(&p).is_err());
    }

    #[test]
    fn jfif_app0_with_thumbnail() {
        // 2x2 RGB thumbnail = 12 bytes of pixel data following the
        // fixed 14-byte header.
        let mut p = jfif_app0_no_thumb(1, 2, 1, 72, 72);
        // Overwrite the thumbnail-count bytes.
        let last_two = p.len() - 2;
        p[last_two] = 2;
        p[last_two + 1] = 2;
        // Append RGB pixels.
        for i in 0..12 {
            p.push(i as u8);
        }
        let info = parse_jfif_app0(&p).expect("parse JFIF 2x2 thumb");
        assert_eq!(info.thumbnail_width, 2);
        assert_eq!(info.thumbnail_height, 2);
        assert!(info.has_thumbnail());
        assert_eq!(info.thumbnail_byte_len(), 12);
    }

    #[test]
    fn jfif_app0_rejects_truncated_thumbnail() {
        // Claim a 4x4 thumbnail (48 RGB bytes) but supply only 10 bytes.
        let mut p = jfif_app0_no_thumb(1, 2, 1, 72, 72);
        let last_two = p.len() - 2;
        p[last_two] = 4;
        p[last_two + 1] = 4;
        p.resize(p.len() + 10, 0);
        assert!(parse_jfif_app0(&p).is_err());
    }

    #[test]
    fn jfif_app0_pre_v102_version_bytes_are_reported_verbatim() {
        // v1.00 / v1.01 streams pre-date T.871 but the parser must
        // not reject them; downstream callers decide whether to fall
        // back to the older spec.
        let p = jfif_app0_no_thumb(1, 0, 1, 96, 96);
        let info = parse_jfif_app0(&p).expect("parse JFIF v1.00");
        assert_eq!(info.version_major, 1);
        assert_eq!(info.version_minor, 0);
        let p = jfif_app0_no_thumb(1, 1, 1, 96, 96);
        let info = parse_jfif_app0(&p).expect("parse JFIF v1.01");
        assert_eq!(info.version_minor, 1);
    }
}
