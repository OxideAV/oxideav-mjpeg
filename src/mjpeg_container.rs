//! Raw Motion-JPEG container (`.mjpeg` / `.mjpg`).
//!
//! A raw MJPEG file is a concatenation of self-contained JPEG bitstreams,
//! one per video frame:
//!
//! ```text
//!   SOI .. EOI | SOI .. EOI | SOI .. EOI | …
//! ```
//!
//! There is no wrapping header, no per-frame timestamp, and no global
//! frame-rate field — those are assumed (or recovered from an outer
//! container like AVI / MOV / MKV when MJPEG is carried inside one).
//! For raw `.mjpeg` files on disk we synthesise a default 25 fps stream
//! with frame `i` carrying `pts = i` in a `time_base = 1/25` clock.
//! Callers that know better can post-process the resulting stream's
//! `time_base` to swap in a different rate.
//!
//! Per the workspace task brief, the demuxer:
//!
//! - Walks the input frame-by-frame using a JPEG-aware byte scanner.
//!   The scanner is the *exact* same one the still-image container
//!   uses — it tracks marker boundaries by walking SOI → (length-
//!   prefixed segments) → SOS → entropy data → EOI, transparently
//!   collapsing `0xFF 0xFF…` fill runs and treating `0xFF 0x00` as a
//!   stuffed literal. This makes the SOI scanner immune to false
//!   positives from compressed entropy data that happens to contain
//!   a literal `0xFF 0xD8` byte sequence.
//! - Maintains a lazy `(pts, byte_offset)` seek index. Every Nth
//!   frame (N = [`SEEK_INDEX_INTERVAL`]) gets a waypoint pushed; the
//!   first frame always lands one, so backward seeks have an anchor.
//! - Implements [`Demuxer::seek_to`] via binary search on the lazy
//!   index + linear scan forward to the exact target frame. If the
//!   target is past any waypoint yet recorded, the demuxer scans
//!   forward from the closest known waypoint, adding new waypoints
//!   as it goes. Targets past EOF clamp to the last waypoint and
//!   `next_packet` returns `Error::Eof` from there.
//!
//! Probe: a raw `.mjpeg` starts with `FF D8 FF` (same as a `.jpg`), so
//! the probe returns a moderate score (60) instead of 100. The
//! still-image container's probe wins on `.jpg` / `.jpeg` extensions;
//! extension routing tips raw `.mjpeg` / `.mjpg` toward this demuxer.

use std::io::{Read, Seek, SeekFrom};

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, Demuxer, Error, MediaType, Packet, PixelFormat,
    ProbeData, Rational, ReadSeek, Result, StreamInfo, TimeBase,
};
use oxideav_core::{ContainerRegistry, ProbeScore};

use crate::jpeg::markers::{self, EOI, SOI};
use crate::jpeg::parser::{parse_sof, SofInfo};

/// Default frame rate assumed for headerless raw `.mjpeg` files.
/// 25 fps matches the historical "PAL" cadence used when timing
/// is otherwise unspecified.
const DEFAULT_FRAME_RATE: u32 = 25;

/// Build a `(pts, byte_offset)` waypoint every Nth frame. 5 keeps the
/// index small (one entry per ~0.2 s at 25 fps) while still giving
/// seek targets a tight upper bound for the forward-scan step.
const SEEK_INDEX_INTERVAL: u64 = 5;

/// Register the raw MJPEG container under the name `mjpeg-raw`.
///
/// Distinguished from the still-image `jpeg` container — `mjpeg-raw`
/// owns the `.mjpeg` / `.mjpg` extensions and yields one packet per
/// frame in the stream; `jpeg` owns `.jpg` / `.jpeg` and yields a
/// single packet for the whole file.
pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("mjpeg-raw", open_demuxer);
    reg.register_extension("mjpeg", "mjpeg-raw");
    reg.register_extension("mjpg", "mjpeg-raw");
    reg.register_probe("mjpeg-raw", probe);
}

/// Probe: `FF D8 FF` opens any JPEG. We return a moderate score so the
/// still-image probe (which also matches at 100) wins on `.jpg` /
/// `.jpeg`, while extension routing tips raw `.mjpeg` / `.mjpg` here.
pub fn probe(p: &ProbeData) -> ProbeScore {
    if p.buf.len() >= 3 && p.buf[0] == 0xFF && p.buf[1] == 0xD8 && p.buf[2] == 0xFF {
        // 60 is below the still container's 100 so probing alone
        // routes to the still container; extension hints flip the
        // call back to "mjpeg-raw" via `container_for_extension`.
        60
    } else {
        0
    }
}

fn open_demuxer(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> Result<Box<dyn Demuxer>> {
    input.seek(SeekFrom::Start(0))?;

    // Walk the first JPEG to learn the canvas size (every frame in an
    // MJPEG stream is supposed to share the same dimensions; we trust
    // the first frame and don't re-validate later ones).
    let (first_start, first_end) = scan_one_frame(input.as_mut(), 0)?;
    let mut first_bytes = vec![0u8; (first_end - first_start) as usize];
    input.seek(SeekFrom::Start(first_start))?;
    input.read_exact(&mut first_bytes)?;
    let sof = scan_for_sof(&first_bytes)?;

    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.width = Some(sof.width as u32);
    params.height = Some(sof.height as u32);
    params.pixel_format = Some(pixel_format_for_sof(&sof));
    params.frame_rate = Some(Rational::new(DEFAULT_FRAME_RATE as i64, 1));

    let time_base = TimeBase::new(1, DEFAULT_FRAME_RATE as i64);
    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: None,
        start_time: Some(0),
        params,
    };

    let mut idx = SeekIndex::default();
    idx.maybe_push(0, first_start);

    // Rewind so `next_packet` re-reads the first frame.
    input.seek(SeekFrom::Start(0))?;

    Ok(Box::new(MjpegDemuxer {
        input,
        stream,
        time_base,
        next_pts: 0,
        next_offset: 0,
        index: idx,
        eof: false,
    }))
}

/// Walk a JPEG segment starting at `start_offset` in `input` and return
/// `(start_offset, end_exclusive)` of the SOI..EOI span. Leaves the
/// cursor at `end_exclusive`.
///
/// State machine (matches T.81 §B.1.1.2 / §B.1.1.4):
///
/// * **Outside a scan** — most markers are length-prefixed segments
///   (DQT, DHT, SOFn, APPn, COM, DRI, DAC, …). We read the 2-byte
///   `Lp` length field and *skip the whole segment body* in one
///   `Seek` rather than byte-walking it. This is critical: APP1
///   segments often embed thumbnail JPEGs that contain literal
///   `FF D8 .. FF D9` byte sequences. A byte-blind scanner would
///   false-match an SOI / EOI inside such a segment. SOS (start of
///   scan) is treated separately — we read its variable-length
///   header then transition to the entropy-scan reader.
/// * **Inside a scan** (after SOS, until the next non-RST marker)
///   — stuffed `0xFF 0x00` is a literal `0xFF`; `0xFF` followed by
///   `RSTn` (`D0..D7`) is a restart marker (no length, ignored).
///   Any other marker `0xFF XX` *ends* the scan. If `XX = EOI` we
///   return; otherwise we fold the new marker back into the
///   outside-scan dispatch loop (DHT / DQT / DRI / SOS / SOFn can
///   all follow a scan in progressive JPEGs).
fn scan_one_frame<R: Read + Seek + ?Sized>(input: &mut R, start_offset: u64) -> Result<(u64, u64)> {
    input.seek(SeekFrom::Start(start_offset))?;
    let mut pos = start_offset;
    let mut buf = [0u8; 2];

    // Expect SOI at frame start: 0xFF 0xD8.
    pos = read_marker(input, pos, &mut buf)?;
    if buf[0] != SOI {
        return Err(Error::invalid("MJPEG: expected SOI at frame start"));
    }

    loop {
        // Read the next marker byte (length-prefixed segments live
        // here; entropy scans are entered via SOS below).
        pos = read_marker(input, pos, &mut buf)?;
        let marker = buf[0];

        if marker == EOI {
            return Ok((start_offset, pos));
        }
        if marker == SOI {
            // Two SOIs in a row would mean we ran off the end of one
            // frame into the next. Caller should have stopped at the
            // previous EOI. Treat as malformed.
            return Err(Error::invalid("MJPEG: unexpected SOI inside frame"));
        }
        if markers::is_rst(marker) {
            // RSTn outside a scan is unusual but not fatal — skip.
            continue;
        }
        if marker == markers::SOS {
            // Read the SOS length field + body in one seek, then enter
            // the entropy-data state machine.
            pos = skip_length_prefixed_segment(input, pos)?;
            pos = walk_entropy_scan(input, pos)?;
            // After the entropy scan, the cursor is positioned right
            // after the *2-byte* marker that terminated the scan and
            // `buf[0]` carries the marker byte. Inspect it and fold
            // back into the outer dispatch loop.
            pos = post_scan_marker(input, pos, &mut buf)?;
            // Re-handle the marker in the same way as the outer loop:
            // EOI returns, otherwise length-prefixed (DHT/DQT/SOFn/SOS).
            // To avoid duplicating logic, just loop and reuse it.
            //
            // We've already consumed the marker bytes; rewind 2 so the
            // next iteration's `read_marker` sees them again.
            input.seek(SeekFrom::Start(pos - 2))?;
            pos -= 2;
            continue;
        }
        // Any other marker is length-prefixed: APP0..APP15, COM, DQT,
        // DHT, DRI, DAC, SOFn, ...
        pos = skip_length_prefixed_segment(input, pos)?;
    }
}

/// Read a 2-byte marker `(0xFF, marker)` starting at the current
/// cursor. Collapses runs of `0xFF` fill bytes (T.81 §B.1.1.2 allows
/// any number of `0xFF` fill bytes preceding a marker). On success
/// `buf[0]` holds the marker byte; the cursor advances by the bytes
/// consumed and the function returns the updated `pos`.
fn read_marker<R: Read + ?Sized>(input: &mut R, mut pos: u64, buf: &mut [u8; 2]) -> Result<u64> {
    // First byte must be 0xFF.
    let mut one = [0u8; 1];
    let n = input.read(&mut one)?;
    if n == 0 {
        return Err(Error::Eof);
    }
    pos += 1;
    if one[0] != 0xFF {
        return Err(Error::invalid("MJPEG: expected 0xFF marker prefix"));
    }
    // Skip any fill 0xFF run.
    loop {
        let n = input.read(&mut one)?;
        if n == 0 {
            return Err(Error::invalid("MJPEG: truncated marker"));
        }
        pos += 1;
        if one[0] != 0xFF {
            buf[0] = one[0];
            buf[1] = 0xFF; // unused
            return Ok(pos);
        }
        // Another 0xFF — fill byte.
    }
}

/// Skip a length-prefixed marker segment whose 2-byte length field
/// starts at the current cursor. The length field includes its own 2
/// bytes (T.81 §B.1.1.4), so the body size is `Lp - 2`. Returns the
/// updated `pos`.
fn skip_length_prefixed_segment<R: Read + Seek + ?Sized>(input: &mut R, pos: u64) -> Result<u64> {
    let mut lp = [0u8; 2];
    let n = input.read(&mut lp)?;
    if n < 2 {
        return Err(Error::invalid("MJPEG: truncated segment length field"));
    }
    let len = u16::from_be_bytes(lp) as u64;
    if len < 2 {
        return Err(Error::invalid("MJPEG: segment length < 2"));
    }
    // Seek over the segment body (length includes its own 2 bytes).
    let body = len - 2;
    input.seek(SeekFrom::Current(body as i64))?;
    Ok(pos + 2 + body)
}

/// Walk entropy-coded scan data until the next non-RST / non-stuffed
/// marker. Leaves the cursor positioned right at the marker prefix
/// `0xFF`, ready for [`post_scan_marker`] to consume it. Returns the
/// updated `pos`.
fn walk_entropy_scan<R: Read + Seek + ?Sized>(input: &mut R, mut pos: u64) -> Result<u64> {
    let mut buf = [0u8; 1];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            return Err(Error::invalid("MJPEG: EOF inside entropy scan"));
        }
        pos += 1;
        if buf[0] != 0xFF {
            continue;
        }
        // Saw 0xFF — peek next byte without committing.
        let n = input.read(&mut buf)?;
        if n == 0 {
            return Err(Error::invalid("MJPEG: truncated marker in scan"));
        }
        pos += 1;
        // Collapse fill bytes.
        while buf[0] == 0xFF {
            let n = input.read(&mut buf)?;
            if n == 0 {
                return Err(Error::invalid("MJPEG: truncated marker in scan"));
            }
            pos += 1;
        }
        if buf[0] == 0x00 {
            // Stuffed literal — continue entropy scan.
            continue;
        }
        if markers::is_rst(buf[0]) {
            // Restart marker, no length — continue entropy scan.
            continue;
        }
        // Real marker — rewind 2 bytes so the caller can re-read
        // the `(0xFF, marker)` prefix via `post_scan_marker`.
        input.seek(SeekFrom::Current(-2))?;
        return Ok(pos - 2);
    }
}

/// Re-read the `(0xFF, marker)` prefix left by [`walk_entropy_scan`].
/// `buf[0]` carries the marker byte on return.
fn post_scan_marker<R: Read + ?Sized>(
    input: &mut R,
    mut pos: u64,
    buf: &mut [u8; 2],
) -> Result<u64> {
    let n = input.read(buf)?;
    if n < 2 {
        return Err(Error::invalid("MJPEG: truncated post-scan marker"));
    }
    pos += 2;
    if buf[0] != 0xFF {
        return Err(Error::invalid("MJPEG: post-scan marker missing 0xFF"));
    }
    buf[0] = buf[1];
    Ok(pos)
}

fn scan_for_sof(data: &[u8]) -> Result<SofInfo> {
    if data.len() < 2 || data[0] != 0xFF || data[1] != SOI {
        return Err(Error::invalid("MJPEG: first frame missing SOI"));
    }
    let body = &data[2..];
    let mut walker = crate::jpeg::parser::MarkerWalker::new(body);
    loop {
        let Some(marker) = walker.next_marker()? else {
            return Err(Error::invalid("MJPEG: SOF not found before EOF"));
        };
        if markers::is_sof(marker) {
            let payload = walker.read_segment_payload()?;
            return Ok(parse_sof(payload)?);
        }
        if marker == SOI || marker == EOI || markers::is_rst(marker) {
            continue;
        }
        let _ = walker.read_segment_payload()?;
    }
}

fn pixel_format_for_sof(sof: &SofInfo) -> PixelFormat {
    match sof.components.len() {
        1 => PixelFormat::Gray8,
        3 => {
            let y = sof.components[0];
            match (y.h_factor, y.v_factor) {
                (2, 2) => PixelFormat::Yuv420P,
                (2, 1) => PixelFormat::Yuv422P,
                (1, 1) => PixelFormat::Yuv444P,
                _ => PixelFormat::Yuv420P,
            }
        }
        _ => PixelFormat::Yuv420P,
    }
}

/// Lazy `(pts, byte_offset)` index. Always sorted by pts; entries are
/// pushed in chronological order by `next_packet`. Seek targets that
/// fall between waypoints land at the latest waypoint with `pts <=
/// target` and let the demuxer scan forward from there.
#[derive(Default, Debug)]
struct SeekIndex {
    pts: Vec<i64>,
    offsets: Vec<u64>,
}

impl SeekIndex {
    fn maybe_push(&mut self, pts: i64, offset: u64) {
        if matches!(self.pts.last(), Some(&p) if p == pts) {
            return;
        }
        self.pts.push(pts);
        self.offsets.push(offset);
    }

    /// Largest waypoint `<= target`. Returns `(0, 0)` when the index
    /// is empty.
    fn lookup(&self, target: i64) -> (i64, u64) {
        if self.pts.is_empty() {
            return (0, 0);
        }
        let idx = self.pts.partition_point(|&p| p <= target);
        if idx == 0 {
            (self.pts[0], self.offsets[0])
        } else {
            (self.pts[idx - 1], self.offsets[idx - 1])
        }
    }
}

pub(crate) struct MjpegDemuxer {
    input: Box<dyn ReadSeek>,
    stream: StreamInfo,
    time_base: TimeBase,
    /// Pts to be emitted by the *next* `next_packet`. Advances by 1 per
    /// frame (frame index in the 1/DEFAULT_FRAME_RATE clock).
    next_pts: i64,
    /// Byte offset where the next frame's SOI is expected. Updated by
    /// `next_packet`; trusted by `seek_to` when restoring after seek.
    next_offset: u64,
    /// Lazy `(pts, byte_offset)` waypoint index.
    index: SeekIndex,
    eof: bool,
}

impl Demuxer for MjpegDemuxer {
    fn format_name(&self) -> &str {
        "mjpeg-raw"
    }

    fn streams(&self) -> &[StreamInfo] {
        std::slice::from_ref(&self.stream)
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.eof {
            return Err(Error::Eof);
        }
        self.input.seek(SeekFrom::Start(self.next_offset))?;
        let frame_start = self.next_offset;
        let frame_pts = self.next_pts;

        let (start, end) = match scan_one_frame(self.input.as_mut(), frame_start) {
            Ok(span) => span,
            Err(Error::Eof) => {
                self.eof = true;
                return Err(Error::Eof);
            }
            Err(e) => return Err(e),
        };
        debug_assert_eq!(start, frame_start);
        let len = (end - start) as usize;
        let mut data = vec![0u8; len];
        self.input.seek(SeekFrom::Start(start))?;
        self.input.read_exact(&mut data)?;

        // Advance cursors.
        self.next_pts = frame_pts + 1;
        self.next_offset = end;

        // Record a waypoint every SEEK_INDEX_INTERVAL frames. The
        // first frame (pts == 0) was already pushed at open time.
        let frame_idx = frame_pts as u64;
        if frame_idx % SEEK_INDEX_INTERVAL == 0 {
            self.index.maybe_push(frame_pts, frame_start);
        }

        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = Some(frame_pts);
        pkt.dts = Some(frame_pts);
        pkt.duration = Some(1);
        pkt.flags.keyframe = true;
        Ok(pkt)
    }

    /// Seek to the frame at `pts` (in the stream's `time_base = 1/25`
    /// clock — i.e. `pts` is the zero-based frame index).
    ///
    /// Algorithm:
    /// 1. Clamp negative pts to 0.
    /// 2. Look up the largest indexed waypoint with `pts <= target`.
    ///    Reset the demuxer cursor to that waypoint.
    /// 3. If the waypoint pts already equals the target, return.
    /// 4. Otherwise scan forward frame-by-frame, advancing the cursor
    ///    and adding new waypoints, until we either reach the target
    ///    or hit EOF. EOF lands the demuxer right after the last
    ///    complete frame; `next_packet` then returns `Error::Eof`.
    fn seek_to(&mut self, stream_index: u32, pts: i64) -> Result<i64> {
        if stream_index != 0 {
            return Err(Error::invalid(format!(
                "MJPEG: stream index {stream_index} out of range (only stream 0 exists)"
            )));
        }
        let target = pts.max(0);

        // Anchor: largest indexed waypoint with pts <= target.
        let (anchor_pts, anchor_offset) = self.index.lookup(target);
        self.next_pts = anchor_pts;
        self.next_offset = anchor_offset;
        self.eof = false;

        // If the anchor already hits the target, we're done.
        if anchor_pts == target {
            return Ok(target);
        }

        // Otherwise scan forward, *without* materialising packet
        // payloads — we only need to advance frame spans. This also
        // grows the index with any waypoints we cross.
        let mut cur_pts = anchor_pts;
        let mut cur_offset = anchor_offset;
        let mut landed_pts = anchor_pts;
        while cur_pts < target {
            self.input.seek(SeekFrom::Start(cur_offset))?;
            let span = scan_one_frame(self.input.as_mut(), cur_offset);
            let (start, end) = match span {
                Ok(s) => s,
                Err(Error::Eof) | Err(Error::InvalidData(_)) => {
                    // Out-of-range seek (target past EOF). Clamp to
                    // the last frame we successfully scanned.
                    self.eof = true;
                    self.next_offset = cur_offset;
                    self.next_pts = cur_pts;
                    return Ok(landed_pts);
                }
                Err(e) => return Err(e),
            };
            // We just consumed a frame at (cur_pts, start..end). Push a
            // waypoint if this frame index is a multiple of the
            // interval; the open-demuxer call already seeds (0, 0).
            let frame_idx = cur_pts as u64;
            if frame_idx % SEEK_INDEX_INTERVAL == 0 {
                self.index.maybe_push(cur_pts, start);
            }
            landed_pts = cur_pts;
            cur_pts += 1;
            cur_offset = end;
        }

        // `cur_pts == target` here. Position the demuxer so the next
        // `next_packet` emits the target frame.
        self.next_pts = cur_pts;
        self.next_offset = cur_offset;
        self.input.seek(SeekFrom::Start(cur_offset))?;
        // Record this waypoint too if it lines up.
        if (cur_pts as u64) % SEEK_INDEX_INTERVAL == 0 {
            self.index.maybe_push(cur_pts, cur_offset);
        }
        Ok(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal "fake MJPEG" stream with `n_frames` each
    /// consisting of `SOI APP0(len=2) EOI`. Sufficient to exercise the
    /// frame scanner without needing a real JPEG payload.
    fn fake_mjpeg(n_frames: usize) -> Vec<u8> {
        // Per-frame: FF D8  FF E0 00 02  FF D9  (8 bytes)
        let frame = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x02, 0xFF, 0xD9];
        let mut buf = Vec::with_capacity(n_frames * frame.len());
        for _ in 0..n_frames {
            buf.extend_from_slice(&frame);
        }
        buf
    }

    #[test]
    fn probe_score_below_still_container() {
        // FF D8 FF is the JPEG magic; still container claims 100, we
        // claim 60 so still wins on bare-content probing.
        let p = ProbeData {
            buf: &[0xFF, 0xD8, 0xFF, 0xE0],
            ext: None,
        };
        assert_eq!(probe(&p), 60);
    }

    #[test]
    fn scan_one_frame_returns_soi_to_eoi_span() {
        let bytes = fake_mjpeg(3);
        let mut cur = Cursor::new(bytes.clone());
        let (s, e) = scan_one_frame(&mut cur, 0).expect("first frame");
        assert_eq!((s, e), (0, 8));
        let (s2, e2) = scan_one_frame(&mut cur, e).expect("second frame");
        assert_eq!((s2, e2), (8, 16));
    }

    #[test]
    fn scan_one_frame_ignores_stuffed_ff_d8_in_entropy_data() {
        // Frame with an embedded `FF 00 D8` byte pattern that would
        // false-match an SOI scanner. Layout:
        //   SOI  (FF D8)
        //   SOS-like marker (FF DA) with a 2-byte length=2 prefix
        //   entropy bytes: FF 00 D8 D8 (D8 is a literal byte here)
        //   EOI (FF D9)
        let bytes = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xDA, 0x00, 0x02, // SOS, length=2 (just the header)
            0xFF, 0x00, 0xD8, 0xD8, // stuffed FF + literal D8 D8
            0xFF, 0xD9, // EOI
        ];
        let mut cur = Cursor::new(bytes.clone());
        let (s, e) = scan_one_frame(&mut cur, 0).expect("scan");
        assert_eq!((s, e), (0, bytes.len() as u64));
    }

    #[test]
    fn seek_index_lookup_returns_largest_le() {
        let mut idx = SeekIndex::default();
        idx.maybe_push(0, 0);
        idx.maybe_push(5, 100);
        idx.maybe_push(10, 200);
        assert_eq!(idx.lookup(0), (0, 0));
        assert_eq!(idx.lookup(4), (0, 0));
        assert_eq!(idx.lookup(5), (5, 100));
        assert_eq!(idx.lookup(7), (5, 100));
        assert_eq!(idx.lookup(10), (10, 200));
        assert_eq!(idx.lookup(99), (10, 200));
    }
}
