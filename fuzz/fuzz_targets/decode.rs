#![no_main]

//! Robustness fuzz target: feed arbitrary bytes into the public JPEG
//! decoder and assert it never panics.
//!
//! The decoder must convert every malformed, truncated, or otherwise
//! adversarial byte string into either a successful [`Frame::Video`]
//! (rare, the chance of a random byte stream forming a valid baseline
//! JPEG by accident is vanishingly small) or a clean
//! [`oxideav_core::Error`] return value. Any `panic!`, `unwrap` on a
//! parser-derived `Option`/`Result`, slice index out-of-bounds, or
//! integer-overflow trap is a real bug — file the crashing input under
//! `fuzz/artifacts/decode/` and fix the source path that raised it in
//! the *same commit* per repo policy.
//!
//! Classic crash surfaces this target exercises (verbatim from the
//! round brief):
//!   * Huffman table overflow (DHT with `Tc`/`Th` out of range, code
//!     lengths summing past 256, or a code-length distribution that
//!     would over-fill the Huffman tree).
//!   * DRI restart-interval arithmetic (a non-zero `Ri` paired with a
//!     scan that emits fewer MCUs than `Ri` between RSTn markers, or
//!     RSTn sequence wrap-around).
//!   * EXIF nested IFD (an APP1 marker whose TIFF header points an
//!     `ExifIFD`/`GPSIFD`/`Interop` pointer back at itself, blowing
//!     the parser's recursion budget — the JPEG decoder *skips* APP
//!     segments wholesale, but EXIF chaining still has to be safe
//!     against pathological lengths).
//!   * JFIF density (an APP0 segment with `Xdensity`/`Ydensity` set
//!     to zero or values that would overflow downstream pixel-ratio
//!     calculations).
//!   * Progressive AC/DC scan (`Ss`/`Se`/`Ah`/`Al` outside legal
//!     ranges, or a successive-approximation refinement on an
//!     un-initialised coefficient buffer).
//!   * SOF0/SOF2 component mismatch (an SOS referencing a component
//!     index not declared in the SOF, or a SOF declaring duplicate
//!     `Ci` ids).
//!   * 12-bit support paths (an SOF0/SOF1 with `P=12` taking the
//!     wide-sample IDCT path; the buffer allocator and pixel-format
//!     mapping have to handle this without overflowing 8-bit
//!     assumptions).
//!
//! No oracle, no round-trip — the bar is "does it panic". Any
//! resulting [`Frame::Video`] is discarded; the decoder's *output*
//! correctness is covered by the encode/decode round-trip targets in
//! this same `fuzz_targets/` directory.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Packet, TimeBase};

/// Cap on the per-call input size. Beyond this libFuzzer is mostly
/// burning corpus space on inputs the decoder rejects on the first
/// marker scan anyway, and the per-iteration cost grows linearly with
/// input length on otherwise-valid Huffman/scan paths. 64 KiB keeps a
/// single iteration under a few ms even on the slowest branches, so
/// the daily 7-minute slice gets a good corpus depth.
const MAX_INPUT: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT {
        return;
    }

    // The decoder factory needs codec id "mjpeg"; width/height are
    // optional and the JPEG SOFn segment carries the real values
    // anyway, so we leave them unset to exercise the "no container
    // hint" path that the still-image demuxer takes.
    let mut params = CodecParameters::video(CodecId::new("mjpeg"));
    params.width = None;
    params.height = None;

    let Ok(mut dec) = oxideav_mjpeg::decoder::make_decoder(&params) else {
        return;
    };

    // Single-packet decode. JPEG is a frame-aligned format (one
    // SOI..EOI per packet), so feeding the whole fuzz input as one
    // packet is the realistic shape — chunking would mostly find
    // bugs in the harness rather than the decoder.
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());

    // `send_packet` is permitted to return Err (most fuzz inputs
    // aren't valid JPEGs); we just must not panic. Same for
    // `receive_frame` — Err is fine, panic is not.
    if dec.send_packet(&pkt).is_err() {
        return;
    }
    let _ = dec.receive_frame();
});
