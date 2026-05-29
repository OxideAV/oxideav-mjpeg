#![no_main]

//! Fuzz the RFC 2435 RTP/JPEG **packetizer** (`oxideav_mjpeg::rtp::packetize`).
//!
//! The packetizer accepts a complete external JPEG interchange stream and
//! parses it from scratch (`fn parse_jpeg`) to extract the well-known §4.1
//! fields and the entropy-coded scan span. That parser walks a sequence of
//! length-prefixed JPEG segments (SOI / SOF0 / SOF1 / DQT / DHT / DRI /
//! APPn / COM / SOS / RSTn / EOI) and indexes into the input buffer at
//! offsets driven by the wire-controlled length field. Any of the
//! following are panic surfaces if not bounds-checked:
//!
//! * SOF0/SOF1 length below the 8-byte fixed header → `len - 2` underflow.
//! * SOF0/SOF1 with `Nf = 3` but a segment too short to carry three
//!   component records (each `id(1) H|V(1) Tq(1)` = 3 bytes) → OOB on
//!   `jpeg[c0 + 7]` etc.
//! * DQT / SOS with `len < 2` → `len - 2` underflow, or `pos + len`
//!   overshooting `jpeg.len()` and corrupting the subsequent scan slice.
//! * Catch-all length-prefixed segments with `len = 0` → `pos += 0`
//!   infinite loop.
//! * `parts.scan_start > parts.scan_end` ⇒ later `&jpeg[start..end]` panic
//!   inside `packetize`.
//!
//! The `QMode` argument also has its own validation (Quality must be
//! 1..=99, InBand must be 128..=255) but those are pre-parse and trivially
//! exhaustive; the harness still exercises both branches per iteration so
//! the in-band table assembly path runs.
//!
//! Contract: every call into `packetize` must return `Ok(_)` or `Err(_)`;
//! it must never panic, slice-OOB, integer-overflow in debug, or allocate
//! a buffer the input couldn't plausibly back. Returned packets are
//! sanity-checked for shape only (first fragment offset 0, last fragment
//! has the marker bit set, internal payload length within `max_payload`).
//! Round-trip correctness (re-feeding into `JpegDepacketizer` and
//! comparing to the original frame) is covered by the unit tests in
//! `src/rtp.rs`, not this fuzz oracle.

use libfuzzer_sys::fuzz_target;
use oxideav_mjpeg::rtp::{packetize, QMode};

/// Discard inputs larger than this. The packetizer copies the scan into
/// per-fragment payload buffers; capping the input keeps each iteration
/// quick while still exercising the parse paths.
const MAX_INPUT_LEN: usize = 16 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_LEN {
        return;
    }

    // The first byte chooses the QMode + max_payload encoding so a single
    // fuzz input exercises both QMode arms and a range of MTU sizes.
    let knob = data[0];
    let jpeg = &data[1..];

    // QMode selection:
    //   bit 0: 0 → Quality, 1 → InBand.
    //   bit 1: when Quality, use the IJG-valid range (1..=99) vs a value
    //          outside it (so the validation branch runs).
    //   bits 2..=3: max_payload bucket (small / typical / large / huge).
    let qmode = if knob & 1 == 0 {
        let q = if knob & 0b10 == 0 {
            ((knob >> 4) % 99) + 1 // 1..=99
        } else {
            // 0 or 100..=255 exercise the Quality rejection branch.
            if knob & 0b1000 == 0 {
                0
            } else {
                (knob >> 4).saturating_add(100)
            }
        };
        QMode::Quality(q)
    } else {
        // 128..=255; high bit of knob selects 255 (dynamic) or some
        // middle static value.
        let q = if knob & 0b10 == 0 { 128 + (knob >> 4) } else { 255 };
        QMode::InBand(q)
    };

    let max_payload: usize = match (knob >> 2) & 0b11 {
        0 => 16,    // pathologically small — exercises the header-room rejection.
        1 => 256,   // small.
        2 => 1400,  // typical MTU.
        _ => 8_192, // large.
    };

    match packetize(jpeg, max_payload, qmode) {
        Ok(packets) => {
            // Shape invariants for any well-formed packet sequence.
            assert!(
                !packets.is_empty(),
                "packetize Ok with no packets — malformed return"
            );
            assert!(
                packets.last().unwrap().marker,
                "final packet must carry the marker bit"
            );
            for (i, p) in packets.iter().enumerate() {
                // Main JPEG header (§3.1) is 8 bytes; payload always >= 8.
                assert!(p.payload.len() >= 8, "packet {i} payload too short");
                // No payload may exceed the caller's stated MTU.
                assert!(
                    p.payload.len() <= max_payload,
                    "packet {i}: payload {} exceeds max_payload {max_payload}",
                    p.payload.len()
                );
                // Only the final packet may set the marker bit.
                if i + 1 < packets.len() {
                    assert!(!p.marker, "non-final packet {i} carries marker bit");
                }
                // §3.1.2 fragment offset is the bytes 1..=3, big-endian. The
                // first fragment must offset 0.
                if i == 0 {
                    let off = ((p.payload[1] as u32) << 16)
                        | ((p.payload[2] as u32) << 8)
                        | (p.payload[3] as u32);
                    assert_eq!(off, 0, "first packet fragment offset must be 0");
                }
            }
        }
        Err(_) => {
            // The overwhelming majority of fuzz inputs fall here — the JPEG
            // wire format is highly structured and arbitrary bytes rarely
            // satisfy the SOI / SOF / SOS sequencing. That's expected.
        }
    }
});
