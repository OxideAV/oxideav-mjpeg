#![no_main]

//! Fuzz the RFC 2435 RTP/JPEG depacketizer (`oxideav_mjpeg::rtp`).
//!
//! The RTP/JPEG parsing surface has several attacker-controlled fields
//! that drive variable-length sub-headers and a fragment-offset
//! reassembly buffer:
//!
//! * §3.1 main header — `Q ∈ 128..=254` triggers the §3.1.8
//!   Quantization Table header on the first fragment; `Q ∈ 1..=99`
//!   uses the IJG-scaled tables; `Q = 0` is reserved and `Q = 255`
//!   forbids `Length = 0` per the RFC.
//! * §3.1.3 type field — the `0x40` restart bit pulls in a §3.1.7
//!   4-byte Restart Marker header whose `restart_interval` MUST NOT
//!   be zero.
//! * §3.1.2 fragment offset (24-bit) — the depacketizer keys the
//!   reassembly buffer on this offset; a hostile sequence of offsets
//!   could in principle force the buffer to grow unboundedly. The
//!   harness caps payload size so the fuzzer cannot legitimately
//!   request a multi-GiB buffer, but any panic inside `resize` or
//!   `copy_from_slice` is still a bug we want flagged.
//! * §3.1.8 quantization-table header — `precision`'s low two bits
//!   select 64-byte vs 128-byte tables; `length` smaller than what
//!   `precision` demands must be rejected without OOB-reading the
//!   payload.
//! * §4.2 in-band table caching — a static `Q ∈ 128..=254` carrying
//!   tables once should let later frames with `Length = 0` decode.
//!   `Q = 255` must never be cached. A previously-cached `Q`
//!   followed by a `Length = 0` frame on a *different* static Q must
//!   not be served the wrong table.
//!
//! Contract: every call into `parse_main_header`,
//! `parse_restart_header`, and `JpegDepacketizer::push` must return
//! `Ok(_)` or `Err(_)` — never panic, slice OOB, integer-overflow on
//! a debug build, or allocate a buffer the input couldn't plausibly
//! back. Return values are inspected only to decide whether to
//! continue feeding more fragments; their *contents* are not
//! validated here (round-trip correctness is the packetize+depacketize
//! test in `src/rtp.rs`'s unit tests, not the fuzz oracle).
//!
//! The harness splits the fuzz input into multiple "packets" so the
//! fragment-reassembly state machine, the marker-bit branch, and the
//! cross-frame table cache all get exercised in a single run.

use libfuzzer_sys::fuzz_target;
use oxideav_mjpeg::rtp::{parse_main_header, parse_restart_header, JpegDepacketizer, Progress};

/// Discard inputs larger than this. The depacketizer copies each
/// fragment's scan bytes into a per-frame buffer keyed on the
/// 24-bit fragment offset; an attacker that supplies a multi-MiB
/// fragment offset would force an O(offset) buffer allocation
/// regardless of how few input bytes actually came from the fuzzer.
/// Capping the input to 16 KiB keeps each iteration fast while
/// still exposing the parse/cache paths.
const MAX_INPUT_LEN: usize = 16 * 1024;

/// Maximum number of "packets" we split one fuzz input into. Past
/// this point we're just spinning on the same code path.
const MAX_PACKETS: usize = 32;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > MAX_INPUT_LEN {
        return;
    }

    // ------------------------------------------------------------------
    // Direct header parsers — exercise their bounds checks on arbitrary
    // bytes. Each call must return `Ok(_)` or `Err(_)` without panicking.
    // ------------------------------------------------------------------
    let _ = parse_main_header(data);
    let _ = parse_restart_header(data);

    // ------------------------------------------------------------------
    // Stateful depacketizer — split the fuzz input into a sequence of
    // packets and feed them through `push`, alternating the marker
    // bit so the assemble path runs whenever the in-progress frame
    // closes.
    //
    // The first byte chooses the split scheme:
    //   - low 3 bits: packet-count nibble (1..=8 packets per call)
    //   - bit 3: whether the marker bit is set on the *final* packet
    //     (otherwise we leave the reassembly mid-frame).
    //
    // Remaining bytes form the packet stream, segmented evenly.
    // ------------------------------------------------------------------
    let split_byte = data[0];
    let body = &data[1..];
    if body.is_empty() {
        return;
    }

    let n_packets = ((split_byte & 0x07) as usize + 1).min(MAX_PACKETS);
    let final_marker = (split_byte & 0x08) != 0;

    let chunk = body.len().div_ceil(n_packets).max(1);

    let mut dp = JpegDepacketizer::new();
    let mut sent_any_frame = false;

    for (i, pkt) in body.chunks(chunk).enumerate() {
        // Marker bit on the last fragment only (or never, depending
        // on `final_marker`). Mid-frame markers test the "marker
        // closes whatever fragments have arrived so far" branch.
        let marker = if i + 1 == n_packets {
            final_marker
        } else {
            false
        };

        match dp.push(pkt, marker) {
            Ok(Progress::Frame(jpeg)) => {
                // Cheap sanity invariants on the assembled stream:
                // it must at least look like SOI..EOI. The decoder
                // itself isn't run here — the `decode` fuzz target
                // covers panic-freedom on arbitrary bytes already,
                // and feeding every assembled depack output through
                // the decoder would dominate the harness with
                // already-fuzzed code.
                assert!(jpeg.len() >= 4, "assembled JPEG impossibly short");
                assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "missing SOI on assembled frame");
                assert_eq!(
                    &jpeg[jpeg.len() - 2..],
                    &[0xFF, 0xD9],
                    "missing EOI on assembled frame"
                );
                sent_any_frame = true;
            }
            Ok(Progress::NeedMore) | Err(_) => {
                // `NeedMore` is the dominant outcome for sub-marker
                // packets; `Err(_)` is the dominant outcome for the
                // overwhelming majority of fuzz inputs (the wire
                // format is highly structured). Either is fine.
            }
        }
    }

    // After the loop, exercise the reset path so the cache-retention
    // invariant (§4.2: `reset()` keeps the table cache, `new()`
    // drops it) is also reached on every iteration.
    dp.reset();

    // If at least one frame was emitted, the cache may now hold a
    // static-Q table pair; push one more empty fragment to probe the
    // "no in-progress frame, non-zero offset" rejection path on a
    // depacketizer with a live cache.
    if sent_any_frame {
        let _ = dp.push(&[0x00, 0x00, 0x00, 0x01, 0x00, 0xC8, 0x00, 0x00], false);
    }
});
