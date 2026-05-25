#![no_main]

//! Decode arbitrary fuzz-supplied bytes through the public MJPEG /
//! JPEG decoder.
//!
//! The contract under test is purely that the call *returns*: any
//! malformed input must yield `Err(_)` or a benign `Ok(_)` frame —
//! never a panic, an `unwrap()` on `None`, a slice OOB, a debug-build
//! integer overflow, or an OOM-class `Vec::with_capacity` /
//! `vec![0; n]` allocation that exceeds what the input could
//! plausibly back. Return values are intentionally discarded.
//!
//! Coverage of the panic surfaces that exist on the decoder's
//! attacker-controlled paths:
//!
//! * `BitReader::get_bits(n)` with `n` derived from a Huffman-decoded
//!   byte. A crafted DHT can deliver any value in `0..=255` for a
//!   SSSS field that the decoder hands straight to `get_bits` — the
//!   shift `self.bits >> (32 - n)` underflows when `n > 24`.
//! * 4-bit selector indexing into the 4-wide `state.dc_huff` /
//!   `ac_huff` / `quant` / `arith_dc` / `arith_ac` arrays. SOS Tdj
//!   and Taj nibbles span `0..=15`, the Tq nibble in SOF spans
//!   `0..=15`, but each table array is 4-wide.
//! * SOS `Ns = 0` (empty scan-component list → empty `prev_dc` and
//!   table-lookup vecs whose first index then panics) and `Ns > 4`
//!   (beyond the spec's component cap).
//! * Repeated SOF segments. T.81 specifies a single SOF per frame;
//!   accepting a second SOF after the coefficient buffer is sized
//!   for the first would mismatch geometry and OOB the per-block
//!   accumulator.
//! * Pixel-budget DoS. A SOF declaring `Wt = Ht = 65535, Nf = 4`
//!   asks for `~17 GiB` of buffers from a 16-byte SOF segment. The
//!   decoder caps total samples at `MAX_PIXEL_BUDGET`.
//!
//! The harness deliberately routes through the `Decoder` trait
//! (`make_decoder` + `send_packet` + `receive_frame`) rather than
//! `decode_jpeg` directly, so the fuzzer also exercises the trait
//! plumbing in `registry.rs`. The crate's standalone build (without
//! the `registry` feature) reaches the same decoder via the same
//! marker walker, so a panic found here applies to either build
//! configuration.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Packet, TimeBase};

/// Discard inputs above this size. The decoder allocates a small
/// constant per-segment regardless of input length, so a huge input
/// would only slow the fuzzer down without exploring new code
/// paths. 64 KiB easily fits a baseline JPEG plus a generous
/// progressive multi-scan tail.
const MAX_INPUT_LEN: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    // The minimum well-formed JPEG is `FF D8 FF D9` (SOI + EOI). Below
    // that the SOI check trivially rejects; let the fuzzer spend its
    // time on richer inputs.
    if data.len() < 4 || data.len() > MAX_INPUT_LEN {
        return;
    }

    let params = CodecParameters::video(CodecId::new("mjpeg"));
    let Ok(mut dec) = oxideav_mjpeg::decoder::make_decoder(&params) else {
        return;
    };

    // `send_packet` only stores the packet; the actual parse runs
    // inside `receive_frame`. A return of `Err` is the expected
    // outcome for the overwhelming majority of fuzz inputs; we don't
    // inspect the success-path frame because the fuzz oracle here is
    // "no panic", not "correct bytes".
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
    if dec.send_packet(&pkt).is_err() {
        return;
    }
    let _ = dec.receive_frame();
});
