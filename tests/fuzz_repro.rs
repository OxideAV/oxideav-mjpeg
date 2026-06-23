#![cfg(feature = "registry")]
//! Decoder fuzz-crash regressions.
//!
//! Each test pins one byte string that previously made the public decode
//! path panic (the `decode` cargo-fuzz harness contract is "never panic").
//! The assertion is only that decoding returns — `Ok` or `Err` are both
//! acceptable; a panic fails the test.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

/// Decode `bytes` through the public `Decoder` trait, returning `true` when
/// it produced a frame and `false` when it returned an error. A panic
/// (index-out-of-bounds, overflow, unwrap, …) fails the test.
fn decode_no_panic(bytes: Vec<u8>) -> bool {
    let mut params = CodecParameters::video(CodecId::new("mjpeg"));
    params.width = Some(64);
    params.height = Some(64);
    let mut dec = match oxideav_mjpeg::decoder::make_decoder(&params) {
        Ok(d) => d,
        Err(_) => return false,
    };
    if dec
        .send_packet(&Packet::new(0, TimeBase::new(1, 30), bytes))
        .is_err()
    {
        return false;
    }
    matches!(dec.receive_frame(), Ok(Frame::Video(_)))
}

/// Regression for fuzz finding
/// `crash-b98641910fbef3137721accd5fed81901c743a03` (Fuzz run 28013962630,
/// `decode` target).
///
/// Failure mode: the DHT segment declared an **over-subscribed** Huffman
/// code (more short codes than the code space admits). The canonical-code
/// walk in `HuffTable::build` drove the code counter past `2^len`, and the
/// `FAST_BITS`-wide fast-lookup fill computed an index of 512 into the
/// 512-entry `fast` table → `index out of bounds: the len is 512 but the
/// index is 512` at `src/jpeg/huffman.rs`.
///
/// Fix: `HuffTable::build` now rejects an over-subscribed table (one whose
/// Kraft sum exceeds 1) up front with `InvalidData`, so the decoder errors
/// cleanly instead of panicking.
#[test]
fn over_subscribed_dht_does_not_panic() {
    // The minimised crash input (201 bytes). A leading SOI, a marker, then
    // a DHT whose BITS over-fills the length-1 code space.
    const CRASH_B64: &str = "/9i+/8QAxAAAAAAAAAAAAAAAAAABAAAAAAAAAYEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQdv//////////wAAAAAAAP8AAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAIqKioqKioqKioqKioqKAQEBioqKJYqKioqK//8k//////////////////8F//////9B2/////////////////////////8AAAAAAACKAACK";
    let bytes = base64_decode(CRASH_B64);
    // Must return (Ok or Err) without panicking.
    let _ = decode_no_panic(bytes);
}

/// Minimal standard-alphabet base64 decoder (no external dep). Ignores
/// padding length mismatches the way the fuzz harness's input never has.
fn base64_decode(s: &str) -> Vec<u8> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut nbits = 0u32;
    for &c in s.as_bytes() {
        let Some(v) = val(c) else { continue };
        acc = (acc << 6) | v as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    out
}
