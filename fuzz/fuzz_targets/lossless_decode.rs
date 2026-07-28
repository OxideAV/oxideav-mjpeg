#![no_main]

//! Decode arbitrary fuzz-supplied entropy bytes through the lossless
//! (SOF3 Huffman / SOF11 arithmetic) predictive decoder.
//!
//! The harness wraps fuzz input in a minimal lossless JPEG envelope so
//! the spatial-predictive code paths (`decode_lossless_scan`, the
//! predictor select in T.81 Table H.1, the point-transform output
//! shift, the `Pt`-aware neighbour reconstruction, and the SSSS=16
//! half-modulus special case) actually run on every iteration — the
//! generic `decode` target almost never composes a valid SOF3 header,
//! DHT, and SOS with matching selectors out of random bytes.
//!
//! The contract under test is the robustness bar shared with the other
//! envelope targets: any input must yield `Err(_)` or a benign `Ok(_)`
//! frame — never a panic, a slice OOB, a debug-build overflow, or an
//! OOM-class allocation.
//!
//! Panic surfaces specific to the lossless path that this target
//! drives per-iteration:
//!
//! * Precision `P ∈ 2..=16` — the full ladder, including the 16-bit
//!   `Di = 32768` half-modulus case (§H.1.2.2) and the sub-byte
//!   precisions where the modulo arithmetic wraps in `< 8` bits.
//!   Output plane sizing switches at `P > 8` (2-byte LE samples).
//! * The SOS `Ss` field reinterpreted as predictor select. Raw
//!   `0..=15` nibble — values `8..=15` must be rejected, `0` is
//!   invalid for a first (non-differential) lossless scan.
//! * The SOS `Al` field reinterpreted as point transform. Raw
//!   `0..=15` nibble — `Pt >= P` must be rejected, valid `Pt`
//!   left-shifts every reconstructed sample on output.
//! * DNL resolution (§B.2.5): SOF `Y = 0` plus a DNL segment after
//!   the first scan. Both the well-formed path (`NL` from the fuzz
//!   dims), the `NL = 0` reject, and a missing/mismatched DNL marker
//!   after the scan are reachable via control bits.
//! * Restart-marker bookkeeping in a lossless scan (DRI = 2 samples)
//!   where the entropy tail is fuzz-truncated mid-interval.
//! * The arithmetic lossless statistical model (SOF11): the
//!   two-dimensional (Da, Db) context selection reuses the Annex H
//!   DC conditioning — driven here both with the default `(L, U) =
//!   (0, 1)` and with a DAC override.
//!
//! Routed through the public `Decoder` trait (`make_decoder` +
//! `send_packet` + `receive_frame`) so `registry.rs` plumbing is on
//! the same path.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Packet, TimeBase};

/// Discard inputs above this size. Lossless entropy decode is
/// sample-serial; the envelope images are at most 16×16 so a few KiB
/// of entropy tail saturates every scan.
const MAX_INPUT_LEN: usize = 8 * 1024;

/// Fixed canonical DC Huffman table: 15 codes of lengths 1..=15 plus
/// two of length 16, covering values 0..=16 — every SSSS category a
/// lossless scan can emit, including the 16-bit-precision SSSS=16.
/// The tree is exactly full (Kraft sum = 1), so all-ones bit padding
/// decodes as the deepest symbol instead of dying in the table walk.
const DHT_BITS: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2];

/// Build a minimal lossless JPEG around fuzz-controlled entropy bytes.
///
/// Layout:
///   SOI                          FF D8
///   DHT (Huffman mode only)      FF C4 ... (fixed canonical table)
///   [optional DAC (arith mode)]  FF CC 00 04 00 <L:U>
///   [optional DRI]               FF DD 00 04 00 02
///   SOF3 / SOF11                 FF C3 / FF CB ...
///   SOS                          FF DA ... Ss=pred Se=0 Ah=0 Al=pt
///   <entropy bytes>              fuzz-controlled tail
///   [DNL when Y=0 mode]          FF DC 00 04 <NL:u16-be>
///   EOI                          FF D9
///
/// Control bytes:
///   data[0] bit 0 — arithmetic (SOF11) vs Huffman (SOF3)
///           bit 1 — 3 components vs 1 component
///           bit 2 — DNL mode: SOF Y = 0 + DNL segment after the scan
///           bit 3 — DRI segment (restart interval = 2 samples)
///           bit 4 — DAC segment (arith mode only)
///           bit 5 — corrupt the DNL: NL = 0 (DNL mode only)
///   data[1] low nibble  — SOS Ss (predictor select, raw 0..=15)
///           high nibble — SOS Al (point transform, raw 0..=15)
///   data[2] — precision P = 2 + (byte mod 15)  → 2..=16
///   data[3] — width  = 1 + high nibble, height = 1 + low nibble
///   data[4..] — entropy tail (0xFF bytes stuffed on even popcount)
fn build_lossless_envelope(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 5 {
        return None;
    }
    let ctrl = data[0];
    let arith = ctrl & 0x01 != 0;
    let three_comp = ctrl & 0x02 != 0;
    let dnl_mode = ctrl & 0x04 != 0;
    let with_dri = ctrl & 0x08 != 0;
    let with_dac = arith && (ctrl & 0x10 != 0);
    let dnl_zero = ctrl & 0x20 != 0;

    let pred = data[1] & 0x0F;
    let pt = data[1] >> 4;
    let precision = 2 + (data[2] % 15);
    let width = 1 + (data[3] >> 4) as u16;
    let height = 1 + (data[3] & 0x0F) as u16;
    let sof_y = if dnl_mode { 0 } else { height };

    let mut j: Vec<u8> = Vec::with_capacity(320 + data.len());

    // SOI
    j.extend_from_slice(&[0xFF, 0xD8]);

    if !arith {
        // DHT — one DC-class table at destination 0.
        // Ld = 2 + 1 + 16 + 17 = 36.
        j.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x24, 0x00]);
        j.extend_from_slice(&DHT_BITS);
        for v in 0u8..=16 {
            j.push(v);
        }
    } else if with_dac {
        // DAC — one DC entry, Tb = 0, L = 0 / U = 2 (off the (0, 1)
        // default so the conditioning shifts run on non-trivial args).
        j.extend_from_slice(&[0xFF, 0xCC, 0x00, 0x04, 0x00, 0x20]);
    }

    if with_dri {
        // DRI — restart every 2 lossless MCUs (= 2 samples).
        j.extend_from_slice(&[0xFF, 0xDD, 0x00, 0x04, 0x00, 0x02]);
    }

    // SOF3 (Huffman lossless) or SOF11 (arithmetic lossless).
    let sof = if arith { 0xCB } else { 0xC3 };
    if !three_comp {
        // Nf = 1 → Lf = 11.
        j.extend_from_slice(&[
            0xFF,
            sof,
            0x00,
            0x0B,
            precision,
            (sof_y >> 8) as u8,
            sof_y as u8,
            (width >> 8) as u8,
            width as u8,
            0x01,
            0x01,
            0x11,
            0x00,
        ]);
    } else {
        // Nf = 3 → Lf = 17; all components H = V = 1 (lossless
        // interleave is sample-per-component, no subsampling).
        j.extend_from_slice(&[
            0xFF,
            sof,
            0x00,
            0x11,
            precision,
            (sof_y >> 8) as u8,
            sof_y as u8,
            (width >> 8) as u8,
            width as u8,
            0x03,
            0x01,
            0x11,
            0x00,
            0x02,
            0x11,
            0x00,
            0x03,
            0x11,
            0x00,
        ]);
    }

    // SOS — Ss carries the predictor select, Se must be 0 for
    // lossless, Ah = 0, Al carries Pt. Table selectors all 0.
    let ss = pred;
    let ahal = pt & 0x0F;
    if !three_comp {
        j.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, ss, 0x00, ahal]);
    } else {
        j.extend_from_slice(&[
            0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, ss, 0x00, ahal,
        ]);
    }

    // Entropy tail with a deterministic 0xFF-stuffing knob: an FF at
    // an even tail offset gets a 0x00 stuffed behind it (legitimate
    // stuffed byte, §B.1.1.5); an FF at an odd offset is left raw so
    // the next fuzz byte decides whether an embedded marker appears
    // mid-scan (driving the marker-trap / unexpected-marker paths).
    for (i, &b) in data[4..].iter().enumerate() {
        j.push(b);
        if b == 0xFF && i % 2 == 0 {
            j.push(0x00);
        }
    }

    if dnl_mode {
        let nl = if dnl_zero { 0u16 } else { height };
        j.extend_from_slice(&[0xFF, 0xDC, 0x00, 0x04, (nl >> 8) as u8, nl as u8]);
    }

    // EOI
    j.extend_from_slice(&[0xFF, 0xD9]);

    Some(j)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 || data.len() > MAX_INPUT_LEN {
        return;
    }

    let Some(jpeg) = build_lossless_envelope(data) else {
        return;
    };

    let params = CodecParameters::video(CodecId::new("mjpeg"));
    let Ok(mut dec) = oxideav_mjpeg::decoder::make_decoder(&params) else {
        return;
    };

    let pkt = Packet::new(0, TimeBase::new(1, 30), jpeg);
    if dec.send_packet(&pkt).is_err() {
        return;
    }
    let _ = dec.receive_frame();
});
