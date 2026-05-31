#![no_main]

//! Decode arbitrary fuzz-supplied bytes through the SOF9 (extended
//! sequential, arithmetic-coded) entropy decoder.
//!
//! The harness wraps fuzz input in a minimal SOF9 JPEG envelope so the
//! arithmetic-specific code paths in `src/jpeg/arith.rs` and
//! `decode_arith_scan` (`src/decoder.rs`) actually run on every
//! iteration — the generic `decode` target almost never reaches them
//! because random bytes rarely line up into a valid SOF9 marker
//! sequence with the matching component count and DQT selectors.
//!
//! The contract under test is the same as the other robustness
//! targets: any malformed input must yield `Err(_)` or a benign
//! `Ok(_)` frame — never a panic, an `unwrap()` on `None`, a slice
//! OOB, a debug-build integer overflow, or an OOM-class allocation.
//!
//! Coverage of the panic surfaces specific to the arithmetic path:
//!
//! * `ArithDecoder::new` / `Initdec` (`src/jpeg/arith.rs` §D.2.4).
//!   Pulls two bytes from the scan; an empty entropy buffer makes
//!   the underlying `ByteSource` yield padding zeros — neither the
//!   preroll nor the subsequent `Renorm_d` may panic.
//! * `Context` indexing into the per-component DC `bins[0..49]` and
//!   AC `bins[0..245]` arrays. `DcStats::dc_context()` returns a
//!   value in `{0, 4, 8, 12, 16}`; `decode_dc_diff` then reads
//!   `bins[s0 + 1..=s0 + 3]`, and `decode_magnitude` walks up to
//!   `x1_base + 14 + bit-count` on the DC side and `217 + 14 +
//!   bit-count` on the AC side. Magnitude bin overflow is gated by
//!   the `category > 15` guard which the harness must drive.
//! * `decode_ac` zero-run bookkeeping. The loop advances `k` per
//!   zero coefficient and asserts `k <= se`; a maliciously crafted
//!   entropy stream that keeps the Q-coder in the "zero" branch
//!   must hit the `arith AC: run past Se` `Err`, not OOB
//!   `bins[3 * (k - 1)]`.
//! * `decode_arith_scan` restart handling. With a non-zero DRI the
//!   scan walks the MCU grid and at every `restart_interval`
//!   boundary calls `locate_next_marker_after` to advance the
//!   ByteSource past `RSTn`. A scan that's shorter than the MCU
//!   product (because the fuzzer ran out of input) must surface
//!   `arithmetic scan: missing restart marker mid-scan` rather
//!   than ranging past `scan.len()`.
//! * Optional `DAC` segment (`parse_dac` + `state.arith_dc` /
//!   `arith_ac` writeback). The harness optionally injects a DAC
//!   payload to drive `(L, U, Kx)` away from the spec defaults
//!   `(0, 1, 5)`, so `DcStats::dc_context` evaluates the
//!   `1i32 << (l - 1)` / `1i32 << u` shift expressions on
//!   fuzz-controlled `l` / `u` values up to the DAC parser's
//!   `u <= 15` cap.
//!
//! The harness deliberately routes through the public `Decoder`
//! trait (`make_decoder` + `send_packet` + `receive_frame`) so the
//! `registry.rs` plumbing is exercised on the same iterations.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Packet, TimeBase};

/// Discard inputs above this size. The arithmetic decoder is a
/// bit-serial Q-coder — large inputs only slow the fuzzer down.
/// 16 KiB easily covers a small grayscale SOF9 frame plus a few
/// restart intervals.
const MAX_INPUT_LEN: usize = 16 * 1024;

/// Build a minimal SOF9 JPEG around fuzz-controlled entropy bytes.
///
/// Layout:
///   SOI                        FF D8
///   DQT (Tq = 0, identity)     FF DB 00 43 00 [64 × 0x01]
///   [optional DAC]             FF CC 00 04 <Tc:Tb> <Cs>
///   SOF9 (1 or 3 components)   FF C9 ...
///   [optional DRI]             FF DD 00 04 <ri:u16-be>
///   SOS                        FF DA ...
///   <entropy bytes>            fuzz-controlled tail
///   EOI                        FF D9
///
/// The first byte of `data` is a control nibble selecting:
///   bit 0 — 1 component (grayscale) vs 3 components (4:4:4 YCbCr)
///   bit 1 — include a DAC segment
///   bit 2 — include a DRI segment (restart interval = 1 MCU)
///   bit 3 — sampling factor: 0 → H=V=1, 1 → H=2 V=1 (3-comp luma only)
///   bits 4..=7 — width / height nibble (1..=8 → image is 8..=64 px)
fn build_sof9_envelope(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }
    let ctrl = data[0];
    let three_comp = ctrl & 0x01 != 0;
    let with_dac = ctrl & 0x02 != 0;
    let with_dri = ctrl & 0x04 != 0;
    let h2_luma = (ctrl & 0x08 != 0) && three_comp;
    let dim_nib = ((ctrl >> 4) & 0x0F).clamp(1, 8);
    let dim: u16 = (dim_nib as u16) * 8; // 8..=64 px square

    // Reserve a generous upper bound to avoid reallocating.
    let mut j: Vec<u8> = Vec::with_capacity(256 + data.len());

    // SOI
    j.extend_from_slice(&[0xFF, 0xD8]);

    // DQT — single identity table at destination 0, 8-bit precision.
    j.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    j.extend_from_slice(&[0x01; 64]);

    // Optional DAC: drive (L, U) or (Kx) off the defaults so
    // `dc_context()` / `AcStats::kx` exercise non-trivial bin maps.
    // The DAC parser validates `tc <= 1`, `tb <= 3`, `u <= 15`,
    // `l <= u`, and AC `Cs ∈ 1..=63`; we feed values that pass
    // those checks unconditionally.
    if with_dac && data.len() >= 3 {
        // One DC entry: Tc=0 Tb=0, low nibble L=0, high nibble U=2
        // (slightly tighter than the default (0,1)).
        j.extend_from_slice(&[0xFF, 0xCC, 0x00, 0x04, 0x00, 0x20]);
    }

    // Optional DRI: restart every 1 MCU. Combined with a too-short
    // entropy tail this drives the "missing restart marker mid-scan"
    // branch.
    if with_dri {
        j.extend_from_slice(&[0xFF, 0xDD, 0x00, 0x04, 0x00, 0x01]);
    }

    // SOF9 — extended sequential, arithmetic-coded.
    if !three_comp {
        // Grayscale: Nf = 1, component id 1 with H=V=1, Tq=0.
        // Lf = 8 + 3*Nf = 11.
        j.extend_from_slice(&[
            0xFF,
            0xC9,
            0x00,
            0x0B,
            0x08,
            (dim >> 8) as u8,
            dim as u8,
            (dim >> 8) as u8,
            dim as u8,
            0x01,
            0x01,
            0x11,
            0x00,
        ]);
    } else {
        // 3-component 4:4:4 (or 4:2:2 with h2_luma): Nf = 3.
        // Lf = 8 + 3*Nf = 17.
        let luma_hv = if h2_luma { 0x21 } else { 0x11 };
        j.extend_from_slice(&[
            0xFF,
            0xC9,
            0x00,
            0x11,
            0x08,
            (dim >> 8) as u8,
            dim as u8,
            (dim >> 8) as u8,
            dim as u8,
            0x03,
            0x01,
            luma_hv,
            0x00,
            0x02,
            0x11,
            0x00,
            0x03,
            0x11,
            0x00,
        ]);
    }

    // SOS — interleaved scan over all SOF components, Ss=0, Se=63,
    // Ah=Al=0. Tdj/Taj nibbles set to 0/0 (DAC entry 0 picked up if
    // present; defaults otherwise).
    if !three_comp {
        // Ls = 6 + 2*Ns = 8.
        j.extend_from_slice(&[
            0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, // component 1, Td=0 Ta=0
            0x00, 0x3F, 0x00, // Ss, Se, Ah/Al
        ]);
    } else {
        // Ls = 6 + 2*Ns = 12.
        j.extend_from_slice(&[
            0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x3F, 0x00,
        ]);
    }

    // Entropy data — the fuzz tail. Stuff any literal 0xFF bytes with
    // a following 0x00 so the byte source's marker-trap path is
    // exercised on legitimate stuffed bytes (we still pass raw
    // unstuffed FFs occasionally by emitting `FF Mn` markers when
    // the fuzzer happens to feed a non-zero byte after an FF).
    let tail = &data[1..];
    for &b in tail {
        j.push(b);
        if b == 0xFF {
            // Half the time stuff, half the time let the next byte
            // determine whether this becomes a (possibly invalid)
            // embedded marker. We pick stuffing based on the byte
            // value to give the fuzzer a deterministic control knob.
            if b.count_ones() % 2 == 0 {
                j.push(0x00);
            }
        }
    }

    // EOI
    j.extend_from_slice(&[0xFF, 0xD9]);

    Some(j)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_LEN {
        return;
    }

    let Some(jpeg) = build_sof9_envelope(data) else {
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
