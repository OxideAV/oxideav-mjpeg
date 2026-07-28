#![no_main]

//! Bit-exact self-roundtrip through the lossless JPEG encoders and the
//! public decoder, on fuzz-derived pixels and fuzz-chosen parameters.
//!
//! Unlike the `lossless_decode` robustness envelope (whose bar is "no
//! panic" on garbage entropy), this target carries a hard oracle:
//! lossless JPEG (T.81 Annex H) guarantees exact reconstruction, so
//! for every parameter combination the encoder accepts, the decoded
//! samples must equal the source samples after the point transform —
//! `(s >> Pt) << Pt` — with no tolerance window at all. Any 1-LSB
//! excursion is a real bug in the predictor arithmetic, the modulo
//! reduction, the SSSS=16 half-modulus case, the restart re-seed, or
//! the Pt output shift on either side.
//!
//! Parameter space sampled per iteration:
//!
//! * entropy coder — Huffman (SOF3) vs arithmetic Q-coder (SOF11)
//! * layout — grayscale (`Nf = 1`, full precision ladder 2..=16) vs
//!   three-component (`Nf = 3`, precision 8 → packed `Rgb24` output
//!   or precision 16 → packed `Rgb48Le` output)
//! * predictor 1..=7 (T.81 Table H.1)
//! * point transform `Pt < P`
//! * restart interval 0 / 3 / 6 / 9 / 12 samples
//! * image geometry (width 1..=12, height from available input)
//!
//! Decoder output shaping follows `shape_lossless_frame`'s documented
//! policy: encoder *input* samples are 1 byte for P ≤ 8 and 2-byte LE
//! above, while decoder *output* samples are 1 byte only at exactly
//! P = 8 (`Gray8` / packed `Rgb24`) and 2-byte LE for every other
//! precision (`Gray16Le`-class widening / packed `Rgb48Le`).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_mjpeg::encoder::{
    encode_lossless_arith_jpeg_grayscale_with_opts, encode_lossless_arith_jpeg_rgb_with_opts,
    encode_lossless_jpeg_grayscale_with_opts, encode_lossless_jpeg_rgb_with_opts,
};
use oxideav_mjpeg::registry::make_decoder;

/// Cap the fuzz input: 6 header bytes + up to 1024 px × 3 comp ×
/// 2 bytes/sample.
const MAX_INPUT_LEN: usize = 6 + 1024 * 6;
/// Total pixel cap — keeps the sample-serial encode/decode fast.
const MAX_PIXELS: usize = 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 || data.len() > MAX_INPUT_LEN {
        return;
    }

    let ctrl = data[0];
    let arith = ctrl & 0x01 != 0;
    let rgb = ctrl & 0x02 != 0;
    let predictor = 1 + data[1] % 7;
    let precision: u8 = if rgb {
        // 3-component output shaping is packed at P = 8 (Rgb24) and
        // P = 16 (Rgb48Le); the planar mid-precision shapes are pinned
        // by the integration suite instead.
        if data[2] & 1 == 0 {
            8
        } else {
            16
        }
    } else {
        2 + data[2] % 15
    };
    let pt = data[3] % precision;
    let restart_interval = (data[4] % 5) as u16 * 3;
    let width = 1 + (data[5] % 12) as usize;

    let ncomp: usize = if rgb { 3 } else { 1 };
    // Encoder-side input sample width (T.81 sample layout).
    let bps: usize = if precision <= 8 { 1 } else { 2 };
    // Decoder-side output sample width: 1 byte only at exactly P = 8;
    // every other precision widens onto a 16-bit LE container.
    let out_bps: usize = if precision == 8 { 1 } else { 2 };
    let tail = &data[6..];
    let px_avail = tail.len() / (ncomp * bps);
    let total_px = px_avail.min(MAX_PIXELS);
    let height = total_px / width;
    if height == 0 {
        return;
    }
    let n = width * height;
    let max = ((1u32 << precision) - 1) as u16;

    // Carve per-component sample planes out of the tail, masking each
    // sample into the declared precision range (the encoder rejects
    // out-of-range samples up front, which would starve the oracle).
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(ncomp);
    let mut expected: Vec<Vec<u16>> = Vec::with_capacity(ncomp);
    for c in 0..ncomp {
        let mut plane = vec![0u8; n * bps];
        let mut exp = vec![0u16; n];
        for i in 0..n {
            let off = (c * n + i) * bps;
            let raw = if bps == 1 {
                tail[off] as u16
            } else {
                (tail[off] as u16) | ((tail[off + 1] as u16) << 8)
            };
            let s = raw & max;
            if bps == 1 {
                plane[i] = s as u8;
            } else {
                plane[i * 2] = (s & 0xFF) as u8;
                plane[i * 2 + 1] = (s >> 8) as u8;
            }
            // Decoder contract: reconstruct (s >> Pt) << Pt exactly.
            exp[i] = (s >> pt) << pt;
        }
        planes.push(plane);
        expected.push(exp);
    }

    let w = width as u32;
    let h = height as u32;
    let stride = width * bps;
    let jpeg = if rgb {
        let refs = [
            planes[0].as_slice(),
            planes[1].as_slice(),
            planes[2].as_slice(),
        ];
        let strides = [stride; 3];
        if arith {
            encode_lossless_arith_jpeg_rgb_with_opts(
                w,
                h,
                refs,
                strides,
                precision,
                predictor,
                restart_interval,
                pt,
            )
        } else {
            encode_lossless_jpeg_rgb_with_opts(
                w,
                h,
                refs,
                strides,
                precision,
                predictor,
                restart_interval,
                pt,
            )
        }
    } else if arith {
        encode_lossless_arith_jpeg_grayscale_with_opts(
            w,
            h,
            &planes[0],
            stride,
            precision,
            predictor,
            restart_interval,
            pt,
        )
    } else {
        encode_lossless_jpeg_grayscale_with_opts(
            w,
            h,
            &planes[0],
            stride,
            precision,
            predictor,
            restart_interval,
            pt,
        )
    };
    // Every sampled parameter combination is valid per the encoder
    // docs; a reject here would itself be a contract break.
    let jpeg = jpeg.unwrap_or_else(|e| {
        panic!(
            "lossless encode rejected valid params \
             (arith={arith} rgb={rgb} P={precision} pred={predictor} \
             Pt={pt} ri={restart_interval} {w}x{h}): {e:?}"
        )
    });

    let params = CodecParameters::video(CodecId::new("mjpeg"));
    let mut dec = make_decoder(&params).expect("make_decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), jpeg))
        .expect("send_packet");
    let frame = dec.receive_frame().unwrap_or_else(|e| {
        panic!(
            "decoder rejected a stream our lossless encoder produced \
             (arith={arith} rgb={rgb} P={precision} pred={predictor} \
             Pt={pt} ri={restart_interval} {w}x{h}): {e:?}"
        )
    });
    let Frame::Video(v) = frame else {
        panic!("expected a video frame");
    };

    // Exact-reconstruction oracle.
    assert_eq!(v.planes.len(), 1, "expected one packed output plane");
    let out = &v.planes[0].data;
    let ostride = v.planes[0].stride;
    let px_bytes = ncomp * out_bps;
    for j in 0..height {
        for i in 0..width {
            for c in 0..ncomp {
                let base = j * ostride + i * px_bytes + c * out_bps;
                let got = if out_bps == 1 {
                    out[base] as u16
                } else {
                    (out[base] as u16) | ((out[base + 1] as u16) << 8)
                };
                let want = expected[c][j * width + i];
                assert_eq!(
                    got, want,
                    "lossless mismatch at ({i},{j}) comp {c} \
                     (arith={arith} rgb={rgb} P={precision} pred={predictor} \
                     Pt={pt} ri={restart_interval} {w}x{h})"
                );
            }
        }
    }
});
