#![no_main]

//! Encoder-in-loop roundtrip through the hierarchical (T.81 Annex J)
//! JPEG paths: `DHP` frame-header parsing, the non-differential first
//! stage, `EXP` reference-plane upsampling (§J.1.1.2), differential
//! `SOF5/SOF7/SOF13/SOF15` refinement stages, and the final
//! reconstruction clamp — none of which any other fuzz target reaches
//! (a random byte stream essentially never forms a valid multi-frame
//! hierarchical sequence).
//!
//! Eight modes are sampled per iteration, split into two oracle
//! classes:
//!
//! * **Bit-exact modes** — spatial-lossless pyramids (Huffman and
//!   arithmetic, grayscale and 3-component) and DCT pyramids
//!   *terminated by a differential lossless stage* (§K.7.2). The
//!   encoders document exact reconstruction, so every decoded sample
//!   must equal its source sample with zero tolerance.
//! * **Shape modes** — plain lossy DCT pyramids (Huffman and
//!   arithmetic-coded). No pixel oracle beyond "decodes Ok with the
//!   declared geometry": quantisation error is unbounded on synthetic
//!   noise, but the decode must still succeed and produce the right
//!   plane count/stride, and must never panic.
//!
//! Encoder-side parameter rejects (e.g. the `levels` geometry rule
//! rejecting a pyramid that would shrink a dimension to zero) simply
//! skip the iteration — the reject paths themselves are exercised in
//! the process.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_mjpeg::encoder::{
    encode_hierarchical_dct_arith_jpeg_grayscale, encode_hierarchical_dct_jpeg_grayscale,
    encode_hierarchical_dct_jpeg_grayscale_lossless_final, encode_hierarchical_dct_jpeg_yuv444,
    encode_hierarchical_dct_jpeg_yuv444_lossless_final,
    encode_hierarchical_lossless_arith_jpeg_grayscale, encode_hierarchical_lossless_jpeg_grayscale,
    encode_hierarchical_lossless_jpeg_rgb,
};
use oxideav_mjpeg::registry::make_decoder;

/// 5 header bytes + up to 1024 px × 3 comp × 2 bytes/sample.
const MAX_INPUT_LEN: usize = 5 + 1024 * 6;
/// Total pixel cap. Hierarchical encode runs a full decoder mirror per
/// stage, so keep images small for throughput.
const MAX_PIXELS: usize = 1024;

fn decode(jpeg: Vec<u8>) -> Option<oxideav_core::VideoFrame> {
    let params = CodecParameters::video(CodecId::new("mjpeg"));
    let mut dec = make_decoder(&params).ok()?;
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), jpeg))
        .ok()?;
    match dec.receive_frame() {
        Ok(Frame::Video(v)) => Some(v),
        _ => None,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 || data.len() > MAX_INPUT_LEN {
        return;
    }

    let mode = data[0] % 8;
    let levels = 1 + data[1] % 3;
    let predictor = 1 + data[2] % 7;
    let quality = 1 + data[2] % 100;
    // Grayscale lossless pyramids run the full precision ladder; the
    // 3-component pyramid sticks to the packed output precisions.
    let precision_gray = 2 + data[3] % 15;
    let precision_rgb: u8 = if data[3] & 1 == 0 { 8 } else { 16 };
    let width = 1 + (data[4] % 12) as usize;

    // `bps` is the encoder-side input sample width (1 byte for P <= 8,
    // 2-byte LE above); the decoder-side output width is computed
    // separately below — `shape_lossless_frame` emits 1-byte samples
    // only at exactly P = 8 and widens every other precision onto a
    // 16-bit LE container.
    let (ncomp, bps, exact): (usize, usize, bool) = match mode {
        0 => (1, if precision_gray <= 8 { 1 } else { 2 }, true), // lossless gray huff
        1 => (1, if precision_gray <= 8 { 1 } else { 2 }, true), // lossless gray arith
        2 => (3, if precision_rgb == 8 { 1 } else { 2 }, true),  // lossless rgb huff
        3 => (1, 1, true),                                       // DCT gray, lossless final
        4 => (3, 1, true),                                       // DCT yuv444, lossless final
        5 => (1, 1, false),                                      // DCT gray, lossy
        6 => (3, 1, false),                                      // DCT yuv444, lossy
        _ => (1, 1, false), // DCT gray arith (± lossless final)
    };
    let arith_lossless_final = data[1] & 0x80 != 0;
    let exact = if mode == 7 {
        arith_lossless_final
    } else {
        exact
    };

    let tail = &data[5..];
    let px_avail = tail.len() / (ncomp * bps);
    let total_px = px_avail.min(MAX_PIXELS);
    let height = total_px / width;
    if height == 0 {
        return;
    }
    let n = width * height;
    let precision = match mode {
        0 | 1 => precision_gray,
        2 => precision_rgb,
        _ => 8,
    };
    let max = ((1u32 << precision) - 1) as u16;
    let out_bps: usize = if precision == 8 { 1 } else { 2 };

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
            exp[i] = s;
        }
        planes.push(plane);
        expected.push(exp);
    }

    let w = width as u32;
    let h = height as u32;
    let stride = width * bps;
    let encoded = match mode {
        0 => encode_hierarchical_lossless_jpeg_grayscale(
            w, h, &planes[0], stride, precision, predictor, levels,
        ),
        1 => encode_hierarchical_lossless_arith_jpeg_grayscale(
            w, h, &planes[0], stride, precision, predictor, levels,
        ),
        2 => encode_hierarchical_lossless_jpeg_rgb(
            w,
            h,
            [&planes[0], &planes[1], &planes[2]],
            [stride; 3],
            precision,
            predictor,
            levels,
        ),
        3 => encode_hierarchical_dct_jpeg_grayscale_lossless_final(
            w, h, &planes[0], stride, quality, levels,
        ),
        4 => encode_hierarchical_dct_jpeg_yuv444_lossless_final(
            w,
            h,
            [&planes[0], &planes[1], &planes[2]],
            [stride; 3],
            quality,
            levels,
        ),
        5 => encode_hierarchical_dct_jpeg_grayscale(w, h, &planes[0], stride, quality, levels),
        6 => encode_hierarchical_dct_jpeg_yuv444(
            w,
            h,
            [&planes[0], &planes[1], &planes[2]],
            [stride; 3],
            quality,
            levels,
        ),
        _ => encode_hierarchical_dct_arith_jpeg_grayscale(
            w,
            h,
            &planes[0],
            stride,
            quality,
            levels,
            arith_lossless_final,
        ),
    };
    // The `levels` geometry rule may legitimately reject a pyramid for
    // this image size; skip such iterations.
    let Ok(jpeg) = encoded else {
        return;
    };

    // A stream our own hierarchical encoder emitted must decode.
    let v = decode(jpeg).unwrap_or_else(|| {
        panic!(
            "decoder rejected a hierarchical stream our encoder produced \
             (mode={mode} levels={levels} P={precision} pred={predictor} \
             q={quality} {w}x{h})"
        )
    });

    // Geometry oracle (all modes): plane count and visible extent.
    let out_planar_3 = ncomp == 3 && (mode == 4 || mode == 6);
    if out_planar_3 {
        assert_eq!(v.planes.len(), 3, "mode={mode}: expected planar output");
    } else {
        assert_eq!(v.planes.len(), 1, "mode={mode}: expected packed output");
    }

    if !exact {
        return;
    }

    // Bit-exact oracle for the lossless / lossless-final modes.
    for c in 0..ncomp {
        for j in 0..height {
            for i in 0..width {
                let got = if out_planar_3 {
                    v.planes[c].data[j * v.planes[c].stride + i] as u16
                } else {
                    let base = j * v.planes[0].stride + i * ncomp * out_bps + c * out_bps;
                    if out_bps == 1 {
                        v.planes[0].data[base] as u16
                    } else {
                        (v.planes[0].data[base] as u16) | ((v.planes[0].data[base + 1] as u16) << 8)
                    }
                };
                let want = expected[c][j * width + i];
                assert_eq!(
                    got, want,
                    "hierarchical mismatch at ({i},{j}) comp {c} \
                     (mode={mode} levels={levels} P={precision} \
                     pred={predictor} q={quality} {w}x{h})"
                );
            }
        }
    }
});
