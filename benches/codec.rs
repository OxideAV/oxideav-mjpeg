//! Criterion benchmarks for the oxideav-mjpeg encode + decode hot paths.
//!
//! Round 209 (depth-mode benchmarks): the codec has accumulated
//! several distinct entropy / coefficient paths (baseline SOF0
//! Huffman + 8x8 DCT-II/IDCT, progressive SOF2 multi-scan, lossless
//! SOF3 per-sample predictor) and the encoder runs each block
//! through `forward_dct` → `quantise` → Huffman, while the decoder
//! runs Huffman → `dequantise` → `inverse_dct` per block. This bench
//! file is the A/B harness for those hot paths, organised so any
//! future tuning (SIMD-able coefficient scaling, predictor-loop
//! tightening, Huffman-table layout tweaks) has a self-contained
//! reproducible reference.
//!
//! Every fixture is built deterministically in-bench from an
//! xorshift32 seed plus cheap arithmetic gradients — no committed
//! payload files, no `docs/` reads, no external library calls — so
//! the numbers are reproducible against any future tweak. Matches
//! the shape of the flac / tta / tiff / huffyuv / pcx benches so
//! cross-codec numbers stay comparable.
//!
//! Scenarios:
//!
//!   - **baseline_encode_yuv420_256x256_q75** — the canonical JPEG
//!     encode workload: 256x256 4:2:0 YUV at quality 75. Exercises
//!     the full SOF0 path (forward DCT, AAN-style quantise, Huffman
//!     run-length encode, marker emission). Sets the baseline for
//!     the encoder's per-MCU throughput.
//!   - **baseline_encode_yuv444_64x64_q75** — small 4:4:4 fixture;
//!     useful for measuring per-call overhead (header emission,
//!     Huffman-table construction) against the per-block cost.
//!   - **baseline_decode_yuv420_256x256_q75** — the inverse: parse
//!     SOI/DQT/SOF0/DHT/SOS, Huffman-decode each block, dequantise,
//!     inverse-DCT, reassemble planes. Driven by the public
//!     `Decoder` trait so the bench tracks the same code path
//!     application callers exercise.
//!   - **progressive_encode_yuv420_64x64_q75** — SOF2 spectral-
//!     selection decomposition. Exercises the 7-SOS scan emission
//!     and the per-scan coefficient walk.
//!   - **lossless_encode_gray_pred1_256x256** — SOF3 grayscale
//!     encode with the simplest predictor (Ra / left). Measures
//!     the per-sample residual + Huffman-magnitude path without
//!     the 2-D predictor overhead.
//!   - **lossless_encode_gray_pred4_256x256** — SOF3 grayscale
//!     encode with predictor 4 (Ra + Rb − Rc), the most expensive
//!     2-D variant in T.81 Table H.1. A/B against `pred1` measures
//!     the predictor-loop cost.
//!
//! Run with:
//!     cargo bench -p oxideav-mjpeg --bench codec

#![cfg(feature = "registry")]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase, VideoFrame};
use oxideav_mjpeg::encoder::{
    encode_jpeg, encode_jpeg_progressive, encode_lossless_jpeg_grayscale,
};
use oxideav_mjpeg::registry::make_decoder;

// ---- deterministic fixtures -------------------------------------------

/// xorshift32 — same shape as the flac/tta/tiff benches so
/// cross-codec numbers stay comparable.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build a (width, height) 4:2:0 / 4:2:2 / 4:4:4 YUV frame whose
/// luma is a gentle gradient + low-amplitude noise (closer to a
/// natural image than pure random; the entropy coder gets meaningful
/// run-length opportunities). Chroma planes carry slow ramps so the
/// DC predictor sees realistic deltas. Strides are tight (= plane
/// width).
fn make_natural_frame(w: u32, h: u32, pix: PixelFormat) -> VideoFrame {
    let (cw, ch): (u32, u32) = match pix {
        PixelFormat::Yuv444P => (w, h),
        PixelFormat::Yuv422P => (w.div_ceil(2), h),
        PixelFormat::Yuv420P => (w.div_ceil(2), h.div_ceil(2)),
        _ => panic!("unsupported pixel format in bench fixture"),
    };
    let y_stride = w as usize;
    let mut y = vec![0u8; y_stride * h as usize];
    let mut rng = 0xA5A5_5A5Au32;
    for j in 0..h as usize {
        for i in 0..w as usize {
            // Gradient + horizontal triangle + small noise jitter.
            let base = ((i + j) as i32) & 0xFF;
            let phase = (i as i32) & 31;
            let tri = if phase < 16 { phase } else { 31 - phase };
            let noise = (xorshift32(&mut rng) & 0x07) as i32 - 4;
            let val = (base + tri + noise).clamp(0, 255) as u8;
            y[j * y_stride + i] = val;
        }
    }
    let cb_stride = cw as usize;
    let cr_stride = cw as usize;
    let mut cb = vec![0u8; cb_stride * ch as usize];
    let mut cr = vec![0u8; cr_stride * ch as usize];
    for j in 0..ch as usize {
        for i in 0..cw as usize {
            let cb_v = 128 + ((i as i32) - (cw as i32) / 2) / 4;
            let cr_v = 128 + ((j as i32) - (ch as i32) / 2) / 4;
            cb[j * cb_stride + i] = cb_v.clamp(0, 255) as u8;
            cr[j * cr_stride + i] = cr_v.clamp(0, 255) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: y_stride,
                data: y,
            },
            VideoPlane {
                stride: cb_stride,
                data: cb,
            },
            VideoPlane {
                stride: cr_stride,
                data: cr,
            },
        ],
    }
}

/// Build a width*height 8-bit grayscale buffer that's a gentle
/// gradient plus a low-amplitude horizontal sinusoid; tight stride
/// (= width). The predictor-loop bench needs spatial correlation to
/// produce small residuals — pure random input would degenerate the
/// per-sample magnitude category to its worst case and stop measuring
/// the predictor.
fn make_natural_grayscale(w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w as usize) * (h as usize)];
    let mut rng = 0x1234_ABCDu32;
    for j in 0..h as usize {
        for i in 0..w as usize {
            let base = ((i + j) as i32) & 0xFF;
            let phase = (i as i32) & 31;
            let tri = if phase < 16 { phase } else { 31 - phase };
            let noise = (xorshift32(&mut rng) & 0x07) as i32 - 4;
            let val = (base + tri + noise).clamp(0, 255) as u8;
            out[j * (w as usize) + i] = val;
        }
    }
    out
}

// ---- benches: baseline encode -----------------------------------------

fn bench_baseline_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_encode");

    let (w, h) = (256u32, 256u32);
    let frame_420 = make_natural_frame(w, h, PixelFormat::Yuv420P);
    let pixels = (w as u64) * (h as u64);
    group.throughput(Throughput::Elements(pixels));
    group.bench_function("yuv420_256x256_q75", |b| {
        b.iter(|| {
            let out = encode_jpeg(
                black_box(&frame_420),
                black_box(w),
                black_box(h),
                black_box(PixelFormat::Yuv420P),
                black_box(75),
            )
            .expect("baseline 4:2:0 encode");
            black_box(out);
        })
    });

    let (w_s, h_s) = (64u32, 64u32);
    let frame_444 = make_natural_frame(w_s, h_s, PixelFormat::Yuv444P);
    group.throughput(Throughput::Elements((w_s as u64) * (h_s as u64)));
    group.bench_function("yuv444_64x64_q75", |b| {
        b.iter(|| {
            let out = encode_jpeg(
                black_box(&frame_444),
                black_box(w_s),
                black_box(h_s),
                black_box(PixelFormat::Yuv444P),
                black_box(75),
            )
            .expect("baseline 4:4:4 encode");
            black_box(out);
        })
    });

    group.finish();
}

// ---- benches: baseline decode -----------------------------------------

fn bench_baseline_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_decode");

    let (w, h) = (256u32, 256u32);
    let frame_420 = make_natural_frame(w, h, PixelFormat::Yuv420P);
    let jpeg = encode_jpeg(&frame_420, w, h, PixelFormat::Yuv420P, 75)
        .expect("baseline 4:2:0 encode for decode bench");

    let mut params = CodecParameters::video(CodecId::new("mjpeg"));
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::Yuv420P);

    group.throughput(Throughput::Elements((w as u64) * (h as u64)));
    group.bench_function("yuv420_256x256_q75", |b| {
        b.iter(|| {
            let mut dec = make_decoder(&params).expect("make_decoder");
            let pkt = Packet::new(0, TimeBase::new(1, 25), jpeg.clone());
            dec.send_packet(black_box(&pkt)).expect("send_packet");
            let frame = dec.receive_frame().expect("receive_frame");
            if let Frame::Video(vf) = &frame {
                black_box(vf.planes.len());
            }
            black_box(frame);
        })
    });

    group.finish();
}

// ---- benches: progressive encode --------------------------------------

fn bench_progressive_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("progressive_encode");

    let (w, h) = (64u32, 64u32);
    let frame = make_natural_frame(w, h, PixelFormat::Yuv420P);
    group.throughput(Throughput::Elements((w as u64) * (h as u64)));
    group.bench_function("yuv420_64x64_q75", |b| {
        b.iter(|| {
            let out = encode_jpeg_progressive(
                black_box(&frame),
                black_box(w),
                black_box(h),
                black_box(PixelFormat::Yuv420P),
                black_box(75),
            )
            .expect("progressive encode");
            black_box(out);
        })
    });

    group.finish();
}

// ---- benches: lossless encode (predictor A/B) -------------------------

fn bench_lossless_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("lossless_encode");

    let (w, h) = (256u32, 256u32);
    let samples = make_natural_grayscale(w, h);
    let stride = w as usize;
    let pixels = (w as u64) * (h as u64);
    group.throughput(Throughput::Elements(pixels));

    for predictor in [1u8, 4u8] {
        let label = match predictor {
            1 => "gray_pred1_256x256",
            4 => "gray_pred4_256x256",
            _ => unreachable!(),
        };
        group.bench_function(label, |b| {
            b.iter(|| {
                let out = encode_lossless_jpeg_grayscale(
                    black_box(w),
                    black_box(h),
                    black_box(&samples),
                    black_box(stride),
                    black_box(8),
                    black_box(predictor),
                )
                .expect("lossless grayscale encode");
                black_box(out);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_baseline_encode,
    bench_baseline_decode,
    bench_progressive_encode,
    bench_lossless_encode,
);
criterion_main!(benches);
