//! Restart-interval-parallel decode (`DecodeOptions::threads`): every
//! result — picture or error — equals the single-threaded decode.

use oxideav_mjpeg::{
    ChromaSubsampling, DecodeOptions, EncodeOptions, JpegImage, decode_with, encode_rgb8,
};

fn picture(w: u32, h: u32) -> Vec<u8> {
    let mut s = 0x9E37_79B9u32;
    (0..w * h * 3)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let (x, y) = ((i / 3) % w, (i / 3) / w);
            ((x * 3 + y * 5 + i % 3 * 40) as u8).wrapping_add((s % 23) as u8)
        })
        .collect()
}

fn decode_n(jpeg: &[u8], threads: usize) -> Result<JpegImage, String> {
    decode_with(jpeg, &DecodeOptions::new().with_threads(threads)).map_err(|e| format!("{e:?}"))
}

/// Byte offsets of the `RSTm` markers of an entropy-coded stream.
fn rst_offsets(jpeg: &[u8]) -> Vec<usize> {
    (0..jpeg.len() - 1)
        .filter(|&i| jpeg[i] == 0xFF && (0xD0..=0xD7).contains(&jpeg[i + 1]))
        .collect()
}

#[test]
fn threaded_decode_equals_serial_on_every_layout_and_interval() {
    for (w, h) in [(64u32, 48u32), (97, 61), (333, 219)] {
        let rgb = picture(w, h);
        for chroma in [
            ChromaSubsampling::Yuv444,
            ChromaSubsampling::Yuv422,
            ChromaSubsampling::Yuv420,
            ChromaSubsampling::Yuv411,
            ChromaSubsampling::GrayscaleOnly,
        ] {
            for ri in [1u16, 3, 7, 40] {
                let opts = EncodeOptions::new()
                    .with_quality(80)
                    .with_chroma(chroma)
                    .with_restart_interval(ri);
                let jpeg = encode_rgb8(w, h, &rgb, &opts).unwrap();
                let serial = decode_n(&jpeg, 1).unwrap();
                for t in [2, 3, 8] {
                    assert_eq!(
                        decode_n(&jpeg, t).unwrap(),
                        serial,
                        "{w}x{h} {chroma:?} ri {ri} threads {t}"
                    );
                }
            }
        }
    }
}

#[test]
fn damaged_intervals_fall_back_to_the_serial_result() {
    let (w, h) = (160u32, 96u32);
    let jpeg = encode_rgb8(
        w,
        h,
        &picture(w, h),
        &EncodeOptions::new().with_restart_interval(4),
    )
    .unwrap();
    let rst = rst_offsets(&jpeg);
    assert!(rst.len() > 4);
    let mut damaged: Vec<Vec<u8>> = Vec::new();
    // A missing marker, a short interval (bytes cut before a marker), an
    // extra marker, and a renumbered one.
    let mut v = jpeg.clone();
    v.drain(rst[2]..rst[2] + 2);
    damaged.push(v);
    let mut v = jpeg.clone();
    v.drain(rst[3] - 6..rst[3]);
    damaged.push(v);
    let mut v = jpeg.clone();
    v.splice(rst[1]..rst[1], [0xFF, 0xD5]);
    damaged.push(v);
    let mut v = jpeg.clone();
    v[rst[0] + 1] = 0xD6;
    damaged.push(v);
    for (i, d) in damaged.iter().enumerate() {
        let serial = decode_n(d, 1);
        for t in [2, 4] {
            assert_eq!(decode_n(d, t), serial, "damage {i}, threads {t}");
        }
    }
}

#[test]
fn threads_option_defaults_to_one_and_clamps() {
    assert_eq!(DecodeOptions::new().threads, 1);
    assert_eq!(DecodeOptions::new().with_threads(0).threads, 1);
    assert_eq!(DecodeOptions::new().with_threads(6).threads, 6);
}

#[cfg(feature = "registry")]
#[test]
fn registry_decoder_takes_the_execution_context_budget() {
    use oxideav_core::{CodecId, CodecParameters, ExecutionContext, Frame, Packet, TimeBase};
    let (w, h) = (128u32, 64u32);
    let jpeg = encode_rgb8(
        w,
        h,
        &picture(w, h),
        &EncodeOptions::new().with_restart_interval(2),
    )
    .unwrap();
    let frame = |threads: Option<usize>| {
        let mut dec =
            oxideav_mjpeg::registry::make_decoder(&CodecParameters::video(CodecId::new("mjpeg")))
                .unwrap();
        if let Some(t) = threads {
            dec.set_execution_context(&ExecutionContext::with_threads(t));
        }
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), jpeg.clone()))
            .unwrap();
        match dec.receive_frame().unwrap() {
            Frame::Video(v) => v
                .planes
                .into_iter()
                .map(|p| (p.stride, p.data))
                .collect::<Vec<_>>(),
            _ => panic!("video frame"),
        }
    };
    assert_eq!(frame(Some(4)), frame(None));
}

/// `EncodeOptions::threads` splits the colour conversion and the
/// transform + quantisation of a picture between workers; every block
/// is computed on its own, so the stream is the serial one byte for
/// byte — every layout, process, quantiser, odd sizes (partial chroma
/// cells, partial MCUs) and more workers than block rows.
#[test]
fn threaded_encode_equals_serial() {
    use oxideav_mjpeg::{JpegProcess, Quantisation, encode};
    for (w, h) in [(1u32, 1u32), (7, 3), (64, 48), (97, 61), (333, 219)] {
        let rgb = picture(w, h);
        for chroma in [
            ChromaSubsampling::Yuv444,
            ChromaSubsampling::Yuv422,
            ChromaSubsampling::Yuv420,
            ChromaSubsampling::Yuv411,
            ChromaSubsampling::GrayscaleOnly,
        ] {
            for (process, quantisation, arithmetic) in [
                (JpegProcess::Sequential, Quantisation::RateDistortion, false),
                (JpegProcess::Sequential, Quantisation::Nearest, false),
                (
                    JpegProcess::Progressive,
                    Quantisation::RateDistortion,
                    false,
                ),
                (JpegProcess::Sequential, Quantisation::Nearest, true),
            ] {
                let opts = EncodeOptions::new()
                    .with_quality(85)
                    .with_chroma(chroma)
                    .with_process(process)
                    .with_quantisation(quantisation)
                    .with_arithmetic(arithmetic);
                assert_eq!(opts.threads, 1, "serial by default");
                let serial = encode_rgb8(w, h, &rgb, &opts).unwrap();
                for t in [0usize, 2, 3, 8, 64] {
                    let out = encode_rgb8(w, h, &rgb, &opts.clone().with_threads(t)).unwrap();
                    assert!(
                        out == serial,
                        "{w}x{h} {chroma:?} {process:?} {quantisation:?} threads {t}"
                    );
                }
                // The planar entry point too.
                let img = decode_with(&serial, &DecodeOptions::new()).unwrap();
                let a = encode(&img, &opts).unwrap();
                let b = encode(&img, &opts.clone().with_threads(5)).unwrap();
                assert!(a == b, "{w}x{h} {chroma:?} planar");
            }
        }
    }
}

/// The registry encoder takes its budget from
/// `Encoder::set_execution_context` and writes the same packet.
#[cfg(feature = "registry")]
#[test]
fn registry_encoder_thread_budget_keeps_the_bytes() {
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{CodecId, CodecParameters, ExecutionContext, Frame, VideoFrame};

    let (w, h) = (160u32, 96u32);
    let rgb = picture(w, h);
    let mut p = CodecParameters::video(CodecId::new("mjpeg"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(oxideav_core::PixelFormat::Rgb24);
    let frame = Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: w as usize * 3,
            data: rgb,
        }],
    });
    let mut packets = Vec::new();
    for threads in [1usize, 4] {
        let mut enc = oxideav_mjpeg::registry::make_encoder(&p).unwrap();
        let mut ctx = ExecutionContext::serial();
        ctx.threads = threads;
        enc.set_execution_context(&ctx);
        enc.send_frame(&frame).unwrap();
        packets.push(enc.receive_packet().unwrap().data);
    }
    assert!(packets[0] == packets[1]);
}

/// `ConvertOptions::threads` / `DecodeOptions::threads` split the RGB
/// conversion into row bands (and build the two chroma planes on two
/// workers): the pixels do not depend on the worker count, for every
/// reconstruction, layout and odd size.
#[test]
fn threaded_rgb_conversion_equals_serial() {
    use oxideav_mjpeg::{ChromaUpsampling, ConvertOptions, decode, decode_rgb8_with};
    assert_eq!(ConvertOptions::new().threads, 1);
    for (w, h) in [(1u32, 1u32), (5, 3), (97, 61), (333, 219)] {
        let rgb = picture(w, h);
        for chroma in [
            ChromaSubsampling::Yuv444,
            ChromaSubsampling::Yuv422,
            ChromaSubsampling::Yuv420,
            ChromaSubsampling::Yuv411,
        ] {
            let opts = EncodeOptions::new().with_quality(80).with_chroma(chroma);
            let jpeg = encode_rgb8(w, h, &rgb, &opts).unwrap();
            let img = decode(&jpeg).unwrap();
            for up in [
                ChromaUpsampling::LumaGuided,
                ChromaUpsampling::Linear,
                ChromaUpsampling::Replicate,
                ChromaUpsampling::Triangular,
            ] {
                let serial = ConvertOptions::new().with_upsampling(up);
                let want = img.to_rgb8_with(&serial);
                let want_a = img.to_rgba8_with(&serial);
                for t in [0usize, 2, 3, 8, 500] {
                    let tag = format!("{w}x{h} {chroma:?} {up:?} threads {t}");
                    let o = serial.with_threads(t);
                    assert!(img.to_rgb8_with(&o) == want, "{tag}");
                    assert!(img.to_rgba8_with(&o) == want_a, "{tag} rgba");
                    let d = DecodeOptions::new().with_upsampling(up).with_threads(t);
                    assert!(
                        decode_rgb8_with(&jpeg, &d).unwrap().data == want,
                        "{tag} decode"
                    );
                }
            }
        }
    }
}
