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
