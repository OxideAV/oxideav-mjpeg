#![no_main]

//! oxideav-mjpeg encode → libjpeg-turbo decode cross-validation.
//!
//! Encodes a synthesised YUV444P frame with `oxideav-mjpeg` at
//! quality 75, then decodes the resulting JPEG with libjpeg-turbo
//! (oracle) and oxideav-mjpeg (sanity check). Both decoded outputs
//! are projected to packed RGB and compared with `CROSS_TOLERANCE`.
//!
//! Skips silently when libturbojpeg isn't installed on the host.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    CodecId, CodecParameters, Decoder as _, Frame, Packet, PixelFormat, TimeBase, VideoFrame,
};
use oxideav_mjpeg::encoder::encode_jpeg;
use oxideav_mjpeg_fuzz::{
    assert_within, libjpeg, rgb_image_from_fuzz_input, rgb_to_yuv444p_frame, CROSS_TOLERANCE,
};

fuzz_target!(|data: &[u8]| {
    if !libjpeg::available() {
        return;
    }

    let Some((width, height, rgb)) = rgb_image_from_fuzz_input(data) else {
        return;
    };

    let frame = rgb_to_yuv444p_frame(rgb, width, height);
    let pix = PixelFormat::Yuv444P;
    let bitstream = match encode_jpeg(&frame, width, height, pix, 75) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Oracle: libjpeg-turbo decode of the oxideav-encoded bitstream.
    let oracle = match libjpeg::decode_to_rgb(&bitstream) {
        Some(d) => d,
        None => return,
    };
    if oracle.width != width || oracle.height != height {
        return;
    }

    // Sanity: the oxideav decoder MUST also accept its own encoder's
    // output (the self-roundtrip target asserts the lossy fidelity
    // separately; here we only need the decoded RGB to grade the
    // libjpeg-decoded RGB against). Decoding through oxideav also
    // exercises the bytes that libjpeg ignores (e.g. JFIF APP0
    // metadata).
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(width);
    dec_params.height = Some(height);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params)
        .expect("oxideav-mjpeg decoder construction failed");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), bitstream))
        .expect("oxideav-mjpeg send_packet failed");
    let Frame::Video(v) = dec.receive_frame().expect("oxideav-mjpeg decode failed") else {
        panic!("expected video frame");
    };
    let actual = yuv444p_to_rgb(&v, width, height);

    assert_within(
        &oracle.rgb,
        &actual,
        CROSS_TOLERANCE,
        "oxideav encode → libjpeg decode",
    );
});

/// Convert a 4:4:4 YUV `VideoFrame` to packed RGB. No chroma
/// upsampling needed — every luma sample has its own colocated
/// chroma sample, so this is the tightest possible YUV→RGB path.
fn yuv444p_to_rgb(frame: &VideoFrame, width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 3];
    let ys = frame.planes[0].stride;
    let us = frame.planes[1].stride;
    let vs = frame.planes[2].stride;
    for j in 0..h {
        for i in 0..w {
            let yy = frame.planes[0].data[j * ys + i] as f32;
            let cb = frame.planes[1].data[j * us + i] as f32 - 128.0;
            let cr = frame.planes[2].data[j * vs + i] as f32 - 128.0;
            let r = yy + 1.402 * cr;
            let g = yy - 0.344136 * cb - 0.714136 * cr;
            let b = yy + 1.772 * cb;
            let p = (j * w + i) * 3;
            out[p] = r.round().clamp(0.0, 255.0) as u8;
            out[p + 1] = g.round().clamp(0.0, 255.0) as u8;
            out[p + 2] = b.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}
