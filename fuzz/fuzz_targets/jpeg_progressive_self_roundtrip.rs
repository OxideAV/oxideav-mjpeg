#![no_main]

//! oxideav-mjpeg progressive (SOF2) encode → oxideav-mjpeg decode.
//!
//! Same shape as `jpeg_self_roundtrip` but exercises the
//! `encode_jpeg_progressive` entry point: a DC-first interleaved
//! scan followed by per-component AC-band scans (`Ah=Al=0`,
//! spectral-selection only). The decoder takes the
//! coefficient-accumulator path used for progressive bitstreams
//! and runs the inverse DCT once at EOI. Pixel-fidelity bound is
//! `SELF_TOLERANCE` LSBs per YUV plane — same as the baseline
//! self-roundtrip, since both encoders share quant tables and the
//! progressive scan decomposition adds no additional quantisation.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Decoder as _, Frame, Packet, PixelFormat, TimeBase};
use oxideav_mjpeg::encoder::encode_jpeg_progressive;
use oxideav_mjpeg_fuzz::{
    assert_within, rgb_image_from_fuzz_input, rgb_to_yuv444p_frame, SELF_TOLERANCE,
};

fuzz_target!(|data: &[u8]| {
    let Some((width, height, rgb)) = rgb_image_from_fuzz_input(data) else {
        return;
    };

    let frame = rgb_to_yuv444p_frame(rgb, width, height);
    let pix = PixelFormat::Yuv444P;

    let bitstream = match encode_jpeg_progressive(&frame, width, height, pix, 100) {
        Ok(b) => b,
        Err(_) => return,
    };

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

    for plane in 0..3 {
        let src = &frame.planes[plane];
        let dst = &v.planes[plane];
        let w = width as usize;
        let h = height as usize;
        let mut a = Vec::with_capacity(w * h);
        let mut b = Vec::with_capacity(w * h);
        for j in 0..h {
            for i in 0..w {
                a.push(src.data[j * src.stride + i]);
                b.push(dst.data[j * dst.stride + i]);
            }
        }
        assert_within(
            &a,
            &b,
            SELF_TOLERANCE,
            &format!("progressive self-roundtrip plane {plane}"),
        );
    }
});
