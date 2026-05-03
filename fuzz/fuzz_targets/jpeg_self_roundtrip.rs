#![no_main]

//! oxideav-mjpeg encode → oxideav-mjpeg decode round-trip.
//!
//! Synthesises a YUV444P frame from the fuzz input via RGB, runs it
//! through `encode_jpeg` at quality 100 (where the quant tables are
//! all-ones and the only loss is IDCT rounding), decodes via the
//! public decoder trait, and asserts the round-tripped YUV samples
//! agree with the source to within `SELF_TOLERANCE` LSBs per plane.
//! Comparing in YUV space — the unit JPEG actually operates on —
//! keeps the tolerance meaningful: an RGB-space comparison would
//! also fold in the BT.601 colour-conversion rounding, which can
//! push past ±2 LSB even on a bit-perfect codec.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Decoder as _, Frame, Packet, PixelFormat, TimeBase};
use oxideav_mjpeg::encoder::encode_jpeg;
use oxideav_mjpeg_fuzz::{
    assert_within, rgb_image_from_fuzz_input, rgb_to_yuv444p_frame, SELF_TOLERANCE,
};

fuzz_target!(|data: &[u8]| {
    let Some((width, height, rgb)) = rgb_image_from_fuzz_input(data) else {
        return;
    };

    let frame = rgb_to_yuv444p_frame(rgb, width, height);
    let pix = PixelFormat::Yuv444P;

    let bitstream = match encode_jpeg(&frame, width, height, pix, 100) {
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

    // Compare each YUV plane within the visible area against the
    // source frame, plane-by-plane. The encoder may pad to MCU
    // boundaries, so we walk only `width * height` samples.
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
            &format!("self-roundtrip plane {plane}"),
        );
    }
});
