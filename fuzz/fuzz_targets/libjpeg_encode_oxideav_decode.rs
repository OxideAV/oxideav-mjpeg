#![no_main]

//! libjpeg-turbo encode → oxideav-mjpeg decode cross-validation.
//!
//! Encodes the synthesised RGB image with `tjCompress2` (RGB in,
//! 4:2:0 chroma, quality 75), then decodes the resulting JPEG with
//! both libjpeg-turbo (oracle) and oxideav-mjpeg (system under
//! test). Both decoded outputs are projected to packed RGB and
//! compared byte-by-byte with `CROSS_TOLERANCE`-LSB tolerance — see
//! the constant's docstring for the per-source slack budget.
//!
//! Skips silently when libturbojpeg isn't installed on the host.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Decoder as _, Frame, Packet, TimeBase, VideoFrame};
use oxideav_mjpeg_fuzz::{assert_within, libjpeg, rgb_image_from_fuzz_input, CROSS_TOLERANCE};

fuzz_target!(|data: &[u8]| {
    if !libjpeg::available() {
        return;
    }

    let Some((width, height, rgb)) = rgb_image_from_fuzz_input(data) else {
        return;
    };

    // Encode RGB via libjpeg-turbo (4:2:0 subsampling, quality 75).
    let bitstream = match libjpeg::encode_rgb(rgb, width, height, 75) {
        Some(b) => b,
        None => return,
    };

    // Reference: libjpeg-turbo decoded back to RGB. Used as the
    // oracle the oxideav decoder is graded against.
    let oracle = match libjpeg::decode_to_rgb(&bitstream) {
        Some(d) => d,
        None => return,
    };
    if oracle.width != width || oracle.height != height {
        return;
    }

    // Subject under test: same bitstream, oxideav-mjpeg decoder.
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

    // oxideav-mjpeg decodes 4:2:0 to `Yuv420P` planes. Convert back
    // to packed RGB with the same BT.601 full-range coefficients
    // libjpeg-turbo uses on its decode side.
    let actual = yuv_planar_to_rgb(&v, width, height);
    assert_within(
        &oracle.rgb,
        &actual,
        CROSS_TOLERANCE,
        "libjpeg encode → oxideav decode",
    );
});

/// Convert any-subsampling YUV planes to packed RGB with BT.601
/// full-range coefficients. Performs nearest-neighbour chroma
/// upsampling — its ±1 LSB worst-case is folded into
/// `CROSS_TOLERANCE`.
fn yuv_planar_to_rgb(frame: &VideoFrame, width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let ys = frame.planes[0].stride;
    let us = frame.planes[1].stride.max(1);
    let vs = frame.planes[2].stride.max(1);
    let cb_height = frame.planes[1].data.len() / us;
    let cr_height = frame.planes[2].data.len() / vs;
    let h_factor_u = (h + cb_height.max(1) - 1) / cb_height.max(1);
    let h_factor_v = (h + cr_height.max(1) - 1) / cr_height.max(1);
    let w_factor_u = (w + us - 1) / us;
    let w_factor_v = (w + vs - 1) / vs;

    let mut out = vec![0u8; w * h * 3];
    for j in 0..h {
        let cu_j = (j / h_factor_u.max(1)).min(cb_height.saturating_sub(1));
        let cv_j = (j / h_factor_v.max(1)).min(cr_height.saturating_sub(1));
        for i in 0..w {
            let cu_i = (i / w_factor_u.max(1)).min(us.saturating_sub(1));
            let cv_i = (i / w_factor_v.max(1)).min(vs.saturating_sub(1));
            let yy = frame.planes[0].data[j * ys + i] as f32;
            let cb = frame.planes[1].data[cu_j * us + cu_i] as f32 - 128.0;
            let cr = frame.planes[2].data[cv_j * vs + cv_i] as f32 - 128.0;
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
