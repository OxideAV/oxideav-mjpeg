#![no_main]

//! libjpeg-turbo encode → oxideav-mjpeg decode cross-validation.
//!
//! Encodes the synthesised RGB image with `tjCompress2` (RGB in,
//! 4:2:0 chroma, quality 75), then decodes the resulting JPEG with
//! both libjpeg-turbo (oracle, via `tjDecompressToYUVPlanes`) and
//! oxideav-mjpeg (system under test). Both decoders return YUV
//! planes; we compare them directly with a `YUV_TOLERANCE`-LSB
//! per-pixel budget.
//!
//! Comparing in YUV space — not in RGB after a chroma upsample —
//! is deliberate. The decoder is responsible for the IDCT and
//! dequantisation; once that lands in YUV planes a conformant
//! decoder must agree with libjpeg byte-for-byte (within ≤1 LSB
//! IDCT slack). Choice of chroma upsampling method ("fancy"
//! interpolation vs nearest-neighbour) is a *consumer* concern,
//! not a decoder concern, and applying it on the harness side
//! introduces unbounded drift on tiny images with abrupt chroma
//! transitions (e.g. a 5×1 image where each chroma sample covers
//! a different colour). See `YUV_TOLERANCE` docstring + the
//! `crash-3e708f209e638ec4086b4fd729469c3aefe0750b` regression
//! test in `tests/cross_decode_repro.rs` for the worked example
//! that motivated the YUV-space comparison.
//!
//! Skips silently when libturbojpeg isn't installed on the host.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_mjpeg_fuzz::{assert_plane_within, libjpeg, rgb_image_from_fuzz_input, YUV_TOLERANCE};

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

    // Reference: libjpeg-turbo decoded back to 4:2:0 YUV planes
    // (no chroma upsampling, no YUV→RGB matrix). This is the
    // tightest possible per-plane oracle.
    let oracle = match libjpeg::decode_to_yuv420(&bitstream) {
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
    assert_eq!(
        v.planes.len(),
        3,
        "oxideav-mjpeg returned {} planes for a 4:2:0 JPEG",
        v.planes.len()
    );

    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    assert_plane_within(
        &oracle.y,
        &v.planes[0].data,
        v.planes[0].stride,
        w,
        h,
        YUV_TOLERANCE,
        "libjpeg vs oxideav decode (Y plane)",
    );
    assert_plane_within(
        &oracle.cb,
        &v.planes[1].data,
        v.planes[1].stride,
        cw,
        ch,
        YUV_TOLERANCE,
        "libjpeg vs oxideav decode (Cb plane)",
    );
    assert_plane_within(
        &oracle.cr,
        &v.planes[2].data,
        v.planes[2].stride,
        cw,
        ch,
        YUV_TOLERANCE,
        "libjpeg vs oxideav decode (Cr plane)",
    );
});
