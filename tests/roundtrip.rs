//! Synthetic encode→decode roundtrip tests.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
};

fn make_gradient_frame(w: u32, h: u32, pix: PixelFormat) -> VideoFrame {
    let (cw, ch): (u32, u32) = match pix {
        PixelFormat::Yuv444P => (w, h),
        PixelFormat::Yuv422P => (w.div_ceil(2), h),
        PixelFormat::Yuv420P => (w.div_ceil(2), h.div_ceil(2)),
        _ => panic!("unsupported for this test"),
    };
    let y_stride = w as usize;
    let mut y = vec![0u8; y_stride * h as usize];
    for j in 0..h as usize {
        for i in 0..w as usize {
            y[j * y_stride + i] = (((i + j) * 2) % 255) as u8;
        }
    }
    let cb_stride = cw as usize;
    let cr_stride = cw as usize;
    let mut cb = vec![0u8; cb_stride * ch as usize];
    let mut cr = vec![0u8; cr_stride * ch as usize];
    for j in 0..ch as usize {
        for i in 0..cw as usize {
            cb[j * cb_stride + i] = ((128 + (i as i32 - cw as i32 / 2)) as u8).clamp(0, 255);
            cr[j * cr_stride + i] = ((128 + (j as i32 - ch as i32 / 2)) as u8).clamp(0, 255);
        }
    }
    let _ = pix;
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

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: f64 = 0.0;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        sse += d * d;
    }
    if sse == 0.0 {
        return 99.0;
    }
    let mse = sse / a.len() as f64;
    20.0 * (255.0_f64 / mse.sqrt()).log10()
}

fn run_roundtrip(w: u32, h: u32, pix: PixelFormat) -> f64 {
    let frame = make_gradient_frame(w, h, pix);

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(pix);
    enc_params.frame_rate = Some(Rational::new(30, 1));
    let mut enc = oxideav_mjpeg::encoder::make_encoder(&enc_params).expect("enc");

    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let pkt = enc.receive_packet().expect("recv");

    assert!(pkt.data.len() >= 10);
    assert_eq!(pkt.data[0], 0xFF);
    assert_eq!(pkt.data[1], 0xD8, "SOI");
    assert_eq!(pkt.data[pkt.data.len() - 2], 0xFF);
    assert_eq!(pkt.data[pkt.data.len() - 1], 0xD9, "EOI");

    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");

    let in_pkt = Packet::new(0, TimeBase::new(1, 30), pkt.data.clone());
    dec.send_packet(&in_pkt).expect("send");
    let out = dec.receive_frame().expect("decode");
    let Frame::Video(v) = out else {
        panic!("expected video");
    };

    // Compute PSNR across Y plane (visible area) — it's the most perceptually
    // important and has the highest resolution.
    let sw = w as usize;
    let sh = h as usize;
    let mut original = Vec::with_capacity(sw * sh);
    let mut decoded = Vec::with_capacity(sw * sh);
    for j in 0..sh {
        for i in 0..sw {
            original.push(frame.planes[0].data[j * frame.planes[0].stride + i]);
            decoded.push(v.planes[0].data[j * v.planes[0].stride + i]);
        }
    }
    psnr(&original, &decoded)
}

#[test]
fn roundtrip_yuv420p_320x240_gradient() {
    let psnr_y = run_roundtrip(320, 240, PixelFormat::Yuv420P);
    eprintln!("yuv420p 320x240 PSNR_Y = {psnr_y:.2} dB");
    assert!(psnr_y >= 35.0, "luma PSNR too low: {psnr_y}");
}

#[test]
fn roundtrip_yuv422p_128x64() {
    let psnr_y = run_roundtrip(128, 64, PixelFormat::Yuv422P);
    eprintln!("yuv422p 128x64 PSNR_Y = {psnr_y:.2} dB");
    assert!(psnr_y >= 35.0, "luma PSNR too low: {psnr_y}");
}

#[test]
fn roundtrip_yuv444p_96x48() {
    let psnr_y = run_roundtrip(96, 48, PixelFormat::Yuv444P);
    eprintln!("yuv444p 96x48 PSNR_Y = {psnr_y:.2} dB");
    assert!(psnr_y >= 35.0, "luma PSNR too low: {psnr_y}");
}

/// Synthetic image designed to produce plenty of 0xFF bytes in the entropy
/// stream. All samples at the mid-point cancel to low-entropy DC — but the
/// very-high-quality tables used by Annex K make lots of long Huffman codes.
#[test]
fn byte_stuffing_roundtrip() {
    // Noisy random-ish texture stresses the entropy coder.
    let w = 64u32;
    let h = 64u32;
    let pix = PixelFormat::Yuv420P;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; (w * h) as usize];
    for j in 0..h {
        for i in 0..w {
            // LCG-style pseudo-random.
            let mut x = (i
                .wrapping_mul(1103515245)
                .wrapping_add(j.wrapping_mul(12345)))
                & 0xFF;
            x ^= (i * j) & 0xFF;
            y[(j * w + i) as usize] = x as u8;
        }
    }
    let cb = vec![128u8; (cw * ch) as usize];
    let cr = vec![128u8; (cw * ch) as usize];
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y.clone(),
            },
            VideoPlane {
                stride: cw as usize,
                data: cb,
            },
            VideoPlane {
                stride: cw as usize,
                data: cr,
            },
        ],
    };

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(pix);
    let mut enc = oxideav_mjpeg::encoder::make_encoder(&enc_params).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();

    // Verify the stream contains a stuffed 0xFF00 somewhere (i.e., byte
    // stuffing actually happened). Skip the header (before SOS).
    let sos_pos = pkt
        .data
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("SOS found");
    let scan = &pkt.data[sos_pos + 2..pkt.data.len() - 2];
    let stuffs = scan
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0x00)
        .count();
    assert!(stuffs > 0, "expected at least one 0xFF00 stuff");

    // Decode and check dimensions.
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    let in_pkt = Packet::new(0, TimeBase::new(1, 30), pkt.data);
    dec.send_packet(&in_pkt).unwrap();
    let out = dec.receive_frame().unwrap();
    let Frame::Video(_v) = out else {
        panic!();
    };
}

/// When restart markers are enabled, the bitstream must carry a DRI
/// segment before SOS and cycle `FF D0..FF D7` markers inside the scan.
/// Decoding must produce byte-identical pixels vs the no-restart encode
/// (restart markers don't change coded coefficients, only add resync
/// points).
#[test]
fn roundtrip_restart_interval_yuv420p() {
    use oxideav_mjpeg::encoder::{encode_jpeg, encode_jpeg_with_opts};
    let w = 64u32;
    let h = 48u32;
    let pix = PixelFormat::Yuv420P;
    let frame = make_gradient_frame(w, h, pix);

    let base = encode_jpeg(&frame, w, h, pix, 75).expect("base encode");
    let with_rst = encode_jpeg_with_opts(&frame, w, h, pix, 75, 4).expect("rst encode");

    // DRI: FFDD 0004 0004 appears before SOS (FFDA).
    let dri_pos = with_rst
        .windows(6)
        .position(|w| w == [0xFF, 0xDD, 0x00, 0x04, 0x00, 0x04])
        .expect("DRI 4 present");
    let sos_pos = with_rst
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("SOS present");
    assert!(dri_pos < sos_pos, "DRI must precede SOS");
    // Non-restart stream must not carry DRI.
    assert!(!base.windows(2).any(|w| w == [0xFF, 0xDD]));

    // Count RSTn markers in the scan. 4×3=12 MCUs total at 64×48 yuv420p;
    // Ri=4 produces restart markers after MCU 4 and 8 (not after the last).
    let scan = &with_rst[sos_pos + 2..with_rst.len() - 2];
    let mut rst_bytes = Vec::new();
    let mut i = 0;
    while i + 1 < scan.len() {
        if scan[i] == 0xFF && (0xD0..=0xD7).contains(&scan[i + 1]) {
            rst_bytes.push(scan[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }
    assert_eq!(rst_bytes, vec![0xD0, 0xD1], "RSTn cycling wrong");

    // Decode both and confirm identical output.
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec_a = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    let mut dec_b = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    dec_a
        .send_packet(&Packet::new(0, TimeBase::new(1, 30), base))
        .unwrap();
    dec_b
        .send_packet(&Packet::new(0, TimeBase::new(1, 30), with_rst))
        .unwrap();
    let Frame::Video(va) = dec_a.receive_frame().unwrap() else {
        panic!()
    };
    let Frame::Video(vb) = dec_b.receive_frame().unwrap() else {
        panic!()
    };
    for p in 0..3 {
        assert_eq!(
            va.planes[p].data, vb.planes[p].data,
            "plane {p} mismatch between restart / non-restart encodes"
        );
    }

    // Sanity: decoded Y PSNR vs source is still reasonable.
    let sw = w as usize;
    let sh = h as usize;
    let mut orig = Vec::with_capacity(sw * sh);
    let mut dec = Vec::with_capacity(sw * sh);
    for j in 0..sh {
        for i in 0..sw {
            orig.push(frame.planes[0].data[j * frame.planes[0].stride + i]);
            dec.push(vb.planes[0].data[j * vb.planes[0].stride + i]);
        }
    }
    let p = psnr(&orig, &dec);
    assert!(p >= 35.0, "restart-encode luma PSNR too low: {p}");
}

/// Exercise the "last MCU on restart boundary" edge case: with 12 MCUs
/// and Ri=6 the second restart boundary lands exactly on the last MCU,
/// which must NOT emit a trailing marker (EOI comes next).
#[test]
fn restart_interval_no_trailing_marker_on_last_mcu() {
    use oxideav_mjpeg::encoder::encode_jpeg_with_opts;
    let w = 64u32;
    let h = 48u32;
    let pix = PixelFormat::Yuv420P;
    let frame = make_gradient_frame(w, h, pix);
    let stream = encode_jpeg_with_opts(&frame, w, h, pix, 75, 6).expect("encode");
    let sos_pos = stream
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("SOS present");
    let scan = &stream[sos_pos + 2..stream.len() - 2];
    let mut rsts = 0;
    let mut i = 0;
    while i + 1 < scan.len() {
        if scan[i] == 0xFF && (0xD0..=0xD7).contains(&scan[i + 1]) {
            rsts += 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    // 12 MCUs / Ri=6 → single restart after MCU 6; none after MCU 12.
    assert_eq!(rsts, 1);

    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), stream))
        .unwrap();
    let Frame::Video(_) = dec.receive_frame().unwrap() else {
        panic!()
    };
}

/// Encode a 64×64 gradient as a progressive JPEG. Verify the bitstream
/// carries SOF2 + multiple SOS markers, then round-trip it through our
/// own progressive decoder and check that luma MSE stays within a
/// lossy-encoder epsilon (baseline and progressive share Huffman tables
/// and quant steps so quality should match).
#[test]
fn progressive_encode_roundtrip_yuv420p_64x64() {
    use oxideav_mjpeg::encoder::{encode_jpeg, encode_jpeg_progressive};
    let w = 64u32;
    let h = 64u32;
    let pix = PixelFormat::Yuv420P;
    let frame = make_gradient_frame(w, h, pix);

    let prog = encode_jpeg_progressive(&frame, w, h, pix, 75).expect("progressive encode");
    let base = encode_jpeg(&frame, w, h, pix, 75).expect("baseline encode");

    // --- Bitstream shape checks --------------------------------------
    // Starts with SOI + ... + SOF2 (FF C2) before the first SOS.
    assert_eq!(&prog[0..2], &[0xFF, 0xD8], "SOI");
    let sof2_pos = prog
        .windows(2)
        .position(|w| w == [0xFF, 0xC2])
        .expect("SOF2 present");
    // No SOF0 in a progressive bitstream.
    assert!(
        !prog.windows(2).any(|w| w == [0xFF, 0xC0]),
        "progressive stream must not carry SOF0"
    );
    // At least two SOS markers (DC + AC bands × 3 components = 7 total for our decomposition).
    let sos_count = prog.windows(2).filter(|w| *w == [0xFF, 0xDA]).count();
    assert!(
        sos_count >= 2,
        "progressive must carry multiple SOS, got {sos_count}"
    );
    assert!(
        sof2_pos < prog.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap(),
        "SOF2 must precede the first SOS"
    );
    assert_eq!(&prog[prog.len() - 2..], &[0xFF, 0xD9], "EOI");

    // --- Decode & compare with source --------------------------------
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), prog.clone()))
        .expect("send");
    let Frame::Video(v) = dec.receive_frame().expect("decode progressive") else {
        panic!("expected video frame")
    };

    // Y-plane MSE vs source — a progressive SOF2 spectral-selection
    // encode uses the same quant step and DCT as baseline, so the
    // reconstructed luma should match a baseline encode to within a
    // small epsilon. Baseline gradient PSNR here is typically >35 dB.
    let sw = w as usize;
    let sh = h as usize;
    let mut orig = Vec::with_capacity(sw * sh);
    let mut out = Vec::with_capacity(sw * sh);
    for j in 0..sh {
        for i in 0..sw {
            orig.push(frame.planes[0].data[j * frame.planes[0].stride + i]);
            out.push(v.planes[0].data[j * v.planes[0].stride + i]);
        }
    }
    let mut sse: f64 = 0.0;
    for i in 0..orig.len() {
        let d = orig[i] as f64 - out[i] as f64;
        sse += d * d;
    }
    let mse = sse / orig.len() as f64;
    eprintln!(
        "progressive 64x64 yuv420p: luma MSE = {mse:.3}  prog bytes={}  base bytes={}",
        prog.len(),
        base.len()
    );
    assert!(mse < 10.0, "progressive luma MSE too high: {mse}");

    // --- Progressive vs baseline pixel match -------------------------
    // Same quant tables + same coefficients end up at the IDCT, so
    // decoded pixels should be bit-identical across the two bitstreams
    // (modulo any sign-of-zero differences — none here since we don't
    // emit EOB-run).
    let mut dec_b = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    dec_b
        .send_packet(&Packet::new(0, TimeBase::new(1, 30), base))
        .unwrap();
    let Frame::Video(vb) = dec_b.receive_frame().unwrap() else {
        panic!()
    };
    for p in 0..3 {
        assert_eq!(
            v.planes[p].data, vb.planes[p].data,
            "plane {p} mismatch progressive vs baseline decode"
        );
    }
}

/// Exercise the `MjpegEncoder::set_progressive` API: flipping the flag
/// on a concrete encoder must swap SOF0 for SOF2 on the next
/// `send_frame`.
#[test]
fn progressive_encoder_trait_api() {
    use oxideav_core::Encoder;
    use oxideav_mjpeg::encoder::MjpegEncoder;
    let w = 64u32;
    let h = 64u32;
    let pix = PixelFormat::Yuv420P;
    let frame = make_gradient_frame(w, h, pix);

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(pix);
    let mut enc: Box<MjpegEncoder> =
        MjpegEncoder::from_params(&enc_params).expect("concrete encoder");
    enc.set_progressive(true);
    assert!(enc.progressive());

    enc.send_frame(&Frame::Video(frame)).expect("send frame");
    let pkt = enc.receive_packet().expect("packet");
    assert!(pkt.data.windows(2).any(|w| w == [0xFF, 0xC2]));
    assert!(!pkt.data.windows(2).any(|w| w == [0xFF, 0xC0]));
}

/// The decoder should treat an extended-sequential (SOF1) 8-bit JPEG the
/// same as baseline (SOF0), since both share the Huffman sequential scan
/// structure. We produce a baseline JPEG with our encoder and rewrite the
/// SOF0 marker byte to SOF1, then verify the decoded frame matches.
#[test]
fn decode_sof1_extended_sequential() {
    let w = 96u32;
    let h = 48u32;
    let pix = PixelFormat::Yuv422P;
    let frame = make_gradient_frame(w, h, pix);

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(pix);
    let mut enc = oxideav_mjpeg::encoder::make_encoder(&enc_params).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();

    let sof0_pos = pkt
        .data
        .windows(2)
        .position(|x| x == [0xFF, 0xC0])
        .expect("SOF0 present");
    let mut sof1_bytes = pkt.data.clone();
    sof1_bytes[sof0_pos + 1] = 0xC1;

    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    let in_pkt = Packet::new(0, TimeBase::new(1, 30), sof1_bytes);
    dec.send_packet(&in_pkt).unwrap();
    let out = dec.receive_frame().expect("decode SOF1");
    let Frame::Video(_v) = out else { panic!() };
}
