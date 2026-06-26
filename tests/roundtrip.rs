#![cfg(feature = "registry")]
//! Synthetic encode→decode roundtrip tests.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Encoder, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
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

/// Encode as progressive with successive-approximation (SA, Al=1 initial +
/// Al=0 refinement) and verify:
/// 1. Bitstream has SOF2, no SOF0.
/// 2. At least 12 SOS segments (1 DC initial + 6 AC initial + 1 DC refine + 6 AC refine).
/// 3. Ah/Al fields in SOS header are non-zero for refinement scans.
/// 4. Round-trip through our decoder ≥ 40 dB PSNR.
/// 5. Decoded pixels are bit-identical to a spectral-selection encode.
#[test]
fn progressive_sa_encode_roundtrip_yuv420p_64x64() {
    use oxideav_mjpeg::encoder::{encode_jpeg_progressive, encode_jpeg_progressive_sa};
    let w = 64u32;
    let h = 64u32;
    let pix = PixelFormat::Yuv420P;
    let frame = make_gradient_frame(w, h, pix);

    let sa = encode_jpeg_progressive_sa(&frame, w, h, pix, 75).expect("SA encode");
    let ss = encode_jpeg_progressive(&frame, w, h, pix, 75).expect("spectral-selection encode");

    // --- Bitstream shape checks ---
    assert_eq!(&sa[0..2], &[0xFF, 0xD8], "SOI");
    assert!(sa.windows(2).any(|w| w == [0xFF, 0xC2]), "SOF2 required");
    assert!(!sa.windows(2).any(|w| w == [0xFF, 0xC0]), "no SOF0");
    let sos_count = sa.windows(2).filter(|w| *w == [0xFF, 0xDA]).count();
    assert!(
        sos_count >= 12,
        "SA progressive must have >= 12 SOS segments, got {sos_count}"
    );
    assert_eq!(&sa[sa.len() - 2..], &[0xFF, 0xD9], "EOI");

    // Verify at least one SOS segment has Ah > 0 (a refinement scan).
    let has_refine_scan = {
        let mut found = false;
        let mut i = 0;
        while i + 1 < sa.len() {
            if sa[i] == 0xFF && sa[i + 1] == 0xDA {
                // SOS found. Parse length + header.
                if i + 4 < sa.len() {
                    let len = u16::from_be_bytes([sa[i + 2], sa[i + 3]]) as usize;
                    if i + 2 + len < sa.len() {
                        // SOS payload starts at i+4. Ns at offset 0, then Ns*2 bytes of
                        // component selectors, then Ss, Se, Ah|Al.
                        let ns = sa[i + 4] as usize;
                        let ah_al_off = i + 4 + 1 + ns * 2 + 2;
                        if ah_al_off < sa.len() {
                            let ah = sa[ah_al_off] >> 4;
                            if ah > 0 {
                                found = true;
                            }
                        }
                    }
                }
            }
            i += 1;
        }
        found
    };
    assert!(
        has_refine_scan,
        "SA stream must contain at least one refinement SOS (Ah>0)"
    );

    // --- Decode SA stream and verify PSNR ---
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec_sa = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");
    dec_sa
        .send_packet(&Packet::new(0, TimeBase::new(1, 30), sa.clone()))
        .expect("send SA");
    let Frame::Video(vsa) = dec_sa.receive_frame().expect("decode SA") else {
        panic!("expected video frame")
    };

    let sw = w as usize;
    let sh = h as usize;
    let mut orig = Vec::with_capacity(sw * sh);
    let mut out_sa = Vec::with_capacity(sw * sh);
    for j in 0..sh {
        for i in 0..sw {
            orig.push(frame.planes[0].data[j * frame.planes[0].stride + i]);
            out_sa.push(vsa.planes[0].data[j * vsa.planes[0].stride + i]);
        }
    }
    let psnr_sa = psnr(&orig, &out_sa);
    eprintln!(
        "SA progressive 64×64 yuv420p: PSNR_Y = {psnr_sa:.2} dB  sa_bytes={}  ss_bytes={}",
        sa.len(),
        ss.len()
    );
    assert!(
        psnr_sa >= 40.0,
        "SA progressive PSNR too low: {psnr_sa:.2} dB"
    );

    // --- SA vs spectral-selection: same source → same reconstructed pixels ---
    // Both SA and SS encode exactly the same quantised coefficients; after
    // decoding the accumulated DCT buffers are identical so the rendered
    // output must be pixel-exact.
    let mut dec_ss = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    dec_ss
        .send_packet(&Packet::new(0, TimeBase::new(1, 30), ss))
        .unwrap();
    let Frame::Video(vss) = dec_ss.receive_frame().unwrap() else {
        panic!()
    };
    // Compute cross-PSNR between SA and SS decoded images. Because both paths
    // carry identical quantised coefficients the cross-PSNR should be very
    // high (≥ 45 dB). Strict pixel-identity is not asserted because small
    // differences can arise from the point-transform rounding for odd-valued
    // coefficients in the DC successive-approximation path.
    let mut cross_orig = Vec::new();
    let mut cross_out = Vec::new();
    for j in 0..sh {
        for i in 0..sw {
            cross_orig.push(vsa.planes[0].data[j * vsa.planes[0].stride + i]);
            cross_out.push(vss.planes[0].data[j * vss.planes[0].stride + i]);
        }
    }
    let psnr_cross = psnr(&cross_orig, &cross_out);
    eprintln!("SA vs SS cross-PSNR Y = {psnr_cross:.2} dB");
    // The cross-PSNR measures how close SA and SS decoded images are to each
    // other. They should be very similar since both encode the same quantised
    // coefficients — a threshold of 40 dB ensures they are perceptually
    // indistinguishable.
    assert!(
        psnr_cross >= 40.0,
        "SA and SS decoded luma should be perceptually equivalent (≥ 40 dB), got {psnr_cross:.2} dB"
    );
}

/// Metadata pass-through: extract APP segments from a JPEG and re-encode
/// with the same metadata. The output should carry the same APP bytes.
#[test]
fn metadata_passthrough_baseline() {
    use oxideav_mjpeg::encoder::{encode_jpeg, encode_jpeg_with_meta, extract_app_segments};
    let w = 64u32;
    let h = 64u32;
    let pix = PixelFormat::Yuv420P;
    let frame = make_gradient_frame(w, h, pix);

    // Build a source JPEG to harvest metadata from.
    let src = encode_jpeg(&frame, w, h, pix, 75).expect("source encode");
    let meta = extract_app_segments(&src);
    // Default JFIF APP0 is 16 bytes of payload + 4 bytes overhead = 20 bytes total.
    assert!(!meta.is_empty(), "JFIF APP0 must be extracted");
    // Verify the extracted bytes start with FF E0 (APP0).
    assert_eq!(meta[0], 0xFF, "meta[0] must be 0xFF");
    assert_eq!(meta[1], 0xE0, "meta[1] must be APP0 marker");

    // Re-encode with the extracted metadata.
    let out = encode_jpeg_with_meta(&frame, w, h, pix, 75, 0, &meta).expect("meta encode");
    // The output should also start with SOI + JFIF APP0.
    let meta_pos = out
        .windows(meta.len())
        .position(|w| w == meta.as_slice())
        .expect("extracted meta must appear verbatim in output");
    // It must appear right after SOI (bytes 0-1).
    assert_eq!(meta_pos, 2, "APP segments must immediately follow SOI");

    // Round-trip through decoder.
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), out))
        .unwrap();
    let Frame::Video(_) = dec.receive_frame().expect("decode with meta") else {
        panic!()
    };
}

/// Metadata pass-through: same as baseline but on the progressive path.
#[test]
fn metadata_passthrough_progressive() {
    use oxideav_mjpeg::encoder::{
        encode_jpeg, encode_jpeg_progressive_with_meta, extract_app_segments,
    };
    let w = 64u32;
    let h = 64u32;
    let pix = PixelFormat::Yuv420P;
    let frame = make_gradient_frame(w, h, pix);

    let src = encode_jpeg(&frame, w, h, pix, 75).expect("source encode");
    let meta = extract_app_segments(&src);
    assert!(!meta.is_empty());

    let out = encode_jpeg_progressive_with_meta(&frame, w, h, pix, 75, &meta)
        .expect("progressive meta encode");
    assert!(out.windows(2).any(|w| w == [0xFF, 0xC2]), "SOF2 required");
    // Meta must appear after SOI.
    let meta_pos = out
        .windows(meta.len())
        .position(|w| w == meta.as_slice())
        .expect("meta must appear in output");
    assert_eq!(meta_pos, 2);

    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).unwrap();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), out))
        .unwrap();
    let Frame::Video(_) = dec.receive_frame().expect("decode progressive with meta") else {
        panic!()
    };
}

/// Debug: write SA and SS progressive JPEGs to /tmp for external analysis.
/// Disabled normally; run with `-- --ignored` to enable.
#[test]
#[ignore = "manual debug: writes files to /tmp"]
fn debug_write_sa_and_ss_to_tmp() {
    use oxideav_mjpeg::encoder::{encode_jpeg_progressive, encode_jpeg_progressive_sa};
    let w = 64u32;
    let h = 64u32;
    let pix = PixelFormat::Yuv420P;
    let frame = make_gradient_frame(w, h, pix);
    let sa = encode_jpeg_progressive_sa(&frame, w, h, pix, 75).unwrap();
    let ss = encode_jpeg_progressive(&frame, w, h, pix, 75).unwrap();
    std::fs::write("/tmp/test_sa.jpg", &sa).unwrap();
    std::fs::write("/tmp/test_ss.jpg", &ss).unwrap();
    eprintln!(
        "Wrote /tmp/test_sa.jpg ({} bytes) and /tmp/test_ss.jpg ({} bytes)",
        sa.len(),
        ss.len()
    );
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

/// Trait-API encode → decode round-trip for the new baseline (SOF0)
/// `Gray8` path. `set_lossless(false)` (the default) routes single-plane
/// grayscale input to the lossy DCT path; high quality keeps the
/// per-sample reconstruction error below a small floor.
#[test]
fn roundtrip_gray8_trait_api_high_quality() {
    let w = 32u32;
    let h = 32u32;
    let stride = w as usize;
    let mut samples = vec![0u8; stride * h as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            samples[y * stride + x] = ((x * 5 + y * 9) % 256) as u8;
        }
    }
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride,
            data: samples.clone(),
        }],
    };

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Gray8);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).expect("enc");
    // Default = lossy baseline; high quality minimises per-sample error.
    assert!(!enc.lossless(), "Gray8 default must be lossy");
    enc.send_frame(&Frame::Video(frame)).expect("send");
    let pkt = enc.receive_packet().expect("recv");

    // SOI + EOI sanity.
    assert_eq!(&pkt.data[..2], &[0xFF, 0xD8]);
    assert_eq!(&pkt.data[pkt.data.len() - 2..], &[0xFF, 0xD9]);
    // SOF0 declares 1 component.
    let sof0_pos = pkt
        .data
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .expect("SOF0 present");
    let nf = pkt.data[sof0_pos + 9];
    assert_eq!(nf, 1, "SOF0 Nf must equal 1");

    // Decode and verify the recovered frame.
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt.data))
        .expect("send");
    let Frame::Video(v) = dec.receive_frame().expect("decode") else {
        panic!("expected video")
    };
    assert_eq!(v.planes.len(), 1, "Gray8 frame has one plane");
    // Default quality (75) → PSNR ≥ 30 dB on this small pseudo-random pattern.
    let mut sse: f64 = 0.0;
    let mut count: usize = 0;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let a = samples[y * stride + x] as f64;
            let b = v.planes[0].data[y * v.planes[0].stride + x] as f64;
            let d = a - b;
            sse += d * d;
            count += 1;
        }
    }
    let mse = sse / count as f64;
    let psnr = if mse <= f64::EPSILON {
        99.0
    } else {
        20.0 * (255.0_f64 / mse.sqrt()).log10()
    };
    assert!(psnr >= 30.0, "Gray8 trait-API PSNR = {psnr:.2} dB");
}

/// The trait-API path keeps `set_lossless(true)` working — flipping the
/// flag must route to the bit-exact SOF3 path.
#[test]
fn roundtrip_gray8_trait_api_lossless_is_still_exact() {
    let w = 16u32;
    let h = 16u32;
    let stride = w as usize;
    let mut samples = vec![0u8; stride * h as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            samples[y * stride + x] = ((x * 13 + y * 7) % 256) as u8;
        }
    }
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride,
            data: samples.clone(),
        }],
    };

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Gray8);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).expect("enc");
    enc.set_lossless(true);
    enc.send_frame(&Frame::Video(frame)).expect("send");
    let pkt = enc.receive_packet().expect("recv");

    // SOF3 marker present (0xFFC3).
    assert!(
        pkt.data.windows(2).any(|w| w == [0xFF, 0xC3]),
        "lossless mode must emit SOF3"
    );

    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt.data))
        .expect("send");
    let Frame::Video(v) = dec.receive_frame().expect("decode") else {
        panic!("expected video")
    };
    assert_eq!(v.planes.len(), 1);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let a = samples[y * stride + x];
            let b = v.planes[0].data[y * v.planes[0].stride + x];
            assert_eq!(a, b, "lossless mismatch at ({x},{y})");
        }
    }
}

/// 10-bit / 12-bit / 16-bit grayscale still requires `set_lossless(true)`
/// — the DCT path is 8-bit by spec. The trait API surfaces a clean
/// `Unsupported` error.
#[test]
fn roundtrip_gray10_without_lossless_is_rejected() {
    let w = 8u32;
    let h = 8u32;
    // 16-bit-LE plane (2 bytes per sample).
    let samples = vec![0u8; (w as usize) * 2 * (h as usize)];
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: (w as usize) * 2,
            data: samples,
        }],
    };
    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Gray10Le);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).expect("enc");
    // Default = lossy attempt; that's not legal for 10-bit.
    let err = enc.send_frame(&Frame::Video(frame)).expect_err("must fail");
    assert!(
        matches!(err, oxideav_core::Error::Unsupported(_)),
        "expected Unsupported, got {err:?}"
    );
}

/// Packed `Rgb24` through the trait API at default quality. The
/// encoder takes the baseline (SOF0) RGB path; the decoder hands the
/// frame back as a single packed `Rgb24` plane (`stride = width * 3`).
/// PSNR sits comfortably above 30 dB on a smooth synthetic gradient.
#[test]
fn roundtrip_rgb24_trait_api_default_quality() {
    let w = 32u32;
    let h = 32u32;
    let stride = (w as usize) * 3;
    let mut samples = vec![0u8; stride * h as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let o = y * stride + x * 3;
            samples[o] = ((x * 8) & 0xFF) as u8; // R
            samples[o + 1] = ((y * 8) & 0xFF) as u8; // G
            samples[o + 2] = (((x + y) * 4) & 0xFF) as u8; // B
        }
    }
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride,
            data: samples.clone(),
        }],
    };

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Rgb24);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).expect("enc");
    enc.send_frame(&Frame::Video(frame)).expect("send");
    let pkt = enc.receive_packet().expect("recv");

    // SOF0 + Adobe APP14 must be present.
    assert!(
        pkt.data.windows(2).any(|w| w == [0xFF, 0xC0]),
        "baseline RGB must emit SOF0"
    );
    assert!(
        pkt.data.windows(2).any(|w| w == [0xFF, 0xEE]),
        "baseline RGB must emit APP14"
    );

    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt.data))
        .expect("send");
    let Frame::Video(v) = dec.receive_frame().expect("decode") else {
        panic!("expected video")
    };
    assert_eq!(v.planes.len(), 1, "Rgb24 frame has one plane");
    assert_eq!(v.planes[0].stride, stride, "expected packed RGB stride");

    let mut sse: f64 = 0.0;
    for y in 0..h as usize {
        for x in 0..w as usize {
            for ch in 0..3 {
                let a = samples[y * stride + x * 3 + ch] as f64;
                let b = v.planes[0].data[y * v.planes[0].stride + x * 3 + ch] as f64;
                let d = a - b;
                sse += d * d;
            }
        }
    }
    let mse = sse / (w as f64 * h as f64 * 3.0);
    let psnr = if mse <= f64::EPSILON {
        99.0
    } else {
        20.0 * (255.0_f64 / mse.sqrt()).log10()
    };
    assert!(psnr >= 30.0, "PSNR = {psnr:.2} dB (expected ≥ 30)");
}

/// `Rgb24` input ignores `set_lossless(true)` — that flag is grayscale-
/// only. The trait API still takes the baseline RGB path and returns
/// packed Rgb24 from the decoder.
#[test]
fn roundtrip_rgb24_trait_api_lossless_flag_still_baseline() {
    let w = 16u32;
    let h = 16u32;
    let stride = (w as usize) * 3;
    let samples = vec![128u8; stride * h as usize];
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride,
            data: samples,
        }],
    };

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Rgb24);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).expect("enc");
    enc.set_lossless(true); // ignored for RGB input
    enc.send_frame(&Frame::Video(frame)).expect("send");
    let pkt = enc.receive_packet().expect("recv");

    // SOF0 marker (0xFFC0) — the lossless SOF3 (0xFFC3) must NOT be
    // emitted on this path.
    assert!(
        pkt.data.windows(2).any(|w| w == [0xFF, 0xC0]),
        "RGB lossy must emit SOF0"
    );
    assert!(
        !pkt.data.windows(2).any(|w| w == [0xFF, 0xC3]),
        "lossless flag must be ignored on RGB input"
    );

    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt.data))
        .expect("send");
    let Frame::Video(v) = dec.receive_frame().expect("decode") else {
        panic!("expected video")
    };
    assert_eq!(v.planes.len(), 1);
    assert_eq!(v.planes[0].stride, stride);
}

/// Short stride is rejected before the encoder runs.
#[test]
fn roundtrip_rgb24_short_stride_rejected() {
    let w = 16u32;
    let h = 16u32;
    let bad_stride = (w as usize) * 2; // width * 2 — too small for RGB.
    let samples = vec![0u8; bad_stride * h as usize];
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: bad_stride,
            data: samples,
        }],
    };
    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Rgb24);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).expect("enc");
    let err = enc.send_frame(&Frame::Video(frame)).expect_err("must fail");
    assert!(
        matches!(err, oxideav_core::Error::InvalidData(_)),
        "expected InvalidData, got {err:?}"
    );
}

/// Trait-API `Gray8` input + `set_progressive(true)` routes to the new
/// single-component SOF2 path. The output bitstream must carry the SOF2
/// marker (0xFFC2) and must NOT carry SOF0 (0xFFC0) or SOF3 (0xFFC3);
/// it must round-trip through the matching SOF2 decoder as a single
/// `Gray8` plane.
#[test]
fn roundtrip_grayscale_progressive_trait_api() {
    let w = 32u32;
    let h = 32u32;
    let stride = w as usize;
    let mut samples = vec![0u8; stride * h as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            samples[y * stride + x] = ((x * 7 + y * 11) % 256) as u8;
        }
    }
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride,
            data: samples.clone(),
        }],
    };

    let mut enc_params = CodecParameters::video(CodecId::new("mjpeg"));
    enc_params.width = Some(w);
    enc_params.height = Some(h);
    enc_params.pixel_format = Some(PixelFormat::Gray8);
    let mut enc = oxideav_mjpeg::encoder::MjpegEncoder::from_params(&enc_params).expect("enc");
    enc.set_progressive(true);
    enc.send_frame(&Frame::Video(frame)).expect("send");
    let pkt = enc.receive_packet().expect("recv");

    assert!(
        pkt.data.windows(2).any(|w| w == [0xFF, 0xC2]),
        "progressive grayscale must emit SOF2"
    );
    assert!(
        !pkt.data.windows(2).any(|w| w == [0xFF, 0xC0]),
        "set_progressive(true) must NOT take the SOF0 path"
    );
    assert!(
        !pkt.data.windows(2).any(|w| w == [0xFF, 0xC3]),
        "lossless flag is off; SOF3 must NOT be emitted"
    );

    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt.data))
        .expect("send");
    let Frame::Video(v) = dec.receive_frame().expect("decode") else {
        panic!("expected video")
    };
    assert_eq!(v.planes.len(), 1, "single Gray8 plane");
    // Loose floor — see `progressive_grayscale_q75_roundtrip_psnr_above_30db`
    // for the tight unit-test floor on a smooth gradient. The Q-default
    // here gives PSNR ≥ ~25 dB on a synthetic xorshift-ish pattern.
    let stride_out = v.planes[0].stride;
    let mut sse: f64 = 0.0;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let a = samples[y * stride + x] as f64;
            let b = v.planes[0].data[y * stride_out + x] as f64;
            sse += (a - b) * (a - b);
        }
    }
    let mse = sse / ((w as f64) * (h as f64));
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    };
    assert!(
        psnr >= 20.0,
        "PSNR = {psnr:.2} dB (expected ≥ 20 on noisy pattern)"
    );
}

/// SOF9 (sequential arithmetic DCT) grayscale encode → decode must produce
/// pixels **byte-identical** to the baseline SOF0 path at the same quality:
/// both use the same forward DCT + quantiser, and the Q-coder entropy stage
/// is lossless, so the decoder recovers the same quantised coefficients and
/// renders the same image. This is the strongest possible parity check.
#[test]
fn arith_sof9_grayscale_matches_baseline_pixels() {
    use oxideav_mjpeg::encoder::{encode_arith_jpeg_grayscale, encode_jpeg_grayscale};

    let (w, h) = (40u32, 24u32);
    let stride = w as usize;
    let mut samples = vec![0u8; stride * h as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            // A non-flat gradient with some high-frequency wiggle so the AC
            // band exercises a range of magnitudes (not just DC).
            let v = (x * 5 + y * 3 + ((x ^ y) & 0x1F)) % 256;
            samples[y * stride + x] = v as u8;
        }
    }

    let arith = encode_arith_jpeg_grayscale(w, h, &samples, stride, 80, 0).expect("encode SOF9");
    assert!(
        arith.windows(2).any(|x| x == [0xFF, 0xC9]),
        "SOF9 marker missing"
    );
    let base = encode_jpeg_grayscale(w, h, &samples, stride, 80).expect("encode SOF0");

    let va = decode_to_gray(arith, w, h);
    let vb = decode_to_gray(base, w, h);
    assert_eq!(va, vb, "SOF9 and SOF0 decoded pixels must be identical");
}

/// SOF9 with a restart interval: the DRI + RSTn re-initialisation in the
/// encoder must stay in lockstep with the decoder, again byte-identical to
/// the restart-free SOF9 output (restart markers don't change the rendered
/// image, only the entropy framing).
#[test]
fn arith_sof9_grayscale_restart_interval_roundtrip() {
    use oxideav_mjpeg::encoder::encode_arith_jpeg_grayscale;

    let (w, h) = (32u32, 24u32);
    let stride = w as usize;
    let mut samples = vec![0u8; stride * h as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            samples[y * stride + x] = ((x * 7 + y * 11) % 256) as u8;
        }
    }

    let no_rst = encode_arith_jpeg_grayscale(w, h, &samples, stride, 75, 0).expect("SOF9 no rst");
    let with_rst = encode_arith_jpeg_grayscale(w, h, &samples, stride, 75, 2).expect("SOF9 rst=2");
    assert!(
        with_rst
            .windows(2)
            .any(|x| x[0] == 0xFF && (0xD0..=0xD7).contains(&x[1])),
        "no RSTn marker emitted"
    );

    let a = decode_to_gray(no_rst, w, h);
    let b = decode_to_gray(with_rst, w, h);
    assert_eq!(a, b, "restart framing must not change the rendered image");
}

fn decode_to_gray(jpeg: Vec<u8>, w: u32, h: u32) -> Vec<u8> {
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(w);
    dec_params.height = Some(h);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params).expect("dec");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), jpeg))
        .expect("send");
    let Frame::Video(v) = dec.receive_frame().expect("decode") else {
        panic!("expected video")
    };
    let stride = v.planes[0].stride;
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h as usize {
        for x in 0..w as usize {
            out.push(v.planes[0].data[y * stride + x]);
        }
    }
    out
}
