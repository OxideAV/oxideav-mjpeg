//! `DecodeOptions::deblock`: opt-in block-edge smoothing.

use oxideav_mjpeg::{DecodeOptions, EncodeOptions, decode_rgb8, decode_rgb8_with, encode_rgb8};

/// A smooth two-dimensional ramp with a soft disc — the content where
/// coarse quantisation shows block edges.
fn smooth(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let r = ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt();
            let disc = (1.0 - (r * 4.0).min(1.0)) * 60.0;
            v.push((40.0 + 150.0 * fx + disc) as u8);
            v.push((60.0 + 120.0 * fy) as u8);
            v.push((200.0 - 100.0 * fx * fy + disc * 0.5) as u8);
        }
    }
    v
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let se: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum();
    10.0 * (255.0f64.powi(2) / (se / a.len() as f64)).log10()
}

#[test]
fn deblocking_is_off_by_default_and_helps_at_low_quality() {
    assert!(!DecodeOptions::new().deblock);
    let (w, h) = (96u32, 80u32);
    let src = smooth(w, h);
    for (q, progressive) in [(15u8, false), (30, false), (30, true)] {
        let jpeg = encode_rgb8(
            w,
            h,
            &src,
            &EncodeOptions::new()
                .with_quality(q)
                .with_progressive(progressive),
        )
        .unwrap();
        let plain = decode_rgb8(&jpeg).unwrap();
        let smoothed = decode_rgb8_with(&jpeg, &DecodeOptions::new().with_deblock(true)).unwrap();
        assert_ne!(plain.as_bytes(), smoothed.as_bytes(), "q{q}");
        let (a, b) = (
            psnr(plain.as_bytes(), &src),
            psnr(smoothed.as_bytes(), &src),
        );
        assert!(
            b > a,
            "q{q} progressive {progressive}: {b:.2} dB vs {a:.2} dB"
        );
    }
}
