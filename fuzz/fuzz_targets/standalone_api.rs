//! Robustness target for the still-image contract surface: arbitrary
//! bytes into `probe` / `info` / `decode_with` / `decode_rgba8`, and —
//! when the bytes do decode — back out through `encode` and
//! `encode_rgb8` so the writer sees fuzz-shaped images too. The bar is
//! "no panic"; every function must return `Err` on hostile input.
//!
//! Attack surfaces this covers beyond the `decode` target (which goes
//! through the framework `Decoder`):
//!
//! * the header scan (`scan_header`): APP0 / APP1 / APP2 / APP14
//!   identifier checks, ICC chunk reassembly with inconsistent `total`
//!   / `seq_no`, DHP-before-SOF, truncated segment lengths, `Y = 0`
//!   DNL resolution;
//! * the shape inference (`infer_shape`) against every SOF variant and
//!   component count, and its agreement with the decoder (a decode
//!   that succeeds must carry the inferred layout);
//! * the limit checks in `decode_with` (small `max_pixels` on the odd
//!   iterations) and strict mode on the even ones;
//! * `to_rgba8` over every native layout the decoder can shape, with
//!   the planes exactly as the decoder produced them;
//! * `encode` of a decoded image (layout → writer mapping) and
//!   `encode_rgb8` of its RGB conversion.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_mjpeg::{DecodeOptions, EncodeOptions};

/// Discard inputs above this size; the surfaces above are reached by
/// small inputs and large ones only slow the fuzzer down.
const MAX_INPUT_LEN: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_LEN {
        return;
    }
    let probed = oxideav_mjpeg::probe(data);
    let info = oxideav_mjpeg::info(data);

    // Every decode in this target is capped at 1 Mpixel so a fuzz-shaped
    // 65535 × 65535 frame header costs a `LimitExceeded`, not seconds of
    // allocation; the cap itself is one of the surfaces under test.
    let cap = DecodeOptions::new().with_max_pixels(1 << 20);
    let opts = if data.len() % 2 == 1 {
        cap.clone().with_max_pixels(4096)
    } else {
        cap.clone().with_strict(true)
    };
    if oxideav_mjpeg::decode_with(data, &opts).is_ok() && opts.strict {
        // A stream that passes strict mode starts `SOI, marker` — the
        // signature sniff must agree (the lenient walker tolerates
        // stray bytes after SOI, strict mode and `probe` do not).
        assert!(probed, "strict decode succeeded on a stream probe rejected");
    }

    let Ok(img) = oxideav_mjpeg::decode_with(data, &cap) else {
        // The one-call path with default limits must fail the same way
        // for anything the header scan rejects; a frame that only broke
        // the 1 Mpixel cap is skipped (too slow to decode here).
        if !matches!(info, Ok(ref i) if u64::from(i.width) * u64::from(i.height) > 1 << 20) {
            assert!(oxideav_mjpeg::decode_rgba8(data).is_err());
        }
        return;
    };
    let info = info.expect("decode succeeded but info failed");
    assert_eq!(info.format, img.format, "info / decode layout disagree");
    assert_eq!((info.width, info.height), (img.width, img.height));
    assert_eq!(info.precision, img.precision);
    assert_eq!(info.color, img.color);

    let rgba = oxideav_mjpeg::decode_rgba8(data).expect("decode ok, decode_rgba8 failed");
    assert_eq!(
        rgba.data.len(),
        img.width as usize * img.height as usize * 4
    );
    let rgb = img.to_rgb8();
    assert_eq!(rgb.len(), img.width as usize * img.height as usize * 3);

    // Re-encode what we decoded: every decodable layout is either
    // writable or refused with `Unsupported`, never a panic. Keep the
    // pixel budget small so the DCT / statistics passes stay cheap.
    if img.width as usize * img.height as usize <= 4096 {
        let lossless = EncodeOptions::new().with_lossless(1);
        match oxideav_mjpeg::encode(&img, &lossless) {
            Ok(bytes) => {
                let back = oxideav_mjpeg::decode(&bytes).expect("own lossless stream rejected");
                assert_eq!(
                    back.planes, img.planes,
                    "lossless round trip changed planes"
                );
            }
            Err(oxideav_mjpeg::Error::Unsupported(_))
            | Err(oxideav_mjpeg::Error::InvalidData(_)) => {}
            Err(e) => panic!("unexpected encode error: {e}"),
        }
        let _ = oxideav_mjpeg::encode(&img, &EncodeOptions::new().with_quality(50));
        let _ = oxideav_mjpeg::encode_rgb8(img.width, img.height, &rgb, &EncodeOptions::new());
    }
});
