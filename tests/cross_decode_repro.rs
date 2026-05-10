#![cfg(feature = "registry")]
//! Per-input cross-decode regression tests.
//!
//! Each test pins one libjpeg-encoded JPEG bitstream that previously
//! tripped a fuzz finding and asserts that:
//!
//!   1. `oxideav-mjpeg` decodes it without error;
//!   2. the produced YUV planes agree with libjpeg's
//!      `tjDecompressToYUVPlanes` output to within 2 LSB per
//!      sample.
//!
//! YUV-plane comparison is deliberate — see the
//! `libjpeg_encode_oxideav_decode` fuzz target docstring for why
//! comparing in RGB after a chroma upsample produces unbounded
//! drift on tiny synthetic images. The decoder is responsible for
//! IDCT + dequantisation only; chroma upsampling is a consumer
//! choice and lives outside the codec.
//!
//! Skips when libturbojpeg isn't installed.

#![allow(unsafe_code)]

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

/// Regression test for fuzz finding
/// `crash-3e708f209e638ec4086b4fd729469c3aefe0750b` (CI run
/// 25623290527, branch master, commit 416178a).
///
/// Original failure mode: the
/// `libjpeg_encode_oxideav_decode` harness was comparing in
/// *RGB* space after a nearest-neighbour chroma upsampler. On a
/// 5×1 image where each chroma sample (the image is 4:2:0, so
/// chroma is 3×1) covers a hard colour edge, libjpeg's "fancy"
/// upsampler interpolates across columns while the harness's
/// nearest-neighbour replicates — the resulting RGB drift was
/// 14–25 LSB on the affected B-channel pixels, well past the
/// ±5 LSB cross-tolerance budget.
///
/// Root cause: harness assertion lived in the wrong colour space.
/// The decoder itself was bit-correct: this test verifies the
/// per-plane match against libjpeg's own YUV decode.
#[test]
fn fuzz_repro_5x1_libjpeg_encode_yuv_match() {
    if !libjpeg::available() {
        eprintln!(
            "cross_decode_repro: libturbojpeg not found; skipping. \
             Install with `brew install jpeg-turbo` (macOS) or \
             `apt install libturbojpeg0-dev` (Debian/Ubuntu)."
        );
        return;
    }

    // The original fuzz input bytes (16 bytes). The harness shape
    // logic carves these into a 5×1 RGB image.
    let fuzz_data: &[u8] = &[
        121, 121, 121, 121, 121, 121, 121, 121, 121, 4, 7, 0, 0, 0, 10, 56,
    ];
    let (shape, rgb) = fuzz_data.split_first().unwrap();
    let pixel_count = (rgb.len() / 3).min(2048);
    let width = ((*shape as usize) % 64) + 1;
    let width = width.min(pixel_count);
    let height = pixel_count / width;
    assert_eq!((width, height), (5, 1), "fuzz shape carving regressed");
    let used_len = width * height * 3;
    let rgb = &rgb[..used_len];
    let (width_u32, height_u32) = (width as u32, height as u32);

    // Encode via libjpeg-turbo (4:2:0, q=75) — same path the fuzz
    // target takes.
    let bitstream =
        libjpeg::encode_rgb(rgb, width_u32, height_u32, 75).expect("libturbojpeg encode failed");

    // Reference: libjpeg-turbo decoded into 4:2:0 YUV planes,
    // trimmed to the visible region.
    let oracle = libjpeg::decode_to_yuv420(&bitstream, width_u32, height_u32)
        .expect("libturbojpeg YUV decode failed");

    // Subject under test.
    let mut params = CodecParameters::video(CodecId::new("mjpeg"));
    params.width = Some(width_u32);
    params.height = Some(height_u32);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&params)
        .expect("oxideav-mjpeg decoder construction failed");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), bitstream))
        .expect("oxideav-mjpeg send_packet failed");
    let Frame::Video(actual) = dec
        .receive_frame()
        .expect("oxideav-mjpeg receive_frame failed")
    else {
        panic!("expected video frame");
    };
    assert_eq!(actual.planes.len(), 3);

    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    assert_plane_within(&oracle.y, &actual.planes[0], width, height, 2, "Y");
    assert_plane_within(&oracle.cb, &actual.planes[1], cw, ch, 2, "Cb");
    assert_plane_within(&oracle.cr, &actual.planes[2], cw, ch, 2, "Cr");

    // Spot-check the specific pixel (byte 5) the original fuzz
    // failure flagged: pixel 1's chroma slot (chroma column 0)
    // must be at the JFIF grey baseline (128). If this regresses
    // the decoder really did break.
    assert_eq!(
        actual.planes[1].data[0], 128,
        "Cb[0] must be 128 (grey baseline)"
    );
    assert_eq!(
        actual.planes[2].data[0], 131,
        "Cr[0] (drift from grey is expected)"
    );
}

/// Compare the visible `width × height` region of `actual` (which
/// may carry MCU stride padding past the visible columns) against
/// the tightly packed `expected_tight` plane within `tolerance`
/// LSBs per pixel.
fn assert_plane_within(
    expected_tight: &[u8],
    actual: &oxideav_core::frame::VideoPlane,
    width: usize,
    height: usize,
    tolerance: i32,
    label: &str,
) {
    assert_eq!(
        expected_tight.len(),
        width * height,
        "{label}: oracle plane sized {} for {}x{} (need {})",
        expected_tight.len(),
        width,
        height,
        width * height
    );
    let stride = actual.stride;
    let need = if height == 0 {
        0
    } else {
        stride * (height - 1) + width
    };
    assert!(
        actual.data.len() >= need,
        "{label}: actual plane too small ({} < {} for stride {} {}x{})",
        actual.data.len(),
        need,
        stride,
        width,
        height
    );
    for j in 0..height {
        for i in 0..width {
            let e = expected_tight[j * width + i] as i32;
            let a = actual.data[j * stride + i] as i32;
            let d = (e - a).abs();
            assert!(
                d <= tolerance,
                "{label}: pixel ({i},{j}) differs by {d} > {tolerance} \
                 (expected {e}, actual {a})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// libturbojpeg shim — matches `fuzz/src/lib.rs` so this test does not depend
// on the (publish=false, separate-workspace) fuzz crate.
// ---------------------------------------------------------------------------

mod libjpeg {
    use libloading::{Library, Symbol};
    use std::os::raw::{c_int, c_uchar, c_ulong};
    use std::sync::OnceLock;

    const CANDIDATES: &[&str] = &[
        "libturbojpeg.dylib",
        "libturbojpeg.0.dylib",
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "turbojpeg.dll",
    ];

    const TJPF_RGB: c_int = 0;
    const TJSAMP_420: c_int = 2;
    const TJFLAG_ACCURATEDCT: c_int = 4096;

    type TjHandle = *mut std::ffi::c_void;

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: libturbojpeg is well-behaved at load time.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    pub fn available() -> bool {
        lib().is_some()
    }

    pub fn encode_rgb(rgb: &[u8], width: u32, height: u32, quality: u8) -> Option<Vec<u8>> {
        type InitFn = unsafe extern "C" fn() -> TjHandle;
        type DestroyFn = unsafe extern "C" fn(TjHandle) -> c_int;
        type FreeFn = unsafe extern "C" fn(*mut c_uchar) -> c_int;
        type CompressFn = unsafe extern "C" fn(
            handle: TjHandle,
            src_buf: *const c_uchar,
            width: c_int,
            pitch: c_int,
            height: c_int,
            pixel_format: c_int,
            jpeg_buf: *mut *mut c_uchar,
            jpeg_size: *mut c_ulong,
            jpeg_subsamp: c_int,
            jpeg_qual: c_int,
            flags: c_int,
        ) -> c_int;

        let l = lib()?;
        let stride = (width as usize).checked_mul(3)?;
        if rgb.len() < stride.checked_mul(height as usize)? {
            return None;
        }
        unsafe {
            let init: Symbol<InitFn> = l.get(b"tjInitCompress").ok()?;
            let destroy: Symbol<DestroyFn> = l.get(b"tjDestroy").ok()?;
            let free: Symbol<FreeFn> = l.get(b"tjFree").ok()?;
            let compress: Symbol<CompressFn> = l.get(b"tjCompress2").ok()?;
            let handle = init();
            if handle.is_null() {
                return None;
            }
            let mut jpeg_buf: *mut c_uchar = std::ptr::null_mut();
            let mut jpeg_size: c_ulong = 0;
            let rc = compress(
                handle,
                rgb.as_ptr(),
                width as c_int,
                stride as c_int,
                height as c_int,
                TJPF_RGB,
                &mut jpeg_buf,
                &mut jpeg_size,
                TJSAMP_420,
                quality as c_int,
                TJFLAG_ACCURATEDCT,
            );
            let result = if rc == 0 && !jpeg_buf.is_null() && jpeg_size > 0 {
                Some(std::slice::from_raw_parts(jpeg_buf, jpeg_size as usize).to_vec())
            } else {
                None
            };
            if !jpeg_buf.is_null() {
                let _ = free(jpeg_buf);
            }
            let _ = destroy(handle);
            result
        }
    }

    pub struct DecodedYuv420 {
        pub y: Vec<u8>,
        pub cb: Vec<u8>,
        pub cr: Vec<u8>,
    }

    /// Decode a JPEG byte string to 4:2:0 YUV planes via
    /// `tjDecompressToYUVPlanes`. Output planes are tight (no row
    /// padding) and trimmed to the visible region. libjpeg writes
    /// data per the MCU-aligned `tjPlaneWidth` × `tjPlaneHeight`
    /// dimensions; we allocate the full MCU-aligned plane for the
    /// FFI call to be sound, then copy the visible region into a
    /// tight buffer for the caller.
    pub fn decode_to_yuv420(data: &[u8], width: u32, height: u32) -> Option<DecodedYuv420> {
        type InitFn = unsafe extern "C" fn() -> TjHandle;
        type DestroyFn = unsafe extern "C" fn(TjHandle) -> c_int;
        type DecYuvPlanesFn = unsafe extern "C" fn(
            handle: TjHandle,
            jpeg_buf: *const c_uchar,
            jpeg_size: c_ulong,
            dst_planes: *mut *mut c_uchar,
            width: c_int,
            strides: *mut c_int,
            height: c_int,
            flags: c_int,
        ) -> c_int;
        type PlaneWidthFn = unsafe extern "C" fn(c_int, c_int, c_int) -> c_int;
        type PlaneHeightFn = unsafe extern "C" fn(c_int, c_int, c_int) -> c_int;

        let l = lib()?;
        unsafe {
            let init: Symbol<InitFn> = l.get(b"tjInitDecompress").ok()?;
            let destroy: Symbol<DestroyFn> = l.get(b"tjDestroy").ok()?;
            let dec: Symbol<DecYuvPlanesFn> = l.get(b"tjDecompressToYUVPlanes").ok()?;
            let plane_w: Symbol<PlaneWidthFn> = l.get(b"tjPlaneWidth").ok()?;
            let plane_h: Symbol<PlaneHeightFn> = l.get(b"tjPlaneHeight").ok()?;

            let handle = init();
            if handle.is_null() {
                return None;
            }
            let yw_full = plane_w(0, width as c_int, TJSAMP_420) as usize;
            let yh_full = plane_h(0, height as c_int, TJSAMP_420) as usize;
            let cw_full = plane_w(1, width as c_int, TJSAMP_420) as usize;
            let ch_full = plane_h(1, height as c_int, TJSAMP_420) as usize;
            let mut y_full = vec![0u8; yw_full * yh_full];
            let mut cb_full = vec![0u8; cw_full * ch_full];
            let mut cr_full = vec![0u8; cw_full * ch_full];
            let mut planes: [*mut c_uchar; 3] = [
                y_full.as_mut_ptr(),
                cb_full.as_mut_ptr(),
                cr_full.as_mut_ptr(),
            ];
            let mut strides: [c_int; 3] = [yw_full as c_int, cw_full as c_int, cw_full as c_int];
            let rc = dec(
                handle,
                data.as_ptr(),
                data.len() as c_ulong,
                planes.as_mut_ptr(),
                width as c_int,
                strides.as_mut_ptr(),
                height as c_int,
                TJFLAG_ACCURATEDCT,
            );
            let _ = destroy(handle);
            if rc != 0 {
                return None;
            }
            let yw = width as usize;
            let yh = height as usize;
            let cw = yw.div_ceil(2);
            let ch = yh.div_ceil(2);
            let mut y = vec![0u8; yw * yh];
            let mut cb = vec![0u8; cw * ch];
            let mut cr = vec![0u8; cw * ch];
            for j in 0..yh {
                y[j * yw..j * yw + yw].copy_from_slice(&y_full[j * yw_full..j * yw_full + yw]);
            }
            for j in 0..ch {
                cb[j * cw..j * cw + cw].copy_from_slice(&cb_full[j * cw_full..j * cw_full + cw]);
                cr[j * cw..j * cw + cw].copy_from_slice(&cr_full[j * cw_full..j * cw_full + cw]);
            }
            Some(DecodedYuv420 { y, cb, cr })
        }
    }
}
