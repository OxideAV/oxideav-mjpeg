//! Runtime libjpeg-turbo interop for the cross-decode fuzz harnesses.
//!
//! libjpeg-turbo (`libturbojpeg`) is loaded via `dlopen` at first call —
//! there is no `turbojpeg-sys`-style build-script dep that would pull
//! libjpeg / libjpeg-turbo source into the workspace's cargo dep tree.
//! Each harness checks [`libjpeg::available`] up front and `return`s
//! early when the shared library isn't installed, so fuzz binaries
//! built on a host without libturbojpeg simply do nothing instead of
//! panicking.
//!
//! Install libturbojpeg with `brew install jpeg-turbo` (macOS) or
//! `apt install libturbojpeg0-dev` (Debian/Ubuntu). The loader probes
//! the conventional shared-object names for both platforms.
//!
//! The interop uses TurboJPEG's `tjCompress2` / `tjDecompress2` byte
//! buffer entry points operating on packed RGB (TJPF_RGB). RGB rather
//! than RGBA because JPEG itself has no alpha channel — alpha would be
//! ignored on encode and synthesised opaque on decode, so feeding RGBA
//! through this wrapper would only invite confusion.

#![allow(unsafe_code)]

pub mod libjpeg {
    use libloading::{Library, Symbol};
    use std::os::raw::{c_int, c_uchar, c_ulong};
    use std::sync::OnceLock;

    /// Conventional libturbojpeg shared-object names the loader will
    /// try in order. Covers macOS (`.dylib`), Linux (versioned + plain
    /// `.so`), and Windows (`.dll`).
    const CANDIDATES: &[&str] = &[
        "libturbojpeg.dylib",
        "libturbojpeg.0.dylib",
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "turbojpeg.dll",
    ];

    /// TJPF_RGB — packed 24-bit R,G,B, no padding. Matches our 3-byte
    /// stride assumptions in the harness.
    const TJPF_RGB: c_int = 0;

    /// TJSAMP_420 — 4:2:0 chroma subsampling. Matches what
    /// `oxideav-mjpeg`'s baseline encoder emits when given a
    /// `PixelFormat::Yuv420P` frame.
    const TJSAMP_420: c_int = 2;

    /// TJFLAG_ACCURATEDCT — request the integer DCT path. Improves
    /// pixel agreement against our reference IDCT, which uses an exact
    /// integer transform.
    const TJFLAG_ACCURATEDCT: c_int = 4096;

    /// Opaque TurboJPEG instance handle.
    type TjHandle = *mut std::ffi::c_void;

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: `Library::new` is documented as unsafe
                // because the loaded library may run code at load
                // time. We accept that risk for fuzz tooling —
                // libturbojpeg is a well-behaved shared library.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    /// True iff a libturbojpeg shared library was successfully loaded.
    /// Cross-decode fuzz harnesses early-return when this is false so
    /// the binary still runs without an oracle (the assertions just
    /// don't fire).
    pub fn available() -> bool {
        lib().is_some()
    }

    /// Encode a packed RGB image as a JPEG via `tjCompress2`. `quality`
    /// is libjpeg's 1..=100 scale. Returns `None` if libturbojpeg
    /// isn't available, the encoder couldn't be initialised, or the
    /// encode call returned a non-zero status.
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

    /// A JPEG image as decoded by libturbojpeg, normalised to packed
    /// RGB.
    pub struct DecodedRgb {
        pub width: u32,
        pub height: u32,
        /// Tightly packed R,G,B bytes, length `width * height * 3`.
        pub rgb: Vec<u8>,
    }

    /// Decode a JPEG byte string to packed RGB via `tjDecompressHeader2`
    /// + `tjDecompress2`. Returns `None` on libturbojpeg unavailable,
    /// header parse failure, allocation overflow, or decode failure.
    pub fn decode_to_rgb(data: &[u8]) -> Option<DecodedRgb> {
        type InitFn = unsafe extern "C" fn() -> TjHandle;
        type DestroyFn = unsafe extern "C" fn(TjHandle) -> c_int;
        type HeaderFn = unsafe extern "C" fn(
            handle: TjHandle,
            jpeg_buf: *const c_uchar,
            jpeg_size: c_ulong,
            width: *mut c_int,
            height: *mut c_int,
            jpeg_subsamp: *mut c_int,
        ) -> c_int;
        type DecompressFn = unsafe extern "C" fn(
            handle: TjHandle,
            jpeg_buf: *const c_uchar,
            jpeg_size: c_ulong,
            dst_buf: *mut c_uchar,
            width: c_int,
            pitch: c_int,
            height: c_int,
            pixel_format: c_int,
            flags: c_int,
        ) -> c_int;

        let l = lib()?;
        unsafe {
            let init: Symbol<InitFn> = l.get(b"tjInitDecompress").ok()?;
            let destroy: Symbol<DestroyFn> = l.get(b"tjDestroy").ok()?;
            let header: Symbol<HeaderFn> = l.get(b"tjDecompressHeader2").ok()?;
            let decompress: Symbol<DecompressFn> = l.get(b"tjDecompress2").ok()?;

            let handle = init();
            if handle.is_null() {
                return None;
            }
            let mut w: c_int = 0;
            let mut h: c_int = 0;
            let mut subsamp: c_int = 0;
            let header_rc = header(
                handle,
                data.as_ptr(),
                data.len() as c_ulong,
                &mut w,
                &mut h,
                &mut subsamp,
            );
            if header_rc != 0 || w <= 0 || h <= 0 {
                let _ = destroy(handle);
                return None;
            }
            let stride = (w as usize).checked_mul(3)?;
            let size = stride.checked_mul(h as usize)?;
            let mut buf = vec![0u8; size];
            let dec_rc = decompress(
                handle,
                data.as_ptr(),
                data.len() as c_ulong,
                buf.as_mut_ptr(),
                w,
                stride as c_int,
                h,
                TJPF_RGB,
                TJFLAG_ACCURATEDCT,
            );
            let _ = destroy(handle);
            if dec_rc != 0 {
                return None;
            }
            Some(DecodedRgb {
                width: w as u32,
                height: h as u32,
                rgb: buf,
            })
        }
    }
}

// ---------------------------------------------------------------------
// Shared helpers used by every fuzz target.
// ---------------------------------------------------------------------

use oxideav_core::frame::VideoPlane;
use oxideav_core::{PixelFormat, VideoFrame};

/// Maximum width allowed for synthesized fuzz inputs. Bounds the per-row
/// MCU count so encoder/decoder allocations stay small in CI.
pub const MAX_WIDTH: usize = 64;
/// Maximum total pixel count allowed for synthesized fuzz inputs. Caps
/// total memory regardless of the chosen aspect ratio.
pub const MAX_PIXELS: usize = 2048;

/// Carve a synthetic packed-RGB image out of a fuzz input. Returns
/// `(width, height, &rgb_bytes)` when the input has at least one shape
/// byte plus 3 bytes per pixel; otherwise `None`.
///
/// Layout matches the webp fuzz harnesses: the first byte selects width
/// (mod `MAX_WIDTH`) and the rest is treated as packed RGB samples.
pub fn rgb_image_from_fuzz_input(data: &[u8]) -> Option<(u32, u32, &[u8])> {
    let (&shape, rgb) = data.split_first()?;

    let pixel_count = (rgb.len() / 3).min(MAX_PIXELS);
    if pixel_count == 0 {
        return None;
    }

    let width = ((shape as usize) % MAX_WIDTH) + 1;
    let width = width.min(pixel_count);
    let height = pixel_count / width;
    let used_len = width * height * 3;
    let rgb = &rgb[..used_len];

    Some((width as u32, height as u32, rgb))
}

/// Convert a packed RGB buffer to a planar `Yuv444P` `VideoFrame` using
/// BT.601 full-range coefficients, matching what `oxideav-mjpeg`'s
/// encoder is calibrated against.
///
/// 4:4:4 (no chroma subsampling) keeps the colour-conversion error
/// bounded so the cross-decode tolerance stays tight; if the harness
/// used 4:2:0 the chroma resampling alone could blow past the ±2 LSB
/// tolerance even on bit-perfect codecs.
pub fn rgb_to_yuv444p_frame(rgb: &[u8], width: u32, height: u32) -> VideoFrame {
    let stride = width as usize;
    let n = stride * height as usize;
    let mut y = vec![0u8; n];
    let mut u = vec![0u8; n];
    let mut v = vec![0u8; n];
    for j in 0..height as usize {
        for i in 0..width as usize {
            let p = (j * stride + i) * 3;
            let r = rgb[p] as f32;
            let g = rgb[p + 1] as f32;
            let b = rgb[p + 2] as f32;
            // BT.601 full-range (JFIF) — matches libjpeg / libjpeg-turbo
            // and our own internal RGB→YCbCr conversion.
            let yy = 0.299 * r + 0.587 * g + 0.114 * b;
            let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
            let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
            y[j * stride + i] = yy.round().clamp(0.0, 255.0) as u8;
            u[j * stride + i] = cb.round().clamp(0.0, 255.0) as u8;
            v[j * stride + i] = cr.round().clamp(0.0, 255.0) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride, data: y },
            VideoPlane { stride, data: u },
            VideoPlane { stride, data: v },
        ],
    }
}

/// Convert a `Yuv444P` `VideoFrame` back to packed RGB using the
/// inverse BT.601 full-range coefficients, so a `rgb -> yuv -> rgb`
/// round-trip through the harness alone is bit-near-identity.
pub fn yuv444p_frame_to_rgb(frame: &VideoFrame, width: u32, height: u32) -> Vec<u8> {
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

/// Tolerance window for *self*-roundtrip assertions (oxideav encoder
/// then decoder, both in YUV space). JPEG is lossy by definition even
/// on a self-roundtrip — quant tables ≥ 1 floor the coefficients
/// before the inverse DCT — so byte-equality cannot hold. ±2 LSB on
/// YUV is what the user-spec asks for and what the encoder-decoder
/// pair achieves at quality 100 on smooth content; high-frequency
/// fuzz inputs at quality 100 sit comfortably inside this bound.
pub const SELF_TOLERANCE: i32 = 2;

/// Tolerance window for *cross*-decode assertions (libjpeg encode →
/// oxideav decode, or oxideav encode → libjpeg decode). RGB-space
/// comparison adds three sources of slack vs the self-roundtrip:
///
///  1. IDCT precision differences between the two implementations
///     (≤ 1 LSB on YUV per JPEG conformance).
///  2. Chroma upsampling method differences ("fancy" smoothing vs
///     nearest-neighbour) — ≤ ~2 LSB on chroma rows in 4:2:0 mode.
///  3. YUV→RGB rounding done by libjpeg vs by our harness — ≤ ~1
///     LSB per RGB component.
///
/// ±5 LSB covers the worst-case sum without admitting real decoder
/// bugs (which typically manifest as tens-of-LSB excursions or DC
/// shifts across whole MCUs).
pub const CROSS_TOLERANCE: i32 = 5;

/// Assert two equally-sized byte buffers agree to within `tolerance`
/// LSBs at every position. Panics with the failing index, expected,
/// and actual bytes on the first mismatch.
pub fn assert_within(expected: &[u8], actual: &[u8], tolerance: i32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch: expected={} actual={}",
        expected.len(),
        actual.len()
    );
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let d = (*e as i32 - *a as i32).abs();
        assert!(
            d <= tolerance,
            "{context}: byte {i} differs by {d} > {tolerance} (expected {e}, actual {a})"
        );
    }
}
