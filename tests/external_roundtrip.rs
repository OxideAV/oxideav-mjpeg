//! End-to-end cross-codec roundtrip test for `oxideav-mjpeg`.
//!
//! The pipeline is the round-trip pattern from the task tracker:
//!
//!   1. oxideav-mjpeg encode  (Yuv444P, q=75)        → JPEG bytes
//!   2. libturbojpeg decode   (RGB)                  → RGB pixels
//!   3. libturbojpeg encode   (q=75, 4:2:0)          → JPEG bytes
//!   4. oxideav-mjpeg decode  (Yuv420P planes)       → final RGB
//!
//! The final RGB image is compared against the synthesised input
//! with a per-component LSB tolerance — JPEG is lossy and we have
//! traversed the codec twice (so chroma subsampling is applied
//! once on the libturbojpeg encode, plus two integer DCT passes,
//! plus two YUV→RGB conversions). The starting tolerance mirrors
//! `CROSS_TOLERANCE` from the fuzz crate (±5 LSB) but allows for
//! the doubled lossy path: it widens to ±20 LSB by default and
//! the assertion reports peak drift on failure so CI can tighten
//! it once measured.
//!
//! # libturbojpeg
//!
//! The libturbojpeg shim (`tjCompress2` / `tjDecompress2`) is
//! `dlopen`'d at runtime — there is no `turbojpeg-sys`-style build
//! dependency, no libjpeg / libjpeg-turbo source pulled into our
//! dep tree (workspace policy bars external library source). When
//! the shared library isn't installed on the host the test prints
//! a one-line skip notice and exits without failing, so this file
//! is safe to ship as part of the standard `cargo test` matrix.
//!
//! Install the shared library with `brew install jpeg-turbo` on
//! macOS or `apt install libturbojpeg0-dev` on Debian/Ubuntu. The
//! loader probes the conventional `.dylib` / `.so` / `.dll` names
//! for all three platforms.

#![allow(unsafe_code)]

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase, VideoFrame};
use oxideav_mjpeg::encoder::encode_jpeg_with_opts;

/// Test image dimensions. 640×480 is the constraint from the task
/// brief — it is large enough to exercise multiple MCU rows in
/// 4:2:0 (each MCU is 16×16 pixels at this subsampling) without
/// blowing past the per-test memory budget.
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

/// libjpeg-style quality factor used on both encodes (oxideav and
/// libturbojpeg).  q=75 is the default that both encoders are
/// calibrated for — it is also what the fuzz harness uses, so the
/// observed drift here is directly comparable to fuzz reports.
const QUALITY: u8 = 75;

/// Per-component RGB tolerance after the full four-step pipeline.
///
/// The fuzz crate documents ±5 LSB for a *single* cross-decode
/// (one IDCT precision delta + one chroma upsample method delta +
/// one YUV→RGB rounding pass).  This test runs through that path
/// twice with an extra 4:2:0 chroma downsample on the libturbojpeg
/// encode — the worst-case sum is ~4× higher.  Start at ±20 LSB
/// and let CI confirm the actual drift; if measured peak is much
/// smaller this can be tightened.  If the first run reports a
/// larger drift, the assertion message includes the peak so the
/// constant can be adjusted in a follow-up.
const TOLERANCE: i32 = 20;

#[test]
fn external_roundtrip_640x480() {
    if !libjpeg::available() {
        eprintln!(
            "external_roundtrip: libturbojpeg not found; skipping. \
             Install with `brew install jpeg-turbo` (macOS) or \
             `apt install libturbojpeg0-dev` (Debian/Ubuntu)."
        );
        return;
    }

    // ---- Step 0: synthesise the input image ------------------------------
    //
    // A pure-noise input would scatter energy across every DCT
    // basis function, blowing the lossy comparison budget on the
    // first pass alone.  Use a smooth gradient with sparse
    // mid-frequency features (overlapping circles) instead — the
    // gradient gives chroma a defined slow-varying signal that
    // 4:2:0 subsampling can faithfully represent, and the circles
    // exercise non-trivial AC coefficients without dominating the
    // spectrum.  Both layers are deterministic so the test result
    // is reproducible run-to-run.
    let rgb_in = synth_smooth_rgb(WIDTH, HEIGHT);

    // ---- Step 1: oxideav encode (Yuv444P) ---------------------------------
    //
    // 4:4:4 keeps the colour-conversion error bounded on the
    // *first* hop so the budget is spent on the unavoidable
    // 4:2:0 downsample inside libturbojpeg's encode (step 3), not
    // on ours.
    let yuv = rgb_to_yuv444p_frame(&rgb_in, WIDTH, HEIGHT);
    let jpeg_a = encode_jpeg_with_opts(&yuv, WIDTH, HEIGHT, PixelFormat::Yuv444P, QUALITY, 0)
        .expect("oxideav encode_jpeg_with_opts failed");
    assert!(jpeg_a.len() >= 4, "oxideav-encoded JPEG is too short");
    assert_eq!(&jpeg_a[..2], &[0xFF, 0xD8], "missing SOI on oxideav encode");
    assert_eq!(
        &jpeg_a[jpeg_a.len() - 2..],
        &[0xFF, 0xD9],
        "missing EOI on oxideav encode"
    );

    // ---- Step 2: libturbojpeg decode -------------------------------------
    //
    // Use libturbojpeg as the cross-decode oracle on the bytes the
    // oxideav encoder produced.  This is also exercised by the
    // `oxideav_encode_libjpeg_decode` fuzz target — the assertion
    // here is just that the pipeline rolls forward (no panic /
    // size mismatch).
    let mid = libjpeg::decode_to_rgb(&jpeg_a)
        .expect("libturbojpeg failed to decode oxideav-produced JPEG");
    assert_eq!(mid.width, WIDTH, "libturbojpeg returned wrong width");
    assert_eq!(mid.height, HEIGHT, "libturbojpeg returned wrong height");
    assert_eq!(
        mid.rgb.len(),
        (WIDTH * HEIGHT * 3) as usize,
        "libturbojpeg returned wrong-sized RGB buffer"
    );

    // ---- Step 3: libturbojpeg encode (4:2:0, q=75) -----------------------
    //
    // Re-encode through libturbojpeg with the conventional 4:2:0
    // chroma layout most decoders see in the wild.  The shim is
    // hard-wired to TJSAMP_420 + TJFLAG_ACCURATEDCT to match the
    // oracle path used in the fuzz harness.
    let jpeg_b =
        libjpeg::encode_rgb(&mid.rgb, WIDTH, HEIGHT, QUALITY).expect("libturbojpeg encode failed");
    assert!(jpeg_b.len() >= 4, "libturbojpeg-produced JPEG is too short");

    // ---- Step 4: oxideav decode ------------------------------------------
    //
    // Decode the libturbojpeg-encoded bytes through `oxideav-mjpeg`.
    // libturbojpeg emits a baseline (SOF0) 4:2:0 JPEG, which our
    // decoder lowers to `Yuv420P` planes.
    let mut dec_params = CodecParameters::video(CodecId::new("mjpeg"));
    dec_params.width = Some(WIDTH);
    dec_params.height = Some(HEIGHT);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&dec_params)
        .expect("oxideav-mjpeg decoder construction failed");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), jpeg_b))
        .expect("oxideav-mjpeg send_packet failed");
    let Frame::Video(out) = dec
        .receive_frame()
        .expect("oxideav-mjpeg receive_frame failed")
    else {
        panic!("oxideav decoder returned a non-video Frame");
    };

    // Convert the decoded YUV420P planes back to packed RGB
    // (nearest-neighbour chroma upsample, matching the fuzz
    // harness's `yuv_planar_to_rgb`).
    let rgb_out = yuv_planar_to_rgb(&out, WIDTH, HEIGHT);

    // ---- Final assertion: per-component RGB tolerance --------------------
    //
    // Walk both buffers in parallel and report the peak drift on
    // failure so the tolerance can be tightened (or, if it is
    // genuinely too tight, widened) in a follow-up.
    assert_eq!(rgb_in.len(), rgb_out.len(), "output RGB has wrong size");
    let mut peak: i32 = 0;
    let mut peak_idx = 0usize;
    for (i, (e, a)) in rgb_in.iter().zip(rgb_out.iter()).enumerate() {
        let d = (*e as i32 - *a as i32).abs();
        if d > peak {
            peak = d;
            peak_idx = i;
        }
    }
    eprintln!(
        "external_roundtrip: peak RGB drift = {peak} LSB at byte {peak_idx} (tolerance {TOLERANCE})"
    );
    assert!(
        peak <= TOLERANCE,
        "external roundtrip drift {peak} > tolerance {TOLERANCE} (at byte {peak_idx})"
    );
}

// ---------------------------------------------------------------------------
// Synthetic image generator
// ---------------------------------------------------------------------------

/// Build a deterministic mid-frequency RGB test image. Combines:
///
///   * a smooth diagonal gradient (slow varying — friendly to JPEG
///     low-frequency basis functions, cheap for chroma to encode);
///   * a few overlapping soft circles with different colours
///     (sparse mid-frequency features that exercise the AC bands
///     without saturating them).
///
/// Pure random noise is deliberately avoided — JPEG smears it
/// across every DCT coefficient and the comparison budget would
/// be spent before the second pass even starts. This pattern
/// produces a roundtrip drift that sits comfortably in the LSBs,
/// matching real-world photographic content.
fn synth_smooth_rgb(width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 3];

    // Five soft circles. (cx, cy, radius, (r, g, b)).
    #[allow(clippy::type_complexity)]
    let circles: &[(f32, f32, f32, (u8, u8, u8))] = &[
        (160.0, 120.0, 80.0, (220, 60, 60)),
        (480.0, 120.0, 90.0, (60, 220, 90)),
        (320.0, 240.0, 110.0, (60, 90, 220)),
        (160.0, 360.0, 75.0, (220, 200, 60)),
        (480.0, 360.0, 95.0, (180, 60, 220)),
    ];

    for j in 0..h {
        for i in 0..w {
            let fx = i as f32;
            let fy = j as f32;

            // Diagonal gradient in HSL-ish space — three offset
            // ramps so each channel has a different phase. Keeps
            // the colour cube exercised without ever clipping.
            let g_r = (fx + fy) * (255.0 / (width as f32 + height as f32));
            let g_g = ((width as f32 - fx) + fy) * (255.0 / (width as f32 + height as f32));
            let g_b = (fx + (height as f32 - fy)) * (255.0 / (width as f32 + height as f32));

            let mut r = g_r;
            let mut g = g_g;
            let mut b = g_b;

            // Composite the circles with a smooth radial falloff
            // (1 - (d/R)^2), clamped to 0 outside the circle.
            for &(cx, cy, radius, (cr, cg, cb)) in circles {
                let dx = fx - cx;
                let dy = fy - cy;
                let d2 = dx * dx + dy * dy;
                let r2 = radius * radius;
                if d2 < r2 {
                    let weight = 1.0 - (d2 / r2);
                    // weighted average — preserves the gradient
                    // outside the circle and blends smoothly into
                    // the circle colour at the centre.
                    r = r * (1.0 - weight) + cr as f32 * weight;
                    g = g * (1.0 - weight) + cg as f32 * weight;
                    b = b * (1.0 - weight) + cb as f32 * weight;
                }
            }

            let p = (j * w + i) * 3;
            out[p] = r.round().clamp(0.0, 255.0) as u8;
            out[p + 1] = g.round().clamp(0.0, 255.0) as u8;
            out[p + 2] = b.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Colour-space helpers (mirror fuzz/src/lib.rs — kept inline so the test
// file does not dev-depend on the fuzz crate, per task brief).
// ---------------------------------------------------------------------------

/// Convert a packed RGB buffer to a planar `Yuv444P` `VideoFrame`
/// using BT.601 full-range coefficients (matches JFIF, libjpeg,
/// and `oxideav-mjpeg`'s internal conversion).
fn rgb_to_yuv444p_frame(rgb: &[u8], width: u32, height: u32) -> VideoFrame {
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

/// Convert any-subsampling YUV planes to packed RGB with BT.601
/// full-range coefficients. Performs nearest-neighbour chroma
/// upsampling — same fold as the libjpeg-encode-oxideav-decode
/// fuzz target.
fn yuv_planar_to_rgb(frame: &VideoFrame, width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let ys = frame.planes[0].stride;
    let us = frame.planes[1].stride.max(1);
    let vs = frame.planes[2].stride.max(1);
    let cb_height = frame.planes[1].data.len() / us;
    let cr_height = frame.planes[2].data.len() / vs;
    let h_factor_u = h.div_ceil(cb_height.max(1));
    let h_factor_v = h.div_ceil(cr_height.max(1));
    let w_factor_u = w.div_ceil(us);
    let w_factor_v = w.div_ceil(vs);

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

// ---------------------------------------------------------------------------
// libturbojpeg shim — copied inline from `fuzz/src/lib.rs` so this test
// file does not depend on the fuzz crate (which is `publish = false` and
// not part of the workspace member set seen by `cargo test`).
// ---------------------------------------------------------------------------

mod libjpeg {
    use libloading::{Library, Symbol};
    use std::os::raw::{c_int, c_uchar, c_ulong};
    use std::sync::OnceLock;

    /// Conventional libturbojpeg shared-object names the loader will
    /// try in order. Covers macOS, Linux (versioned + plain `.so`),
    /// and Windows.
    const CANDIDATES: &[&str] = &[
        "libturbojpeg.dylib",
        "libturbojpeg.0.dylib",
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "turbojpeg.dll",
    ];

    /// TJPF_RGB — packed 24-bit R,G,B, no padding.
    const TJPF_RGB: c_int = 0;
    /// TJSAMP_420 — 4:2:0 chroma subsampling on encode.
    const TJSAMP_420: c_int = 2;
    /// TJFLAG_ACCURATEDCT — request the integer DCT path on
    /// encode/decode for tighter pixel agreement against the
    /// reference IDCT.
    const TJFLAG_ACCURATEDCT: c_int = 4096;

    type TjHandle = *mut std::ffi::c_void;

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: `Library::new` is documented as unsafe
                // because the loaded library may run code at load
                // time. Acceptable for test tooling.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    /// True iff a libturbojpeg shared library was successfully loaded.
    pub fn available() -> bool {
        lib().is_some()
    }

    /// Encode a packed RGB image as a JPEG via `tjCompress2`. Returns
    /// `None` on libturbojpeg unavailable, init failure, or encode
    /// failure.
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

    /// A JPEG image as decoded by libturbojpeg, normalised to packed RGB.
    pub struct DecodedRgb {
        pub width: u32,
        pub height: u32,
        /// Tightly packed R,G,B bytes, length `width * height * 3`.
        pub rgb: Vec<u8>,
    }

    /// Decode a JPEG byte string to packed RGB via `tjDecompressHeader2`
    /// and `tjDecompress2`. Returns `None` on libturbojpeg unavailable,
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
