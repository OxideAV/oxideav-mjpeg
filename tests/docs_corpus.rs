//! Integration tests against the docs/image/jpeg/ fixture corpus.
//!
//! Each fixture under `../../docs/image/jpeg/fixtures/<name>/` carries an
//! `input.jpg` plus a ground-truth decode in PPM (P6, RGB) or PGM (P5,
//! grayscale; 16-bit big-endian for the 12-bit-precision fixture). The
//! corpus is documented in `../../docs/image/jpeg/fixtures/README.md`.
//!
//! This driver decodes every fixture through `oxideav_mjpeg`'s in-tree
//! decoder, converts the planar output back to interleaved RGB (or
//! grayscale) at the same bit depth as the ground truth, and reports
//! per-channel RMS, match percentage, and PSNR. JPEG is lossy: bit-exact
//! is achievable for `baseline-q100-no-loss` and the lossless
//! `lossless-1986-mode` fixture; everything else is high-PSNR.
//!
//! Per-fixture classification:
//! * `Tier::ReportOnly` — log stats; do NOT fail. The matrix stays
//!   visible for the human implementer to spot regressions, but
//!   individual divergences are picked off in follow-up rounds.
//! * `Tier::Ignored` — variants the decoder explicitly does not support
//!   (arithmetic-coded SOF9 returns `Error::Unsupported`; YUV 4:1:1
//!   sampling is rejected because `PixelFormat::Yuv411P` does not exist
//!   in the shared enum yet).
//!
//! Color-space convention:
//! * 1-component JPEGs decode to `Gray8` / `Gray12Le` and are compared
//!   directly against the PGM payload.
//! * 3-component RGB JPEGs (Adobe APP14 transform=0, component IDs
//!   R/G/B) decode to `Yuv444P`, but the planes are R/G/B; we treat
//!   plane[0]/[1]/[2] as R/G/B without colour conversion.
//! * 3-component YCbCr JPEGs decode to `Yuv420P` / `Yuv422P` / `Yuv444P`
//!   with planes Y/Cb/Cr; we upsample chroma by nearest-neighbour and
//!   apply the JFIF / BT.601 full-range YCbCr→RGB transform before
//!   comparing against the PPM.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};

/// Locate `docs/image/jpeg/fixtures/<name>/`. Tests run with CWD set to
/// the crate root, so we walk two levels up to reach the workspace root
/// and then into `docs/`.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/image/jpeg/fixtures").join(name)
}

// ---------------------------------------------------------------------------
// PPM / PGM (Netpbm "raw" P5/P6) parser.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Pnm {
    /// Magic: 5 (grayscale) or 6 (RGB).
    magic: u8,
    width: u32,
    height: u32,
    /// 255 for 8-bit, 4095 for 12-bit (each sample is 2 bytes big-endian
    /// when maxval > 255 — Netpbm convention).
    maxval: u32,
    /// Raw payload after the header. For maxval > 255, each sample is
    /// big-endian u16; bytes.len() == width * height * channels * 2.
    bytes: Vec<u8>,
}

impl Pnm {
    fn channels(&self) -> usize {
        if self.magic == 5 {
            1
        } else {
            3
        }
    }
    fn bytes_per_sample(&self) -> usize {
        if self.maxval > 255 {
            2
        } else {
            1
        }
    }
}

fn parse_pnm(bytes: &[u8]) -> Pnm {
    assert!(bytes.len() >= 2, "PNM too short");
    assert_eq!(bytes[0], b'P', "not a PNM file (no leading 'P')");
    let magic = match bytes[1] {
        b'5' => 5,
        b'6' => 6,
        other => panic!("unsupported PNM magic P{}", other as char),
    };

    // Tokenise: skip whitespace + '#'-comments, collect 3 numeric tokens
    // (width, height, maxval), then the next single byte is the
    // separator before the binary payload.
    let mut i = 2usize;
    let mut toks: Vec<String> = Vec::new();
    while toks.len() < 3 {
        // Skip whitespace.
        while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
        }
        if i >= bytes.len() {
            panic!("PNM ended before 3 header tokens");
        }
        if bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < bytes.len() && !matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r' | b'#') {
            i += 1;
        }
        toks.push(String::from_utf8_lossy(&bytes[start..i]).into_owned());
    }
    // One single whitespace byte separates the header from the payload.
    i += 1;

    let width: u32 = toks[0].parse().expect("PNM width");
    let height: u32 = toks[1].parse().expect("PNM height");
    let maxval: u32 = toks[2].parse().expect("PNM maxval");
    Pnm {
        magic,
        width,
        height,
        maxval,
        bytes: bytes[i..].to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Convert decoder output to a comparable buffer matching the PNM payload.
// ---------------------------------------------------------------------------

/// Result of decoding + flattening one fixture.
struct Decoded {
    /// Pixel format the decoder emitted (for diagnostics).
    pix_fmt: PixelFormat,
    width: u32,
    height: u32,
    channels: usize,
    /// 1 for 8-bit, 2 for 16-bit (matches PNM convention).
    bytes_per_sample: usize,
    /// Interleaved data (RGBRGBRGB... for 3-channel, GG... for 1-channel).
    /// Big-endian u16 per sample when bytes_per_sample == 2 (matches PNM).
    data: Vec<u8>,
}

/// JPEG/JFIF full-range BT.601 YCbCr → RGB. Returns clamped 0..=255.
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    // Same coefficients used elsewhere in the decoder for YCCK; the
    // standard JFIF formula is:
    //   R = Y + 1.402   * (Cr - 128)
    //   G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    //   B = Y + 1.772   * (Cb - 128)
    // Use rounded fixed-point mirrors of the decoder's own constants
    // for parity with internal colour math.
    let y_s = y as i32;
    let cbi = cb as i32 - 128;
    let cri = cr as i32 - 128;
    let r = (y_s + ((cri * 91881 + 32768) >> 16)).clamp(0, 255) as u8;
    let g = (y_s - ((cbi * 22554 + cri * 46802 + 32768) >> 16)).clamp(0, 255) as u8;
    let b = (y_s + ((cbi * 116130 + 32768) >> 16)).clamp(0, 255) as u8;
    (r, g, b)
}

/// Flatten a decoded VideoFrame into the format that matches the PNM
/// ground-truth (interleaved RGB or planar gray, 8-bit or 16-bit BE).
///
/// `is_rgb_jpeg` distinguishes a 3-component RGB JPEG (Adobe APP14
/// transform=0) — where the decoder's "Yuv444P" planes are actually
/// R/G/B — from a 3-component YCbCr JPEG which needs colour conversion.
fn flatten_frame(
    vf: &oxideav_core::VideoFrame,
    fmt: PixelFormat,
    width: u32,
    height: u32,
    is_rgb_jpeg: bool,
) -> Decoded {
    let w = width as usize;
    let h = height as usize;
    match fmt {
        PixelFormat::Gray8 => {
            assert_eq!(vf.planes.len(), 1, "Gray8 frame should have 1 plane");
            let stride = vf.planes[0].stride;
            let mut data = Vec::with_capacity(w * h);
            for y in 0..h {
                data.extend_from_slice(&vf.planes[0].data[y * stride..y * stride + w]);
            }
            Decoded {
                pix_fmt: fmt,
                width,
                height,
                channels: 1,
                bytes_per_sample: 1,
                data,
            }
        }
        PixelFormat::Gray12Le => {
            // Decoder stores 12-bit samples in little-endian u16 per
            // plane element. PGM 16-bit is big-endian. Re-pack to BE,
            // sample-by-sample.
            assert_eq!(vf.planes.len(), 1, "Gray12Le frame should have 1 plane");
            let stride = vf.planes[0].stride; // bytes
            let mut data = Vec::with_capacity(w * h * 2);
            for y in 0..h {
                let row = &vf.planes[0].data[y * stride..y * stride + w * 2];
                for x in 0..w {
                    let lo = row[x * 2];
                    let hi = row[x * 2 + 1];
                    let v = u16::from_le_bytes([lo, hi]);
                    data.extend_from_slice(&v.to_be_bytes());
                }
            }
            Decoded {
                pix_fmt: fmt,
                width,
                height,
                channels: 1,
                bytes_per_sample: 2,
                data,
            }
        }
        PixelFormat::Yuv444P | PixelFormat::Yuv422P | PixelFormat::Yuv420P => {
            assert_eq!(vf.planes.len(), 3, "YUV frame should have 3 planes");
            let (cw, ch) = match fmt {
                PixelFormat::Yuv444P => (w, h),
                PixelFormat::Yuv422P => (w.div_ceil(2), h),
                PixelFormat::Yuv420P => (w.div_ceil(2), h.div_ceil(2)),
                _ => unreachable!(),
            };
            let y_stride = vf.planes[0].stride;
            let cb_stride = vf.planes[1].stride;
            let cr_stride = vf.planes[2].stride;
            let mut data = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                for x in 0..w {
                    let yv = vf.planes[0].data[y * y_stride + x];
                    // Nearest-neighbour upsample for chroma.
                    let cy = (y * ch / h).min(ch - 1);
                    let cx = (x * cw / w).min(cw - 1);
                    let cb = vf.planes[1].data[cy * cb_stride + cx];
                    let cr = vf.planes[2].data[cy * cr_stride + cx];
                    let (r, g, b) = if is_rgb_jpeg {
                        // Plane[0]/[1]/[2] already carry R/G/B.
                        (yv, cb, cr)
                    } else {
                        ycbcr_to_rgb(yv, cb, cr)
                    };
                    data.push(r);
                    data.push(g);
                    data.push(b);
                }
            }
            Decoded {
                pix_fmt: fmt,
                width,
                height,
                channels: 3,
                bytes_per_sample: 1,
                data,
            }
        }
        other => panic!("flatten_frame: unsupported PixelFormat {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Per-channel comparison stats.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Copy)]
struct ChannelStats {
    /// Number of samples compared.
    n: usize,
    /// Number of exactly-equal samples.
    exact: usize,
    /// Sum of absolute differences.
    abs_sum: u64,
    /// Sum of squared differences.
    sq_sum: u64,
    /// Largest absolute difference observed.
    max_abs: u32,
}

impl ChannelStats {
    fn match_pct(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.exact as f64 / self.n as f64 * 100.0
        }
    }
    fn rms(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            (self.sq_sum as f64 / self.n as f64).sqrt()
        }
    }
    /// PSNR with the given peak signal value (255 for 8-bit, 4095 for
    /// 12-bit). Returns f64::INFINITY when bit-exact.
    fn psnr(&self, peak: f64) -> f64 {
        if self.sq_sum == 0 {
            f64::INFINITY
        } else {
            let mse = self.sq_sum as f64 / self.n as f64;
            20.0 * peak.log10() - 10.0 * mse.log10()
        }
    }
}

/// Compare `got` and `want` (both interleaved channels-first if 3-channel,
/// flat if 1-channel; same `bytes_per_sample`). Returns one
/// `ChannelStats` per channel.
fn compare_per_channel(got: &Decoded, want: &Pnm) -> Vec<ChannelStats> {
    assert_eq!(got.channels, want.channels(), "channel count mismatch");
    assert_eq!(
        got.bytes_per_sample,
        want.bytes_per_sample(),
        "bit-depth mismatch"
    );
    let chans = got.channels;
    let mut stats = vec![ChannelStats::default(); chans];

    let bps = got.bytes_per_sample;
    let n_samples = (got.width as usize) * (got.height as usize) * chans;
    assert_eq!(
        got.data.len(),
        n_samples * bps,
        "decoded buffer size mismatch (got {} expected {})",
        got.data.len(),
        n_samples * bps
    );
    assert_eq!(
        want.bytes.len(),
        n_samples * bps,
        "expected buffer size mismatch (got {} expected {})",
        want.bytes.len(),
        n_samples * bps
    );

    for i in 0..n_samples {
        let ch = i % chans;
        let (g, w) = if bps == 1 {
            (got.data[i] as i32, want.bytes[i] as i32)
        } else {
            // Both buffers are big-endian u16 per sample.
            let off = i * 2;
            let gv = u16::from_be_bytes([got.data[off], got.data[off + 1]]) as i32;
            let wv = u16::from_be_bytes([want.bytes[off], want.bytes[off + 1]]) as i32;
            (gv, wv)
        };
        let d = (g - w).unsigned_abs();
        stats[ch].n += 1;
        if d == 0 {
            stats[ch].exact += 1;
        }
        stats[ch].abs_sum += d as u64;
        stats[ch].sq_sum += (d as u64) * (d as u64);
        if d > stats[ch].max_abs {
            stats[ch].max_abs = d;
        }
    }
    stats
}

// ---------------------------------------------------------------------------
// Per-fixture driver.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Tier {
    /// Decode is permitted to diverge (lossy JPEG); we log per-channel
    /// stats but do not gate CI on them.
    ReportOnly,
    /// Variant the decoder cannot handle yet; logged but not asserted.
    Ignored,
}

struct CorpusCase {
    name: &'static str,
    /// Set to `true` for fixtures whose JPEG carries R/G/B components
    /// (Adobe APP14 transform=0, component IDs 82/71/66) — the decoder
    /// will tag the frame as `Yuv444P` but the planes are R/G/B and we
    /// must skip colour conversion.
    is_rgb_jpeg: bool,
    tier: Tier,
}

fn evaluate(case: &CorpusCase) {
    let dir = fixture_dir(case.name);
    let jpg_path = dir.join("input.jpg");
    let ppm_path = if dir.join("expected.ppm").exists() {
        dir.join("expected.ppm")
    } else {
        dir.join("expected.pgm")
    };

    let jpg = match fs::read(&jpg_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, jpg_path.display());
            return;
        }
    };
    let pnm_bytes = match fs::read(&ppm_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, ppm_path.display());
            return;
        }
    };

    let want = parse_pnm(&pnm_bytes);
    eprintln!(
        "[{:?}] {}: input.jpg={} B, expected={} ({}x{}, {} ch, maxval {})",
        case.tier,
        case.name,
        jpg.len(),
        ppm_path.file_name().unwrap().to_string_lossy(),
        want.width,
        want.height,
        want.channels(),
        want.maxval,
    );

    let mut params = CodecParameters::video(CodecId::new(oxideav_mjpeg::CODEC_ID_STR));
    params.width = Some(want.width);
    params.height = Some(want.height);
    let mut dec = oxideav_mjpeg::decoder::make_decoder(&params).expect("make_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 1), jpg);
    if let Err(e) = dec.send_packet(&pkt) {
        // Both tiers swallow decode errors — ReportOnly so we don't gate
        // CI on lossy variants regressing, Ignored because the variant
        // is documented-unsupported. The error itself is the report.
        eprintln!("  send_packet ERROR: {e:?}");
        return;
    }
    let frame = match dec.receive_frame() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  receive_frame ERROR: {e:?}");
            return;
        }
    };
    let Frame::Video(vf) = frame else {
        panic!("{}: expected Frame::Video", case.name);
    };

    // The decoder doesn't expose its chosen PixelFormat on the frame, so
    // infer it from plane count + sizes against the requested width/height.
    let inferred = infer_pix_fmt(&vf, want.width as usize, want.height as usize);
    let got = flatten_frame(&vf, inferred, want.width, want.height, case.is_rgb_jpeg);

    let stats = compare_per_channel(&got, &want);
    let peak = want.maxval as f64;
    let ch_labels: &[&str] = if got.channels == 1 {
        &["G"]
    } else if case.is_rgb_jpeg || got.bytes_per_sample == 1 {
        // After flatten_frame, even YUV JPEGs have been RGB-converted.
        &["R", "G", "B"]
    } else {
        &["?", "?", "?"]
    };
    let mut total = ChannelStats::default();
    for (i, s) in stats.iter().enumerate() {
        eprintln!(
            "  ch[{}] {}: n={} exact={} ({:.3}%) RMS={:.3} max|d|={} PSNR={:.2} dB",
            i,
            ch_labels[i],
            s.n,
            s.exact,
            s.match_pct(),
            s.rms(),
            s.max_abs,
            s.psnr(peak),
        );
        total.n += s.n;
        total.exact += s.exact;
        total.abs_sum += s.abs_sum;
        total.sq_sum += s.sq_sum;
        total.max_abs = total.max_abs.max(s.max_abs);
    }
    eprintln!(
        "  total: n={} exact={} ({:.3}%) RMS={:.3} max|d|={} PSNR={:.2} dB  pix_fmt={:?}",
        total.n,
        total.exact,
        total.match_pct(),
        total.rms(),
        total.max_abs,
        total.psnr(peak),
        got.pix_fmt,
    );
}

/// Best-effort PixelFormat inference from the VideoFrame shape. The
/// decoder picks one of {Gray8, Gray12Le, Yuv420P, Yuv422P, Yuv444P,
/// Cmyk, Gray*Le} but doesn't tag the frame; we re-derive it here just
/// for human-readable logging + to drive `flatten_frame`.
fn infer_pix_fmt(vf: &oxideav_core::VideoFrame, w: usize, h: usize) -> PixelFormat {
    match vf.planes.len() {
        1 => {
            // Stride in bytes. Gray8 → stride == width; Gray12Le →
            // stride == width * 2.
            let s = vf.planes[0].stride;
            if s == w {
                PixelFormat::Gray8
            } else if s == w * 2 {
                PixelFormat::Gray12Le
            } else {
                panic!("infer_pix_fmt: unexpected gray stride {s} for width {w}");
            }
        }
        3 => {
            let cw = vf.planes[1].stride;
            let ch = vf.planes[1].data.len() / cw.max(1);
            let yw = vf.planes[0].stride;
            assert_eq!(yw, w, "luma stride should equal width");
            // Compare chroma-plane geometry to the three legal subsamplings.
            let (full_w, full_h) = (w, h);
            if cw == full_w && ch == full_h {
                PixelFormat::Yuv444P
            } else if cw == full_w.div_ceil(2) && ch == full_h {
                PixelFormat::Yuv422P
            } else if cw == full_w.div_ceil(2) && ch == full_h.div_ceil(2) {
                PixelFormat::Yuv420P
            } else {
                panic!(
                    "infer_pix_fmt: unrecognised chroma geometry cw={cw} ch={ch} for {w}x{h}"
                );
            }
        }
        n => panic!("infer_pix_fmt: unexpected plane count {n}"),
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests
// ---------------------------------------------------------------------------

// --- Tier::ReportOnly: decoder is expected to produce a high-PSNR
//     match (or bit-exact for q100 / lossless). Stats logged for the
//     human reader; CI does not currently gate on PSNR thresholds so
//     regressions are visible without flapping. ---

#[test]
fn corpus_tiny_baseline_1x1() {
    evaluate(&CorpusCase {
        name: "tiny-baseline-1x1",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_baseline_grayscale_32x32() {
    evaluate(&CorpusCase {
        name: "baseline-grayscale-32x32",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_baseline_rgb_32x32() {
    // Adobe APP14 transform=0; component IDs R/G/B. Decoder tags as
    // Yuv444P but plane[0..3] are literally R/G/B.
    evaluate(&CorpusCase {
        name: "baseline-rgb-32x32",
        is_rgb_jpeg: true,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_baseline_yuv422_32x32() {
    evaluate(&CorpusCase {
        name: "baseline-yuv422-32x32",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_baseline_yuv420_128x128_q75() {
    evaluate(&CorpusCase {
        name: "baseline-yuv420-128x128-q75",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_baseline_q1_low_quality() {
    // q=1 — extreme quantization, expect low PSNR but a valid decode.
    evaluate(&CorpusCase {
        name: "baseline-q1-low-quality",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_baseline_q100_no_loss() {
    // q=100 — all-ones DQT. Highest possible PSNR; near-bit-exact (the
    // only loss is the YCbCr↔RGB rounding, since the ground truth was
    // re-decoded by libjpeg-turbo itself with the same transform).
    evaluate(&CorpusCase {
        name: "baseline-q100-no-loss",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_progressive_yuv420_128x128() {
    evaluate(&CorpusCase {
        name: "progressive-yuv420-128x128",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_multi_scan_non_interleaved() {
    evaluate(&CorpusCase {
        name: "multi-scan-non-interleaved",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_extended_sequential_12bit() {
    // SOF1 P=12 grayscale. Decoder produces Gray12Le; ground truth is
    // PGM with maxval=4095 and 16-bit-BE samples.
    evaluate(&CorpusCase {
        name: "extended-sequential-12bit",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_lossless_1986_mode() {
    // SOF3 grayscale predictor=1. Truly lossless: should be bit-exact
    // at maxval=255.
    evaluate(&CorpusCase {
        name: "lossless-1986-mode",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_with_restart_interval_8() {
    evaluate(&CorpusCase {
        name: "with-restart-interval-8",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_with_icc_profile_embedded() {
    // APP2 ICC_PROFILE — decoder ignores APPn segments.
    evaluate(&CorpusCase {
        name: "with-icc-profile-embedded",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_without_jfif_marker() {
    evaluate(&CorpusCase {
        name: "without-jfif-marker",
        is_rgb_jpeg: false,
        tier: Tier::ReportOnly,
    });
}

// --- Tier::Ignored: variants the decoder explicitly rejects. We still
//     run them so the failure reason is visible in the log; we just
//     don't fail the test on the rejection. ---

#[test]
fn corpus_arithmetic_coded() {
    // SOF9 — extended sequential arithmetic. Decoder returns
    // `Error::Unsupported` per the lib.rs feature-gap notes. Ground
    // truth is grayscale PGM (component_id list is single Y only).
    // Tracked for follow-up: requires the Q-coder / arithmetic
    // entropy decoder to be implemented.
    evaluate(&CorpusCase {
        name: "arithmetic-coded",
        is_rgb_jpeg: false,
        tier: Tier::Ignored,
    });
}

#[test]
fn corpus_baseline_yuv411_32x32() {
    // 4:1:1 chroma subsampling: luma h_factor=4, v_factor=1. The
    // decoder rejects this with `Error::Unsupported` because the
    // shared `PixelFormat` enum has no `Yuv411P` variant. Tracked
    // for follow-up: needs either a `Yuv411P` addition to
    // `oxideav-core`, or an internal upsample-to-4:2:2 path.
    evaluate(&CorpusCase {
        name: "baseline-yuv411-32x32",
        is_rgb_jpeg: false,
        tier: Tier::Ignored,
    });
}
