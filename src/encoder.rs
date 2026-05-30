//! Baseline / progressive JPEG encoder.
//!
//! The default path produces a standalone baseline JPEG per video frame:
//! SOI, JFIF APP0, DQT, SOF0, DHT (Annex K tables), SOS, entropy scan, EOI.
//! Handles 4:2:0 / 4:2:2 / 4:4:4 YUV planar input. Restart markers
//! (DRI + `RSTn`) are emitted when [`MjpegEncoder::set_restart_interval`]
//! (or [`encode_jpeg_with_opts`]) is called with a non-zero MCU count;
//! by default the encoder matches the historical behaviour and emits
//! none.
//!
//! Progressive (SOF2) emission is available via
//! [`MjpegEncoder::set_progressive`] (or [`encode_jpeg_progressive`]). The
//! spectral-selection decomposition used here is:
//!   1. Interleaved DC-first scan (`Ss=0, Se=0, Ah=0, Al=0`) covering every
//!      component.
//!   2. A per-component AC scan over the low-frequency band
//!      (`Ss=1, Se=5, Ah=0, Al=0`).
//!   3. A per-component AC scan over the high-frequency band
//!      (`Ss=6, Se=63, Ah=0, Al=0`).
//!
//! Full successive-approximation (SA) progressive emission is available via
//! [`encode_jpeg_progressive_sa`]. The 2-bit-point-transform decomposition is:
//!   1. DC initial scan (`Ss=0, Se=0, Ah=0, Al=1`) — sends `coef >> 1`.
//!   2. Per-component AC initial low band (`Ss=1, Se=5, Ah=0, Al=1`).
//!   3. Per-component AC initial high band (`Ss=6, Se=63, Ah=0, Al=1`).
//!   4. DC refinement scan (`Ss=0, Se=0, Ah=1, Al=0`) — sends the dropped bit.
//!   5. Per-component AC refinement low band (`Ss=1, Se=5, Ah=1, Al=0`).
//!   6. Per-component AC refinement high band (`Ss=6, Se=63, Ah=1, Al=0`).
//!
//! Metadata (EXIF, ICC, XMP …) pass-through is available on both the
//! baseline and progressive paths via [`encode_jpeg_with_meta`] /
//! [`encode_jpeg_progressive_with_meta`].  The caller supplies a pre-assembled
//! byte slice that will be inserted verbatim between SOI and the first DQT
//! segment.  Use [`extract_app_segments`] to harvest the APP segments from a
//! previously-decoded JPEG when building a transcode pipeline.

use crate::error::{MjpegError as Error, Result};

// When the `registry` feature is on, the public `encode_jpeg_*`
// functions accept the framework's `oxideav_core::VideoFrame` /
// `PixelFormat` types directly so callers that already operate on
// trait-API plumbing don't have to translate. The inner code only
// touches plane `(stride, data)` tuples and 4:4:4/4:2:2/4:2:0 YUV
// pixel-format discriminants — both type families share that shape,
// so the same encode helpers compile against either alias unchanged.
//
// With `--no-default-features` the same names resolve to the
// crate-local [`MjpegFrame`] / [`MjpegPixelFormat`] / [`MjpegPlane`]
// types so the standalone build never references `oxideav-core`.
#[cfg(feature = "registry")]
use oxideav_core::{PixelFormat, VideoFrame};

#[cfg(not(feature = "registry"))]
use crate::image::{MjpegFrame as VideoFrame, MjpegPixelFormat as PixelFormat};

// Re-export the framework-side `Encoder` factory and concrete
// `MjpegEncoder` at their historical paths so consumers (and
// integration tests) that import `oxideav_mjpeg::encoder::make_encoder`
// or `oxideav_mjpeg::encoder::MjpegEncoder` keep compiling.
#[cfg(feature = "registry")]
pub use crate::registry::{make_encoder, MjpegEncoder};

use crate::jpeg::dct::fdct8x8;
use crate::jpeg::huffman::{
    DefaultHuffman, HuffTable, STD_AC_CHROMA_BITS, STD_AC_CHROMA_VALS, STD_AC_LUMA_BITS,
    STD_AC_LUMA_VALS, STD_DC_CHROMA_BITS, STD_DC_CHROMA_VALS, STD_DC_LUMA_BITS, STD_DC_LUMA_VALS,
};
use crate::jpeg::markers;
use crate::jpeg::quant::{scale_for_quality, DEFAULT_CHROMA_Q50, DEFAULT_LUMA_Q50};
use crate::jpeg::zigzag::ZIGZAG;

/// Quality factor 1..=100 (libjpeg style). 75 is a sensible default.
pub const DEFAULT_QUALITY: u8 = 75;

// ---- Encoding ------------------------------------------------------------

/// Encode a single `VideoFrame` (YUV 4:4:4 / 4:2:2 / 4:2:0) as a complete,
/// self-contained baseline JPEG byte stream (`FFD8 … FFD9`). `quality` is
/// the libjpeg-style 1..=100 factor. Exposed publicly so sibling crates
/// (e.g. `oxideav-amv`, which wraps the same bitstream with a custom
/// container-level header) can reuse the encoder without going through the
/// `Encoder` trait's stateful packet/frame plumbing.
///
/// Does not emit restart markers. For a restart-marker-aware variant see
/// [`encode_jpeg_with_opts`].
pub fn encode_jpeg(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
) -> Result<Vec<u8>> {
    encode_jpeg_with_opts(frame, width, height, pix, quality, 0)
}

/// Extract all APP0..APP15 and COM segments from a JPEG byte stream as a
/// single concatenated byte slice (allocated). The returned bytes are the
/// raw marker + length + payload that can be injected verbatim into another
/// JPEG between SOI and DQT via [`encode_jpeg_with_meta`] /
/// [`encode_jpeg_progressive_with_meta`].
///
/// Segments that are not APP or COM markers are skipped. Returns an empty
/// `Vec` if the input is not a valid JPEG or carries no metadata.
pub fn extract_app_segments(jpeg: &[u8]) -> Vec<u8> {
    if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != markers::SOI {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut pos = 2usize;
    while pos + 3 < jpeg.len() {
        if jpeg[pos] != 0xFF {
            break;
        }
        // Skip fill bytes.
        let mut p = pos + 1;
        while p < jpeg.len() && jpeg[p] == 0xFF {
            p += 1;
        }
        if p >= jpeg.len() {
            break;
        }
        let marker = jpeg[p];
        p += 1; // now points at length MSB
                // SOI/EOI/RST have no length.
        if marker == markers::SOI || marker == markers::EOI || markers::is_rst(marker) {
            pos = p;
            continue;
        }
        if p + 2 > jpeg.len() {
            break;
        }
        let len = u16::from_be_bytes([jpeg[p], jpeg[p + 1]]) as usize;
        if len < 2 || p + len > jpeg.len() {
            break;
        }
        // Collect APP0..APP15 and COM.
        if markers::is_app(marker) || marker == markers::COM {
            // marker_byte + length_hi + length_lo + payload
            out.push(0xFF);
            out.push(marker);
            out.extend_from_slice(&jpeg[p..p + len]);
        }
        // Stop when we hit DQT / SOF / SOS — those are codec segments, not meta.
        if marker == markers::DQT
            || markers::is_sof(marker)
            || marker == markers::SOS
            || marker == markers::DHT
        {
            break;
        }
        pos = p + len;
    }
    out
}

/// Like [`encode_jpeg`] but also emits a DRI segment and cycles
/// `RST0..=RST7` markers every `restart_interval` MCUs during the scan.
/// Passing `0` disables restart marker emission (equivalent to
/// [`encode_jpeg`]).
pub fn encode_jpeg_with_opts(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
    restart_interval: u16,
) -> Result<Vec<u8>> {
    encode_jpeg_with_meta(frame, width, height, pix, quality, restart_interval, &[])
}

/// Like [`encode_jpeg_with_opts`] but inserts `meta` verbatim between the
/// SOI and the first DQT segment. `meta` must contain only APP0..APP15 or
/// COM segments (each starting with `0xFF 0xEn` / `0xFF 0xFE` followed by
/// a big-endian length). Use [`extract_app_segments`] to harvest metadata
/// from a source JPEG.
///
/// When `meta` is empty this is bit-for-bit identical to
/// [`encode_jpeg_with_opts`].
pub fn encode_jpeg_with_meta(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
    restart_interval: u16,
    meta: &[u8],
) -> Result<Vec<u8>> {
    let width = width as usize;
    let height = height as usize;
    let (h_factor, v_factor) = match pix {
        PixelFormat::Yuv444P => (1u8, 1u8),
        PixelFormat::Yuv422P => (2, 1),
        PixelFormat::Yuv420P => (2, 2),
        _ => {
            return Err(Error::unsupported(
                "MJPEG encoder: unsupported pixel format",
            ))
        }
    };
    if frame.planes.len() != 3 {
        return Err(Error::invalid("MJPEG encoder: expected 3 planes"));
    }

    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let chroma_q = scale_for_quality(&DEFAULT_CHROMA_Q50, quality);

    let huff = DefaultHuffman::build()?;

    let mut out: Vec<u8> = Vec::with_capacity(16_384);
    // SOI.
    out.push(0xFF);
    out.push(markers::SOI);
    // Metadata segments (JFIF APP0 fallback when caller provides nothing).
    if meta.is_empty() {
        write_jfif_app0(&mut out);
    } else {
        out.extend_from_slice(meta);
    }
    // DQT (both tables, precision 0 = 8-bit).
    write_dqt(&mut out, 0, &luma_q);
    write_dqt(&mut out, 1, &chroma_q);
    // SOF0.
    write_sof0(&mut out, width as u16, height as u16, h_factor, v_factor);
    // DHT: all 4 default tables.
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_dht(&mut out, 0, 1, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    write_dht(&mut out, 1, 1, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);
    // DRI — placed right before SOS per T.81 F.2.2.4.
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    // SOS.
    write_sos(&mut out);
    // Scan data.
    write_scan(
        &mut out,
        frame,
        pix,
        width,
        height,
        h_factor,
        v_factor,
        &luma_q,
        &chroma_q,
        &huff,
        restart_interval,
    )?;
    // EOI.
    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

/// Encode a single `VideoFrame` (YUV 4:4:4 / 4:2:2 / 4:2:0) as a standalone
/// **progressive** JPEG byte stream (SOF2). The scan decomposition is
/// three scans: one interleaved DC-first scan followed by two
/// non-interleaved AC band scans per component (`Ss=1..=5` then
/// `Ss=6..=63`), all at `Ah=0, Al=0` — i.e. spectral selection without
/// successive-approximation refinement. Uses the Annex K Huffman tables.
/// Quality is the libjpeg-style factor 1..=100.
pub fn encode_jpeg_progressive(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
) -> Result<Vec<u8>> {
    encode_jpeg_progressive_with_meta(frame, width, height, pix, quality, &[])
}

/// Like [`encode_jpeg_progressive`] but inserts `meta` verbatim between the
/// SOI and the first DQT segment. See [`extract_app_segments`] and
/// [`encode_jpeg_with_meta`] for details.
pub fn encode_jpeg_progressive_with_meta(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
    meta: &[u8],
) -> Result<Vec<u8>> {
    encode_jpeg_progressive_inner(frame, width, height, pix, quality, meta, false)
}

/// Encode a single `VideoFrame` (YUV 4:4:4 / 4:2:2 / 4:2:0) as a
/// **progressive** JPEG with full successive-approximation (SA) scan
/// decomposition. Uses a 1-bit point transform (`Al=1` initial, `Al=0`
/// refinement). The 6+3+3 = 12-scan structure is:
///
/// - Scan 1: Interleaved DC initial (`Ss=0,Se=0, Ah=0,Al=1`) — all 3 components.
/// - Scans 2-4: Per-component AC initial low band (`Ss=1..5, Ah=0,Al=1`) × 3.
/// - Scans 5-7: Per-component AC initial high band (`Ss=6..63, Ah=0,Al=1`) × 3.
/// - Scan 8: Interleaved DC refinement (`Ss=0,Se=0, Ah=1,Al=0`) — all 3 components.
/// - Scans 9-11: Per-component AC refinement low band (`Ss=1..5, Ah=1,Al=0`) × 3.
/// - Scans 12-14: Per-component AC refinement high band (`Ss=6..63, Ah=1,Al=0`) × 3.
///
/// Output round-trips through any conformant SOF2 decoder (ffmpeg, libjpeg,
/// ImageMagick). The reconstructed image is identical to the spectral-
/// selection-only output — successive approximation only changes the bit
/// order, not the quantised coefficients.
pub fn encode_jpeg_progressive_sa(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
) -> Result<Vec<u8>> {
    encode_jpeg_progressive_sa_with_meta(frame, width, height, pix, quality, &[])
}

/// Like [`encode_jpeg_progressive_sa`] but inserts `meta` verbatim between
/// SOI and the first DQT.
pub fn encode_jpeg_progressive_sa_with_meta(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
    meta: &[u8],
) -> Result<Vec<u8>> {
    encode_jpeg_progressive_inner(frame, width, height, pix, quality, meta, true)
}

/// Internal progressive encoder. `use_sa` selects the successive-approximation
/// (2-pass) decomposition; `false` gives spectral-selection-only.
#[allow(clippy::too_many_arguments)]
fn encode_jpeg_progressive_inner(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
    meta: &[u8],
    use_sa: bool,
) -> Result<Vec<u8>> {
    let width = width as usize;
    let height = height as usize;
    let (h_factor, v_factor) = match pix {
        PixelFormat::Yuv444P => (1u8, 1u8),
        PixelFormat::Yuv422P => (2, 1),
        PixelFormat::Yuv420P => (2, 2),
        _ => {
            return Err(Error::unsupported(
                "MJPEG progressive encoder: unsupported pixel format",
            ))
        }
    };
    if frame.planes.len() != 3 {
        return Err(Error::invalid(
            "MJPEG progressive encoder: expected 3 planes",
        ));
    }

    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let chroma_q = scale_for_quality(&DEFAULT_CHROMA_Q50, quality);
    let huff = DefaultHuffman::build()?;

    // Compute DCT+quantise coefficients for every block in every component,
    // keyed by block index (block_y * blocks_x + block_x).
    let mcu_w_px = 8 * h_factor as usize;
    let mcu_h_px = 8 * v_factor as usize;
    let mcus_x = width.div_ceil(mcu_w_px);
    let mcus_y = height.div_ceil(mcu_h_px);

    // Luma: h_factor*mcus_x blocks wide × v_factor*mcus_y tall.
    let luma_blocks_x = mcus_x * h_factor as usize;
    let luma_blocks_y = mcus_y * v_factor as usize;
    let mut y_coefs = vec![[0i32; 64]; luma_blocks_x * luma_blocks_y];
    fill_coef_grid(
        &mut y_coefs,
        &frame.planes[0].data,
        frame.planes[0].stride,
        width,
        height,
        luma_blocks_x,
        luma_blocks_y,
        &luma_q,
    );

    // Chroma planes — always 1×1 block per MCU for 4:4:4/4:2:2/4:2:0.
    let (c_w, c_h) = match pix {
        PixelFormat::Yuv444P => (width, height),
        PixelFormat::Yuv422P => (width.div_ceil(2), height),
        PixelFormat::Yuv420P => (width.div_ceil(2), height.div_ceil(2)),
        _ => unreachable!(),
    };
    let chroma_blocks_x = mcus_x;
    let chroma_blocks_y = mcus_y;
    let mut cb_coefs = vec![[0i32; 64]; chroma_blocks_x * chroma_blocks_y];
    let mut cr_coefs = vec![[0i32; 64]; chroma_blocks_x * chroma_blocks_y];
    fill_coef_grid(
        &mut cb_coefs,
        &frame.planes[1].data,
        frame.planes[1].stride,
        c_w,
        c_h,
        chroma_blocks_x,
        chroma_blocks_y,
        &chroma_q,
    );
    fill_coef_grid(
        &mut cr_coefs,
        &frame.planes[2].data,
        frame.planes[2].stride,
        c_w,
        c_h,
        chroma_blocks_x,
        chroma_blocks_y,
        &chroma_q,
    );

    // Header.
    let mut out: Vec<u8> = Vec::with_capacity(16_384);
    out.push(0xFF);
    out.push(markers::SOI);
    if meta.is_empty() {
        write_jfif_app0(&mut out);
    } else {
        out.extend_from_slice(meta);
    }
    write_dqt(&mut out, 0, &luma_q);
    write_dqt(&mut out, 1, &chroma_q);
    write_sof2(&mut out, width as u16, height as u16, h_factor, v_factor);
    // All four default Huffman tables; scans pick per (class, id).
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_dht(&mut out, 0, 1, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    write_dht(&mut out, 1, 1, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);

    if use_sa {
        // ---- Successive-approximation (SA) decomposition, Al=1/Ah=0 first ----
        // Phase 1 — initial scans (Al=1): encoder sends coef >> 1 (drops LSB).

        // DC initial, interleaved, Ah=0 Al=1.
        write_sos_progressive_dc_sa(&mut out, 0, 1);
        write_dc_scan_interleaved_sa(
            &mut out,
            &y_coefs,
            luma_blocks_x,
            h_factor as usize,
            v_factor as usize,
            mcus_x,
            mcus_y,
            &cb_coefs,
            &cr_coefs,
            chroma_blocks_x,
            &huff,
            1, // Al = 1
        );

        // AC initial: low and high bands.
        for &(ss, se) in &[(1u8, 5u8), (6u8, 63u8)] {
            write_sos_progressive_ac_sa(&mut out, 1, 0, ss, se, 0, 1);
            write_ac_scan_sa(
                &mut out,
                &y_coefs,
                luma_blocks_x * luma_blocks_y,
                &huff.luma_ac,
                ss as usize,
                se as usize,
                1,
            );
            write_sos_progressive_ac_sa(&mut out, 2, 1, ss, se, 0, 1);
            write_ac_scan_sa(
                &mut out,
                &cb_coefs,
                chroma_blocks_x * chroma_blocks_y,
                &huff.chroma_ac,
                ss as usize,
                se as usize,
                1,
            );
            write_sos_progressive_ac_sa(&mut out, 3, 1, ss, se, 0, 1);
            write_ac_scan_sa(
                &mut out,
                &cr_coefs,
                chroma_blocks_x * chroma_blocks_y,
                &huff.chroma_ac,
                ss as usize,
                se as usize,
                1,
            );
        }

        // Phase 2 — refinement scans (Ah=1, Al=0): encoder sends the dropped LSB.

        // DC refinement, interleaved, Ah=1 Al=0.
        write_sos_progressive_dc_sa(&mut out, 1, 0);
        write_dc_refine_scan_interleaved(
            &mut out,
            &y_coefs,
            luma_blocks_x,
            h_factor as usize,
            v_factor as usize,
            mcus_x,
            mcus_y,
            &cb_coefs,
            &cr_coefs,
            chroma_blocks_x,
        );

        // AC refinement: low and high bands.
        for &(ss, se) in &[(1u8, 5u8), (6u8, 63u8)] {
            write_sos_progressive_ac_sa(&mut out, 1, 0, ss, se, 1, 0);
            write_ac_refine_scan(
                &mut out,
                &y_coefs,
                luma_blocks_x * luma_blocks_y,
                &huff.luma_ac,
                ss as usize,
                se as usize,
            );
            write_sos_progressive_ac_sa(&mut out, 2, 1, ss, se, 1, 0);
            write_ac_refine_scan(
                &mut out,
                &cb_coefs,
                chroma_blocks_x * chroma_blocks_y,
                &huff.chroma_ac,
                ss as usize,
                se as usize,
            );
            write_sos_progressive_ac_sa(&mut out, 3, 1, ss, se, 1, 0);
            write_ac_refine_scan(
                &mut out,
                &cr_coefs,
                chroma_blocks_x * chroma_blocks_y,
                &huff.chroma_ac,
                ss as usize,
                se as usize,
            );
        }
    } else {
        // ---- Spectral-selection-only decomposition (Ah=0, Al=0) -------------

        // Scan 1: interleaved DC-first (Ss=0, Se=0, Ah=0, Al=0). All 3 components.
        write_sos_progressive_dc_interleaved(&mut out);
        write_dc_scan_interleaved(
            &mut out,
            &y_coefs,
            luma_blocks_x,
            h_factor as usize,
            v_factor as usize,
            mcus_x,
            mcus_y,
            &cb_coefs,
            &cr_coefs,
            chroma_blocks_x,
            &huff,
        );

        // AC bands, low then high.
        for &(ss, se) in &[(1u8, 5u8), (6u8, 63u8)] {
            // Luma AC.
            write_sos_progressive_ac(&mut out, 1, 0, ss, se);
            write_ac_scan(
                &mut out,
                &y_coefs,
                luma_blocks_x * luma_blocks_y,
                &huff.luma_ac,
                ss as usize,
                se as usize,
            );
            // Cb AC.
            write_sos_progressive_ac(&mut out, 2, 1, ss, se);
            write_ac_scan(
                &mut out,
                &cb_coefs,
                chroma_blocks_x * chroma_blocks_y,
                &huff.chroma_ac,
                ss as usize,
                se as usize,
            );
            // Cr AC.
            write_sos_progressive_ac(&mut out, 3, 1, ss, se);
            write_ac_scan(
                &mut out,
                &cr_coefs,
                chroma_blocks_x * chroma_blocks_y,
                &huff.chroma_ac,
                ss as usize,
                se as usize,
            );
        }
    }

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

/// Perform forward DCT + quantisation on every 8×8 block of a plane,
/// storing the quantised coefficients in natural (row-major) order.
/// Edge blocks are filled by replicating the last valid pixel.
#[allow(clippy::too_many_arguments)]
fn fill_coef_grid(
    coefs: &mut [[i32; 64]],
    plane: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    blocks_x: usize,
    blocks_y: usize,
    quant: &[u16; 64],
) {
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut block = [0.0f32; 64];
            fill_block(&mut block, plane, stride, w, h, bx * 8, by * 8);
            fdct8x8(&mut block);
            let mut q = [0i32; 64];
            for k in 0..64 {
                let v = block[k] / quant[k] as f32;
                q[k] = if v >= 0.0 {
                    (v + 0.5) as i32
                } else {
                    -((-v + 0.5) as i32)
                };
            }
            coefs[by * blocks_x + bx] = q;
        }
    }
}

fn write_sof2(out: &mut Vec<u8>, width: u16, height: u16, h: u8, v: u8) {
    // Same shape as SOF0 — spec only changes the marker byte for a
    // progressive frame.
    let mut payload = Vec::with_capacity(8 + 9);
    payload.push(8); // precision
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(3); // components
    payload.push(1);
    payload.push((h << 4) | v);
    payload.push(0);
    payload.push(2);
    payload.push(0x11);
    payload.push(1);
    payload.push(3);
    payload.push(0x11);
    payload.push(1);
    write_length_prefix(out, markers::SOF2, &payload);
}

fn write_sos_progressive_dc_interleaved(out: &mut Vec<u8>) {
    // 3 components, Ss=0, Se=0, Ah|Al=0. Y uses DC=0, Cb/Cr use DC=1.
    let payload: [u8; 10] = [3, 1, 0x00, 2, 0x10, 3, 0x10, 0, 0, 0x00];
    write_length_prefix(out, markers::SOS, &payload);
}

fn write_sos_progressive_ac(out: &mut Vec<u8>, comp_id: u8, ac_table: u8, ss: u8, se: u8) {
    // Single-component AC scan. Td (DC table selector, high nibble) is
    // unused here but the field is always present; zero it. Ta (AC
    // selector) goes in the low nibble.
    let payload: [u8; 6] = [
        1,               // Ns
        comp_id,         // Cs
        ac_table & 0x0F, // Td | Ta
        ss,              // Ss
        se,              // Se
        0x00,            // Ah | Al
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

/// Interleaved DC scan SOS header for successive-approximation progressive.
/// `ah` is the previous approximation level; `al` is the current point-transform.
fn write_sos_progressive_dc_sa(out: &mut Vec<u8>, ah: u8, al: u8) {
    // 3 components, Ss=0, Se=0, Ah|Al. Y uses DC=0, Cb/Cr use DC=1.
    let payload: [u8; 10] = [3, 1, 0x00, 2, 0x10, 3, 0x10, 0, 0, (ah << 4) | (al & 0x0F)];
    write_length_prefix(out, markers::SOS, &payload);
}

/// Single-component AC scan SOS header with explicit Ah/Al.
fn write_sos_progressive_ac_sa(
    out: &mut Vec<u8>,
    comp_id: u8,
    ac_table: u8,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
) {
    let payload: [u8; 6] = [1, comp_id, ac_table & 0x0F, ss, se, (ah << 4) | (al & 0x0F)];
    write_length_prefix(out, markers::SOS, &payload);
}

/// Emit the interleaved DC-first progressive scan. Walks the MCU grid
/// identically to the baseline interleaved encoder, but only codes each
/// block's DC coefficient with the DC Huffman table.
#[allow(clippy::too_many_arguments)]
fn write_dc_scan_interleaved(
    out: &mut Vec<u8>,
    y_coefs: &[[i32; 64]],
    luma_blocks_x: usize,
    h_factor: usize,
    v_factor: usize,
    mcus_x: usize,
    mcus_y: usize,
    cb_coefs: &[[i32; 64]],
    cr_coefs: &[[i32; 64]],
    chroma_blocks_x: usize,
    huff: &DefaultHuffman,
) {
    let mut bw = BitWriter::new(out);
    let mut prev_dc_y: i32 = 0;
    let mut prev_dc_cb: i32 = 0;
    let mut prev_dc_cr: i32 = 0;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // Luma blocks in MCU.
            for by in 0..v_factor {
                for bx in 0..h_factor {
                    let bi_x = mx * h_factor + bx;
                    let bi_y = my * v_factor + by;
                    let bi = bi_y * luma_blocks_x + bi_x;
                    encode_dc(&mut bw, y_coefs[bi][0], &mut prev_dc_y, &huff.luma_dc);
                }
            }
            // Cb.
            let cbi = my * chroma_blocks_x + mx;
            encode_dc(&mut bw, cb_coefs[cbi][0], &mut prev_dc_cb, &huff.chroma_dc);
            // Cr.
            encode_dc(&mut bw, cr_coefs[cbi][0], &mut prev_dc_cr, &huff.chroma_dc);
        }
    }
    bw.finish();
}

fn encode_dc(bw: &mut BitWriter<'_>, dc: i32, prev_dc: &mut i32, dc_huff: &HuffTable) {
    let dc_diff = dc - *prev_dc;
    *prev_dc = dc;
    let (size, bits) = category(dc_diff);
    let hc = dc_huff.encode[size as usize];
    bw.write_bits(hc.code as u32, hc.len as u32);
    if size > 0 {
        bw.write_bits(bits, size as u32);
    }
}

/// Emit a non-interleaved AC-band scan over one component's blocks. Walks
/// `block_count` blocks in natural order and codes only zigzag coefficients
/// in `[ss..=se]` using the supplied AC Huffman table. Ah=0 (initial scan).
///
/// No EOB-run (EOBn) is emitted: every fully-zero band gets a plain EOB
/// (RS=0x00). This is valid and trades a handful of extra bits for much
/// simpler encoder logic.
fn write_ac_scan(
    out: &mut Vec<u8>,
    coefs: &[[i32; 64]],
    block_count: usize,
    ac_huff: &HuffTable,
    ss: usize,
    se: usize,
) {
    let mut bw = BitWriter::new(out);
    for bi in 0..block_count {
        let block = &coefs[bi];
        let mut run: u32 = 0;
        for k in ss..=se {
            let v = block[ZIGZAG[k]];
            if v == 0 {
                run += 1;
            } else {
                while run >= 16 {
                    let zc = ac_huff.encode[0xF0];
                    bw.write_bits(zc.code as u32, zc.len as u32);
                    run -= 16;
                }
                let (sz, bv) = category(v);
                let rs = ((run as u8) << 4) | sz;
                let ac = ac_huff.encode[rs as usize];
                bw.write_bits(ac.code as u32, ac.len as u32);
                if sz > 0 {
                    bw.write_bits(bv, sz as u32);
                }
                run = 0;
            }
        }
        if run > 0 {
            // Trailing zeros in the band → plain EOB.
            let eob = ac_huff.encode[0x00];
            bw.write_bits(eob.code as u32, eob.len as u32);
        }
    }
    bw.finish();
}

/// Emit an interleaved DC scan with successive-approximation point transform `al`.
/// In the initial pass (Ah=0, Al=al) each DC coefficient is right-shifted by `al`
/// before Huffman coding; in the predictor the full shifted value is used.
#[allow(clippy::too_many_arguments)]
fn write_dc_scan_interleaved_sa(
    out: &mut Vec<u8>,
    y_coefs: &[[i32; 64]],
    luma_blocks_x: usize,
    h_factor: usize,
    v_factor: usize,
    mcus_x: usize,
    mcus_y: usize,
    cb_coefs: &[[i32; 64]],
    cr_coefs: &[[i32; 64]],
    chroma_blocks_x: usize,
    huff: &DefaultHuffman,
    al: u8,
) {
    let mut bw = BitWriter::new(out);
    let mut prev_dc_y: i32 = 0;
    let mut prev_dc_cb: i32 = 0;
    let mut prev_dc_cr: i32 = 0;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            for by in 0..v_factor {
                for bx in 0..h_factor {
                    let bi_x = mx * h_factor + bx;
                    let bi_y = my * v_factor + by;
                    let bi = bi_y * luma_blocks_x + bi_x;
                    encode_dc_sa(&mut bw, y_coefs[bi][0], &mut prev_dc_y, &huff.luma_dc, al);
                }
            }
            let cbi = my * chroma_blocks_x + mx;
            encode_dc_sa(
                &mut bw,
                cb_coefs[cbi][0],
                &mut prev_dc_cb,
                &huff.chroma_dc,
                al,
            );
            encode_dc_sa(
                &mut bw,
                cr_coefs[cbi][0],
                &mut prev_dc_cr,
                &huff.chroma_dc,
                al,
            );
        }
    }
    bw.finish();
}

/// Encode one DC coefficient with a point-transform shift. The predictor
/// operates on the shifted values so DPCM is consistent.
fn encode_dc_sa(bw: &mut BitWriter<'_>, dc: i32, prev_dc: &mut i32, dc_huff: &HuffTable, al: u8) {
    let dc_shifted = dc >> al;
    let dc_diff = dc_shifted - *prev_dc;
    *prev_dc = dc_shifted;
    let (size, bits) = category(dc_diff);
    let hc = dc_huff.encode[size as usize];
    bw.write_bits(hc.code as u32, hc.len as u32);
    if size > 0 {
        bw.write_bits(bits, size as u32);
    }
}

/// Emit a DC refinement scan (Ah=1, Al=0). For each block emit exactly one
/// raw correction bit: the bit at position `1` (i.e., `(dc >> 0) & 1` which
/// is the LSB that was dropped in the initial Ah=0/Al=1 scan).
#[allow(clippy::too_many_arguments)]
fn write_dc_refine_scan_interleaved(
    out: &mut Vec<u8>,
    y_coefs: &[[i32; 64]],
    luma_blocks_x: usize,
    h_factor: usize,
    v_factor: usize,
    mcus_x: usize,
    mcus_y: usize,
    cb_coefs: &[[i32; 64]],
    cr_coefs: &[[i32; 64]],
    chroma_blocks_x: usize,
) {
    let mut bw = BitWriter::new(out);
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            for by in 0..v_factor {
                for bx in 0..h_factor {
                    let bi_x = mx * h_factor + bx;
                    let bi_y = my * v_factor + by;
                    let bi = bi_y * luma_blocks_x + bi_x;
                    // Correction bit is the dropped bit from the initial scan (bit 0 of dc).
                    let dc = y_coefs[bi][0];
                    bw.write_bits(dc.unsigned_abs() & 1, 1);
                }
            }
            let cbi = my * chroma_blocks_x + mx;
            let dc_cb = cb_coefs[cbi][0];
            bw.write_bits(dc_cb.unsigned_abs() & 1, 1);
            let dc_cr = cr_coefs[cbi][0];
            bw.write_bits(dc_cr.unsigned_abs() & 1, 1);
        }
    }
    bw.finish();
}

/// AC initial scan with point-transform `al`. Encodes `coef >> al`; any
/// coefficient whose magnitude rounds to zero is treated as zero in the
/// first pass (it becomes a new nonzero in the refinement scan).
fn write_ac_scan_sa(
    out: &mut Vec<u8>,
    coefs: &[[i32; 64]],
    block_count: usize,
    ac_huff: &HuffTable,
    ss: usize,
    se: usize,
    al: u8,
) {
    let mut bw = BitWriter::new(out);
    for bi in 0..block_count {
        let block = &coefs[bi];
        let mut run: u32 = 0;
        for k in ss..=se {
            let v = block[ZIGZAG[k]] >> al; // point-transform
            if v == 0 {
                run += 1;
            } else {
                while run >= 16 {
                    let zc = ac_huff.encode[0xF0];
                    bw.write_bits(zc.code as u32, zc.len as u32);
                    run -= 16;
                }
                let (sz, bv) = category(v);
                let rs = ((run as u8) << 4) | sz;
                let ac = ac_huff.encode[rs as usize];
                bw.write_bits(ac.code as u32, ac.len as u32);
                if sz > 0 {
                    bw.write_bits(bv, sz as u32);
                }
                run = 0;
            }
        }
        if run > 0 {
            let eob = ac_huff.encode[0x00];
            bw.write_bits(eob.code as u32, eob.len as u32);
        }
    }
    bw.finish();
}

/// AC refinement scan (Ah=1, Al=0) over one component's blocks, per T.81
/// §G.1.2.3. For each block walks zigzag positions `[ss..=se]`:
///
/// * Positions where `|coef >> 1|` was already nonzero (coded in the
///   first pass): emit a raw correction bit = `|coef| & 1`.
/// * Positions where `|coef >> 1| == 0` but `|coef| == 1` (new nonzero):
///   code them as RS=0x01 (run=0, size=1) with a sign bit.
/// * Run of zeros that precede a new nonzero or a band boundary: coded
///   in the RS run-length nibble or as ZRL (0xF0) runs.
/// * EOB when the rest of the band has no new nonzeros — emitted after
///   refining all pre-existing nonzeros in the band.
///
/// Delegates per-block work to [`emit_ac_refine_block`].
fn write_ac_refine_scan(
    out: &mut Vec<u8>,
    coefs: &[[i32; 64]],
    block_count: usize,
    ac_huff: &HuffTable,
    ss: usize,
    se: usize,
) {
    let mut bw = BitWriter::new(out);
    for bi in 0..block_count {
        emit_ac_refine_block(&mut bw, &coefs[bi], ss, se, ac_huff);
    }
    bw.finish();
}

/// Canonical per-block AC refinement encoder following T.81 §G.1.2.3.
///
/// The decoder walks zigzag positions counting zero-history slots (positions
/// where the accumulated coefficient is currently zero).  For each new-nonzero
/// event the encoder writes:
///
///   RS = (run_zeros << 4) | 1
///   sign bit  (1 = positive, 0 = negative)
///
/// Correction bits for pre-existing nonzeros (`|coef >> 1| != 0`) are emitted
/// INLINE as the decoder walks past them — one bit per such position, in
/// traversal order.  They are interleaved with zero-history counting, not
/// appended after the RS code.
///
/// After the last new-nonzero event the tail correction bits are emitted for
/// any remaining pre-existing nonzeros.
///
/// When no new nonzero exists the block is coded as EOB (RS=0x00) followed
/// by the correction bits for all pre-existing nonzeros in the band.
fn emit_ac_refine_block(
    bw: &mut BitWriter<'_>,
    block: &[i32; 64],
    ss: usize,
    se: usize,
    ac_huff: &HuffTable,
) {
    // Determine whether there are any "new nonzero" positions (|coef >> 1| == 0
    // but coef != 0) in the band.
    let has_new = (ss..=se).any(|k| {
        let v = block[ZIGZAG[k]];
        (v >> 1) == 0 && v != 0
    });

    if !has_new {
        // EOB: Huffman symbol, then correction bits for all pre-existing nonzeros.
        let eob = ac_huff.encode[0x00];
        bw.write_bits(eob.code as u32, eob.len as u32);
        for k in ss..=se {
            let v = block[ZIGZAG[k]];
            if (v >> 1) != 0 {
                bw.write_bits(v.unsigned_abs() & 1, 1);
            }
        }
        return;
    }

    // Walk the band emitting events. For each new-nonzero, we need to:
    //   1. Count zero-history slots (not pre-existing nonzeros) = `run`.
    //   2. Emit ZRL tokens if run >= 16.
    //   3. Emit RS = (remaining_run << 4) | 1.
    //   4. Emit sign bit.
    // Correction bits for pre-existing nonzeros are emitted INLINE as we
    // count the zero-history run — which means they must be emitted AFTER the
    // RS code but in the exact positions the decoder will encounter them while
    // walking. Since the decoder walks sequentially and reads corr bits for
    // pre-existing nonzeros as it goes, we simulate the walk and emit bits in
    // the same order.
    //
    // To handle this correctly: after emitting RS + sign, replay the walk from
    // the position of the previous anchor, emitting correction bits for any
    // pre-existing nonzeros encountered before the current new-nonzero.

    // We collect (new_nonzero_pos, sign) events in a first pass, then emit them
    // in a second pass that simulates the decoder walk.

    // Phase 1: collect new-nonzero positions and signs.
    let mut new_nz: Vec<(usize, u32)> = Vec::new(); // (zigzag_k, sign)
    for k in ss..=se {
        let v = block[ZIGZAG[k]];
        if (v >> 1) == 0 && v != 0 {
            let sign = if v > 0 { 1u32 } else { 0u32 };
            new_nz.push((k, sign));
        }
    }

    // Phase 2: simulate the decoder's walk to emit bits in the same order.
    //
    // KEY: after each new-nonzero event, the decoder sets that position to ±1.
    // For subsequent events, that position is now "nonzero" in the decoder's
    // view, so the encoder must NOT count it as a zero-history slot and MUST
    // emit a correction bit for it (correction bit for ±1 is always 0 since
    // the LSB of `1` that we "dropped" in the initial scan is 0 — the initial
    // scan had `1 >> 1 = 0`, so the dropped bit is 0).
    //
    // We track which positions have been newly set by previous events.
    let mut decoder_nonzero = [false; 64]; // tracks decoder's view of the band
                                           // Pre-populate with positions that were nonzero in the first pass.
    for k in ss..=se {
        let v = block[ZIGZAG[k]];
        if (v >> 1) != 0 {
            decoder_nonzero[k] = true;
        }
    }

    let mut k = ss;
    for (event_k, sign) in &new_nz {
        // Count zero-history slots from k up to (but not including) event_k,
        // using the decoder's current view of nonzero-ness.
        let mut run: u32 = 0;
        let mut pos = k;
        while pos < *event_k {
            if !decoder_nonzero[pos] {
                run += 1;
            }
            pos += 1;
        }

        // Emit ZRL tokens if run >= 16. During each ZRL, the decoder walks 16
        // zero-history positions and reads corr bits for pre-existing nonzeros.
        while run >= 16 {
            let zrl = ac_huff.encode[0xF0];
            bw.write_bits(zrl.code as u32, zrl.len as u32);
            // Walk 16 zero-history positions from k, emitting corr bits inline.
            let mut zrl_run = 0u32;
            while zrl_run < 16 {
                if decoder_nonzero[k] {
                    // Pre-existing nonzero: emit correction bit.
                    let v = block[ZIGZAG[k]];
                    bw.write_bits(v.unsigned_abs() & 1, 1);
                } else {
                    zrl_run += 1;
                }
                k += 1;
            }
            run -= 16;
        }

        // Emit RS = (run << 4) | 1 and sign bit.
        let rs = ((run as u8) << 4) | 1;
        let hc = ac_huff.encode[rs as usize];
        bw.write_bits(hc.code as u32, hc.len as u32);
        bw.write_bits(*sign, 1);

        // Emit correction bits for pre-existing nonzeros while the decoder
        // counts the remaining `run` zero-history slots up to event_k.
        while k < *event_k {
            if decoder_nonzero[k] {
                let v = block[ZIGZAG[k]];
                bw.write_bits(v.unsigned_abs() & 1, 1);
            }
            k += 1;
        }
        // Advance past the new nonzero itself and mark it as nonzero in the
        // decoder's view (its correction bit = 0 always, since the initial
        // scan encoded it as zero → dropped bit = 0).
        decoder_nonzero[*event_k] = true;
        k = event_k + 1;
    }

    // Tail: emit correction bits for all remaining (pre-existing) nonzeros.
    while k <= se {
        if decoder_nonzero[k] {
            // Was pre-existing nonzero: emit actual correction bit.
            // (Newly-set positions in decoder_nonzero are from previous events
            // and their correction bit = 0, but they appear before k so they're
            // never reached here.)
            let v = block[ZIGZAG[k]];
            // Check if this position was pre-existing (|v >> 1| != 0) or newly set.
            if (v >> 1) != 0 {
                bw.write_bits(v.unsigned_abs() & 1, 1);
            } else {
                // Newly set by a previous event (|v| == 1): correction bit = 0.
                // This shouldn't happen here since k > last event_k.
                bw.write_bits(0, 1);
            }
        }
        k += 1;
    }
}

fn write_length_prefix(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    let len = (payload.len() + 2) as u16;
    out.push(0xFF);
    out.push(marker);
    out.push((len >> 8) as u8);
    out.push(len as u8);
    out.extend_from_slice(payload);
}

fn write_jfif_app0(out: &mut Vec<u8>) {
    // "JFIF\0" + version 1.01 + density unit 0 (aspect) + 1x1 density + 0 thumbnail.
    let payload = [
        b'J', b'F', b'I', b'F', 0, // identifier
        1, 1, // version
        0, // units
        0, 1, // Xdensity
        0, 1, // Ydensity
        0, 0, // thumbnail w, h
    ];
    write_length_prefix(out, markers::APP0, &payload);
}

fn write_dqt(out: &mut Vec<u8>, table_id: u8, nat_order: &[u16; 64]) {
    let mut payload = Vec::with_capacity(1 + 64);
    payload.push(table_id & 0x0F); // precision=0, id=table_id
    for k in 0..64 {
        payload.push(nat_order[ZIGZAG[k]].min(255) as u8);
    }
    write_length_prefix(out, markers::DQT, &payload);
}

fn write_sof0(out: &mut Vec<u8>, width: u16, height: u16, h: u8, v: u8) {
    let mut payload = Vec::with_capacity(8 + 9);
    payload.push(8); // precision
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(3); // components
                     // Y
    payload.push(1);
    payload.push((h << 4) | v);
    payload.push(0);
    // Cb
    payload.push(2);
    payload.push(0x11);
    payload.push(1);
    // Cr
    payload.push(3);
    payload.push(0x11);
    payload.push(1);
    write_length_prefix(out, markers::SOF0, &payload);
}

fn write_dht(out: &mut Vec<u8>, class: u8, id: u8, bits: &[u8; 16], values: &[u8]) {
    let mut payload = Vec::with_capacity(1 + 16 + values.len());
    payload.push(((class & 0x01) << 4) | (id & 0x0F));
    payload.extend_from_slice(bits);
    payload.extend_from_slice(values);
    write_length_prefix(out, markers::DHT, &payload);
}

fn write_dri(out: &mut Vec<u8>, restart_interval: u16) {
    out.push(0xFF);
    out.push(markers::DRI);
    out.push(0x00);
    out.push(0x04);
    out.extend_from_slice(&restart_interval.to_be_bytes());
}

fn write_sos(out: &mut Vec<u8>) {
    let payload: [u8; 10] = [
        3, // components
        1, 0x00, // Y uses DC=0 AC=0
        2, 0x11, // Cb uses DC=1 AC=1
        3, 0x11, // Cr uses DC=1 AC=1
        0, 63, 0, // Ss, Se, Ah|Al
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

#[allow(clippy::too_many_arguments)]
fn write_scan(
    out: &mut Vec<u8>,
    frame: &VideoFrame,
    pix: PixelFormat,
    width: usize,
    height: usize,
    h_factor: u8,
    v_factor: u8,
    luma_q: &[u16; 64],
    chroma_q: &[u16; 64],
    huff: &DefaultHuffman,
    restart_interval: u16,
) -> Result<()> {
    let mcu_w_px = 8 * h_factor as usize;
    let mcu_h_px = 8 * v_factor as usize;
    let mcus_x = width.div_ceil(mcu_w_px);
    let mcus_y = height.div_ceil(mcu_h_px);
    let total_mcus = mcus_x.saturating_mul(mcus_y);

    let y_plane = &frame.planes[0];
    let cb_plane = &frame.planes[1];
    let cr_plane = &frame.planes[2];

    let (c_w, c_h) = match pix {
        PixelFormat::Yuv444P => (width, height),
        PixelFormat::Yuv422P => (width.div_ceil(2), height),
        PixelFormat::Yuv420P => (width.div_ceil(2), height.div_ceil(2)),
        _ => unreachable!(),
    };

    let mut prev_dc_y: i32 = 0;
    let mut prev_dc_cb: i32 = 0;
    let mut prev_dc_cr: i32 = 0;

    let ri = restart_interval as usize;
    let mut rst_counter: u8 = 0;
    let mut mcus_since_restart: usize = 0;
    let mut mcu_index: usize = 0;

    let mut bw = BitWriter::new(out);

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // Luma blocks.
            for by in 0..v_factor as usize {
                for bx in 0..h_factor as usize {
                    let x0 = mx * mcu_w_px + bx * 8;
                    let y0 = my * mcu_h_px + by * 8;
                    let mut blk = [0.0f32; 64];
                    fill_block(
                        &mut blk,
                        &y_plane.data,
                        y_plane.stride,
                        width,
                        height,
                        x0,
                        y0,
                    );
                    encode_block(
                        &mut bw,
                        &mut blk,
                        luma_q,
                        &mut prev_dc_y,
                        &huff.luma_dc,
                        &huff.luma_ac,
                    );
                }
            }
            // Chroma: one block per component per MCU.
            let cb_x0 = mx * 8;
            let cb_y0 = my * 8;
            let mut blk_cb = [0.0f32; 64];
            fill_block(
                &mut blk_cb,
                &cb_plane.data,
                cb_plane.stride,
                c_w,
                c_h,
                cb_x0,
                cb_y0,
            );
            encode_block(
                &mut bw,
                &mut blk_cb,
                chroma_q,
                &mut prev_dc_cb,
                &huff.chroma_dc,
                &huff.chroma_ac,
            );
            let mut blk_cr = [0.0f32; 64];
            fill_block(
                &mut blk_cr,
                &cr_plane.data,
                cr_plane.stride,
                c_w,
                c_h,
                cb_x0,
                cb_y0,
            );
            encode_block(
                &mut bw,
                &mut blk_cr,
                chroma_q,
                &mut prev_dc_cr,
                &huff.chroma_dc,
                &huff.chroma_ac,
            );

            mcu_index += 1;
            mcus_since_restart += 1;

            // Emit a restart marker after every `ri` MCUs, but never after
            // the very last MCU of the image (decoders expect EOI next).
            if ri != 0 && mcus_since_restart == ri && mcu_index < total_mcus {
                bw.flush_to_byte();
                bw.emit_raw_marker(markers::RST0 + (rst_counter & 0x07));
                rst_counter = rst_counter.wrapping_add(1);
                prev_dc_y = 0;
                prev_dc_cb = 0;
                prev_dc_cr = 0;
                mcus_since_restart = 0;
            }
        }
    }

    bw.finish();
    Ok(())
}

/// Fill an 8×8 f32 block from a plane, repeating edge pixels for any MCU
/// blocks that extend past the picture boundary. Subtracts 128 (level shift).
fn fill_block(
    dst: &mut [f32; 64],
    plane: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
) {
    for j in 0..8 {
        let y = (y0 + j).min(h.saturating_sub(1));
        for i in 0..8 {
            let x = (x0 + i).min(w.saturating_sub(1));
            let v = plane[y * stride + x] as i32;
            dst[j * 8 + i] = (v - 128) as f32;
        }
    }
}

fn encode_block(
    bw: &mut BitWriter<'_>,
    block: &mut [f32; 64],
    quant: &[u16; 64],
    prev_dc: &mut i32,
    dc_huff: &HuffTable,
    ac_huff: &HuffTable,
) {
    fdct8x8(block);
    // Quantise.
    let mut q = [0i32; 64];
    for k in 0..64 {
        let v = block[k] / quant[k] as f32;
        q[k] = if v >= 0.0 {
            (v + 0.5) as i32
        } else {
            -((-v + 0.5) as i32)
        };
    }
    // DC.
    let dc_diff = q[0] - *prev_dc;
    *prev_dc = q[0];
    let (size, bits) = category(dc_diff);
    let hc = dc_huff.encode[size as usize];
    bw.write_bits(hc.code as u32, hc.len as u32);
    if size > 0 {
        bw.write_bits(bits, size as u32);
    }
    // AC in zigzag order.
    let mut run: u32 = 0;
    for k in 1..64 {
        let v = q[ZIGZAG[k]];
        if v == 0 {
            run += 1;
        } else {
            while run >= 16 {
                // ZRL.
                let zc = ac_huff.encode[0xF0];
                bw.write_bits(zc.code as u32, zc.len as u32);
                run -= 16;
            }
            let (sz, bv) = category(v);
            let rs = ((run as u8) << 4) | sz;
            let ac = ac_huff.encode[rs as usize];
            bw.write_bits(ac.code as u32, ac.len as u32);
            if sz > 0 {
                bw.write_bits(bv, sz as u32);
            }
            run = 0;
        }
    }
    if run > 0 {
        let eob = ac_huff.encode[0x00];
        bw.write_bits(eob.code as u32, eob.len as u32);
    }
}

/// JPEG category: `(size, bits)` where `size` = min bits to hold |v|, and
/// `bits` is the value encoded per Annex F (positive → plain, negative →
/// bitwise complement of |v| fitted to `size` bits, equivalently
/// `v + (2^size - 1)`).
fn category(v: i32) -> (u8, u32) {
    if v == 0 {
        return (0, 0);
    }
    let abs = v.unsigned_abs();
    let size = 32 - abs.leading_zeros();
    debug_assert!(size <= 16);
    let bits = if v > 0 {
        abs
    } else {
        (1u32 << size).wrapping_sub(1).wrapping_add(v as u32)
    };
    (size as u8, bits)
}

// ---- Bit writer with 0xFF stuffing --------------------------------------

struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    buf: u32,
    nbits: u32,
}

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            buf: 0,
            nbits: 0,
        }
    }

    fn write_bits(&mut self, value: u32, len: u32) {
        if len == 0 {
            return;
        }
        // Mask value to `len` bits.
        let v = value & ((1u32 << len) - 1);
        self.buf = (self.buf << len) | v;
        self.nbits += len;
        while self.nbits >= 8 {
            self.nbits -= 8;
            let b = ((self.buf >> self.nbits) & 0xFF) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0x00);
            }
        }
    }

    fn finish(&mut self) {
        self.flush_to_byte();
    }

    /// Byte-align the bitstream by padding any partial-byte with 1-bits
    /// (T.81 Annex F.1.2.3), then emit the completed byte with 0xFF
    /// stuffing. Used before writing `RSTn` markers.
    fn flush_to_byte(&mut self) {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.buf = (self.buf << pad) | ((1u32 << pad) - 1);
            self.nbits = 0;
            let b = (self.buf & 0xFF) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0x00);
            }
        }
    }

    /// Emit a raw two-byte `FF xx` marker (no 0-stuffing). Caller must
    /// have byte-aligned the stream first via [`Self::flush_to_byte`].
    fn emit_raw_marker(&mut self, marker: u8) {
        debug_assert_eq!(self.nbits, 0);
        self.out.push(0xFF);
        self.out.push(marker);
    }
}

/// Test-only: emit a YUV 4:4:4 / 4:2:2 / 4:2:0 JPEG using **non-interleaved**
/// scans — one SOS per component. Exercises the decoder's non-interleaved
/// code path. Output is a conformant baseline JPEG (same headers as
/// [`encode_jpeg`]) with 3 SOS segments instead of 1.
#[cfg(test)]
pub(crate) fn encode_jpeg_non_interleaved(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
) -> Result<Vec<u8>> {
    let width = width as usize;
    let height = height as usize;
    let (h_factor, v_factor) = match pix {
        PixelFormat::Yuv444P => (1u8, 1u8),
        PixelFormat::Yuv422P => (2, 1),
        PixelFormat::Yuv420P => (2, 2),
        _ => {
            return Err(Error::unsupported(
                "non-interleaved helper: unsupported format",
            ))
        }
    };
    if frame.planes.len() != 3 {
        return Err(Error::invalid("non-interleaved helper: expected 3 planes"));
    }

    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let chroma_q = scale_for_quality(&DEFAULT_CHROMA_Q50, quality);
    let huff = DefaultHuffman::build()?;

    let mut out: Vec<u8> = Vec::with_capacity(16_384);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_dqt(&mut out, 0, &luma_q);
    write_dqt(&mut out, 1, &chroma_q);
    write_sof0(&mut out, width as u16, height as u16, h_factor, v_factor);
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_dht(&mut out, 0, 1, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    write_dht(&mut out, 1, 1, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);

    // Per-component block counts cover the full MCU grid (padded).
    let mcu_w_px = 8 * h_factor as usize;
    let mcu_h_px = 8 * v_factor as usize;
    let mcus_x = width.div_ceil(mcu_w_px);
    let mcus_y = height.div_ceil(mcu_h_px);

    let (c_w, c_h) = match pix {
        PixelFormat::Yuv444P => (width, height),
        PixelFormat::Yuv422P => (width.div_ceil(2), height),
        PixelFormat::Yuv420P => (width.div_ceil(2), height.div_ceil(2)),
        _ => unreachable!(),
    };

    // Y scan: blocks per row = mcus_x * h_factor, rows = mcus_y * v_factor.
    write_non_interleaved_sos(&mut out, 1, 0, 0);
    write_component_scan(
        &mut out,
        &frame.planes[0].data,
        frame.planes[0].stride,
        width,
        height,
        mcus_x * h_factor as usize,
        mcus_y * v_factor as usize,
        &luma_q,
        &huff.luma_dc,
        &huff.luma_ac,
    );

    // Cb scan.
    write_non_interleaved_sos(&mut out, 2, 1, 1);
    write_component_scan(
        &mut out,
        &frame.planes[1].data,
        frame.planes[1].stride,
        c_w,
        c_h,
        mcus_x,
        mcus_y,
        &chroma_q,
        &huff.chroma_dc,
        &huff.chroma_ac,
    );

    // Cr scan.
    write_non_interleaved_sos(&mut out, 3, 1, 1);
    write_component_scan(
        &mut out,
        &frame.planes[2].data,
        frame.planes[2].stride,
        c_w,
        c_h,
        mcus_x,
        mcus_y,
        &chroma_q,
        &huff.chroma_dc,
        &huff.chroma_ac,
    );

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

#[cfg(test)]
fn write_non_interleaved_sos(out: &mut Vec<u8>, comp_id: u8, dc_table: u8, ac_table: u8) {
    let payload = [
        1, // Ns
        comp_id,
        ((dc_table & 0x0F) << 4) | (ac_table & 0x0F),
        0,
        63,
        0, // Ss, Se, Ah|Al
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn write_component_scan(
    out: &mut Vec<u8>,
    plane: &[u8],
    plane_stride: usize,
    plane_w: usize,
    plane_h: usize,
    blocks_x: usize,
    blocks_y: usize,
    quant: &[u16; 64],
    dc_huff: &HuffTable,
    ac_huff: &HuffTable,
) {
    let mut bw = BitWriter::new(out);
    let mut prev_dc: i32 = 0;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut blk = [0.0f32; 64];
            fill_block(
                &mut blk,
                plane,
                plane_stride,
                plane_w,
                plane_h,
                bx * 8,
                by * 8,
            );
            encode_block(&mut bw, &mut blk, quant, &mut prev_dc, dc_huff, ac_huff);
        }
    }
    bw.finish();
}

/// Emit a 4-component JPEG (1:1:1:1 sampling, all components at full
/// resolution) from four raw planar component buffers.
///
/// This is the per-plane "back-end" entry point used by both the
/// integrated [`MjpegEncoder`](crate::registry::MjpegEncoder) and the
/// packed-CMYK [`encode_jpeg_cmyk`] convenience wrapper. Most callers
/// should reach for `encode_jpeg_cmyk` instead, which accepts the same
/// `[C, M, Y, K]`-packed buffer the decoder produces.
///
/// Parameters:
/// * `planes` — four equal-size sample buffers, one per component, in
///   the order the caller wants the SOS scan to see them. For plain
///   CMYK that is `[C, M, Y, K]`; for Adobe YCCK with
///   `adobe_transform = Some(2)` it is `[Y, Cb, Cr, K]`.
/// * `plane_strides` — bytes per row for each plane (typically `width`).
/// * `quality` — IJG quality factor in the same `1..=100` range the
///   YUV encoder uses (clamped internally to `1..=100`). Q ≥ 90 keeps
///   visible quantisation artefacts well within the per-component
///   `PSNR ≥ 30 dB` floor that the round-trip tests enforce.
/// * `adobe_transform`:
///   * `None` — no APP14 segment; samples written verbatim. The
///     [`crate::decoder::decode_jpeg`] side treats the result as plain
///     "regular" CMYK and round-trips it unchanged.
///   * `Some(0)` — Adobe CMYK convention. Every sample is inverted on
///     the wire (`store = 255 − input`) and the decoder un-inverts it
///     on output, so the function takes the same `[C, M, Y, K]` input
///     as the no-APP14 path.
///   * `Some(2)` — Adobe YCCK. Components 0..2 are written verbatim
///     (interpreted as Y/Cb/Cr); component 3 (K) is inverted on the
///     wire. The decoder converts the YCbCr triple back to RGB then
///     CMY (BT.601, full-range) and flips K to recover the original
///     CMYK quadruple.
///
/// The bitstream layout — DQT(0) + DQT(1), SOF0 with `Nf = 4` and
/// `H_i = V_i = 1` for every component, four Annex K Huffman tables
/// (luma DC/AC at `Th = 0`, chroma DC/AC at `Th = 1`), one
/// interleaved SOS scan with `Ss = 0, Se = 63, Ah = Al = 0` — exactly
/// matches the per-component layout the decoder's CMYK path expects.
///
/// Errors: `Error::InvalidData` if any plane is shorter than
/// `plane_strides[i] * height`.
#[allow(clippy::too_many_arguments)]
pub fn encode_jpeg_cmyk_1111(
    width: u32,
    height: u32,
    planes: &[&[u8]; 4],
    plane_strides: &[usize; 4],
    quality: u8,
    adobe_transform: Option<u8>,
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    for (i, p) in planes.iter().enumerate() {
        if p.len() < plane_strides[i] * h {
            return Err(Error::invalid(
                "cmyk helper: plane shorter than height*stride",
            ));
        }
    }

    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let chroma_q = scale_for_quality(&DEFAULT_CHROMA_Q50, quality);
    let huff = DefaultHuffman::build()?;

    // For Adobe conventions we invert the stored samples (so the decoder
    // can un-invert them and match the original inputs). For transform=2
    // we also invert K only.
    let needs_invert_all = matches!(adobe_transform, Some(0));
    let needs_invert_k_only = matches!(adobe_transform, Some(2));
    let maybe_invert = |plane_idx: usize, samples: &[u8]| -> Vec<u8> {
        let must = needs_invert_all || (needs_invert_k_only && plane_idx == 3);
        if !must {
            return samples.to_vec();
        }
        samples.iter().map(|&b| 255 - b).collect()
    };
    let owned: [Vec<u8>; 4] = [
        maybe_invert(0, planes[0]),
        maybe_invert(1, planes[1]),
        maybe_invert(2, planes[2]),
        maybe_invert(3, planes[3]),
    ];

    let mut out: Vec<u8> = Vec::with_capacity(16_384);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    if let Some(tx) = adobe_transform {
        write_adobe_app14(&mut out, tx);
    }
    // Two quant tables suffice (one "luma-ish", one "chroma-ish"):
    //   component 0 (Y or C)  → qt 0
    //   components 1,2,3      → qt 1
    write_dqt(&mut out, 0, &luma_q);
    write_dqt(&mut out, 1, &chroma_q);
    write_sof0_4comp(&mut out, w as u16, h as u16);
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_dht(&mut out, 0, 1, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    write_dht(&mut out, 1, 1, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);
    write_sos_4comp(&mut out);

    // One interleaved scan, one block per component per MCU (1:1:1:1).
    let mcus_x = w.div_ceil(8);
    let mcus_y = h.div_ceil(8);
    let mut bw = BitWriter::new(&mut out);
    let mut prev_dc = [0i32; 4];
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            for ci in 0..4 {
                let mut blk = [0.0f32; 64];
                fill_block(
                    &mut blk,
                    &owned[ci],
                    plane_strides[ci],
                    w,
                    h,
                    mx * 8,
                    my * 8,
                );
                let (qt, dc_t, ac_t) = if ci == 0 {
                    (&luma_q, &huff.luma_dc, &huff.luma_ac)
                } else {
                    (&chroma_q, &huff.chroma_dc, &huff.chroma_ac)
                };
                encode_block(&mut bw, &mut blk, qt, &mut prev_dc[ci], dc_t, ac_t);
            }
        }
    }
    bw.finish();

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

fn write_adobe_app14(out: &mut Vec<u8>, transform: u8) {
    // "Adobe" + version 100 + flags0/flags1 (0) + transform.
    let payload = [
        b'A', b'd', b'o', b'b', b'e', //
        0, 100, // DCTEncodeVersion = 100
        0, 0, // APP14Flags0
        0, 0, // APP14Flags1
        transform,
    ];
    write_length_prefix(out, 0xEE, &payload);
}

fn write_sof0_4comp(out: &mut Vec<u8>, width: u16, height: u16) {
    // 1:1:1:1 sampling, qt 0 for component 1, qt 1 for the rest.
    let mut payload = Vec::with_capacity(8 + 12);
    payload.push(8); // precision
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(4); // Nf
    for (id, qt) in [(1, 0u8), (2, 1), (3, 1), (4, 1)] {
        payload.push(id);
        payload.push(0x11); // H=1 V=1
        payload.push(qt);
    }
    write_length_prefix(out, markers::SOF0, &payload);
}

fn write_sos_4comp(out: &mut Vec<u8>) {
    let payload: [u8; 12] = [
        4, // Ns
        1, 0x00, // comp 1 → DC=0 AC=0
        2, 0x11, // comp 2 → DC=1 AC=1
        3, 0x11, // comp 3 → DC=1 AC=1
        4, 0x11, // comp 4 → DC=1 AC=1
        0, 63, 0, // Ss, Se, Ah|Al
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

fn write_sof2_4comp(out: &mut Vec<u8>, width: u16, height: u16) {
    // Progressive (SOF2) variant of `write_sof0_4comp`: identical 4-component
    // 1:1:1:1 layout, only the marker byte changes. Component 1 binds quant
    // table 0; the remaining three share table 1 (mirroring the existing
    // baseline 4-component encoder's `(luma_q, chroma_q)` policy).
    let mut payload = Vec::with_capacity(8 + 12);
    payload.push(8); // precision
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(4); // Nf
    for (id, qt) in [(1, 0u8), (2, 1), (3, 1), (4, 1)] {
        payload.push(id);
        payload.push(0x11); // H=1 V=1
        payload.push(qt);
    }
    write_length_prefix(out, markers::SOF2, &payload);
}

fn write_sos_progressive_dc_4comp_interleaved(out: &mut Vec<u8>) {
    // Interleaved DC scan over all four 1:1:1:1 components. Td selects
    // DC=0 for component 1 (matching the SOS used by the baseline
    // 4-component encoder, and the Annex K luma DC table emitted at
    // (Tc=0, Th=0)) and DC=1 for components 2/3/4 (the Annex K chroma
    // DC table emitted at (Tc=0, Th=1)). Ss=Se=0 selects the DC band;
    // Ah=Al=0 (spectral-selection only, no SA point-transform).
    let payload: [u8; 12] = [
        4, // Ns
        1, 0x00, // comp 1 → DC=0
        2, 0x10, // comp 2 → DC=1
        3, 0x10, // comp 3 → DC=1
        4, 0x10, // comp 4 → DC=1
        0, 0, 0x00, // Ss, Se, Ah|Al
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

/// Emit a 4-component (CMYK / YCCK) **progressive** (SOF2) JPEG from
/// four raw planar component buffers.
///
/// Companion to [`encode_jpeg_cmyk_1111`] using SOF2 instead of SOF0,
/// with the same spectral-selection-only scan decomposition the
/// three-component progressive YUV helpers use: one interleaved DC
/// scan (`Ss = Se = 0, Ah = Al = 0`) followed by per-component AC
/// bands `[1..=5]` then `[6..=63]` for each of the four components.
/// Total scans: 1 + 4 + 4 = 9.
///
/// All four components are declared `H_i = V_i = 1` so the MCU equals
/// one data unit per component. Component 1 binds quant table 0;
/// components 2/3/4 share quant table 1 (same `(luma_q, chroma_q)`
/// policy as the baseline 4-component helper).
///
/// The `adobe_transform` argument follows the same semantics as
/// [`encode_jpeg_cmyk_1111`]:
///   * `None`     — no APP14 segment, samples passed through unchanged.
///   * `Some(0)`  — Adobe CMYK: every component inverted on the wire.
///   * `Some(2)`  — Adobe YCCK: K plane inverted; components 0..2 carry
///     YCbCr-encoded data, decoder converts back via BT.601.
///
/// Most callers should reach for the packed-buffer convenience wrapper
/// [`encode_jpeg_cmyk_progressive`] instead; this entry point is the
/// per-plane back-end used by both that wrapper and the registry
/// encoder.
///
/// Errors: `Error::InvalidData` if any plane is shorter than
/// `plane_strides[i] * height`.
#[allow(clippy::too_many_arguments)]
pub fn encode_jpeg_progressive_cmyk_1111(
    width: u32,
    height: u32,
    planes: &[&[u8]; 4],
    plane_strides: &[usize; 4],
    quality: u8,
    adobe_transform: Option<u8>,
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    for (i, p) in planes.iter().enumerate() {
        if p.len() < plane_strides[i] * h {
            return Err(Error::invalid(
                "cmyk progressive helper: plane shorter than height*stride",
            ));
        }
    }

    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let chroma_q = scale_for_quality(&DEFAULT_CHROMA_Q50, quality);
    let huff = DefaultHuffman::build()?;

    let needs_invert_all = matches!(adobe_transform, Some(0));
    let needs_invert_k_only = matches!(adobe_transform, Some(2));
    let maybe_invert = |plane_idx: usize, samples: &[u8]| -> Vec<u8> {
        let must = needs_invert_all || (needs_invert_k_only && plane_idx == 3);
        if !must {
            return samples.to_vec();
        }
        samples.iter().map(|&b| 255 - b).collect()
    };
    let owned: [Vec<u8>; 4] = [
        maybe_invert(0, planes[0]),
        maybe_invert(1, planes[1]),
        maybe_invert(2, planes[2]),
        maybe_invert(3, planes[3]),
    ];

    // 1:1:1:1 sampling → one block per component per MCU; mcus_x/y is the
    // ceiling-divided block grid.
    let mcus_x = w.div_ceil(8);
    let mcus_y = h.div_ceil(8);
    let blocks = mcus_x * mcus_y;

    // Per-component coefficient grid (component 1 uses luma_q; 2/3/4 use chroma_q).
    let qts: [&[u16; 64]; 4] = [&luma_q, &chroma_q, &chroma_q, &chroma_q];
    let mut comp_coefs: [Vec<[i32; 64]>; 4] = [
        vec![[0i32; 64]; blocks],
        vec![[0i32; 64]; blocks],
        vec![[0i32; 64]; blocks],
        vec![[0i32; 64]; blocks],
    ];
    for ci in 0..4 {
        fill_coef_grid(
            &mut comp_coefs[ci],
            &owned[ci],
            plane_strides[ci],
            w,
            h,
            mcus_x,
            mcus_y,
            qts[ci],
        );
    }

    // Header.
    let mut out: Vec<u8> = Vec::with_capacity(16_384);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    if let Some(tx) = adobe_transform {
        write_adobe_app14(&mut out, tx);
    }
    write_dqt(&mut out, 0, &luma_q);
    write_dqt(&mut out, 1, &chroma_q);
    write_sof2_4comp(&mut out, w as u16, h as u16);
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_dht(&mut out, 0, 1, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    write_dht(&mut out, 1, 1, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);

    // Scan 1 — interleaved DC across all four components (Ss=Se=0, Ah=Al=0).
    // Component 1 uses DC table 0 (luma); components 2/3/4 use DC table 1
    // (chroma).
    write_sos_progressive_dc_4comp_interleaved(&mut out);
    {
        let mut bw = BitWriter::new(&mut out);
        let mut prev_dc = [0i32; 4];
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                let bi = my * mcus_x + mx;
                for ci in 0..4 {
                    let dc_t = if ci == 0 {
                        &huff.luma_dc
                    } else {
                        &huff.chroma_dc
                    };
                    encode_dc(&mut bw, comp_coefs[ci][bi][0], &mut prev_dc[ci], dc_t);
                }
            }
        }
        bw.finish();
    }

    // Scans 2..=9 — per-component AC bands. Component 1 uses AC table 0
    // (luma); components 2/3/4 use AC table 1 (chroma). Two bands per
    // component: low [1..=5] then high [6..=63].
    for &(ss, se) in &[(1u8, 5u8), (6u8, 63u8)] {
        for ci in 0..4 {
            let (comp_id, ac_table, ac_huff) = if ci == 0 {
                (1u8, 0u8, &huff.luma_ac)
            } else {
                ((ci as u8) + 1, 1u8, &huff.chroma_ac)
            };
            write_sos_progressive_ac(&mut out, comp_id, ac_table, ss, se);
            write_ac_scan(
                &mut out,
                &comp_coefs[ci],
                blocks,
                ac_huff,
                ss as usize,
                se as usize,
            );
        }
    }

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

// ---- packed-CMYK convenience wrappers ----------------------------------
//
// These accept the same packed `[C, M, Y, K]` interleaved buffer the
// decoder produces (one plane, `stride` bytes per row, 4 bytes per pixel).
// They de-interleave the source into four owned planes once, then
// delegate to the planar back-ends above.

/// Decompose a packed 4-byte-per-pixel CMYK buffer into four owned
/// per-component planes. Each output plane is `width * height` bytes
/// with `stride = width`. `packed_stride` is the number of bytes per
/// row in the source buffer; the function errors out if
/// `packed_stride < width * 4` or the buffer is shorter than
/// `packed_stride * height`.
fn unpack_cmyk(
    width: u32,
    height: u32,
    packed: &[u8],
    packed_stride: usize,
) -> Result<[Vec<u8>; 4]> {
    let w = width as usize;
    let h = height as usize;
    if packed_stride < w * 4 {
        return Err(Error::invalid(
            "encode_jpeg_cmyk: packed stride must be at least width * 4",
        ));
    }
    if packed.len() < packed_stride * h {
        return Err(Error::invalid(
            "encode_jpeg_cmyk: packed buffer shorter than height * stride",
        ));
    }
    let mut c = vec![0u8; w * h];
    let mut m = vec![0u8; w * h];
    let mut y = vec![0u8; w * h];
    let mut k = vec![0u8; w * h];
    for j in 0..h {
        let row = &packed[j * packed_stride..j * packed_stride + w * 4];
        for i in 0..w {
            let o = i * 4;
            c[j * w + i] = row[o];
            m[j * w + i] = row[o + 1];
            y[j * w + i] = row[o + 2];
            k[j * w + i] = row[o + 3];
        }
    }
    Ok([c, m, y, k])
}

/// Emit a 4-component CMYK / YCCK baseline (SOF0) JPEG from a packed
/// `[C, M, Y, K]` interleaved input buffer.
///
/// The buffer layout matches what
/// [`crate::decoder::decode_jpeg`] produces for a 4-component JPEG
/// (one packed plane, `stride` bytes per row, 4 bytes per pixel in
/// `C, M, Y, K` order), so round-tripping a decoded CMYK frame back
/// into a new JPEG is a single call.
///
/// `quality` is the IJG quality factor (`1..=100`, clamped). All four
/// components are coded at full resolution (`H = V = 1` each); the
/// chroma quant table is shared by components 2/3/4.
///
/// `adobe_transform` controls the Adobe APP14 colour-transform marker
/// emitted between SOI and DQT:
///   * `None` — no APP14, samples passed through unchanged ("regular"
///     CMYK).
///   * `Some(0)` — Adobe CMYK: every sample is inverted on the wire.
///     The decoder un-inverts the result, so the caller still passes
///     the same `[C, M, Y, K]` buffer it would pass with `None`.
///   * `Some(2)` — Adobe YCCK: the function interprets the four
///     packed components as `[Y, Cb, Cr, K]` rather than `[C, M, Y, K]`
///     and inverts only the K plane on the wire. Callers wanting a
///     YCCK output from CMYK input should perform the BT.601 CMY→RGB→
///     YCbCr conversion themselves before calling.
///
/// Errors:
/// * `Error::InvalidData` if `stride < width * 4` or the buffer is
///   shorter than `height * stride`.
/// * `Error::InvalidData` if `adobe_transform` is `Some(t)` with
///   `t` not equal to `0` or `2` — only those two transform values
///   round-trip through this crate's decoder.
pub fn encode_jpeg_cmyk(
    width: u32,
    height: u32,
    packed: &[u8],
    stride: usize,
    quality: u8,
    adobe_transform: Option<u8>,
) -> Result<Vec<u8>> {
    if let Some(t) = adobe_transform {
        if t != 0 && t != 2 {
            return Err(Error::invalid(
                "encode_jpeg_cmyk: adobe_transform must be 0 (CMYK) or 2 (YCCK)",
            ));
        }
    }
    let planes = unpack_cmyk(width, height, packed, stride)?;
    let refs: [&[u8]; 4] = [&planes[0], &planes[1], &planes[2], &planes[3]];
    let w = width as usize;
    let strides = [w; 4];
    encode_jpeg_cmyk_1111(width, height, &refs, &strides, quality, adobe_transform)
}

/// Emit a 4-component CMYK / YCCK **progressive** (SOF2) JPEG from a
/// packed `[C, M, Y, K]` interleaved input buffer.
///
/// SOF2 companion of [`encode_jpeg_cmyk`] using the same packed
/// buffer convention. The scan decomposition is the 9-segment
/// spectral-selection-only layout described on
/// [`encode_jpeg_progressive_cmyk_1111`] (one interleaved DC scan plus
/// per-component AC bands `[1..=5]` and `[6..=63]`, `Ah = Al = 0`
/// throughout).
///
/// Errors mirror [`encode_jpeg_cmyk`].
pub fn encode_jpeg_cmyk_progressive(
    width: u32,
    height: u32,
    packed: &[u8],
    stride: usize,
    quality: u8,
    adobe_transform: Option<u8>,
) -> Result<Vec<u8>> {
    if let Some(t) = adobe_transform {
        if t != 0 && t != 2 {
            return Err(Error::invalid(
                "encode_jpeg_cmyk_progressive: adobe_transform must be 0 (CMYK) or 2 (YCCK)",
            ));
        }
    }
    let planes = unpack_cmyk(width, height, packed, stride)?;
    let refs: [&[u8]; 4] = [&planes[0], &planes[1], &planes[2], &planes[3]];
    let w = width as usize;
    let strides = [w; 4];
    encode_jpeg_progressive_cmyk_1111(width, height, &refs, &strides, quality, adobe_transform)
}

/// Test-only: emit a single-component, 12-bit precision baseline JPEG.
/// `samples` carries u16 values in `[0, 4095]`. Caller must keep inputs
/// moderate enough that per-block DC/AC Huffman categories stay ≤ 11 —
/// we reuse the 8-bit Annex K tables rather than carrying a separate
/// 12-bit table set.
#[cfg(test)]
pub(crate) fn encode_grayscale_jpeg_12bit(
    width: u32,
    height: u32,
    samples: &[u16],
    stride: usize,
    quality: u8,
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if samples.len() < stride * h {
        return Err(Error::invalid(
            "12-bit gray helper: samples buffer too short",
        ));
    }
    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let huff = DefaultHuffman::build()?;

    let mut out: Vec<u8> = Vec::with_capacity(16_384);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_dqt(&mut out, 0, &luma_q);
    write_sof_grayscale_12bit(&mut out, w as u16, h as u16);
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_sos_grayscale(&mut out);

    let mcus_x = w.div_ceil(8);
    let mcus_y = h.div_ceil(8);
    let mut bw = BitWriter::new(&mut out);
    let mut prev_dc: i32 = 0;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            let mut blk = [0.0f32; 64];
            fill_block_u16_levelshift_2048(&mut blk, samples, stride, w, h, mx * 8, my * 8);
            encode_block(
                &mut bw,
                &mut blk,
                &luma_q,
                &mut prev_dc,
                &huff.luma_dc,
                &huff.luma_ac,
            );
        }
    }
    bw.finish();

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

#[cfg(test)]
fn fill_block_u16_levelshift_2048(
    dst: &mut [f32; 64],
    plane: &[u16],
    stride: usize,
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
) {
    for j in 0..8 {
        let y = (y0 + j).min(h.saturating_sub(1));
        for i in 0..8 {
            let x = (x0 + i).min(w.saturating_sub(1));
            let v = plane[y * stride + x] as i32;
            dst[j * 8 + i] = (v - 2048) as f32;
        }
    }
}

#[cfg(test)]
fn write_sof_grayscale_12bit(out: &mut Vec<u8>, width: u16, height: u16) {
    let mut payload = Vec::with_capacity(8 + 3);
    payload.push(12); // precision
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(1); // Nf
    payload.push(1); // component id
    payload.push(0x11); // H=1 V=1
    payload.push(0); // qt = 0
    write_length_prefix(out, markers::SOF0, &payload);
}

#[cfg(test)]
fn write_sos_grayscale(out: &mut Vec<u8>) {
    let payload: [u8; 6] = [
        1, // Ns
        1, 0x00, // comp 1 → DC=0 AC=0
        0, 63, 0, // Ss, Se, Ah|Al
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

/// Test-only: emit a three-component (YUV) 12-bit precision extended-sequential
/// JPEG (SOF1). `y_samples` / `cb_samples` / `cr_samples` carry u16 values in
/// `[0, 4095]`. Caller must keep inputs moderate enough that per-block DC/AC
/// Huffman categories stay ≤ 11 — we reuse the 8-bit Annex K luma/chroma
/// Huffman tables rather than carrying a separate 12-bit table set. The
/// `(h_factor, v_factor)` argument controls the luma sampling factor:
///   * `(1, 1)` → 4:4:4 (`Yuv444P12Le`)
///   * `(2, 1)` → 4:2:2 (`Yuv422P12Le`)
///   * `(2, 2)` → 4:2:0 (`Yuv420P12Le`)
///
/// Chroma is always declared `H=V=1`. Plane strides are the row width in
/// samples (i.e. `stride_y = y_w`, `stride_c = c_w`).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_yuv_jpeg_12bit(
    width: u32,
    height: u32,
    y_samples: &[u16],
    cb_samples: &[u16],
    cr_samples: &[u16],
    h_factor: u8,
    v_factor: u8,
    quality: u8,
) -> Result<Vec<u8>> {
    // Permit the three supported decode subsamplings plus 4:1:1 (luma 4×1)
    // so the test surface can exercise the decoder's reject path for
    // non-PixelFormat-mapped sampling factors. Beyond that we don't carry
    // encoder helpers for arbitrary `(H, V)` grids.
    if !matches!((h_factor, v_factor), (1, 1) | (2, 1) | (2, 2) | (4, 1)) {
        return Err(Error::invalid(
            "12-bit YUV helper: unsupported sampling factor",
        ));
    }
    let w = width as usize;
    let h = height as usize;
    let c_w = w.div_ceil(h_factor as usize);
    let c_h = h.div_ceil(v_factor as usize);
    if y_samples.len() < w * h || cb_samples.len() < c_w * c_h || cr_samples.len() < c_w * c_h {
        return Err(Error::invalid(
            "12-bit YUV helper: samples buffer too short",
        ));
    }
    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let chroma_q = scale_for_quality(&DEFAULT_CHROMA_Q50, quality);
    let huff = DefaultHuffman::build()?;

    let mut out: Vec<u8> = Vec::with_capacity(32_768);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_dqt(&mut out, 0, &luma_q);
    write_dqt(&mut out, 1, &chroma_q);
    write_sof1_yuv_12bit(&mut out, w as u16, h as u16, h_factor, v_factor);
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_dht(&mut out, 0, 1, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    write_dht(&mut out, 1, 1, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);
    write_sos_yuv(&mut out);

    // MCU geometry: an MCU covers `h_factor * 8` × `v_factor * 8` luma
    // pixels plus one 8×8 block per chroma component.
    let mcu_w_px = 8 * h_factor as usize;
    let mcu_h_px = 8 * v_factor as usize;
    let mcus_x = w.div_ceil(mcu_w_px);
    let mcus_y = h.div_ceil(mcu_h_px);

    let mut bw = BitWriter::new(&mut out);
    let mut prev_dc_y: i32 = 0;
    let mut prev_dc_cb: i32 = 0;
    let mut prev_dc_cr: i32 = 0;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // Luma blocks: `h_factor` × `v_factor` per MCU.
            for jy in 0..v_factor as usize {
                for ix in 0..h_factor as usize {
                    let x0 = mx * mcu_w_px + ix * 8;
                    let y0 = my * mcu_h_px + jy * 8;
                    let mut blk = [0.0f32; 64];
                    fill_block_u16_levelshift_2048(&mut blk, y_samples, w, w, h, x0, y0);
                    encode_block(
                        &mut bw,
                        &mut blk,
                        &luma_q,
                        &mut prev_dc_y,
                        &huff.luma_dc,
                        &huff.luma_ac,
                    );
                }
            }
            // Chroma blocks: one each per MCU at the chroma's own resolution.
            let cx0 = mx * 8;
            let cy0 = my * 8;
            let mut cb_blk = [0.0f32; 64];
            fill_block_u16_levelshift_2048(&mut cb_blk, cb_samples, c_w, c_w, c_h, cx0, cy0);
            encode_block(
                &mut bw,
                &mut cb_blk,
                &chroma_q,
                &mut prev_dc_cb,
                &huff.chroma_dc,
                &huff.chroma_ac,
            );
            let mut cr_blk = [0.0f32; 64];
            fill_block_u16_levelshift_2048(&mut cr_blk, cr_samples, c_w, c_w, c_h, cx0, cy0);
            encode_block(
                &mut bw,
                &mut cr_blk,
                &chroma_q,
                &mut prev_dc_cr,
                &huff.chroma_dc,
                &huff.chroma_ac,
            );
        }
    }
    bw.finish();

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

#[cfg(test)]
fn write_sof1_yuv_12bit(out: &mut Vec<u8>, width: u16, height: u16, h_factor: u8, v_factor: u8) {
    let mut payload = Vec::with_capacity(8 + 9);
    payload.push(12); // precision = 12
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(3); // Nf = 3 components
                     // Y component (id=1) carries the variable sampling factor; chroma is
                     // always H=V=1.
    payload.push(1);
    payload.push((h_factor << 4) | v_factor);
    payload.push(0); // qt = 0 (luma)
    payload.push(2);
    payload.push(0x11);
    payload.push(1); // qt = 1 (chroma)
    payload.push(3);
    payload.push(0x11);
    payload.push(1);
    write_length_prefix(out, markers::SOF1, &payload);
}

#[cfg(test)]
fn write_sos_yuv(out: &mut Vec<u8>) {
    let payload: [u8; 10] = [
        3, // Ns
        1, 0x00, // Y → DC=0 AC=0
        2, 0x11, // Cb → DC=1 AC=1
        3, 0x11, // Cr → DC=1 AC=1
        0, 63, 0, // Ss, Se, Ah|Al
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

// ---- Test-only progressive (SOF2) 12-bit YUV encoder ---------------------
//
// Exercises the decoder's SOF2-at-P=12 path. T.81 §G.1.1 permits
// progressive at precision 8 *or* 12; the progressive scan-builder code is
// precision-agnostic (operates in i32 coefficient space), so the only
// "new" emission machinery here is the SOF2 segment with `P = 12` and the
// 2048-centre level shift in the per-block DCT input.
//
// The scan decomposition is the simple spectral-selection-only layout
// (Ah = Al = 0): interleaved DC scan, then three single-component AC
// scans (Y, Cb, Cr) over [1..=5] then [6..=63]. This matches the
// non-SA branch of `encode_jpeg_progressive_inner`, which the decoder
// already round-trips at P = 8.

/// Variant of `fill_coef_grid` that takes 12-bit-precision `u16` samples
/// and applies the spec level shift of `2 ^ (P − 1) = 2048` per
/// T.81 §A.3.1.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fill_coef_grid_u16_2048(
    coefs: &mut [[i32; 64]],
    plane: &[u16],
    stride: usize,
    w: usize,
    h: usize,
    blocks_x: usize,
    blocks_y: usize,
    quant: &[u16; 64],
) {
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut block = [0.0f32; 64];
            fill_block_u16_levelshift_2048(&mut block, plane, stride, w, h, bx * 8, by * 8);
            fdct8x8(&mut block);
            let mut q = [0i32; 64];
            for k in 0..64 {
                let v = block[k] / quant[k] as f32;
                q[k] = if v >= 0.0 {
                    (v + 0.5) as i32
                } else {
                    -((-v + 0.5) as i32)
                };
            }
            coefs[by * blocks_x + bx] = q;
        }
    }
}

/// SOF2 (progressive) header at `P = 12`, three components. Shape mirrors
/// `write_sof2` but with the precision byte set to 12.
#[cfg(test)]
fn write_sof2_yuv_12bit(out: &mut Vec<u8>, width: u16, height: u16, h: u8, v: u8) {
    let mut payload = Vec::with_capacity(8 + 9);
    payload.push(12); // precision = 12
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(3); // components
    payload.push(1);
    payload.push((h << 4) | v);
    payload.push(0); // luma qt
    payload.push(2);
    payload.push(0x11);
    payload.push(1); // chroma qt
    payload.push(3);
    payload.push(0x11);
    payload.push(1);
    write_length_prefix(out, markers::SOF2, &payload);
}

/// Emit a three-component progressive (SOF2) JPEG at `P = 12` precision.
/// Uses the spectral-selection-only scan decomposition (interleaved DC
/// pass + Y/Cb/Cr AC bands `[1..=5]` then `[6..=63]`, all with
/// `Ah = Al = 0`). Caller's samples must lie close to the 2048 midpoint so
/// post-DCT coefficient magnitudes stay within the Annex K AC table's
/// SSSS ≤ 10 envelope — same constraint as `encode_yuv_jpeg_12bit` for
/// the baseline path.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_yuv_jpeg_progressive_12bit(
    width: u32,
    height: u32,
    y_samples: &[u16],
    cb_samples: &[u16],
    cr_samples: &[u16],
    h_factor: u8,
    v_factor: u8,
    quality: u8,
) -> Result<Vec<u8>> {
    if !matches!((h_factor, v_factor), (1, 1) | (2, 1) | (2, 2)) {
        return Err(Error::invalid(
            "12-bit progressive YUV helper: unsupported sampling factor",
        ));
    }
    let w = width as usize;
    let h = height as usize;
    let c_w = w.div_ceil(h_factor as usize);
    let c_h = h.div_ceil(v_factor as usize);
    if y_samples.len() < w * h || cb_samples.len() < c_w * c_h || cr_samples.len() < c_w * c_h {
        return Err(Error::invalid(
            "12-bit progressive YUV helper: samples buffer too short",
        ));
    }

    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let chroma_q = scale_for_quality(&DEFAULT_CHROMA_Q50, quality);
    let huff = DefaultHuffman::build()?;

    // MCU + per-component block-grid geometry mirroring the 8-bit
    // progressive encoder.
    let mcu_w_px = 8 * h_factor as usize;
    let mcu_h_px = 8 * v_factor as usize;
    let mcus_x = w.div_ceil(mcu_w_px);
    let mcus_y = h.div_ceil(mcu_h_px);

    let luma_blocks_x = mcus_x * h_factor as usize;
    let luma_blocks_y = mcus_y * v_factor as usize;
    let mut y_coefs = vec![[0i32; 64]; luma_blocks_x * luma_blocks_y];
    fill_coef_grid_u16_2048(
        &mut y_coefs,
        y_samples,
        w,
        w,
        h,
        luma_blocks_x,
        luma_blocks_y,
        &luma_q,
    );

    let chroma_blocks_x = mcus_x;
    let chroma_blocks_y = mcus_y;
    let mut cb_coefs = vec![[0i32; 64]; chroma_blocks_x * chroma_blocks_y];
    let mut cr_coefs = vec![[0i32; 64]; chroma_blocks_x * chroma_blocks_y];
    fill_coef_grid_u16_2048(
        &mut cb_coefs,
        cb_samples,
        c_w,
        c_w,
        c_h,
        chroma_blocks_x,
        chroma_blocks_y,
        &chroma_q,
    );
    fill_coef_grid_u16_2048(
        &mut cr_coefs,
        cr_samples,
        c_w,
        c_w,
        c_h,
        chroma_blocks_x,
        chroma_blocks_y,
        &chroma_q,
    );

    // Header.
    let mut out: Vec<u8> = Vec::with_capacity(32_768);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_dqt(&mut out, 0, &luma_q);
    write_dqt(&mut out, 1, &chroma_q);
    write_sof2_yuv_12bit(&mut out, w as u16, h as u16, h_factor, v_factor);
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    write_dht(&mut out, 0, 1, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    write_dht(&mut out, 1, 1, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);

    // Scan 1: interleaved DC (Ss=0, Se=0, Ah=Al=0).
    write_sos_progressive_dc_interleaved(&mut out);
    write_dc_scan_interleaved(
        &mut out,
        &y_coefs,
        luma_blocks_x,
        h_factor as usize,
        v_factor as usize,
        mcus_x,
        mcus_y,
        &cb_coefs,
        &cr_coefs,
        chroma_blocks_x,
        &huff,
    );

    // AC bands, low then high, per component (Y / Cb / Cr).
    for &(ss, se) in &[(1u8, 5u8), (6u8, 63u8)] {
        write_sos_progressive_ac(&mut out, 1, 0, ss, se);
        write_ac_scan(
            &mut out,
            &y_coefs,
            luma_blocks_x * luma_blocks_y,
            &huff.luma_ac,
            ss as usize,
            se as usize,
        );
        write_sos_progressive_ac(&mut out, 2, 1, ss, se);
        write_ac_scan(
            &mut out,
            &cb_coefs,
            chroma_blocks_x * chroma_blocks_y,
            &huff.chroma_ac,
            ss as usize,
            se as usize,
        );
        write_sos_progressive_ac(&mut out, 3, 1, ss, se);
        write_ac_scan(
            &mut out,
            &cr_coefs,
            chroma_blocks_x * chroma_blocks_y,
            &huff.chroma_ac,
            ss as usize,
            se as usize,
        );
    }

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

// ---- Lossless JPEG encoder (SOF3, single-component grayscale) -----------

/// Custom DC Huffman table for the lossless (SOF3) encoder.
///
/// Annex K's standard luma DC table (`STD_DC_LUMA_BITS` / `STD_DC_LUMA_VALS`)
/// only covers SSSS symbols 0..=11, which is enough for 8-bit lossless but
/// breaks at higher precisions where the residual magnitude category can
/// reach 16 (T.81 Table H.2). We use a single canonical table that covers
/// every SSSS value in 0..=16 and is therefore valid for any precision
/// `P ∈ 2..=16`.
///
/// Layout (Kraft sum 1.0):
///   * 15 symbols at code length 4 — SSSS 0..=14 (the common cases).
///   * 2 symbols at code length 5 — SSSS 15 and 16 (large residuals).
///
/// `STD_DC_LOSSLESS_BITS[L-1]` is the count of codes of length `L`, and
/// `STD_DC_LOSSLESS_VALS` is the canonical-order symbol list per
/// T.81 Annex C.2.
const STD_DC_LOSSLESS_BITS: [u8; 16] = [0, 0, 0, 15, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const STD_DC_LOSSLESS_VALS: [u8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

/// Category for a lossless residual computed modulo 2^16 (T.81 §H.1.2.1).
/// Returns `(ssss, bits, extra_bits_count)` where `extra_bits_count` is the
/// number of magnitude bits to write after the Huffman code:
///
///   * `ssss == 0`       → no extra bits (residual is 0).
///   * `1 <= ssss <= 15` → `ssss` extra bits per Annex F category code.
///   * `ssss == 16`      → no extra bits; encodes the special diff value
///     32768 (§H.1.2.2, Table H.2).
///
/// Caller passes the *unreduced* signed difference `diff = actual - pred`.
/// The function handles the mod-2^16 reduction internally: any `diff`
/// whose 16-bit-wrapped magnitude equals 0x8000 is mapped to `ssss == 16`
/// regardless of original sign (since +32768 and -32768 alias under
/// mod-2^16 arithmetic).
fn category_lossless(diff: i32) -> (u8, u32) {
    // Reduce to the canonical signed 16-bit representative: positive
    // diffs > 32767 wrap below 0, and the lone half-modulus point
    // (±32768) becomes the special SSSS=16 case.
    let diff16 = (diff as i64) & 0xFFFF;
    // Treat 0x8000 as the half-modulus point regardless of sign.
    if diff16 == 0x8000 {
        return (16, 0);
    }
    // Canonical signed: in -32767..=32767.
    let signed: i32 = if diff16 >= 0x8000 {
        (diff16 - 0x1_0000) as i32
    } else {
        diff16 as i32
    };
    category(signed)
}

/// Encode a single-component grayscale image as a standalone **lossless**
/// JPEG (SOF3) byte stream.
///
/// * `samples` — row-major luminance samples; row `y` starts at byte
///   `y * stride`. For 8-bit precision this carries one byte per sample;
///   for `precision > 8` it must contain little-endian 16-bit samples
///   (two bytes per sample) and `stride` is therefore in bytes.
/// * `precision` — input bits per sample, in `2..=16`. Outside this
///   range the function returns [`Error::Unsupported`].
/// * `predictor` — selector value 1..=7 from T.81 Table H.1. Predictor
///   1 (Ra) is the safest default; predictors 2 / 3 / 4..=7 may compress
///   better on horizontally or vertically smooth images.
///
/// Output is bit-exact: the decoder side recovers every input sample
/// verbatim. Point transform is fixed at `Pt = 0` and no restart markers
/// are emitted. For non-zero point transform or restart-marker emission
/// see [`encode_lossless_jpeg_grayscale_with_opts`]. The encoder reuses
/// the [`HuffTable`] machinery to build the wide-symbol DC table at
/// construction time so the output stays universal for every precision
/// in the valid range.
pub fn encode_lossless_jpeg_grayscale(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    precision: u8,
    predictor: u8,
) -> Result<Vec<u8>> {
    encode_lossless_jpeg_grayscale_with_opts(
        width, height, samples, stride, precision, predictor, 0, 0,
    )
}

/// Like [`encode_lossless_jpeg_grayscale`] but also accepts a non-zero
/// point transform `point_transform` (`Pt` in T.81 — the SOS `Al` field)
/// and a `restart_interval` measured in MCUs (= samples, since a lossless
/// MCU is one residual). Both default-to-zero options match the
/// historical zero-restart, zero-Pt output.
///
/// * `point_transform` — `0..=15`, but must be strictly less than
///   `precision` (T.81 §H.1.2: a non-zero `Pt` divides every input
///   sample by `2^Pt`, so `Pt = precision` would discard every bit).
///   With `Pt > 0` the encoder right-shifts every input sample by `Pt`
///   before predictive coding; the decoder side then left-shifts the
///   reconstructed sample back by `Pt` on output (the low `Pt` bits of
///   the original sample are lost).
/// * `restart_interval` — number of samples between successive `RSTn`
///   markers (`0` disables restart emission, matching
///   [`encode_lossless_jpeg_grayscale`]). On each boundary the encoder
///   byte-aligns the bitstream, writes a fresh `RST0..=RST7` marker
///   (cycling modulo 8 per T.81 §F.1.1.5.2), and re-seeds the
///   single-sample predictor history to the image-origin value
///   `2^(precision − Pt − 1)` (T.81 §H.1.2.1 — every restart interval
///   starts with the same scan-origin defaults as the image start).
///
/// Bit-exact roundtrip vs. the SOF3 decoder for every supported
/// precision, predictor, restart interval, and `Pt`.
#[allow(clippy::too_many_arguments)]
pub fn encode_lossless_jpeg_grayscale_with_opts(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    precision: u8,
    predictor: u8,
    restart_interval: u16,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if !(2..=16).contains(&precision) {
        return Err(Error::unsupported(format!(
            "lossless encoder: precision {precision} out of range 2..=16"
        )));
    }
    if !(1..=7).contains(&predictor) {
        return Err(Error::invalid(format!(
            "lossless encoder: predictor {predictor} not in 1..=7"
        )));
    }
    if point_transform >= precision {
        return Err(Error::invalid(format!(
            "lossless encoder: point_transform {point_transform} must be < precision {precision}"
        )));
    }
    if point_transform > 15 {
        return Err(Error::invalid(format!(
            "lossless encoder: point_transform {point_transform} > 15 (SOS Al is 4 bits)"
        )));
    }
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid("lossless encoder: zero-size image"));
    }
    let bytes_per_sample = if precision <= 8 { 1 } else { 2 };
    if stride < w * bytes_per_sample {
        return Err(Error::invalid(
            "lossless encoder: stride smaller than width*bytes_per_sample",
        ));
    }
    if samples.len() < stride * h {
        return Err(Error::invalid(
            "lossless encoder: samples shorter than stride*h",
        ));
    }

    // Build the wide-symbol DC Huffman table once per call. The
    // STD_DC_LOSSLESS_* constants give a Kraft-complete (sum=1) layout
    // valid for any precision in 2..=16, so no per-image table tuning
    // is required for spec correctness.
    let dc_huff = HuffTable::build(&STD_DC_LOSSLESS_BITS, &STD_DC_LOSSLESS_VALS)?;

    let mut out: Vec<u8> = Vec::with_capacity(16_384 + 2 * w * h);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_sof_lossless(&mut out, w as u16, h as u16, precision);
    write_dht(&mut out, 0, 0, &STD_DC_LOSSLESS_BITS, &STD_DC_LOSSLESS_VALS);
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    write_sos_lossless(&mut out, predictor, point_transform);

    // Decode samples into a flat u16 buffer once so the predictor loop
    // doesn't have to branch on `precision` per pixel. With Pt > 0 the
    // wire samples are `actual >> Pt` (T.81 §H.1.2). We apply the
    // shift here so every subsequent value already lives in the
    // `precision - Pt` range, matching the decoder's `sample_bits`.
    let pt = point_transform as u32;
    let mut src = vec![0u16; w * h];
    if precision <= 8 {
        for y in 0..h {
            for x in 0..w {
                src[y * w + x] = (samples[y * stride + x] as u16) >> pt;
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                let lo = samples[y * stride + x * 2] as u16;
                let hi = samples[y * stride + x * 2 + 1] as u16;
                src[y * w + x] = (lo | (hi << 8)) >> pt;
            }
        }
    }

    // Validate that no pre-shift input sample exceeds the declared
    // precision (the shifted value automatically fits in `precision - Pt`
    // bits, but a stray out-of-range input is still a caller bug).
    let max_sample: u32 = (1u32 << precision) - 1;
    if precision <= 8 {
        for y in 0..h {
            for x in 0..w {
                let v = samples[y * stride + x] as u32;
                if v > max_sample {
                    return Err(Error::invalid(format!(
                        "lossless encoder: sample {v} exceeds precision-{precision} max {max_sample}"
                    )));
                }
            }
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                let lo = samples[y * stride + x * 2] as u32;
                let hi = samples[y * stride + x * 2 + 1] as u32;
                let v = lo | (hi << 8);
                if v > max_sample {
                    return Err(Error::invalid(format!(
                        "lossless encoder: sample {v} exceeds precision-{precision} max {max_sample}"
                    )));
                }
            }
        }
    }

    // Default prediction for the first sample of the scan and after each
    // restart interval (T.81 §H.1.2.1). With Pt the working precision is
    // `precision − Pt`, so the origin is `2^(precision − Pt − 1)`.
    let sample_bits = precision as u32 - pt;
    let origin: i32 = 1i32 << (sample_bits - 1);

    // Restart bookkeeping. A lossless MCU is exactly one sample (per
    // §H.1.2 with H_i=V_i=1), so the restart interval counts samples.
    // `samples_since_restart` is reset on every `RSTn` emission; the
    // encoder cycles RST0..=RST7 modulo 8 per §F.1.1.5.2.
    let ri = restart_interval as u32;
    let mut rst_counter: u8 = 0;
    let mut samples_since_restart: u32 = 0;
    let total_samples: u64 = w as u64 * h as u64;
    let mut sample_index: u64 = 0;
    // `reset_pred` is true at scan start and immediately after each
    // RSTn so the next sample uses `origin` (not Ra/Rb/Rc from the
    // previous interval — those neighbours are still present in `src`
    // but the spec mandates we forget them at the restart boundary).
    let mut reset_pred = true;

    let mut bw = BitWriter::new(&mut out);
    for y in 0..h {
        for x in 0..w {
            let actual = src[y * w + x] as i32;
            let pred: i32 = if reset_pred {
                origin
            } else if y == 0 {
                // First line uses Ra regardless of selector (Table H.1
                // applies after the first line / restart).
                src[y * w + x - 1] as i32
            } else if x == 0 {
                // Start of a non-first line uses Rb.
                src[(y - 1) * w + x] as i32
            } else {
                let ra = src[y * w + x - 1] as i32;
                let rb = src[(y - 1) * w + x] as i32;
                let rc = src[(y - 1) * w + x - 1] as i32;
                match predictor {
                    1 => ra,
                    2 => rb,
                    3 => rc,
                    4 => ra + rb - rc,
                    // Per H.1 footnote: divide-by-2 is an arithmetic shift.
                    5 => ra + ((rb - rc) >> 1),
                    6 => rb + ((ra - rc) >> 1),
                    7 => (ra + rb) >> 1,
                    _ => unreachable!(),
                }
            };
            let diff = actual - pred;
            let (s, bits) = category_lossless(diff);
            let hc = dc_huff.encode[s as usize];
            debug_assert!(
                hc.len != 0,
                "DC Huffman code for SSSS={s} must be present; \
                 STD_DC_LOSSLESS_VALS covers 0..=16"
            );
            bw.write_bits(hc.code as u32, hc.len as u32);
            // SSSS == 0 emits no extra bits (zero residual); SSSS == 16
            // also emits no extras (special-case Di = 32768 per H.1.2.2).
            if s != 0 && s != 16 {
                bw.write_bits(bits, s as u32);
            }

            reset_pred = false;
            sample_index += 1;
            samples_since_restart += 1;

            // Emit an RSTn marker after every `ri` samples, but never
            // after the very last sample (the decoder expects EOI next,
            // and the spec disallows a trailing RST per §F.1.1.5.2).
            if ri != 0 && samples_since_restart == ri && sample_index < total_samples {
                bw.flush_to_byte();
                bw.emit_raw_marker(markers::RST0 + (rst_counter & 0x07));
                rst_counter = rst_counter.wrapping_add(1);
                samples_since_restart = 0;
                reset_pred = true;
            }
        }
    }
    bw.finish();

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

/// Back-compat wrapper kept for the existing 8-bit roundtrip test in
/// `decoder::lossless_tests`. New callers should prefer
/// [`encode_lossless_jpeg_grayscale`].
#[cfg(test)]
pub(crate) fn encode_lossless_grayscale_jpeg_8bit(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
) -> Result<Vec<u8>> {
    encode_lossless_jpeg_grayscale(width, height, samples, stride, 8, 1)
}

/// Encode three planar 8-bit colour channels as a standalone **multi-component
/// lossless** JPEG (SOF3, three-component interleaved scan) byte stream.
///
/// Annex H of T.81 specifies that "each component in the scan is modeled
/// independently, using predictions derived from neighbouring samples of
/// that component" (§H.1.2). This is the natural extension of the
/// single-component grayscale encoder: three independent predictor planes
/// share a single Huffman table and a single predictor-selector value, and
/// their samples are interleaved one-per-MCU-position in the entropy
/// stream (since each component is declared with sampling factors
/// `H_i = V_i = 1`, a single MCU is exactly one sample per component).
///
/// * `planes` — three plane slices in `[ch0, ch1, ch2]` order. The
///   encoder's input is colour-agnostic: the caller passes `[R, G, B]`,
///   `[G, B, R]`, or any other three independent monochrome planes that
///   share the same `width` × `height`. Component IDs in the bitstream
///   are 1, 2, 3 in that order.
/// * `strides` — bytes per row per plane. For 8-bit precision this is
///   typically `width`; for `precision > 8` it is `width * 2` (the plane
///   carries little-endian 16-bit samples).
/// * `precision` — input bits per sample, in `2..=16`.
/// * `predictor` — selector value `1..=7` from Table H.1, **shared by
///   every component** (T.81 §B.2.3: the scan-header predictor selector
///   `Ss` applies to the whole scan).
///
/// Output is bit-exact for every supported precision. Point transform is
/// fixed at `Pt = 0` and no restart markers are emitted. For non-zero
/// point transform or restart-marker emission see
/// [`encode_lossless_jpeg_rgb_with_opts`]. The DC Huffman table built
/// once at the top of the call is shared across all three components via
/// DHT `Td = 0`.
pub fn encode_lossless_jpeg_rgb(
    width: u32,
    height: u32,
    planes: [&[u8]; 3],
    strides: [usize; 3],
    precision: u8,
    predictor: u8,
) -> Result<Vec<u8>> {
    encode_lossless_jpeg_rgb_with_opts(width, height, planes, strides, precision, predictor, 0, 0)
}

/// Like [`encode_lossless_jpeg_rgb`] but also accepts `restart_interval`
/// (in MCUs — and a 3-component lossless MCU is exactly one sample per
/// component, i.e. one pixel position — per T.81 §H.1.2 with
/// `H_i = V_i = 1`) and a `point_transform` (`Pt` in T.81, the low
/// nibble of the SOS `Ah|Al` field). Both default-to-zero options match
/// the historical zero-restart, zero-Pt output.
///
/// * `point_transform` — `0..=15` and strictly less than `precision`.
///   With `Pt > 0` every input sample is right-shifted by `Pt` before
///   prediction; the decoder side later left-shifts the reconstructed
///   sample by the same `Pt` when materialising the output plane.
/// * `restart_interval` — number of MCUs (= pixel positions) between
///   successive `RSTn` markers. On every boundary the encoder
///   byte-aligns the bitstream, emits the next `RST0..=RST7` marker
///   (cycling modulo 8 per T.81 §F.1.1.5.2), and re-seeds **every**
///   component's predictor history to the per-component origin
///   `2^(precision − Pt − 1)` (T.81 §H.1.2.1: each restart interval
///   starts with the same scan-origin defaults as the image start,
///   independently per component).
///
/// Bit-exact roundtrip vs. the SOF3 three-component decoder for every
/// supported predictor / restart interval / Pt combination at every
/// precision in `2..=16`. Decoder output shape varies by precision —
/// see `decode_lossless_scan` in `src/decoder.rs` for the per-precision
/// `PixelFormat` mapping (P=8 → packed Rgb24, P∈{10,12,14} → planar
/// Gbrp*Le, every other P → packed Rgb48Le).
#[allow(clippy::too_many_arguments)]
pub fn encode_lossless_jpeg_rgb_with_opts(
    width: u32,
    height: u32,
    planes: [&[u8]; 3],
    strides: [usize; 3],
    precision: u8,
    predictor: u8,
    restart_interval: u16,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if !(2..=16).contains(&precision) {
        return Err(Error::unsupported(format!(
            "lossless RGB encoder: precision {precision} out of range 2..=16"
        )));
    }
    if !(1..=7).contains(&predictor) {
        return Err(Error::invalid(format!(
            "lossless RGB encoder: predictor {predictor} not in 1..=7"
        )));
    }
    if point_transform >= precision {
        return Err(Error::invalid(format!(
            "lossless RGB encoder: point_transform {point_transform} must be < precision {precision}"
        )));
    }
    if point_transform > 15 {
        return Err(Error::invalid(format!(
            "lossless RGB encoder: point_transform {point_transform} > 15 (SOS Al is 4 bits)"
        )));
    }
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid("lossless RGB encoder: zero-size image"));
    }
    let bytes_per_sample = if precision <= 8 { 1 } else { 2 };
    for c in 0..3 {
        if strides[c] < w * bytes_per_sample {
            return Err(Error::invalid(
                "lossless RGB encoder: stride smaller than width*bytes_per_sample",
            ));
        }
        if planes[c].len() < strides[c] * h {
            return Err(Error::invalid(
                "lossless RGB encoder: plane shorter than stride*h",
            ));
        }
    }

    // Single shared DC Huffman table (Td = 0). The Kraft-complete layout
    // in STD_DC_LOSSLESS_* covers SSSS 0..=16, valid for any precision.
    let dc_huff = HuffTable::build(&STD_DC_LOSSLESS_BITS, &STD_DC_LOSSLESS_VALS)?;

    let mut out: Vec<u8> = Vec::with_capacity(16_384 + 3 * 2 * w * h);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_sof_lossless_multi(&mut out, w as u16, h as u16, precision, 3);
    write_dht(&mut out, 0, 0, &STD_DC_LOSSLESS_BITS, &STD_DC_LOSSLESS_VALS);
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    write_sos_lossless_multi(&mut out, predictor, 3, point_transform);

    let pt = point_transform as u32;

    // Decode each plane into a flat u16 buffer once so the predictor
    // loop is precision-uniform, and apply the point-transform shift
    // up-front so all downstream arithmetic lives in `precision - Pt`.
    let mut src: [Vec<u16>; 3] = [vec![0u16; w * h], vec![0u16; w * h], vec![0u16; w * h]];
    if precision <= 8 {
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    src[c][y * w + x] = (planes[c][y * strides[c] + x] as u16) >> pt;
                }
            }
        }
    } else {
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let lo = planes[c][y * strides[c] + x * 2] as u16;
                    let hi = planes[c][y * strides[c] + x * 2 + 1] as u16;
                    src[c][y * w + x] = (lo | (hi << 8)) >> pt;
                }
            }
        }
    }

    // Validate pre-shift sample range against declared precision.
    let max_sample: u32 = (1u32 << precision) - 1;
    if precision <= 8 {
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let v = planes[c][y * strides[c] + x] as u32;
                    if v > max_sample {
                        return Err(Error::invalid(format!(
                            "lossless RGB encoder: sample {v} in plane {c} exceeds precision-{precision} max {max_sample}"
                        )));
                    }
                }
            }
        }
    } else {
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let lo = planes[c][y * strides[c] + x * 2] as u32;
                    let hi = planes[c][y * strides[c] + x * 2 + 1] as u32;
                    let v = lo | (hi << 8);
                    if v > max_sample {
                        return Err(Error::invalid(format!(
                            "lossless RGB encoder: sample {v} in plane {c} exceeds precision-{precision} max {max_sample}"
                        )));
                    }
                }
            }
        }
    }

    // Default prediction for the first sample of each component at scan
    // start and after every RSTn (T.81 §H.1.2.1): `2^(P − Pt − 1)`.
    let sample_bits = precision as u32 - pt;
    let origin: i32 = 1i32 << (sample_bits - 1);

    // Restart bookkeeping. The lossless 3-component MCU is one pixel
    // (each component's `H_i = V_i = 1`, so all three samples land in a
    // single MCU). `mcus_since_restart` counts pixels.
    let ri = restart_interval as u32;
    let mut rst_counter: u8 = 0;
    let mut mcus_since_restart: u32 = 0;
    let total_mcus: u64 = w as u64 * h as u64;
    let mut mcu_index: u64 = 0;
    let mut reset_pred = true;

    let mut bw = BitWriter::new(&mut out);
    for y in 0..h {
        for x in 0..w {
            // Annex H interleaved-MCU order with Hi=Vi=1 across all
            // components: at each (y, x) position, emit one residual for
            // component 0, then 1, then 2 (component IDs in scan-header
            // order).
            for c in 0..3 {
                let plane = &src[c];
                let actual = plane[y * w + x] as i32;
                let pred: i32 = if reset_pred {
                    origin
                } else if y == 0 {
                    // First line: forced predictor 1 (Ra) per H.1.2.1.
                    plane[y * w + x - 1] as i32
                } else if x == 0 {
                    // First column: forced predictor 2 (Rb) per H.1.2.1.
                    plane[(y - 1) * w + x] as i32
                } else {
                    let ra = plane[y * w + x - 1] as i32;
                    let rb = plane[(y - 1) * w + x] as i32;
                    let rc = plane[(y - 1) * w + x - 1] as i32;
                    match predictor {
                        1 => ra,
                        2 => rb,
                        3 => rc,
                        4 => ra + rb - rc,
                        // H.1 footnote: divide-by-2 is an arithmetic shift.
                        5 => ra + ((rb - rc) >> 1),
                        6 => rb + ((ra - rc) >> 1),
                        7 => (ra + rb) >> 1,
                        _ => unreachable!(),
                    }
                };
                let diff = actual - pred;
                let (s, bits) = category_lossless(diff);
                let hc = dc_huff.encode[s as usize];
                debug_assert!(hc.len != 0, "DC Huffman code for SSSS={s} must be present");
                bw.write_bits(hc.code as u32, hc.len as u32);
                // SSSS == 0 emits no residual bits; SSSS == 16 is the
                // half-modulus special case (T.81 §H.1.2.2).
                if s != 0 && s != 16 {
                    bw.write_bits(bits, s as u32);
                }
            }

            reset_pred = false;
            mcu_index += 1;
            mcus_since_restart += 1;

            // Emit an RSTn marker after every `ri` MCUs, but never after
            // the very last MCU (the decoder expects EOI next).
            if ri != 0 && mcus_since_restart == ri && mcu_index < total_mcus {
                bw.flush_to_byte();
                bw.emit_raw_marker(markers::RST0 + (rst_counter & 0x07));
                rst_counter = rst_counter.wrapping_add(1);
                mcus_since_restart = 0;
                reset_pred = true;
            }
        }
    }
    bw.finish();

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

fn write_sof_lossless(out: &mut Vec<u8>, width: u16, height: u16, precision: u8) {
    let mut payload = Vec::with_capacity(8 + 3);
    payload.push(precision);
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(1); // Nf — single grayscale component
    payload.push(1); // component id
    payload.push(0x11); // H=1 V=1
    payload.push(0); // Tq=0 (lossless ignores quant tables, but the byte is still present)
    write_length_prefix(out, 0xC3 /* SOF3 */, &payload);
}

fn write_sos_lossless(out: &mut Vec<u8>, predictor: u8, point_transform: u8) {
    debug_assert!(point_transform <= 0x0F, "Pt must fit in 4 bits");
    let payload: [u8; 6] = [
        1, // Ns
        1,
        0x00, // comp 1 → DC=0 AC=0 (AC unused for lossless)
        predictor,
        0,                      // Ss = predictor selector, Se = 0
        point_transform & 0x0F, // Ah | Al — Ah=0 (lossless), Al=Pt (low nibble)
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

/// Write an SOF3 segment for an `nf`-component lossless frame with every
/// component declared `H_i = V_i = 1` (the natural interleaved layout for
/// multi-component lossless per T.81 §H.1.2). Component identifiers start
/// at 1 and increment by 1.
fn write_sof_lossless_multi(out: &mut Vec<u8>, width: u16, height: u16, precision: u8, nf: u8) {
    debug_assert!((1..=4).contains(&nf));
    let mut payload = Vec::with_capacity(8 + 3 * nf as usize);
    payload.push(precision);
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(nf);
    for ci in 1..=nf {
        payload.push(ci); // component identifier
        payload.push(0x11); // H=1 V=1
        payload.push(0); // Tq (ignored for lossless)
    }
    write_length_prefix(out, markers::SOF3, &payload);
}

/// Write an interleaved SOS segment for an `ns`-component lossless scan.
/// Every component selects DC table 0 (AC unused), `Ss` carries the
/// shared predictor selector, `Se = 0`, `Ah | Al` packs `Ah = 0` (no
/// successive approximation in lossless) and the supplied
/// `point_transform` in the low nibble (`Al = Pt`).
fn write_sos_lossless_multi(out: &mut Vec<u8>, predictor: u8, ns: u8, point_transform: u8) {
    debug_assert!((1..=4).contains(&ns));
    debug_assert!(point_transform <= 0x0F, "Pt must fit in 4 bits");
    let mut payload = Vec::with_capacity(3 + 2 * ns as usize);
    payload.push(ns);
    for ci in 1..=ns {
        payload.push(ci); // component identifier
        payload.push(0x00); // DC=0 AC=0
    }
    payload.push(predictor); // Ss = predictor selector
    payload.push(0); // Se
    payload.push(point_transform & 0x0F); // Ah | Al: Ah=0, Al=Pt
    write_length_prefix(out, markers::SOS, &payload);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_zero() {
        assert_eq!(category(0), (0, 0));
    }

    #[test]
    fn category_small() {
        // v=+1: size=1, bits=1
        assert_eq!(category(1), (1, 1));
        // v=-1: size=1, bits=0
        assert_eq!(category(-1), (1, 0));
        // v=+5: size=3, bits=5
        assert_eq!(category(5), (3, 5));
        // v=-5: size=3, bits=(-5 + 7) = 2
        assert_eq!(category(-5), (3, 2));
    }

    #[test]
    fn bit_writer_stuffs_ff() {
        let mut buf = Vec::new();
        let mut bw = BitWriter::new(&mut buf);
        bw.write_bits(0xFF, 8);
        bw.finish();
        assert_eq!(buf, vec![0xFF, 0x00]);
    }

    #[test]
    fn category_lossless_special_case_half_modulus() {
        // The half-modulus point ±32768 collapses to SSSS=16 with zero
        // extra bits per T.81 §H.1.2.2 (Table H.2).
        assert_eq!(category_lossless(32768), (16, 0));
        assert_eq!(category_lossless(-32768), (16, 0));
        // Adjacent values fall back to the normal magnitude scheme:
        // 32767 fits in 15 bits, encodes plain.
        assert_eq!(category_lossless(32767).0, 15);
        assert_eq!(category_lossless(-32767).0, 15);
    }

    #[test]
    fn category_lossless_mod_2_to_16_aliases() {
        // The encoder reduces diff modulo 2^16, so values like 65536+x
        // (which the i32-domain calculation might produce when actual
        // is near 2^16 and pred is small) must collapse to x.
        for v in [-65535i32, -65000, -16384, 0, 16384, 65000, 65535] {
            let (s1, b1) = category_lossless(v);
            let (s2, b2) = category_lossless(v + 0x1_0000);
            assert_eq!((s1, b1), (s2, b2), "mod-2^16 alias mismatch at v={v}");
        }
    }

    #[test]
    fn lossless_dc_huff_table_kraft_complete() {
        // STD_DC_LOSSLESS_BITS must satisfy the Kraft equality so every
        // SSSS in 0..=16 has a unique prefix-free code.
        let mut kraft_num: u32 = 0; // numerator over 2^16
        for (i, &n) in STD_DC_LOSSLESS_BITS.iter().enumerate() {
            let len = (i + 1) as u32;
            kraft_num += (n as u32) * (1u32 << (16 - len));
        }
        assert_eq!(kraft_num, 1 << 16, "Kraft inequality must equal exactly 1");
        // Symbol coverage: all 17 SSSS values present.
        assert_eq!(STD_DC_LOSSLESS_VALS.len(), 17);
        let mut seen = [false; 17];
        for &v in &STD_DC_LOSSLESS_VALS {
            assert!((v as usize) < 17, "symbol {v} out of expected 0..=16 range");
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b), "every SSSS in 0..=16 must appear");
    }
}
