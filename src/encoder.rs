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

/// Quality factor 1..=100, scaled against the Annex K Q=50 base tables.
/// 75 is a sensible default.
pub const DEFAULT_QUALITY: u8 = 75;

// ---- Encoding ------------------------------------------------------------

/// Encode a single `VideoFrame` (YUV 4:4:4 / 4:2:2 / 4:2:0) as a complete,
/// self-contained baseline JPEG byte stream (`FFD8 … FFD9`). `quality` is
/// the 1..=100 factor scaled against the Annex K Q=50 base tables. Exposed
/// publicly so sibling crates (e.g. `oxideav-amv`, which wraps the same
/// bitstream with a custom container-level header) can reuse the encoder
/// without going through the `Encoder` trait's stateful packet/frame
/// plumbing.
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
/// Quality is the 1..=100 factor scaled against the Annex K Q=50 base tables.
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
/// Output round-trips through any conformant SOF2 decoder. The reconstructed
/// image is identical to the spectral-selection-only output — successive
/// approximation only changes the bit order, not the quantised coefficients.
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

// ---- Baseline (SOF0) single-component grayscale (Gray8) encoder ---------
//
// Single-component sequential JPEG at 8-bit precision. Mirrors the layout
// of [`encode_jpeg`] but emits only one component (`Y`, id 1, `H=V=1`),
// one DQT (luma), the standard DC/AC luma Huffman pair, and a one-entry
// SOS. The MCU is one 8x8 block of luma; the scan walks the picture in
// raster order with the same per-block path the YUV encoder uses
// (`fill_block` → `encode_block`).

/// Encode a single-component grayscale image as a standalone **baseline
/// sequential** JPEG byte stream (`FF D8 … FF D9`). The bitstream layout
/// is the usual SOI / JFIF APP0 / DQT (luma quantiser scaled by `quality`)
/// / SOF0 (one component, `H = V = 1`, precision 8) / DHT (Annex K luma
/// DC and AC) / optional DRI / SOS / entropy scan / EOI sequence — the
/// exact same shape as [`encode_jpeg`] reduced to one luma component.
///
/// Inputs:
/// - `samples`: tightly-packed grayscale samples in row-major order, each
///   byte covering one pixel.
/// - `stride`: bytes per row; must be `>= width`.
/// - `quality`: 1..=100, scaled against the Annex K Q=50 base table.
///
/// Round-trips through the matching SOF0 decoder (which emits
/// `PixelFormat::Gray8`) with the usual DCT-quantise-IDCT distortion
/// floor: PSNR climbs above ~40 dB by quality 75 and approaches 60 dB
/// at quality 100 on smooth content.
///
/// No restart markers and no metadata segments are emitted; for the
/// DRI + `RSTn` and APP-pass-through variants see
/// [`encode_jpeg_grayscale_with_opts`] and
/// [`encode_jpeg_grayscale_with_meta`].
pub fn encode_jpeg_grayscale(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    quality: u8,
) -> Result<Vec<u8>> {
    encode_jpeg_grayscale_with_meta(width, height, samples, stride, quality, 0, &[])
}

/// Like [`encode_jpeg_grayscale`] but also emits a DRI segment and cycles
/// `RST0..=RST7` markers every `restart_interval` MCUs during the scan.
/// Passing `0` disables restart marker emission (equivalent to
/// [`encode_jpeg_grayscale`]).
pub fn encode_jpeg_grayscale_with_opts(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    quality: u8,
    restart_interval: u16,
) -> Result<Vec<u8>> {
    encode_jpeg_grayscale_with_meta(
        width,
        height,
        samples,
        stride,
        quality,
        restart_interval,
        &[],
    )
}

/// Like [`encode_jpeg_grayscale_with_opts`] but inserts `meta` verbatim
/// between the SOI and the first DQT segment. `meta` must contain only
/// APP0..APP15 or COM segments (each starting with `0xFF 0xEn` /
/// `0xFF 0xFE` followed by a big-endian length). Use
/// [`extract_app_segments`] to harvest metadata from a source JPEG.
///
/// When `meta` is empty this is bit-for-bit identical to
/// [`encode_jpeg_grayscale_with_opts`].
pub fn encode_jpeg_grayscale_with_meta(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    quality: u8,
    restart_interval: u16,
    meta: &[u8],
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid("grayscale encoder: zero-size image"));
    }
    if stride < w {
        return Err(Error::invalid(
            "grayscale encoder: stride smaller than width",
        ));
    }
    if samples.len() < stride * h {
        return Err(Error::invalid(
            "grayscale encoder: samples shorter than stride*h",
        ));
    }

    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let huff = DefaultHuffman::build()?;

    let mut out: Vec<u8> = Vec::with_capacity(8_192);
    // SOI.
    out.push(0xFF);
    out.push(markers::SOI);
    // Metadata segments (JFIF APP0 fallback when caller provides nothing).
    if meta.is_empty() {
        write_jfif_app0(&mut out);
    } else {
        out.extend_from_slice(meta);
    }
    // DQT — one table, table id 0, precision 0 (8-bit).
    write_dqt(&mut out, 0, &luma_q);
    // SOF0 — single component, `H = V = 1`.
    write_sof0_grayscale_8bit(&mut out, w as u16, h as u16);
    // DHT — Annex K luma DC + AC.
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    // DRI before SOS per T.81 §F.2.2.4.
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    // SOS — one component (id 1), DC=0, AC=0.
    write_sos_grayscale_8bit(&mut out);

    // Scan: one 8x8 block per MCU, walked in raster order. The MCU is
    // exactly one luma data unit because the single component declares
    // `H = V = 1`.
    let mcus_x = w.div_ceil(8);
    let mcus_y = h.div_ceil(8);
    let total_mcus = mcus_x.saturating_mul(mcus_y);
    let ri = restart_interval as usize;
    let mut rst_counter: u8 = 0;
    let mut mcus_since_restart: usize = 0;
    let mut mcu_index: usize = 0;

    let mut bw = BitWriter::new(&mut out);
    let mut prev_dc: i32 = 0;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            let mut blk = [0.0f32; 64];
            fill_block(&mut blk, samples, stride, w, h, mx * 8, my * 8);
            encode_block(
                &mut bw,
                &mut blk,
                &luma_q,
                &mut prev_dc,
                &huff.luma_dc,
                &huff.luma_ac,
            );

            mcu_index += 1;
            mcus_since_restart += 1;

            // Restart boundary — same semantics as `write_scan`: never
            // emit `RSTn` after the very last MCU because the decoder
            // expects EOI immediately past the final residual.
            if ri != 0 && mcus_since_restart == ri && mcu_index < total_mcus {
                bw.flush_to_byte();
                bw.emit_raw_marker(markers::RST0 + (rst_counter & 0x07));
                rst_counter = rst_counter.wrapping_add(1);
                prev_dc = 0;
                mcus_since_restart = 0;
            }
        }
    }
    bw.finish();

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

fn write_sof0_grayscale_8bit(out: &mut Vec<u8>, width: u16, height: u16) {
    let mut payload = Vec::with_capacity(8 + 3);
    payload.push(8); // P = 8
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(1); // Nf = 1
    payload.push(1); // component id 1 (`Y`)
    payload.push(0x11); // H = 1, V = 1
    payload.push(0); // quantiser table 0
    write_length_prefix(out, markers::SOF0, &payload);
}

fn write_sos_grayscale_8bit(out: &mut Vec<u8>) {
    let payload: [u8; 6] = [
        1, // Ns = 1
        1, 0x00, // component 1 → DC = 0, AC = 0
        0, 63, 0, // Ss = 0, Se = 63, Ah = 0 | Al = 0
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

// ---- Progressive (SOF2) single-component grayscale encoder ---------------
//
// T.81 §G.1.1 permits the progressive coding process at every component
// count `Nf ∈ 1..=4` (one image-wide DCT coefficient grid per component,
// shipped across multiple SOS scans). The 3-component YUV path in
// [`encode_jpeg_progressive`] above generalises straight to a single
// component: every block carries one DC and 63 ACs; the spectral-
// selection decomposition keeps the same `(Ss, Se)` band split. We emit
// three single-component SOS scans:
//
// 1. DC-only       — `Ss=0, Se=0, Ah=0, Al=0`
// 2. AC low band   — `Ss=1, Se=5, Ah=0, Al=0`
// 3. AC high band  — `Ss=6, Se=63, Ah=0, Al=0`
//
// All blocks use the Annex K luma quantiser, the Annex K luma DC + AC
// Huffman tables, and the standard `H = V = 1` shape (so the MCU is
// exactly one 8×8 block, matching the baseline grayscale encoder above).
// No DRI / `RSTn` emission on this path; the 3-component progressive
// path doesn't expose restart markers either, so this stays consistent.

/// Encode a single-component (`Gray8`) buffer as a complete,
/// self-contained progressive (SOF2) JPEG byte stream (`FFD8 … FFD9`).
///
/// * `samples` — row-major 8-bit luma; byte `samples[y * stride + x]` is
///   the pixel at `(x, y)`.
/// * `stride`  — bytes between successive rows; must be at least
///   `width`. Pass `width` for a tightly-packed buffer.
/// * `quality` — JPEG quality factor `1..=100` scaled against the
///   Annex K Q=50 luma base table. Only the luma quantiser is used; the
///   chroma table is never emitted.
///
/// The bitstream layout is `SOI / JFIF APP0 / DQT (luma) / SOF2
/// (Nf = 1, H = V = 1, P = 8) / DHT (Annex K luma DC + AC) / SOS_DC /
/// dc-scan / SOS_AC_low (Ss=1, Se=5) / ac-low-scan / SOS_AC_high
/// (Ss=6, Se=63) / ac-high-scan / EOI`. The output round-trips through
/// any conformant SOF2 decoder — including the matching SOF2 path in
/// `crate::decoder` — as a single-plane `Gray8` frame.
///
/// No restart markers and no metadata segments beyond the default JFIF
/// APP0 are emitted; for APP-pass-through see
/// [`encode_jpeg_progressive_grayscale_with_meta`].
pub fn encode_jpeg_progressive_grayscale(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    quality: u8,
) -> Result<Vec<u8>> {
    encode_jpeg_progressive_grayscale_with_meta(width, height, samples, stride, quality, &[])
}

/// Like [`encode_jpeg_progressive_grayscale`] but inserts `meta`
/// verbatim between the SOI and the first DQT segment. `meta` must
/// contain only APP0..APP15 or COM segments (each starting with
/// `0xFF 0xEn` / `0xFF 0xFE` followed by a big-endian length). Use
/// [`extract_app_segments`] to harvest metadata from a source JPEG.
///
/// When `meta` is empty this is bit-for-bit identical to
/// [`encode_jpeg_progressive_grayscale`].
pub fn encode_jpeg_progressive_grayscale_with_meta(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    quality: u8,
    meta: &[u8],
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid(
            "progressive grayscale encoder: zero-size image",
        ));
    }
    if stride < w {
        return Err(Error::invalid(
            "progressive grayscale encoder: stride smaller than width",
        ));
    }
    if samples.len() < stride * h {
        return Err(Error::invalid(
            "progressive grayscale encoder: samples shorter than stride*h",
        ));
    }

    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let huff = DefaultHuffman::build()?;

    // Build the DCT-quantised coefficient grid: one [i32; 64] per block,
    // walked in raster order. The progressive path needs the *entire*
    // image's coefficients available before the AC scans run, since each
    // AC scan walks the block grid independently.
    let blocks_x = w.div_ceil(8);
    let blocks_y = h.div_ceil(8);
    let mut coefs = vec![[0i32; 64]; blocks_x * blocks_y];
    fill_coef_grid(
        &mut coefs, samples, stride, w, h, blocks_x, blocks_y, &luma_q,
    );

    let mut out: Vec<u8> = Vec::with_capacity(8_192);
    // SOI.
    out.push(0xFF);
    out.push(markers::SOI);
    // Metadata segments (JFIF APP0 fallback when caller provides nothing).
    if meta.is_empty() {
        write_jfif_app0(&mut out);
    } else {
        out.extend_from_slice(meta);
    }
    // DQT — one luma table at id 0.
    write_dqt(&mut out, 0, &luma_q);
    // SOF2 — single component, `H = V = 1`, P = 8, Tq = 0.
    write_sof2_grayscale_8bit(&mut out, w as u16, h as u16);
    // DHT — Annex K luma DC + AC.
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);

    // ---- Scan 1: DC-only, Ss = 0, Se = 0, Ah = 0, Al = 0 ----
    // The 3-component variant uses `write_sos_progressive_dc_interleaved`
    // because all three DCs ship in one scan; the single-component case
    // is identical to a generic AC SOS at (Ss=0, Se=0) — one Cs, no
    // interleaving. Td|Ta nibbles both reference the only luma table id 0
    // — the high nibble (Td, the DC table) is the one actually used in
    // this DC-only scan; the low nibble (Ta, AC table) is unread by the
    // decoder when Ss = 0, Se = 0.
    write_sos_progressive_ac(&mut out, 1, 0, 0, 0);
    {
        let mut bw = BitWriter::new(&mut out);
        let mut prev_dc: i32 = 0;
        for bi in 0..coefs.len() {
            encode_dc(&mut bw, coefs[bi][0], &mut prev_dc, &huff.luma_dc);
        }
        bw.finish();
    }

    // ---- Scan 2: AC low band, Ss = 1, Se = 5 ----
    write_sos_progressive_ac(&mut out, 1, 0, 1, 5);
    write_ac_scan(&mut out, &coefs, coefs.len(), &huff.luma_ac, 1, 5);

    // ---- Scan 3: AC high band, Ss = 6, Se = 63 ----
    write_sos_progressive_ac(&mut out, 1, 0, 6, 63);
    write_ac_scan(&mut out, &coefs, coefs.len(), &huff.luma_ac, 6, 63);

    // EOI.
    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

fn write_sof2_grayscale_8bit(out: &mut Vec<u8>, width: u16, height: u16) {
    // Same component-record shape as `write_sof0_grayscale_8bit`; the
    // only on-wire difference vs. SOF0 is the marker byte. T.81 §G.1.1
    // permits the progressive process at every `Nf ∈ 1..=4`.
    let mut payload = Vec::with_capacity(8 + 3);
    payload.push(8); // P = 8
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(1); // Nf = 1
    payload.push(1); // component id 1 (`Y`)
    payload.push(0x11); // H = 1, V = 1
    payload.push(0); // quantiser table 0
    write_length_prefix(out, markers::SOF2, &payload);
}

// ---- Baseline (SOF0) packed-RGB24 encoder --------------------------------
//
// Three-component, all `H_i = V_i = 1` (MCU = 1 luma block per component =
// one 8×8 pixel tile). Component IDs are ASCII `'R' / 'G' / 'B'` =
// `82 / 71 / 66`, matching the convention exercised by the
// `baseline-rgb-32x32` clean-room fixture under
// `docs/image/jpeg/fixtures/baseline-rgb-32x32/`. All three components
// bind quantiser table 0 (the luma table) so a single DQT segment carries
// the whole image, and all three use the Annex K luma DC + luma AC
// Huffman tables (DC table 0, AC table 0).
//
// An Adobe APP14 segment with `transform = 0` is emitted alongside the
// JFIF APP0 so any conformant decoder that honours the APP14 colour-
// transform flag treats the three planes as plain R / G / B (instead of
// Y / Cb / Cr). The companion baseline-decoder change in
// `crate::decoder` also keys on the component IDs as a fallback, so
// dropping the APP14 (e.g. via custom `meta`) still round-trips.

/// Encode a single packed-RGB24 buffer as a complete, self-contained
/// baseline (SOF0) JPEG byte stream (`FFD8 … FFD9`).
///
/// * `samples` — row-major packed RGB triples: byte 0 = `R`, byte 1 =
///   `G`, byte 2 = `B`, repeating for `width` pixels per row.
/// * `stride` — bytes between successive rows; must be at least
///   `width * 3`. Pass `width * 3` for a tightly-packed buffer.
/// * `quality` — JPEG quality factor `1..=100` scaled against the
///   Annex K Q=50 luma base table. Only the luma quantiser is used; the
///   chroma table is never emitted.
///
/// The output round-trips through any baseline decoder that recognises
/// the Adobe APP14 `transform = 0` flag or that interprets a 3-component
/// SOS with IDs `'R'/'G'/'B'` as plain RGB. Conformant decoders that
/// only know JFIF YCbCr will return the three planes "as is" and treat
/// them as Y / Cb / Cr — callers in that situation should pair the
/// JPEG with explicit colour-space metadata at the container level.
///
/// No restart markers and no metadata segments beyond the default JFIF
/// APP0 + Adobe APP14 are emitted; for the `DRI + RSTn` and APP-
/// pass-through variants see [`encode_jpeg_rgb24_with_opts`] and
/// [`encode_jpeg_rgb24_with_meta`].
pub fn encode_jpeg_rgb24(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    quality: u8,
) -> Result<Vec<u8>> {
    encode_jpeg_rgb24_with_meta(width, height, samples, stride, quality, 0, &[])
}

/// Like [`encode_jpeg_rgb24`] but also emits a DRI segment and cycles
/// `RST0..=RST7` markers every `restart_interval` MCUs during the scan.
/// Passing `0` disables restart marker emission (equivalent to
/// [`encode_jpeg_rgb24`]).
pub fn encode_jpeg_rgb24_with_opts(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    quality: u8,
    restart_interval: u16,
) -> Result<Vec<u8>> {
    encode_jpeg_rgb24_with_meta(
        width,
        height,
        samples,
        stride,
        quality,
        restart_interval,
        &[],
    )
}

/// Like [`encode_jpeg_rgb24_with_opts`] but inserts `meta` verbatim
/// between the SOI and the first DQT segment, replacing the default
/// JFIF APP0 + Adobe APP14 pair. `meta` must contain only APP0..APP15
/// or COM segments (each starting with `0xFF 0xEn` / `0xFF 0xFE`
/// followed by a big-endian length).
///
/// When `meta` is empty this is bit-for-bit identical to
/// [`encode_jpeg_rgb24_with_opts`].
///
/// Note: when `meta` is non-empty the encoder does **not** synthesise
/// an Adobe APP14 segment — the caller is expected to either include
/// one in `meta` or rely on the `'R'/'G'/'B'` component-id fallback to
/// signal the colour space. The companion decoder accepts either.
pub fn encode_jpeg_rgb24_with_meta(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    quality: u8,
    restart_interval: u16,
    meta: &[u8],
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid("RGB24 encoder: zero-size image"));
    }
    let min_stride = w.checked_mul(3).ok_or_else(|| {
        Error::invalid("RGB24 encoder: width * 3 overflow when computing min stride")
    })?;
    if stride < min_stride {
        return Err(Error::invalid(
            "RGB24 encoder: stride smaller than width * 3",
        ));
    }
    if samples.len() < stride.saturating_mul(h) {
        return Err(Error::invalid(
            "RGB24 encoder: samples shorter than stride*h",
        ));
    }

    // Single quantiser table — the luma Q=50 base scaled by `quality`.
    // All three components bind table 0, so the chroma table is never
    // referenced and is not emitted.
    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);
    let huff = DefaultHuffman::build()?;

    let mut out: Vec<u8> = Vec::with_capacity(16_384);
    // SOI.
    out.push(0xFF);
    out.push(markers::SOI);
    // Metadata: caller-supplied `meta` overrides the default APP0 +
    // APP14 pair entirely (so callers passing their own APP14 don't
    // end up with two of them).
    if meta.is_empty() {
        write_jfif_app0(&mut out);
        // Adobe APP14 transform = 0 → "three components are R/G/B"
        // (matching the `baseline-rgb-32x32` clean-room fixture).
        write_adobe_app14(&mut out, 0);
    } else {
        out.extend_from_slice(meta);
    }
    // Single DQT — luma table only, ID 0, precision 0 (8-bit).
    write_dqt(&mut out, 0, &luma_q);
    // SOF0 — three components at IDs 82/71/66 ('R'/'G'/'B'), each H=V=1,
    // all binding QT 0.
    write_sof0_rgb24_8bit(&mut out, w as u16, h as u16);
    // DHT — Annex K luma DC + luma AC only (both classes at ID 0). All
    // three SOS components reference this single pair.
    write_dht(&mut out, 0, 0, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(&mut out, 1, 0, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    // DRI before SOS per T.81 §F.2.2.4.
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    // SOS — three components (R/G/B), each DC=0/AC=0, full AC band, no SA.
    write_sos_rgb24_8bit(&mut out);

    // Scan: one MCU = one 8×8 R block + one 8×8 G block + one 8×8 B block
    // (interleaved). Walked in raster order over MCUs.
    let mcus_x = w.div_ceil(8);
    let mcus_y = h.div_ceil(8);
    let total_mcus = mcus_x.saturating_mul(mcus_y);
    let ri = restart_interval as usize;
    let mut rst_counter: u8 = 0;
    let mut mcus_since_restart: usize = 0;
    let mut mcu_index: usize = 0;

    let mut bw = BitWriter::new(&mut out);
    let mut prev_dc = [0i32; 3];
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            for ch in 0..3 {
                let mut blk = [0.0f32; 64];
                fill_block_packed_channel(&mut blk, samples, stride, w, h, mx * 8, my * 8, ch);
                encode_block(
                    &mut bw,
                    &mut blk,
                    &luma_q,
                    &mut prev_dc[ch],
                    &huff.luma_dc,
                    &huff.luma_ac,
                );
            }

            mcu_index += 1;
            mcus_since_restart += 1;

            // Restart boundary — same semantics as `write_scan`: never
            // emit `RSTn` after the very last MCU because the decoder
            // expects EOI immediately past the final residual.
            if ri != 0 && mcus_since_restart == ri && mcu_index < total_mcus {
                bw.flush_to_byte();
                bw.emit_raw_marker(markers::RST0 + (rst_counter & 0x07));
                rst_counter = rst_counter.wrapping_add(1);
                for d in prev_dc.iter_mut() {
                    *d = 0;
                }
                mcus_since_restart = 0;
            }
        }
    }
    bw.finish();

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

/// SOF0 payload for a 3-component packed-RGB24 baseline JPEG. Component
/// IDs are `'R' / 'G' / 'B'` (82 / 71 / 66), every component declares
/// `H = V = 1`, every component binds quant table 0.
fn write_sof0_rgb24_8bit(out: &mut Vec<u8>, width: u16, height: u16) {
    let mut payload = Vec::with_capacity(8 + 9);
    payload.push(8); // P = 8
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(3); // Nf = 3
                     // R
    payload.push(b'R');
    payload.push(0x11); // H=1 V=1
    payload.push(0); // qt = 0
                     // G
    payload.push(b'G');
    payload.push(0x11);
    payload.push(0);
    // B
    payload.push(b'B');
    payload.push(0x11);
    payload.push(0);
    write_length_prefix(out, markers::SOF0, &payload);
}

/// SOS payload for the 3-component packed-RGB24 baseline JPEG. Every
/// SOS component binds DC table 0 + AC table 0, spectral band
/// `Ss = 0 .. Se = 63`, `Ah = Al = 0` (no successive approximation).
fn write_sos_rgb24_8bit(out: &mut Vec<u8>) {
    let payload: [u8; 10] = [
        3, // Ns = 3
        b'R', 0x00, // component R → DC=0 AC=0
        b'G', 0x00, // component G → DC=0 AC=0
        b'B', 0x00, // component B → DC=0 AC=0
        0, 63, 0, // Ss, Se, Ah|Al
    ];
    write_length_prefix(out, markers::SOS, &payload);
}

/// Fill an 8×8 f32 block from one channel of a packed 3-byte-per-pixel
/// RGB buffer. `channel` selects R/G/B (0/1/2). Edge pixels are
/// replicated for MCU blocks that extend past the picture boundary.
/// Subtracts 128 (level shift) before returning, matching `fill_block`.
#[allow(clippy::too_many_arguments)]
fn fill_block_packed_channel(
    dst: &mut [f32; 64],
    plane: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    channel: usize,
) {
    for j in 0..8 {
        let y = (y0 + j).min(h.saturating_sub(1));
        for i in 0..8 {
            let x = (x0 + i).min(w.saturating_sub(1));
            let v = plane[y * stride + x * 3 + channel] as i32;
            dst[j * 8 + i] = (v - 128) as f32;
        }
    }
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

// ---- Lossless arithmetic JPEG encoder (SOF11, single-component) ----------

/// Write an SOF11 segment for a single-component arithmetic lossless frame.
/// The payload shape is identical to SOF3 (precision / Y / X / Nf /
/// per-component id+sampling+Tq) per T.81 §B.2.2 — only the marker code
/// (`0xCB` vs `0xC3`) distinguishes the lossless arithmetic class.
fn write_sof11_lossless(out: &mut Vec<u8>, width: u16, height: u16, precision: u8) {
    let mut payload = Vec::with_capacity(8 + 3);
    payload.push(precision);
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(1); // Nf — single grayscale component
    payload.push(1); // component id
    payload.push(0x11); // H=1 V=1
    payload.push(0); // Tq (ignored for lossless)
    write_length_prefix(out, markers::SOF11, &payload);
}

/// Write a SOF11 segment for an `nf`-component lossless, arithmetic-coded
/// frame with every component declared `H_i = V_i = 1` (the interleaved
/// layout for multi-component lossless per T.81 §H.1.2). Component
/// identifiers start at 1 and increment by 1. This is the arithmetic-coder
/// counterpart of [`write_sof_lossless_multi`] (which writes SOF3).
fn write_sof11_lossless_multi(out: &mut Vec<u8>, width: u16, height: u16, precision: u8, nf: u8) {
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
    write_length_prefix(out, markers::SOF11, &payload);
}

/// Write a SOF11 segment for a three-component lossless, arithmetic-coded
/// YUV-class frame: the luma component declares the oversampling factors
/// `H_1 × V_1` and the two chroma components are `1 × 1`. This is the
/// arithmetic-coder counterpart of [`write_sof_lossless_yuv`] (which writes
/// SOF3) — precision is fixed at `P = 8` for the subsampled YUV-class path.
fn write_sof11_lossless_yuv(
    out: &mut Vec<u8>,
    width: u16,
    height: u16,
    h_factor: u8,
    v_factor: u8,
) {
    debug_assert!((1..=4).contains(&h_factor) && (1..=4).contains(&v_factor));
    let mut payload = Vec::with_capacity(8 + 3 * 3);
    payload.push(8); // precision (YUV-class lossless is P = 8)
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(3); // Nf
    payload.push(1); // luma component id
    payload.push((h_factor << 4) | (v_factor & 0x0F));
    payload.push(0); // Tq
    payload.push(2); // Cb id
    payload.push(0x11); // 1×1
    payload.push(0);
    payload.push(3); // Cr id
    payload.push(0x11); // 1×1
    payload.push(0);
    write_length_prefix(out, markers::SOF11, &payload);
}

// ---- Sequential arithmetic DCT (SOF9) grayscale encoder ------------------
//
// T.81 §F.2 / Annex D: the SOF9 (extended sequential DCT, arithmetic-coded)
// process shares the SOF0/SOF1 MCU layout but replaces the Huffman entropy
// stage with the binary Q-coder. Each 8×8 block is forward-DCT'd, quantised,
// and emitted as a DC difference (§F.1.4.1) plus a full AC band (§F.1.4) under
// the per-component DC/AC statistics areas. No DAC segment is written, so the
// decoder applies the default conditioning `(L, U) = (0, 1)` and `Kx = 5`
// (§F.1.4.4.1.3). The output round-trips bit-exact through this crate's SOF9
// decode path.

/// Write a SOF9 (extended sequential DCT, arithmetic) frame header for a
/// single 8-bit grayscale component, `H = V = 1`, quantiser table 0.
fn write_sof9_grayscale_8bit(out: &mut Vec<u8>, width: u16, height: u16) {
    let mut payload = Vec::with_capacity(8 + 3);
    payload.push(8); // P = 8
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(1); // Nf = 1
    payload.push(1); // component id 1
    payload.push(0x11); // H = 1, V = 1
    payload.push(0); // quantiser table 0
    write_length_prefix(out, markers::SOF9, &payload);
}

/// Encode a single-component grayscale image as a standalone **sequential,
/// arithmetic-coded DCT** JPEG (SOF9) byte stream — the Q-coder counterpart
/// of [`encode_jpeg_grayscale`]. Forward-DCT + quality-scaled quantisation
/// are identical to the baseline path; only the entropy stage differs.
///
/// `restart_interval` (in MCUs) emits a DRI segment and re-initialises the
/// Q-coder + per-component statistics + DC predictor at every `RST0..=RST7`
/// boundary (§F.2.4.4 / §F.1.4.4.1.5). Passing `0` disables restart markers.
/// Default arithmetic conditioning is used (no DAC). Output is bit-exact
/// through the SOF9 decode path.
pub fn encode_arith_jpeg_grayscale(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    quality: u8,
    restart_interval: u16,
) -> Result<Vec<u8>> {
    use crate::jpeg::arith::{encode_block as arith_encode_block, AcStats, ArithEncoder, DcStats};

    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid("arith-DCT encoder: zero-size image"));
    }
    if stride < w {
        return Err(Error::invalid(
            "arith-DCT encoder: stride smaller than width",
        ));
    }
    if samples.len() < stride * h {
        return Err(Error::invalid(
            "arith-DCT encoder: samples shorter than stride*h",
        ));
    }

    let luma_q = scale_for_quality(&DEFAULT_LUMA_Q50, quality);

    let mut out: Vec<u8> = Vec::with_capacity(8_192);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_dqt(&mut out, 0, &luma_q);
    write_sof9_grayscale_8bit(&mut out, w as u16, h as u16);
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    // SOS — one component (id 1), DC table 0, AC table 0, Ss=0 Se=63 Ah=Al=0.
    write_sos_grayscale_8bit(&mut out);

    let mcus_x = w.div_ceil(8);
    let mcus_y = h.div_ceil(8);
    let total_mcus = mcus_x.saturating_mul(mcus_y);
    let ri = restart_interval as usize;
    let mut rst_counter: u8 = 0;
    let mut mcus_since_restart: usize = 0;
    let mut mcu_index: usize = 0;

    let mut dc = DcStats::new();
    let mut ac = AcStats::new();
    let mut enc = ArithEncoder::new();

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            let mut blk = [0.0f32; 64];
            fill_block(&mut blk, samples, stride, w, h, mx * 8, my * 8);
            fdct8x8(&mut blk);
            // Quantise to natural order, then reorder into zigzag-index
            // order for the arithmetic AC band coder (which indexes by the
            // zigzag position `k`).
            let mut zz = [0i32; 64];
            for k in 0..64 {
                let v = blk[ZIGZAG[k]] / luma_q[ZIGZAG[k]] as f32;
                zz[k] = if v >= 0.0 {
                    (v + 0.5) as i32
                } else {
                    -((-v + 0.5) as i32)
                };
            }
            arith_encode_block(&mut enc, &mut dc, &mut ac, &zz);

            mcu_index += 1;
            mcus_since_restart += 1;

            if ri != 0 && mcus_since_restart == ri && mcu_index < total_mcus {
                // Flush the arithmetic segment, emit RSTn, then re-init the
                // coder + stats + DC predictor for the next interval.
                out.extend_from_slice(&std::mem::take(&mut enc).finish());
                out.push(0xFF);
                out.push(markers::RST0 + (rst_counter & 0x07));
                rst_counter = rst_counter.wrapping_add(1);
                dc = DcStats::new();
                ac = AcStats::new();
                mcus_since_restart = 0;
            }
        }
    }
    out.extend_from_slice(&enc.finish());

    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

/// Encode a single-component grayscale image as a standalone **lossless,
/// arithmetic-coded** JPEG (SOF11) byte stream — the entropy-coder
/// counterpart of [`encode_lossless_jpeg_grayscale`].
///
/// The spatial model is identical to the Huffman lossless path (T.81
/// Annex H predictors over `Ra` / `Rb` / `Rc`), but the prediction
/// differences are coded with the Q-coder arithmetic statistical model of
/// §H.1.2.3 (Table H.3) instead of a Huffman magnitude category. No DAC
/// segment is emitted, so the decoder applies the default conditioning
/// bounds `(L, U) = (0, 1)` per §H.1.2.3.3.
///
/// * `samples` — row-major luminance samples; row `y` starts at byte
///   `y * stride`. For 8-bit precision this carries one byte per sample;
///   for `precision > 8` it must contain little-endian 16-bit samples
///   (two bytes per sample) and `stride` is therefore in bytes.
/// * `precision` — input bits per sample, in `2..=16`.
/// * `predictor` — selector value `1..=7` from T.81 Table H.1.
///
/// Output is bit-exact: the SOF11 decoder recovers every input sample
/// verbatim, including the half-modulus `Di = 32768` case (§H.1.2.2).
/// Point transform is fixed at `Pt = 0` and no restart markers are
/// emitted. For non-zero point transform or restart-marker emission see
/// [`encode_lossless_arith_jpeg_grayscale_with_opts`].
pub fn encode_lossless_arith_jpeg_grayscale(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    precision: u8,
    predictor: u8,
) -> Result<Vec<u8>> {
    encode_lossless_arith_jpeg_grayscale_with_opts(
        width, height, samples, stride, precision, predictor, 0, 0,
    )
}

/// Like [`encode_lossless_arith_jpeg_grayscale`] but also accepts a
/// non-zero point transform `point_transform` (`Pt` — the SOS `Al` field)
/// and a `restart_interval` measured in samples (a lossless MCU is one
/// residual per §H.1.2). Both default-to-zero options reproduce the
/// historical zero-restart, zero-`Pt` output.
///
/// * `point_transform` — `0..=15`, strictly less than `precision`. With
///   `Pt > 0` every input sample is right-shifted by `Pt` before
///   predictive coding; the decoder left-shifts the reconstructed sample
///   back by the same `Pt` on output (the low `Pt` bits are lost).
/// * `restart_interval` — samples between successive `RSTn` markers (`0`
///   disables restart emission). On each boundary the encoder flushes the
///   arithmetic segment, writes a fresh `RST0..=RST7` marker (cycling
///   modulo 8 per §F.1.1.5.2), and re-initialises the statistical model,
///   the neighbour-difference history, and the predictor to the
///   scan-origin default `2^(precision − Pt − 1)` (§H.1.1 / §H.1.2.3.4).
#[allow(clippy::too_many_arguments)]
pub fn encode_lossless_arith_jpeg_grayscale_with_opts(
    width: u32,
    height: u32,
    samples: &[u8],
    stride: usize,
    precision: u8,
    predictor: u8,
    restart_interval: u16,
    point_transform: u8,
) -> Result<Vec<u8>> {
    use crate::jpeg::arith::{encode_lossless_diff, ArithEncoder, LosslessStats};

    if !(2..=16).contains(&precision) {
        return Err(Error::unsupported(format!(
            "lossless-arith encoder: precision {precision} out of range 2..=16"
        )));
    }
    if !(1..=7).contains(&predictor) {
        return Err(Error::invalid(format!(
            "lossless-arith encoder: predictor {predictor} not in 1..=7"
        )));
    }
    if point_transform >= precision {
        return Err(Error::invalid(format!(
            "lossless-arith encoder: point_transform {point_transform} must be < precision {precision}"
        )));
    }
    if point_transform > 15 {
        return Err(Error::invalid(format!(
            "lossless-arith encoder: point_transform {point_transform} > 15 (SOS Al is 4 bits)"
        )));
    }
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid("lossless-arith encoder: zero-size image"));
    }
    let bytes_per_sample = if precision <= 8 { 1 } else { 2 };
    if stride < w * bytes_per_sample {
        return Err(Error::invalid(
            "lossless-arith encoder: stride smaller than width*bytes_per_sample",
        ));
    }
    if samples.len() < stride * h {
        return Err(Error::invalid(
            "lossless-arith encoder: samples shorter than stride*h",
        ));
    }

    // Decode samples into a flat u16 buffer once and apply the point
    // transform up-front so every subsequent value lives in the
    // `precision - Pt` range (matching the decoder's `sample_bits`).
    let pt = point_transform as u32;
    let max_sample: u32 = (1u32 << precision) - 1;
    let mut src = vec![0u32; w * h];
    if precision <= 8 {
        for y in 0..h {
            for x in 0..w {
                let v = samples[y * stride + x] as u32;
                if v > max_sample {
                    return Err(Error::invalid(format!(
                        "lossless-arith encoder: sample {v} exceeds precision-{precision} max {max_sample}"
                    )));
                }
                src[y * w + x] = v >> pt;
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
                        "lossless-arith encoder: sample {v} exceeds precision-{precision} max {max_sample}"
                    )));
                }
                src[y * w + x] = v >> pt;
            }
        }
    }

    let mut out: Vec<u8> = Vec::with_capacity(16_384 + 2 * w * h);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_sof11_lossless(&mut out, w as u16, h as u16, precision);
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    write_sos_lossless(&mut out, predictor, point_transform);

    // Default prediction for the first sample of the scan and after each
    // restart interval (§H.1.2.1). The working precision is `precision − Pt`.
    let sample_bits = precision as u32 - pt;
    let origin: u32 = 1u32 << (sample_bits - 1);

    // Neighbour-difference history feeding `L_Context(Da, Db)` /
    // `X1_Context(Db)`: `prev_diff[x]` is the reduced difference one row up,
    // `cur_diff[x]` one column left in the current row (§H.1.2.3.2).
    let mut prev_diff = vec![0i32; w];
    let mut cur_diff = vec![0i32; w];
    let mut stats = LosslessStats::new();
    let mut enc = ArithEncoder::new();

    let ri = restart_interval as u32;
    let mut rst_counter: u8 = 0;
    let mut samples_since_restart: u32 = 0;
    let total_samples: u64 = w as u64 * h as u64;
    let mut sample_index: u64 = 0;
    // `reset_pred` forces `origin` at scan start and at the first sample of
    // every restart interval; `first_line_y` tracks the row where the
    // current interval began so the "first line uses Ra" rule (§H.1.2.1)
    // applies per-interval, not just to image row 0.
    let mut reset_pred = true;
    let mut first_line_y = 0usize;

    for y in 0..h {
        for x in 0..w {
            // Emit RSTn before the boundary sample (never after the last
            // sample — the decoder expects EOI there per §F.1.1.5.2).
            if ri != 0 && samples_since_restart == ri && sample_index < total_samples {
                out.extend_from_slice(&std::mem::take(&mut enc).finish());
                out.push(0xFF);
                out.push(markers::RST0 + (rst_counter & 0x07));
                rst_counter = rst_counter.wrapping_add(1);
                samples_since_restart = 0;
                stats.reset();
                prev_diff.fill(0);
                cur_diff.fill(0);
                reset_pred = true;
                first_line_y = y;
            }

            let actual = src[y * w + x];
            let pred: u32 = if reset_pred {
                origin
            } else if y == first_line_y {
                // First line of the interval uses Ra regardless of selector.
                src[y * w + x - 1]
            } else if x == 0 {
                // Start of a non-first line uses Rb.
                src[(y - 1) * w + x]
            } else {
                let ra = src[y * w + x - 1];
                let rb = src[(y - 1) * w + x];
                let rc = src[(y - 1) * w + x - 1];
                match predictor {
                    1 => ra,
                    2 => rb,
                    3 => rc,
                    4 => ra.wrapping_add(rb).wrapping_sub(rc),
                    // §H.1 footnote: divide-by-2 is an arithmetic shift.
                    5 => ra.wrapping_add(rb.wrapping_sub(rc) >> 1),
                    6 => rb.wrapping_add(ra.wrapping_sub(rc) >> 1),
                    7 => (ra.wrapping_add(rb)) >> 1,
                    _ => unreachable!(),
                }
            };
            // Modulo-2^16 difference (§H.1.2.1) reduced to the canonical
            // -32768..=32767 representative (the ±32768 alias becomes the
            // last `Sz` decision of Table H.3 inside `encode_lossless_diff`).
            let dm = (actual.wrapping_sub(pred) & 0xFFFF) as i32;
            let dm = if dm >= 0x8000 { dm - 0x10000 } else { dm };
            let da = if x == 0 { 0 } else { cur_diff[x - 1] };
            let db = prev_diff[x];
            encode_lossless_diff(&mut enc, &mut stats, da, db, dm)?;
            cur_diff[x] = dm;

            reset_pred = false;
            sample_index += 1;
            samples_since_restart += 1;
        }
        std::mem::swap(&mut prev_diff, &mut cur_diff);
    }

    out.extend_from_slice(&enc.finish());
    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

/// Encode three planar colour channels as a standalone **multi-component
/// lossless, arithmetic-coded** JPEG (SOF11, three-component interleaved
/// scan) byte stream — the Q-coder counterpart of
/// [`encode_lossless_jpeg_rgb`].
///
/// The spatial model is identical to the Huffman three-component path (the
/// Annex H Table H.1 predictors `1..=7` applied independently per component
/// over each component's own `Ra` / `Rb` / `Rc`), but every prediction
/// difference is coded with the Q-coder arithmetic statistical model of
/// §H.1.2.3 (Table H.3 — `L_Context(Da, Db)` / `X1_Context(Db)`
/// conditioning over each component's neighbouring differences) rather than
/// a Huffman magnitude category. Each component is modelled independently
/// (T.81 §H.1.2: "each component in the scan is modeled independently"), so
/// the encoder keeps one statistics area and one difference-history pair per
/// component while sharing a single arithmetic-coded entropy segment — one
/// residual per component is emitted per pixel position in scan-component
/// order (each component declared `H_i = V_i = 1`, so a lossless MCU is one
/// pixel). No DAC segment is emitted, so the decoder applies the default
/// conditioning bounds `(L, U) = (0, 1)` per §H.1.2.3.3.
///
/// * `planes` — three plane slices in `[ch0, ch1, ch2]` order. The encoder
///   is colour-agnostic: component IDs in the bitstream are 1, 2, 3 in that
///   order and the decoder hands the planes back in the same scan order
///   (see [`encode_lossless_jpeg_rgb`] for the per-precision decoder output
///   shape: `P = 8` → packed `Rgb24`, `P ∈ {10, 12, 14}` → planar
///   `Gbrp*Le`, every other `P` → packed `Rgb48Le`).
/// * `strides` — bytes per row per plane (`width` for `P ≤ 8`,
///   `width * 2` for the little-endian 16-bit wider precisions).
/// * `precision` — input bits per sample, in `2..=16`.
/// * `predictor` — selector value `1..=7` from Table H.1, shared by every
///   component (T.81 §B.2.3: the scan-header predictor selector applies to
///   the whole scan).
///
/// Output is bit-exact for every supported precision and predictor,
/// including the half-modulus `Di = 32768` case (§H.1.2.2). Point transform
/// is fixed at `Pt = 0` and no restart markers are emitted; for non-zero
/// point transform or restart-marker emission see
/// [`encode_lossless_arith_jpeg_rgb_with_opts`].
pub fn encode_lossless_arith_jpeg_rgb(
    width: u32,
    height: u32,
    planes: [&[u8]; 3],
    strides: [usize; 3],
    precision: u8,
    predictor: u8,
) -> Result<Vec<u8>> {
    encode_lossless_arith_jpeg_rgb_with_opts(
        width, height, planes, strides, precision, predictor, 0, 0,
    )
}

/// Like [`encode_lossless_arith_jpeg_rgb`] but also accepts a
/// `restart_interval` (in MCUs — and a 3-component lossless MCU is exactly
/// one pixel position, T.81 §H.1.2 with `H_i = V_i = 1`) and a
/// `point_transform` (`Pt`, the SOS `Al` field's low nibble). Both
/// default-to-zero options reproduce the historical zero-restart, zero-`Pt`
/// output.
///
/// * `point_transform` — `0..=15`, strictly less than `precision`. With
///   `Pt > 0` every input sample is right-shifted by `Pt` before predictive
///   coding; the decoder left-shifts the reconstructed sample back by the
///   same `Pt` on output (the low `Pt` bits are lost).
/// * `restart_interval` — MCUs (= pixel positions) between successive
///   `RSTn` markers (`0` disables restart emission). On each boundary the
///   encoder flushes the arithmetic segment, writes a fresh `RST0..=RST7`
///   marker (cycling modulo 8 per §F.1.1.5.2), and re-initialises **every**
///   component's statistical model, its neighbour-difference history, and
///   its predictor to the scan-origin default `2^(precision − Pt − 1)`
///   (§H.1.1 / §H.1.2.3.4 — each interval restarts with the same defaults
///   as the image start, independently per component).
#[allow(clippy::too_many_arguments)]
pub fn encode_lossless_arith_jpeg_rgb_with_opts(
    width: u32,
    height: u32,
    planes: [&[u8]; 3],
    strides: [usize; 3],
    precision: u8,
    predictor: u8,
    restart_interval: u16,
    point_transform: u8,
) -> Result<Vec<u8>> {
    use crate::jpeg::arith::{encode_lossless_diff, ArithEncoder, LosslessStats};

    if !(2..=16).contains(&precision) {
        return Err(Error::unsupported(format!(
            "lossless-arith RGB encoder: precision {precision} out of range 2..=16"
        )));
    }
    if !(1..=7).contains(&predictor) {
        return Err(Error::invalid(format!(
            "lossless-arith RGB encoder: predictor {predictor} not in 1..=7"
        )));
    }
    if point_transform >= precision {
        return Err(Error::invalid(format!(
            "lossless-arith RGB encoder: point_transform {point_transform} must be < precision {precision}"
        )));
    }
    if point_transform > 15 {
        return Err(Error::invalid(format!(
            "lossless-arith RGB encoder: point_transform {point_transform} > 15 (SOS Al is 4 bits)"
        )));
    }
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid(
            "lossless-arith RGB encoder: zero-size image",
        ));
    }
    let bytes_per_sample = if precision <= 8 { 1 } else { 2 };
    for c in 0..3 {
        if strides[c] < w * bytes_per_sample {
            return Err(Error::invalid(
                "lossless-arith RGB encoder: stride smaller than width*bytes_per_sample",
            ));
        }
        if planes[c].len() < strides[c] * h {
            return Err(Error::invalid(
                "lossless-arith RGB encoder: plane shorter than stride*h",
            ));
        }
    }

    // Decode each plane into a flat u32 buffer once and apply the point
    // transform up-front so all downstream values live in the
    // `precision - Pt` range (matching the decoder's `sample_bits`).
    let pt = point_transform as u32;
    let max_sample: u32 = (1u32 << precision) - 1;
    let mut src: [Vec<u32>; 3] = [vec![0u32; w * h], vec![0u32; w * h], vec![0u32; w * h]];
    if precision <= 8 {
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let v = planes[c][y * strides[c] + x] as u32;
                    if v > max_sample {
                        return Err(Error::invalid(format!(
                            "lossless-arith RGB encoder: sample {v} in plane {c} exceeds precision-{precision} max {max_sample}"
                        )));
                    }
                    src[c][y * w + x] = v >> pt;
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
                            "lossless-arith RGB encoder: sample {v} in plane {c} exceeds precision-{precision} max {max_sample}"
                        )));
                    }
                    src[c][y * w + x] = v >> pt;
                }
            }
        }
    }

    let mut out: Vec<u8> = Vec::with_capacity(16_384 + 3 * 2 * w * h);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_sof11_lossless_multi(&mut out, w as u16, h as u16, precision, 3);
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    write_sos_lossless_multi(&mut out, predictor, 3, point_transform);

    // Default prediction for each component's first sample at scan start and
    // after each restart interval (§H.1.2.1). The working precision is
    // `precision − Pt`.
    let sample_bits = precision as u32 - pt;
    let origin: u32 = 1u32 << (sample_bits - 1);

    // Per-component conditioning history feeding `L_Context(Da, Db)` /
    // `X1_Context(Db)`: `prev_diff[c][x]` is the reduced difference one row
    // up, `cur_diff[c][x]` one column left in the current row (§H.1.2.3.2).
    let mut prev_diff: [Vec<i32>; 3] = [vec![0i32; w], vec![0i32; w], vec![0i32; w]];
    let mut cur_diff: [Vec<i32>; 3] = [vec![0i32; w], vec![0i32; w], vec![0i32; w]];
    let mut stats: [LosslessStats; 3] = [
        LosslessStats::new(),
        LosslessStats::new(),
        LosslessStats::new(),
    ];
    let mut enc = ArithEncoder::new();

    let ri = restart_interval as u32;
    let mut rst_counter: u8 = 0;
    let mut mcus_since_restart: u32 = 0;
    let total_mcus: u64 = w as u64 * h as u64;
    let mut mcu_index: u64 = 0;
    // `reset_pred` forces `origin` at scan start and at the first MCU of
    // every restart interval; `first_line_y` tracks the row where the
    // current interval began so the "first line uses Ra" rule (§H.1.2.1)
    // applies per-interval, not just to image row 0.
    let mut reset_pred = true;
    let mut first_line_y = 0usize;

    for y in 0..h {
        for x in 0..w {
            // Emit RSTn before the boundary MCU (never after the last MCU —
            // the decoder expects EOI there per §F.1.1.5.2).
            if ri != 0 && mcus_since_restart == ri && mcu_index < total_mcus {
                out.extend_from_slice(&std::mem::take(&mut enc).finish());
                out.push(0xFF);
                out.push(markers::RST0 + (rst_counter & 0x07));
                rst_counter = rst_counter.wrapping_add(1);
                mcus_since_restart = 0;
                for s in stats.iter_mut() {
                    s.reset();
                }
                for r in prev_diff.iter_mut() {
                    r.fill(0);
                }
                for r in cur_diff.iter_mut() {
                    r.fill(0);
                }
                reset_pred = true;
                first_line_y = y;
            }

            // Interleaved-MCU order with Hi=Vi=1 across all components: at
            // each (y, x) position, emit one residual for component 0, then
            // 1, then 2 (component IDs in scan-header order).
            for c in 0..3 {
                let plane = &src[c];
                let actual = plane[y * w + x];
                let pred: u32 = if reset_pred {
                    origin
                } else if y == first_line_y {
                    // First line of the interval uses Ra regardless of selector.
                    plane[y * w + x - 1]
                } else if x == 0 {
                    // Start of a non-first line uses Rb.
                    plane[(y - 1) * w + x]
                } else {
                    let ra = plane[y * w + x - 1];
                    let rb = plane[(y - 1) * w + x];
                    let rc = plane[(y - 1) * w + x - 1];
                    match predictor {
                        1 => ra,
                        2 => rb,
                        3 => rc,
                        4 => ra.wrapping_add(rb).wrapping_sub(rc),
                        // §H.1 footnote: divide-by-2 is an arithmetic shift.
                        5 => ra.wrapping_add(rb.wrapping_sub(rc) >> 1),
                        6 => rb.wrapping_add(ra.wrapping_sub(rc) >> 1),
                        7 => (ra.wrapping_add(rb)) >> 1,
                        _ => unreachable!(),
                    }
                };
                // Modulo-2^16 difference (§H.1.2.1) reduced to the canonical
                // -32768..=32767 representative (the ±32768 alias becomes the
                // last `Sz` decision of Table H.3 inside `encode_lossless_diff`).
                let dm = (actual.wrapping_sub(pred) & 0xFFFF) as i32;
                let dm = if dm >= 0x8000 { dm - 0x10000 } else { dm };
                let da = if x == 0 { 0 } else { cur_diff[c][x - 1] };
                let db = prev_diff[c][x];
                encode_lossless_diff(&mut enc, &mut stats[c], da, db, dm)?;
                cur_diff[c][x] = dm;
            }

            reset_pred = false;
            mcu_index += 1;
            mcus_since_restart += 1;
        }
        std::mem::swap(&mut prev_diff, &mut cur_diff);
    }

    out.extend_from_slice(&enc.finish());
    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

/// Encode four planar 8-bit colour channels (C, M, Y, K — or any four
/// independent monochrome planes) as a standalone **four-component lossless,
/// arithmetic-coded** JPEG (SOF11, interleaved scan) byte stream — the
/// Q-coder counterpart of [`encode_lossless_jpeg_cmyk`].
///
/// The spatial model is identical to the Huffman four-component path (the
/// Annex H Table H.1 predictors `1..=7` applied independently per component
/// over each component's own `Ra` / `Rb` / `Rc`), but every prediction
/// difference is coded with the Q-coder arithmetic statistical model of
/// §H.1.2.3 (Table H.3 — `L_Context(Da, Db)` / `X1_Context(Db)`
/// conditioning over each component's neighbouring differences) rather than a
/// Huffman magnitude category. Each component is modelled independently (T.81
/// §H.1.2), so the encoder keeps one statistics area and one
/// difference-history pair per component while sharing a single
/// arithmetic-coded entropy segment — one residual per component is emitted
/// per pixel position in scan-component order (component IDs 1, 2, 3, 4; each
/// declared `H_i = V_i = 1`, so a lossless MCU is one pixel). No DAC segment
/// is emitted, so the decoder applies the default conditioning bounds
/// `(L, U) = (0, 1)` per §H.1.2.3.3.
///
/// * `planes` — four plane slices in scan order. The encoder is colour-
///   agnostic: the caller decides what the four planes represent and the
///   decoder hands them back in the same SOS scan order, then applies the
///   APP14 colour transform (if any) on output.
/// * `strides` — bytes per row per plane; must be at least `width`.
/// * `predictor` — selector value `1..=7` from Table H.1, shared by every
///   component (T.81 §B.2.3).
/// * `adobe_transform` — Adobe APP14 colour-transform flag, identical to
///   [`encode_lossless_jpeg_cmyk`]:
///   * `None`     — no APP14 segment; samples passed through unchanged
///     ("regular" CMYK on the decoder side).
///   * `Some(0)`  — Adobe CMYK convention: every input byte is inverted
///     before predictive coding; the decoder un-inverts on output.
///   * `Some(2)`  — Adobe YCCK: the caller passes `[Y, Cb, Cr, K]` planes
///     and the encoder inverts only the K plane before coding so the
///     decoder's YCCK → CMYK un-inversion lands on the same K value.
///
/// Precision is fixed at 8 bits (the only depth the workspace `PixelFormat`
/// enum's `Cmyk` variant covers). Point transform is fixed at `Pt = 0` and no
/// restart markers are emitted. For non-zero point transform or
/// restart-marker emission see
/// [`encode_lossless_arith_jpeg_cmyk_with_opts`].
pub fn encode_lossless_arith_jpeg_cmyk(
    width: u32,
    height: u32,
    planes: [&[u8]; 4],
    strides: [usize; 4],
    predictor: u8,
    adobe_transform: Option<u8>,
) -> Result<Vec<u8>> {
    encode_lossless_arith_jpeg_cmyk_with_opts(
        width,
        height,
        planes,
        strides,
        predictor,
        adobe_transform,
        0,
        0,
    )
}

/// Like [`encode_lossless_arith_jpeg_cmyk`] but also accepts a
/// `restart_interval` (in MCUs — a four-component lossless MCU is exactly one
/// pixel position, T.81 §H.1.2 with `H_i = V_i = 1`) and a `point_transform`
/// (`Pt`, the SOS `Al` field's low nibble). Both default-to-zero options
/// reproduce the historical zero-restart, zero-`Pt` output.
///
/// * `point_transform` — `0..=7`, strictly less than the fixed precision 8.
///   With `Pt > 0` every (post-Adobe-inversion) input sample is
///   right-shifted by `Pt` before predictive coding; the decoder left-shifts
///   the reconstructed sample back by the same `Pt` on output.
/// * `restart_interval` — MCUs (= pixel positions) between successive `RSTn`
///   markers (`0` disables restart emission). On each boundary the encoder
///   flushes the arithmetic segment, writes a fresh `RST0..=RST7` marker
///   (cycling modulo 8 per §F.1.1.5.2), and re-initialises **every**
///   component's statistical model, its neighbour-difference history, and its
///   predictor to the scan-origin default `2^(8 − Pt − 1)` (§H.1.1 /
///   §H.1.2.3.4 — each interval restarts with the same defaults as the image
///   start, independently per component).
#[allow(clippy::too_many_arguments)]
pub fn encode_lossless_arith_jpeg_cmyk_with_opts(
    width: u32,
    height: u32,
    planes: [&[u8]; 4],
    strides: [usize; 4],
    predictor: u8,
    adobe_transform: Option<u8>,
    restart_interval: u16,
    point_transform: u8,
) -> Result<Vec<u8>> {
    use crate::jpeg::arith::{encode_lossless_diff, ArithEncoder, LosslessStats};

    if !(1..=7).contains(&predictor) {
        return Err(Error::invalid(format!(
            "lossless-arith CMYK encoder: predictor {predictor} not in 1..=7"
        )));
    }
    if point_transform >= 8 {
        return Err(Error::invalid(format!(
            "lossless-arith CMYK encoder: point_transform {point_transform} must be < precision 8"
        )));
    }
    match adobe_transform {
        None | Some(0) | Some(2) => {}
        Some(other) => {
            return Err(Error::invalid(format!(
                "lossless-arith CMYK encoder: adobe_transform = {other} (only None / Some(0) / Some(2) are supported)"
            )));
        }
    }
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid(
            "lossless-arith CMYK encoder: zero-size image",
        ));
    }
    for c in 0..4 {
        if strides[c] < w {
            return Err(Error::invalid(
                "lossless-arith CMYK encoder: stride smaller than width",
            ));
        }
        if planes[c].len() < strides[c] * h {
            return Err(Error::invalid(
                "lossless-arith CMYK encoder: plane shorter than stride*h",
            ));
        }
    }

    // Apply the on-the-wire transform implied by `adobe_transform` so the
    // residuals coded below match what the matching decoder expects. The
    // decoder will re-invert (and YCCK→CMYK convert) on output (mirrors the
    // SOF3 four-component encoder's inversion policy).
    let invert_all = matches!(adobe_transform, Some(0));
    let invert_k_only = matches!(adobe_transform, Some(2));
    let pt = point_transform as u32;

    // Decode each plane into a flat u32 buffer once, applying the Adobe
    // inversion and the point transform up-front so all downstream values
    // live in the `8 − Pt` range (matching the decoder's `sample_bits`).
    let mut src: [Vec<u32>; 4] = [
        vec![0u32; w * h],
        vec![0u32; w * h],
        vec![0u32; w * h],
        vec![0u32; w * h],
    ];
    for c in 0..4 {
        let invert_this = invert_all || (invert_k_only && c == 3);
        for y in 0..h {
            for x in 0..w {
                let v = planes[c][y * strides[c] + x] as u32;
                let v = if invert_this { 255 - v } else { v };
                src[c][y * w + x] = v >> pt;
            }
        }
    }

    let mut out: Vec<u8> = Vec::with_capacity(16_384 + 4 * w * h);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    if let Some(tx) = adobe_transform {
        write_adobe_app14(&mut out, tx);
    }
    write_sof11_lossless_multi(&mut out, w as u16, h as u16, 8, 4);
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    write_sos_lossless_multi(&mut out, predictor, 4, point_transform);

    // Default prediction for each component's first sample at scan start and
    // after each restart interval (§H.1.2.1). Working precision is `8 − Pt`.
    let sample_bits = 8u32 - pt;
    let origin: u32 = 1u32 << (sample_bits - 1);

    // Per-component conditioning history feeding `L_Context(Da, Db)` /
    // `X1_Context(Db)`: `prev_diff[c][x]` is the reduced difference one row
    // up, `cur_diff[c][x]` one column left in the current row (§H.1.2.3.2).
    let mut prev_diff: [Vec<i32>; 4] = [vec![0i32; w], vec![0i32; w], vec![0i32; w], vec![0i32; w]];
    let mut cur_diff: [Vec<i32>; 4] = [vec![0i32; w], vec![0i32; w], vec![0i32; w], vec![0i32; w]];
    let mut stats: [LosslessStats; 4] = [
        LosslessStats::new(),
        LosslessStats::new(),
        LosslessStats::new(),
        LosslessStats::new(),
    ];
    let mut enc = ArithEncoder::new();

    let ri = restart_interval as u32;
    let mut rst_counter: u8 = 0;
    let mut mcus_since_restart: u32 = 0;
    let total_mcus: u64 = w as u64 * h as u64;
    let mut mcu_index: u64 = 0;
    let mut reset_pred = true;
    let mut first_line_y = 0usize;

    for y in 0..h {
        for x in 0..w {
            // Emit RSTn before the boundary MCU (never after the last MCU —
            // the decoder expects EOI there per §F.1.1.5.2).
            if ri != 0 && mcus_since_restart == ri && mcu_index < total_mcus {
                out.extend_from_slice(&std::mem::take(&mut enc).finish());
                out.push(0xFF);
                out.push(markers::RST0 + (rst_counter & 0x07));
                rst_counter = rst_counter.wrapping_add(1);
                mcus_since_restart = 0;
                for s in stats.iter_mut() {
                    s.reset();
                }
                for r in prev_diff.iter_mut() {
                    r.fill(0);
                }
                for r in cur_diff.iter_mut() {
                    r.fill(0);
                }
                reset_pred = true;
                first_line_y = y;
            }

            // Interleaved-MCU order with Hi=Vi=1 across all components: at
            // each (y, x) position, emit one residual per component in
            // scan-header (component-id) order.
            for c in 0..4 {
                let plane = &src[c];
                let actual = plane[y * w + x];
                let pred: u32 = if reset_pred {
                    origin
                } else if y == first_line_y {
                    // First line of the interval uses Ra regardless of selector.
                    plane[y * w + x - 1]
                } else if x == 0 {
                    // Start of a non-first line uses Rb.
                    plane[(y - 1) * w + x]
                } else {
                    let ra = plane[y * w + x - 1];
                    let rb = plane[(y - 1) * w + x];
                    let rc = plane[(y - 1) * w + x - 1];
                    match predictor {
                        1 => ra,
                        2 => rb,
                        3 => rc,
                        4 => ra.wrapping_add(rb).wrapping_sub(rc),
                        // §H.1 footnote: divide-by-2 is an arithmetic shift.
                        5 => ra.wrapping_add(rb.wrapping_sub(rc) >> 1),
                        6 => rb.wrapping_add(ra.wrapping_sub(rc) >> 1),
                        7 => (ra.wrapping_add(rb)) >> 1,
                        _ => unreachable!(),
                    }
                };
                // Modulo-2^16 difference (§H.1.2.1) reduced to the canonical
                // -32768..=32767 representative (the ±32768 alias becomes the
                // last `Sz` decision of Table H.3 inside `encode_lossless_diff`).
                let dm = (actual.wrapping_sub(pred) & 0xFFFF) as i32;
                let dm = if dm >= 0x8000 { dm - 0x10000 } else { dm };
                let da = if x == 0 { 0 } else { cur_diff[c][x - 1] };
                let db = prev_diff[c][x];
                encode_lossless_diff(&mut enc, &mut stats[c], da, db, dm)?;
                cur_diff[c][x] = dm;
            }

            reset_pred = false;
            mcu_index += 1;
            mcus_since_restart += 1;
        }
        std::mem::swap(&mut prev_diff, &mut cur_diff);
    }

    out.extend_from_slice(&enc.finish());
    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
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

/// Encode three planar **8-bit YUV-class** channels as a standalone
/// **subsampled three-component lossless** JPEG (SOF3, interleaved scan)
/// byte stream, with the luma component oversampled relative to the two
/// chroma components.
///
/// This is the lossless counterpart of the lossy 4:2:2 / 4:2:0 / 4:1:1
/// paths: the luma (first) component declares sampling factors
/// `(h_factor, v_factor)` and both chroma components declare `1×1`, so a
/// minimum coded unit carries `h_factor × v_factor` luma samples followed
/// by one Cb and one Cr sample (T.81 A.2.3 interleaved data ordering, with
/// the lossless data unit equal to one sample per E.1.1). Supported luma
/// factors and the pixel format they round-trip to:
///
/// | `(h_factor, v_factor)` | chroma resolution        | decoder format |
/// |------------------------|--------------------------|----------------|
/// | `(1, 1)`               | full (= luma)            | `Yuv444P`      |
/// | `(2, 1)`               | half width               | `Yuv422P`      |
/// | `(2, 2)`               | half width, half height  | `Yuv420P`      |
/// | `(4, 1)`               | quarter width            | `Yuv411P`      |
///
/// * `y_plane` / `y_stride` — full-resolution luma, `width × height`.
/// * `cb_plane` / `cr_plane` (+ strides) — chroma at the subsampled
///   resolution `ceil(width × h / h) × ceil(height × v / v)` i.e.
///   `ceil(width / (h_max / 1)) × …`. Concretely: width is divided by
///   `h_factor` (rounding up) and height by `v_factor` (rounding up).
/// * `predictor` — Table H.1 selector `1..=7`, shared by every component.
/// * `restart_interval` — `RSTn` cadence in **MCUs** (0 disables).
/// * `point_transform` — `Pt`, `0..=15` and `< 8`.
///
/// Per A.2.4 each component is padded to a whole number of MCUs before
/// predictive coding by replicating its right-most column / bottom row; the
/// decoder crops the added samples back off on output. Bit-exact round-trip
/// vs. `decode_lossless_scan` for every supported factor / predictor /
/// restart / Pt combination.
#[allow(clippy::too_many_arguments)]
pub fn encode_lossless_jpeg_yuv_with_opts(
    width: u32,
    height: u32,
    y_plane: &[u8],
    y_stride: usize,
    cb_plane: &[u8],
    cb_stride: usize,
    cr_plane: &[u8],
    cr_stride: usize,
    h_factor: u8,
    v_factor: u8,
    predictor: u8,
    restart_interval: u16,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if !(1..=7).contains(&predictor) {
        return Err(Error::invalid(format!(
            "lossless YUV encoder: predictor {predictor} not in 1..=7"
        )));
    }
    if point_transform >= 8 {
        return Err(Error::invalid(format!(
            "lossless YUV encoder: point_transform {point_transform} must be < precision 8"
        )));
    }
    if !matches!((h_factor, v_factor), (1, 1) | (2, 1) | (2, 2) | (4, 1)) {
        return Err(Error::unsupported(format!(
            "lossless YUV encoder: unsupported luma sampling {h_factor}x{v_factor} (supported: 1x1, 2x1, 2x2, 4x1)"
        )));
    }
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid("lossless YUV encoder: zero-size image"));
    }
    let hf = h_factor as usize;
    let vf = v_factor as usize;
    let cw = w.div_ceil(hf); // chroma width
    let ch = h.div_ceil(vf); // chroma height
                             // Bounds: each supplied plane must hold its (sub)sampled extent.
    if y_stride < w || y_plane.len() < y_stride * h {
        return Err(Error::invalid("lossless YUV encoder: Y plane too small"));
    }
    if cb_stride < cw || cb_plane.len() < cb_stride * ch {
        return Err(Error::invalid("lossless YUV encoder: Cb plane too small"));
    }
    if cr_stride < cw || cr_plane.len() < cr_stride * ch {
        return Err(Error::invalid("lossless YUV encoder: Cr plane too small"));
    }

    let dc_huff = HuffTable::build(&STD_DC_LOSSLESS_BITS, &STD_DC_LOSSLESS_VALS)?;

    let mut out: Vec<u8> = Vec::with_capacity(16_384 + 2 * w * h);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_sof_lossless_yuv(&mut out, w as u16, h as u16, h_factor, v_factor);
    write_dht(&mut out, 0, 0, &STD_DC_LOSSLESS_BITS, &STD_DC_LOSSLESS_VALS);
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    write_sos_lossless_multi(&mut out, predictor, 3, point_transform);

    let pt = point_transform as u32;

    // MCU grid: ceil(width / h_max) × ceil(height / v_max) with
    // h_max = h_factor, v_max = v_factor (luma carries the oversampling,
    // chroma are 1×1). Each component is padded out to a whole number of
    // MCUs by edge replication (A.2.4 NOTE).
    let mcus_x = w.div_ceil(hf);
    let mcus_y = h.div_ceil(vf);
    let y_gw = mcus_x * hf; // padded luma grid width
    let y_gh = mcus_y * vf;
    let c_gw = mcus_x; // padded chroma grid (1×1) width
    let c_gh = mcus_y;

    // Build MCU-padded, point-transform-shifted component grids. Edge
    // replication: a sample beyond the component's true extent takes the
    // value of the nearest in-bounds sample (clamp the index).
    let build_grid =
        |plane: &[u8], stride: usize, src_w: usize, src_h: usize, gw: usize, gh: usize| {
            let mut grid = vec![0u8; gw * gh];
            for gy in 0..gh {
                let sy = gy.min(src_h - 1);
                for gx in 0..gw {
                    let sx = gx.min(src_w - 1);
                    grid[gy * gw + gx] = plane[sy * stride + sx] >> pt;
                }
            }
            grid
        };
    let grids: [Vec<u8>; 3] = [
        build_grid(y_plane, y_stride, w, h, y_gw, y_gh),
        build_grid(cb_plane, cb_stride, cw, ch, c_gw, c_gh),
        build_grid(cr_plane, cr_stride, cw, ch, c_gw, c_gh),
    ];
    let grid_w = [y_gw, c_gw, c_gw];
    let comp_hf = [hf, 1, 1];
    let comp_vf = [vf, 1, 1];

    // Origin / restart bookkeeping (identical model to the RGB path; only
    // the MCU sample layout differs).
    let sample_bits = 8u32 - pt;
    let origin: i32 = 1i32 << (sample_bits - 1);
    let ri = restart_interval as u32;
    let mut rst_counter: u8 = 0;
    let mut mcus_since_restart: u32 = 0;
    let total_mcus: u64 = mcus_x as u64 * mcus_y as u64;
    let mut mcu_index: u64 = 0;
    let mut reset_pred = true;

    let mut bw = BitWriter::new(&mut out);
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // T.81 A.2.3: within an MCU emit component 0's H×V block (row
            // major), then component 1's single sample, then component 2's.
            for c in 0..3 {
                let gw = grid_w[c];
                let grid = &grids[c];
                for sy in 0..comp_vf[c] {
                    for sx in 0..comp_hf[c] {
                        let gx = mx * comp_hf[c] + sx;
                        let gy = my * comp_vf[c] + sy;
                        let actual = grid[gy * gw + gx] as i32;
                        let pred: i32 = if reset_pred {
                            origin
                        } else if gy == 0 {
                            grid[gy * gw + gx - 1] as i32
                        } else if gx == 0 {
                            grid[(gy - 1) * gw + gx] as i32
                        } else {
                            let ra = grid[gy * gw + gx - 1] as i32;
                            let rb = grid[(gy - 1) * gw + gx] as i32;
                            let rc = grid[(gy - 1) * gw + gx - 1] as i32;
                            match predictor {
                                1 => ra,
                                2 => rb,
                                3 => rc,
                                4 => ra + rb - rc,
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
                        if s != 0 && s != 16 {
                            bw.write_bits(bits, s as u32);
                        }
                    }
                }
            }

            reset_pred = false;
            mcu_index += 1;
            mcus_since_restart += 1;

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

/// Convenience wrapper for [`encode_lossless_jpeg_yuv_with_opts`] with the
/// default predictor 1 (Ra / left), no restart markers, and `Pt = 0`.
#[allow(clippy::too_many_arguments)]
pub fn encode_lossless_jpeg_yuv(
    width: u32,
    height: u32,
    y_plane: &[u8],
    y_stride: usize,
    cb_plane: &[u8],
    cb_stride: usize,
    cr_plane: &[u8],
    cr_stride: usize,
    h_factor: u8,
    v_factor: u8,
    predictor: u8,
) -> Result<Vec<u8>> {
    encode_lossless_jpeg_yuv_with_opts(
        width, height, y_plane, y_stride, cb_plane, cb_stride, cr_plane, cr_stride, h_factor,
        v_factor, predictor, 0, 0,
    )
}

/// Encode three planar 8-bit YUV-class channels as a standalone
/// **three-component lossless, arithmetic-coded** JPEG (SOF11, subsampled
/// interleaved scan) byte stream — the Q-coder counterpart of
/// [`encode_lossless_jpeg_yuv_with_opts`].
///
/// The luma component may be oversampled `1×1` / `2×1` / `2×2` / `4×1` with
/// both chroma components at `1×1`; the scan walks the T.81 §A.2.3
/// interleaved-MCU ordering (component 0's `H_1 × V_1` block, then component
/// 1's single sample, then component 2's, per MCU). Each component is modelled
/// independently per §H.1.2 over its **own** sample grid (its own `Ra` / `Rb` /
/// `Rc` neighbours and its own `L_Context(Da, Db)` / `X1_Context(Db)`
/// difference history), and every prediction difference is coded with the
/// Q-coder statistical model of §H.1.2.3 (Table H.3). The per-component grids
/// are padded out to a whole number of MCUs by edge replication (§A.2.4 NOTE);
/// the decoder crops each component back to its true extent on output. No DAC
/// segment is emitted, so the decoder applies the default conditioning bounds
/// `(L, U) = (0, 1)` per §H.1.2.3.3.
///
/// Precision is fixed at 8 bits (the only depth the subsampled YUV-class path
/// covers). The `_with_opts` form adds a `restart_interval` (in MCUs — one
/// `H_1 × V_1` luma block plus two chroma samples) and a `point_transform`
/// (`Pt`, the SOS `Al` field's low nibble): on each restart boundary the
/// encoder flushes the arithmetic segment, writes a fresh `RST0..=RST7` marker
/// (cycling modulo 8 per §F.1.1.5.2), and re-initialises every component's
/// statistical model, difference history, and predictor to the scan-origin
/// default `2^(8 − Pt − 1)` (§H.1.1 / §H.1.2.3.4). With `Pt > 0` every input
/// sample is right-shifted by `Pt` before predictive coding.
#[allow(clippy::too_many_arguments)]
pub fn encode_lossless_arith_jpeg_yuv_with_opts(
    width: u32,
    height: u32,
    y_plane: &[u8],
    y_stride: usize,
    cb_plane: &[u8],
    cb_stride: usize,
    cr_plane: &[u8],
    cr_stride: usize,
    h_factor: u8,
    v_factor: u8,
    predictor: u8,
    restart_interval: u16,
    point_transform: u8,
) -> Result<Vec<u8>> {
    use crate::jpeg::arith::{encode_lossless_diff, ArithEncoder, LosslessStats};

    if !(1..=7).contains(&predictor) {
        return Err(Error::invalid(format!(
            "lossless-arith YUV encoder: predictor {predictor} not in 1..=7"
        )));
    }
    if point_transform >= 8 {
        return Err(Error::invalid(format!(
            "lossless-arith YUV encoder: point_transform {point_transform} must be < precision 8"
        )));
    }
    if !matches!((h_factor, v_factor), (1, 1) | (2, 1) | (2, 2) | (4, 1)) {
        return Err(Error::unsupported(format!(
            "lossless-arith YUV encoder: unsupported luma sampling {h_factor}x{v_factor} (supported: 1x1, 2x1, 2x2, 4x1)"
        )));
    }
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid(
            "lossless-arith YUV encoder: zero-size image",
        ));
    }
    let hf = h_factor as usize;
    let vf = v_factor as usize;
    let cw = w.div_ceil(hf); // chroma width
    let ch = h.div_ceil(vf); // chroma height
    if y_stride < w || y_plane.len() < y_stride * h {
        return Err(Error::invalid(
            "lossless-arith YUV encoder: Y plane too small",
        ));
    }
    if cb_stride < cw || cb_plane.len() < cb_stride * ch {
        return Err(Error::invalid(
            "lossless-arith YUV encoder: Cb plane too small",
        ));
    }
    if cr_stride < cw || cr_plane.len() < cr_stride * ch {
        return Err(Error::invalid(
            "lossless-arith YUV encoder: Cr plane too small",
        ));
    }

    let mut out: Vec<u8> = Vec::with_capacity(16_384 + 3 * w * h);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    write_sof11_lossless_yuv(&mut out, w as u16, h as u16, h_factor, v_factor);
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    write_sos_lossless_multi(&mut out, predictor, 3, point_transform);

    let pt = point_transform as u32;

    // MCU grid: ceil(width / h_max) × ceil(height / v_max) with
    // h_max = h_factor, v_max = v_factor (luma carries the oversampling,
    // chroma are 1×1). Each component is padded out to a whole number of MCUs
    // by edge replication (§A.2.4 NOTE).
    let mcus_x = w.div_ceil(hf);
    let mcus_y = h.div_ceil(vf);
    let y_gw = mcus_x * hf; // padded luma grid width
    let y_gh = mcus_y * vf;
    let c_gw = mcus_x; // padded chroma grid (1×1) width
    let c_gh = mcus_y;

    // Build MCU-padded, point-transform-shifted component grids. Edge
    // replication: a sample beyond the component's true extent takes the value
    // of the nearest in-bounds sample (clamp the index).
    let build_grid =
        |plane: &[u8], stride: usize, src_w: usize, src_h: usize, gw: usize, gh: usize| {
            let mut grid = vec![0u32; gw * gh];
            for gy in 0..gh {
                let sy = gy.min(src_h - 1);
                for gx in 0..gw {
                    let sx = gx.min(src_w - 1);
                    grid[gy * gw + gx] = (plane[sy * stride + sx] as u32) >> pt;
                }
            }
            grid
        };
    let grids: [Vec<u32>; 3] = [
        build_grid(y_plane, y_stride, w, h, y_gw, y_gh),
        build_grid(cb_plane, cb_stride, cw, ch, c_gw, c_gh),
        build_grid(cr_plane, cr_stride, cw, ch, c_gw, c_gh),
    ];
    let grid_w = [y_gw, c_gw, c_gw];
    let comp_hf = [hf, 1, 1];
    let comp_vf = [vf, 1, 1];

    // Default prediction for each component's first sample at scan start and
    // after each restart interval (§H.1.2.1). Working precision is `8 − Pt`.
    let sample_bits = 8u32 - pt;
    let origin: u32 = 1u32 << (sample_bits - 1);

    // Per-component conditioning history as a full per-component difference
    // grid (one entry per grid sample). The §A.2.3 MCU walk visits a
    // component's samples out of plain raster order (a luma block spans `vf`
    // rows per MCU and successive MCUs fill columns left-to-right), so the
    // `Da` (left) / `Db` (above) conditioning neighbours are addressed by
    // absolute grid coordinate — exactly as the predictor reads the sample
    // grid — rather than by a sliding two-row window. Unwritten cells hold the
    // §H.1.2.3.1 "zero outside the reconstructed region" default. Reset to all
    // zero at scan start and at each restart (§H.1.2.3.4).
    let mut diff_grid: [Vec<i32>; 3] = [
        vec![0i32; grids[0].len()],
        vec![0i32; grids[1].len()],
        vec![0i32; grids[2].len()],
    ];
    let mut stats: [LosslessStats; 3] = [
        LosslessStats::new(),
        LosslessStats::new(),
        LosslessStats::new(),
    ];
    let mut enc = ArithEncoder::new();

    let ri = restart_interval as u32;
    let mut rst_counter: u8 = 0;
    let mut mcus_since_restart: u32 = 0;
    let total_mcus: u64 = mcus_x as u64 * mcus_y as u64;
    let mut mcu_index: u64 = 0;
    // `reset_pred` forces `origin` at scan start and at the first MCU of every
    // restart interval. The "first line uses Ra" rule (§H.1.2.1) is tracked
    // per component via `first_row[c]` — the grid row index where the current
    // interval began for that component.
    let mut reset_pred = true;
    let mut first_row: [usize; 3] = [0, 0, 0];

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // Emit RSTn before the boundary MCU (never after the last MCU —
            // the decoder expects EOI there per §F.1.1.5.2).
            if ri != 0 && mcus_since_restart == ri && mcu_index < total_mcus {
                out.extend_from_slice(&std::mem::take(&mut enc).finish());
                out.push(0xFF);
                out.push(markers::RST0 + (rst_counter & 0x07));
                rst_counter = rst_counter.wrapping_add(1);
                mcus_since_restart = 0;
                for s in stats.iter_mut() {
                    s.reset();
                }
                for g in diff_grid.iter_mut() {
                    g.fill(0);
                }
                reset_pred = true;
                first_row = [my * comp_vf[0], my, my];
            }

            // §A.2.3 interleaved-MCU order: component 0's H×V block (row
            // major), then component 1's single sample, then component 2's.
            for c in 0..3 {
                let gw = grid_w[c];
                let grid = &grids[c];
                for sy in 0..comp_vf[c] {
                    for sx in 0..comp_hf[c] {
                        let gx = mx * comp_hf[c] + sx;
                        let gy = my * comp_vf[c] + sy;
                        let actual = grid[gy * gw + gx];
                        let pred: u32 = if reset_pred && mx == 0 && sx == 0 && gy == first_row[c] {
                            // Very first sample of the scan / interval.
                            origin
                        } else if gy == first_row[c] {
                            // First grid line of the scan / interval uses Ra.
                            grid[gy * gw + gx - 1]
                        } else if gx == 0 {
                            // Start of a non-first line uses Rb.
                            grid[(gy - 1) * gw + gx]
                        } else {
                            let ra = grid[gy * gw + gx - 1];
                            let rb = grid[(gy - 1) * gw + gx];
                            let rc = grid[(gy - 1) * gw + gx - 1];
                            match predictor {
                                1 => ra,
                                2 => rb,
                                3 => rc,
                                4 => ra.wrapping_add(rb).wrapping_sub(rc),
                                5 => ra.wrapping_add(rb.wrapping_sub(rc) >> 1),
                                6 => rb.wrapping_add(ra.wrapping_sub(rc) >> 1),
                                7 => (ra.wrapping_add(rb)) >> 1,
                                _ => unreachable!(),
                            }
                        };
                        // Modulo-2^16 difference (§H.1.2.1) reduced to the
                        // canonical -32768..=32767 representative.
                        let dm = (actual.wrapping_sub(pred) & 0xFFFF) as i32;
                        let dm = if dm >= 0x8000 { dm - 0x10000 } else { dm };
                        let dg = &diff_grid[c];
                        let da = if gx == 0 { 0 } else { dg[gy * gw + gx - 1] };
                        let db = if gy == 0 { 0 } else { dg[(gy - 1) * gw + gx] };
                        encode_lossless_diff(&mut enc, &mut stats[c], da, db, dm)?;
                        diff_grid[c][gy * gw + gx] = dm;
                    }
                }
            }

            reset_pred = false;
            mcu_index += 1;
            mcus_since_restart += 1;
        }
    }

    out.extend_from_slice(&enc.finish());
    out.push(0xFF);
    out.push(markers::EOI);
    Ok(out)
}

/// Convenience wrapper for [`encode_lossless_arith_jpeg_yuv_with_opts`] with
/// the default predictor 1 (Ra / left), no restart markers, and `Pt = 0`.
#[allow(clippy::too_many_arguments)]
pub fn encode_lossless_arith_jpeg_yuv(
    width: u32,
    height: u32,
    y_plane: &[u8],
    y_stride: usize,
    cb_plane: &[u8],
    cb_stride: usize,
    cr_plane: &[u8],
    cr_stride: usize,
    h_factor: u8,
    v_factor: u8,
    predictor: u8,
) -> Result<Vec<u8>> {
    encode_lossless_arith_jpeg_yuv_with_opts(
        width, height, y_plane, y_stride, cb_plane, cb_stride, cr_plane, cr_stride, h_factor,
        v_factor, predictor, 0, 0,
    )
}

/// Encode four planar 8-bit colour channels as a standalone **four-component
/// lossless** JPEG (SOF3, four-component interleaved scan) byte stream.
///
/// The four-component lossless path is the natural extension of the
/// three-component (RGB-class) encoder to a fourth independently-modelled
/// component (T.81 §H.1.2 — "each component in the scan is modeled
/// independently"): four predictor planes share a single DC Huffman table
/// and a single predictor selector, and one sample per component is
/// interleaved per MCU position (each component declared with sampling
/// factors `H_i = V_i = 1`, so each MCU is exactly one sample-quadruple).
///
/// * `planes` — four plane slices in scan order. Component IDs in the
///   bitstream are 1, 2, 3, 4 in that order. The encoder is colour-
///   agnostic: the caller decides what the four planes represent and the
///   decoder hands them back in the same SOS scan order, then applies the
///   APP14 colour transform (if any) on output.
/// * `strides` — bytes per row per plane; must be at least `width`.
/// * `predictor` — selector value `1..=7` from Table H.1, shared by every
///   component (T.81 §B.2.3).
/// * `adobe_transform` — Adobe APP14 colour-transform flag:
///   * `None`     — no APP14 segment; samples passed through unchanged
///     ("regular" CMYK on the decoder side).
///   * `Some(0)`  — Adobe CMYK convention. The encoder inverts every
///     input byte before predictive coding, so the on-wire samples
///     match what an Adobe-CMYK consumer expects. The matching decoder
///     un-inverts on output.
///   * `Some(2)`  — Adobe YCCK. The caller passes `[Y, Cb, Cr, K]`
///     planes (already in YCbCr space); the encoder inverts only the K
///     plane before predictive coding so the decoder's YCCK → CMYK
///     un-inversion path lands on the same K value.
///
/// Precision is fixed at 8 bits (the only depth the workspace
/// `PixelFormat` enum's `Cmyk` variant covers). Point transform is fixed
/// at `Pt = 0` and no restart markers are emitted. For non-zero point
/// transform or restart-marker emission see
/// [`encode_lossless_jpeg_cmyk_with_opts`].
pub fn encode_lossless_jpeg_cmyk(
    width: u32,
    height: u32,
    planes: [&[u8]; 4],
    strides: [usize; 4],
    predictor: u8,
    adobe_transform: Option<u8>,
) -> Result<Vec<u8>> {
    encode_lossless_jpeg_cmyk_with_opts(
        width,
        height,
        planes,
        strides,
        predictor,
        adobe_transform,
        0,
        0,
    )
}

/// Like [`encode_lossless_jpeg_cmyk`] but also accepts `restart_interval`
/// (in MCUs — and a four-component lossless MCU is exactly one pixel
/// position) and a `point_transform` (`Pt` in T.81). Both default-to-zero
/// options match the historical zero-restart, zero-Pt output of
/// [`encode_lossless_jpeg_cmyk`].
///
/// * `point_transform` — `0..=7` and strictly less than 8 (the fixed
///   precision). With `Pt > 0` every input sample is right-shifted by
///   `Pt` before prediction; the decoder side later left-shifts the
///   reconstructed sample by the same `Pt` on output.
/// * `restart_interval` — number of MCUs (= pixel positions) between
///   successive `RSTn` markers. On every boundary the encoder
///   byte-aligns the bitstream, emits the next `RST0..=RST7` marker
///   (cycling modulo 8 per T.81 §F.1.1.5.2), and re-seeds every
///   component's predictor history to the per-component origin
///   `2^(8 − Pt − 1)` (T.81 §H.1.2.1).
#[allow(clippy::too_many_arguments)]
pub fn encode_lossless_jpeg_cmyk_with_opts(
    width: u32,
    height: u32,
    planes: [&[u8]; 4],
    strides: [usize; 4],
    predictor: u8,
    adobe_transform: Option<u8>,
    restart_interval: u16,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if !(1..=7).contains(&predictor) {
        return Err(Error::invalid(format!(
            "lossless CMYK encoder: predictor {predictor} not in 1..=7"
        )));
    }
    if point_transform >= 8 {
        return Err(Error::invalid(format!(
            "lossless CMYK encoder: point_transform {point_transform} must be < precision 8"
        )));
    }
    match adobe_transform {
        None | Some(0) | Some(2) => {}
        Some(other) => {
            return Err(Error::invalid(format!(
                "lossless CMYK encoder: adobe_transform = {other} (only None / Some(0) / Some(2) are supported)"
            )));
        }
    }
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err(Error::invalid("lossless CMYK encoder: zero-size image"));
    }
    for c in 0..4 {
        if strides[c] < w {
            return Err(Error::invalid(
                "lossless CMYK encoder: stride smaller than width",
            ));
        }
        if planes[c].len() < strides[c] * h {
            return Err(Error::invalid(
                "lossless CMYK encoder: plane shorter than stride*h",
            ));
        }
    }

    // Apply the on-the-wire transform implied by `adobe_transform` so the
    // residuals coded below match what the matching decoder expects. The
    // decoder will re-invert (and YCCK→CMYK convert) on output.
    let invert_all = matches!(adobe_transform, Some(0));
    let invert_k_only = matches!(adobe_transform, Some(2));

    // Single shared DC Huffman table (Td = 0).
    let dc_huff = HuffTable::build(&STD_DC_LOSSLESS_BITS, &STD_DC_LOSSLESS_VALS)?;

    let mut out: Vec<u8> = Vec::with_capacity(16_384 + 4 * w * h);
    out.push(0xFF);
    out.push(markers::SOI);
    write_jfif_app0(&mut out);
    if let Some(tx) = adobe_transform {
        write_adobe_app14(&mut out, tx);
    }
    write_sof_lossless_multi(&mut out, w as u16, h as u16, 8, 4);
    write_dht(&mut out, 0, 0, &STD_DC_LOSSLESS_BITS, &STD_DC_LOSSLESS_VALS);
    if restart_interval != 0 {
        write_dri(&mut out, restart_interval);
    }
    write_sos_lossless_multi(&mut out, predictor, 4, point_transform);

    let pt = point_transform as u32;

    // Apply the adobe-transform inversion (if any) up-front, then the
    // point-transform right-shift, so the predictor loop sees pre-coded
    // samples in the 0..2^(8 - Pt) range.
    let mut src: [Vec<u16>; 4] = [
        vec![0u16; w * h],
        vec![0u16; w * h],
        vec![0u16; w * h],
        vec![0u16; w * h],
    ];
    for c in 0..4 {
        let invert_this = invert_all || (invert_k_only && c == 3);
        for y in 0..h {
            for x in 0..w {
                let v = planes[c][y * strides[c] + x] as u16;
                let v = if invert_this { 255 - v } else { v };
                src[c][y * w + x] = v >> pt;
            }
        }
    }

    // Default prediction for the first sample of each component at scan
    // start and after every RSTn (T.81 §H.1.2.1): `2^(P − Pt − 1)` with
    // `P = 8`.
    let sample_bits = 8u32 - pt;
    let origin: i32 = 1i32 << (sample_bits - 1);

    let ri = restart_interval as u32;
    let mut rst_counter: u8 = 0;
    let mut mcus_since_restart: u32 = 0;
    let total_mcus: u64 = w as u64 * h as u64;
    let mut mcu_index: u64 = 0;
    let mut reset_pred = true;

    let mut bw = BitWriter::new(&mut out);
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let plane = &src[c];
                let actual = plane[y * w + x] as i32;
                let pred: i32 = if reset_pred {
                    origin
                } else if y == 0 {
                    plane[y * w + x - 1] as i32
                } else if x == 0 {
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
                if s != 0 && s != 16 {
                    bw.write_bits(bits, s as u32);
                }
            }

            reset_pred = false;
            mcu_index += 1;
            mcus_since_restart += 1;

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

/// Write an SOF3 segment for a subsampled three-component (YUV-class)
/// lossless frame: the luma component (id 1) declares `(h_factor,
/// v_factor)`, both chroma components (ids 2, 3) declare `1×1`. Tq is 0 for
/// every component (lossless ignores quantisation tables).
fn write_sof_lossless_yuv(out: &mut Vec<u8>, width: u16, height: u16, h_factor: u8, v_factor: u8) {
    debug_assert!((1..=4).contains(&h_factor) && (1..=4).contains(&v_factor));
    let mut payload = Vec::with_capacity(8 + 3 * 3);
    payload.push(8); // precision (YUV-class lossless is P = 8)
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(3); // Nf
    payload.push(1); // luma component id
    payload.push((h_factor << 4) | (v_factor & 0x0F));
    payload.push(0); // Tq
    payload.push(2); // Cb id
    payload.push(0x11); // 1×1
    payload.push(0);
    payload.push(3); // Cr id
    payload.push(0x11); // 1×1
    payload.push(0);
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

    // ---- Baseline grayscale (SOF0 single-component, P=8) ----------

    #[test]
    fn baseline_grayscale_emits_well_formed_jpeg() {
        // Smooth gradient → small DCT magnitudes, easy on the entropy
        // coder. We only care that the output is a complete SOI..EOI
        // bytestream with a SOF0 (`Nf = 1`) frame header.
        let w = 16usize;
        let h = 16usize;
        let mut samples = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                samples[y * w + x] = ((x + y) * 8) as u8;
            }
        }
        let jpeg =
            encode_jpeg_grayscale(w as u32, h as u32, &samples, w, 75).expect("encode grayscale");
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "expected SOI prefix");
        assert_eq!(
            &jpeg[jpeg.len() - 2..],
            &[0xFF, 0xD9],
            "expected EOI suffix"
        );
        // Walk segments up to SOS (the scan body that follows isn't
        // length-prefixed) and find SOF0 + check its single-component
        // shape.
        let mut found_sof0 = false;
        let mut i = 2;
        while i + 3 < jpeg.len() {
            assert_eq!(jpeg[i], 0xFF, "expected marker prefix at {i}");
            let marker = jpeg[i + 1];
            // Bail out once we reach SOS — the entropy scan that
            // follows isn't length-prefixed.
            if marker == 0xDA {
                break;
            }
            let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            if marker == 0xC0 {
                // SOF0 payload: P, Y, X, Nf, then component records.
                let p = i + 4;
                let precision = jpeg[p];
                let nf = jpeg[p + 5];
                assert_eq!(precision, 8, "SOF0 precision must be 8");
                assert_eq!(nf, 1, "SOF0 must declare a single component");
                let comp_id = jpeg[p + 6];
                let hv = jpeg[p + 7];
                let tq = jpeg[p + 8];
                assert_eq!(comp_id, 1, "component id");
                assert_eq!(hv, 0x11, "H = V = 1");
                assert_eq!(tq, 0, "single quant table id");
                found_sof0 = true;
            }
            i += 2 + len;
        }
        assert!(found_sof0, "expected SOF0 segment in output");
    }

    #[test]
    fn baseline_grayscale_high_quality_roundtrip_is_near_lossless() {
        // High quality → tiny quantiser → near bit-exact reconstruction.
        // We assert per-sample max-diff stays small; PSNR shoots up so
        // we don't need a separate dB check.
        let w = 32usize;
        let h = 32usize;
        let mut samples = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                samples[y * w + x] = (((x * 7 + y * 11) % 256) as u8).clamp(0, 255);
            }
        }
        let jpeg =
            encode_jpeg_grayscale(w as u32, h as u32, &samples, w, 100).expect("encode grayscale");
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode");
        assert_eq!(frame.planes.len(), 1, "Gray8 frame has one plane");
        let recovered = &frame.planes[0].data;
        assert!(recovered.len() >= w * h, "recovered plane too short");
        let stride = frame.planes[0].stride;
        let mut max_diff: u32 = 0;
        for y in 0..h {
            for x in 0..w {
                let a = samples[y * w + x];
                let b = recovered[y * stride + x];
                let d = (a as i32 - b as i32).unsigned_abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        // Annex K luma table at Q=100 scales to all-1 quantisers, so any
        // residual delta comes from f32 DCT/IDCT rounding only. ±4 LSB
        // is a generous ceiling; we typically see ≤ 2.
        assert!(max_diff <= 4, "Q=100 max diff = {max_diff} (expected ≤ 4)");
    }

    #[test]
    fn baseline_grayscale_q75_roundtrip_psnr_above_30db() {
        // The standard quality default. Loose floor: PSNR ≥ 30 dB on a
        // smooth-ish synthetic pattern. The same Annex K matrices the
        // 3-component YUV path uses, exercised on a single plane.
        let w = 64usize;
        let h = 64usize;
        let mut samples = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let r = ((x as i32 - 32).abs() + (y as i32 - 32).abs()) as u32;
                samples[y * w + x] = (128 + (r as i32 - 32).clamp(-127, 127)) as u8;
            }
        }
        let jpeg = encode_jpeg_grayscale(w as u32, h as u32, &samples, w, DEFAULT_QUALITY)
            .expect("encode grayscale");
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode");
        let recovered = &frame.planes[0].data;
        let stride = frame.planes[0].stride;
        let mut sse: f64 = 0.0;
        for y in 0..h {
            for x in 0..w {
                let a = samples[y * w + x] as f64;
                let b = recovered[y * stride + x] as f64;
                let d = a - b;
                sse += d * d;
            }
        }
        let mse = sse / (w * h) as f64;
        let psnr = if mse <= f64::EPSILON {
            99.0
        } else {
            20.0 * (255.0_f64 / mse.sqrt()).log10()
        };
        assert!(psnr >= 30.0, "Q=75 PSNR = {psnr:.2} dB (expected ≥ 30)");
    }

    #[test]
    fn baseline_grayscale_rejects_short_stride() {
        let samples = vec![0u8; 16 * 16];
        let err =
            encode_jpeg_grayscale(16, 16, &samples, 8, 75).expect_err("stride < width must fail");
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn baseline_grayscale_rejects_short_buffer() {
        // Stride OK but the caller passed too few rows.
        let samples = vec![0u8; 16 * 8];
        let err = encode_jpeg_grayscale(16, 16, &samples, 16, 75)
            .expect_err("samples shorter than stride*h must fail");
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn baseline_grayscale_with_opts_emits_dri_and_restart() {
        // restart_interval = 4 MCUs on a 16x16 image (4 MCUs total per
        // row → restart fires once per row). Decode must succeed and
        // round-trip.
        let w = 32u32;
        let h = 32u32;
        let mut samples = vec![0u8; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                samples[y * w as usize + x] = ((x + y) as u8).wrapping_mul(3);
            }
        }
        let jpeg = encode_jpeg_grayscale_with_opts(w, h, &samples, w as usize, 80, 4)
            .expect("encode grayscale with DRI");
        // DRI marker = 0xFFDD.
        let mut found_dri = false;
        let mut found_rst = false;
        for i in 0..jpeg.len().saturating_sub(1) {
            if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDD {
                found_dri = true;
            }
            if jpeg[i] == 0xFF && (0xD0..=0xD7).contains(&jpeg[i + 1]) {
                found_rst = true;
            }
        }
        assert!(found_dri, "DRI segment must be present when interval > 0");
        assert!(found_rst, "at least one RSTn marker must be present");
        // Decode succeeds and shape matches.
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode DRI grayscale");
        assert_eq!(frame.planes.len(), 1);
    }

    #[test]
    fn baseline_grayscale_with_meta_embeds_app_segments() {
        // Build a fake APP1 (EXIF placeholder) segment and confirm it
        // appears verbatim in the output ahead of DQT.
        let w = 16u32;
        let h = 16u32;
        let samples = vec![100u8; (w * h) as usize];
        let app1_body = b"FAKEEXIF";
        let mut meta = Vec::new();
        meta.push(0xFF);
        meta.push(0xE1); // APP1
        let len = (2 + app1_body.len()) as u16;
        meta.extend_from_slice(&len.to_be_bytes());
        meta.extend_from_slice(app1_body);
        let jpeg = encode_jpeg_grayscale_with_meta(w, h, &samples, w as usize, 90, 0, &meta)
            .expect("encode with meta");
        // First marker after SOI must be APP1; default JFIF APP0 must
        // be absent on this path.
        assert_eq!(&jpeg[2..4], &[0xFF, 0xE1], "APP1 must follow SOI");
        // The APP1 body bytes must be present verbatim.
        assert!(
            jpeg.windows(app1_body.len()).any(|w| w == app1_body),
            "APP1 body not embedded"
        );
        // And the decoder still consumes the output cleanly.
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode with meta");
        assert_eq!(frame.planes.len(), 1);
    }

    // ---- Progressive (SOF2) single-component grayscale --------------

    /// Confirm the progressive grayscale encoder emits a well-formed
    /// JPEG bytestream: SOI prefix, EOI suffix, single SOF2 segment
    /// with `Nf = 1, P = 8, H = V = 1, Tq = 0`, single DQT, the Annex K
    /// luma DC + luma AC DHT pair only (no chroma tables), and exactly
    /// three SOS scans laid out as the `(Ss, Se)` pairs `(0, 0)`,
    /// `(1, 5)`, `(6, 63)`.
    #[test]
    fn progressive_grayscale_emits_well_formed_sof2_scan_layout() {
        let w = 16usize;
        let h = 16usize;
        let mut samples = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                samples[y * w + x] = ((x + y) * 8) as u8;
            }
        }
        let jpeg = encode_jpeg_progressive_grayscale(w as u32, h as u32, &samples, w, 75)
            .expect("encode progressive grayscale");
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "expected SOI prefix");
        assert_eq!(
            &jpeg[jpeg.len() - 2..],
            &[0xFF, 0xD9],
            "expected EOI suffix"
        );

        // Walk segments up to the first SOS, then re-scan the rest of
        // the bytestream for additional SOS markers (the scan body
        // between SOS segments isn't length-prefixed, so we use the
        // unstuffed `0xFF 0xDA` pattern).
        let mut found_sof2 = false;
        let mut dqt_count = 0usize;
        let mut dht_pairs: Vec<(u8, u8)> = Vec::new();
        let mut sos_ssse: Vec<(u8, u8)> = Vec::new();
        let mut i = 2;
        while i + 3 < jpeg.len() {
            assert_eq!(jpeg[i], 0xFF, "expected marker prefix at {i}");
            let marker = jpeg[i + 1];
            if marker == 0xDA {
                // Capture this SOS's Ss/Se and skip past it via the
                // length-prefixed payload so we land on the next
                // segment / scan boundary.
                let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
                let payload = &jpeg[i + 4..i + 2 + len];
                // payload: [Ns, (Cs, Td|Ta) * Ns, Ss, Se, Ah|Al]
                let ns = payload[0] as usize;
                let ss = payload[1 + 2 * ns];
                let se = payload[2 + 2 * ns];
                sos_ssse.push((ss, se));
                // Skip past the SOS marker and its payload, then walk
                // the scan body until the next `FF xx` with `xx != 00`
                // and `xx not in RST0..=RST7` — that's the next SOS or
                // EOI.
                let mut j = i + 2 + len;
                while j + 1 < jpeg.len() {
                    if jpeg[j] == 0xFF && jpeg[j + 1] != 0x00 {
                        let m = jpeg[j + 1];
                        // RSTn (0xD0..=0xD7) doesn't end the scan.
                        if (0xD0..=0xD7).contains(&m) {
                            j += 2;
                            continue;
                        }
                        break;
                    }
                    j += 1;
                }
                i = j;
                continue;
            }
            let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            let payload = &jpeg[i + 4..i + 2 + len];
            match marker {
                0xC2 => {
                    // SOF2 — single-component shape.
                    assert_eq!(payload[0], 8, "SOF2 precision must be 8");
                    assert_eq!(payload[5], 1, "SOF2 must declare Nf = 1");
                    assert_eq!(payload[6], 1, "component id");
                    assert_eq!(payload[7], 0x11, "H = V = 1");
                    assert_eq!(payload[8], 0, "quant table id");
                    found_sof2 = true;
                }
                0xDB => {
                    dqt_count += 1;
                }
                0xC4 => {
                    // DHT: first byte = class<<4 | id.
                    let class = payload[0] >> 4;
                    let id = payload[0] & 0x0F;
                    dht_pairs.push((class, id));
                }
                _ => {}
            }
            i += 2 + len;
        }
        assert!(found_sof2, "expected SOF2 segment in output");
        assert_eq!(dqt_count, 1, "exactly one DQT expected (luma only)");
        assert_eq!(dht_pairs.len(), 2, "exactly two DHTs expected");
        assert!(
            dht_pairs.contains(&(0, 0)),
            "luma DC DHT (class=0, id=0) missing"
        );
        assert!(
            dht_pairs.contains(&(1, 0)),
            "luma AC DHT (class=1, id=0) missing"
        );
        assert_eq!(
            sos_ssse,
            vec![(0, 0), (1, 5), (6, 63)],
            "SOS scans must be DC / AC-low / AC-high in order",
        );
    }

    /// Progressive grayscale round-trip at Q = 100 — the matching SOF2
    /// decoder must reconstruct the input to within a few LSBs (Annex
    /// K table at Q=100 scales to all-1 quantisers, so any residual
    /// delta comes from f32 DCT/IDCT rounding only).
    #[test]
    fn progressive_grayscale_high_quality_roundtrip_is_near_lossless() {
        let w = 32usize;
        let h = 32usize;
        let mut samples = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                samples[y * w + x] = (((x * 7 + y * 11) % 256) as u8).clamp(0, 255);
            }
        }
        let jpeg = encode_jpeg_progressive_grayscale(w as u32, h as u32, &samples, w, 100)
            .expect("encode progressive grayscale");
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode");
        assert_eq!(frame.planes.len(), 1, "Gray8 frame has one plane");
        let recovered = &frame.planes[0].data;
        let stride = frame.planes[0].stride;
        let mut max_diff: u32 = 0;
        for y in 0..h {
            for x in 0..w {
                let a = samples[y * w + x];
                let b = recovered[y * stride + x];
                let d = (a as i32 - b as i32).unsigned_abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        assert!(max_diff <= 4, "Q=100 max diff = {max_diff} (expected ≤ 4)");
    }

    /// Progressive grayscale round-trip at Q = 75 — PSNR ≥ 30 dB on a
    /// smooth synthetic gradient. Matches the floor `encode_jpeg_progressive`
    /// achieves on YUV at the same quality.
    #[test]
    fn progressive_grayscale_q75_roundtrip_psnr_above_30db() {
        let w = 64usize;
        let h = 64usize;
        let mut samples = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let r = ((x as i32 - 32).abs() + (y as i32 - 32).abs()) as u32;
                samples[y * w + x] = (128 + (r as i32 - 32).clamp(-127, 127)) as u8;
            }
        }
        let jpeg =
            encode_jpeg_progressive_grayscale(w as u32, h as u32, &samples, w, DEFAULT_QUALITY)
                .expect("encode progressive grayscale");
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode");
        let recovered = &frame.planes[0].data;
        let stride = frame.planes[0].stride;
        let mut sse: f64 = 0.0;
        for y in 0..h {
            for x in 0..w {
                let a = samples[y * w + x] as f64;
                let b = recovered[y * stride + x] as f64;
                sse += (a - b) * (a - b);
            }
        }
        let mse = sse / ((w * h) as f64);
        // PSNR = 10*log10(255^2 / mse).
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0_f64 * 255.0 / mse).log10()
        };
        assert!(psnr >= 30.0, "Q=75 PSNR = {psnr:.2} dB (expected ≥ 30)");
    }

    /// The progressive grayscale encoder rejects strides shorter than
    /// `width` and buffers shorter than `stride * height`.
    #[test]
    fn progressive_grayscale_rejects_short_stride() {
        let samples = vec![0u8; 64];
        let err = encode_jpeg_progressive_grayscale(16, 4, &samples, 8, 75).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn progressive_grayscale_rejects_short_buffer() {
        let samples = vec![0u8; 30];
        let err = encode_jpeg_progressive_grayscale(8, 8, &samples, 8, 75).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    /// The `_with_meta` variant places the caller-supplied metadata
    /// segments right after SOI, suppressing the default JFIF APP0,
    /// and the bytestream still round-trips through the decoder.
    #[test]
    fn progressive_grayscale_with_meta_embeds_app_segments() {
        let w = 16u32;
        let h = 16u32;
        let samples = vec![100u8; (w * h) as usize];
        let app1_body = b"FAKEEXIF";
        let mut meta = Vec::new();
        meta.push(0xFF);
        meta.push(0xE1); // APP1
        let len = (2 + app1_body.len()) as u16;
        meta.extend_from_slice(&len.to_be_bytes());
        meta.extend_from_slice(app1_body);
        let jpeg =
            encode_jpeg_progressive_grayscale_with_meta(w, h, &samples, w as usize, 90, &meta)
                .expect("encode with meta");
        // First marker after SOI must be APP1; default JFIF APP0 must
        // be absent on this path.
        assert_eq!(&jpeg[2..4], &[0xFF, 0xE1], "APP1 must follow SOI");
        assert!(
            jpeg.windows(app1_body.len()).any(|win| win == app1_body),
            "APP1 body not embedded"
        );
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode with meta");
        assert_eq!(frame.planes.len(), 1);
    }

    // ---- Baseline RGB24 (SOF0 three-component, P=8) --------------

    /// Encode `samples` and walk the segment list confirming the
    /// SOF0 three-component shape (IDs 82/71/66, every H=V=1, every
    /// component binding qt 0), the presence of an Adobe APP14
    /// segment with transform = 0, and that only one DQT + the luma
    /// DC + luma AC DHT pair are emitted.
    fn walk_rgb24_header(jpeg: &[u8]) {
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "expected SOI prefix");
        assert_eq!(
            &jpeg[jpeg.len() - 2..],
            &[0xFF, 0xD9],
            "expected EOI suffix"
        );
        let mut found_sof0 = false;
        let mut found_app14 = false;
        let mut dqt_count = 0usize;
        let mut dht_pairs: Vec<(u8, u8)> = Vec::new();
        let mut i = 2;
        while i + 3 < jpeg.len() {
            assert_eq!(jpeg[i], 0xFF, "expected marker prefix at {i}");
            let marker = jpeg[i + 1];
            if marker == 0xDA {
                break;
            }
            let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            let payload = &jpeg[i + 4..i + 2 + len];
            match marker {
                0xC0 => {
                    // SOF0: P, Y, X, Nf, then component records.
                    assert_eq!(payload[0], 8, "P must be 8");
                    assert_eq!(payload[5], 3, "Nf must be 3");
                    let comps = [
                        (payload[6], payload[7], payload[8]),
                        (payload[9], payload[10], payload[11]),
                        (payload[12], payload[13], payload[14]),
                    ];
                    assert_eq!(comps[0].0, b'R', "component 0 id must be 'R'");
                    assert_eq!(comps[1].0, b'G', "component 1 id must be 'G'");
                    assert_eq!(comps[2].0, b'B', "component 2 id must be 'B'");
                    for c in &comps {
                        assert_eq!(c.1, 0x11, "H = V = 1 on every component");
                        assert_eq!(c.2, 0, "qt 0 on every component");
                    }
                    found_sof0 = true;
                }
                0xEE if payload.len() >= 12 && &payload[0..5] == b"Adobe" => {
                    // APP14 — confirm Adobe magic + transform = 0.
                    assert_eq!(payload[11], 0, "Adobe APP14 transform must be 0");
                    found_app14 = true;
                }
                0xDB => {
                    dqt_count += 1;
                    assert_eq!(payload[0] & 0xF0, 0, "DQT precision must be 0 (8-bit)");
                    assert_eq!(payload[0] & 0x0F, 0, "DQT table id must be 0");
                }
                0xC4 => {
                    let class = payload[0] >> 4;
                    let id = payload[0] & 0x0F;
                    dht_pairs.push((class, id));
                }
                _ => {}
            }
            i += 2 + len;
        }
        assert!(found_sof0, "expected SOF0 segment");
        assert!(found_app14, "expected Adobe APP14 transform=0 segment");
        assert_eq!(dqt_count, 1, "expected exactly one DQT (luma only)");
        assert_eq!(
            dht_pairs,
            vec![(0u8, 0u8), (1u8, 0u8)],
            "expected luma DC + luma AC only"
        );
    }

    #[test]
    fn baseline_rgb24_header_is_well_formed() {
        let w = 16usize;
        let h = 16usize;
        let mut samples = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                samples[o] = ((x * 8) & 0xFF) as u8;
                samples[o + 1] = ((y * 8) & 0xFF) as u8;
                samples[o + 2] = (((x + y) * 4) & 0xFF) as u8;
            }
        }
        let jpeg =
            encode_jpeg_rgb24(w as u32, h as u32, &samples, w * 3, 75).expect("encode rgb24");
        walk_rgb24_header(&jpeg);
    }

    #[test]
    fn baseline_rgb24_q100_roundtrip_is_near_lossless() {
        // Annex K luma table at Q=100 reduces to all-1 quantisers — any
        // residual delta is f32 DCT/IDCT rounding only. Mirror the
        // grayscale ±4 LSB ceiling; per-channel diff is checked
        // independently.
        let w = 32usize;
        let h = 32usize;
        let mut samples = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                samples[o] = ((x * 7 + y * 11) % 256) as u8;
                samples[o + 1] = ((x * 11 + y * 7) % 256) as u8;
                samples[o + 2] = ((x * 13 + y * 5) % 256) as u8;
            }
        }
        let jpeg =
            encode_jpeg_rgb24(w as u32, h as u32, &samples, w * 3, 100).expect("encode rgb24");
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode");
        assert_eq!(frame.planes.len(), 1, "Rgb24 frame has one plane");
        let recovered = &frame.planes[0].data;
        let stride = frame.planes[0].stride;
        assert_eq!(stride, w * 3, "expected packed RGB stride");
        let mut max_diff: u32 = 0;
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    let a = samples[(y * w + x) * 3 + ch];
                    let b = recovered[y * stride + x * 3 + ch];
                    let d = (a as i32 - b as i32).unsigned_abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                }
            }
        }
        assert!(max_diff <= 4, "Q=100 max diff = {max_diff} (expected ≤ 4)");
    }

    #[test]
    fn baseline_rgb24_q75_roundtrip_psnr_above_30db() {
        // PSNR floor on a smooth gradient at the default quality. RGB
        // baseline binds every component to the luma quant table, so the
        // chroma channels see the same numeric quantisation the luma
        // path sees — looser than YUV 4:2:0 baseline at the same Q.
        let w = 64usize;
        let h = 64usize;
        let mut samples = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                let r = ((x as i32 - 32).abs() + (y as i32 - 32).abs()) as u32;
                samples[o] = (128 + (r as i32 - 32).clamp(-127, 127)) as u8;
                samples[o + 1] = ((x * 4) & 0xFF) as u8;
                samples[o + 2] = ((y * 4) & 0xFF) as u8;
            }
        }
        let jpeg = encode_jpeg_rgb24(w as u32, h as u32, &samples, w * 3, DEFAULT_QUALITY)
            .expect("encode rgb24");
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode");
        let recovered = &frame.planes[0].data;
        let stride = frame.planes[0].stride;
        let mut sse: f64 = 0.0;
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    let a = samples[(y * w + x) * 3 + ch] as f64;
                    let b = recovered[y * stride + x * 3 + ch] as f64;
                    let d = a - b;
                    sse += d * d;
                }
            }
        }
        let mse = sse / (w * h * 3) as f64;
        let psnr = if mse <= f64::EPSILON {
            99.0
        } else {
            20.0 * (255.0_f64 / mse.sqrt()).log10()
        };
        assert!(psnr >= 30.0, "Q=75 PSNR = {psnr:.2} dB (expected ≥ 30)");
    }

    #[test]
    fn baseline_rgb24_rejects_short_stride() {
        let samples = vec![0u8; 16 * 16 * 3];
        let err =
            encode_jpeg_rgb24(16, 16, &samples, 16, 75).expect_err("stride < width * 3 must fail");
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn baseline_rgb24_rejects_short_buffer() {
        // Stride OK but the caller passed too few rows.
        let samples = vec![0u8; 16 * 8 * 3];
        let err = encode_jpeg_rgb24(16, 16, &samples, 16 * 3, 75)
            .expect_err("samples shorter than stride*h must fail");
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
    }

    #[test]
    fn baseline_rgb24_with_opts_emits_dri_and_restart() {
        // 32×32 image at 4 MCUs per row, restart_interval = 4 MCUs →
        // RSTn fires three times per row. DRI segment must be present,
        // at least one RSTn marker emitted, decoder still round-trips.
        let w = 32u32;
        let h = 32u32;
        let mut samples = vec![0u8; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let o = (y * w as usize + x) * 3;
                samples[o] = ((x + y) as u8).wrapping_mul(3);
                samples[o + 1] = ((x + y) as u8).wrapping_mul(5);
                samples[o + 2] = ((x + y) as u8).wrapping_mul(7);
            }
        }
        let jpeg = encode_jpeg_rgb24_with_opts(w, h, &samples, (w * 3) as usize, 80, 4)
            .expect("encode rgb24 with DRI");
        let mut found_dri = false;
        let mut found_rst = false;
        for i in 0..jpeg.len().saturating_sub(1) {
            if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDD {
                found_dri = true;
            }
            if jpeg[i] == 0xFF && (0xD0..=0xD7).contains(&jpeg[i + 1]) {
                found_rst = true;
            }
        }
        assert!(found_dri, "DRI segment must be present when interval > 0");
        assert!(found_rst, "at least one RSTn marker must be present");
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode DRI rgb24");
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.planes[0].stride, (w * 3) as usize);
    }

    #[test]
    fn baseline_rgb24_with_meta_embeds_app_segments() {
        // Caller-supplied meta replaces the default APP0 + APP14 pair
        // entirely. The component-id fallback (`'R'/'G'/'B'`) still
        // signals RGB to the decoder so it round-trips as `Rgb24`.
        let w = 16u32;
        let h = 16u32;
        let samples = vec![100u8; (w * h * 3) as usize];
        let app1_body = b"FAKEEXIF";
        let mut meta = Vec::new();
        meta.push(0xFF);
        meta.push(0xE1); // APP1
        let len = (2 + app1_body.len()) as u16;
        meta.extend_from_slice(&len.to_be_bytes());
        meta.extend_from_slice(app1_body);
        let jpeg = encode_jpeg_rgb24_with_meta(w, h, &samples, (w * 3) as usize, 90, 0, &meta)
            .expect("encode rgb24 with meta");
        assert_eq!(&jpeg[2..4], &[0xFF, 0xE1], "APP1 must follow SOI");
        assert!(
            jpeg.windows(app1_body.len()).any(|win| win == app1_body),
            "APP1 body not embedded"
        );
        // No Adobe APP14 was emitted on this path; the decoder still
        // recognises RGB via component IDs.
        let frame = crate::decoder::decode_jpeg(&jpeg, None).expect("decode with meta");
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.planes[0].stride, (w * 3) as usize);
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
