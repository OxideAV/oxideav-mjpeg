//! Baseline-JPEG encoder.
//!
//! Produces a standalone, self-contained JPEG packet per video frame:
//! SOI, JFIF APP0, DQT, SOF0, DHT (Annex K tables), SOS, entropy scan, EOI.
//! Handles 4:2:0 / 4:2:2 / 4:4:4 YUV planar input. Restart markers
//! (DRI + `RSTn`) are emitted when [`MjpegEncoder::set_restart_interval`]
//! (or [`encode_jpeg_with_opts`]) is called with a non-zero MCU count;
//! by default the encoder matches the historical behaviour and emits
//! none.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

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

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("MJPEG encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("MJPEG encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    match pix {
        PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P => {}
        _ => {
            return Err(Error::unsupported(format!(
                "MJPEG encoder: pixel format {:?} not supported",
                pix
            )))
        }
    }

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(pix);

    Ok(Box::new(MjpegEncoder {
        output_params,
        width,
        height,
        pix,
        quality: DEFAULT_QUALITY,
        restart_interval: 0,
        time_base: params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num)),
        pending: VecDeque::new(),
        eof: false,
    }))
}

/// Baseline-JPEG encoder. Emits one self-contained JPEG bitstream per
/// video frame.
pub struct MjpegEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    pix: PixelFormat,
    quality: u8,
    /// MCU-per-restart-interval count. 0 disables DRI / `RSTn` emission.
    restart_interval: u16,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl MjpegEncoder {
    /// Set the restart interval in MCUs (JPEG DRI field). `0` disables
    /// restart marker emission (matches the default).
    ///
    /// Values are clamped to `u16::MAX` since the JPEG DRI field is a
    /// 16-bit big-endian unsigned integer.
    pub fn set_restart_interval(&mut self, mcus: u32) {
        self.restart_interval = mcus.min(u16::MAX as u32) as u16;
    }

    /// Current restart interval (MCUs between `RSTn` markers; 0 = off).
    pub fn restart_interval(&self) -> u16 {
        self.restart_interval
    }
}

impl Encoder for MjpegEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(v) => {
                if v.width != self.width || v.height != self.height {
                    return Err(Error::invalid(
                        "MJPEG encoder: frame dimensions do not match encoder config",
                    ));
                }
                if v.format != self.pix {
                    return Err(Error::invalid(format!(
                        "MJPEG encoder: frame format {:?} does not match encoder format {:?}",
                        v.format, self.pix
                    )));
                }
                let data = encode_jpeg_with_opts(v, self.quality, self.restart_interval)?;
                let mut pkt = Packet::new(0, self.time_base, data);
                pkt.pts = v.pts;
                pkt.dts = v.pts;
                pkt.flags.keyframe = true;
                self.pending.push_back(pkt);
                Ok(())
            }
            _ => Err(Error::invalid("MJPEG encoder: video frames only")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

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
pub fn encode_jpeg(frame: &VideoFrame, quality: u8) -> Result<Vec<u8>> {
    encode_jpeg_with_opts(frame, quality, 0)
}

/// Like [`encode_jpeg`] but also emits a DRI segment and cycles
/// `RST0..=RST7` markers every `restart_interval` MCUs during the scan.
/// Passing `0` disables restart marker emission (equivalent to
/// [`encode_jpeg`]).
pub fn encode_jpeg_with_opts(
    frame: &VideoFrame,
    quality: u8,
    restart_interval: u16,
) -> Result<Vec<u8>> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let (h_factor, v_factor) = match frame.format {
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
    // JFIF APP0.
    write_jfif_app0(&mut out);
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

    let (c_w, c_h) = match frame.format {
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
pub(crate) fn encode_jpeg_non_interleaved(frame: &VideoFrame, quality: u8) -> Result<Vec<u8>> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let (h_factor, v_factor) = match frame.format {
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

    let (c_w, c_h) = match frame.format {
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

/// Test-only: emit a 4-component JPEG (1:1:1:1 sampling, all components at
/// full resolution). The caller supplies four raw sample planes and an
/// optional Adobe APP14 transform flag:
/// * `None` → no APP14 written (treated as plain/"regular" CMYK by this
///   crate's decoder).
/// * `Some(0)` → APP14 transform=0, i.e. Adobe CMYK where the stored
///   samples are inverted (the test encoder inverts its inputs to match
///   that convention before writing).
/// * `Some(2)` → APP14 transform=2, YCCK. The caller supplies the four
///   components in Adobe's stored order (Y, Cb, Cr, K); K is inverted
///   on the way out because Adobe stores it that way.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_jpeg_cmyk_1111(
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
}
