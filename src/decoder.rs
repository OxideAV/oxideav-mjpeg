//! JPEG packet decoder — baseline (SOF0) and progressive (SOF2).

use oxideav_codec::Decoder;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
};

use crate::jpeg::dct::idct8x8;
use crate::jpeg::huffman::{parse_dht, HuffTable};
use crate::jpeg::markers::{self, *};
use crate::jpeg::parser::{parse_dri, parse_sof, parse_sos, MarkerWalker, SofInfo, SosInfo};
use crate::jpeg::quant::{parse_dqt, QuantTable};
use crate::jpeg::zigzag::ZIGZAG;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let codec_id = params.codec_id.clone();
    Ok(Box::new(MjpegDecoder {
        codec_id,
        pending: None,
        eof: false,
    }))
}

struct MjpegDecoder {
    codec_id: CodecId,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for MjpegDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "MJPEG decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let vf = decode_jpeg(&pkt.data, pkt.pts, pkt.time_base)?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ---- Decoding state ------------------------------------------------------

struct JpegState {
    quant: [Option<QuantTable>; 4],
    dc_huff: [Option<HuffTable>; 4],
    ac_huff: [Option<HuffTable>; 4],
    restart_interval: u16,
    sof: Option<SofInfo>,
    /// True when SOF2 (progressive) was parsed.
    progressive: bool,
}

impl JpegState {
    fn new() -> Self {
        Self {
            quant: Default::default(),
            dc_huff: Default::default(),
            ac_huff: Default::default(),
            restart_interval: 0,
            sof: None,
            progressive: false,
        }
    }
}

fn decode_jpeg(data: &[u8], pts: Option<i64>, time_base: TimeBase) -> Result<VideoFrame> {
    // Verify SOI.
    if data.len() < 2 || data[0] != 0xFF || data[1] != markers::SOI {
        return Err(Error::invalid("JPEG: missing SOI"));
    }

    let mut walker = MarkerWalker::new(&data[2..]);
    let mut state = JpegState::new();

    // Progressive coefficient accumulator, only populated once SOF2 is seen.
    // One [i32;64] per MCU-block per component, in natural order.
    let mut prog_coefs: Vec<Vec<[i32; 64]>> = Vec::new();

    loop {
        let Some(marker) = walker.next_marker()? else {
            return Err(Error::invalid("JPEG: unexpected EOF before EOI"));
        };
        match marker {
            EOI => {
                if state.progressive {
                    return render_progressive(&state, &prog_coefs, pts, time_base);
                }
                return Err(Error::invalid("JPEG: EOI before SOS"));
            }
            SOI => continue,
            m if markers::is_rst(m) => continue,
            DQT => {
                let p = walker.read_segment_payload()?;
                parse_dqt(p, &mut state.quant)?;
            }
            DHT => {
                let p = walker.read_segment_payload()?;
                parse_dht(p, &mut state.dc_huff, &mut state.ac_huff)?;
            }
            DRI => {
                let p = walker.read_segment_payload()?;
                state.restart_interval = parse_dri(p)?;
            }
            // SOF0 (baseline) and SOF1 (extended sequential) share the same
            // Huffman sequential scan structure; for 8-bit precision the
            // decode path is identical, and the extended-sequential
            // allowance of up to 4 DC/AC Huffman tables falls out of our
            // existing 4-entry table arrays. Treat SOF1 as SOF0.
            SOF0 | SOF1 => {
                let p = walker.read_segment_payload()?;
                state.sof = Some(parse_sof(p)?);
            }
            SOF2 => {
                let p = walker.read_segment_payload()?;
                let sof = parse_sof(p)?;
                prog_coefs = init_progressive_coefs(&sof)?;
                state.sof = Some(sof);
                state.progressive = true;
            }
            SOF3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                let _ = walker.read_segment_payload();
                return Err(Error::unsupported(
                    "JPEG: only baseline/extended-sequential/progressive SOFs are supported (no hierarchical, lossless, or arithmetic-coded variants)",
                ));
            }
            SOS => {
                let p = walker.read_segment_payload()?;
                let sos = parse_sos(p)?;
                let scan = walker.read_scan_data()?;
                if state.progressive {
                    decode_progressive_scan(&state, &sos, scan, &mut prog_coefs)?;
                    // Continue — more scans or EOI follow.
                } else {
                    return decode_scan(&state, &sos, scan, pts, time_base);
                }
            }
            COM => {
                let _ = walker.read_segment_payload()?;
            }
            m if markers::is_app(m) => {
                let _ = walker.read_segment_payload()?;
            }
            _ => {
                // Ignore unknown length-prefixed markers but still skip the segment.
                let _ = walker.read_segment_payload();
            }
        }
    }
}

/// Allocate a coefficient-accumulator plane for each component in the SOF.
/// Dimensions match the MCU grid so that blocks line up with decoded scans.
fn init_progressive_coefs(sof: &SofInfo) -> Result<Vec<Vec<[i32; 64]>>> {
    if sof.precision != 8 {
        return Err(Error::unsupported("progressive: precision != 8"));
    }
    if sof.components.is_empty() {
        return Err(Error::invalid("SOF: no components"));
    }
    if sof.components.len() > 3 {
        return Err(Error::unsupported("progressive: 4+ components"));
    }
    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1);
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1);
    if h_max == 0 || v_max == 0 {
        return Err(Error::invalid("SOF: sampling factor = 0"));
    }
    let width = sof.width as usize;
    let height = sof.height as usize;
    let mcu_w_px = 8 * h_max as usize;
    let mcu_h_px = 8 * v_max as usize;
    let mcus_x = width.div_ceil(mcu_w_px);
    let mcus_y = height.div_ceil(mcu_h_px);

    let mut out = Vec::with_capacity(sof.components.len());
    for c in &sof.components {
        let blocks_x = mcus_x * c.h_factor as usize;
        let blocks_y = mcus_y * c.v_factor as usize;
        out.push(vec![[0i32; 64]; blocks_x * blocks_y]);
    }
    Ok(out)
}

// ---- Bit reader with 0xFF00 stuff handling -------------------------------

struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    /// 32-bit bit-buffer (MSB-aligned).
    bits: u32,
    /// Number of valid bits currently in `bits`.
    nbits: u32,
    /// Set when we hit a restart marker while refilling.
    pub saw_rst: Option<u8>,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            bits: 0,
            nbits: 0,
            saw_rst: None,
        }
    }

    fn next_byte_with_stuff(&mut self) -> Result<Option<u8>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        if b == 0xFF {
            // Collapse any run of 0xFF fill bytes.
            while self.pos < self.buf.len() && self.buf[self.pos] == 0xFF {
                self.pos += 1;
            }
            if self.pos >= self.buf.len() {
                return Err(Error::invalid("scan: 0xFF at end without followup"));
            }
            let next = self.buf[self.pos];
            self.pos += 1;
            if next == 0x00 {
                // Stuffed zero → literal 0xFF.
                return Ok(Some(0xFF));
            }
            if markers::is_rst(next) {
                self.saw_rst = Some(next);
                // Stop feeding bits; caller must handle restart.
                return Ok(None);
            }
            // Any other marker means scan ended. Back up and stop.
            self.pos -= 2;
            return Ok(None);
        }
        Ok(Some(b))
    }

    fn fill(&mut self, needed: u32) -> Result<()> {
        while self.nbits < needed {
            match self.next_byte_with_stuff()? {
                Some(b) => {
                    self.bits |= (b as u32) << (24 - self.nbits);
                    self.nbits += 8;
                }
                None => {
                    // If we ran out of bits mid-decode, pad with zeros — this
                    // matches libjpeg's behaviour at scan end. The caller is
                    // responsible for noticing `saw_rst` / EOI and stopping.
                    self.bits |= 0;
                    self.nbits = needed;
                    break;
                }
            }
        }
        Ok(())
    }

    fn get_bits(&mut self, n: u32) -> Result<u32> {
        self.fill(n)?;
        let v = self.bits >> (32 - n);
        self.bits <<= n;
        self.nbits -= n;
        Ok(v)
    }

    fn reset_at_restart(&mut self) {
        self.bits = 0;
        self.nbits = 0;
        self.saw_rst = None;
    }
}

/// Read a Huffman symbol using a linear walk up the `min_code` table (fine
/// for a textbook implementation — no fast path).
fn decode_huff(br: &mut BitReader<'_>, t: &HuffTable) -> Result<u8> {
    let mut code: i32 = 0;
    for l in 0..16 {
        let bit = br.get_bits(1)? as i32;
        code = (code << 1) | bit;
        if code <= t.max_code[l] {
            let idx = (t.val_offset[l] + code) as usize;
            if idx >= t.values.len() {
                return Err(Error::invalid("huffman: value index OOB"));
            }
            return Ok(t.values[idx]);
        }
    }
    Err(Error::invalid("huffman: no matching code (length > 16)"))
}

/// Read `size` bits and sign-extend per JPEG Annex F rules.
fn extend(value: i32, size: u32) -> i32 {
    if size == 0 {
        return 0;
    }
    let vt = 1 << (size - 1);
    if value < vt {
        value - ((1 << size) - 1)
    } else {
        value
    }
}

fn decode_scan(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    pts: Option<i64>,
    time_base: TimeBase,
) -> Result<VideoFrame> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("SOS before SOF"))?;
    if sof.precision != 8 {
        return Err(Error::unsupported("precision != 8"));
    }
    if sof.components.is_empty() {
        return Err(Error::invalid("SOF: no components"));
    }
    if sof.components.len() > 3 {
        return Err(Error::unsupported("4+ components"));
    }
    // We only handle an SOS that lists every SOF component (interleaved scan).
    if sos.components.len() != sof.components.len() {
        return Err(Error::unsupported("non-interleaved scan"));
    }

    let n_comp = sof.components.len();
    let grayscale = n_comp == 1;

    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1);
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1);
    if h_max == 0 || v_max == 0 {
        return Err(Error::invalid("SOF: sampling factor = 0"));
    }

    // Output pixel format (subsampling is implied).
    let pix_fmt = if grayscale {
        PixelFormat::Gray8
    } else if n_comp == 3 {
        let y = sof.components[0];
        let cb = sof.components[1];
        let cr = sof.components[2];
        if cb.h_factor != cr.h_factor || cb.v_factor != cr.v_factor {
            return Err(Error::unsupported(
                "chroma components have different sampling factors",
            ));
        }
        if cb.h_factor != 1 || cb.v_factor != 1 {
            return Err(Error::unsupported(
                "chroma components must have factor 1 (luma carries the oversampling)",
            ));
        }
        match (y.h_factor, y.v_factor) {
            (1, 1) => PixelFormat::Yuv444P,
            (2, 1) => PixelFormat::Yuv422P,
            (2, 2) => PixelFormat::Yuv420P,
            _ => {
                return Err(Error::unsupported(format!(
                    "luma sampling {}x{}",
                    y.h_factor, y.v_factor
                )))
            }
        }
    } else {
        return Err(Error::unsupported("2-component JPEG"));
    };

    let width = sof.width as usize;
    let height = sof.height as usize;
    let mcu_w_px = 8 * h_max as usize;
    let mcu_h_px = 8 * v_max as usize;
    let mcus_x = width.div_ceil(mcu_w_px);
    let mcus_y = height.div_ceil(mcu_h_px);

    // Component-level output buffers sized to full MCU coverage.
    let mut comp_buf: Vec<Vec<u8>> = Vec::with_capacity(n_comp);
    let mut comp_stride: Vec<usize> = Vec::with_capacity(n_comp);
    let mut comp_w_full: Vec<usize> = Vec::with_capacity(n_comp);
    let mut comp_h_full: Vec<usize> = Vec::with_capacity(n_comp);
    for c in &sof.components {
        let w_full = mcus_x * 8 * c.h_factor as usize;
        let h_full = mcus_y * 8 * c.v_factor as usize;
        comp_buf.push(vec![0u8; w_full * h_full]);
        comp_stride.push(w_full);
        comp_w_full.push(w_full);
        comp_h_full.push(h_full);
    }

    // Map SOS component id → index in SOF component vector.
    let sos_map: Vec<usize> = sos
        .components
        .iter()
        .map(|sc| {
            sof.components
                .iter()
                .position(|fc| fc.id == sc.id)
                .ok_or_else(|| Error::invalid("SOS: component id not in SOF"))
        })
        .collect::<Result<Vec<_>>>()?;

    // Resolve Huffman tables.
    let dc_tables: Vec<&HuffTable> = sos
        .components
        .iter()
        .map(|sc| {
            state.dc_huff[sc.dc_table as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("SOS: DC Huffman table missing"))
        })
        .collect::<Result<Vec<_>>>()?;
    let ac_tables: Vec<&HuffTable> = sos
        .components
        .iter()
        .map(|sc| {
            state.ac_huff[sc.ac_table as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("SOS: AC Huffman table missing"))
        })
        .collect::<Result<Vec<_>>>()?;

    let quant_tables: Vec<&QuantTable> = sof
        .components
        .iter()
        .map(|c| {
            state.quant[c.qt_id as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("quant table missing for component"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut br = BitReader::new(scan);
    let mut prev_dc = vec![0i32; n_comp];
    let mut mcus_since_restart: u32 = 0;
    let mut expected_rst: u8 = RST0;

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if state.restart_interval != 0
                && mcus_since_restart != 0
                && mcus_since_restart % state.restart_interval as u32 == 0
            {
                // Expect a restart marker before this MCU.
                // Drain bits until the reader hits one. If we already consumed
                // the marker during fill, br.saw_rst is set.
                // Drain any partial bits (align to byte).
                br.bits = 0;
                br.nbits = 0;
                // Scan forward for a marker byte.
                while br.saw_rst.is_none() {
                    // Try to pull one byte; if it's a restart, next_byte records it.
                    let prev = br.pos;
                    match br.next_byte_with_stuff()? {
                        Some(_) => {
                            // Plain byte — unexpected, but keep looking.
                            // (Some encoders pad with zeros before the RST marker.)
                        }
                        None => {
                            if br.saw_rst.is_none() {
                                // No marker visible; maybe off the end.
                                if prev == br.pos {
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                if let Some(m) = br.saw_rst {
                    if m != expected_rst {
                        // Some encoders mis-order; accept anyway but warn with InvalidData
                        // only if you want strictness. We'll accept silently.
                    }
                    expected_rst = if expected_rst == RST7 {
                        RST0
                    } else {
                        expected_rst + 1
                    };
                    for p in prev_dc.iter_mut() {
                        *p = 0;
                    }
                    br.reset_at_restart();
                }
            }

            // For each component, decode H*V blocks.
            for (sidx, sof_idx) in sos_map.iter().enumerate() {
                let c = sof.components[*sof_idx];
                for by in 0..c.v_factor as usize {
                    for bx in 0..c.h_factor as usize {
                        let mut block = [0i32; 64];
                        decode_block(
                            &mut br,
                            dc_tables[sidx],
                            ac_tables[sidx],
                            &mut prev_dc[*sof_idx],
                            &mut block,
                        )?;
                        // Dequantise.
                        let qt = quant_tables[*sof_idx];
                        let mut fblock = [0.0f32; 64];
                        for k in 0..64 {
                            // `block` is natural-order; `qt.values` is natural-order.
                            fblock[k] = (block[k] * qt.values[k] as i32) as f32;
                        }
                        idct8x8(&mut fblock);
                        // Write 8×8 to component buffer at position (mx * H + bx, my * V + by).
                        let dst_x0 = mx * 8 * c.h_factor as usize + bx * 8;
                        let dst_y0 = my * 8 * c.v_factor as usize + by * 8;
                        let stride = comp_stride[*sof_idx];
                        let buf = &mut comp_buf[*sof_idx];
                        for j in 0..8 {
                            for i in 0..8 {
                                let v = fblock[j * 8 + i] + 128.0;
                                let px = if v <= 0.0 {
                                    0
                                } else if v >= 255.0 {
                                    255
                                } else {
                                    v.round() as u8
                                };
                                buf[(dst_y0 + j) * stride + dst_x0 + i] = px;
                            }
                        }
                    }
                }
            }
            mcus_since_restart += 1;
        }
    }

    // Build output VideoFrame.
    let out_format = pix_fmt;
    let mut planes: Vec<VideoPlane> = Vec::new();
    match out_format {
        PixelFormat::Gray8 => {
            let stride = width;
            let mut data = vec![0u8; stride * height];
            let src_stride = comp_stride[0];
            for y in 0..height {
                data[y * stride..y * stride + width]
                    .copy_from_slice(&comp_buf[0][y * src_stride..y * src_stride + width]);
            }
            planes.push(VideoPlane { stride, data });
        }
        PixelFormat::Yuv444P | PixelFormat::Yuv422P | PixelFormat::Yuv420P => {
            let (c_w, c_h) = match out_format {
                PixelFormat::Yuv444P => (width, height),
                PixelFormat::Yuv422P => (width.div_ceil(2), height),
                PixelFormat::Yuv420P => (width.div_ceil(2), height.div_ceil(2)),
                _ => unreachable!(),
            };
            // Y plane.
            let y_stride = width;
            let mut y_data = vec![0u8; y_stride * height];
            let src_stride_y = comp_stride[0];
            for y in 0..height {
                y_data[y * y_stride..y * y_stride + width]
                    .copy_from_slice(&comp_buf[0][y * src_stride_y..y * src_stride_y + width]);
            }
            planes.push(VideoPlane {
                stride: y_stride,
                data: y_data,
            });
            // Cb, Cr planes (each same size).
            for ci in [1usize, 2] {
                let src_stride = comp_stride[ci];
                let stride = c_w;
                let mut data = vec![0u8; stride * c_h];
                for y in 0..c_h {
                    data[y * stride..y * stride + c_w]
                        .copy_from_slice(&comp_buf[ci][y * src_stride..y * src_stride + c_w]);
                }
                planes.push(VideoPlane { stride, data });
            }
        }
        _ => unreachable!(),
    }

    Ok(VideoFrame {
        format: out_format,
        width: width as u32,
        height: height as u32,
        pts,
        time_base,
        planes,
    })
}

fn decode_block(
    br: &mut BitReader<'_>,
    dc: &HuffTable,
    ac: &HuffTable,
    prev_dc: &mut i32,
    out_natural: &mut [i32; 64],
) -> Result<()> {
    // DC.
    let t = decode_huff(br, dc)? as u32;
    let dc_diff = if t == 0 {
        0
    } else {
        let bits = br.get_bits(t)? as i32;
        extend(bits, t)
    };
    *prev_dc = prev_dc.wrapping_add(dc_diff);
    out_natural[0] = *prev_dc;
    // AC: zigzag run-length decode into natural order.
    let mut k: usize = 1;
    while k < 64 {
        let rs = decode_huff(br, ac)?;
        let run = (rs >> 4) as usize;
        let size = (rs & 0x0F) as u32;
        if size == 0 {
            if run == 15 {
                // ZRL — 16 zeros.
                k += 16;
                continue;
            }
            // EOB.
            break;
        }
        k += run;
        if k >= 64 {
            return Err(Error::invalid("JPEG AC: run out of block"));
        }
        let bits = br.get_bits(size)? as i32;
        let val = extend(bits, size);
        out_natural[ZIGZAG[k]] = val;
        k += 1;
    }
    Ok(())
}

// ---- Progressive JPEG (SOF2) --------------------------------------------
//
// A progressive JPEG ships its DCT coefficients in multiple scans. Each
// scan specifies a spectral range `[Ss, Se]` plus successive-approximation
// shift values `Ah`/`Al`:
//
//   * Scans with `Ss == 0 && Se == 0` are DC scans. A DC scan with `Ah == 0`
//     is the first DC pass (decodes `category + sign-magnitude bits`, shifted
//     left by `Al`). A subsequent DC scan with `Ah > 0` refines one more bit.
//   * Scans with `Ss > 0` are AC scans for a single component (non-interleaved
//     per T.81) across `[Ss..=Se]`. `Ah == 0` is the first pass; `Ah > 0` is
//     a refinement scan that adds a correction bit to every already-nonzero
//     AC coefficient and, optionally, creates new coefficients from runs.
//
// The output of each scan is accumulated into the per-component coefficient
// plane. Once EOI is reached, we dequantise + inverse-DCT each block.

fn decode_progressive_scan(
    state: &JpegState,
    sos: &SosInfo,
    scan: &[u8],
    coefs: &mut [Vec<[i32; 64]>],
) -> Result<()> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("SOS before SOF"))?;

    // Validate spectral range + Ah/Al.
    if sos.ss > 63 || sos.se > 63 || sos.ss > sos.se {
        return Err(Error::invalid("progressive: invalid Ss/Se"));
    }
    let is_dc_scan = sos.ss == 0;
    if is_dc_scan && sos.se != 0 {
        return Err(Error::invalid("progressive: DC scan must have Se=0"));
    }
    if !is_dc_scan && sos.components.len() != 1 {
        return Err(Error::invalid(
            "progressive: AC scans must be non-interleaved",
        ));
    }
    if sos.ah > 13 || sos.al > 13 {
        return Err(Error::invalid("progressive: Ah/Al out of range"));
    }

    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1) as usize;
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1) as usize;
    let width = sof.width as usize;
    let height = sof.height as usize;
    let mcus_x = width.div_ceil(8 * h_max);
    let mcus_y = height.div_ceil(8 * v_max);

    // Map SOS component id → SOF index.
    let sos_map: Vec<usize> = sos
        .components
        .iter()
        .map(|sc| {
            sof.components
                .iter()
                .position(|fc| fc.id == sc.id)
                .ok_or_else(|| Error::invalid("progressive SOS: unknown component"))
        })
        .collect::<Result<Vec<_>>>()?;

    // Resolve Huffman tables per selected component.
    let dc_tables: Vec<Option<&HuffTable>> = sos
        .components
        .iter()
        .map(|sc| {
            if is_dc_scan {
                state.dc_huff[sc.dc_table as usize].as_ref()
            } else {
                None
            }
        })
        .collect();
    let ac_tables: Vec<Option<&HuffTable>> = sos
        .components
        .iter()
        .map(|sc| {
            if is_dc_scan {
                None
            } else {
                state.ac_huff[sc.ac_table as usize].as_ref()
            }
        })
        .collect();
    if is_dc_scan && dc_tables.iter().any(|t| t.is_none()) {
        return Err(Error::invalid("progressive DC: Huffman table missing"));
    }
    if !is_dc_scan && ac_tables[0].is_none() {
        return Err(Error::invalid("progressive AC: Huffman table missing"));
    }

    let mut br = BitReader::new(scan);
    let mut prev_dc = vec![0i32; sos.components.len()];
    let mut eob_run: u32 = 0;
    let mut mcus_since_restart: u32 = 0;
    let mut expected_rst: u8 = RST0;

    // How many MCUs does this scan step through?
    let (scan_mcus_x, scan_mcus_y) = if is_dc_scan && sos.components.len() > 1 {
        (mcus_x, mcus_y)
    } else {
        // Non-interleaved scan: a "MCU" is one data-unit for the component.
        let ci = sos_map[0];
        let c = sof.components[ci];
        if sos.components.len() == 1 && sof.components.len() == 1 {
            // Single-component image: MCU grid equals block grid exactly.
            (mcus_x * c.h_factor as usize, mcus_y * c.v_factor as usize)
        } else {
            // For a non-interleaved scan, iterate the component's full block grid.
            (mcus_x * c.h_factor as usize, mcus_y * c.v_factor as usize)
        }
    };

    for my in 0..scan_mcus_y {
        for mx in 0..scan_mcus_x {
            if state.restart_interval != 0
                && mcus_since_restart != 0
                && mcus_since_restart % state.restart_interval as u32 == 0
            {
                br.bits = 0;
                br.nbits = 0;
                while br.saw_rst.is_none() {
                    let prev = br.pos;
                    match br.next_byte_with_stuff()? {
                        Some(_) => {}
                        None => {
                            if prev == br.pos {
                                break;
                            }
                            break;
                        }
                    }
                }
                if br.saw_rst.is_some() {
                    expected_rst = if expected_rst == RST7 {
                        RST0
                    } else {
                        expected_rst + 1
                    };
                    for p in prev_dc.iter_mut() {
                        *p = 0;
                    }
                    eob_run = 0;
                    br.reset_at_restart();
                }
            }

            if is_dc_scan {
                // DC scan may be interleaved across components.
                if sos.components.len() > 1 {
                    // Interleaved: iterate components with H*V blocks each.
                    for (sidx, &sof_idx) in sos_map.iter().enumerate() {
                        let c = sof.components[sof_idx];
                        for by in 0..c.v_factor as usize {
                            for bx in 0..c.h_factor as usize {
                                let blocks_x = mcus_x * c.h_factor as usize;
                                let bidx_x = mx * c.h_factor as usize + bx;
                                let bidx_y = my * c.v_factor as usize + by;
                                let bi = bidx_y * blocks_x + bidx_x;
                                prog_decode_dc(
                                    &mut br,
                                    dc_tables[sidx].unwrap(),
                                    &mut prev_dc[sidx],
                                    &mut coefs[sof_idx][bi],
                                    sos.ah,
                                    sos.al,
                                )?;
                            }
                        }
                    }
                } else {
                    // Non-interleaved single-component DC scan: one block per MCU.
                    let sof_idx = sos_map[0];
                    let c = sof.components[sof_idx];
                    let blocks_x = mcus_x * c.h_factor as usize;
                    let bi = my * blocks_x + mx;
                    prog_decode_dc(
                        &mut br,
                        dc_tables[0].unwrap(),
                        &mut prev_dc[0],
                        &mut coefs[sof_idx][bi],
                        sos.ah,
                        sos.al,
                    )?;
                }
            } else {
                // AC scan is always non-interleaved.
                let sof_idx = sos_map[0];
                let c = sof.components[sof_idx];
                let blocks_x = mcus_x * c.h_factor as usize;
                let bi = my * blocks_x + mx;
                let ac = ac_tables[0].unwrap();
                if sos.ah == 0 {
                    prog_decode_ac_first(
                        &mut br,
                        ac,
                        &mut coefs[sof_idx][bi],
                        sos.ss as usize,
                        sos.se as usize,
                        sos.al,
                        &mut eob_run,
                    )?;
                } else {
                    prog_decode_ac_refine(
                        &mut br,
                        ac,
                        &mut coefs[sof_idx][bi],
                        sos.ss as usize,
                        sos.se as usize,
                        sos.al,
                        &mut eob_run,
                    )?;
                }
            }
            mcus_since_restart += 1;
        }
    }
    Ok(())
}

/// Progressive DC decode for one block. `Ah == 0` decodes a category + sign
/// bits and stores `diff << Al` into coefficient 0. `Ah > 0` reads one extra
/// bit and ORs it into the existing DC value at bit `Al`.
fn prog_decode_dc(
    br: &mut BitReader<'_>,
    dc: &HuffTable,
    prev_dc: &mut i32,
    block: &mut [i32; 64],
    ah: u8,
    al: u8,
) -> Result<()> {
    if ah == 0 {
        let t = decode_huff(br, dc)? as u32;
        let dc_diff = if t == 0 {
            0
        } else {
            let bits = br.get_bits(t)? as i32;
            extend(bits, t)
        };
        *prev_dc = prev_dc.wrapping_add(dc_diff);
        block[0] = *prev_dc << al;
    } else {
        // Successive-approximation DC refinement: read 1 bit, shift into the
        // existing DC at position `Al`.
        let bit = br.get_bits(1)? as i32;
        block[0] |= bit << al;
    }
    Ok(())
}

/// First-pass AC scan over `[ss..=se]`. Uses an EOBn symbol scheme where the
/// low-nibble `s` with high-nibble `r` can also indicate the run-length of
/// all-zero blocks (`r=15, s=0` → ZRL; `r<15, s=0` → EOBn where `n = r`).
fn prog_decode_ac_first(
    br: &mut BitReader<'_>,
    ac: &HuffTable,
    block: &mut [i32; 64],
    ss: usize,
    se: usize,
    al: u8,
    eob_run: &mut u32,
) -> Result<()> {
    if *eob_run > 0 {
        *eob_run -= 1;
        return Ok(());
    }
    let mut k = ss;
    while k <= se {
        let rs = decode_huff(br, ac)?;
        let r = (rs >> 4) as usize;
        let s = (rs & 0x0F) as u32;
        if s == 0 {
            if r != 15 {
                // EOBn: skip 2^r - 1 additional blocks.
                let extra = if r == 0 { 0 } else { br.get_bits(r as u32)? };
                *eob_run = (1u32 << r) + extra - 1;
                return Ok(());
            }
            // ZRL → 16 zeros.
            k += 16;
        } else {
            k += r;
            if k > se {
                return Err(Error::invalid("progressive AC: run out of band"));
            }
            let bits = br.get_bits(s)? as i32;
            let val = extend(bits, s) << al;
            block[ZIGZAG[k]] = val;
            k += 1;
        }
    }
    Ok(())
}

/// Successive-approximation AC refinement scan. Each already-nonzero AC in
/// `[ss..=se]` gains one correction bit; new coefficients appear at positions
/// where the first pass coded a zero but this pass decodes a fresh sign.
fn prog_decode_ac_refine(
    br: &mut BitReader<'_>,
    ac: &HuffTable,
    block: &mut [i32; 64],
    ss: usize,
    se: usize,
    al: u8,
    eob_run: &mut u32,
) -> Result<()> {
    let p1: i32 = 1 << al;
    let m1: i32 = -1 << al;

    let mut k = ss;
    if *eob_run == 0 {
        while k <= se {
            let rs = decode_huff(br, ac)?;
            let mut r = (rs >> 4) as usize;
            let s = (rs & 0x0F) as u32;
            let new_val: i32;
            if s == 0 {
                if r != 15 {
                    // EOBn: refine remaining non-zeros then schedule n-1 EOB blocks.
                    let extra = if r == 0 { 0 } else { br.get_bits(r as u32)? };
                    *eob_run = (1u32 << r) + extra;
                    break;
                }
                // ZRL: skip 16 zero-history positions (refining any current nonzeros).
                new_val = 0;
            } else if s == 1 {
                // A new nonzero: read the sign bit.
                let sign_bit = br.get_bits(1)?;
                new_val = if sign_bit == 0 { m1 } else { p1 };
            } else {
                return Err(Error::invalid("progressive AC refine: bad s"));
            }

            // Walk through zeros until we've skipped `r` zero-history positions,
            // refining any existing non-zero we pass along the way.
            loop {
                if k > se {
                    return Err(Error::invalid("progressive AC refine: k past se"));
                }
                let pos = ZIGZAG[k];
                if block[pos] != 0 {
                    // Refine existing coefficient with one additional bit.
                    let bit = br.get_bits(1)? as i32;
                    if bit != 0 && (block[pos] & p1) == 0 {
                        if block[pos] >= 0 {
                            block[pos] += p1;
                        } else {
                            block[pos] += m1;
                        }
                    }
                } else if r == 0 {
                    break;
                } else {
                    r -= 1;
                }
                k += 1;
            }

            if new_val != 0 && k <= se {
                block[ZIGZAG[k]] = new_val;
            }
            k += 1;
        }
    }

    // Handle EOB-run tail: continue refining nonzero history for the rest of
    // the band (no new coefficients allowed).
    if *eob_run > 0 {
        while k <= se {
            let pos = ZIGZAG[k];
            if block[pos] != 0 {
                let bit = br.get_bits(1)? as i32;
                if bit != 0 && (block[pos] & p1) == 0 {
                    if block[pos] >= 0 {
                        block[pos] += p1;
                    } else {
                        block[pos] += m1;
                    }
                }
            }
            k += 1;
        }
        *eob_run -= 1;
    }

    Ok(())
}

/// After all progressive scans land, dequantise + IDCT each component block
/// and emit a VideoFrame.
fn render_progressive(
    state: &JpegState,
    coefs: &[Vec<[i32; 64]>],
    pts: Option<i64>,
    time_base: TimeBase,
) -> Result<VideoFrame> {
    let sof = state
        .sof
        .as_ref()
        .ok_or_else(|| Error::invalid("progressive: EOI before SOF"))?;
    let n_comp = sof.components.len();
    let grayscale = n_comp == 1;
    let width = sof.width as usize;
    let height = sof.height as usize;
    let h_max = sof.components.iter().map(|c| c.h_factor).max().unwrap_or(1) as usize;
    let v_max = sof.components.iter().map(|c| c.v_factor).max().unwrap_or(1) as usize;
    let mcus_x = width.div_ceil(8 * h_max);
    let mcus_y = height.div_ceil(8 * v_max);

    // Determine output pixel format (same rules as baseline).
    let out_format = if grayscale {
        PixelFormat::Gray8
    } else if n_comp == 3 {
        let y = sof.components[0];
        let cb = sof.components[1];
        let cr = sof.components[2];
        if cb.h_factor != cr.h_factor || cb.v_factor != cr.v_factor {
            return Err(Error::unsupported(
                "chroma components have different sampling factors",
            ));
        }
        if cb.h_factor != 1 || cb.v_factor != 1 {
            return Err(Error::unsupported("chroma components must have factor 1"));
        }
        match (y.h_factor, y.v_factor) {
            (1, 1) => PixelFormat::Yuv444P,
            (2, 1) => PixelFormat::Yuv422P,
            (2, 2) => PixelFormat::Yuv420P,
            _ => {
                return Err(Error::unsupported(format!(
                    "luma sampling {}x{}",
                    y.h_factor, y.v_factor
                )))
            }
        }
    } else {
        return Err(Error::unsupported("2-component JPEG"));
    };

    // Resolve quant tables.
    let quant_tables: Vec<&QuantTable> = sof
        .components
        .iter()
        .map(|c| {
            state.quant[c.qt_id as usize]
                .as_ref()
                .ok_or_else(|| Error::invalid("quant table missing for component"))
        })
        .collect::<Result<Vec<_>>>()?;

    // Per-component decoded sample buffer (full MCU coverage).
    let mut comp_buf: Vec<Vec<u8>> = Vec::with_capacity(n_comp);
    let mut comp_stride: Vec<usize> = Vec::with_capacity(n_comp);
    for c in &sof.components {
        let w_full = mcus_x * 8 * c.h_factor as usize;
        let h_full = mcus_y * 8 * c.v_factor as usize;
        comp_buf.push(vec![0u8; w_full * h_full]);
        comp_stride.push(w_full);
    }

    for (ci, c) in sof.components.iter().enumerate() {
        let blocks_x = mcus_x * c.h_factor as usize;
        let blocks_y = mcus_y * c.v_factor as usize;
        let qt = quant_tables[ci];
        let stride = comp_stride[ci];
        let buf = &mut comp_buf[ci];
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block = &coefs[ci][by * blocks_x + bx];
                let mut fblock = [0.0f32; 64];
                for k in 0..64 {
                    fblock[k] = (block[k] * qt.values[k] as i32) as f32;
                }
                idct8x8(&mut fblock);
                let dst_x0 = bx * 8;
                let dst_y0 = by * 8;
                for j in 0..8 {
                    for i in 0..8 {
                        let v = fblock[j * 8 + i] + 128.0;
                        let px = if v <= 0.0 {
                            0
                        } else if v >= 255.0 {
                            255
                        } else {
                            v.round() as u8
                        };
                        buf[(dst_y0 + j) * stride + dst_x0 + i] = px;
                    }
                }
            }
        }
    }

    // Build output.
    let mut planes: Vec<VideoPlane> = Vec::new();
    match out_format {
        PixelFormat::Gray8 => {
            let stride = width;
            let mut data = vec![0u8; stride * height];
            let src_stride = comp_stride[0];
            for y in 0..height {
                data[y * stride..y * stride + width]
                    .copy_from_slice(&comp_buf[0][y * src_stride..y * src_stride + width]);
            }
            planes.push(VideoPlane { stride, data });
        }
        PixelFormat::Yuv444P | PixelFormat::Yuv422P | PixelFormat::Yuv420P => {
            let (c_w, c_h) = match out_format {
                PixelFormat::Yuv444P => (width, height),
                PixelFormat::Yuv422P => (width.div_ceil(2), height),
                PixelFormat::Yuv420P => (width.div_ceil(2), height.div_ceil(2)),
                _ => unreachable!(),
            };
            let y_stride = width;
            let mut y_data = vec![0u8; y_stride * height];
            let src_stride_y = comp_stride[0];
            for y in 0..height {
                y_data[y * y_stride..y * y_stride + width]
                    .copy_from_slice(&comp_buf[0][y * src_stride_y..y * src_stride_y + width]);
            }
            planes.push(VideoPlane {
                stride: y_stride,
                data: y_data,
            });
            for ci in [1usize, 2] {
                let src_stride = comp_stride[ci];
                let stride = c_w;
                let mut data = vec![0u8; stride * c_h];
                for y in 0..c_h {
                    data[y * stride..y * stride + c_w]
                        .copy_from_slice(&comp_buf[ci][y * src_stride..y * src_stride + c_w]);
                }
                planes.push(VideoPlane { stride, data });
            }
        }
        _ => unreachable!(),
    }

    Ok(VideoFrame {
        format: out_format,
        width: width as u32,
        height: height as u32,
        pts,
        time_base,
        planes,
    })
}

#[cfg(test)]
mod prog_tests {
    use super::*;
    use crate::jpeg::huffman::{HuffTable, STD_DC_LUMA_BITS, STD_DC_LUMA_VALS};
    use crate::jpeg::parser::SofComponent;

    fn make_state(progressive: bool, width: u16, height: u16) -> JpegState {
        let mut s = JpegState::new();
        s.sof = Some(SofInfo {
            precision: 8,
            height,
            width,
            components: vec![SofComponent {
                id: 1,
                h_factor: 1,
                v_factor: 1,
                qt_id: 0,
            }],
        });
        s.progressive = progressive;
        s.quant[0] = Some(QuantTable { values: [1u16; 64] });
        s.dc_huff[0] = Some(HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS).unwrap());
        s
    }

    /// `init_progressive_coefs` should size the accumulator to the MCU grid.
    #[test]
    fn accumulator_sizes() {
        let sof = SofInfo {
            precision: 8,
            height: 16,
            width: 16,
            components: vec![
                SofComponent {
                    id: 1,
                    h_factor: 2,
                    v_factor: 2,
                    qt_id: 0,
                },
                SofComponent {
                    id: 2,
                    h_factor: 1,
                    v_factor: 1,
                    qt_id: 1,
                },
                SofComponent {
                    id: 3,
                    h_factor: 1,
                    v_factor: 1,
                    qt_id: 1,
                },
            ],
        };
        let coefs = init_progressive_coefs(&sof).unwrap();
        // 16x16 image, h_max=v_max=2 → 1 MCU. Y is 2x2 blocks, chroma is 1.
        assert_eq!(coefs[0].len(), 4);
        assert_eq!(coefs[1].len(), 1);
        assert_eq!(coefs[2].len(), 1);
        for blk in &coefs[0] {
            assert!(blk.iter().all(|&v| v == 0));
        }
    }

    /// Synthesize a DC-first scan buffer for a single-component 8×8 image
    /// with DC diff = 5 and feed it through `prog_decode_dc`. `Al = 2`, so
    /// the stored coefficient should be `5 << 2 = 20`.
    #[test]
    fn dc_first_pass_shifts_by_al() {
        let t = HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS).unwrap();
        // Emit category=3 (encoded with Huffman) followed by 3-bit value 0b101 (=5).
        // Category=3 in the standard luma DC table: (per Annex K) the code for
        // symbol 3 is "100" (3 bits).
        let code = t.encode[3];
        let mut bw = ProgTestBitWriter::new();
        bw.put(code.code as u32, code.len as u32);
        bw.put(0b101, 3);
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let mut prev_dc = 0i32;
        let mut block = [0i32; 64];
        prog_decode_dc(&mut br, &t, &mut prev_dc, &mut block, 0, 2).unwrap();
        assert_eq!(prev_dc, 5);
        assert_eq!(block[0], 20);
    }

    /// DC refinement (Ah>0) just appends a single bit at position `Al`.
    #[test]
    fn dc_refine_appends_bit() {
        // Build a buffer with a single `1` bit MSB-aligned.
        let mut bw = ProgTestBitWriter::new();
        bw.put(1, 1);
        let buf = bw.finish();

        let t = HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS).unwrap();
        let mut br = BitReader::new(&buf);
        let mut prev_dc = 0i32;
        let mut block = [0i32; 64];
        block[0] = 0b1000;
        prog_decode_dc(&mut br, &t, &mut prev_dc, &mut block, 1, 0).unwrap();
        assert_eq!(block[0], 0b1001);
    }

    /// `make_state` is currently used only by the test below.
    #[test]
    fn state_helper_is_coherent() {
        let s = make_state(true, 8, 8);
        assert!(s.progressive);
        assert!(s.sof.is_some());
    }

    // ---- tiny bit-stream writer (MSB first, byte-stuffing for 0xFF) --------

    struct ProgTestBitWriter {
        out: Vec<u8>,
        bits: u32,
        nbits: u32,
    }

    impl ProgTestBitWriter {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                bits: 0,
                nbits: 0,
            }
        }
        fn put(&mut self, val: u32, n: u32) {
            self.bits = (self.bits << n) | (val & ((1u32 << n) - 1));
            self.nbits += n;
            while self.nbits >= 8 {
                self.nbits -= 8;
                let b = ((self.bits >> self.nbits) & 0xFF) as u8;
                self.out.push(b);
                if b == 0xFF {
                    self.out.push(0x00);
                }
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.nbits > 0 {
                let pad = 8 - self.nbits;
                let b = (((self.bits << pad) | ((1u32 << pad) - 1)) & 0xFF) as u8;
                self.out.push(b);
                if b == 0xFF {
                    self.out.push(0x00);
                }
            }
            self.out
        }
    }

    /// First-pass AC decode with an explicit stream: symbol (r=0,s=1) + sign
    /// bit 1, then EOB (r=0,s=0) → one nonzero coefficient at `k=1`.
    #[test]
    fn ac_first_single_coef_then_eob() {
        use crate::jpeg::huffman::{STD_AC_LUMA_BITS, STD_AC_LUMA_VALS};
        let ac = HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS).unwrap();
        let mut bw = ProgTestBitWriter::new();
        // Symbol 0x01 (r=0, s=1) encoded.
        let c = ac.encode[0x01];
        bw.put(c.code as u32, c.len as u32);
        bw.put(1, 1); // coefficient value = +1
                      // EOB (r=0, s=0) → symbol 0x00.
        let c0 = ac.encode[0x00];
        bw.put(c0.code as u32, c0.len as u32);
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let mut block = [0i32; 64];
        let mut eob = 0u32;
        prog_decode_ac_first(&mut br, &ac, &mut block, 1, 63, 0, &mut eob).unwrap();
        // Coefficient at zigzag k=1 → natural position ZIGZAG[1] = 1.
        assert_eq!(block[ZIGZAG[1]], 1);
        for i in 0..64 {
            if i != ZIGZAG[1] {
                assert_eq!(block[i], 0, "unexpected nonzero at {i}");
            }
        }
    }

    /// `prog_decode_ac_first` with Al=2 must left-shift the decoded value.
    #[test]
    fn ac_first_pass_shifts_by_al() {
        use crate::jpeg::huffman::{STD_AC_LUMA_BITS, STD_AC_LUMA_VALS};
        let ac = HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS).unwrap();
        let mut bw = ProgTestBitWriter::new();
        let c = ac.encode[0x01];
        bw.put(c.code as u32, c.len as u32);
        bw.put(1, 1); // +1
        let c0 = ac.encode[0x00];
        bw.put(c0.code as u32, c0.len as u32);
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let mut block = [0i32; 64];
        let mut eob = 0u32;
        prog_decode_ac_first(&mut br, &ac, &mut block, 1, 63, 2, &mut eob).unwrap();
        assert_eq!(block[ZIGZAG[1]], 4); // 1 << 2
    }

    /// Refinement of an already-nonzero coefficient must add one correction
    /// bit at position `Al`.
    #[test]
    fn ac_refine_extends_existing_coef() {
        use crate::jpeg::huffman::{STD_AC_LUMA_BITS, STD_AC_LUMA_VALS};
        let ac = HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS).unwrap();
        // Refinement stream: EOBn with r=0, s=0 → EOB for just this block,
        // but we also need to feed refinement bits for existing nonzeros
        // beforehand. Simplest is: EOBn r=0 (eob_run = 1), which sets up the
        // "refine remaining non-zeros" tail in the same call.
        let mut bw = ProgTestBitWriter::new();
        // Symbol 0x00: r=0, s=0 → EOB0: eob_run = 1. Then refinement tail
        // will consume one bit per existing nonzero in [ss..=se]. Block has
        // one existing nonzero at k=1; give it a `1` refinement bit.
        let c0 = ac.encode[0x00];
        bw.put(c0.code as u32, c0.len as u32);
        bw.put(1, 1); // refine bit for the existing coefficient
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let mut block = [0i32; 64];
        block[ZIGZAG[1]] = 4; // existing coef = +4 (from Al=2 first pass)
        let mut eob = 0u32;
        prog_decode_ac_refine(&mut br, &ac, &mut block, 1, 63, 1, &mut eob).unwrap();
        // 4 (binary 0b100) + 0b10 (1 << Al=1) = 0b110 = 6.
        assert_eq!(block[ZIGZAG[1]], 6);
    }
}
