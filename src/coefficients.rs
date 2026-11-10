//! Quantised DCT coefficients — the level below samples (#21).
//!
//! A DCT-based JPEG stream (T.81 Annex F sequential, Annex G
//! progressive, Huffman or arithmetic) codes, per component, a grid of
//! 8 × 8 blocks of quantised DCT coefficients `Sq_vu` (§A.3.4) together
//! with the quantisation tables `Q_vu` (§B.2.4.1) the decoder multiplies
//! them by before the inverse DCT (§A.3.5: `R_vu = Sq_vu × Q_vu`).
//! [`read_coefficients`] stops there: it runs the entropy decoder over
//! every scan and returns the coefficients and tables without
//! dequantising or transforming, so a caller can run its own inverse
//! DCT, inspect the quantisation, or transform the image losslessly in
//! the coefficient domain. [`encode_coefficients`] goes the other way —
//! it entropy-codes a [`JpegCoefficients`] record (Huffman, sequential
//! or progressive, typical or optimal tables) without touching a single
//! coefficient, so `encode_coefficients(&read_coefficients(b)?, …)` is a
//! lossless transcode of `b`. [`decode_coefficients`] renders a record
//! to samples with the crate's own inverse DCT.
//!
//! # Layout
//!
//! Each [`CoefficientComponent`] holds its blocks in raster order on
//! the **MCU-padded** grid of §A.2.4: `blocks_wide = mcus_x × Hi` by
//! `blocks_high = mcus_y × Vi` (`mcus_x = ⌈X / (8 × Hmax)⌉`, `mcus_y =
//! ⌈Y / (8 × Vmax)⌉`) — for the usual `1 × 1` single-component frame
//! that is `⌈X / 8⌉ × ⌈Y / 8⌉`. [`JpegCoefficients::block_grid`] computes it;
//! [`JpegCoefficients::block_extent`] gives the `⌈xi / 8⌉ × ⌈yi / 8⌉`
//! blocks that cover real samples (§A.1.1) — the rest are padding,
//! coded only by interleaved scans. Within a block the 64 coefficients
//! are in natural order, row-major by vertical frequency: index
//! `v × 8 + u` holds `Sq_vu` (`[0]` is the DC coefficient), not the
//! zig-zag order of Figure A.6. Quantisation tables use the same natural
//! order.
//!
//! Hierarchical sequences (DHP) and the lossless processes (SOF3 /
//! SOF11) are refused with [`Error::Unsupported`](crate::Error): the
//! former code differential frames, the latter carry no DCT
//! coefficients.
//!
//! Builds without `oxideav-core`.

use crate::error::{MjpegError as Error, Result};
use crate::image::{DecodeOptions, JpegImage, Metadata};
use crate::jpeg::inspect::SofKind;
use crate::t81::{EncodeOptions, HuffmanTables};

/// One component of a [`JpegCoefficients`] record.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct CoefficientComponent {
    /// Component identifier `Ci` (§B.2.2).
    pub id: u8,
    /// Horizontal sampling factor `Hi` (1..=4).
    pub h: u8,
    /// Vertical sampling factor `Vi` (1..=4).
    pub v: u8,
    /// Quantisation table destination `Tqi` (0..=3).
    pub quant_id: u8,
    /// The blocks on the MCU-padded grid, raster order
    /// ([`JpegCoefficients::block_grid`]); each block in natural order
    /// (`v × 8 + u`).
    pub blocks: Vec<[i32; 64]>,
}

impl CoefficientComponent {
    /// A component with the given identifier, sampling factors and
    /// quantisation destination, and no blocks yet.
    pub fn new(id: u8, h: u8, v: u8, quant_id: u8) -> Self {
        Self {
            id,
            h,
            v,
            quant_id,
            blocks: Vec::new(),
        }
    }

    /// Set the blocks (MCU-padded grid, raster order, natural order
    /// within each block).
    #[must_use]
    pub fn with_blocks(mut self, blocks: Vec<[i32; 64]>) -> Self {
        self.blocks = blocks;
        self
    }
}

/// The quantised DCT coefficients and quantisation tables of one
/// DCT-based frame (see the [module docs](self) for the layout).
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct JpegCoefficients {
    /// Frame width `X`.
    pub width: u16,
    /// Frame height `Y` (the DNL-supplied line count when the frame
    /// header codes `Y = 0`).
    pub height: u16,
    /// Sample precision `P`: 8 or 12.
    pub precision: u8,
    /// The frame's coding process as read (`None` for a record built by
    /// hand); informational — [`encode_coefficients`] codes with the
    /// process its options select.
    pub kind: Option<SofKind>,
    /// Quantisation tables by destination, natural order (`v × 8 + u`).
    pub quant_tables: [Option<[u16; 64]>; 4],
    /// The frame components, in frame-header order.
    pub components: Vec<CoefficientComponent>,
    /// The Adobe APP14 colour-transform flag the stream carried (T.872
    /// §6.5.3), if any: `0` RGB / complemented CMYK, `1` YCbCr, `2`
    /// YCCK. [`encode_coefficients`] writes it back (the coefficients
    /// are not colour-converted).
    pub adobe_transform: Option<u8>,
    /// The `JFIF` `APP0` segment was present.
    pub jfif: bool,
    /// The ICC / Exif / XMP / `COM` metadata of the stream
    /// ([`crate::read_metadata`]).
    pub metadata: Metadata,
}

impl JpegCoefficients {
    /// An empty record for a `width × height` frame at precision `P`.
    pub fn new(width: u16, height: u16, precision: u8) -> Self {
        Self {
            width,
            height,
            precision,
            ..Self::default()
        }
    }

    /// Set quantisation table `tq` (natural order).
    #[must_use]
    pub fn with_quant_table(mut self, tq: usize, table: [u16; 64]) -> Self {
        self.quant_tables[tq & 3] = Some(table);
        self
    }

    /// Append a component.
    #[must_use]
    pub fn with_component(mut self, c: CoefficientComponent) -> Self {
        self.components.push(c);
        self
    }

    /// Set the Adobe APP14 colour-transform flag.
    #[must_use]
    pub fn with_adobe_transform(mut self, t: Option<u8>) -> Self {
        self.adobe_transform = t;
        self
    }

    /// Set the metadata.
    #[must_use]
    pub fn with_metadata(mut self, m: Metadata) -> Self {
        self.metadata = m;
        self
    }

    fn max_factors(&self) -> (usize, usize) {
        let h = self
            .components
            .iter()
            .map(|c| c.h)
            .max()
            .unwrap_or(1)
            .max(1);
        let v = self
            .components
            .iter()
            .map(|c| c.v)
            .max()
            .unwrap_or(1)
            .max(1);
        (h as usize, v as usize)
    }

    /// `(blocks_wide, blocks_high)` of component `ci`'s MCU-padded block
    /// grid — the shape of [`CoefficientComponent::blocks`].
    pub fn block_grid(&self, ci: usize) -> (usize, usize) {
        let c = &self.components[ci];
        let (hm, vm) = self.max_factors();
        let mcus_x = (self.width as usize).div_ceil(8 * hm);
        let mcus_y = (self.height as usize).div_ceil(8 * vm);
        (mcus_x * c.h as usize, mcus_y * c.v as usize)
    }

    /// `(⌈xi / 8⌉, ⌈yi / 8⌉)`: the blocks of component `ci` that cover
    /// real samples (§A.1.1 `xi = ⌈X × Hi / Hmax⌉`).
    pub fn block_extent(&self, ci: usize) -> (usize, usize) {
        let c = &self.components[ci];
        let (hm, vm) = self.max_factors();
        let xi = (self.width as usize * c.h as usize).div_ceil(hm);
        let yi = (self.height as usize * c.v as usize).div_ceil(vm);
        (xi.div_ceil(8), yi.div_ceil(8))
    }

    /// Check the record: geometry, sampling factors, table references
    /// (every entry `Qk ≥ 1`; 16-bit entries are kept even at `P = 8`,
    /// where Table B.4 asks for 8-bit ones, so such streams transcode
    /// as read),
    /// block counts, and the coefficient magnitudes the Huffman coding
    /// of Tables F.1 / F.2 can carry: AC `|Sq| < 2^(P + 2)` (category ≤
    /// 10 at `P = 8`, ≤ 14 at `P = 12`), DC `|Sq| < 2^(P + 3)` (the DC
    /// difference between neighbouring blocks must still fit category
    /// 11 / 15; a record that breaks that is refused when it is coded).
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(Error::invalid("JPEG coefficients: zero frame dimension"));
        }
        if self.precision != 8 && self.precision != 12 {
            return Err(Error::invalid(format!(
                "JPEG coefficients: precision {} (DCT processes use 8 or 12)",
                self.precision
            )));
        }
        if self.components.is_empty() || self.components.len() > 4 {
            return Err(Error::invalid(format!(
                "JPEG coefficients: {} components (1..=4)",
                self.components.len()
            )));
        }
        let ac_limit = 1i32 << (self.precision + 2);
        let dc_limit = ac_limit << 1;
        for (ci, c) in self.components.iter().enumerate() {
            if !(1..=4).contains(&c.h) || !(1..=4).contains(&c.v) {
                return Err(Error::invalid(format!(
                    "JPEG coefficients: component {ci} sampling {}x{} (1..=4)",
                    c.h, c.v
                )));
            }
            if c.quant_id > 3 || self.quant_tables[c.quant_id as usize].is_none() {
                return Err(Error::invalid(format!(
                    "JPEG coefficients: component {ci} references undefined quantisation table {}",
                    c.quant_id
                )));
            }
            let (bw, bh) = self.block_grid(ci);
            if c.blocks.len() != bw * bh {
                return Err(Error::invalid(format!(
                    "JPEG coefficients: component {ci} carries {} blocks, its {bw}x{bh} grid needs {}",
                    c.blocks.len(),
                    bw * bh
                )));
            }
            let bad = c.blocks.iter().any(|b| {
                b[0] <= -dc_limit
                    || b[0] >= dc_limit
                    || b[1..].iter().any(|&q| q <= -ac_limit || q >= ac_limit)
            });
            if bad {
                return Err(Error::invalid(format!(
                    "JPEG coefficients: component {ci} has a coefficient outside DC ±{} / AC ±{} (P = {})",
                    dc_limit - 1,
                    ac_limit - 1,
                    self.precision
                )));
            }
        }
        for t in self.quant_tables.iter().flatten() {
            if t.contains(&0) {
                return Err(Error::invalid(
                    "JPEG coefficients: a quantisation table entry is zero (B.2.4.1: Qk ≥ 1)",
                ));
            }
        }
        Ok(())
    }
}

/// Read the quantised DCT coefficients of a DCT-based JPEG stream
/// (sequential or progressive, Huffman or arithmetic, `P` = 8 or 12;
/// see the [module docs](self)). Hierarchical and lossless frames are
/// [`Error::Unsupported`](crate::Error).
pub fn read_coefficients(bytes: &[u8]) -> Result<JpegCoefficients> {
    read_coefficients_with(bytes, &DecodeOptions::default())
}

/// [`read_coefficients`] honouring the geometry limits, `max_bytes` and
/// the §B.5 tables stream of `opts` (the reconstruction knobs do not
/// apply).
pub fn read_coefficients_with(bytes: &[u8], opts: &DecodeOptions) -> Result<JpegCoefficients> {
    if let Some(max) = opts.max_bytes
        && bytes.len() as u64 > max
    {
        return Err(Error::limit(format!(
            "JPEG: input of {} bytes exceeds max_bytes = {max}",
            bytes.len()
        )));
    }
    let limits = crate::decoder::DecodeLimits {
        max_width: opts.max_width.unwrap_or(u32::MAX),
        max_height: opts.max_height.unwrap_or(u32::MAX),
        max_pixels: opts.max_pixels.unwrap_or(u64::MAX),
    };
    let mut out = crate::decoder::read_coefficients_raw(opts.tables.as_deref(), bytes, limits)?;
    let hdr = crate::api::header_flags(bytes)?;
    out.jfif = hdr.0;
    out.metadata = hdr.1;
    Ok(out)
}

/// Render a coefficient record to samples with this crate's
/// dequantisation and inverse DCT — the image [`crate::decode_with`]
/// would produce for a stream carrying these coefficients (the
/// `idct` / `upsampling` knobs of `opts` apply; colour is labelled as
/// `decode` labels it, metadata is copied from the record).
pub fn decode_coefficients(coefs: &JpegCoefficients, opts: &DecodeOptions) -> Result<JpegImage> {
    coefs.validate()?;
    let mut img = crate::decoder::render_coefficients(coefs, opts.idct, opts.upsampling)?;
    img.format = img.format.full_range_label();
    img.color = crate::api::color_for(img.format);
    img.metadata = coefs.metadata.clone();
    Ok(img)
}

/// Entropy-code a coefficient record as one JPEG interchange stream,
/// coefficients and quantisation tables untouched.
///
/// From `opts` only these apply: `process` ([`JpegProcess::Sequential`](crate::t81::JpegProcess::Sequential)
/// → `SOF0` / `SOF1`, [`JpegProcess::Progressive`](crate::t81::JpegProcess::Progressive) → `SOF2`; lossless
/// is refused), `tables` (Annex K typical or K.2 optimal Huffman tables;
/// `P = 12` always uses optimal ones), `restart_interval`, `metadata`
/// (merged over the record's own, the option winning per kind) and
/// `jfif`. Component identifiers, sampling factors, quantisation
/// destinations and the Adobe flag come from the record; a record
/// without an Adobe flag gets a JFIF `APP0` for one or three components
/// when its source had one (or was built by hand). Each component uses
/// entropy table destination 0 if it is the first, 1 otherwise. A
/// single-component frame is written with sampling factors `1 × 1`
/// (§A.2.2 codes it non-interleaved whatever it declares).
pub fn encode_coefficients(coefs: &JpegCoefficients, opts: &EncodeOptions) -> Result<Vec<u8>> {
    coefs.validate()?;
    if !opts.process.is_dct() {
        return Err(Error::unsupported(
            "JPEG coefficients: a lossless process codes samples, not DCT coefficients",
        ));
    }
    let nf = coefs.components.len();
    let jfif = coefs.adobe_transform.is_none()
        && (nf == 1 || nf == 3)
        && (coefs.jfif || coefs.kind.is_none());
    let optimal = opts.tables == HuffmanTables::Optimal || coefs.precision > 8;
    let meta = coefs.metadata.with_overrides(&opts.metadata);
    crate::t81::encode_coefficient_frame(coefs, opts, optimal, jfif, &meta)
}
