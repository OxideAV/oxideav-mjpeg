#![no_main]

//! Writer → reader round-trip through the general T.81 encoder
//! (`oxideav_mjpeg::t81`) and the public decoder, on fuzz-derived
//! pixels and fuzz-chosen options.
//!
//! Parameter space sampled per iteration:
//!
//! * process — sequential DCT (SOF0 / SOF1), progressive DCT (SOF2),
//!   lossless (SOF3, predictor 1..=7, point transform)
//! * precision — 8 / 12 for the DCT processes, 2..=16 for lossless
//!   (three- and four-component shapes stay within what the decoder
//!   shapes: subsampled YUV-class and CMYK-class at P = 8)
//! * layout — 1 / 3 / 4 components with every §A.1.1 sampling
//!   combination the §B.2.3 bound admits (4×2 luma, mixed chroma,
//!   chroma oversampled relative to luma, …)
//! * tables — Annex K.3 typical vs Annex K.2 optimal
//! * restart intervals (row-aligned for lossless, Table B.7)
//! * §B.5 abbreviated pair (tables-only stream + table-less frame,
//!   decoded through `decode_jpeg_with_tables`)
//! * colour signalling — JFIF / RGB (Adobe APP14 + 'R'/'G'/'B') / CMYK
//!   (no APP14, Adobe inverted, YCCK)
//!
//! Oracle: every accepted combination must decode, with the plane
//! geometry the pixel format implies; the lossless process (outside the
//! lossy YCCK convention) must reconstruct `(s >> Pt) << Pt` bit-exact
//! on every component.

use libfuzzer_sys::fuzz_target;
use oxideav_mjpeg::decoder::{decode_jpeg, decode_jpeg_with_tables};
use oxideav_mjpeg::t81::{ColorSignalling, HuffmanTables, JpegEncodeOptions, JpegProcess};

const MAX_PIXELS: usize = 512;

const LAYOUTS_3: &[&[(u8, u8)]] = &[
    &[(1, 1), (1, 1), (1, 1)],
    &[(2, 1), (1, 1), (1, 1)],
    &[(2, 2), (1, 1), (1, 1)],
    &[(4, 1), (1, 1), (1, 1)],
    &[(4, 2), (1, 1), (1, 1)],
    &[(1, 2), (1, 1), (1, 1)],
    &[(2, 2), (2, 1), (1, 2)],
    &[(1, 1), (2, 2), (2, 2)],
];
const LAYOUTS_4: &[&[(u8, u8)]] = &[
    &[(1, 1), (1, 1), (1, 1), (1, 1)],
    &[(2, 2), (1, 1), (1, 1), (1, 1)],
    &[(4, 1), (1, 1), (1, 1), (1, 1)],
];

/// Per-component full-resolution samples of a decoded frame (planar
/// with subsampled chroma, packed RGB / CMYK, planar GBR), or `None`
/// when the geometry is not one of the shapes the decoder documents.
fn decoded_components(
    f: &oxideav_core::VideoFrame,
    w: usize,
    h: usize,
    layout: &[(u8, u8)],
) -> Option<Vec<Vec<u32>>> {
    let nf = layout.len();
    let h_max = layout.iter().map(|s| s.0).max()? as usize;
    let v_max = layout.iter().map(|s| s.1).max()? as usize;
    let bps = if f.planes.len() == nf {
        f.planes[0].stride / w
    } else if f.planes.len() == 1 {
        f.planes[0].stride / (w * nf)
    } else {
        return None;
    };
    if !matches!(bps, 1 | 2) {
        return None;
    }
    let read = |data: &[u8], o: usize| -> u32 {
        if bps == 2 {
            u32::from(data[o]) | u32::from(data[o + 1]) << 8
        } else {
            u32::from(data[o])
        }
    };
    let mut out = Vec::with_capacity(nf);
    if f.planes.len() == nf {
        for (c, pl) in f.planes.iter().enumerate() {
            let cw = pl.stride / bps;
            let ch = pl.data.len() / pl.stride;
            // Either the frame grid (4:4:4 output) or the component's own
            // A.1.1 extent (native planar chroma).
            let (hi, vi) = if cw == w && ch == h {
                (h_max, v_max)
            } else {
                let (hi, vi) = (layout[c].0 as usize, layout[c].1 as usize);
                if cw != (w * hi).div_ceil(h_max) || ch != (h * vi).div_ceil(v_max) {
                    return None;
                }
                (hi, vi)
            };
            let mut full = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    full.push(read(
                        &pl.data,
                        (y * vi / v_max) * pl.stride + (x * hi / h_max) * bps,
                    ));
                }
            }
            out.push(full);
        }
    } else {
        let pl = &f.planes[0];
        for c in 0..nf {
            let mut full = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    full.push(read(&pl.data, y * pl.stride + (x * nf + c) * bps));
                }
            }
            out.push(full);
        }
    }
    Some(out)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    let process_sel = data[0] % 3;
    let nf = [1usize, 3, 4][(data[1] & 0x03) as usize % 3];
    let lossless = process_sel == 2;
    let layout: Vec<(u8, u8)> = match nf {
        1 => vec![(1, 1)],
        3 => LAYOUTS_3[(data[1] >> 2) as usize % LAYOUTS_3.len()].to_vec(),
        // The lossless decoder shapes four-component frames as packed
        // CMYK at H_i = V_i = 1 only (subsampled CMYK-class lossless
        // still returns `Unsupported`); the DCT paths upsample any layout.
        _ if lossless => LAYOUTS_4[0].to_vec(),
        _ => LAYOUTS_4[(data[1] >> 2) as usize % LAYOUTS_4.len()].to_vec(),
    };
    let subsampled = layout.iter().any(|&s| s != (1, 1));
    // Precision within what the decoder shapes for this layout.
    let precision: u8 = if lossless {
        if nf == 4 || subsampled {
            8
        } else {
            2 + data[2] % 15
        }
    } else if nf == 4 {
        8
    } else if data[2] & 1 == 0 {
        8
    } else {
        12
    };
    let tables = if data[3] & 1 == 0 {
        HuffmanTables::Typical
    } else {
        HuffmanTables::Optimal
    };
    let abbreviated = data[3] & 2 != 0;
    let predictor = 1 + data[4] % 7;
    let pt = if lossless { (data[4] >> 3) % precision.min(4) } else { 0 };
    let quality = 50 + data[5] % 51;
    let width = 1 + (data[6] % 16) as usize;
    let sig_sel = data[7] % 3;

    let h_max = layout.iter().map(|s| s.0).max().unwrap() as usize;
    let v_max = layout.iter().map(|s| s.1).max().unwrap() as usize;
    let bytes_per_sample = if precision > 8 { 2 } else { 1 };
    // Samples per pixel across components, at most `MAX_PIXELS` pixels.
    let payload = &data[8..];
    let per_pixel: usize = layout
        .iter()
        .map(|&(h, v)| h as usize * v as usize)
        .sum::<usize>()
        .max(1);
    let height = (payload.len() / (width * per_pixel * bytes_per_sample))
        .clamp(1, MAX_PIXELS / width);
    let max = (1u32 << precision) - 1;
    let mut cursor = 0usize;
    let mut planes: Vec<Vec<u16>> = Vec::with_capacity(nf);
    for &(h, v) in &layout {
        let cw = (width * h as usize).div_ceil(h_max);
        let ch = (height * v as usize).div_ceil(v_max);
        let mut p = Vec::with_capacity(cw * ch);
        for _ in 0..cw * ch {
            let v = if bytes_per_sample == 2 {
                let lo = payload.get(cursor).copied().unwrap_or(0) as u32;
                let hi = payload.get(cursor + 1).copied().unwrap_or(0) as u32;
                cursor += 2;
                lo | hi << 8
            } else {
                let b = payload.get(cursor).copied().unwrap_or(0) as u32;
                cursor += 1;
                b
            };
            p.push((v & max) as u16);
        }
        planes.push(p);
    }

    let mcur = if nf > 1 { width.div_ceil(h_max) } else { width };
    let restart_interval: u16 = if lossless {
        (mcur * (data[3] >> 2 & 3) as usize) as u16
    } else {
        (data[3] >> 2 & 3) as u16
    };
    let signalling = match (nf, sig_sel) {
        (3, 0) => ColorSignalling::Rgb,
        (4, 1) => ColorSignalling::Cmyk {
            adobe_transform: Some(0),
        },
        (4, 2) => ColorSignalling::Cmyk {
            adobe_transform: Some(2),
        },
        _ => ColorSignalling::Auto,
    };
    // The decoder shapes subsampled three-component lossless frames as
    // YUV-class planes; RGB signalling there would be a lie.
    let signalling = if subsampled && signalling == ColorSignalling::Rgb {
        ColorSignalling::Auto
    } else {
        signalling
    };
    let process = match process_sel {
        0 => JpegProcess::Sequential,
        1 => JpegProcess::Progressive,
        _ => JpegProcess::Lossless {
            predictor,
            point_transform: pt,
        },
    };
    let opts = JpegEncodeOptions {
        quality,
        tables,
        process,
        precision,
        restart_interval,
        abbreviated,
        signalling,
        sampling: layout.clone(),
        table_ids: Vec::new(),
    };
    let refs: Vec<&[u16]> = planes.iter().map(|p| p.as_slice()).collect();
    let out = opts
        .encode(width as u32, height as u32, &refs)
        .unwrap_or_else(|e| panic!("t81 encode refused a legal combination ({opts:?}): {e}"));
    let decoded = match &out.tables {
        Some(t) => decode_jpeg_with_tables(t, &out.data, None),
        None => decode_jpeg(&out.data, None),
    }
    .unwrap_or_else(|e| panic!("decoder rejected the t81 stream ({opts:?}): {e}"));
    let got = decoded_components(&decoded, width, height, &layout)
        .unwrap_or_else(|| panic!("unexpected decoded geometry ({opts:?})"));
    if lossless
        && signalling
            != (ColorSignalling::Cmyk {
                adobe_transform: Some(2),
            })
    {
        // Adobe-inverted CMYK codes `max − v`; the point transform acts on
        // that coded value, so reconstruction is `max − (((max − v) >> Pt)
        // << Pt)` — the inversion and the shift do not commute.
        let inverted = signalling
            == (ColorSignalling::Cmyk {
                adobe_transform: Some(0),
            });
        for (c, &(h, v)) in layout.iter().enumerate() {
            let cw = (width * h as usize).div_ceil(h_max);
            for y in 0..height {
                for x in 0..width {
                    let sx = x * h as usize / h_max;
                    let sy = y * v as usize / v_max;
                    let src = u32::from(planes[c][sy * cw + sx]);
                    let wire = if inverted { max - src } else { src };
                    let recon = (wire >> pt) << pt;
                    let want = if inverted { max - recon } else { recon };
                    let have = got[c][y * width + x];
                    assert_eq!(
                        have, want,
                        "lossless mismatch at component {c} ({x},{y}) ({opts:?})"
                    );
                }
            }
        }
    }
});
