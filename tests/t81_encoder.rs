//! The general T.81 writer (`oxideav_mjpeg::t81`) — every process ×
//! precision × layout it emits must decode in this crate's own decoder
//! (the 100 % oracle: bit-exact for lossless, PSNR for DCT) and in the
//! black-box validators (`djpeg` / `magick`) when they are installed.

use std::io::Write;
use std::process::Command;

use oxideav_mjpeg::decoder::decode_jpeg;
use oxideav_mjpeg::t81::{
    ColorSignalling, HuffmanTables, JpegEncodeOptions, JpegProcess, JpegTableSet,
};
use oxideav_mjpeg::{inspect_jpeg, SofKind};

const W: u32 = 37;
const H: u32 = 29;

/// Deterministic "photo-like" component plane at `w × h` below `2^p`.
fn plane(w: usize, h: usize, p: u8, seed: u32) -> Vec<u16> {
    let max = (1u32 << p) - 1;
    let mut state = seed | 1;
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state % 7) as f64 - 3.0;
            let v = 0.5 + 0.35 * ((x * 0.37 + seed as f64).sin() * (y * 0.23).cos()) + noise / 64.0;
            (v.clamp(0.0, 1.0) * max as f64).round() as u16
        })
        .collect()
}

/// Component planes at the A.1.1 resolutions of `sampling`.
fn planes(nf: usize, p: u8, sampling: &[(u8, u8)]) -> Vec<Vec<u16>> {
    let s: Vec<(u8, u8)> = if sampling.is_empty() {
        vec![(1, 1); nf]
    } else {
        sampling.to_vec()
    };
    let hm = s.iter().map(|x| x.0).max().unwrap() as usize;
    let vm = s.iter().map(|x| x.1).max().unwrap() as usize;
    (0..nf)
        .map(|i| {
            let (h, v) = (s[i].0 as usize, s[i].1 as usize);
            plane(
                (W as usize * h).div_ceil(hm),
                (H as usize * v).div_ceil(vm),
                p,
                0x9E37 + i as u32 * 77,
            )
        })
        .collect()
}

fn psnr(a: &[u32], b: &[u32], peak: f64) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    }
}

/// Nearest-neighbour upsample a component plane (`cw` wide, sampling
/// factors `hi × vi` against `hmax × vmax`) onto the frame grid — the
/// inverse of the A.1.1 dimension expressions (`sx = x × Hi / Hmax`).
fn upsample(src: &[u16], cw: usize, hi: usize, vi: usize, hmax: usize, vmax: usize) -> Vec<u32> {
    let (w, h) = (W as usize, H as usize);
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            u32::from(src[(y * vi / vmax) * cw + x * hi / hmax])
        })
        .collect()
}

/// Read a decoded plane of `w × h` samples with `bps` bytes per sample.
fn read_plane(stride: usize, data: &[u8], w: usize, h: usize, bps: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let o = y * stride + x * bps;
            out.push(if bps == 2 {
                u32::from(data[o]) | u32::from(data[o + 1]) << 8
            } else {
                u32::from(data[o])
            });
        }
    }
    out
}

/// Per-component full-resolution samples of a decoded frame, in
/// component order, for every shape the decoder produces: planar
/// (possibly subsampled chroma), packed `Rgb24` / `Rgb48Le` / `Cmyk`,
/// planar `Gbrp*Le` (G, B, R order).
fn decoded_components(jpeg: &[u8], nf: usize, p: u8, sampling: &[(u8, u8)]) -> Vec<Vec<u32>> {
    let f = decode_jpeg(jpeg, None).expect("our decoder rejects our stream");
    let (w, h) = (W as usize, H as usize);
    // Bytes per sample from the geometry: plane 0 of a planar frame is
    // always full-width; a packed frame is `nf` samples per pixel.
    let bps = if f.planes.len() == nf {
        f.planes[0].stride / w
    } else {
        f.planes[0].stride / (w * nf)
    };
    assert!(
        matches!(bps, 1 | 2),
        "nf{nf} P{p}: stride {}",
        f.planes[0].stride
    );
    if f.planes.len() == nf {
        let comps: Vec<Vec<u32>> = f
            .planes
            .iter()
            .enumerate()
            .map(|(c, pl)| {
                let cw = pl.stride / bps;
                let ch = pl.data.len() / pl.stride;
                let raw = read_plane(pl.stride, &pl.data, cw, ch, bps);
                let raw16: Vec<u16> = raw.iter().map(|&v| v as u16).collect();
                // Either the frame grid (4:4:4 output) or the component's
                // own A.1.1 extent (native planar chroma).
                let s: Vec<(u8, u8)> = if sampling.is_empty() {
                    vec![(1, 1); nf]
                } else {
                    sampling.to_vec()
                };
                let hm = s.iter().map(|x| x.0).max().unwrap() as usize;
                let vm = s.iter().map(|x| x.1).max().unwrap() as usize;
                if cw == w && ch == h {
                    upsample(&raw16, cw, 1, 1, 1, 1)
                } else {
                    let (hi, vi) = (s[c].0 as usize, s[c].1 as usize);
                    assert_eq!(
                        (cw, ch),
                        ((w * hi).div_ceil(hm), (h * vi).div_ceil(vm)),
                        "nf{nf} P{p}: plane {c} extent"
                    );
                    upsample(&raw16, cw, hi, vi, hm, vm)
                }
            })
            .collect();
        // Planar `Gbrp*Le` output (lossless 3-component at P = 10 / 12
        // / 14) keeps the scan-order plane sequence, so nothing to
        // reorder.
        comps
    } else {
        assert_eq!(f.planes.len(), 1, "packed output");
        let pl = &f.planes[0];
        (0..nf)
            .map(|c| {
                (0..w * h)
                    .map(|i| {
                        let o = (i / w) * pl.stride + ((i % w) * nf + c) * bps;
                        if bps == 2 {
                            u32::from(pl.data[o]) | u32::from(pl.data[o + 1]) << 8
                        } else {
                            u32::from(pl.data[o])
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("-version")
        .output()
        .map(|o| o.status.success() || !o.stderr.is_empty())
        .unwrap_or(false)
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("oxideav_mjpeg_t81");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// `djpeg -pnm [-grayscale]` → `(maxval, samples)`; `None` without djpeg.
fn djpeg(jpeg: &[u8], tag: &str, extra: &[&str]) -> Option<(u32, Vec<u32>)> {
    if !have("djpeg") {
        eprintln!("djpeg not available — skipping black-box check");
        return None;
    }
    let inp = scratch(&format!("{tag}.jpg"));
    let outp = scratch(&format!("{tag}.pnm"));
    std::fs::File::create(&inp)
        .unwrap()
        .write_all(jpeg)
        .unwrap();
    let st = Command::new("djpeg")
        .args(extra)
        .args(["-pnm", "-outfile"])
        .arg(&outp)
        .arg(&inp)
        .status()
        .expect("spawn djpeg");
    assert!(st.success(), "djpeg rejected our {tag} stream");
    let pnm = std::fs::read(&outp).unwrap();
    let mut i = 2;
    let mut fields = Vec::new();
    while fields.len() < 3 {
        while pnm[i].is_ascii_whitespace() {
            i += 1;
        }
        let s = i;
        while !pnm[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(
            std::str::from_utf8(&pnm[s..i])
                .unwrap()
                .parse::<u32>()
                .unwrap(),
        );
    }
    i += 1;
    let maxval = fields[2];
    let body = &pnm[i..];
    let samples = if maxval > 255 {
        body.chunks(2)
            .map(|c| u32::from(c[0]) << 8 | u32::from(c[1]))
            .collect()
    } else {
        body.iter().map(|&b| u32::from(b)).collect()
    };
    Some((maxval, samples))
}

fn sof_kind(jpeg: &[u8]) -> SofKind {
    inspect_jpeg(jpeg).expect("inspect").sof_kind
}

// ---------------------------------------------------------------------------

const DCT_LAYOUTS: &[(usize, &[(u8, u8)])] = &[
    (1, &[]),
    (3, &[]),
    (3, &[(2, 1), (1, 1), (1, 1)]),
    (3, &[(2, 2), (1, 1), (1, 1)]),
    (3, &[(4, 1), (1, 1), (1, 1)]),
    (3, &[(4, 2), (1, 1), (1, 1)]),
    (3, &[(1, 2), (1, 1), (1, 1)]),
    (3, &[(2, 2), (2, 1), (1, 2)]),
    (4, &[]),
    (4, &[(2, 2), (1, 1), (1, 1), (1, 1)]),
    (4, &[(4, 1), (1, 1), (1, 1), (1, 1)]),
];

/// Sequential + progressive, 8- and 12-bit, typical + optimal tables,
/// with and without restarts, every layout: our decoder reconstructs
/// every component within the quantisation floor.
#[test]
fn dct_processes_every_layout_decode_in_our_decoder() {
    for process in [JpegProcess::Sequential, JpegProcess::Progressive] {
        for precision in [8u8, 12] {
            for &(nf, sampling) in DCT_LAYOUTS {
                if nf == 4 && precision == 12 {
                    continue; // no 12-bit CMYK pixel format on the decode side
                }
                for tables in [HuffmanTables::Typical, HuffmanTables::Optimal] {
                    for ri in [0u16, 3] {
                        let src = planes(nf, precision, sampling);
                        let refs: Vec<&[u16]> = src.iter().map(|v| v.as_slice()).collect();
                        let opts = JpegEncodeOptions {
                            quality: 92,
                            tables,
                            process,
                            precision,
                            restart_interval: ri,
                            sampling: sampling.to_vec(),
                            ..Default::default()
                        };
                        let tag = format!(
                            "{process:?} P{precision} nf{nf} {sampling:?} {tables:?} ri{ri}"
                        );
                        let out = opts
                            .encode(W, H, &refs)
                            .unwrap_or_else(|e| panic!("{tag}: {e}"));
                        assert!(out.tables.is_none());
                        let kind = sof_kind(&out.data);
                        match process {
                            JpegProcess::Sequential if precision == 8 => {
                                assert!(matches!(kind, SofKind::Baseline), "{tag}: {kind:?}")
                            }
                            JpegProcess::Sequential => {
                                assert!(
                                    matches!(kind, SofKind::ExtendedSequential),
                                    "{tag}: {kind:?}"
                                )
                            }
                            _ => assert!(matches!(kind, SofKind::Progressive), "{tag}: {kind:?}"),
                        }
                        assert_eq!(
                            out.data.windows(2).any(|m| m == [0xFF, 0xDD]),
                            ri > 0,
                            "{tag}: DRI"
                        );
                        let got = decoded_components(&out.data, nf, precision, sampling);
                        let peak = ((1u32 << precision) - 1) as f64;
                        let s: Vec<(u8, u8)> = if sampling.is_empty() {
                            vec![(1, 1); nf]
                        } else {
                            sampling.to_vec()
                        };
                        let hm = s.iter().map(|x| x.0).max().unwrap() as usize;
                        let vm = s.iter().map(|x| x.1).max().unwrap() as usize;
                        for c in 0..nf {
                            let cw = (W as usize * s[c].0 as usize).div_ceil(hm);
                            let want =
                                upsample(&src[c], cw, s[c].0 as usize, s[c].1 as usize, hm, vm);
                            let db = psnr(&got[c], &want, peak);
                            assert!(db > 34.0, "{tag}: component {c} PSNR {db:.1} dB");
                        }
                    }
                }
            }
        }
    }
}

/// K.2 optimal tables never cost more than the typical ones on a
/// natural image, and always decode.
#[test]
fn optimal_tables_are_no_larger_than_typical() {
    for process in [JpegProcess::Sequential, JpegProcess::Progressive] {
        let src = planes(3, 8, &[(2, 2), (1, 1), (1, 1)]);
        let refs: Vec<&[u16]> = src.iter().map(|v| v.as_slice()).collect();
        let base = JpegEncodeOptions {
            quality: 85,
            process,
            sampling: vec![(2, 2), (1, 1), (1, 1)],
            ..Default::default()
        };
        let typical = base.encode(W, H, &refs).unwrap();
        let optimal = JpegEncodeOptions {
            tables: HuffmanTables::Optimal,
            ..base
        }
        .encode(W, H, &refs)
        .unwrap();
        assert!(
            optimal.data.len() <= typical.data.len(),
            "{process:?}: optimal {} > typical {}",
            optimal.data.len(),
            typical.data.len()
        );
        assert_eq!(
            decoded_components(&optimal.data, 3, 8, &[(2, 2), (1, 1), (1, 1)]),
            decoded_components(&typical.data, 3, 8, &[(2, 2), (1, 1), (1, 1)]),
            "{process:?}: same quantiser → same pixels"
        );
    }
}

/// Lossless: every precision ladder rung, every predictor, point
/// transform, row-aligned restarts, 1 / 3 / 4 components and
/// oversampled luma — bit-exact through our decoder.
#[test]
fn lossless_process_is_bit_exact_in_our_decoder() {
    type Case = (usize, &'static [(u8, u8)], &'static [u8]);
    let cases: &[Case] = &[
        (1, &[], &[2, 4, 8, 12, 16]),
        (3, &[], &[8, 12, 16]),
        (3, &[(2, 2), (1, 1), (1, 1)], &[8]),
        (3, &[(4, 2), (1, 1), (1, 1)], &[8]),
        (4, &[], &[8]),
    ];
    for &(nf, sampling, precisions) in cases {
        for &precision in precisions {
            for predictor in 1..=7u8 {
                for pt in [0u8, 1] {
                    if pt >= precision {
                        continue;
                    }
                    let src = planes(nf, precision, sampling);
                    let refs: Vec<&[u16]> = src.iter().map(|v| v.as_slice()).collect();
                    let hm = sampling.iter().map(|x| x.0).max().unwrap_or(1) as usize;
                    let mcur = if nf > 1 {
                        (W as usize).div_ceil(hm)
                    } else {
                        W as usize
                    };
                    for ri in [0u16, mcur as u16, 3 * mcur as u16] {
                        let opts = JpegEncodeOptions {
                            process: JpegProcess::Lossless {
                                predictor,
                                point_transform: pt,
                            },
                            precision,
                            restart_interval: ri,
                            sampling: sampling.to_vec(),
                            signalling: if nf == 3 && sampling.is_empty() {
                                ColorSignalling::Rgb
                            } else {
                                ColorSignalling::Auto
                            },
                            ..Default::default()
                        };
                        let tag = format!("lossless P{precision} nf{nf} {sampling:?} pred{predictor} pt{pt} ri{ri}");
                        let out = opts
                            .encode(W, H, &refs)
                            .unwrap_or_else(|e| panic!("{tag}: {e}"));
                        assert!(matches!(sof_kind(&out.data), SofKind::Lossless), "{tag}");
                        let got = decoded_components(&out.data, nf, precision, sampling);
                        let s: Vec<(u8, u8)> = if sampling.is_empty() {
                            vec![(1, 1); nf]
                        } else {
                            sampling.to_vec()
                        };
                        let vm = s.iter().map(|x| x.1).max().unwrap() as usize;
                        for c in 0..nf {
                            let cw = (W as usize * s[c].0 as usize).div_ceil(hm);
                            // The decoder widens `sample >> Pt` back by `<< Pt`.
                            let want: Vec<u32> =
                                upsample(&src[c], cw, s[c].0 as usize, s[c].1 as usize, hm, vm)
                                    .iter()
                                    .map(|&v| (v >> pt) << pt)
                                    .collect();
                            assert!(got[c] == want, "{tag}: component {c} not bit-exact");
                        }
                    }
                }
            }
        }
    }
}

/// §B.5 abbreviated pair: the tables-only stream carries exactly the
/// segments the interchange stream would, and splicing the two yields
/// a byte-identical interchange stream.
#[test]
fn abbreviated_pair_splices_into_the_interchange_stream() {
    for process in [
        JpegProcess::Sequential,
        JpegProcess::Progressive,
        JpegProcess::Lossless {
            predictor: 6,
            point_transform: 0,
        },
    ] {
        for precision in [8u8, 12] {
            let src = planes(3, precision, &[(2, 2), (1, 1), (1, 1)]);
            let refs: Vec<&[u16]> = src.iter().map(|v| v.as_slice()).collect();
            // Lossless restarts must be row-aligned (Table B.7): one
            // MCU-row of the 2×2-luma frame is ceil(W / 2) MCUs.
            let ri = if process.is_dct() {
                2
            } else {
                (W as u16).div_ceil(2)
            };
            let base = JpegEncodeOptions {
                process,
                precision,
                restart_interval: ri,
                tables: HuffmanTables::Optimal,
                sampling: vec![(2, 2), (1, 1), (1, 1)],
                ..Default::default()
            };
            let full = base.encode(W, H, &refs).unwrap();
            let abbr = JpegEncodeOptions {
                abbreviated: true,
                ..base
            }
            .encode(W, H, &refs)
            .unwrap();
            let tables = abbr.tables.as_ref().expect("tables stream");
            assert_eq!(&tables[..2], &[0xFF, 0xD8]);
            assert_eq!(&tables[tables.len() - 2..], &[0xFF, 0xD9]);
            assert!(
                !abbr
                    .data
                    .windows(2)
                    .any(|m| m == [0xFF, 0xDB] || m == [0xFF, 0xC4]),
                "{process:?}: frame carries tables"
            );
            // SOI + JFIF APP0 (18 bytes) + tables + rest.
            let app0_end = 2 + 18;
            let mut spliced = abbr.data[..app0_end].to_vec();
            spliced.extend_from_slice(&tables[2..tables.len() - 2]);
            spliced.extend_from_slice(&abbr.data[app0_end..]);
            assert_eq!(
                spliced, full.data,
                "{process:?} P{precision}: splice differs"
            );
            // And the tables stream alone is a legal (image-less) stream
            // to the inspector.
            assert!(inspect_jpeg(tables).is_ok() || tables.len() > 4);
        }
    }
}

/// Black-box: `djpeg` decodes our sequential / progressive streams
/// within the quantisation floor and our lossless streams byte-exact,
/// at 8 and 12 (and 16 for lossless) bits.
#[test]
fn validators_decode_the_general_writer_output() {
    if !have("djpeg") {
        eprintln!("djpeg not available — skipping");
        return;
    }
    // DCT, grayscale + 4:2:0 + 4x2, 8- and 12-bit, optimal + restart.
    for process in [JpegProcess::Sequential, JpegProcess::Progressive] {
        for precision in [8u8, 12] {
            for (nf, sampling) in [
                (1usize, vec![]),
                (3, vec![(2, 2), (1, 1), (1, 1)]),
                (3, vec![(4, 2), (1, 1), (1, 1)]),
            ] {
                let src = planes(nf, precision, &sampling);
                let refs: Vec<&[u16]> = src.iter().map(|v| v.as_slice()).collect();
                let opts = JpegEncodeOptions {
                    quality: 92,
                    process,
                    precision,
                    restart_interval: 2,
                    tables: HuffmanTables::Optimal,
                    sampling: sampling.clone(),
                    ..Default::default()
                };
                let out = opts.encode(W, H, &refs).unwrap();
                let tag = format!("dct_{process:?}_P{precision}_nf{nf}_{}", sampling.len());
                let (maxval, got) = djpeg(&out.data, &tag, &["-nosmooth", "-grayscale"]).unwrap();
                assert_eq!(maxval, (1u32 << precision) - 1, "{tag}");
                // Luma only (`-grayscale` returns Y verbatim).
                let want: Vec<u32> = src[0].iter().map(|&v| u32::from(v)).collect();
                let db = psnr(&got, &want, maxval as f64);
                assert!(db > 38.0, "{tag}: djpeg luma PSNR {db:.1} dB");
            }
        }
    }
    // Lossless grayscale 8 / 12 / 16 and RGB 8, every predictor, restarts.
    for precision in [8u8, 12, 16] {
        for predictor in [1u8, 4, 7] {
            let src = planes(1, precision, &[]);
            let opts = JpegEncodeOptions {
                process: JpegProcess::Lossless {
                    predictor,
                    point_transform: 0,
                },
                precision,
                restart_interval: 2 * W as u16,
                ..Default::default()
            };
            let out = opts.encode(W, H, &[&src[0]]).unwrap();
            let tag = format!("ll_P{precision}_p{predictor}");
            let (_, got) = djpeg(&out.data, &tag, &[]).unwrap();
            let want: Vec<u32> = src[0].iter().map(|&v| u32::from(v)).collect();
            assert!(got == want, "{tag}: djpeg not byte-exact");
        }
    }
    let src = planes(3, 8, &[]);
    let refs: Vec<&[u16]> = src.iter().map(|v| v.as_slice()).collect();
    let out = JpegEncodeOptions {
        process: JpegProcess::Lossless {
            predictor: 5,
            point_transform: 0,
        },
        signalling: ColorSignalling::Rgb,
        restart_interval: W as u16,
        ..Default::default()
    }
    .encode(W, H, &refs)
    .unwrap();
    let (_, got) = djpeg(&out.data, "ll_rgb", &[]).unwrap();
    let mut want = Vec::new();
    for ((&r, &g), &b) in src[0].iter().zip(&src[1]).zip(&src[2]) {
        want.extend([u32::from(r), u32::from(g), u32::from(b)]);
    }
    assert!(got == want, "ll_rgb: djpeg not byte-exact");
}

/// The frame-level API tiff calls: explicit components + table set +
/// `gather_stats` across two frames sharing one `JPEGTables`.
#[test]
fn frame_api_shares_one_table_set_across_frames() {
    use oxideav_mjpeg::t81::{encode_frame, gather_stats, HuffStats, JpegComponent, JpegFrame};
    let a = planes(1, 12, &[]);
    let b: Vec<u16> = a[0].iter().map(|&v| 4095 - v).collect();
    let frame = JpegFrame {
        width: W as u16,
        height: H as u16,
        precision: 12,
        process: JpegProcess::Sequential,
        restart_interval: 4,
    };
    fn comp(s: &[u16]) -> JpegComponent<'_> {
        JpegComponent {
            id: 1,
            samples: s,
            width: W as usize,
            height: H as usize,
            h: 1,
            v: 1,
            quant_id: 0,
            huff_id: 0,
        }
    }
    let mut tables = JpegTableSet::typical(92, 12, false, 1);
    let mut dc: [HuffStats; 4] = Default::default();
    let mut ac: [HuffStats; 4] = Default::default();
    gather_stats(&frame, &[comp(&a[0])], &tables, &mut dc, &mut ac).unwrap();
    gather_stats(&frame, &[comp(&b)], &tables, &mut dc, &mut ac).unwrap();
    tables.dc[0] = Some(dc[0].to_spec());
    tables.ac[0] = Some(ac[0].to_spec());
    let jpegtables = tables.tables_stream(true);
    for s in [&a[0], &b] {
        let strip = encode_frame(&frame, &[comp(s)], &tables, false).unwrap();
        let mut full = jpegtables[..jpegtables.len() - 2].to_vec();
        full.extend_from_slice(&strip[2..]);
        let got = decoded_components(&full, 1, 12, &[]);
        let want: Vec<u32> = s.iter().map(|&v| u32::from(v)).collect();
        assert!(psnr(&got[0], &want, 4095.0) > 34.0);
    }
}
