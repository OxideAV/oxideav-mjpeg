//! Coefficient-level access (#21): `read_coefficients` /
//! `decode_coefficients` / `encode_coefficients`.
//!
//! * Every DCT stream of `fixtures/libjpeg/` (sequential, progressive,
//!   arithmetic, every sampling layout, restarts, 8 and 12 bits, grey,
//!   YCbCr, YCCK, CMYK) and the arithmetic / Huffman twins of
//!   `fixtures/arith/`: rendering the coefficients gives `decode`'s
//!   samples, re-coding them (sequential typical / optimal, progressive)
//!   gives a stream with the very same coefficients and samples, and the
//!   arithmetic twin of a Huffman stream reads to identical coefficients.
//! * `fixtures/coefficients/flip_src.jpg` is `cjpeg -quality 85 -sample
//!   2x2` of a 48×32 frame; `flip_h.jpg` and `transpose.jpg` are
//!   `jpegtran -flip horizontal -copy none` / `jpegtran -transpose -copy
//!   none` of it (black-box lossless transforms). The same transforms
//!   done here in the coefficient domain — horizontal: mirror the block
//!   columns and negate the odd horizontal frequencies; transpose: swap
//!   block rows / columns, swap `u` / `v`, transpose the quantisation
//!   tables — must reproduce the validator's coefficients exactly.
//! * When `djpeg` is installed, our re-coded streams decode in it to the
//!   bytes the source decodes to.

use oxideav_mjpeg::{
    CoefficientComponent, DecodeOptions, EncodeOptions, HuffmanTables, JpegCoefficients,
    JpegProcess, SofKind, decode, decode_coefficients, encode_coefficients, read_coefficients,
};

const LIB: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/libjpeg");
const ARITH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/arith");
const COEF: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/coefficients");

fn jpegs(dir: &str) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "jpg"))
        .map(|p| {
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect();
    v.sort();
    v
}

fn recode_options() -> Vec<(&'static str, EncodeOptions)> {
    let seq = EncodeOptions::default();
    let mut opt = EncodeOptions::default();
    opt.tables = HuffmanTables::Optimal;
    let mut prog = EncodeOptions::default();
    prog.process = JpegProcess::Progressive;
    let mut rst = EncodeOptions::default();
    rst.restart_interval = 3;
    vec![("seq", seq), ("optimal", opt), ("prog", prog), ("rst", rst)]
}

#[test]
fn coefficients_render_and_recode_losslessly() {
    let mut checked = 0;
    for dir in [LIB, ARITH] {
        for (name, bytes) in jpegs(dir) {
            let coefs = read_coefficients(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
            coefs.validate().unwrap();
            let want = decode(&bytes).unwrap();
            let rendered = decode_coefficients(&coefs, &DecodeOptions::default()).unwrap();
            assert_eq!(rendered.format, want.format, "{name}");
            assert_eq!(rendered.planes, want.planes, "{name}: render");
            assert_eq!(rendered.metadata, want.metadata, "{name}: metadata");
            for (tag, opts) in recode_options() {
                let out = encode_coefficients(&coefs, &opts)
                    .unwrap_or_else(|e| panic!("{name} {tag}: {e}"));
                let again = read_coefficients(&out).unwrap();
                if tag == "prog" {
                    // Progressive AC scans are non-interleaved and cover
                    // only the blocks over real samples (§A.2.2); the
                    // §A.2.4 padding blocks keep their DC alone.
                    for (ci, (a, b)) in again.components.iter().zip(&coefs.components).enumerate() {
                        let (bw, bh) = coefs.block_grid(ci);
                        let (ew, eh) = coefs.block_extent(ci);
                        for by in 0..bh {
                            for bx in 0..bw {
                                let (x, y) = (&a.blocks[by * bw + bx], &b.blocks[by * bw + bx]);
                                if bx < ew && by < eh {
                                    assert_eq!(x, y, "{name} prog: block ({bx}, {by}) of {ci}");
                                } else {
                                    assert_eq!(x[0], y[0], "{name} prog: padding DC");
                                }
                            }
                        }
                    }
                } else {
                    assert_eq!(
                        again.components, coefs.components,
                        "{name} {tag}: coefficients"
                    );
                }
                assert_eq!(
                    again.quant_tables, coefs.quant_tables,
                    "{name} {tag}: tables"
                );
                assert_eq!(again.adobe_transform, coefs.adobe_transform, "{name} {tag}");
                let got = decode(&out).unwrap();
                assert_eq!(got.format, want.format, "{name} {tag}");
                assert_eq!(got.planes, want.planes, "{name} {tag}: samples");
            }
            checked += 1;
        }
    }
    assert!(checked >= 100, "only {checked} streams");
}

#[test]
fn arithmetic_twins_read_to_the_same_coefficients() {
    let mut pairs = 0;
    for (name, bytes) in jpegs(ARITH) {
        let Some(rest) = name.strip_prefix("arith_") else {
            continue;
        };
        let twin = if rest.starts_with("cmyk") {
            "huff_cmyk.jpg".to_string()
        } else {
            format!("huff_{rest}")
        };
        let Ok(huff) = std::fs::read(format!("{ARITH}/{twin}")) else {
            continue;
        };
        let a = read_coefficients(&bytes).unwrap();
        let h = read_coefficients(&huff).unwrap();
        assert_eq!(a.components, h.components, "{name}");
        assert_eq!(a.quant_tables, h.quant_tables, "{name}");
        assert!(matches!(
            a.kind,
            Some(SofKind::ExtendedSequentialArith) | Some(SofKind::ProgressiveArith)
        ));
        pairs += 1;
    }
    assert!(pairs >= 8, "{pairs} pairs");
}

#[test]
fn record_shape_follows_the_frame_header() {
    let bytes = std::fs::read(format!("{COEF}/flip_src.jpg")).unwrap();
    let c = read_coefficients(&bytes).unwrap();
    assert_eq!((c.width, c.height, c.precision), (48, 32, 8));
    assert_eq!(c.kind, Some(SofKind::Baseline));
    assert!(c.jfif);
    assert_eq!(c.components.len(), 3);
    assert_eq!((c.components[0].h, c.components[0].v), (2, 2));
    assert_eq!(c.block_grid(0), (6, 4));
    assert_eq!(c.block_grid(1), (3, 2));
    assert_eq!(c.block_extent(1), (3, 2));
    // The DQT entries in natural order: `cjpeg -quality 85` scales the
    // Annex K tables, whose (0, 0) entries are 16 and 17.
    let q0 = c.quant_tables[0].unwrap();
    let q1 = c.quant_tables[1].unwrap();
    assert_eq!((q0[0], q1[0]), (5, 5));
}

/// Horizontal mirror in the coefficient domain (frame width a multiple
/// of the MCU width, so no edge blocks move).
fn flip_horizontal(c: &JpegCoefficients) -> JpegCoefficients {
    let mut out = c.clone();
    for (ci, comp) in out.components.iter_mut().enumerate() {
        let (bw, bh) = c.block_grid(ci);
        let src = &c.components[ci].blocks;
        for by in 0..bh {
            for bx in 0..bw {
                let mut b = src[by * bw + (bw - 1 - bx)];
                for v in 0..8 {
                    for u in (1..8).step_by(2) {
                        b[v * 8 + u] = -b[v * 8 + u];
                    }
                }
                comp.blocks[by * bw + bx] = b;
            }
        }
    }
    out
}

/// Transpose in the coefficient domain.
fn transpose(c: &JpegCoefficients) -> JpegCoefficients {
    let mut out = JpegCoefficients::new(c.height, c.width, c.precision);
    for (tq, t) in c.quant_tables.iter().enumerate() {
        if let Some(t) = t {
            let mut tt = [0u16; 64];
            for v in 0..8 {
                for u in 0..8 {
                    tt[u * 8 + v] = t[v * 8 + u];
                }
            }
            out = out.with_quant_table(tq, tt);
        }
    }
    for (ci, comp) in c.components.iter().enumerate() {
        let (bw, bh) = c.block_grid(ci);
        let mut blocks = vec![[0i32; 64]; bw * bh];
        for by in 0..bh {
            for bx in 0..bw {
                let b = &comp.blocks[by * bw + bx];
                let t = &mut blocks[bx * bh + by];
                for v in 0..8 {
                    for u in 0..8 {
                        t[u * 8 + v] = b[v * 8 + u];
                    }
                }
            }
        }
        out = out.with_component(
            CoefficientComponent::new(comp.id, comp.v, comp.h, comp.quant_id).with_blocks(blocks),
        );
    }
    out
}

#[test]
fn coefficient_domain_transforms_match_the_validator() {
    let src = read_coefficients(&std::fs::read(format!("{COEF}/flip_src.jpg")).unwrap()).unwrap();
    let flipped = read_coefficients(&std::fs::read(format!("{COEF}/flip_h.jpg")).unwrap()).unwrap();
    let ours = flip_horizontal(&src);
    assert_eq!(ours.components, flipped.components);
    assert_eq!(ours.quant_tables, flipped.quant_tables);

    let tr = read_coefficients(&std::fs::read(format!("{COEF}/transpose.jpg")).unwrap()).unwrap();
    let ours = transpose(&src);
    assert_eq!((ours.width, ours.height), (tr.width, tr.height));
    assert_eq!(ours.components, tr.components);
    assert_eq!(ours.quant_tables, tr.quant_tables);
    // And it codes: our transposed stream decodes to the validator's.
    let out = encode_coefficients(&ours, &EncodeOptions::default()).unwrap();
    assert_eq!(
        decode(&out).unwrap().planes,
        decode(&std::fs::read(format!("{COEF}/transpose.jpg")).unwrap())
            .unwrap()
            .planes
    );
}

#[test]
fn malformed_records_and_frames_without_coefficients_are_refused() {
    let src = read_coefficients(&std::fs::read(format!("{COEF}/flip_src.jpg")).unwrap()).unwrap();
    let mut short = src.clone();
    short.components[0].blocks.pop();
    assert!(short.validate().is_err());
    assert!(encode_coefficients(&short, &EncodeOptions::default()).is_err());
    let mut big = src.clone();
    big.components[1].blocks[0][5] = 1024; // P = 8: AC |Sq| < 2^10
    assert!(encode_coefficients(&big, &EncodeOptions::default()).is_err());
    let mut no_table = src.clone();
    no_table.quant_tables[1] = None;
    assert!(decode_coefficients(&no_table, &DecodeOptions::default()).is_err());
    let mut lossless = EncodeOptions::default();
    lossless.process = JpegProcess::Lossless {
        predictor: 1,
        point_transform: 0,
    };
    assert!(encode_coefficients(&src, &lossless).is_err());

    // A lossless stream has no DCT coefficients.
    let mut o = EncodeOptions::default();
    o.process = JpegProcess::Lossless {
        predictor: 1,
        point_transform: 0,
    };
    let gray = vec![100u16; 16 * 8];
    let ll = o.encode(16, 8, &[&gray]).unwrap().data;
    assert!(matches!(
        read_coefficients(&ll),
        Err(oxideav_mjpeg::Error::Unsupported(_))
    ));
}

#[test]
fn a_hand_built_record_codes_and_renders() {
    // One grey 16×8 frame: a flat block of DC 20 (× Q00 2 = 40 → level
    // 128 + 40 / 8 = 133) and a block with only DC −8 (→ 128 − 2 = 126).
    let mut q = [1u16; 64];
    q[0] = 2;
    let mut a = [0i32; 64];
    a[0] = 20;
    let mut b = [0i32; 64];
    b[0] = -8;
    let rec = JpegCoefficients::new(16, 8, 8)
        .with_quant_table(0, q)
        .with_component(CoefficientComponent::new(1, 1, 1, 0).with_blocks(vec![a, b]));
    let img = decode_coefficients(&rec, &DecodeOptions::default()).unwrap();
    let row = &img.planes[0].data[..16];
    assert!(row[..8].iter().all(|&s| s == 133), "{row:?}");
    assert!(row[8..].iter().all(|&s| s == 126), "{row:?}");
    let out = encode_coefficients(&rec, &EncodeOptions::default()).unwrap();
    assert_eq!(decode(&out).unwrap().planes, img.planes);
    assert!(oxideav_mjpeg::info(&out).unwrap().has_jfif);
}

/// Black-box: `djpeg` decodes our re-coded streams to the same bytes as
/// the sources.
#[test]
fn djpeg_decodes_recoded_streams_identically() {
    let dir = std::env::temp_dir().join(format!("oxideav-mjpeg-coef-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let djpeg = |bytes: &[u8], name: &str| -> Option<Vec<u8>> {
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        let o = std::process::Command::new("djpeg")
            .arg("-pnm")
            .arg(&p)
            .output()
            .ok()?;
        o.status.success().then_some(o.stdout)
    };
    let mut checked = 0;
    for (name, bytes) in jpegs(LIB)
        .into_iter()
        .filter(|(n, _)| !n.starts_with("p12"))
    {
        let Some(want) = djpeg(&bytes, "src.jpg") else {
            let _ = std::fs::remove_dir_all(&dir);
            return; // validator not installed
        };
        let coefs = read_coefficients(&bytes).unwrap();
        for (tag, opts) in recode_options() {
            let out = encode_coefficients(&coefs, &opts).unwrap();
            let got = djpeg(&out, "out.jpg").expect("djpeg reads our stream");
            assert!(got == want, "{name} {tag}: djpeg output differs");
        }
        checked += 1;
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(checked >= 80);
}

/// `fixtures/coefficients/gray_2x2.jpg` is `cjpeg -grayscale -sample 2x2`
/// of a 61×47 frame: one component declaring `H = V = 2`. Its grid
/// follows the MCU rule (`⌈61 / 16⌉ × 2` blocks wide), and the re-code is
/// written `1 × 1` with the same samples.
#[test]
fn single_component_with_sampling_factors_round_trips() {
    let bytes = std::fs::read(format!("{COEF}/gray_2x2.jpg")).unwrap();
    let c = read_coefficients(&bytes).unwrap();
    assert_eq!((c.components[0].h, c.components[0].v), (2, 2));
    assert_eq!(c.block_grid(0), (8, 6));
    assert_eq!(c.block_extent(0), (8, 6));
    let want = decode(&bytes).unwrap();
    let r = decode_coefficients(&c, &DecodeOptions::default()).unwrap();
    assert_eq!(r.planes, want.planes);
    for (tag, opts) in recode_options() {
        let out = encode_coefficients(&c, &opts).unwrap();
        let again = read_coefficients(&out).unwrap();
        assert_eq!(
            (again.components[0].h, again.components[0].v),
            (1, 1),
            "{tag}"
        );
        assert_eq!(decode(&out).unwrap().planes, want.planes, "{tag}");
    }
}
