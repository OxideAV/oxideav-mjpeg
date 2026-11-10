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
    let arith = EncodeOptions::default().with_arithmetic(true);
    let mut arith_prog = arith.clone();
    arith_prog.process = JpegProcess::Progressive;
    let mut arith_rst = arith.clone();
    arith_rst.restart_interval = 3;
    vec![
        ("seq", seq),
        ("optimal", opt),
        ("prog", prog),
        ("rst", rst),
        ("arith", arith),
        ("arith_prog", arith_prog),
        ("arith_rst", arith_rst),
    ]
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
                if tag.ends_with("prog") {
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
                if tag.starts_with("arith") {
                    let kind = if tag.ends_with("prog") {
                        SofKind::ProgressiveArith
                    } else {
                        SofKind::ExtendedSequentialArith
                    };
                    assert_eq!(again.kind, Some(kind), "{name} {tag}");
                    assert!(
                        !out.windows(2).any(|m| m == [0xFF, 0xC4]),
                        "{name} {tag}: no DHT"
                    );
                }
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

/// The entropy-coded data behind the first `SOS` header, up to `EOI`,
/// and the restart interval in force (0 without `DRI`).
fn first_scan(jpeg: &[u8]) -> (Vec<u8>, u16, Vec<u8>) {
    let (mut i, mut ri) = (2, 0u16);
    loop {
        let m = jpeg[i + 1];
        let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        if m == 0xDD {
            ri = u16::from_be_bytes([jpeg[i + 4], jpeg[i + 5]]);
        }
        if m == 0xDA {
            let header = jpeg[i + 4..i + 2 + len].to_vec();
            return (jpeg[i + 2 + len..jpeg.len() - 2].to_vec(), ri, header);
        }
        i += 2 + len;
    }
}

/// Byte-level round trip of the arithmetic coder (#21): the sequential
/// `fixtures/arith/arith_*.jpg` streams are a black-box lossless
/// transcoder's arithmetic output. Reading their coefficients and
/// writing them back with arithmetic coding and the same restart
/// interval reproduces the entropy-coded segment byte for byte (when
/// the scan header names the same table destinations), so the
/// statistics model, the conditioning and the termination all agree
/// with the validator.
#[test]
fn arithmetic_recode_reproduces_the_validator_scan_bytes() {
    let mut exact = 0;
    for (name, bytes) in jpegs(ARITH) {
        let coefs = read_coefficients(&bytes).unwrap();
        if coefs.kind != Some(SofKind::ExtendedSequentialArith) {
            continue;
        }
        let (want, ri, header) = first_scan(&bytes);
        let opts = EncodeOptions::default()
            .with_arithmetic(true)
            .with_restart_interval(ri);
        let out = encode_coefficients(&coefs, &opts).unwrap();
        let (got, _, our_header) = first_scan(&out);
        assert_eq!(
            read_coefficients(&out).unwrap().components,
            coefs.components
        );
        if header != our_header {
            // Other table destinations: other statistics areas shared.
            continue;
        }
        assert!(
            got == want,
            "{name}: scan bytes differ from the validator's"
        );
        exact += 1;
    }
    assert!(exact >= 5, "{exact} byte-exact streams");
}

/// The library transforms (#21) reproduce the helpers above — which
/// match the black-box lossless transcoder — and obey the group laws on
/// the coefficients themselves.
#[test]
fn library_transforms_are_exact() {
    let src = read_coefficients(&std::fs::read(format!("{COEF}/flip_src.jpg")).unwrap()).unwrap();
    assert_eq!(src.mcu_size(), (16, 16));
    assert_eq!(src.flipped_horizontal().unwrap(), flip_horizontal(&src));
    let t = src.transposed();
    let want = transpose(&src);
    assert_eq!(t.components, want.components);
    assert_eq!(t.quant_tables, want.quant_tables);
    assert_eq!((t.width, t.height), (src.height, src.width));
    assert_eq!(t.metadata, src.metadata);
    // Involutions and the rotation group.
    assert_eq!(t.transposed(), src);
    assert_eq!(
        src.flipped_horizontal()
            .unwrap()
            .flipped_horizontal()
            .unwrap(),
        src
    );
    assert_eq!(
        src.flipped_vertical().unwrap().flipped_vertical().unwrap(),
        src
    );
    assert_eq!(src.rotated(0).unwrap(), src);
    assert_eq!(src.rotated(4).unwrap(), src);
    let r1 = src.rotated(1).unwrap();
    assert_eq!((r1.width, r1.height), (src.height, src.width));
    assert_eq!(r1.rotated(3).unwrap(), src);
    assert_eq!(r1.rotated(1).unwrap(), src.rotated(2).unwrap());
    assert_eq!(src.rotated(2).unwrap().rotated(2).unwrap(), src);
    assert_eq!(
        src.rotated(2).unwrap(),
        src.flipped_vertical()
            .unwrap()
            .flipped_horizontal()
            .unwrap()
    );
    // A vertical mirror is the transpose of a horizontal one.
    assert_eq!(
        src.flipped_vertical().unwrap(),
        src.transposed().flipped_horizontal().unwrap().transposed()
    );
    // Every result codes and reads back.
    for c in [
        src.rotated(1).unwrap(),
        src.rotated(2).unwrap(),
        src.rotated(3).unwrap(),
        src.flipped_vertical().unwrap(),
    ] {
        let out = encode_coefficients(&c, &EncodeOptions::default()).unwrap();
        assert_eq!(read_coefficients(&out).unwrap().components, c.components);
    }
    // The pixels follow (4:2:0 planar output: luma is plane 0). The
    // inverse DCT is evaluated in another order on the mirrored block,
    // so a sample may round the other way.
    let base = decode_coefficients(&src, &DecodeOptions::default()).unwrap();
    let flip = decode_coefficients(
        &src.flipped_horizontal().unwrap(),
        &DecodeOptions::default(),
    )
    .unwrap();
    let (w, h) = (src.width as usize, src.height as usize);
    for y in 0..h {
        for x in 0..w {
            let a = base.planes[0].data[y * base.planes[0].stride + x] as i32;
            let b = flip.planes[0].data[y * flip.planes[0].stride + (w - 1 - x)] as i32;
            assert!((a - b).abs() <= 1, "({x}, {y}): {a} vs {b}");
        }
    }
}

/// Crops copy whole blocks: an MCU-aligned origin, any size inside the
/// frame; mirrors refuse a frame with a partial MCU on that axis until
/// it is trimmed.
#[test]
fn crops_are_exact_and_mirrors_refuse_partial_mcus() {
    let src = read_coefficients(&std::fs::read(format!("{COEF}/flip_src.jpg")).unwrap()).unwrap();
    // 48 × 32 at 4:2:0: three by two MCUs.
    let c = src.cropped(16, 16, 32, 16).unwrap();
    assert_eq!((c.width, c.height), (32, 16));
    assert_eq!(c.block_grid(0), (4, 2));
    assert_eq!(c.block_grid(1), (2, 1));
    let (bw, _) = src.block_grid(0);
    assert_eq!(
        c.components[0].blocks[0],
        src.components[0].blocks[2 * bw + 2]
    );
    assert_eq!(c.components[1].blocks[1], src.components[1].blocks[3 + 2]);
    assert_eq!(c.quant_tables, src.quant_tables);
    assert_eq!(src.cropped(0, 0, 48, 32).unwrap(), src);
    // The crop decodes to the same samples as that region of the source.
    let whole = decode_coefficients(&src, &DecodeOptions::default()).unwrap();
    let part = decode_coefficients(&c, &DecodeOptions::default()).unwrap();
    for y in 0..16 {
        assert_eq!(
            part.planes[0].data[y * part.planes[0].stride..][..32],
            whole.planes[0].data[(16 + y) * whole.planes[0].stride + 16..][..32]
        );
    }
    // A ragged size keeps the blocks of its last MCU as padding.
    let ragged = src.cropped(16, 0, 21, 19).unwrap();
    assert_eq!(ragged.block_grid(0), (4, 4));
    ragged.validate().unwrap();
    let out = encode_coefficients(&ragged, &EncodeOptions::default()).unwrap();
    let img = decode(&out).unwrap();
    assert_eq!((img.width, img.height), (21, 19));
    for y in 0..19 {
        assert_eq!(
            img.planes[0].data[y * img.planes[0].stride..][..21],
            whole.planes[0].data[y * whole.planes[0].stride + 16..][..21]
        );
    }
    // Mirrors on the ragged axis are refused, the transpose is not.
    assert!(ragged.flipped_horizontal().is_err());
    assert!(ragged.flipped_vertical().is_err());
    assert!(ragged.rotated(1).is_err());
    ragged.transposed().validate().unwrap();
    let trimmed = ragged.cropped(0, 0, 16, 16).unwrap();
    trimmed.rotated(1).unwrap().validate().unwrap();
    // Bad regions.
    assert!(src.cropped(8, 0, 16, 16).is_err());
    assert!(src.cropped(0, 16, 48, 17).is_err());
    assert!(src.cropped(0, 0, 0, 16).is_err());
}

/// The same transforms against the black-box lossless transcoder, when
/// it is installed: vertical mirror, the three rotations and a crop.
#[test]
fn library_transforms_match_the_black_box_transcoder() {
    let dir = std::env::temp_dir().join(format!("oxideav-mjpeg-xform-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = format!("{COEF}/flip_src.jpg");
    let src = read_coefficients(&std::fs::read(&path).unwrap()).unwrap();
    let run = |args: &[&str]| -> Option<JpegCoefficients> {
        let o = std::process::Command::new("jpegtran")
            .args(["-copy", "none"])
            .args(args)
            .arg(&path)
            .output()
            .ok()?;
        o.status
            .success()
            .then(|| read_coefficients(&o.stdout).unwrap())
    };
    let cases: [(&[&str], JpegCoefficients); 5] = [
        (&["-flip", "vertical"], src.flipped_vertical().unwrap()),
        (&["-rotate", "90"], src.rotated(1).unwrap()),
        (&["-rotate", "180"], src.rotated(2).unwrap()),
        (&["-rotate", "270"], src.rotated(3).unwrap()),
        (
            &["-crop", "32x16+16+16"],
            src.cropped(16, 16, 32, 16).unwrap(),
        ),
    ];
    for (args, ours) in cases {
        let Some(want) = run(args) else {
            eprintln!("jpegtran not available — skipping the black-box transform check");
            break;
        };
        assert_eq!(
            (ours.width, ours.height),
            (want.width, want.height),
            "{args:?}"
        );
        assert!(ours.components == want.components, "{args:?}: coefficients");
        assert_eq!(ours.quant_tables, want.quant_tables, "{args:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
