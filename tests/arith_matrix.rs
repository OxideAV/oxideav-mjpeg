//! Arithmetic-coded DCT streams (T.81 Annex D, §F.1.4, §G.1.3) against
//! black-box tools (#21).
//!
//! * **Decode matrix.** A black-box encoder writes sequential (`SOF9`)
//!   and progressive (`SOF10`, with successive approximation) arithmetic
//!   streams over sizes × sampling layouts × restart intervals ×
//!   qualities. For each one this crate's coefficients must equal those
//!   of the Huffman twin a black-box lossless transcoder makes from it,
//!   and the reference-compatible RGB decode must equal the black-box
//!   decoder's pixels byte for byte.
//! * **Encode matrix.** This crate's arithmetic writer
//!   ([`EntropyCoding::Arithmetic`]) over processes × layouts × restart
//!   intervals × conditioning (`DAC` with non-default `L` / `U` / `Kx`):
//!   our decoder must return the Huffman twin's planes, and the
//!   black-box decoder must return the same pixels for both streams.
//!
//! Every check skips (with a note on stderr) when the tools are not
//! installed.

use std::path::PathBuf;
use std::process::Command;

use oxideav_mjpeg::{
    ArithConditioning, ChromaSubsampling, DecodeOptions, EncodeOptions, EntropyCoding, JpegProcess,
    SofKind, decode, decode_rgb8_with, encode_rgb8, read_coefficients,
};

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("-version")
        .output()
        .map(|o| o.status.success() || !o.stderr.is_empty())
        .unwrap_or(false)
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "oxideav-mjpeg-arith-matrix-{tag}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A picture with smooth areas, edges and noise (all coefficient
/// magnitudes and both chroma signs occur).
fn picture(w: usize, h: usize) -> Vec<u8> {
    let mut seed = 0x1234_5678u32;
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let n = (seed >> 24) as i32 - 128;
            let edge = if (x / 9 + y / 7) % 2 == 0 { 60 } else { -40 };
            let base = (x * 255 / w.max(1)) as i32;
            let r = base + edge + n / 3;
            let g = (y * 255 / h.max(1)) as i32 + n / 5 - edge / 2;
            let b = 255 - base + n / 2;
            for v in [r, g, b] {
                out.push(v.clamp(0, 255) as u8);
            }
        }
    }
    out
}

fn ppm(w: usize, h: usize, rgb: &[u8]) -> Vec<u8> {
    let mut v = format!("P6\n{w} {h}\n255\n").into_bytes();
    v.extend_from_slice(rgb);
    v
}

/// Body of a binary PNM (after the three header fields).
fn pnm_body(pnm: &[u8]) -> &[u8] {
    let (mut i, mut fields) = (2, 0);
    while fields < 3 {
        while pnm[i].is_ascii_whitespace() {
            i += 1;
        }
        while !pnm[i].is_ascii_whitespace() {
            i += 1;
        }
        fields += 1;
    }
    &pnm[i + 1..]
}

fn run(tool: &str, args: &[&str], input: &PathBuf) -> Vec<u8> {
    let o = Command::new(tool)
        .args(args)
        .arg(input)
        .output()
        .unwrap_or_else(|e| panic!("spawn {tool}: {e}"));
    assert!(
        o.status.success(),
        "{tool} {args:?} failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    o.stdout
}

#[test]
fn black_box_arithmetic_streams_decode_to_the_validator_output() {
    if !(have("cjpeg") && have("djpeg") && have("jpegtran")) {
        eprintln!("black-box JPEG tools not available — skipping the arithmetic decode matrix");
        return;
    }
    let dir = scratch("decode");
    let (src, jpg) = (dir.join("d.ppm"), dir.join("d.jpg"));
    let compat = DecodeOptions::libjpeg_compatible();
    let mut checked = 0;
    for (w, h) in [(1usize, 1usize), (7, 5), (17, 13), (64, 48), (131, 97)] {
        let rgb = picture(w, h);
        std::fs::write(&src, ppm(w, h, &rgb)).unwrap();
        for sample in ["1x1", "2x1", "2x2", "1x2", "4x1", "gray"] {
            for progressive in [false, true] {
                for restart in ["0", "1", "3B"] {
                    for quality in ["8", "75", "100"] {
                        let mut args = vec!["-arithmetic", "-quality", quality];
                        if sample == "gray" {
                            args.push("-grayscale");
                        } else {
                            args.extend(["-sample", sample]);
                        }
                        if progressive {
                            args.push("-progressive");
                        }
                        if restart != "0" {
                            args.extend(["-restart", restart]);
                        }
                        let tag = format!("{w}x{h} {args:?}");
                        let arith = run("cjpeg", &args, &src);
                        std::fs::write(&jpg, &arith).unwrap();
                        let ours =
                            read_coefficients(&arith).unwrap_or_else(|e| panic!("{tag}: {e}"));
                        let want_kind = if progressive {
                            SofKind::ProgressiveArith
                        } else {
                            SofKind::ExtendedSequentialArith
                        };
                        assert_eq!(ours.kind, Some(want_kind), "{tag}");
                        // Coefficients: the lossless Huffman transcode.
                        let huff = run("jpegtran", &["-copy", "none"], &jpg);
                        let twin = read_coefficients(&huff).unwrap();
                        assert!(
                            ours.components == twin.components,
                            "{tag}: coefficients differ from the Huffman twin"
                        );
                        assert_eq!(ours.quant_tables, twin.quant_tables, "{tag}");
                        // Pixels: the black-box decoder.
                        let pnm = run("djpeg", &["-pnm"], &jpg);
                        let got = decode_rgb8_with(&arith, &compat).unwrap();
                        let body = pnm_body(&pnm);
                        if sample == "gray" {
                            let grey: Vec<u8> = got.data.chunks(3).map(|p| p[0]).collect();
                            assert!(grey == body, "{tag}: grey pixels differ");
                        } else {
                            assert!(got.data == body, "{tag}: pixels differ");
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(checked, 5 * 6 * 2 * 3 * 3);
}

#[test]
fn our_arithmetic_streams_decode_in_the_black_box_decoder() {
    let tools = have("djpeg");
    if !tools {
        eprintln!(
            "black-box decoder not available — checking the arithmetic writer against our decoder only"
        );
    }
    let dir = scratch("encode");
    let jpg = dir.join("e.jpg");
    let conditionings = [
        ArithConditioning::default(),
        ArithConditioning::new().with_dc_bounds(0, 0).with_ac_kx(1),
        ArithConditioning::new().with_dc_bounds(1, 3).with_ac_kx(9),
        ArithConditioning::new()
            .with_dc_bounds(4, 15)
            .with_ac_kx(63),
    ];
    let mut checked = 0;
    for (w, h) in [(1u32, 1u32), (19, 11), (67, 45)] {
        let rgb = picture(w as usize, h as usize);
        for chroma in [
            ChromaSubsampling::Yuv444,
            ChromaSubsampling::Yuv422,
            ChromaSubsampling::Yuv420,
            ChromaSubsampling::Yuv411,
            ChromaSubsampling::GrayscaleOnly,
        ] {
            for process in [JpegProcess::Sequential, JpegProcess::Progressive] {
                for ri in [0u16, 1, 5] {
                    let base = EncodeOptions::new()
                        .with_quality(90)
                        .with_chroma(chroma)
                        .with_process(process)
                        .with_restart_interval(ri);
                    let huff = encode_rgb8(w, h, &rgb, &base).unwrap();
                    let want = decode(&huff).unwrap();
                    let want_bb = tools.then(|| {
                        std::fs::write(&jpg, &huff).unwrap();
                        run("djpeg", &["-pnm"], &jpg)
                    });
                    for cond in conditionings {
                        let tag = format!("{w}x{h} {chroma:?} {process:?} ri{ri} {cond:?}");
                        let opts = base
                            .clone()
                            .with_entropy(EntropyCoding::Arithmetic)
                            .with_conditioning(cond);
                        let out = encode_rgb8(w, h, &rgb, &opts).unwrap();
                        let has_dac = out.windows(2).any(|m| m == [0xFF, 0xCC]);
                        // (A two-byte window can also sit inside entropy
                        // data only as FF 00 / RSTm, never FF CC.)
                        assert_eq!(has_dac, cond != ArithConditioning::default(), "{tag}");
                        assert!(!out.windows(2).any(|m| m == [0xFF, 0xC4]), "{tag}: DHT");
                        let got = decode(&out).unwrap_or_else(|e| panic!("{tag}: {e}"));
                        assert!(got.planes == want.planes, "{tag}: our decode differs");
                        if let Some(want_bb) = &want_bb {
                            std::fs::write(&jpg, &out).unwrap();
                            let bb = run("djpeg", &["-pnm"], &jpg);
                            assert!(&bb == want_bb, "{tag}: black-box decode differs");
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(checked, 3 * 5 * 2 * 3 * 4);
    // Out-of-range conditioning is refused.
    for bad in [
        ArithConditioning::new().with_dc_bounds(3, 2),
        ArithConditioning::new().with_dc_bounds(0, 16),
        ArithConditioning::new().with_ac_kx(0),
        ArithConditioning::new().with_ac_kx(64),
    ] {
        let o = EncodeOptions::new()
            .with_arithmetic(true)
            .with_conditioning(bad);
        assert!(encode_rgb8(8, 8, &picture(8, 8), &o).is_err(), "{bad:?}");
    }
    // The lossless process codes arithmetically too (`SOF11`): the same
    // samples as its Huffman form.
    let rgb = picture(8, 8);
    let o = EncodeOptions::new().with_lossless(1);
    let huffman = encode_rgb8(8, 8, &rgb, &o).expect("lossless");
    let arith = encode_rgb8(8, 8, &rgb, &o.with_arithmetic(true)).expect("lossless arithmetic");
    assert!(arith.windows(2).any(|m| m == [0xFF, 0xCB]), "SOF11");
    let planes = |j: &[u8]| oxideav_mjpeg::decode(j).expect("decode").planes;
    assert_eq!(planes(&arith), planes(&huffman));
}
