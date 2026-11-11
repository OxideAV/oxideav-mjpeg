//! Golden pins for every decode mode, independent of the encoder's
//! transform: the streams are entropy-coded from synthetic quantised
//! coefficients (`encode_coefficients`), so what a decode returns
//! depends only on those coefficients, the quantisation tables and the
//! decoder — dequantisation, IDCT (float and integer), sample store,
//! chroma reconstruction (luma-guided, linear, replicated, triangular),
//! YCbCr conversion (exact and fixed-point), deblocking and threading.
//!
//! Performance work on any of those stages must leave these hashes
//! alone. Runs with and without the `registry` feature.
//!
//! Re-record after an *intentional* output change with
//!
//! ```text
//! GOLDEN_RECORD=1 cargo test --test golden_modes -- --nocapture
//! ```

use oxideav_mjpeg::coefficients::{CoefficientComponent, JpegCoefficients};
use oxideav_mjpeg::{
    ChromaUpsampling, DecodeOptions, EncodeOptions, JpegImage, decode_coefficients,
    decode_rgb8_with, decode_rgba8_with, decode_with, encode_coefficients, read_coefficients,
};

const W: u16 = 99;
const H: u16 = 53;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn hash_image(img: &JpegImage) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in &img.planes {
        fnv(&mut h, &(p.stride as u64).to_le_bytes());
        fnv(&mut h, &p.data);
    }
    h
}

fn hash_bytes(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    fnv(&mut h, b);
    h
}

/// A frame of pseudo-random coefficient blocks: a quarter with only a DC
/// level, a quarter confined to the lowest diagonals, a quarter to the
/// first six, the rest anywhere, with levels that shrink with frequency.
fn coefficients(sampling: &[(u8, u8)], seed: u32) -> JpegCoefficients {
    let mut state = seed;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let quant = |base: u16, slope: u16| -> [u16; 64] {
        std::array::from_fn(|i| base + slope * ((i % 8) + (i / 8)) as u16)
    };
    let mut c = JpegCoefficients::new(W, H, 8)
        .with_quant_table(0, quant(4, 3))
        .with_quant_table(1, quant(6, 4));
    let hmax = sampling.iter().map(|s| s.0).max().unwrap() as usize;
    let vmax = sampling.iter().map(|s| s.1).max().unwrap() as usize;
    let mcus_x = (W as usize).div_ceil(8 * hmax);
    let mcus_y = (H as usize).div_ceil(8 * vmax);
    for (ci, &(h, v)) in sampling.iter().enumerate() {
        let n = mcus_x * h as usize * mcus_y * v as usize;
        let mut blocks = vec![[0i32; 64]; n];
        let mut dc = 0i32;
        for b in blocks.iter_mut() {
            dc = (dc + (rnd() % 31) as i32 - 15).clamp(-120, 120);
            b[0] = dc;
            let reach = [0usize, 2, 5, 14][(rnd() % 4) as usize];
            for (i, level) in b.iter_mut().enumerate().skip(1) {
                let d = i % 8 + i / 8;
                if d <= reach && rnd() % 3 == 0 {
                    let amp = (24 / (d as i32 + 1)).max(1);
                    *level = (rnd() % (2 * amp as u32 + 1)) as i32 - amp;
                }
            }
        }
        let comp = CoefficientComponent::new(ci as u8 + 1, h, v, u8::from(ci > 0));
        c = c.with_component(comp.with_blocks(blocks));
    }
    // A non-interleaved scan (the AC scans of a progressive frame) codes
    // only the blocks that cover real samples (T.81 §A.2.2), so the
    // blocks that merely complete an MCU carry no AC coefficients here.
    for ci in 0..c.components.len() {
        let (grid_w, _) = c.block_grid(ci);
        let (real_w, real_h) = c.block_extent(ci);
        for (i, b) in c.components[ci].blocks.iter_mut().enumerate() {
            if i % grid_w >= real_w || i / grid_w >= real_h {
                b[1..].fill(0);
            }
        }
    }
    c
}

/// `(layout name, sampling factors, [hash; 9])`: planar default, planar
/// reference-compatible (integer IDCT), planar deblocked, then RGB with
/// the luma-guided / linear / replicated / triangular reconstructions,
/// RGB reference-compatible, RGBA default.
type Pin = (&'static str, &'static [(u8, u8)], [u64; 9]);

const PINS: &[Pin] = &[
    (
        "grey",
        &[(1, 1)],
        [
            0xad2beb9331545494,
            0xbb6e127981a5a89e,
            0x309af16c7cfc8ad5,
            0xf4051e2cf599157d,
            0xf4051e2cf599157d,
            0xf4051e2cf599157d,
            0xf4051e2cf599157d,
            0xe277003e01a9d25b,
            0x9272e30f272ddb86,
        ],
    ),
    (
        "444",
        &[(1, 1), (1, 1), (1, 1)],
        [
            0x743b31753747d2ba,
            0xcce7a0b964c84394,
            0x0a66774b076127c5,
            0x7014fb25a5c75d50,
            0x7014fb25a5c75d50,
            0x7014fb25a5c75d50,
            0x7014fb25a5c75d50,
            0x5d242e84d0537b96,
            0xe485ee9d1f69d0b3,
        ],
    ),
    (
        "422",
        &[(2, 1), (1, 1), (1, 1)],
        [
            0x3edf3747c918d072,
            0x0a43b7b43d0af5d1,
            0xb06bb615af818ebe,
            0x2ca8c7b96c460d0a,
            0x2642e94adbf073bc,
            0xd7c25c619c25e32c,
            0x2cca56fe6e60bcb6,
            0x59d911a15dddea07,
            0x9e41374f705ae6cd,
        ],
    ),
    (
        "420",
        &[(2, 2), (1, 1), (1, 1)],
        [
            0x072ed73a884516fa,
            0x4db72ecf99be0938,
            0xcdde51e94b391ade,
            0x9d7864e1c13795bc,
            0xb1ec0635ca822fb4,
            0x4b2dab51a6b92891,
            0x4a8b933805dd24e8,
            0x8439ba1975e7aaa2,
            0x752c233196a66bd5,
        ],
    ),
    (
        "411",
        &[(4, 1), (1, 1), (1, 1)],
        [
            0xa4808df101dc6c2b,
            0x16f50e7b560f111e,
            0xb1cd912011f51506,
            0xc8e41b9cf72eb772,
            0xf7c2f1ca8a500904,
            0xdb7deb68546fbc79,
            0xdb7deb68546fbc79,
            0xf0922c5c01816da5,
            0x2484372ac2ac3801,
        ],
    ),
    (
        "440",
        &[(1, 2), (1, 1), (1, 1)],
        [
            0x825da271ebd948e4,
            0x5d2cda8196304520,
            0x75b171c74a176063,
            0x5e6afc56af6990e7,
            0x27e85db4cfab4fd7,
            0xa70274bb771dae58,
            0x995f1e8b9927797b,
            0xb5340012d0bb3c41,
            0xcd8059b65bb0fb34,
        ],
    ),
    (
        "mixed",
        &[(2, 2), (2, 1), (1, 1)],
        [
            0x38e5d275b8e0d1dc,
            0x66cae204e2f87f62,
            0x08ce37d854752fe8,
            0xf61fa22c1050e567,
            0xa151ede3ca66c937,
            0xa06405c839ae2a81,
            0xaf1c0c0626839466,
            0xaafe08c90b604056,
            0xe565755154160470,
        ],
    ),
];

fn modes(stream: &[u8]) -> [u64; 9] {
    let d = DecodeOptions::new;
    let planar = |o: &DecodeOptions| hash_image(&decode_with(stream, o).expect("decode"));
    let rgb = |o: &DecodeOptions| {
        hash_bytes(decode_rgb8_with(stream, o).expect("decode_rgb8").as_bytes())
    };
    [
        planar(&d()),
        planar(&DecodeOptions::libjpeg_compatible()),
        planar(&d().with_deblock(true)),
        rgb(&d()),
        rgb(&d().with_upsampling(ChromaUpsampling::Linear)),
        rgb(&d().with_upsampling(ChromaUpsampling::Replicate)),
        rgb(&d().with_upsampling(ChromaUpsampling::Triangular)),
        rgb(&DecodeOptions::libjpeg_compatible()),
        hash_bytes(
            decode_rgba8_with(stream, &d())
                .expect("decode_rgba8")
                .as_bytes(),
        ),
    ]
}

#[test]
fn every_decode_mode_is_golden() {
    let record = std::env::var_os("GOLDEN_RECORD").is_some();
    for (li, &(name, sampling, want)) in PINS.iter().enumerate() {
        let coefs = coefficients(
            sampling,
            0x9E37_79B9 ^ (li as u32).wrapping_mul(0x85EB_CA6B),
        );
        let base = EncodeOptions::new();
        let streams = [
            ("sequential", base.clone().with_optimal_tables(false)),
            ("optimal", base.clone().with_optimal_tables(true)),
            ("restart", base.clone().with_restart_interval(3)),
            ("progressive", base.clone().with_progressive(true)),
            ("arithmetic", base.clone().with_arithmetic(true)),
        ]
        .map(|(tag, o)| (tag, encode_coefficients(&coefs, &o).expect(tag)));
        let got = modes(&streams[0].1);
        if record {
            let row: Vec<String> = got.iter().map(|h| format!("0x{h:016x}")).collect();
            println!("\n    (\"{name}\", &{sampling:?}, [{}]),", row.join(", "));
        } else {
            assert_eq!(got, want, "{name}: decode output changed");
        }
        for (tag, stream) in &streams {
            // The coefficients survive every entropy coding, so every
            // stream decodes to the pinned output — whichever path the
            // decoder takes for it (block by block, or from the
            // coefficient store).
            let back = read_coefficients(stream).expect("read_coefficients");
            for (a, b) in back.components.iter().zip(&coefs.components) {
                assert!(a.blocks == b.blocks, "{name} {tag}: coefficients");
            }
            assert_eq!(modes(stream), got, "{name} {tag}");
            // Rendering the record directly is the same picture.
            let direct = decode_coefficients(&coefs, &DecodeOptions::new()).expect("render");
            let decoded = decode_with(stream, &DecodeOptions::new()).expect("decode");
            assert!(direct.planes == decoded.planes, "{name} {tag}: render");
            // Workers change nothing.
            for threads in [2, 5] {
                let o = DecodeOptions::new().with_threads(threads);
                let serial = decode_rgb8_with(stream, &DecodeOptions::new()).expect("rgb");
                let split = decode_rgb8_with(stream, &o).expect("rgb");
                assert!(
                    serial.as_bytes() == split.as_bytes(),
                    "{name} {tag}: {threads} workers"
                );
                let p = decode_with(stream, &o).expect("decode");
                assert!(
                    p.planes == decoded.planes,
                    "{name} {tag}: {threads} workers (planar)"
                );
            }
        }
    }
}
