//! Baseline JPEG bitstream primitives.

#[doc(hidden)]
pub mod arith;
#[doc(hidden)]
pub mod color;
#[doc(hidden)]
pub mod dct;
#[doc(hidden)]
pub mod huffman;
pub(crate) mod idct_int;
pub mod inspect;
#[doc(hidden)]
pub mod markers;
#[doc(hidden)]
pub mod parser;
#[doc(hidden)]
pub mod quant;
pub(crate) mod upsample;
#[doc(hidden)]
pub mod zigzag;
