//! Inference binary: load a trained checkpoint, embed SMILES batches
//! from one of several supported input formats, write embeddings to one
//! of several supported output formats.
//!
//! All file-format knowledge lives in the library
//! ([`molecular_autoencoder::source_for_path`] and
//! [`molecular_autoencoder::sink_for_path`]). This bin is just clap +
//! backend selection + the source-→encoder-→sink loop.

use std::error::Error as StdError;

use clap::Parser;

mod cli;
mod run;

/// Unified error type used across the embed bin.
pub type AppResult<T> = Result<T, Box<dyn StdError>>;

#[cfg(feature = "cuda")]
fn main() -> AppResult<()> {
    type Backend = burn::backend::Cuda<f32, i32>;
    let args = cli::Args::parse();
    let device = burn::backend::cuda::CudaDevice::new(args.cuda_device);
    run::run::<Backend>(args, device)
}

#[cfg(not(feature = "cuda"))]
fn main() -> AppResult<()> {
    type Backend = burn::backend::NdArray<f32, i64>;
    let args = cli::Args::parse();
    let device = burn::backend::ndarray::NdArrayDevice::default();
    run::run::<Backend>(args, device)
}
