//! Training binary for the molecular autoencoder.
//!
//! Selects an autodiff backend, parses CLI arguments via [`clap`], and
//! delegates to [`run::run`] for the full training pipeline (preprocessing,
//! dataloading, training, validation, and checkpointing).

use std::error::Error as StdError;

use clap::Parser;

mod checkpoint;
mod cli;
mod dataloader;
mod metrics;
mod preprocess;
mod ranking;
mod reporter;
mod run;
mod training;

/// Unified error type used across the training bin.
pub type AppResult<T> = Result<T, Box<dyn StdError>>;

/// Wraps a `String` message into the unified error type.
pub fn invalid_input(message: impl Into<String>) -> Box<dyn StdError> {
    Box::new(molecular_autoencoder::Error::InvalidBatch(message.into()))
}

#[cfg(feature = "cuda")]
fn main() -> AppResult<()> {
    type Backend = burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>;
    let args = cli::Args::parse();
    let device = burn::backend::cuda::CudaDevice::new(args.cuda_device);
    run::run::<Backend>(args, device)
}

#[cfg(not(feature = "cuda"))]
fn main() -> AppResult<()> {
    type Backend = burn::backend::Autodiff<burn::backend::NdArray<f32, i64>>;
    eprintln!(
        "warning: running with ndarray backend for smoke testing; enable `cuda-fusion` or \
         `cuda-no-fusion` for the intended training path"
    );
    let args = cli::Args::parse();
    let device = burn::backend::ndarray::NdArrayDevice::default();
    run::run::<Backend>(args, device)
}
