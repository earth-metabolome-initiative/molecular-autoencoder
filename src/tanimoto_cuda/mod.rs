//! CUDA kernels for counted sparse fingerprint Tanimoto ranking metrics.

#![allow(
    clippy::cast_possible_truncation,
    clippy::comparison_chain,
    clippy::too_many_arguments,
    clippy::trivially_copy_pass_by_ref
)]

mod api;
#[cfg(feature = "train")]
mod autodiff;
mod cube_backend;
#[cfg(feature = "cuda-fusion")]
mod fusion;
mod kernels;
#[cfg(test)]
mod tests;

pub use api::{
    CountedTanimotoKernelBackend, CountedTanimotoRankingKernelConfig,
    CountedTanimotoRankingKernelConfigBuilder, counted_tanimoto_similarity_ranking_kernel,
};
