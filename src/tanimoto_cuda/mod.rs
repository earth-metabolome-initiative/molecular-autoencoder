//! CUDA kernels for counted sparse fingerprint Tanimoto ranking metrics.

#![allow(
    clippy::cast_possible_truncation,
    clippy::comparison_chain,
    clippy::too_many_arguments,
    clippy::trivially_copy_pass_by_ref
)]

/// Public kernel API: trait, free function, and config builder.
mod api;
/// Autodiff forwarder so `Autodiff<B>` inherits the kernel impl with a
/// no-grad backward step.
#[cfg(feature = "train")]
mod autodiff;
/// Pure-Rust per-anchor ranking algorithm shared by the CPU reference
/// (test oracle) and the NdArray backend.
mod cpu;
/// `CubeBackend<R, _, _, _>` impl that dispatches to the `#[cube]`
/// kernel on any CubeCL runtime (CUDA / CubeCL-CPU / Wgpu).
mod cube_backend;
/// Fusion-aware forwarder for `Fusion<B>`.
#[cfg(feature = "cuda-fusion")]
mod fusion;
/// `#[cube]` kernel source compiled by the CubeCL toolchain at runtime.
mod kernels;
/// NdArray backend impl that loops over anchors on the CPU.
#[cfg(feature = "ndarray")]
mod ndarray_backend;
/// Cross-backend regression test against the CPU reference algorithm.
#[cfg(test)]
mod tests;

pub use api::{
    CountedTanimotoKernelBackend, CountedTanimotoRankingKernelConfig,
    CountedTanimotoRankingKernelConfigBuilder, counted_tanimoto_similarity_ranking_kernel,
};
