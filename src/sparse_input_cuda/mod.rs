//! EmbeddingBag-style kernel for the encoder's sparse-input linear layer.
//!
//! Replaces the dense `scatter`-then-matmul pattern that nsys flagged
//! at 22% of GPU time (`select_assign_kernel_f_f32_i_i32`).
//!
//! The forward streams over each row's K active indices in registers
//! (one CUDA thread per `(batch, hidden)` cell), so the `[batch, K,
//! hidden]` intermediate that OOMed our prior composed-ops attempt
//! never materialises. The backward currently runs through Burn's
//! composed dense ops (scatter to reconstruct the input, then dense
//! matmul) because `cubecl-cpu` does not yet implement `atomic<f32>`
//! and the cubecl `FloatAtomicAddFamily` boilerplate is substantial; a
//! custom backward kernel can slot in behind the same trait later
//! without touching call sites. The win we capture today is the
//! forward-side kernel (`select_assign` gone from the train path).

#![allow(
    clippy::cast_possible_truncation,
    clippy::comparison_chain,
    clippy::too_many_arguments,
    clippy::trivially_copy_pass_by_ref,
    clippy::used_underscore_binding,
    missing_docs,
    clippy::missing_docs_in_private_items
)]

/// Public trait + free function entry point. Always compiled so the
/// trait exists for the model's bound regardless of which backend
/// implementation is enabled.
mod api;
/// `Autodiff<B>` forwarder: wraps the trait so autograd-aware backends
/// inherit the kernel with a hand-written backward.
#[cfg(feature = "train")]
mod autodiff;
/// `CubeBackend<R, _, _, _>` impl that dispatches to the `#[cube]`
/// kernels on any CubeCL runtime (CUDA, CubeCL-CPU).
#[cfg(any(feature = "cuda", feature = "cubecl-cpu"))]
mod cube_backend;
/// Fusion forwarder so `Fusion<B>` (cuda-fusion path) records the op
/// in the fusion graph and replays it on the inner backend.
#[cfg(feature = "cuda-fusion")]
mod fusion;
/// `#[cube]` kernel sources compiled by the CubeCL toolchain at runtime.
#[cfg(any(feature = "cuda", feature = "cubecl-cpu"))]
mod kernels;
/// NdArray composed-ops fallback. Replicates the previous dense
/// scatter-then-matmul implementation so ndarray-only builds (tests,
/// wasm) still compile and pass.
#[cfg(feature = "ndarray")]
mod ndarray_backend;
/// Cross-backend regression tests against an explicit dense reference.
/// Requires a cubecl runtime (either `cuda` for GPU or `cubecl-cpu`
/// for CPU CI); skipped on ndarray-only test builds.
#[cfg(all(test, any(feature = "cuda", feature = "cubecl-cpu")))]
mod tests;

pub use api::{SparseInputLinearKernelBackend, sparse_input_linear_forward};
