//! Tanimoto-ranking attachment for training and validation batches.
//!
//! On CUDA, runs the counted-Tanimoto sampling kernel to fill candidate
//! indices, best-position labels, and gap weights. On other backends the
//! attachment is a no-op (the geometry loss path is CUDA-only today).

use burn::tensor::backend::Backend;
use molecular_autoencoder::{MoleculeAutoencoderBatch, TanimotoRankingRuntimeConfig};

#[cfg(feature = "cuda")]
use molecular_autoencoder::{
    TanimotoRankingBatch,
    tanimoto_cuda::{
        CountedTanimotoKernelBackend, CountedTanimotoRankingKernelConfig,
        counted_tanimoto_similarity_ranking_kernel,
    },
};

/// Backend trait alias used by training loops to allow CUDA-only metric paths.
#[cfg(feature = "cuda")]
pub trait TanimotoMetricBackend: Backend + CountedTanimotoKernelBackend {}

#[cfg(feature = "cuda")]
impl<B> TanimotoMetricBackend for B where B: Backend + CountedTanimotoKernelBackend {}

#[cfg(not(feature = "cuda"))]
pub trait TanimotoMetricBackend: Backend {}

#[cfg(not(feature = "cuda"))]
impl<B> TanimotoMetricBackend for B where B: Backend {}

/// Per-batch deterministic seed derived from `(epoch, batch_index)`.
pub fn tanimoto_ranking_seed(epoch: usize, batch_index: usize) -> u64 {
    ((epoch as u64) << 32) ^ (batch_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Fills the Tanimoto ranking labels on the batch when the geometry loss is
/// enabled (weight > 0, batch size >= 3) and a CUDA backend is present.
#[cfg(feature = "cuda")]
pub fn attach_tanimoto_ranking<B>(
    batch: &mut MoleculeAutoencoderBatch<B>,
    config: TanimotoRankingRuntimeConfig,
    seed: u64,
) where
    B: TanimotoMetricBackend,
{
    if config.weight <= 0.0 || batch.fingerprints.batch_size() < 3 {
        return;
    }
    let (candidate_index, best_candidate_position, top2_gap) =
        counted_tanimoto_similarity_ranking_kernel(
            batch.fingerprints.indices.clone(),
            batch.fingerprints.counts.clone(),
            batch.fingerprints.mask.clone(),
            CountedTanimotoRankingKernelConfig {
                batch_items: batch.fingerprints.batch_size(),
                candidates_per_anchor: config.candidates_per_anchor,
                seed,
                epsilon: 1.0e-8,
            },
        );
    batch.tanimoto_ranking = TanimotoRankingBatch {
        candidate_index,
        best_candidate_position,
        top2_gap,
    };
}

#[cfg(not(feature = "cuda"))]
pub fn attach_tanimoto_ranking<B>(
    _batch: &mut MoleculeAutoencoderBatch<B>,
    _config: TanimotoRankingRuntimeConfig,
    _seed: u64,
) where
    B: TanimotoMetricBackend,
{
}

/// Returns the effective device-prefetch depth after applying CUDA safety.
///
/// CUDA tensor construction from a background thread can trigger
/// `CUDA_ERROR_ILLEGAL_ADDRESS` in Burn/CubeCL, so requested prefetch is
/// forced to zero with a warning. Host-side prefetch via `--loader-workers`
/// stays enabled.
pub fn effective_device_prefetch_batches(requested: usize) -> usize {
    #[cfg(feature = "cuda")]
    {
        if requested > 0 {
            eprintln!(
                "warning: disabling --device-prefetch-batches={requested} for CUDA; \
                 Burn/CubeCL CUDA tensor construction from a background thread can cause \
                 CUDA_ERROR_ILLEGAL_ADDRESS. Host batch prefetch remains enabled via \
                 --loader-workers."
            );
        }
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        requested
    }
}
