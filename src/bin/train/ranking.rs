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

/// Backend trait alias used by training loops to allow CUDA-only metric paths.
#[cfg(not(feature = "cuda"))]
pub trait TanimotoMetricBackend: Backend {}

#[cfg(not(feature = "cuda"))]
impl<B> TanimotoMetricBackend for B where B: Backend {}

/// Per-batch deterministic seed derived from `(epoch, batch_index, peer_id)`.
///
/// The `peer_id` term keeps the per-batch RNG distinct across DDP workers, so
/// the contrastive anchor selection on rank 1 does not mirror rank 0's draws
/// for the same `(epoch, batch_index)`.
pub fn tanimoto_ranking_seed(epoch: usize, batch_index: usize, peer_id: u32) -> u64 {
    let base = ((epoch as u64) << 32) ^ (batch_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    base ^ (u64::from(peer_id)).wrapping_mul(0xbf58_476d_1ce4_e5b9)
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
    if config.weight() <= 0.0 || batch.fingerprints.batch_size() < 3 {
        return;
    }
    // `MoleculeAutoencoderConfig::validate` already enforces
    // `candidates_per_anchor >= 2` whenever the geometry loss weight is
    // positive, so the kernel builder cannot fail here.
    // `MoleculeAutoencoderConfig::validate` already enforces
    // `candidates_per_anchor >= 2` whenever the geometry loss weight is
    // positive, and `epsilon` falls back to the builder default
    // (`DEFAULT_KERNEL_EPSILON` in `tanimoto_cuda::api`), so the kernel
    // builder cannot fail here.
    let kernel_config = CountedTanimotoRankingKernelConfig::builder()
        .batch_items(batch.fingerprints.batch_size())
        .candidates_per_anchor(config.candidates_per_anchor())
        .seed(seed)
        .build()
        .expect("Tanimoto kernel config is valid");
    let (candidate_index, best_candidate_position, top2_gap, candidate_scores) =
        counted_tanimoto_similarity_ranking_kernel(
            batch.fingerprints.indices.clone(),
            batch.fingerprints.counts.clone(),
            batch.fingerprints.mask.clone(),
            kernel_config,
        );
    batch.tanimoto_ranking = TanimotoRankingBatch {
        candidate_index,
        best_candidate_position,
        top2_gap,
        candidate_scores,
    };
}

/// No-op stand-in used on non-CUDA backends — the geometry loss path is
/// CUDA-only today.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanimoto_ranking_seed_is_deterministic() {
        assert_eq!(
            tanimoto_ranking_seed(7, 31, 2),
            tanimoto_ranking_seed(7, 31, 2)
        );
    }

    #[test]
    fn tanimoto_ranking_seed_differs_across_epoch_batch_or_peer() {
        let base = tanimoto_ranking_seed(0, 0, 0);
        assert_ne!(base, tanimoto_ranking_seed(1, 0, 0));
        assert_ne!(base, tanimoto_ranking_seed(0, 1, 0));
        assert_ne!(base, tanimoto_ranking_seed(0, 0, 1));
    }

    #[test]
    fn tanimoto_ranking_seed_decorrelates_neighboring_peers() {
        // Two peers running the same (epoch, batch) must not draw identical
        // contrastive partners — guard the constant-multiplier reuse.
        let peer0 = tanimoto_ranking_seed(3, 17, 0);
        let peer1 = tanimoto_ranking_seed(3, 17, 1);
        let peer2 = tanimoto_ranking_seed(3, 17, 2);
        assert_ne!(peer0, peer1);
        assert_ne!(peer0, peer2);
        assert_ne!(peer1, peer2);
    }

    #[test]
    fn effective_device_prefetch_batches_passes_requested_value_on_cpu() {
        // Under non-CUDA builds the requested depth flows through unchanged;
        // under CUDA builds the helper warns and returns 0 (printed to stderr).
        let result = effective_device_prefetch_batches(4);
        #[cfg(feature = "cuda")]
        assert_eq!(result, 0);
        #[cfg(not(feature = "cuda"))]
        assert_eq!(result, 4);
    }

    #[test]
    fn effective_device_prefetch_batches_zero_request_is_idempotent() {
        assert_eq!(effective_device_prefetch_batches(0), 0);
    }
}
