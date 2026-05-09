use burn::tensor::backend::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{Int as TensorInt, Tensor as BurnTensor, TensorPrimitive};

/// Backend extension for sampled counted Tanimoto ranking labels.
pub trait CountedTanimotoKernelBackend: Backend {
    /// Build a sampled similarity-ranking batch on the device.
    fn counted_tanimoto_similarity_ranking_kernel(
        indices: IntTensor<Self>,
        counts: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        config: CountedTanimotoRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>);
}

/// Scalar options for the GPU-only counted Tanimoto ranking metric.
#[derive(Clone, Copy, Debug)]
pub struct CountedTanimotoRankingKernelConfig {
    /// Number of batch items to score.
    pub batch_items: usize,
    /// Random candidate partners sampled for each anchor.
    pub candidates_per_anchor: usize,
    /// Per-batch deterministic seed.
    pub seed: u64,
    /// Numerical stabilizer used by the scorer.
    pub epsilon: f64,
}

/// Build sampled counted Tanimoto ranking labels on the backend.
pub fn counted_tanimoto_similarity_ranking_kernel<B: CountedTanimotoKernelBackend>(
    indices: BurnTensor<B, 2, TensorInt>,
    counts: BurnTensor<B, 2>,
    mask: BurnTensor<B, 2>,
    config: CountedTanimotoRankingKernelConfig,
) -> (
    BurnTensor<B, 1, TensorInt>,
    BurnTensor<B, 1, TensorInt>,
    BurnTensor<B, 2>,
) {
    let (partner_a, partner_b, target_delta) = B::counted_tanimoto_similarity_ranking_kernel(
        indices.into_primitive(),
        counts.into_primitive().tensor(),
        mask.into_primitive().tensor(),
        config,
    );

    (
        BurnTensor::new(partner_a),
        BurnTensor::new(partner_b),
        BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(target_delta))
            .reshape([config.batch_items, 1]),
    )
}
