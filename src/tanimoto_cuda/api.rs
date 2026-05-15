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
/// Construct via [`CountedTanimotoRankingKernelConfig::builder`].
#[derive(Clone, Copy, Debug)]
pub struct CountedTanimotoRankingKernelConfig {
    batch_items: usize,
    candidates_per_anchor: usize,
    seed: u64,
    epsilon: f64,
}

impl CountedTanimotoRankingKernelConfig {
    /// Starts a fluent builder.
    #[must_use]
    pub fn builder() -> CountedTanimotoRankingKernelConfigBuilder {
        CountedTanimotoRankingKernelConfigBuilder::new()
    }

    /// Number of batch items to score.
    #[must_use]
    pub const fn batch_items(self) -> usize {
        self.batch_items
    }

    /// Random candidate partners sampled for each anchor.
    #[must_use]
    pub const fn candidates_per_anchor(self) -> usize {
        self.candidates_per_anchor
    }

    /// Per-batch deterministic seed.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Numerical stabilizer used by the scorer.
    #[must_use]
    pub const fn epsilon(self) -> f64 {
        self.epsilon
    }

    /// Effective number of candidate partners scored for each anchor.
    #[must_use]
    pub fn effective_candidates_per_anchor(self) -> usize {
        self.candidates_per_anchor
            .max(2)
            .min(self.batch_items.saturating_sub(1))
    }
}

/// Fluent builder for [`CountedTanimotoRankingKernelConfig`].
#[derive(Clone, Copy, Debug)]
pub struct CountedTanimotoRankingKernelConfigBuilder {
    batch_items: Option<usize>,
    candidates_per_anchor: usize,
    seed: u64,
    epsilon: f64,
}

impl Default for CountedTanimotoRankingKernelConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CountedTanimotoRankingKernelConfigBuilder {
    /// Creates a builder with sensible defaults; `batch_items` must be set
    /// before `build`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            batch_items: None,
            candidates_per_anchor: 4,
            seed: 0,
            epsilon: 1.0e-8,
        }
    }

    /// Sets the number of batch items to score.
    #[must_use]
    pub const fn batch_items(mut self, value: usize) -> Self {
        self.batch_items = Some(value);
        self
    }

    /// Sets the number of random candidate partners per anchor.
    #[must_use]
    pub const fn candidates_per_anchor(mut self, value: usize) -> Self {
        self.candidates_per_anchor = value;
        self
    }

    /// Sets the per-batch deterministic seed.
    #[must_use]
    pub const fn seed(mut self, value: u64) -> Self {
        self.seed = value;
        self
    }

    /// Sets the numerical stabilizer.
    #[must_use]
    pub const fn epsilon(mut self, value: f64) -> Self {
        self.epsilon = value;
        self
    }

    /// Validates and builds the immutable kernel config.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::Error::ConfigInvalid`] when `batch_items` was not
    /// set, when `epsilon` is non-finite or non-positive, or when
    /// `candidates_per_anchor` is below the kernel's minimum of 2.
    pub fn build(self) -> crate::Result<CountedTanimotoRankingKernelConfig> {
        let batch_items = self
            .batch_items
            .ok_or_else(|| crate::Error::ConfigInvalid {
                message: "counted Tanimoto kernel batch_items must be set".to_string(),
            })?;
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(crate::Error::ConfigInvalid {
                message: format!(
                    "counted Tanimoto kernel epsilon must be positive and finite, got {}",
                    self.epsilon
                ),
            });
        }
        if self.candidates_per_anchor < 2 {
            return Err(crate::Error::ConfigInvalid {
                message: format!(
                    "counted Tanimoto kernel candidates_per_anchor must be at least 2, got {}",
                    self.candidates_per_anchor
                ),
            });
        }
        Ok(CountedTanimotoRankingKernelConfig {
            batch_items,
            candidates_per_anchor: self.candidates_per_anchor,
            seed: self.seed,
            epsilon: self.epsilon,
        })
    }
}

/// Build sampled counted Tanimoto ranking labels on the backend.
pub fn counted_tanimoto_similarity_ranking_kernel<B: CountedTanimotoKernelBackend>(
    indices: BurnTensor<B, 2, TensorInt>,
    counts: BurnTensor<B, 2>,
    mask: BurnTensor<B, 2>,
    config: CountedTanimotoRankingKernelConfig,
) -> (
    BurnTensor<B, 2, TensorInt>,
    BurnTensor<B, 1, TensorInt>,
    BurnTensor<B, 2>,
) {
    let (candidate_index, best_candidate_position, top2_gap) =
        B::counted_tanimoto_similarity_ranking_kernel(
            indices.into_primitive(),
            counts.into_primitive().tensor(),
            mask.into_primitive().tensor(),
            config,
        );

    (
        BurnTensor::new(candidate_index),
        BurnTensor::new(best_candidate_position),
        BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(top2_gap))
            .reshape([config.batch_items(), 1]),
    )
}
