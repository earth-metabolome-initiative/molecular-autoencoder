use burn::tensor::backend::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{Int as TensorInt, Tensor as BurnTensor, TensorPrimitive};

/// Backend extension for sampled counted Tanimoto ranking labels.
pub trait CountedTanimotoKernelBackend: Backend {
    /// Build a sampled similarity-ranking batch on the device.
    ///
    /// Returns, in order:
    /// 1. `[batch, candidates]` candidate row indices,
    /// 2. `[batch]` position of the best-scoring candidate within each
    ///    anchor's candidate list,
    /// 3. `[batch]` clamped score gap between the top-1 and top-2
    ///    candidates,
    /// 4. `[batch, candidates]` per-candidate counted-Tanimoto similarity
    ///    score (the same values internally used to derive the top-1 /
    ///    top-2 picks). Surfaced so downstream ranking-quality metrics
    ///    can correlate them with the latent cosine logits.
    fn counted_tanimoto_similarity_ranking_kernel(
        indices: IntTensor<Self>,
        counts: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        config: CountedTanimotoRankingKernelConfig,
    ) -> (
        IntTensor<Self>,
        IntTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );
}

/// Scalar options for the GPU-only counted Tanimoto ranking metric.
/// Construct via [`CountedTanimotoRankingKernelConfig::builder`].
#[derive(Clone, Copy, Debug)]
pub struct CountedTanimotoRankingKernelConfig {
    /// Number of anchors the kernel will score.
    batch_items: usize,
    /// Random candidate partners drawn per anchor.
    candidates_per_anchor: usize,
    /// Per-batch deterministic RNG seed.
    seed: u64,
    /// Numerical stabiliser added to the Tanimoto denominator.
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
    /// Pending anchor count; `None` until `batch_items` is called.
    batch_items: Option<usize>,
    /// Pending candidate partners per anchor.
    candidates_per_anchor: usize,
    /// Pending per-batch RNG seed.
    seed: u64,
    /// Pending denominator stabiliser.
    epsilon: f64,
}

impl Default for CountedTanimotoRankingKernelConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Default numerical stabilizer for the counted-Tanimoto kernel.
pub const DEFAULT_KERNEL_EPSILON: f64 = 1.0e-8;

impl CountedTanimotoRankingKernelConfigBuilder {
    /// Creates a builder with sensible defaults; `batch_items` must be set
    /// before `build`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            batch_items: None,
            candidates_per_anchor: crate::model::DEFAULT_TANIMOTO_RANKING_CANDIDATES_PER_ANCHOR,
            seed: 0,
            epsilon: DEFAULT_KERNEL_EPSILON,
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

/// Output tuple of [`counted_tanimoto_similarity_ranking_kernel`]:
/// `(candidate_index, best_candidate_position, top2_gap,
///   candidate_scores)`. Pulled out so clippy's `type_complexity` lint
/// stays quiet on the call signature.
pub type CountedTanimotoRankingKernelOutputs<B> = (
    BurnTensor<B, 2, TensorInt>,
    BurnTensor<B, 1, TensorInt>,
    BurnTensor<B, 2>,
    BurnTensor<B, 2>,
);

/// Build sampled counted Tanimoto ranking labels on the backend.
pub fn counted_tanimoto_similarity_ranking_kernel<B: CountedTanimotoKernelBackend>(
    indices: BurnTensor<B, 2, TensorInt>,
    counts: BurnTensor<B, 2>,
    mask: BurnTensor<B, 2>,
    config: CountedTanimotoRankingKernelConfig,
) -> CountedTanimotoRankingKernelOutputs<B> {
    let (candidate_index, best_candidate_position, top2_gap, candidate_scores) =
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
        BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(candidate_scores)),
    )
}

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[test]
    fn build_without_batch_items_is_rejected() {
        let err = CountedTanimotoRankingKernelConfig::builder()
            .build()
            .expect_err("missing batch_items must fail");
        assert!(matches!(err, crate::Error::ConfigInvalid { .. }));
    }

    #[test]
    fn nan_epsilon_is_rejected() {
        let err = CountedTanimotoRankingKernelConfig::builder()
            .batch_items(4)
            .epsilon(f64::NAN)
            .build()
            .expect_err("NaN epsilon must fail");
        assert!(matches!(err, crate::Error::ConfigInvalid { .. }));
    }

    #[test]
    fn negative_epsilon_is_rejected() {
        let err = CountedTanimotoRankingKernelConfig::builder()
            .batch_items(4)
            .epsilon(-1.0e-9)
            .build()
            .expect_err("non-positive epsilon must fail");
        assert!(matches!(err, crate::Error::ConfigInvalid { .. }));
    }

    #[test]
    fn zero_epsilon_is_rejected() {
        let err = CountedTanimotoRankingKernelConfig::builder()
            .batch_items(4)
            .epsilon(0.0)
            .build()
            .expect_err("zero epsilon must fail");
        assert!(matches!(err, crate::Error::ConfigInvalid { .. }));
    }

    #[test]
    fn single_candidate_per_anchor_is_rejected() {
        let err = CountedTanimotoRankingKernelConfig::builder()
            .batch_items(4)
            .candidates_per_anchor(1)
            .build()
            .expect_err("candidates_per_anchor < 2 must fail");
        assert!(matches!(err, crate::Error::ConfigInvalid { .. }));
    }

    #[test]
    fn build_preserves_all_fields() {
        let config = CountedTanimotoRankingKernelConfig::builder()
            .batch_items(16)
            .candidates_per_anchor(8)
            .seed(0xdead_beef)
            .epsilon(1.0e-6)
            .build()
            .expect("valid builder must produce a config");
        assert_eq!(config.batch_items(), 16);
        assert_eq!(config.candidates_per_anchor(), 8);
        assert_eq!(config.seed(), 0xdead_beef);
        assert!((config.epsilon() - 1.0e-6).abs() < f64::EPSILON);
    }

    #[test]
    fn effective_candidates_saturates_to_batch_minus_one() {
        let config = CountedTanimotoRankingKernelConfig::builder()
            .batch_items(3)
            .candidates_per_anchor(8)
            .build()
            .expect("batch_items=3 is valid");
        assert_eq!(config.effective_candidates_per_anchor(), 2);
    }

    #[test]
    fn effective_candidates_clamps_to_at_least_two_when_possible() {
        let config = CountedTanimotoRankingKernelConfig::builder()
            .batch_items(8)
            .candidates_per_anchor(2)
            .build()
            .expect("min candidates is valid");
        assert_eq!(config.effective_candidates_per_anchor(), 2);
    }

    #[test]
    fn effective_candidates_is_zero_for_singleton_batch() {
        let config = CountedTanimotoRankingKernelConfig::builder()
            .batch_items(1)
            .candidates_per_anchor(4)
            .build()
            .expect("batch_items=1 is technically valid for the builder");
        assert_eq!(config.effective_candidates_per_anchor(), 0);
    }
}
