//! Latent geometry support losses for molecular embeddings.

use burn::{
    prelude::*,
    tensor::{Int, TensorData},
};

/// Metric-derived Tanimoto ranking partners and score gaps.
#[derive(Debug, Clone)]
pub struct TanimotoRankingBatch<B: Backend> {
    /// First partner row index for each anchor.
    pub partner_a_index: Tensor<B, 1, Int>,
    /// Second partner row index for each anchor.
    pub partner_b_index: Tensor<B, 1, Int>,
    /// Metric score delta `tanimoto(anchor, a) - tanimoto(anchor, b)`.
    pub target_delta: Tensor<B, 2>,
}

impl<B: Backend> TanimotoRankingBatch<B> {
    /// Creates an empty ranking batch with no valid metric gaps.
    #[must_use]
    pub fn zeros(batch_size: usize, device: &B::Device) -> Self {
        let indices = vec![0_i64; batch_size];
        Self {
            partner_a_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(indices.clone(), [batch_size]),
                device,
            ),
            partner_b_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(indices, [batch_size]),
                device,
            ),
            target_delta: Tensor::<B, 2>::zeros([batch_size, 1], device),
        }
    }
}

/// Tanimoto-ranking objective result and diagnostics.
pub struct TanimotoRankingOutput<B: Backend> {
    /// Weighted or unweighted ranking loss, depending on caller.
    pub loss: Tensor<B, 1>,
    /// Number of valid anchors whose metric gap exceeded the threshold.
    pub valid_pairs: Tensor<B, 1>,
    /// Fraction of valid anchors whose latent similarity preserved metric order.
    pub accuracy: Tensor<B, 1>,
}

/// In-batch ranking objective for counted-fingerprint Tanimoto metric order.
#[must_use]
pub fn tanimoto_ranking_output<B: Backend>(
    latent: Tensor<B, 2>,
    batch: TanimotoRankingBatch<B>,
    max_pairs: usize,
    margin: f64,
    min_gap: f64,
) -> TanimotoRankingOutput<B> {
    let [batch_size, _latent_width] = latent.dims();
    if batch_size < 3 {
        return zero_tanimoto_ranking_output(&latent.device());
    }

    let pair_count = if max_pairs == 0 {
        batch_size
    } else {
        batch_size.min(max_pairs)
    };
    if pair_count == 0 {
        return zero_tanimoto_ranking_output(&latent.device());
    }

    let anchor_latent = latent.clone().narrow(0, 0, pair_count);
    let latent_a = latent
        .clone()
        .select(0, batch.partner_a_index.narrow(0, 0, pair_count));
    let latent_b = latent
        .clone()
        .select(0, batch.partner_b_index.narrow(0, 0, pair_count));
    let latent_delta = row_cosine_similarity(anchor_latent.clone(), latent_a)
        - row_cosine_similarity(anchor_latent, latent_b);
    let target_delta = batch.target_delta.narrow(0, 0, pair_count).detach();
    let target_gap = target_delta.clone().abs();
    let target_direction = target_delta / (target_gap.clone() + 1.0e-6);
    let valid = target_gap.clone().greater_elem(min_gap).float();
    let valid_pairs = valid.clone().sum();
    let gap_weights = target_gap * valid.clone();
    let gap_weight_sum = gap_weights.clone().sum().clamp_min(1.0e-6);
    let ordered_delta = target_direction * latent_delta;
    let hinge = (margin - ordered_delta.clone()).clamp_min(0.0);
    let accuracy = (ordered_delta.greater_elem(0.0).float() * valid.clone()).sum()
        / valid_pairs.clone().clamp_min(1.0);

    TanimotoRankingOutput {
        loss: (hinge * gap_weights).sum() / gap_weight_sum,
        valid_pairs,
        accuracy,
    }
}

/// Weighted Tanimoto-ranking output, or zero diagnostics when disabled.
#[must_use]
pub fn weighted_tanimoto_ranking_output<B: Backend>(
    latent: Tensor<B, 2>,
    batch: TanimotoRankingBatch<B>,
    max_pairs: usize,
    margin: f64,
    min_gap: f64,
    weight: f64,
) -> TanimotoRankingOutput<B> {
    if weight > 0.0 {
        let mut output = tanimoto_ranking_output(latent, batch, max_pairs, margin, min_gap);
        output.loss = output.loss * weight;
        output
    } else {
        zero_tanimoto_ranking_output(&latent.device())
    }
}

fn zero_tanimoto_ranking_output<B: Backend>(device: &B::Device) -> TanimotoRankingOutput<B> {
    TanimotoRankingOutput {
        loss: Tensor::zeros([1], device),
        valid_pairs: Tensor::zeros([1], device),
        accuracy: Tensor::zeros([1], device),
    }
}

fn row_cosine_similarity<B: Backend>(left: Tensor<B, 2>, right: Tensor<B, 2>) -> Tensor<B, 2> {
    let numerator = (left.clone() * right.clone()).sum_dim(1);
    let left_norm = (left.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    let right_norm = (right.powf_scalar(2.0).sum_dim(1) + 1.0e-6).sqrt();
    (numerator / (left_norm * right_norm))
        .clamp_min(-1.0)
        .clamp_max(1.0)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;

    type B = burn::backend::NdArray<f32, i64>;

    #[test]
    fn zero_batch_has_expected_shapes() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batch = TanimotoRankingBatch::<B>::zeros(3, &device);

        assert_eq!(batch.partner_a_index.dims(), [3]);
        assert_eq!(batch.partner_b_index.dims(), [3]);
        assert_eq!(batch.target_delta.dims(), [3, 1]);
    }

    #[test]
    fn ranking_loss_rewards_preserved_order() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::from_floats([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], &device);
        let batch = TanimotoRankingBatch {
            partner_a_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![1_i64], [1]),
                &device,
            ),
            partner_b_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![2_i64], [1]),
                &device,
            ),
            target_delta: Tensor::<B, 2>::from_data(
                TensorData::new(vec![0.8_f32], [1, 1]),
                &device,
            ),
        };

        let output = tanimoto_ranking_output(latent, batch, 1, 0.05, 0.01);

        assert!(output.loss.into_scalar() < 0.05);
        assert_eq!(output.valid_pairs.into_scalar(), 1.0);
        assert_eq!(output.accuracy.into_scalar(), 1.0);
    }

    #[test]
    fn ranking_loss_ignores_small_metric_gaps() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::ones([3, 2], &device);
        let batch = TanimotoRankingBatch {
            partner_a_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![1_i64], [1]),
                &device,
            ),
            partner_b_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![2_i64], [1]),
                &device,
            ),
            target_delta: Tensor::<B, 2>::from_data(
                TensorData::new(vec![0.001_f32], [1, 1]),
                &device,
            ),
        };

        let output = tanimoto_ranking_output(latent, batch, 1, 0.05, 0.01);

        assert_eq!(output.loss.into_scalar(), 0.0);
        assert_eq!(output.valid_pairs.into_scalar(), 0.0);
    }
}
