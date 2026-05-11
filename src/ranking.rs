//! Latent geometry support losses for molecular embeddings.

use burn::{
    prelude::*,
    tensor::{Int, TensorData, activation::log_softmax},
};

/// Metric-derived Tanimoto geometry candidates and score gaps.
#[derive(Debug, Clone)]
pub struct TanimotoRankingBatch<B: Backend> {
    /// Candidate partner row indices for each anchor, shaped `[batch, candidates]`.
    pub candidate_index: Tensor<B, 2, Int>,
    /// Position of the highest-scoring candidate in each anchor's candidate row.
    pub best_candidate_position: Tensor<B, 1, Int>,
    /// Counted Tanimoto score gap between the best and runner-up candidates.
    pub top2_gap: Tensor<B, 2>,
}

impl<B: Backend> TanimotoRankingBatch<B> {
    /// Creates an empty ranking batch with no valid metric gaps.
    #[must_use]
    pub fn zeros(batch_size: usize, device: &B::Device) -> Self {
        let indices = vec![0_i64; batch_size];
        Self {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![0_i64; batch_size * 2], [batch_size, 2]),
                device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(indices, [batch_size]),
                device,
            ),
            top2_gap: Tensor::<B, 2>::zeros([batch_size, 1], device),
        }
    }
}

/// Tanimoto geometry objective result and diagnostics.
pub struct TanimotoRankingOutput<B: Backend> {
    /// Weighted or unweighted softmax cross-entropy loss, depending on caller.
    pub loss: Tensor<B, 1>,
    /// Number of valid anchors whose metric gap exceeded the threshold.
    pub valid_pairs: Tensor<B, 1>,
    /// Fraction of valid anchors whose latent nearest candidate matched the metric.
    pub accuracy: Tensor<B, 1>,
}

/// In-batch softmax ranking objective for counted-fingerprint Tanimoto metric order.
#[must_use]
pub fn tanimoto_ranking_output<B: Backend>(
    latent: Tensor<B, 2>,
    batch: TanimotoRankingBatch<B>,
    max_pairs: usize,
    latent_temperature: f64,
    _metric_temperature: f64,
    min_gap: f64,
) -> TanimotoRankingOutput<B> {
    let [batch_size, latent_width] = latent.dims();
    if batch_size < 3 {
        return zero_tanimoto_ranking_output(&latent.device());
    }
    let [_candidate_rows, candidate_count] = batch.candidate_index.dims();
    if candidate_count < 2 {
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
    let candidate_index = batch.candidate_index.narrow(0, 0, pair_count);
    let flat_candidate_index = candidate_index.reshape([pair_count * candidate_count]);
    let candidate_latent = latent.clone().select(0, flat_candidate_index).reshape([
        pair_count,
        candidate_count,
        latent_width,
    ]);
    let anchor_latent =
        anchor_latent
            .unsqueeze_dim::<3>(1)
            .expand([pair_count, candidate_count, latent_width]);
    let logits =
        row_cosine_similarity_3d(anchor_latent, candidate_latent) / latent_temperature.max(1.0e-6);
    let log_probs = log_softmax(logits.clone(), 1);
    let target = batch
        .best_candidate_position
        .clone()
        .narrow(0, 0, pair_count)
        .one_hot::<2>(candidate_count)
        .float();
    let cross_entropy = (log_probs * target * -1.0).sum_dim(1);
    let target_gap = batch.top2_gap.narrow(0, 0, pair_count).detach();
    let valid = target_gap.clone().greater_elem(min_gap).float();
    let valid_pairs = valid.clone().sum();
    let gap_weights = target_gap * valid.clone();
    let gap_weight_sum = gap_weights.clone().sum().clamp_min(1.0e-6);
    let predicted_best = logits.argmax(1);
    let metric_best = batch
        .best_candidate_position
        .narrow(0, 0, pair_count)
        .reshape([pair_count, 1]);
    let accuracy = (predicted_best.equal(metric_best).float() * valid.clone()).sum()
        / valid_pairs.clone().clamp_min(1.0);

    TanimotoRankingOutput {
        loss: (cross_entropy * gap_weights).sum() / gap_weight_sum,
        valid_pairs,
        accuracy,
    }
}

/// Weighted Tanimoto geometry output, or zero diagnostics when disabled.
#[must_use]
pub fn weighted_tanimoto_ranking_output<B: Backend>(
    latent: Tensor<B, 2>,
    batch: TanimotoRankingBatch<B>,
    max_pairs: usize,
    latent_temperature: f64,
    metric_temperature: f64,
    min_gap: f64,
    weight: f64,
) -> TanimotoRankingOutput<B> {
    if weight > 0.0 {
        let mut output = tanimoto_ranking_output(
            latent,
            batch,
            max_pairs,
            latent_temperature,
            metric_temperature,
            min_gap,
        );
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

fn row_cosine_similarity_3d<B: Backend>(left: Tensor<B, 3>, right: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, candidate_count, _latent_width] = left.dims();
    let numerator = (left.clone() * right.clone()).sum_dim(2);
    let left_norm = (left.powf_scalar(2.0).sum_dim(2) + 1.0e-6).sqrt();
    let right_norm = (right.powf_scalar(2.0).sum_dim(2) + 1.0e-6).sqrt();
    (numerator / (left_norm * right_norm))
        .clamp_min(-1.0)
        .clamp_max(1.0)
        .reshape([batch_size, candidate_count])
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;

    type B = burn::backend::NdArray<f32, i64>;

    #[test]
    fn zero_batch_has_expected_shapes() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batch = TanimotoRankingBatch::<B>::zeros(3, &device);

        assert_eq!(batch.candidate_index.dims(), [3, 2]);
        assert_eq!(batch.best_candidate_position.dims(), [3]);
        assert_eq!(batch.top2_gap.dims(), [3, 1]);
    }

    #[test]
    fn ranking_loss_rewards_preserved_order() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::from_floats([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], &device);
        let preserved_batch = TanimotoRankingBatch {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![1_i64, 2], [1, 2]),
                &device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![0_i64], [1]),
                &device,
            ),
            top2_gap: Tensor::<B, 2>::from_data(TensorData::new(vec![0.8_f32], [1, 1]), &device),
        };
        let reversed_batch = TanimotoRankingBatch {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![2_i64, 1], [1, 2]),
                &device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![0_i64], [1]),
                &device,
            ),
            top2_gap: Tensor::<B, 2>::from_data(TensorData::new(vec![0.8_f32], [1, 1]), &device),
        };

        let preserved =
            tanimoto_ranking_output(latent.clone(), preserved_batch, 1, 0.10, 0.10, 0.01);
        let reversed = tanimoto_ranking_output(latent, reversed_batch, 1, 0.10, 0.10, 0.01);

        assert!(preserved.loss.into_scalar() < reversed.loss.into_scalar());
        assert_eq!(preserved.valid_pairs.into_scalar(), 1.0);
        assert_eq!(preserved.accuracy.into_scalar(), 1.0);
        assert_eq!(reversed.accuracy.into_scalar(), 0.0);
    }

    #[test]
    fn ranking_loss_ignores_small_metric_gaps() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::ones([3, 2], &device);
        let batch = TanimotoRankingBatch {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![1_i64, 2], [1, 2]),
                &device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![0_i64], [1]),
                &device,
            ),
            top2_gap: Tensor::<B, 2>::from_data(TensorData::new(vec![0.001_f32], [1, 1]), &device),
        };

        let output = tanimoto_ranking_output(latent, batch, 1, 0.10, 0.10, 0.01);

        assert_eq!(output.loss.into_scalar(), 0.0);
        assert_eq!(output.valid_pairs.into_scalar(), 0.0);
    }

    #[test]
    fn ranking_loss_weights_softmax_cross_entropy_by_top2_gap() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::from_floats([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &device);
        let batch = TanimotoRankingBatch {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![1_i64, 2, 2, 0, 0, 1], [3, 2]),
                &device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![0_i64, 0, 0], [3]),
                &device,
            ),
            top2_gap: Tensor::<B, 2>::from_floats([[0.9], [0.1], [0.0]], &device),
        };

        let loss = tanimoto_ranking_output(latent, batch, 0, 0.5, 0.5, 0.01)
            .loss
            .into_scalar();
        let expected = (0.9 * softmax_ce_for_class0(2.0, 0.0)
            + 0.1 * softmax_ce_for_class0(0.0, 2.0))
            / (0.9 + 0.1);

        assert!(
            (loss - expected).abs() < 1.0e-3,
            "expected gap-weighted softmax CE near {expected}, got {loss}"
        );
    }

    #[test]
    fn ranking_pairs_per_batch_limits_anchors() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::from_floats([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &device);
        let batch = TanimotoRankingBatch {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![1_i64, 2, 2, 0, 0, 1], [3, 2]),
                &device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![0_i64, 0, 0], [3]),
                &device,
            ),
            top2_gap: Tensor::<B, 2>::from_floats([[0.9], [0.9], [0.9]], &device),
        };

        let output = tanimoto_ranking_output(latent, batch, 1, 0.5, 0.5, 0.01);

        assert_eq!(output.valid_pairs.into_scalar(), 1.0);
    }

    #[test]
    fn ranking_loss_is_zero_for_too_small_batches() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::zeros([2, 2], &device);
        let batch = TanimotoRankingBatch::zeros(2, &device);

        let output = tanimoto_ranking_output(latent, batch, 0, 0.10, 0.10, 0.01);

        assert_eq!(output.loss.into_scalar(), 0.0);
        assert_eq!(output.valid_pairs.into_scalar(), 0.0);
        assert_eq!(output.accuracy.into_scalar(), 0.0);
    }

    fn softmax_ce_for_class0(logit0: f32, logit1: f32) -> f32 {
        let max_logit = logit0.max(logit1);
        max_logit + ((logit0 - max_logit).exp() + (logit1 - max_logit).exp()).ln() - logit0
    }
}
