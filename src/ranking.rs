//! Latent geometry support losses for molecular embeddings.

use burn::{
    prelude::*,
    tensor::{Int, TensorData, activation::sigmoid},
};

/// Metric-derived Tanimoto geometry partners and score gaps.
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

/// Tanimoto geometry objective result and diagnostics.
pub struct TanimotoRankingOutput<B: Backend> {
    /// Weighted or unweighted pairwise logistic loss, depending on caller.
    pub loss: Tensor<B, 1>,
    /// Number of valid anchors whose metric gap exceeded the threshold.
    pub valid_pairs: Tensor<B, 1>,
    /// Fraction of valid anchors whose latent similarity preserved metric order.
    pub accuracy: Tensor<B, 1>,
}

/// In-batch pairwise logistic objective for counted-fingerprint Tanimoto metric order.
#[must_use]
pub fn tanimoto_ranking_output<B: Backend>(
    latent: Tensor<B, 2>,
    batch: TanimotoRankingBatch<B>,
    max_pairs: usize,
    latent_temperature: f64,
    metric_temperature: f64,
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
    let target_direction = target_delta.clone() / (target_gap.clone() + 1.0e-6);
    let valid = target_gap.clone().greater_elem(min_gap).float();
    let valid_pairs = valid.clone().sum();
    let gap_weights = target_gap * valid.clone();
    let gap_weight_sum = gap_weights.clone().sum().clamp_min(1.0e-6);
    let ordered_delta = target_direction * latent_delta.clone();
    let latent_logit = latent_delta / latent_temperature.max(1.0e-6);
    let metric_target = sigmoid(target_delta / metric_temperature.max(1.0e-6));
    let binary_cross_entropy = latent_logit.clone().clamp_min(0.0)
        - latent_logit.clone() * metric_target
        + ((latent_logit.abs() * -1.0).exp() + 1.0).log();
    let accuracy = (ordered_delta.greater_elem(0.0).float() * valid.clone()).sum()
        / valid_pairs.clone().clamp_min(1.0);

    TanimotoRankingOutput {
        loss: (binary_cross_entropy * gap_weights).sum() / gap_weight_sum,
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
        let preserved_batch = TanimotoRankingBatch {
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
        let reversed_batch = TanimotoRankingBatch {
            partner_a_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![2_i64], [1]),
                &device,
            ),
            partner_b_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![1_i64], [1]),
                &device,
            ),
            target_delta: Tensor::<B, 2>::from_data(
                TensorData::new(vec![0.8_f32], [1, 1]),
                &device,
            ),
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

        let output = tanimoto_ranking_output(latent, batch, 1, 0.10, 0.10, 0.01);

        assert_eq!(output.loss.into_scalar(), 0.0);
        assert_eq!(output.valid_pairs.into_scalar(), 0.0);
    }

    #[test]
    fn ranking_loss_weights_pairwise_logistic_by_metric_gap() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::from_floats([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &device);
        let batch = TanimotoRankingBatch {
            partner_a_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![1_i64, 2, 0], [3]),
                &device,
            ),
            partner_b_index: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![2_i64, 0, 1], [3]),
                &device,
            ),
            target_delta: Tensor::<B, 2>::from_floats([[0.9], [0.1], [0.0]], &device),
        };

        let loss = tanimoto_ranking_output(latent, batch, 0, 0.5, 0.5, 0.01)
            .loss
            .into_scalar();
        let expected = (0.9 * bce_with_logits(1.0 / 0.5, sigmoid_scalar(0.9 / 0.5))
            + 0.1 * bce_with_logits(-1.0 / 0.5, sigmoid_scalar(0.1 / 0.5)))
            / (0.9 + 0.1);

        assert!(
            (loss - expected).abs() < 1.0e-3,
            "expected gap-weighted pairwise logistic loss near {expected}, got {loss}"
        );
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

    fn sigmoid_scalar(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn bce_with_logits(logit: f32, target: f32) -> f32 {
        logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p()
    }
}
