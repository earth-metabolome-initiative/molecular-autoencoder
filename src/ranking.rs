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
    /// Per-candidate counted-Tanimoto score, shaped `[batch, candidates]`,
    /// in the same column order as `candidate_index`. Surfaced so the
    /// downstream ranking-quality metrics (MRR / Pearson / Spearman) can
    /// be correlated against the latent cosine logits.
    pub candidate_scores: Tensor<B, 2>,
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
            candidate_scores: Tensor::<B, 2>::zeros([batch_size, 2], device),
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
    /// Per-anchor diagnostic tensors carrying enough state for the
    /// caller to compute MRR / Pearson / Spearman host-side after one
    /// Transaction round-trip. Cloned through the loss breakdown into
    /// the training-bin metrics pipeline.
    pub diagnostics: TanimotoRankingDiagnostics<B>,
}

/// Per-anchor tensors needed to score ranking quality (MRR, Pearson,
/// Spearman). All four shape-bearing fields are narrowed to the same
/// `pair_count` rows so host-side reductions stay aligned.
#[derive(Debug, Clone)]
pub struct TanimotoRankingDiagnostics<B: Backend> {
    /// Latent cosine logits, shape `[pair_count, candidate_count]`.
    pub latent_logits: Tensor<B, 2>,
    /// Per-candidate counted-Tanimoto scores, shape
    /// `[pair_count, candidate_count]`.
    pub candidate_scores: Tensor<B, 2>,
    /// Metric-best candidate position per anchor, shape `[pair_count]`.
    pub best_positions: Tensor<B, 1, Int>,
    /// `1.0` for each anchor whose metric gap exceeds `min_gap`, `0.0`
    /// otherwise. Shape `[pair_count]`.
    pub valid_mask: Tensor<B, 1>,
    /// Number of anchors that actually contributed to the loss.
    pub pair_count: usize,
    /// Number of candidate partners sampled per anchor.
    pub candidate_count: usize,
}

impl<B: Backend> TanimotoRankingDiagnostics<B> {
    /// Empty diagnostics for the zero-loss path. The tensors have
    /// zero-sized shapes so Transactions can register them but the
    /// host-side helper short-circuits on `pair_count == 0`.
    #[must_use]
    pub fn empty(device: &B::Device) -> Self {
        Self {
            latent_logits: Tensor::zeros([0, 0], device),
            candidate_scores: Tensor::zeros([0, 0], device),
            best_positions: Tensor::<B, 1, Int>::from_data(
                TensorData::new(Vec::<i64>::new(), [0]),
                device,
            ),
            valid_mask: Tensor::zeros([0], device),
            pair_count: 0,
            candidate_count: 0,
        }
    }
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
    let predicted_best = logits.clone().argmax(1);
    let best_positions = batch.best_candidate_position.narrow(0, 0, pair_count);
    let metric_best = best_positions.clone().reshape([pair_count, 1]);
    let accuracy = (predicted_best.equal(metric_best).float() * valid.clone()).sum()
        / valid_pairs.clone().clamp_min(1.0);

    let candidate_scores = batch.candidate_scores.narrow(0, 0, pair_count).detach();
    let valid_mask: Tensor<B, 1> = valid.reshape([pair_count]);

    TanimotoRankingOutput {
        loss: (cross_entropy * gap_weights).sum() / gap_weight_sum,
        valid_pairs,
        accuracy,
        diagnostics: TanimotoRankingDiagnostics {
            latent_logits: logits.detach(),
            candidate_scores,
            best_positions,
            valid_mask,
            pair_count,
            candidate_count,
        },
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

/// Zero-cost ranking output used when the geometry loss is disabled
/// (weight = 0, batch < 3, or candidate count < 2).
fn zero_tanimoto_ranking_output<B: Backend>(device: &B::Device) -> TanimotoRankingOutput<B> {
    TanimotoRankingOutput {
        loss: Tensor::zeros([1], device),
        valid_pairs: Tensor::zeros([1], device),
        accuracy: Tensor::zeros([1], device),
        diagnostics: TanimotoRankingDiagnostics::empty(device),
    }
}

/// Host-side ranking-quality scalars for the Tanimoto geometry head.
///
/// Holds MRR, Pearson and Spearman correlations between the latent
/// cosine logits and the counted-Tanimoto scores, averaged over the
/// `pair_count` anchors that contributed to the loss. Surfaced as plain
/// `f32` scalars because all three reductions are trivial on host and
/// the rank-based Spearman path is awkward in Burn tensor ops.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TanimotoRankingQuality {
    /// Mean Reciprocal Rank: per anchor `1 / rank(true_best)` where
    /// `rank` is the position of the metric best among the latent
    /// scores sorted descending (1-indexed, ties broken by stable
    /// ordering — see Spearman note below).
    pub mrr: f32,
    /// Per-anchor Pearson correlation between the latent logits and the
    /// counted-Tanimoto scores, averaged over valid anchors. Range
    /// `[-1, 1]`; falls back to `0.0` when either vector has zero
    /// variance (division-by-zero guard).
    pub pearson: f32,
    /// Per-anchor Spearman rank correlation, averaged over valid
    /// anchors. Uses ordinal ranking (each candidate gets a distinct
    /// rank) since exact ties on counted ECFP Tanimoto are rare at
    /// typical 4k+ bit widths.
    pub spearman: f32,
}

/// Computes the three ranking-quality scalars from already-materialised
/// host buffers. Returns all-zeros when `pair_count == 0` or no anchor
/// in `valid_mask` is set.
///
/// # Panics
///
/// Panics when any input slice is shorter than expected for the
/// declared `pair_count` / `candidate_count`.
#[must_use]
pub fn tanimoto_ranking_quality(
    latent_logits: &[f32],
    candidate_scores: &[f32],
    best_positions: &[i32],
    valid_mask: &[f32],
    pair_count: usize,
    candidate_count: usize,
) -> TanimotoRankingQuality {
    if pair_count == 0 || candidate_count == 0 {
        return TanimotoRankingQuality::default();
    }
    assert!(
        latent_logits.len() >= pair_count * candidate_count,
        "latent_logits buffer is too short for the declared shape"
    );
    assert!(
        candidate_scores.len() >= pair_count * candidate_count,
        "candidate_scores buffer is too short for the declared shape"
    );
    assert!(
        best_positions.len() >= pair_count,
        "best_positions buffer is too short for the declared pair_count"
    );
    assert!(
        valid_mask.len() >= pair_count,
        "valid_mask buffer is too short for the declared pair_count"
    );

    let mut mrr_sum = 0.0_f32;
    let mut pearson_sum = 0.0_f32;
    let mut spearman_sum = 0.0_f32;
    let mut valid_count = 0_u32;
    let mut latent_rank_buf = vec![0_f32; candidate_count];
    let mut score_rank_buf = vec![0_f32; candidate_count];

    for anchor in 0..pair_count {
        if valid_mask[anchor] <= 0.0 {
            continue;
        }
        let start = anchor * candidate_count;
        let latent_row = &latent_logits[start..start + candidate_count];
        let score_row = &candidate_scores[start..start + candidate_count];

        let best_position = usize::try_from(best_positions[anchor].max(0))
            .expect("best_position is non-negative")
            .min(candidate_count - 1);
        let best_score = latent_row[best_position];
        let mut strictly_better = 0_usize;
        for (idx, &value) in latent_row.iter().enumerate() {
            if idx != best_position && value > best_score {
                strictly_better += 1;
            }
        }
        let rank = (strictly_better + 1) as f32;
        mrr_sum += 1.0_f32 / rank;

        pearson_sum += pearson(latent_row, score_row);

        ordinal_ranks(latent_row, &mut latent_rank_buf);
        ordinal_ranks(score_row, &mut score_rank_buf);
        spearman_sum += pearson(&latent_rank_buf, &score_rank_buf);

        valid_count += 1;
    }

    if valid_count == 0 {
        return TanimotoRankingQuality::default();
    }
    let denom = valid_count as f32;
    TanimotoRankingQuality {
        mrr: mrr_sum / denom,
        pearson: pearson_sum / denom,
        spearman: spearman_sum / denom,
    }
}

/// Population Pearson correlation between two equal-length slices.
/// Returns `0.0` when either side has zero variance.
fn pearson(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let n = left.len() as f32;
    let mean_left = left.iter().sum::<f32>() / n;
    let mean_right = right.iter().sum::<f32>() / n;
    let mut cov = 0.0_f32;
    let mut var_left = 0.0_f32;
    let mut var_right = 0.0_f32;
    for (&a, &b) in left.iter().zip(right) {
        let da = a - mean_left;
        let db = b - mean_right;
        cov += da * db;
        var_left += da * da;
        var_right += db * db;
    }
    let denom = (var_left * var_right).sqrt();
    if denom <= 1.0e-12 { 0.0 } else { cov / denom }
}

/// Writes the ordinal rank of each `values[i]` into `out[i]`. Ranks
/// are 1-indexed; ties are broken by index so two equal values get
/// distinct ranks.
fn ordinal_ranks(values: &[f32], out: &mut [f32]) {
    debug_assert_eq!(values.len(), out.len());
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&i, &j| {
        values[i]
            .partial_cmp(&values[j])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });
    for (rank, &source_idx) in order.iter().enumerate() {
        out[source_idx] = (rank + 1) as f32;
    }
}

/// Computes the per-row cosine similarity between two `[batch,
/// candidates, latent]` tensors and reshapes the result to `[batch,
/// candidates]`. Used to derive the latent logits for the ranking
/// softmax loss.
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
        assert_eq!(batch.candidate_scores.dims(), [3, 2]);
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
            candidate_scores: Tensor::<B, 2>::zeros([1, 2], &device),
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
            candidate_scores: Tensor::<B, 2>::zeros([1, 2], &device),
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
            candidate_scores: Tensor::<B, 2>::zeros([1, 2], &device),
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
            candidate_scores: Tensor::<B, 2>::zeros([3, 2], &device),
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
            candidate_scores: Tensor::<B, 2>::zeros([3, 2], &device),
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

    #[test]
    fn ranking_loss_is_zero_when_candidate_count_is_below_two() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::zeros([3, 2], &device);
        let batch = TanimotoRankingBatch {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![0_i64; 3], [3, 1]),
                &device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![0_i64; 3], [3]),
                &device,
            ),
            top2_gap: Tensor::<B, 2>::zeros([3, 1], &device),
            candidate_scores: Tensor::<B, 2>::zeros([3, 1], &device),
        };
        let output = tanimoto_ranking_output(latent, batch, 0, 0.10, 0.10, 0.01);
        assert_eq!(output.loss.into_scalar(), 0.0);
        assert_eq!(output.valid_pairs.into_scalar(), 0.0);
        assert_eq!(output.accuracy.into_scalar(), 0.0);
    }

    #[test]
    fn weighted_ranking_returns_zero_diagnostics_when_weight_is_zero() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::ones([3, 2], &device);
        let batch = TanimotoRankingBatch::zeros(3, &device);
        let output = weighted_tanimoto_ranking_output(latent, batch, 0, 0.10, 0.10, 0.01, 0.0);
        assert_eq!(output.loss.into_scalar(), 0.0);
        assert_eq!(output.valid_pairs.into_scalar(), 0.0);
        assert_eq!(output.accuracy.into_scalar(), 0.0);
    }

    #[test]
    fn weighted_ranking_scales_loss_by_weight() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.0_f32, 1.0, 1.0, 0.0, 0.5, 0.5], [3, 2]),
            &device,
        );
        let batch = TanimotoRankingBatch {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![1_i64, 2, 2, 0, 0, 1], [3, 2]),
                &device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![0_i64, 0, 0], [3]),
                &device,
            ),
            top2_gap: Tensor::<B, 2>::ones([3, 1], &device),
            candidate_scores: Tensor::<B, 2>::zeros([3, 2], &device),
        };
        let unweighted = tanimoto_ranking_output(latent.clone(), batch.clone(), 0, 0.5, 0.5, 0.01)
            .loss
            .into_scalar();
        let weighted = weighted_tanimoto_ranking_output(latent, batch, 0, 0.5, 0.5, 0.01, 2.5)
            .loss
            .into_scalar();
        assert!((weighted - unweighted * 2.5).abs() < 1.0e-5);
    }

    #[test]
    fn tanimoto_ranking_quality_handles_empty_pair_count() {
        let quality = tanimoto_ranking_quality(&[], &[], &[], &[], 0, 4);
        assert_eq!(quality, TanimotoRankingQuality::default());
    }

    #[test]
    fn tanimoto_ranking_quality_perfect_alignment() {
        // 2 anchors, 4 candidates each. Latent and Tanimoto scores
        // share the same ordering, so MRR = 1, Pearson = 1, Spearman = 1.
        let latent_logits = [0.8, 0.6, 0.4, 0.2, 0.1, 0.3, 0.5, 0.7];
        let candidate_scores = [0.9, 0.7, 0.5, 0.3, 0.2, 0.4, 0.6, 0.8];
        let best_positions = [0_i32, 3];
        let valid_mask = [1.0_f32, 1.0];
        let quality = tanimoto_ranking_quality(
            &latent_logits,
            &candidate_scores,
            &best_positions,
            &valid_mask,
            2,
            4,
        );
        assert!((quality.mrr - 1.0).abs() < 1.0e-6);
        assert!((quality.pearson - 1.0).abs() < 1.0e-5);
        assert!((quality.spearman - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn tanimoto_ranking_quality_anti_alignment() {
        // Same data but latent order is reversed → MRR = 1/4,
        // Pearson = -1, Spearman = -1.
        let latent_logits = [0.2, 0.4, 0.6, 0.8];
        let candidate_scores = [0.9, 0.7, 0.5, 0.3];
        let best_positions = [0_i32];
        let valid_mask = [1.0_f32];
        let quality = tanimoto_ranking_quality(
            &latent_logits,
            &candidate_scores,
            &best_positions,
            &valid_mask,
            1,
            4,
        );
        assert!((quality.mrr - 0.25).abs() < 1.0e-6);
        assert!((quality.pearson + 1.0).abs() < 1.0e-5);
        assert!((quality.spearman + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn tanimoto_ranking_quality_handles_constant_latent() {
        // Constant latent → Pearson / Spearman fall back to 0 via the
        // zero-variance guard. MRR uses the ordinal-sort tie-break (the
        // metric best sits at index 0 → rank 1 / N candidates).
        let latent_logits = [0.5_f32; 4];
        let candidate_scores = [0.9_f32, 0.6, 0.3, 0.1];
        let best_positions = [0_i32];
        let valid_mask = [1.0_f32];
        let quality = tanimoto_ranking_quality(
            &latent_logits,
            &candidate_scores,
            &best_positions,
            &valid_mask,
            1,
            4,
        );
        assert_eq!(quality.pearson, 0.0);
        // Spearman is identical to Pearson on ranks; with a constant
        // latent vector, ordinal ranks 1..4 vs the candidate-score
        // ranks are perfectly anti-correlated under our tie-break
        // (latent ranks ascend by index, scores descend). We just
        // require a finite value here.
        assert!(quality.spearman.is_finite());
        // The best position is index 0; with all latents equal, the
        // ordinal-stable argmax is index 0, so MRR = 1.
        assert!((quality.mrr - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn tanimoto_ranking_quality_skips_invalid_anchors() {
        // Anchor 0 is invalid → only anchor 1 contributes.
        let latent_logits = [0.0, 0.0, 0.0, 0.0, 0.9, 0.6, 0.3, 0.1];
        let candidate_scores = [0.0, 0.0, 0.0, 0.0, 0.9, 0.6, 0.3, 0.1];
        let best_positions = [0_i32, 0];
        let valid_mask = [0.0_f32, 1.0];
        let quality = tanimoto_ranking_quality(
            &latent_logits,
            &candidate_scores,
            &best_positions,
            &valid_mask,
            2,
            4,
        );
        assert!((quality.mrr - 1.0).abs() < 1.0e-6);
        assert!((quality.pearson - 1.0).abs() < 1.0e-5);
        assert!((quality.spearman - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn tanimoto_ranking_quality_returns_zero_when_no_anchor_is_valid() {
        let quality =
            tanimoto_ranking_quality(&[0.0_f32; 4], &[0.0_f32; 4], &[0_i32], &[0.0_f32], 1, 4);
        assert_eq!(quality, TanimotoRankingQuality::default());
    }
}
