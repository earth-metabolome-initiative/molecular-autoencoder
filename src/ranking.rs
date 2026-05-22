//! Latent geometry support losses for molecular embeddings.

use burn::{
    prelude::*,
    tensor::{
        Int, TensorData,
        activation::{log_softmax, sigmoid},
    },
};

use crate::model::TanimotoLossKind;

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

/// Raw violation-classifier tensors staged for host-side quality metrics.
///
/// Both tensors are shaped `[batch, class_count]` so the host-side
/// MCC / accuracy helper can iterate them in lockstep after a
/// Transaction round-trip.
#[derive(Debug, Clone)]
pub struct ViolationDiagnostics<B: Backend> {
    /// Per-sample violation-classifier logits, shape `[batch, classes]`.
    pub logits: Tensor<B, 2>,
    /// Per-sample multi-hot violation targets, shape `[batch, classes]`.
    pub targets: Tensor<B, 2>,
    /// Number of samples that contributed to the loss (== batch size).
    pub batch_size: usize,
    /// Number of violation classes (== `VIOLATION_CLASS_COUNT`).
    pub class_count: usize,
}

impl<B: Backend> ViolationDiagnostics<B> {
    /// Empty diagnostics for the zero-loss / pre-sync path. The tensors
    /// have zero-sized shapes so Transactions can register them but the
    /// host-side helper short-circuits on `batch_size == 0`.
    #[must_use]
    pub fn empty(device: &B::Device) -> Self {
        Self {
            logits: Tensor::zeros([0, 0], device),
            targets: Tensor::zeros([0, 0], device),
            batch_size: 0,
            class_count: 0,
        }
    }
}

/// Macro-averaged classification-quality scalars for the violation head.
///
/// Both scalars treat each of the `class_count` independent binary
/// heads equally regardless of class frequency, which matters because
/// the per-class positive rate is heavily skewed (most rows have label
/// `0x00`, and even on negative rows only one or two bits typically
/// co-fire).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ViolationClassificationQuality {
    /// Matthews correlation coefficient averaged over classes (macro-
    /// MCC). Each class is treated as an independent binary classifier
    /// with threshold `logit > 0` (== `sigmoid(logit) > 0.5`), and the
    /// per-class MCC is averaged. Range `[-1, 1]`, `0.0` indicates
    /// random-quality predictions. A class whose confusion-matrix
    /// denominator is zero (e.g. the head predicts the same label
    /// everywhere, or no positives exist in the batch) contributes
    /// `0.0`, matching the convention used by `sklearn` and most
    /// retrieval benchmarks.
    pub mcc_macro: f32,
    /// Fraction of `(sample, class)` cells where the thresholded
    /// prediction matches the target bit, averaged over the entire
    /// `batch * class_count` matrix. Falls back to `0.0` when the
    /// batch is empty.
    pub accuracy: f32,
}

/// Computes [`ViolationClassificationQuality`] from already-materialised
/// host buffers. Returns all-zeros when the batch is empty.
///
/// `logits` and `targets` are expected in `[batch * class_count]`
/// row-major layout; `targets` is the multi-hot float vector
/// (`1.0` for set bits, `0.0` otherwise) emitted by the batcher from
/// each sample's `ViolationLabel` bitset.
///
/// # Panics
///
/// Panics when either slice is shorter than `batch_size * class_count`.
#[must_use]
pub fn violation_classification_quality(
    logits: &[f32],
    targets: &[f32],
    batch_size: usize,
    class_count: usize,
) -> ViolationClassificationQuality {
    if batch_size == 0 || class_count == 0 {
        return ViolationClassificationQuality::default();
    }
    let expected = batch_size * class_count;
    assert!(
        logits.len() >= expected,
        "logits buffer is too short for the declared batch * class_count shape"
    );
    assert!(
        targets.len() >= expected,
        "targets buffer is too short for the declared batch * class_count shape"
    );

    let mut mcc_sum = 0.0_f64;
    let mut correct = 0_u64;
    for class in 0..class_count {
        let mut tp = 0_u64;
        let mut tn = 0_u64;
        let mut fp = 0_u64;
        let mut fn_count = 0_u64;
        for sample in 0..batch_size {
            let offset = sample * class_count + class;
            let predicted = logits[offset] > 0.0;
            let actual = targets[offset] > 0.5;
            match (predicted, actual) {
                (true, true) => tp += 1,
                (false, false) => tn += 1,
                (true, false) => fp += 1,
                (false, true) => fn_count += 1,
            }
        }
        correct += tp + tn;
        mcc_sum += binary_mcc(tp, tn, fp, fn_count);
    }
    ViolationClassificationQuality {
        mcc_macro: (mcc_sum / class_count as f64) as f32,
        accuracy: correct as f32 / expected as f32,
    }
}

/// Matthews correlation coefficient for a single 2x2 confusion matrix.
///
/// Returns `0.0` when any of the four marginal sums (`TP + FP`,
/// `TP + FN`, `TN + FP`, `TN + FN`) is zero — matching the convention
/// `sklearn.metrics.matthews_corrcoef` uses for the all-positive /
/// all-negative degenerate cases.
fn binary_mcc(tp: u64, tn: u64, fp: u64, fn_count: u64) -> f64 {
    let row_pos = tp + fp;
    let row_neg = tn + fn_count;
    let col_pos = tp + fn_count;
    let col_neg = tn + fp;
    if row_pos == 0 || row_neg == 0 || col_pos == 0 || col_neg == 0 {
        return 0.0;
    }
    let tp = tp as f64;
    let tn = tn as f64;
    let fp = fp as f64;
    let fn_ = fn_count as f64;
    let numerator = tp * tn - fp * fn_;
    let denominator = (row_pos as f64 * row_neg as f64 * col_pos as f64 * col_neg as f64).sqrt();
    numerator / denominator
}

/// In-batch ranking objective for counted-fingerprint Tanimoto metric
/// order. Dispatches between gap-weighted softmax cross-entropy and the
/// SoftRank Spearman surrogate via `loss_kind`.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "configuration knobs are passed flat through the model's forward path rather than bundled in a config struct, matching the rest of this module's signatures"
)]
pub fn tanimoto_ranking_output<B: Backend>(
    latent: Tensor<B, 2>,
    batch: TanimotoRankingBatch<B>,
    max_pairs: usize,
    latent_temperature: f64,
    _metric_temperature: f64,
    min_gap: f64,
    loss_kind: TanimotoLossKind,
    softrank_temperature: f64,
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
    let valid_mask: Tensor<B, 1> = valid.clone().reshape([pair_count]);

    let loss = match loss_kind {
        TanimotoLossKind::SoftmaxCe => {
            let log_probs = log_softmax(logits.clone(), 1);
            let target = best_positions.clone().one_hot::<2>(candidate_count).float();
            let cross_entropy = (log_probs * target * -1.0).sum_dim(1);
            (cross_entropy * gap_weights).sum() / gap_weight_sum
        }
        TanimotoLossKind::SoftRankSpearman => {
            // Differentiable Spearman surrogate via Taylor-style soft ranks.
            // Gradient flows through `logits`; the teacher side is detached.
            let logits_soft_rank = soft_rank(logits.clone(), softrank_temperature);
            let teacher_soft_rank = soft_rank(candidate_scores.clone(), softrank_temperature);
            let softrank_per_row =
                pearson_per_row(logits_soft_rank, teacher_soft_rank, candidate_count);
            let softrank_one_minus = softrank_per_row.ones_like() - softrank_per_row;
            (softrank_one_minus * gap_weights).sum() / gap_weight_sum
        }
    };

    TanimotoRankingOutput {
        loss,
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
#[allow(
    clippy::too_many_arguments,
    reason = "configuration knobs are passed flat through the model's forward path rather than bundled in a config struct, matching the rest of this module's signatures"
)]
pub fn weighted_tanimoto_ranking_output<B: Backend>(
    latent: Tensor<B, 2>,
    batch: TanimotoRankingBatch<B>,
    max_pairs: usize,
    latent_temperature: f64,
    metric_temperature: f64,
    min_gap: f64,
    weight: f64,
    loss_kind: TanimotoLossKind,
    softrank_temperature: f64,
) -> TanimotoRankingOutput<B> {
    if weight > 0.0 {
        let mut output = tanimoto_ranking_output(
            latent,
            batch,
            max_pairs,
            latent_temperature,
            metric_temperature,
            min_gap,
            loss_kind,
            softrank_temperature,
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

/// Per-row Pearson correlation between two `[pair_count, candidate_count]`
/// tensors, returned as a `[pair_count, 1]` column.
///
/// Used by the SoftRank Spearman surrogate to score the agreement between
/// the soft ranks of the latent logits and the soft ranks of the teacher
/// scores. The denominator is clamped via `clamp_min(1.0e-12)` so
/// degenerate zero-variance rows contribute `0` rather than `NaN`.
fn pearson_per_row<B: Backend>(
    left: Tensor<B, 2>,
    right: Tensor<B, 2>,
    candidate_count: usize,
) -> Tensor<B, 2> {
    let inv_n = 1.0 / candidate_count as f64;
    let left_mean = left.clone().mean_dim(1);
    let right_mean = right.clone().mean_dim(1);
    let left_centered = left - left_mean;
    let right_centered = right - right_mean;
    let covariance = (left_centered.clone() * right_centered.clone()).sum_dim(1) * inv_n;
    let left_var = left_centered.powf_scalar(2.0).sum_dim(1) * inv_n;
    let right_var = right_centered.powf_scalar(2.0).sum_dim(1) * inv_n;
    let denominator = (left_var * right_var).sqrt().clamp_min(1.0e-12);
    covariance / denominator
}

/// Taylor-style soft-rank operator from
/// [Blondel et al. 2020, "Fast Differentiable Sorting and Ranking"][1]
/// in its pairwise-sigmoid (non-projection) form.
///
/// Given a `[pair_count, candidate_count]` tensor of values and a
/// temperature `tau`, returns a same-shape tensor where each row is the
/// soft rank of the corresponding input row. Smaller `tau` makes the
/// pairwise sigmoid sharper, approaching the hard ranks (plus the
/// constant `0.5` per row from the `j == i` diagonal, which is
/// invariant under Pearson and so cancels out of the downstream loss).
///
/// [1]: https://arxiv.org/abs/2002.08871
fn soft_rank<B: Backend>(values: Tensor<B, 2>, tau: f64) -> Tensor<B, 2> {
    let [pair_count, candidate_count] = values.dims();
    let left = values.clone().unsqueeze_dim::<3>(2);
    let right = values.unsqueeze_dim::<3>(1);
    let pairwise = sigmoid((left - right) / tau.max(1.0e-6));
    pairwise.sum_dim(2).reshape([pair_count, candidate_count])
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;

    type B = burn::backend::NdArray<f32, i64>;

    /// Test shim: calls [`tanimoto_ranking_output`] with the shipping
    /// `SoftmaxCe` baseline and a fixed SoftRank temperature so existing
    /// SoftmaxCe-only tests keep their original signature. SoftRank tests
    /// call [`tanimoto_ranking_output`] directly with the explicit knobs.
    fn softmax_ce_ranking_output(
        latent: Tensor<B, 2>,
        batch: TanimotoRankingBatch<B>,
        max_pairs: usize,
        latent_temperature: f64,
        metric_temperature: f64,
        min_gap: f64,
    ) -> TanimotoRankingOutput<B> {
        tanimoto_ranking_output(
            latent,
            batch,
            max_pairs,
            latent_temperature,
            metric_temperature,
            min_gap,
            TanimotoLossKind::SoftmaxCe,
            0.20,
        )
    }

    /// Test shim for [`weighted_tanimoto_ranking_output`] under the
    /// `SoftmaxCe` baseline.
    fn softmax_ce_weighted_ranking_output(
        latent: Tensor<B, 2>,
        batch: TanimotoRankingBatch<B>,
        max_pairs: usize,
        latent_temperature: f64,
        metric_temperature: f64,
        min_gap: f64,
        weight: f64,
    ) -> TanimotoRankingOutput<B> {
        weighted_tanimoto_ranking_output(
            latent,
            batch,
            max_pairs,
            latent_temperature,
            metric_temperature,
            min_gap,
            weight,
            TanimotoLossKind::SoftmaxCe,
            0.20,
        )
    }

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
            softmax_ce_ranking_output(latent.clone(), preserved_batch, 1, 0.10, 0.10, 0.01);
        let reversed = softmax_ce_ranking_output(latent, reversed_batch, 1, 0.10, 0.10, 0.01);

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

        let output = softmax_ce_ranking_output(latent, batch, 1, 0.10, 0.10, 0.01);

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

        let loss = softmax_ce_ranking_output(latent, batch, 0, 0.5, 0.5, 0.01)
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

        let output = softmax_ce_ranking_output(latent, batch, 1, 0.5, 0.5, 0.01);

        assert_eq!(output.valid_pairs.into_scalar(), 1.0);
    }

    #[test]
    fn ranking_loss_is_zero_for_too_small_batches() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::zeros([2, 2], &device);
        let batch = TanimotoRankingBatch::zeros(2, &device);

        let output = softmax_ce_ranking_output(latent, batch, 0, 0.10, 0.10, 0.01);

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
        let output = softmax_ce_ranking_output(latent, batch, 0, 0.10, 0.10, 0.01);
        assert_eq!(output.loss.into_scalar(), 0.0);
        assert_eq!(output.valid_pairs.into_scalar(), 0.0);
        assert_eq!(output.accuracy.into_scalar(), 0.0);
    }

    #[test]
    fn weighted_ranking_returns_zero_diagnostics_when_weight_is_zero() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::ones([3, 2], &device);
        let batch = TanimotoRankingBatch::zeros(3, &device);
        let output = softmax_ce_weighted_ranking_output(latent, batch, 0, 0.10, 0.10, 0.01, 0.0);
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
        let unweighted =
            softmax_ce_ranking_output(latent.clone(), batch.clone(), 0, 0.5, 0.5, 0.01)
                .loss
                .into_scalar();
        let weighted = softmax_ce_weighted_ranking_output(latent, batch, 0, 0.5, 0.5, 0.01, 2.5)
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

    #[test]
    fn violation_classification_quality_perfect_predictions_score_one() {
        // 2 samples, 4 classes; predictions match targets exactly.
        // Class 0: tgt [1, 0] logits [+5, -5] -> TP=1, TN=1
        // Class 1: tgt [0, 1] logits [-5, +5] -> TP=1, TN=1
        // Class 2: tgt [1, 1] logits [+5, +5] -> TP=2 (no negatives -> MCC=0 by convention)
        // Class 3: tgt [0, 0] logits [-5, -5] -> TN=2 (no positives -> MCC=0)
        // Macro MCC = (1 + 1 + 0 + 0) / 4 = 0.5, accuracy = 1.0
        let logits = vec![5.0, -5.0, 5.0, -5.0, -5.0, 5.0, 5.0, -5.0];
        let targets = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let quality = violation_classification_quality(&logits, &targets, 2, 4);
        assert!((quality.accuracy - 1.0).abs() < 1.0e-6);
        assert!((quality.mcc_macro - 0.5).abs() < 1.0e-5);
    }

    #[test]
    fn violation_classification_quality_inverted_predictions_score_negative_one() {
        // 2 samples, 1 class. Targets [1, 0]; predictions [-5, +5] are
        // completely inverted -> TP=0, TN=0, FP=1, FN=1, MCC=-1.
        let quality = violation_classification_quality(&[-5.0, 5.0], &[1.0, 0.0], 2, 1);
        assert_eq!(quality.accuracy, 0.0);
        assert!((quality.mcc_macro + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn violation_classification_quality_returns_zero_when_class_has_no_variance() {
        // Single class, all-positive targets, model also predicts all
        // positive: TP=4, FN=0, TN=0, FP=0. Denominator factors include
        // (TN+FP)=0 -> MCC=0 by convention.
        let quality = violation_classification_quality(&[5.0; 4], &[1.0_f32; 4], 4, 1);
        assert_eq!(quality.accuracy, 1.0);
        assert_eq!(quality.mcc_macro, 0.0);
    }

    #[test]
    fn violation_classification_quality_handles_empty_inputs() {
        let quality = violation_classification_quality(&[], &[], 0, 0);
        assert_eq!(quality, ViolationClassificationQuality::default());
        let quality = violation_classification_quality(&[1.0_f32; 4], &[1.0_f32; 4], 0, 8);
        assert_eq!(quality, ViolationClassificationQuality::default());
    }

    #[test]
    fn violation_classification_quality_mixed_classes_average_per_class_mcc() {
        // 4 samples, 2 classes.
        // Class 0: targets [1, 0, 1, 0], logits [+5, -5, -5, +5]
        //   -> TP=1, TN=1, FP=1, FN=1
        //   -> num = 1 - 1 = 0, MCC=0
        // Class 1: targets [1, 1, 0, 0], logits [+5, +5, -5, -5] (perfect)
        //   -> TP=2, TN=2, FP=0, FN=0
        //   -> MCC=1
        // Macro = (0 + 1)/2 = 0.5; accuracy = (2+4)/8 = 0.75
        let logits = vec![5.0, 5.0, -5.0, 5.0, -5.0, -5.0, 5.0, -5.0];
        let targets = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let quality = violation_classification_quality(&logits, &targets, 4, 2);
        assert!((quality.mcc_macro - 0.5).abs() < 1.0e-5);
        assert!((quality.accuracy - 0.75).abs() < 1.0e-6);
    }

    #[test]
    fn binary_mcc_matches_textbook_formula() {
        // Wikipedia worked example: TP=90, TN=80, FP=10, FN=20.
        // MCC = (90*80 - 10*20) / sqrt(100 * 110 * 90 * 100)
        //     = (7200 - 200) / sqrt(99_000_000)
        //     = 7000 / 9949.87... = 0.7035...
        let mcc = binary_mcc(90, 80, 10, 20);
        assert!((mcc - 0.703_526).abs() < 1.0e-5, "got {mcc}");
    }

    #[test]
    fn soft_rank_matches_hand_computed_values() {
        // Hand computation for values [1.0, 3.0, 2.0]:
        // soft_rank[i] = sum_j sigmoid((v[i] - v[j]) / tau), so j==i
        // contributes a constant 0.5 per row.
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let values = Tensor::<B, 2>::from_floats([[1.0, 3.0, 2.0]], &device);
        let ranks_tau_one = soft_rank(values.clone(), 1.0)
            .into_data()
            .to_vec::<f32>()
            .expect("f32 data");
        // Hand-precise via sigmoid(-x) = 1/(1+e^x):
        // sigmoid(-2) = 0.119_202_922
        // sigmoid(-1) = 0.268_941_421
        // Row 0 (v=1.0): 0.5 + 0.119_202_922 + 0.268_941_421 = 0.888_144_343
        // Row 1 (v=3.0): 0.880_797_078 + 0.5 + 0.731_058_578 = 2.111_855_656
        // Row 2 (v=2.0): 0.731_058_578 + 0.268_941_421 + 0.5  = 1.500_000_000
        assert!((ranks_tau_one[0] - 0.888_144).abs() < 1.0e-4);
        assert!((ranks_tau_one[1] - 2.111_856).abs() < 1.0e-4);
        assert!((ranks_tau_one[2] - 1.5).abs() < 1.0e-6);

        // Tiny tau sharpens the pairwise sigmoids: each summand collapses
        // to 0 or 1 depending on whether v[i] > v[j], plus the constant
        // 0.5 from j == i. Hard ranks of [1.0, 3.0, 2.0] are
        // [0, 2, 1] (zero-indexed by how many values it strictly beats),
        // so the limiting soft ranks at tau -> 0 are [0.5, 2.5, 1.5].
        let ranks_tau_tiny = soft_rank(values, 0.01)
            .into_data()
            .to_vec::<f32>()
            .expect("f32 data");
        assert!((ranks_tau_tiny[0] - 0.5).abs() < 1.0e-3);
        assert!((ranks_tau_tiny[1] - 2.5).abs() < 1.0e-3);
        assert!((ranks_tau_tiny[2] - 1.5).abs() < 1.0e-6);
    }

    #[test]
    fn pearson_per_row_handles_perfect_and_inverse_correlation() {
        // Three rows: identical -> +1, reversed -> -1, constant -> 0
        // (zero-variance row, denominator clamp routes through to 0).
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let left = Tensor::<B, 2>::from_floats(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            &device,
        );
        let right = Tensor::<B, 2>::from_floats(
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [5.0, 5.0, 5.0]],
            &device,
        );
        let values = pearson_per_row(left, right, 3)
            .into_data()
            .to_vec::<f32>()
            .expect("f32 data");
        assert!((values[0] - 1.0).abs() < 1.0e-5, "got {}", values[0]);
        assert!((values[1] + 1.0).abs() < 1.0e-5, "got {}", values[1]);
        assert!(values[2].abs() < 1.0e-3, "got {}", values[2]);
    }

    #[test]
    fn softrank_loss_is_zero_when_latent_matches_teacher_ranking() {
        // Three anchors, two candidates each. Build a latent whose
        // pairwise cosine ordering on each anchor row matches the
        // teacher's candidate_scores ordering exactly; the SoftRank
        // Spearman surrogate should land near zero (Pearson of two
        // identically-ranked rows is +1, so `1 - Pearson` ~ 0).
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // Latent rows i=0..2 are the anchors; rows i=3..8 are the
        // candidate partners. Each anchor sees [near, far] candidates
        // where "near" has the higher cosine to the anchor.
        let latent = Tensor::<B, 2>::from_floats(
            [
                [1.0, 0.0],   // anchor 0
                [0.0, 1.0],   // anchor 1
                [1.0, 1.0],   // anchor 2
                [1.0, 0.0],   // anchor 0's "near"
                [-1.0, 0.0],  // anchor 0's "far"
                [0.0, 1.0],   // anchor 1's "near"
                [0.0, -1.0],  // anchor 1's "far"
                [1.0, 1.0],   // anchor 2's "near"
                [-1.0, -1.0], // anchor 2's "far"
            ],
            &device,
        );
        let batch = TanimotoRankingBatch {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![3_i64, 4, 5, 6, 7, 8], [3, 2]),
                &device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![0_i64, 0, 0], [3]),
                &device,
            ),
            top2_gap: Tensor::<B, 2>::from_floats([[0.9], [0.9], [0.9]], &device),
            // Teacher: position 0 has higher Tanimoto than position 1
            // on every anchor, matching the latent ordering above.
            candidate_scores: Tensor::<B, 2>::from_floats(
                [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
                &device,
            ),
        };

        // `max_pairs = 3` limits the anchor sweep to the first 3 latent
        // rows, matching the metadata shape; the trailing 6 latent rows
        // are referenced only via candidate_index.
        let loss = tanimoto_ranking_output(
            latent,
            batch,
            3,
            0.10,
            0.10,
            0.01,
            TanimotoLossKind::SoftRankSpearman,
            0.05,
        )
        .loss
        .into_scalar();

        // Soft ranks at finite tau are never exactly hard, but with
        // candidate_count == 2 and a wide separation between the
        // teacher's two scores, `1 - Pearson` collapses to within 1e-4
        // of zero.
        assert!(loss.abs() < 1.0e-4, "expected ~0, got {loss}");
    }

    #[test]
    fn softrank_loss_with_k4_responds_to_ordering_quality() {
        // Larger K (=4) so the soft ranks are no longer trivially hard;
        // verifies that a perfectly-aligned latent gives a near-zero
        // loss and that flipping the latent ordering produces a strictly
        // larger loss. Catches regressions that only manifest when
        // multiple candidates compete inside the pairwise sigmoid cube.
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // One anchor (row 0) + four candidates (rows 1..=4). The four
        // candidate vectors are placed on a circle around the anchor at
        // monotonically increasing angles so their cosine similarities
        // to the anchor are strictly decreasing.
        let latent = Tensor::<B, 2>::from_floats(
            [
                [1.0, 0.0],       // anchor
                [1.0, 0.0],       // candidate 0, cos = 1.0
                [0.866_025, 0.5], // candidate 1, cos ≈ 0.866 (30°)
                [0.5, 0.866_025], // candidate 2, cos ≈ 0.5   (60°)
                [0.0, 1.0],       // candidate 3, cos = 0.0   (90°)
            ],
            &device,
        );
        let aligned = TanimotoRankingBatch {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
                &device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![0_i64], [1]),
                &device,
            ),
            top2_gap: Tensor::<B, 2>::from_data(TensorData::new(vec![0.5_f32], [1, 1]), &device),
            // Teacher ranking matches the latent ordering exactly.
            candidate_scores: Tensor::<B, 2>::from_floats([[0.9, 0.7, 0.4, 0.1]], &device),
        };
        let misaligned = TanimotoRankingBatch {
            candidate_index: Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
                &device,
            ),
            best_candidate_position: Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![0_i64], [1]),
                &device,
            ),
            top2_gap: Tensor::<B, 2>::from_data(TensorData::new(vec![0.5_f32], [1, 1]), &device),
            // Teacher ranking is reversed.
            candidate_scores: Tensor::<B, 2>::from_floats([[0.1, 0.4, 0.7, 0.9]], &device),
        };

        let aligned_loss = tanimoto_ranking_output(
            latent.clone(),
            aligned,
            1,
            0.10,
            0.10,
            0.01,
            TanimotoLossKind::SoftRankSpearman,
            0.20,
        )
        .loss
        .into_scalar();
        let misaligned_loss = tanimoto_ranking_output(
            latent,
            misaligned,
            1,
            0.10,
            0.10,
            0.01,
            TanimotoLossKind::SoftRankSpearman,
            0.20,
        )
        .loss
        .into_scalar();

        // Aligned soft ranks should give Pearson ~ +1, so `1 - Pearson`
        // is small (finite tau leaves a tiny residual that is still well
        // below 0.05 for K = 4).
        assert!(
            aligned_loss < 0.05,
            "aligned SoftRank loss should be near zero, got {aligned_loss}"
        );
        // Reversed ordering should yield Pearson ~ -1, so `1 - Pearson`
        // approaches 2 (the maximum). The aligned case must be strictly
        // smaller; we leave plenty of margin to absorb tau-driven slack.
        assert!(
            misaligned_loss > 1.5,
            "misaligned SoftRank loss should be near 2, got {misaligned_loss}"
        );
        assert!(
            misaligned_loss > aligned_loss,
            "misaligned loss {misaligned_loss} should exceed aligned loss {aligned_loss}"
        );
    }
}
