//! Evaluation metrics for counted fingerprints.

use burn::{prelude::*, tensor::Tensor};

use crate::batch::SparseFingerprintBatch;

/// Backend-native per-row count-vector Tanimoto for dense `[batch, width]` count tensors.
///
/// This runs on the active Burn backend, so CUDA training and validation can compute
/// Tanimoto without copying dense fingerprints back to the CPU.
///
/// # Panics
///
/// Panics when the two count tensors have different shapes.
#[must_use]
pub fn batch_count_tanimoto<B: Backend>(
    left_counts: Tensor<B, 2>,
    right_counts: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let left_dims = left_counts.dims();
    assert_eq!(
        left_dims,
        right_counts.dims(),
        "count tensors must share a shape"
    );

    let batch_size = left_dims[0];
    let device = left_counts.device();
    let numerator = left_counts
        .clone()
        .min_pair(right_counts.clone())
        .sum_dim(1);
    let denominator = left_counts.max_pair(right_counts).sum_dim(1);
    let raw = numerator / denominator.clone().clamp_min(1.0e-12);
    let ones = Tensor::<B, 2>::ones([batch_size, 1], &device);
    raw.mask_where(denominator.lower_equal_elem(0.0), ones)
        .squeeze_dim(1)
}

/// Backend-native per-row Tanimoto from reconstructed `log1p(count)` predictions.
#[must_use]
pub fn batch_log_count_reconstruction_tanimoto<B: Backend>(
    target_counts: Tensor<B, 2>,
    predicted_log_counts: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let predicted_counts = (predicted_log_counts.exp() - 1.0).clamp_min(0.0);
    batch_count_tanimoto(target_counts, predicted_counts)
}

/// Backend-native per-row Tanimoto from target and reconstructed `log1p(count)` tensors.
#[must_use]
pub fn batch_log_count_tanimoto<B: Backend>(
    target_log_counts: Tensor<B, 2>,
    predicted_log_counts: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let target_counts = (target_log_counts.exp() - 1.0).clamp_min(0.0);
    batch_log_count_reconstruction_tanimoto(target_counts, predicted_log_counts)
}

/// Backend-native per-row Tanimoto from sparse target and dense reconstructed log-count tensors.
///
/// # Panics
///
/// Panics when the predicted dense tensor shape does not match the sparse target
/// batch size and fingerprint width.
#[must_use]
pub fn batch_sparse_log_count_tanimoto<B: Backend>(
    target: &SparseFingerprintBatch<B>,
    predicted_log_counts: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let predicted_dims = predicted_log_counts.dims();
    assert_eq!(
        predicted_dims[0],
        target.batch_size(),
        "predicted and target batches must have the same row count"
    );
    assert_eq!(
        predicted_dims[1], target.fingerprint_size,
        "predicted reconstruction width must match sparse fingerprint width"
    );

    let batch_size = predicted_dims[0];
    let device = predicted_log_counts.device();
    let predicted_counts = (predicted_log_counts.exp() - 1.0).clamp_min(0.0);
    let predicted_total = predicted_counts.clone().sum_dim(1);
    let predicted_nonzero =
        predicted_counts.gather(1, target.indices.clone()) * target.mask.clone();
    let target_counts =
        (target.log_counts.clone().exp() - 1.0).clamp_min(0.0) * target.mask.clone();
    let intersection = target_counts.clone().min_pair(predicted_nonzero).sum_dim(1);
    let denominator = target_counts.sum_dim(1) + predicted_total - intersection.clone();
    let raw = intersection / denominator.clone().clamp_min(1.0e-12);
    let ones = Tensor::<B, 2>::ones([batch_size, 1], &device);
    raw.mask_where(denominator.lower_equal_elem(0.0), ones)
        .squeeze_dim(1)
}

/// CPU reference count-vector Tanimoto similarity using `sum(min) / sum(max)`.
///
/// # Panics
///
/// Panics when the vectors have different widths.
#[must_use]
pub fn count_tanimoto(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "count vectors must share a width");
    let mut numerator = 0.0_f32;
    let mut denominator = 0.0_f32;
    for (&a, &b) in left.iter().zip(right) {
        numerator += a.min(b);
        denominator += a.max(b);
    }
    if denominator == 0.0 {
        1.0
    } else {
        numerator / denominator
    }
}

/// CPU-side reconstruction metrics for a dense counted fingerprint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CountReconstructionMetrics {
    /// Mean squared error over `log1p(count)`.
    pub log_mse: f32,
    /// Count-vector Tanimoto similarity after clipping negative predictions to zero.
    pub count_tanimoto: f32,
    /// Agreement of nonzero/zero bit presence.
    pub bit_agreement: f32,
    /// Mean absolute error over nonzero target bins.
    pub nonzero_count_mae: f32,
}

impl CountReconstructionMetrics {
    /// Computes metrics from target counts and predicted log-counts.
    ///
    /// # Panics
    ///
    /// Panics when target and prediction slices have different widths.
    #[must_use]
    pub fn from_target_counts_and_predicted_log_counts(
        target_counts: &[f32],
        predicted_log_counts: &[f32],
    ) -> Self {
        assert_eq!(
            target_counts.len(),
            predicted_log_counts.len(),
            "target and predicted vectors must share a width"
        );

        let mut log_mse = 0.0_f32;
        let mut bit_agreement = 0.0_f32;
        let mut nonzero_abs = 0.0_f32;
        let mut nonzero_count = 0.0_f32;
        let mut predicted_counts = Vec::with_capacity(target_counts.len());

        for (&target_count, &predicted_log_count) in target_counts.iter().zip(predicted_log_counts)
        {
            let target_log = target_count.ln_1p();
            let predicted_count = predicted_log_count.exp_m1().max(0.0);
            predicted_counts.push(predicted_count);
            let delta = predicted_log_count - target_log;
            log_mse += delta * delta;
            if (target_count > 0.0) == (predicted_count >= 0.5) {
                bit_agreement += 1.0;
            }
            if target_count > 0.0 {
                nonzero_abs += (predicted_count - target_count).abs();
                nonzero_count += 1.0;
            }
        }

        let width = target_counts.len() as f32;
        Self {
            log_mse: log_mse / width,
            count_tanimoto: count_tanimoto(target_counts, &predicted_counts),
            bit_agreement: bit_agreement / width,
            nonzero_count_mae: if nonzero_count == 0.0 {
                0.0
            } else {
                nonzero_abs / nonzero_count
            },
        }
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    type TestBackend = burn::backend::NdArray<f32, i64>;

    fn tensor(values: Vec<f32>, shape: [usize; 2]) -> Tensor<TestBackend, 2> {
        Tensor::from_data(
            TensorData::new(values, shape),
            &burn::backend::ndarray::NdArrayDevice::default(),
        )
    }

    fn tensor_values(tensor: Tensor<TestBackend, 1>) -> Vec<f32> {
        tensor
            .to_data()
            .as_slice::<f32>()
            .expect("tensor data should be f32")
            .to_vec()
    }

    #[test]
    fn tanimoto_handles_empty_vectors() {
        assert_eq!(count_tanimoto(&[0.0, 0.0], &[0.0, 0.0]), 1.0);
    }

    #[test]
    fn tanimoto_uses_min_over_max_counts() {
        let value = count_tanimoto(&[2.0, 0.0, 1.0], &[1.0, 3.0, 1.0]);
        assert!((value - (1.0 / 3.0)).abs() < 1.0e-6);
    }

    #[test]
    fn batch_tanimoto_matches_cpu_reference_per_row() {
        let left = tensor(vec![2.0, 0.0, 1.0, 0.0, 0.0, 0.0], [2, 3]);
        let right = tensor(vec![1.0, 3.0, 1.0, 0.0, 0.0, 0.0], [2, 3]);
        let values = tensor_values(batch_count_tanimoto(left, right));

        assert!((values[0] - (1.0 / 3.0)).abs() < 1.0e-6);
        assert_eq!(values[1], 1.0);
    }

    #[test]
    fn batch_log_count_tanimoto_clips_negative_reconstructions() {
        let target = tensor(vec![2.0, 0.0, 1.0, 0.0], [1, 4]);
        let predicted_log = tensor(vec![1.0_f32.ln_1p(), 1.0_f32.ln_1p(), 0.0, -1.0], [1, 4]);
        let values = tensor_values(batch_log_count_reconstruction_tanimoto(
            target,
            predicted_log,
        ));

        assert!((values[0] - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn sparse_batch_tanimoto_matches_dense_metric() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let sparse = SparseFingerprintBatch {
            indices: Tensor::<TestBackend, 2, burn::tensor::Int>::from_data([[0_i64, 2]], &device),
            counts: tensor(vec![2.0, 1.0], [1, 2]),
            log_counts: tensor(vec![2.0_f32.ln_1p(), 1.0_f32.ln_1p()], [1, 2]),
            mask: tensor(vec![1.0, 1.0], [1, 2]),
            fingerprint_size: 4,
            max_nnz: 2,
        };
        let target = tensor(vec![2.0, 0.0, 1.0, 0.0], [1, 4]);
        let predicted_log = tensor(vec![1.0_f32.ln_1p(), 1.0_f32.ln_1p(), 0.0, -1.0], [1, 4]);
        let dense = tensor_values(batch_log_count_reconstruction_tanimoto(
            target,
            predicted_log.clone(),
        ));
        let sparse = tensor_values(batch_sparse_log_count_tanimoto(&sparse, predicted_log));

        assert!((dense[0] - sparse[0]).abs() < 1.0e-6);
    }

    #[test]
    fn reconstruction_metrics_are_perfect_for_exact_log_counts() {
        let target = [2.0, 0.0, 1.0, 0.0];
        let predicted = target.map(f32::ln_1p);
        let metrics = CountReconstructionMetrics::from_target_counts_and_predicted_log_counts(
            &target, &predicted,
        );

        assert_eq!(metrics.log_mse, 0.0);
        assert_eq!(metrics.count_tanimoto, 1.0);
        assert_eq!(metrics.bit_agreement, 1.0);
        assert_eq!(metrics.nonzero_count_mae, 0.0);
    }

    #[test]
    fn reconstruction_metrics_capture_imperfect_predictions() {
        let target = [2.0, 0.0, 1.0, 0.0];
        let predicted = [1.0_f32.ln_1p(), 1.0_f32.ln_1p(), 0.0, -1.0];
        let metrics = CountReconstructionMetrics::from_target_counts_and_predicted_log_counts(
            &target, &predicted,
        );
        let expected_log_mse = ((1.0_f32.ln_1p() - 2.0_f32.ln_1p()).powi(2)
            + 1.0_f32.ln_1p().powi(2)
            + 1.0_f32.ln_1p().powi(2)
            + 1.0)
            / 4.0;

        assert!((metrics.log_mse - expected_log_mse).abs() < 1.0e-6);
        assert!((metrics.count_tanimoto - 0.25).abs() < 1.0e-6);
        assert_eq!(metrics.bit_agreement, 0.5);
        assert_eq!(metrics.nonzero_count_mae, 1.0);
    }

    #[test]
    fn reconstruction_metrics_clip_negative_log_counts_to_zero() {
        let metrics = CountReconstructionMetrics::from_target_counts_and_predicted_log_counts(
            &[0.0, 0.0],
            &[-1.0, -2.0],
        );

        assert!(metrics.log_mse > 0.0);
        assert_eq!(metrics.count_tanimoto, 1.0);
        assert_eq!(metrics.bit_agreement, 1.0);
        assert_eq!(metrics.nonzero_count_mae, 0.0);
    }

    #[test]
    #[should_panic(expected = "count vectors must share a width")]
    fn tanimoto_panics_for_mismatched_widths() {
        let _ = count_tanimoto(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "count tensors must share a shape")]
    fn batch_tanimoto_panics_for_mismatched_shapes() {
        let left = tensor(vec![1.0, 2.0], [1, 2]);
        let right = tensor(vec![1.0, 2.0], [2, 1]);

        let _ = batch_count_tanimoto(left, right);
    }

    #[test]
    #[should_panic(expected = "target and predicted vectors must share a width")]
    fn reconstruction_metrics_panic_for_mismatched_widths() {
        let _ = CountReconstructionMetrics::from_target_counts_and_predicted_log_counts(
            &[1.0],
            &[1.0, 2.0],
        );
    }
}
