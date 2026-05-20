//! Per-batch metrics extraction and dataloader profiling.

use std::time::Duration;

use burn::{
    prelude::*,
    tensor::{TensorData, Transaction},
};
use molecular_autoencoder::{DataSplit, MoleculeLossBreakdown};

use crate::{AppResult, dataloader::BatchLoadProfile, invalid_input};

/// Scalar loss components captured from a single training/eval batch.
#[derive(Debug, Clone, Copy)]
pub struct BatchLossMetrics {
    pub loss: f32,
    pub reconstruction: f32,
    pub reconstruction_bce: f32,
    pub descriptors: f32,
    pub tanimoto_ranking: f32,
    pub tanimoto_ranking_accuracy: f32,
    pub count_tanimoto: f32,
    pub binary_tanimoto: f32,
}

/// Returns `"train"` or `"valid"` for the given split.
pub const fn split_label(split: DataSplit) -> &'static str {
    match split {
        DataSplit::Train => "train",
        DataSplit::Validation => "valid",
    }
}

/// Whether to sample expensive per-batch metrics on this step.
pub const fn should_sample_metrics(batch_index: usize, every: usize) -> bool {
    batch_index == 1 || (every != 0 && batch_index.is_multiple_of(every))
}

/// Computes batch loss metrics for the training path.
pub fn loss_metrics<B>(
    loss: Tensor<B, 1>,
    losses: &MoleculeLossBreakdown<B>,
    count_tanimoto: Tensor<B, 1>,
    binary_tanimoto: Tensor<B, 1>,
) -> AppResult<BatchLossMetrics>
where
    B: Backend<FloatElem = f32>,
{
    let data = Transaction::default()
        .register(loss)
        .register(losses.reconstruction.clone())
        .register(losses.reconstruction_bce.clone())
        .register(losses.descriptors.clone())
        .register(losses.tanimoto_ranking.clone())
        .register(losses.tanimoto_ranking_accuracy.clone())
        .register(count_tanimoto)
        .register(binary_tanimoto)
        .try_execute()
        .map_err(|source| invalid_input(format!("failed to read training metrics: {source}")))?;
    if data.len() != 8 {
        return Err(invalid_input(format!(
            "training metrics transaction returned {} tensors instead of 8",
            data.len()
        )));
    }
    Ok(BatchLossMetrics {
        loss: scalar_data(&data[0], "loss")?,
        reconstruction: scalar_data(&data[1], "reconstruction loss")?,
        reconstruction_bce: scalar_data(&data[2], "reconstruction BCE loss")?,
        descriptors: scalar_data(&data[3], "descriptor loss")?,
        tanimoto_ranking: scalar_data(&data[4], "Tanimoto geometry loss")?,
        tanimoto_ranking_accuracy: scalar_data(&data[5], "Tanimoto geometry accuracy")?,
        count_tanimoto: scalar_data(&data[6], "count Tanimoto")?,
        binary_tanimoto: scalar_data(&data[7], "binary Tanimoto")?,
    })
}

/// Computes batch loss metrics plus the count- and binary-Tanimoto scalars
/// for validation.
pub fn validation_metrics<B>(
    loss: Tensor<B, 1>,
    losses: &MoleculeLossBreakdown<B>,
    count_tanimoto: Tensor<B, 1>,
    binary_tanimoto: Tensor<B, 1>,
) -> AppResult<(BatchLossMetrics, f32, f32)>
where
    B: Backend<FloatElem = f32>,
{
    let data = Transaction::default()
        .register(loss)
        .register(losses.reconstruction.clone())
        .register(losses.reconstruction_bce.clone())
        .register(losses.descriptors.clone())
        .register(losses.tanimoto_ranking.clone())
        .register(losses.tanimoto_ranking_accuracy.clone())
        .register(count_tanimoto)
        .register(binary_tanimoto)
        .try_execute()
        .map_err(|source| invalid_input(format!("failed to read validation metrics: {source}")))?;
    if data.len() != 8 {
        return Err(invalid_input(format!(
            "validation metrics transaction returned {} tensors instead of 8",
            data.len()
        )));
    }
    let count_tanimoto = scalar_data(&data[6], "count Tanimoto")?;
    let binary_tanimoto = scalar_data(&data[7], "binary Tanimoto")?;
    Ok((
        BatchLossMetrics {
            loss: scalar_data(&data[0], "loss")?,
            reconstruction: scalar_data(&data[1], "reconstruction loss")?,
            reconstruction_bce: scalar_data(&data[2], "reconstruction BCE loss")?,
            descriptors: scalar_data(&data[3], "descriptor loss")?,
            tanimoto_ranking: scalar_data(&data[4], "Tanimoto geometry loss")?,
            tanimoto_ranking_accuracy: scalar_data(&data[5], "Tanimoto geometry accuracy")?,
            count_tanimoto,
            binary_tanimoto,
        },
        count_tanimoto,
        binary_tanimoto,
    ))
}

fn scalar_data(data: &TensorData, label: &str) -> AppResult<f32> {
    let values = data
        .as_slice::<f32>()
        .map_err(|source| invalid_input(format!("{label} tensor is not f32: {source}")))?;
    values
        .first()
        .copied()
        .ok_or_else(|| invalid_input(format!("{label} tensor did not contain an f32 scalar")))
}

/// Examples-per-second helper, returning 0.0 if elapsed is zero.
pub fn examples_per_second(examples: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        examples as f64 / elapsed.as_secs_f64()
    }
}

/// Milliseconds-per-batch helper, returning 0.0 if `batches` is zero.
pub fn millis_per_batch(duration: Duration, batches: usize) -> f64 {
    if batches == 0 {
        0.0
    } else {
        duration.as_secs_f64() * 1000.0 / batches as f64
    }
}

/// Periodically logs aggregate dataloader timing per phase.
pub struct LoaderProfileReporter {
    phase: &'static str,
    every: usize,
    summary: LoaderProfileSummary,
}

#[derive(Debug, Clone, Copy, Default)]
struct LoaderProfileSummary {
    batches: usize,
    rows: usize,
    wait: Duration,
    profile: BatchLoadProfile,
}

impl LoaderProfileReporter {
    pub fn new(phase: &'static str, every: usize) -> Self {
        Self {
            phase,
            every,
            summary: LoaderProfileSummary::default(),
        }
    }

    pub fn record(&mut self, wait: Duration, rows: usize, profile: BatchLoadProfile) {
        if self.every == 0 {
            return;
        }

        self.summary.batches += 1;
        self.summary.rows += rows;
        self.summary.wait += wait;
        self.summary.profile.shard_read += profile.shard_read;
        self.summary.profile.row_scan += profile.row_scan;
        self.summary.profile.sparse_allocation += profile.sparse_allocation;
        self.summary.profile.sparse_fill += profile.sparse_fill;
        self.summary.profile.tensor_build += profile.tensor_build;
        self.summary.profile.producer_total += profile.producer_total;

        if self.summary.batches >= self.every {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.summary.batches == 0 {
            return;
        }
        let batches = self.summary.batches;
        let profile = self.summary.profile;
        eprintln!(
            "loader_profile phase={} batches={} rows={} wait_ms_per_batch={:.3} producer_ms_per_batch={:.3} shard_read_ms_per_batch={:.3} row_scan_ms_per_batch={:.3} sparse_alloc_ms_per_batch={:.3} sparse_fill_ms_per_batch={:.3} tensor_build_ms_per_batch={:.3}",
            self.phase,
            batches,
            self.summary.rows,
            millis_per_batch(self.summary.wait, batches),
            millis_per_batch(profile.producer_total, batches),
            millis_per_batch(profile.shard_read, batches),
            millis_per_batch(profile.row_scan, batches),
            millis_per_batch(profile.sparse_allocation, batches),
            millis_per_batch(profile.sparse_fill, batches),
            millis_per_batch(profile.tensor_build, batches),
        );
        self.summary = LoaderProfileSummary::default();
    }
}

impl Drop for LoaderProfileReporter {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_label_maps_each_variant_once() {
        assert_eq!(split_label(DataSplit::Train), "train");
        assert_eq!(split_label(DataSplit::Validation), "valid");
    }

    #[test]
    fn should_sample_metrics_always_fires_on_first_batch() {
        assert!(should_sample_metrics(1, 0));
        assert!(should_sample_metrics(1, 100));
    }

    #[test]
    fn should_sample_metrics_respects_period_after_first_batch() {
        assert!(should_sample_metrics(20, 10));
        assert!(should_sample_metrics(30, 10));
        assert!(!should_sample_metrics(15, 10));
        // every=0 disables sampling for non-first batches.
        assert!(!should_sample_metrics(2, 0));
        assert!(!should_sample_metrics(100, 0));
    }

    #[test]
    fn examples_per_second_returns_zero_for_zero_elapsed() {
        assert_eq!(examples_per_second(1000, Duration::ZERO), 0.0);
    }

    #[test]
    fn examples_per_second_divides_by_seconds() {
        let throughput = examples_per_second(500, Duration::from_secs_f64(2.0));
        assert!((throughput - 250.0).abs() < 1.0e-9);
    }

    #[test]
    fn millis_per_batch_returns_zero_when_no_batches() {
        assert_eq!(millis_per_batch(Duration::from_secs(3), 0), 0.0);
    }

    #[test]
    fn millis_per_batch_converts_to_milliseconds() {
        let result = millis_per_batch(Duration::from_secs(4), 8);
        assert!((result - 500.0).abs() < 1.0e-9);
    }

    fn fixture_profile() -> BatchLoadProfile {
        BatchLoadProfile {
            shard_read: Duration::from_millis(1),
            row_scan: Duration::from_millis(2),
            sparse_allocation: Duration::from_millis(3),
            sparse_fill: Duration::from_millis(4),
            tensor_build: Duration::from_millis(5),
            producer_total: Duration::from_millis(15),
        }
    }

    #[test]
    fn loader_profile_reporter_with_every_zero_is_a_noop() {
        let mut reporter = LoaderProfileReporter::new("train", 0);
        reporter.record(Duration::from_millis(10), 32, fixture_profile());
        // record() bailed out early; summary stays empty.
        assert_eq!(reporter.summary.batches, 0);
        assert_eq!(reporter.summary.rows, 0);
        assert!(reporter.summary.wait.is_zero());
    }

    #[test]
    fn loader_profile_reporter_flushes_after_period() {
        let mut reporter = LoaderProfileReporter::new("train", 2);
        reporter.record(Duration::from_millis(1), 8, fixture_profile());
        assert_eq!(reporter.summary.batches, 1);
        // Second call hits `batches >= every` and flushes (eprintln side effect).
        reporter.record(Duration::from_millis(2), 8, fixture_profile());
        // Post-flush the summary is reset.
        assert_eq!(reporter.summary.batches, 0);
        assert_eq!(reporter.summary.rows, 0);
        assert!(reporter.summary.wait.is_zero());
    }

    #[test]
    fn loader_profile_reporter_flush_is_idempotent_when_empty() {
        let mut reporter = LoaderProfileReporter::new("valid", 4);
        reporter.flush();
        assert_eq!(reporter.summary.batches, 0);
    }

    #[test]
    fn loader_profile_reporter_drop_flushes_partial_summary() {
        let mut reporter = LoaderProfileReporter::new("train", 100);
        reporter.record(Duration::from_millis(1), 4, fixture_profile());
        assert_eq!(reporter.summary.batches, 1);
        // Drop runs flush(); just ensure it doesn't panic.
        drop(reporter);
    }
}
