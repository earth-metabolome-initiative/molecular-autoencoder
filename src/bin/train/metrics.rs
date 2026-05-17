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
