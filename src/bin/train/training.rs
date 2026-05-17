//! Training and evaluation per-epoch loops.

use std::time::{Duration, Instant};

use burn::{
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use molecular_autoencoder::{
    DataSplit, MoleculeAutoencoder, MoleculeAutoencoderBatch, MoleculeAutoencoderBatcher,
    TanimotoRankingRuntimeConfig, batch_sparse_log_count_binary_tanimoto,
    batch_sparse_log_count_tanimoto,
};

use crate::{
    AppResult,
    checkpoint::{CachedShardInfo, CheckpointState},
    cli::Args,
    dataloader::{BatchIterationContext, for_each_batch},
    invalid_input,
    metrics::{BatchLossMetrics, loss_metrics, should_sample_metrics, validation_metrics},
    ranking::{TanimotoMetricBackend, attach_tanimoto_ranking, tanimoto_ranking_seed},
    reporter::TrainingReporter,
};

/// Aggregate metrics collected over a single training epoch.
#[derive(Debug, Clone, Copy, Default)]
pub struct EpochSummary {
    pub batches: usize,
    pub loss_batches: usize,
    pub examples: usize,
    pub loss_sum: f64,
    pub data_time: Duration,
    pub step_time: Duration,
}

impl EpochSummary {
    fn record(
        &mut self,
        examples: usize,
        loss: Option<f32>,
        data_time: Duration,
        step_time: Duration,
    ) {
        self.batches += 1;
        self.examples += examples;
        if let Some(loss) = loss {
            self.loss_batches += 1;
            self.loss_sum += f64::from(loss);
        }
        self.data_time += data_time;
        self.step_time += step_time;
    }

    pub fn mean_loss(self) -> f32 {
        if self.loss_batches == 0 {
            0.0
        } else {
            (self.loss_sum / self.loss_batches as f64) as f32
        }
    }
}

/// Aggregate metrics collected over a single validation pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvaluationSummary {
    pub batches: usize,
    pub loss_batches: usize,
    pub examples: usize,
    pub loss_sum: f64,
    pub count_tanimoto_sum: f64,
    pub binary_tanimoto_sum: f64,
    pub data_time: Duration,
    pub step_time: Duration,
}

impl EvaluationSummary {
    fn record(
        &mut self,
        examples: usize,
        loss: Option<f32>,
        count_tanimoto: f32,
        binary_tanimoto: f32,
        data_time: Duration,
        step_time: Duration,
    ) {
        self.batches += 1;
        self.examples += examples;
        if let Some(loss) = loss {
            self.loss_batches += 1;
            self.loss_sum += f64::from(loss);
        }
        self.count_tanimoto_sum += f64::from(count_tanimoto);
        self.binary_tanimoto_sum += f64::from(binary_tanimoto);
        self.data_time += data_time;
        self.step_time += step_time;
    }

    pub fn mean_loss(self) -> f32 {
        if self.loss_batches == 0 {
            0.0
        } else {
            (self.loss_sum / self.loss_batches as f64) as f32
        }
    }

    pub fn mean_count_tanimoto(self) -> f32 {
        if self.batches == 0 {
            0.0
        } else {
            (self.count_tanimoto_sum / self.batches as f64) as f32
        }
    }

    pub fn mean_binary_tanimoto(self) -> f32 {
        if self.batches == 0 {
            0.0
        } else {
            (self.binary_tanimoto_sum / self.batches as f64) as f32
        }
    }
}

/// Per-epoch training context owned by the run loop.
pub struct TrainEpochContext<'a, B: Backend> {
    pub shards: &'a [CachedShardInfo],
    pub batcher: MoleculeAutoencoderBatcher,
    pub device: &'a B::Device,
    pub tanimoto_ranking: TanimotoRankingRuntimeConfig,
    pub args: &'a Args,
    pub loader_profile_every: usize,
    pub validation_per_mille: u16,
    pub state: &'a mut CheckpointState,
    pub reporter: &'a mut TrainingReporter,
    pub epoch: usize,
}

/// Per-validation context owned by the run loop.
pub struct EvaluationContext<'a, B: Backend> {
    pub shards: &'a [CachedShardInfo],
    pub batcher: MoleculeAutoencoderBatcher,
    pub device: &'a B::Device,
    pub batch_size: usize,
    pub max_batches: Option<usize>,
    pub loader_workers: usize,
    pub device_prefetch_batches: usize,
    pub loader_profile_every: usize,
    pub tanimoto_ranking: TanimotoRankingRuntimeConfig,
    pub validation_per_mille: u16,
    pub reporter: &'a mut TrainingReporter,
    pub epoch: usize,
    pub epoch_total: usize,
}

/// Runs the training loop for a single epoch and returns the updated model
/// alongside the epoch summary.
pub fn train_epoch<B, O>(
    model: MoleculeAutoencoder<B>,
    optimizer: &mut O,
    context: TrainEpochContext<'_, B>,
) -> AppResult<(MoleculeAutoencoder<B>, EpochSummary)>
where
    B: AutodiffBackend<FloatElem = f32> + TanimotoMetricBackend,
    B::Device: Clone + Send + Sync,
    MoleculeAutoencoderBatch<B>: Send,
    O: Optimizer<MoleculeAutoencoder<B>, B>,
{
    let TrainEpochContext {
        shards,
        batcher,
        device,
        tanimoto_ranking,
        args,
        loader_profile_every,
        validation_per_mille,
        state,
        reporter,
        epoch,
    } = context;
    let mut model = Some(model);
    let mut summary = EpochSummary::default();

    for_each_batch(
        BatchIterationContext {
            shards,
            split: DataSplit::Train,
            batcher,
            device,
            batch_size: args.batch_size,
            max_batches: args.max_train_batches,
            loader_workers: args.loader_workers,
            device_prefetch_batches: args.device_prefetch_batches,
            loader_profile_every,
            validation_per_mille,
        },
        |mut batch, batch_size, data_time| {
            attach_tanimoto_ranking(
                &mut batch,
                tanimoto_ranking,
                tanimoto_ranking_seed(epoch, state.global_step),
            );
            let current = model
                .take()
                .ok_or_else(|| invalid_input("training model was not available"))?;
            let step_start = Instant::now();
            let next_step = state.global_step + 1;
            let sample_metrics = should_sample_metrics(next_step, args.metric_every);
            let MoleculeAutoencoderBatch {
                fingerprints,
                descriptor_targets,
                tanimoto_ranking,
                ..
            } = batch;
            let fingerprints_for_metrics = sample_metrics.then(|| fingerprints.clone());
            let output = current.forward(&fingerprints);
            let reconstructed_for_metrics =
                sample_metrics.then(|| output.reconstructed_log_counts.clone());
            let losses = current.loss_from_output(
                output,
                fingerprints,
                descriptor_targets,
                tanimoto_ranking,
            );
            let loss = losses.total();
            let batch_metrics = match (fingerprints_for_metrics, reconstructed_for_metrics) {
                (Some(fingerprints), Some(reconstructed)) => {
                    let count_tanimoto =
                        batch_sparse_log_count_tanimoto(&fingerprints, reconstructed.clone())
                            .mean();
                    let binary_tanimoto =
                        batch_sparse_log_count_binary_tanimoto(&fingerprints, reconstructed).mean();
                    Some(loss_metrics(
                        loss.clone(),
                        &losses,
                        count_tanimoto,
                        binary_tanimoto,
                    )?)
                }
                _ => None,
            };
            if batch_metrics.is_some_and(|metrics: BatchLossMetrics| !metrics.loss.is_finite()) {
                return Err(invalid_input(format!(
                    "non-finite training loss at global step {}",
                    state.global_step
                )));
            }
            let grads = GradientsParams::from_grads(loss.backward(), &current);
            model = Some(optimizer.step(args.learning_rate, current, grads));
            state.global_step = next_step;
            let step_time = step_start.elapsed();
            summary.record(
                batch_size,
                batch_metrics.map(|metrics| metrics.loss),
                data_time,
                step_time,
            );
            Ok(reporter.train_batch(
                epoch,
                args.epochs,
                state.global_step,
                batch_size,
                batch_metrics,
                data_time,
                step_time,
            ))
        },
    )?;

    model
        .map(|model| (model, summary))
        .ok_or_else(|| invalid_input("training model was consumed unexpectedly"))
}

/// Runs the validation pass; returns `None` when no validation batches were
/// produced.
pub fn evaluate<B>(
    model: &MoleculeAutoencoder<B>,
    context: EvaluationContext<'_, B>,
) -> AppResult<Option<EvaluationSummary>>
where
    B: Backend<FloatElem = f32> + TanimotoMetricBackend,
    B::Device: Clone + Send + Sync,
    MoleculeAutoencoderBatch<B>: Send,
{
    let EvaluationContext {
        shards,
        batcher,
        device,
        batch_size,
        max_batches,
        loader_workers,
        device_prefetch_batches,
        loader_profile_every,
        tanimoto_ranking,
        validation_per_mille,
        reporter,
        epoch,
        epoch_total,
    } = context;
    let mut summary = EvaluationSummary::default();
    for_each_batch(
        BatchIterationContext {
            shards,
            split: DataSplit::Validation,
            batcher,
            device,
            batch_size,
            max_batches,
            loader_workers,
            device_prefetch_batches,
            loader_profile_every,
            validation_per_mille,
        },
        |mut batch, rows, data_time| {
            attach_tanimoto_ranking(
                &mut batch,
                tanimoto_ranking,
                tanimoto_ranking_seed(epoch, summary.batches),
            );
            let step_start = Instant::now();
            let MoleculeAutoencoderBatch {
                fingerprints,
                descriptor_targets,
                tanimoto_ranking,
                ..
            } = batch;
            let output = model.forward(&fingerprints);
            let reconstructed = output.reconstructed_log_counts.clone();
            let fingerprints_for_metrics = fingerprints.clone();
            let losses =
                model.loss_from_output(output, fingerprints, descriptor_targets, tanimoto_ranking);
            let total_loss = losses.total();
            let count_tanimoto_tensor =
                batch_sparse_log_count_tanimoto(&fingerprints_for_metrics, reconstructed.clone())
                    .mean();
            let binary_tanimoto_tensor =
                batch_sparse_log_count_binary_tanimoto(&fingerprints_for_metrics, reconstructed)
                    .mean();
            let (batch_metrics, count_tanimoto, binary_tanimoto) = validation_metrics(
                total_loss.clone(),
                &losses,
                count_tanimoto_tensor,
                binary_tanimoto_tensor,
            )?;
            if !batch_metrics.loss.is_finite() {
                return Err(invalid_input("non-finite validation loss"));
            }
            let step_time = step_start.elapsed();
            summary.record(
                rows,
                Some(batch_metrics.loss),
                count_tanimoto,
                binary_tanimoto,
                data_time,
                step_time,
            );
            Ok(reporter.valid_batch(
                epoch,
                epoch_total,
                summary.batches,
                rows,
                Some(batch_metrics),
                count_tanimoto,
                binary_tanimoto,
            ))
        },
    )?;

    if summary.batches == 0 {
        Ok(None)
    } else {
        Ok(Some(summary))
    }
}
