//! Training and evaluation per-epoch loops.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender, TryRecvError},
    },
    time::{Duration, Instant},
};

use burn::tensor::activation::softplus;
use burn::{
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
#[cfg(feature = "multi-gpu")]
use burn_collective::{PeerId, ReduceOperation};
use molecular_autoencoder::{
    DataSplit, MoleculeAutoencoder, MoleculeAutoencoderBatch, MoleculeAutoencoderBatcher,
    ReconstructionLossKind, TanimotoRankingRuntimeConfig, batch_sparse_log_count_binary_tanimoto,
    batch_sparse_log_count_tanimoto,
};

use crate::{
    AppResult,
    checkpoint::{AdaptiveSchedulerState, CachedShardInfo, CheckpointState},
    cli::Args,
    dataloader::{BatchControl, BatchIterationContext, for_each_batch},
    invalid_input,
    metrics::{
        BatchLossMetrics, PolaritySplitTensors, loss_metrics, should_sample_metrics,
        validation_metrics,
    },
    ranking::{TanimotoMetricBackend, attach_tanimoto_ranking, tanimoto_ranking_seed},
    reporter::TrainingReporter,
};

/// Aggregate metrics collected over a single training epoch.
#[derive(Debug, Clone, Copy, Default)]
pub struct EpochSummary {
    /// Number of batches processed.
    pub batches: usize,
    /// Number of batches that contributed to `loss_sum`.
    pub loss_batches: usize,
    /// Total examples seen.
    pub examples: usize,
    /// Sum of per-batch loss scalars (only over `loss_batches`).
    pub loss_sum: f64,
    /// Cumulative time spent waiting on the dataloader.
    pub data_time: Duration,
    /// Cumulative time spent in the forward + backward + optimizer step.
    pub step_time: Duration,
}

impl EpochSummary {
    /// Adds one batch's contribution to the running totals.
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

    /// Mean per-batch loss over the batches that produced a sampled
    /// loss scalar. Returns `0.0` when no batch was sampled.
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
    /// Number of validation batches processed.
    pub batches: usize,
    /// Number of batches that contributed a loss scalar.
    pub loss_batches: usize,
    /// Total validation examples seen.
    pub examples: usize,
    /// Sum of per-batch validation loss scalars.
    pub loss_sum: f64,
    /// Sum of counted-Tanimoto scores over every positive
    /// (reconstruction) row seen across all batches.
    pub count_tanimoto_reconstruction_sum: f64,
    /// Sum of counted-Tanimoto scores over every negative
    /// (correction / denoising) row seen across all batches.
    pub count_tanimoto_correction_sum: f64,
    /// Sum of binary-Tanimoto scores over every positive (reconstruction)
    /// row seen across all batches.
    pub binary_tanimoto_reconstruction_sum: f64,
    /// Sum of binary-Tanimoto scores over every negative (correction)
    /// row seen across all batches.
    pub binary_tanimoto_correction_sum: f64,
    /// Sum of reconstruction-head per-row scores over every positive
    /// (reconstruction) row seen across all batches.
    pub recon_loss_reconstruction_sum: f64,
    /// Sum of reconstruction-head per-row scores over every negative
    /// (correction) row seen across all batches.
    pub recon_loss_correction_sum: f64,
    /// Sum of bit-presence BCE per-row scores (already multiplied by
    /// `bce_weight`) over every positive (reconstruction) row seen
    /// across all batches.
    pub bce_reconstruction_sum: f64,
    /// Sum of bit-presence BCE per-row scores (already multiplied by
    /// `bce_weight`) over every negative (correction) row seen across
    /// all batches.
    pub bce_correction_sum: f64,
    /// Total positive (reconstruction) rows seen across all batches.
    pub reconstruction_rows: usize,
    /// Total negative (correction) rows seen across all batches.
    pub correction_rows: usize,
    /// Cumulative dataloader wait time across validation batches.
    pub data_time: Duration,
    /// Cumulative forward-only step time across validation batches.
    pub step_time: Duration,
}

impl EvaluationSummary {
    /// Adds one validation batch's contribution to the running totals.
    fn record(
        &mut self,
        examples: usize,
        loss: Option<f32>,
        metrics: &BatchLossMetrics,
        data_time: Duration,
        step_time: Duration,
    ) {
        self.batches += 1;
        self.examples += examples;
        if let Some(loss) = loss {
            self.loss_batches += 1;
            self.loss_sum += f64::from(loss);
        }
        if let Some(mean) = metrics.count_tanimoto_reconstruction {
            self.count_tanimoto_reconstruction_sum +=
                f64::from(mean) * metrics.reconstruction_rows as f64;
        }
        if let Some(mean) = metrics.binary_tanimoto_reconstruction {
            self.binary_tanimoto_reconstruction_sum +=
                f64::from(mean) * metrics.reconstruction_rows as f64;
        }
        if let Some(mean) = metrics.count_tanimoto_correction {
            self.count_tanimoto_correction_sum += f64::from(mean) * metrics.correction_rows as f64;
        }
        if let Some(mean) = metrics.binary_tanimoto_correction {
            self.binary_tanimoto_correction_sum += f64::from(mean) * metrics.correction_rows as f64;
        }
        if let Some(mean) = metrics.recon_loss_reconstruction {
            self.recon_loss_reconstruction_sum +=
                f64::from(mean) * metrics.reconstruction_rows as f64;
        }
        if let Some(mean) = metrics.recon_loss_correction {
            self.recon_loss_correction_sum += f64::from(mean) * metrics.correction_rows as f64;
        }
        if let Some(mean) = metrics.bce_reconstruction {
            self.bce_reconstruction_sum += f64::from(mean) * metrics.reconstruction_rows as f64;
        }
        if let Some(mean) = metrics.bce_correction {
            self.bce_correction_sum += f64::from(mean) * metrics.correction_rows as f64;
        }
        self.reconstruction_rows += metrics.reconstruction_rows;
        self.correction_rows += metrics.correction_rows;
        self.data_time += data_time;
        self.step_time += step_time;
    }

    /// Mean validation loss over sampled batches.
    pub fn mean_loss(self) -> f32 {
        if self.loss_batches == 0 {
            0.0
        } else {
            (self.loss_sum / self.loss_batches as f64) as f32
        }
    }

    /// Row-weighted mean counted-Tanimoto over positive (reconstruction)
    /// rows. Returns `None` when no positive rows were seen.
    pub fn mean_count_tanimoto_reconstruction(self) -> Option<f32> {
        weighted_mean(
            self.count_tanimoto_reconstruction_sum,
            self.reconstruction_rows,
        )
    }

    /// Row-weighted mean counted-Tanimoto over negative (correction)
    /// rows. Returns `None` when no negative rows were seen.
    pub fn mean_count_tanimoto_correction(self) -> Option<f32> {
        weighted_mean(self.count_tanimoto_correction_sum, self.correction_rows)
    }

    /// Row-weighted mean binary-Tanimoto over positive (reconstruction)
    /// rows. Returns `None` when no positive rows were seen.
    pub fn mean_binary_tanimoto_reconstruction(self) -> Option<f32> {
        weighted_mean(
            self.binary_tanimoto_reconstruction_sum,
            self.reconstruction_rows,
        )
    }

    /// Row-weighted mean binary-Tanimoto over negative (correction)
    /// rows. Returns `None` when no negative rows were seen.
    pub fn mean_binary_tanimoto_correction(self) -> Option<f32> {
        weighted_mean(self.binary_tanimoto_correction_sum, self.correction_rows)
    }

    /// Row-weighted mean reconstruction-head loss over positive
    /// (reconstruction) rows. Returns `None` when no positive rows
    /// were seen.
    pub fn mean_recon_loss_reconstruction(self) -> Option<f32> {
        weighted_mean(self.recon_loss_reconstruction_sum, self.reconstruction_rows)
    }

    /// Row-weighted mean reconstruction-head loss over negative
    /// (correction) rows. Returns `None` when no negative rows were
    /// seen.
    pub fn mean_recon_loss_correction(self) -> Option<f32> {
        weighted_mean(self.recon_loss_correction_sum, self.correction_rows)
    }

    /// Row-weighted mean bit-presence BCE (already multiplied by
    /// `bce_weight`) over positive (reconstruction) rows. Returns
    /// `None` when no positive rows were seen.
    pub fn mean_bce_reconstruction(self) -> Option<f32> {
        weighted_mean(self.bce_reconstruction_sum, self.reconstruction_rows)
    }

    /// Row-weighted mean bit-presence BCE (already multiplied by
    /// `bce_weight`) over negative (correction) rows. Returns `None`
    /// when no negative rows were seen.
    pub fn mean_bce_correction(self) -> Option<f32> {
        weighted_mean(self.bce_correction_sum, self.correction_rows)
    }
}

/// `Some(sum / rows)` when `rows > 0`, `None` otherwise.
fn weighted_mean(sum: f64, rows: usize) -> Option<f32> {
    if rows == 0 {
        None
    } else {
        Some((sum / rows as f64) as f32)
    }
}

/// Payload routed from a worker peer to the lead peer's training reporter.
#[derive(Debug, Clone)]
pub struct TrainBatchPayload {
    /// Epoch index (1-based) the batch belongs to.
    pub epoch: usize,
    /// Total epochs in the training run.
    pub epoch_total: usize,
    /// Monotonic global step counter after this batch.
    pub global_step: usize,
    /// Number of examples in the batch.
    pub batch_size: usize,
    /// Sampled loss / accuracy / quality scalars, when this batch was
    /// chosen by `should_sample_metrics`.
    pub metrics: Option<BatchLossMetrics>,
    /// Time spent waiting on the dataloader.
    pub data_time: Duration,
    /// Forward + backward + optimizer step time.
    pub step_time: Duration,
}

/// Payload routed from a worker peer to the lead peer's validation reporter.
#[derive(Debug, Clone)]
pub struct ValidBatchPayload {
    /// Epoch index this validation batch belongs to.
    pub epoch: usize,
    /// Total epochs in the training run.
    pub epoch_total: usize,
    /// Validation batch index within the epoch.
    pub batch_index: usize,
    /// Number of validation examples in the batch.
    pub batch_size: usize,
    /// Sampled loss / accuracy / quality scalars, when sampled.
    pub metrics: Option<BatchLossMetrics>,
}

/// Inter-peer message envelope (worker → lead).
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "multi-gpu"), allow(dead_code))]
pub enum BatchUpdate {
    /// One training batch completed on a worker peer.
    Train(TrainBatchPayload),
    /// One validation batch completed on a worker peer.
    Valid(ValidBatchPayload),
    /// Sentinel from one worker that finished its training epoch.
    TrainEpochComplete,
    /// Sentinel from one worker that finished its validation pass.
    ValidEpochComplete,
}

/// Routes batch-level metrics either to the local reporter (lead peer, peer 0)
/// or to an mpsc channel drained by peer 0 (worker peers).
pub enum BatchSink<'a> {
    /// Peer 0 routes metrics straight into the in-process reporter.
    Lead {
        /// Borrowed mutable reporter used to render TUI / indicatif
        /// progress for both this peer's batches and any drained
        /// worker-peer updates.
        reporter: &'a mut TrainingReporter,
        /// Shared atomic flag toggled when the renderer requests an
        /// early stop (e.g. Ctrl-C).
        stop_flag: Arc<AtomicBool>,
        /// Channel drained between own batches to surface worker peers'
        /// metrics on the lead reporter. `None` in single-GPU runs.
        receiver: Option<&'a Receiver<BatchUpdate>>,
    },
    /// Non-lead peer forwards its batch updates over an mpsc channel
    /// owned by peer 0.
    #[cfg_attr(not(feature = "multi-gpu"), allow(dead_code))]
    Worker {
        /// Sender half drained by the lead peer.
        tx: Sender<BatchUpdate>,
        /// Shared stop flag toggled when the lead reporter requests an
        /// early stop.
        stop_flag: Arc<AtomicBool>,
    },
}

impl BatchSink<'_> {
    /// Routes a freshly-completed training batch to either the local
    /// reporter (lead peer) or the worker→lead channel.
    fn train_batch(&mut self, payload: TrainBatchPayload) -> BatchControl {
        match self {
            Self::Lead {
                reporter,
                stop_flag,
                receiver,
            } => {
                let control = forward_train(reporter, stop_flag, payload);
                if let Some(rx) = receiver {
                    drain_worker_updates(reporter, stop_flag, rx);
                }
                control
            }
            Self::Worker { tx, stop_flag } => {
                let _ = tx.send(BatchUpdate::Train(payload));
                if stop_flag.load(Ordering::Acquire) {
                    BatchControl::Stop
                } else {
                    BatchControl::Continue
                }
            }
        }
    }

    /// Validation counterpart of [`Self::train_batch`].
    fn valid_batch(&mut self, payload: ValidBatchPayload) -> BatchControl {
        match self {
            Self::Lead {
                reporter,
                stop_flag,
                receiver,
            } => {
                let control = forward_valid(reporter, stop_flag, payload);
                if let Some(rx) = receiver {
                    drain_worker_updates(reporter, stop_flag, rx);
                }
                control
            }
            Self::Worker { tx, stop_flag } => {
                let _ = tx.send(BatchUpdate::Valid(payload));
                if stop_flag.load(Ordering::Acquire) {
                    BatchControl::Stop
                } else {
                    BatchControl::Continue
                }
            }
        }
    }
}

/// Renders one training batch through the reporter and propagates any
/// requested stop into the shared atomic flag.
fn forward_train(
    reporter: &mut TrainingReporter,
    stop_flag: &AtomicBool,
    payload: TrainBatchPayload,
) -> BatchControl {
    let control = reporter.train_batch(
        payload.epoch,
        payload.epoch_total,
        payload.global_step,
        payload.batch_size,
        payload.metrics,
        payload.data_time,
        payload.step_time,
    );
    if matches!(control, BatchControl::Stop) {
        stop_flag.store(true, Ordering::Release);
    }
    control
}

/// Renders one validation batch through the reporter and propagates
/// any requested stop into the shared atomic flag.
fn forward_valid(
    reporter: &mut TrainingReporter,
    stop_flag: &AtomicBool,
    payload: ValidBatchPayload,
) -> BatchControl {
    let control = reporter.valid_batch(
        payload.epoch,
        payload.epoch_total,
        payload.batch_index,
        payload.batch_size,
        payload.metrics,
    );
    if matches!(control, BatchControl::Stop) {
        stop_flag.store(true, Ordering::Release);
    }
    control
}

/// Non-blocking drain: forwards every pending worker-side `BatchUpdate` to the
/// lead reporter, propagating any `BatchControl::Stop` into the shared flag.
fn drain_worker_updates(
    reporter: &mut TrainingReporter,
    stop_flag: &AtomicBool,
    receiver: &Receiver<BatchUpdate>,
) {
    loop {
        match receiver.try_recv() {
            Ok(BatchUpdate::Train(payload)) => {
                let _ = forward_train(reporter, stop_flag, payload);
            }
            Ok(BatchUpdate::Valid(payload)) => {
                let _ = forward_valid(reporter, stop_flag, payload);
            }
            Ok(BatchUpdate::TrainEpochComplete | BatchUpdate::ValidEpochComplete) => {
                // Sentinels are observed by the orchestrator on the rendezvous path; the
                // reporter does not need to see them.
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
        }
    }
}

/// Per-epoch training context owned by the run loop or per-worker DDP loop.
pub struct TrainEpochContext<'a, B: Backend> {
    /// Per-peer slice of the cached shards iterator.
    pub shards: &'a [CachedShardInfo],
    /// Pre-built batcher pinned to the model's fingerprint shape.
    pub batcher: MoleculeAutoencoderBatcher,
    /// Backend device this peer is training on.
    pub device: &'a B::Device,
    /// Runtime view of the Tanimoto geometry hyperparameters.
    pub tanimoto_ranking: TanimotoRankingRuntimeConfig,
    /// Parsed CLI arguments.
    pub args: &'a Args,
    /// Loader-profile flush interval (`0` disables profiling).
    pub loader_profile_every: usize,
    /// Per-mille validation split fraction.
    pub validation_per_mille: u16,
    /// Mutable checkpoint state updated as batches complete.
    pub state: &'a mut CheckpointState,
    /// Batch-update sink: in-process reporter on peer 0, mpsc sender
    /// on worker peers.
    pub sink: BatchSink<'a>,
    /// Epoch index (1-based).
    pub epoch: usize,
    /// Local peer identifier (0 = lead).
    pub peer_id: u32,
    /// Total DDP peer count.
    pub world_size: u32,
    /// Optional adaptive reconstruction-loss scheduler. When `Some`,
    /// the train loop calls `update` on every metric-sampled batch and
    /// mutates the model's active loss kind when a hysteresis threshold
    /// is crossed.
    pub adaptive_loss: Option<&'a mut AdaptiveReconstructionLossController>,
}

/// Per-validation context owned by the run loop or per-worker DDP loop.
pub struct EvaluationContext<'a, B: Backend> {
    /// Per-peer slice of the cached shards iterator.
    pub shards: &'a [CachedShardInfo],
    /// Batcher pinned to the model's fingerprint shape.
    pub batcher: MoleculeAutoencoderBatcher,
    /// Backend device for the validation pass.
    pub device: &'a B::Device,
    /// Validation batch size.
    pub batch_size: usize,
    /// Optional cap on the number of validation batches per epoch.
    pub max_batches: Option<usize>,
    /// Number of host-side dataloader worker threads.
    pub loader_workers: usize,
    /// Device prefetch depth (`0` keeps everything on the host
    /// thread).
    pub device_prefetch_batches: usize,
    /// Loader-profile flush interval (`0` disables profiling).
    pub loader_profile_every: usize,
    /// Runtime view of the Tanimoto geometry hyperparameters.
    pub tanimoto_ranking: TanimotoRankingRuntimeConfig,
    /// Per-mille validation split fraction.
    pub validation_per_mille: u16,
    /// Batch-update sink.
    pub sink: BatchSink<'a>,
    /// Epoch index (1-based).
    pub epoch: usize,
    /// Total epochs in the training run.
    pub epoch_total: usize,
    /// Local peer identifier.
    pub peer_id: u32,
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
        mut sink,
        epoch,
        peer_id,
        world_size,
        adaptive_loss,
    } = context;
    let mut adaptive_loss = adaptive_loss;
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
                tanimoto_ranking_seed(epoch, state.global_step, peer_id),
            );
            let current = model
                .take()
                .ok_or_else(|| invalid_input("training model was not available"))?;
            let step_start = Instant::now();
            let next_step = state.global_step + 1;
            let sample_metrics = should_sample_metrics(next_step, args.metric_every);
            let MoleculeAutoencoderBatch {
                fingerprints,
                target_fingerprints,
                descriptor_targets,
                violation_targets,
                tanimoto_ranking,
                ..
            } = batch;
            // Reconstruction-quality metrics compare the decoder output
            // to the decoder target (which equals the encoder input on
            // positive rows and the source positive's fingerprint on
            // negative rows under the denoising framing). The Tanimoto
            // means are split by polarity so positive (reconstruction)
            // and negative (correction) rows are reported separately.
            let fingerprints_for_metrics = sample_metrics.then(|| target_fingerprints.clone());
            let polarity_for_metrics = sample_metrics.then(|| polarity_mask(&violation_targets));
            let output = current.forward(&fingerprints);
            let reconstructed_for_metrics =
                sample_metrics.then(|| output.reconstructed_log_counts.clone());
            let losses = current.loss_from_output(
                output,
                fingerprints,
                target_fingerprints,
                descriptor_targets,
                tanimoto_ranking,
                violation_targets,
            );
            let loss = losses.total();
            let batch_metrics = match (
                fingerprints_for_metrics,
                reconstructed_for_metrics,
                polarity_for_metrics,
            ) {
                (Some(fingerprints), Some(reconstructed), Some(correction_mask)) => {
                    let metric_log_counts =
                        adapt_to_log1p_count(reconstructed, current.reconstruction_loss_kind());
                    let count_per_row =
                        batch_sparse_log_count_tanimoto(&fingerprints, metric_log_counts.clone());
                    let binary_per_row =
                        batch_sparse_log_count_binary_tanimoto(&fingerprints, metric_log_counts);
                    let split = polarity_split(
                        correction_mask,
                        count_per_row,
                        binary_per_row,
                        losses.reconstruction_per_row.clone(),
                        losses.reconstruction_bce_per_row.clone(),
                        losses.huber_balance.clone(),
                    );
                    Some(loss_metrics(loss.clone(), &losses, split)?)
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
            let grads = all_reduce_mean::<B>(grads, peer_id, world_size)?;
            model = Some(optimizer.step(args.learning_rate, current, grads));
            state.global_step = next_step;
            // Adaptive reconstruction-loss scheduler: feed the latest
            // active-vs-inactive Huber ratio into the controller, then
            // mutate the freshly-stepped model in place if a transition
            // fires. Only runs on metric-sampled batches because that
            // is the only path where balance diagnostics get computed.
            if let (Some(controller), Some(metrics)) =
                (adaptive_loss.as_deref_mut(), batch_metrics.as_ref())
            {
                let inactive = metrics.inactive_huber_mean.max(1.0e-6);
                let ratio = metrics.active_huber_mean / inactive;
                if let Some(new_kind) = controller.update(ratio, epoch)
                    && let Some(active_model) = model.as_mut()
                    && active_model.reconstruction_loss_kind() != new_kind
                {
                    println!(
                        "reconstruction_loss_kind_swap epoch={epoch} step={} from={:?} to={:?} ema={:.3}",
                        state.global_step,
                        active_model.reconstruction_loss_kind(),
                        new_kind,
                        controller.ema(),
                    );
                    active_model.set_reconstruction_loss_kind(new_kind);
                }
                state.adaptive_scheduler = Some(controller.snapshot());
            }
            let step_time = step_start.elapsed();
            summary.record(
                batch_size,
                batch_metrics.map(|metrics| metrics.loss),
                data_time,
                step_time,
            );
            Ok(sink.train_batch(TrainBatchPayload {
                epoch,
                epoch_total: args.epochs,
                global_step: state.global_step,
                batch_size,
                metrics: batch_metrics,
                data_time,
                step_time,
            }))
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
        mut sink,
        epoch,
        epoch_total,
        peer_id,
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
                tanimoto_ranking_seed(epoch, summary.batches, peer_id),
            );
            let step_start = Instant::now();
            let MoleculeAutoencoderBatch {
                fingerprints,
                target_fingerprints,
                descriptor_targets,
                violation_targets,
                tanimoto_ranking,
                ..
            } = batch;
            let output = model.forward(&fingerprints);
            let reconstructed = output.reconstructed_log_counts.clone();
            // Reconstruction quality is measured against the decoder
            // target, not the encoder input (denoising framing). The
            // Tanimoto means are split by polarity so positives
            // (reconstruction) and negatives (correction) are reported
            // separately.
            let fingerprints_for_metrics = target_fingerprints.clone();
            let correction_mask = polarity_mask(&violation_targets);
            let losses = model.loss_from_output(
                output,
                fingerprints,
                target_fingerprints,
                descriptor_targets,
                tanimoto_ranking,
                violation_targets,
            );
            let total_loss = losses.total();
            let metric_log_counts =
                adapt_to_log1p_count(reconstructed, model.reconstruction_loss_kind());
            let count_per_row = batch_sparse_log_count_tanimoto(
                &fingerprints_for_metrics,
                metric_log_counts.clone(),
            );
            let binary_per_row = batch_sparse_log_count_binary_tanimoto(
                &fingerprints_for_metrics,
                metric_log_counts,
            );
            let split = polarity_split(
                correction_mask,
                count_per_row,
                binary_per_row,
                losses.reconstruction_per_row.clone(),
                losses.reconstruction_bce_per_row.clone(),
                losses.huber_balance.clone(),
            );
            let batch_metrics = validation_metrics(total_loss.clone(), &losses, split)?;
            if !batch_metrics.loss.is_finite() {
                return Err(invalid_input("non-finite validation loss"));
            }
            let step_time = step_start.elapsed();
            summary.record(
                rows,
                Some(batch_metrics.loss),
                &batch_metrics,
                data_time,
                step_time,
            );
            Ok(sink.valid_batch(ValidBatchPayload {
                epoch,
                epoch_total,
                batch_index: summary.batches,
                batch_size: rows,
                metrics: Some(batch_metrics),
            }))
        },
    )?;

    if summary.batches == 0 {
        Ok(None)
    } else {
        Ok(Some(summary))
    }
}

/// Mean-reduces parameter gradients across all DDP peers. When `world_size <=
/// 1` or the `multi-gpu` feature is disabled, this is a no-op identity.
#[cfg(feature = "multi-gpu")]
fn all_reduce_mean<B>(
    grads: GradientsParams,
    peer_id: u32,
    world_size: u32,
) -> AppResult<GradientsParams>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    if world_size <= 1 {
        return Ok(grads);
    }
    grads
        .all_reduce::<B::InnerBackend>(PeerId::from(peer_id), ReduceOperation::Mean)
        .map_err(|err| invalid_input(format!("gradient all-reduce failed: {err:?}")))
}

/// Single-peer no-op variant of [`all_reduce_mean`] used when the
/// `multi-gpu` feature is disabled. Preserves the signature so the hot
/// path stays cfg-agnostic.
#[cfg(not(feature = "multi-gpu"))]
#[allow(
    clippy::extra_unused_type_parameters,
    clippy::unnecessary_wraps,
    reason = "the multi-gpu cfg uses `B::InnerBackend` and returns Result; the cfg-off variant must keep the same signature so the hot path stays cfg-agnostic"
)]
fn all_reduce_mean<B>(
    grads: GradientsParams,
    _peer_id: u32,
    _world_size: u32,
) -> AppResult<GradientsParams>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    Ok(grads)
}

/// Returns a `[batch]` mask that is `1.0` on negative (correction)
/// rows and `0.0` on positive (reconstruction) rows.
///
/// Polarity is derived from `violation_targets`, the multi-hot
/// `[batch, 8]` matrix expanded from each sample's
/// `finge_rs::ViolationLabel` bitset: a row is a negative iff any
/// violation class bit is set.
fn polarity_mask<B: Backend>(violation_targets: &Tensor<B, 2>) -> Tensor<B, 1> {
    violation_targets
        .clone()
        .sum_dim(1)
        .squeeze_dim(1)
        .greater_elem(0.0)
        .float()
}

/// Reinterprets the model's raw decoder output as `log1p(count)` so it
/// can be fed straight into the Tanimoto metric helpers, which assume
/// that shape (they undo via `exp(z) - 1`).
///
/// Under [`ReconstructionLossKind::Huber`] the model is already
/// emitting `log1p(count)`, so the output passes through unchanged.
/// Under [`ReconstructionLossKind::Poisson`] the output is the
/// Poisson `log(rate)`; `softplus` converts it to `log1p(exp(z))`,
/// which satisfies `exp(softplus(z)) − 1 == exp(z)` and so feeds the
/// existing Tanimoto helpers their expected count decoding.
fn adapt_to_log1p_count<B: Backend>(
    reconstructed: Tensor<B, 2>,
    kind: ReconstructionLossKind,
) -> Tensor<B, 2> {
    match kind {
        ReconstructionLossKind::Huber | ReconstructionLossKind::HuberSymmetric => reconstructed,
        ReconstructionLossKind::Poisson => softplus(reconstructed, 1.0),
    }
}

/// Splits the per-row Tanimoto, Huber, and BCE vectors into
/// polarity-specific sums and row counts, ready to be flushed through
/// the metrics [`Transaction`].
///
/// `correction_mask` carries `1.0` on negative rows and `0.0` on
/// positive rows (the convention used by [`polarity_mask`]).
#[allow(
    clippy::too_many_arguments,
    reason = "polarity_split bundles every per-row metric the train binary tracks; flattening to a struct would just shuffle the same arity across the call site"
)]
fn polarity_split<B: Backend>(
    correction_mask: Tensor<B, 1>,
    count_per_row: Tensor<B, 1>,
    binary_per_row: Tensor<B, 1>,
    huber_per_row: Tensor<B, 1>,
    bce_per_row: Tensor<B, 1>,
    huber_balance: molecular_autoencoder::HuberBalanceDiagnostics<B>,
) -> PolaritySplitTensors<B> {
    let device = correction_mask.device();
    let reconstruction_mask = correction_mask.clone() * -1.0 + 1.0;
    let reconstruction_count = reconstruction_mask.clone().sum();
    let correction_count = correction_mask.clone().sum();
    let count_reconstruction_sum = (count_per_row.clone() * reconstruction_mask.clone()).sum();
    let count_correction_sum = (count_per_row * correction_mask.clone()).sum();
    let binary_reconstruction_sum = (binary_per_row.clone() * reconstruction_mask.clone()).sum();
    let binary_correction_sum = (binary_per_row * correction_mask.clone()).sum();
    let recon_loss_reconstruction_sum = (huber_per_row.clone() * reconstruction_mask.clone()).sum();
    let recon_loss_correction_sum = (huber_per_row * correction_mask.clone()).sum();
    let bce_reconstruction_sum = (bce_per_row.clone() * reconstruction_mask).sum();
    let bce_correction_sum = (bce_per_row * correction_mask).sum();
    // Active/inactive Huber sums are batch-wide (no polarity split):
    // the diagnostic measures cell-type imbalance, not row-polarity
    // imbalance. When balance diagnostics are disabled the per-row
    // tensors are zero-shape so we substitute scalar zeros.
    let (
        active_huber_total_sum,
        inactive_huber_total_sum,
        active_cell_total_count,
        inactive_cell_total_count,
    ) = if huber_balance.active_huber_sum_per_row.dims()[0] == 0 {
        (
            Tensor::<B, 1>::zeros([1], &device),
            Tensor::<B, 1>::zeros([1], &device),
            Tensor::<B, 1>::zeros([1], &device),
            Tensor::<B, 1>::zeros([1], &device),
        )
    } else {
        (
            huber_balance.active_huber_sum_per_row.sum(),
            huber_balance.inactive_huber_sum_per_row.sum(),
            huber_balance.active_count_per_row.sum(),
            huber_balance.inactive_count_per_row.sum(),
        )
    };
    PolaritySplitTensors {
        count_reconstruction_sum,
        count_correction_sum,
        binary_reconstruction_sum,
        binary_correction_sum,
        recon_loss_reconstruction_sum,
        recon_loss_correction_sum,
        bce_reconstruction_sum,
        bce_correction_sum,
        reconstruction_count,
        correction_count,
        active_huber_total_sum,
        inactive_huber_total_sum,
        active_cell_total_count,
        inactive_cell_total_count,
    }
}

/// Adaptive reconstruction-loss-kind scheduler driven by an EMA of the
/// active-bin / inactive-bin symmetric-Huber ratio.
///
/// The controller is created by [`AdaptiveReconstructionLossController::from_args`]
/// (returning `None` when `--reconstruction-loss-adaptive` is unset) and
/// owned by the run loop. Each metric-sampled batch feeds a fresh
/// `active / inactive` ratio into [`Self::update`]; the controller
/// updates its EMA, applies hysteresis against the two configured
/// thresholds, and returns `Some(new_kind)` when a transition should
/// fire. The fixed-epoch warmup configured via
/// `--reconstruction-loss-warmup-epochs` is honoured as a safety floor:
/// while `epoch <= warmup_floor_epochs` the controller is forced to the
/// warmup kind regardless of the EMA.
#[derive(Debug, Clone)]
pub struct AdaptiveReconstructionLossController {
    /// EMA of `active_huber_mean / inactive_huber_mean` across
    /// metric-sampled batches.
    ema: f32,
    /// Reconstruction-loss kind the controller currently wants the
    /// model to use.
    current_kind: molecular_autoencoder::ReconstructionLossKind,
    /// Per-update EMA decay (closer to `1.0` = more smoothing).
    decay: f32,
    /// EMA below this value while in the Warmup state triggers a
    /// transition to Main.
    switch_to_main_at: f32,
    /// EMA above this value while in the Main state triggers a
    /// transition back to Warmup.
    switch_to_warmup_at: f32,
    /// Kind to use during the warmup (= the `HuberSymmetric` arm of the
    /// curriculum).
    warmup_kind: molecular_autoencoder::ReconstructionLossKind,
    /// Kind to use after the warmup (= the `Huber + BCE` arm).
    main_kind: molecular_autoencoder::ReconstructionLossKind,
    /// Number of epochs during which the controller is pinned to
    /// `warmup_kind` regardless of the EMA. `0` disables the floor.
    warmup_floor_epochs: usize,
}

impl AdaptiveReconstructionLossController {
    /// Builds a controller from the parsed CLI args, or returns `None`
    /// when adaptive scheduling is disabled or the warmup kind is unset
    /// (which `validate_adaptive_loss` already rejects upstream).
    #[must_use]
    pub fn from_args(args: &Args) -> Option<Self> {
        if !args.reconstruction_loss_adaptive {
            return None;
        }
        let warmup_kind: molecular_autoencoder::ReconstructionLossKind =
            args.reconstruction_loss_warmup?.into();
        let main_kind: molecular_autoencoder::ReconstructionLossKind =
            args.reconstruction_loss.into();
        let midpoint = ((args.reconstruction_loss_adaptive_switch_to_main
            + args.reconstruction_loss_adaptive_switch_to_warmup)
            * 0.5) as f32;
        Some(Self {
            ema: midpoint,
            current_kind: warmup_kind,
            decay: args.reconstruction_loss_adaptive_ema_decay as f32,
            switch_to_main_at: args.reconstruction_loss_adaptive_switch_to_main as f32,
            switch_to_warmup_at: args.reconstruction_loss_adaptive_switch_to_warmup as f32,
            warmup_kind,
            main_kind,
            warmup_floor_epochs: args.reconstruction_loss_warmup_epochs,
        })
    }

    /// Restores `ema` + `current_kind` from a persisted snapshot. When
    /// `persisted` is `None` the controller keeps its bootstrapped EMA
    /// (the midpoint between the two thresholds) and `current_kind`.
    pub fn seed_from_checkpoint(&mut self, persisted: Option<AdaptiveSchedulerState>) {
        if let Some(state) = persisted {
            self.ema = state.ema;
            self.current_kind = state.current_kind;
        }
    }

    /// Currently-active reconstruction-loss kind according to the
    /// controller.
    #[must_use]
    pub const fn current_kind(&self) -> molecular_autoencoder::ReconstructionLossKind {
        self.current_kind
    }

    /// Current EMA value (exposed mainly for diagnostics / tests).
    #[must_use]
    pub const fn ema(&self) -> f32 {
        self.ema
    }

    /// Updates the EMA with one fresh `active_huber / inactive_huber`
    /// ratio sample and returns `Some(new_kind)` when a hysteresis
    /// threshold has been crossed.
    ///
    /// While `epoch <= warmup_floor_epochs` the controller is forced
    /// to the warmup kind regardless of EMA (returns `Some(warmup_kind)`
    /// only on the first call where it actually changes the state).
    pub fn update(
        &mut self,
        sample: f32,
        epoch: usize,
    ) -> Option<molecular_autoencoder::ReconstructionLossKind> {
        if sample.is_finite() {
            self.ema = self.decay * self.ema + (1.0 - self.decay) * sample;
        }
        if epoch <= self.warmup_floor_epochs {
            if self.current_kind != self.warmup_kind {
                self.current_kind = self.warmup_kind;
                return Some(self.warmup_kind);
            }
            return None;
        }
        if self.current_kind == self.warmup_kind && self.ema < self.switch_to_main_at {
            self.current_kind = self.main_kind;
            return Some(self.main_kind);
        }
        if self.current_kind == self.main_kind && self.ema > self.switch_to_warmup_at {
            self.current_kind = self.warmup_kind;
            return Some(self.warmup_kind);
        }
        None
    }

    /// Captures the controller's current state for persistence.
    #[must_use]
    pub const fn snapshot(&self) -> AdaptiveSchedulerState {
        AdaptiveSchedulerState {
            ema: self.ema,
            current_kind: self.current_kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::data::dataloader::batcher::Batcher;
    use burn::module::AutodiffModule;
    use burn::optim::AdamConfig;
    use clap::Parser;
    use molecular_autoencoder::{
        DataSplit, FingerprintTargets, MoleculeAutoencoderConfig, SparseMoleculeShard,
        features::REGRESSION_TARGET_WIDTH,
    };
    use std::sync::mpsc;

    use crate::cli::Args;
    use crate::reporter::TrainingReporter;

    type TestBackend = Autodiff<burn::backend::NdArray<f32, i64>>;

    const FINGERPRINT_WIDTH: usize = 256;
    const HIDDEN_WIDTH: usize = 16;
    const LATENT_WIDTH: usize = 8;

    fn fixture_args() -> Args {
        let mut args = Args::try_parse_from(["train", "ignored", "ignored"]).expect("parse");
        args.batch_size = 2;
        args.loader_workers = 0;
        args.device_prefetch_batches = 0;
        args.max_train_batches = Some(2);
        args.metric_every = 1;
        // Disable Tanimoto ranking — it needs a CUDA kernel.
        args.tanimoto_ranking_weight = 0.0;
        args.descriptor_weight = 0.05;
        args.epochs = 1;
        args
    }

    fn write_fixture_shard(dir: &std::path::Path) -> CachedShardInfo {
        let mut shard = SparseMoleculeShard::new(FINGERPRINT_WIDTH, REGRESSION_TARGET_WIDTH);
        for (cid, indices, split) in [
            (1_u64, vec![3_u16, 11], DataSplit::Train),
            (2, vec![5_u16, 17], DataSplit::Train),
            (3, vec![1_u16, 23], DataSplit::Train),
            (4, vec![7_u16, 29], DataSplit::Validation),
            (5, vec![9_u16, 31], DataSplit::Validation),
        ] {
            let fingerprint = FingerprintTargets::new(
                indices.clone(),
                vec![1_u16; indices.len()],
                FINGERPRINT_WIDTH,
            )
            .expect("valid fingerprint");
            shard
                .push(
                    cid,
                    &fingerprint,
                    &[0.0; REGRESSION_TARGET_WIDTH],
                    split,
                    0,
                    None,
                )
                .expect("push row");
        }
        let path = dir.join("shard-00000.maeshard");
        shard.write_to_path(&path).expect("write shard");
        CachedShardInfo {
            path,
            row_count: shard.len(),
        }
    }

    fn build_model_and_optimizer() -> (
        MoleculeAutoencoder<TestBackend>,
        impl Optimizer<MoleculeAutoencoder<TestBackend>, TestBackend>,
    ) {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = MoleculeAutoencoderConfig::symmetric(
            FINGERPRINT_WIDTH,
            LATENT_WIDTH,
            vec![HIDDEN_WIDTH],
        );
        let model = config.init::<TestBackend>(&device);
        let optimizer = AdamConfig::new().init::<TestBackend, MoleculeAutoencoder<TestBackend>>();
        (model, optimizer)
    }

    fn fixture_reporter() -> TrainingReporter {
        TrainingReporter::new(5, 400, 2, Some(2), Some(2), None)
    }

    #[test]
    fn train_epoch_completes_one_pass_and_updates_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shard = write_fixture_shard(dir.path());
        let shards = vec![shard];
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(FINGERPRINT_WIDTH, REGRESSION_TARGET_WIDTH);
        let args = fixture_args();
        let (model, mut optimizer) = build_model_and_optimizer();
        let mut state = CheckpointState {
            completed_epoch: 0,
            global_step: 0,
            best_validation_loss: None,
            adaptive_scheduler: None,
        };
        let mut reporter = fixture_reporter();
        let stop_flag = Arc::new(AtomicBool::new(false));

        let (next_model, summary) = train_epoch(
            model,
            &mut optimizer,
            TrainEpochContext {
                shards: &shards,
                batcher,
                device: &device,
                tanimoto_ranking: TanimotoRankingRuntimeConfig::default(),
                args: &args,
                loader_profile_every: 0,
                validation_per_mille: 400,
                state: &mut state,
                sink: BatchSink::Lead {
                    reporter: &mut reporter,
                    stop_flag: stop_flag.clone(),
                    receiver: None,
                },
                epoch: 1,
                peer_id: 0,
                world_size: 1,
                adaptive_loss: None,
            },
        )
        .expect("train epoch");

        assert!(summary.batches >= 1, "should run at least one batch");
        assert_eq!(state.global_step, summary.batches);
        assert!(summary.mean_loss().is_finite());
        // Smoke: model is returned and can serve another forward call.
        let inference = next_model.valid();
        let batch = batcher.batch(
            vec![molecular_autoencoder::MoleculeAutoencoderSample {
                cid: 99,
                fingerprint_indices: vec![3, 11],
                fingerprint_counts: vec![1, 1],
                descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                ..Default::default()
            }],
            &device,
        );
        let out = inference.forward(&batch.fingerprints);
        let latent_data = out.latent.to_data();
        let latent = latent_data.as_slice::<f32>().expect("latent f32");
        assert!(latent.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn train_epoch_with_loader_workers_exercises_threaded_dataloader() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shard = write_fixture_shard(dir.path());
        let shards = vec![shard];
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(FINGERPRINT_WIDTH, REGRESSION_TARGET_WIDTH);
        let mut args = fixture_args();
        // Drive the multi-threaded dataloader + device prefetch paths.
        args.loader_workers = 1;
        args.device_prefetch_batches = 1;
        args.loader_profile_every = 1;
        let (model, mut optimizer) = build_model_and_optimizer();
        let mut state = CheckpointState {
            completed_epoch: 0,
            global_step: 0,
            best_validation_loss: None,
            adaptive_scheduler: None,
        };
        let mut reporter = fixture_reporter();
        let stop_flag = Arc::new(AtomicBool::new(false));

        let (_next_model, summary) = train_epoch(
            model,
            &mut optimizer,
            TrainEpochContext {
                shards: &shards,
                batcher,
                device: &device,
                tanimoto_ranking: TanimotoRankingRuntimeConfig::default(),
                args: &args,
                loader_profile_every: 1,
                validation_per_mille: 400,
                state: &mut state,
                sink: BatchSink::Lead {
                    reporter: &mut reporter,
                    stop_flag: stop_flag.clone(),
                    receiver: None,
                },
                epoch: 1,
                peer_id: 0,
                world_size: 1,
                adaptive_loss: None,
            },
        )
        .expect("threaded train_epoch");
        assert!(summary.batches >= 1);
    }

    #[test]
    fn evaluate_returns_summary_for_validation_split() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shard = write_fixture_shard(dir.path());
        let shards = vec![shard];
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(FINGERPRINT_WIDTH, REGRESSION_TARGET_WIDTH);
        let args = fixture_args();
        let (model, _optimizer) = build_model_and_optimizer();
        let mut reporter = fixture_reporter();
        let stop_flag = Arc::new(AtomicBool::new(false));

        let inference = model.valid();
        let summary = evaluate(
            &inference,
            EvaluationContext {
                shards: &shards,
                batcher,
                device: &device,
                batch_size: args.batch_size,
                max_batches: Some(args.max_valid_batches),
                loader_workers: 0,
                device_prefetch_batches: 0,
                loader_profile_every: 0,
                tanimoto_ranking: TanimotoRankingRuntimeConfig::default(),
                validation_per_mille: 400,
                sink: BatchSink::Lead {
                    reporter: &mut reporter,
                    stop_flag: stop_flag.clone(),
                    receiver: None,
                },
                epoch: 1,
                epoch_total: 1,
                peer_id: 0,
            },
        )
        .expect("evaluate")
        .expect("validation produced at least one batch");

        assert!(summary.batches >= 1);
        assert!(summary.mean_loss().is_finite());
        // The default fixture shard only has positive rows so the
        // reconstruction polarity should always be populated and the
        // correction polarity should always be absent.
        let count_reconstruction = summary
            .mean_count_tanimoto_reconstruction()
            .expect("reconstruction Tanimoto should be set on positive-only fixture");
        let binary_reconstruction = summary
            .mean_binary_tanimoto_reconstruction()
            .expect("binary reconstruction Tanimoto should be set on positive-only fixture");
        assert!((0.0..=1.0).contains(&count_reconstruction));
        assert!((0.0..=1.0).contains(&binary_reconstruction));
        assert!(summary.mean_count_tanimoto_correction().is_none());
        assert!(summary.mean_binary_tanimoto_correction().is_none());
    }

    #[test]
    fn epoch_summary_record_and_mean_loss_aggregate_correctly() {
        let mut summary = EpochSummary::default();
        summary.record(
            8,
            Some(0.4),
            Duration::from_millis(2),
            Duration::from_millis(3),
        );
        summary.record(4, None, Duration::from_millis(1), Duration::from_millis(2));
        summary.record(
            2,
            Some(0.8),
            Duration::from_millis(1),
            Duration::from_millis(1),
        );

        assert_eq!(summary.batches, 3);
        assert_eq!(summary.examples, 14);
        assert_eq!(summary.loss_batches, 2);
        assert!((summary.mean_loss() - 0.6).abs() < 1.0e-6);
        assert_eq!(summary.data_time, Duration::from_millis(4));
        assert_eq!(summary.step_time, Duration::from_millis(6));
    }

    #[test]
    fn epoch_summary_mean_loss_is_zero_with_no_loss_batches() {
        let mut summary = EpochSummary::default();
        summary.record(4, None, Duration::ZERO, Duration::ZERO);
        assert_eq!(summary.mean_loss(), 0.0);
    }

    #[test]
    fn evaluation_summary_means_handle_zero_batches() {
        let summary = EvaluationSummary::default();
        assert_eq!(summary.mean_loss(), 0.0);
        assert!(summary.mean_count_tanimoto_reconstruction().is_none());
        assert!(summary.mean_binary_tanimoto_reconstruction().is_none());
        assert!(summary.mean_count_tanimoto_correction().is_none());
        assert!(summary.mean_binary_tanimoto_correction().is_none());
    }

    fn fixture_batch_metrics(
        loss: f32,
        count_reconstruction: Option<f32>,
        binary_reconstruction: Option<f32>,
        count_correction: Option<f32>,
        binary_correction: Option<f32>,
        reconstruction_rows: usize,
        correction_rows: usize,
    ) -> BatchLossMetrics {
        BatchLossMetrics {
            loss,
            reconstruction: 0.0,
            reconstruction_bce: 0.0,
            descriptors: 0.0,
            tanimoto_ranking: 0.0,
            tanimoto_ranking_accuracy: 0.0,
            tanimoto_ranking_mrr: 0.0,
            tanimoto_ranking_pearson: 0.0,
            tanimoto_ranking_spearman: 0.0,
            count_tanimoto_reconstruction: count_reconstruction,
            count_tanimoto_correction: count_correction,
            binary_tanimoto_reconstruction: binary_reconstruction,
            binary_tanimoto_correction: binary_correction,
            recon_loss_reconstruction: None,
            recon_loss_correction: None,
            bce_reconstruction: None,
            bce_correction: None,
            reconstruction_rows,
            correction_rows,
            active_huber_mean: 0.0,
            inactive_huber_mean: 0.0,
            violation_bce: 0.0,
            violation_mcc: 0.0,
            violation_mcc_per_class: [0.0; molecular_autoencoder::VIOLATION_CLASS_COUNT],
        }
    }

    #[test]
    fn evaluation_summary_aggregates_loss_and_tanimoto() {
        let mut summary = EvaluationSummary::default();
        // First batch: 3 positives + 1 negative. Per-polarity means
        // collapse to the per-row Tanimoto on the corresponding rows.
        summary.record(
            4,
            Some(0.3),
            &fixture_batch_metrics(0.3, Some(0.8), Some(0.7), Some(0.4), Some(0.3), 3, 1),
            Duration::from_millis(1),
            Duration::from_millis(2),
        );
        // Second batch: 2 positives + 2 negatives.
        summary.record(
            4,
            Some(0.5),
            &fixture_batch_metrics(0.5, Some(0.6), Some(0.5), Some(0.2), Some(0.1), 2, 2),
            Duration::from_millis(1),
            Duration::from_millis(2),
        );
        assert_eq!(summary.batches, 2);
        assert_eq!(summary.reconstruction_rows, 5);
        assert_eq!(summary.correction_rows, 3);
        assert!((summary.mean_loss() - 0.4).abs() < 1.0e-6);
        // Row-weighted means: positives = (0.8*3 + 0.6*2) / 5 = 0.72,
        // negatives = (0.4*1 + 0.2*2) / 3 ≈ 0.2666667.
        let count_reconstruction = summary
            .mean_count_tanimoto_reconstruction()
            .expect("positives present");
        let count_correction = summary
            .mean_count_tanimoto_correction()
            .expect("negatives present");
        let binary_reconstruction = summary
            .mean_binary_tanimoto_reconstruction()
            .expect("positives present");
        let binary_correction = summary
            .mean_binary_tanimoto_correction()
            .expect("negatives present");
        assert!((count_reconstruction - 0.72).abs() < 1.0e-6);
        assert!((count_correction - ((0.4 + 0.4) / 3.0)).abs() < 1.0e-6);
        // binary positives = (0.7*3 + 0.5*2) / 5 = 0.62
        assert!((binary_reconstruction - 0.62).abs() < 1.0e-6);
        // binary negatives = (0.3*1 + 0.1*2) / 3 ≈ 0.1666667
        assert!((binary_correction - ((0.3 + 0.2) / 3.0)).abs() < 1.0e-6);
    }

    #[test]
    fn batch_sink_worker_path_dispatches_train_and_valid_messages() {
        let (tx, rx) = mpsc::channel::<BatchUpdate>();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let mut sink = BatchSink::Worker {
            tx,
            stop_flag: stop_flag.clone(),
        };
        let train_payload = TrainBatchPayload {
            epoch: 1,
            epoch_total: 1,
            global_step: 1,
            batch_size: 2,
            metrics: None,
            data_time: Duration::ZERO,
            step_time: Duration::ZERO,
        };
        let valid_payload = ValidBatchPayload {
            epoch: 1,
            epoch_total: 1,
            batch_index: 1,
            batch_size: 2,
            metrics: None,
        };
        assert!(matches!(
            sink.train_batch(train_payload),
            BatchControl::Continue
        ));
        assert!(matches!(
            sink.valid_batch(valid_payload),
            BatchControl::Continue
        ));
        // The receiver should have got both messages.
        assert!(matches!(
            rx.recv().expect("train msg"),
            BatchUpdate::Train(_)
        ));
        assert!(matches!(
            rx.recv().expect("valid msg"),
            BatchUpdate::Valid(_)
        ));

        // Setting the stop flag causes the next batch to return Stop.
        stop_flag.store(true, Ordering::Release);
        let payload = TrainBatchPayload {
            epoch: 1,
            epoch_total: 1,
            global_step: 1,
            batch_size: 2,
            metrics: None,
            data_time: Duration::ZERO,
            step_time: Duration::ZERO,
        };
        assert!(matches!(sink.train_batch(payload), BatchControl::Stop));
    }

    #[test]
    fn drain_worker_updates_forwards_pending_messages_to_reporter() {
        let (tx, rx) = mpsc::channel::<BatchUpdate>();
        let mut reporter = fixture_reporter();
        let stop_flag = AtomicBool::new(false);
        tx.send(BatchUpdate::Train(TrainBatchPayload {
            epoch: 1,
            epoch_total: 1,
            global_step: 1,
            batch_size: 2,
            metrics: None,
            data_time: Duration::ZERO,
            step_time: Duration::ZERO,
        }))
        .expect("send train");
        tx.send(BatchUpdate::Valid(ValidBatchPayload {
            epoch: 1,
            epoch_total: 1,
            batch_index: 1,
            batch_size: 2,
            metrics: None,
        }))
        .expect("send valid");
        tx.send(BatchUpdate::TrainEpochComplete)
            .expect("send sentinel");
        drop(tx);
        drain_worker_updates(&mut reporter, &stop_flag, &rx);
        // No assertion — this test ensures the drain handles all three
        // variants and the disconnected-channel case without panic.
    }

    #[test]
    fn all_reduce_mean_is_identity_for_world_size_one() {
        // The single-GPU short-circuit returns the input grads unchanged.
        // We can't easily construct a non-empty GradientsParams here, but
        // we can verify the cfg-off path on an empty value compiles + runs.
        let grads = GradientsParams::new();
        let result = all_reduce_mean::<TestBackend>(grads, 0, 1).expect("identity");
        assert_eq!(result.len(), 0);
    }

    /// Builds an adaptive controller with the warmup curriculum
    /// configured and the requested thresholds + decay, bypassing the
    /// fixed-epoch floor unless the caller explicitly sets it.
    fn adaptive_controller_with(
        switch_to_main: f64,
        switch_to_warmup: f64,
        decay: f64,
        warmup_floor_epochs: usize,
    ) -> AdaptiveReconstructionLossController {
        let mut args = fixture_args();
        args.reconstruction_loss_adaptive = true;
        args.reconstruction_loss_warmup =
            Some(crate::cli::ReconstructionLossKindArg::HuberSymmetric);
        args.reconstruction_loss = crate::cli::ReconstructionLossKindArg::Huber;
        args.reconstruction_loss_warmup_epochs = warmup_floor_epochs;
        args.reconstruction_loss_adaptive_switch_to_main = switch_to_main;
        args.reconstruction_loss_adaptive_switch_to_warmup = switch_to_warmup;
        args.reconstruction_loss_adaptive_ema_decay = decay;
        AdaptiveReconstructionLossController::from_args(&args)
            .expect("from_args returns Some when adaptive is enabled")
    }

    #[test]
    fn controller_swaps_to_main_when_ema_drops_below_threshold() {
        // decay = 0.0 makes `update` overwrite the EMA with the latest
        // sample, so the threshold check fires on the first call.
        let mut controller = adaptive_controller_with(1.0, 2.0, 0.0, 0);
        assert_eq!(
            controller.current_kind(),
            molecular_autoencoder::ReconstructionLossKind::HuberSymmetric
        );
        let transition = controller.update(0.5, 1);
        assert_eq!(
            transition,
            Some(molecular_autoencoder::ReconstructionLossKind::Huber)
        );
        assert_eq!(
            controller.current_kind(),
            molecular_autoencoder::ReconstructionLossKind::Huber
        );
    }

    #[test]
    fn controller_swaps_back_to_warmup_above_high_threshold() {
        let mut controller = adaptive_controller_with(1.0, 2.0, 0.0, 0);
        // Drop below the low threshold to move to Main first.
        assert!(controller.update(0.5, 1).is_some());
        // Now climb above the high threshold and verify the reverse
        // transition fires.
        let transition = controller.update(3.0, 1);
        assert_eq!(
            transition,
            Some(molecular_autoencoder::ReconstructionLossKind::HuberSymmetric)
        );
        assert_eq!(
            controller.current_kind(),
            molecular_autoencoder::ReconstructionLossKind::HuberSymmetric
        );
    }

    #[test]
    fn controller_hysteresis_prevents_oscillation_at_boundary() {
        // With thresholds 1.0 / 2.0, EMA samples that stay inside the
        // dead band (1.0 .. 2.0) must not trigger any transitions.
        let mut controller = adaptive_controller_with(1.0, 2.0, 0.0, 0);
        for sample in [1.2_f32, 1.5, 1.9, 1.1, 1.7, 1.3] {
            assert!(
                controller.update(sample, 1).is_none(),
                "in-band sample {sample} must not flip the state"
            );
        }
        assert_eq!(
            controller.current_kind(),
            molecular_autoencoder::ReconstructionLossKind::HuberSymmetric
        );
    }

    #[test]
    fn controller_warmup_floor_suppresses_switch_until_epoch_n() {
        // floor = 3 — until epoch 3 (inclusive), even sub-threshold
        // ratios must keep the controller pinned to Warmup. From epoch
        // 4 onward the same ratio triggers a swap.
        let mut controller = adaptive_controller_with(1.0, 2.0, 0.0, 3);
        for epoch in 1..=3 {
            assert!(controller.update(0.1, epoch).is_none());
            assert_eq!(
                controller.current_kind(),
                molecular_autoencoder::ReconstructionLossKind::HuberSymmetric
            );
        }
        let transition = controller.update(0.1, 4);
        assert_eq!(
            transition,
            Some(molecular_autoencoder::ReconstructionLossKind::Huber)
        );
    }

    #[test]
    fn controller_snapshot_round_trips_through_serde_json() {
        let mut controller = adaptive_controller_with(1.0, 2.0, 0.5, 0);
        // Drive the EMA away from its bootstrap midpoint so the
        // round-trip actually tests preservation of a non-default value.
        let _ = controller.update(3.0, 1);
        let snapshot = controller.snapshot();
        let encoded = serde_json::to_string(&snapshot).expect("serialize snapshot");
        let decoded: AdaptiveSchedulerState =
            serde_json::from_str(&encoded).expect("deserialize snapshot");
        assert!((decoded.ema - snapshot.ema).abs() < f32::EPSILON);
        assert_eq!(decoded.current_kind, snapshot.current_kind);
    }
}
