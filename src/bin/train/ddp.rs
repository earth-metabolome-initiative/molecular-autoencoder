//! Multi-GPU DDP orchestrator: one worker thread per CUDA device with
//! per-step gradient all-reduce via `burn-collective`.
//!
//! The lead peer (peer 0) stays on the main thread so it owns the
//! [`TrainingReporter`] (TUI / indicatif state is not `Send`). Worker peers
//! 1..N run in `std::thread::spawn` closures and forward per-batch metrics to
//! the lead via an `mpsc` channel; the lead drains the channel between its
//! own batches inside [`BatchSink::Lead`].

use std::{
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::{Duration, Instant},
};

use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamConfig, Optimizer},
    record::{DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};
use burn_collective::{CollectiveConfig, PeerId, finish_collective, register};
use molecular_autoencoder::{
    MoleculeAutoencoder, MoleculeAutoencoderBatch, MoleculeAutoencoderBatcher,
    MoleculeAutoencoderConfig, ShardManifest, TanimotoRankingRuntimeConfig,
};

use crate::{
    AppResult,
    checkpoint::{CachedShardInfo, CheckpointState, read_state_json, save_checkpoint, shard_infos},
    cli::Args,
    invalid_input,
    metrics::{examples_per_second, format_summary_metric, millis_per_batch},
    preprocess::prepare_manifest,
    ranking::{TanimotoMetricBackend, effective_device_prefetch_batches},
    reporter::TrainingReporter,
    training::{
        BatchSink, BatchUpdate, EvaluationContext, TrainEpochContext, evaluate, train_epoch,
    },
};

/// Multi-peer DDP entry point. Assumes `devices.len() >= 2`; the caller in
/// [`crate::run::run`] dispatches single-device runs through the legacy path.
pub fn run_multi<B>(args: Args, devices: Vec<B::Device>) -> AppResult<()>
where
    B: AutodiffBackend<FloatElem = f32> + TanimotoMetricBackend + 'static,
    B::Device: Clone + Send + Sync + 'static,
    MoleculeAutoencoder<B>:
        AutodiffModule<B, InnerModule = MoleculeAutoencoder<B::InnerBackend>> + Send + 'static,
    MoleculeAutoencoderBatch<B>: Send,
    B::InnerBackend: TanimotoMetricBackend<FloatElem = f32>,
    MoleculeAutoencoderBatch<B::InnerBackend>: Send,
    <B::InnerBackend as burn::tensor::backend::BackendTypes>::Device: Clone + Send + Sync,
{
    let world_size = u32::try_from(devices.len())
        .map_err(|_| invalid_input("world_size does not fit in u32"))?;

    let mut args = args;
    let requested_device_prefetch_batches = args.device_prefetch_batches;
    args.device_prefetch_batches =
        effective_device_prefetch_batches(requested_device_prefetch_batches);
    let paths = args.manifest_paths()?;
    let max_valid_batches = args.max_valid_batches();

    let manifest_path = prepare_manifest(&args, &paths)?;
    let manifest = ShardManifest::read_from_path(&manifest_path)?;
    let shards = shard_infos(&manifest_path, &manifest)?;
    let fingerprint_size = manifest.preprocessing().counted_ecfp().size();
    let ecfp_radius = manifest.preprocessing().counted_ecfp().radius();
    let validation_per_mille = manifest.preprocessing().validation_per_mille();

    let recorder = DefaultRecorder::default();
    let model_config_path = args.checkpoint_dir.join("model-config.json");
    let state_path = args.checkpoint_dir.join("state.json");

    std::fs::create_dir_all(&args.checkpoint_dir)?;
    let model_config = if args.resume && model_config_path.exists() {
        let config = MoleculeAutoencoderConfig::load_json(&model_config_path)?;
        config.validate(fingerprint_size)?;
        config
    } else {
        let config = args.to_model_config(fingerprint_size, ecfp_radius)?;
        config.save_json(&model_config_path)?;
        config
    };

    let batcher = MoleculeAutoencoderBatcher::new(
        model_config.encoder().input_width(),
        model_config.descriptor_width(),
    );
    let tanimoto_ranking = model_config.tanimoto_ranking_runtime();
    let initial_state = if args.resume {
        read_state_json(&state_path)?
    } else {
        CheckpointState {
            completed_epoch: 0,
            global_step: 0,
            best_validation_loss: None,
        }
    };

    let lead_device = devices
        .first()
        .cloned()
        .expect("devices is non-empty by caller contract");
    let mut canonical_model: MoleculeAutoencoder<B> = model_config.init::<B>(&lead_device);
    if args.resume {
        canonical_model = canonical_model.load_file(
            args.checkpoint_dir.join("model"),
            &recorder,
            &lead_device,
        )?;
    }

    let mut reporter = TrainingReporter::new(
        manifest.row_count(),
        validation_per_mille,
        args.batch_size,
        args.max_train_batches,
        max_valid_batches,
        args.resume.then_some(initial_state.completed_epoch),
    );
    let loader_profile_every = if reporter.is_active() {
        0
    } else {
        args.loader_profile_every
    };

    println!(
        "training[ddp] manifest={} shards={} world_size={} checkpoint_dir={} start_epoch={} epochs={} batch_size_per_peer={} lr={}",
        manifest_path.display(),
        shards.len(),
        world_size,
        args.checkpoint_dir.display(),
        initial_state.completed_epoch + 1,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    );

    let collective_cfg = CollectiveConfig::default().with_num_devices(devices.len());
    let stop_flag = Arc::new(AtomicBool::new(false));
    let train_barrier = Arc::new(Barrier::new(devices.len()));
    let valid_barrier = Arc::new(Barrier::new(devices.len()));

    let (batch_tx, batch_rx) = mpsc::channel::<BatchUpdate>();

    let lead_shards = shards_for_peer(&shards, 0, world_size);
    let lead_state = initial_state.clone();

    let mut handles = Vec::with_capacity(devices.len() - 1);
    for (peer_index, peer_device) in devices.iter().enumerate().skip(1) {
        let peer_id = u32::try_from(peer_index).expect("peer_index fits u32");
        let replica = canonical_model.clone().fork(peer_device);
        let ctx = WorkerCtx {
            peer_id,
            world_size,
            device: peer_device.clone(),
            model: replica,
            shards: shards_for_peer(&shards, peer_id, world_size),
            args: args.clone(),
            tanimoto_ranking,
            collective_cfg: collective_cfg.clone(),
            stop_flag: stop_flag.clone(),
            batch_tx: batch_tx.clone(),
            train_barrier: train_barrier.clone(),
            valid_barrier: valid_barrier.clone(),
            validation_per_mille,
            max_valid_batches,
            initial_state: initial_state.clone(),
            batcher,
        };
        handles.push(thread::spawn(move || -> Result<(), String> {
            worker_loop::<B>(ctx).map_err(|err| err.to_string())
        }));
    }
    // Lead peer keeps the original Sender so the channel stays open even if all
    // worker threads exit; explicitly drop our extra clone to avoid extending
    // the channel beyond the workers' lifetime.
    drop(batch_tx);

    let lead_result = lead_loop::<B>(LeadCtx {
        world_size,
        device: lead_device.clone(),
        model: canonical_model,
        shards: lead_shards,
        args: &args,
        tanimoto_ranking,
        collective_cfg: collective_cfg.clone(),
        stop_flag: stop_flag.clone(),
        batch_rx: &batch_rx,
        train_barrier: train_barrier.clone(),
        valid_barrier: valid_barrier.clone(),
        validation_per_mille,
        max_valid_batches,
        state: lead_state,
        batcher,
        loader_profile_every,
        recorder: &recorder,
        reporter: &mut reporter,
    });

    // Surface a worker error after the lead loop, so any panic message is
    // shown even when the lead returned successfully.
    let mut first_worker_error: Option<Box<dyn std::error::Error>> = None;
    for (peer_index, handle) in handles.into_iter().enumerate() {
        let peer_id = peer_index + 1;
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(message)) => {
                eprintln!("ddp peer {peer_id} returned error: {message}");
                first_worker_error
                    .get_or_insert_with(|| invalid_input(format!("ddp peer {peer_id}: {message}")));
                stop_flag.store(true, Ordering::Release);
            }
            Err(_) => {
                eprintln!("ddp peer {peer_id} panicked");
                first_worker_error
                    .get_or_insert_with(|| invalid_input(format!("ddp peer {peer_id} panicked")));
                stop_flag.store(true, Ordering::Release);
            }
        }
    }

    reporter.finish()?;

    match (lead_result, first_worker_error) {
        (Err(err), _) | (Ok(()), Some(err)) => Err(err),
        (Ok(()), None) => Ok(()),
    }
}

/// Partitions a shard list into the strided slice owned by `peer_id`.
fn shards_for_peer(
    shards: &[CachedShardInfo],
    peer_id: u32,
    world_size: u32,
) -> Vec<CachedShardInfo> {
    shards
        .iter()
        .enumerate()
        .filter(|(index, _)| u32::try_from(*index).is_ok_and(|i| i % world_size == peer_id))
        .map(|(_, shard)| shard.clone())
        .collect()
}

/// Per-worker state passed into [`worker_loop`] when spawning a peer thread.
struct WorkerCtx<B>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    /// Zero-based peer index inside the distributed group.
    peer_id: u32,
    /// Total number of peers participating in the run.
    world_size: u32,
    /// Backend device the worker owns.
    device: B::Device,
    /// Initial model the worker starts from.
    model: MoleculeAutoencoder<B>,
    /// Worker-local subset of cached shards.
    shards: Vec<CachedShardInfo>,
    /// Resolved CLI arguments shared by every worker.
    args: Args,
    /// Tanimoto ranking runtime config (weight, candidates, epsilon).
    tanimoto_ranking: TanimotoRankingRuntimeConfig,
    /// Collective backend configuration for cross-peer comms.
    collective_cfg: CollectiveConfig,
    /// Global stop flag observed across peers (e.g. on Ctrl-C).
    stop_flag: Arc<AtomicBool>,
    /// Channel used to forward per-batch metrics to the lead process.
    batch_tx: Sender<BatchUpdate>,
    /// Barrier synchronizing peers at the end of each training epoch.
    train_barrier: Arc<Barrier>,
    /// Barrier synchronizing peers at the end of each validation pass.
    valid_barrier: Arc<Barrier>,
    /// Validation-split share expressed in parts per mille.
    validation_per_mille: u16,
    /// Optional cap on the number of validation batches per epoch.
    max_valid_batches: Option<usize>,
    /// Checkpoint state the worker resumes from.
    initial_state: CheckpointState,
    /// Batcher used to assemble samples into tensors.
    batcher: MoleculeAutoencoderBatcher,
}

/// Per-peer training loop: registers with the collective backend, runs
/// `train_epoch`/`evaluate` and forwards updates back to the lead via
/// `batch_tx`.
fn worker_loop<B>(ctx: WorkerCtx<B>) -> AppResult<()>
where
    B: AutodiffBackend<FloatElem = f32> + TanimotoMetricBackend,
    B::Device: Clone + Send + Sync,
    MoleculeAutoencoderBatch<B>: Send,
    MoleculeAutoencoder<B>: AutodiffModule<B, InnerModule = MoleculeAutoencoder<B::InnerBackend>>,
    B::InnerBackend: TanimotoMetricBackend<FloatElem = f32>,
    MoleculeAutoencoderBatch<B::InnerBackend>: Send,
    <B::InnerBackend as burn::tensor::backend::BackendTypes>::Device: Clone + Send + Sync,
{
    let WorkerCtx {
        peer_id,
        world_size,
        device,
        model,
        shards,
        args,
        tanimoto_ranking,
        collective_cfg,
        stop_flag,
        batch_tx,
        train_barrier,
        valid_barrier,
        validation_per_mille,
        max_valid_batches,
        initial_state,
        batcher,
    } = ctx;

    register::<B::InnerBackend>(PeerId::from(peer_id), device.clone(), collective_cfg)
        .map_err(|err| invalid_input(format!("collective register failed: {err:?}")))?;

    let recorder = DefaultRecorder::default();
    let mut model = model;
    let mut optimizer = AdamConfig::new().init::<B, MoleculeAutoencoder<B>>();
    if args.resume {
        optimizer = optimizer.load_record(<DefaultRecorder as Recorder<B>>::load(
            &recorder,
            args.checkpoint_dir.join("optimizer"),
            &device,
        )?);
    }
    let mut state = initial_state;

    for epoch in (state.completed_epoch + 1)..=args.epochs {
        let sink = BatchSink::Worker {
            tx: batch_tx.clone(),
            stop_flag: stop_flag.clone(),
        };
        let (next_model, _summary) = train_epoch(
            model,
            &mut optimizer,
            TrainEpochContext {
                shards: &shards,
                batcher,
                device: &device,
                tanimoto_ranking,
                args: &args,
                loader_profile_every: 0,
                validation_per_mille,
                state: &mut state,
                sink,
                epoch,
                peer_id,
                world_size,
            },
        )?;
        model = next_model;

        let _ = batch_tx.send(BatchUpdate::TrainEpochComplete);
        train_barrier.wait();

        if args.validate_every != 0 && epoch % args.validate_every == 0 {
            let valid_model = model.valid();
            let _ = evaluate(
                &valid_model,
                EvaluationContext {
                    shards: &shards,
                    batcher,
                    device: &device,
                    batch_size: args.batch_size,
                    max_batches: max_valid_batches,
                    loader_workers: args.loader_workers,
                    device_prefetch_batches: args.device_prefetch_batches,
                    loader_profile_every: 0,
                    tanimoto_ranking,
                    validation_per_mille,
                    sink: BatchSink::Worker {
                        tx: batch_tx.clone(),
                        stop_flag: stop_flag.clone(),
                    },
                    epoch,
                    epoch_total: args.epochs,
                    peer_id,
                },
            )?;
            let _ = batch_tx.send(BatchUpdate::ValidEpochComplete);
            valid_barrier.wait();
        }

        if stop_flag.load(Ordering::Acquire) {
            break;
        }
        state.completed_epoch = epoch;
    }

    finish_collective::<B::InnerBackend>(PeerId::from(peer_id))
        .map_err(|err| invalid_input(format!("collective finish failed: {err:?}")))?;
    Ok(())
}

/// Lead process state passed into [`lead_loop`] alongside the spawned peers.
#[allow(clippy::struct_field_names)]
struct LeadCtx<'a, B>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    /// Total number of peers participating in the run.
    world_size: u32,
    /// Backend device the lead trains on (peer 0).
    device: B::Device,
    /// Initial model the lead starts from.
    model: MoleculeAutoencoder<B>,
    /// Lead-local subset of cached shards.
    shards: Vec<CachedShardInfo>,
    /// Resolved CLI arguments shared by every peer.
    args: &'a Args,
    /// Tanimoto ranking runtime config.
    tanimoto_ranking: TanimotoRankingRuntimeConfig,
    /// Collective backend configuration for cross-peer comms.
    collective_cfg: CollectiveConfig,
    /// Global stop flag shared with workers.
    stop_flag: Arc<AtomicBool>,
    /// Channel that delivers per-batch updates from worker peers.
    batch_rx: &'a Receiver<BatchUpdate>,
    /// Barrier synchronizing peers at the end of each training epoch.
    train_barrier: Arc<Barrier>,
    /// Barrier synchronizing peers at the end of each validation pass.
    valid_barrier: Arc<Barrier>,
    /// Validation-split share expressed in parts per mille.
    validation_per_mille: u16,
    /// Optional cap on the number of validation batches per epoch.
    max_valid_batches: Option<usize>,
    /// Checkpoint state the lead persists at the end of each epoch.
    state: CheckpointState,
    /// Batcher used to assemble samples into tensors.
    batcher: MoleculeAutoencoderBatcher,
    /// Period (in batches) between loader-profile log lines.
    loader_profile_every: usize,
    /// Recorder used to persist model / optimizer state.
    recorder: &'a DefaultRecorder,
    /// Live training reporter (TUI or indicatif fallback).
    reporter: &'a mut TrainingReporter,
}

/// Lead-process training loop: drives `train_epoch` / `evaluate` on peer 0,
/// drains worker metrics from `batch_rx`, and persists checkpoints.
fn lead_loop<B>(ctx: LeadCtx<'_, B>) -> AppResult<()>
where
    B: AutodiffBackend<FloatElem = f32> + TanimotoMetricBackend,
    B::Device: Clone + Send + Sync,
    MoleculeAutoencoderBatch<B>: Send,
    MoleculeAutoencoder<B>: AutodiffModule<B, InnerModule = MoleculeAutoencoder<B::InnerBackend>>,
    B::InnerBackend: TanimotoMetricBackend<FloatElem = f32>,
    MoleculeAutoencoderBatch<B::InnerBackend>: Send,
    <B::InnerBackend as burn::tensor::backend::BackendTypes>::Device: Clone + Send + Sync,
{
    let LeadCtx {
        world_size,
        device,
        mut model,
        shards,
        args,
        tanimoto_ranking,
        collective_cfg,
        stop_flag,
        batch_rx,
        train_barrier,
        valid_barrier,
        validation_per_mille,
        max_valid_batches,
        mut state,
        batcher,
        loader_profile_every,
        recorder,
        reporter,
    } = ctx;

    register::<B::InnerBackend>(PeerId::from(0u32), device.clone(), collective_cfg)
        .map_err(|err| invalid_input(format!("collective register failed: {err:?}")))?;

    let mut optimizer = AdamConfig::new().init::<B, MoleculeAutoencoder<B>>();
    if args.resume {
        optimizer = optimizer.load_record(<DefaultRecorder as Recorder<B>>::load(
            recorder,
            args.checkpoint_dir.join("optimizer"),
            &device,
        )?);
    }

    let workers = world_size.saturating_sub(1) as usize;

    for epoch in (state.completed_epoch + 1)..=args.epochs {
        let epoch_start = Instant::now();
        let (next_model, summary) = train_epoch(
            model,
            &mut optimizer,
            TrainEpochContext {
                shards: &shards,
                batcher,
                device: &device,
                tanimoto_ranking,
                args,
                loader_profile_every,
                validation_per_mille,
                state: &mut state,
                sink: BatchSink::Lead {
                    reporter: &mut *reporter,
                    stop_flag: stop_flag.clone(),
                    receiver: Some(batch_rx),
                },
                epoch,
                peer_id: 0,
                world_size,
            },
        )?;
        model = next_model;

        // Drain remaining worker batches and their epoch sentinels.
        drain_until_epoch_complete(batch_rx, reporter, &stop_flag, workers, true);
        train_barrier.wait();

        let elapsed = epoch_start.elapsed();
        if !reporter.is_active() {
            println!(
                "epoch={epoch} train_loss={:.6} train_batches={} train_examples={} examples_per_sec={:.2} data_wait_ms_per_batch={:.3} train_step_ms_per_batch={:.3}",
                summary.mean_loss(),
                summary.batches,
                summary.examples,
                examples_per_second(summary.examples, elapsed),
                millis_per_batch(summary.data_time, summary.batches),
                millis_per_batch(summary.step_time, summary.batches),
            );
        }

        if args.validate_every != 0 && epoch % args.validate_every == 0 {
            let valid_model = model.valid();
            let outcome = evaluate(
                &valid_model,
                EvaluationContext {
                    shards: &shards,
                    batcher,
                    device: &device,
                    batch_size: args.batch_size,
                    max_batches: max_valid_batches,
                    loader_workers: args.loader_workers,
                    device_prefetch_batches: args.device_prefetch_batches,
                    loader_profile_every,
                    tanimoto_ranking,
                    validation_per_mille,
                    sink: BatchSink::Lead {
                        reporter: &mut *reporter,
                        stop_flag: stop_flag.clone(),
                        receiver: Some(batch_rx),
                    },
                    epoch,
                    epoch_total: args.epochs,
                    peer_id: 0,
                },
            )?;
            drain_until_epoch_complete(batch_rx, reporter, &stop_flag, workers, false);
            valid_barrier.wait();
            if let Some(valid) = outcome
                && !reporter.is_active()
            {
                let valid_loss = valid.mean_loss();
                println!(
                    "epoch={epoch} valid_loss={valid_loss:.6} valid_count_tanimoto_reconstruction={} valid_count_tanimoto_correction={} valid_binary_tanimoto_reconstruction={} valid_binary_tanimoto_correction={} valid_reconstruction_rows={} valid_correction_rows={} valid_batches={} valid_examples={} valid_data_wait_ms_per_batch={:.3} valid_step_ms_per_batch={:.3}",
                    format_summary_metric(valid.mean_count_tanimoto_reconstruction()),
                    format_summary_metric(valid.mean_count_tanimoto_correction()),
                    format_summary_metric(valid.mean_binary_tanimoto_reconstruction()),
                    format_summary_metric(valid.mean_binary_tanimoto_correction()),
                    valid.reconstruction_rows,
                    valid.correction_rows,
                    valid.batches,
                    valid.examples,
                    millis_per_batch(valid.data_time, valid.batches),
                    millis_per_batch(valid.step_time, valid.batches),
                );
                state.best_validation_loss = Some(
                    state
                        .best_validation_loss
                        .map_or(valid_loss, |best| best.min(valid_loss)),
                );
            }
        }

        if reporter.should_stop() || stop_flag.load(Ordering::Acquire) {
            stop_flag.store(true, Ordering::Release);
            break;
        }

        state.completed_epoch = epoch;
        if args.checkpoint_every != 0 && epoch % args.checkpoint_every == 0 {
            save_checkpoint(&args.checkpoint_dir, recorder, &model, &optimizer, &state)?;
            if !reporter.is_active() {
                println!(
                    "checkpoint_saved epoch={epoch} global_step={} dir={}",
                    state.global_step,
                    args.checkpoint_dir.display(),
                );
            }
        }
    }

    save_checkpoint(&args.checkpoint_dir, recorder, &model, &optimizer, &state)?;
    finish_collective::<B::InnerBackend>(PeerId::from(0u32))
        .map_err(|err| invalid_input(format!("collective finish failed: {err:?}")))?;
    Ok(())
}

/// Drains pending worker updates until the expected number of epoch sentinels
/// has been observed. Forwards every per-batch payload to the reporter.
fn drain_until_epoch_complete(
    rx: &Receiver<BatchUpdate>,
    reporter: &mut TrainingReporter,
    stop_flag: &AtomicBool,
    expected_sentinels: usize,
    train: bool,
) {
    if expected_sentinels == 0 {
        return;
    }
    let mut remaining = expected_sentinels;
    while remaining > 0 {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(BatchUpdate::Train(payload)) => {
                let control = reporter.train_batch(
                    payload.epoch,
                    payload.epoch_total,
                    payload.global_step,
                    payload.batch_size,
                    payload.metrics,
                    payload.data_time,
                    payload.step_time,
                );
                if matches!(control, crate::dataloader::BatchControl::Stop) {
                    stop_flag.store(true, Ordering::Release);
                }
            }
            Ok(BatchUpdate::Valid(payload)) => {
                let control = reporter.valid_batch(
                    payload.epoch,
                    payload.epoch_total,
                    payload.batch_index,
                    payload.batch_size,
                    payload.metrics,
                );
                if matches!(control, crate::dataloader::BatchControl::Stop) {
                    stop_flag.store(true, Ordering::Release);
                }
            }
            Ok(BatchUpdate::TrainEpochComplete) if train => {
                remaining -= 1;
            }
            Ok(BatchUpdate::ValidEpochComplete) if !train => {
                remaining -= 1;
            }
            Ok(BatchUpdate::TrainEpochComplete | BatchUpdate::ValidEpochComplete) => {
                // Wrong phase sentinel: forward through and keep waiting for
                // the phase we expected. Out-of-phase arrivals only happen if
                // a worker errors mid-validation, so this branch is rare.
            }
            Err(_) => {
                // Timeout: check stop flag, then keep waiting. A worker may
                // have died; we don't want to deadlock.
                if stop_flag.load(Ordering::Acquire) {
                    return;
                }
            }
        }
    }
}
