//! End-to-end training orchestrator.

use std::{
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamConfig, Optimizer},
    record::{DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};
use molecular_autoencoder::{
    MoleculeAutoencoder, MoleculeAutoencoderBatch, MoleculeAutoencoderBatcher,
    MoleculeAutoencoderConfig, ShardManifest,
};

use crate::{
    AppResult,
    checkpoint::{CheckpointState, read_state_json, save_checkpoint, shard_infos},
    cli::Args,
    invalid_input,
    metrics::{examples_per_second, format_summary_metric, millis_per_batch},
    preprocess::prepare_manifest,
    ranking::{TanimotoMetricBackend, effective_device_prefetch_batches},
    reporter::TrainingReporter,
    training::{BatchSink, EvaluationContext, TrainEpochContext, evaluate, train_epoch},
};

/// Trains the molecular autoencoder according to the parsed CLI arguments.
///
/// `devices` carries one entry per participating peer. With a single entry the
/// orchestrator runs the legacy single-GPU training loop; with multiple entries
/// it spawns one worker thread per device and synchronises gradients via
/// `burn-collective`.
pub fn run<B>(args: Args, devices: Vec<B::Device>) -> AppResult<()>
where
    B: AutodiffBackend<FloatElem = f32> + TanimotoMetricBackend,
    B::Device: Clone + Send + Sync,
    MoleculeAutoencoderBatch<B>: Send,
    MoleculeAutoencoder<B>: AutodiffModule<B, InnerModule = MoleculeAutoencoder<B::InnerBackend>>,
    B::InnerBackend: TanimotoMetricBackend<FloatElem = f32>,
    MoleculeAutoencoderBatch<B::InnerBackend>: Send,
    <B::InnerBackend as burn::tensor::backend::BackendTypes>::Device: Clone + Send + Sync,
{
    if devices.is_empty() {
        return Err(invalid_input("run::run requires at least one device"));
    }
    let device = devices
        .first()
        .cloned()
        .expect("non-empty devices checked above");
    if devices.len() > 1 {
        #[cfg(feature = "multi-gpu")]
        {
            return crate::ddp::run_multi::<B>(args, devices);
        }
        #[cfg(not(feature = "multi-gpu"))]
        {
            return Err(invalid_input(
                "multi-device training requires the `multi-gpu` feature (auto-enabled with cuda)",
            ));
        }
    }
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

    let mut model = model_config.init::<B>(&device);
    let mut optimizer = AdamConfig::new().init::<B, MoleculeAutoencoder<B>>();
    let mut state = if args.resume {
        model = model.load_file(args.checkpoint_dir.join("model"), &recorder, &device)?;
        optimizer = optimizer.load_record(<DefaultRecorder as Recorder<B>>::load(
            &recorder,
            args.checkpoint_dir.join("optimizer"),
            &device,
        )?);
        read_state_json(&state_path)?
    } else {
        CheckpointState {
            completed_epoch: 0,
            global_step: 0,
            best_validation_loss: None,
        }
    };

    let batcher = MoleculeAutoencoderBatcher::new(
        model_config.encoder().input_width(),
        model_config.descriptor_width(),
    );
    let tanimoto_ranking = model_config.tanimoto_ranking_runtime();
    let mut reporter = TrainingReporter::new(
        manifest.row_count(),
        manifest.preprocessing().validation_per_mille(),
        args.batch_size,
        args.max_train_batches,
        max_valid_batches,
        args.resume.then_some(state.completed_epoch),
    );
    let loader_profile_every = if reporter.is_active() {
        0
    } else {
        args.loader_profile_every
    };
    println!(
        "training manifest={} shards={} checkpoint_dir={} start_epoch={} epochs={} batch_size={} loader_workers={} device_prefetch_batches={} requested_device_prefetch_batches={} metric_every={} loader_profile_every={} lr={} latent_noise_std={} descriptor_weight={} tanimoto_ranking_weight={} tanimoto_ranking_latent_temperature={} tanimoto_ranking_metric_temperature={} tanimoto_ranking_min_gap={} tanimoto_ranking_candidates={} tanimoto_ranking_pairs_per_batch={} tanimoto_ranking_loss={:?} tanimoto_ranking_softrank_temperature={} bce_weight={} bce_zero_weight={} bce_nonzero_weight={}",
        manifest_path.display(),
        shards.len(),
        args.checkpoint_dir.display(),
        state.completed_epoch + 1,
        args.epochs,
        args.batch_size,
        args.loader_workers,
        args.device_prefetch_batches,
        requested_device_prefetch_batches,
        args.metric_every,
        loader_profile_every,
        args.learning_rate,
        model_config.latent_noise_std(),
        model_config.auxiliary_weights().descriptors(),
        tanimoto_ranking.weight(),
        tanimoto_ranking.latent_temperature(),
        tanimoto_ranking.metric_temperature(),
        tanimoto_ranking.min_gap(),
        tanimoto_ranking.candidates_per_anchor(),
        tanimoto_ranking.pairs_per_batch(),
        tanimoto_ranking.loss_kind(),
        tanimoto_ranking.softrank_temperature(),
        model_config.reconstruction_loss().bce_weight(),
        model_config.reconstruction_loss().bce_zero_weight(),
        model_config.reconstruction_loss().bce_nonzero_weight()
    );

    let stop_flag = Arc::new(AtomicBool::new(false));

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
                args: &args,
                loader_profile_every,
                validation_per_mille: manifest.preprocessing().validation_per_mille(),
                state: &mut state,
                sink: BatchSink::Lead {
                    reporter: &mut reporter,
                    stop_flag: stop_flag.clone(),
                    receiver: None,
                },
                epoch,
                peer_id: 0,
                world_size: 1,
            },
        )?;
        model = next_model;

        if summary.batches == 0 {
            if reporter.should_stop() {
                break;
            }
            return Err(invalid_input("no training batches were produced"));
        }

        let elapsed = epoch_start.elapsed();
        if !reporter.is_active() {
            println!(
                "epoch={epoch} train_loss={:.6} train_batches={} train_examples={} examples_per_sec={:.2} data_wait_ms_per_batch={:.3} train_step_ms_per_batch={:.3}",
                summary.mean_loss(),
                summary.batches,
                summary.examples,
                examples_per_second(summary.examples, elapsed),
                millis_per_batch(summary.data_time, summary.batches),
                millis_per_batch(summary.step_time, summary.batches)
            );
        }

        if args.validate_every != 0 && epoch % args.validate_every == 0 {
            let valid_model = model.valid();
            match evaluate(
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
                    validation_per_mille: manifest.preprocessing().validation_per_mille(),
                    sink: BatchSink::Lead {
                        reporter: &mut reporter,
                        stop_flag: stop_flag.clone(),
                        receiver: None,
                    },
                    epoch,
                    epoch_total: args.epochs,
                    peer_id: 0,
                },
            )? {
                Some(valid) => {
                    let valid_loss = valid.mean_loss();
                    if !reporter.is_active() {
                        println!(
                            "epoch={epoch} valid_loss={valid_loss:.6} valid_count_tanimoto_reconstruction={} valid_count_tanimoto_correction={} valid_binary_tanimoto_reconstruction={} valid_binary_tanimoto_correction={} valid_huber_reconstruction={} valid_huber_correction={} valid_bce_reconstruction={} valid_bce_correction={} valid_reconstruction_rows={} valid_correction_rows={} valid_batches={} valid_examples={} valid_data_wait_ms_per_batch={:.3} valid_step_ms_per_batch={:.3}",
                            format_summary_metric(valid.mean_count_tanimoto_reconstruction()),
                            format_summary_metric(valid.mean_count_tanimoto_correction()),
                            format_summary_metric(valid.mean_binary_tanimoto_reconstruction()),
                            format_summary_metric(valid.mean_binary_tanimoto_correction()),
                            format_summary_metric(valid.mean_huber_reconstruction()),
                            format_summary_metric(valid.mean_huber_correction()),
                            format_summary_metric(valid.mean_bce_reconstruction()),
                            format_summary_metric(valid.mean_bce_correction()),
                            valid.reconstruction_rows,
                            valid.correction_rows,
                            valid.batches,
                            valid.examples,
                            millis_per_batch(valid.data_time, valid.batches),
                            millis_per_batch(valid.step_time, valid.batches)
                        );
                    }
                    state.best_validation_loss = Some(
                        state
                            .best_validation_loss
                            .map_or(valid_loss, |best| best.min(valid_loss)),
                    );
                }
                None => {
                    if !reporter.is_active() {
                        println!("epoch={epoch} validation_skipped=no_validation_rows");
                    }
                }
            }
        }

        if reporter.should_stop() {
            break;
        }

        state.completed_epoch = epoch;
        if args.checkpoint_every != 0 && epoch % args.checkpoint_every == 0 {
            save_checkpoint(&args.checkpoint_dir, &recorder, &model, &optimizer, &state)?;
            if !reporter.is_active() {
                println!(
                    "checkpoint_saved epoch={epoch} global_step={} dir={}",
                    state.global_step,
                    args.checkpoint_dir.display()
                );
            }
        }
    }

    save_checkpoint(&args.checkpoint_dir, &recorder, &model, &optimizer, &state)?;
    reporter.finish()?;
    Ok(())
}
