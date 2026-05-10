//! Train the molecular autoencoder from cached dataset shard files.

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
use std::{
    env,
    error::Error as StdError,
    fs::File,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant},
};

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
use std::io::IsTerminal;

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{DefaultRecorder, Recorder},
    tensor::{TensorData, Transaction, backend::AutodiffBackend},
};
#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "cuda")]
use molecular_autoencoder::TanimotoRankingBatch;
#[cfg(feature = "cuda")]
use molecular_autoencoder::tanimoto_cuda::{
    CountedTanimotoKernelBackend, CountedTanimotoRankingKernelConfig,
    counted_tanimoto_similarity_ranking_kernel,
};
#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "datasets",
    any(feature = "cuda", feature = "ndarray")
))]
use molecular_autoencoder::{
    DEFAULT_PREPROCESS_CHUNK_ROWS, DEFAULT_PREPROCESS_THREADS, DEFAULT_ROWS_PER_SHARD,
    DatasetPreprocessOptions, MoleculeRecord, PreprocessingConfig,
    features::REGRESSION_TARGET_WIDTH, molecule_records_from_smiles_dataset,
    preprocess_dataset_record_chunks,
};
#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "datasets"),
    any(feature = "cuda", feature = "ndarray")
))]
const DEFAULT_PREPROCESS_THREADS: usize = 64;
#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "datasets"),
    any(feature = "cuda", feature = "ndarray")
))]
const DEFAULT_ROWS_PER_SHARD: usize = 10_000_000;
#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
use molecular_autoencoder::{
    DataSplit, MoleculeAutoencoder, MoleculeAutoencoderBatch, MoleculeAutoencoderBatcher,
    MoleculeAutoencoderConfig, MoleculeAutoencoderHostBatch, MoleculeAutoencoderSample,
    MoleculeBatchBuildProfile, MoleculeLossBreakdown, MoleculeShardRow, SHARD_MANIFEST_VERSION,
    ShardManifest, SparseMoleculeShard, TanimotoRankingConfig, batch_sparse_log_count_tanimoto,
};
#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
use serde::{Deserialize, Serialize};
#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "datasets",
    any(feature = "cuda", feature = "ndarray")
))]
use smiles_parser::prelude::{
    DatasetCollectionSource, DatasetFetchOptions, DatasetSource, GzipMode, PUBCHEM_SMILES,
    SmilesDatasetRecordSource, Zinc20Smiles,
};

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
type AppResult<T> = Result<T, Box<dyn StdError>>;

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
const DEFAULT_LOADER_WORKERS: usize = 2;

#[cfg(all(feature = "std", feature = "train", feature = "cuda"))]
const DEFAULT_DEVICE_PREFETCH_BATCHES: usize = 0;

#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "cuda"),
    feature = "ndarray"
))]
const DEFAULT_DEVICE_PREFETCH_BATCHES: usize = 0;

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
const DEFAULT_METRIC_EVERY: usize = 10;

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
const ZINC20_FIRST_CHUNK: u8 = 1;

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
const ZINC20_LAST_CHUNK: u8 = 20;

#[cfg(all(feature = "std", feature = "train", feature = "cuda"))]
trait TanimotoMetricBackend: Backend + CountedTanimotoKernelBackend {}

#[cfg(all(feature = "std", feature = "train", feature = "cuda"))]
impl<B> TanimotoMetricBackend for B where B: Backend + CountedTanimotoKernelBackend {}

#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "cuda"),
    feature = "ndarray"
))]
trait TanimotoMetricBackend: Backend {}

#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "cuda"),
    feature = "ndarray"
))]
impl<B> TanimotoMetricBackend for B where B: Backend {}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceSelection {
    PubChem,
    Zinc20 { first_chunk: u8, last_chunk: u8 },
    PubChemAndZinc20 { first_chunk: u8, last_chunk: u8 },
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
impl SourceSelection {
    fn with_zinc20_chunks(self, first_chunk: u8, last_chunk: u8) -> AppResult<Self> {
        let updated = match self {
            Self::PubChem => {
                return Err(invalid_input(
                    "--zinc20-chunks is only valid for `zinc20` or `all` sources",
                ));
            }
            Self::Zinc20 { .. } => Self::Zinc20 {
                first_chunk,
                last_chunk,
            },
            Self::PubChemAndZinc20 { .. } => Self::PubChemAndZinc20 {
                first_chunk,
                last_chunk,
            },
        };
        Ok(updated)
    }

    fn manifest_source(self) -> String {
        match self {
            Self::PubChem => "pubchem-smiles".to_string(),
            Self::Zinc20 {
                first_chunk,
                last_chunk,
            } => format!("zinc20-smiles chunks {first_chunk}..={last_chunk}"),
            Self::PubChemAndZinc20 {
                first_chunk,
                last_chunk,
            } => format!("pubchem-smiles + zinc20-smiles chunks {first_chunk}..={last_chunk}"),
        }
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone)]
struct Args {
    manifest_path: PathBuf,
    cache_dir: Option<PathBuf>,
    source_selection: Option<SourceSelection>,
    checkpoint_dir: PathBuf,
    rows_per_shard: usize,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    latent_width: usize,
    latent_noise_std: f64,
    hidden_widths: Vec<usize>,
    checkpoint_every: usize,
    validate_every: usize,
    max_train_batches: Option<usize>,
    max_valid_batches: Option<usize>,
    loader_workers: usize,
    device_prefetch_batches: usize,
    loader_profile_every: usize,
    metric_every: usize,
    descriptor_weight: f64,
    tanimoto_ranking_weight: f64,
    tanimoto_ranking_latent_temperature: f64,
    tanimoto_ranking_metric_temperature: f64,
    tanimoto_ranking_min_gap: f64,
    tanimoto_ranking_candidates: usize,
    tanimoto_ranking_pairs_per_batch: usize,
    resume: bool,
    force_preprocess: bool,
    preprocess_threads: usize,
    cuda_device: usize,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointState {
    completed_epoch: usize,
    global_step: usize,
    best_validation_loss: Option<f32>,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Copy, Default)]
struct EpochSummary {
    batches: usize,
    loss_batches: usize,
    examples: usize,
    loss_sum: f64,
    data_time: Duration,
    step_time: Duration,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Copy, Default)]
struct EvaluationSummary {
    batches: usize,
    loss_batches: usize,
    examples: usize,
    loss_sum: f64,
    tanimoto_sum: f64,
    data_time: Duration,
    step_time: Duration,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[cfg_attr(not(feature = "tui"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
struct BatchLossMetrics {
    loss: f32,
    reconstruction: f32,
    descriptors: f32,
    tanimoto_ranking: f32,
    tanimoto_ranking_accuracy: f32,
    tanimoto_ranking_pairs: f32,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchControl {
    Continue,
    Stop,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
struct TrainEpochContext<'a, B: Backend> {
    shards: &'a [CachedShardInfo],
    batcher: MoleculeAutoencoderBatcher,
    device: &'a B::Device,
    tanimoto_ranking: TanimotoRankingRuntimeConfig,
    args: &'a Args,
    loader_profile_every: usize,
    validation_per_mille: u16,
    state: &'a mut CheckpointState,
    reporter: &'a mut TrainingReporter,
    epoch: usize,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
struct EvaluationContext<'a, B: Backend> {
    shards: &'a [CachedShardInfo],
    batcher: MoleculeAutoencoderBatcher,
    device: &'a B::Device,
    batch_size: usize,
    max_batches: Option<usize>,
    loader_workers: usize,
    device_prefetch_batches: usize,
    loader_profile_every: usize,
    tanimoto_ranking: TanimotoRankingRuntimeConfig,
    validation_per_mille: u16,
    reporter: &'a mut TrainingReporter,
    epoch: usize,
    epoch_total: usize,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
struct BatchIterationContext<'a, B: Backend> {
    shards: &'a [CachedShardInfo],
    split: DataSplit,
    batcher: MoleculeAutoencoderBatcher,
    device: &'a B::Device,
    batch_size: usize,
    max_batches: Option<usize>,
    loader_workers: usize,
    device_prefetch_batches: usize,
    loader_profile_every: usize,
    tanimoto_ranking: TanimotoRankingRuntimeConfig,
    validation_per_mille: u16,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Copy, Default)]
struct TanimotoRankingRuntimeConfig {
    weight: f64,
    latent_temperature: f64,
    metric_temperature: f64,
    min_gap: f64,
    candidates_per_anchor: usize,
    pairs_per_batch: usize,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Copy, Default)]
struct BatchLoadProfile {
    shard_read: Duration,
    row_scan: Duration,
    sparse_allocation: Duration,
    sparse_fill: Duration,
    tensor_build: Duration,
    producer_total: Duration,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone)]
struct CachedShardInfo {
    path: PathBuf,
    row_count: usize,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone)]
struct CachedBatchPlan {
    shard: CachedShardInfo,
    start_row: usize,
    end_row: usize,
    split: DataSplit,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone)]
struct CachedBatchPlanDataset {
    plans: Vec<CachedBatchPlan>,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone)]
struct CachedLoaderBatch {
    host: MoleculeAutoencoderHostBatch,
    rows: usize,
    profile: BatchLoadProfile,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
struct DeviceLoaderBatch<B: Backend> {
    batch: MoleculeAutoencoderBatch<B>,
    rows: usize,
    profile: BatchLoadProfile,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
enum HostLoaderMessage {
    Batch(Box<CachedLoaderBatch>),
    Error(String),
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
enum DeviceLoaderMessage<B: Backend> {
    Batch(DeviceLoaderBatch<B>),
    Error(String),
}

#[cfg(all(feature = "std", feature = "train", feature = "cuda"))]
fn main() -> AppResult<()> {
    type B = burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>;

    let args = Args::parse()?;
    let device = burn::backend::cuda::CudaDevice::new(args.cuda_device);
    run::<B>(args, device)
}

#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "cuda"),
    feature = "ndarray"
))]
fn main() -> AppResult<()> {
    type B = burn::backend::Autodiff<burn::backend::NdArray<f32, i64>>;

    eprintln!(
        "warning: running with ndarray for smoke testing; enable `cuda-fusion` or \
         `cuda-no-fusion` for the intended training path"
    );
    let args = Args::parse()?;
    let device = burn::backend::ndarray::NdArrayDevice::default();
    run::<B>(args, device)
}

#[cfg(not(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
)))]
fn main() {
    eprintln!(
        "enable `std`, `train`, and either `cuda-fusion`, `cuda-no-fusion`, or `ndarray` \
         to run this example; include `tui` for the terminal training UI"
    );
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn run<B>(args: Args, device: B::Device) -> AppResult<()>
where
    B: AutodiffBackend<FloatElem = f32> + TanimotoMetricBackend,
    B::Device: Clone + Send + Sync,
    MoleculeAutoencoderBatch<B>: Send,
    MoleculeAutoencoder<B>: AutodiffModule<B, InnerModule = MoleculeAutoencoder<B::InnerBackend>>,
    B::InnerBackend: TanimotoMetricBackend<FloatElem = f32>,
    MoleculeAutoencoderBatch<B::InnerBackend>: Send,
    <B::InnerBackend as burn::tensor::backend::BackendTypes>::Device: Clone + Send + Sync,
{
    let mut args = args;
    let requested_device_prefetch_batches = args.device_prefetch_batches;
    args.device_prefetch_batches =
        effective_device_prefetch_batches(requested_device_prefetch_batches);

    let manifest_path = prepare_manifest(&args)?;
    let manifest = ShardManifest::read_from_path(&manifest_path)?;
    let shards = shard_infos(&manifest_path, &manifest)?;
    let fingerprint_size = manifest.preprocessing.counted_ecfp.size;
    let recorder = DefaultRecorder::default();
    let model_config_path = args.checkpoint_dir.join("model-config.json");
    let state_path = args.checkpoint_dir.join("state.json");

    std::fs::create_dir_all(&args.checkpoint_dir)?;
    let model_config = if args.resume && model_config_path.exists() {
        read_json(&model_config_path)?
    } else {
        let mut config = MoleculeAutoencoderConfig::symmetric(
            fingerprint_size,
            args.latent_width,
            args.hidden_widths.clone(),
        );
        config.auxiliary_weights.descriptors = args.descriptor_weight;
        config.auxiliary_weights.tanimoto_ranking = args.tanimoto_ranking_weight;
        config.latent_noise_std = args.latent_noise_std;
        config.tanimoto_ranking = TanimotoRankingConfig {
            latent_temperature: args.tanimoto_ranking_latent_temperature,
            metric_temperature: args.tanimoto_ranking_metric_temperature,
            min_gap: args.tanimoto_ranking_min_gap,
            candidates_per_anchor: args.tanimoto_ranking_candidates,
            pairs_per_batch: args.tanimoto_ranking_pairs_per_batch,
        };
        write_json(&model_config_path, &config)?;
        config
    };
    validate_model_config(&model_config, fingerprint_size)?;

    let mut model = model_config.init::<B>(&device);
    let mut optimizer = AdamConfig::new().init::<B, MoleculeAutoencoder<B>>();
    let mut state = if args.resume {
        model = model.load_file(args.checkpoint_dir.join("model"), &recorder, &device)?;
        optimizer = optimizer.load_record(<DefaultRecorder as Recorder<B>>::load(
            &recorder,
            args.checkpoint_dir.join("optimizer"),
            &device,
        )?);
        read_json(&state_path)?
    } else {
        CheckpointState {
            completed_epoch: 0,
            global_step: 0,
            best_validation_loss: None,
        }
    };

    let batcher = MoleculeAutoencoderBatcher::new(
        model_config.encoder.input_width,
        model_config.descriptor_width,
    );
    let tanimoto_ranking = TanimotoRankingRuntimeConfig::from_model_config(&model_config);
    let mut reporter = TrainingReporter::new(
        manifest.row_count,
        manifest.preprocessing.validation_per_mille,
        args.batch_size,
        args.max_train_batches,
        args.max_valid_batches,
        args.resume.then_some(state.completed_epoch),
    );
    let loader_profile_every = if reporter.is_active() {
        0
    } else {
        args.loader_profile_every
    };
    println!(
        "training manifest={} shards={} checkpoint_dir={} start_epoch={} epochs={} batch_size={} loader_workers={} device_prefetch_batches={} requested_device_prefetch_batches={} metric_every={} loader_profile_every={} lr={} latent_noise_std={} descriptor_weight={} tanimoto_ranking_weight={} tanimoto_ranking_latent_temperature={} tanimoto_ranking_metric_temperature={} tanimoto_ranking_min_gap={} tanimoto_ranking_candidates={} tanimoto_ranking_pairs_per_batch={}",
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
        model_config.latent_noise_std,
        model_config.auxiliary_weights.descriptors,
        tanimoto_ranking.weight,
        tanimoto_ranking.latent_temperature,
        tanimoto_ranking.metric_temperature,
        tanimoto_ranking.min_gap,
        tanimoto_ranking.candidates_per_anchor,
        tanimoto_ranking.pairs_per_batch
    );

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
                validation_per_mille: manifest.preprocessing.validation_per_mille,
                state: &mut state,
                reporter: &mut reporter,
                epoch,
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
                    max_batches: args.max_valid_batches,
                    loader_workers: args.loader_workers,
                    device_prefetch_batches: args.device_prefetch_batches,
                    loader_profile_every,
                    tanimoto_ranking,
                    validation_per_mille: manifest.preprocessing.validation_per_mille,
                    reporter: &mut reporter,
                    epoch,
                    epoch_total: args.epochs,
                },
            )? {
                Some(valid) => {
                    let valid_loss = valid.mean_loss();
                    if !reporter.is_active() {
                        println!(
                            "epoch={epoch} valid_loss={valid_loss:.6} valid_count_tanimoto={:.6} valid_batches={} valid_examples={} valid_data_wait_ms_per_batch={:.3} valid_step_ms_per_batch={:.3}",
                            valid.mean_tanimoto(),
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

#[cfg(all(feature = "std", feature = "train", feature = "cuda"))]
fn effective_device_prefetch_batches(requested: usize) -> usize {
    if requested > 0 {
        eprintln!(
            "warning: disabling --device-prefetch-batches={requested} for CUDA; \
             Burn/CubeCL CUDA tensor construction from a background thread can cause \
             CUDA_ERROR_ILLEGAL_ADDRESS. Host batch prefetch remains enabled via \
             --loader-workers."
        );
    }
    0
}

#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "cuda"),
    feature = "ndarray"
))]
fn effective_device_prefetch_batches(requested: usize) -> usize {
    requested
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
impl TanimotoRankingRuntimeConfig {
    fn from_model_config(config: &MoleculeAutoencoderConfig) -> Self {
        Self {
            weight: config.auxiliary_weights.tanimoto_ranking,
            latent_temperature: config.tanimoto_ranking.latent_temperature,
            metric_temperature: config.tanimoto_ranking.metric_temperature,
            min_gap: config.tanimoto_ranking.min_gap,
            candidates_per_anchor: config.tanimoto_ranking.candidates_per_anchor,
            pairs_per_batch: config.tanimoto_ranking.pairs_per_batch,
        }
    }

    #[cfg(feature = "cuda")]
    fn enabled(self) -> bool {
        self.weight > 0.0
    }
}

#[cfg(all(feature = "std", feature = "train", feature = "cuda"))]
fn attach_tanimoto_ranking<B>(
    batch: &mut MoleculeAutoencoderBatch<B>,
    config: TanimotoRankingRuntimeConfig,
    seed: u64,
) where
    B: TanimotoMetricBackend,
{
    if !config.enabled() || batch.fingerprints.batch_size() < 3 {
        return;
    }
    let (partner_a_index, partner_b_index, target_delta) =
        counted_tanimoto_similarity_ranking_kernel(
            batch.fingerprints.indices.clone(),
            batch.fingerprints.counts.clone(),
            batch.fingerprints.mask.clone(),
            CountedTanimotoRankingKernelConfig {
                batch_items: batch.fingerprints.batch_size(),
                candidates_per_anchor: config.candidates_per_anchor,
                seed,
                epsilon: 1.0e-8,
            },
        );
    batch.tanimoto_ranking = TanimotoRankingBatch {
        partner_a_index,
        partner_b_index,
        target_delta,
    };
}

#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "cuda"),
    feature = "ndarray"
))]
fn attach_tanimoto_ranking<B>(
    _batch: &mut MoleculeAutoencoderBatch<B>,
    _config: TanimotoRankingRuntimeConfig,
    _seed: u64,
) where
    B: TanimotoMetricBackend,
{
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn tanimoto_ranking_seed(epoch: usize, batch_index: usize) -> u64 {
    ((epoch as u64) << 32) ^ (batch_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn train_epoch<B, O>(
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
            tanimoto_ranking,
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
            let losses = current.loss(batch);
            let loss = losses.total();
            let next_step = state.global_step + 1;
            let batch_metrics = should_sample_metrics(next_step, args.metric_every)
                .then(|| loss_metrics(loss.clone(), &losses))
                .transpose()?;
            if batch_metrics.is_some_and(|metrics| !metrics.loss.is_finite()) {
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

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn evaluate<B>(
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
            tanimoto_ranking,
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
            let tanimoto_tensor =
                batch_sparse_log_count_tanimoto(&fingerprints_for_metrics, reconstructed).mean();
            let (batch_metrics, tanimoto) =
                validation_metrics(total_loss.clone(), &losses, tanimoto_tensor)?;
            if !batch_metrics.loss.is_finite() {
                return Err(invalid_input("non-finite validation loss"));
            }
            let step_time = step_start.elapsed();
            summary.record(
                rows,
                Some(batch_metrics.loss),
                tanimoto,
                data_time,
                step_time,
            );
            Ok(reporter.valid_batch(
                epoch,
                epoch_total,
                summary.batches,
                rows,
                Some(batch_metrics),
                tanimoto,
            ))
        },
    )?;

    if summary.batches == 0 {
        Ok(None)
    } else {
        Ok(Some(summary))
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn for_each_batch<B, F>(context: BatchIterationContext<'_, B>, consume: F) -> AppResult<()>
where
    B: Backend,
    B::Device: Clone + Send + Sync,
    MoleculeAutoencoderBatch<B>: Send,
    F: FnMut(MoleculeAutoencoderBatch<B>, usize, Duration) -> AppResult<BatchControl>,
{
    let BatchIterationContext {
        shards,
        split,
        batcher,
        device,
        batch_size,
        max_batches,
        loader_workers,
        device_prefetch_batches,
        loader_profile_every,
        tanimoto_ranking,
        validation_per_mille,
    } = context;

    if batch_size == 0 {
        return Err(invalid_input("batch size must be greater than zero"));
    }
    if loader_workers > 0 {
        return for_each_batch_dataloader(
            BatchIterationContext {
                shards,
                split,
                batcher,
                device,
                batch_size,
                max_batches,
                loader_workers,
                device_prefetch_batches,
                loader_profile_every,
                tanimoto_ranking,
                validation_per_mille,
            },
            consume,
        );
    }
    for_each_batch_sync(
        BatchIterationContext {
            shards,
            split,
            batcher,
            device,
            batch_size,
            max_batches,
            loader_workers,
            device_prefetch_batches,
            loader_profile_every,
            tanimoto_ranking,
            validation_per_mille,
        },
        consume,
    )
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn for_each_batch_dataloader<B, F>(
    context: BatchIterationContext<'_, B>,
    mut consume: F,
) -> AppResult<()>
where
    B: Backend,
    B::Device: Clone + Send + Sync,
    MoleculeAutoencoderBatch<B>: Send,
    F: FnMut(MoleculeAutoencoderBatch<B>, usize, Duration) -> AppResult<BatchControl>,
{
    let BatchIterationContext {
        shards,
        split,
        batcher,
        device,
        batch_size,
        max_batches,
        loader_workers,
        device_prefetch_batches,
        loader_profile_every,
        validation_per_mille,
        ..
    } = context;
    let dataset =
        CachedBatchPlanDataset::new(shards, split, batch_size, max_batches, validation_per_mille)?;
    if dataset.plans.is_empty() {
        return Ok(());
    }

    let plans = Arc::new(dataset.plans);
    let next_plan = AtomicUsize::new(0);
    let (sender, receiver) = mpsc::sync_channel(loader_workers.max(1));
    let phase = split_label(split);

    thread::scope(|scope| {
        for _ in 0..loader_workers {
            let sender = sender.clone();
            let plans = Arc::clone(&plans);
            let next_plan = &next_plan;
            scope.spawn(move || {
                produce_host_loader_batches(batcher, plans, next_plan, sender);
            });
        }
        drop(sender);

        if device_prefetch_batches > 0 {
            let (device_sender, device_receiver) = mpsc::sync_channel(device_prefetch_batches);
            let device = device.clone();
            scope.spawn(move || {
                upload_host_loader_batches::<B>(receiver, batcher, device, device_sender);
            });
            consume_device_loader_batches(
                device_receiver,
                phase,
                loader_profile_every,
                &mut consume,
            )
        } else {
            consume_host_loader_batches(
                receiver,
                batcher,
                device,
                phase,
                loader_profile_every,
                &mut consume,
            )
        }
    })
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn for_each_batch_sync<B, F>(context: BatchIterationContext<'_, B>, mut consume: F) -> AppResult<()>
where
    B: Backend,
    F: FnMut(MoleculeAutoencoderBatch<B>, usize, Duration) -> AppResult<BatchControl>,
{
    let BatchIterationContext {
        shards,
        split,
        batcher,
        device,
        batch_size,
        max_batches,
        loader_profile_every,
        ..
    } = context;

    let mut pending = Vec::with_capacity(batch_size);
    let mut batches = 0_usize;
    let mut data_start = Instant::now();
    let mut shard_read_pending = Duration::ZERO;
    let mut row_scan_pending = Duration::ZERO;
    let mut loader_profile = LoaderProfileReporter::new(split_label(split), loader_profile_every);

    'shards: for shard_info in shards {
        let shard_path = &shard_info.path;
        let read_start = Instant::now();
        let shard = SparseMoleculeShard::read_from_path(shard_path)?;
        shard_read_pending += read_start.elapsed();
        validate_shard_shape(&shard, batcher, shard_path)?;
        let mut scan_start = Instant::now();
        for row_index in 0..shard.len() {
            let row = shard
                .row(row_index)
                .ok_or_else(|| invalid_input("shard row disappeared during iteration"))?;
            if row.split != split {
                continue;
            }
            pending.push(sample_from_row(row));
            if pending.len() == batch_size {
                row_scan_pending += scan_start.elapsed();
                let items = std::mem::replace(&mut pending, Vec::with_capacity(batch_size));
                let (batch, build_profile) = batcher.batch_profiled(items, device);
                let profile = load_profile(
                    std::mem::take(&mut shard_read_pending),
                    std::mem::take(&mut row_scan_pending),
                    build_profile,
                );
                let data_time = data_start.elapsed();
                loader_profile.record(data_time, batch_size, profile);
                if consume(batch, batch_size, data_time)? == BatchControl::Stop {
                    break 'shards;
                }
                batches += 1;
                if max_batches.is_some_and(|limit| batches >= limit) {
                    break 'shards;
                }
                data_start = Instant::now();
                scan_start = Instant::now();
            }
        }
        row_scan_pending += scan_start.elapsed();
    }

    if !pending.is_empty() && max_batches.is_none_or(|limit| batches < limit) {
        let rows = pending.len();
        let (batch, build_profile) = batcher.batch_profiled(pending, device);
        let profile = load_profile(shard_read_pending, row_scan_pending, build_profile);
        let data_time = data_start.elapsed();
        loader_profile.record(data_time, rows, profile);
        let _ = consume(batch, rows, data_time)?;
    }

    Ok(())
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn load_profile(
    shard_read: Duration,
    row_scan: Duration,
    build_profile: MoleculeBatchBuildProfile,
) -> BatchLoadProfile {
    BatchLoadProfile {
        shard_read,
        row_scan,
        sparse_allocation: build_profile.sparse_allocation,
        sparse_fill: build_profile.sparse_fill,
        tensor_build: build_profile.tensor_build,
        producer_total: shard_read + row_scan + build_profile.total(),
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
impl CachedBatchPlanDataset {
    fn new(
        shards: &[CachedShardInfo],
        split: DataSplit,
        batch_size: usize,
        max_batches: Option<usize>,
        validation_per_mille: u16,
    ) -> AppResult<Self> {
        let rows_per_plan = source_rows_per_batch(split, batch_size, validation_per_mille)?;
        let mut plans = Vec::new();
        for shard in shards {
            let mut start_row = 0_usize;
            while start_row < shard.row_count {
                let end_row = start_row.saturating_add(rows_per_plan).min(shard.row_count);
                plans.push(CachedBatchPlan {
                    shard: shard.clone(),
                    start_row,
                    end_row,
                    split,
                });
                if max_batches.is_some_and(|limit| plans.len() >= limit) {
                    return Ok(Self { plans });
                }
                start_row = end_row;
            }
        }
        Ok(Self { plans })
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn produce_host_loader_batches(
    batcher: MoleculeAutoencoderBatcher,
    plans: Arc<Vec<CachedBatchPlan>>,
    next_plan: &AtomicUsize,
    sender: std::sync::mpsc::SyncSender<HostLoaderMessage>,
) {
    loop {
        let index = next_plan.fetch_add(1, Ordering::Relaxed);
        let Some(plan) = plans.get(index).cloned() else {
            return;
        };
        match build_cached_loader_batch(batcher, plan) {
            Ok(Some(batch)) => {
                if sender
                    .send(HostLoaderMessage::Batch(Box::new(batch)))
                    .is_err()
                {
                    return;
                }
            }
            Ok(None) => {}
            Err(message) => {
                let _ = sender.send(HostLoaderMessage::Error(message));
                return;
            }
        }
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn upload_host_loader_batches<B>(
    receiver: Receiver<HostLoaderMessage>,
    batcher: MoleculeAutoencoderBatcher,
    device: B::Device,
    sender: std::sync::mpsc::SyncSender<DeviceLoaderMessage<B>>,
) where
    B: Backend,
    MoleculeAutoencoderBatch<B>: Send,
{
    loop {
        match receiver.recv() {
            Ok(HostLoaderMessage::Batch(loader_batch)) => {
                let loader_batch = *loader_batch;
                let (batch, tensor_profile) =
                    batcher.batch_host_profiled(loader_batch.host, &device);
                let mut profile = loader_batch.profile;
                profile.tensor_build += tensor_profile.tensor_build;
                profile.producer_total += tensor_profile.tensor_build;
                if sender
                    .send(DeviceLoaderMessage::Batch(DeviceLoaderBatch {
                        batch,
                        rows: loader_batch.rows,
                        profile,
                    }))
                    .is_err()
                {
                    return;
                }
            }
            Ok(HostLoaderMessage::Error(message)) => {
                let _ = sender.send(DeviceLoaderMessage::Error(message));
                return;
            }
            Err(_) => return,
        }
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn consume_device_loader_batches<B, F>(
    receiver: Receiver<DeviceLoaderMessage<B>>,
    phase: &'static str,
    loader_profile_every: usize,
    consume: &mut F,
) -> AppResult<()>
where
    B: Backend,
    F: FnMut(MoleculeAutoencoderBatch<B>, usize, Duration) -> AppResult<BatchControl>,
{
    let mut loader_profile = LoaderProfileReporter::new(phase, loader_profile_every);
    loop {
        let wait_start = Instant::now();
        match receiver.recv() {
            Ok(DeviceLoaderMessage::Batch(loader_batch)) => {
                let data_wait = wait_start.elapsed();
                loader_profile.record(data_wait, loader_batch.rows, loader_batch.profile);
                if consume(loader_batch.batch, loader_batch.rows, data_wait)? == BatchControl::Stop
                {
                    drop(receiver);
                    return Ok(());
                }
            }
            Ok(DeviceLoaderMessage::Error(message)) => {
                drop(receiver);
                return Err(invalid_input(message));
            }
            Err(_) => return Ok(()),
        }
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn consume_host_loader_batches<B, F>(
    receiver: Receiver<HostLoaderMessage>,
    batcher: MoleculeAutoencoderBatcher,
    device: &B::Device,
    phase: &'static str,
    loader_profile_every: usize,
    consume: &mut F,
) -> AppResult<()>
where
    B: Backend,
    F: FnMut(MoleculeAutoencoderBatch<B>, usize, Duration) -> AppResult<BatchControl>,
{
    let mut loader_profile = LoaderProfileReporter::new(phase, loader_profile_every);
    loop {
        let wait_start = Instant::now();
        match receiver.recv() {
            Ok(HostLoaderMessage::Batch(loader_batch)) => {
                let loader_batch = *loader_batch;
                let data_wait = wait_start.elapsed();
                let upload_start = Instant::now();
                let (batch, tensor_profile) =
                    batcher.batch_host_profiled(loader_batch.host, device);
                let upload_time = upload_start.elapsed();
                let mut profile = loader_batch.profile;
                profile.tensor_build += tensor_profile.tensor_build;
                profile.producer_total += tensor_profile.tensor_build;
                let data_time = data_wait + upload_time;
                loader_profile.record(data_time, loader_batch.rows, profile);
                if consume(batch, loader_batch.rows, data_time)? == BatchControl::Stop {
                    drop(receiver);
                    return Ok(());
                }
            }
            Ok(HostLoaderMessage::Error(message)) => {
                drop(receiver);
                return Err(invalid_input(message));
            }
            Err(_) => return Ok(()),
        }
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn build_cached_loader_batch(
    batcher: MoleculeAutoencoderBatcher,
    plan: CachedBatchPlan,
) -> Result<Option<CachedLoaderBatch>, String> {
    let read_start = Instant::now();
    let shard =
        SparseMoleculeShard::read_range_from_path(&plan.shard.path, plan.start_row, plan.end_row)
            .map_err(|source| source.to_string())?;
    let shard_read = read_start.elapsed();
    validate_shard_shape(&shard, batcher, &plan.shard.path).map_err(|source| source.to_string())?;

    let mut pending = Vec::with_capacity(shard.len());
    let row_scan_start = Instant::now();
    for row_index in 0..shard.len() {
        let row = shard
            .row(row_index)
            .ok_or_else(|| "shard row disappeared during dataloader batching".to_string())?;
        if row.split == plan.split {
            pending.push(sample_from_row(row));
        }
    }
    let row_scan = row_scan_start.elapsed();

    if pending.is_empty() {
        return Ok(None);
    }

    let rows = pending.len();
    let (host, build_profile) = batcher.host_batch_profiled(pending);
    Ok(Some(CachedLoaderBatch {
        host,
        rows,
        profile: load_profile(shard_read, row_scan, build_profile),
    }))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn source_rows_per_batch(
    split: DataSplit,
    batch_size: usize,
    validation_per_mille: u16,
) -> AppResult<usize> {
    let validation_per_mille = usize::from(validation_per_mille.min(1000));
    let selected_per_mille = match split {
        DataSplit::Train => 1000_usize.saturating_sub(validation_per_mille),
        DataSplit::Validation => validation_per_mille,
    };
    if selected_per_mille == 0 {
        return Err(invalid_input(format!(
            "no rows are assigned to the {} split",
            split_label(split)
        )));
    }
    Ok(batch_size
        .saturating_mul(1000)
        .div_ceil(selected_per_mille)
        .max(1))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
struct LoaderProfileReporter {
    phase: &'static str,
    every: usize,
    summary: LoaderProfileSummary,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Copy, Default)]
struct LoaderProfileSummary {
    batches: usize,
    rows: usize,
    wait: Duration,
    profile: BatchLoadProfile,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
impl LoaderProfileReporter {
    fn new(phase: &'static str, every: usize) -> Self {
        Self {
            phase,
            every,
            summary: LoaderProfileSummary::default(),
        }
    }

    fn record(&mut self, wait: Duration, rows: usize, profile: BatchLoadProfile) {
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

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
impl Drop for LoaderProfileReporter {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn split_label(split: DataSplit) -> &'static str {
    match split {
        DataSplit::Train => "train",
        DataSplit::Validation => "valid",
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn should_sample_metrics(batch_index: usize, every: usize) -> bool {
    batch_index == 1 || (every != 0 && batch_index.is_multiple_of(every))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn loss_metrics<B>(
    loss: Tensor<B, 1>,
    losses: &MoleculeLossBreakdown<B>,
) -> AppResult<BatchLossMetrics>
where
    B: Backend<FloatElem = f32>,
{
    let data = Transaction::default()
        .register(loss)
        .register(losses.reconstruction.clone())
        .register(losses.descriptors.clone())
        .register(losses.tanimoto_ranking.clone())
        .register(losses.tanimoto_ranking_accuracy.clone())
        .register(losses.tanimoto_ranking_pairs.clone())
        .try_execute()
        .map_err(|source| invalid_input(format!("failed to read training metrics: {source}")))?;
    if data.len() != 6 {
        return Err(invalid_input(format!(
            "training metrics transaction returned {} tensors instead of 6",
            data.len()
        )));
    }
    Ok(BatchLossMetrics {
        loss: scalar_data(&data[0], "loss")?,
        reconstruction: scalar_data(&data[1], "reconstruction loss")?,
        descriptors: scalar_data(&data[2], "descriptor loss")?,
        tanimoto_ranking: scalar_data(&data[3], "Tanimoto geometry loss")?,
        tanimoto_ranking_accuracy: scalar_data(&data[4], "Tanimoto geometry accuracy")?,
        tanimoto_ranking_pairs: scalar_data(&data[5], "Tanimoto geometry pairs")?,
    })
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn validation_metrics<B>(
    loss: Tensor<B, 1>,
    losses: &MoleculeLossBreakdown<B>,
    tanimoto: Tensor<B, 1>,
) -> AppResult<(BatchLossMetrics, f32)>
where
    B: Backend<FloatElem = f32>,
{
    let data = Transaction::default()
        .register(loss)
        .register(losses.reconstruction.clone())
        .register(losses.descriptors.clone())
        .register(losses.tanimoto_ranking.clone())
        .register(losses.tanimoto_ranking_accuracy.clone())
        .register(losses.tanimoto_ranking_pairs.clone())
        .register(tanimoto)
        .try_execute()
        .map_err(|source| invalid_input(format!("failed to read validation metrics: {source}")))?;
    if data.len() != 7 {
        return Err(invalid_input(format!(
            "validation metrics transaction returned {} tensors instead of 7",
            data.len()
        )));
    }
    Ok((
        BatchLossMetrics {
            loss: scalar_data(&data[0], "loss")?,
            reconstruction: scalar_data(&data[1], "reconstruction loss")?,
            descriptors: scalar_data(&data[2], "descriptor loss")?,
            tanimoto_ranking: scalar_data(&data[3], "Tanimoto geometry loss")?,
            tanimoto_ranking_accuracy: scalar_data(&data[4], "Tanimoto geometry accuracy")?,
            tanimoto_ranking_pairs: scalar_data(&data[5], "Tanimoto geometry pairs")?,
        },
        scalar_data(&data[6], "count Tanimoto")?,
    ))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn scalar_data(data: &TensorData, label: &str) -> AppResult<f32> {
    let values = data
        .as_slice::<f32>()
        .map_err(|source| invalid_input(format!("{label} tensor is not f32: {source}")))?;
    values
        .first()
        .copied()
        .ok_or_else(|| invalid_input(format!("{label} tensor did not contain an f32 scalar")))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn save_checkpoint<B, O>(
    checkpoint_dir: &Path,
    recorder: &DefaultRecorder,
    model: &MoleculeAutoencoder<B>,
    optimizer: &O,
    state: &CheckpointState,
) -> AppResult<()>
where
    B: AutodiffBackend,
    O: Optimizer<MoleculeAutoencoder<B>, B>,
{
    model
        .clone()
        .save_file(checkpoint_dir.join("model"), recorder)?;
    <DefaultRecorder as Recorder<B>>::record(
        recorder,
        optimizer.to_record(),
        checkpoint_dir.join("optimizer"),
    )?;
    write_json(&checkpoint_dir.join("state.json"), state)?;
    Ok(())
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn shard_infos(manifest_path: &Path, manifest: &ShardManifest) -> AppResult<Vec<CachedShardInfo>> {
    if manifest.shards.is_empty() {
        return Err(invalid_input("manifest does not list any shards"));
    }
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(manifest
        .shards
        .iter()
        .map(|entry| CachedShardInfo {
            path: manifest_dir.join(&entry.path),
            row_count: entry.row_count,
        })
        .collect())
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn prepare_manifest(args: &Args) -> AppResult<PathBuf> {
    let Some(cache_dir) = &args.cache_dir else {
        return Ok(args.manifest_path.clone());
    };
    let Some(source_selection) = args.source_selection else {
        return Ok(args.manifest_path.clone());
    };
    let manifest_path = cache_dir.join("manifest.json");
    let expected_source = source_selection.manifest_source();

    if manifest_path.exists() && !args.force_preprocess {
        let manifest = ShardManifest::read_from_path(&manifest_path)?;
        if manifest.manifest_version == SHARD_MANIFEST_VERSION && manifest.source == expected_source
        {
            println!(
                "using_cached_manifest={} source={}",
                manifest_path.display(),
                manifest.source
            );
            return Ok(manifest_path);
        }
        if manifest.manifest_version == SHARD_MANIFEST_VERSION {
            println!(
                "ignoring_cached_manifest={} reason=source_mismatch found={} expected={}",
                manifest_path.display(),
                manifest.source,
                expected_source
            );
        } else {
            println!(
                "ignoring_cached_manifest={} reason=manifest_version_mismatch found={} expected={}",
                manifest_path.display(),
                manifest.manifest_version,
                SHARD_MANIFEST_VERSION
            );
        }
    }

    preprocess_cache(
        cache_dir,
        args.rows_per_shard,
        args.preprocess_threads,
        source_selection,
    )?;
    Ok(manifest_path)
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "datasets",
    any(feature = "cuda", feature = "ndarray")
))]
fn preprocess_cache(
    cache_dir: &Path,
    rows_per_shard: usize,
    preprocess_threads: usize,
    source_selection: SourceSelection,
) -> AppResult<()> {
    if rows_per_shard == 0 {
        return Err(invalid_input("rows per shard must be greater than zero"));
    }

    std::fs::create_dir_all(cache_dir)?;
    let source_cache_dir = cache_dir.join("source");
    let config = PreprocessingConfig::default();
    let mut manifest = ShardManifest::new(source_selection.manifest_source(), config.clone());
    let mut shard = SparseMoleculeShard::new(config.counted_ecfp.size, REGRESSION_TARGET_WIDTH);
    let mut shard_index = 0_usize;
    let mut records_seen_total = 0_usize;
    let start = Instant::now();

    println!(
        "preprocess_start source={} cache_dir={} rows_per_shard={} chunk_rows={} threads={}",
        source_selection.manifest_source(),
        cache_dir.display(),
        rows_per_shard,
        DEFAULT_PREPROCESS_CHUNK_ROWS,
        preprocess_threads
    );

    let progress = PreprocessProgress::new(
        cache_dir,
        source_selection,
        rows_per_shard,
        preprocess_threads,
    );
    {
        let mut context = PreprocessContext {
            cache_dir,
            source_cache_dir: &source_cache_dir,
            rows_per_shard,
            preprocess_threads,
            config: &config,
            manifest: &mut manifest,
            shard: &mut shard,
            shard_index: &mut shard_index,
            progress: &progress,
            records_seen_total: &mut records_seen_total,
        };
        match source_selection {
            SourceSelection::PubChem => context.preprocess_pubchem_source()?,
            SourceSelection::Zinc20 {
                first_chunk,
                last_chunk,
            } => context.preprocess_zinc20_source(first_chunk, last_chunk)?,
            SourceSelection::PubChemAndZinc20 {
                first_chunk,
                last_chunk,
            } => {
                context.preprocess_pubchem_source()?;
                context.preprocess_zinc20_source(first_chunk, last_chunk)?;
            }
        }
    }
    if !shard.is_empty() {
        write_preprocessed_shard(cache_dir, shard_index, &mut manifest, &shard)?;
        progress.shard_written(&manifest, 0);
    }
    manifest.write_to_path(cache_dir.join("manifest.json"))?;
    progress.finish(records_seen_total, &manifest, start.elapsed());
    Ok(())
}

#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "datasets"),
    any(feature = "cuda", feature = "ndarray")
))]
fn preprocess_cache(
    _cache_dir: &Path,
    _rows_per_shard: usize,
    _preprocess_threads: usize,
    _source_selection: SourceSelection,
) -> AppResult<()> {
    Err(invalid_input(
        "source input requires the `datasets` feature; rebuild with `--features std,cuda-fusion,train,datasets`",
    ))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "datasets",
    any(feature = "cuda", feature = "ndarray")
))]
struct PreprocessContext<'a> {
    cache_dir: &'a Path,
    source_cache_dir: &'a Path,
    rows_per_shard: usize,
    preprocess_threads: usize,
    config: &'a PreprocessingConfig,
    manifest: &'a mut ShardManifest,
    shard: &'a mut SparseMoleculeShard,
    shard_index: &'a mut usize,
    progress: &'a PreprocessProgress,
    records_seen_total: &'a mut usize,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "datasets",
    any(feature = "cuda", feature = "ndarray")
))]
impl PreprocessContext<'_> {
    fn preprocess_pubchem_source(&mut self) -> AppResult<()> {
        let source_progress = spinner(format!(
            "opening PubChem records cache at {}",
            self.source_cache_dir.display()
        ));
        let records = PUBCHEM_SMILES.iter_records_with_options(&DatasetFetchOptions {
            cache_dir: Some(self.source_cache_dir.to_path_buf()),
            gzip_mode: GzipMode::KeepCompressed,
            ..DatasetFetchOptions::default()
        })?;
        source_progress.finish_with_message(format!(
            "PubChem records ready source={}",
            PUBCHEM_SMILES.url()
        ));
        let records = molecule_records_from_smiles_dataset(PUBCHEM_SMILES.id(), records);
        self.preprocess_records("PubChem", records)
    }

    fn preprocess_zinc20_source(&mut self, first_chunk: u8, last_chunk: u8) -> AppResult<()> {
        let dataset = Zinc20Smiles::chunk_range(first_chunk, last_chunk)?;
        let label = format!("ZINC20 chunks {first_chunk}..={last_chunk}");
        let source_progress = spinner(format!(
            "opening {label} records cache at {}",
            self.source_cache_dir.display()
        ));
        let records = dataset.iter_records_with_options(&DatasetFetchOptions {
            cache_dir: Some(self.source_cache_dir.to_path_buf()),
            gzip_mode: GzipMode::Decompress,
            ..DatasetFetchOptions::default()
        })?;
        source_progress.finish_with_message(format!("{label} records ready"));
        let records = molecule_records_from_smiles_dataset(dataset.id(), records);
        self.preprocess_records(&label, records)
    }

    fn preprocess_records<I>(&mut self, dataset_label: &str, records: I) -> AppResult<()>
    where
        I: IntoIterator<Item = molecular_autoencoder::Result<MoleculeRecord>>,
    {
        let source_records_seen = preprocess_dataset_record_chunks(
            records,
            self.config,
            DatasetPreprocessOptions {
                chunk_rows: DEFAULT_PREPROCESS_CHUNK_ROWS,
                threads: Some(self.preprocess_threads),
            },
            |chunk| {
                for result in chunk.results {
                    match result {
                        Ok(targets) => {
                            self.shard.push_targets(&targets, self.config.descriptors)?;
                            if self.shard.len() >= self.rows_per_shard {
                                write_preprocessed_shard(
                                    self.cache_dir,
                                    *self.shard_index,
                                    self.manifest,
                                    self.shard,
                                )?;
                                self.progress.shard_written(self.manifest, 0);
                                *self.shard = SparseMoleculeShard::new(
                                    self.config.counted_ecfp.size,
                                    REGRESSION_TARGET_WIDTH,
                                );
                                *self.shard_index += 1;
                            }
                        }
                        Err(error) => {
                            self.manifest.error_count += 1;
                            eprintln!("{error}");
                        }
                    }
                }
                self.progress.record(
                    dataset_label,
                    *self.records_seen_total + chunk.records_seen,
                    self.manifest,
                    self.shard.len(),
                );
                Ok(())
            },
        )?;
        *self.records_seen_total += source_records_seen;
        Ok(())
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "datasets",
    any(feature = "cuda", feature = "ndarray")
))]
fn write_preprocessed_shard(
    cache_dir: &Path,
    shard_index: usize,
    manifest: &mut ShardManifest,
    shard: &SparseMoleculeShard,
) -> molecular_autoencoder::Result<()> {
    let filename = format!("shard-{shard_index:05}.maeshard");
    let path = cache_dir.join(&filename);
    shard.write_to_path(&path)?;
    manifest.push_shard(filename, shard);
    Ok(())
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn validate_model_config(
    config: &MoleculeAutoencoderConfig,
    fingerprint_size: usize,
) -> AppResult<()> {
    if config.encoder.input_width != fingerprint_size
        || config.decoder.output_width != fingerprint_size
    {
        return Err(invalid_input(format!(
            "model fingerprint width does not match manifest width {fingerprint_size}"
        )));
    }
    if !(config.auxiliary_weights.descriptors.is_finite()
        && config.auxiliary_weights.descriptors >= 0.0)
    {
        return Err(invalid_input(
            "descriptor weight must be finite and non-negative",
        ));
    }
    if !(config.auxiliary_weights.tanimoto_ranking.is_finite()
        && config.auxiliary_weights.tanimoto_ranking >= 0.0)
    {
        return Err(invalid_input(
            "Tanimoto geometry weight must be finite and non-negative",
        ));
    }
    if !(config.latent_noise_std.is_finite() && config.latent_noise_std >= 0.0) {
        return Err(invalid_input(
            "latent noise std must be finite and non-negative",
        ));
    }
    if !(config.tanimoto_ranking.latent_temperature.is_finite()
        && config.tanimoto_ranking.latent_temperature > 0.0)
    {
        return Err(invalid_input(
            "Tanimoto geometry latent temperature must be finite and positive",
        ));
    }
    if !(config.tanimoto_ranking.metric_temperature.is_finite()
        && config.tanimoto_ranking.metric_temperature > 0.0)
    {
        return Err(invalid_input(
            "Tanimoto geometry metric temperature must be finite and positive",
        ));
    }
    if !(config.tanimoto_ranking.min_gap.is_finite() && config.tanimoto_ranking.min_gap >= 0.0) {
        return Err(invalid_input(
            "Tanimoto geometry min gap must be finite and non-negative",
        ));
    }
    if config.auxiliary_weights.tanimoto_ranking > 0.0
        && config.tanimoto_ranking.candidates_per_anchor < 2
    {
        return Err(invalid_input(
            "Tanimoto geometry candidates must be at least 2 when enabled",
        ));
    }
    Ok(())
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn validate_shard_shape(
    shard: &SparseMoleculeShard,
    batcher: MoleculeAutoencoderBatcher,
    path: &Path,
) -> AppResult<()> {
    if shard.fingerprint_size != batcher.fingerprint_size {
        return Err(invalid_input(format!(
            "{} fingerprint width {} does not match model width {}",
            path.display(),
            shard.fingerprint_size,
            batcher.fingerprint_size
        )));
    }
    if shard.descriptor_width != batcher.descriptor_width {
        return Err(invalid_input(format!(
            "{} descriptor width {} does not match model width {}",
            path.display(),
            shard.descriptor_width,
            batcher.descriptor_width
        )));
    }
    Ok(())
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn sample_from_row(row: MoleculeShardRow<'_>) -> MoleculeAutoencoderSample {
    MoleculeAutoencoderSample {
        cid: row.cid,
        fingerprint_indices: row.fingerprint_indices.to_vec(),
        fingerprint_counts: row.fingerprint_counts.to_vec(),
        descriptor_targets: row.descriptor_targets.to_vec(),
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn read_json<T>(path: &Path) -> AppResult<T>
where
    T: for<'de> Deserialize<'de>,
{
    Ok(serde_json::from_reader(File::open(path)?)?)
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn write_json<T>(path: &Path, value: &T) -> AppResult<()>
where
    T: Serialize,
{
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, value)?;
    Ok(())
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn invalid_input(message: impl Into<String>) -> Box<dyn StdError> {
    Box::new(molecular_autoencoder::Error::InvalidBatch(message.into()))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn examples_per_second(examples: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        examples as f64 / elapsed.as_secs_f64()
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn millis_per_batch(duration: Duration, batches: usize) -> f64 {
    if batches == 0 {
        0.0
    } else {
        duration.as_secs_f64() * 1000.0 / batches as f64
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "datasets",
    any(feature = "cuda", feature = "ndarray")
))]
fn spinner(message: impl Into<String>) -> ProgressBar {
    let bar = ProgressBar::new_spinner();
    bar.set_style(spinner_style());
    bar.enable_steady_tick(Duration::from_millis(100));
    bar.set_message(message.into());
    bar
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "datasets",
    any(feature = "cuda", feature = "ndarray")
))]
fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} {elapsed_precise} {wide_msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {wide_msg}",
    )
    .map_or_else(
        |_| ProgressStyle::default_bar(),
        |style| style.progress_chars("=> "),
    )
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "datasets",
    any(feature = "cuda", feature = "ndarray")
))]
struct PreprocessProgress {
    bar: ProgressBar,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "datasets",
    any(feature = "cuda", feature = "ndarray")
))]
impl PreprocessProgress {
    fn new(
        cache_dir: &Path,
        source_selection: SourceSelection,
        rows_per_shard: usize,
        threads: usize,
    ) -> Self {
        let bar = spinner(format!(
            "preprocessing {} into {} rows_per_shard={rows_per_shard} chunk_rows={} threads={}",
            source_selection.manifest_source(),
            cache_dir.display(),
            DEFAULT_PREPROCESS_CHUNK_ROWS,
            threads
        ));
        Self { bar }
    }

    fn record(
        &self,
        dataset_label: &str,
        records_seen: usize,
        manifest: &ShardManifest,
        current_shard_rows: usize,
    ) {
        if records_seen == 1 || records_seen.is_multiple_of(1024) {
            self.bar.set_message(format!(
                "preprocessing {dataset_label} records={} rows={} errors={} shards={} current_shard_rows={}",
                records_seen,
                manifest.row_count + current_shard_rows,
                manifest.error_count,
                manifest.shards.len(),
                current_shard_rows
            ));
        }
    }

    fn shard_written(&self, manifest: &ShardManifest, current_shard_rows: usize) {
        self.bar.set_message(format!(
            "wrote shard={} rows={} errors={} next_shard_rows={}",
            manifest.shards.len(),
            manifest.row_count,
            manifest.error_count,
            current_shard_rows
        ));
    }

    fn finish(self, records_seen: usize, manifest: &ShardManifest, elapsed: Duration) {
        self.bar.finish_with_message(format!(
            "preprocess_done records_seen={} rows={} errors={} shards={} elapsed_sec={:.2}",
            records_seen,
            manifest.row_count,
            manifest.error_count,
            manifest.shards.len(),
            elapsed.as_secs_f64()
        ));
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
struct IndicatifTrainingBars {
    bar: Option<ProgressBar>,
    phase: Option<ProgressPhase>,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProgressPhase {
    kind: ProgressKind,
    epoch: usize,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressKind {
    Train,
    Valid,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
impl IndicatifTrainingBars {
    fn new() -> Self {
        Self {
            bar: None,
            phase: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn train_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        processed: usize,
        total: usize,
        metrics: Option<BatchLossMetrics>,
        data_time: Duration,
        step_time: Duration,
    ) {
        let bar = self.phase_bar(ProgressKind::Train, epoch, epoch_total, total);
        bar.set_position(processed as u64);
        let data_ms = data_time.as_secs_f64() * 1000.0;
        let step_ms = step_time.as_secs_f64() * 1000.0;
        match metrics {
            Some(metrics) => bar.set_message(format!(
                "loss={:.4} recon={:.4} desc={:.4} tanrank={:.4} acc={:.3} pairs={:.0} data_ms={data_ms:.1} step_ms={step_ms:.1}",
                metrics.loss,
                metrics.reconstruction,
                metrics.descriptors,
                metrics.tanimoto_ranking,
                metrics.tanimoto_ranking_accuracy,
                metrics.tanimoto_ranking_pairs,
            )),
            None => bar.set_message(format!("data_ms={data_ms:.1} step_ms={step_ms:.1}")),
        }
    }

    fn valid_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        processed: usize,
        total: usize,
        metrics: Option<BatchLossMetrics>,
        tanimoto: f32,
    ) {
        let bar = self.phase_bar(ProgressKind::Valid, epoch, epoch_total, total);
        bar.set_position(processed as u64);
        match metrics {
            Some(metrics) => bar.set_message(format!(
                "loss={:.4} recon={:.4} desc={:.4} tanrank={:.4} acc={:.3} pairs={:.0} tanimoto={tanimoto:.4}",
                metrics.loss,
                metrics.reconstruction,
                metrics.descriptors,
                metrics.tanimoto_ranking,
                metrics.tanimoto_ranking_accuracy,
                metrics.tanimoto_ranking_pairs,
            )),
            None => bar.set_message(format!("tanimoto={tanimoto:.4}")),
        }
    }

    fn finish(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
        self.phase = None;
    }

    fn phase_bar(
        &mut self,
        kind: ProgressKind,
        epoch: usize,
        epoch_total: usize,
        total: usize,
    ) -> &ProgressBar {
        let phase = ProgressPhase { kind, epoch };
        if self.phase != Some(phase) {
            self.finish();
            let bar = ProgressBar::new(total as u64);
            bar.set_style(bar_style());
            bar.set_prefix(match kind {
                ProgressKind::Train => format!("train {epoch}/{epoch_total}"),
                ProgressKind::Valid => format!("valid {epoch}/{epoch_total}"),
            });
            self.bar = Some(bar);
            self.phase = Some(phase);
        }

        let Some(bar) = self.bar.as_ref() else {
            panic!("phase bar is initialized before use");
        };
        bar
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
struct TrainingReporter {
    renderer: Option<Box<dyn burn::train::renderer::MetricsRenderer>>,
    bars: Option<IndicatifTrainingBars>,
    interrupter: burn::train::Interrupter,
    metric_ids: ReporterMetricIds,
    train_total: usize,
    valid_total: usize,
    train_processed: usize,
    valid_processed: usize,
    train_epoch: Option<usize>,
    valid_epoch: Option<usize>,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
struct ReporterMetricIds {
    loss: burn::train::metric::MetricId,
    reconstruction: burn::train::metric::MetricId,
    descriptors: burn::train::metric::MetricId,
    tanimoto_ranking: burn::train::metric::MetricId,
    tanimoto_ranking_accuracy: burn::train::metric::MetricId,
    tanimoto_ranking_pairs: burn::train::metric::MetricId,
    tanimoto: burn::train::metric::MetricId,
    data_wait_ms: burn::train::metric::MetricId,
    step_ms: burn::train::metric::MetricId,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
impl TrainingReporter {
    fn new(
        row_count: usize,
        validation_per_mille: u16,
        batch_size: usize,
        max_train_batches: Option<usize>,
        max_valid_batches: Option<usize>,
        checkpoint: Option<usize>,
    ) -> Self {
        let interrupter = burn::train::Interrupter::new();
        let use_tui = std::io::stdout().is_terminal();
        let mut renderer = use_tui.then(|| {
            Box::new(burn::train::renderer::tui::TuiMetricsRendererWrapper::new(
                interrupter.clone(),
                checkpoint,
            )) as Box<dyn burn::train::renderer::MetricsRenderer>
        });
        let bars = (!use_tui).then(IndicatifTrainingBars::new);
        let metric_ids = ReporterMetricIds::new();

        if let Some(renderer) = renderer.as_mut() {
            for metric in ReporterMetric::ALL {
                renderer.register_metric(metric.definition(metric_ids.id(metric)));
            }
        }

        Self {
            renderer,
            bars,
            interrupter,
            metric_ids,
            train_total: progress_total(
                train_row_estimate(row_count, validation_per_mille),
                batch_size,
                max_train_batches,
            ),
            valid_total: progress_total(
                valid_row_estimate(row_count, validation_per_mille),
                batch_size,
                max_valid_batches,
            ),
            train_processed: 0,
            valid_processed: 0,
            train_epoch: None,
            valid_epoch: None,
        }
    }

    fn is_active(&self) -> bool {
        self.renderer.is_some()
    }

    fn should_stop(&self) -> bool {
        self.interrupter.should_stop()
    }

    #[allow(clippy::too_many_arguments)]
    fn train_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        iteration: usize,
        examples: usize,
        metrics: Option<BatchLossMetrics>,
        data_time: Duration,
        step_time: Duration,
    ) -> BatchControl {
        if self.train_epoch != Some(epoch) {
            self.train_epoch = Some(epoch);
            self.train_processed = 0;
        }
        self.train_processed = self
            .train_processed
            .saturating_add(examples)
            .min(self.train_total);

        if let Some(renderer) = self.renderer.as_mut() {
            if let Some(metrics) = metrics {
                renderer.update_train(metric_state(
                    self.metric_ids.loss.clone(),
                    metrics.loss,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.reconstruction.clone(),
                    metrics.reconstruction,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.descriptors.clone(),
                    metrics.descriptors,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.tanimoto_ranking.clone(),
                    metrics.tanimoto_ranking,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.tanimoto_ranking_accuracy.clone(),
                    metrics.tanimoto_ranking_accuracy,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.tanimoto_ranking_pairs.clone(),
                    metrics.tanimoto_ranking_pairs,
                    1,
                ));
            }
            renderer.update_train(metric_state(
                self.metric_ids.data_wait_ms.clone(),
                data_time.as_secs_f64() * 1000.0,
                1,
            ));
            renderer.update_train(metric_state(
                self.metric_ids.step_ms.clone(),
                step_time.as_secs_f64() * 1000.0,
                1,
            ));
            renderer.render_train(
                training_progress(
                    self.train_processed,
                    self.train_total,
                    epoch,
                    epoch_total,
                    iteration,
                ),
                progress_indicators(
                    self.train_processed,
                    self.train_total,
                    epoch,
                    epoch_total,
                    iteration,
                ),
            );
        }
        if let Some(bars) = self.bars.as_mut() {
            bars.train_batch(
                epoch,
                epoch_total,
                self.train_processed,
                self.train_total,
                metrics,
                data_time,
                step_time,
            );
        }

        if self.should_stop() {
            BatchControl::Stop
        } else {
            BatchControl::Continue
        }
    }

    fn valid_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        iteration: usize,
        examples: usize,
        metrics: Option<BatchLossMetrics>,
        tanimoto: f32,
    ) -> BatchControl {
        if self.valid_epoch != Some(epoch) {
            self.valid_epoch = Some(epoch);
            self.valid_processed = 0;
        }
        self.valid_processed = self
            .valid_processed
            .saturating_add(examples)
            .min(self.valid_total);

        if let Some(renderer) = self.renderer.as_mut() {
            if let Some(metrics) = metrics {
                renderer.update_valid(metric_state(
                    self.metric_ids.loss.clone(),
                    metrics.loss,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.reconstruction.clone(),
                    metrics.reconstruction,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.descriptors.clone(),
                    metrics.descriptors,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.tanimoto_ranking.clone(),
                    metrics.tanimoto_ranking,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.tanimoto_ranking_accuracy.clone(),
                    metrics.tanimoto_ranking_accuracy,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.tanimoto_ranking_pairs.clone(),
                    metrics.tanimoto_ranking_pairs,
                    1,
                ));
            }
            renderer.update_valid(metric_state(
                self.metric_ids.tanimoto.clone(),
                tanimoto,
                examples,
            ));
            renderer.render_valid(
                training_progress(
                    self.valid_processed,
                    self.valid_total,
                    epoch,
                    epoch_total,
                    iteration,
                ),
                progress_indicators(
                    self.valid_processed,
                    self.valid_total,
                    epoch,
                    epoch_total,
                    iteration,
                ),
            );
        }
        if let Some(bars) = self.bars.as_mut() {
            bars.valid_batch(
                epoch,
                epoch_total,
                self.valid_processed,
                self.valid_total,
                metrics,
                tanimoto,
            );
        }

        if self.should_stop() {
            BatchControl::Stop
        } else {
            BatchControl::Continue
        }
    }

    fn finish(&mut self) -> AppResult<()> {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.on_train_end(None)?;
        }
        if let Some(bars) = self.bars.as_mut() {
            bars.finish();
        }
        Ok(())
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "tui"),
    any(feature = "cuda", feature = "ndarray")
))]
struct TrainingReporter {
    bars: IndicatifTrainingBars,
    train_total: usize,
    valid_total: usize,
    train_processed: usize,
    valid_processed: usize,
    train_epoch: Option<usize>,
    valid_epoch: Option<usize>,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    not(feature = "tui"),
    any(feature = "cuda", feature = "ndarray")
))]
impl TrainingReporter {
    fn new(
        row_count: usize,
        validation_per_mille: u16,
        batch_size: usize,
        max_train_batches: Option<usize>,
        max_valid_batches: Option<usize>,
        _checkpoint: Option<usize>,
    ) -> Self {
        Self {
            bars: IndicatifTrainingBars::new(),
            train_total: progress_total(
                train_row_estimate(row_count, validation_per_mille),
                batch_size,
                max_train_batches,
            ),
            valid_total: progress_total(
                valid_row_estimate(row_count, validation_per_mille),
                batch_size,
                max_valid_batches,
            ),
            train_processed: 0,
            valid_processed: 0,
            train_epoch: None,
            valid_epoch: None,
        }
    }

    #[allow(clippy::unused_self)]
    fn is_active(&self) -> bool {
        false
    }

    #[allow(clippy::unused_self)]
    fn should_stop(&self) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn train_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        _iteration: usize,
        examples: usize,
        metrics: Option<BatchLossMetrics>,
        data_time: Duration,
        step_time: Duration,
    ) -> BatchControl {
        if self.train_epoch != Some(epoch) {
            self.train_epoch = Some(epoch);
            self.train_processed = 0;
        }
        self.train_processed = self
            .train_processed
            .saturating_add(examples)
            .min(self.train_total);
        self.bars.train_batch(
            epoch,
            epoch_total,
            self.train_processed,
            self.train_total,
            metrics,
            data_time,
            step_time,
        );
        BatchControl::Continue
    }

    fn valid_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        _iteration: usize,
        examples: usize,
        metrics: Option<BatchLossMetrics>,
        tanimoto: f32,
    ) -> BatchControl {
        if self.valid_epoch != Some(epoch) {
            self.valid_epoch = Some(epoch);
            self.valid_processed = 0;
        }
        self.valid_processed = self
            .valid_processed
            .saturating_add(examples)
            .min(self.valid_total);
        self.bars.valid_batch(
            epoch,
            epoch_total,
            self.valid_processed,
            self.valid_total,
            metrics,
            tanimoto,
        );
        BatchControl::Continue
    }

    #[allow(clippy::unnecessary_wraps)]
    fn finish(&mut self) -> AppResult<()> {
        self.bars.finish();
        Ok(())
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
#[derive(Debug, Clone, Copy)]
enum ReporterMetric {
    Loss,
    Reconstruction,
    Descriptors,
    TanimotoRanking,
    TanimotoRankingAccuracy,
    TanimotoRankingPairs,
    Tanimoto,
    DataWaitMs,
    StepMs,
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
impl ReporterMetric {
    const ALL: [Self; 9] = [
        Self::Loss,
        Self::Reconstruction,
        Self::Descriptors,
        Self::TanimotoRanking,
        Self::TanimotoRankingAccuracy,
        Self::TanimotoRankingPairs,
        Self::Tanimoto,
        Self::DataWaitMs,
        Self::StepMs,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Loss => "Loss",
            Self::Reconstruction => "Reconstruction Loss",
            Self::Descriptors => "Descriptor Loss",
            Self::TanimotoRanking => "Tanimoto Geometry Loss",
            Self::TanimotoRankingAccuracy => "Tanimoto Geometry Accuracy",
            Self::TanimotoRankingPairs => "Tanimoto Geometry Pairs",
            Self::Tanimoto => "Count Tanimoto",
            Self::DataWaitMs => "Data Wait",
            Self::StepMs => "Step Time",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Loss => "weighted total loss",
            Self::Reconstruction => "counted ECFP reconstruction loss",
            Self::Descriptors => "descriptor side-task loss",
            Self::TanimotoRanking => "latent counted-Tanimoto pairwise logistic loss",
            Self::TanimotoRankingAccuracy => "latent ordering accuracy for sampled Tanimoto pairs",
            Self::TanimotoRankingPairs => "valid sampled Tanimoto geometry anchors",
            Self::Tanimoto => "counted fingerprint Tanimoto reconstruction metric",
            Self::DataWaitMs => "time spent waiting for the next queued batch",
            Self::StepMs => "batch training or validation step time",
        }
    }

    const fn higher_is_better(self) -> bool {
        matches!(
            self,
            Self::Tanimoto | Self::TanimotoRankingAccuracy | Self::TanimotoRankingPairs
        )
    }

    fn definition(
        self,
        metric_id: burn::train::metric::MetricId,
    ) -> burn::train::metric::MetricDefinition {
        burn::train::metric::MetricDefinition {
            metric_id,
            name: self.name().to_string(),
            description: Some(self.description().to_string()),
            attributes: burn::train::metric::NumericAttributes {
                unit: self.unit().map(str::to_string),
                higher_is_better: self.higher_is_better(),
            }
            .into(),
        }
    }

    const fn unit(self) -> Option<&'static str> {
        match self {
            Self::DataWaitMs | Self::StepMs => Some("ms"),
            Self::TanimotoRankingPairs => Some("pairs"),
            _ => None,
        }
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
impl ReporterMetricIds {
    fn new() -> Self {
        Self {
            loss: metric_id(ReporterMetric::Loss),
            reconstruction: metric_id(ReporterMetric::Reconstruction),
            descriptors: metric_id(ReporterMetric::Descriptors),
            tanimoto_ranking: metric_id(ReporterMetric::TanimotoRanking),
            tanimoto_ranking_accuracy: metric_id(ReporterMetric::TanimotoRankingAccuracy),
            tanimoto_ranking_pairs: metric_id(ReporterMetric::TanimotoRankingPairs),
            tanimoto: metric_id(ReporterMetric::Tanimoto),
            data_wait_ms: metric_id(ReporterMetric::DataWaitMs),
            step_ms: metric_id(ReporterMetric::StepMs),
        }
    }

    fn id(&self, metric: ReporterMetric) -> burn::train::metric::MetricId {
        match metric {
            ReporterMetric::Loss => self.loss.clone(),
            ReporterMetric::Reconstruction => self.reconstruction.clone(),
            ReporterMetric::Descriptors => self.descriptors.clone(),
            ReporterMetric::TanimotoRanking => self.tanimoto_ranking.clone(),
            ReporterMetric::TanimotoRankingAccuracy => self.tanimoto_ranking_accuracy.clone(),
            ReporterMetric::TanimotoRankingPairs => self.tanimoto_ranking_pairs.clone(),
            ReporterMetric::Tanimoto => self.tanimoto.clone(),
            ReporterMetric::DataWaitMs => self.data_wait_ms.clone(),
            ReporterMetric::StepMs => self.step_ms.clone(),
        }
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
fn metric_id(metric: ReporterMetric) -> burn::train::metric::MetricId {
    burn::train::metric::MetricId::new(std::sync::Arc::new(metric.name().to_string()))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
fn metric_state(
    metric_id: burn::train::metric::MetricId,
    value: impl Into<f64>,
    count: usize,
) -> burn::train::renderer::MetricState {
    let value = value.into();
    let numeric = burn::train::metric::NumericEntry::Aggregated {
        aggregated_value: value,
        count,
    };
    let serialized = burn::train::metric::SerializedEntry::new(
        burn::train::metric::format_float(value, 4),
        numeric.serialize(),
    );
    burn::train::renderer::MetricState::Numeric(
        burn::train::metric::MetricEntry::new(metric_id, serialized),
        numeric,
    )
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
fn training_progress(
    processed: usize,
    total: usize,
    epoch: usize,
    epoch_total: usize,
    iteration: usize,
) -> burn::train::renderer::TrainingProgress {
    burn::train::renderer::TrainingProgress {
        progress: Some(burn::data::dataloader::Progress {
            items_processed: processed,
            items_total: total,
        }),
        global_progress: burn::data::dataloader::Progress {
            items_processed: epoch,
            items_total: epoch_total,
        },
        iteration: Some(iteration),
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    feature = "tui",
    any(feature = "cuda", feature = "ndarray")
))]
fn progress_indicators(
    processed: usize,
    total: usize,
    epoch: usize,
    epoch_total: usize,
    iteration: usize,
) -> Vec<burn::train::renderer::ProgressType> {
    vec![
        burn::train::renderer::ProgressType::Detailed {
            tag: "Items".to_string(),
            progress: burn::data::dataloader::Progress {
                items_processed: processed,
                items_total: total,
            },
        },
        burn::train::renderer::ProgressType::Detailed {
            tag: "Epoch".to_string(),
            progress: burn::data::dataloader::Progress {
                items_processed: epoch,
                items_total: epoch_total,
            },
        },
        burn::train::renderer::ProgressType::Value {
            tag: "Iteration".to_string(),
            value: iteration,
        },
    ]
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn progress_total(row_count: usize, batch_size: usize, max_batches: Option<usize>) -> usize {
    max_batches
        .map_or(row_count, |batches| batches.saturating_mul(batch_size))
        .max(1)
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn valid_row_estimate(row_count: usize, validation_per_mille: u16) -> usize {
    row_count.saturating_mul(usize::from(validation_per_mille.min(1000))) / 1000
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn train_row_estimate(row_count: usize, validation_per_mille: u16) -> usize {
    row_count.saturating_sub(valid_row_estimate(row_count, validation_per_mille))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
impl Args {
    fn parse() -> AppResult<Self> {
        let mut values = env::args().skip(1);
        let Some(first) = values.next() else {
            return Err(invalid_input(Self::usage()));
        };
        let (manifest_path, cache_dir, source_selection, checkpoint_dir) = match first.as_str() {
            "pubchem" => {
                let cache_dir = PathBuf::from(next_value(&mut values, "pubchem <cache-dir>")?);
                let checkpoint_dir =
                    PathBuf::from(next_value(&mut values, "pubchem <checkpoint-dir>")?);
                (
                    cache_dir.join("manifest.json"),
                    Some(cache_dir),
                    Some(SourceSelection::PubChem),
                    checkpoint_dir,
                )
            }
            "zinc20" => {
                let cache_dir = PathBuf::from(next_value(&mut values, "zinc20 <cache-dir>")?);
                let checkpoint_dir =
                    PathBuf::from(next_value(&mut values, "zinc20 <checkpoint-dir>")?);
                (
                    cache_dir.join("manifest.json"),
                    Some(cache_dir),
                    Some(SourceSelection::Zinc20 {
                        first_chunk: ZINC20_FIRST_CHUNK,
                        last_chunk: ZINC20_LAST_CHUNK,
                    }),
                    checkpoint_dir,
                )
            }
            "all" | "pubchem-zinc20" => {
                let cache_dir = PathBuf::from(next_value(&mut values, "all <cache-dir>")?);
                let checkpoint_dir =
                    PathBuf::from(next_value(&mut values, "all <checkpoint-dir>")?);
                (
                    cache_dir.join("manifest.json"),
                    Some(cache_dir),
                    Some(SourceSelection::PubChemAndZinc20 {
                        first_chunk: ZINC20_FIRST_CHUNK,
                        last_chunk: ZINC20_LAST_CHUNK,
                    }),
                    checkpoint_dir,
                )
            }
            "manifest" => {
                let manifest_path = PathBuf::from(next_value(&mut values, "manifest <path>")?);
                let checkpoint_dir =
                    PathBuf::from(next_value(&mut values, "manifest <checkpoint-dir>")?);
                (manifest_path, None, None, checkpoint_dir)
            }
            manifest_path => {
                let checkpoint_dir = PathBuf::from(next_value(&mut values, "<checkpoint-dir>")?);
                (PathBuf::from(manifest_path), None, None, checkpoint_dir)
            }
        };
        let mut args = Self {
            manifest_path,
            cache_dir,
            source_selection,
            checkpoint_dir,
            rows_per_shard: DEFAULT_ROWS_PER_SHARD,
            epochs: 10,
            batch_size: 4096,
            learning_rate: 1.0e-3,
            latent_width: 512,
            latent_noise_std: 0.02,
            hidden_widths: vec![4096, 2048, 1024],
            checkpoint_every: 1,
            validate_every: 1,
            max_train_batches: None,
            max_valid_batches: Some(64),
            loader_workers: DEFAULT_LOADER_WORKERS,
            device_prefetch_batches: DEFAULT_DEVICE_PREFETCH_BATCHES,
            loader_profile_every: 0,
            metric_every: DEFAULT_METRIC_EVERY,
            descriptor_weight: 0.05,
            tanimoto_ranking_weight: 0.10,
            tanimoto_ranking_latent_temperature: 0.10,
            tanimoto_ranking_metric_temperature: 0.10,
            tanimoto_ranking_min_gap: 0.05,
            tanimoto_ranking_candidates: 4,
            tanimoto_ranking_pairs_per_batch: 0,
            resume: false,
            force_preprocess: false,
            preprocess_threads: DEFAULT_PREPROCESS_THREADS,
            cuda_device: 0,
        };

        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--epochs" => args.epochs = parse_value(&mut values, &flag)?,
                "--batch-size" => args.batch_size = parse_value(&mut values, &flag)?,
                "--learning-rate" => args.learning_rate = parse_value(&mut values, &flag)?,
                "--latent-width" => args.latent_width = parse_value(&mut values, &flag)?,
                "--latent-noise-std" => {
                    args.latent_noise_std = parse_value(&mut values, &flag)?;
                }
                "--hidden-widths" => {
                    let value = next_value(&mut values, &flag)?;
                    args.hidden_widths = parse_hidden_widths(&value)?;
                }
                "--rows-per-shard" => args.rows_per_shard = parse_value(&mut values, &flag)?,
                "--zinc20-chunks" => {
                    let value = next_value(&mut values, &flag)?;
                    let (first_chunk, last_chunk) = parse_zinc20_chunks(&value)?;
                    let Some(selection) = args.source_selection else {
                        return Err(invalid_input(
                            "--zinc20-chunks is only valid with a dataset source",
                        ));
                    };
                    args.source_selection =
                        Some(selection.with_zinc20_chunks(first_chunk, last_chunk)?);
                }
                "--checkpoint-every" => args.checkpoint_every = parse_value(&mut values, &flag)?,
                "--validate-every" => args.validate_every = parse_value(&mut values, &flag)?,
                "--max-train-batches" => {
                    args.max_train_batches = Some(parse_value(&mut values, &flag)?);
                }
                "--max-valid-batches" => {
                    args.max_valid_batches = Some(parse_value(&mut values, &flag)?);
                }
                "--loader-workers" => args.loader_workers = parse_value(&mut values, &flag)?,
                "--device-prefetch-batches" => {
                    args.device_prefetch_batches = parse_value(&mut values, &flag)?;
                }
                "--loader-profile-every" => {
                    args.loader_profile_every = parse_value(&mut values, &flag)?;
                }
                "--metric-every" => args.metric_every = parse_positive_value(&mut values, &flag)?,
                "--descriptor-weight" => {
                    args.descriptor_weight = parse_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-weight" => {
                    args.tanimoto_ranking_weight = parse_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-margin" | "--tanimoto-ranking-latent-temperature" => {
                    args.tanimoto_ranking_latent_temperature = parse_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-metric-temperature" => {
                    args.tanimoto_ranking_metric_temperature = parse_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-min-gap" => {
                    args.tanimoto_ranking_min_gap = parse_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-candidates" => {
                    args.tanimoto_ranking_candidates = parse_positive_value(&mut values, &flag)?;
                }
                "--tanimoto-ranking-pairs-per-batch" => {
                    args.tanimoto_ranking_pairs_per_batch = parse_value(&mut values, &flag)?;
                }
                "--full-validation" => args.max_valid_batches = None,
                "--resume" => args.resume = true,
                "--force-preprocess" => args.force_preprocess = true,
                "--preprocess-threads" => {
                    args.preprocess_threads = parse_positive_value(&mut values, &flag)?;
                }
                "--cuda-device" => args.cuda_device = parse_value(&mut values, &flag)?,
                "--help" | "-h" => return Err(invalid_input(Self::usage())),
                unknown => {
                    return Err(invalid_input(format!(
                        "unknown argument `{unknown}`\n{}",
                        Self::usage()
                    )));
                }
            }
        }

        Ok(args)
    }

    fn usage() -> String {
        "usage: train_cached_shards pubchem <cache-dir> <checkpoint-dir> \
         OR train_cached_shards zinc20 <cache-dir> <checkpoint-dir> \
         OR train_cached_shards all <cache-dir> <checkpoint-dir> \
         OR train_cached_shards manifest <manifest.json> <checkpoint-dir> \
         [--epochs N] [--batch-size N] [--learning-rate LR] [--latent-width N] \
         [--latent-noise-std STD] \
         [--hidden-widths 4096,2048,1024] [--checkpoint-every N] \
         [--validate-every N] [--max-train-batches N] [--max-valid-batches N] \
         [--loader-workers N] [--device-prefetch-batches N] \
         [--metric-every N] [--loader-profile-every N] \
         [--descriptor-weight W] \
         [--tanimoto-ranking-weight W] \
         [--tanimoto-ranking-latent-temperature T] \
         [--tanimoto-ranking-metric-temperature T] \
         [--tanimoto-ranking-min-gap G] [--tanimoto-ranking-candidates N] \
         [--tanimoto-ranking-pairs-per-batch N] \
         [--full-validation] \
         [--rows-per-shard N] [--zinc20-chunks FIRST-LAST] \
         [--preprocess-threads N] [--force-preprocess] \
         [--resume] [--cuda-device N]"
            .to_string()
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
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

    fn mean_loss(self) -> f32 {
        if self.loss_batches == 0 {
            0.0
        } else {
            (self.loss_sum / self.loss_batches as f64) as f32
        }
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
impl EvaluationSummary {
    fn record(
        &mut self,
        examples: usize,
        loss: Option<f32>,
        tanimoto: f32,
        data_time: Duration,
        step_time: Duration,
    ) {
        self.batches += 1;
        self.examples += examples;
        if let Some(loss) = loss {
            self.loss_batches += 1;
            self.loss_sum += f64::from(loss);
        }
        self.tanimoto_sum += f64::from(tanimoto);
        self.data_time += data_time;
        self.step_time += step_time;
    }

    fn mean_loss(self) -> f32 {
        if self.loss_batches == 0 {
            0.0
        } else {
            (self.loss_sum / self.loss_batches as f64) as f32
        }
    }

    fn mean_tanimoto(self) -> f32 {
        if self.batches == 0 {
            0.0
        } else {
            (self.tanimoto_sum / self.batches as f64) as f32
        }
    }
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn next_value(values: &mut impl Iterator<Item = String>, flag: &str) -> AppResult<String> {
    values
        .next()
        .ok_or_else(|| invalid_input(format!("missing value for `{flag}`")))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn parse_value<T>(values: &mut impl Iterator<Item = String>, flag: &str) -> AppResult<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = next_value(values, flag)?;
    value
        .parse::<T>()
        .map_err(|source| invalid_input(format!("invalid value for `{flag}`: {source}")))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn parse_positive_value(values: &mut impl Iterator<Item = String>, flag: &str) -> AppResult<usize> {
    let value = parse_value::<usize>(values, flag)?;
    if value == 0 {
        return Err(invalid_input(format!("`{flag}` must be greater than zero")));
    }
    Ok(value)
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn parse_zinc20_chunks(value: &str) -> AppResult<(u8, u8)> {
    let value = value.trim();
    let (first, last) = if let Some((first, last)) = value.split_once("..=") {
        (first, last)
    } else if let Some((first, last)) = value.split_once('-') {
        (first, last)
    } else {
        (value, value)
    };
    let first_chunk = first.parse::<u8>().map_err(|source| {
        invalid_input(format!("invalid ZINC20 first chunk `{first}`: {source}"))
    })?;
    let last_chunk = last
        .parse::<u8>()
        .map_err(|source| invalid_input(format!("invalid ZINC20 last chunk `{last}`: {source}")))?;
    if first_chunk < ZINC20_FIRST_CHUNK
        || last_chunk > ZINC20_LAST_CHUNK
        || first_chunk > last_chunk
    {
        return Err(invalid_input(format!(
            "ZINC20 chunks must be an inclusive range within {ZINC20_FIRST_CHUNK}..={ZINC20_LAST_CHUNK}"
        )));
    }
    Ok((first_chunk, last_chunk))
}

#[cfg(all(
    feature = "std",
    feature = "train",
    any(feature = "cuda", feature = "ndarray")
))]
fn parse_hidden_widths(value: &str) -> AppResult<Vec<usize>> {
    let widths = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|source| invalid_input(format!("invalid hidden width `{part}`: {source}")))
        })
        .collect::<AppResult<Vec<_>>>()?;

    if widths.is_empty() {
        Err(invalid_input("at least one hidden width is required"))
    } else {
        Ok(widths)
    }
}
