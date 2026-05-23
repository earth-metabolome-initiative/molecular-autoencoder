//! Command-line argument parsing for the `train` bin.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use molecular_autoencoder::{
    DEFAULT_BCE_NONZERO_WEIGHT, DEFAULT_BCE_WEIGHT, DEFAULT_BCE_ZERO_WEIGHT,
    DEFAULT_DESCRIPTOR_WEIGHT, DEFAULT_ECFP_RADIUS, DEFAULT_HIDDEN_WIDTHS,
    DEFAULT_LATENT_NOISE_STD, DEFAULT_LATENT_WIDTH, DEFAULT_NEGATIVES_PER_POSITIVE,
    DEFAULT_RECONSTRUCTION_POISSON_MAX_LOG_COUNT, DEFAULT_TANIMOTO_RANKING_CANDIDATES_PER_ANCHOR,
    DEFAULT_TANIMOTO_RANKING_LATENT_TEMPERATURE, DEFAULT_TANIMOTO_RANKING_METRIC_TEMPERATURE,
    DEFAULT_TANIMOTO_RANKING_MIN_GAP, DEFAULT_TANIMOTO_RANKING_PAIRS_PER_BATCH,
    DEFAULT_TANIMOTO_RANKING_SOFTRANK_TEMPERATURE, DEFAULT_TANIMOTO_RANKING_WEIGHT,
    DEFAULT_VIOLATION_WEIGHT, MoleculeAutoencoderConfig, ReconstructionLossKind,
    SmilesQualityFilter, TanimotoLossKind,
};

use crate::{AppResult, invalid_input};

/// Default Rayon workers for dataset preprocessing.
pub const DEFAULT_PREPROCESS_THREADS: usize = 64;
/// Default rows written per cached shard.
pub const DEFAULT_ROWS_PER_SHARD: usize = 10_000_000;
/// Default number of host-side dataloader worker threads.
pub const DEFAULT_LOADER_WORKERS: usize = 2;
/// Default device-side prefetch depth.
pub const DEFAULT_DEVICE_PREFETCH_BATCHES: usize = 0;
/// Default sampling stride for expensive training metrics.
pub const DEFAULT_METRIC_EVERY: usize = 10;
/// Smallest valid ZINC20 chunk index.
pub const ZINC20_FIRST_CHUNK: u8 = 1;
/// Largest valid ZINC20 chunk index.
pub const ZINC20_LAST_CHUNK: u8 = 20;

/// Supported source datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DatasetKind {
    /// PubChem SMILES dataset (~123M records).
    Pubchem,
    /// ZINC20 SMILES dataset (~1G records over 20 chunks).
    Zinc20,
}

/// Inclusive ZINC20 chunk range, parsed from `FIRST-LAST` or `FIRST..=LAST`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Zinc20ChunkRange {
    /// First chunk (inclusive).
    pub first: u8,
    /// Last chunk (inclusive).
    pub last: u8,
}

impl std::str::FromStr for Zinc20ChunkRange {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        let (first, last) = if let Some((first, last)) = value.split_once("..=") {
            (first, last)
        } else if let Some((first, last)) = value.split_once('-') {
            (first, last)
        } else {
            (value, value)
        };
        let first_chunk = first
            .parse::<u8>()
            .map_err(|source| format!("invalid ZINC20 first chunk `{first}`: {source}"))?;
        let last_chunk = last
            .parse::<u8>()
            .map_err(|source| format!("invalid ZINC20 last chunk `{last}`: {source}"))?;
        if first_chunk < ZINC20_FIRST_CHUNK
            || last_chunk > ZINC20_LAST_CHUNK
            || first_chunk > last_chunk
        {
            return Err(format!(
                "ZINC20 chunks must be an inclusive range within {ZINC20_FIRST_CHUNK}..={ZINC20_LAST_CHUNK}"
            ));
        }
        Ok(Self {
            first: first_chunk,
            last: last_chunk,
        })
    }
}

/// Datasets selected for ingestion, after deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceSelection {
    /// Only PubChem records.
    PubChem,
    /// Only ZINC20 records over an inclusive chunk range.
    Zinc20(Zinc20ChunkRange),
    /// Both PubChem and a ZINC20 chunk range.
    PubChemAndZinc20(Zinc20ChunkRange),
}

impl SourceSelection {
    /// Builds the source selection from clap-parsed flags.
    pub fn from_flags(
        datasets: &[DatasetKind],
        zinc20_chunks: Option<Zinc20ChunkRange>,
    ) -> AppResult<Option<Self>> {
        let mut want_pubchem = false;
        let mut want_zinc20 = false;
        for kind in datasets {
            match kind {
                DatasetKind::Pubchem => {
                    if want_pubchem {
                        return Err(invalid_input("--datasets lists `pubchem` more than once"));
                    }
                    want_pubchem = true;
                }
                DatasetKind::Zinc20 => {
                    if want_zinc20 {
                        return Err(invalid_input("--datasets lists `zinc20` more than once"));
                    }
                    want_zinc20 = true;
                }
            }
        }
        if !want_zinc20 && zinc20_chunks.is_some() {
            return Err(invalid_input(
                "--zinc20-chunks requires --datasets to include `zinc20`",
            ));
        }
        let chunks = zinc20_chunks.unwrap_or(Zinc20ChunkRange {
            first: ZINC20_FIRST_CHUNK,
            last: ZINC20_LAST_CHUNK,
        });
        Ok(match (want_pubchem, want_zinc20) {
            (false, false) => None,
            (true, false) => Some(Self::PubChem),
            (false, true) => Some(Self::Zinc20(chunks)),
            (true, true) => Some(Self::PubChemAndZinc20(chunks)),
        })
    }

    /// Returns the manifest source label that identifies this selection.
    pub fn manifest_source(self) -> String {
        match self {
            Self::PubChem => "pubchem-smiles".to_string(),
            Self::Zinc20(range) => {
                format!("zinc20-smiles chunks {}..={}", range.first, range.last)
            }
            Self::PubChemAndZinc20(range) => format!(
                "pubchem-smiles + zinc20-smiles chunks {}..={}",
                range.first, range.last
            ),
        }
    }
}

/// Fully-parsed training CLI configuration.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "train",
    version,
    about = "Train the molecular autoencoder on cached SMILES shards",
    disable_help_subcommand = true
)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "clap flag aggregator: each bool is a discrete CLI switch and refactoring to enums would obscure the user-facing interface"
)]
pub struct Args {
    /// Cached shards directory, or path to an existing manifest.json file.
    pub shards: PathBuf,

    /// Directory for model, optimizer, and state checkpoints.
    pub checkpoint_dir: PathBuf,

    /// Datasets to ingest into the cache (comma-separated). Omit to train
    /// from an existing manifest without re-preprocessing.
    #[arg(long, value_delimiter = ',', value_enum)]
    pub datasets: Vec<DatasetKind>,

    /// Inclusive ZINC20 chunk range (e.g. `1-5` or `1..=5`).
    #[arg(long, value_name = "FIRST-LAST")]
    pub zinc20_chunks: Option<Zinc20ChunkRange>,

    /// Rows written per cached shard.
    #[arg(long, default_value_t = DEFAULT_ROWS_PER_SHARD)]
    pub rows_per_shard: usize,

    /// Training epochs.
    #[arg(long, default_value_t = 10)]
    pub epochs: usize,

    /// Mini-batch size.
    #[arg(long, default_value_t = 4096)]
    pub batch_size: usize,

    /// Adam learning rate.
    #[arg(long, default_value_t = 1.0e-3)]
    pub learning_rate: f64,

    /// Latent embedding width.
    #[arg(long, default_value_t = DEFAULT_LATENT_WIDTH)]
    pub latent_width: usize,

    /// Decoder-side latent denoising noise as fraction of batch latent std.
    #[arg(long, default_value_t = DEFAULT_LATENT_NOISE_STD)]
    pub latent_noise_std: f64,

    /// Encoder hidden widths, comma-separated.
    #[arg(long, value_delimiter = ',', default_values_t = DEFAULT_HIDDEN_WIDTHS)]
    pub hidden_widths: Vec<usize>,

    /// Checkpoint every N epochs (0 disables).
    #[arg(long, default_value_t = 1)]
    pub checkpoint_every: usize,

    /// Validate every N epochs (0 disables).
    #[arg(long, default_value_t = 1)]
    pub validate_every: usize,

    /// Cap training batches per epoch.
    #[arg(long)]
    pub max_train_batches: Option<usize>,

    /// Cap validation batches per pass.
    #[arg(long, default_value_t = 64)]
    pub max_valid_batches: usize,

    /// Disable the validation batch cap entirely.
    #[arg(long)]
    pub full_validation: bool,

    /// Host-side dataloader worker threads (0 forces the sync path).
    #[arg(long, default_value_t = DEFAULT_LOADER_WORKERS)]
    pub loader_workers: usize,

    /// Device-side prefetch depth (forced to 0 on CUDA).
    #[arg(long, default_value_t = DEFAULT_DEVICE_PREFETCH_BATCHES)]
    pub device_prefetch_batches: usize,

    /// Periodic loader-profile flush interval (0 disables).
    #[arg(long, default_value_t = 0)]
    pub loader_profile_every: usize,

    /// Stride for expensive batch-level metrics (must be > 0).
    #[arg(long, default_value_t = DEFAULT_METRIC_EVERY, value_parser = positive_usize)]
    pub metric_every: usize,

    /// Descriptor regression loss weight.
    #[arg(long, default_value_t = DEFAULT_DESCRIPTOR_WEIGHT)]
    pub descriptor_weight: f64,

    /// Latent Tanimoto geometry loss weight.
    #[arg(long, default_value_t = DEFAULT_TANIMOTO_RANKING_WEIGHT)]
    pub tanimoto_ranking_weight: f64,

    /// Multi-label violation-classification loss weight (latent head).
    #[arg(long, default_value_t = DEFAULT_VIOLATION_WEIGHT)]
    pub violation_weight: f64,

    /// Average number of `MutatorMix`-generated negative rows the
    /// preprocessor emits per source positive (fractional values use a
    /// per-record Bernoulli draw). Has no effect on training runs that
    /// reuse an already-preprocessed shard manifest.
    #[arg(long, default_value_t = DEFAULT_NEGATIVES_PER_POSITIVE)]
    pub negatives_per_positive: f32,

    /// Latent cosine-logit temperature (alias: --tanimoto-ranking-margin).
    #[arg(long, alias = "tanimoto-ranking-margin", default_value_t = DEFAULT_TANIMOTO_RANKING_LATENT_TEMPERATURE)]
    pub tanimoto_ranking_latent_temperature: f64,

    /// Compatibility metric temperature (unused by the softmax loss).
    #[arg(long, default_value_t = DEFAULT_TANIMOTO_RANKING_METRIC_TEMPERATURE)]
    pub tanimoto_ranking_metric_temperature: f64,

    /// Minimum counted-Tanimoto gap an anchor must clear to contribute.
    #[arg(long, default_value_t = DEFAULT_TANIMOTO_RANKING_MIN_GAP)]
    pub tanimoto_ranking_min_gap: f64,

    /// Random candidate partners sampled per anchor (must be >= 2).
    #[arg(long, default_value_t = DEFAULT_TANIMOTO_RANKING_CANDIDATES_PER_ANCHOR, value_parser = positive_usize)]
    pub tanimoto_ranking_candidates: usize,

    /// Max anchors used by the geometry loss (0 uses all rows).
    #[arg(long, default_value_t = DEFAULT_TANIMOTO_RANKING_PAIRS_PER_BATCH)]
    pub tanimoto_ranking_pairs_per_batch: usize,

    /// Resume from the checkpoint directory.
    #[arg(long)]
    pub resume: bool,

    /// Force a fresh preprocessing pass even when a cached manifest exists.
    #[arg(long)]
    pub force_preprocess: bool,

    /// Rayon worker threads used during preprocessing (must be > 0).
    #[arg(long, default_value_t = DEFAULT_PREPROCESS_THREADS, value_parser = positive_usize)]
    pub preprocess_threads: usize,

    /// CUDA device ordinals (comma-separated). When omitted, every visible
    /// CUDA device is used. Ignored on the ndarray backend.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    pub cuda_devices: Vec<usize>,

    /// Minimum heavy-atom count quality bound.
    #[arg(long)]
    pub min_heavy_atoms: Option<u32>,

    /// Maximum heavy-atom count quality bound.
    #[arg(long)]
    pub max_heavy_atoms: Option<u32>,

    /// Minimum molecular mass (Da).
    #[arg(long)]
    pub min_molecular_mass: Option<f32>,

    /// Maximum molecular mass (Da).
    #[arg(long)]
    pub max_molecular_mass: Option<f32>,

    /// Minimum total formal charge.
    #[arg(long, allow_negative_numbers = true)]
    pub min_formal_charge: Option<i32>,

    /// Maximum total formal charge.
    #[arg(long, allow_negative_numbers = true)]
    pub max_formal_charge: Option<i32>,

    /// Maximum number of disconnected components (set 1 to drop salts/mixtures).
    #[arg(long)]
    pub max_connected_components: Option<u32>,

    /// Morgan/ECFP radius for counted fingerprints (must be >= 1).
    #[arg(long, default_value_t = DEFAULT_ECFP_RADIUS, value_parser = positive_u8)]
    pub ecfp_radius: u8,

    /// Auxiliary BCE-with-logits reconstruction loss weight (`0.0` disables it).
    #[arg(long, default_value_t = DEFAULT_BCE_WEIGHT)]
    pub bce_weight: f64,

    /// Per-position weight applied to inactive bins inside the BCE term.
    #[arg(long, default_value_t = DEFAULT_BCE_ZERO_WEIGHT)]
    pub bce_zero_weight: f64,

    /// Per-position weight applied to active bins inside the BCE term.
    #[arg(long, default_value_t = DEFAULT_BCE_NONZERO_WEIGHT)]
    pub bce_nonzero_weight: f64,

    /// Tanimoto-geometry ranking loss kind. `softrank-spearman`
    /// (default) is the gap-weighted differentiable Spearman surrogate
    /// `1 - Pearson(soft_rank(logits), soft_rank(teacher_scores))`,
    /// which has out-performed the legacy `softmax-ce` baseline at
    /// `--tanimoto-ranking-candidates 8`. Pass `softmax-ce` to opt
    /// back into the legacy gap-weighted softmax cross-entropy on the
    /// top-1 target.
    #[arg(long, value_enum, default_value_t = TanimotoLossKindArg::SoftrankSpearman)]
    pub tanimoto_ranking_loss: TanimotoLossKindArg,

    /// Pairwise-sigmoid temperature for the SoftRank Spearman
    /// surrogate (consulted only when `--tanimoto-ranking-loss` is
    /// `softrank-spearman`).
    #[arg(long, default_value_t = DEFAULT_TANIMOTO_RANKING_SOFTRANK_TEMPERATURE)]
    pub tanimoto_ranking_softrank_temperature: f64,

    /// Reconstruction loss kind. `huber` is the shipping baseline
    /// (weighted Huber on `log1p(count)` plus the auxiliary bit-presence
    /// BCE term); `poisson` swaps in Poisson NLL on raw counts, which
    /// structurally avoids the "predict zero everywhere" trivial
    /// minimum the one-sided Huber + one-sided BCE pair invites.
    #[arg(long, value_enum, default_value_t = ReconstructionLossKindArg::Huber)]
    pub reconstruction_loss: ReconstructionLossKindArg,

    /// Clamp ceiling applied to predicted log-rates before `exp` inside
    /// the Poisson NLL branch. Prevents an early-training logit spike
    /// from overflowing `f32` in `exp(z)`. Ignored under
    /// `--reconstruction-loss huber`.
    #[arg(long, default_value_t = DEFAULT_RECONSTRUCTION_POISSON_MAX_LOG_COUNT)]
    pub reconstruction_poisson_max_log_count: f64,

    /// Optional curriculum warmup: when set, the model is trained under
    /// this reconstruction loss kind for the first
    /// `--reconstruction-loss-warmup-epochs` epochs before swapping to
    /// `--reconstruction-loss` for the rest of the run. Omit (or pair
    /// with `--reconstruction-loss-warmup-epochs 0`) to disable the
    /// warmup phase.
    #[arg(long, value_enum)]
    pub reconstruction_loss_warmup: Option<ReconstructionLossKindArg>,

    /// Number of epochs to spend under
    /// `--reconstruction-loss-warmup` before swapping to the main
    /// `--reconstruction-loss`. `0` (default) disables the warmup
    /// phase; ignored when `--reconstruction-loss-warmup` is unset.
    #[arg(long, default_value_t = 0)]
    pub reconstruction_loss_warmup_epochs: usize,

    /// Enable the adaptive reconstruction-loss-kind scheduler. When
    /// set, the model tracks an EMA of `active-bin / inactive-bin`
    /// symmetric-Huber error and swaps between
    /// `--reconstruction-loss-warmup` (Warmup) and
    /// `--reconstruction-loss` (Main) based on hysteresis thresholds.
    /// Requires `--reconstruction-loss-warmup` to be set.
    #[arg(long, default_value_t = false)]
    pub reconstruction_loss_adaptive: bool,

    /// While in Warmup, swap to the Main kind when the EMA drops below
    /// this ratio.
    #[arg(long, default_value_t = 1.0)]
    pub reconstruction_loss_adaptive_switch_to_main: f64,

    /// While in Main, swap back to the Warmup kind when the EMA climbs
    /// above this ratio.
    #[arg(long, default_value_t = 2.0)]
    pub reconstruction_loss_adaptive_switch_to_warmup: f64,

    /// EMA decay applied to each metric-sampled batch's
    /// `active / inactive` ratio. Half-life is roughly
    /// `ln(0.5) / ln(decay)` sampled batches; default 0.95 gives
    /// ~14 samples ≈ ~140 batches at default `--metric-every 10`.
    #[arg(long, default_value_t = 0.95)]
    pub reconstruction_loss_adaptive_ema_decay: f64,
}

/// CLI mirror of [`molecular_autoencoder::ReconstructionLossKind`]
/// derived as [`clap::ValueEnum`] so the library does not need a
/// `clap` dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ReconstructionLossKindArg {
    /// Weighted Huber on `log1p(count)` (with the auxiliary BCE term).
    Huber,
    /// Two-sided weighted Huber on `log1p(count)` (penalizes both leak
    /// directions at zero-target bins). BCE branch is force-zeroed.
    HuberSymmetric,
    /// Poisson NLL on raw counts (model output reinterpreted as
    /// `log(rate)`).
    Poisson,
}

impl From<ReconstructionLossKindArg> for ReconstructionLossKind {
    fn from(value: ReconstructionLossKindArg) -> Self {
        match value {
            ReconstructionLossKindArg::Huber => Self::Huber,
            ReconstructionLossKindArg::HuberSymmetric => Self::HuberSymmetric,
            ReconstructionLossKindArg::Poisson => Self::Poisson,
        }
    }
}

/// CLI mirror of [`molecular_autoencoder::TanimotoLossKind`] derived as
/// [`clap::ValueEnum`] so the library does not need a `clap` dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TanimotoLossKindArg {
    /// Gap-weighted softmax cross-entropy against the one-hot top-1
    /// candidate (legacy baseline).
    SoftmaxCe,
    /// Gap-weighted `1 - Pearson(soft_rank(logits),
    /// soft_rank(teacher_scores))` SoftRank Spearman surrogate
    /// (current default).
    SoftrankSpearman,
}

impl From<TanimotoLossKindArg> for TanimotoLossKind {
    fn from(value: TanimotoLossKindArg) -> Self {
        match value {
            TanimotoLossKindArg::SoftmaxCe => Self::SoftmaxCe,
            TanimotoLossKindArg::SoftrankSpearman => Self::SoftRankSpearman,
        }
    }
}

impl Args {
    /// Resolves the active reconstruction-loss kind for a given
    /// 1-based training epoch, honouring the
    /// `--reconstruction-loss-warmup` curriculum.
    ///
    /// Falls back to `--reconstruction-loss` when the warmup variant is
    /// unset or the warmup window has elapsed.
    pub fn reconstruction_loss_kind_for_epoch(&self, epoch: usize) -> ReconstructionLossKind {
        match self.reconstruction_loss_warmup {
            Some(warmup) if epoch <= self.reconstruction_loss_warmup_epochs => warmup.into(),
            _ => self.reconstruction_loss.into(),
        }
    }

    /// Validates the adaptive-scheduler flags after clap parsing.
    /// `--reconstruction-loss-adaptive` requires a Warmup kind to swap
    /// between, and the two ratio thresholds must form a proper
    /// hysteresis window (`switch_to_main < switch_to_warmup`).
    ///
    /// # Errors
    ///
    /// Returns [`crate::AppError`] (via `invalid_input`) when the
    /// adaptive flag is set without `--reconstruction-loss-warmup`,
    /// or when the thresholds are inverted, non-finite, or the EMA
    /// decay sits outside `(0, 1)`.
    pub fn validate_adaptive_loss(&self) -> AppResult<()> {
        if !self.reconstruction_loss_adaptive {
            return Ok(());
        }
        if self.reconstruction_loss_warmup.is_none() {
            return Err(invalid_input(
                "--reconstruction-loss-adaptive requires --reconstruction-loss-warmup to be set so the controller knows which kind to swap back to",
            ));
        }
        let switch_to_main = self.reconstruction_loss_adaptive_switch_to_main;
        let switch_to_warmup = self.reconstruction_loss_adaptive_switch_to_warmup;
        if !switch_to_main.is_finite() || !switch_to_warmup.is_finite() {
            return Err(invalid_input(format!(
                "adaptive thresholds must be finite, got switch-to-main={switch_to_main}, switch-to-warmup={switch_to_warmup}",
            )));
        }
        if switch_to_main >= switch_to_warmup {
            return Err(invalid_input(format!(
                "--reconstruction-loss-adaptive-switch-to-main ({switch_to_main}) must be strictly less than --reconstruction-loss-adaptive-switch-to-warmup ({switch_to_warmup}) to form a hysteresis window",
            )));
        }
        let decay = self.reconstruction_loss_adaptive_ema_decay;
        if !decay.is_finite() || !(0.0..1.0).contains(&decay) {
            return Err(invalid_input(format!(
                "--reconstruction-loss-adaptive-ema-decay must lie in (0, 1), got {decay}",
            )));
        }
        Ok(())
    }

    /// Resolves the manifest path and source selection from the raw flags.
    pub fn manifest_paths(&self) -> AppResult<ResolvedPaths> {
        let shards_path = self.shards.clone();
        let (manifest_path, cache_dir) = if shards_path.is_dir() || !shards_path.exists() {
            (shards_path.join("manifest.json"), Some(shards_path))
        } else {
            (shards_path, None)
        };
        let source_selection = SourceSelection::from_flags(&self.datasets, self.zinc20_chunks)?;
        Ok(ResolvedPaths {
            manifest_path,
            cache_dir,
            source_selection,
        })
    }

    /// Resolves the CUDA device ordinals to use for training.
    ///
    /// When `--cuda-devices` is empty, enumerates every visible CUDA device via
    /// `cudarc`. When it is specified, validates each index against the visible
    /// count and rejects out-of-range entries.
    ///
    /// Honours `CUDA_VISIBLE_DEVICES` because `cudarc` does.
    ///
    /// # Errors
    ///
    /// Returns an error when the CUDA driver fails to initialise, when no
    /// device is visible, or when `--cuda-devices` references an unavailable
    /// index.
    #[cfg(feature = "cuda")]
    pub fn resolved_cuda_devices(&self) -> AppResult<Vec<usize>> {
        cudarc::driver::result::init()
            .map_err(|err| invalid_input(format!("CUDA driver init failed: {err}")))?;
        let raw = cudarc::driver::result::device::get_count()
            .map_err(|err| invalid_input(format!("CUDA enumeration failed: {err}")))?;
        let visible = usize::try_from(raw)
            .map_err(|_| invalid_input(format!("CUDA reported negative device count {raw}")))?;
        if visible == 0 {
            return Err(invalid_input("no CUDA devices visible"));
        }
        if self.cuda_devices.is_empty() {
            Ok((0..visible).collect())
        } else {
            for &index in &self.cuda_devices {
                if index >= visible {
                    return Err(invalid_input(format!(
                        "--cuda-devices: index {index} is not less than visible count {visible}"
                    )));
                }
            }
            let mut sorted = self.cuda_devices.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != self.cuda_devices.len() {
                return Err(invalid_input("--cuda-devices contains duplicate indices"));
            }
            Ok(sorted)
        }
    }

    /// Returns `None` when `--full-validation` is set; otherwise the cap.
    pub fn max_valid_batches(&self) -> Option<usize> {
        if self.full_validation {
            None
        } else {
            Some(self.max_valid_batches)
        }
    }

    /// Builds and validates a [`MoleculeAutoencoderConfig`] from the CLI
    /// components via [`MoleculeAutoencoderConfig::builder`].
    ///
    /// # Errors
    ///
    /// Returns the same `ConfigInvalid` payload the builder emits when any
    /// invariant fails (negative weights, non-finite temperatures, etc.).
    pub fn to_model_config(
        &self,
        fingerprint_size: usize,
        ecfp_radius: u8,
    ) -> AppResult<MoleculeAutoencoderConfig> {
        MoleculeAutoencoderConfig::builder()
            .fingerprint_size(fingerprint_size)
            .latent_width(self.latent_width)
            .hidden_widths(self.hidden_widths.clone())
            .descriptor_weight(self.descriptor_weight)
            .tanimoto_ranking_weight(self.tanimoto_ranking_weight)
            .violation_weight(self.violation_weight)
            .latent_noise_std(self.latent_noise_std)
            .tanimoto_ranking_latent_temperature(self.tanimoto_ranking_latent_temperature)
            .tanimoto_ranking_metric_temperature(self.tanimoto_ranking_metric_temperature)
            .tanimoto_ranking_min_gap(self.tanimoto_ranking_min_gap)
            .tanimoto_ranking_candidates(self.tanimoto_ranking_candidates)
            .tanimoto_ranking_pairs_per_batch(self.tanimoto_ranking_pairs_per_batch)
            .tanimoto_ranking_loss_kind(self.tanimoto_ranking_loss.into())
            .tanimoto_ranking_softrank_temperature(self.tanimoto_ranking_softrank_temperature)
            .ecfp_radius(ecfp_radius)
            .bce_weight(self.bce_weight)
            .bce_zero_weight(self.bce_zero_weight)
            .bce_nonzero_weight(self.bce_nonzero_weight)
            .reconstruction_loss_kind(self.reconstruction_loss.into())
            .reconstruction_poisson_max_log_count(self.reconstruction_poisson_max_log_count)
            .build()
            .map_err(Into::into)
    }

    /// Builds an immutable [`SmilesQualityFilter`] from the CLI bounds,
    /// validating any inverted ranges at build time.
    pub fn quality_filter(&self) -> AppResult<SmilesQualityFilter> {
        let mut builder = SmilesQualityFilter::builder();
        if let Some(value) = self.min_heavy_atoms {
            builder = builder.min_heavy_atoms(value);
        }
        if let Some(value) = self.max_heavy_atoms {
            builder = builder.max_heavy_atoms(value);
        }
        if let Some(value) = self.min_molecular_mass {
            builder = builder.min_molecular_mass(value);
        }
        if let Some(value) = self.max_molecular_mass {
            builder = builder.max_molecular_mass(value);
        }
        if let Some(value) = self.min_formal_charge {
            builder = builder.min_formal_charge(value);
        }
        if let Some(value) = self.max_formal_charge {
            builder = builder.max_formal_charge(value);
        }
        if let Some(value) = self.max_connected_components {
            builder = builder.max_connected_components(value);
        }
        builder.build().map_err(Into::into)
    }
}

/// Manifest path and dataset selection resolved from CLI inputs.
#[derive(Debug, Clone)]
pub struct ResolvedPaths {
    /// Final manifest path used to drive shard discovery.
    pub manifest_path: PathBuf,
    /// Optional cache directory backing dataset downloads.
    pub cache_dir: Option<PathBuf>,
    /// Dataset source selection (raw paths vs registered dataset name).
    pub source_selection: Option<SourceSelection>,
}

/// `clap` value parser that accepts strictly positive `usize` values.
fn positive_usize(value: &str) -> Result<usize, String> {
    let parsed: usize = value
        .parse()
        .map_err(|err: std::num::ParseIntError| err.to_string())?;
    if parsed == 0 {
        Err("must be greater than zero".to_string())
    } else {
        Ok(parsed)
    }
}

/// `clap` value parser that accepts strictly positive `u8` values.
fn positive_u8(value: &str) -> Result<u8, String> {
    let parsed: u8 = value
        .parse()
        .map_err(|err: std::num::ParseIntError| err.to_string())?;
    if parsed == 0 {
        Err("must be greater than zero".to_string())
    } else {
        Ok(parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn command_definition_is_valid() {
        Args::command().debug_assert();
    }

    #[test]
    fn zinc20_chunks_parses_dash_and_inclusive_forms() {
        assert_eq!(
            "1-5".parse::<Zinc20ChunkRange>().expect("dash form"),
            Zinc20ChunkRange { first: 1, last: 5 }
        );
        assert_eq!(
            "2..=4".parse::<Zinc20ChunkRange>().expect("inclusive form"),
            Zinc20ChunkRange { first: 2, last: 4 }
        );
        assert_eq!(
            "7".parse::<Zinc20ChunkRange>().expect("single chunk"),
            Zinc20ChunkRange { first: 7, last: 7 }
        );
    }

    #[test]
    fn zinc20_chunks_rejects_out_of_range_and_inverted() {
        assert!("0-5".parse::<Zinc20ChunkRange>().is_err());
        assert!("1-21".parse::<Zinc20ChunkRange>().is_err());
        assert!("5-3".parse::<Zinc20ChunkRange>().is_err());
    }

    #[test]
    fn source_selection_dedups_and_validates_zinc_chunk_dependency() {
        assert_eq!(
            SourceSelection::from_flags(&[DatasetKind::Pubchem], None).expect("pubchem"),
            Some(SourceSelection::PubChem)
        );
        assert_eq!(
            SourceSelection::from_flags(&[DatasetKind::Zinc20, DatasetKind::Pubchem], None)
                .expect("both"),
            Some(SourceSelection::PubChemAndZinc20(Zinc20ChunkRange {
                first: ZINC20_FIRST_CHUNK,
                last: ZINC20_LAST_CHUNK,
            }))
        );
        assert!(
            SourceSelection::from_flags(&[DatasetKind::Pubchem, DatasetKind::Pubchem], None)
                .is_err()
        );
        assert!(
            SourceSelection::from_flags(
                &[DatasetKind::Pubchem],
                Some(Zinc20ChunkRange { first: 1, last: 5 })
            )
            .is_err()
        );
        assert_eq!(
            SourceSelection::from_flags(&[], None).expect("no datasets"),
            None
        );
    }

    #[test]
    fn parse_minimal_positional_arguments() {
        let args = Args::try_parse_from(["train", "shards", "runs"]).expect("minimal parse");
        assert_eq!(args.shards, PathBuf::from("shards"));
        assert_eq!(args.checkpoint_dir, PathBuf::from("runs"));
        assert!(args.datasets.is_empty());
        assert_eq!(args.batch_size, 4096);
        assert_eq!(args.hidden_widths, vec![4096, 2048, 1024]);
        assert!(!args.full_validation);
        assert_eq!(args.max_valid_batches(), Some(64));
    }

    #[test]
    fn parse_full_validation_clears_max_cap() {
        let args = Args::try_parse_from(["train", "s", "c", "--full-validation"]).expect("parse");
        assert_eq!(args.max_valid_batches(), None);
    }

    #[test]
    fn parse_quality_filter_through_builder_validates() {
        let args = Args::try_parse_from([
            "train",
            "s",
            "c",
            "--min-heavy-atoms",
            "5",
            "--max-heavy-atoms",
            "50",
            "--min-formal-charge",
            "-1",
            "--max-formal-charge",
            "1",
            "--max-connected-components",
            "1",
        ])
        .expect("parse filter flags");
        let filter = args.quality_filter().expect("valid filter");
        assert_eq!(filter.min_heavy_atoms(), Some(5));
        assert_eq!(filter.max_heavy_atoms(), Some(50));
        assert_eq!(filter.min_formal_charge(), Some(-1));
        assert_eq!(filter.max_formal_charge(), Some(1));
        assert_eq!(filter.max_connected_components(), Some(1));
        assert!(filter.is_active());
    }

    #[test]
    fn parse_quality_filter_rejects_inverted_range() {
        let args = Args::try_parse_from([
            "train",
            "s",
            "c",
            "--min-heavy-atoms",
            "50",
            "--max-heavy-atoms",
            "5",
        ])
        .expect("parse");
        assert!(args.quality_filter().is_err());
    }

    #[test]
    fn to_model_config_threads_overrides_through_builder() {
        let args = Args::try_parse_from([
            "train",
            "s",
            "c",
            "--latent-width",
            "16",
            "--hidden-widths",
            "32,24",
            "--descriptor-weight",
            "0.07",
            "--tanimoto-ranking-weight",
            "0.13",
            "--tanimoto-ranking-candidates",
            "5",
        ])
        .expect("parse");
        let config = args
            .to_model_config(64, 3)
            .expect("builder accepts overrides");

        assert_eq!(config.encoder().input_width(), 64);
        assert_eq!(config.encoder().latent_width(), 16);
        assert_eq!(config.encoder().hidden_widths(), &[32, 24]);
        assert_eq!(config.auxiliary_weights().descriptors(), 0.07);
        assert_eq!(config.auxiliary_weights().tanimoto_ranking(), 0.13);
        assert_eq!(config.tanimoto_ranking().candidates_per_anchor(), 5);
        assert_eq!(config.ecfp_radius(), 3);
        assert_eq!(
            config.reconstruction_loss().bce_weight(),
            DEFAULT_BCE_WEIGHT
        );
    }

    #[test]
    fn to_model_config_propagates_builder_validation_errors() {
        // candidates_per_anchor=1 is parseable but invalid when the geometry
        // loss is active; the builder must surface the validation error.
        let args = Args::try_parse_from([
            "train",
            "s",
            "c",
            "--tanimoto-ranking-weight",
            "0.5",
            "--tanimoto-ranking-candidates",
            "1",
        ])
        .expect("parse");
        let error = args
            .to_model_config(64, DEFAULT_ECFP_RADIUS)
            .expect_err("builder must reject");
        assert!(error.to_string().contains("candidates_per_anchor"));
    }

    #[test]
    fn parse_datasets_csv_through_clap() {
        let args = Args::try_parse_from(["train", "s", "c", "--datasets", "pubchem,zinc20"])
            .expect("parse datasets");
        assert_eq!(
            args.datasets,
            vec![DatasetKind::Pubchem, DatasetKind::Zinc20]
        );
    }

    #[test]
    fn source_selection_zinc20_alone_uses_default_chunk_range_when_unset() {
        let selection = SourceSelection::from_flags(&[DatasetKind::Zinc20], None)
            .expect("zinc20 only")
            .expect("non-empty");
        assert_eq!(
            selection,
            SourceSelection::Zinc20(Zinc20ChunkRange {
                first: ZINC20_FIRST_CHUNK,
                last: ZINC20_LAST_CHUNK,
            })
        );
    }

    #[test]
    fn source_selection_rejects_zinc20_duplicate() {
        assert!(
            SourceSelection::from_flags(&[DatasetKind::Zinc20, DatasetKind::Zinc20], None).is_err()
        );
    }

    #[test]
    fn manifest_source_formats_each_variant() {
        assert_eq!(SourceSelection::PubChem.manifest_source(), "pubchem-smiles");
        let range = Zinc20ChunkRange { first: 1, last: 5 };
        assert_eq!(
            SourceSelection::Zinc20(range).manifest_source(),
            "zinc20-smiles chunks 1..=5"
        );
        assert_eq!(
            SourceSelection::PubChemAndZinc20(range).manifest_source(),
            "pubchem-smiles + zinc20-smiles chunks 1..=5"
        );
    }

    #[test]
    fn manifest_paths_picks_manifest_inside_existing_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let args =
            Args::try_parse_from(["train", dir.path().to_str().expect("utf-8"), "checkpoints"])
                .expect("parse");
        let resolved = args.manifest_paths().expect("manifest paths");
        assert_eq!(resolved.manifest_path, dir.path().join("manifest.json"));
        assert_eq!(resolved.cache_dir, Some(dir.path().to_path_buf()));
        assert!(resolved.source_selection.is_none());
    }

    #[test]
    fn manifest_paths_uses_explicit_manifest_file_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest = dir.path().join("custom-manifest.json");
        std::fs::write(&manifest, "{}").expect("write placeholder");
        let args =
            Args::try_parse_from(["train", manifest.to_str().expect("utf-8"), "checkpoints"])
                .expect("parse");
        let resolved = args.manifest_paths().expect("manifest paths");
        assert_eq!(resolved.manifest_path, manifest);
        assert!(resolved.cache_dir.is_none());
    }

    #[test]
    fn manifest_paths_treats_missing_path_as_cache_dir() {
        let args = Args::try_parse_from(["train", "/tmp/does-not-exist-xyz", "checkpoints"])
            .expect("parse");
        let resolved = args.manifest_paths().expect("manifest paths");
        assert_eq!(
            resolved.manifest_path,
            PathBuf::from("/tmp/does-not-exist-xyz/manifest.json")
        );
        assert_eq!(
            resolved.cache_dir,
            Some(PathBuf::from("/tmp/does-not-exist-xyz"))
        );
    }

    #[test]
    fn quality_filter_threads_mass_bounds_through_builder() {
        let args = Args::try_parse_from([
            "train",
            "s",
            "c",
            "--min-molecular-mass",
            "120.0",
            "--max-molecular-mass",
            "500.0",
        ])
        .expect("parse");
        let filter = args.quality_filter().expect("valid filter");
        assert_eq!(filter.min_molecular_mass(), Some(120.0));
        assert_eq!(filter.max_molecular_mass(), Some(500.0));
    }

    #[test]
    fn positive_usize_parser_rejects_zero() {
        assert!(positive_usize("0").is_err());
        assert_eq!(positive_usize("64").expect("nonzero"), 64);
        assert!(positive_usize("not-a-number").is_err());
    }

    #[test]
    fn positive_u8_parser_rejects_zero() {
        assert!(positive_u8("0").is_err());
        assert_eq!(positive_u8("3").expect("nonzero"), 3);
        assert!(positive_u8("not-a-number").is_err());
    }

    #[test]
    fn reconstruction_loss_warmup_schedules_kind_switch_at_boundary() {
        let args = Args::try_parse_from([
            "train",
            "shards",
            "checkpoints",
            "--reconstruction-loss",
            "huber",
            "--reconstruction-loss-warmup",
            "huber-symmetric",
            "--reconstruction-loss-warmup-epochs",
            "2",
        ])
        .expect("parse warmup args");
        // First two epochs run under the warmup kind, then the loop
        // swaps to the main kind from epoch 3 onward.
        assert_eq!(
            args.reconstruction_loss_kind_for_epoch(1),
            ReconstructionLossKind::HuberSymmetric
        );
        assert_eq!(
            args.reconstruction_loss_kind_for_epoch(2),
            ReconstructionLossKind::HuberSymmetric
        );
        assert_eq!(
            args.reconstruction_loss_kind_for_epoch(3),
            ReconstructionLossKind::Huber
        );
        assert_eq!(
            args.reconstruction_loss_kind_for_epoch(100),
            ReconstructionLossKind::Huber
        );
    }

    #[test]
    fn adaptive_loss_requires_warmup_kind_to_be_set() {
        let args = Args::try_parse_from([
            "train",
            "shards",
            "checkpoints",
            "--reconstruction-loss-adaptive",
        ])
        .expect("parse");
        let err = args
            .validate_adaptive_loss()
            .expect_err("adaptive without warmup must be rejected");
        assert!(format!("{err:?}").contains("--reconstruction-loss-warmup"));
    }

    #[test]
    fn adaptive_loss_rejects_inverted_thresholds() {
        let args = Args::try_parse_from([
            "train",
            "shards",
            "checkpoints",
            "--reconstruction-loss-warmup",
            "huber-symmetric",
            "--reconstruction-loss-adaptive",
            "--reconstruction-loss-adaptive-switch-to-main",
            "2.0",
            "--reconstruction-loss-adaptive-switch-to-warmup",
            "1.0",
        ])
        .expect("parse");
        let err = args
            .validate_adaptive_loss()
            .expect_err("inverted thresholds must be rejected");
        assert!(format!("{err:?}").contains("hysteresis"));
    }

    #[test]
    fn adaptive_loss_accepts_valid_configuration() {
        let args = Args::try_parse_from([
            "train",
            "shards",
            "checkpoints",
            "--reconstruction-loss-warmup",
            "huber-symmetric",
            "--reconstruction-loss-adaptive",
            "--reconstruction-loss-adaptive-switch-to-main",
            "1.0",
            "--reconstruction-loss-adaptive-switch-to-warmup",
            "2.5",
            "--reconstruction-loss-adaptive-ema-decay",
            "0.9",
        ])
        .expect("parse");
        args.validate_adaptive_loss().expect("valid args");
    }

    #[test]
    fn adaptive_loss_rejects_out_of_range_decay() {
        let args = Args::try_parse_from([
            "train",
            "shards",
            "checkpoints",
            "--reconstruction-loss-warmup",
            "huber-symmetric",
            "--reconstruction-loss-adaptive",
            "--reconstruction-loss-adaptive-ema-decay",
            "1.2",
        ])
        .expect("parse");
        let err = args
            .validate_adaptive_loss()
            .expect_err("decay above 1 must be rejected");
        assert!(format!("{err:?}").contains("ema-decay"));
    }

    #[test]
    fn reconstruction_loss_warmup_disabled_when_warmup_kind_unset() {
        let args = Args::try_parse_from([
            "train",
            "shards",
            "checkpoints",
            "--reconstruction-loss",
            "poisson",
            "--reconstruction-loss-warmup-epochs",
            "5",
        ])
        .expect("parse args without warmup kind");
        // Setting only the epoch count is a no-op without a warmup kind.
        assert_eq!(
            args.reconstruction_loss_kind_for_epoch(1),
            ReconstructionLossKind::Poisson
        );
        assert_eq!(
            args.reconstruction_loss_kind_for_epoch(6),
            ReconstructionLossKind::Poisson
        );
    }
}
