//! Dense counted-fingerprint autoencoder model and losses.

#[cfg(feature = "std")]
use std::path::Path;

use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    tensor::{Distribution, Tensor},
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use crate::{Error, Result};
use crate::{
    batch::{MoleculeAutoencoderBatch, SparseFingerprintBatch},
    features::REGRESSION_TARGET_WIDTH,
    fingerprints::{DEFAULT_ECFP_RADIUS, DEFAULT_ECFP_SIZE},
    ranking::{TanimotoRankingDiagnostics, weighted_tanimoto_ranking_output},
};

/// Default latent embedding width for the v1 architecture.
pub const DEFAULT_LATENT_WIDTH: usize = 512;

/// Default encoder hidden widths for the v1 architecture (decoder mirrors).
pub const DEFAULT_HIDDEN_WIDTHS: [usize; 3] = [4096, 2048, 1024];

/// Default descriptor regression loss weight.
pub const DEFAULT_DESCRIPTOR_WEIGHT: f64 = 0.05;

/// Default latent Tanimoto geometry loss weight.
pub const DEFAULT_TANIMOTO_RANKING_WEIGHT: f64 = 0.10;

/// Default decoder-side latent denoising noise as a fraction of batch std.
pub const DEFAULT_LATENT_NOISE_STD: f64 = 0.02;

/// Default latent cosine-logit temperature in the ranking softmax.
///
/// With cosine similarity in `[-1, 1]`, dividing by `0.05` gives a logit range
/// of `[-20, 20]`. A typical winner-vs-runner-up cosine gap of `0.3` maps to
/// `6` nats of softmax advantage, enough to put `> 99%` mass on the winner
/// without collapsing the gradient on already-correct anchors. Matches the
/// usual sampled-softmax contrastive temperature (SimCLR, Sentence-BERT,
/// MoCo, ...) for `k ≈ 16` candidates.
pub const DEFAULT_TANIMOTO_RANKING_LATENT_TEMPERATURE: f64 = 0.05;

/// Default compatibility metric temperature (unused by the softmax loss).
pub const DEFAULT_TANIMOTO_RANKING_METRIC_TEMPERATURE: f64 = 0.10;

/// Default minimum counted-Tanimoto gap below which an anchor is dropped.
pub const DEFAULT_TANIMOTO_RANKING_MIN_GAP: f64 = 0.05;

/// Default number of random candidate partners sampled per anchor.
pub const DEFAULT_TANIMOTO_RANKING_CANDIDATES_PER_ANCHOR: usize = 32;

/// Default max anchors per batch for the ranking loss (`0` uses all rows).
pub const DEFAULT_TANIMOTO_RANKING_PAIRS_PER_BATCH: usize = 0;

/// Default SoftRank Spearman surrogate temperature.
///
/// `0.20` keeps the pairwise sigmoid in a less-saturated regime than the
/// `0.10` spectral default — Tanimoto similarities live in `[0, 1]`
/// natively, so the divided-by-`latent_temperature` logits are on a
/// different scale than the spectral cosine logits the SoftRank surrogate
/// was originally tuned against. Sweep `{0.05, 0.10, 0.20, 0.50}` when
/// A/B-ing against the softmax CE baseline.
pub const DEFAULT_TANIMOTO_RANKING_SOFTRANK_TEMPERATURE: f64 = 0.20;

/// Default Huber transition point in log-count space.
pub const DEFAULT_RECONSTRUCTION_BETA: f64 = 1.0;

/// Default weight for zero-count bins in the reconstruction loss.
pub const DEFAULT_RECONSTRUCTION_ZERO_WEIGHT: f64 = 0.05;

/// Default weight for nonzero-count bins in the reconstruction loss.
pub const DEFAULT_RECONSTRUCTION_NONZERO_WEIGHT: f64 = 1.0;

/// Default weight for the bit-presence BCE auxiliary reconstruction loss.
pub const DEFAULT_BCE_WEIGHT: f64 = 1.0;

/// Default per-position weight applied to inactive bins inside the BCE term.
///
/// Higher than the Huber zero weight so the BCE actually drives non-target
/// logits negative instead of letting them leak slightly above zero.
pub const DEFAULT_BCE_ZERO_WEIGHT: f64 = 4.0;

/// Default per-position weight applied to active bins inside the BCE term.
pub const DEFAULT_BCE_NONZERO_WEIGHT: f64 = 1.0;

/// Default clamp ceiling applied to predicted log-rates before the
/// `exp` step inside the Poisson NLL reconstruction loss.
///
/// `exp(20) ≈ 4.85e8` is well within `f32` range even when summed across
/// `[batch=10240, width=4096]` cells (≈ 2e16 worst case), but it is high
/// enough that the optimizer can drive z anywhere a sensible learned
/// log-rate would land for typical small-integer ECFP counts. Real
/// training settles deep into the negatives at inactive bins and within
/// `[0, log(small_int)]` at active bins, so the clamp only fires on
/// pathological early-training spikes.
pub const DEFAULT_RECONSTRUCTION_POISSON_MAX_LOG_COUNT: f64 = 20.0;

/// Selectable reconstruction loss kind.
///
/// Switches between the existing weighted Huber on `log1p(count)` (with
/// the auxiliary bit-presence BCE term) and a Poisson NLL on the raw
/// counts. Under Poisson the model output is reinterpreted as
/// `log(rate)`; the metric path applies `softplus` before decoding so
/// the existing Tanimoto helpers (which expect `log1p(count)`) stay
/// numerically equivalent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReconstructionLossKind {
    /// Weighted Huber on `log1p(count)` (with the auxiliary bit-presence
    /// BCE term applied alongside per `bce_weight`). Shipping baseline.
    #[default]
    Huber,
    /// Poisson NLL on raw counts (`exp(z) - count · z` per cell, averaged
    /// over `batch × width`). No `bce_*` knobs apply (Poisson already
    /// handles both presence and magnitude); `--bce-weight 0` is the
    /// natural pairing to keep the BCE term out of the total loss.
    Poisson,
}

// Serde requires `#[serde(default = "fn")]` to point at a callable, not a
// `const`, so these tiny shim fns exist solely so deserializers can default
// missing JSON fields. Their bodies must stay in sync with the public consts
// above; do not edit one without the other.

/// Serde fallback for the descriptor-head loss weight.
const fn default_descriptor_weight() -> f64 {
    DEFAULT_DESCRIPTOR_WEIGHT
}

/// Serde fallback for the Tanimoto ranking loss weight.
const fn default_tanimoto_ranking_weight() -> f64 {
    DEFAULT_TANIMOTO_RANKING_WEIGHT
}

/// Serde fallback for the latent-noise standard deviation.
const fn default_latent_noise_std() -> f64 {
    DEFAULT_LATENT_NOISE_STD
}

/// Serde fallback for the latent cosine-logit temperature.
const fn default_tanimoto_ranking_latent_temperature() -> f64 {
    DEFAULT_TANIMOTO_RANKING_LATENT_TEMPERATURE
}

/// Serde fallback for the metric-side temperature kept for diagnostics.
const fn default_tanimoto_ranking_metric_temperature() -> f64 {
    DEFAULT_TANIMOTO_RANKING_METRIC_TEMPERATURE
}

/// Serde fallback for the SoftRank Spearman surrogate temperature.
const fn default_tanimoto_ranking_softrank_temperature() -> f64 {
    DEFAULT_TANIMOTO_RANKING_SOFTRANK_TEMPERATURE
}

/// Selectable Tanimoto-geometry ranking loss kind.
///
/// Picks between the existing gap-weighted softmax cross-entropy on the
/// top-1 target and the Taylor-style SoftRank Spearman surrogate
/// `1 - Pearson(soft_rank(logits), soft_rank(teacher_scores))`.
/// Diagnostics (MRR / Pearson / Spearman / accuracy) are computed
/// identically in both modes; only the gradient signal changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TanimotoLossKind {
    /// Gap-weighted softmax cross-entropy against the one-hot top-1
    /// candidate. Tested and shipping baseline.
    #[default]
    SoftmaxCe,
    /// Gap-weighted `1 − Pearson(soft_rank(logits),
    /// soft_rank(teacher_scores))` SoftRank Spearman surrogate, used to
    /// learn the full per-candidate ordering instead of just the
    /// argmax. Experimental.
    ///
    /// `#[serde(rename = ...)]` pins the on-disk spelling to the same
    /// `softrank-spearman` form the CLI accepts; the default kebab-case
    /// rule would produce `soft-rank-spearman` (three words), which
    /// would not round-trip with the CLI flag.
    #[serde(rename = "softrank-spearman")]
    SoftRankSpearman,
}

/// Serde fallback for the Tanimoto ranking loss kind. Returns the
/// shipping baseline (`SoftmaxCe`) so deserializing an older config
/// keeps its existing behavior.
const fn default_tanimoto_ranking_loss_kind() -> TanimotoLossKind {
    TanimotoLossKind::SoftmaxCe
}

/// Serde fallback for the counted-ECFP radius hyperparameter.
const fn default_ecfp_radius() -> u8 {
    DEFAULT_ECFP_RADIUS
}

/// Serde fallback for the bit-presence BCE term weight.
const fn default_bce_weight() -> f64 {
    DEFAULT_BCE_WEIGHT
}

/// Serde fallback for the zero-target BCE per-position weight.
const fn default_bce_zero_weight() -> f64 {
    DEFAULT_BCE_ZERO_WEIGHT
}

/// Serde fallback for the active-bin BCE per-position weight.
const fn default_bce_nonzero_weight() -> f64 {
    DEFAULT_BCE_NONZERO_WEIGHT
}

/// Serde fallback for the reconstruction loss kind. Returns the
/// shipping baseline (`Huber`) so old configs deserialise to the
/// existing behaviour.
const fn default_reconstruction_loss_kind() -> ReconstructionLossKind {
    ReconstructionLossKind::Huber
}

/// Serde fallback for the Poisson clamp ceiling.
const fn default_reconstruction_poisson_max_log_count() -> f64 {
    DEFAULT_RECONSTRUCTION_POISSON_MAX_LOG_COUNT
}

/// Encoder model configuration. Construct via [`EncoderConfig::builder`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Width of the dense input vector fed to the first encoder layer
    /// (equals the counted ECFP fingerprint size).
    input_width: usize,
    /// Width of each ReLU-activated hidden layer between input and
    /// latent. Must be non-empty.
    hidden_widths: Vec<usize>,
    /// Width of the latent embedding produced by the final linear layer.
    latent_width: usize,
}

impl EncoderConfig {
    /// Starts a fluent builder.
    #[must_use]
    pub fn builder() -> EncoderConfigBuilder {
        EncoderConfigBuilder::new()
    }

    /// Input feature width.
    #[must_use]
    pub const fn input_width(&self) -> usize {
        self.input_width
    }

    /// Hidden layer widths.
    #[must_use]
    pub fn hidden_widths(&self) -> &[usize] {
        &self.hidden_widths
    }

    /// Latent embedding width.
    #[must_use]
    pub const fn latent_width(&self) -> usize {
        self.latent_width
    }

    /// Creates an initialized encoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let first_width = self.hidden_widths[0];
        let mut input_width = first_width;
        let mut layers = Vec::with_capacity(self.hidden_widths.len().saturating_sub(1));
        for &hidden_width in self.hidden_widths.iter().skip(1) {
            layers.push(LinearConfig::new(input_width, hidden_width).init(device));
            input_width = hidden_width;
        }

        Encoder {
            input: SparseInputLinear {
                linear: LinearConfig::new(self.input_width, first_width).init(device),
            },
            layers,
            latent: LinearConfig::new(input_width, self.latent_width).init(device),
            activation: Relu::new(),
        }
    }
}

/// Fluent builder for [`EncoderConfig`].
#[derive(Debug, Clone)]
pub struct EncoderConfigBuilder {
    /// Pending input width; `None` until [`Self::input_width`] is called.
    input_width: Option<usize>,
    /// Pending hidden layer widths; `None` until set.
    hidden_widths: Option<Vec<usize>>,
    /// Pending latent width; `None` until set.
    latent_width: Option<usize>,
}

impl Default for EncoderConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderConfigBuilder {
    /// Creates an empty builder; all fields must be set before `build`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            input_width: None,
            hidden_widths: None,
            latent_width: None,
        }
    }

    /// Sets the input feature width.
    #[must_use]
    pub const fn input_width(mut self, value: usize) -> Self {
        self.input_width = Some(value);
        self
    }

    /// Sets the hidden layer widths.
    #[must_use]
    pub fn hidden_widths(mut self, value: Vec<usize>) -> Self {
        self.hidden_widths = Some(value);
        self
    }

    /// Sets the latent embedding width.
    #[must_use]
    pub const fn latent_width(mut self, value: usize) -> Self {
        self.latent_width = Some(value);
        self
    }

    /// Validates and builds the immutable [`EncoderConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when any required field is missing,
    /// when the latent or input width is zero, or when `hidden_widths` is
    /// empty / contains zeros.
    pub fn build(self) -> Result<EncoderConfig> {
        let input_width = self.input_width.ok_or_else(|| Error::ConfigInvalid {
            message: "encoder input_width must be set".to_string(),
        })?;
        let hidden_widths = self.hidden_widths.ok_or_else(|| Error::ConfigInvalid {
            message: "encoder hidden_widths must be set".to_string(),
        })?;
        let latent_width = self.latent_width.ok_or_else(|| Error::ConfigInvalid {
            message: "encoder latent_width must be set".to_string(),
        })?;
        if input_width == 0 {
            return Err(Error::ConfigInvalid {
                message: "encoder input_width must be greater than zero".to_string(),
            });
        }
        if latent_width == 0 {
            return Err(Error::ConfigInvalid {
                message: "encoder latent_width must be greater than zero".to_string(),
            });
        }
        if hidden_widths.is_empty() {
            return Err(Error::ConfigInvalid {
                message: "encoder hidden_widths must not be empty".to_string(),
            });
        }
        if hidden_widths.contains(&0) {
            return Err(Error::ConfigInvalid {
                message: "encoder hidden widths must all be greater than zero".to_string(),
            });
        }
        Ok(EncoderConfig {
            input_width,
            hidden_widths,
            latent_width,
        })
    }
}

/// Decoder model configuration. Construct via [`DecoderConfig::builder`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Width of the latent embedding fed to the first decoder layer.
    latent_width: usize,
    /// Width of each ReLU-activated hidden layer between latent and
    /// output. May be empty (linear projection from latent to output).
    hidden_widths: Vec<usize>,
    /// Reconstructed log-count width emitted by the decoder; equals the
    /// counted ECFP fingerprint size.
    output_width: usize,
}

impl DecoderConfig {
    /// Starts a fluent builder.
    #[must_use]
    pub fn builder() -> DecoderConfigBuilder {
        DecoderConfigBuilder::new()
    }

    /// Latent embedding width.
    #[must_use]
    pub const fn latent_width(&self) -> usize {
        self.latent_width
    }

    /// Hidden layer widths.
    #[must_use]
    pub fn hidden_widths(&self) -> &[usize] {
        &self.hidden_widths
    }

    /// Output reconstruction width.
    #[must_use]
    pub const fn output_width(&self) -> usize {
        self.output_width
    }

    /// Creates an initialized decoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        let mut input_width = self.latent_width;
        let mut layers = Vec::with_capacity(self.hidden_widths.len());
        for &hidden_width in &self.hidden_widths {
            layers.push(LinearConfig::new(input_width, hidden_width).init(device));
            input_width = hidden_width;
        }

        Decoder {
            layers,
            output: LinearConfig::new(input_width, self.output_width).init(device),
            activation: Relu::new(),
        }
    }
}

/// Fluent builder for [`DecoderConfig`].
#[derive(Debug, Clone)]
pub struct DecoderConfigBuilder {
    /// Pending latent width; `None` until [`Self::latent_width`] is called.
    latent_width: Option<usize>,
    /// Pending hidden layer widths; `None` defaults to an empty Vec.
    hidden_widths: Option<Vec<usize>>,
    /// Pending output width; `None` until [`Self::output_width`] is called.
    output_width: Option<usize>,
}

impl Default for DecoderConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DecoderConfigBuilder {
    /// Creates an empty builder; all required fields must be set before
    /// `build`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            latent_width: None,
            hidden_widths: None,
            output_width: None,
        }
    }

    /// Sets the latent embedding width.
    #[must_use]
    pub const fn latent_width(mut self, value: usize) -> Self {
        self.latent_width = Some(value);
        self
    }

    /// Sets the hidden layer widths (may be empty).
    #[must_use]
    pub fn hidden_widths(mut self, value: Vec<usize>) -> Self {
        self.hidden_widths = Some(value);
        self
    }

    /// Sets the output reconstruction width.
    #[must_use]
    pub const fn output_width(mut self, value: usize) -> Self {
        self.output_width = Some(value);
        self
    }

    /// Validates and builds the immutable [`DecoderConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when `latent_width` or `output_width`
    /// is missing or zero, or when any hidden width is zero.
    pub fn build(self) -> Result<DecoderConfig> {
        let latent_width = self.latent_width.ok_or_else(|| Error::ConfigInvalid {
            message: "decoder latent_width must be set".to_string(),
        })?;
        let output_width = self.output_width.ok_or_else(|| Error::ConfigInvalid {
            message: "decoder output_width must be set".to_string(),
        })?;
        let hidden_widths = self.hidden_widths.unwrap_or_default();
        if latent_width == 0 {
            return Err(Error::ConfigInvalid {
                message: "decoder latent_width must be greater than zero".to_string(),
            });
        }
        if output_width == 0 {
            return Err(Error::ConfigInvalid {
                message: "decoder output_width must be greater than zero".to_string(),
            });
        }
        if hidden_widths.contains(&0) {
            return Err(Error::ConfigInvalid {
                message: "decoder hidden widths must all be greater than zero".to_string(),
            });
        }
        Ok(DecoderConfig {
            latent_width,
            hidden_widths,
            output_width,
        })
    }
}

/// Counted ECFP reconstruction loss configuration. Construct via
/// [`ReconstructionLossConfig::builder`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionLossConfig {
    /// Huber transition point in `log1p(count)` space.
    beta: f64,
    /// Per-position weight for zero-count target bins.
    zero_weight: f64,
    /// Per-position weight for nonzero-count target bins.
    nonzero_weight: f64,
    /// Total weight for the auxiliary bit-presence BCE-with-logits term;
    /// `0.0` disables the BCE branch entirely.
    #[serde(default = "default_bce_weight")]
    bce_weight: f64,
    /// Per-position weight applied to inactive bins inside the BCE term.
    #[serde(default = "default_bce_zero_weight")]
    bce_zero_weight: f64,
    /// Per-position weight applied to active bins inside the BCE term.
    #[serde(default = "default_bce_nonzero_weight")]
    bce_nonzero_weight: f64,
    /// Which reconstruction-loss kind the training loop should optimise.
    /// Defaults to the shipping Huber baseline.
    #[serde(default = "default_reconstruction_loss_kind")]
    loss_kind: ReconstructionLossKind,
    /// Clamp ceiling applied to predicted log-rates before `exp` inside
    /// the Poisson NLL branch. Ignored under [`ReconstructionLossKind::Huber`].
    #[serde(default = "default_reconstruction_poisson_max_log_count")]
    poisson_max_log_count: f64,
}

impl Default for ReconstructionLossConfig {
    fn default() -> Self {
        ReconstructionLossConfigBuilder::new()
            .build()
            .expect("default reconstruction loss config is valid")
    }
}

impl ReconstructionLossConfig {
    /// Starts a fluent builder.
    #[must_use]
    pub fn builder() -> ReconstructionLossConfigBuilder {
        ReconstructionLossConfigBuilder::new()
    }

    /// Huber transition point in log-count space.
    #[must_use]
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    /// Weight for zero-count bins.
    #[must_use]
    pub const fn zero_weight(&self) -> f64 {
        self.zero_weight
    }

    /// Weight for nonzero-count bins.
    #[must_use]
    pub const fn nonzero_weight(&self) -> f64 {
        self.nonzero_weight
    }

    /// Auxiliary bit-presence BCE-with-logits loss weight. Set to `0.0` to
    /// disable the BCE term entirely.
    #[must_use]
    pub const fn bce_weight(&self) -> f64 {
        self.bce_weight
    }

    /// Per-position weight applied to inactive bins inside the BCE term.
    #[must_use]
    pub const fn bce_zero_weight(&self) -> f64 {
        self.bce_zero_weight
    }

    /// Per-position weight applied to active bins inside the BCE term.
    #[must_use]
    pub const fn bce_nonzero_weight(&self) -> f64 {
        self.bce_nonzero_weight
    }

    /// Active reconstruction loss kind.
    #[must_use]
    pub const fn loss_kind(&self) -> ReconstructionLossKind {
        self.loss_kind
    }

    /// Clamp ceiling applied to predicted log-rates before `exp` in the
    /// Poisson branch.
    #[must_use]
    pub const fn poisson_max_log_count(&self) -> f64 {
        self.poisson_max_log_count
    }
}

/// Fluent builder for [`ReconstructionLossConfig`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructionLossConfigBuilder {
    /// Pending Huber transition point.
    beta: f64,
    /// Pending zero-bin weight.
    zero_weight: f64,
    /// Pending nonzero-bin weight.
    nonzero_weight: f64,
    /// Pending overall BCE term weight.
    bce_weight: f64,
    /// Pending per-position BCE weight for inactive bins.
    bce_zero_weight: f64,
    /// Pending per-position BCE weight for active bins.
    bce_nonzero_weight: f64,
    /// Pending reconstruction loss kind.
    loss_kind: ReconstructionLossKind,
    /// Pending Poisson NLL exp clamp ceiling.
    poisson_max_log_count: f64,
}

impl Default for ReconstructionLossConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ReconstructionLossConfigBuilder {
    /// Creates a builder seeded with the v1 defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            beta: DEFAULT_RECONSTRUCTION_BETA,
            zero_weight: DEFAULT_RECONSTRUCTION_ZERO_WEIGHT,
            nonzero_weight: DEFAULT_RECONSTRUCTION_NONZERO_WEIGHT,
            bce_weight: DEFAULT_BCE_WEIGHT,
            bce_zero_weight: DEFAULT_BCE_ZERO_WEIGHT,
            bce_nonzero_weight: DEFAULT_BCE_NONZERO_WEIGHT,
            loss_kind: ReconstructionLossKind::Huber,
            poisson_max_log_count: DEFAULT_RECONSTRUCTION_POISSON_MAX_LOG_COUNT,
        }
    }

    /// Sets the Huber transition point.
    #[must_use]
    pub const fn beta(mut self, value: f64) -> Self {
        self.beta = value;
        self
    }

    /// Sets the zero-count bin weight.
    #[must_use]
    pub const fn zero_weight(mut self, value: f64) -> Self {
        self.zero_weight = value;
        self
    }

    /// Sets the nonzero-count bin weight.
    #[must_use]
    pub const fn nonzero_weight(mut self, value: f64) -> Self {
        self.nonzero_weight = value;
        self
    }

    /// Sets the BCE auxiliary loss weight (`0.0` disables the BCE term).
    #[must_use]
    pub const fn bce_weight(mut self, value: f64) -> Self {
        self.bce_weight = value;
        self
    }

    /// Sets the per-position weight applied to inactive bins inside the BCE term.
    #[must_use]
    pub const fn bce_zero_weight(mut self, value: f64) -> Self {
        self.bce_zero_weight = value;
        self
    }

    /// Sets the per-position weight applied to active bins inside the BCE term.
    #[must_use]
    pub const fn bce_nonzero_weight(mut self, value: f64) -> Self {
        self.bce_nonzero_weight = value;
        self
    }

    /// Sets the reconstruction loss kind (Huber baseline vs Poisson NLL).
    #[must_use]
    pub const fn loss_kind(mut self, value: ReconstructionLossKind) -> Self {
        self.loss_kind = value;
        self
    }

    /// Sets the clamp ceiling applied to predicted log-rates before
    /// `exp` in the Poisson branch.
    #[must_use]
    pub const fn poisson_max_log_count(mut self, value: f64) -> Self {
        self.poisson_max_log_count = value;
        self
    }

    /// Validates and builds the immutable [`ReconstructionLossConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when `beta` is non-finite or
    /// non-positive, or when either weight is negative or non-finite.
    pub fn build(self) -> Result<ReconstructionLossConfig> {
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(Error::ConfigInvalid {
                message: format!(
                    "reconstruction loss beta must be positive and finite, got {}",
                    self.beta
                ),
            });
        }
        for (label, value) in [
            ("zero_weight", self.zero_weight),
            ("nonzero_weight", self.nonzero_weight),
            ("bce_weight", self.bce_weight),
            ("bce_zero_weight", self.bce_zero_weight),
            ("bce_nonzero_weight", self.bce_nonzero_weight),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::ConfigInvalid {
                    message: format!("{label} must be finite and non-negative, got {value}"),
                });
            }
        }
        if !self.poisson_max_log_count.is_finite() || self.poisson_max_log_count <= 0.0 {
            return Err(Error::ConfigInvalid {
                message: format!(
                    "poisson_max_log_count must be positive and finite, got {}",
                    self.poisson_max_log_count
                ),
            });
        }
        Ok(ReconstructionLossConfig {
            beta: self.beta,
            zero_weight: self.zero_weight,
            nonzero_weight: self.nonzero_weight,
            bce_weight: self.bce_weight,
            bce_zero_weight: self.bce_zero_weight,
            bce_nonzero_weight: self.bce_nonzero_weight,
            loss_kind: self.loss_kind,
            poisson_max_log_count: self.poisson_max_log_count,
        })
    }
}

/// Default weight applied to the violation-classification side task.
pub const DEFAULT_VIOLATION_WEIGHT: f64 = 0.5;

/// Serde fallback for the violation-classification loss weight.
const fn default_violation_weight() -> f64 {
    DEFAULT_VIOLATION_WEIGHT
}

/// Auxiliary side-task weights. Construct via [`AuxiliaryLossWeights::builder`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AuxiliaryLossWeights {
    /// Weight applied to the scalar descriptor regression side task.
    #[serde(default = "default_descriptor_weight")]
    descriptors: f64,
    /// Weight applied to the latent Tanimoto geometry side task.
    #[serde(default = "default_tanimoto_ranking_weight")]
    tanimoto_ranking: f64,
    /// Weight applied to the multi-label violation-classification side
    /// task that reads from the latent embedding.
    #[serde(default = "default_violation_weight")]
    violation: f64,
}

impl Default for AuxiliaryLossWeights {
    fn default() -> Self {
        AuxiliaryLossWeightsBuilder::new()
            .build()
            .expect("default auxiliary loss weights are valid")
    }
}

impl AuxiliaryLossWeights {
    /// Starts a fluent builder.
    #[must_use]
    pub fn builder() -> AuxiliaryLossWeightsBuilder {
        AuxiliaryLossWeightsBuilder::new()
    }

    /// Weight for scalar descriptor regression.
    #[must_use]
    pub const fn descriptors(&self) -> f64 {
        self.descriptors
    }

    /// Weight for preserving counted-fingerprint Tanimoto ordering in latent
    /// space.
    #[must_use]
    pub const fn tanimoto_ranking(&self) -> f64 {
        self.tanimoto_ranking
    }

    /// Weight for the multi-label violation-classification side task.
    #[must_use]
    pub const fn violation(&self) -> f64 {
        self.violation
    }
}

/// Fluent builder for [`AuxiliaryLossWeights`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuxiliaryLossWeightsBuilder {
    /// Pending descriptor regression weight.
    descriptors: f64,
    /// Pending latent Tanimoto geometry weight.
    tanimoto_ranking: f64,
    /// Pending violation-classification weight.
    violation: f64,
}

impl Default for AuxiliaryLossWeightsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AuxiliaryLossWeightsBuilder {
    /// Creates a builder seeded with the v1 defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            descriptors: DEFAULT_DESCRIPTOR_WEIGHT,
            tanimoto_ranking: DEFAULT_TANIMOTO_RANKING_WEIGHT,
            violation: DEFAULT_VIOLATION_WEIGHT,
        }
    }

    /// Sets the descriptor regression loss weight.
    #[must_use]
    pub const fn descriptors(mut self, value: f64) -> Self {
        self.descriptors = value;
        self
    }

    /// Sets the latent Tanimoto geometry loss weight.
    #[must_use]
    pub const fn tanimoto_ranking(mut self, value: f64) -> Self {
        self.tanimoto_ranking = value;
        self
    }

    /// Sets the violation-classification loss weight.
    #[must_use]
    pub const fn violation(mut self, value: f64) -> Self {
        self.violation = value;
        self
    }

    /// Validates and builds the immutable [`AuxiliaryLossWeights`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when any weight is negative or
    /// non-finite.
    pub fn build(self) -> Result<AuxiliaryLossWeights> {
        for (label, value) in [
            ("descriptors", self.descriptors),
            ("tanimoto_ranking", self.tanimoto_ranking),
            ("violation", self.violation),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::ConfigInvalid {
                    message: format!("{label} weight must be finite and non-negative, got {value}"),
                });
            }
        }
        Ok(AuxiliaryLossWeights {
            descriptors: self.descriptors,
            tanimoto_ranking: self.tanimoto_ranking,
            violation: self.violation,
        })
    }
}

/// Counted Tanimoto sampled softmax latent-geometry loss configuration.
/// Construct via [`TanimotoRankingConfig::builder`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TanimotoRankingConfig {
    /// Temperature divisor applied to the latent cosine logits before
    /// the softmax. Older configs surface this as `margin`; the aliases
    /// are kept for backwards compatibility.
    #[serde(
        default = "default_tanimoto_ranking_latent_temperature",
        alias = "margin",
        alias = "tanimoto_ranking_margin"
    )]
    latent_temperature: f64,
    /// Compatibility temperature retained for diagnostics; unused by
    /// the current softmax loss.
    #[serde(default = "default_tanimoto_ranking_metric_temperature")]
    metric_temperature: f64,
    /// Minimum counted-Tanimoto top-1 / top-2 gap an anchor must clear
    /// to contribute to the loss.
    min_gap: f64,
    /// Random candidate partners sampled per anchor by the GPU kernel.
    candidates_per_anchor: usize,
    /// Maximum anchors used per batch by the geometry loss; `0` uses
    /// every row.
    pairs_per_batch: usize,
    /// Which ranking loss the training loop should optimise. Defaults to
    /// the shipping softmax cross-entropy baseline.
    #[serde(default = "default_tanimoto_ranking_loss_kind")]
    loss_kind: TanimotoLossKind,
    /// Temperature for the pairwise sigmoid inside the SoftRank Spearman
    /// surrogate. Ignored when [`Self::loss_kind`] is
    /// [`TanimotoLossKind::SoftmaxCe`].
    #[serde(default = "default_tanimoto_ranking_softrank_temperature")]
    softrank_temperature: f64,
}

impl Default for TanimotoRankingConfig {
    fn default() -> Self {
        TanimotoRankingConfigBuilder::new()
            .build()
            .expect("default Tanimoto ranking config is valid")
    }
}

impl TanimotoRankingConfig {
    /// Starts a fluent builder.
    #[must_use]
    pub fn builder() -> TanimotoRankingConfigBuilder {
        TanimotoRankingConfigBuilder::new()
    }

    /// Temperature applied to latent cosine logits.
    #[must_use]
    pub const fn latent_temperature(&self) -> f64 {
        self.latent_temperature
    }

    /// Deprecated compatibility temperature retained for diagnostics.
    #[must_use]
    pub const fn metric_temperature(&self) -> f64 {
        self.metric_temperature
    }

    /// Minimum counted Tanimoto gap required for an anchor to contribute.
    #[must_use]
    pub const fn min_gap(&self) -> f64 {
        self.min_gap
    }

    /// Random candidate partners sampled per anchor by the GPU kernel.
    #[must_use]
    pub const fn candidates_per_anchor(&self) -> usize {
        self.candidates_per_anchor
    }

    /// Maximum anchors used by the latent geometry loss; `0` means all rows.
    #[must_use]
    pub const fn pairs_per_batch(&self) -> usize {
        self.pairs_per_batch
    }

    /// Ranking loss kind selected for this configuration.
    #[must_use]
    pub const fn loss_kind(&self) -> TanimotoLossKind {
        self.loss_kind
    }

    /// SoftRank Spearman surrogate temperature.
    #[must_use]
    pub const fn softrank_temperature(&self) -> f64 {
        self.softrank_temperature
    }
}

/// Fluent builder for [`TanimotoRankingConfig`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TanimotoRankingConfigBuilder {
    /// Pending latent cosine-logit temperature.
    latent_temperature: f64,
    /// Pending compatibility metric temperature.
    metric_temperature: f64,
    /// Pending minimum metric gap.
    min_gap: f64,
    /// Pending candidate partners per anchor.
    candidates_per_anchor: usize,
    /// Pending maximum anchors per batch (`0` for all).
    pairs_per_batch: usize,
    /// Pending ranking loss kind.
    loss_kind: TanimotoLossKind,
    /// Pending SoftRank Spearman surrogate temperature.
    softrank_temperature: f64,
}

impl Default for TanimotoRankingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TanimotoRankingConfigBuilder {
    /// Creates a builder seeded with the v1 defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            latent_temperature: DEFAULT_TANIMOTO_RANKING_LATENT_TEMPERATURE,
            metric_temperature: DEFAULT_TANIMOTO_RANKING_METRIC_TEMPERATURE,
            min_gap: DEFAULT_TANIMOTO_RANKING_MIN_GAP,
            candidates_per_anchor: DEFAULT_TANIMOTO_RANKING_CANDIDATES_PER_ANCHOR,
            pairs_per_batch: DEFAULT_TANIMOTO_RANKING_PAIRS_PER_BATCH,
            loss_kind: TanimotoLossKind::SoftmaxCe,
            softrank_temperature: DEFAULT_TANIMOTO_RANKING_SOFTRANK_TEMPERATURE,
        }
    }

    /// Sets the latent cosine-logit temperature.
    #[must_use]
    pub const fn latent_temperature(mut self, value: f64) -> Self {
        self.latent_temperature = value;
        self
    }

    /// Sets the (compatibility) metric temperature.
    #[must_use]
    pub const fn metric_temperature(mut self, value: f64) -> Self {
        self.metric_temperature = value;
        self
    }

    /// Sets the minimum counted-Tanimoto gap.
    #[must_use]
    pub const fn min_gap(mut self, value: f64) -> Self {
        self.min_gap = value;
        self
    }

    /// Sets the number of random candidate partners per anchor.
    #[must_use]
    pub const fn candidates_per_anchor(mut self, value: usize) -> Self {
        self.candidates_per_anchor = value;
        self
    }

    /// Sets the maximum anchors per batch (`0` uses all rows).
    #[must_use]
    pub const fn pairs_per_batch(mut self, value: usize) -> Self {
        self.pairs_per_batch = value;
        self
    }

    /// Sets the ranking loss kind.
    #[must_use]
    pub const fn loss_kind(mut self, value: TanimotoLossKind) -> Self {
        self.loss_kind = value;
        self
    }

    /// Sets the SoftRank Spearman surrogate temperature.
    #[must_use]
    pub const fn softrank_temperature(mut self, value: f64) -> Self {
        self.softrank_temperature = value;
        self
    }

    /// Validates and builds the immutable [`TanimotoRankingConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when either temperature is non-finite
    /// or non-positive, or when `min_gap` is negative / non-finite.
    pub fn build(self) -> Result<TanimotoRankingConfig> {
        if !self.latent_temperature.is_finite() || self.latent_temperature <= 0.0 {
            return Err(Error::ConfigInvalid {
                message: format!(
                    "tanimoto ranking latent temperature must be positive and finite, got {}",
                    self.latent_temperature
                ),
            });
        }
        if !self.metric_temperature.is_finite() || self.metric_temperature <= 0.0 {
            return Err(Error::ConfigInvalid {
                message: format!(
                    "tanimoto ranking metric temperature must be positive and finite, got {}",
                    self.metric_temperature
                ),
            });
        }
        if !self.min_gap.is_finite() || self.min_gap < 0.0 {
            return Err(Error::ConfigInvalid {
                message: format!(
                    "tanimoto ranking min_gap must be non-negative and finite, got {}",
                    self.min_gap
                ),
            });
        }
        if !self.softrank_temperature.is_finite() || self.softrank_temperature <= 0.0 {
            return Err(Error::ConfigInvalid {
                message: format!(
                    "tanimoto ranking softrank_temperature must be positive and finite, got {}",
                    self.softrank_temperature
                ),
            });
        }
        Ok(TanimotoRankingConfig {
            latent_temperature: self.latent_temperature,
            metric_temperature: self.metric_temperature,
            min_gap: self.min_gap,
            candidates_per_anchor: self.candidates_per_anchor,
            pairs_per_batch: self.pairs_per_batch,
            loss_kind: self.loss_kind,
            softrank_temperature: self.softrank_temperature,
        })
    }
}

/// Flat runtime view of the Tanimoto geometry side task.
///
/// Bundles the per-task weight stored in [`AuxiliaryLossWeights`] with the
/// shape and temperature parameters in [`TanimotoRankingConfig`] so training
/// loops can pass a single value around without re-reading the model config.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TanimotoRankingRuntimeConfig {
    /// Weight applied to the Tanimoto geometry loss component.
    weight: f64,
    /// Temperature divisor for the latent cosine-similarity logits.
    latent_temperature: f64,
    /// Compatibility temperature kept for diagnostics; unused by the
    /// current softmax loss.
    metric_temperature: f64,
    /// Minimum top-1 / top-2 metric gap required for an anchor to
    /// contribute to the loss.
    min_gap: f64,
    /// Random candidate partners sampled per anchor by the GPU kernel.
    candidates_per_anchor: usize,
    /// Maximum anchors used per batch; `0` uses every row.
    pairs_per_batch: usize,
    /// Active ranking loss kind for this run.
    loss_kind: TanimotoLossKind,
    /// SoftRank Spearman surrogate temperature (ignored when `loss_kind`
    /// is [`TanimotoLossKind::SoftmaxCe`]).
    softrank_temperature: f64,
}

impl TanimotoRankingRuntimeConfig {
    /// Weight applied to the Tanimoto geometry loss component.
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }

    /// Temperature applied to latent cosine logits.
    #[must_use]
    pub const fn latent_temperature(&self) -> f64 {
        self.latent_temperature
    }

    /// Compatibility field kept for diagnostics; unused by the softmax loss.
    #[must_use]
    pub const fn metric_temperature(&self) -> f64 {
        self.metric_temperature
    }

    /// Minimum counted Tanimoto gap required for an anchor to contribute.
    #[must_use]
    pub const fn min_gap(&self) -> f64 {
        self.min_gap
    }

    /// Random candidate partners sampled per anchor by the GPU kernel.
    #[must_use]
    pub const fn candidates_per_anchor(&self) -> usize {
        self.candidates_per_anchor
    }

    /// Maximum anchors used by the latent geometry loss; `0` means all rows.
    #[must_use]
    pub const fn pairs_per_batch(&self) -> usize {
        self.pairs_per_batch
    }

    /// Ranking loss kind selected for this run.
    #[must_use]
    pub const fn loss_kind(&self) -> TanimotoLossKind {
        self.loss_kind
    }

    /// SoftRank Spearman surrogate temperature.
    #[must_use]
    pub const fn softrank_temperature(&self) -> f64 {
        self.softrank_temperature
    }
}

/// Full molecule autoencoder configuration. Construct via
/// [`MoleculeAutoencoderConfig::builder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoleculeAutoencoderConfig {
    /// Encoder MLP shape (fingerprint → latent).
    encoder: EncoderConfig,
    /// Decoder MLP shape (latent → reconstructed fingerprint).
    decoder: DecoderConfig,
    /// Width of the scalar descriptor regression side head.
    descriptor_width: usize,
    /// Per-bin reconstruction loss parameters (Huber β + BCE knobs).
    reconstruction_loss: ReconstructionLossConfig,
    /// Side-task weights bundled together for clarity.
    auxiliary_weights: AuxiliaryLossWeights,
    /// Hyperparameters for the latent Tanimoto geometry loss.
    #[serde(default)]
    tanimoto_ranking: TanimotoRankingConfig,
    /// Standard deviation of the Gaussian noise injected into the
    /// decoder-side latent during training (`0.0` disables denoising).
    #[serde(default = "default_latent_noise_std")]
    latent_noise_std: f64,
    /// ECFP radius used by the preprocessing pipeline (must match the
    /// shards being trained on).
    #[serde(default = "default_ecfp_radius")]
    ecfp_radius: u8,
}

impl MoleculeAutoencoderConfig {
    /// Starting v1 configuration: 4k counted ECFP input and 512-d latent.
    #[must_use]
    pub fn v1_counted_ecfp() -> Self {
        Self::symmetric(DEFAULT_ECFP_SIZE, 512, vec![4096, 2048, 1024])
    }

    /// Creates a symmetric MLP autoencoder configuration.
    #[must_use]
    pub fn symmetric(input_width: usize, latent_width: usize, hidden_widths: Vec<usize>) -> Self {
        let mut decoder_hidden_widths = hidden_widths.clone();
        decoder_hidden_widths.reverse();
        Self {
            encoder: EncoderConfig {
                input_width,
                hidden_widths,
                latent_width,
            },
            decoder: DecoderConfig {
                latent_width,
                hidden_widths: decoder_hidden_widths,
                output_width: input_width,
            },
            descriptor_width: REGRESSION_TARGET_WIDTH,
            reconstruction_loss: ReconstructionLossConfig::default(),
            auxiliary_weights: AuxiliaryLossWeights::default(),
            tanimoto_ranking: TanimotoRankingConfig::default(),
            latent_noise_std: default_latent_noise_std(),
            ecfp_radius: default_ecfp_radius(),
        }
    }

    /// Starts a fluent builder for an autoencoder config.
    #[must_use]
    pub fn builder() -> MoleculeAutoencoderConfigBuilder {
        MoleculeAutoencoderConfigBuilder::new()
    }

    /// Encoder configuration.
    #[must_use]
    pub const fn encoder(&self) -> &EncoderConfig {
        &self.encoder
    }

    /// Decoder configuration.
    #[must_use]
    pub const fn decoder(&self) -> &DecoderConfig {
        &self.decoder
    }

    /// Descriptor regression target width.
    #[must_use]
    pub const fn descriptor_width(&self) -> usize {
        self.descriptor_width
    }

    /// Main reconstruction loss configuration.
    #[must_use]
    pub const fn reconstruction_loss(&self) -> &ReconstructionLossConfig {
        &self.reconstruction_loss
    }

    /// Side-task loss weights.
    #[must_use]
    pub const fn auxiliary_weights(&self) -> &AuxiliaryLossWeights {
        &self.auxiliary_weights
    }

    /// Tanimoto sampled-softmax geometry side-task configuration.
    #[must_use]
    pub const fn tanimoto_ranking(&self) -> &TanimotoRankingConfig {
        &self.tanimoto_ranking
    }

    /// Decoder-side latent Gaussian denoising noise as a fraction of batch
    /// latent standard deviation. `0.0` disables latent denoising.
    #[must_use]
    pub const fn latent_noise_std(&self) -> f64 {
        self.latent_noise_std
    }

    /// Morgan/ECFP radius the cached fingerprints were generated with.
    /// Inference callers need this to reconstruct fingerprints at the same
    /// radius the model was trained against.
    #[must_use]
    pub const fn ecfp_radius(&self) -> u8 {
        self.ecfp_radius
    }

    /// Returns the flat runtime view of the Tanimoto geometry side task.
    #[must_use]
    pub const fn tanimoto_ranking_runtime(&self) -> TanimotoRankingRuntimeConfig {
        TanimotoRankingRuntimeConfig {
            weight: self.auxiliary_weights.tanimoto_ranking,
            latent_temperature: self.tanimoto_ranking.latent_temperature,
            metric_temperature: self.tanimoto_ranking.metric_temperature,
            min_gap: self.tanimoto_ranking.min_gap,
            candidates_per_anchor: self.tanimoto_ranking.candidates_per_anchor,
            pairs_per_batch: self.tanimoto_ranking.pairs_per_batch,
            loss_kind: self.tanimoto_ranking.loss_kind,
            softrank_temperature: self.tanimoto_ranking.softrank_temperature,
        }
    }

    /// Checks the configuration against an expected sparse-input width and
    /// the loss-weight invariants the training pipeline depends on.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when the encoder or decoder shape does
    /// not match `expected_input_width`, or when any loss weight, temperature,
    /// gap, or noise term is outside its valid range.
    #[cfg(feature = "std")]
    pub fn validate(&self, expected_input_width: usize) -> Result<()> {
        let bail = |message: String| -> Result<()> { Err(Error::ConfigInvalid { message }) };

        if self.encoder.input_width != expected_input_width {
            return bail(format!(
                "encoder input width {} does not match expected width {expected_input_width}",
                self.encoder.input_width
            ));
        }
        if self.decoder.output_width != expected_input_width {
            return bail(format!(
                "decoder output width {} does not match expected width {expected_input_width}",
                self.decoder.output_width
            ));
        }
        if self.encoder.hidden_widths.is_empty() {
            return bail("encoder hidden widths must not be empty".to_string());
        }
        if self.encoder.latent_width == 0 || self.decoder.latent_width == 0 {
            return bail("latent width must be greater than zero".to_string());
        }
        if self.encoder.latent_width != self.decoder.latent_width {
            return bail(format!(
                "encoder latent width {} does not match decoder latent width {}",
                self.encoder.latent_width, self.decoder.latent_width
            ));
        }
        if self.descriptor_width == 0 {
            return bail("descriptor width must be greater than zero".to_string());
        }
        if !self.reconstruction_loss.beta.is_finite() || self.reconstruction_loss.beta <= 0.0 {
            return bail(format!(
                "reconstruction loss beta must be positive and finite, got {}",
                self.reconstruction_loss.beta
            ));
        }
        if self.reconstruction_loss.zero_weight < 0.0
            || self.reconstruction_loss.nonzero_weight < 0.0
        {
            return bail("reconstruction loss weights must be non-negative".to_string());
        }
        if self.auxiliary_weights.descriptors < 0.0 || self.auxiliary_weights.tanimoto_ranking < 0.0
        {
            return bail("auxiliary loss weights must be non-negative".to_string());
        }
        if self.tanimoto_ranking.latent_temperature <= 0.0
            || !self.tanimoto_ranking.latent_temperature.is_finite()
        {
            return bail(format!(
                "tanimoto ranking latent temperature must be positive and finite, got {}",
                self.tanimoto_ranking.latent_temperature
            ));
        }
        if self.tanimoto_ranking.metric_temperature <= 0.0
            || !self.tanimoto_ranking.metric_temperature.is_finite()
        {
            return bail(format!(
                "tanimoto ranking metric temperature must be positive and finite, got {}",
                self.tanimoto_ranking.metric_temperature
            ));
        }
        if self.auxiliary_weights.tanimoto_ranking > 0.0
            && self.tanimoto_ranking.candidates_per_anchor < 2
        {
            return bail(format!(
                "tanimoto ranking candidates_per_anchor must be at least 2 when the geometry loss is enabled, got {}",
                self.tanimoto_ranking.candidates_per_anchor
            ));
        }
        if self.tanimoto_ranking.min_gap < 0.0 || !self.tanimoto_ranking.min_gap.is_finite() {
            return bail(format!(
                "tanimoto ranking min_gap must be non-negative and finite, got {}",
                self.tanimoto_ranking.min_gap
            ));
        }
        if !self.latent_noise_std.is_finite() || self.latent_noise_std < 0.0 {
            return bail(format!(
                "latent noise std must be non-negative and finite, got {}",
                self.latent_noise_std
            ));
        }
        if self.ecfp_radius == 0 {
            return bail("ecfp radius must be greater than zero".to_string());
        }
        Ok(())
    }

    /// Reads a configuration from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file cannot be opened and
    /// [`Error::Json`] when the contents are not valid JSON for this type.
    #[cfg(feature = "std")]
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|source| Error::io(path, source))?;
        serde_json::from_reader(file).map_err(|source| Error::Json {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Parses a configuration from raw JSON bytes.
    ///
    /// Bytes-based counterpart of [`load_json`](Self::load_json) for callers
    /// that load configuration from a non-filesystem source (HTTP body,
    /// embedded asset, browser fetch).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Json`] when the bytes are not valid JSON for this
    /// configuration shape. The `path` field on the error is left as a
    /// synthetic `<bytes>` marker so downstream display still works.
    #[cfg(feature = "std")]
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|source| Error::Json {
            path: std::path::PathBuf::from("<bytes>"),
            source,
        })
    }

    /// Writes the configuration to a JSON file, pretty-printed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file cannot be created and
    /// [`Error::Json`] when serialization fails.
    #[cfg(feature = "std")]
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|source| Error::io(path, source))?;
        serde_json::to_writer_pretty(file, self).map_err(|source| Error::Json {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Creates an initialized autoencoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MoleculeAutoencoder<B> {
        MoleculeAutoencoder {
            encoder: self.encoder.init(device),
            decoder: self.decoder.init(device),
            descriptor_head: LinearConfig::new(self.encoder.latent_width, self.descriptor_width)
                .init(device),
            violation_head: LinearConfig::new(
                self.encoder.latent_width,
                crate::fingerprints::VIOLATION_CLASS_COUNT,
            )
            .init(device),
            reconstruction_beta: self.reconstruction_loss.beta,
            zero_weight: self.reconstruction_loss.zero_weight,
            nonzero_weight: self.reconstruction_loss.nonzero_weight,
            bce_weight: self.reconstruction_loss.bce_weight,
            bce_zero_weight: self.reconstruction_loss.bce_zero_weight,
            bce_nonzero_weight: self.reconstruction_loss.bce_nonzero_weight,
            reconstruction_loss_kind: self.reconstruction_loss.loss_kind,
            reconstruction_poisson_max_log_count: self.reconstruction_loss.poisson_max_log_count,
            descriptor_weight: self.auxiliary_weights.descriptors,
            tanimoto_ranking_weight: self.auxiliary_weights.tanimoto_ranking,
            violation_weight: self.auxiliary_weights.violation,
            tanimoto_ranking_latent_temperature: self.tanimoto_ranking.latent_temperature,
            tanimoto_ranking_metric_temperature: self.tanimoto_ranking.metric_temperature,
            tanimoto_ranking_min_gap: self.tanimoto_ranking.min_gap,
            tanimoto_ranking_pairs_per_batch: self.tanimoto_ranking.pairs_per_batch,
            tanimoto_ranking_loss_kind: self.tanimoto_ranking.loss_kind,
            tanimoto_ranking_softrank_temperature: self.tanimoto_ranking.softrank_temperature,
            latent_noise_std: self.latent_noise_std,
        }
    }
}

impl Default for MoleculeAutoencoderConfig {
    fn default() -> Self {
        Self::v1_counted_ecfp()
    }
}

/// Fluent builder for [`MoleculeAutoencoderConfig`].
///
/// Defaults match [`MoleculeAutoencoderConfig::v1_counted_ecfp`]; setters only
/// need to be called for fields the caller wants to override.
/// [`build`](Self::build) validates the assembled configuration via
/// [`MoleculeAutoencoderConfig::validate`] before returning it.
#[derive(Debug, Clone)]
pub struct MoleculeAutoencoderConfigBuilder {
    /// Counted ECFP fingerprint width that the encoder reads in.
    fingerprint_size: usize,
    /// Width of the latent embedding produced by the encoder.
    latent_width: usize,
    /// Hidden-layer widths shared by the encoder and (mirrored) the
    /// decoder.
    hidden_widths: Vec<usize>,
    /// Weight applied to the descriptor regression side task.
    descriptor_weight: f64,
    /// Weight applied to the latent Tanimoto geometry side task.
    tanimoto_ranking_weight: f64,
    /// Weight applied to the multi-label violation-classification side
    /// task on top of the latent embedding.
    violation_weight: f64,
    /// Standard deviation of the Gaussian noise injected into the
    /// decoder-side latent at training time.
    latent_noise_std: f64,
    /// Temperature divisor for latent cosine logits in the ranking loss.
    latent_temperature: f64,
    /// Compatibility temperature retained for diagnostics.
    metric_temperature: f64,
    /// Minimum metric gap an anchor must clear to contribute.
    min_gap: f64,
    /// Random candidate partners sampled per anchor.
    candidates_per_anchor: usize,
    /// Maximum anchors used per batch by the geometry loss.
    pairs_per_batch: usize,
    /// ECFP radius matching the cached preprocessing pipeline.
    ecfp_radius: u8,
    /// Auxiliary bit-presence BCE term weight.
    bce_weight: f64,
    /// Per-position BCE weight for inactive bins.
    bce_zero_weight: f64,
    /// Per-position BCE weight for active bins.
    bce_nonzero_weight: f64,
    /// Ranking loss kind (softmax CE baseline vs SoftRank Spearman).
    tanimoto_ranking_loss_kind: TanimotoLossKind,
    /// SoftRank Spearman surrogate temperature.
    tanimoto_ranking_softrank_temperature: f64,
    /// Reconstruction loss kind (Huber baseline vs Poisson NLL).
    reconstruction_loss_kind: ReconstructionLossKind,
    /// Clamp ceiling applied to predicted log-rates inside the Poisson
    /// branch.
    reconstruction_poisson_max_log_count: f64,
}

impl Default for MoleculeAutoencoderConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MoleculeAutoencoderConfigBuilder {
    /// Creates a builder seeded with the v1 4k-ECFP / 512-latent defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fingerprint_size: DEFAULT_ECFP_SIZE,
            latent_width: DEFAULT_LATENT_WIDTH,
            hidden_widths: DEFAULT_HIDDEN_WIDTHS.to_vec(),
            descriptor_weight: DEFAULT_DESCRIPTOR_WEIGHT,
            tanimoto_ranking_weight: DEFAULT_TANIMOTO_RANKING_WEIGHT,
            violation_weight: DEFAULT_VIOLATION_WEIGHT,
            latent_noise_std: DEFAULT_LATENT_NOISE_STD,
            latent_temperature: DEFAULT_TANIMOTO_RANKING_LATENT_TEMPERATURE,
            metric_temperature: DEFAULT_TANIMOTO_RANKING_METRIC_TEMPERATURE,
            min_gap: DEFAULT_TANIMOTO_RANKING_MIN_GAP,
            candidates_per_anchor: DEFAULT_TANIMOTO_RANKING_CANDIDATES_PER_ANCHOR,
            pairs_per_batch: DEFAULT_TANIMOTO_RANKING_PAIRS_PER_BATCH,
            ecfp_radius: DEFAULT_ECFP_RADIUS,
            bce_weight: DEFAULT_BCE_WEIGHT,
            bce_zero_weight: DEFAULT_BCE_ZERO_WEIGHT,
            bce_nonzero_weight: DEFAULT_BCE_NONZERO_WEIGHT,
            tanimoto_ranking_loss_kind: TanimotoLossKind::SoftmaxCe,
            tanimoto_ranking_softrank_temperature: DEFAULT_TANIMOTO_RANKING_SOFTRANK_TEMPERATURE,
            reconstruction_loss_kind: ReconstructionLossKind::Huber,
            reconstruction_poisson_max_log_count: DEFAULT_RECONSTRUCTION_POISSON_MAX_LOG_COUNT,
        }
    }

    /// Sets the counted ECFP input width.
    #[must_use]
    pub const fn fingerprint_size(mut self, value: usize) -> Self {
        self.fingerprint_size = value;
        self
    }

    /// Sets the latent embedding width.
    #[must_use]
    pub const fn latent_width(mut self, value: usize) -> Self {
        self.latent_width = value;
        self
    }

    /// Sets the encoder hidden layer widths (decoder mirrors in reverse).
    #[must_use]
    pub fn hidden_widths(mut self, value: Vec<usize>) -> Self {
        self.hidden_widths = value;
        self
    }

    /// Sets the descriptor regression loss weight.
    #[must_use]
    pub const fn descriptor_weight(mut self, value: f64) -> Self {
        self.descriptor_weight = value;
        self
    }

    /// Sets the latent Tanimoto geometry loss weight.
    #[must_use]
    pub const fn tanimoto_ranking_weight(mut self, value: f64) -> Self {
        self.tanimoto_ranking_weight = value;
        self
    }

    /// Sets the multi-label violation-classification loss weight.
    #[must_use]
    pub const fn violation_weight(mut self, value: f64) -> Self {
        self.violation_weight = value;
        self
    }

    /// Sets the decoder-side latent noise std as a fraction of batch std.
    #[must_use]
    pub const fn latent_noise_std(mut self, value: f64) -> Self {
        self.latent_noise_std = value;
        self
    }

    /// Sets the latent cosine-logit temperature in the geometry loss.
    #[must_use]
    pub const fn tanimoto_ranking_latent_temperature(mut self, value: f64) -> Self {
        self.latent_temperature = value;
        self
    }

    /// Sets the (compatibility) metric temperature.
    #[must_use]
    pub const fn tanimoto_ranking_metric_temperature(mut self, value: f64) -> Self {
        self.metric_temperature = value;
        self
    }

    /// Sets the minimum counted-Tanimoto gap an anchor must clear.
    #[must_use]
    pub const fn tanimoto_ranking_min_gap(mut self, value: f64) -> Self {
        self.min_gap = value;
        self
    }

    /// Sets the number of candidate partners sampled per anchor.
    #[must_use]
    pub const fn tanimoto_ranking_candidates(mut self, value: usize) -> Self {
        self.candidates_per_anchor = value;
        self
    }

    /// Sets the maximum number of anchors per batch (`0` uses all rows).
    #[must_use]
    pub const fn tanimoto_ranking_pairs_per_batch(mut self, value: usize) -> Self {
        self.pairs_per_batch = value;
        self
    }

    /// Sets the Tanimoto ranking loss kind (softmax CE baseline vs
    /// SoftRank Spearman surrogate).
    #[must_use]
    pub const fn tanimoto_ranking_loss_kind(mut self, value: TanimotoLossKind) -> Self {
        self.tanimoto_ranking_loss_kind = value;
        self
    }

    /// Sets the SoftRank Spearman surrogate temperature.
    #[must_use]
    pub const fn tanimoto_ranking_softrank_temperature(mut self, value: f64) -> Self {
        self.tanimoto_ranking_softrank_temperature = value;
        self
    }

    /// Sets the reconstruction loss kind (Huber baseline vs Poisson NLL).
    #[must_use]
    pub const fn reconstruction_loss_kind(mut self, value: ReconstructionLossKind) -> Self {
        self.reconstruction_loss_kind = value;
        self
    }

    /// Sets the Poisson NLL exp clamp ceiling. Ignored when
    /// [`Self::reconstruction_loss_kind`] is
    /// [`ReconstructionLossKind::Huber`].
    #[must_use]
    pub const fn reconstruction_poisson_max_log_count(mut self, value: f64) -> Self {
        self.reconstruction_poisson_max_log_count = value;
        self
    }

    /// Sets the Morgan/ECFP radius the cached fingerprints were built with.
    #[must_use]
    pub const fn ecfp_radius(mut self, value: u8) -> Self {
        self.ecfp_radius = value;
        self
    }

    /// Sets the BCE auxiliary reconstruction loss weight (`0.0` disables it).
    #[must_use]
    pub const fn bce_weight(mut self, value: f64) -> Self {
        self.bce_weight = value;
        self
    }

    /// Sets the per-position weight applied to inactive bins inside the BCE term.
    #[must_use]
    pub const fn bce_zero_weight(mut self, value: f64) -> Self {
        self.bce_zero_weight = value;
        self
    }

    /// Sets the per-position weight applied to active bins inside the BCE term.
    #[must_use]
    pub const fn bce_nonzero_weight(mut self, value: f64) -> Self {
        self.bce_nonzero_weight = value;
        self
    }

    /// Validates the configured fields and builds the immutable config.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] for any failing invariant; the same
    /// checks are reused for the JSON-loaded path via
    /// [`MoleculeAutoencoderConfig::validate`].
    #[cfg(feature = "std")]
    pub fn build(self) -> Result<MoleculeAutoencoderConfig> {
        let fingerprint_size = self.fingerprint_size;
        let mut decoder_hidden_widths = self.hidden_widths.clone();
        decoder_hidden_widths.reverse();
        let config = MoleculeAutoencoderConfig {
            encoder: EncoderConfig {
                input_width: fingerprint_size,
                hidden_widths: self.hidden_widths,
                latent_width: self.latent_width,
            },
            decoder: DecoderConfig {
                latent_width: self.latent_width,
                hidden_widths: decoder_hidden_widths,
                output_width: fingerprint_size,
            },
            descriptor_width: REGRESSION_TARGET_WIDTH,
            reconstruction_loss: ReconstructionLossConfig::builder()
                .bce_weight(self.bce_weight)
                .bce_zero_weight(self.bce_zero_weight)
                .bce_nonzero_weight(self.bce_nonzero_weight)
                .loss_kind(self.reconstruction_loss_kind)
                .poisson_max_log_count(self.reconstruction_poisson_max_log_count)
                .build()?,
            auxiliary_weights: AuxiliaryLossWeights {
                descriptors: self.descriptor_weight,
                tanimoto_ranking: self.tanimoto_ranking_weight,
                violation: self.violation_weight,
            },
            tanimoto_ranking: TanimotoRankingConfig {
                latent_temperature: self.latent_temperature,
                metric_temperature: self.metric_temperature,
                min_gap: self.min_gap,
                candidates_per_anchor: self.candidates_per_anchor,
                pairs_per_batch: self.pairs_per_batch,
                loss_kind: self.tanimoto_ranking_loss_kind,
                softrank_temperature: self.tanimoto_ranking_softrank_temperature,
            },
            latent_noise_std: self.latent_noise_std,
            ecfp_radius: self.ecfp_radius,
        };
        config.validate(fingerprint_size)?;
        Ok(config)
    }
}

/// Encoder module.
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    /// Sparse-aware input projection.
    input: SparseInputLinear<B>,
    /// Stacked hidden ReLU-activated linear layers.
    layers: Vec<Linear<B>>,
    /// Final linear projection into the latent space.
    latent: Linear<B>,
    /// Shared ReLU activation reused across hidden layers.
    activation: Relu,
}

impl<B: Backend> Encoder<B> {
    /// Encodes sparse log-count features.
    pub fn forward(&self, fingerprints: &SparseFingerprintBatch<B>) -> Tensor<B, 2> {
        let mut features = self.activation.forward(self.input.forward(fingerprints));
        for layer in &self.layers {
            features = self.activation.forward(layer.forward(features));
        }
        self.latent.forward(features)
    }

    /// Encodes dense log-count features using the same parameters.
    pub fn forward_dense(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut features = self.activation.forward(self.input.forward_dense(features));
        for layer in &self.layers {
            features = self.activation.forward(layer.forward(features));
        }
        self.latent.forward(features)
    }
}

/// Linear input projection for padded sparse counted fingerprints.
#[derive(Module, Debug)]
pub struct SparseInputLinear<B: Backend> {
    /// Underlying dense linear layer (`xW + b`).
    linear: Linear<B>,
}

impl<B: Backend> SparseInputLinear<B> {
    /// Applies `xW + b` after materializing dense input fingerprints on the backend device.
    pub fn forward(&self, fingerprints: &SparseFingerprintBatch<B>) -> Tensor<B, 2> {
        self.forward_dense(fingerprints.to_dense_log_counts())
    }

    /// Applies the same projection to an already dense tensor.
    pub fn forward_dense(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(features)
    }
}

/// Decoder module.
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    /// Stacked hidden ReLU-activated linear layers.
    layers: Vec<Linear<B>>,
    /// Final linear projection back to the reconstruction width.
    output: Linear<B>,
    /// Shared ReLU activation reused across hidden layers.
    activation: Relu,
}

impl<B: Backend> Decoder<B> {
    /// Decodes latent embeddings into log-count reconstructions.
    pub fn forward(&self, mut latent: Tensor<B, 2>) -> Tensor<B, 2> {
        for layer in &self.layers {
            latent = self.activation.forward(layer.forward(latent));
        }
        self.output.forward(latent)
    }
}

/// Output returned by the molecular autoencoder.
#[derive(Debug)]
pub struct MoleculeAutoencoderOutput<B: Backend> {
    /// Latent molecular embedding.
    pub latent: Tensor<B, 2>,
    /// Reconstructed `log1p(counted ECFP)`.
    pub reconstructed_log_counts: Tensor<B, 2>,
    /// Scalar descriptor predictions.
    pub descriptor_predictions: Tensor<B, 2>,
    /// Multi-label violation-classification logits with shape
    /// `[batch, 8]`. Sigmoid + BCE against the per-sample violation
    /// bitset trains the encoder to surface chemistry-validity signals
    /// on the latent.
    pub violation_logits: Tensor<B, 2>,
}

/// Weighted loss components.
#[derive(Debug)]
pub struct MoleculeLossBreakdown<B: Backend> {
    /// Main counted ECFP reconstruction-head loss (Huber or Poisson per the active `ReconstructionLossKind`).
    pub reconstruction: Tensor<B, 1>,
    /// Per-row counterpart of [`Self::reconstruction`], shape `[batch]`,
    /// surfaced so the train-time metrics pipeline can split the
    /// reported reconstruction loss by sample polarity (positive
    /// "reconstruction" rows vs. negative "correction / denoising"
    /// rows) without affecting the gradient.
    pub reconstruction_per_row: Tensor<B, 1>,
    /// Auxiliary bit-presence BCE-with-logits loss over the same logits as
    /// the reconstruction term. Already multiplied by the configured
    /// `bce_weight`; zero when the BCE term is disabled.
    pub reconstruction_bce: Tensor<B, 1>,
    /// Per-row counterpart of [`Self::reconstruction_bce`], shape
    /// `[batch]`, already multiplied by `bce_weight`. Zero everywhere
    /// when the BCE term is disabled.
    pub reconstruction_bce_per_row: Tensor<B, 1>,
    /// Weighted descriptor regression loss.
    pub descriptors: Tensor<B, 1>,
    /// Weighted latent Tanimoto sampled softmax cross-entropy loss.
    pub tanimoto_ranking: Tensor<B, 1>,
    /// Latent Tanimoto geometry best-candidate accuracy.
    pub tanimoto_ranking_accuracy: Tensor<B, 1>,
    /// Number of valid anchors contributing to the Tanimoto geometry loss.
    pub tanimoto_ranking_pairs: Tensor<B, 1>,
    /// Per-anchor tensors the metric pipeline scores host-side to
    /// compute MRR / Pearson / Spearman after the next Transaction
    /// round-trip. Empty (zero-sized shapes) when the ranking loss is
    /// disabled or the batch is too small to contribute.
    pub tanimoto_diagnostics: TanimotoRankingDiagnostics<B>,
    /// Weighted multi-label BCE-with-logits loss between
    /// [`MoleculeAutoencoderOutput::violation_logits`] and the
    /// per-sample [`finge_rs::ViolationClass`] bitset.
    pub violation_bce: Tensor<B, 1>,
    /// Raw violation-classifier logits + targets staged for host-side
    /// classification-quality scoring (macro-MCC, accuracy) after the
    /// next Transaction round-trip. Empty shapes when the head is
    /// disabled or the batch is too small to contribute.
    pub violation_diagnostics: crate::ranking::ViolationDiagnostics<B>,
}

impl<B: Backend> MoleculeLossBreakdown<B> {
    /// Returns the weighted total loss.
    pub fn total(&self) -> Tensor<B, 1> {
        self.reconstruction.clone()
            + self.reconstruction_bce.clone()
            + self.descriptors.clone()
            + self.tanimoto_ranking.clone()
            + self.violation_bce.clone()
    }
}

/// Dense counted-fingerprint molecular autoencoder.
#[derive(Module, Debug)]
pub struct MoleculeAutoencoder<B: Backend> {
    /// Encoder branch (fingerprint → latent).
    encoder: Encoder<B>,
    /// Decoder branch (latent → reconstructed log-count fingerprint).
    decoder: Decoder<B>,
    /// Side head producing scalar descriptor regression predictions
    /// from the latent.
    descriptor_head: Linear<B>,
    /// Multi-label violation-classification head reading from the
    /// latent. Output shape is `[batch, 8]`; trained with BCE-with-
    /// logits against the per-sample violation bitset.
    violation_head: Linear<B>,
    /// Huber transition point in `log1p(count)` space.
    reconstruction_beta: f64,
    /// Per-position weight for zero-count target bins.
    zero_weight: f64,
    /// Per-position weight for nonzero-count target bins.
    nonzero_weight: f64,
    /// Total weight for the auxiliary BCE-with-logits bit-presence term.
    bce_weight: f64,
    /// Per-position BCE weight for inactive bins.
    bce_zero_weight: f64,
    /// Per-position BCE weight for active bins.
    bce_nonzero_weight: f64,
    /// Side-task weight for the descriptor regression head.
    descriptor_weight: f64,
    /// Side-task weight for the latent Tanimoto geometry head.
    tanimoto_ranking_weight: f64,
    /// Side-task weight for the multi-label violation classifier.
    violation_weight: f64,
    /// Temperature divisor for latent cosine logits inside the ranking
    /// loss.
    tanimoto_ranking_latent_temperature: f64,
    /// Compatibility temperature retained for diagnostics.
    tanimoto_ranking_metric_temperature: f64,
    /// Minimum metric gap required for an anchor to contribute.
    tanimoto_ranking_min_gap: f64,
    /// Maximum anchors used per batch by the geometry loss.
    tanimoto_ranking_pairs_per_batch: usize,
    /// Active Tanimoto ranking loss kind (softmax CE vs SoftRank
    /// Spearman surrogate). Stored on the module so the forward path
    /// can dispatch without re-reading the model config.
    tanimoto_ranking_loss_kind: TanimotoLossKind,
    /// SoftRank Spearman surrogate temperature; ignored under
    /// [`TanimotoLossKind::SoftmaxCe`].
    tanimoto_ranking_softrank_temperature: f64,
    /// Active reconstruction loss kind (Huber baseline vs Poisson NLL).
    /// Stored on the module so the forward path can dispatch without
    /// re-reading the model config.
    reconstruction_loss_kind: ReconstructionLossKind,
    /// Clamp ceiling applied to predicted log-rates before `exp` in the
    /// Poisson branch; ignored under
    /// [`ReconstructionLossKind::Huber`].
    reconstruction_poisson_max_log_count: f64,
    /// Standard deviation of the Gaussian noise injected into the
    /// decoder-side latent at training time.
    latent_noise_std: f64,
}

impl<B: Backend> MoleculeAutoencoder<B> {
    /// Reconstructs the per-bin reconstruction-loss config from the
    /// scalar fields stored on the module (Burn modules can't hold
    /// nested config structs directly).
    fn reconstruction_loss_config(&self) -> ReconstructionLossConfig {
        ReconstructionLossConfig {
            beta: self.reconstruction_beta,
            zero_weight: self.zero_weight,
            nonzero_weight: self.nonzero_weight,
            bce_weight: self.bce_weight,
            bce_zero_weight: self.bce_zero_weight,
            bce_nonzero_weight: self.bce_nonzero_weight,
            loss_kind: self.reconstruction_loss_kind,
            poisson_max_log_count: self.reconstruction_poisson_max_log_count,
        }
    }

    /// Active reconstruction loss kind. Exposed so the train binary's
    /// metric path can adapt the model output (Huber expects
    /// `log1p(count)`, Poisson outputs `log(rate)`).
    #[must_use]
    pub const fn reconstruction_loss_kind(&self) -> ReconstructionLossKind {
        self.reconstruction_loss_kind
    }

    /// Runs the full autoencoder from sparse counted fingerprints.
    pub fn forward(
        &self,
        fingerprints: &SparseFingerprintBatch<B>,
    ) -> MoleculeAutoencoderOutput<B> {
        self.forward_with_decoder_latent_noise(fingerprints, false)
    }

    /// Runs the full autoencoder, optionally perturbing the decoder-side latent.
    pub(crate) fn forward_with_decoder_latent_noise(
        &self,
        fingerprints: &SparseFingerprintBatch<B>,
        use_latent_noise: bool,
    ) -> MoleculeAutoencoderOutput<B> {
        let latent = self.encoder.forward(fingerprints);
        let decoder_latent = if use_latent_noise {
            apply_latent_noise(latent.clone(), self.latent_noise_std)
        } else {
            latent.clone()
        };
        MoleculeAutoencoderOutput {
            reconstructed_log_counts: self.decoder.forward(decoder_latent),
            descriptor_predictions: self.descriptor_head.forward(latent.clone()),
            violation_logits: self.violation_head.forward(latent.clone()),
            latent,
        }
    }

    /// Runs the full autoencoder from a dense `log1p(counted ECFP)` tensor.
    pub fn forward_dense(&self, input_log_counts: Tensor<B, 2>) -> MoleculeAutoencoderOutput<B> {
        let latent = self.encoder.forward_dense(input_log_counts);
        MoleculeAutoencoderOutput {
            reconstructed_log_counts: self.decoder.forward(latent.clone()),
            descriptor_predictions: self.descriptor_head.forward(latent.clone()),
            violation_logits: self.violation_head.forward(latent.clone()),
            latent,
        }
    }

    /// Computes all weighted loss components for a batch.
    pub fn loss(&self, batch: MoleculeAutoencoderBatch<B>) -> MoleculeLossBreakdown<B> {
        let MoleculeAutoencoderBatch {
            fingerprints,
            target_fingerprints,
            descriptor_targets,
            violation_targets,
            tanimoto_ranking,
            ..
        } = batch;
        let output = self.forward(&fingerprints);
        self.loss_from_output(
            output,
            fingerprints,
            target_fingerprints,
            descriptor_targets,
            tanimoto_ranking,
            violation_targets,
        )
    }

    /// Computes all weighted losses from a precomputed model output.
    ///
    /// `target_fingerprints` is the decoder reconstruction target. For
    /// positive rows this batch is row-identical to `fingerprints`; for
    /// negative rows it carries the source positive's fingerprint
    /// (denoising / correction framing), so the decoder is trained to
    /// map invalid inputs back onto valid chemistry. `violation_targets`
    /// is a multi-hot `[batch, 8]` float matrix expanded from each
    /// sample's [`finge_rs::ViolationLabel`] bitset.
    pub fn loss_from_output(
        &self,
        output: MoleculeAutoencoderOutput<B>,
        fingerprints: SparseFingerprintBatch<B>,
        target_fingerprints: SparseFingerprintBatch<B>,
        descriptor_targets: Tensor<B, 2>,
        tanimoto_ranking: crate::ranking::TanimotoRankingBatch<B>,
        violation_targets: Tensor<B, 2>,
    ) -> MoleculeLossBreakdown<B> {
        let device = output.reconstructed_log_counts.device();
        let batch_size = output.reconstructed_log_counts.dims()[0];
        let latent = output.latent;
        // Under Huber the auxiliary BCE term is meaningful; under
        // Poisson the NLL already handles both presence and magnitude,
        // so the BCE branch is force-zeroed regardless of bce_weight.
        let reconstruction_bce_per_row = if self.bce_weight > 0.0
            && self.reconstruction_loss_kind == ReconstructionLossKind::Huber
        {
            weighted_sparse_log_count_bce_loss_per_row(
                output.reconstructed_log_counts.clone(),
                &target_fingerprints,
                self.bce_zero_weight,
                self.bce_nonzero_weight,
            ) * self.bce_weight
        } else {
            Tensor::zeros([batch_size], &device)
        };
        let reconstruction_bce = reconstruction_bce_per_row.clone().mean();
        let reconstruction_per_row = match self.reconstruction_loss_kind {
            ReconstructionLossKind::Huber => weighted_sparse_log_count_huber_loss_per_row(
                output.reconstructed_log_counts,
                target_fingerprints,
                self.reconstruction_loss_config(),
            ),
            ReconstructionLossKind::Poisson => weighted_sparse_log_count_poisson_loss_per_row(
                output.reconstructed_log_counts,
                target_fingerprints,
                self.reconstruction_poisson_max_log_count,
            ),
        };
        let reconstruction = reconstruction_per_row.clone().mean();
        let _ = fingerprints; // retained for symmetry; unused once recon
        // is computed against the decoder target.
        let descriptors = if self.descriptor_weight > 0.0 {
            (output.descriptor_predictions - descriptor_targets)
                .powf_scalar(2.0)
                .mean()
                * self.descriptor_weight
        } else {
            Tensor::zeros([1], &device)
        };
        let tanimoto = weighted_tanimoto_ranking_output(
            latent,
            tanimoto_ranking,
            self.tanimoto_ranking_pairs_per_batch,
            self.tanimoto_ranking_latent_temperature,
            self.tanimoto_ranking_metric_temperature,
            self.tanimoto_ranking_min_gap,
            self.tanimoto_ranking_weight,
            self.tanimoto_ranking_loss_kind,
            self.tanimoto_ranking_softrank_temperature,
        );
        let violation_logits = output.violation_logits;
        let violation_dims = violation_logits.dims();
        let violation_diagnostics = crate::ranking::ViolationDiagnostics {
            logits: violation_logits.clone(),
            targets: violation_targets.clone(),
            batch_size: violation_dims[0],
            class_count: violation_dims[1],
        };
        let violation_bce = if self.violation_weight > 0.0 {
            sigmoid_binary_cross_entropy(violation_logits, violation_targets)
                * self.violation_weight
        } else {
            Tensor::zeros([1], &device)
        };

        MoleculeLossBreakdown {
            reconstruction,
            reconstruction_per_row,
            reconstruction_bce,
            reconstruction_bce_per_row,
            descriptors,
            tanimoto_ranking: tanimoto.loss,
            tanimoto_ranking_accuracy: tanimoto.accuracy,
            tanimoto_ranking_pairs: tanimoto.valid_pairs,
            tanimoto_diagnostics: tanimoto.diagnostics,
            violation_bce,
            violation_diagnostics,
        }
    }
}

/// Numerically stable BCE-with-logits between `logits` and a multi-hot
/// `targets` matrix, both shaped `[batch, classes]`. Reduces to the
/// per-element mean, matching the standard PyTorch reduction.
fn sigmoid_binary_cross_entropy<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    // log(1 + exp(-|x|)) + max(x, 0) - x * y
    // Avoids the overflow path when |x| is large; equivalent to
    // softplus(x) - x * y when applied to positive logits.
    let zero = logits.clone() * 0.0;
    let max_part = logits.clone().max_pair(zero);
    let abs_neg = -logits.clone().abs();
    let softplus = abs_neg.exp().log1p();
    let cross = max_part - logits * targets + softplus;
    cross.mean()
}

/// Applies Gaussian denoising noise to the decoder-side latent input.
///
/// The noise scale is relative to the current batch latent standard deviation,
/// and the scale is detached so the encoder is not rewarded for inflating or
/// shrinking latent variance just to control the injected noise.
#[must_use]
pub fn apply_latent_noise<B: Backend>(latent: Tensor<B, 2>, std_fraction: f64) -> Tensor<B, 2> {
    if std_fraction <= 0.0 {
        return latent;
    }

    let [batch_size, latent_width] = latent.dims();
    let device = latent.device();
    let mean = latent
        .clone()
        .mean_dim(0)
        .expand([batch_size, latent_width]);
    let centered = latent.clone() - mean;
    let scale = (centered.powf_scalar(2.0).mean() + 1.0e-6)
        .sqrt()
        .detach()
        .reshape([1, 1])
        .expand([batch_size, latent_width]);
    let noise = Tensor::<B, 2>::random(
        [batch_size, latent_width],
        Distribution::Normal(0.0, std_fraction),
        &device,
    );

    latent + noise * scale
}

/// Element-wise Huber loss given a precomputed `delta = |ẑ - z|`:
/// quadratic for `delta <= beta`, linear otherwise.
fn huber_from_delta<B: Backend>(delta: Tensor<B, 2>, beta: f64) -> Tensor<B, 2> {
    let quadratic = delta.clone().powf_scalar(2.0) * (0.5 / beta);
    let linear = delta.clone() - (0.5 * beta);
    linear.mask_where(delta.lower_equal_elem(beta), quadratic)
}

/// Weighted Huber loss over `log1p(count)` reconstruction targets.
///
/// The per-bin delta is one-sided at inactive targets (where `z_i = 0`): only
/// positive overshoot `max(ẑ_i, 0)` is penalized so the Huber does not fight
/// the BCE term, which wants those logits firmly negative. Active bins keep
/// the symmetric `|ẑ_i - z_i|`.
pub fn weighted_log_count_huber_loss<B: Backend>(
    predicted_log_counts: Tensor<B, 2>,
    target_log_counts: Tensor<B, 2>,
    config: ReconstructionLossConfig,
) -> Tensor<B, 1> {
    let nonzero = target_log_counts.clone().greater_elem(0.0).float();
    let one_minus_nonzero = nonzero.clone() * -1.0 + 1.0;
    let delta_active = (predicted_log_counts.clone() - target_log_counts).abs();
    let delta_zero = predicted_log_counts.clamp_min(0.0);
    let delta = delta_active * nonzero.clone() + delta_zero * one_minus_nonzero.clone();
    let huber = huber_from_delta(delta, config.beta);
    let weights = nonzero * config.nonzero_weight + one_minus_nonzero * config.zero_weight;
    (huber * weights).mean()
}

/// Numerically stable element-wise softplus used by the BCE
/// reconstruction branch.
fn softplus<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    // Numerically stable softplus: max(x, 0) + log(1 + exp(-|x|)).
    let neg_abs = x.clone().abs() * -1.0;
    x.clamp_min(0.0) + (neg_abs.exp() + 1.0).log()
}

/// Per-position weighted bit-presence loss against the sparse fingerprint
/// target.
///
/// Uses a continuous one-sided form of BCE-with-logits: per-bin loss is
/// `[softplus(s_i · ẑ_i) − ln 2]_+` where `s_i = +1` at inactive bins and
/// `s_i = −1` at active bins. This is the standard BCE softplus shape
/// shifted down so it bottoms out at `ẑ_i = 0` and clamped at zero so the
/// "correctly classified" side never contributes a gradient. Concretely:
///
/// - Inactive bins (`y_i = 0`): only positive leaks `ẑ_i > 0` pay a penalty,
///   so the BCE does not fight the Huber once `ẑ_i ≤ 0`.
/// - Active bins (`y_i = 1`): only negative dips `ẑ_i < 0` pay a penalty,
///   so the BCE stops pulling once the sign is right and the Huber alone
///   refines the magnitude toward `log1p(target_count)`.
///
/// Computation mirrors [`weighted_sparse_log_count_huber_loss`] — assume the
/// target bit is zero everywhere, then correct at the sparse active indices.
///
/// # Panics
///
/// Panics when the predicted dense reconstruction shape does not match the
/// sparse target batch size and fingerprint width.
#[must_use]
pub fn weighted_sparse_log_count_bce_loss<B: Backend>(
    predicted_log_counts: Tensor<B, 2>,
    target: &SparseFingerprintBatch<B>,
    zero_weight: f64,
    nonzero_weight: f64,
) -> Tensor<B, 1> {
    weighted_sparse_log_count_bce_loss_per_row(
        predicted_log_counts,
        target,
        zero_weight,
        nonzero_weight,
    )
    .mean()
}

/// Per-row variant of [`weighted_sparse_log_count_bce_loss`].
///
/// Returns a `Tensor<B, 1>` of shape `[batch]` where each element is the
/// per-cell mean BCE for that row, so the scalar version is `.mean()` of
/// this and positive (reconstruction) rows can be compared against
/// negative (correction / denoising) rows without a polarity mix-up.
///
/// # Panics
///
/// Panics when the predicted dense reconstruction shape does not match
/// the sparse target batch size and fingerprint width.
#[must_use]
pub fn weighted_sparse_log_count_bce_loss_per_row<B: Backend>(
    predicted_log_counts: Tensor<B, 2>,
    target: &SparseFingerprintBatch<B>,
    zero_weight: f64,
    nonzero_weight: f64,
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

    let log2 = std::f64::consts::LN_2;
    let width = predicted_dims[1] as f64;
    // y_i = 0: penalty grows when ẑ_i > 0 (positive leak).
    let zero_bce = (softplus(predicted_log_counts.clone()) - log2).clamp_min(0.0);
    let zero_per_row: Tensor<B, 1> = zero_bce.sum_dim(1).squeeze_dim(1);
    let predicted_nonzero = predicted_log_counts.gather(1, target.indices.clone());
    // y_i = 1: penalty grows when ẑ_i < 0 (negative dip).
    let active_bce = (softplus(predicted_nonzero.clone() * -1.0) - log2).clamp_min(0.0);
    // Subtract the "assumed y_i = 0" contribution at the same active indices,
    // matching the sparse correction pattern used by the Huber loss above.
    let zero_at_active_bce = (softplus(predicted_nonzero) - log2).clamp_min(0.0);
    let correction =
        (active_bce * nonzero_weight - zero_at_active_bce * zero_weight) * target.mask.clone();
    let correction_per_row: Tensor<B, 1> = correction.sum_dim(1).squeeze_dim(1);

    (zero_per_row * zero_weight + correction_per_row) / width
}

/// Weighted Huber loss over sparse `log1p(count)` reconstruction targets.
///
/// At inactive bins (`z_i = 0`) the Huber delta is one-sided: only the
/// positive part `max(ẑ_i, 0)` contributes, so the Huber is silent when the
/// BCE drives `ẑ_i` negative. At active bins the symmetric `|ẑ_i - z_i|` is
/// used as before.
///
/// # Panics
///
/// Panics when the predicted dense reconstruction shape does not match the
/// sparse target batch size and fingerprint width.
pub fn weighted_sparse_log_count_huber_loss<B: Backend>(
    predicted_log_counts: Tensor<B, 2>,
    target: SparseFingerprintBatch<B>,
    config: ReconstructionLossConfig,
) -> Tensor<B, 1> {
    weighted_sparse_log_count_huber_loss_per_row(predicted_log_counts, target, config).mean()
}

/// Per-row variant of [`weighted_sparse_log_count_huber_loss`].
///
/// Returns a `Tensor<B, 1>` of shape `[batch]` where each element is the
/// per-cell mean weighted Huber for that row, so the scalar version is
/// `.mean()` of this and positive (reconstruction) rows can be compared
/// against negative (correction / denoising) rows without a polarity
/// mix-up.
///
/// # Panics
///
/// Panics when the predicted dense reconstruction shape does not match
/// the sparse target batch size and fingerprint width.
pub fn weighted_sparse_log_count_huber_loss_per_row<B: Backend>(
    predicted_log_counts: Tensor<B, 2>,
    target: SparseFingerprintBatch<B>,
    config: ReconstructionLossConfig,
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

    let width = predicted_dims[1] as f64;
    let zero_huber = huber_from_delta(predicted_log_counts.clone().clamp_min(0.0), config.beta);
    let zero_per_row: Tensor<B, 1> = zero_huber.sum_dim(1).squeeze_dim(1);
    let predicted_nonzero = predicted_log_counts.gather(1, target.indices.clone());
    let nonzero_huber = huber_from_delta(
        (predicted_nonzero.clone() - target.log_counts).abs(),
        config.beta,
    );
    let zero_nonzero_huber = huber_from_delta(predicted_nonzero.clamp_min(0.0), config.beta);
    let correction = (nonzero_huber * config.nonzero_weight
        - zero_nonzero_huber * config.zero_weight)
        * target.mask;
    let correction_per_row: Tensor<B, 1> = correction.sum_dim(1).squeeze_dim(1);

    (zero_per_row * config.zero_weight + correction_per_row) / width
}

/// Poisson NLL on raw counts, with the model output interpreted as
/// `log(rate)`.
///
/// Per-cell loss is `exp(z) - count · z` (the count-dependent `log(count!)`
/// constant is dropped since it has no gradient). Sparse formulation:
/// the `count · z` term is non-zero only at active bins, so we sum
/// `exp(z)` densely and subtract a sparse correction.
///
/// Predicted log-rates are clamped to `clamp_max(max_log_count)` before
/// `exp` to keep the `f32` reduction safe even when the optimizer briefly
/// drives a logit very high during early training.
///
/// # Panics
///
/// Panics when the predicted dense reconstruction shape does not match
/// the sparse target batch size and fingerprint width.
#[must_use]
pub fn weighted_sparse_log_count_poisson_loss<B: Backend>(
    predicted_log_counts: Tensor<B, 2>,
    target: SparseFingerprintBatch<B>,
    max_log_count: f64,
) -> Tensor<B, 1> {
    weighted_sparse_log_count_poisson_loss_per_row(predicted_log_counts, target, max_log_count)
        .mean()
}

/// Per-row variant of [`weighted_sparse_log_count_poisson_loss`].
///
/// Returns a `Tensor<B, 1>` of shape `[batch]` where each element is the
/// per-cell mean Poisson NLL for that row, matching the shape contract
/// of [`weighted_sparse_log_count_huber_loss_per_row`] so the train-time
/// polarity split can sum it against the same gap weights.
///
/// # Panics
///
/// Panics when the predicted dense reconstruction shape does not match
/// the sparse target batch size and fingerprint width.
pub fn weighted_sparse_log_count_poisson_loss_per_row<B: Backend>(
    predicted_log_counts: Tensor<B, 2>,
    target: SparseFingerprintBatch<B>,
    max_log_count: f64,
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

    let width = predicted_dims[1] as f64;
    let clamped = predicted_log_counts.clamp_max(max_log_count);
    // exp(z) summed over every cell; this is the "all cells are inactive
    // (count = 0)" baseline contribution to the NLL.
    let exp_z_per_row: Tensor<B, 1> = clamped.clone().exp().sum_dim(1).squeeze_dim(1);
    // At active bins the NLL gains a -count * z term. Pull the model's
    // logits at the sparse active indices, weight by the (mask-zeroed)
    // raw counts, and sum per row.
    let z_at_active = clamped.gather(1, target.indices.clone());
    let count_z = target.counts * z_at_active * target.mask;
    let count_z_per_row: Tensor<B, 1> = count_z.sum_dim(1).squeeze_dim(1);

    (exp_z_per_row - count_z_per_row) / width
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::data::dataloader::batcher::Batcher;

    use super::*;
    use crate::batch::{MoleculeAutoencoderBatcher, MoleculeAutoencoderSample};

    #[test]
    fn v1_config_uses_4k_ecfp_and_512_latent() {
        let config = MoleculeAutoencoderConfig::v1_counted_ecfp();

        assert_eq!(config.encoder.input_width, 4096);
        assert_eq!(config.encoder.hidden_widths, vec![4096, 2048, 1024]);
        assert_eq!(config.encoder.latent_width, 512);
        assert_eq!(config.decoder.hidden_widths, vec![1024, 2048, 4096]);
        assert_eq!(config.decoder.output_width, 4096);
        assert_eq!(config.descriptor_width, REGRESSION_TARGET_WIDTH);
        assert_eq!(config.auxiliary_weights.descriptors, 0.05);
        assert_eq!(config.auxiliary_weights.tanimoto_ranking, 0.10);
        assert_eq!(
            config.tanimoto_ranking.latent_temperature,
            DEFAULT_TANIMOTO_RANKING_LATENT_TEMPERATURE
        );
        assert_eq!(config.tanimoto_ranking.metric_temperature, 0.10);
        assert_eq!(config.tanimoto_ranking.min_gap, 0.05);
        assert_eq!(
            config.tanimoto_ranking.candidates_per_anchor,
            DEFAULT_TANIMOTO_RANKING_CANDIDATES_PER_ANCHOR
        );
        assert_eq!(config.tanimoto_ranking.pairs_per_batch, 0);
        assert_eq!(config.latent_noise_std, 0.02);
        assert_eq!(config.ecfp_radius, DEFAULT_ECFP_RADIUS);
        assert_eq!(config.reconstruction_loss.bce_weight, DEFAULT_BCE_WEIGHT);
        assert_eq!(
            config.reconstruction_loss.bce_zero_weight,
            DEFAULT_BCE_ZERO_WEIGHT
        );
        assert_eq!(
            config.reconstruction_loss.bce_nonzero_weight,
            DEFAULT_BCE_NONZERO_WEIGHT
        );
    }

    #[test]
    fn tanimoto_ranking_config_accepts_old_margin_key() {
        let config: TanimotoRankingConfig = serde_json::from_str(
            r#"{
                "margin": 0.25,
                "min_gap": 0.05,
                "candidates_per_anchor": 4,
                "pairs_per_batch": 0
            }"#,
        )
        .expect("old margin key should deserialize as latent temperature");

        assert_eq!(config.latent_temperature, 0.25);
        assert_eq!(config.metric_temperature, 0.10);
    }

    #[test]
    fn model_forward_and_loss_have_stable_shapes() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = MoleculeAutoencoderConfig::symmetric(32, 8, vec![16]);
        let model = config.init::<B>(&device);
        let batcher = MoleculeAutoencoderBatcher::new(32, REGRESSION_TARGET_WIDTH);
        let batch = batcher.batch(
            vec![MoleculeAutoencoderSample {
                cid: 1,
                fingerprint_indices: vec![1, 3, 5],
                fingerprint_counts: vec![2, 1, 1],
                descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                ..Default::default()
            }],
            &device,
        );

        let output = model.forward(&batch.fingerprints);
        assert_eq!(output.latent.dims(), [1, 8]);
        assert_eq!(output.reconstructed_log_counts.dims(), [1, 32]);
        assert_eq!(
            output.descriptor_predictions.dims(),
            [1, REGRESSION_TARGET_WIDTH]
        );
        assert_eq!(
            output.violation_logits.dims(),
            [1, crate::fingerprints::VIOLATION_CLASS_COUNT]
        );

        let losses = model.loss(batch);
        assert!(losses.total().into_scalar().is_finite());
        assert_eq!(losses.tanimoto_ranking.dims(), [1]);
        assert_eq!(losses.tanimoto_ranking_accuracy.dims(), [1]);
        assert_eq!(losses.tanimoto_ranking_pairs.dims(), [1]);
        assert_eq!(losses.violation_bce.dims(), [1]);
        assert!(losses.violation_bce.into_scalar().is_finite());
    }

    #[test]
    fn reconstruction_loss_uses_target_fingerprint_for_negative_rows() {
        // A negative row's decoder target is the source positive's
        // fingerprint, not its mutated input. Build a mixed batch where
        // the (target == positive) and (input != positive); the
        // reconstruction loss should be smaller when the decoder
        // predicts the positive than when it predicts the negative.
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(8, REGRESSION_TARGET_WIDTH);
        let batch: MoleculeAutoencoderBatch<B> = batcher.batch(
            vec![MoleculeAutoencoderSample {
                cid: 1,
                fingerprint_indices: vec![5, 7],
                fingerprint_counts: vec![1, 1],
                descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                violation_label: 0b0000_0001,
                target_fingerprint_indices: vec![1, 3],
                target_fingerprint_counts: vec![2, 1],
            }],
            &device,
        );

        // Sanity: target storage holds the source positive bins.
        let target_indices = batch
            .target_fingerprints
            .indices
            .to_data()
            .as_slice::<i64>()
            .expect("i64 target indices")
            .to_vec();
        assert_eq!(target_indices, vec![1, 3]);
        let input_indices = batch
            .fingerprints
            .indices
            .to_data()
            .as_slice::<i64>()
            .expect("i64 input indices")
            .to_vec();
        assert_eq!(input_indices, vec![5, 7]);

        let model = MoleculeAutoencoderConfig::symmetric(8, 4, vec![16]).init::<B>(&device);
        let losses = model.loss(batch);
        assert!(losses.reconstruction.into_scalar().is_finite());
        assert!(losses.violation_bce.into_scalar().is_finite());
    }

    #[test]
    fn sparse_reconstruction_loss_matches_dense_loss() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(4, REGRESSION_TARGET_WIDTH);
        let batch = batcher.batch(
            vec![
                MoleculeAutoencoderSample {
                    cid: 1,
                    fingerprint_indices: vec![0, 2],
                    fingerprint_counts: vec![2, 1],
                    descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                    ..Default::default()
                },
                MoleculeAutoencoderSample {
                    cid: 2,
                    fingerprint_indices: vec![1],
                    fingerprint_counts: vec![3],
                    descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                    ..Default::default()
                },
            ],
            &device,
        );
        let predicted = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.2, -0.1, 0.8, 0.0, 0.0, 1.1, -0.2, 0.3], [2, 4]),
            &device,
        );
        let dense_target = Tensor::<B, 2>::from_data(
            TensorData::new(
                vec![
                    2.0_f32.ln_1p(),
                    0.0,
                    1.0_f32.ln_1p(),
                    0.0,
                    0.0,
                    3.0_f32.ln_1p(),
                    0.0,
                    0.0,
                ],
                [2, 4],
            ),
            &device,
        );
        let config = ReconstructionLossConfig::builder()
            .beta(0.7)
            .zero_weight(0.05)
            .nonzero_weight(1.0)
            .build()
            .expect("reconstruction config");

        let dense = weighted_log_count_huber_loss(predicted.clone(), dense_target, config);
        let sparse = weighted_sparse_log_count_huber_loss(predicted, batch.fingerprints, config);

        let delta = (dense.into_scalar() - sparse.into_scalar()).abs();
        assert!(delta < 1.0e-6, "dense and sparse loss differ by {delta}");
    }

    #[test]
    fn sparse_huber_ignores_negative_predictions_at_zero_targets() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(4, REGRESSION_TARGET_WIDTH);
        let batch = batcher.batch(
            vec![MoleculeAutoencoderSample {
                cid: 1,
                fingerprint_indices: vec![1],
                fingerprint_counts: vec![2],
                descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                ..Default::default()
            }],
            &device,
        );
        let config = ReconstructionLossConfig::builder()
            .beta(1.0)
            .zero_weight(1.0)
            .nonzero_weight(0.0)
            .build()
            .expect("config");
        // Active bin at index 1: predict the target exactly so it contributes nothing.
        // Zero bins at indices 0, 2, 3: predict strongly negative — should be silent.
        let target_log = 2.0_f32.ln_1p();
        let predicted = Tensor::<B, 2>::from_data(
            TensorData::new(vec![-3.0_f32, target_log, -1.5, -0.7], [1, 4]),
            &device,
        );
        let loss = weighted_sparse_log_count_huber_loss(predicted, batch.fingerprints, config)
            .into_scalar();
        assert!(
            loss.abs() < 1.0e-6,
            "expected zero loss with one-sided Huber at zero targets, got {loss}"
        );
    }

    #[test]
    fn sparse_huber_still_penalizes_positive_leak_at_zero_targets() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(4, REGRESSION_TARGET_WIDTH);
        let batch = batcher.batch(
            vec![MoleculeAutoencoderSample {
                cid: 1,
                fingerprint_indices: vec![1],
                fingerprint_counts: vec![2],
                descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                ..Default::default()
            }],
            &device,
        );
        let config = ReconstructionLossConfig::builder()
            .beta(1.0)
            .zero_weight(1.0)
            .nonzero_weight(0.0)
            .build()
            .expect("config");
        // Active bin matches target; the three zero bins leak +0.4 each.
        let target_log = 2.0_f32.ln_1p();
        let predicted = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.4_f32, target_log, 0.4, 0.4], [1, 4]),
            &device,
        );
        let loss = weighted_sparse_log_count_huber_loss(predicted, batch.fingerprints, config)
            .into_scalar();
        // Per-bin Huber with delta=0.4, beta=1.0: 0.5 * 0.4^2 / 1.0 = 0.08.
        // Three zero leaks averaged across N=4: 3 * 0.08 / 4 = 0.06.
        let expected = 3.0 * 0.08 / 4.0;
        assert!(
            (loss - expected).abs() < 1.0e-5,
            "expected {expected}, got {loss}"
        );
    }

    #[test]
    fn sparse_bce_matches_cpu_reference_with_weighted_zeros() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(4, REGRESSION_TARGET_WIDTH);
        let batch = batcher.batch(
            vec![MoleculeAutoencoderSample {
                cid: 1,
                fingerprint_indices: vec![0, 2],
                fingerprint_counts: vec![2, 1],
                descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                ..Default::default()
            }],
            &device,
        );
        // Logits at positions 0 (active), 1 (inactive leak), 2 (active), 3 (clean inactive).
        let predicted = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0_f32, 0.5, -0.2, -1.5], [1, 4]),
            &device,
        );
        let zero_weight = 4.0;
        let nonzero_weight = 1.0;
        let actual = weighted_sparse_log_count_bce_loss(
            predicted.clone(),
            &batch.fingerprints,
            zero_weight,
            nonzero_weight,
        )
        .into_scalar();

        // CPU reference: one-sided shifted BCE.
        //   y_i = 0 (s = +1): penalty = max(0, softplus( logit) - ln 2)
        //   y_i = 1 (s = -1): penalty = max(0, softplus(-logit) - ln 2)
        let logits = [1.0_f32, 0.5, -0.2, -1.5];
        let targets = [1.0_f32, 0.0, 1.0, 0.0];
        let log2 = std::f32::consts::LN_2;
        let per_position: Vec<f32> = logits
            .iter()
            .zip(targets)
            .map(|(&logit, target)| {
                let sign = if target > 0.0 { -1.0_f32 } else { 1.0 };
                let signed = sign * logit;
                let softplus = signed.max(0.0) + (-signed.abs()).exp().ln_1p();
                let shifted = (softplus - log2).max(0.0);
                let weight = if target > 0.0 {
                    nonzero_weight as f32
                } else {
                    zero_weight as f32
                };
                shifted * weight
            })
            .collect();
        let expected = per_position.iter().sum::<f32>() / per_position.len() as f32;
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "sparse BCE {actual} differed from reference {expected}"
        );
    }

    #[test]
    fn sparse_bce_is_silent_on_correctly_classified_bins() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(4, REGRESSION_TARGET_WIDTH);
        let batch = batcher.batch(
            vec![MoleculeAutoencoderSample {
                cid: 1,
                fingerprint_indices: vec![1, 3],
                fingerprint_counts: vec![3, 1],
                descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                ..Default::default()
            }],
            &device,
        );
        // Active bins predict positive; inactive bins predict negative.
        // All bins are on the "correct" side of zero so BCE should be exactly 0.
        let predicted = Tensor::<B, 2>::from_data(
            TensorData::new(vec![-0.1_f32, 0.1, -0.1, 0.1], [1, 4]),
            &device,
        );
        let value = weighted_sparse_log_count_bce_loss(predicted, &batch.fingerprints, 4.0, 1.0)
            .into_scalar();
        assert!(
            value.abs() < 1.0e-6,
            "expected exactly zero BCE for correctly classified bins, got {value}"
        );
    }

    #[test]
    fn sparse_bce_only_penalizes_wrong_direction() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(2, REGRESSION_TARGET_WIDTH);
        let batch = batcher.batch(
            vec![MoleculeAutoencoderSample {
                cid: 1,
                fingerprint_indices: vec![0],
                fingerprint_counts: vec![1],
                descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                ..Default::default()
            }],
            &device,
        );
        // Active bin at index 0 wrongly predicts negative; inactive bin at
        // index 1 stays well below zero. Only the active-bin violation should
        // contribute.
        let predicted =
            Tensor::<B, 2>::from_data(TensorData::new(vec![-2.0_f32, -5.0], [1, 2]), &device);
        let value = weighted_sparse_log_count_bce_loss(predicted, &batch.fingerprints, 4.0, 1.0)
            .into_scalar();
        // Expected: per-bin loss at position 0 only.
        // softplus(2) ≈ 2.1269, minus ln 2 ≈ 0.6931 → 1.4338, weighted by 1.0,
        // averaged over N=2 → ≈ 0.7169.
        let log2 = std::f32::consts::LN_2;
        let softplus = 2.0_f32.max(0.0) + (-2.0_f32.abs()).exp().ln_1p();
        let expected = ((softplus - log2).max(0.0) * 1.0) / 2.0;
        assert!(
            (value - expected).abs() < 1.0e-5,
            "expected {expected}, got {value}"
        );
    }

    #[test]
    fn sparse_bce_zero_when_logits_match_target_bits() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(4, REGRESSION_TARGET_WIDTH);
        let batch = batcher.batch(
            vec![MoleculeAutoencoderSample {
                cid: 1,
                fingerprint_indices: vec![1, 3],
                fingerprint_counts: vec![3, 1],
                descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                ..Default::default()
            }],
            &device,
        );
        // Push inactive positions strongly negative and active positions strongly positive.
        let predicted = Tensor::<B, 2>::from_data(
            TensorData::new(vec![-50.0_f32, 50.0, -50.0, 50.0], [1, 4]),
            &device,
        );
        let value = weighted_sparse_log_count_bce_loss(predicted, &batch.fingerprints, 4.0, 1.0)
            .into_scalar();
        assert!(value.abs() < 1.0e-6, "expected near-zero BCE, got {value}");
    }

    #[test]
    fn latent_noise_is_noop_when_disabled() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

        let output = apply_latent_noise(latent, 0.0);
        let values = output
            .to_data()
            .as_slice::<f32>()
            .expect("latent values")
            .to_vec();

        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn latent_noise_preserves_shape_and_finiteness() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let latent = Tensor::<B, 2>::from_floats(
            [[1.0, 0.0, 2.0], [3.0, 4.0, 5.0], [6.0, 8.0, 10.0]],
            &device,
        );

        let output = apply_latent_noise(latent, 0.02);
        assert_eq!(output.dims(), [3, 3]);
        let values = output
            .to_data()
            .as_slice::<f32>()
            .expect("latent values")
            .to_vec();

        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn noisy_decoder_forward_keeps_side_heads_on_clean_latent() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mut config = MoleculeAutoencoderConfig::symmetric(8, 4, vec![6]);
        config.latent_noise_std = 10.0;
        let model = config.init::<B>(&device);
        let batcher = MoleculeAutoencoderBatcher::new(8, REGRESSION_TARGET_WIDTH);
        let batch = batcher.batch(
            vec![
                MoleculeAutoencoderSample {
                    cid: 1,
                    fingerprint_indices: vec![0, 2, 4],
                    fingerprint_counts: vec![2, 1, 3],
                    descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                    ..Default::default()
                },
                MoleculeAutoencoderSample {
                    cid: 2,
                    fingerprint_indices: vec![1, 3, 5],
                    fingerprint_counts: vec![1, 2, 1],
                    descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                    ..Default::default()
                },
            ],
            &device,
        );

        let clean = model.forward(&batch.fingerprints);
        let noisy = model.forward_with_decoder_latent_noise(&batch.fingerprints, true);
        let clean_descriptors = clean
            .descriptor_predictions
            .to_data()
            .as_slice::<f32>()
            .expect("descriptor values")
            .to_vec();
        let noisy_descriptors = noisy
            .descriptor_predictions
            .to_data()
            .as_slice::<f32>()
            .expect("descriptor values")
            .to_vec();
        let clean_reconstruction = clean
            .reconstructed_log_counts
            .to_data()
            .as_slice::<f32>()
            .expect("reconstruction values")
            .to_vec();
        let noisy_reconstruction = noisy
            .reconstructed_log_counts
            .to_data()
            .as_slice::<f32>()
            .expect("reconstruction values")
            .to_vec();

        assert_eq!(clean_descriptors, noisy_descriptors);
        assert!(
            clean_reconstruction
                .iter()
                .zip(noisy_reconstruction)
                .any(|(clean, noisy)| (clean - noisy).abs() > 1.0e-6)
        );
    }

    #[test]
    fn builder_matches_symmetric_with_overrides() {
        let config = MoleculeAutoencoderConfig::builder()
            .fingerprint_size(64)
            .latent_width(16)
            .hidden_widths(vec![32, 24])
            .descriptor_weight(0.07)
            .tanimoto_ranking_weight(0.13)
            .latent_noise_std(0.03)
            .tanimoto_ranking_latent_temperature(0.20)
            .tanimoto_ranking_metric_temperature(0.30)
            .tanimoto_ranking_min_gap(0.04)
            .tanimoto_ranking_candidates(6)
            .tanimoto_ranking_pairs_per_batch(8)
            .ecfp_radius(3)
            .bce_weight(0.5)
            .bce_zero_weight(3.0)
            .bce_nonzero_weight(2.0)
            .build()
            .expect("valid config");

        assert_eq!(config.encoder.input_width, 64);
        assert_eq!(config.encoder.hidden_widths, vec![32, 24]);
        assert_eq!(config.encoder.latent_width, 16);
        assert_eq!(config.decoder.hidden_widths, vec![24, 32]);
        assert_eq!(config.decoder.output_width, 64);
        assert_eq!(config.auxiliary_weights.descriptors, 0.07);
        assert_eq!(config.auxiliary_weights.tanimoto_ranking, 0.13);
        assert_eq!(config.latent_noise_std, 0.03);
        assert_eq!(config.tanimoto_ranking.latent_temperature, 0.20);
        assert_eq!(config.tanimoto_ranking.metric_temperature, 0.30);
        assert_eq!(config.tanimoto_ranking.min_gap, 0.04);
        assert_eq!(config.tanimoto_ranking.candidates_per_anchor, 6);
        assert_eq!(config.tanimoto_ranking.pairs_per_batch, 8);
        assert_eq!(config.ecfp_radius, 3);
        assert_eq!(config.reconstruction_loss.bce_weight, 0.5);
        assert_eq!(config.reconstruction_loss.bce_zero_weight, 3.0);
        assert_eq!(config.reconstruction_loss.bce_nonzero_weight, 2.0);
    }

    #[test]
    fn builder_defaults_match_v1_counted_ecfp() {
        let from_builder = MoleculeAutoencoderConfig::builder()
            .build()
            .expect("default builder is valid");

        assert_eq!(from_builder, MoleculeAutoencoderConfig::v1_counted_ecfp());
    }

    #[test]
    fn reconstruction_loss_builder_rejects_zero_beta_and_negative_weights() {
        assert!(matches!(
            ReconstructionLossConfig::builder()
                .beta(0.0)
                .build()
                .expect_err("zero beta"),
            crate::Error::ConfigInvalid { message } if message.contains("beta")
        ));
        assert!(matches!(
            ReconstructionLossConfig::builder()
                .zero_weight(-0.1)
                .build()
                .expect_err("negative zero weight"),
            crate::Error::ConfigInvalid { message } if message.contains("zero_weight")
        ));
    }

    #[test]
    fn auxiliary_loss_weights_builder_rejects_negative_weights() {
        assert!(matches!(
            AuxiliaryLossWeights::builder()
                .descriptors(-0.5)
                .build()
                .expect_err("negative descriptor weight"),
            crate::Error::ConfigInvalid { message } if message.contains("descriptors")
        ));
    }

    #[test]
    fn encoder_builder_requires_all_fields_and_rejects_empty_hidden_widths() {
        let missing = EncoderConfig::builder()
            .input_width(32)
            .latent_width(8)
            .build()
            .expect_err("missing hidden_widths");
        assert!(matches!(
            missing,
            crate::Error::ConfigInvalid { message } if message.contains("hidden_widths must be set")
        ));

        let empty = EncoderConfig::builder()
            .input_width(32)
            .latent_width(8)
            .hidden_widths(vec![])
            .build()
            .expect_err("empty hidden_widths");
        assert!(matches!(
            empty,
            crate::Error::ConfigInvalid { message } if message.contains("must not be empty")
        ));

        let zero_layer = EncoderConfig::builder()
            .input_width(32)
            .latent_width(8)
            .hidden_widths(vec![16, 0, 4])
            .build()
            .expect_err("zero hidden width");
        assert!(matches!(
            zero_layer,
            crate::Error::ConfigInvalid { message } if message.contains("greater than zero")
        ));
    }

    #[test]
    fn builder_rejects_invalid_temperature_or_weight() {
        let bad_temperature = MoleculeAutoencoderConfig::builder()
            .tanimoto_ranking_latent_temperature(0.0)
            .build()
            .expect_err("zero latent temperature must be rejected");
        assert!(matches!(
            bad_temperature,
            crate::Error::ConfigInvalid { message } if message.contains("latent temperature")
        ));

        let bad_candidates = MoleculeAutoencoderConfig::builder()
            .tanimoto_ranking_weight(0.10)
            .tanimoto_ranking_candidates(1)
            .build()
            .expect_err("single candidate with positive weight must be rejected");
        assert!(matches!(
            bad_candidates,
            crate::Error::ConfigInvalid { message } if message.contains("candidates_per_anchor")
        ));
    }

    #[test]
    fn tanimoto_ranking_runtime_view_packs_weight_and_shape() {
        let config = MoleculeAutoencoderConfig::builder()
            .fingerprint_size(32)
            .latent_width(8)
            .hidden_widths(vec![16])
            .descriptor_weight(0.05)
            .tanimoto_ranking_weight(0.11)
            .latent_noise_std(0.02)
            .tanimoto_ranking_latent_temperature(0.15)
            .tanimoto_ranking_metric_temperature(0.25)
            .tanimoto_ranking_min_gap(0.05)
            .tanimoto_ranking_candidates(4)
            .tanimoto_ranking_pairs_per_batch(0)
            .build()
            .expect("valid config");
        let runtime = config.tanimoto_ranking_runtime();

        assert_eq!(runtime.weight, 0.11);
        assert_eq!(runtime.latent_temperature, 0.15);
        assert_eq!(runtime.metric_temperature, 0.25);
        assert_eq!(runtime.min_gap, 0.05);
        assert_eq!(runtime.candidates_per_anchor, 4);
        assert_eq!(runtime.pairs_per_batch, 0);
    }

    #[test]
    fn validate_accepts_default_configuration() {
        let config = MoleculeAutoencoderConfig::v1_counted_ecfp();

        config
            .validate(config.encoder.input_width)
            .expect("default configuration should validate");
    }

    #[test]
    fn validate_rejects_input_width_mismatch() {
        let config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        let error = config
            .validate(config.encoder.input_width + 1)
            .expect_err("mismatched width should be rejected");

        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("encoder input width")
        ));
    }

    #[test]
    fn validate_rejects_negative_weights_and_temperatures() {
        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.auxiliary_weights.descriptors = -0.1;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("negative weight should be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("auxiliary")
        ));

        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.tanimoto_ranking.latent_temperature = 0.0;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("zero temperature should be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("latent temperature")
        ));

        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.latent_noise_std = -0.5;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("negative noise std should be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("latent noise")
        ));
    }

    #[test]
    fn load_and_save_json_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model-config.json");
        let config = MoleculeAutoencoderConfig::builder()
            .fingerprint_size(128)
            .latent_width(32)
            .hidden_widths(vec![64, 48])
            .descriptor_weight(0.06)
            .tanimoto_ranking_weight(0.12)
            .latent_noise_std(0.01)
            .tanimoto_ranking_latent_temperature(0.18)
            .tanimoto_ranking_metric_temperature(0.28)
            .tanimoto_ranking_min_gap(0.03)
            .tanimoto_ranking_candidates(5)
            .tanimoto_ranking_pairs_per_batch(7)
            .build()
            .expect("valid config");

        config.save_json(&path).expect("save");
        let loaded = MoleculeAutoencoderConfig::load_json(&path).expect("load");

        assert_eq!(loaded, config);
    }

    #[test]
    fn sparse_encoder_matches_dense_encoder() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = EncoderConfig {
            input_width: 4,
            hidden_widths: vec![3],
            latent_width: 2,
        };
        let encoder = config.init::<B>(&device);
        let batcher = MoleculeAutoencoderBatcher::new(4, REGRESSION_TARGET_WIDTH);
        let batch = batcher.batch(
            vec![MoleculeAutoencoderSample {
                cid: 1,
                fingerprint_indices: vec![0, 2],
                fingerprint_counts: vec![2, 1],
                descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
                ..Default::default()
            }],
            &device,
        );
        let dense = Tensor::<B, 2>::from_data(
            TensorData::new(vec![2.0_f32.ln_1p(), 0.0, 1.0_f32.ln_1p(), 0.0], [1, 4]),
            &device,
        );

        let sparse_latent = encoder.forward(&batch.fingerprints);
        let dense_latent = encoder.forward_dense(dense);
        let sparse_values = sparse_latent
            .to_data()
            .as_slice::<f32>()
            .expect("latent should be f32")
            .to_vec();
        let dense_values = dense_latent
            .to_data()
            .as_slice::<f32>()
            .expect("latent should be f32")
            .to_vec();

        for (sparse, dense) in sparse_values.iter().zip(dense_values) {
            assert!((sparse - dense).abs() < 1.0e-5);
        }
    }

    #[test]
    fn decoder_builder_rejects_missing_or_zero_fields() {
        let err = DecoderConfig::builder()
            .build()
            .expect_err("missing latent_width should fail");
        assert!(matches!(
            err,
            crate::Error::ConfigInvalid { message } if message.contains("latent_width")
        ));

        let err = DecoderConfig::builder()
            .latent_width(8)
            .build()
            .expect_err("missing output_width should fail");
        assert!(matches!(
            err,
            crate::Error::ConfigInvalid { message } if message.contains("output_width")
        ));

        let err = DecoderConfig::builder()
            .latent_width(0)
            .output_width(64)
            .build()
            .expect_err("zero latent_width should fail");
        assert!(matches!(
            err,
            crate::Error::ConfigInvalid { message } if message.contains("latent_width")
        ));

        let err = DecoderConfig::builder()
            .latent_width(8)
            .output_width(0)
            .build()
            .expect_err("zero output_width should fail");
        assert!(matches!(
            err,
            crate::Error::ConfigInvalid { message } if message.contains("output_width")
        ));

        let err = DecoderConfig::builder()
            .latent_width(8)
            .output_width(64)
            .hidden_widths(vec![32, 0, 16])
            .build()
            .expect_err("zero hidden width should fail");
        assert!(matches!(
            err,
            crate::Error::ConfigInvalid { message } if message.contains("hidden widths")
        ));

        let config = DecoderConfig::builder()
            .latent_width(8)
            .output_width(64)
            .hidden_widths(vec![32, 16])
            .build()
            .expect("valid builder must produce a config");
        assert_eq!(config.latent_width(), 8);
        assert_eq!(config.output_width(), 64);
        assert_eq!(config.hidden_widths(), &[32, 16]);
    }

    #[test]
    fn validate_rejects_decoder_output_width_mismatch() {
        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.decoder.output_width += 1;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("decoder output width mismatch must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("decoder output width")
        ));
    }

    #[test]
    fn validate_rejects_empty_encoder_hidden_widths() {
        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.encoder.hidden_widths.clear();
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("empty encoder hidden widths must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("encoder hidden widths")
        ));
    }

    #[test]
    fn validate_rejects_zero_latent_width() {
        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.encoder.latent_width = 0;
        config.decoder.latent_width = 0;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("zero latent width must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("latent width")
        ));
    }

    #[test]
    fn validate_rejects_encoder_decoder_latent_width_mismatch() {
        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.decoder.latent_width = config.encoder.latent_width + 1;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("latent width mismatch must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("latent width")
        ));
    }

    #[test]
    fn validate_rejects_zero_descriptor_width() {
        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.descriptor_width = 0;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("zero descriptor width must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("descriptor")
        ));
    }

    #[test]
    fn validate_rejects_invalid_reconstruction_beta_and_weights() {
        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.reconstruction_loss.beta = 0.0;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("zero beta must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("beta")
        ));

        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.reconstruction_loss.beta = f64::INFINITY;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("infinite beta must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("beta")
        ));

        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.reconstruction_loss.zero_weight = -1.0;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("negative reconstruction weight must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("reconstruction loss weights")
        ));
    }

    #[test]
    fn validate_rejects_invalid_tanimoto_ranking_configuration() {
        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.tanimoto_ranking.metric_temperature = -1.0;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("non-positive metric temperature must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("metric temperature")
        ));

        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.tanimoto_ranking.min_gap = -0.1;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("negative min_gap must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("min_gap")
        ));

        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.auxiliary_weights.tanimoto_ranking = 1.0;
        config.tanimoto_ranking.candidates_per_anchor = 1;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("candidates < 2 with positive ranking weight must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("candidates_per_anchor")
        ));
    }

    #[test]
    fn validate_rejects_zero_ecfp_radius_and_nonfinite_noise() {
        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.ecfp_radius = 0;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("zero ecfp radius must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("ecfp radius")
        ));

        let mut config = MoleculeAutoencoderConfig::v1_counted_ecfp();
        config.latent_noise_std = f64::NAN;
        let error = config
            .validate(config.encoder.input_width)
            .expect_err("NaN latent noise std must be rejected");
        assert!(matches!(
            error,
            crate::Error::ConfigInvalid { message } if message.contains("latent noise")
        ));
    }

    #[test]
    fn tanimoto_ranking_config_accessors_expose_builder_values() {
        let config = TanimotoRankingConfig::builder()
            .latent_temperature(0.42)
            .metric_temperature(0.84)
            .min_gap(0.05)
            .candidates_per_anchor(7)
            .pairs_per_batch(11)
            .loss_kind(TanimotoLossKind::SoftRankSpearman)
            .softrank_temperature(0.33)
            .build()
            .expect("valid builder");
        assert!((config.latent_temperature() - 0.42).abs() < 1.0e-12);
        assert!((config.metric_temperature() - 0.84).abs() < 1.0e-12);
        assert!((config.min_gap() - 0.05).abs() < 1.0e-12);
        assert_eq!(config.candidates_per_anchor(), 7);
        assert_eq!(config.pairs_per_batch(), 11);
        assert_eq!(config.loss_kind(), TanimotoLossKind::SoftRankSpearman);
        assert!((config.softrank_temperature() - 0.33).abs() < 1.0e-12);
    }

    #[test]
    fn reconstruction_loss_kind_serde_round_trip_matches_cli_spelling() {
        // Both variants must round-trip with the CLI flag spellings so
        // a config dumped during training reloads cleanly through the
        // CLI parser.
        assert_eq!(
            serde_json::to_string(&ReconstructionLossKind::Huber).expect("serialize"),
            "\"huber\""
        );
        assert_eq!(
            serde_json::to_string(&ReconstructionLossKind::Poisson).expect("serialize"),
            "\"poisson\""
        );
        let parsed: ReconstructionLossKind =
            serde_json::from_str("\"poisson\"").expect("deserialize");
        assert_eq!(parsed, ReconstructionLossKind::Poisson);
    }

    #[test]
    fn poisson_loss_hand_computed_values_match_formula() {
        type B = burn::backend::NdArray<f32, i64>;
        // Two-row batch with a 4-bin "fingerprint". Hand-construct the
        // sparse target so position 0 has count=2 in row 0, position 2
        // has count=1 in row 1.
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let predicted_log_counts =
            Tensor::<B, 2>::from_floats([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], &device);
        let indices = Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![0_i64, 0, 2, 0], [2, 2]),
            &device,
        );
        let counts = Tensor::<B, 2>::from_data(
            TensorData::new(vec![2.0_f32, 0.0, 1.0, 0.0], [2, 2]),
            &device,
        );
        let log_counts =
            Tensor::<B, 2>::from_data(TensorData::new(vec![0.0_f32; 4], [2, 2]), &device);
        let mask = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0_f32, 0.0, 1.0, 0.0], [2, 2]),
            &device,
        );
        let target = crate::batch::SparseFingerprintBatch::<B> {
            indices,
            counts,
            log_counts,
            mask,
            fingerprint_size: 4,
            max_nnz: 2,
        };
        // Per-cell NLL with z=0 everywhere: exp(0)=1 at every cell, and
        // the count*z subtraction is 0 (z=0). So per-row NLL =
        // (4 * 1 - 0) / 4 = 1.0 for both rows.
        let per_row =
            weighted_sparse_log_count_poisson_loss_per_row(predicted_log_counts, target, 20.0)
                .into_data()
                .to_vec::<f32>()
                .expect("f32 data");
        assert!((per_row[0] - 1.0).abs() < 1.0e-6, "row 0 = {}", per_row[0]);
        assert!((per_row[1] - 1.0).abs() < 1.0e-6, "row 1 = {}", per_row[1]);
    }

    #[test]
    fn poisson_loss_optimum_lands_at_log_count() {
        type B = burn::backend::NdArray<f32, i64>;
        // Row 0 active position has count=2 -> Poisson NLL minimum at
        // z = log(2). Picking z = log(2) at the active bin and z very
        // negative elsewhere should give a clearly lower per-row NLL
        // than the trivial-zero baseline of 1.0 / cell.
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let predicted_log_counts =
            Tensor::<B, 2>::from_floats([[2.0_f32.ln(), -10.0, -10.0, -10.0]], &device);
        let indices = Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![0_i64, 0], [1, 2]),
            &device,
        );
        let counts =
            Tensor::<B, 2>::from_data(TensorData::new(vec![2.0_f32, 0.0], [1, 2]), &device);
        let log_counts =
            Tensor::<B, 2>::from_data(TensorData::new(vec![0.0_f32; 2], [1, 2]), &device);
        let mask = Tensor::<B, 2>::from_data(TensorData::new(vec![1.0_f32, 0.0], [1, 2]), &device);
        let target = crate::batch::SparseFingerprintBatch::<B> {
            indices,
            counts,
            log_counts,
            mask,
            fingerprint_size: 4,
            max_nnz: 2,
        };
        let per_row =
            weighted_sparse_log_count_poisson_loss_per_row(predicted_log_counts, target, 20.0)
                .into_data()
                .to_vec::<f32>()
                .expect("f32 data");
        // Per-row NLL ≈ (exp(log(2)) + 3·exp(-10) - 2·log(2)) / 4
        //            ≈ (2 + 0 - 1.386) / 4 ≈ 0.153.
        assert!(
            per_row[0] < 0.2,
            "optimized row should be well below the trivial-zero 1.0 baseline, got {}",
            per_row[0]
        );
    }

    #[test]
    fn tanimoto_loss_kind_serde_round_trip_matches_cli_spelling() {
        // The CLI flag spells the SoftRank variant `softrank-spearman`
        // (two words); the library variant is `SoftRankSpearman` (three
        // words in Rust camel-case). Without the explicit
        // `#[serde(rename = "softrank-spearman")]` override, serde's
        // default kebab-case rule would emit `soft-rank-spearman` and
        // configs written by the trainer would not round-trip with the
        // CLI parser.
        assert_eq!(
            serde_json::to_string(&TanimotoLossKind::SoftmaxCe).expect("serialize"),
            "\"softmax-ce\""
        );
        assert_eq!(
            serde_json::to_string(&TanimotoLossKind::SoftRankSpearman).expect("serialize"),
            "\"softrank-spearman\""
        );
        let parsed: TanimotoLossKind =
            serde_json::from_str("\"softrank-spearman\"").expect("deserialize");
        assert_eq!(parsed, TanimotoLossKind::SoftRankSpearman);
    }

    #[test]
    fn tanimoto_ranking_builder_rejects_non_finite_or_non_positive_fields() {
        let err = TanimotoRankingConfig::builder()
            .latent_temperature(0.0)
            .build()
            .expect_err("zero latent temperature must be rejected");
        assert!(matches!(
            err,
            crate::Error::ConfigInvalid { message } if message.contains("latent temperature")
        ));

        let err = TanimotoRankingConfig::builder()
            .metric_temperature(f64::NAN)
            .build()
            .expect_err("NaN metric temperature must be rejected");
        assert!(matches!(
            err,
            crate::Error::ConfigInvalid { message } if message.contains("metric temperature")
        ));

        let err = TanimotoRankingConfig::builder()
            .min_gap(-1.0)
            .build()
            .expect_err("negative min_gap must be rejected");
        assert!(matches!(
            err,
            crate::Error::ConfigInvalid { message } if message.contains("min_gap")
        ));
    }
}
