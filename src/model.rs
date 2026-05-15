//! Dense counted-fingerprint autoencoder model and losses.

#[cfg(feature = "std")]
use std::path::Path;

use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    tensor::{Distribution, Tensor},
};
use serde::{Deserialize, Serialize};

use crate::{
    batch::{MoleculeAutoencoderBatch, SparseFingerprintBatch},
    features::REGRESSION_TARGET_WIDTH,
    fingerprints::DEFAULT_ECFP_SIZE,
    ranking::weighted_tanimoto_ranking_output,
};
#[cfg(feature = "std")]
use crate::{Error, Result};

const fn default_descriptor_weight() -> f64 {
    0.05
}

const fn default_tanimoto_ranking_weight() -> f64 {
    0.10
}

const fn default_latent_noise_std() -> f64 {
    0.02
}

const fn default_tanimoto_ranking_latent_temperature() -> f64 {
    0.10
}

const fn default_tanimoto_ranking_metric_temperature() -> f64 {
    0.10
}

/// Encoder model configuration. Construct via [`EncoderConfig::builder`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderConfig {
    input_width: usize,
    hidden_widths: Vec<usize>,
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
    input_width: Option<usize>,
    hidden_widths: Option<Vec<usize>>,
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
    latent_width: usize,
    hidden_widths: Vec<usize>,
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
    latent_width: Option<usize>,
    hidden_widths: Option<Vec<usize>>,
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
    beta: f64,
    zero_weight: f64,
    nonzero_weight: f64,
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
}

/// Fluent builder for [`ReconstructionLossConfig`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructionLossConfigBuilder {
    beta: f64,
    zero_weight: f64,
    nonzero_weight: f64,
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
            beta: 1.0,
            zero_weight: 0.05,
            nonzero_weight: 1.0,
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
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::ConfigInvalid {
                    message: format!("{label} must be finite and non-negative, got {value}"),
                });
            }
        }
        Ok(ReconstructionLossConfig {
            beta: self.beta,
            zero_weight: self.zero_weight,
            nonzero_weight: self.nonzero_weight,
        })
    }
}

/// Auxiliary side-task weights. Construct via [`AuxiliaryLossWeights::builder`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AuxiliaryLossWeights {
    #[serde(default = "default_descriptor_weight")]
    descriptors: f64,
    #[serde(default = "default_tanimoto_ranking_weight")]
    tanimoto_ranking: f64,
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
}

/// Fluent builder for [`AuxiliaryLossWeights`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuxiliaryLossWeightsBuilder {
    descriptors: f64,
    tanimoto_ranking: f64,
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
            descriptors: default_descriptor_weight(),
            tanimoto_ranking: default_tanimoto_ranking_weight(),
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

    /// Validates and builds the immutable [`AuxiliaryLossWeights`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when either weight is negative or
    /// non-finite.
    pub fn build(self) -> Result<AuxiliaryLossWeights> {
        for (label, value) in [
            ("descriptors", self.descriptors),
            ("tanimoto_ranking", self.tanimoto_ranking),
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
        })
    }
}

/// Counted Tanimoto sampled softmax latent-geometry loss configuration.
/// Construct via [`TanimotoRankingConfig::builder`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TanimotoRankingConfig {
    #[serde(
        default = "default_tanimoto_ranking_latent_temperature",
        alias = "margin",
        alias = "tanimoto_ranking_margin"
    )]
    latent_temperature: f64,
    #[serde(default = "default_tanimoto_ranking_metric_temperature")]
    metric_temperature: f64,
    min_gap: f64,
    candidates_per_anchor: usize,
    pairs_per_batch: usize,
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
}

/// Fluent builder for [`TanimotoRankingConfig`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TanimotoRankingConfigBuilder {
    latent_temperature: f64,
    metric_temperature: f64,
    min_gap: f64,
    candidates_per_anchor: usize,
    pairs_per_batch: usize,
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
            latent_temperature: default_tanimoto_ranking_latent_temperature(),
            metric_temperature: default_tanimoto_ranking_metric_temperature(),
            min_gap: 0.05,
            candidates_per_anchor: 4,
            pairs_per_batch: 0,
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
        Ok(TanimotoRankingConfig {
            latent_temperature: self.latent_temperature,
            metric_temperature: self.metric_temperature,
            min_gap: self.min_gap,
            candidates_per_anchor: self.candidates_per_anchor,
            pairs_per_batch: self.pairs_per_batch,
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
    weight: f64,
    latent_temperature: f64,
    metric_temperature: f64,
    min_gap: f64,
    candidates_per_anchor: usize,
    pairs_per_batch: usize,
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
}

/// Full molecule autoencoder configuration. Construct via
/// [`MoleculeAutoencoderConfig::builder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoleculeAutoencoderConfig {
    encoder: EncoderConfig,
    decoder: DecoderConfig,
    descriptor_width: usize,
    reconstruction_loss: ReconstructionLossConfig,
    auxiliary_weights: AuxiliaryLossWeights,
    #[serde(default)]
    tanimoto_ranking: TanimotoRankingConfig,
    #[serde(default = "default_latent_noise_std")]
    latent_noise_std: f64,
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
        if self.reconstruction_loss.zero_weight < 0.0 || self.reconstruction_loss.nonzero_weight < 0.0
        {
            return bail("reconstruction loss weights must be non-negative".to_string());
        }
        if self.auxiliary_weights.descriptors < 0.0
            || self.auxiliary_weights.tanimoto_ranking < 0.0
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
            reconstruction_beta: self.reconstruction_loss.beta,
            zero_weight: self.reconstruction_loss.zero_weight,
            nonzero_weight: self.reconstruction_loss.nonzero_weight,
            descriptor_weight: self.auxiliary_weights.descriptors,
            tanimoto_ranking_weight: self.auxiliary_weights.tanimoto_ranking,
            tanimoto_ranking_latent_temperature: self.tanimoto_ranking.latent_temperature,
            tanimoto_ranking_metric_temperature: self.tanimoto_ranking.metric_temperature,
            tanimoto_ranking_min_gap: self.tanimoto_ranking.min_gap,
            tanimoto_ranking_pairs_per_batch: self.tanimoto_ranking.pairs_per_batch,
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
    fingerprint_size: usize,
    latent_width: usize,
    hidden_widths: Vec<usize>,
    descriptor_weight: f64,
    tanimoto_ranking_weight: f64,
    latent_noise_std: f64,
    latent_temperature: f64,
    metric_temperature: f64,
    min_gap: f64,
    candidates_per_anchor: usize,
    pairs_per_batch: usize,
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
            latent_width: 512,
            hidden_widths: vec![4096, 2048, 1024],
            descriptor_weight: default_descriptor_weight(),
            tanimoto_ranking_weight: default_tanimoto_ranking_weight(),
            latent_noise_std: default_latent_noise_std(),
            latent_temperature: default_tanimoto_ranking_latent_temperature(),
            metric_temperature: default_tanimoto_ranking_metric_temperature(),
            min_gap: 0.05,
            candidates_per_anchor: 4,
            pairs_per_batch: 0,
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
            reconstruction_loss: ReconstructionLossConfig::default(),
            auxiliary_weights: AuxiliaryLossWeights {
                descriptors: self.descriptor_weight,
                tanimoto_ranking: self.tanimoto_ranking_weight,
            },
            tanimoto_ranking: TanimotoRankingConfig {
                latent_temperature: self.latent_temperature,
                metric_temperature: self.metric_temperature,
                min_gap: self.min_gap,
                candidates_per_anchor: self.candidates_per_anchor,
                pairs_per_batch: self.pairs_per_batch,
            },
            latent_noise_std: self.latent_noise_std,
        };
        config.validate(fingerprint_size)?;
        Ok(config)
    }
}

/// Encoder module.
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    input: SparseInputLinear<B>,
    layers: Vec<Linear<B>>,
    latent: Linear<B>,
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
    layers: Vec<Linear<B>>,
    output: Linear<B>,
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
}

/// Weighted loss components.
#[derive(Debug)]
pub struct MoleculeLossBreakdown<B: Backend> {
    /// Main counted ECFP reconstruction loss.
    pub reconstruction: Tensor<B, 1>,
    /// Weighted descriptor regression loss.
    pub descriptors: Tensor<B, 1>,
    /// Weighted latent Tanimoto sampled softmax cross-entropy loss.
    pub tanimoto_ranking: Tensor<B, 1>,
    /// Latent Tanimoto geometry best-candidate accuracy.
    pub tanimoto_ranking_accuracy: Tensor<B, 1>,
    /// Number of valid anchors contributing to the Tanimoto geometry loss.
    pub tanimoto_ranking_pairs: Tensor<B, 1>,
}

impl<B: Backend> MoleculeLossBreakdown<B> {
    /// Returns the weighted total loss.
    pub fn total(&self) -> Tensor<B, 1> {
        self.reconstruction.clone() + self.descriptors.clone() + self.tanimoto_ranking.clone()
    }
}

/// Dense counted-fingerprint molecular autoencoder.
#[derive(Module, Debug)]
pub struct MoleculeAutoencoder<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
    descriptor_head: Linear<B>,
    reconstruction_beta: f64,
    zero_weight: f64,
    nonzero_weight: f64,
    descriptor_weight: f64,
    tanimoto_ranking_weight: f64,
    tanimoto_ranking_latent_temperature: f64,
    tanimoto_ranking_metric_temperature: f64,
    tanimoto_ranking_min_gap: f64,
    tanimoto_ranking_pairs_per_batch: usize,
    latent_noise_std: f64,
}

impl<B: Backend> MoleculeAutoencoder<B> {
    fn reconstruction_loss_config(&self) -> ReconstructionLossConfig {
        ReconstructionLossConfig {
            beta: self.reconstruction_beta,
            zero_weight: self.zero_weight,
            nonzero_weight: self.nonzero_weight,
        }
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
            latent,
        }
    }

    /// Runs the full autoencoder from a dense `log1p(counted ECFP)` tensor.
    pub fn forward_dense(&self, input_log_counts: Tensor<B, 2>) -> MoleculeAutoencoderOutput<B> {
        let latent = self.encoder.forward_dense(input_log_counts);
        MoleculeAutoencoderOutput {
            reconstructed_log_counts: self.decoder.forward(latent.clone()),
            descriptor_predictions: self.descriptor_head.forward(latent.clone()),
            latent,
        }
    }

    /// Computes all weighted loss components for a batch.
    pub fn loss(&self, batch: MoleculeAutoencoderBatch<B>) -> MoleculeLossBreakdown<B> {
        let MoleculeAutoencoderBatch {
            fingerprints,
            descriptor_targets,
            tanimoto_ranking,
            ..
        } = batch;
        let output = self.forward(&fingerprints);
        self.loss_from_output(output, fingerprints, descriptor_targets, tanimoto_ranking)
    }

    /// Computes all weighted losses from a precomputed model output.
    pub fn loss_from_output(
        &self,
        output: MoleculeAutoencoderOutput<B>,
        fingerprints: SparseFingerprintBatch<B>,
        descriptor_targets: Tensor<B, 2>,
        tanimoto_ranking: crate::ranking::TanimotoRankingBatch<B>,
    ) -> MoleculeLossBreakdown<B> {
        let device = output.reconstructed_log_counts.device();
        let latent = output.latent;
        let reconstruction = weighted_sparse_log_count_huber_loss(
            output.reconstructed_log_counts,
            fingerprints,
            self.reconstruction_loss_config(),
        );
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
        );

        MoleculeLossBreakdown {
            reconstruction,
            descriptors,
            tanimoto_ranking: tanimoto.loss,
            tanimoto_ranking_accuracy: tanimoto.accuracy,
            tanimoto_ranking_pairs: tanimoto.valid_pairs,
        }
    }
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

fn huber_from_delta<B: Backend>(delta: Tensor<B, 2>, beta: f64) -> Tensor<B, 2> {
    let quadratic = delta.clone().powf_scalar(2.0) * (0.5 / beta);
    let linear = delta.clone() - (0.5 * beta);
    linear.mask_where(delta.lower_equal_elem(beta), quadratic)
}

/// Weighted Huber loss over `log1p(count)` reconstruction targets.
pub fn weighted_log_count_huber_loss<B: Backend>(
    predicted_log_counts: Tensor<B, 2>,
    target_log_counts: Tensor<B, 2>,
    config: ReconstructionLossConfig,
) -> Tensor<B, 1> {
    let delta = (predicted_log_counts - target_log_counts.clone()).abs();
    let huber = huber_from_delta(delta, config.beta);
    let nonzero = target_log_counts.greater_elem(0.0).float();
    let weights =
        nonzero.clone() * config.nonzero_weight + (nonzero * -1.0 + 1.0) * config.zero_weight;
    (huber * weights).mean()
}

/// Weighted Huber loss over sparse `log1p(count)` reconstruction targets.
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

    let denominator = (predicted_dims[0] * predicted_dims[1]) as f64;
    let zero_huber = huber_from_delta(predicted_log_counts.clone().abs(), config.beta);
    let zero_total = zero_huber.sum() * config.zero_weight;
    let predicted_nonzero = predicted_log_counts.gather(1, target.indices.clone());
    let nonzero_huber = huber_from_delta(
        (predicted_nonzero.clone() - target.log_counts).abs(),
        config.beta,
    );
    let zero_nonzero_huber = huber_from_delta(predicted_nonzero.abs(), config.beta);
    let correction = (nonzero_huber * config.nonzero_weight
        - zero_nonzero_huber * config.zero_weight)
        * target.mask;

    (zero_total + correction.sum()) / denominator
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
        assert_eq!(config.tanimoto_ranking.latent_temperature, 0.10);
        assert_eq!(config.tanimoto_ranking.metric_temperature, 0.10);
        assert_eq!(config.tanimoto_ranking.min_gap, 0.05);
        assert_eq!(config.tanimoto_ranking.candidates_per_anchor, 4);
        assert_eq!(config.tanimoto_ranking.pairs_per_batch, 0);
        assert_eq!(config.latent_noise_std, 0.02);
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

        let losses = model.loss(batch);
        assert!(losses.total().into_scalar().is_finite());
        assert_eq!(losses.tanimoto_ranking.dims(), [1]);
        assert_eq!(losses.tanimoto_ranking_accuracy.dims(), [1]);
        assert_eq!(losses.tanimoto_ranking_pairs.dims(), [1]);
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
                },
                MoleculeAutoencoderSample {
                    cid: 2,
                    fingerprint_indices: vec![1],
                    fingerprint_counts: vec![3],
                    descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
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
        let config = ReconstructionLossConfig {
            beta: 0.7,
            zero_weight: 0.05,
            nonzero_weight: 1.0,
        };

        let dense = weighted_log_count_huber_loss(predicted.clone(), dense_target, config);
        let sparse = weighted_sparse_log_count_huber_loss(predicted, batch.fingerprints, config);

        let delta = (dense.into_scalar() - sparse.into_scalar()).abs();
        assert!(delta < 1.0e-6, "dense and sparse loss differ by {delta}");
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
                },
                MoleculeAutoencoderSample {
                    cid: 2,
                    fingerprint_indices: vec![1, 3, 5],
                    fingerprint_counts: vec![1, 2, 1],
                    descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
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
}
