//! Dense counted-fingerprint autoencoder model and losses.

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

/// Encoder model configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Input feature width.
    pub input_width: usize,
    /// Hidden layer widths.
    pub hidden_widths: Vec<usize>,
    /// Latent embedding width.
    pub latent_width: usize,
}

impl EncoderConfig {
    /// Creates an initialized encoder.
    ///
    /// # Panics
    ///
    /// Panics when `hidden_widths` is empty. The sparse input projection needs
    /// at least one hidden width.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        assert!(
            !self.hidden_widths.is_empty(),
            "encoder hidden widths must not be empty for sparse input"
        );
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

/// Decoder model configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Latent embedding width.
    pub latent_width: usize,
    /// Hidden layer widths.
    pub hidden_widths: Vec<usize>,
    /// Output reconstruction width.
    pub output_width: usize,
}

impl DecoderConfig {
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

/// Counted ECFP reconstruction loss configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionLossConfig {
    /// Huber transition point in log-count space.
    pub beta: f64,
    /// Weight for zero-count bins.
    pub zero_weight: f64,
    /// Weight for nonzero-count bins.
    pub nonzero_weight: f64,
}

impl Default for ReconstructionLossConfig {
    fn default() -> Self {
        Self {
            beta: 1.0,
            zero_weight: 0.05,
            nonzero_weight: 1.0,
        }
    }
}

/// Auxiliary side-task weights.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AuxiliaryLossWeights {
    /// Weight for scalar descriptor regression.
    #[serde(default = "default_descriptor_weight")]
    pub descriptors: f64,
    /// Weight for preserving counted-fingerprint Tanimoto ordering in latent space.
    #[serde(default = "default_tanimoto_ranking_weight")]
    pub tanimoto_ranking: f64,
}

impl Default for AuxiliaryLossWeights {
    fn default() -> Self {
        Self {
            descriptors: default_descriptor_weight(),
            tanimoto_ranking: default_tanimoto_ranking_weight(),
        }
    }
}

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

/// Counted Tanimoto pairwise latent-geometry loss configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TanimotoRankingConfig {
    /// Temperature applied to latent cosine gaps in the pairwise logistic loss.
    #[serde(
        default = "default_tanimoto_ranking_latent_temperature",
        alias = "margin",
        alias = "tanimoto_ranking_margin"
    )]
    pub latent_temperature: f64,
    /// Temperature applied to counted Tanimoto gaps in the pairwise logistic loss.
    #[serde(default = "default_tanimoto_ranking_metric_temperature")]
    pub metric_temperature: f64,
    /// Minimum counted Tanimoto gap required for an anchor to contribute.
    pub min_gap: f64,
    /// Random candidate partners sampled for each anchor by the GPU metric kernel.
    pub candidates_per_anchor: usize,
    /// Maximum anchors used by the latent geometry loss; `0` means all rows.
    pub pairs_per_batch: usize,
}

impl Default for TanimotoRankingConfig {
    fn default() -> Self {
        Self {
            latent_temperature: default_tanimoto_ranking_latent_temperature(),
            metric_temperature: default_tanimoto_ranking_metric_temperature(),
            min_gap: 0.05,
            candidates_per_anchor: 4,
            pairs_per_batch: 0,
        }
    }
}

/// Full molecule autoencoder configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoleculeAutoencoderConfig {
    /// Encoder configuration.
    pub encoder: EncoderConfig,
    /// Decoder configuration.
    pub decoder: DecoderConfig,
    /// Descriptor regression target width.
    pub descriptor_width: usize,
    /// Main reconstruction loss.
    pub reconstruction_loss: ReconstructionLossConfig,
    /// Side-task loss weights.
    pub auxiliary_weights: AuxiliaryLossWeights,
    /// Counted Tanimoto pairwise latent-geometry side task.
    #[serde(default)]
    pub tanimoto_ranking: TanimotoRankingConfig,
    /// Decoder-side latent Gaussian noise as a fraction of batch latent standard deviation.
    ///
    /// This is applied only during training, after the encoder and before the decoder.
    /// A value of `0.0` disables latent denoising.
    #[serde(default = "default_latent_noise_std")]
    pub latent_noise_std: f64,
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
    /// Weighted latent Tanimoto pairwise logistic loss.
    pub tanimoto_ranking: Tensor<B, 1>,
    /// Latent Tanimoto geometry ordering accuracy.
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
    /// Encoder module.
    pub encoder: Encoder<B>,
    /// Decoder module.
    pub decoder: Decoder<B>,
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
