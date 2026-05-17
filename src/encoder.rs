//! Inference helper that loads a trained checkpoint and embeds SMILES.
//!
//! This wraps the primitives the training loop already uses
//! ([`MoleculeAutoencoderConfig::load_json`], [`Module::load_file`],
//! [`compute_fingerprint_targets`], [`MoleculeAutoencoderBatcher`],
//! [`MoleculeAutoencoder::forward`], and
//! [`batch_sparse_log_count_tanimoto`]) into a single
//! "load → encode → metrics" pipeline. The bin layer is just a clap
//! shell around this type; downstream library consumers can use it
//! directly without depending on any of the I/O traits in
//! [`crate::embed`].

use std::path::Path;

use burn::{module::Module, prelude::*, record::DefaultRecorder, tensor::Transaction};
use smiles_parser::prelude::Smiles;

use crate::{
    CountedEcfpConfig, Error, MoleculeAutoencoder, MoleculeAutoencoderBatch,
    MoleculeAutoencoderBatcher, MoleculeAutoencoderConfig, MoleculeAutoencoderSample, Result,
    batch_sparse_log_count_tanimoto, compute_fingerprint_targets,
    features::REGRESSION_TARGET_WIDTH,
};

/// One molecule's encoder output: the latent embedding plus the two
/// reconstruction-quality signals used for out-of-distribution detection.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodingRow {
    /// The input SMILES, copied through so callers can correlate outputs
    /// to inputs without holding the original `Vec<String>` alongside.
    pub smiles: String,
    /// Latent embedding; length is [`MoleculeEncoder::latent_width`].
    pub latent: Vec<f32>,
    /// Per-row counted-Tanimoto similarity between the input sparse
    /// fingerprint and the decoder's reconstructed counts. Range `[0, 1]`;
    /// low values flag inputs the model can't reproduce well.
    pub reconstruction_count_tanimoto: f32,
    /// Per-row mean-squared error over `log1p(count)` predictions. Higher
    /// values mean the reconstruction is further from the input.
    pub reconstruction_log_mse: f32,
}

/// Loads a trained checkpoint and embeds SMILES batches on demand.
///
/// Construct via [`MoleculeEncoder::from_checkpoint`]. The encoder owns a
/// [`finge_rs::smiles_support::SmilesRdkitScratch`] for fingerprint
/// preparation, so individual `encode` calls reuse the allocation. The
/// fingerprint radius is sourced from
/// [`MoleculeAutoencoderConfig::ecfp_radius`] so inference matches the
/// preprocessing that produced the cached shards.
pub struct MoleculeEncoder<B: Backend> {
    model: MoleculeAutoencoder<B>,
    config: MoleculeAutoencoderConfig,
    batcher: MoleculeAutoencoderBatcher,
    ecfp_config: CountedEcfpConfig,
    scratch: finge_rs::smiles_support::SmilesRdkitScratch,
    device: B::Device,
}

impl<B: Backend<FloatElem = f32>> MoleculeEncoder<B> {
    /// Loads `model-config.json` and `model.mpk` from a training run
    /// directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the config file cannot be read or fails
    /// validation, when the model weights cannot be deserialized, or when
    /// the encoder's input width does not match the configured fingerprint
    /// width.
    pub fn from_checkpoint(checkpoint_dir: &Path, device: B::Device) -> Result<Self> {
        let config_path = checkpoint_dir.join("model-config.json");
        let config = MoleculeAutoencoderConfig::load_json(&config_path)?;
        let fingerprint_size = config.encoder().input_width();
        config.validate(fingerprint_size)?;

        let recorder = DefaultRecorder::default();
        let mut model = config.init::<B>(&device);
        model = model
            .load_file(checkpoint_dir.join("model"), &recorder, &device)
            .map_err(|source| Error::Io {
                path: checkpoint_dir.join("model"),
                source: std::io::Error::other(source.to_string()),
            })?;

        let batcher = MoleculeAutoencoderBatcher::new(fingerprint_size, config.descriptor_width());
        let ecfp_config = CountedEcfpConfig::builder()
            .radius(config.ecfp_radius())
            .size(fingerprint_size)
            .build()?;

        Ok(Self {
            model,
            config,
            batcher,
            ecfp_config,
            scratch: finge_rs::smiles_support::SmilesRdkitScratch::default(),
            device,
        })
    }

    /// Latent embedding width (column count of [`EncodingRow::latent`]).
    #[must_use]
    pub fn latent_width(&self) -> usize {
        self.config.encoder().latent_width()
    }

    /// Counted-ECFP fingerprint width used internally.
    #[must_use]
    pub fn fingerprint_size(&self) -> usize {
        self.config.encoder().input_width()
    }

    /// Returns the underlying model config (e.g. for emitting a sink schema).
    #[must_use]
    pub fn config(&self) -> &MoleculeAutoencoderConfig {
        &self.config
    }

    /// Encodes a batch of SMILES into latent embeddings plus reconstruction
    /// metrics.
    ///
    /// Per-row failures (unparseable SMILES, fingerprint preparation error)
    /// yield `Ok(None)` for that row so callers can skip the failure and
    /// keep processing the rest of the batch. The returned `Vec` has the
    /// same length as `smiles` and preserves order.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is empty, or when the device-side
    /// tensor transaction fails to materialize the output scalars.
    pub fn encode(&mut self, smiles: &[String]) -> Result<Vec<Option<EncodingRow>>> {
        if smiles.is_empty() {
            return Ok(Vec::new());
        }

        let mut samples = Vec::with_capacity(smiles.len());
        let mut row_to_sample = Vec::with_capacity(smiles.len());
        for text in smiles {
            let Some(sample) = self.try_build_sample(text) else {
                row_to_sample.push(None);
                continue;
            };
            row_to_sample.push(Some(samples.len()));
            samples.push(sample);
        }

        if samples.is_empty() {
            return Ok(smiles.iter().map(|_| None).collect());
        }

        let (batch, _profile): (MoleculeAutoencoderBatch<B>, _) =
            self.batcher.batch_profiled(samples, &self.device);
        let output = self.model.forward(&batch.fingerprints);

        let tanimoto = batch_sparse_log_count_tanimoto(
            &batch.fingerprints,
            output.reconstructed_log_counts.clone(),
        );

        let dense_target = batch.fingerprints.to_dense_log_counts();
        let log_mse_2d: Tensor<B, 2> = (dense_target - output.reconstructed_log_counts.clone())
            .powf_scalar(2.0)
            .mean_dim(1);
        let log_mse: Tensor<B, 1> = log_mse_2d.squeeze_dim(1);

        let [latent, tanimoto_data, log_mse_data] = Transaction::default()
            .register(output.latent)
            .register(tanimoto)
            .register(log_mse)
            .try_execute()
            .map_err(|source| Error::InvalidBatch(format!("encode transaction failed: {source}")))?
            .try_into()
            .map_err(|_| {
                Error::InvalidBatch("encode transaction returned wrong number of tensors".into())
            })?;

        let latent_width = self.latent_width();
        let latent_values = latent
            .as_slice::<f32>()
            .map_err(|source| Error::InvalidBatch(format!("latent tensor not f32: {source}")))?;
        let tanimoto_values = tanimoto_data
            .as_slice::<f32>()
            .map_err(|source| Error::InvalidBatch(format!("tanimoto tensor not f32: {source}")))?;
        let log_mse_values = log_mse_data
            .as_slice::<f32>()
            .map_err(|source| Error::InvalidBatch(format!("log_mse tensor not f32: {source}")))?;

        let rows = row_to_sample
            .into_iter()
            .enumerate()
            .map(|(input_index, sample_index)| {
                sample_index.map(|index| {
                    let start = index * latent_width;
                    let end = start + latent_width;
                    EncodingRow {
                        smiles: smiles[input_index].clone(),
                        latent: latent_values[start..end].to_vec(),
                        reconstruction_count_tanimoto: tanimoto_values[index],
                        reconstruction_log_mse: log_mse_values[index],
                    }
                })
            })
            .collect();

        Ok(rows)
    }

    fn try_build_sample(&mut self, smiles: &str) -> Option<MoleculeAutoencoderSample> {
        let parsed: Smiles = smiles.parse().ok()?;
        let fingerprint =
            compute_fingerprint_targets(0, &parsed, self.ecfp_config, &mut self.scratch).ok()?;
        Some(MoleculeAutoencoderSample {
            cid: 0,
            fingerprint_indices: fingerprint.indices().to_vec(),
            fingerprint_counts: fingerprint.counts().to_vec(),
            descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
        })
    }
}
