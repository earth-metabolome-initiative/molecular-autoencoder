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

// `reqwest::Response` and the wasm-bindgen futures it produces are not
// `Send` on wasm32 (browser fetch values aren't shareable between
// threads), so the async constructors below intentionally return futures
// without `Send` when targeting wasm. Single-threaded JS is the only
// runtime there; on native targets the lint stays active.
#![cfg_attr(target_arch = "wasm32", allow(clippy::future_not_send))]

use std::path::Path;

use burn::{
    module::Module,
    prelude::*,
    record::{DefaultRecorder, FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::Transaction,
};
use smiles_parser::prelude::Smiles;

use crate::{
    CountedEcfpConfig, Error, MoleculeAutoencoder, MoleculeAutoencoderBatch,
    MoleculeAutoencoderBatcher, MoleculeAutoencoderConfig, MoleculeAutoencoderSample, Result,
    batch_sparse_log_count_binary_tanimoto, batch_sparse_log_count_tanimoto,
    compute_fingerprint_targets, features::REGRESSION_TARGET_WIDTH,
};

#[cfg(feature = "http")]
mod http_loader {
    use std::path::PathBuf;

    use sha2::{Digest, Sha256};

    use crate::{Error, Result};

    /// Filename for the JSON config artifact when laid out at a URL or in
    /// a HuggingFace repository.
    pub const CONFIG_FILENAME: &str = "model-config.json";
    /// Filename for the model record (Burn `NamedMpk` format).
    pub const RECORD_FILENAME: &str = "model.mpk";

    /// Cache scope derived from two URLs: a 16-hex-char prefix of the
    /// SHA-256 of `config_url || '\n' || record_url`. Lets two calls to
    /// `from_url` with different URLs share no cache entries while keeping
    /// the directory name compact.
    pub fn url_cache_scope(config_url: &str, record_url: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(config_url.as_bytes());
        hasher.update(b"\n");
        hasher.update(record_url.as_bytes());
        let digest = hasher.finalize();
        format!("url-{}", hex::encode(&digest[..8]))
    }

    /// Resolves the platform default cache root, or `None` when the
    /// target has no addressable filesystem (wasm32) or the OS exposes
    /// no per-user cache location.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn default_cache_root() -> Option<PathBuf> {
        dirs::cache_dir().map(|root| root.join("molecular-autoencoder"))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn default_cache_root() -> Option<PathBuf> {
        None
    }

    /// HuggingFace base URL ("https://huggingface.co"). Pulled out as a
    /// constant so a mirror or private endpoint can be swapped in via a
    /// `with_base_url` constructor later without touching call sites.
    pub const HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";

    /// Builds the canonical `(config_url, record_url)` pair for a
    /// HuggingFace repository revision. Extracted so the URL pattern is
    /// directly unit-testable instead of buried inside an async fn.
    pub fn huggingface_artifact_urls(repo: &str, revision: &str) -> (String, String) {
        let base = format!("{HUGGINGFACE_BASE_URL}/{repo}/resolve/{revision}");
        (
            format!("{base}/{CONFIG_FILENAME}"),
            format!("{base}/{RECORD_FILENAME}"),
        )
    }

    /// Cache scope key for HuggingFace artifacts. Stable across runs and
    /// human-readable so `~/.cache/molecular-autoencoder/` stays
    /// browsable.
    pub fn huggingface_cache_scope(repo: &str, revision: &str) -> String {
        format!("huggingface/{repo}/{revision}")
    }

    /// Like [`fetch_bytes`], but with an explicit cache root override.
    /// Pass `None` to disable the on-disk cache entirely (useful in tests
    /// or in CLIs that want full control over where downloads land).
    pub async fn fetch_bytes_with_cache_root(
        url: &str,
        scope: &str,
        filename: &str,
        cache_root: Option<PathBuf>,
    ) -> Result<Vec<u8>> {
        #[cfg(not(target_arch = "wasm32"))]
        let cache_path = cache_root.map(|root| root.join(scope).join(filename));
        #[cfg(target_arch = "wasm32")]
        let _ = (cache_root, scope, filename);

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = cache_path.as_ref()
            && path.exists()
        {
            return std::fs::read(path).map_err(|source| Error::io(path, source));
        }

        let response = reqwest::get(url).await.map_err(|source| Error::Network {
            url: url.to_string(),
            message: source.to_string(),
        })?;
        if !response.status().is_success() {
            return Err(Error::Network {
                url: url.to_string(),
                message: format!("HTTP status {}", response.status()),
            });
        }
        let bytes = response.bytes().await.map_err(|source| Error::Network {
            url: url.to_string(),
            message: source.to_string(),
        })?;

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = cache_path.as_ref() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|source| Error::io(parent, source))?;
            }
            std::fs::write(path, &bytes).map_err(|source| Error::io(path, source))?;
        }

        Ok(bytes.to_vec())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn url_cache_scope_is_deterministic_and_url_sensitive() {
            let base = url_cache_scope("https://a.example/c", "https://a.example/r");
            assert_eq!(
                base,
                url_cache_scope("https://a.example/c", "https://a.example/r")
            );
            // Any single-character change in either URL changes the scope.
            assert_ne!(
                base,
                url_cache_scope("https://a.example/C", "https://a.example/r")
            );
            assert_ne!(
                base,
                url_cache_scope("https://a.example/c", "https://a.example/R")
            );
            assert!(base.starts_with("url-"));
            assert_eq!(base.len(), "url-".len() + 16);
        }

        #[cfg(not(target_arch = "wasm32"))]
        #[test]
        fn default_cache_root_is_under_molecular_autoencoder() {
            let root = default_cache_root().expect("OS exposes a cache dir");
            assert!(
                root.ends_with("molecular-autoencoder"),
                "got {}",
                root.display()
            );
        }

        #[test]
        fn huggingface_urls_follow_resolve_pattern() {
            let (config, record) =
                huggingface_artifact_urls("EarthMetabolomeInitiative/mae", "main");
            assert_eq!(
                config,
                "https://huggingface.co/EarthMetabolomeInitiative/mae/resolve/main/model-config.json"
            );
            assert_eq!(
                record,
                "https://huggingface.co/EarthMetabolomeInitiative/mae/resolve/main/model.mpk"
            );
        }

        #[test]
        fn huggingface_scope_is_path_friendly() {
            assert_eq!(
                huggingface_cache_scope("Org/repo", "v1.0"),
                "huggingface/Org/repo/v1.0"
            );
        }
    }
}

/// One molecule's encoder output: the latent embedding plus the
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
    /// Per-row binary (bit-presence) Tanimoto similarity between the input
    /// fingerprint and the decoder's reconstructed counts. Range `[0, 1]`;
    /// useful for spotting reconstructions that hit the right bits but
    /// miss the magnitudes.
    pub reconstruction_binary_tanimoto: f32,
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
        config.validate(config.encoder().input_width())?;

        let recorder = DefaultRecorder::default();
        let model = config
            .init::<B>(&device)
            .load_file(checkpoint_dir.join("model"), &recorder, &device)
            .map_err(|source| Error::Io {
                path: checkpoint_dir.join("model"),
                source: std::io::Error::other(source.to_string()),
            })?;

        Self::assemble(config, model, device)
    }

    /// Builds the encoder from already-resident config and model-record
    /// bytes, without touching the filesystem.
    ///
    /// `config_json` is the verbatim contents of `model-config.json`;
    /// `model_record` is the verbatim contents of `model.mpk` (the
    /// [`NamedMpkFileRecorder`](burn::record::NamedMpkFileRecorder) format
    /// emitted by [`crate::checkpoint`-style](crate) writers). Loading is
    /// identical to [`from_checkpoint`](Self::from_checkpoint) but the
    /// caller is in charge of acquiring the bytes — which means this is
    /// the entry point that works in WASM, embedded contexts, or anywhere
    /// without an addressable filesystem.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Json`] when `config_json` cannot be deserialized
    /// or fails validation, [`Error::ConfigInvalid`] when the deserialized
    /// configuration is internally inconsistent, and [`Error::Io`] (with a
    /// synthetic `<bytes>` path) when the model record cannot be decoded.
    pub fn from_bytes(
        config_json: &[u8],
        model_record: Vec<u8>,
        device: B::Device,
    ) -> Result<Self> {
        let config = MoleculeAutoencoderConfig::from_json_bytes(config_json)?;
        config.validate(config.encoder().input_width())?;

        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let record =
            Recorder::<B>::load(&recorder, model_record, &device).map_err(|source| Error::Io {
                path: std::path::PathBuf::from("<bytes>"),
                source: std::io::Error::other(source.to_string()),
            })?;
        let model = config.init::<B>(&device).load_record(record);

        Self::assemble(config, model, device)
    }

    /// Shared assembly tail used by every constructor: builds the
    /// fingerprint config + batcher + scratch buffer around an
    /// already-validated config and an already-loaded model. Keeps the
    /// `from_checkpoint` / `from_bytes` / URL-based paths from drifting
    /// over time.
    fn assemble(
        config: MoleculeAutoencoderConfig,
        model: MoleculeAutoencoder<B>,
        device: B::Device,
    ) -> Result<Self> {
        let fingerprint_size = config.encoder().input_width();
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

    /// Builds the encoder by fetching `model-config.json` and `model.mpk`
    /// from the supplied URLs.
    ///
    /// On native targets, the downloaded artifacts are mirrored under
    /// `~/.cache/molecular-autoencoder/<hash>/` so repeated calls with the
    /// same URLs short-circuit the network round-trip. On `wasm32` the
    /// caller relies on the surrounding fetch layer (e.g. the browser's
    /// HTTP cache) for any reuse. Use
    /// [`from_url_with_cache_root`](Self::from_url_with_cache_root) to
    /// point the cache somewhere else or disable it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Network`] when either GET fails or returns a
    /// non-success status, [`Error::Json`] when the config payload is not
    /// valid JSON, and [`Error::Io`] when the model record cannot be
    /// decoded or — on native — when the cache directory cannot be
    /// written.
    #[cfg(feature = "http")]
    pub async fn from_url(config_url: &str, record_url: &str, device: B::Device) -> Result<Self> {
        Self::from_url_with_cache_root(
            config_url,
            record_url,
            device,
            http_loader::default_cache_root(),
        )
        .await
    }

    /// Like [`from_url`](Self::from_url), but with an explicit cache
    /// root. Pass `None` to disable on-disk caching entirely.
    ///
    /// # Errors
    ///
    /// Same conditions as [`from_url`](Self::from_url).
    #[cfg(feature = "http")]
    pub async fn from_url_with_cache_root(
        config_url: &str,
        record_url: &str,
        device: B::Device,
        cache_root: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        let scope = http_loader::url_cache_scope(config_url, record_url);
        Self::from_scoped_urls(config_url, record_url, &scope, cache_root, device).await
    }

    /// Internal helper: fetch the canonical pair of artifacts under a
    /// caller-chosen cache `scope` and assemble the encoder. Centralises
    /// the fetch + `from_bytes` plumbing so the URL- and HuggingFace-
    /// flavoured constructors share one body.
    #[cfg(feature = "http")]
    async fn from_scoped_urls(
        config_url: &str,
        record_url: &str,
        scope: &str,
        cache_root: Option<std::path::PathBuf>,
        device: B::Device,
    ) -> Result<Self> {
        let config_bytes = http_loader::fetch_bytes_with_cache_root(
            config_url,
            scope,
            http_loader::CONFIG_FILENAME,
            cache_root.clone(),
        )
        .await?;
        let record_bytes = http_loader::fetch_bytes_with_cache_root(
            record_url,
            scope,
            http_loader::RECORD_FILENAME,
            cache_root,
        )
        .await?;
        Self::from_bytes(&config_bytes, record_bytes, device)
    }

    /// Builds the encoder by fetching the model artifacts from a
    /// HuggingFace repository revision.
    ///
    /// Resolves `https://huggingface.co/{repo}/resolve/{revision}/model-config.json`
    /// and `.../model.mpk` and delegates to the URL-based loader. On
    /// native targets the cache scope is keyed by repo + revision so each
    /// published revision occupies its own directory. Use
    /// [`from_huggingface_with_cache_root`](Self::from_huggingface_with_cache_root)
    /// to override the cache location.
    ///
    /// # Errors
    ///
    /// Same conditions as [`from_url`](Self::from_url) — typically
    /// [`Error::Network`] when the repository or revision is unreachable
    /// and [`Error::Json`] / [`Error::Io`] when the artifacts are not
    /// shaped as expected.
    #[cfg(feature = "huggingface")]
    pub async fn from_huggingface(repo: &str, revision: &str, device: B::Device) -> Result<Self> {
        Self::from_huggingface_with_cache_root(
            repo,
            revision,
            device,
            http_loader::default_cache_root(),
        )
        .await
    }

    /// Like [`from_huggingface`](Self::from_huggingface), with an
    /// explicit cache root override.
    ///
    /// # Errors
    ///
    /// Same conditions as [`from_huggingface`](Self::from_huggingface).
    #[cfg(feature = "huggingface")]
    pub async fn from_huggingface_with_cache_root(
        repo: &str,
        revision: &str,
        device: B::Device,
        cache_root: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        let (config_url, record_url) = http_loader::huggingface_artifact_urls(repo, revision);
        let scope = http_loader::huggingface_cache_scope(repo, revision);
        Self::from_scoped_urls(&config_url, &record_url, &scope, cache_root, device).await
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

        let count_tanimoto = batch_sparse_log_count_tanimoto(
            &batch.fingerprints,
            output.reconstructed_log_counts.clone(),
        );
        let binary_tanimoto = batch_sparse_log_count_binary_tanimoto(
            &batch.fingerprints,
            output.reconstructed_log_counts.clone(),
        );

        let dense_target = batch.fingerprints.to_dense_log_counts();
        let log_mse_2d: Tensor<B, 2> = (dense_target - output.reconstructed_log_counts.clone())
            .powf_scalar(2.0)
            .mean_dim(1);
        let log_mse: Tensor<B, 1> = log_mse_2d.squeeze_dim(1);

        let [
            latent,
            count_tanimoto_data,
            binary_tanimoto_data,
            log_mse_data,
        ] = Transaction::default()
            .register(output.latent)
            .register(count_tanimoto)
            .register(binary_tanimoto)
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
        let count_tanimoto_values = count_tanimoto_data.as_slice::<f32>().map_err(|source| {
            Error::InvalidBatch(format!("count tanimoto tensor not f32: {source}"))
        })?;
        let binary_tanimoto_values = binary_tanimoto_data.as_slice::<f32>().map_err(|source| {
            Error::InvalidBatch(format!("binary tanimoto tensor not f32: {source}"))
        })?;
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
                        reconstruction_count_tanimoto: count_tanimoto_values[index],
                        reconstruction_binary_tanimoto: binary_tanimoto_values[index],
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
