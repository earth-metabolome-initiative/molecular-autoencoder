//! Library components for molecular representation autoencoders.
//!
//! The crate is library-first: preprocessing, cached batch types, model
//! definitions, losses, and metrics live here, while long-running orchestration
//! belongs in examples.

pub mod batch;
pub mod data;
pub mod error;
pub mod features;
pub mod fingerprints;
pub mod metrics;
pub mod model;
pub mod ranking;
#[cfg(feature = "cuda")]
pub mod tanimoto_cuda;
#[cfg(feature = "train")]
pub mod training;

pub use batch::{
    MoleculeAutoencoderBatch, MoleculeAutoencoderBatcher, MoleculeAutoencoderSample,
    SparseFingerprintBatch,
};
#[cfg(feature = "std")]
pub use batch::{MoleculeAutoencoderHostBatch, MoleculeBatchBuildProfile};
#[cfg(feature = "datasets")]
pub use data::{
    DEFAULT_PREPROCESS_CHUNK_ROWS, DEFAULT_PREPROCESS_THREADS, DEFAULT_ROWS_PER_SHARD,
    DatasetPreprocessOptions, DatasetPreprocessOptionsBuilder, PreprocessedDatasetChunk,
    molecule_records_from_smiles_dataset, preprocess_dataset_record_chunks,
};
pub use data::{
    DataSplit, MoleculeRecord, MoleculeShardRow, MoleculeTargets, PreprocessingConfig,
    PreprocessingConfigBuilder, SHARD_MANIFEST_VERSION, ShardManifest, ShardManifestEntry,
    SparseMoleculeShard, preprocess_record,
};
pub use error::{Error, Result};
pub use features::{
    DescriptorConfig, DescriptorConfigBuilder, DescriptorTargets, SmilesQualityFilter,
    SmilesQualityFilterBuilder,
};
pub use fingerprints::{
    CountedEcfpConfig, CountedEcfpConfigBuilder, FingerprintTargets, compute_fingerprint_targets,
};
pub use metrics::{
    CountReconstructionMetrics, batch_count_tanimoto, batch_log_count_reconstruction_tanimoto,
    batch_log_count_tanimoto, batch_sparse_log_count_tanimoto, count_tanimoto,
};
pub use model::{
    AuxiliaryLossWeights, AuxiliaryLossWeightsBuilder, DEFAULT_DESCRIPTOR_WEIGHT,
    DEFAULT_HIDDEN_WIDTHS, DEFAULT_LATENT_NOISE_STD, DEFAULT_LATENT_WIDTH,
    DEFAULT_RECONSTRUCTION_BETA, DEFAULT_RECONSTRUCTION_NONZERO_WEIGHT,
    DEFAULT_RECONSTRUCTION_ZERO_WEIGHT, DEFAULT_TANIMOTO_RANKING_CANDIDATES_PER_ANCHOR,
    DEFAULT_TANIMOTO_RANKING_LATENT_TEMPERATURE, DEFAULT_TANIMOTO_RANKING_METRIC_TEMPERATURE,
    DEFAULT_TANIMOTO_RANKING_MIN_GAP, DEFAULT_TANIMOTO_RANKING_PAIRS_PER_BATCH,
    DEFAULT_TANIMOTO_RANKING_WEIGHT, Decoder, DecoderConfig, DecoderConfigBuilder, Encoder,
    EncoderConfig, EncoderConfigBuilder, MoleculeAutoencoder, MoleculeAutoencoderConfig,
    MoleculeAutoencoderConfigBuilder, MoleculeAutoencoderOutput, MoleculeLossBreakdown,
    ReconstructionLossConfig, ReconstructionLossConfigBuilder, TanimotoRankingConfig,
    TanimotoRankingConfigBuilder, TanimotoRankingRuntimeConfig, apply_latent_noise,
    weighted_sparse_log_count_huber_loss,
};
pub use ranking::{
    TanimotoRankingBatch, TanimotoRankingOutput, tanimoto_ranking_output,
    weighted_tanimoto_ranking_output,
};
#[cfg(feature = "train")]
pub use training::{MoleculeAutoencoderTrainingOutput, MoleculeTrainingMetricsExt};
