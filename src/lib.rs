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
pub use data::{
    CidSmilesRecord, DataSplit, MoleculeShardRow, MoleculeTargets, PreprocessingConfig,
    PubChemRecordIter, SHARD_MANIFEST_VERSION, ShardManifest, SparseMoleculeShard,
    parse_pubchem_cid_smiles_line, preprocess_record,
};
#[cfg(feature = "datasets")]
pub use data::{
    DEFAULT_PREPROCESS_CHUNK_ROWS, DEFAULT_PREPROCESS_THREADS, DEFAULT_PUBCHEM_ROWS_PER_SHARD,
    PreprocessedPubChemChunk, PubChemPreprocessOptions, preprocess_cid_smiles_record_chunks,
    pubchem_records_from_path,
};
pub use error::{Error, Result};
pub use features::{DescriptorConfig, DescriptorTargets};
pub use fingerprints::{CountedEcfpConfig, FingerprintTargets, compute_fingerprint_targets};
pub use metrics::{
    CountReconstructionMetrics, batch_count_tanimoto, batch_log_count_reconstruction_tanimoto,
    batch_log_count_tanimoto, batch_sparse_log_count_tanimoto, count_tanimoto,
};
pub use model::{
    AuxiliaryLossWeights, Decoder, DecoderConfig, Encoder, EncoderConfig, MoleculeAutoencoder,
    MoleculeAutoencoderConfig, MoleculeAutoencoderOutput, MoleculeLossBreakdown,
    ReconstructionLossConfig, TanimotoRankingConfig, apply_latent_noise,
    weighted_sparse_log_count_huber_loss,
};
pub use ranking::{
    TanimotoRankingBatch, TanimotoRankingOutput, tanimoto_ranking_output,
    weighted_tanimoto_ranking_output,
};
#[cfg(feature = "train")]
pub use training::{MoleculeAutoencoderTrainingOutput, MoleculeTrainingMetricsExt};
