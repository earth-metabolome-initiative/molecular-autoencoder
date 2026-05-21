//! Dataset SMILES preprocessing and tensor-ready sparse shard IO.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use smiles_parser::prelude::Smiles;

use crate::{
    CountedEcfpConfig, DescriptorConfig, DescriptorTargets, Error, FingerprintTargets, Result,
    SmilesQualityFilter, compute_fingerprint_targets,
};

/// Magic bytes stamped at the start of every cached sparse shard file.
///
/// `MAESH03` is the post-`v2 → v3` bump introduced when negative-molecule
/// rows joined the shard schema: each row now carries a violation-class
/// bitset and, for negatives, a second (CSR-encoded) decoder-target
/// fingerprint. Older `MAESH02` shards must be re-preprocessed —
/// [`SparseMoleculeShard::read_from_path`] surfaces a clear "re-preprocess
/// required" error when it encounters the legacy magic.
const SHARD_MAGIC: [u8; 8] = *b"MAESH03\0";
/// Previous shard magic kept around so the reader can emit a precise
/// "please re-preprocess" error when it encounters a `v2` file.
const LEGACY_SHARD_MAGIC_V2: [u8; 8] = *b"MAESH02\0";
/// FNV-1a 64-bit offset basis used to derive stable molecule IDs from
/// dataset + record string tuples.
const STABLE_ID_FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime used alongside [`STABLE_ID_FNV_OFFSET`].
const STABLE_ID_FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Top-2-bit namespace tag (`0b01`) marking hash-derived stable IDs for
/// datasets without a natively numeric record identifier.
const HASH_STABLE_ID_NAMESPACE: u64 = 0x4000_0000_0000_0000;
/// Bitmask isolating the FNV hash bits within a hash-namespaced ID.
const HASH_STABLE_ID_MASK: u64 = 0x3fff_ffff_ffff_ffff;
/// Top-bit namespace tag (`0b10`) marking ZINC20-derived stable IDs.
const ZINC20_STABLE_ID_NAMESPACE: u64 = 0x8000_0000_0000_0000;
/// Bit shift applied to the ZINC20 tranche suffix within a stable ID.
const ZINC20_STABLE_ID_SUFFIX_SHIFT: u32 = 48;
/// Bitmask isolating the ZINC20 numeric part of a stable ID.
const ZINC20_STABLE_ID_NUMBER_MASK: u64 = 0x0000_ffff_ffff_ffff;
/// Bitmask isolating the 15-bit ZINC20 tranche suffix.
const ZINC20_STABLE_ID_SUFFIX_MASK: u64 = 0x7fff;
/// Current cached-shard manifest schema version.
pub const SHARD_MANIFEST_VERSION: u32 = 3;
/// Default number of dataset records preprocessed per Rayon chunk.
#[cfg(feature = "datasets")]
pub const DEFAULT_PREPROCESS_CHUNK_ROWS: usize = 8192;
/// Default Rayon workers for dataset preprocessing.
#[cfg(feature = "datasets")]
pub const DEFAULT_PREPROCESS_THREADS: usize = 64;
/// Default number of preprocessed rows written per sparse shard.
#[cfg(feature = "datasets")]
pub const DEFAULT_ROWS_PER_SHARD: usize = 10_000_000;

/// One SMILES record from a supported dataset. Construct via
/// [`MoleculeRecord::new`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoleculeRecord {
    /// Source dataset label (e.g. `"pubchem-smiles"`, `"zinc20-smiles"`).
    dataset_id: String,
    /// Dataset-specific identifier exactly as the source supplied it.
    record_id: String,
    /// Deterministic 64-bit ID derived from `dataset_id` + `record_id`
    /// used by shard metadata and the train/validation split function.
    stable_id: u64,
    /// Source SMILES text.
    smiles: String,
}

impl MoleculeRecord {
    /// Creates a dataset molecule record with a deterministic numeric id.
    #[must_use]
    pub fn new(
        dataset_id: impl Into<String>,
        record_id: impl Into<String>,
        smiles: impl Into<String>,
    ) -> Self {
        let dataset_id = dataset_id.into();
        let record_id = record_id.into();
        let stable_id = stable_molecule_id(&dataset_id, &record_id);
        Self {
            dataset_id,
            record_id,
            stable_id,
            smiles: smiles.into(),
        }
    }

    /// Stable source dataset identifier.
    #[must_use]
    pub fn dataset_id(&self) -> &str {
        &self.dataset_id
    }

    /// Dataset-specific record identifier.
    #[must_use]
    pub fn record_id(&self) -> &str {
        &self.record_id
    }

    /// Stable numeric molecule identifier used by shard metadata and splits.
    #[must_use]
    pub const fn stable_id(&self) -> u64 {
        self.stable_id
    }

    /// Source SMILES text.
    #[must_use]
    pub fn smiles(&self) -> &str {
        &self.smiles
    }

    /// Converts a `smiles-parser` dataset record into a preprocessing record.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DatasetRecord`] when the source-specific identifier is
    /// malformed.
    #[cfg(feature = "datasets")]
    pub fn from_dataset_smiles_record(
        dataset_id: &'static str,
        record: smiles_parser::prelude::DatasetSmilesRecord,
    ) -> Result<Self> {
        if dataset_id == "pubchem-smiles"
            && let Err(source) = record.id().parse::<u64>()
        {
            return Err(Error::DatasetRecord {
                message: format!("invalid PubChem CID `{}`: {source}", record.id()),
            });
        }
        if dataset_id == "zinc20-smiles" && zinc20_stable_molecule_id(record.id()).is_none() {
            return Err(Error::DatasetRecord {
                message: format!("invalid ZINC20 identifier `{}`", record.id()),
            });
        }
        Ok(Self::new(dataset_id, record.id(), record.smiles()))
    }
}

/// Maps `smiles-parser` dataset records into molecule preprocessing records.
#[cfg(feature = "datasets")]
pub fn molecule_records_from_smiles_dataset<I>(
    dataset_id: &'static str,
    records: I,
) -> impl Iterator<Item = Result<MoleculeRecord>>
where
    I: IntoIterator<
        Item = std::result::Result<
            smiles_parser::prelude::DatasetSmilesRecord,
            smiles_parser::prelude::DatasetError,
        >,
    >,
{
    records.into_iter().map(move |record| {
        let record = record.map_err(|source| Error::DatasetRecord {
            message: source.to_string(),
        })?;
        MoleculeRecord::from_dataset_smiles_record(dataset_id, record)
    })
}

/// Parallel dataset preprocessing controls. Construct via
/// [`DatasetPreprocessOptions::builder`].
#[cfg(feature = "datasets")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetPreprocessOptions {
    /// Number of dataset records read before each Rayon chunk dispatch.
    chunk_rows: usize,
    /// Optional thread count for the per-chunk Rayon pool; `None` uses
    /// the global pool.
    threads: Option<usize>,
}

#[cfg(feature = "datasets")]
impl Default for DatasetPreprocessOptions {
    fn default() -> Self {
        DatasetPreprocessOptionsBuilder::new()
            .build()
            .expect("default dataset preprocess options are valid")
    }
}

#[cfg(feature = "datasets")]
impl DatasetPreprocessOptions {
    /// Starts a fluent builder.
    #[must_use]
    pub fn builder() -> DatasetPreprocessOptionsBuilder {
        DatasetPreprocessOptionsBuilder::new()
    }

    /// Number of records read before dispatching work to Rayon.
    #[must_use]
    pub const fn chunk_rows(&self) -> usize {
        self.chunk_rows
    }

    /// Optional local Rayon thread count (`None` uses the global pool).
    #[must_use]
    pub const fn threads(&self) -> Option<usize> {
        self.threads
    }
}

/// Fluent builder for [`DatasetPreprocessOptions`].
#[cfg(feature = "datasets")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetPreprocessOptionsBuilder {
    /// Pending per-chunk row count.
    chunk_rows: usize,
    /// Pending Rayon thread count.
    threads: Option<usize>,
}

#[cfg(feature = "datasets")]
impl Default for DatasetPreprocessOptionsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "datasets")]
impl DatasetPreprocessOptionsBuilder {
    /// Creates a builder seeded with the v1 defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            chunk_rows: DEFAULT_PREPROCESS_CHUNK_ROWS,
            threads: Some(DEFAULT_PREPROCESS_THREADS),
        }
    }

    /// Sets the per-chunk row count.
    #[must_use]
    pub const fn chunk_rows(mut self, value: usize) -> Self {
        self.chunk_rows = value;
        self
    }

    /// Sets the local Rayon thread count; `None` uses the global pool.
    #[must_use]
    pub const fn threads(mut self, value: Option<usize>) -> Self {
        self.threads = value;
        self
    }

    /// Validates and builds the immutable options.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when `chunk_rows` is zero or when
    /// `threads` is `Some(0)`.
    pub fn build(self) -> Result<DatasetPreprocessOptions> {
        if self.chunk_rows == 0 {
            return Err(Error::ConfigInvalid {
                message: "preprocess chunk_rows must be greater than zero".to_string(),
            });
        }
        if matches!(self.threads, Some(0)) {
            return Err(Error::ConfigInvalid {
                message: "preprocess threads must be greater than zero when set".to_string(),
            });
        }
        Ok(DatasetPreprocessOptions {
            chunk_rows: self.chunk_rows,
            threads: self.threads,
        })
    }
}

/// Ordered result of one parallel dataset preprocessing chunk.
///
/// Constructed by [`preprocess_dataset_record_chunks`] and consumed via the
/// [`records_seen`](Self::records_seen) / [`into_results`](Self::into_results)
/// accessors.
#[cfg(feature = "datasets")]
#[derive(Debug)]
pub struct PreprocessedDatasetChunk {
    /// Cumulative count of source dataset records consumed up to this
    /// chunk.
    records_seen: usize,
    /// Per-record preprocessing outcome: an `Ok(Vec)` (empty Vec means
    /// the record was rejected by the quality filter, length 1 means a
    /// positive with no negatives, length > 1 means a positive plus
    /// negatives), or an `Err(_)` parse / RDKit failure.
    results: Vec<Result<Vec<MoleculeTargets>>>,
}

#[cfg(feature = "datasets")]
impl PreprocessedDatasetChunk {
    /// Total records read from the source after this chunk.
    #[must_use]
    pub const fn records_seen(&self) -> usize {
        self.records_seen
    }

    /// Per-record preprocessing results, preserving source order.
    ///
    /// `Ok(rows)` with `rows.is_empty()` is a record skipped by the
    /// configured [`SmilesQualityFilter`]; `Ok(rows)` with `!rows.is_empty()`
    /// is the positive row followed by any negative replicas; `Err(_)`
    /// is a parse or fingerprint failure.
    pub fn results(&self) -> &[Result<Vec<MoleculeTargets>>] {
        &self.results
    }

    /// Consumes the chunk, returning its results.
    #[must_use]
    pub fn into_results(self) -> Vec<Result<Vec<MoleculeTargets>>> {
        self.results
    }
}

/// Reads dataset records sequentially, then preprocesses each chunk in parallel.
///
/// # Errors
///
/// Returns an error when the preprocessing options are invalid, the optional
/// Rayon pool cannot be built, an input record is malformed, or the consumer
/// callback returns an error.
#[cfg(feature = "datasets")]
pub fn preprocess_dataset_record_chunks<I, F>(
    records: I,
    config: &PreprocessingConfig,
    options: DatasetPreprocessOptions,
    mut consume: F,
) -> Result<usize>
where
    I: IntoIterator<Item = Result<MoleculeRecord>>,
    F: FnMut(PreprocessedDatasetChunk) -> Result<()>,
{
    let mut records = records.into_iter();
    let mut records_seen = 0_usize;
    let pool = match options.threads {
        Some(threads) => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|source| Error::InvalidBatch(format!("invalid Rayon pool: {source}")))?,
        ),
        None => None,
    };

    loop {
        let mut chunk = Vec::with_capacity(options.chunk_rows);
        for _ in 0..options.chunk_rows {
            let Some(record) = records.next() else {
                break;
            };
            chunk.push(record);
        }
        if chunk.is_empty() {
            return Ok(records_seen);
        }

        records_seen += chunk.len();
        let results = match &pool {
            Some(pool) => pool.install(|| preprocess_record_chunk_parallel(chunk, config)),
            None => preprocess_record_chunk_parallel(chunk, config),
        };
        consume(PreprocessedDatasetChunk {
            records_seen,
            results,
        })?;
    }
}

/// Preprocesses a chunk of molecule records in parallel via Rayon.
/// Each worker thread owns its own `SmilesRdkitScratch` so the
/// fingerprint allocations are reused across records.
#[cfg(feature = "datasets")]
fn preprocess_record_chunk_parallel(
    chunk: Vec<Result<MoleculeRecord>>,
    config: &PreprocessingConfig,
) -> Vec<Result<Vec<MoleculeTargets>>> {
    use rayon::prelude::*;

    chunk
        .into_par_iter()
        .map_init(
            finge_rs::smiles_support::SmilesRdkitScratch::default,
            |scratch, record| record.and_then(|record| preprocess_record(&record, config, scratch)),
        )
        .collect()
}

/// Deterministic split assigned during preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSplit {
    /// Training row.
    Train,
    /// Validation row.
    Validation,
}

impl DataSplit {
    /// Serialises the split as a single byte for on-disk shard storage.
    const fn as_byte(self) -> u8 {
        match self {
            Self::Train => 0,
            Self::Validation => 1,
        }
    }

    /// Parses a split byte from a shard payload. Returns `None` for any
    /// unrecognised tag so callers can flag shard corruption.
    const fn from_byte(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Train),
            1 => Some(Self::Validation),
            _ => None,
        }
    }
}

/// Default number of negatives emitted per positive when the preprocessor
/// is asked to bake violation-classified mutants into the shard.
pub const DEFAULT_NEGATIVES_PER_POSITIVE: f32 = 1.0;

/// Deterministic preprocessing configuration. Construct via
/// [`PreprocessingConfig::builder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Counted-ECFP fingerprint configuration (size + radius).
    counted_ecfp: CountedEcfpConfig,
    /// Descriptor normalisation scales applied to the regression head.
    descriptors: DescriptorConfig,
    /// Validation split fraction in permille (0..=1000).
    validation_per_mille: u16,
    /// Quality bounds applied to candidate SMILES before fingerprinting.
    #[serde(default)]
    quality_filter: SmilesQualityFilter,
    /// Average number of `MutatorMix`-generated negative rows emitted per
    /// positive. Fractional values turn into per-record Bernoulli draws
    /// using a `splitmix64`-derived seed. `0.0` disables negatives.
    #[serde(default = "default_negatives_per_positive")]
    negatives_per_positive: f32,
}

/// Serde fallback for the negatives-per-positive ratio — preserves
/// backward-compatible deserialization of `PreprocessingConfig` JSON
/// emitted before this field existed.
const fn default_negatives_per_positive() -> f32 {
    DEFAULT_NEGATIVES_PER_POSITIVE
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        PreprocessingConfigBuilder::new()
            .build()
            .expect("default preprocessing config is valid")
    }
}

impl PreprocessingConfig {
    /// Starts a fluent builder.
    #[must_use]
    pub fn builder() -> PreprocessingConfigBuilder {
        PreprocessingConfigBuilder::new()
    }

    /// Main counted ECFP configuration.
    #[must_use]
    pub const fn counted_ecfp(&self) -> &CountedEcfpConfig {
        &self.counted_ecfp
    }

    /// Descriptor normalization configuration.
    #[must_use]
    pub const fn descriptors(&self) -> &DescriptorConfig {
        &self.descriptors
    }

    /// Validation split in permille units.
    #[must_use]
    pub const fn validation_per_mille(&self) -> u16 {
        self.validation_per_mille
    }

    /// SMILES quality filter applied before fingerprinting.
    #[must_use]
    pub const fn quality_filter(&self) -> &SmilesQualityFilter {
        &self.quality_filter
    }

    /// Average number of negatives emitted per positive (0.0 disables).
    #[must_use]
    pub const fn negatives_per_positive(&self) -> f32 {
        self.negatives_per_positive
    }

    /// Resolves the integer number of negative rows the preprocessor should
    /// emit for a given source molecule. The integer part of
    /// [`Self::negatives_per_positive`] is emitted deterministically;
    /// any fractional remainder turns into a per-record Bernoulli draw
    /// seeded from `stable_id` via a `splitmix64` mix, so two preprocess
    /// runs over the same dataset produce identical row counts.
    #[must_use]
    pub fn negative_count_for_stable_id(&self, stable_id: u64) -> usize {
        let ratio = self.negatives_per_positive.max(0.0);
        // `trunc` is already non-negative here so the cast lane is safe;
        // saturate at `u32::MAX` to keep the conversion well-defined on
        // any (pathological) input that survives the builder validation.
        let base_raw = ratio.trunc().min(u32::MAX as f32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let base = base_raw as u32 as usize;
        let fractional = ratio - ratio.trunc();
        if fractional <= 0.0 {
            return base;
        }
        let draw = (splitmix64(stable_id.wrapping_add(0xA5A5_A5A5_5A5A_5A5A)) % 1_000_000) as f32
            / 1_000_000.0;
        if draw < fractional { base + 1 } else { base }
    }

    /// Assigns a deterministic split from a stable molecule id.
    #[must_use]
    pub fn split_for_stable_id(&self, stable_id: u64) -> DataSplit {
        let validation_per_mille = u64::from(self.validation_per_mille.min(1000));
        if splitmix64(stable_id) % 1000 < validation_per_mille {
            DataSplit::Validation
        } else {
            DataSplit::Train
        }
    }
}

/// Fluent builder for [`PreprocessingConfig`].
#[derive(Debug, Clone)]
pub struct PreprocessingConfigBuilder {
    /// Pending counted-ECFP fingerprint configuration.
    counted_ecfp: CountedEcfpConfig,
    /// Pending descriptor configuration.
    descriptors: DescriptorConfig,
    /// Pending validation split fraction in permille.
    validation_per_mille: u16,
    /// Pending SMILES quality filter.
    quality_filter: SmilesQualityFilter,
    /// Pending average number of negatives emitted per positive.
    negatives_per_positive: f32,
}

impl Default for PreprocessingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PreprocessingConfigBuilder {
    /// Creates a builder seeded with the v1 defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counted_ecfp: CountedEcfpConfig::default(),
            descriptors: DescriptorConfig::default(),
            validation_per_mille: 100,
            quality_filter: SmilesQualityFilter::default(),
            negatives_per_positive: DEFAULT_NEGATIVES_PER_POSITIVE,
        }
    }

    /// Sets the counted ECFP configuration.
    #[must_use]
    pub const fn counted_ecfp(mut self, value: CountedEcfpConfig) -> Self {
        self.counted_ecfp = value;
        self
    }

    /// Sets the descriptor normalization configuration.
    #[must_use]
    pub const fn descriptors(mut self, value: DescriptorConfig) -> Self {
        self.descriptors = value;
        self
    }

    /// Sets the validation split as a permille fraction.
    #[must_use]
    pub const fn validation_per_mille(mut self, value: u16) -> Self {
        self.validation_per_mille = value;
        self
    }

    /// Sets the SMILES quality filter.
    #[must_use]
    pub const fn quality_filter(mut self, value: SmilesQualityFilter) -> Self {
        self.quality_filter = value;
        self
    }

    /// Sets the average number of negatives emitted per positive.
    /// Values < 0 are clamped to 0 at build time.
    #[must_use]
    pub const fn negatives_per_positive(mut self, value: f32) -> Self {
        self.negatives_per_positive = value;
        self
    }

    /// Validates and builds the immutable [`PreprocessingConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when `validation_per_mille` exceeds
    /// 1000 or when `negatives_per_positive` is negative, NaN, or
    /// infinite.
    pub fn build(self) -> Result<PreprocessingConfig> {
        if self.validation_per_mille > 1000 {
            return Err(Error::ConfigInvalid {
                message: format!(
                    "validation_per_mille must be in 0..=1000, got {}",
                    self.validation_per_mille
                ),
            });
        }
        if !self.negatives_per_positive.is_finite() || self.negatives_per_positive < 0.0 {
            return Err(Error::ConfigInvalid {
                message: format!(
                    "negatives_per_positive must be a finite non-negative float, got {}",
                    self.negatives_per_positive
                ),
            });
        }
        Ok(PreprocessingConfig {
            counted_ecfp: self.counted_ecfp,
            descriptors: self.descriptors,
            validation_per_mille: self.validation_per_mille,
            quality_filter: self.quality_filter,
            negatives_per_positive: self.negatives_per_positive,
        })
    }
}

/// Parsed, fingerprinted, descriptor-rich molecule target.
///
/// Constructed by [`preprocess_record`]; consumed via the accessor methods
/// below.
///
/// A row is a *positive* when `violation_label == 0` (and then
/// `target_fingerprint == fingerprint`), or a *negative* when
/// `violation_label != 0` (the bitset of fired
/// [`finge_rs::ViolationClass`] predicates) — in which case
/// `fingerprint` is the mutated input the encoder sees and
/// `target_fingerprint` is the original positive's fingerprint the
/// decoder is trained to reconstruct (denoising / correction framing).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoleculeTargets {
    /// Stable numeric molecule identifier (the FNV / ZINC20-packed ID).
    cid: u64,
    /// Original SMILES string for traceability.
    source_smiles: String,
    /// Counted-ECFP fingerprint vector (encoder input).
    fingerprint: FingerprintTargets,
    /// Decoder reconstruction target. Equal to [`Self::fingerprint`] for
    /// positives; equal to the source positive's fingerprint for
    /// negatives (denoising framing).
    #[serde(default)]
    target_fingerprint: Option<FingerprintTargets>,
    /// Scalar descriptor regression targets.
    descriptors: DescriptorTargets,
    /// Deterministic train/validation split assignment.
    split: DataSplit,
    /// Multi-label violation bitset. `0` for positives.
    #[serde(default)]
    violation_label: u8,
}

impl MoleculeTargets {
    /// Stable numeric molecule identifier.
    #[must_use]
    pub const fn cid(&self) -> u64 {
        self.cid
    }

    /// Source SMILES text.
    #[must_use]
    pub fn source_smiles(&self) -> &str {
        &self.source_smiles
    }

    /// Sparse counted ECFP target (the encoder input).
    #[must_use]
    pub const fn fingerprint(&self) -> &FingerprintTargets {
        &self.fingerprint
    }

    /// Decoder reconstruction target. Returns the input fingerprint for
    /// positive rows; returns the source-positive fingerprint for
    /// negative rows.
    #[must_use]
    pub fn target_fingerprint(&self) -> &FingerprintTargets {
        self.target_fingerprint
            .as_ref()
            .unwrap_or(&self.fingerprint)
    }

    /// Returns the explicit target fingerprint if this row carries one
    /// (i.e. is a negative). Positive rows return `None` so callers can
    /// alias the decoder target to the encoder input in shard storage.
    #[must_use]
    pub const fn explicit_target_fingerprint(&self) -> Option<&FingerprintTargets> {
        self.target_fingerprint.as_ref()
    }

    /// Molecule descriptor targets.
    #[must_use]
    pub const fn descriptors(&self) -> &DescriptorTargets {
        &self.descriptors
    }

    /// Deterministic data split.
    #[must_use]
    pub const fn split(&self) -> DataSplit {
        self.split
    }

    /// Violation-class bitset. `0` for positives.
    #[must_use]
    pub const fn violation_label(&self) -> u8 {
        self.violation_label
    }

    /// Convenience: `true` when [`Self::violation_label`] is non-zero.
    #[must_use]
    pub const fn is_negative(&self) -> bool {
        self.violation_label != 0
    }
}

/// Parses and fingerprints one dataset record.
///
/// Applies the configured quality filter before doing fingerprint work,
/// then emits one positive row plus zero or more negative rows derived
/// from `MutatorMix` (per
/// [`PreprocessingConfig::negatives_per_positive`]).
///
/// Returns an empty `Vec` when the record is rejected by the quality
/// filter; callers should treat that as a silent skip. The first
/// element of a non-empty result is always the positive; subsequent
/// elements are negatives with `violation_label != 0` and
/// `target_fingerprint == positive.fingerprint`.
///
/// # Errors
///
/// Returns an error when the SMILES cannot be parsed or fingerprint
/// preparation fails for the positive. Negative-side `MutatorError`s
/// are absorbed into a silent skip (the negative slot is dropped, the
/// positive emits normally).
pub fn preprocess_record(
    record: &MoleculeRecord,
    config: &PreprocessingConfig,
    scratch: &mut finge_rs::smiles_support::SmilesRdkitScratch,
) -> Result<Vec<MoleculeTargets>> {
    let smiles = record
        .smiles
        .parse::<Smiles>()
        .map_err(|source| Error::SmilesParse {
            molecule_id: record.stable_id,
            message: format!("{}:{}: {source}", record.dataset_id, record.record_id),
        })?;
    let descriptors = DescriptorTargets::from_smiles(&smiles);
    if !config.quality_filter.accepts(&descriptors) {
        return Ok(Vec::new());
    }
    let fingerprint =
        compute_fingerprint_targets(record.stable_id, &smiles, config.counted_ecfp, scratch)?;
    let split = config.split_for_stable_id(record.stable_id);

    let negative_count = config.negative_count_for_stable_id(record.stable_id);
    let mut out = Vec::with_capacity(1 + negative_count);
    out.push(MoleculeTargets {
        cid: record.stable_id,
        source_smiles: record.smiles.clone(),
        fingerprint: fingerprint.clone(),
        target_fingerprint: None,
        descriptors: descriptors.clone(),
        split,
        violation_label: 0,
    });

    for replica in 0..negative_count {
        // Seed each negative deterministically from
        // (stable_id, replica). Two preprocess runs over the same
        // dataset produce identical negative rows.
        let seed = splitmix64(
            record
                .stable_id
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(replica as u64),
        );
        match crate::fingerprints::compute_negative_fingerprint_targets(
            record.stable_id,
            &smiles,
            seed,
            config.counted_ecfp,
            scratch,
        ) {
            Ok(Some((neg_fingerprint, label))) => {
                out.push(MoleculeTargets {
                    cid: record.stable_id,
                    source_smiles: record.smiles.clone(),
                    fingerprint: neg_fingerprint,
                    target_fingerprint: Some(fingerprint.clone()),
                    descriptors: descriptors.clone(),
                    split,
                    violation_label: label.as_u8(),
                });
            }
            Ok(None) => {
                // MutatorMix gave up on this seed; skip silently and
                // let the realized negative ratio dip below the
                // configured target.
            }
            Err(error) => {
                return Err(error);
            }
        }
    }
    Ok(out)
}

/// One row view into a sparse molecule shard.
#[derive(Debug, Clone, Copy)]
pub struct MoleculeShardRow<'a> {
    /// Stable molecule identifier for this row.
    cid: u64,
    /// Sparse fingerprint indices for this row (no padding).
    fingerprint_indices: &'a [u16],
    /// Sparse fingerprint counts aligned with `fingerprint_indices`.
    fingerprint_counts: &'a [u16],
    /// Scalar descriptor regression targets for this row.
    descriptor_targets: &'a [f32],
    /// Train / validation split assignment for this row.
    split: DataSplit,
    /// [`finge_rs::ViolationClass`] bitset for this row (`0` = positive).
    violation_label: u8,
    /// Decoder-target sparse fingerprint indices when this row stores
    /// an explicit target (i.e. a negative). Empty for positives — in
    /// that case the decoder target aliases the encoder input.
    target_fingerprint_indices: &'a [u16],
    /// Decoder-target sparse fingerprint counts, aligned with
    /// [`Self::target_fingerprint_indices`].
    target_fingerprint_counts: &'a [u16],
}

impl<'a> MoleculeShardRow<'a> {
    /// Stable numeric molecule identifier.
    #[must_use]
    pub const fn cid(&self) -> u64 {
        self.cid
    }

    /// Sparse fingerprint indices.
    #[must_use]
    pub const fn fingerprint_indices(&self) -> &'a [u16] {
        self.fingerprint_indices
    }

    /// Sparse fingerprint counts.
    #[must_use]
    pub const fn fingerprint_counts(&self) -> &'a [u16] {
        self.fingerprint_counts
    }

    /// Normalized descriptor regression targets.
    #[must_use]
    pub const fn descriptor_targets(&self) -> &'a [f32] {
        self.descriptor_targets
    }

    /// Deterministic data split.
    #[must_use]
    pub const fn split(&self) -> DataSplit {
        self.split
    }

    /// Multi-label violation bitset (`0` for positives).
    #[must_use]
    pub const fn violation_label(&self) -> u8 {
        self.violation_label
    }

    /// `true` when this row is a mutated negative (label bitset != 0).
    #[must_use]
    pub const fn is_negative(&self) -> bool {
        self.violation_label != 0
    }

    /// Decoder-target sparse fingerprint indices. For positive rows the
    /// shard stores no explicit target and this accessor aliases to
    /// [`Self::fingerprint_indices`].
    #[must_use]
    pub const fn target_fingerprint_indices(&self) -> &'a [u16] {
        if self.target_fingerprint_indices.is_empty() && self.violation_label == 0 {
            self.fingerprint_indices
        } else {
            self.target_fingerprint_indices
        }
    }

    /// Decoder-target sparse fingerprint counts. Aliases the encoder
    /// input on positive rows; mirrors the explicit decoder target on
    /// negative rows.
    #[must_use]
    pub const fn target_fingerprint_counts(&self) -> &'a [u16] {
        if self.target_fingerprint_counts.is_empty() && self.violation_label == 0 {
            self.fingerprint_counts
        } else {
            self.target_fingerprint_counts
        }
    }
}

/// Tensor-ready sparse molecule shard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseMoleculeShard {
    /// Dense fingerprint width (bit count) used to recover dense tensors.
    fingerprint_size: usize,
    /// Width of the descriptor regression target row.
    descriptor_width: usize,
    /// Per-row stable molecule IDs, length `len()`.
    cids: Vec<u64>,
    /// CSR-style row offsets into `indices` / `counts`, length `len() + 1`.
    row_offsets: Vec<u64>,
    /// Concatenated sparse fingerprint indices (encoder input).
    indices: Vec<u16>,
    /// Concatenated sparse fingerprint counts (aligned with `indices`).
    counts: Vec<u16>,
    /// Dense descriptor targets concatenated `row_count * descriptor_width`.
    descriptor_targets: Vec<f32>,
    /// Per-row train / validation split assignments.
    splits: Vec<DataSplit>,
    /// Per-row [`finge_rs::ViolationClass`] bitset. `0` for positives.
    /// Length matches [`Self::len`].
    #[serde(default)]
    violation_labels: Vec<u8>,
    /// CSR-style row offsets into `target_indices` / `target_counts`.
    /// Length `len() + 1`. Positive rows produce an empty slice (the
    /// reader aliases the decoder target to the encoder input), so the
    /// offset increment is 0 for positives and equals the source
    /// positive's `nnz` for negatives.
    #[serde(default)]
    target_row_offsets: Vec<u64>,
    /// Concatenated sparse fingerprint indices (decoder target).
    /// Empty for shards with no negative rows.
    #[serde(default)]
    target_indices: Vec<u16>,
    /// Concatenated sparse fingerprint counts (decoder target, aligned
    /// with `target_indices`).
    #[serde(default)]
    target_counts: Vec<u16>,
}

impl SparseMoleculeShard {
    /// Creates an empty shard.
    #[must_use]
    pub fn new(fingerprint_size: usize, descriptor_width: usize) -> Self {
        Self {
            fingerprint_size,
            descriptor_width,
            cids: Vec::new(),
            row_offsets: vec![0],
            indices: Vec::new(),
            counts: Vec::new(),
            descriptor_targets: Vec::new(),
            splits: Vec::new(),
            violation_labels: Vec::new(),
            target_row_offsets: vec![0],
            target_indices: Vec::new(),
            target_counts: Vec::new(),
        }
    }

    /// Full dense fingerprint width.
    #[must_use]
    pub const fn fingerprint_size(&self) -> usize {
        self.fingerprint_size
    }

    /// Descriptor regression target width.
    #[must_use]
    pub const fn descriptor_width(&self) -> usize {
        self.descriptor_width
    }

    /// Stable numeric molecule identifiers, one per row.
    #[must_use]
    pub fn cids(&self) -> &[u64] {
        &self.cids
    }

    /// Sparse row offsets, length `row_count + 1`.
    #[must_use]
    pub fn row_offsets(&self) -> &[u64] {
        &self.row_offsets
    }

    /// Concatenated sparse fingerprint indices.
    #[must_use]
    pub fn indices(&self) -> &[u16] {
        &self.indices
    }

    /// Concatenated sparse fingerprint counts.
    #[must_use]
    pub fn counts(&self) -> &[u16] {
        &self.counts
    }

    /// Flattened descriptor targets with shape `[rows, descriptor_width]`.
    #[must_use]
    pub fn descriptor_targets(&self) -> &[f32] {
        &self.descriptor_targets
    }

    /// Deterministic split, one per row.
    #[must_use]
    pub fn splits(&self) -> &[DataSplit] {
        &self.splits
    }

    /// Per-row [`finge_rs::ViolationClass`] bitset. `0` indicates the row
    /// is a positive (decoder target aliases the encoder input).
    #[must_use]
    pub fn violation_labels(&self) -> &[u8] {
        &self.violation_labels
    }

    /// CSR-style row offsets into the decoder-target sparse storage.
    /// Length `len() + 1`. Positive rows produce a zero-width slice; the
    /// reader aliases their decoder target to the encoder input.
    #[must_use]
    pub fn target_row_offsets(&self) -> &[u64] {
        &self.target_row_offsets
    }

    /// Concatenated decoder-target sparse fingerprint indices.
    #[must_use]
    pub fn target_indices(&self) -> &[u16] {
        &self.target_indices
    }

    /// Concatenated decoder-target sparse fingerprint counts.
    #[must_use]
    pub fn target_counts(&self) -> &[u16] {
        &self.target_counts
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cids.len()
    }

    /// Returns whether the shard has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cids.is_empty()
    }

    /// Appends one preprocessed molecule.
    ///
    /// The `violation_label` parameter is the multi-label bitset for this
    /// row (`0` for positives). `target_fingerprint`, when provided,
    /// stores the decoder reconstruction target separately — required
    /// when the row is a negative (denoising / correction framing).
    /// When `None`, the decoder target aliases the encoder input.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBatch`] when the fingerprint width, descriptor
    /// width, sparse index/count shape, or target fingerprint width does
    /// not match the shard.
    pub fn push(
        &mut self,
        cid: u64,
        fingerprint: &FingerprintTargets,
        descriptor_targets: &[f32],
        split: DataSplit,
        violation_label: u8,
        target_fingerprint: Option<&FingerprintTargets>,
    ) -> Result<()> {
        if fingerprint.fingerprint_size() != self.fingerprint_size {
            return Err(Error::InvalidBatch(format!(
                "fingerprint width {} does not match shard width {}",
                fingerprint.fingerprint_size(),
                self.fingerprint_size
            )));
        }
        if descriptor_targets.len() != self.descriptor_width {
            return Err(Error::InvalidBatch(format!(
                "descriptor width {} does not match shard width {}",
                descriptor_targets.len(),
                self.descriptor_width
            )));
        }
        if let Some(target) = target_fingerprint
            && target.fingerprint_size() != self.fingerprint_size
        {
            return Err(Error::InvalidBatch(format!(
                "target fingerprint width {} does not match shard width {}",
                target.fingerprint_size(),
                self.fingerprint_size
            )));
        }
        self.cids.push(cid);
        self.indices.extend_from_slice(fingerprint.indices());
        self.counts.extend_from_slice(fingerprint.counts());
        self.descriptor_targets
            .extend_from_slice(descriptor_targets);
        self.splits.push(split);
        self.row_offsets.push(self.indices.len() as u64);
        self.violation_labels.push(violation_label);
        if let Some(target) = target_fingerprint {
            self.target_indices.extend_from_slice(target.indices());
            self.target_counts.extend_from_slice(target.counts());
        }
        self.target_row_offsets
            .push(self.target_indices.len() as u64);
        Ok(())
    }

    /// Appends one [`MoleculeTargets`] value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBatch`] when the targets are incompatible with
    /// the shard layout.
    pub fn push_targets(
        &mut self,
        targets: &MoleculeTargets,
        config: DescriptorConfig,
    ) -> Result<()> {
        self.push(
            targets.cid,
            &targets.fingerprint,
            &targets.descriptors.regression_targets(config),
            targets.split,
            targets.violation_label,
            targets.explicit_target_fingerprint(),
        )
    }

    /// Returns a row view.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<MoleculeShardRow<'_>> {
        if row >= self.len() {
            return None;
        }
        let start = self.row_offsets[row] as usize;
        let end = self.row_offsets[row + 1] as usize;
        let descriptor_start = row * self.descriptor_width;
        let descriptor_end = descriptor_start + self.descriptor_width;
        let target_start = self.target_row_offsets[row] as usize;
        let target_end = self.target_row_offsets[row + 1] as usize;
        Some(MoleculeShardRow {
            cid: self.cids[row],
            fingerprint_indices: &self.indices[start..end],
            fingerprint_counts: &self.counts[start..end],
            descriptor_targets: &self.descriptor_targets[descriptor_start..descriptor_end],
            split: self.splits[row],
            violation_label: self.violation_labels[row],
            target_fingerprint_indices: &self.target_indices[target_start..target_end],
            target_fingerprint_counts: &self.target_counts[target_start..target_end],
        })
    }

    /// Writes the shard using a simple little-endian sequential binary format.
    ///
    /// # Errors
    ///
    /// Returns an error when the shard is internally inconsistent or the file
    /// cannot be written.
    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        self.validate_for_path(path)?;
        let file = File::create(path).map_err(|source| Error::io(path, source))?;
        let mut writer = BufWriter::new(file);

        write_all(path, &mut writer, &SHARD_MAGIC)?;
        write_u64(path, &mut writer, self.fingerprint_size as u64)?;
        write_u64(path, &mut writer, self.descriptor_width as u64)?;
        write_u64(path, &mut writer, self.len() as u64)?;
        write_u64(path, &mut writer, self.indices.len() as u64)?;
        write_u64(path, &mut writer, self.target_indices.len() as u64)?;
        for &value in &self.cids {
            write_u64(path, &mut writer, value)?;
        }
        for &value in &self.row_offsets {
            write_u64(path, &mut writer, value)?;
        }
        for &value in &self.indices {
            write_u16(path, &mut writer, value)?;
        }
        for &value in &self.counts {
            write_u16(path, &mut writer, value)?;
        }
        for &value in &self.descriptor_targets {
            write_f32(path, &mut writer, value)?;
        }
        for &split in &self.splits {
            write_all(path, &mut writer, &[split.as_byte()])?;
        }
        write_all(path, &mut writer, &self.violation_labels)?;
        for &value in &self.target_row_offsets {
            write_u64(path, &mut writer, value)?;
        }
        for &value in &self.target_indices {
            write_u16(path, &mut writer, value)?;
        }
        for &value in &self.target_counts {
            write_u16(path, &mut writer, value)?;
        }
        writer.flush().map_err(|source| Error::io(path, source))
    }

    /// Reads a sparse shard from disk.
    ///
    /// # Errors
    ///
    /// Returns an error when the shard cannot be read or does not match the
    /// expected on-disk format.
    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| Error::io(path, source))?;
        let mut reader = BufReader::new(file);

        let mut magic = [0_u8; 8];
        read_exact(path, &mut reader, &mut magic)?;
        if magic == LEGACY_SHARD_MAGIC_V2 {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "shard is the legacy MAESH02 format; \
                          rerun preprocess to regenerate as MAESH03 \
                          (negative-molecule columns required)"
                    .to_string(),
            });
        }
        if magic != SHARD_MAGIC {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "unexpected magic header".to_string(),
            });
        }

        let fingerprint_size = read_u64(path, &mut reader)? as usize;
        let descriptor_width = read_u64(path, &mut reader)? as usize;
        let row_count = read_u64(path, &mut reader)? as usize;
        let nnz = read_u64(path, &mut reader)? as usize;
        let target_nnz = read_u64(path, &mut reader)? as usize;

        let mut cids = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            cids.push(read_u64(path, &mut reader)?);
        }
        let mut row_offsets = Vec::with_capacity(row_count + 1);
        for _ in 0..=row_count {
            row_offsets.push(read_u64(path, &mut reader)?);
        }
        let mut indices = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            indices.push(read_u16(path, &mut reader)?);
        }
        let mut counts = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            counts.push(read_u16(path, &mut reader)?);
        }
        let descriptor_len =
            row_count
                .checked_mul(descriptor_width)
                .ok_or_else(|| Error::ShardFormat {
                    path: path.to_path_buf(),
                    message: "descriptor shape overflows usize".to_string(),
                })?;
        let mut descriptor_targets = Vec::with_capacity(descriptor_len);
        for _ in 0..descriptor_len {
            descriptor_targets.push(read_f32(path, &mut reader)?);
        }
        let mut split_bytes = vec![0_u8; row_count];
        read_exact(path, &mut reader, &mut split_bytes)?;
        let splits = split_bytes
            .into_iter()
            .map(|byte| {
                DataSplit::from_byte(byte).ok_or_else(|| Error::ShardFormat {
                    path: path.to_path_buf(),
                    message: format!("invalid split byte {byte}"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut violation_labels = vec![0_u8; row_count];
        read_exact(path, &mut reader, &mut violation_labels)?;
        let mut target_row_offsets = Vec::with_capacity(row_count + 1);
        for _ in 0..=row_count {
            target_row_offsets.push(read_u64(path, &mut reader)?);
        }
        let mut target_indices = Vec::with_capacity(target_nnz);
        for _ in 0..target_nnz {
            target_indices.push(read_u16(path, &mut reader)?);
        }
        let mut target_counts = Vec::with_capacity(target_nnz);
        for _ in 0..target_nnz {
            target_counts.push(read_u16(path, &mut reader)?);
        }

        let shard = Self {
            fingerprint_size,
            descriptor_width,
            cids,
            row_offsets,
            indices,
            counts,
            descriptor_targets,
            splits,
            violation_labels,
            target_row_offsets,
            target_indices,
            target_counts,
        };
        shard.validate_for_path(path)?;
        Ok(shard)
    }

    /// Reads a contiguous row range from a sparse shard without loading the full file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, the requested row range is
    /// invalid, or the shard contents are malformed.
    pub fn read_range_from_path(
        path: impl AsRef<Path>,
        start_row: usize,
        end_row: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|source| Error::io(path, source))?;
        let header = read_shard_header(path, &mut file)?;
        if start_row > end_row || end_row > header.row_count {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: format!(
                    "row range {start_row}..{end_row} exceeds shard row count {}",
                    header.row_count
                ),
            });
        }

        let row_count = end_row - start_row;
        if row_count == 0 {
            return Ok(Self::new(header.fingerprint_size, header.descriptor_width));
        }

        let layout = ShardFileLayout::new(
            header.row_count,
            header.nnz,
            header.target_nnz,
            header.descriptor_width,
        )?;
        let cids = read_u64_vec_at(
            path,
            &mut file,
            checked_add(
                path,
                layout.cids,
                checked_mul(path, start_row, 8, "molecule-id byte offset")?,
                "molecule-id byte offset",
            )?,
            row_count,
        )?;
        let absolute_offsets = read_u64_vec_at(
            path,
            &mut file,
            checked_add(
                path,
                layout.row_offsets,
                checked_mul(path, start_row, 8, "row-offset byte offset")?,
                "row-offset byte offset",
            )?,
            row_count + 1,
        )?;
        let sparse_start = *absolute_offsets.first().ok_or_else(|| Error::ShardFormat {
            path: path.to_path_buf(),
            message: "row range did not include a start sparse offset".to_string(),
        })?;
        let sparse_end = *absolute_offsets.last().ok_or_else(|| Error::ShardFormat {
            path: path.to_path_buf(),
            message: "row range did not include an end sparse offset".to_string(),
        })?;
        let sparse_len = sparse_end
            .checked_sub(sparse_start)
            .ok_or_else(|| Error::ShardFormat {
                path: path.to_path_buf(),
                message: "row-range sparse offsets are not monotonic".to_string(),
            })? as usize;
        let row_offsets = absolute_offsets
            .into_iter()
            .map(|offset| offset - sparse_start)
            .collect::<Vec<_>>();
        let sparse_start = sparse_start as usize;

        let indices = read_u16_vec_at(
            path,
            &mut file,
            checked_add(
                path,
                layout.indices,
                checked_mul(path, sparse_start, 2, "index byte offset")?,
                "index byte offset",
            )?,
            sparse_len,
        )?;
        let counts = read_u16_vec_at(
            path,
            &mut file,
            checked_add(
                path,
                layout.counts,
                checked_mul(path, sparse_start, 2, "count byte offset")?,
                "count byte offset",
            )?,
            sparse_len,
        )?;
        let descriptor_len = row_count
            .checked_mul(header.descriptor_width)
            .ok_or_else(|| Error::ShardFormat {
                path: path.to_path_buf(),
                message: "descriptor range shape overflows usize".to_string(),
            })?;
        let descriptor_targets = read_f32_vec_at(
            path,
            &mut file,
            checked_add(
                path,
                layout.descriptors,
                checked_mul_u64(
                    path,
                    checked_mul(
                        path,
                        start_row,
                        header.descriptor_width,
                        "descriptor row offset",
                    )?,
                    4,
                    "descriptor byte offset",
                )?,
                "descriptor byte offset",
            )?,
            descriptor_len,
        )?;
        let mut split_bytes = vec![0_u8; row_count];
        read_exact_at(
            path,
            &mut file,
            checked_add(path, layout.splits, start_row as u64, "split byte offset")?,
            &mut split_bytes,
        )?;
        let splits = split_bytes
            .into_iter()
            .map(|byte| {
                DataSplit::from_byte(byte).ok_or_else(|| Error::ShardFormat {
                    path: path.to_path_buf(),
                    message: format!("invalid split byte {byte}"),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut violation_labels = vec![0_u8; row_count];
        read_exact_at(
            path,
            &mut file,
            checked_add(
                path,
                layout.violation_labels,
                start_row as u64,
                "violation-label byte offset",
            )?,
            &mut violation_labels,
        )?;

        let target_absolute_offsets = read_u64_vec_at(
            path,
            &mut file,
            checked_add(
                path,
                layout.target_row_offsets,
                checked_mul(path, start_row, 8, "target row-offset byte offset")?,
                "target row-offset byte offset",
            )?,
            row_count + 1,
        )?;
        let target_sparse_start =
            *target_absolute_offsets
                .first()
                .ok_or_else(|| Error::ShardFormat {
                    path: path.to_path_buf(),
                    message: "row range did not include a start target sparse offset".to_string(),
                })?;
        let target_sparse_end =
            *target_absolute_offsets
                .last()
                .ok_or_else(|| Error::ShardFormat {
                    path: path.to_path_buf(),
                    message: "row range did not include an end target sparse offset".to_string(),
                })?;
        let target_sparse_len = target_sparse_end
            .checked_sub(target_sparse_start)
            .ok_or_else(|| Error::ShardFormat {
                path: path.to_path_buf(),
                message: "row-range target sparse offsets are not monotonic".to_string(),
            })? as usize;
        let target_row_offsets = target_absolute_offsets
            .into_iter()
            .map(|offset| offset - target_sparse_start)
            .collect::<Vec<_>>();
        let target_sparse_start = target_sparse_start as usize;

        let target_indices = read_u16_vec_at(
            path,
            &mut file,
            checked_add(
                path,
                layout.target_indices,
                checked_mul(path, target_sparse_start, 2, "target index byte offset")?,
                "target index byte offset",
            )?,
            target_sparse_len,
        )?;
        let target_counts = read_u16_vec_at(
            path,
            &mut file,
            checked_add(
                path,
                layout.target_counts,
                checked_mul(path, target_sparse_start, 2, "target count byte offset")?,
                "target count byte offset",
            )?,
            target_sparse_len,
        )?;

        let shard = Self {
            fingerprint_size: header.fingerprint_size,
            descriptor_width: header.descriptor_width,
            cids,
            row_offsets,
            indices,
            counts,
            descriptor_targets,
            splits,
            violation_labels,
            target_row_offsets,
            target_indices,
            target_counts,
        };
        shard.validate_for_path(path)?;
        Ok(shard)
    }

    /// Cross-checks all internal length invariants of a freshly-loaded
    /// shard and surfaces any mismatch as [`Error::ShardFormat`] tagged
    /// with the originating path.
    fn validate_for_path(&self, path: &Path) -> Result<()> {
        if self.row_offsets.len() != self.len() + 1 {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "row offset length does not match row count".to_string(),
            });
        }
        if self.indices.len() != self.counts.len() {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "indices and counts have different lengths".to_string(),
            });
        }
        if self.row_offsets.first().copied() != Some(0)
            || self.row_offsets.last().copied() != Some(self.indices.len() as u64)
        {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "row offsets do not cover sparse values".to_string(),
            });
        }
        if self.descriptor_targets.len() != self.len() * self.descriptor_width {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "descriptor target length does not match shape".to_string(),
            });
        }
        if self.splits.len() != self.len() {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "split length does not match row count".to_string(),
            });
        }
        if self.violation_labels.len() != self.len() {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "violation-label length does not match row count".to_string(),
            });
        }
        if self.target_row_offsets.len() != self.len() + 1 {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "target row offset length does not match row count".to_string(),
            });
        }
        if self.target_indices.len() != self.target_counts.len() {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "target indices and target counts have different lengths".to_string(),
            });
        }
        if self.target_row_offsets.first().copied() != Some(0)
            || self.target_row_offsets.last().copied() != Some(self.target_indices.len() as u64)
        {
            return Err(Error::ShardFormat {
                path: path.to_path_buf(),
                message: "target row offsets do not cover sparse values".to_string(),
            });
        }
        // Positive rows (label = 0) must not carry an explicit decoder
        // target; the row reader aliases them to the encoder input. Any
        // deviation indicates a writer bug.
        for row in 0..self.len() {
            let target_start = self.target_row_offsets[row];
            let target_end = self.target_row_offsets[row + 1];
            let has_explicit_target = target_end != target_start;
            if self.violation_labels[row] == 0 && has_explicit_target {
                return Err(Error::ShardFormat {
                    path: path.to_path_buf(),
                    message: format!(
                        "row {row} is a positive (label = 0) but carries an explicit \
                         decoder-target fingerprint"
                    ),
                });
            }
        }
        Ok(())
    }
}

/// One shard entry in a manifest. Constructed internally by
/// [`ShardManifest::push_shard`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardManifestEntry {
    /// Shard filename relative to the manifest directory.
    path: String,
    /// Number of rows in the shard.
    row_count: usize,
    /// Number of sparse nonzero entries across all rows.
    nnz: usize,
}

impl ShardManifestEntry {
    /// Shard path relative to the manifest.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Number of rows in the shard.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    /// Number of sparse nonzero entries.
    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.nnz
    }
}

/// Preprocessing manifest for cached dataset shards. Construct via
/// [`ShardManifest::new`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardManifest {
    /// Manifest schema version stamped at write time.
    manifest_version: u32,
    /// Human-readable label for the source dataset selection.
    source: String,
    /// Preprocessing configuration that produced these shards.
    preprocessing: PreprocessingConfig,
    /// Ordered list of shard entries, one per output file.
    shards: Vec<ShardManifestEntry>,
    /// Total number of rows written across all shards.
    row_count: usize,
    /// Number of source records skipped by the quality filter.
    skipped_count: usize,
    /// Number of source records that failed parsing or fingerprinting.
    error_count: usize,
}

impl ShardManifest {
    /// Creates an empty manifest.
    #[must_use]
    pub fn new(source: impl Into<String>, preprocessing: PreprocessingConfig) -> Self {
        Self {
            manifest_version: SHARD_MANIFEST_VERSION,
            source: source.into(),
            preprocessing,
            shards: Vec::new(),
            row_count: 0,
            skipped_count: 0,
            error_count: 0,
        }
    }

    /// Manifest schema version.
    #[must_use]
    pub const fn manifest_version(&self) -> u32 {
        self.manifest_version
    }

    /// Input source label or URL.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Preprocessing configuration that produced these shards.
    #[must_use]
    pub const fn preprocessing(&self) -> &PreprocessingConfig {
        &self.preprocessing
    }

    /// Output shard entries, ordered as written.
    #[must_use]
    pub fn shards(&self) -> &[ShardManifestEntry] {
        &self.shards
    }

    /// Total rows written.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    /// Records skipped before writing.
    #[must_use]
    pub const fn skipped_count(&self) -> usize {
        self.skipped_count
    }

    /// Records that failed parsing or preprocessing.
    #[must_use]
    pub const fn error_count(&self) -> usize {
        self.error_count
    }

    /// Increments the skipped-record counter.
    pub const fn record_skipped(&mut self) {
        self.skipped_count += 1;
    }

    /// Increments the error counter.
    pub const fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Appends one shard entry and updates totals.
    pub fn push_shard(&mut self, path: impl Into<String>, shard: &SparseMoleculeShard) {
        self.row_count += shard.len();
        self.shards.push(ShardManifestEntry {
            path: path.into(),
            row_count: shard.len(),
            nnz: shard.indices.len(),
        });
    }

    /// Writes the manifest as pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest cannot be created or serialized.
    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path).map_err(|source| Error::io(path, source))?;
        serde_json::to_writer_pretty(file, self).map_err(|source| Error::Json {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Reads a JSON manifest.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest cannot be opened or parsed.
    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| Error::io(path, source))?;
        serde_json::from_reader(file).map_err(|source| Error::Json {
            path: path.to_path_buf(),
            source,
        })
    }
}

/// Derives the deterministic 64-bit stable molecule identifier from a
/// `(dataset_id, record_id)` pair. Recognises the PubChem CID and ZINC20
/// formats natively; everything else falls back to a namespaced FNV-1a
/// hash so the value is collision-resistant within a dataset.
fn stable_molecule_id(dataset_id: &str, record_id: &str) -> u64 {
    if dataset_id == "pubchem-smiles"
        && let Ok(cid) = record_id.parse::<u64>()
    {
        return cid;
    }

    if dataset_id == "zinc20-smiles"
        && let Some(stable_id) = zinc20_stable_molecule_id(record_id)
    {
        return stable_id;
    }

    let mut hash = STABLE_ID_FNV_OFFSET;
    for byte in dataset_id
        .bytes()
        .chain(std::iter::once(0xff))
        .chain(record_id.bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(STABLE_ID_FNV_PRIME);
    }
    HASH_STABLE_ID_NAMESPACE | (splitmix64(hash) & HASH_STABLE_ID_MASK)
}

/// Parses a `ZINC{number}_{suffix}` identifier and packs it into a
/// 64-bit ID under [`ZINC20_STABLE_ID_NAMESPACE`]. Returns `None` for
/// malformed strings so the caller can drop the record.
fn zinc20_stable_molecule_id(record_id: &str) -> Option<u64> {
    let rest = record_id.strip_prefix("ZINC")?;
    let (number, suffix) = rest.split_once('_').unwrap_or((rest, "0"));
    let number = number.parse::<u64>().ok()?;
    let suffix = suffix.parse::<u64>().ok()?;
    if number > ZINC20_STABLE_ID_NUMBER_MASK || suffix > ZINC20_STABLE_ID_SUFFIX_MASK {
        return None;
    }
    Some(ZINC20_STABLE_ID_NAMESPACE | (suffix << ZINC20_STABLE_ID_SUFFIX_SHIFT) | number)
}

/// `splitmix64` finaliser used both for shard hashing and for the
/// per-row train/validation split assignment. Cheap, deterministic, and
/// passes basic bit-mixing tests.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// On-disk header read from / written to the start of every shard file.
#[derive(Debug, Clone, Copy)]
struct ShardFileHeader {
    /// Dense fingerprint width recovered from the shard.
    fingerprint_size: usize,
    /// Descriptor regression row width recovered from the shard.
    descriptor_width: usize,
    /// Number of rows packed in the shard.
    row_count: usize,
    /// Total nonzero sparse entries across all rows (encoder input).
    nnz: usize,
    /// Total nonzero sparse entries across all decoder-target rows
    /// (sum of negatives only; zero for shards with no negatives).
    target_nnz: usize,
}

/// Pre-computed byte offsets into a shard file, one per logical section.
/// All values are absolute byte positions relative to the start of the
/// file so seeking is constant-time.
#[derive(Debug, Clone, Copy)]
struct ShardFileLayout {
    /// Byte offset of the stable-ID section.
    cids: u64,
    /// Byte offset of the CSR row-offset section.
    row_offsets: u64,
    /// Byte offset of the concatenated sparse-index section.
    indices: u64,
    /// Byte offset of the concatenated sparse-count section.
    counts: u64,
    /// Byte offset of the descriptor regression target block.
    descriptors: u64,
    /// Byte offset of the per-row train/validation split byte block.
    splits: u64,
    /// Byte offset of the per-row violation-label byte block.
    violation_labels: u64,
    /// Byte offset of the decoder-target CSR row-offset section.
    target_row_offsets: u64,
    /// Byte offset of the decoder-target sparse-index section.
    target_indices: u64,
    /// Byte offset of the decoder-target sparse-count section.
    target_counts: u64,
}

impl ShardFileLayout {
    /// Computes the absolute byte offsets of every shard section from
    /// the row count, sparse-NNZ, target-sparse-NNZ, and descriptor
    /// width. Returns [`Error::ShardFormat`] when any multiplication
    /// overflows `u64`.
    fn new(
        row_count: usize,
        nnz: usize,
        target_nnz: usize,
        descriptor_width: usize,
    ) -> Result<Self> {
        let path = Path::new("<shard layout>");
        // Header: 8-byte magic + 5 × u64 = 48 bytes.
        let cids = 48_u64;
        let row_offsets = checked_add(
            path,
            cids,
            checked_mul(path, row_count, 8, "molecule-id section length")?,
            "row-offset section start",
        )?;
        let indices = checked_add(
            path,
            row_offsets,
            checked_mul(path, row_count + 1, 8, "row-offset section length")?,
            "index section start",
        )?;
        let counts = checked_add(
            path,
            indices,
            checked_mul(path, nnz, 2, "index section length")?,
            "count section start",
        )?;
        let descriptors = checked_add(
            path,
            counts,
            checked_mul(path, nnz, 2, "count section length")?,
            "descriptor section start",
        )?;
        let descriptor_values =
            row_count
                .checked_mul(descriptor_width)
                .ok_or_else(|| Error::ShardFormat {
                    path: path.to_path_buf(),
                    message: "descriptor section shape overflows usize".to_string(),
                })?;
        let splits = checked_add(
            path,
            descriptors,
            checked_mul(path, descriptor_values, 4, "descriptor section length")?,
            "split section start",
        )?;
        let violation_labels = checked_add(
            path,
            splits,
            row_count as u64,
            "violation-label section start",
        )?;
        let target_row_offsets = checked_add(
            path,
            violation_labels,
            row_count as u64,
            "target row-offset section start",
        )?;
        let target_indices = checked_add(
            path,
            target_row_offsets,
            checked_mul(path, row_count + 1, 8, "target row-offset section length")?,
            "target index section start",
        )?;
        let target_counts = checked_add(
            path,
            target_indices,
            checked_mul(path, target_nnz, 2, "target index section length")?,
            "target count section start",
        )?;

        Ok(Self {
            cids,
            row_offsets,
            indices,
            counts,
            descriptors,
            splits,
            violation_labels,
            target_row_offsets,
            target_indices,
            target_counts,
        })
    }
}

/// Writes `bytes` to `writer`, tagging any IO failure with `path`.
fn write_all(path: &Path, writer: &mut impl Write, bytes: &[u8]) -> Result<()> {
    writer
        .write_all(bytes)
        .map_err(|source| Error::io(path, source))
}

/// Writes a `u64` to `writer` in little-endian byte order.
fn write_u64(path: &Path, writer: &mut impl Write, value: u64) -> Result<()> {
    write_all(path, writer, &value.to_le_bytes())
}

/// Writes a `u16` to `writer` in little-endian byte order.
fn write_u16(path: &Path, writer: &mut impl Write, value: u16) -> Result<()> {
    write_all(path, writer, &value.to_le_bytes())
}

/// Writes an `f32` to `writer` in little-endian byte order.
fn write_f32(path: &Path, writer: &mut impl Write, value: f32) -> Result<()> {
    write_all(path, writer, &value.to_le_bytes())
}

/// Reads exactly `bytes.len()` bytes from `reader`, tagging any IO
/// failure with `path`.
fn read_exact(path: &Path, reader: &mut impl Read, bytes: &mut [u8]) -> Result<()> {
    reader
        .read_exact(bytes)
        .map_err(|source| Error::io(path, source))
}

/// Reads a little-endian `u64` from `reader`.
fn read_u64(path: &Path, reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    read_exact(path, reader, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Reads a little-endian `u16` from `reader`.
fn read_u16(path: &Path, reader: &mut impl Read) -> Result<u16> {
    let mut bytes = [0_u8; 2];
    read_exact(path, reader, &mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

/// Reads a little-endian `f32` from `reader`.
fn read_f32(path: &Path, reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0_u8; 4];
    read_exact(path, reader, &mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

/// Seeks to the start of `reader` and validates the magic header, then
/// reads the 40-byte fixed-shape descriptor (`MAESH03` adds a
/// `target_nnz` field to the legacy 32-byte header).
fn read_shard_header(path: &Path, reader: &mut (impl Read + Seek)) -> Result<ShardFileHeader> {
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|source| Error::io(path, source))?;
    let mut magic = [0_u8; 8];
    read_exact(path, reader, &mut magic)?;
    if magic == LEGACY_SHARD_MAGIC_V2 {
        return Err(Error::ShardFormat {
            path: path.to_path_buf(),
            message: "shard is the legacy MAESH02 format; \
                      rerun preprocess to regenerate as MAESH03 \
                      (negative-molecule columns required)"
                .to_string(),
        });
    }
    if magic != SHARD_MAGIC {
        return Err(Error::ShardFormat {
            path: path.to_path_buf(),
            message: "unexpected magic header".to_string(),
        });
    }

    Ok(ShardFileHeader {
        fingerprint_size: read_u64(path, reader)? as usize,
        descriptor_width: read_u64(path, reader)? as usize,
        row_count: read_u64(path, reader)? as usize,
        nnz: read_u64(path, reader)? as usize,
        target_nnz: read_u64(path, reader)? as usize,
    })
}

/// Seeks `reader` to `offset` then reads exactly `bytes.len()` bytes.
fn read_exact_at(
    path: &Path,
    reader: &mut (impl Read + Seek),
    offset: u64,
    bytes: &mut [u8],
) -> Result<()> {
    reader
        .seek(SeekFrom::Start(offset))
        .map_err(|source| Error::io(path, source))?;
    read_exact(path, reader, bytes)
}

/// Reads `len` little-endian `u64`s starting at `offset` in `reader`.
fn read_u64_vec_at(
    path: &Path,
    reader: &mut (impl Read + Seek),
    offset: u64,
    len: usize,
) -> Result<Vec<u64>> {
    let mut bytes = vec![0_u8; checked_mul(path, len, 8, "u64 vector byte length")? as usize];
    read_exact_at(path, reader, offset, &mut bytes)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

/// Reads `len` little-endian `u16`s starting at `offset` in `reader`.
fn read_u16_vec_at(
    path: &Path,
    reader: &mut (impl Read + Seek),
    offset: u64,
    len: usize,
) -> Result<Vec<u16>> {
    let mut bytes = vec![0_u8; checked_mul(path, len, 2, "u16 vector byte length")? as usize];
    read_exact_at(path, reader, offset, &mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

/// Reads `len` little-endian `f32`s starting at `offset` in `reader`.
fn read_f32_vec_at(
    path: &Path,
    reader: &mut (impl Read + Seek),
    offset: u64,
    len: usize,
) -> Result<Vec<f32>> {
    let mut bytes = vec![0_u8; checked_mul(path, len, 4, "f32 vector byte length")? as usize];
    read_exact_at(path, reader, offset, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Multiplies two `usize`s and converts to `u64`, surfacing overflow as
/// [`Error::ShardFormat`] tagged with `label`.
fn checked_mul(path: &Path, left: usize, right: usize, label: &str) -> Result<u64> {
    left.checked_mul(right)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| Error::ShardFormat {
            path: path.to_path_buf(),
            message: format!("{label} overflows u64"),
        })
}

/// `u64 * u64` with the same overflow contract as [`checked_mul`].
fn checked_mul_u64(path: &Path, left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_mul(right).ok_or_else(|| Error::ShardFormat {
        path: path.to_path_buf(),
        message: format!("{label} overflows u64"),
    })
}

/// `u64 + u64` with the same overflow contract as [`checked_mul`].
fn checked_add(path: &Path, left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right).ok_or_else(|| Error::ShardFormat {
        path: path.to_path_buf(),
        message: format!("{label} overflows u64"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::REGRESSION_TARGET_WIDTH;
    use std::{
        fs::OpenOptions,
        io::{Seek, SeekFrom, Write},
    };

    fn test_fingerprint() -> FingerprintTargets {
        FingerprintTargets::new(vec![3, 9], vec![1, 4], 4096).expect("valid test fingerprint")
    }

    fn test_shard() -> SparseMoleculeShard {
        let mut shard = SparseMoleculeShard::new(4096, REGRESSION_TARGET_WIDTH);
        shard
            .push(
                702,
                &test_fingerprint(),
                &[0.0; REGRESSION_TARGET_WIDTH],
                DataSplit::Train,
                0,
                None,
            )
            .expect("push");
        shard
    }

    #[test]
    fn molecule_record_preserves_pubchem_cid_as_stable_id() {
        let record = MoleculeRecord::new("pubchem-smiles", "702", "CCO");

        assert_eq!(record.dataset_id, "pubchem-smiles");
        assert_eq!(record.record_id, "702");
        assert_eq!(record.stable_id, 702);
        assert_eq!(record.smiles, "CCO");
    }

    #[test]
    fn molecule_record_packs_zinc20_ids_deterministically() {
        let first = MoleculeRecord::new("zinc20-smiles", "ZINC000000000001_1", "CCO");
        let second = MoleculeRecord::new("zinc20-smiles", "ZINC000000000001_1", "CCO");
        let different_number = MoleculeRecord::new("zinc20-smiles", "ZINC000000000002_1", "CCO");
        let different_suffix = MoleculeRecord::new("zinc20-smiles", "ZINC000000000001_2", "CCO");

        assert_eq!(first.stable_id, 0x8001_0000_0000_0001);
        assert_eq!(first.stable_id, second.stable_id);
        assert_ne!(first.stable_id, different_number.stable_id);
        assert_ne!(first.stable_id, different_suffix.stable_id);
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn dataset_record_conversion_rejects_invalid_pubchem_cids() {
        let record =
            smiles_parser::prelude::DatasetSmilesRecord::new("not-a-cid".to_string(), "CCO".into());
        let error = MoleculeRecord::from_dataset_smiles_record("pubchem-smiles", record)
            .expect_err("invalid pubchem cid");

        assert!(matches!(
            error,
            Error::DatasetRecord { message } if message.contains("invalid PubChem CID")
        ));
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn dataset_record_conversion_rejects_invalid_zinc20_ids() {
        let record =
            smiles_parser::prelude::DatasetSmilesRecord::new("not-zinc".to_string(), "CCO".into());
        let error = MoleculeRecord::from_dataset_smiles_record("zinc20-smiles", record)
            .expect_err("invalid zinc20 id");

        assert!(matches!(
            error,
            Error::DatasetRecord { message } if message.contains("invalid ZINC20 identifier")
        ));
    }

    #[test]
    fn molecule_record_hashes_unknown_dataset_ids_deterministically() {
        let first = MoleculeRecord::new("fixture-smiles", "record-1", "CCO");
        let second = MoleculeRecord::new("fixture-smiles", "record-1", "CCO");
        let different = MoleculeRecord::new("fixture-smiles", "record-2", "CCO");

        assert_eq!(first.stable_id >> 62, 1);
        assert_eq!(first.stable_id, second.stable_id);
        assert_ne!(first.stable_id, different.stable_id);
    }

    #[test]
    fn split_is_deterministic() {
        let config = PreprocessingConfig::default();

        assert_eq!(
            config.split_for_stable_id(123),
            config.split_for_stable_id(123)
        );
    }

    #[test]
    fn preprocess_record_reports_invalid_smiles() {
        let config = PreprocessingConfig::default();
        let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
        let record = MoleculeRecord::new("pubchem-smiles", "1", "(");
        let error = preprocess_record(&record, &config, &mut scratch).expect_err("invalid smiles");

        assert!(matches!(
            error,
            Error::SmilesParse {
                molecule_id: 1,
                message
            } if !message.is_empty()
        ));
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn parallel_preprocessing_preserves_order_and_reports_record_errors() {
        let config = PreprocessingConfig::builder()
            .negatives_per_positive(0.0)
            .build()
            .expect("valid");
        let mut chunks = Vec::new();
        let records = vec![
            Ok(MoleculeRecord::new("pubchem-smiles", "702", "CCO")),
            Ok(MoleculeRecord::new("pubchem-smiles", "703", "(")),
            Ok(MoleculeRecord::new("pubchem-smiles", "704", "O")),
            Err(Error::DatasetRecord {
                message: "bad record".to_string(),
            }),
        ];

        let records_seen = preprocess_dataset_record_chunks(
            records,
            &config,
            DatasetPreprocessOptions::builder()
                .chunk_rows(2)
                .threads(Some(2))
                .build()
                .expect("valid preprocess options"),
            |chunk| {
                chunks.push(chunk);
                Ok(())
            },
        )
        .expect("preprocess chunks");

        assert_eq!(records_seen, 4);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].records_seen, 2);
        assert_eq!(chunks[1].records_seen, 4);

        let results = chunks
            .into_iter()
            .flat_map(|chunk| chunk.results)
            .collect::<Vec<_>>();
        assert_eq!(results.len(), 4);
        let ethanol = results[0].as_ref().expect("ethanol ok");
        assert_eq!(ethanol.len(), 1, "negatives disabled by default");
        assert_eq!(ethanol[0].cid, 702);
        assert!(matches!(
            &results[1],
            Err(Error::SmilesParse {
                molecule_id: 703,
                message
            }) if !message.is_empty()
        ));
        let water = results[2].as_ref().expect("water ok");
        assert_eq!(water.len(), 1);
        assert_eq!(water[0].cid, 704);
        assert!(matches!(
            &results[3],
            Err(Error::DatasetRecord { message }) if message == "bad record"
        ));
    }

    #[test]
    fn preprocess_record_skips_records_rejected_by_quality_filter() {
        let config = PreprocessingConfig {
            quality_filter: crate::SmilesQualityFilter::builder()
                .min_heavy_atoms(100)
                .build()
                .expect("valid filter"),
            ..PreprocessingConfig::default()
        };
        let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
        let record = MoleculeRecord::new("pubchem-smiles", "702", "CCO");

        let outcome = preprocess_record(&record, &config, &mut scratch).expect("preprocess ok");

        assert!(outcome.is_empty(), "filtered record must produce no rows");
    }

    #[test]
    fn preprocessing_config_builder_rejects_validation_per_mille_over_thousand() {
        let error = PreprocessingConfig::builder()
            .validation_per_mille(1001)
            .build()
            .expect_err("validation_per_mille > 1000");
        assert!(matches!(
            error,
            Error::ConfigInvalid { message } if message.contains("validation_per_mille")
        ));
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn dataset_preprocess_options_builder_rejects_zero_chunk_or_threads() {
        let zero_chunk_error = DatasetPreprocessOptions::builder()
            .chunk_rows(0)
            .build()
            .expect_err("zero chunk size should be rejected");
        assert!(matches!(
            zero_chunk_error,
            Error::ConfigInvalid { message } if message.contains("chunk_rows")
        ));

        let zero_threads_error = DatasetPreprocessOptions::builder()
            .threads(Some(0))
            .build()
            .expect_err("zero thread count should be rejected");
        assert!(matches!(
            zero_threads_error,
            Error::ConfigInvalid { message } if message.contains("threads")
        ));
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn dataset_preprocessing_defaults_to_sixty_four_threads() {
        let options = DatasetPreprocessOptions::default();

        assert_eq!(options.chunk_rows(), DEFAULT_PREPROCESS_CHUNK_ROWS);
        assert_eq!(options.threads(), Some(DEFAULT_PREPROCESS_THREADS));
    }

    #[test]
    fn sparse_shard_roundtrip_preserves_rows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("shard.maeshard");
        let shard = test_shard();

        shard.write_to_path(&path).expect("write");
        let loaded = SparseMoleculeShard::read_from_path(&path).expect("read");
        let row = loaded.row(0).expect("row");

        assert_eq!(loaded, shard);
        assert_eq!(row.cid, 702);
        assert_eq!(row.fingerprint_indices, &[3, 9]);
        assert_eq!(row.fingerprint_counts, &[1, 4]);
        // Positive row: violation label is zero and the decoder target
        // aliases the encoder input (no explicit target slice).
        assert_eq!(row.violation_label(), 0);
        assert!(!row.is_negative());
        assert_eq!(row.target_fingerprint_indices(), row.fingerprint_indices());
        assert_eq!(row.target_fingerprint_counts(), row.fingerprint_counts());
    }

    #[test]
    fn sparse_shard_roundtrip_preserves_mixed_positive_and_negative_rows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mixed.maeshard");
        let mut shard = SparseMoleculeShard::new(4096, REGRESSION_TARGET_WIDTH);
        let positive_fp = test_fingerprint();
        let negative_fp =
            FingerprintTargets::new(vec![7, 11], vec![3, 2], 4096).expect("valid fingerprint");

        shard
            .push(
                17,
                &positive_fp,
                &[0.0; REGRESSION_TARGET_WIDTH],
                DataSplit::Train,
                0,
                None,
            )
            .expect("push positive");
        shard
            .push(
                17,
                &negative_fp,
                &[0.0; REGRESSION_TARGET_WIDTH],
                DataSplit::Train,
                0b0001_0010,
                Some(&positive_fp),
            )
            .expect("push negative");

        shard.write_to_path(&path).expect("write");
        let loaded = SparseMoleculeShard::read_from_path(&path).expect("read");
        assert_eq!(loaded, shard);

        let positive_row = loaded.row(0).expect("positive row");
        let negative_row = loaded.row(1).expect("negative row");
        assert!(!positive_row.is_negative());
        assert_eq!(positive_row.target_fingerprint_indices(), &[3, 9]);
        assert!(negative_row.is_negative());
        assert_eq!(negative_row.violation_label(), 0b0001_0010);
        // Negative row: decoder target points at the source positive's
        // fingerprint, not its own (mutated) one.
        assert_eq!(negative_row.fingerprint_indices(), &[7, 11]);
        assert_eq!(negative_row.target_fingerprint_indices(), &[3, 9]);
        assert_eq!(negative_row.target_fingerprint_counts(), &[1, 4]);
    }

    #[test]
    fn sparse_shard_range_read_preserves_mixed_positive_and_negative_rows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mixed-range.maeshard");
        let mut shard = SparseMoleculeShard::new(4096, REGRESSION_TARGET_WIDTH);
        let positive_fp = test_fingerprint();
        let negative_fp =
            FingerprintTargets::new(vec![7, 11], vec![3, 2], 4096).expect("valid fingerprint");
        shard
            .push(
                10,
                &positive_fp,
                &[0.0; REGRESSION_TARGET_WIDTH],
                DataSplit::Train,
                0,
                None,
            )
            .expect("push positive 0");
        shard
            .push(
                10,
                &negative_fp,
                &[1.0; REGRESSION_TARGET_WIDTH],
                DataSplit::Train,
                0b0000_0100,
                Some(&positive_fp),
            )
            .expect("push negative");
        shard
            .push(
                20,
                &positive_fp,
                &[2.0; REGRESSION_TARGET_WIDTH],
                DataSplit::Validation,
                0,
                None,
            )
            .expect("push positive 2");
        shard.write_to_path(&path).expect("write");

        // Pull rows [1, 3): one negative + one positive. The slice must
        // carry the negative's explicit target and the positive's
        // aliased target side-by-side.
        let range = SparseMoleculeShard::read_range_from_path(&path, 1, 3).expect("range");
        assert_eq!(range.len(), 2);
        let neg = range.row(0).expect("negative");
        let pos = range.row(1).expect("positive");
        assert!(neg.is_negative());
        assert_eq!(neg.violation_label(), 0b0000_0100);
        assert_eq!(neg.target_fingerprint_indices(), &[3, 9]);
        assert!(!pos.is_negative());
        assert_eq!(pos.target_fingerprint_indices(), pos.fingerprint_indices());
    }

    #[test]
    fn sparse_shard_rejects_legacy_v2_magic_with_repreprocess_hint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("legacy.maeshard");
        std::fs::write(&path, b"MAESH02\0").expect("write legacy magic");
        let error = SparseMoleculeShard::read_from_path(&path).expect_err("legacy magic");
        assert!(matches!(
            error,
            Error::ShardFormat { message, .. } if message.contains("MAESH02")
                && message.contains("rerun preprocess")
        ));
    }

    #[test]
    fn sparse_shard_range_read_preserves_requested_rows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("shard.maeshard");
        let mut shard = SparseMoleculeShard::new(4096, REGRESSION_TARGET_WIDTH);
        for row in 0_u16..4 {
            let fingerprint =
                FingerprintTargets::new(vec![row + 1, row + 11], vec![row + 2, row + 3], 4096)
                    .expect("valid fingerprint");
            shard
                .push(
                    1000 + u64::from(row),
                    &fingerprint,
                    &[f32::from(row); REGRESSION_TARGET_WIDTH],
                    if row % 2 == 0 {
                        DataSplit::Train
                    } else {
                        DataSplit::Validation
                    },
                    0,
                    None,
                )
                .expect("push");
        }
        shard.write_to_path(&path).expect("write");

        let range = SparseMoleculeShard::read_range_from_path(&path, 1, 3).expect("range");

        assert_eq!(range.len(), 2);
        assert_eq!(range.row_offsets, vec![0, 2, 4]);
        assert_eq!(range.cids, vec![1001, 1002]);
        assert_eq!(range.indices, vec![2, 12, 3, 13]);
        assert_eq!(range.counts, vec![3, 4, 4, 5]);
        assert_eq!(range.splits, vec![DataSplit::Validation, DataSplit::Train]);
        assert_eq!(range.descriptor_targets[0], 1.0);
        assert_eq!(range.descriptor_targets[REGRESSION_TARGET_WIDTH], 2.0);
    }

    #[test]
    fn sparse_shard_row_returns_none_out_of_bounds() {
        let shard = test_shard();

        assert!(shard.row(1).is_none());
    }

    #[test]
    fn sparse_shard_rejects_invalid_push_inputs() {
        let mut shard = SparseMoleculeShard::new(4096, REGRESSION_TARGET_WIDTH);
        let wrong_width =
            FingerprintTargets::new(vec![3, 9], vec![1, 4], 2048).expect("valid fingerprint");
        let error = shard
            .push(
                702,
                &wrong_width,
                &[0.0; REGRESSION_TARGET_WIDTH],
                DataSplit::Train,
                0,
                None,
            )
            .expect_err("width mismatch");
        assert!(
            matches!(error, Error::InvalidBatch(message) if message.contains("fingerprint width"))
        );

        let error = shard
            .push(
                702,
                &test_fingerprint(),
                &[0.0; REGRESSION_TARGET_WIDTH - 1],
                DataSplit::Train,
                0,
                None,
            )
            .expect_err("descriptor width mismatch");
        assert!(
            matches!(error, Error::InvalidBatch(message) if message.contains("descriptor width"))
        );
    }

    #[test]
    fn fingerprint_targets_constructor_rejects_invalid_inputs() {
        let length_mismatch =
            FingerprintTargets::new(vec![3, 9], vec![1], 4096).expect_err("length mismatch");
        assert!(matches!(
            length_mismatch,
            Error::InvalidBatch(message) if message.contains("different lengths")
        ));

        let index_overflow =
            FingerprintTargets::new(vec![4096], vec![1], 4096).expect_err("index >= width");
        assert!(matches!(
            index_overflow,
            Error::InvalidBatch(message) if message.contains("exceeds configured width")
        ));
    }

    #[test]
    fn sparse_shard_rejects_corrupt_magic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad.maeshard");
        std::fs::write(&path, b"NOTSHARD").expect("write bad shard");
        let error = SparseMoleculeShard::read_from_path(&path).expect_err("bad magic");

        assert!(matches!(
            error,
            Error::ShardFormat { message, .. } if message == "unexpected magic header"
        ));
    }

    #[test]
    fn sparse_shard_rejects_invalid_split_byte() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad-split.maeshard");
        let shard = test_shard();
        shard.write_to_path(&path).expect("write shard");
        // The split byte for row 0 sits immediately after the descriptor
        // block. v3's appended sections (violation labels + target CSR)
        // shifted the split section so we can no longer use a fixed
        // negative offset from the end; compute it from the layout.
        let split_offset = 48 // magic + 5 × u64 header
            + 8 // cids (1 row)
            + 16 // row_offsets (row_count + 1)
            + 4 // indices (nnz = 2, 2 bytes each)
            + 4 // counts (nnz = 2, 2 bytes each)
            + REGRESSION_TARGET_WIDTH * 4; // descriptors (row_count × width)
        let mut file = OpenOptions::new().write(true).open(&path).expect("open");
        file.seek(SeekFrom::Start(split_offset as u64))
            .expect("seek to split byte");
        file.write_all(&[9]).expect("overwrite split");
        drop(file);

        let error = SparseMoleculeShard::read_from_path(&path).expect_err("invalid split");
        assert!(matches!(
            error,
            Error::ShardFormat { message, .. } if message.contains("invalid split byte")
        ));
    }

    #[test]
    fn sparse_shard_rejects_invalid_serialized_shapes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("invalid.maeshard");
        let mut shard = test_shard();

        shard.row_offsets.pop();
        let error = shard.write_to_path(&path).expect_err("invalid row offsets");
        assert!(matches!(
            error,
            Error::ShardFormat { message, .. } if message.contains("row offset length")
        ));

        let mut shard = test_shard();
        shard.counts.pop();
        let error = shard
            .write_to_path(&path)
            .expect_err("invalid count length");
        assert!(matches!(
            error,
            Error::ShardFormat { message, .. } if message.contains("different lengths")
        ));

        let mut shard = test_shard();
        shard.row_offsets[0] = 1;
        let error = shard.write_to_path(&path).expect_err("invalid offsets");
        assert!(matches!(
            error,
            Error::ShardFormat { message, .. } if message.contains("row offsets")
        ));

        let mut shard = test_shard();
        shard.descriptor_targets.pop();
        let error = shard
            .write_to_path(&path)
            .expect_err("invalid descriptor shape");
        assert!(matches!(
            error,
            Error::ShardFormat { message, .. } if message.contains("descriptor target length")
        ));

        let mut shard = test_shard();
        shard.splits.pop();
        let error = shard
            .write_to_path(&path)
            .expect_err("invalid split length");
        assert!(matches!(
            error,
            Error::ShardFormat { message, .. } if message.contains("split length")
        ));
    }

    #[test]
    fn shard_manifest_roundtrip_preserves_entries_and_totals() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("manifest.json");
        let config = PreprocessingConfig::default();
        let shard = test_shard();
        let mut manifest = ShardManifest::new("dataset-fixture", config);
        manifest.skipped_count = 2;
        manifest.error_count = 3;
        manifest.push_shard("shard-00000.maeshard", &shard);

        manifest.write_to_path(&path).expect("write manifest");
        let loaded = ShardManifest::read_from_path(&path).expect("read manifest");

        assert_eq!(loaded, manifest);
        assert_eq!(loaded.row_count, 1);
        assert_eq!(loaded.shards[0].nnz, 2);
    }

    #[test]
    fn molecule_record_accessors_return_constructed_values() {
        let record = MoleculeRecord::new("pubchem-smiles", "42", "CCO");
        assert_eq!(record.dataset_id(), "pubchem-smiles");
        assert_eq!(record.record_id(), "42");
        assert_eq!(record.stable_id(), 42);
        assert_eq!(record.smiles(), "CCO");
    }

    #[test]
    fn shard_manifest_accessors_and_counters_track_state() {
        let config = PreprocessingConfig::default();
        let mut manifest = ShardManifest::new("fixture", config);
        let shard = test_shard();

        assert_eq!(manifest.manifest_version(), SHARD_MANIFEST_VERSION);
        assert_eq!(manifest.source(), "fixture");
        assert_eq!(manifest.row_count(), 0);
        assert_eq!(manifest.skipped_count(), 0);
        assert_eq!(manifest.error_count(), 0);
        assert!(manifest.shards().is_empty());
        assert_eq!(manifest.preprocessing(), &PreprocessingConfig::default());

        manifest.record_skipped();
        manifest.record_skipped();
        manifest.record_error();
        manifest.push_shard("shard-00000.maeshard", &shard);

        assert_eq!(manifest.skipped_count(), 2);
        assert_eq!(manifest.error_count(), 1);
        assert_eq!(manifest.row_count(), shard.len());
        assert_eq!(manifest.shards().len(), 1);
        let entry = &manifest.shards()[0];
        assert_eq!(entry.path(), "shard-00000.maeshard");
        assert_eq!(entry.row_count(), shard.len());
        assert_eq!(entry.nnz(), shard.indices.len());
    }

    #[test]
    fn molecule_targets_accessors_round_trip_preprocessed_record() {
        let config = PreprocessingConfig::builder()
            .negatives_per_positive(0.0)
            .build()
            .expect("valid");
        let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
        let record = MoleculeRecord::new("pubchem-smiles", "702", "CCO");
        let rows = preprocess_record(&record, &config, &mut scratch).expect("preprocess ok");
        assert_eq!(
            rows.len(),
            1,
            "default config emits one positive and zero negatives"
        );
        let targets = &rows[0];

        assert_eq!(targets.cid(), 702);
        assert_eq!(targets.source_smiles(), "CCO");
        assert_eq!(
            targets.fingerprint().fingerprint_size(),
            config.counted_ecfp().size()
        );
        assert!(targets.descriptors().carbon_count() >= 2);
        assert_eq!(targets.violation_label(), 0);
        assert!(!targets.is_negative());
        // The target fingerprint of a positive aliases the input.
        assert!(targets.explicit_target_fingerprint().is_none());
        assert_eq!(
            targets.target_fingerprint().indices(),
            targets.fingerprint().indices()
        );
        // The split tag must be one of the two enum variants.
        assert!(matches!(
            targets.split(),
            DataSplit::Train | DataSplit::Validation
        ));
    }

    #[test]
    fn preprocess_record_emits_negatives_with_labels_and_distinct_inputs() {
        let config = PreprocessingConfig::builder()
            .negatives_per_positive(2.0)
            .build()
            .expect("valid config");
        let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
        let record = MoleculeRecord::new("pubchem-smiles", "996", "c1ccccc1O");
        let rows = preprocess_record(&record, &config, &mut scratch).expect("preprocess ok");

        assert_eq!(
            rows.len(),
            3,
            "1 positive + 2 negatives expected for phenol"
        );
        let positive = &rows[0];
        assert!(!positive.is_negative());
        assert!(positive.explicit_target_fingerprint().is_none());

        for negative in &rows[1..] {
            assert!(negative.is_negative(), "label bitset must be non-zero");
            // Decoder target on a negative is the source positive's
            // fingerprint (denoising framing).
            let target = negative
                .explicit_target_fingerprint()
                .expect("negative carries an explicit target");
            assert_eq!(target.indices(), positive.fingerprint().indices());
            // Same descriptors as the positive (source molecule).
            assert_eq!(
                negative.descriptors().carbon_count(),
                positive.descriptors().carbon_count()
            );
            assert_eq!(negative.split(), positive.split());
        }
    }

    #[test]
    fn negative_count_for_stable_id_is_deterministic_and_zero_when_disabled() {
        let zero = PreprocessingConfig::builder()
            .negatives_per_positive(0.0)
            .build()
            .expect("valid");
        assert_eq!(zero.negatives_per_positive(), 0.0);
        assert_eq!(zero.negative_count_for_stable_id(0xABCD), 0);

        let two = PreprocessingConfig::builder()
            .negatives_per_positive(2.0)
            .build()
            .expect("valid");
        assert_eq!(two.negative_count_for_stable_id(0xABCD), 2);
        // Determinism: same stable_id → same count across calls.
        assert_eq!(
            two.negative_count_for_stable_id(0xABCD),
            two.negative_count_for_stable_id(0xABCD)
        );

        // Fractional ratio: roughly half of a large sweep should emit 1.
        let half = PreprocessingConfig::builder()
            .negatives_per_positive(0.5)
            .build()
            .expect("valid");
        let mut ones = 0_usize;
        for id in 0..1_000_u64 {
            if half.negative_count_for_stable_id(id) == 1 {
                ones += 1;
            }
        }
        assert!(
            (350..=650).contains(&ones),
            "Bernoulli draw should land near 50%, got {ones} of 1000"
        );
    }

    #[test]
    fn preprocessing_config_builder_rejects_negative_negatives_per_positive() {
        let error = PreprocessingConfig::builder()
            .negatives_per_positive(-0.1)
            .build()
            .expect_err("must reject negative ratio");
        assert!(matches!(
            error,
            Error::ConfigInvalid { message } if message.contains("negatives_per_positive")
        ));
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn dataset_preprocess_options_accessors_round_trip_builder() {
        let options = DatasetPreprocessOptions::builder()
            .chunk_rows(256)
            .threads(Some(4))
            .build()
            .expect("valid options");
        assert_eq!(options.chunk_rows(), 256);
        assert_eq!(options.threads(), Some(4));

        let unbounded = DatasetPreprocessOptions::builder()
            .chunk_rows(512)
            .threads(None)
            .build()
            .expect("None threads is valid");
        assert_eq!(unbounded.threads(), None);
    }
}
