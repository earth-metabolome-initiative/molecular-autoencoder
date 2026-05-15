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

const SHARD_MAGIC: [u8; 8] = *b"MAESH02\0";
const STABLE_ID_FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const STABLE_ID_FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const HASH_STABLE_ID_NAMESPACE: u64 = 0x4000_0000_0000_0000;
const HASH_STABLE_ID_MASK: u64 = 0x3fff_ffff_ffff_ffff;
const ZINC20_STABLE_ID_NAMESPACE: u64 = 0x8000_0000_0000_0000;
const ZINC20_STABLE_ID_SUFFIX_SHIFT: u32 = 48;
const ZINC20_STABLE_ID_NUMBER_MASK: u64 = 0x0000_ffff_ffff_ffff;
const ZINC20_STABLE_ID_SUFFIX_MASK: u64 = 0x7fff;
/// Current cached-shard manifest schema version.
pub const SHARD_MANIFEST_VERSION: u32 = 2;
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
    dataset_id: String,
    record_id: String,
    stable_id: u64,
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
    chunk_rows: usize,
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
    chunk_rows: usize,
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
    records_seen: usize,
    results: Vec<Result<Option<MoleculeTargets>>>,
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
    /// `Ok(Some(_))` is an accepted molecule, `Ok(None)` is a record skipped
    /// by the configured [`SmilesQualityFilter`], and `Err(_)` is a parse or
    /// fingerprint failure.
    pub fn results(&self) -> &[Result<Option<MoleculeTargets>>] {
        &self.results
    }

    /// Consumes the chunk, returning its results.
    #[must_use]
    pub fn into_results(self) -> Vec<Result<Option<MoleculeTargets>>> {
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

#[cfg(feature = "datasets")]
fn preprocess_record_chunk_parallel(
    chunk: Vec<Result<MoleculeRecord>>,
    config: &PreprocessingConfig,
) -> Vec<Result<Option<MoleculeTargets>>> {
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
    const fn as_byte(self) -> u8 {
        match self {
            Self::Train => 0,
            Self::Validation => 1,
        }
    }

    const fn from_byte(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Train),
            1 => Some(Self::Validation),
            _ => None,
        }
    }
}

/// Deterministic preprocessing configuration. Construct via
/// [`PreprocessingConfig::builder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    counted_ecfp: CountedEcfpConfig,
    descriptors: DescriptorConfig,
    validation_per_mille: u16,
    #[serde(default)]
    quality_filter: SmilesQualityFilter,
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
    counted_ecfp: CountedEcfpConfig,
    descriptors: DescriptorConfig,
    validation_per_mille: u16,
    quality_filter: SmilesQualityFilter,
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

    /// Validates and builds the immutable [`PreprocessingConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when `validation_per_mille` exceeds
    /// 1000.
    pub fn build(self) -> Result<PreprocessingConfig> {
        if self.validation_per_mille > 1000 {
            return Err(Error::ConfigInvalid {
                message: format!(
                    "validation_per_mille must be in 0..=1000, got {}",
                    self.validation_per_mille
                ),
            });
        }
        Ok(PreprocessingConfig {
            counted_ecfp: self.counted_ecfp,
            descriptors: self.descriptors,
            validation_per_mille: self.validation_per_mille,
            quality_filter: self.quality_filter,
        })
    }
}

/// Parsed, fingerprinted, descriptor-rich molecule target.
///
/// Constructed by [`preprocess_record`]; consumed via the accessor methods
/// below.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoleculeTargets {
    cid: u64,
    source_smiles: String,
    fingerprint: FingerprintTargets,
    descriptors: DescriptorTargets,
    split: DataSplit,
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

    /// Sparse counted ECFP target.
    #[must_use]
    pub const fn fingerprint(&self) -> &FingerprintTargets {
        &self.fingerprint
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
}

/// Parses and fingerprints one dataset record, applying the configured
/// quality filter before doing fingerprint work.
///
/// # Errors
///
/// Returns an error when the SMILES cannot be parsed or fingerprint preparation
/// fails. Returns `Ok(None)` when the record is rejected by the quality
/// filter; callers should treat that as a silent skip.
pub fn preprocess_record(
    record: &MoleculeRecord,
    config: &PreprocessingConfig,
    scratch: &mut finge_rs::smiles_support::SmilesRdkitScratch,
) -> Result<Option<MoleculeTargets>> {
    let smiles = record
        .smiles
        .parse::<Smiles>()
        .map_err(|source| Error::SmilesParse {
            molecule_id: record.stable_id,
            message: format!("{}:{}: {source}", record.dataset_id, record.record_id),
        })?;
    let descriptors = DescriptorTargets::from_smiles(&smiles);
    if !config.quality_filter.accepts(&descriptors) {
        return Ok(None);
    }
    let fingerprint =
        compute_fingerprint_targets(record.stable_id, &smiles, config.counted_ecfp, scratch)?;
    let split = config.split_for_stable_id(record.stable_id);

    Ok(Some(MoleculeTargets {
        cid: record.stable_id,
        source_smiles: record.smiles.clone(),
        fingerprint,
        descriptors,
        split,
    }))
}

/// One row view into a sparse molecule shard.
#[derive(Debug, Clone, Copy)]
pub struct MoleculeShardRow<'a> {
    cid: u64,
    fingerprint_indices: &'a [u16],
    fingerprint_counts: &'a [u16],
    descriptor_targets: &'a [f32],
    split: DataSplit,
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
}

/// Tensor-ready sparse molecule shard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseMoleculeShard {
    fingerprint_size: usize,
    descriptor_width: usize,
    cids: Vec<u64>,
    row_offsets: Vec<u64>,
    indices: Vec<u16>,
    counts: Vec<u16>,
    descriptor_targets: Vec<f32>,
    splits: Vec<DataSplit>,
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
    /// # Errors
    ///
    /// Returns [`Error::InvalidBatch`] when the fingerprint width, descriptor
    /// width, or sparse index/count shape does not match the shard.
    pub fn push(
        &mut self,
        cid: u64,
        fingerprint: &FingerprintTargets,
        descriptor_targets: &[f32],
        split: DataSplit,
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
        self.cids.push(cid);
        self.indices.extend_from_slice(fingerprint.indices());
        self.counts.extend_from_slice(fingerprint.counts());
        self.descriptor_targets
            .extend_from_slice(descriptor_targets);
        self.splits.push(split);
        self.row_offsets.push(self.indices.len() as u64);
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
        Some(MoleculeShardRow {
            cid: self.cids[row],
            fingerprint_indices: &self.indices[start..end],
            fingerprint_counts: &self.counts[start..end],
            descriptor_targets: &self.descriptor_targets[descriptor_start..descriptor_end],
            split: self.splits[row],
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

        let shard = Self {
            fingerprint_size,
            descriptor_width,
            cids,
            row_offsets,
            indices,
            counts,
            descriptor_targets,
            splits,
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

        let layout = ShardFileLayout::new(header.row_count, header.nnz, header.descriptor_width)?;
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

        let shard = Self {
            fingerprint_size: header.fingerprint_size,
            descriptor_width: header.descriptor_width,
            cids,
            row_offsets,
            indices,
            counts,
            descriptor_targets,
            splits,
        };
        shard.validate_for_path(path)?;
        Ok(shard)
    }

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
        Ok(())
    }
}

/// One shard entry in a manifest. Constructed internally by
/// [`ShardManifest::push_shard`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardManifestEntry {
    path: String,
    row_count: usize,
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
    manifest_version: u32,
    source: String,
    preprocessing: PreprocessingConfig,
    shards: Vec<ShardManifestEntry>,
    row_count: usize,
    skipped_count: usize,
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

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Debug, Clone, Copy)]
struct ShardFileHeader {
    fingerprint_size: usize,
    descriptor_width: usize,
    row_count: usize,
    nnz: usize,
}

#[derive(Debug, Clone, Copy)]
struct ShardFileLayout {
    cids: u64,
    row_offsets: u64,
    indices: u64,
    counts: u64,
    descriptors: u64,
    splits: u64,
}

impl ShardFileLayout {
    fn new(row_count: usize, nnz: usize, descriptor_width: usize) -> Result<Self> {
        let path = Path::new("<shard layout>");
        let cids = 40_u64;
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

        Ok(Self {
            cids,
            row_offsets,
            indices,
            counts,
            descriptors,
            splits,
        })
    }
}

fn write_all(path: &Path, writer: &mut impl Write, bytes: &[u8]) -> Result<()> {
    writer
        .write_all(bytes)
        .map_err(|source| Error::io(path, source))
}

fn write_u64(path: &Path, writer: &mut impl Write, value: u64) -> Result<()> {
    write_all(path, writer, &value.to_le_bytes())
}

fn write_u16(path: &Path, writer: &mut impl Write, value: u16) -> Result<()> {
    write_all(path, writer, &value.to_le_bytes())
}

fn write_f32(path: &Path, writer: &mut impl Write, value: f32) -> Result<()> {
    write_all(path, writer, &value.to_le_bytes())
}

fn read_exact(path: &Path, reader: &mut impl Read, bytes: &mut [u8]) -> Result<()> {
    reader
        .read_exact(bytes)
        .map_err(|source| Error::io(path, source))
}

fn read_u64(path: &Path, reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    read_exact(path, reader, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u16(path: &Path, reader: &mut impl Read) -> Result<u16> {
    let mut bytes = [0_u8; 2];
    read_exact(path, reader, &mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_f32(path: &Path, reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0_u8; 4];
    read_exact(path, reader, &mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_shard_header(path: &Path, reader: &mut (impl Read + Seek)) -> Result<ShardFileHeader> {
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|source| Error::io(path, source))?;
    let mut magic = [0_u8; 8];
    read_exact(path, reader, &mut magic)?;
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
    })
}

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

fn checked_mul(path: &Path, left: usize, right: usize, label: &str) -> Result<u64> {
    left.checked_mul(right)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| Error::ShardFormat {
            path: path.to_path_buf(),
            message: format!("{label} overflows u64"),
        })
}

fn checked_mul_u64(path: &Path, left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_mul(right).ok_or_else(|| Error::ShardFormat {
        path: path.to_path_buf(),
        message: format!("{label} overflows u64"),
    })
}

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
        let config = PreprocessingConfig::default();
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
        assert_eq!(
            results[0]
                .as_ref()
                .expect("ethanol ok")
                .as_ref()
                .expect("ethanol kept")
                .cid,
            702
        );
        assert!(matches!(
            &results[1],
            Err(Error::SmilesParse {
                molecule_id: 703,
                message
            }) if !message.is_empty()
        ));
        assert_eq!(
            results[2]
                .as_ref()
                .expect("water ok")
                .as_ref()
                .expect("water kept")
                .cid,
            704
        );
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

        assert!(outcome.is_none());
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
        let mut file = OpenOptions::new().write(true).open(&path).expect("open");
        file.seek(SeekFrom::End(-1)).expect("seek");
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
}
