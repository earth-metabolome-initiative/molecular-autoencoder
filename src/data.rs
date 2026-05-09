//! PubChem materialized-record parsing and tensor-ready sparse shard IO.

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use smiles_parser::prelude::Smiles;

use crate::{
    CountedEcfpConfig, DescriptorConfig, DescriptorTargets, Error, FingerprintTargets, Result,
    compute_fingerprint_targets,
};

const SHARD_MAGIC: [u8; 8] = *b"MAESH02\0";
/// Current cached-shard manifest schema version.
pub const SHARD_MANIFEST_VERSION: u32 = 2;
/// Default number of PubChem records preprocessed per Rayon chunk.
#[cfg(feature = "datasets")]
pub const DEFAULT_PREPROCESS_CHUNK_ROWS: usize = 8192;
/// Default Rayon workers for PubChem preprocessing.
#[cfg(feature = "datasets")]
pub const DEFAULT_PREPROCESS_THREADS: usize = 64;
/// Default number of preprocessed PubChem rows written per sparse shard.
#[cfg(feature = "datasets")]
pub const DEFAULT_PUBCHEM_ROWS_PER_SHARD: usize = 10_000_000;

/// One record from a materialized PubChem CID-SMILES artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CidSmilesRecord {
    /// PubChem compound identifier.
    pub cid: u64,
    /// Source SMILES text.
    pub smiles: String,
}

/// Parses one materialized PubChem `CID-SMILES` line.
///
/// # Errors
///
/// Returns [`Error::PubChemRecord`] when the line is missing a CID, contains an
/// invalid CID, or has no SMILES field.
pub fn parse_pubchem_cid_smiles_line(line_number: usize, line: &str) -> Result<CidSmilesRecord> {
    let line = line.trim_end_matches(['\n', '\r']);
    let mut fields = line.splitn(2, char::is_whitespace);
    let cid = fields
        .next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| Error::PubChemRecord {
            line_number,
            message: "missing CID".to_string(),
        })?
        .parse::<u64>()
        .map_err(|source| Error::PubChemRecord {
            line_number,
            message: format!("invalid CID: {source}"),
        })?;
    let smiles = fields
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| Error::PubChemRecord {
            line_number,
            message: "missing SMILES".to_string(),
        })?;

    Ok(CidSmilesRecord {
        cid,
        smiles: smiles.to_string(),
    })
}

/// Streaming parser for materialized PubChem CID-SMILES artifacts.
#[derive(Debug)]
pub struct PubChemRecordIter<R: BufRead> {
    reader: R,
    line_number: usize,
    buffer: String,
}

impl<R: BufRead> PubChemRecordIter<R> {
    /// Creates an iterator from an existing buffered reader.
    #[must_use]
    pub fn from_reader(reader: R) -> Self {
        Self {
            reader,
            line_number: 0,
            buffer: String::new(),
        }
    }
}

impl<R: BufRead> Iterator for PubChemRecordIter<R> {
    type Item = Result<CidSmilesRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buffer.clear();
            let read = match self.reader.read_line(&mut self.buffer) {
                Ok(read) => read,
                Err(source) => {
                    return Some(Err(Error::io(PathBuf::from("<pubchem stream>"), source)));
                }
            };
            if read == 0 {
                return None;
            }
            self.line_number += 1;
            if self.buffer.trim().is_empty() {
                continue;
            }
            return Some(parse_pubchem_cid_smiles_line(
                self.line_number,
                &self.buffer,
            ));
        }
    }
}

/// Opens a plain-text or gzip-compressed materialized PubChem CID-SMILES artifact.
///
/// Upstream acquisition/cache should be handled by `smiles-parser`; this helper
/// keeps CIDs for deterministic splits and shard metadata.
///
/// # Errors
///
/// Returns [`Error::Io`] when the source artifact cannot be opened.
#[cfg(feature = "datasets")]
pub fn pubchem_records_from_path(
    path: impl AsRef<Path>,
) -> Result<PubChemRecordIter<Box<dyn BufRead>>> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| Error::io(path, source))?;
    let reader: Box<dyn BufRead> = if path.extension().and_then(|ext| ext.to_str()) == Some("gz") {
        Box::new(BufReader::new(flate2::read::GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    Ok(PubChemRecordIter::from_reader(reader))
}

/// Parallel PubChem preprocessing controls.
#[cfg(feature = "datasets")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PubChemPreprocessOptions {
    /// Number of records read before dispatching work to Rayon.
    pub chunk_rows: usize,
    /// Optional local Rayon thread count.
    pub threads: Option<usize>,
}

#[cfg(feature = "datasets")]
impl Default for PubChemPreprocessOptions {
    fn default() -> Self {
        Self {
            chunk_rows: DEFAULT_PREPROCESS_CHUNK_ROWS,
            threads: Some(DEFAULT_PREPROCESS_THREADS),
        }
    }
}

/// Ordered result of one parallel PubChem preprocessing chunk.
#[cfg(feature = "datasets")]
#[derive(Debug)]
pub struct PreprocessedPubChemChunk {
    /// Total records read from the source after this chunk.
    pub records_seen: usize,
    /// Per-record preprocessing results, preserving source order.
    pub results: Vec<Result<MoleculeTargets>>,
}

/// Reads PubChem records sequentially, then preprocesses each chunk in parallel.
///
/// # Errors
///
/// Returns an error when the preprocessing options are invalid, the optional
/// Rayon pool cannot be built, an input record is malformed, or the consumer
/// callback returns an error.
#[cfg(feature = "datasets")]
pub fn preprocess_cid_smiles_record_chunks<I, F>(
    records: I,
    config: &PreprocessingConfig,
    options: PubChemPreprocessOptions,
    mut consume: F,
) -> Result<usize>
where
    I: IntoIterator<Item = Result<CidSmilesRecord>>,
    F: FnMut(PreprocessedPubChemChunk) -> Result<()>,
{
    validate_preprocess_options(options)?;
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
        consume(PreprocessedPubChemChunk {
            records_seen,
            results,
        })?;
    }
}

#[cfg(feature = "datasets")]
fn validate_preprocess_options(options: PubChemPreprocessOptions) -> Result<()> {
    if options.chunk_rows == 0 {
        return Err(Error::InvalidBatch(
            "preprocess chunk size must be greater than zero".to_string(),
        ));
    }
    if options.threads == Some(0) {
        return Err(Error::InvalidBatch(
            "preprocess threads must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

#[cfg(feature = "datasets")]
fn preprocess_record_chunk_parallel(
    chunk: Vec<Result<CidSmilesRecord>>,
    config: &PreprocessingConfig,
) -> Vec<Result<MoleculeTargets>> {
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

/// Deterministic preprocessing configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Main counted ECFP configuration.
    pub counted_ecfp: CountedEcfpConfig,
    /// Descriptor normalization configuration.
    pub descriptors: DescriptorConfig,
    /// Validation split in permille units.
    pub validation_per_mille: u16,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            counted_ecfp: CountedEcfpConfig::default(),
            descriptors: DescriptorConfig::default(),
            validation_per_mille: 100,
        }
    }
}

impl PreprocessingConfig {
    /// Assigns a deterministic split from CID using a fixed SplitMix64 hash.
    #[must_use]
    pub fn split_for_cid(&self, cid: u64) -> DataSplit {
        let validation_per_mille = u64::from(self.validation_per_mille.min(1000));
        if splitmix64(cid) % 1000 < validation_per_mille {
            DataSplit::Validation
        } else {
            DataSplit::Train
        }
    }
}

/// Parsed, fingerprinted, descriptor-rich molecule target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoleculeTargets {
    /// PubChem compound identifier.
    pub cid: u64,
    /// Source SMILES text.
    pub source_smiles: String,
    /// Sparse counted ECFP target.
    pub fingerprint: FingerprintTargets,
    /// Descriptor targets.
    pub descriptors: DescriptorTargets,
    /// Deterministic data split.
    pub split: DataSplit,
}

/// Parses and fingerprints one PubChem record.
///
/// # Errors
///
/// Returns an error when the SMILES cannot be parsed or fingerprint preparation
/// fails.
pub fn preprocess_record(
    record: &CidSmilesRecord,
    config: &PreprocessingConfig,
    scratch: &mut finge_rs::smiles_support::SmilesRdkitScratch,
) -> Result<MoleculeTargets> {
    let smiles = record
        .smiles
        .parse::<Smiles>()
        .map_err(|source| Error::SmilesParse {
            cid: record.cid,
            message: source.to_string(),
        })?;
    let descriptors = DescriptorTargets::from_smiles(&smiles);
    let fingerprint =
        compute_fingerprint_targets(record.cid, &smiles, config.counted_ecfp, scratch)?;
    let split = config.split_for_cid(record.cid);

    Ok(MoleculeTargets {
        cid: record.cid,
        source_smiles: record.smiles.clone(),
        fingerprint,
        descriptors,
        split,
    })
}

/// One row view into a sparse molecule shard.
#[derive(Debug, Clone, Copy)]
pub struct MoleculeShardRow<'a> {
    /// PubChem compound identifier.
    pub cid: u64,
    /// Sparse fingerprint indices.
    pub fingerprint_indices: &'a [u16],
    /// Sparse fingerprint counts.
    pub fingerprint_counts: &'a [u16],
    /// Normalized descriptor regression targets.
    pub descriptor_targets: &'a [f32],
    /// Deterministic data split.
    pub split: DataSplit,
}

/// Tensor-ready sparse molecule shard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseMoleculeShard {
    /// Full dense fingerprint width.
    pub fingerprint_size: usize,
    /// Descriptor regression target width.
    pub descriptor_width: usize,
    /// PubChem CIDs, one per row.
    pub cids: Vec<u64>,
    /// Sparse row offsets, length `row_count + 1`.
    pub row_offsets: Vec<u64>,
    /// Concatenated sparse fingerprint indices.
    pub indices: Vec<u16>,
    /// Concatenated sparse fingerprint counts.
    pub counts: Vec<u16>,
    /// Flattened descriptor targets with shape `[rows, descriptor_width]`.
    pub descriptor_targets: Vec<f32>,
    /// Deterministic split, one per row.
    pub splits: Vec<DataSplit>,
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
        if fingerprint.fingerprint_size != self.fingerprint_size {
            return Err(Error::InvalidBatch(format!(
                "fingerprint width {} does not match shard width {}",
                fingerprint.fingerprint_size, self.fingerprint_size
            )));
        }
        if descriptor_targets.len() != self.descriptor_width {
            return Err(Error::InvalidBatch(format!(
                "descriptor width {} does not match shard width {}",
                descriptor_targets.len(),
                self.descriptor_width
            )));
        }
        if fingerprint.indices.len() != fingerprint.counts.len() {
            return Err(Error::InvalidBatch(
                "fingerprint indices and counts have different lengths".to_string(),
            ));
        }

        self.cids.push(cid);
        self.indices.extend_from_slice(&fingerprint.indices);
        self.counts.extend_from_slice(&fingerprint.counts);
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
                checked_mul(path, start_row, 8, "CID byte offset")?,
                "CID byte offset",
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

/// One shard entry in a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardManifestEntry {
    /// Shard path relative to the manifest.
    pub path: String,
    /// Number of rows in the shard.
    pub row_count: usize,
    /// Number of sparse nonzero entries.
    pub nnz: usize,
}

/// Preprocessing manifest for a cached PubChem run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardManifest {
    /// Manifest schema version.
    pub manifest_version: u32,
    /// Input source label or URL.
    pub source: String,
    /// Preprocessing configuration.
    pub preprocessing: PreprocessingConfig,
    /// Output shards.
    pub shards: Vec<ShardManifestEntry>,
    /// Total rows written.
    pub row_count: usize,
    /// Records skipped before writing.
    pub skipped_count: usize,
    /// Records that failed parsing or preprocessing.
    pub error_count: usize,
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
            checked_mul(path, row_count, 8, "CID section length")?,
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
        FingerprintTargets {
            indices: vec![3, 9],
            counts: vec![1, 4],
            fingerprint_size: 4096,
        }
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
    fn parses_pubchem_cid_smiles_lines() {
        let record = parse_pubchem_cid_smiles_line(7, "702\tCCO\n").expect("record");

        assert_eq!(record.cid, 702);
        assert_eq!(record.smiles, "CCO");
    }

    #[test]
    fn rejects_malformed_pubchem_records() {
        let missing_cid = parse_pubchem_cid_smiles_line(1, "\tCCO").expect_err("missing cid");
        assert!(matches!(
            missing_cid,
            Error::PubChemRecord {
                line_number: 1,
                message
            } if message == "missing CID"
        ));

        let invalid_cid = parse_pubchem_cid_smiles_line(2, "abc CCO").expect_err("invalid cid");
        assert!(matches!(
            invalid_cid,
            Error::PubChemRecord {
                line_number: 2,
                message
            } if message.contains("invalid CID")
        ));

        let missing_smiles = parse_pubchem_cid_smiles_line(3, "702").expect_err("missing smiles");
        assert!(matches!(
            missing_smiles,
            Error::PubChemRecord {
                line_number: 3,
                message
            } if message == "missing SMILES"
        ));
    }

    #[test]
    fn pubchem_iterator_skips_empty_lines() {
        let input = std::io::Cursor::new(b"\n702\tCCO\n");
        let mut iter = PubChemRecordIter::from_reader(input);

        assert_eq!(iter.next().expect("row").expect("valid").cid, 702);
        assert!(iter.next().is_none());
    }

    #[test]
    fn pubchem_iterator_reports_malformed_line_numbers() {
        let input = std::io::Cursor::new(b"\nabc CCO\n");
        let mut iter = PubChemRecordIter::from_reader(input);
        let error = iter.next().expect("row").expect_err("invalid row");

        assert!(matches!(
            error,
            Error::PubChemRecord {
                line_number: 2,
                message
            } if message.contains("invalid CID")
        ));
    }

    #[test]
    fn split_is_deterministic() {
        let config = PreprocessingConfig::default();

        assert_eq!(config.split_for_cid(123), config.split_for_cid(123));
    }

    #[test]
    fn preprocess_record_reports_invalid_smiles() {
        let config = PreprocessingConfig::default();
        let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
        let record = CidSmilesRecord {
            cid: 1,
            smiles: "(".to_string(),
        };
        let error = preprocess_record(&record, &config, &mut scratch).expect_err("invalid smiles");

        assert!(matches!(
            error,
            Error::SmilesParse {
                cid: 1,
                message
            } if !message.is_empty()
        ));
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn parallel_preprocessing_preserves_order_and_reports_record_errors() {
        let input = std::io::Cursor::new(b"702\tCCO\n703\t(\n704\tO\nbad C\n");
        let config = PreprocessingConfig::default();
        let mut chunks = Vec::new();

        let records_seen = preprocess_cid_smiles_record_chunks(
            PubChemRecordIter::from_reader(input),
            &config,
            PubChemPreprocessOptions {
                chunk_rows: 2,
                threads: Some(2),
            },
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
        assert_eq!(results[0].as_ref().expect("ethanol").cid, 702);
        assert!(matches!(
            &results[1],
            Err(Error::SmilesParse {
                cid: 703,
                message
            }) if !message.is_empty()
        ));
        assert_eq!(results[2].as_ref().expect("water").cid, 704);
        assert!(matches!(
            &results[3],
            Err(Error::PubChemRecord {
                line_number: 4,
                message
            }) if message.contains("invalid CID")
        ));
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn parallel_preprocessing_rejects_invalid_options() {
        let config = PreprocessingConfig::default();

        let zero_chunk_error = preprocess_cid_smiles_record_chunks(
            std::iter::empty::<Result<CidSmilesRecord>>(),
            &config,
            PubChemPreprocessOptions {
                chunk_rows: 0,
                threads: None,
            },
            |_| Ok(()),
        )
        .expect_err("zero chunk size should be rejected");
        assert!(matches!(
            zero_chunk_error,
            Error::InvalidBatch(message) if message.contains("chunk size")
        ));

        let zero_threads_error = preprocess_cid_smiles_record_chunks(
            std::iter::empty::<Result<CidSmilesRecord>>(),
            &config,
            PubChemPreprocessOptions {
                chunk_rows: 1,
                threads: Some(0),
            },
            |_| Ok(()),
        )
        .expect_err("zero thread count should be rejected");
        assert!(matches!(
            zero_threads_error,
            Error::InvalidBatch(message) if message.contains("threads")
        ));
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn pubchem_preprocessing_defaults_to_sixty_four_threads() {
        let options = PubChemPreprocessOptions::default();

        assert_eq!(options.chunk_rows, DEFAULT_PREPROCESS_CHUNK_ROWS);
        assert_eq!(options.threads, Some(DEFAULT_PREPROCESS_THREADS));
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
            let fingerprint = FingerprintTargets {
                indices: vec![row + 1, row + 11],
                counts: vec![row + 2, row + 3],
                fingerprint_size: 4096,
            };
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
        let wrong_width = FingerprintTargets {
            fingerprint_size: 2048,
            ..test_fingerprint()
        };
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

        let mismatched_sparse = FingerprintTargets {
            counts: vec![1],
            ..test_fingerprint()
        };
        let error = shard
            .push(
                702,
                &mismatched_sparse,
                &[0.0; REGRESSION_TARGET_WIDTH],
                DataSplit::Train,
            )
            .expect_err("sparse length mismatch");
        assert!(
            matches!(error, Error::InvalidBatch(message) if message.contains("different lengths"))
        );
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
        let mut manifest = ShardManifest::new("CID-SMILES.gz", config);
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
