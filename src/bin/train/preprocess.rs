//! Dataset preprocessing pipeline: SMILES → cached sparse shards + manifest.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use indicatif::{ProgressBar, ProgressStyle};
use molecular_autoencoder::{
    DEFAULT_PREPROCESS_CHUNK_ROWS, DatasetPreprocessOptions, MoleculeRecord, PreprocessingConfig,
    SHARD_MANIFEST_VERSION, ShardManifest, SparseMoleculeShard,
    features::REGRESSION_TARGET_WIDTH, molecule_records_from_smiles_dataset,
    preprocess_dataset_record_chunks,
};
use smiles_parser::prelude::{
    DatasetCollectionSource, DatasetFetchOptions, DatasetSource, GzipMode, PUBCHEM_SMILES,
    SmilesDatasetRecordSource, Zinc20Smiles,
};

use crate::{
    AppResult,
    cli::{Args, SourceSelection},
    invalid_input,
};

/// Resolves the manifest path the training loop should read, building cached
/// shards from scratch when the cache is missing, stale, or `--force-preprocess`
/// is set.
pub fn prepare_manifest(args: &Args) -> AppResult<PathBuf> {
    let Some(cache_dir) = &args.cache_dir else {
        return Ok(args.manifest_path.clone());
    };
    let Some(source_selection) = args.source_selection else {
        return Ok(args.manifest_path.clone());
    };
    let manifest_path = cache_dir.join("manifest.json");
    let expected_source = source_selection.manifest_source();

    if manifest_path.exists() && !args.force_preprocess {
        let manifest = ShardManifest::read_from_path(&manifest_path)?;
        if manifest.manifest_version == SHARD_MANIFEST_VERSION && manifest.source == expected_source
        {
            println!(
                "using_cached_manifest={} source={}",
                manifest_path.display(),
                manifest.source
            );
            return Ok(manifest_path);
        }
        if manifest.manifest_version == SHARD_MANIFEST_VERSION {
            println!(
                "ignoring_cached_manifest={} reason=source_mismatch found={} expected={}",
                manifest_path.display(),
                manifest.source,
                expected_source
            );
        } else {
            println!(
                "ignoring_cached_manifest={} reason=manifest_version_mismatch found={} expected={}",
                manifest_path.display(),
                manifest.manifest_version,
                SHARD_MANIFEST_VERSION
            );
        }
    }

    preprocess_cache(
        cache_dir,
        args.rows_per_shard,
        args.preprocess_threads,
        source_selection,
    )?;
    Ok(manifest_path)
}

fn preprocess_cache(
    cache_dir: &Path,
    rows_per_shard: usize,
    preprocess_threads: usize,
    source_selection: SourceSelection,
) -> AppResult<()> {
    if rows_per_shard == 0 {
        return Err(invalid_input("rows per shard must be greater than zero"));
    }

    std::fs::create_dir_all(cache_dir)?;
    let source_cache_dir = cache_dir.join("source");
    let config = PreprocessingConfig::default();
    let mut manifest = ShardManifest::new(source_selection.manifest_source(), config.clone());
    let mut shard = SparseMoleculeShard::new(config.counted_ecfp.size, REGRESSION_TARGET_WIDTH);
    let mut shard_index = 0_usize;
    let mut records_seen_total = 0_usize;
    let start = Instant::now();

    println!(
        "preprocess_start source={} cache_dir={} rows_per_shard={} chunk_rows={} threads={}",
        source_selection.manifest_source(),
        cache_dir.display(),
        rows_per_shard,
        DEFAULT_PREPROCESS_CHUNK_ROWS,
        preprocess_threads
    );

    let progress = PreprocessProgress::new(
        cache_dir,
        source_selection,
        rows_per_shard,
        preprocess_threads,
    );
    {
        let mut context = PreprocessContext {
            cache_dir,
            source_cache_dir: &source_cache_dir,
            rows_per_shard,
            preprocess_threads,
            config: &config,
            manifest: &mut manifest,
            shard: &mut shard,
            shard_index: &mut shard_index,
            progress: &progress,
            records_seen_total: &mut records_seen_total,
        };
        match source_selection {
            SourceSelection::PubChem => context.preprocess_pubchem_source()?,
            SourceSelection::Zinc20 {
                first_chunk,
                last_chunk,
            } => context.preprocess_zinc20_source(first_chunk, last_chunk)?,
            SourceSelection::PubChemAndZinc20 {
                first_chunk,
                last_chunk,
            } => {
                context.preprocess_pubchem_source()?;
                context.preprocess_zinc20_source(first_chunk, last_chunk)?;
            }
        }
    }
    if !shard.is_empty() {
        write_preprocessed_shard(cache_dir, shard_index, &mut manifest, &shard)?;
        progress.shard_written(&manifest, 0);
    }
    manifest.write_to_path(cache_dir.join("manifest.json"))?;
    progress.finish(records_seen_total, &manifest, start.elapsed());
    Ok(())
}

struct PreprocessContext<'a> {
    cache_dir: &'a Path,
    source_cache_dir: &'a Path,
    rows_per_shard: usize,
    preprocess_threads: usize,
    config: &'a PreprocessingConfig,
    manifest: &'a mut ShardManifest,
    shard: &'a mut SparseMoleculeShard,
    shard_index: &'a mut usize,
    progress: &'a PreprocessProgress,
    records_seen_total: &'a mut usize,
}

impl PreprocessContext<'_> {
    fn preprocess_pubchem_source(&mut self) -> AppResult<()> {
        let source_progress = spinner(format!(
            "opening PubChem records cache at {}",
            self.source_cache_dir.display()
        ));
        let records = PUBCHEM_SMILES.iter_records_with_options(&DatasetFetchOptions {
            cache_dir: Some(self.source_cache_dir.to_path_buf()),
            gzip_mode: GzipMode::KeepCompressed,
            ..DatasetFetchOptions::default()
        })?;
        source_progress.finish_with_message(format!(
            "PubChem records ready source={}",
            PUBCHEM_SMILES.url()
        ));
        let records = molecule_records_from_smiles_dataset(PUBCHEM_SMILES.id(), records);
        self.preprocess_records("PubChem", records)
    }

    fn preprocess_zinc20_source(&mut self, first_chunk: u8, last_chunk: u8) -> AppResult<()> {
        let dataset = Zinc20Smiles::chunk_range(first_chunk, last_chunk)?;
        let label = format!("ZINC20 chunks {first_chunk}..={last_chunk}");
        let source_progress = spinner(format!(
            "opening {label} records cache at {}",
            self.source_cache_dir.display()
        ));
        let records = dataset.iter_records_with_options(&DatasetFetchOptions {
            cache_dir: Some(self.source_cache_dir.to_path_buf()),
            gzip_mode: GzipMode::Decompress,
            ..DatasetFetchOptions::default()
        })?;
        source_progress.finish_with_message(format!("{label} records ready"));
        let records = molecule_records_from_smiles_dataset(dataset.id(), records);
        self.preprocess_records(&label, records)
    }

    fn preprocess_records<I>(&mut self, dataset_label: &str, records: I) -> AppResult<()>
    where
        I: IntoIterator<Item = molecular_autoencoder::Result<MoleculeRecord>>,
    {
        let source_records_seen = preprocess_dataset_record_chunks(
            records,
            self.config,
            DatasetPreprocessOptions {
                chunk_rows: DEFAULT_PREPROCESS_CHUNK_ROWS,
                threads: Some(self.preprocess_threads),
            },
            |chunk| {
                for result in chunk.results {
                    match result {
                        Ok(targets) => {
                            self.shard.push_targets(&targets, self.config.descriptors)?;
                            if self.shard.len() >= self.rows_per_shard {
                                write_preprocessed_shard(
                                    self.cache_dir,
                                    *self.shard_index,
                                    self.manifest,
                                    self.shard,
                                )?;
                                self.progress.shard_written(self.manifest, 0);
                                *self.shard = SparseMoleculeShard::new(
                                    self.config.counted_ecfp.size,
                                    REGRESSION_TARGET_WIDTH,
                                );
                                *self.shard_index += 1;
                            }
                        }
                        Err(error) => {
                            self.manifest.error_count += 1;
                            eprintln!("{error}");
                        }
                    }
                }
                self.progress.record(
                    dataset_label,
                    *self.records_seen_total + chunk.records_seen,
                    self.manifest,
                    self.shard.len(),
                );
                Ok(())
            },
        )?;
        *self.records_seen_total += source_records_seen;
        Ok(())
    }
}

fn write_preprocessed_shard(
    cache_dir: &Path,
    shard_index: usize,
    manifest: &mut ShardManifest,
    shard: &SparseMoleculeShard,
) -> molecular_autoencoder::Result<()> {
    let filename = format!("shard-{shard_index:05}.maeshard");
    let path = cache_dir.join(&filename);
    shard.write_to_path(&path)?;
    manifest.push_shard(filename, shard);
    Ok(())
}

struct PreprocessProgress {
    bar: ProgressBar,
}

impl PreprocessProgress {
    fn new(
        cache_dir: &Path,
        source_selection: SourceSelection,
        rows_per_shard: usize,
        threads: usize,
    ) -> Self {
        let bar = spinner(format!(
            "preprocessing {} into {} rows_per_shard={rows_per_shard} chunk_rows={} threads={}",
            source_selection.manifest_source(),
            cache_dir.display(),
            DEFAULT_PREPROCESS_CHUNK_ROWS,
            threads
        ));
        Self { bar }
    }

    fn record(
        &self,
        dataset_label: &str,
        records_seen: usize,
        manifest: &ShardManifest,
        current_shard_rows: usize,
    ) {
        if records_seen == 1 || records_seen.is_multiple_of(1024) {
            self.bar.set_message(format!(
                "preprocessing {dataset_label} records={} rows={} errors={} shards={} current_shard_rows={}",
                records_seen,
                manifest.row_count + current_shard_rows,
                manifest.error_count,
                manifest.shards.len(),
                current_shard_rows
            ));
        }
    }

    fn shard_written(&self, manifest: &ShardManifest, current_shard_rows: usize) {
        self.bar.set_message(format!(
            "wrote shard={} rows={} errors={} next_shard_rows={}",
            manifest.shards.len(),
            manifest.row_count,
            manifest.error_count,
            current_shard_rows
        ));
    }

    fn finish(self, records_seen: usize, manifest: &ShardManifest, elapsed: Duration) {
        self.bar.finish_with_message(format!(
            "preprocess_done records_seen={} rows={} errors={} shards={} elapsed_sec={:.2}",
            records_seen,
            manifest.row_count,
            manifest.error_count,
            manifest.shards.len(),
            elapsed.as_secs_f64()
        ));
    }
}

fn spinner(message: impl Into<String>) -> ProgressBar {
    let bar = ProgressBar::new_spinner();
    bar.set_style(spinner_style());
    bar.enable_steady_tick(Duration::from_millis(100));
    bar.set_message(message.into());
    bar
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} {elapsed_precise} {wide_msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
}
