//! Dataset preprocessing pipeline: SMILES → cached sparse shards + manifest.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use indicatif::{ProgressBar, ProgressStyle};
use molecular_autoencoder::{
    CountedEcfpConfig, DEFAULT_PREPROCESS_CHUNK_ROWS, DatasetPreprocessOptions, MoleculeRecord,
    PreprocessingConfig, SHARD_MANIFEST_VERSION, ShardManifest, SmilesQualityFilter,
    SparseMoleculeShard, features::REGRESSION_TARGET_WIDTH, molecule_records_from_smiles_dataset,
    preprocess_dataset_record_chunks,
};
use smiles_parser::prelude::{
    DatasetCollectionSource, DatasetFetchOptions, DatasetSource, GzipMode, PUBCHEM_SMILES,
    SmilesDatasetRecordSource, Zinc20Smiles,
};

use crate::{
    AppResult,
    cli::{Args, ResolvedPaths, SourceSelection},
    invalid_input,
};

/// Resolves the manifest path the training loop should read, building cached
/// shards from scratch when the cache is missing, stale, or `--force-preprocess`
/// is set.
pub fn prepare_manifest(args: &Args, paths: &ResolvedPaths) -> AppResult<PathBuf> {
    let Some(cache_dir) = paths.cache_dir.as_ref() else {
        return Ok(paths.manifest_path.clone());
    };
    let Some(source_selection) = paths.source_selection else {
        return Ok(paths.manifest_path.clone());
    };
    let manifest_path = cache_dir.join("manifest.json");
    let expected_source = source_selection.manifest_source();

    if manifest_path.exists() && !args.force_preprocess {
        let manifest = ShardManifest::read_from_path(&manifest_path)?;
        if manifest.manifest_version() == SHARD_MANIFEST_VERSION
            && manifest.source() == expected_source
        {
            println!(
                "using_cached_manifest={} source={}",
                manifest_path.display(),
                manifest.source()
            );
            return Ok(manifest_path);
        }
        if manifest.manifest_version() == SHARD_MANIFEST_VERSION {
            println!(
                "ignoring_cached_manifest={} reason=source_mismatch found={} expected={}",
                manifest_path.display(),
                manifest.source(),
                expected_source
            );
        } else {
            println!(
                "ignoring_cached_manifest={} reason=manifest_version_mismatch found={} expected={}",
                manifest_path.display(),
                manifest.manifest_version(),
                SHARD_MANIFEST_VERSION
            );
        }
    }

    preprocess_cache(
        cache_dir,
        args.rows_per_shard,
        args.preprocess_threads,
        source_selection,
        args.quality_filter()?,
        args.ecfp_radius,
        args.negatives_per_positive,
    )?;
    Ok(manifest_path)
}

/// Streams the selected dataset through the preprocessor and writes one or
/// more shard files plus a manifest under `cache_dir`.
fn preprocess_cache(
    cache_dir: &Path,
    rows_per_shard: usize,
    preprocess_threads: usize,
    source_selection: SourceSelection,
    quality_filter: SmilesQualityFilter,
    ecfp_radius: u8,
    negatives_per_positive: f32,
) -> AppResult<()> {
    if rows_per_shard == 0 {
        return Err(invalid_input("rows per shard must be greater than zero"));
    }

    std::fs::create_dir_all(cache_dir)?;
    let source_cache_dir = cache_dir.join("source");
    let counted_ecfp = CountedEcfpConfig::builder()
        .radius(ecfp_radius)
        .build()
        .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;
    let config = PreprocessingConfig::builder()
        .counted_ecfp(counted_ecfp)
        .quality_filter(quality_filter)
        .negatives_per_positive(negatives_per_positive)
        .build()
        .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;
    let mut manifest = ShardManifest::new(source_selection.manifest_source(), config.clone());
    let mut shard = SparseMoleculeShard::new(config.counted_ecfp().size(), REGRESSION_TARGET_WIDTH);
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
            SourceSelection::Zinc20(range) => {
                context.preprocess_zinc20_source(range.first, range.last)?;
            }
            SourceSelection::PubChemAndZinc20(range) => {
                context.preprocess_pubchem_source()?;
                context.preprocess_zinc20_source(range.first, range.last)?;
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

/// Shared mutable state threaded through every source-specific preprocess
/// routine.
struct PreprocessContext<'a> {
    /// Directory where shard files and the manifest are written.
    cache_dir: &'a Path,
    /// Directory used to cache raw dataset downloads.
    source_cache_dir: &'a Path,
    /// Maximum number of rows per shard before flushing.
    rows_per_shard: usize,
    /// Thread count for the parallel chunk preprocessor.
    preprocess_threads: usize,
    /// Preprocessing configuration (ECFP / descriptors / filter).
    config: &'a PreprocessingConfig,
    /// Manifest accumulator updated as each shard is written.
    manifest: &'a mut ShardManifest,
    /// Current in-flight shard buffer being filled with rows.
    shard: &'a mut SparseMoleculeShard,
    /// Counter for the next shard filename / index.
    shard_index: &'a mut usize,
    /// Progress bar used to surface live preprocessing stats.
    progress: &'a PreprocessProgress,
    /// Total records observed across all sources for the run.
    records_seen_total: &'a mut usize,
}

impl PreprocessContext<'_> {
    /// Streams the PubChem dataset through [`Self::preprocess_records`].
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

    /// Streams the requested ZINC20 chunk range through
    /// [`Self::preprocess_records`].
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

    /// Pulls records from `records`, applies the parallel chunk preprocessor,
    /// and writes shards as they fill.
    fn preprocess_records<I>(&mut self, dataset_label: &str, records: I) -> AppResult<()>
    where
        I: IntoIterator<Item = molecular_autoencoder::Result<MoleculeRecord>>,
    {
        let source_records_seen = preprocess_dataset_record_chunks(
            records,
            self.config,
            DatasetPreprocessOptions::builder()
                .chunk_rows(DEFAULT_PREPROCESS_CHUNK_ROWS)
                .threads(Some(self.preprocess_threads))
                .build()
                .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?,
            |chunk| {
                let records_seen = chunk.records_seen();
                for result in chunk.into_results() {
                    match result {
                        Ok(rows) if rows.is_empty() => {
                            self.manifest.record_skipped();
                        }
                        Ok(rows) => {
                            for targets in &rows {
                                self.shard
                                    .push_targets(targets, *self.config.descriptors())?;
                                if self.shard.len() >= self.rows_per_shard {
                                    write_preprocessed_shard(
                                        self.cache_dir,
                                        *self.shard_index,
                                        self.manifest,
                                        self.shard,
                                    )?;
                                    self.progress.shard_written(self.manifest, 0);
                                    *self.shard = SparseMoleculeShard::new(
                                        self.config.counted_ecfp().size(),
                                        REGRESSION_TARGET_WIDTH,
                                    );
                                    *self.shard_index += 1;
                                }
                            }
                        }
                        Err(error) => {
                            self.manifest.record_error();
                            eprintln!("{error}");
                        }
                    }
                }
                self.progress.record(
                    dataset_label,
                    *self.records_seen_total + records_seen,
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

/// Writes the in-flight shard to disk under `cache_dir` and pushes the
/// resulting entry into the manifest.
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

/// Indicatif spinner that streams live counters during preprocessing.
struct PreprocessProgress {
    /// Underlying spinner driving the live status line.
    bar: ProgressBar,
}

impl PreprocessProgress {
    /// Initializes the spinner with the dataset description and a steady
    /// tick.
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

    /// Updates the spinner with a periodic snapshot of records / rows /
    /// errors.
    fn record(
        &self,
        dataset_label: &str,
        records_seen: usize,
        manifest: &ShardManifest,
        current_shard_rows: usize,
    ) {
        if records_seen == 1 || records_seen.is_multiple_of(1024) {
            self.bar.set_message(format!(
                "preprocessing {dataset_label} records={} rows={} skipped={} errors={} shards={} current_shard_rows={}",
                records_seen,
                manifest.row_count() + current_shard_rows,
                manifest.skipped_count(),
                manifest.error_count(),
                manifest.shards().len(),
                current_shard_rows
            ));
        }
    }

    /// Updates the spinner message after a shard has been flushed to disk.
    fn shard_written(&self, manifest: &ShardManifest, current_shard_rows: usize) {
        self.bar.set_message(format!(
            "wrote shard={} rows={} errors={} next_shard_rows={}",
            manifest.shards().len(),
            manifest.row_count(),
            manifest.error_count(),
            current_shard_rows
        ));
    }

    /// Finalizes the spinner with the run summary line.
    fn finish(self, records_seen: usize, manifest: &ShardManifest, elapsed: Duration) {
        self.bar.finish_with_message(format!(
            "preprocess_done records_seen={} rows={} skipped={} errors={} shards={} elapsed_sec={:.2}",
            records_seen,
            manifest.row_count(),
            manifest.skipped_count(),
            manifest.error_count(),
            manifest.shards().len(),
            elapsed.as_secs_f64()
        ));
    }
}

/// Builds an indicatif spinner with the project-standard style and tick rate.
fn spinner(message: impl Into<String>) -> ProgressBar {
    let bar = ProgressBar::new_spinner();
    bar.set_style(spinner_style());
    bar.enable_steady_tick(Duration::from_millis(100));
    bar.set_message(message.into());
    bar
}

/// Shared indicatif spinner template used by every preprocessor spinner.
fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} {elapsed_precise} {wide_msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Args, ResolvedPaths, Zinc20ChunkRange};
    use clap::Parser;
    use molecular_autoencoder::{DataSplit, FingerprintTargets};

    const FINGERPRINT_WIDTH: usize = 4096;

    fn args_with(checkpoint_dir: &Path, shards: &Path) -> Args {
        Args::try_parse_from([
            "train",
            shards.to_str().expect("utf-8"),
            checkpoint_dir.to_str().expect("utf-8"),
        ])
        .expect("parse")
    }

    fn fixture_shard() -> SparseMoleculeShard {
        let mut shard = SparseMoleculeShard::new(FINGERPRINT_WIDTH, REGRESSION_TARGET_WIDTH);
        let fingerprint = FingerprintTargets::new(vec![3, 11], vec![1, 2], FINGERPRINT_WIDTH)
            .expect("valid fingerprint");
        shard
            .push(
                17,
                &fingerprint,
                &[0.0; REGRESSION_TARGET_WIDTH],
                DataSplit::Train,
                0,
                None,
            )
            .expect("push row");
        shard
    }

    #[test]
    fn prepare_manifest_short_circuits_when_no_cache_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = dir.path().join("custom.json");
        std::fs::write(&manifest_path, "{}").expect("write placeholder");
        let args = args_with(&dir.path().join("runs"), &manifest_path);
        let resolved = ResolvedPaths {
            manifest_path: manifest_path.clone(),
            cache_dir: None,
            source_selection: Some(SourceSelection::PubChem),
        };

        let result = prepare_manifest(&args, &resolved).expect("prepare manifest");
        assert_eq!(result, manifest_path);
    }

    #[test]
    fn prepare_manifest_short_circuits_when_no_source_selection() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = dir.path().join("manifest.json");
        let args = args_with(&dir.path().join("runs"), dir.path());
        let resolved = ResolvedPaths {
            manifest_path: manifest_path.clone(),
            cache_dir: Some(dir.path().to_path_buf()),
            source_selection: None,
        };

        let result = prepare_manifest(&args, &resolved).expect("prepare manifest");
        assert_eq!(result, manifest_path);
    }

    #[test]
    fn prepare_manifest_reuses_matching_cached_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = dir.path().join("manifest.json");

        let mut manifest = ShardManifest::new(
            SourceSelection::PubChem.manifest_source(),
            PreprocessingConfig::default(),
        );
        manifest.push_shard("shard-00000.maeshard", &fixture_shard());
        manifest
            .write_to_path(&manifest_path)
            .expect("write manifest");

        let args = args_with(&dir.path().join("runs"), dir.path());
        let resolved = ResolvedPaths {
            manifest_path: manifest_path.clone(),
            cache_dir: Some(dir.path().to_path_buf()),
            source_selection: Some(SourceSelection::PubChem),
        };

        let result = prepare_manifest(&args, &resolved).expect("prepare manifest");
        assert_eq!(result, manifest_path);
    }

    #[test]
    fn prepare_manifest_logs_source_mismatch_warning_before_rebuilding() {
        // We can't run the full preprocessing pipeline (it would download
        // PubChem), so we just exercise the source-mismatch detection up to
        // the point where it would call preprocess_cache. That path triggers
        // preprocessing, which we short-circuit by forcing `rows_per_shard=0`.
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = dir.path().join("manifest.json");

        let mut manifest = ShardManifest::new(
            // Wrong source: cached PubChem, requesting Zinc20.
            SourceSelection::PubChem.manifest_source(),
            PreprocessingConfig::default(),
        );
        manifest.push_shard("shard-00000.maeshard", &fixture_shard());
        manifest
            .write_to_path(&manifest_path)
            .expect("write manifest");

        let mut args = args_with(&dir.path().join("runs"), dir.path());
        args.rows_per_shard = 0; // force preprocess_cache to bail before downloads
        let resolved = ResolvedPaths {
            manifest_path,
            cache_dir: Some(dir.path().to_path_buf()),
            source_selection: Some(SourceSelection::Zinc20(Zinc20ChunkRange {
                first: 1,
                last: 2,
            })),
        };

        let err = prepare_manifest(&args, &resolved)
            .expect_err("preprocessing aborts before any network fetch");
        assert!(err.to_string().contains("rows per shard"), "got: {err}");
    }

    #[test]
    fn prepare_manifest_force_preprocess_bypasses_cache_check() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = dir.path().join("manifest.json");

        let mut manifest = ShardManifest::new(
            SourceSelection::PubChem.manifest_source(),
            PreprocessingConfig::default(),
        );
        manifest.push_shard("shard-00000.maeshard", &fixture_shard());
        manifest
            .write_to_path(&manifest_path)
            .expect("write manifest");

        let mut args = args_with(&dir.path().join("runs"), dir.path());
        args.force_preprocess = true;
        args.rows_per_shard = 0; // halt preprocess_cache before any IO
        let resolved = ResolvedPaths {
            manifest_path,
            cache_dir: Some(dir.path().to_path_buf()),
            source_selection: Some(SourceSelection::PubChem),
        };

        let err = prepare_manifest(&args, &resolved)
            .expect_err("force_preprocess + rows_per_shard=0 must trigger preprocess_cache");
        assert!(err.to_string().contains("rows per shard"));
    }

    #[test]
    fn write_preprocessed_shard_appends_entry_and_writes_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut manifest = ShardManifest::new(
            SourceSelection::PubChem.manifest_source(),
            PreprocessingConfig::default(),
        );
        let shard = fixture_shard();

        write_preprocessed_shard(dir.path(), 0, &mut manifest, &shard).expect("write");

        assert_eq!(manifest.shards().len(), 1);
        let entry = &manifest.shards()[0];
        assert_eq!(entry.path(), "shard-00000.maeshard");
        assert_eq!(entry.row_count(), shard.len());
        assert!(dir.path().join("shard-00000.maeshard").exists());
    }

    #[test]
    fn preprocess_progress_message_helpers_are_callable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let progress = PreprocessProgress::new(dir.path(), SourceSelection::PubChem, 64, 1);
        let manifest = ShardManifest::new(
            SourceSelection::PubChem.manifest_source(),
            PreprocessingConfig::default(),
        );
        progress.record("PubChem", 1, &manifest, 0);
        progress.record("PubChem", 1024, &manifest, 8);
        progress.shard_written(&manifest, 0);
        progress.finish(1024, &manifest, Duration::from_millis(10));
    }

    #[test]
    fn spinner_returns_a_progress_bar_with_message() {
        let bar = spinner("hello");
        assert_eq!(bar.message(), "hello");
        bar.finish_and_clear();
    }
}
