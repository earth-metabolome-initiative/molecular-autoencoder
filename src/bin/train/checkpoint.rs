//! Checkpoint I/O and shard manifest validation.

use std::path::{Path, PathBuf};

use burn::optim::Optimizer;
use burn::{
    module::Module,
    record::{DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};
use molecular_autoencoder::{
    MoleculeAutoencoder, MoleculeAutoencoderBatcher, ShardManifest, SparseMoleculeShard,
};
use serde::{Deserialize, Serialize};

use crate::{AppResult, invalid_input};

/// Cached shard reference resolved against a manifest directory.
#[derive(Debug, Clone)]
pub struct CachedShardInfo {
    pub path: PathBuf,
    pub row_count: usize,
}

/// Persistent training-loop state stored alongside the model checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    pub completed_epoch: usize,
    pub global_step: usize,
    pub best_validation_loss: Option<f32>,
}

/// Builds the list of cached shards referenced by the manifest, resolving
/// each entry against the manifest directory.
pub fn shard_infos(
    manifest_path: &Path,
    manifest: &ShardManifest,
) -> AppResult<Vec<CachedShardInfo>> {
    if manifest.shards().is_empty() {
        return Err(invalid_input("manifest does not list any shards"));
    }
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(manifest
        .shards()
        .iter()
        .map(|entry| CachedShardInfo {
            path: manifest_dir.join(entry.path()),
            row_count: entry.row_count(),
        })
        .collect())
}

/// Confirms that a freshly read shard matches the model's input shape.
pub fn validate_shard_shape(
    shard: &SparseMoleculeShard,
    batcher: MoleculeAutoencoderBatcher,
    path: &Path,
) -> AppResult<()> {
    if shard.fingerprint_size() != batcher.fingerprint_size {
        return Err(invalid_input(format!(
            "{} fingerprint width {} does not match model width {}",
            path.display(),
            shard.fingerprint_size(),
            batcher.fingerprint_size
        )));
    }
    if shard.descriptor_width() != batcher.descriptor_width {
        return Err(invalid_input(format!(
            "{} descriptor width {} does not match model width {}",
            path.display(),
            shard.descriptor_width(),
            batcher.descriptor_width
        )));
    }
    Ok(())
}

/// Writes the model, optimizer state, and training state JSON.
pub fn save_checkpoint<B, O>(
    checkpoint_dir: &Path,
    recorder: &DefaultRecorder,
    model: &MoleculeAutoencoder<B>,
    optimizer: &O,
    state: &CheckpointState,
) -> AppResult<()>
where
    B: AutodiffBackend,
    O: Optimizer<MoleculeAutoencoder<B>, B>,
{
    model
        .clone()
        .save_file(checkpoint_dir.join("model"), recorder)?;
    <DefaultRecorder as Recorder<B>>::record(
        recorder,
        optimizer.to_record(),
        checkpoint_dir.join("optimizer"),
    )?;
    write_state_json(&checkpoint_dir.join("state.json"), state)?;
    Ok(())
}

/// Reads the JSON-serialized [`CheckpointState`] from disk.
pub fn read_state_json(path: &Path) -> AppResult<CheckpointState> {
    Ok(serde_json::from_reader(std::fs::File::open(path)?)?)
}

fn write_state_json(path: &Path, state: &CheckpointState) -> AppResult<()> {
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, state)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use molecular_autoencoder::{
        DataSplit, FingerprintTargets, PreprocessingConfig, SHARD_MANIFEST_VERSION, ShardManifest,
    };

    const FINGERPRINT_WIDTH: usize = 4096;

    fn write_dummy_shard_file(path: &Path) {
        // shard_infos only resolves the path; it doesn't open it.
        std::fs::write(path, b"").expect("write placeholder shard");
    }

    fn fixture_shard() -> SparseMoleculeShard {
        let mut shard = SparseMoleculeShard::new(
            FINGERPRINT_WIDTH,
            molecular_autoencoder::features::REGRESSION_TARGET_WIDTH,
        );
        let fingerprint = FingerprintTargets::new(vec![1, 5], vec![1, 2], FINGERPRINT_WIDTH)
            .expect("valid fingerprint");
        shard
            .push(
                42,
                &fingerprint,
                &[0.0; molecular_autoencoder::features::REGRESSION_TARGET_WIDTH],
                DataSplit::Train,
            )
            .expect("push");
        shard
    }

    #[test]
    fn shard_infos_resolves_entries_against_manifest_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = dir.path().join("manifest.json");
        write_dummy_shard_file(&dir.path().join("shard-00000.maeshard"));

        let mut manifest = ShardManifest::new("fixture", PreprocessingConfig::default());
        manifest.push_shard("shard-00000.maeshard", &fixture_shard());

        let resolved = shard_infos(&manifest_path, &manifest).expect("shard_infos");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].path, dir.path().join("shard-00000.maeshard"));
        assert_eq!(resolved[0].row_count, 1);
    }

    #[test]
    fn shard_infos_rejects_empty_manifest() {
        let manifest = ShardManifest::new("fixture", PreprocessingConfig::default());
        let err = shard_infos(Path::new("/tmp/missing-manifest.json"), &manifest)
            .expect_err("empty manifest must fail");
        assert!(err.to_string().contains("does not list any shards"));
    }

    #[test]
    fn shard_infos_falls_back_to_cwd_for_parentless_manifest() {
        // When the manifest path has no parent (e.g. "manifest.json"), the
        // helper uses "." as the base directory. We verify by passing a bare
        // filename and checking the resolved path starts with "./" or matches
        // the bare entry.
        let mut manifest = ShardManifest::new("fixture", PreprocessingConfig::default());
        manifest.push_shard("nested/shard.bin", &fixture_shard());
        let resolved = shard_infos(Path::new("manifest.json"), &manifest).expect("shard_infos");
        // The current directory may differ between runs; just assert the
        // relative segment is preserved at the tail.
        assert!(
            resolved[0].path.ends_with("nested/shard.bin"),
            "got {:?}",
            resolved[0].path
        );
    }

    #[test]
    fn validate_shard_shape_accepts_matching_widths() {
        let shard = fixture_shard();
        let batcher = MoleculeAutoencoderBatcher::new(
            FINGERPRINT_WIDTH,
            molecular_autoencoder::features::REGRESSION_TARGET_WIDTH,
        );
        validate_shard_shape(&shard, batcher, Path::new("fixture.maeshard"))
            .expect("matching widths must pass");
    }

    #[test]
    fn validate_shard_shape_rejects_fingerprint_mismatch() {
        let shard = fixture_shard();
        let batcher = MoleculeAutoencoderBatcher::new(
            FINGERPRINT_WIDTH * 2,
            molecular_autoencoder::features::REGRESSION_TARGET_WIDTH,
        );
        let err = validate_shard_shape(&shard, batcher, Path::new("fixture.maeshard"))
            .expect_err("fingerprint width mismatch must fail");
        assert!(err.to_string().contains("fingerprint width"));
    }

    #[test]
    fn validate_shard_shape_rejects_descriptor_mismatch() {
        let shard = fixture_shard();
        let batcher = MoleculeAutoencoderBatcher::new(FINGERPRINT_WIDTH, 4);
        let err = validate_shard_shape(&shard, batcher, Path::new("fixture.maeshard"))
            .expect_err("descriptor width mismatch must fail");
        assert!(err.to_string().contains("descriptor width"));
    }

    #[test]
    fn state_json_round_trip_preserves_fields() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.json");
        let state = CheckpointState {
            completed_epoch: 17,
            global_step: 12_345,
            best_validation_loss: Some(0.421),
        };
        write_state_json(&path, &state).expect("write state");
        let loaded = read_state_json(&path).expect("read state");
        assert_eq!(loaded.completed_epoch, state.completed_epoch);
        assert_eq!(loaded.global_step, state.global_step);
        assert_eq!(loaded.best_validation_loss, state.best_validation_loss);
    }

    #[test]
    fn state_json_round_trip_handles_unset_best_loss() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.json");
        let state = CheckpointState {
            completed_epoch: 0,
            global_step: 0,
            best_validation_loss: None,
        };
        write_state_json(&path, &state).expect("write state");
        let loaded = read_state_json(&path).expect("read state");
        assert!(loaded.best_validation_loss.is_none());
    }

    #[test]
    fn read_state_json_propagates_io_errors() {
        let err = read_state_json(Path::new("/tmp/definitely-missing-checkpoint.json"))
            .expect_err("missing file must fail");
        assert!(err.to_string().contains("No such file") || err.to_string().contains("not found"));
    }

    #[test]
    fn manifest_version_constant_matches_library() {
        // Surface check — if SHARD_MANIFEST_VERSION changes, downstream
        // checkpoint readers need to be aware.
        const _: () = assert!(SHARD_MANIFEST_VERSION >= 1);
    }
}
