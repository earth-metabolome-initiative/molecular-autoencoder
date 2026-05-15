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
