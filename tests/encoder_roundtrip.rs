//! Direct coverage for the `MoleculeEncoder` inference API.
//!
//! Complements `embed_pipeline.rs` (which only runs when the embed-parquet
//! feature is enabled) by exercising the encoder under a minimal `ndarray`
//! matrix — this is the same surface a future WASM inference build will use.

#![cfg(feature = "ndarray")]

use std::path::Path;

use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use burn::record::DefaultRecorder;
use molecular_autoencoder::{Error, MoleculeAutoencoderConfig, MoleculeEncoder};

type B = burn::backend::NdArray<f32, i64>;

const FINGERPRINT_SIZE: usize = 4096;
const LATENT_WIDTH: usize = 16;

fn save_tiny_checkpoint(dir: &Path) -> MoleculeAutoencoderConfig {
    let config = MoleculeAutoencoderConfig::symmetric(FINGERPRINT_SIZE, LATENT_WIDTH, vec![32]);
    config
        .save_json(dir.join("model-config.json"))
        .expect("save config");
    let device = NdArrayDevice::default();
    let model = config.init::<B>(&device);
    model
        .save_file(dir.join("model"), &DefaultRecorder::default())
        .expect("save model");
    config
}

fn load_encoder(dir: &Path) -> MoleculeEncoder<B> {
    MoleculeEncoder::<B>::from_checkpoint(dir, NdArrayDevice::default()).expect("encoder load")
}

#[test]
fn encoder_round_trips_a_small_batch() {
    let dir = tempfile::tempdir().expect("tempdir");
    save_tiny_checkpoint(dir.path());
    let mut encoder = load_encoder(dir.path());

    let smiles: Vec<String> = ["CCO", "c1ccccc1", "O=C=O"]
        .iter()
        .map(ToString::to_string)
        .collect();
    let rows = encoder.encode(&smiles).expect("encode");

    assert_eq!(rows.len(), smiles.len());
    for (input, slot) in smiles.iter().zip(rows.iter()) {
        let row = slot.as_ref().expect("valid smiles must yield a row");
        assert_eq!(row.smiles, *input);
        assert_eq!(row.latent.len(), LATENT_WIDTH);
        assert!(row.latent.iter().all(|v| v.is_finite()));
        assert!((0.0..=1.0).contains(&row.reconstruction_count_tanimoto));
        assert!((0.0..=1.0).contains(&row.reconstruction_binary_tanimoto));
        assert!(row.reconstruction_log_mse.is_finite());
        assert!(row.reconstruction_log_mse >= 0.0);
    }
}

#[test]
fn encoder_reports_invalid_smiles_as_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    save_tiny_checkpoint(dir.path());
    let mut encoder = load_encoder(dir.path());

    let smiles: Vec<String> = ["CCO", "definitely not a valid smiles!@#", "c1ccccc1"]
        .iter()
        .map(ToString::to_string)
        .collect();
    let rows = encoder.encode(&smiles).expect("encode");

    assert_eq!(rows.len(), 3);
    assert!(rows[0].is_some(), "CCO should encode");
    assert!(rows[1].is_none(), "garbage smiles should drop to None");
    assert!(rows[2].is_some(), "benzene should encode");
}

#[test]
fn encoder_returns_empty_vec_for_empty_input() {
    let dir = tempfile::tempdir().expect("tempdir");
    save_tiny_checkpoint(dir.path());
    let mut encoder = load_encoder(dir.path());

    let rows = encoder.encode(&[]).expect("encode empty");
    assert!(rows.is_empty());
}

#[test]
fn encoder_returns_all_none_when_every_smiles_is_invalid() {
    let dir = tempfile::tempdir().expect("tempdir");
    save_tiny_checkpoint(dir.path());
    let mut encoder = load_encoder(dir.path());

    let smiles: Vec<String> = ["not-a-smiles", "also bad!!"]
        .iter()
        .map(ToString::to_string)
        .collect();
    let rows = encoder.encode(&smiles).expect("encode");
    assert_eq!(rows.len(), smiles.len());
    assert!(rows.iter().all(Option::is_none));
}

#[test]
fn encoder_is_deterministic_across_calls() {
    let dir = tempfile::tempdir().expect("tempdir");
    save_tiny_checkpoint(dir.path());
    let mut encoder = load_encoder(dir.path());

    let smiles: Vec<String> = ["CCO", "c1ccccc1"]
        .iter()
        .map(ToString::to_string)
        .collect();
    let first = encoder.encode(&smiles).expect("encode #1");
    let second = encoder.encode(&smiles).expect("encode #2");

    assert_eq!(first.len(), second.len());
    for (left, right) in first.iter().zip(second.iter()) {
        let left_row = left.as_ref().expect("row #1");
        let right_row = right.as_ref().expect("row #2");
        assert_eq!(left_row.latent, right_row.latent);
        assert_eq!(
            left_row.reconstruction_count_tanimoto,
            right_row.reconstruction_count_tanimoto
        );
        assert_eq!(
            left_row.reconstruction_binary_tanimoto,
            right_row.reconstruction_binary_tanimoto
        );
        assert_eq!(
            left_row.reconstruction_log_mse,
            right_row.reconstruction_log_mse
        );
    }
}

#[test]
fn from_checkpoint_errors_when_config_is_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    match MoleculeEncoder::<B>::from_checkpoint(dir.path(), NdArrayDevice::default()) {
        Err(Error::Io { .. } | Error::Json { .. }) => {}
        Err(other) => panic!("unexpected error variant: {other:?}"),
        Ok(_) => panic!("missing model-config.json should not produce an encoder"),
    }
}

#[test]
fn from_checkpoint_errors_when_model_record_is_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = MoleculeAutoencoderConfig::symmetric(FINGERPRINT_SIZE, LATENT_WIDTH, vec![32]);
    config
        .save_json(dir.path().join("model-config.json"))
        .expect("save config");
    // Note: no model.mpk written.
    match MoleculeEncoder::<B>::from_checkpoint(dir.path(), NdArrayDevice::default()) {
        Err(Error::Io { .. }) => {}
        Err(other) => panic!("unexpected error variant: {other:?}"),
        Ok(_) => panic!("missing model record should not produce an encoder"),
    }
}

#[test]
fn encoder_exposes_config_dimensions() {
    let dir = tempfile::tempdir().expect("tempdir");
    save_tiny_checkpoint(dir.path());
    let encoder = load_encoder(dir.path());
    assert_eq!(encoder.latent_width(), LATENT_WIDTH);
    assert_eq!(encoder.fingerprint_size(), FINGERPRINT_SIZE);
    assert_eq!(encoder.config().encoder().latent_width(), LATENT_WIDTH);
}
