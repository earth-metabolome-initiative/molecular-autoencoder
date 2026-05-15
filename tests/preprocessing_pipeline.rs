//! End-to-end preprocessing and finite-loss smoke tests.

#![cfg(feature = "ndarray")]

use burn::data::dataloader::batcher::Batcher;
use molecular_autoencoder::{
    MoleculeAutoencoderBatcher, MoleculeAutoencoderConfig, MoleculeAutoencoderSample,
    MoleculeRecord, PreprocessingConfig, SparseMoleculeShard, features::REGRESSION_TARGET_WIDTH,
    preprocess_record,
};

#[test]
fn preprocessing_batching_and_model_loss_are_finite() {
    type B = burn::backend::NdArray<f32, i64>;

    let config = PreprocessingConfig::default();
    let record = MoleculeRecord::new("pubchem-smiles", "702", "CCO");
    let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
    let targets = preprocess_record(&record, &config, &mut scratch)
        .expect("preprocess ok")
        .expect("not filtered");
    let mut shard =
        SparseMoleculeShard::new(config.counted_ecfp().size(), REGRESSION_TARGET_WIDTH);
    shard
        .push_targets(&targets, *config.descriptors())
        .expect("push row");
    let row = shard.row(0).expect("row");
    let sample = MoleculeAutoencoderSample {
        cid: row.cid(),
        fingerprint_indices: row.fingerprint_indices().to_vec(),
        fingerprint_counts: row.fingerprint_counts().to_vec(),
        descriptor_targets: row.descriptor_targets().to_vec(),
    };

    let device = burn::backend::ndarray::NdArrayDevice::default();
    let batch: molecular_autoencoder::MoleculeAutoencoderBatch<B> =
        MoleculeAutoencoderBatcher::default().batch(vec![sample], &device);
    let model = MoleculeAutoencoderConfig::symmetric(config.counted_ecfp().size(), 16, vec![32])
        .init::<B>(&device);
    let loss = model.loss(batch).total().into_scalar();

    assert!(loss.is_finite());
}
