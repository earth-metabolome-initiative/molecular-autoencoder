//! End-to-end smoke test for the embed pipeline: train a tiny model,
//! save it, embed a handful of SMILES through every supported sink, and
//! verify the encoder output is well-shaped.

#![cfg(all(feature = "ndarray", feature = "embed", feature = "embed-parquet"))]

use std::fs;
use std::path::Path;

use molecular_autoencoder::{
    EncodingSchema, MoleculeAutoencoderConfig, MoleculeEncoder, SinkOptions, SourceOptions,
    sink_for_path, source_for_path,
};

type B = burn::backend::NdArray<f32, i64>;

fn save_tiny_checkpoint(dir: &Path) -> MoleculeAutoencoderConfig {
    use burn::module::Module;
    use burn::record::DefaultRecorder;

    let config = MoleculeAutoencoderConfig::symmetric(4096, 16, vec![32]);
    config
        .save_json(dir.join("model-config.json"))
        .expect("save config");

    let device = burn::backend::ndarray::NdArrayDevice::default();
    let model = config.init::<B>(&device);
    model
        .save_file(dir.join("model"), &DefaultRecorder::default())
        .expect("save model");
    config
}

fn write_smi(path: &Path, smiles: &[&str]) {
    let mut text = String::new();
    for smi in smiles {
        text.push_str(smi);
        text.push('\n');
    }
    fs::write(path, text).expect("write smi fixture");
}

#[test]
fn embed_pipeline_smi_to_tsv_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = save_tiny_checkpoint(dir.path());
    let smi_path = dir.path().join("inputs.smi");
    let smiles = ["CCO", "c1ccccc1", "CC(=O)O"];
    write_smi(&smi_path, &smiles);
    let out_path = dir.path().join("out.tsv");

    let mut source = source_for_path(&smi_path, &SourceOptions::default()).expect("smi source");
    let mut encoder = MoleculeEncoder::<B>::from_checkpoint(
        dir.path(),
        burn::backend::ndarray::NdArrayDevice::default(),
    )
    .expect("encoder load");
    let schema = EncodingSchema {
        latent_width: encoder.latent_width(),
    };
    let mut sink = sink_for_path(&out_path, &SinkOptions::default()).expect("tsv sink");
    sink.open(&schema).expect("sink open");

    let mut batch_buf: Vec<String> = Vec::new();
    while let Some(input) = source.next().expect("read") {
        batch_buf.push(input.smiles);
    }
    let rows = encoder.encode(&batch_buf).expect("encode");
    for slot in rows {
        let row = slot.expect("smi smoke fixture is all valid");
        sink.write(&row.into()).expect("write");
    }
    sink.finish().expect("finish");

    let text = fs::read_to_string(&out_path).expect("read tsv");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1 + smiles.len(), "header + one row per smile");
    assert_eq!(
        lines[0].matches('\t').count(),
        4 + config.encoder().latent_width() - 1
    );
    let first_row = lines[1];
    assert!(first_row.starts_with("CCO\t"));
}

#[test]
fn embed_pipeline_smi_to_parquet_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    save_tiny_checkpoint(dir.path());
    let smi_path = dir.path().join("inputs.smi");
    write_smi(&smi_path, &["CCO", "c1ccccc1"]);
    let out_path = dir.path().join("out.parquet");

    let mut source = source_for_path(&smi_path, &SourceOptions::default()).expect("smi source");
    let mut encoder = MoleculeEncoder::<B>::from_checkpoint(
        dir.path(),
        burn::backend::ndarray::NdArrayDevice::default(),
    )
    .expect("encoder load");
    let schema = EncodingSchema {
        latent_width: encoder.latent_width(),
    };
    let mut sink = sink_for_path(&out_path, &SinkOptions::default()).expect("parquet sink");
    sink.open(&schema).expect("sink open");

    let mut buf: Vec<String> = Vec::new();
    while let Some(input) = source.next().expect("read") {
        buf.push(input.smiles);
    }
    for slot in encoder.encode(&buf).expect("encode") {
        sink.write(&slot.expect("valid smi").into()).expect("write");
    }
    sink.finish().expect("finish");

    let file = fs::File::open(&out_path).expect("reopen parquet");
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("reader")
        .build()
        .expect("build reader");
    let batches: Result<Vec<_>, _> = reader.collect();
    let total: usize = batches
        .expect("collect")
        .iter()
        .map(arrow::record_batch::RecordBatch::num_rows)
        .sum();
    assert_eq!(total, 2);
}
