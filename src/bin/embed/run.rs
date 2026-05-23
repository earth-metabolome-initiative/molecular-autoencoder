//! End-to-end orchestrator: open source → load encoder → encode → write.

use std::time::Instant;

use burn::prelude::Backend;
use molecular_autoencoder::{
    EncodingSchema, MoleculeEncoder, SinkOptions, SourceOptions, sink_for_path, source_for_path,
};

use crate::AppResult;
use crate::cli::Args;

/// Streams SMILES from the configured source through the encoder and into the
/// configured sink, emitting per-batch progress to stderr.
pub fn run<
    B: Backend<FloatElem = f32>
        + molecular_autoencoder::sparse_input_cuda::SparseInputLinearKernelBackend,
>(
    args: Args,
    device: B::Device,
) -> AppResult<()> {
    if args.batch_size == 0 {
        return Err("batch size must be greater than zero".into());
    }

    let source_options = SourceOptions {
        smiles_column: Some(args.smiles_column.clone()),
    };
    let sink_options = SinkOptions {
        parquet_compression: args.parquet_compression.clone(),
    };

    let mut source = source_for_path(&args.input, &source_options)?;
    let mut encoder = MoleculeEncoder::<B>::from_checkpoint(&args.checkpoint, device)?;
    let schema = EncodingSchema {
        latent_width: encoder.latent_width(),
    };
    let mut sink = sink_for_path(&args.output, &sink_options)?;
    sink.open(&schema)?;

    let mut buffer: Vec<String> = Vec::with_capacity(args.batch_size);
    let mut processed = 0_usize;
    let mut skipped = 0_usize;
    let start = Instant::now();

    while let Some(input) = source.next()? {
        buffer.push(input.smiles);
        if buffer.len() >= args.batch_size {
            let (encoded_count, skipped_count) = flush(&mut encoder, &mut sink, &mut buffer)?;
            processed += encoded_count;
            skipped += skipped_count;
        }
    }
    if !buffer.is_empty() {
        let (encoded_count, skipped_count) = flush(&mut encoder, &mut sink, &mut buffer)?;
        processed += encoded_count;
        skipped += skipped_count;
    }

    sink.finish()?;
    eprintln!(
        "embed_done processed={processed} skipped={skipped} elapsed_sec={:.2}",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Encodes the buffered SMILES batch and writes the resulting rows to the
/// sink, returning `(encoded, skipped)` counts and clearing the buffer.
fn flush<
    B: Backend<FloatElem = f32>
        + molecular_autoencoder::sparse_input_cuda::SparseInputLinearKernelBackend,
>(
    encoder: &mut MoleculeEncoder<B>,
    sink: &mut Box<dyn molecular_autoencoder::EncodingSink>,
    buffer: &mut Vec<String>,
) -> AppResult<(usize, usize)> {
    if buffer.is_empty() {
        return Ok((0, 0));
    }
    let rows = encoder.encode(buffer)?;
    let mut written = 0_usize;
    let mut skipped = 0_usize;
    for (slot, smiles) in rows.into_iter().zip(buffer.iter()) {
        if let Some(row) = slot {
            sink.write(&row.into())?;
            written += 1;
        } else {
            eprintln!("embed_skip smiles={smiles}");
            skipped += 1;
        }
    }
    buffer.clear();
    Ok((written, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use molecular_autoencoder::MoleculeAutoencoderConfig;

    type B = burn::backend::NdArray<f32, i64>;

    fn write_checkpoint(dir: &std::path::Path) {
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
    }

    #[test]
    fn run_end_to_end_smi_to_tsv() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ckpt = dir.path().join("ckpt");
        std::fs::create_dir_all(&ckpt).expect("ckpt dir");
        write_checkpoint(&ckpt);

        let input = dir.path().join("inputs.smi");
        // Include one bogus SMILES so the flush() skip branch is exercised.
        std::fs::write(&input, "CCO\nnot-a-smiles\nc1ccccc1\n").expect("write smi");
        let output = dir.path().join("out.tsv");

        let args = crate::cli::Args::try_parse_from([
            "embed",
            input.to_str().expect("utf-8"),
            output.to_str().expect("utf-8"),
            "--checkpoint",
            ckpt.to_str().expect("utf-8"),
            "--batch-size",
            "2",
        ])
        .expect("parse");

        run::<B>(args, burn::backend::ndarray::NdArrayDevice::default()).expect("run");

        let text = std::fs::read_to_string(&output).expect("read tsv");
        // Header + two surviving rows (third skipped).
        assert_eq!(text.lines().count(), 1 + 2);
    }

    #[test]
    fn run_rejects_zero_batch_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("out.tsv");
        let mut args = crate::cli::Args::try_parse_from([
            "embed",
            dir.path().to_str().expect("utf-8"),
            out.to_str().expect("utf-8"),
            "--checkpoint",
            dir.path().to_str().expect("utf-8"),
        ])
        .expect("parse");
        args.batch_size = 0;
        let err = run::<B>(args, burn::backend::ndarray::NdArrayDevice::default())
            .expect_err("zero batch must be rejected");
        assert!(err.to_string().contains("batch size"));
    }
}
