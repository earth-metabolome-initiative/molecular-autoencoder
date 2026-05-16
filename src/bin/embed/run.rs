//! End-to-end orchestrator: open source → load encoder → encode → write.

use std::time::Instant;

use burn::prelude::Backend;
use molecular_autoencoder::{
    EncodingSchema, MoleculeEncoder, SinkOptions, SourceOptions, sink_for_path, source_for_path,
};

use crate::AppResult;
use crate::cli::Args;

pub fn run<B: Backend<FloatElem = f32>>(args: Args, device: B::Device) -> AppResult<()> {
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

fn flush<B: Backend<FloatElem = f32>>(
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
