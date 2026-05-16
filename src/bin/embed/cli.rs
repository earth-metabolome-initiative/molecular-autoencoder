//! Command-line argument parsing for the `embed` bin.

use std::path::PathBuf;

use clap::Parser;

/// Default number of SMILES processed per encoder forward pass.
pub const DEFAULT_BATCH_SIZE: usize = 4096;

/// Inference CLI.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "embed",
    version,
    about = "Embed SMILES with a trained molecular autoencoder",
    disable_help_subcommand = true
)]
pub struct Args {
    /// Input path. Use `-` (or `/dev/stdin`) for stdin (.smi semantics).
    /// Recognised extensions: .smi, .smiles, .tsv, .csv, .mgf, .mgf.zst,
    /// .mgf.gz.
    pub input: PathBuf,

    /// Output path. Use `-` (or `/dev/stdout`) for stdout (TSV).
    /// Recognised extensions: .tsv, .csv, .jsonl, .parquet.
    pub output: PathBuf,

    /// Training run directory containing `model-config.json` and
    /// `model.mpk`.
    #[arg(long)]
    pub checkpoint: PathBuf,

    /// SMILES rows processed per forward pass.
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    pub batch_size: usize,

    /// SMILES column name in delimited (`.tsv`/`.csv`) inputs.
    #[arg(long, default_value = "smiles")]
    pub smiles_column: String,

    /// Parquet compression: snappy | none | zstd | gzip. Ignored when
    /// the output is not Parquet.
    #[arg(long)]
    pub parquet_compression: Option<String>,

    /// CUDA device ordinal (ignored on the ndarray backend).
    #[arg(long, default_value_t = 0)]
    pub cuda_device: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn command_definition_is_valid() {
        Args::command().debug_assert();
    }

    #[test]
    fn parses_minimal_positional_arguments() {
        let args = Args::try_parse_from(["embed", "inputs.smi", "-", "--checkpoint", "runs/foo"])
            .expect("minimal parse");
        assert_eq!(args.input, PathBuf::from("inputs.smi"));
        assert_eq!(args.output, PathBuf::from("-"));
        assert_eq!(args.checkpoint, PathBuf::from("runs/foo"));
        assert_eq!(args.batch_size, DEFAULT_BATCH_SIZE);
        assert_eq!(args.smiles_column, "smiles");
    }
}
