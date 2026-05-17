//! Stdout writer with TSV formatting (mirrors [`TsvSink`]).

use std::fmt::Write as _;
use std::io::Write;

use crate::{
    Error, Result,
    embed::record::{EncodingRecord, EncodingSchema},
    embed::sink::EncodingSink,
};

/// Writes encoded rows to stdout as TSV.
pub struct StdoutSink {
    writer: std::io::Stdout,
    latent_width: Option<usize>,
}

impl Default for StdoutSink {
    fn default() -> Self {
        Self::new()
    }
}

impl StdoutSink {
    /// Creates a new stdout sink.
    #[must_use]
    pub fn new() -> Self {
        Self {
            writer: std::io::stdout(),
            latent_width: None,
        }
    }
}

impl EncodingSink for StdoutSink {
    fn open(&mut self, schema: &EncodingSchema) -> Result<()> {
        let mut header = String::from("smiles\tcount_tanimoto\tbinary_tanimoto\tlog_mse");
        for index in 0..schema.latent_width {
            write!(header, "\tlatent_{index}").map_err(stdout_fmt_error)?;
        }
        header.push('\n');
        self.writer
            .write_all(header.as_bytes())
            .map_err(|source| Error::InvalidBatch(format!("stdout write failed: {source}")))?;
        self.latent_width = Some(schema.latent_width);
        Ok(())
    }

    fn write(&mut self, record: &EncodingRecord) -> Result<()> {
        let row = &record.row;
        let expected = self.latent_width.ok_or_else(|| {
            Error::InvalidBatch("stdout sink wrote before open() was called".to_string())
        })?;
        if row.latent.len() != expected {
            return Err(Error::InvalidBatch(format!(
                "stdout sink expected latent width {expected}, got {}",
                row.latent.len()
            )));
        }
        let mut line = String::with_capacity(64 + 12 * row.latent.len());
        line.push_str(&row.smiles);
        write!(
            line,
            "\t{:.7}\t{:.7}\t{:.7}",
            row.reconstruction_count_tanimoto,
            row.reconstruction_binary_tanimoto,
            row.reconstruction_log_mse
        )
        .map_err(stdout_fmt_error)?;
        for value in &row.latent {
            write!(line, "\t{value:.7}").map_err(stdout_fmt_error)?;
        }
        line.push('\n');
        self.writer
            .write_all(line.as_bytes())
            .map_err(|source| Error::InvalidBatch(format!("stdout write failed: {source}")))?;
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        self.writer
            .flush()
            .map_err(|source| Error::InvalidBatch(format!("stdout flush failed: {source}")))?;
        Ok(())
    }
}

fn stdout_fmt_error(source: std::fmt::Error) -> Error {
    Error::InvalidBatch(format!("stdout format failed: {source}"))
}
