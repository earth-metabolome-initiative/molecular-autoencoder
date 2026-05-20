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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EncodingRow;

    fn sample_row(width: usize) -> EncodingRecord {
        EncodingRow {
            smiles: "CCO".to_string(),
            latent: vec![0.1; width],
            reconstruction_count_tanimoto: 0.5,
            reconstruction_binary_tanimoto: 0.6,
            reconstruction_log_mse: 0.01,
        }
        .into()
    }

    #[test]
    fn open_then_write_then_finish_smoke() {
        // Output goes to test stdout (captured by cargo); we only assert the
        // sink does not error.
        let mut sink = Box::new(StdoutSink::new());
        let schema = EncodingSchema { latent_width: 4 };
        sink.open(&schema).expect("open");
        sink.write(&sample_row(4)).expect("write row");
        sink.finish().expect("finish");
    }

    #[test]
    fn default_is_equivalent_to_new() {
        // Default impl just calls `new` and produces a working sink.
        let mut sink = Box::new(StdoutSink::default());
        sink.open(&EncodingSchema { latent_width: 2 })
            .expect("open");
        sink.finish().expect("finish");
    }

    #[test]
    fn write_before_open_returns_invalid_batch() {
        let mut sink = StdoutSink::new();
        let err = sink.write(&sample_row(4)).expect_err("write before open");
        assert!(matches!(
            err,
            Error::InvalidBatch(msg) if msg.contains("before open")
        ));
    }

    #[test]
    fn write_with_latent_width_mismatch_is_rejected() {
        let mut sink = StdoutSink::new();
        sink.open(&EncodingSchema { latent_width: 4 })
            .expect("open");
        let err = sink.write(&sample_row(3)).expect_err("width mismatch");
        assert!(matches!(
            err,
            Error::InvalidBatch(msg) if msg.contains("latent width")
        ));
    }
}
