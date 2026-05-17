//! `.tsv` / `.csv` writer.
//!
//! Column layout:
//! `smiles  count_tanimoto  binary_tanimoto  log_mse  latent_0  latent_1  ...  latent_{N-1}`
//! where the separator is set at construction time.

use std::{fs::File, path::Path};

use csv::WriterBuilder;

use crate::{
    Error, Result,
    embed::record::{EncodingRecord, EncodingSchema},
    embed::sink::EncodingSink,
};

/// Tab- or comma-separated writer.
pub struct TsvSink {
    writer: csv::Writer<File>,
    latent_width: Option<usize>,
}

impl TsvSink {
    /// Creates a new writer at `path` with the given column `delimiter`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file cannot be created.
    pub fn from_path(path: &Path, delimiter: u8) -> Result<Self> {
        let file = File::create(path).map_err(|source| Error::io(path, source))?;
        let writer = WriterBuilder::new()
            .delimiter(delimiter)
            .has_headers(false)
            .from_writer(file);
        Ok(Self {
            writer,
            latent_width: None,
        })
    }
}

impl EncodingSink for TsvSink {
    fn open(&mut self, schema: &EncodingSchema) -> Result<()> {
        let mut headers = Vec::with_capacity(4 + schema.latent_width);
        headers.push("smiles".to_string());
        headers.push("count_tanimoto".to_string());
        headers.push("binary_tanimoto".to_string());
        headers.push("log_mse".to_string());
        for index in 0..schema.latent_width {
            headers.push(format!("latent_{index}"));
        }
        self.writer
            .write_record(&headers)
            .map_err(|source| Error::InvalidBatch(format!("tsv header write failed: {source}")))?;
        self.latent_width = Some(schema.latent_width);
        Ok(())
    }

    fn write(&mut self, record: &EncodingRecord) -> Result<()> {
        let row = &record.row;
        let expected = self.latent_width.ok_or_else(|| {
            Error::InvalidBatch("tsv sink wrote before open() was called".to_string())
        })?;
        if row.latent.len() != expected {
            return Err(Error::InvalidBatch(format!(
                "tsv sink expected latent width {expected}, got {}",
                row.latent.len()
            )));
        }
        let mut fields = Vec::with_capacity(4 + row.latent.len());
        fields.push(row.smiles.clone());
        fields.push(format_f32(row.reconstruction_count_tanimoto));
        fields.push(format_f32(row.reconstruction_binary_tanimoto));
        fields.push(format_f32(row.reconstruction_log_mse));
        for value in &row.latent {
            fields.push(format_f32(*value));
        }
        self.writer
            .write_record(&fields)
            .map_err(|source| Error::InvalidBatch(format!("tsv row write failed: {source}")))?;
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        self.writer
            .flush()
            .map_err(|source| Error::InvalidBatch(format!("tsv flush failed: {source}")))?;
        Ok(())
    }
}

fn format_f32(value: f32) -> String {
    // 7 significant digits round-trips IEEE-754 f32; cheaper than scientific.
    format!("{value:.7}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EncodingRow;
    use std::io::Read;

    #[test]
    fn header_and_one_row_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out.tsv");
        let mut sink = Box::new(TsvSink::from_path(&path, b'\t').expect("open"));
        let schema = EncodingSchema { latent_width: 3 };
        sink.open(&schema).expect("header");
        let row = EncodingRow {
            smiles: "CCO".into(),
            latent: vec![0.1, 0.2, 0.3],
            reconstruction_count_tanimoto: 0.85,
            reconstruction_binary_tanimoto: 0.93,
            reconstruction_log_mse: 0.042,
        };
        sink.write(&row.into()).expect("write");
        sink.finish().expect("finish");

        let mut text = String::new();
        File::open(&path)
            .expect("reopen")
            .read_to_string(&mut text)
            .expect("read");
        let mut lines = text.lines();
        assert_eq!(
            lines.next().expect("header"),
            "smiles\tcount_tanimoto\tbinary_tanimoto\tlog_mse\tlatent_0\tlatent_1\tlatent_2"
        );
        let data = lines.next().expect("row");
        assert!(data.starts_with("CCO\t"));
        assert_eq!(data.matches('\t').count(), 6);
    }
}
