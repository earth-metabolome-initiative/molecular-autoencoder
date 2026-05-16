//! `.tsv` / `.csv` reader. Picks SMILES out of the column whose header
//! matches the configured name (defaults to `"smiles"`).

use std::{fs::File, path::Path};

use csv::{ReaderBuilder, StringRecord};

use crate::{Error, Result, embed::record::MoleculeInput, embed::source::MoleculeSource};

/// Delimited reader for `.tsv` / `.csv` inputs.
pub struct DelimitedSource {
    reader: csv::Reader<File>,
    smiles_index: usize,
    record_buffer: StringRecord,
}

impl DelimitedSource {
    /// Opens a delimited file at `path` using `delimiter` (typically
    /// `b'\t'` or `b','`) and resolves the SMILES column by header name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file cannot be opened and
    /// [`Error::InvalidBatch`] when the header row is missing or the
    /// configured SMILES column is not present.
    pub fn from_path(path: &Path, delimiter: u8, smiles_column: String) -> Result<Self> {
        let file = File::open(path).map_err(|source| Error::io(path, source))?;
        let mut reader = ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(true)
            .from_reader(file);
        let header = reader.headers().map_err(|source| {
            Error::InvalidBatch(format!(
                "failed to read header row from {}: {source}",
                path.display()
            ))
        })?;
        let smiles_index = header
            .iter()
            .position(|column| column == smiles_column)
            .ok_or_else(|| {
                Error::InvalidBatch(format!(
                    "header row of {} has no column named `{smiles_column}` (got: {})",
                    path.display(),
                    header
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                ))
            })?;
        Ok(Self {
            reader,
            smiles_index,
            record_buffer: StringRecord::new(),
        })
    }
}

impl MoleculeSource for DelimitedSource {
    fn next(&mut self) -> Result<Option<MoleculeInput>> {
        loop {
            let has_next = self
                .reader
                .read_record(&mut self.record_buffer)
                .map_err(|source| {
                    Error::InvalidBatch(format!("failed to read delimited record: {source}"))
                })?;
            if !has_next {
                return Ok(None);
            }
            let Some(smiles) = self.record_buffer.get(self.smiles_index) else {
                return Err(Error::InvalidBatch(format!(
                    "delimited record at position {} is missing column {}",
                    self.reader.position().record(),
                    self.smiles_index
                )));
            };
            let smiles = smiles.trim();
            if smiles.is_empty() {
                continue;
            }
            return Ok(Some(MoleculeInput {
                smiles: smiles.to_owned(),
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("tempfile");
        file.write_all(contents.as_bytes()).expect("write");
        file
    }

    #[test]
    fn tsv_reads_smiles_from_named_column() {
        let file = write_temp("name\tsmiles\nethanol\tCCO\nbenzene\tc1ccccc1\n");
        let mut source =
            DelimitedSource::from_path(file.path(), b'\t', "smiles".to_string()).expect("open");
        assert_eq!(
            source.next().expect("row 0").expect("row present").smiles,
            "CCO"
        );
        assert_eq!(
            source.next().expect("row 1").expect("row present").smiles,
            "c1ccccc1"
        );
        assert!(source.next().expect("eof").is_none());
    }

    #[test]
    fn missing_column_is_rejected() {
        let file = write_temp("name,structure\nethanol,CCO\n");
        let result = DelimitedSource::from_path(file.path(), b',', "smiles".to_string());
        assert!(matches!(result, Err(Error::InvalidBatch(_))));
    }

    #[test]
    fn blank_smiles_rows_are_skipped() {
        let file = write_temp("smiles\nCCO\n\nc1ccccc1\n");
        let mut source =
            DelimitedSource::from_path(file.path(), b'\t', "smiles".to_string()).expect("open");
        // csv crate treats blank as a row with one empty field;
        // delimited source skips empties.
        let first = source.next().expect("row 0").expect("row present");
        assert_eq!(first.smiles, "CCO");
        let second = source.next().expect("row 1").expect("row present");
        assert_eq!(second.smiles, "c1ccccc1");
        assert!(source.next().expect("eof").is_none());
    }
}
