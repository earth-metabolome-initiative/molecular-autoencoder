//! `.smi` / `.smiles` reader: one SMILES per line, blanks and
//! `#`-prefixed comments ignored.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::{Error, Result, embed::record::MoleculeInput};

use super::MoleculeSource;

/// Reads one SMILES per line.
///
/// Trailing whitespace is stripped; blank lines and lines starting with
/// `#` are skipped. Multi-column inputs (e.g. `SMILES\tname`) get the
/// first whitespace-separated token taken as the SMILES; the rest of
/// the line is currently ignored.
pub struct SmilesLineSource {
    reader: BufReader<Box<dyn std::io::Read + Send>>,
    line_buffer: String,
}

impl SmilesLineSource {
    /// Opens a file as a SMILES line source.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when `path` cannot be opened.
    pub fn from_path(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|source| Error::io(path, source))?;
        Ok(Self::new(Box::new(file)))
    }

    /// Wraps any `Read`+`Send` source as a SMILES line reader; useful
    /// for stdin and unit tests.
    #[must_use]
    pub fn new(reader: Box<dyn std::io::Read + Send>) -> Self {
        Self {
            reader: BufReader::new(reader),
            line_buffer: String::new(),
        }
    }
}

impl MoleculeSource for SmilesLineSource {
    fn next(&mut self) -> Result<Option<MoleculeInput>> {
        loop {
            self.line_buffer.clear();
            let bytes = self
                .reader
                .read_line(&mut self.line_buffer)
                .map_err(|source| Error::io(Path::new("<smi>"), source))?;
            if bytes == 0 {
                return Ok(None);
            }
            let trimmed = self.line_buffer.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let smiles = trimmed
                .split_whitespace()
                .next()
                .map_or_else(String::new, str::to_owned);
            if smiles.is_empty() {
                continue;
            }
            return Ok(Some(MoleculeInput { smiles }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(mut source: SmilesLineSource) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(input) = source.next().expect("read smi") {
            out.push(input.smiles);
        }
        out
    }

    #[test]
    fn reads_three_smiles_skipping_blanks_and_comments() {
        let text = "CCO\n\n# a comment\nc1ccccc1\nCC.O\n";
        let source =
            SmilesLineSource::new(Box::new(std::io::Cursor::new(text.as_bytes().to_vec())));
        assert_eq!(collect(source), vec!["CCO", "c1ccccc1", "CC.O"]);
    }

    #[test]
    fn multi_column_input_takes_first_token() {
        let text = "CCO\tethanol\nc1ccccc1\tbenzene\n";
        let source =
            SmilesLineSource::new(Box::new(std::io::Cursor::new(text.as_bytes().to_vec())));
        assert_eq!(collect(source), vec!["CCO", "c1ccccc1"]);
    }
}
