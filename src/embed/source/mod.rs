//! Input-side trait, dispatch, and per-format readers.

mod delimited;
#[cfg(feature = "embed-mgf")]
mod mgf;
mod smi;
mod stdin;

pub use delimited::DelimitedSource;
#[cfg(feature = "embed-mgf")]
pub use mgf::MgfSource;
pub use smi::SmilesLineSource;
pub use stdin::StdinSource;

use std::path::Path;

use crate::{Error, Result, embed::record::MoleculeInput};

/// Streaming source of [`MoleculeInput`] records.
///
/// Implementations are typically one-record-at-a-time and may read
/// lazily from disk. A `None` return signals exhaustion.
pub trait MoleculeSource {
    /// Yields the next record, or `None` when the source is exhausted.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying I/O fails or when a record
    /// is structurally invalid (for example a delimited row missing the
    /// configured SMILES column).
    fn next(&mut self) -> Result<Option<MoleculeInput>>;
}

/// Per-source tuning options threaded in by the bin's CLI.
#[derive(Debug, Clone, Default)]
pub struct SourceOptions {
    /// Column name to look up SMILES in delimited inputs (`.tsv` / `.csv`).
    /// Defaults to `"smiles"` when unset.
    pub smiles_column: Option<String>,
}

impl SourceOptions {
    /// Effective SMILES column name; falls back to `"smiles"`.
    #[must_use]
    pub fn smiles_column(&self) -> &str {
        self.smiles_column.as_deref().unwrap_or("smiles")
    }
}

/// Dispatches to the right concrete [`MoleculeSource`] based on the
/// input path's extension.
///
/// The bare path `-` (or `/dev/stdin`) is treated as stdin with
/// `.smi`-style line semantics. Recognised extensions:
///
/// - `.smi`, `.smiles` → [`SmilesLineSource`]
/// - `.tsv`, `.csv` → [`DelimitedSource`]
/// - `.mgf`, `.mgf.zst`, `.mgf.gz` → [`MgfSource`] (requires the
///   `embed-mgf` feature; compression is handled by `mascot-rs`)
///
/// # Errors
///
/// Returns [`Error::InvalidBatch`] for unknown extensions or when the
/// file cannot be opened.
pub fn source_for_path(path: &Path, options: &SourceOptions) -> Result<Box<dyn MoleculeSource>> {
    if is_stdin_path(path) {
        return Ok(Box::new(StdinSource::new()));
    }

    let ext = compound_extension(path);
    match ext.as_deref() {
        Some("smi" | "smiles") => Ok(Box::new(SmilesLineSource::from_path(path)?)),
        Some("tsv") => Ok(Box::new(DelimitedSource::from_path(
            path,
            b'\t',
            options.smiles_column().to_string(),
        )?)),
        Some("csv") => Ok(Box::new(DelimitedSource::from_path(
            path,
            b',',
            options.smiles_column().to_string(),
        )?)),
        #[cfg(feature = "embed-mgf")]
        Some("mgf" | "mgf.zst" | "mgf.gz" | "mgf.zstd" | "mgf.gzip") => {
            Ok(Box::new(MgfSource::from_path(path)?))
        }
        Some(other) => Err(Error::InvalidBatch(format!(
            "unsupported input extension `.{other}` for {}",
            path.display()
        ))),
        None => Err(Error::InvalidBatch(format!(
            "input path {} has no extension; use `-` for stdin",
            path.display()
        ))),
    }
}

fn is_stdin_path(path: &Path) -> bool {
    path == Path::new("-") || path == Path::new("/dev/stdin")
}

/// Returns the file's compound extension (e.g. `"mgf.zst"` for
/// `library.mgf.zst`), lowercased. Falls back to the single-extension
/// form when no compound suffix is recognised.
fn compound_extension(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    for compound in ["mgf.zst", "mgf.zstd", "mgf.gz", "mgf.gzip"] {
        if name.ends_with(&format!(".{compound}")) {
            return Some(compound.to_string());
        }
    }
    path.extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_extension_picks_mgf_compressed_suffix() {
        assert_eq!(
            compound_extension(Path::new("a.mgf")).as_deref(),
            Some("mgf")
        );
        assert_eq!(
            compound_extension(Path::new("a.MGF.zst")).as_deref(),
            Some("mgf.zst")
        );
        assert_eq!(
            compound_extension(Path::new("a.mgf.gz")).as_deref(),
            Some("mgf.gz")
        );
        assert_eq!(
            compound_extension(Path::new("a.tsv")).as_deref(),
            Some("tsv")
        );
        assert_eq!(compound_extension(Path::new("/dev/stdin")), None);
    }

    #[test]
    fn stdin_dispatch_yields_stdin_source() {
        let source = source_for_path(Path::new("-"), &SourceOptions::default())
            .expect("stdin source dispatch");
        // Can't easily exercise reads on stdin here, but the type
        // dispatch is what we're verifying.
        drop(source);
    }

    #[test]
    fn unknown_extension_is_rejected() {
        let result = source_for_path(Path::new("data.xyz"), &SourceOptions::default());
        assert!(matches!(result, Err(Error::InvalidBatch(_))));
    }
}
