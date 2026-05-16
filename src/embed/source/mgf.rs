//! `.mgf` reader using `mascot-rs`'s streaming path iterator.
//!
//! `MascotGenericFormat::iter_from_path` opens the file, applies the
//! right decompressor for `.mgf` / `.mgf.zst` / `.mgf.gz`, and yields
//! records lazily; we walk that iterator and extract the
//! `metadata().smiles()` field. Records without a `SMILES=` annotation
//! and records that fail to parse are skipped silently — same policy as
//! `mascot_rs::IterMGFProperty::properties`.

use std::path::Path;

use mascot_rs::prelude::{MGFPathIter, MGFVec};

use crate::{Error, Result, embed::record::MoleculeInput, embed::source::MoleculeSource};

/// Streaming `.mgf` reader. Compression is handled transparently for
/// `.mgf.zst`, `.mgf.zstd`, `.mgf.gz`, and `.mgf.gzip`.
pub struct MgfSource {
    iter: MGFPathIter<f64>,
}

impl MgfSource {
    /// Opens the MGF file at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBatch`] when the file cannot be opened or
    /// the decompressor cannot be initialized.
    pub fn from_path(path: &Path) -> Result<Self> {
        let iter = MGFVec::<f64>::iter_from_path(path).map_err(|source| {
            Error::InvalidBatch(format!("failed to open MGF {}: {source}", path.display()))
        })?;
        Ok(Self { iter })
    }
}

impl MoleculeSource for MgfSource {
    fn next(&mut self) -> Result<Option<MoleculeInput>> {
        loop {
            let Some(result) = self.iter.next() else {
                return Ok(None);
            };
            let Ok(record) = result else {
                // Parse errors mirror `mascot_rs`'s default policy: skip.
                continue;
            };
            let Some(smiles) = record.metadata().smiles() else {
                continue;
            };
            return Ok(Some(MoleculeInput {
                smiles: smiles.to_string(),
            }));
        }
    }
}
