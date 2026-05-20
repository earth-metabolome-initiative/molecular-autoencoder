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
    /// Backing `mascot-rs` path iterator yielding parsed MGF records.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fixture_mgf(path: &Path) {
        // Two records: the first carries a SMILES annotation, the second does not.
        // The reader must yield the first and skip the second.
        let body = "BEGIN IONS\n\
            FEATURE_ID=1\n\
            PEPMASS=500.0\n\
            CHARGE=1\n\
            RTINSECONDS=10.0\n\
            MSLEVEL=2\n\
            SMILES=CCO\n\
            IONMODE=Positive\n\
            SOURCE_INSTRUMENT=LC-ESI-Orbitrap\n\
            100.0 2.0\n\
            SCANS=1\n\
            END IONS\n\
            \n\
            BEGIN IONS\n\
            FEATURE_ID=2\n\
            PEPMASS=600.0\n\
            CHARGE=1\n\
            RTINSECONDS=12.0\n\
            MSLEVEL=2\n\
            IONMODE=Positive\n\
            SOURCE_INSTRUMENT=LC-ESI-Orbitrap\n\
            200.0 3.0\n\
            SCANS=2\n\
            END IONS\n";
        std::fs::write(path, body).expect("write mgf");
    }

    #[test]
    fn mgf_source_yields_records_with_smiles_and_skips_blank_ones() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fixture.mgf");
        write_fixture_mgf(&path);

        let mut source = MgfSource::from_path(&path).expect("open mgf");
        let first = source.next().expect("read first").expect("smiles record");
        assert_eq!(first.smiles, "CCO");
        // Second record has no SMILES → it is silently skipped, so the
        // stream now returns Ok(None).
        assert!(source.next().expect("read second").is_none());
    }

    #[test]
    fn mgf_source_reports_open_failure_for_missing_file() {
        match MgfSource::from_path(Path::new("/tmp/definitely-missing-xyz.mgf")) {
            Err(Error::InvalidBatch(msg)) => assert!(msg.contains("failed to open MGF")),
            Err(other) => panic!("unexpected error variant: {other:?}"),
            Ok(_) => panic!("missing file should not produce a source"),
        }
    }
}
