//! Stdin reader using `.smi`-style line semantics.

use crate::{Result, embed::record::MoleculeInput, embed::source::MoleculeSource};

use super::smi::SmilesLineSource;

/// Wraps `std::io::stdin()` as a [`MoleculeSource`] with the same
/// line-by-line SMILES semantics as a `.smi` file.
pub struct StdinSource {
    inner: SmilesLineSource,
}

impl Default for StdinSource {
    fn default() -> Self {
        Self::new()
    }
}

impl StdinSource {
    /// Creates a new stdin reader.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: SmilesLineSource::new(Box::new(std::io::stdin())),
        }
    }
}

impl MoleculeSource for StdinSource {
    fn next(&mut self) -> Result<Option<MoleculeInput>> {
        self.inner.next()
    }
}
