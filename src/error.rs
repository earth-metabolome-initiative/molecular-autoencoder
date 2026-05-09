//! Error types.

use std::{io, path::PathBuf};

use thiserror::Error;

/// Crate result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised by preprocessing, shard loading, and modeling helpers.
#[derive(Debug, Error)]
pub enum Error {
    /// PubChem CID-SMILES line was malformed.
    #[error("invalid PubChem CID-SMILES record at line {line_number}: {message}")]
    PubChemRecord {
        /// One-based line number.
        line_number: usize,
        /// Human-readable parse error.
        message: String,
    },
    /// A SMILES string failed to parse.
    #[error("failed to parse SMILES for CID {cid}: {message}")]
    SmilesParse {
        /// PubChem compound identifier.
        cid: u64,
        /// Underlying parser error text.
        message: String,
    },
    /// RDKit-style fingerprint preparation failed.
    #[error("failed to prepare SMILES for fingerprinting for CID {cid}: {message}")]
    FingerprintPreparation {
        /// PubChem compound identifier.
        cid: u64,
        /// Underlying preparation error text.
        message: String,
    },
    /// I/O failure.
    #[error("failed to access {path}: {source}")]
    Io {
        /// Path being read or written.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// JSON serialization failure.
    #[error("failed to serialize or parse JSON at {path}: {source}")]
    Json {
        /// Path being read or written.
        path: PathBuf,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// Shard contents do not match the expected format.
    #[error("invalid shard {path}: {message}")]
    ShardFormat {
        /// Shard path.
        path: PathBuf,
        /// Human-readable format error.
        message: String,
    },
    /// Invalid batch input.
    #[error("invalid batch: {0}")]
    InvalidBatch(String),
}

impl Error {
    /// Builds an [`Error::Io`] value for a path-specific I/O failure.
    pub fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
