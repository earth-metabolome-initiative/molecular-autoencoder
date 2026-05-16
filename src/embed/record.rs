//! Record types passed between sources, the encoder, and sinks.

use crate::EncodingRow;

/// Input record yielded by a [`MoleculeSource`](super::MoleculeSource).
///
/// SMILES is the only identifier carried through the pipeline; per the
/// project convention there is no separate `id` column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoleculeInput {
    /// Source SMILES text.
    pub smiles: String,
}

/// One row of output written by an [`EncodingSink`](super::EncodingSink).
///
/// Currently identical to [`EncodingRow`] from the encoder layer; carried
/// as a distinct type so the sink trait can grow source-side metadata
/// fields later without affecting the encoder API.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodingRecord {
    /// The four columns produced by the encoder for this molecule.
    pub row: EncodingRow,
}

impl From<EncodingRow> for EncodingRecord {
    fn from(row: EncodingRow) -> Self {
        Self { row }
    }
}

/// Shape information passed to [`EncodingSink::open`](super::EncodingSink::open)
/// once before any record is written.
///
/// Parquet uses [`latent_width`](Self::latent_width) to pin its
/// `FixedSizeList<Float32, N>` schema up front. Streaming text sinks
/// (TSV, JSONL, stdout) ignore the schema and emit columns lazily based
/// on the first record they see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodingSchema {
    /// Length of the latent embedding column across every row.
    pub latent_width: usize,
}
