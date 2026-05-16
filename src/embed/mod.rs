//! Extension-dispatched readers and writers for the embed pipeline.
//!
//! The bin layer doesn't know about file formats: it asks
//! [`source_for_path`] and [`sink_for_path`] for boxed trait objects
//! given the input/output paths the user supplied, then loops
//! `source.next()? → encoder.encode(...) → sink.write(...)`.
//!
//! Adding a new format is one new module + one new arm in the relevant
//! dispatch function.

pub mod record;
pub mod sink;
pub mod source;

pub use record::{EncodingRecord, EncodingSchema, MoleculeInput};
pub use sink::{EncodingSink, SinkOptions, sink_for_path};
pub use source::{MoleculeSource, SourceOptions, source_for_path};
