//! Fingerprint target generation.

use serde::{Deserialize, Serialize};
use smiles_parser::prelude::Smiles;

use crate::{Error, Result};

/// Default ECFP radius for the v1 model.
pub const DEFAULT_ECFP_RADIUS: u8 = 2;

/// Default folded counted ECFP width for the v1 model.
pub const DEFAULT_ECFP_SIZE: usize = 4096;

/// Counted ECFP configuration. Construct via [`CountedEcfpConfig::builder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CountedEcfpConfig {
    radius: u8,
    size: usize,
}

impl Default for CountedEcfpConfig {
    fn default() -> Self {
        CountedEcfpConfigBuilder::new()
            .build()
            .expect("default counted ECFP config is valid")
    }
}

impl CountedEcfpConfig {
    /// Starts a fluent builder.
    #[must_use]
    pub fn builder() -> CountedEcfpConfigBuilder {
        CountedEcfpConfigBuilder::new()
    }

    /// Morgan/ECFP radius.
    #[must_use]
    pub const fn radius(&self) -> u8 {
        self.radius
    }

    /// Folded fingerprint vector width.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Creates the finge-rs fingerprint implementation.
    #[must_use]
    pub const fn to_fingerprint(self) -> finge_rs::CountEcfpFingerprint {
        finge_rs::CountEcfpFingerprint::new(self.radius, self.size)
    }
}

/// Fluent builder for [`CountedEcfpConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountedEcfpConfigBuilder {
    radius: u8,
    size: usize,
}

impl Default for CountedEcfpConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CountedEcfpConfigBuilder {
    /// Creates a builder seeded with the v1 defaults (radius 2, size 4096).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            radius: DEFAULT_ECFP_RADIUS,
            size: DEFAULT_ECFP_SIZE,
        }
    }

    /// Sets the Morgan/ECFP radius.
    #[must_use]
    pub const fn radius(mut self, value: u8) -> Self {
        self.radius = value;
        self
    }

    /// Sets the folded fingerprint vector width.
    #[must_use]
    pub const fn size(mut self, value: usize) -> Self {
        self.size = value;
        self
    }

    /// Validates the configured fields and builds the immutable config.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfigInvalid`] when `radius` is 0, when `size` is 0,
    /// or when `size` exceeds the u16 sparse-index storage limit.
    pub fn build(self) -> Result<CountedEcfpConfig> {
        if self.radius == 0 {
            return Err(Error::ConfigInvalid {
                message: "counted ECFP radius must be greater than zero".to_string(),
            });
        }
        if self.size == 0 {
            return Err(Error::ConfigInvalid {
                message: "counted ECFP size must be greater than zero".to_string(),
            });
        }
        if self.size > usize::from(u16::MAX) + 1 {
            return Err(Error::ConfigInvalid {
                message: format!(
                    "counted ECFP size {} exceeds u16 sparse index storage",
                    self.size
                ),
            });
        }
        Ok(CountedEcfpConfig {
            radius: self.radius,
            size: self.size,
        })
    }
}

/// Sparse counted fingerprint target. Construct via
/// [`FingerprintTargets::new`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FingerprintTargets {
    indices: Vec<u16>,
    counts: Vec<u16>,
    fingerprint_size: usize,
}

impl FingerprintTargets {
    /// Creates a sparse fingerprint target from concatenated index / count
    /// pairs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBatch`] when `indices` and `counts` have
    /// different lengths, or when any index meets or exceeds
    /// `fingerprint_size`.
    pub fn new(indices: Vec<u16>, counts: Vec<u16>, fingerprint_size: usize) -> Result<Self> {
        if indices.len() != counts.len() {
            return Err(Error::InvalidBatch(
                "fingerprint indices and counts have different lengths".to_string(),
            ));
        }
        if indices
            .iter()
            .any(|&index| usize::from(index) >= fingerprint_size)
        {
            return Err(Error::InvalidBatch(format!(
                "fingerprint index exceeds configured width {fingerprint_size}"
            )));
        }
        Ok(Self {
            indices,
            counts,
            fingerprint_size,
        })
    }

    /// Nonzero folded fingerprint indices.
    #[must_use]
    pub fn indices(&self) -> &[u16] {
        &self.indices
    }

    /// Counts for each nonzero folded index.
    #[must_use]
    pub fn counts(&self) -> &[u16] {
        &self.counts
    }

    /// Full dense vector width.
    #[must_use]
    pub const fn fingerprint_size(&self) -> usize {
        self.fingerprint_size
    }

    /// Number of nonzero bins.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Returns a dense count vector.
    #[must_use]
    pub fn to_dense_counts(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.fingerprint_size];
        for (&index, &count) in self.indices.iter().zip(&self.counts) {
            dense[usize::from(index)] = f32::from(count);
        }
        dense
    }

    /// Returns a dense `log1p(count)` vector.
    #[must_use]
    pub fn to_dense_log_counts(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.fingerprint_size];
        for (&index, &count) in self.indices.iter().zip(&self.counts) {
            dense[usize::from(index)] = f32::from(count).ln_1p();
        }
        dense
    }
}

/// Computes counted ECFP targets for a parsed SMILES graph.
///
/// # Errors
///
/// Returns an error when the configured fingerprint width exceeds sparse index
/// storage or when `finge-rs` cannot prepare the molecule for fingerprinting.
pub fn compute_fingerprint_targets(
    molecule_id: u64,
    smiles: &Smiles,
    config: CountedEcfpConfig,
    scratch: &mut finge_rs::smiles_support::SmilesRdkitScratch,
) -> Result<FingerprintTargets> {
    use finge_rs::Fingerprint;

    let size = config.size();
    let prepared = scratch
        .try_prepare(smiles)
        .map_err(|source| Error::FingerprintPreparation {
            molecule_id,
            message: source.to_string(),
        })?;
    let fingerprint = config.to_fingerprint().compute(&prepared);
    let mut indices = Vec::new();
    let mut counts = Vec::new();
    for (index, count) in fingerprint.active_counts() {
        let index = u16::try_from(index).map_err(|_| {
            Error::InvalidBatch(format!("fingerprint index {index} exceeds u16 storage"))
        })?;
        let count = u16::try_from(count).unwrap_or(u16::MAX);
        indices.push(index);
        counts.push(count);
    }

    FingerprintTargets::new(indices, counts, size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counted_ecfp_default_is_4k_radius_2() {
        let config = CountedEcfpConfig::default();

        assert_eq!(config.radius(), 2);
        assert_eq!(config.size(), 4096);
    }

    #[test]
    fn counted_ecfp_builder_rejects_zero_size_and_radius() {
        assert!(CountedEcfpConfig::builder().radius(0).build().is_err());
        assert!(CountedEcfpConfig::builder().size(0).build().is_err());
        assert!(
            CountedEcfpConfig::builder()
                .size(usize::from(u16::MAX) + 2)
                .build()
                .is_err()
        );
    }

    #[test]
    fn counted_ecfp_produces_sparse_counts() {
        let smiles: Smiles = "CCO".parse().expect("valid SMILES");
        let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
        let targets =
            compute_fingerprint_targets(702, &smiles, CountedEcfpConfig::default(), &mut scratch)
                .expect("fingerprint target");

        assert_eq!(targets.indices().len(), targets.counts().len());
        assert_eq!(targets.fingerprint_size(), 4096);
        assert!(!targets.indices().is_empty());
        assert!(
            targets
                .indices()
                .iter()
                .all(|&index| usize::from(index) < 4096)
        );
    }

    #[test]
    fn sparse_targets_expand_to_dense_counts_and_log_counts() {
        let targets = FingerprintTargets::new(vec![1, 3], vec![2, 4], 5).expect("valid targets");

        assert_eq!(targets.nnz(), 2);
        assert_eq!(targets.to_dense_counts(), vec![0.0, 2.0, 0.0, 4.0, 0.0]);
        assert_eq!(
            targets.to_dense_log_counts(),
            vec![0.0, 2.0_f32.ln_1p(), 0.0, 4.0_f32.ln_1p(), 0.0]
        );
    }

    #[test]
    fn counted_ecfp_builder_rejects_oversized_widths_before_compute() {
        let error = CountedEcfpConfig::builder()
            .size(usize::from(u16::MAX) + 2)
            .build()
            .expect_err("oversized fingerprint should be rejected");

        assert!(matches!(error, Error::ConfigInvalid { message } if message.contains("u16")));
    }
}
