//! Fingerprint target generation.

use serde::{Deserialize, Serialize};
use smiles_parser::prelude::Smiles;

use crate::{Error, Result};

/// Default ECFP radius for the v1 model.
pub const DEFAULT_ECFP_RADIUS: u8 = 2;

/// Default folded counted ECFP width for the v1 model.
pub const DEFAULT_ECFP_SIZE: usize = 4096;

/// Counted ECFP configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CountedEcfpConfig {
    /// Morgan/ECFP radius.
    pub radius: u8,
    /// Folded fingerprint vector width.
    pub size: usize,
}

impl Default for CountedEcfpConfig {
    fn default() -> Self {
        Self {
            radius: DEFAULT_ECFP_RADIUS,
            size: DEFAULT_ECFP_SIZE,
        }
    }
}

impl CountedEcfpConfig {
    /// Creates the finge-rs fingerprint implementation.
    #[must_use]
    pub const fn to_fingerprint(self) -> finge_rs::CountEcfpFingerprint {
        finge_rs::CountEcfpFingerprint::new(self.radius, self.size)
    }
}

/// Sparse counted fingerprint target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FingerprintTargets {
    /// Nonzero folded fingerprint indices.
    pub indices: Vec<u16>,
    /// Counts for each nonzero folded index.
    pub counts: Vec<u16>,
    /// Full dense vector width.
    pub fingerprint_size: usize,
}

impl FingerprintTargets {
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
    cid: u64,
    smiles: &Smiles,
    config: CountedEcfpConfig,
    scratch: &mut finge_rs::smiles_support::SmilesRdkitScratch,
) -> Result<FingerprintTargets> {
    use finge_rs::Fingerprint;

    if config.size > usize::from(u16::MAX) + 1 {
        return Err(Error::InvalidBatch(format!(
            "fingerprint width {} exceeds u16 sparse index storage",
            config.size
        )));
    }

    let prepared = scratch
        .try_prepare(smiles)
        .map_err(|source| Error::FingerprintPreparation {
            cid,
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

    Ok(FingerprintTargets {
        indices,
        counts,
        fingerprint_size: config.size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counted_ecfp_default_is_4k_radius_2() {
        let config = CountedEcfpConfig::default();

        assert_eq!(config.radius, 2);
        assert_eq!(config.size, 4096);
    }

    #[test]
    fn counted_ecfp_produces_sparse_counts() {
        let smiles: Smiles = "CCO".parse().expect("valid SMILES");
        let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
        let targets =
            compute_fingerprint_targets(702, &smiles, CountedEcfpConfig::default(), &mut scratch)
                .expect("fingerprint target");

        assert_eq!(targets.indices.len(), targets.counts.len());
        assert_eq!(targets.fingerprint_size, 4096);
        assert!(!targets.indices.is_empty());
        assert!(
            targets
                .indices
                .iter()
                .all(|&index| usize::from(index) < 4096)
        );
    }

    #[test]
    fn sparse_targets_expand_to_dense_counts_and_log_counts() {
        let targets = FingerprintTargets {
            indices: vec![1, 3],
            counts: vec![2, 4],
            fingerprint_size: 5,
        };

        assert_eq!(targets.nnz(), 2);
        assert_eq!(targets.to_dense_counts(), vec![0.0, 2.0, 0.0, 4.0, 0.0]);
        assert_eq!(
            targets.to_dense_log_counts(),
            vec![0.0, 2.0_f32.ln_1p(), 0.0, 4.0_f32.ln_1p(), 0.0]
        );
    }

    #[test]
    fn counted_ecfp_rejects_widths_too_large_for_sparse_storage() {
        let smiles: Smiles = "CCO".parse().expect("valid SMILES");
        let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
        let error = compute_fingerprint_targets(
            702,
            &smiles,
            CountedEcfpConfig {
                radius: 2,
                size: usize::from(u16::MAX) + 2,
            },
            &mut scratch,
        )
        .expect_err("oversized fingerprint should be rejected");

        assert!(matches!(error, Error::InvalidBatch(message) if message.contains("u16")));
    }
}
