//! Fingerprint target generation.

use serde::{Deserialize, Serialize};
use smiles_parser::prelude::Smiles;

use crate::{Error, Result};

/// Default ECFP radius for the v1 model.
pub const DEFAULT_ECFP_RADIUS: u8 = 2;

/// Default folded counted ECFP width for the v1 model.
pub const DEFAULT_ECFP_SIZE: usize = 4096;

/// Number of [`finge_rs::ViolationClass`] variants the multi-label
/// classifier head distinguishes (mirrors `ViolationClass::COUNT`).
pub const VIOLATION_CLASS_COUNT: usize = finge_rs::ViolationClass::COUNT;

/// Maximum number of `MutatorMix::sample` attempts per negative slot before
/// giving up. Mutators can fail on small or all-aliphatic molecules; a
/// handful of re-rolls usually finds a viable target.
const NEGATIVE_FINGERPRINT_RETRIES: u32 = 4;

/// Counted ECFP configuration. Construct via [`CountedEcfpConfig::builder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CountedEcfpConfig {
    /// ECFP Morgan radius (depth of the neighbourhood walks).
    radius: u8,
    /// Folded fingerprint width (number of bits / counters).
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
    /// Pending Morgan radius.
    radius: u8,
    /// Pending folded fingerprint width.
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
    /// Sparse bit indices contributing to the fingerprint.
    indices: Vec<u16>,
    /// Counts aligned with `indices`.
    counts: Vec<u16>,
    /// Dense fingerprint width recovered when materialising counts.
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
    fingerprint_targets_from_active_counts(fingerprint.active_counts(), size)
}

/// Drives [`finge_rs::MutatorMix`] once on a parsed SMILES.
///
/// Returns the counted-ECFP fingerprint of the mutated molecule along
/// with the [`finge_rs::ViolationLabel`] bitset describing every
/// violation class the post-mutation predicates fire on.
///
/// Returns `Ok(None)` when no eligible mutation can be produced for this
/// input even after a small number of retry attempts. Tiny or strictly
/// aliphatic molecules can legitimately reject every mutator (e.g. the
/// topological-pathology mutator needs at least one aromatic bond);
/// callers should treat the `None` arm as a silent skip.
///
/// # Errors
///
/// Returns [`Error::FingerprintPreparation`] when `finge-rs` cannot prepare
/// the SMILES, or [`Error::InvalidBatch`] when the produced fingerprint
/// exceeds the configured sparse-index storage.
pub fn compute_negative_fingerprint_targets(
    molecule_id: u64,
    smiles: &Smiles,
    seed: u64,
    config: CountedEcfpConfig,
    scratch: &mut finge_rs::smiles_support::SmilesRdkitScratch,
) -> Result<Option<(FingerprintTargets, finge_rs::ViolationLabel)>> {
    use finge_rs::Fingerprint;
    use rand_chacha::ChaCha8Rng;
    use rand_chacha::rand_core::SeedableRng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut last_error: Option<finge_rs::MutatorError> = None;

    for _ in 0..NEGATIVE_FINGERPRINT_RETRIES {
        let prepared =
            scratch
                .try_prepare(smiles)
                .map_err(|source| Error::FingerprintPreparation {
                    molecule_id,
                    message: source.to_string(),
                })?;
        // `MutatorMix<'_, G>` captures the prepared graph's lifetime in
        // the mutator/predicate trait objects, so we rebuild the mix
        // inside the loop where the fresh scratch borrow is in scope.
        let mix = finge_rs::MutatorMix::with_default_mutators_and_predicates();
        match mix.sample(prepared, &mut rng) {
            Ok((invalidated, label)) => {
                let fingerprint = config.to_fingerprint().compute(&invalidated);
                let targets = fingerprint_targets_from_active_counts(
                    fingerprint.active_counts(),
                    config.size(),
                )?;
                return Ok(Some((targets, label)));
            }
            Err(error) => last_error = Some(error),
        }
    }

    // Every retry yielded a "no eligible target" outcome. Surfacing as
    // Ok(None) lets the preprocessor degrade gracefully on small or
    // all-aliphatic inputs without poisoning the positive emit.
    let _ = last_error;
    Ok(None)
}

/// Collects sparse `(index, count)` pairs from a finge-rs fingerprint into a
/// [`FingerprintTargets`] of the configured width.
fn fingerprint_targets_from_active_counts(
    pairs: impl IntoIterator<Item = (usize, u32)>,
    size: usize,
) -> Result<FingerprintTargets> {
    let mut indices = Vec::new();
    let mut counts = Vec::new();
    for (index, count) in pairs {
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
    fn negative_fingerprint_emits_label_and_distinct_indices_for_phenol() {
        let smiles: Smiles = "c1ccccc1O".parse().expect("valid SMILES");
        let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
        let positive =
            compute_fingerprint_targets(42, &smiles, CountedEcfpConfig::default(), &mut scratch)
                .expect("positive fingerprint");
        let (negative, label) = compute_negative_fingerprint_targets(
            42,
            &smiles,
            0x00C0_FFEE,
            CountedEcfpConfig::default(),
            &mut scratch,
        )
        .expect("negative fingerprint")
        .expect("phenol accepts at least one mutator");

        assert_eq!(negative.fingerprint_size(), positive.fingerprint_size());
        assert!(!label.is_empty(), "MutatorMix must set at least one bit");
        // A single-mutation negative usually differs from the positive in
        // at least one sparse bin; treat equality as a regression.
        assert_ne!(negative.indices(), positive.indices());
    }

    #[test]
    fn negative_fingerprint_is_deterministic_under_same_seed() {
        let smiles: Smiles = "c1ccc(N)cc1".parse().expect("valid SMILES");
        let mut scratch = finge_rs::smiles_support::SmilesRdkitScratch::default();
        let first = compute_negative_fingerprint_targets(
            7,
            &smiles,
            0xCAFE_BABE,
            CountedEcfpConfig::default(),
            &mut scratch,
        )
        .expect("first sample")
        .expect("aniline accepts mutators");
        let second = compute_negative_fingerprint_targets(
            7,
            &smiles,
            0xCAFE_BABE,
            CountedEcfpConfig::default(),
            &mut scratch,
        )
        .expect("second sample")
        .expect("aniline accepts mutators");

        assert_eq!(first.0.indices(), second.0.indices());
        assert_eq!(first.0.counts(), second.0.counts());
        assert_eq!(first.1.as_u8(), second.1.as_u8());
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
