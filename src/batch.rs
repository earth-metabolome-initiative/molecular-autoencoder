//! Burn batch types for cached molecule shards.

#[cfg(feature = "std")]
use burn::data::dataloader::batcher::Batcher;
#[cfg(any(feature = "std", test))]
use burn::tensor::TensorData;
use burn::{prelude::*, tensor::Int};
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

use crate::{
    features::REGRESSION_TARGET_WIDTH,
    fingerprints::{DEFAULT_ECFP_SIZE, VIOLATION_CLASS_COUNT},
    ranking::TanimotoRankingBatch,
};

/// One vectorized molecule sample.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MoleculeAutoencoderSample {
    /// Stable numeric molecule identifier.
    pub cid: u64,
    /// Sparse counted ECFP indices (encoder input).
    pub fingerprint_indices: Vec<u16>,
    /// Sparse counted ECFP counts (encoder input).
    pub fingerprint_counts: Vec<u16>,
    /// Normalized scalar descriptor targets.
    pub descriptor_targets: Vec<f32>,
    /// Multi-label [`finge_rs::ViolationClass`] bitset for this sample
    /// (`0` for positives).
    #[cfg_attr(feature = "std", doc = "")]
    pub violation_label: u8,
    /// Decoder reconstruction-target sparse indices. Leave empty when
    /// this sample is a positive — the batcher then aliases the decoder
    /// target to the encoder input. Required (and non-empty when the
    /// fingerprint has any active bins) for negatives.
    pub target_fingerprint_indices: Vec<u16>,
    /// Decoder reconstruction-target sparse counts, aligned with
    /// [`Self::target_fingerprint_indices`].
    pub target_fingerprint_counts: Vec<u16>,
}

/// Padded sparse counted-fingerprint tensors.
#[derive(Debug, Clone)]
pub struct SparseFingerprintBatch<B: Backend> {
    /// Sparse counted ECFP indices with shape `[batch, max_nnz]`.
    pub indices: Tensor<B, 2, Int>,
    /// Sparse raw counted ECFP values with shape `[batch, max_nnz]`.
    pub counts: Tensor<B, 2>,
    /// Sparse `log1p(count)` values with shape `[batch, max_nnz]`.
    pub log_counts: Tensor<B, 2>,
    /// Padding mask with shape `[batch, max_nnz]`; active bins are `1`.
    pub mask: Tensor<B, 2>,
    /// Full counted ECFP width.
    pub fingerprint_size: usize,
    /// Number of padded sparse columns in this batch.
    pub max_nnz: usize,
}

impl<B: Backend> SparseFingerprintBatch<B> {
    /// Returns the batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.indices.dims()[0]
    }
}

/// Batched tensors for molecular autoencoder training.
#[derive(Debug, Clone)]
pub struct MoleculeAutoencoderBatch<B: Backend> {
    /// Stable numeric molecule identifiers.
    pub cids: Vec<u64>,
    /// Sparse counted ECFP input fed to the encoder.
    pub fingerprints: SparseFingerprintBatch<B>,
    /// Decoder reconstruction target. Equal (row-by-row) to
    /// [`Self::fingerprints`] for positive rows, distinct for negatives
    /// where it holds the source positive's fingerprint (denoising /
    /// correction framing).
    pub target_fingerprints: SparseFingerprintBatch<B>,
    /// Normalized scalar descriptor targets.
    pub descriptor_targets: Tensor<B, 2>,
    /// Multi-label violation targets with shape `[batch, 8]`, expanded
    /// from each sample's `violation_label` bitset (multi-hot float).
    pub violation_targets: Tensor<B, 2>,
    /// Metric-derived latent Tanimoto ranking labels.
    pub tanimoto_ranking: TanimotoRankingBatch<B>,
}

/// Sparse CPU-side batch prepared before tensor upload.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq)]
pub struct MoleculeAutoencoderHostBatch {
    /// Stable numeric molecule identifiers.
    pub cids: Vec<u64>,
    /// Padded sparse counted ECFP indices with shape `[batch, max_nnz]`.
    pub fingerprint_indices: Vec<i64>,
    /// Padded sparse raw counted ECFP values with shape `[batch, max_nnz]`.
    pub fingerprint_counts: Vec<f32>,
    /// Padded sparse `log1p(counted ECFP)` values with shape `[batch, max_nnz]`.
    pub log_counts: Vec<f32>,
    /// Padded sparse active-entry mask with shape `[batch, max_nnz]`.
    pub fingerprint_mask: Vec<f32>,
    /// Padded decoder-target sparse indices with shape
    /// `[batch, target_max_nnz]`. For positive rows this storage
    /// mirrors the input fingerprint; for negative rows it carries the
    /// source positive's fingerprint (denoising target).
    pub target_fingerprint_indices: Vec<i64>,
    /// Padded decoder-target sparse counts with shape
    /// `[batch, target_max_nnz]`.
    pub target_fingerprint_counts: Vec<f32>,
    /// Padded decoder-target sparse `log1p(count)` values.
    pub target_log_counts: Vec<f32>,
    /// Padded decoder-target sparse active-entry mask.
    pub target_fingerprint_mask: Vec<f32>,
    /// Normalized scalar descriptor targets with shape `[batch, descriptor_width]`.
    pub descriptor_targets: Vec<f32>,
    /// Multi-hot violation targets with shape `[batch, 8]` (row-major,
    /// `1.0` where the bit is set, `0.0` elsewhere).
    pub violation_targets: Vec<f32>,
    /// Number of rows.
    pub batch_size: usize,
    /// Dense counted ECFP width represented by the sparse indices.
    pub fingerprint_size: usize,
    /// Number of padded sparse columns in this batch (encoder input).
    pub max_nnz: usize,
    /// Number of padded sparse columns in the decoder-target batch.
    pub target_max_nnz: usize,
    /// Descriptor regression target width.
    pub descriptor_width: usize,
}

/// Converts sparse samples into Burn tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoleculeAutoencoderBatcher {
    /// Full dense counted ECFP width.
    pub fingerprint_size: usize,
    /// Descriptor regression target width.
    pub descriptor_width: usize,
}

/// CPU-side and tensor-construction timing for one sparse batch build.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoleculeBatchBuildProfile {
    /// Sparse vector allocation and zero-initialization time.
    pub sparse_allocation: Duration,
    /// Sparse row packing into padded CPU buffers.
    pub sparse_fill: Duration,
    /// Burn tensor construction from sparse CPU buffers.
    pub tensor_build: Duration,
}

#[cfg(feature = "std")]
impl MoleculeBatchBuildProfile {
    /// Total measured batch build time.
    #[must_use]
    pub fn total(self) -> Duration {
        self.sparse_allocation + self.sparse_fill + self.tensor_build
    }
}

impl Default for MoleculeAutoencoderBatcher {
    fn default() -> Self {
        Self {
            fingerprint_size: DEFAULT_ECFP_SIZE,
            descriptor_width: REGRESSION_TARGET_WIDTH,
        }
    }
}

impl MoleculeAutoencoderBatcher {
    /// Creates a batcher for a concrete cached shard shape.
    #[must_use]
    pub const fn new(fingerprint_size: usize, descriptor_width: usize) -> Self {
        Self {
            fingerprint_size,
            descriptor_width,
        }
    }

    /// Internal entry point that routes either through the profiled
    /// host-batch path (under `std`) or the no-std unprofiled path.
    /// Used by tests so the shared assertions don't drift between cfgs.
    #[cfg(any(feature = "std", test))]
    fn batch_inner<B: Backend>(
        &self,
        items: Vec<MoleculeAutoencoderSample>,
        device: &B::Device,
    ) -> MoleculeAutoencoderBatch<B> {
        #[cfg(feature = "std")]
        {
            self.batch_profiled(items, device).0
        }
        #[cfg(not(feature = "std"))]
        {
            self.batch_unprofiled(items, device)
        }
    }

    /// Builds a sparse batch and returns loader-side timing.
    #[cfg(feature = "std")]
    pub fn batch_profiled<B: Backend>(
        &self,
        items: Vec<MoleculeAutoencoderSample>,
        device: &B::Device,
    ) -> (MoleculeAutoencoderBatch<B>, MoleculeBatchBuildProfile) {
        let (host, mut profile) = self.host_batch_profiled(items);
        let (batch, tensor_profile) = self.batch_host_profiled(host, device);
        profile.tensor_build = tensor_profile.tensor_build;
        (batch, profile)
    }

    /// Builds a sparse CPU-side batch without uploading tensors to a backend.
    ///
    /// # Panics
    ///
    /// Panics when the input batch is empty, sample sparse arrays have different
    /// lengths, descriptor widths do not match the batcher, or sparse indices
    /// exceed the configured fingerprint width.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn host_batch_profiled(
        &self,
        items: Vec<MoleculeAutoencoderSample>,
    ) -> (MoleculeAutoencoderHostBatch, MoleculeBatchBuildProfile) {
        assert!(!items.is_empty(), "molecule batches are never empty");
        let batch_size = items.len();
        // Decoder-target padding follows the largest target slice across
        // rows. For positive rows the target aliases the input (the
        // sample carries an empty target slice) and the per-row "target
        // nnz" used for padding is the input nnz; for negative rows
        // it's the explicit target slice length.
        let max_nnz = items
            .iter()
            .map(|item| item.fingerprint_indices.len())
            .max()
            .unwrap_or(0)
            .max(1);
        let target_max_nnz = items
            .iter()
            .map(effective_target_nnz)
            .max()
            .unwrap_or(0)
            .max(1);

        let allocation_start = Instant::now();
        let mut cids = Vec::with_capacity(batch_size);
        let mut fingerprint_indices = vec![0_i64; batch_size * max_nnz];
        let mut fingerprint_counts = vec![0.0_f32; batch_size * max_nnz];
        let mut log_counts = vec![0.0_f32; batch_size * max_nnz];
        let mut fingerprint_mask = vec![0.0_f32; batch_size * max_nnz];
        let mut target_fingerprint_indices = vec![0_i64; batch_size * target_max_nnz];
        let mut target_fingerprint_counts = vec![0.0_f32; batch_size * target_max_nnz];
        let mut target_log_counts = vec![0.0_f32; batch_size * target_max_nnz];
        let mut target_fingerprint_mask = vec![0.0_f32; batch_size * target_max_nnz];
        let mut descriptor_targets = Vec::with_capacity(batch_size * self.descriptor_width);
        let mut violation_targets = vec![0.0_f32; batch_size.saturating_mul(VIOLATION_CLASS_COUNT)];
        let sparse_allocation = allocation_start.elapsed();

        let sparse_fill_start = Instant::now();
        for (row, item) in items.into_iter().enumerate() {
            assert_eq!(
                item.fingerprint_indices.len(),
                item.fingerprint_counts.len(),
                "fingerprint indices and counts must have the same length"
            );
            assert_eq!(
                item.target_fingerprint_indices.len(),
                item.target_fingerprint_counts.len(),
                "target fingerprint indices and counts must have the same length"
            );
            assert_eq!(
                item.descriptor_targets.len(),
                self.descriptor_width,
                "descriptor target width does not match the batcher"
            );
            cids.push(item.cid);

            fill_sparse_row(
                self.fingerprint_size,
                row,
                max_nnz,
                &item.fingerprint_indices,
                &item.fingerprint_counts,
                &mut fingerprint_indices,
                &mut fingerprint_counts,
                &mut log_counts,
                &mut fingerprint_mask,
            );

            // Positive rows (empty target slice + label == 0) reuse the
            // encoder input as the decoder target; negative rows carry
            // an explicit target slice.
            let (target_indices_src, target_counts_src) =
                if item.target_fingerprint_indices.is_empty() && item.violation_label == 0 {
                    (
                        item.fingerprint_indices.as_slice(),
                        item.fingerprint_counts.as_slice(),
                    )
                } else {
                    (
                        item.target_fingerprint_indices.as_slice(),
                        item.target_fingerprint_counts.as_slice(),
                    )
                };
            fill_sparse_row(
                self.fingerprint_size,
                row,
                target_max_nnz,
                target_indices_src,
                target_counts_src,
                &mut target_fingerprint_indices,
                &mut target_fingerprint_counts,
                &mut target_log_counts,
                &mut target_fingerprint_mask,
            );

            descriptor_targets.extend_from_slice(&item.descriptor_targets);
            let target_row_offset = row * VIOLATION_CLASS_COUNT;
            for bit in 0..VIOLATION_CLASS_COUNT {
                if (item.violation_label >> bit) & 1 == 1 {
                    violation_targets[target_row_offset + bit] = 1.0;
                }
            }
        }
        let sparse_fill = sparse_fill_start.elapsed();

        (
            MoleculeAutoencoderHostBatch {
                cids,
                fingerprint_indices,
                fingerprint_counts,
                log_counts,
                fingerprint_mask,
                target_fingerprint_indices,
                target_fingerprint_counts,
                target_log_counts,
                target_fingerprint_mask,
                descriptor_targets,
                violation_targets,
                batch_size,
                fingerprint_size: self.fingerprint_size,
                max_nnz,
                target_max_nnz,
                descriptor_width: self.descriptor_width,
            },
            MoleculeBatchBuildProfile {
                sparse_allocation,
                sparse_fill,
                tensor_build: Duration::ZERO,
            },
        )
    }

    /// Uploads a sparse CPU-side batch to the provided backend device.
    ///
    /// # Panics
    ///
    /// Panics when the host batch shape does not match this batcher.
    #[cfg(feature = "std")]
    pub fn batch_host_profiled<B: Backend>(
        &self,
        host: MoleculeAutoencoderHostBatch,
        device: &B::Device,
    ) -> (MoleculeAutoencoderBatch<B>, MoleculeBatchBuildProfile) {
        assert_eq!(
            host.fingerprint_size, self.fingerprint_size,
            "host batch fingerprint width does not match the batcher"
        );
        assert_eq!(
            host.descriptor_width, self.descriptor_width,
            "host batch descriptor width does not match the batcher"
        );
        assert_eq!(
            host.fingerprint_indices.len(),
            host.batch_size * host.max_nnz,
            "host batch sparse-index shape does not match the batcher"
        );
        assert_eq!(
            host.fingerprint_counts.len(),
            host.batch_size * host.max_nnz,
            "host batch sparse-count shape does not match the batcher"
        );
        assert_eq!(
            host.log_counts.len(),
            host.batch_size * host.max_nnz,
            "host batch sparse log-count shape does not match the batcher"
        );
        assert_eq!(
            host.fingerprint_mask.len(),
            host.batch_size * host.max_nnz,
            "host batch sparse-mask shape does not match the batcher"
        );
        assert_eq!(
            host.target_fingerprint_indices.len(),
            host.batch_size * host.target_max_nnz,
            "host batch target sparse-index shape does not match the batcher"
        );
        assert_eq!(
            host.target_fingerprint_counts.len(),
            host.batch_size * host.target_max_nnz,
            "host batch target sparse-count shape does not match the batcher"
        );
        assert_eq!(
            host.target_log_counts.len(),
            host.batch_size * host.target_max_nnz,
            "host batch target sparse log-count shape does not match the batcher"
        );
        assert_eq!(
            host.target_fingerprint_mask.len(),
            host.batch_size * host.target_max_nnz,
            "host batch target sparse-mask shape does not match the batcher"
        );
        assert_eq!(
            host.descriptor_targets.len(),
            host.batch_size * self.descriptor_width,
            "host batch descriptor shape does not match the batcher"
        );
        assert_eq!(
            host.violation_targets.len(),
            host.batch_size * VIOLATION_CLASS_COUNT,
            "host batch violation-target shape does not match the batcher"
        );

        let tensor_build_start = Instant::now();
        let sparse_shape = [host.batch_size, host.max_nnz];
        let target_sparse_shape = [host.batch_size, host.target_max_nnz];
        let batch = MoleculeAutoencoderBatch {
            cids: host.cids,
            fingerprints: SparseFingerprintBatch {
                indices: Tensor::<B, 2, Int>::from_data(
                    TensorData::new(host.fingerprint_indices, sparse_shape),
                    device,
                ),
                counts: Tensor::<B, 2>::from_data(
                    TensorData::new(host.fingerprint_counts, sparse_shape),
                    device,
                ),
                log_counts: Tensor::<B, 2>::from_data(
                    TensorData::new(host.log_counts, sparse_shape),
                    device,
                ),
                mask: Tensor::<B, 2>::from_data(
                    TensorData::new(host.fingerprint_mask, sparse_shape),
                    device,
                ),
                fingerprint_size: self.fingerprint_size,
                max_nnz: host.max_nnz,
            },
            target_fingerprints: SparseFingerprintBatch {
                indices: Tensor::<B, 2, Int>::from_data(
                    TensorData::new(host.target_fingerprint_indices, target_sparse_shape),
                    device,
                ),
                counts: Tensor::<B, 2>::from_data(
                    TensorData::new(host.target_fingerprint_counts, target_sparse_shape),
                    device,
                ),
                log_counts: Tensor::<B, 2>::from_data(
                    TensorData::new(host.target_log_counts, target_sparse_shape),
                    device,
                ),
                mask: Tensor::<B, 2>::from_data(
                    TensorData::new(host.target_fingerprint_mask, target_sparse_shape),
                    device,
                ),
                fingerprint_size: self.fingerprint_size,
                max_nnz: host.target_max_nnz,
            },
            descriptor_targets: Tensor::<B, 2>::from_data(
                TensorData::new(
                    host.descriptor_targets,
                    [host.batch_size, self.descriptor_width],
                ),
                device,
            ),
            violation_targets: Tensor::<B, 2>::from_data(
                TensorData::new(
                    host.violation_targets,
                    [host.batch_size, VIOLATION_CLASS_COUNT],
                ),
                device,
            ),
            tanimoto_ranking: TanimotoRankingBatch::zeros(host.batch_size, device),
        };
        let tensor_build = tensor_build_start.elapsed();

        (
            batch,
            MoleculeBatchBuildProfile {
                sparse_allocation: Duration::ZERO,
                sparse_fill: Duration::ZERO,
                tensor_build,
            },
        )
    }
}

/// Writes one `(row, column)` slot of the sparse host-side buffers
/// while validating the fingerprint index and converting the count into
/// the dense / log-count layouts the batcher emits.
#[cfg(feature = "std")]
#[allow(clippy::too_many_arguments)]
fn fill_sparse_slot(
    fingerprint_size: usize,
    row: usize,
    max_nnz: usize,
    column: usize,
    index: u16,
    count: u16,
    fingerprint_indices: &mut [i64],
    fingerprint_counts: &mut [f32],
    log_counts: &mut [f32],
    fingerprint_mask: &mut [f32],
) {
    let dense_index = usize::from(index);
    assert!(
        dense_index < fingerprint_size,
        "fingerprint index exceeds configured width"
    );
    let offset = row * max_nnz + column;
    fingerprint_indices[offset] = i64::from(index);
    let count = f32::from(count);
    fingerprint_counts[offset] = count;
    log_counts[offset] = count.ln_1p();
    fingerprint_mask[offset] = 1.0;
}

/// Fills one padded sparse row in the host-side buffers, sorting the
/// `(index, count)` pairs by index when they are not already strictly
/// increasing so the downstream Tanimoto kernel's lockstep walk holds.
#[cfg(feature = "std")]
#[allow(clippy::too_many_arguments)]
fn fill_sparse_row(
    fingerprint_size: usize,
    row: usize,
    max_nnz: usize,
    indices: &[u16],
    counts: &[u16],
    fingerprint_indices: &mut [i64],
    fingerprint_counts: &mut [f32],
    log_counts: &mut [f32],
    fingerprint_mask: &mut [f32],
) {
    if indices.windows(2).all(|window| window[0] < window[1]) {
        for (column, (&index, &count)) in indices.iter().zip(counts).enumerate() {
            fill_sparse_slot(
                fingerprint_size,
                row,
                max_nnz,
                column,
                index,
                count,
                fingerprint_indices,
                fingerprint_counts,
                log_counts,
                fingerprint_mask,
            );
        }
    } else {
        let mut pairs = indices
            .iter()
            .copied()
            .zip(counts.iter().copied())
            .collect::<Vec<_>>();
        pairs.sort_unstable_by_key(|&(index, _)| index);
        for (column, (index, count)) in pairs.into_iter().enumerate() {
            fill_sparse_slot(
                fingerprint_size,
                row,
                max_nnz,
                column,
                index,
                count,
                fingerprint_indices,
                fingerprint_counts,
                log_counts,
                fingerprint_mask,
            );
        }
    }
}

/// Number of decoder-target sparse entries this sample contributes to
/// the batch's `target_max_nnz` calculation. Positive rows alias to the
/// encoder input, so their effective target nnz is the input nnz;
/// negative rows use their explicit target slice length.
#[cfg(feature = "std")]
fn effective_target_nnz(item: &MoleculeAutoencoderSample) -> usize {
    if item.target_fingerprint_indices.is_empty() && item.violation_label == 0 {
        item.fingerprint_indices.len()
    } else {
        item.target_fingerprint_indices.len()
    }
}

#[cfg(feature = "std")]
impl<B: Backend> Batcher<B, MoleculeAutoencoderSample, MoleculeAutoencoderBatch<B>>
    for MoleculeAutoencoderBatcher
{
    fn batch(
        &self,
        items: Vec<MoleculeAutoencoderSample>,
        device: &B::Device,
    ) -> MoleculeAutoencoderBatch<B> {
        self.batch_inner(items, device)
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;

    fn sample() -> MoleculeAutoencoderSample {
        MoleculeAutoencoderSample {
            cid: 1,
            fingerprint_indices: vec![1],
            fingerprint_counts: vec![2],
            descriptor_targets: vec![0.1, 0.2],
            ..Default::default()
        }
    }

    #[test]
    fn batcher_preserves_shapes() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(16, 2);
        let batch: MoleculeAutoencoderBatch<B> = batcher.batch_inner(
            vec![
                MoleculeAutoencoderSample {
                    cid: 1,
                    fingerprint_indices: vec![1, 3],
                    fingerprint_counts: vec![2, 1],
                    descriptor_targets: vec![0.1, 0.2],
                    ..Default::default()
                },
                MoleculeAutoencoderSample {
                    cid: 2,
                    fingerprint_indices: vec![2],
                    fingerprint_counts: vec![4],
                    descriptor_targets: vec![0.3, 0.4],
                    ..Default::default()
                },
            ],
            &device,
        );

        assert_eq!(batch.cids, vec![1, 2]);
        assert_eq!(batch.fingerprints.indices.dims(), [2, 2]);
        assert_eq!(batch.fingerprints.counts.dims(), [2, 2]);
        assert_eq!(batch.fingerprints.log_counts.dims(), [2, 2]);
        assert_eq!(batch.fingerprints.mask.dims(), [2, 2]);
        assert_eq!(batch.fingerprints.fingerprint_size, 16);
        assert_eq!(batch.fingerprints.max_nnz, 2);
        assert_eq!(batch.descriptor_targets.dims(), [2, 2]);
    }

    #[test]
    fn batcher_pads_sparse_rows_without_truncating() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(16, 2);
        let batch: MoleculeAutoencoderBatch<B> = batcher.batch_inner(
            vec![
                MoleculeAutoencoderSample {
                    cid: 1,
                    fingerprint_indices: vec![1, 3, 5],
                    fingerprint_counts: vec![2, 1, 4],
                    descriptor_targets: vec![0.1, 0.2],
                    ..Default::default()
                },
                MoleculeAutoencoderSample {
                    cid: 2,
                    fingerprint_indices: vec![2],
                    fingerprint_counts: vec![4],
                    descriptor_targets: vec![0.3, 0.4],
                    ..Default::default()
                },
            ],
            &device,
        );

        let indices = batch
            .fingerprints
            .indices
            .to_data()
            .as_slice::<i64>()
            .expect("indices should be i64")
            .to_vec();
        let mask = batch
            .fingerprints
            .mask
            .to_data()
            .as_slice::<f32>()
            .expect("mask should be f32")
            .to_vec();
        let counts = batch
            .fingerprints
            .counts
            .to_data()
            .as_slice::<f32>()
            .expect("counts should be f32")
            .to_vec();

        assert_eq!(batch.fingerprints.indices.dims(), [2, 3]);
        assert_eq!(indices, vec![1, 3, 5, 2, 0, 0]);
        assert_eq!(counts, vec![2.0, 1.0, 4.0, 4.0, 0.0, 0.0]);
        assert_eq!(mask, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "molecule batches are never empty")]
    fn batcher_panics_for_empty_batches() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let _ = MoleculeAutoencoderBatcher::new(16, 2).batch_inner::<B>(Vec::new(), &device);
    }

    #[test]
    #[should_panic(expected = "fingerprint indices and counts must have the same length")]
    fn batcher_panics_for_mismatched_sparse_lengths() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mut item = sample();
        item.fingerprint_counts.clear();

        let _ = MoleculeAutoencoderBatcher::new(16, 2).batch_inner::<B>(vec![item], &device);
    }

    #[test]
    #[should_panic(expected = "descriptor target width does not match the batcher")]
    fn batcher_panics_for_descriptor_width_mismatch() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mut item = sample();
        item.descriptor_targets.pop();

        let _ = MoleculeAutoencoderBatcher::new(16, 2).batch_inner::<B>(vec![item], &device);
    }

    #[test]
    #[should_panic(expected = "fingerprint index exceeds configured width")]
    fn batcher_panics_for_out_of_range_fingerprint_index() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mut item = sample();
        item.fingerprint_indices = vec![16];

        let _ = MoleculeAutoencoderBatcher::new(16, 2).batch_inner::<B>(vec![item], &device);
    }

    #[cfg(feature = "std")]
    #[test]
    fn batch_build_profile_total_sums_phase_durations() {
        let profile = MoleculeBatchBuildProfile {
            sparse_allocation: std::time::Duration::from_micros(100),
            sparse_fill: std::time::Duration::from_micros(200),
            tensor_build: std::time::Duration::from_micros(300),
        };
        assert_eq!(profile.total(), std::time::Duration::from_micros(600));
    }

    #[test]
    fn batcher_aliases_target_fingerprint_to_input_for_positive_only_batch() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(16, 2);
        let batch: MoleculeAutoencoderBatch<B> = batcher.batch_inner(
            vec![
                MoleculeAutoencoderSample {
                    cid: 1,
                    fingerprint_indices: vec![1, 3],
                    fingerprint_counts: vec![2, 1],
                    descriptor_targets: vec![0.1, 0.2],
                    ..Default::default()
                },
                MoleculeAutoencoderSample {
                    cid: 2,
                    fingerprint_indices: vec![5],
                    fingerprint_counts: vec![4],
                    descriptor_targets: vec![0.3, 0.4],
                    ..Default::default()
                },
            ],
            &device,
        );

        let input_indices = batch
            .fingerprints
            .indices
            .to_data()
            .as_slice::<i64>()
            .expect("i64 indices")
            .to_vec();
        let target_indices = batch
            .target_fingerprints
            .indices
            .to_data()
            .as_slice::<i64>()
            .expect("i64 target indices")
            .to_vec();
        assert_eq!(input_indices, target_indices);

        let violation = batch
            .violation_targets
            .to_data()
            .as_slice::<f32>()
            .expect("f32 violation targets")
            .to_vec();
        assert_eq!(batch.violation_targets.dims(), [2, VIOLATION_CLASS_COUNT]);
        assert!(
            violation.iter().all(|&value| value == 0.0),
            "all-zero label bitsets should produce an all-zero multi-hot target"
        );
    }

    #[test]
    fn batcher_routes_negative_decoder_target_to_explicit_source_fingerprint() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(16, 2);
        let batch: MoleculeAutoencoderBatch<B> = batcher.batch_inner(
            vec![
                MoleculeAutoencoderSample {
                    cid: 1,
                    fingerprint_indices: vec![1, 3],
                    fingerprint_counts: vec![2, 1],
                    descriptor_targets: vec![0.1, 0.2],
                    ..Default::default()
                },
                MoleculeAutoencoderSample {
                    cid: 1,
                    fingerprint_indices: vec![7, 9],
                    fingerprint_counts: vec![3, 4],
                    descriptor_targets: vec![0.1, 0.2],
                    violation_label: 0b0000_1010,
                    target_fingerprint_indices: vec![1, 3],
                    target_fingerprint_counts: vec![2, 1],
                },
            ],
            &device,
        );

        let target_indices = batch
            .target_fingerprints
            .indices
            .to_data()
            .as_slice::<i64>()
            .expect("i64 target indices")
            .to_vec();
        // Row 0 is a positive (target aliases input -> 1, 3),
        // row 1 is a negative whose decoder target points back to the
        // source positive's fingerprint (also 1, 3).
        assert_eq!(target_indices, vec![1, 3, 1, 3]);

        let violation = batch
            .violation_targets
            .to_data()
            .as_slice::<f32>()
            .expect("f32 violation targets")
            .to_vec();
        // Row 0 is fully zero; row 1 sets bits 1 and 3.
        assert_eq!(violation[0..VIOLATION_CLASS_COUNT], [0.0; 8]);
        assert_eq!(
            violation[VIOLATION_CLASS_COUNT..2 * VIOLATION_CLASS_COUNT],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
    }
}
