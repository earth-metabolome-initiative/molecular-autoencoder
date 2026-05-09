//! Burn batch types for cached molecule shards.

#[cfg(feature = "std")]
use burn::data::dataloader::batcher::Batcher;
#[cfg(any(feature = "std", test))]
use burn::tensor::TensorData;
use burn::{
    prelude::*,
    tensor::{IndexingUpdateOp, Int},
};
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

use crate::{
    features::REGRESSION_TARGET_WIDTH, fingerprints::DEFAULT_ECFP_SIZE,
    ranking::TanimotoRankingBatch,
};

/// One vectorized molecule sample.
#[derive(Debug, Clone, PartialEq)]
pub struct MoleculeAutoencoderSample {
    /// Stable numeric molecule identifier.
    pub cid: u64,
    /// Sparse counted ECFP indices.
    pub fingerprint_indices: Vec<u16>,
    /// Sparse counted ECFP counts.
    pub fingerprint_counts: Vec<u16>,
    /// Normalized scalar descriptor targets.
    pub descriptor_targets: Vec<f32>,
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

    /// Materializes a dense `log1p(count)` tensor on the current backend device.
    #[must_use]
    pub fn to_dense_log_counts(&self) -> Tensor<B, 2> {
        let batch_size = self.batch_size();
        let device = self.log_counts.device();
        Tensor::<B, 2>::zeros([batch_size, self.fingerprint_size], &device).scatter(
            1,
            self.indices.clone(),
            self.log_counts.clone() * self.mask.clone(),
            IndexingUpdateOp::Add,
        )
    }
}

/// Batched tensors for molecular autoencoder training.
#[derive(Debug, Clone)]
pub struct MoleculeAutoencoderBatch<B: Backend> {
    /// Stable numeric molecule identifiers.
    pub cids: Vec<u64>,
    /// Sparse counted ECFP input and reconstruction target.
    pub fingerprints: SparseFingerprintBatch<B>,
    /// Normalized scalar descriptor targets.
    pub descriptor_targets: Tensor<B, 2>,
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
    /// Normalized scalar descriptor targets with shape `[batch, descriptor_width]`.
    pub descriptor_targets: Vec<f32>,
    /// Number of rows.
    pub batch_size: usize,
    /// Dense counted ECFP width represented by the sparse indices.
    pub fingerprint_size: usize,
    /// Number of padded sparse columns in this batch.
    pub max_nnz: usize,
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
        let max_nnz = items
            .iter()
            .map(|item| item.fingerprint_indices.len())
            .max()
            .unwrap_or(0)
            .max(1);

        let allocation_start = Instant::now();
        let mut cids = Vec::with_capacity(batch_size);
        let mut fingerprint_indices = vec![0_i64; batch_size * max_nnz];
        let mut fingerprint_counts = vec![0.0_f32; batch_size * max_nnz];
        let mut log_counts = vec![0.0_f32; batch_size * max_nnz];
        let mut fingerprint_mask = vec![0.0_f32; batch_size * max_nnz];
        let mut descriptor_targets = Vec::with_capacity(batch_size * self.descriptor_width);
        let sparse_allocation = allocation_start.elapsed();

        let sparse_fill_start = Instant::now();
        for (row, item) in items.into_iter().enumerate() {
            assert_eq!(
                item.fingerprint_indices.len(),
                item.fingerprint_counts.len(),
                "fingerprint indices and counts must have the same length"
            );
            assert_eq!(
                item.descriptor_targets.len(),
                self.descriptor_width,
                "descriptor target width does not match the batcher"
            );
            cids.push(item.cid);
            if item
                .fingerprint_indices
                .windows(2)
                .all(|window| window[0] < window[1])
            {
                for (column, (&index, &count)) in item
                    .fingerprint_indices
                    .iter()
                    .zip(&item.fingerprint_counts)
                    .enumerate()
                {
                    fill_sparse_slot(
                        self.fingerprint_size,
                        row,
                        max_nnz,
                        column,
                        index,
                        count,
                        &mut fingerprint_indices,
                        &mut fingerprint_counts,
                        &mut log_counts,
                        &mut fingerprint_mask,
                    );
                }
            } else {
                let mut pairs = item
                    .fingerprint_indices
                    .iter()
                    .copied()
                    .zip(item.fingerprint_counts.iter().copied())
                    .collect::<Vec<_>>();
                pairs.sort_unstable_by_key(|&(index, _)| index);
                for (column, (index, count)) in pairs.into_iter().enumerate() {
                    fill_sparse_slot(
                        self.fingerprint_size,
                        row,
                        max_nnz,
                        column,
                        index,
                        count,
                        &mut fingerprint_indices,
                        &mut fingerprint_counts,
                        &mut log_counts,
                        &mut fingerprint_mask,
                    );
                }
            }
            descriptor_targets.extend_from_slice(&item.descriptor_targets);
        }
        let sparse_fill = sparse_fill_start.elapsed();

        (
            MoleculeAutoencoderHostBatch {
                cids,
                fingerprint_indices,
                fingerprint_counts,
                log_counts,
                fingerprint_mask,
                descriptor_targets,
                batch_size,
                fingerprint_size: self.fingerprint_size,
                max_nnz,
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
            host.descriptor_targets.len(),
            host.batch_size * self.descriptor_width,
            "host batch descriptor shape does not match the batcher"
        );

        let tensor_build_start = Instant::now();
        let sparse_shape = [host.batch_size, host.max_nnz];
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
            descriptor_targets: Tensor::<B, 2>::from_data(
                TensorData::new(
                    host.descriptor_targets,
                    [host.batch_size, self.descriptor_width],
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
                },
                MoleculeAutoencoderSample {
                    cid: 2,
                    fingerprint_indices: vec![2],
                    fingerprint_counts: vec![4],
                    descriptor_targets: vec![0.3, 0.4],
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
                },
                MoleculeAutoencoderSample {
                    cid: 2,
                    fingerprint_indices: vec![2],
                    fingerprint_counts: vec![4],
                    descriptor_targets: vec![0.3, 0.4],
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
}
