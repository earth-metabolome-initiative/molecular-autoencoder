//! Multi-threaded sparse-shard dataloader and batch iteration.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant},
};

use burn::tensor::backend::Backend;
use molecular_autoencoder::{
    DataSplit, MoleculeAutoencoderBatch, MoleculeAutoencoderBatcher, MoleculeAutoencoderHostBatch,
    MoleculeAutoencoderSample, MoleculeBatchBuildProfile, MoleculeShardRow, SparseMoleculeShard,
};

use crate::{
    AppResult,
    checkpoint::{CachedShardInfo, validate_shard_shape},
    invalid_input,
    metrics::{LoaderProfileReporter, split_label},
};

/// Signal returned by the per-batch consumer callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchControl {
    /// Keep iterating.
    Continue,
    /// Stop the loader and unwind without consuming further batches.
    Stop,
}

/// Per-component timing for one dataloader batch.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchLoadProfile {
    /// Wall time spent reading shard bytes from disk.
    pub shard_read: Duration,
    /// Wall time spent scanning shard rows to filter by split.
    pub row_scan: Duration,
    /// Wall time spent allocating sparse buffers.
    pub sparse_allocation: Duration,
    /// Wall time spent populating sparse buffers.
    pub sparse_fill: Duration,
    /// Wall time spent constructing backend tensors.
    pub tensor_build: Duration,
    /// Sum of all producer-side stages for one batch.
    pub producer_total: Duration,
}

/// Iteration parameters shared by sync and multi-threaded dataloader paths.
pub struct BatchIterationContext<'a, B: Backend> {
    /// Shards to iterate over in manifest order.
    pub shards: &'a [CachedShardInfo],
    /// Split to keep (train vs. validation).
    pub split: DataSplit,
    /// Batcher used to assemble samples into tensors.
    pub batcher: MoleculeAutoencoderBatcher,
    /// Backend device that owns the produced tensors.
    pub device: &'a B::Device,
    /// Target number of samples per batch.
    pub batch_size: usize,
    /// Optional upper bound on the number of batches to emit.
    pub max_batches: Option<usize>,
    /// Number of host-side loader threads; `0` selects the sync path.
    pub loader_workers: usize,
    /// Depth of the device-side prefetch queue; `0` disables it.
    pub device_prefetch_batches: usize,
    /// Period (in batches) between loader-profile log lines.
    pub loader_profile_every: usize,
    /// Validation-split share encoded as parts per mille.
    pub validation_per_mille: u16,
}

/// One contiguous slice of a shard scheduled for a single batch build.
#[derive(Debug, Clone)]
struct CachedBatchPlan {
    /// Shard the plan reads from.
    shard: CachedShardInfo,
    /// First row index covered by this plan (inclusive).
    start_row: usize,
    /// Last row index covered by this plan (exclusive).
    end_row: usize,
    /// Split to keep when scanning the planned row window.
    split: DataSplit,
}

/// Sequence of [`CachedBatchPlan`] entries that fully cover a dataset run.
#[derive(Debug, Clone)]
struct CachedBatchPlanDataset {
    /// Ordered list of plans consumed by loader threads.
    plans: Vec<CachedBatchPlan>,
}

/// Host-side staged batch produced by a loader worker before tensorization.
#[derive(Debug, Clone)]
struct CachedLoaderBatch {
    /// Host batch ready for tensorization.
    host: MoleculeAutoencoderHostBatch,
    /// Number of samples in the batch.
    rows: usize,
    /// Per-stage producer timing accumulated so far.
    profile: BatchLoadProfile,
}

/// Device-side staged batch carrying backend tensors plus timing.
struct DeviceLoaderBatch<B: Backend> {
    /// Backend-native batch ready to be consumed.
    batch: MoleculeAutoencoderBatch<B>,
    /// Number of samples in the batch.
    rows: usize,
    /// Per-stage producer timing accumulated up to tensorization.
    profile: BatchLoadProfile,
}

/// Channel payload sent from a loader worker to the consumer / uploader.
enum HostLoaderMessage {
    /// Successful host-side batch ready for tensor build.
    Batch(Box<CachedLoaderBatch>),
    /// Loader-side error message; aborts iteration.
    Error(String),
}

/// Channel payload sent from the uploader thread to the consumer.
enum DeviceLoaderMessage<B: Backend> {
    /// Successful device-side batch ready for consumption.
    Batch(DeviceLoaderBatch<B>),
    /// Upload-side error message; aborts iteration.
    Error(String),
}

/// Iterates batches of the requested split, dispatching to the multi-threaded
/// dataloader when `loader_workers > 0` and to the sync path otherwise.
pub fn for_each_batch<B, F>(context: BatchIterationContext<'_, B>, consume: F) -> AppResult<()>
where
    B: Backend,
    B::Device: Clone + Send + Sync,
    MoleculeAutoencoderBatch<B>: Send,
    F: FnMut(MoleculeAutoencoderBatch<B>, usize, Duration) -> AppResult<BatchControl>,
{
    if context.batch_size == 0 {
        return Err(invalid_input("batch size must be greater than zero"));
    }
    if context.loader_workers > 0 {
        for_each_batch_dataloader(context, consume)
    } else {
        for_each_batch_sync(context, consume)
    }
}

/// Multi-threaded dataloader: producer threads read shards in parallel, an
/// optional uploader stages device tensors, and the consumer drains batches.
fn for_each_batch_dataloader<B, F>(
    context: BatchIterationContext<'_, B>,
    mut consume: F,
) -> AppResult<()>
where
    B: Backend,
    B::Device: Clone + Send + Sync,
    MoleculeAutoencoderBatch<B>: Send,
    F: FnMut(MoleculeAutoencoderBatch<B>, usize, Duration) -> AppResult<BatchControl>,
{
    let BatchIterationContext {
        shards,
        split,
        batcher,
        device,
        batch_size,
        max_batches,
        loader_workers,
        device_prefetch_batches,
        loader_profile_every,
        validation_per_mille,
        ..
    } = context;
    let dataset =
        CachedBatchPlanDataset::new(shards, split, batch_size, max_batches, validation_per_mille)?;
    if dataset.plans.is_empty() {
        return Ok(());
    }

    let plans = Arc::new(dataset.plans);
    let next_plan = AtomicUsize::new(0);
    let (sender, receiver) = mpsc::sync_channel(loader_workers.max(1));
    let phase = split_label(split);

    thread::scope(|scope| {
        for _ in 0..loader_workers {
            let sender = sender.clone();
            let plans = Arc::clone(&plans);
            let next_plan = &next_plan;
            scope.spawn(move || {
                produce_host_loader_batches(batcher, plans, next_plan, sender);
            });
        }
        drop(sender);

        if device_prefetch_batches > 0 {
            let (device_sender, device_receiver) = mpsc::sync_channel(device_prefetch_batches);
            let device = device.clone();
            scope.spawn(move || {
                upload_host_loader_batches::<B>(receiver, batcher, device, device_sender);
            });
            consume_device_loader_batches(
                device_receiver,
                phase,
                loader_profile_every,
                &mut consume,
            )
        } else {
            consume_host_loader_batches(
                receiver,
                batcher,
                device,
                phase,
                loader_profile_every,
                &mut consume,
            )
        }
    })
}

/// Single-threaded dataloader used when `loader_workers == 0` — reads shards
/// sequentially and tensorizes batches inline.
fn for_each_batch_sync<B, F>(context: BatchIterationContext<'_, B>, mut consume: F) -> AppResult<()>
where
    B: Backend,
    F: FnMut(MoleculeAutoencoderBatch<B>, usize, Duration) -> AppResult<BatchControl>,
{
    let BatchIterationContext {
        shards,
        split,
        batcher,
        device,
        batch_size,
        max_batches,
        loader_profile_every,
        ..
    } = context;

    let mut pending = Vec::with_capacity(batch_size);
    let mut batches = 0_usize;
    let mut data_start = Instant::now();
    let mut shard_read_pending = Duration::ZERO;
    let mut row_scan_pending = Duration::ZERO;
    let mut loader_profile = LoaderProfileReporter::new(split_label(split), loader_profile_every);

    'shards: for shard_info in shards {
        let shard_path = &shard_info.path;
        let read_start = Instant::now();
        let shard = SparseMoleculeShard::read_from_path(shard_path)?;
        shard_read_pending += read_start.elapsed();
        validate_shard_shape(&shard, batcher, shard_path)?;
        let mut scan_start = Instant::now();
        for row_index in 0..shard.len() {
            let row = shard
                .row(row_index)
                .ok_or_else(|| invalid_input("shard row disappeared during iteration"))?;
            if row.split() != split {
                continue;
            }
            pending.push(sample_from_row(row));
            if pending.len() == batch_size {
                row_scan_pending += scan_start.elapsed();
                let items = std::mem::replace(&mut pending, Vec::with_capacity(batch_size));
                let (batch, build_profile) = batcher.batch_profiled(items, device);
                let profile = load_profile(
                    std::mem::take(&mut shard_read_pending),
                    std::mem::take(&mut row_scan_pending),
                    build_profile,
                );
                let data_time = data_start.elapsed();
                loader_profile.record(data_time, batch_size, profile);
                if consume(batch, batch_size, data_time)? == BatchControl::Stop {
                    break 'shards;
                }
                batches += 1;
                if max_batches.is_some_and(|limit| batches >= limit) {
                    break 'shards;
                }
                data_start = Instant::now();
                scan_start = Instant::now();
            }
        }
        row_scan_pending += scan_start.elapsed();
    }

    if !pending.is_empty() && max_batches.is_none_or(|limit| batches < limit) {
        let rows = pending.len();
        let (batch, build_profile) = batcher.batch_profiled(pending, device);
        let profile = load_profile(shard_read_pending, row_scan_pending, build_profile);
        let data_time = data_start.elapsed();
        loader_profile.record(data_time, rows, profile);
        let _ = consume(batch, rows, data_time)?;
    }

    Ok(())
}

/// Folds shard-read / row-scan timing and the batcher's build profile into a
/// single [`BatchLoadProfile`].
fn load_profile(
    shard_read: Duration,
    row_scan: Duration,
    build_profile: MoleculeBatchBuildProfile,
) -> BatchLoadProfile {
    BatchLoadProfile {
        shard_read,
        row_scan,
        sparse_allocation: build_profile.sparse_allocation,
        sparse_fill: build_profile.sparse_fill,
        tensor_build: build_profile.tensor_build,
        producer_total: shard_read + row_scan + build_profile.total(),
    }
}

impl CachedBatchPlanDataset {
    /// Plans the shard slicing needed to cover the requested split, honoring
    /// the optional `max_batches` cap.
    fn new(
        shards: &[CachedShardInfo],
        split: DataSplit,
        batch_size: usize,
        max_batches: Option<usize>,
        validation_per_mille: u16,
    ) -> AppResult<Self> {
        let rows_per_plan = source_rows_per_batch(split, batch_size, validation_per_mille)?;
        let mut plans = Vec::new();
        for shard in shards {
            let mut start_row = 0_usize;
            while start_row < shard.row_count {
                let end_row = start_row.saturating_add(rows_per_plan).min(shard.row_count);
                plans.push(CachedBatchPlan {
                    shard: shard.clone(),
                    start_row,
                    end_row,
                    split,
                });
                if max_batches.is_some_and(|limit| plans.len() >= limit) {
                    return Ok(Self { plans });
                }
                start_row = end_row;
            }
        }
        Ok(Self { plans })
    }
}

/// Worker entry point: claims the next plan via the shared atomic cursor,
/// builds a host batch, and pushes it onto the channel.
fn produce_host_loader_batches(
    batcher: MoleculeAutoencoderBatcher,
    plans: Arc<Vec<CachedBatchPlan>>,
    next_plan: &AtomicUsize,
    sender: mpsc::SyncSender<HostLoaderMessage>,
) {
    loop {
        let index = next_plan.fetch_add(1, Ordering::Relaxed);
        let Some(plan) = plans.get(index).cloned() else {
            return;
        };
        match build_cached_loader_batch(batcher, plan) {
            Ok(Some(batch)) => {
                if sender
                    .send(HostLoaderMessage::Batch(Box::new(batch)))
                    .is_err()
                {
                    return;
                }
            }
            Ok(None) => {}
            Err(message) => {
                let _ = sender.send(HostLoaderMessage::Error(message));
                return;
            }
        }
    }
}

/// Uploader thread: pulls host batches off the producer channel, tensorizes
/// them onto the backend device, and forwards them to the consumer.
fn upload_host_loader_batches<B>(
    receiver: Receiver<HostLoaderMessage>,
    batcher: MoleculeAutoencoderBatcher,
    device: B::Device,
    sender: mpsc::SyncSender<DeviceLoaderMessage<B>>,
) where
    B: Backend,
    MoleculeAutoencoderBatch<B>: Send,
{
    loop {
        match receiver.recv() {
            Ok(HostLoaderMessage::Batch(loader_batch)) => {
                let loader_batch = *loader_batch;
                let (batch, tensor_profile) =
                    batcher.batch_host_profiled(loader_batch.host, &device);
                let mut profile = loader_batch.profile;
                profile.tensor_build += tensor_profile.tensor_build;
                profile.producer_total += tensor_profile.tensor_build;
                if sender
                    .send(DeviceLoaderMessage::Batch(DeviceLoaderBatch {
                        batch,
                        rows: loader_batch.rows,
                        profile,
                    }))
                    .is_err()
                {
                    return;
                }
            }
            Ok(HostLoaderMessage::Error(message)) => {
                let _ = sender.send(DeviceLoaderMessage::Error(message));
                return;
            }
            Err(_) => return,
        }
    }
}

/// Consumer half of the device-prefetched pipeline: drains
/// [`DeviceLoaderMessage`]s, surfaces loader-profile stats, and invokes the
/// user-supplied callback.
fn consume_device_loader_batches<B, F>(
    receiver: Receiver<DeviceLoaderMessage<B>>,
    phase: &'static str,
    loader_profile_every: usize,
    consume: &mut F,
) -> AppResult<()>
where
    B: Backend,
    F: FnMut(MoleculeAutoencoderBatch<B>, usize, Duration) -> AppResult<BatchControl>,
{
    let mut loader_profile = LoaderProfileReporter::new(phase, loader_profile_every);
    loop {
        let wait_start = Instant::now();
        match receiver.recv() {
            Ok(DeviceLoaderMessage::Batch(loader_batch)) => {
                let data_wait = wait_start.elapsed();
                loader_profile.record(data_wait, loader_batch.rows, loader_batch.profile);
                if consume(loader_batch.batch, loader_batch.rows, data_wait)? == BatchControl::Stop
                {
                    drop(receiver);
                    return Ok(());
                }
            }
            Ok(DeviceLoaderMessage::Error(message)) => {
                drop(receiver);
                return Err(invalid_input(message));
            }
            Err(_) => return Ok(()),
        }
    }
}

/// Consumer half of the no-prefetch pipeline: drains host batches, performs
/// inline tensor upload, and invokes the user-supplied callback.
fn consume_host_loader_batches<B, F>(
    receiver: Receiver<HostLoaderMessage>,
    batcher: MoleculeAutoencoderBatcher,
    device: &B::Device,
    phase: &'static str,
    loader_profile_every: usize,
    consume: &mut F,
) -> AppResult<()>
where
    B: Backend,
    F: FnMut(MoleculeAutoencoderBatch<B>, usize, Duration) -> AppResult<BatchControl>,
{
    let mut loader_profile = LoaderProfileReporter::new(phase, loader_profile_every);
    loop {
        let wait_start = Instant::now();
        match receiver.recv() {
            Ok(HostLoaderMessage::Batch(loader_batch)) => {
                let loader_batch = *loader_batch;
                let data_wait = wait_start.elapsed();
                let upload_start = Instant::now();
                let (batch, tensor_profile) =
                    batcher.batch_host_profiled(loader_batch.host, device);
                let upload_time = upload_start.elapsed();
                let mut profile = loader_batch.profile;
                profile.tensor_build += tensor_profile.tensor_build;
                profile.producer_total += tensor_profile.tensor_build;
                let data_time = data_wait + upload_time;
                loader_profile.record(data_time, loader_batch.rows, profile);
                if consume(batch, loader_batch.rows, data_time)? == BatchControl::Stop {
                    drop(receiver);
                    return Ok(());
                }
            }
            Ok(HostLoaderMessage::Error(message)) => {
                drop(receiver);
                return Err(invalid_input(message));
            }
            Err(_) => return Ok(()),
        }
    }
}

/// Reads the planned shard range from disk and builds the host-side batch,
/// returning `Ok(None)` when no rows match the requested split.
fn build_cached_loader_batch(
    batcher: MoleculeAutoencoderBatcher,
    plan: CachedBatchPlan,
) -> Result<Option<CachedLoaderBatch>, String> {
    let read_start = Instant::now();
    let shard =
        SparseMoleculeShard::read_range_from_path(&plan.shard.path, plan.start_row, plan.end_row)
            .map_err(|source| source.to_string())?;
    let shard_read = read_start.elapsed();
    let path: &std::path::Path = &plan.shard.path;
    validate_shard_shape(&shard, batcher, path).map_err(|source| source.to_string())?;

    let mut pending = Vec::with_capacity(shard.len());
    let row_scan_start = Instant::now();
    for row_index in 0..shard.len() {
        let row = shard
            .row(row_index)
            .ok_or_else(|| "shard row disappeared during dataloader batching".to_string())?;
        if row.split() == plan.split {
            pending.push(sample_from_row(row));
        }
    }
    let row_scan = row_scan_start.elapsed();

    if pending.is_empty() {
        return Ok(None);
    }

    let rows = pending.len();
    let (host, build_profile) = batcher.host_batch_profiled(pending);
    Ok(Some(CachedLoaderBatch {
        host,
        rows,
        profile: load_profile(shard_read, row_scan, build_profile),
    }))
}

/// Returns the number of source rows that must be scanned to yield one
/// `batch_size` batch for the given split, accounting for the validation
/// split share.
fn source_rows_per_batch(
    split: DataSplit,
    batch_size: usize,
    validation_per_mille: u16,
) -> AppResult<usize> {
    let validation_per_mille = usize::from(validation_per_mille.min(1000));
    let selected_per_mille = match split {
        DataSplit::Train => 1000_usize.saturating_sub(validation_per_mille),
        DataSplit::Validation => validation_per_mille,
    };
    if selected_per_mille == 0 {
        return Err(invalid_input(format!(
            "no rows are assigned to the {} split",
            split_label(split)
        )));
    }
    Ok(batch_size
        .saturating_mul(1000)
        .div_ceil(selected_per_mille)
        .max(1))
}

/// Converts a borrowed shard row into the owned sample shape the batcher
/// consumes.
fn sample_from_row(row: MoleculeShardRow<'_>) -> MoleculeAutoencoderSample {
    MoleculeAutoencoderSample {
        cid: row.cid(),
        fingerprint_indices: row.fingerprint_indices().to_vec(),
        fingerprint_counts: row.fingerprint_counts().to_vec(),
        descriptor_targets: row.descriptor_targets().to_vec(),
    }
}
