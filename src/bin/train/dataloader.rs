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
    Continue,
    Stop,
}

/// Per-component timing for one dataloader batch.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchLoadProfile {
    pub shard_read: Duration,
    pub row_scan: Duration,
    pub sparse_allocation: Duration,
    pub sparse_fill: Duration,
    pub tensor_build: Duration,
    pub producer_total: Duration,
}

/// Iteration parameters shared by sync and multi-threaded dataloader paths.
pub struct BatchIterationContext<'a, B: Backend> {
    pub shards: &'a [CachedShardInfo],
    pub split: DataSplit,
    pub batcher: MoleculeAutoencoderBatcher,
    pub device: &'a B::Device,
    pub batch_size: usize,
    pub max_batches: Option<usize>,
    pub loader_workers: usize,
    pub device_prefetch_batches: usize,
    pub loader_profile_every: usize,
    pub validation_per_mille: u16,
}

/// One contiguous slice of a shard scheduled for a single batch build.
#[derive(Debug, Clone)]
struct CachedBatchPlan {
    shard: CachedShardInfo,
    start_row: usize,
    end_row: usize,
    split: DataSplit,
}

#[derive(Debug, Clone)]
struct CachedBatchPlanDataset {
    plans: Vec<CachedBatchPlan>,
}

#[derive(Debug, Clone)]
struct CachedLoaderBatch {
    host: MoleculeAutoencoderHostBatch,
    rows: usize,
    profile: BatchLoadProfile,
}

struct DeviceLoaderBatch<B: Backend> {
    batch: MoleculeAutoencoderBatch<B>,
    rows: usize,
    profile: BatchLoadProfile,
}

enum HostLoaderMessage {
    Batch(Box<CachedLoaderBatch>),
    Error(String),
}

enum DeviceLoaderMessage<B: Backend> {
    Batch(DeviceLoaderBatch<B>),
    Error(String),
}

/// Iterates batches of the requested split, dispatching to the multi-threaded
/// dataloader when `loader_workers > 0` and to the sync path otherwise.
pub fn for_each_batch<B, F>(
    context: BatchIterationContext<'_, B>,
    consume: F,
) -> AppResult<()>
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

fn for_each_batch_sync<B, F>(
    context: BatchIterationContext<'_, B>,
    mut consume: F,
) -> AppResult<()>
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

fn sample_from_row(row: MoleculeShardRow<'_>) -> MoleculeAutoencoderSample {
    MoleculeAutoencoderSample {
        cid: row.cid(),
        fingerprint_indices: row.fingerprint_indices().to_vec(),
        fingerprint_counts: row.fingerprint_counts().to_vec(),
        descriptor_targets: row.descriptor_targets().to_vec(),
    }
}
