//! One-off analyzer for per-row active-bin counts (`nnz`) in cached
//! `*.maeshard` files. Built to understand the long-tail distribution
//! that pads the sparse-input matmul intermediate to `~1500` active
//! bins and triggered the 13 GB OOM on a 24 GB GPU.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --bin analyze_nnz --no-default-features --features std -- \
//!     shards/pubchem/manifest.json
//! ```
//!
//! Streams shards one at a time (memory peak ~= one shard, a few GB).
//! Output: per-row encoder-input and decoder-target nnz histograms,
//! top-20 outlier CIDs, and a "what cap would exclude X%" table.

use std::{collections::BTreeMap, env, path::Path, process::ExitCode};

use molecular_autoencoder::{ShardManifest, SparseMoleculeShard};

/// Number of outlier rows (by nnz) to print per category.
const TOP_N: usize = 20;

/// Per-category running stats: counts, min/max/mean, top-N outliers,
/// and the raw samples needed for percentile + cap-exclusion tables.
#[derive(Default)]
struct NnzStats {
    /// Number of rows observed.
    rows: u64,
    /// Sum of `nnz` across all observed rows (for the mean).
    sum: u64,
    /// Smallest `nnz` seen so far.
    min: u32,
    /// Largest `nnz` seen so far.
    max: u32,
    /// Top-N (nnz, cid) pairs in descending order by nnz.
    top: Vec<(u32, u64)>,
    /// Per-row `nnz` samples (sorted in [`Self::report`] for percentiles).
    samples: Vec<u32>,
}

impl NnzStats {
    /// Empty stats with the `min` seed set to `u32::MAX` so the first
    /// observation always updates it.
    fn new() -> Self {
        Self {
            min: u32::MAX,
            ..Self::default()
        }
    }

    /// Records one row with the given `nnz` and its `cid` (so we can
    /// identify outlier molecules in the top-N table).
    fn observe(&mut self, nnz: u32, cid: u64) {
        self.rows += 1;
        self.sum += u64::from(nnz);
        self.min = self.min.min(nnz);
        self.max = self.max.max(nnz);
        self.samples.push(nnz);

        if self.top.len() < TOP_N {
            self.top.push((nnz, cid));
            self.top
                .sort_unstable_by_key(|&(n, _)| std::cmp::Reverse(n));
        } else if nnz > self.top[TOP_N - 1].0 {
            self.top[TOP_N - 1] = (nnz, cid);
            self.top
                .sort_unstable_by_key(|&(n, _)| std::cmp::Reverse(n));
        }
    }

    /// Arithmetic mean of `nnz`, or `0.0` for an empty category.
    fn mean(&self) -> f64 {
        if self.rows == 0 {
            0.0
        } else {
            self.sum as f64 / self.rows as f64
        }
    }

    /// Returns the requested percentile of `sorted` (which must already
    /// be in ascending order). Uses nearest-rank for simplicity.
    fn percentile(sorted: &[u32], p: f64) -> u32 {
        if sorted.is_empty() {
            return 0;
        }
        let rank_f = (p / 100.0 * (sorted.len() as f64 - 1.0)).round();
        #[allow(
            clippy::cast_sign_loss,
            reason = "p in [0,100] and len > 0 implies rank_f >= 0"
        )]
        let rank = rank_f as usize;
        sorted[rank.min(sorted.len() - 1)]
    }

    /// Returns the fraction of rows whose `nnz` strictly exceeds `cap`.
    fn fraction_above(sorted: &[u32], cap: u32) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let above = sorted.iter().filter(|&&nnz| nnz > cap).count();
        above as f64 / sorted.len() as f64
    }

    /// Sorts the samples and prints the full per-category report
    /// (summary line, percentiles, histogram, top-N, cap table).
    fn report(&mut self, label: &str) {
        if self.rows == 0 {
            println!("\n--- {label} ---\nno rows");
            return;
        }
        self.samples.sort_unstable();
        let samples = self.samples.as_slice();
        println!("\n--- {label} ---");
        println!(
            "rows={} min={} max={} mean={:.2}",
            self.rows,
            self.min,
            self.max,
            self.mean()
        );
        println!(
            "percentiles: p50={} p90={} p95={} p99={} p99.9={} p99.99={}",
            Self::percentile(samples, 50.0),
            Self::percentile(samples, 90.0),
            Self::percentile(samples, 95.0),
            Self::percentile(samples, 99.0),
            Self::percentile(samples, 99.9),
            Self::percentile(samples, 99.99),
        );

        // Log-bucketed histogram so the long tail is visible.
        let mut buckets: BTreeMap<u32, u64> = BTreeMap::new();
        let edges = [
            0_u32,
            10,
            20,
            30,
            50,
            75,
            100,
            150,
            200,
            300,
            500,
            750,
            1000,
            1500,
            2000,
            3000,
            5000,
            u32::MAX,
        ];
        for &nnz in samples {
            let bucket = edges.iter().rposition(|&e| nnz >= e).unwrap_or(0);
            *buckets.entry(edges[bucket]).or_default() += 1;
        }
        println!("histogram (right-open buckets):");
        for window in edges.windows(2) {
            let lo = window[0];
            let hi = window[1];
            let count = buckets.get(&lo).copied().unwrap_or(0);
            let pct = 100.0 * count as f64 / self.rows as f64;
            let label_hi = if hi == u32::MAX {
                "inf".to_string()
            } else {
                hi.to_string()
            };
            println!("  [{lo:>5},{label_hi:>5}): {count:>12} ({pct:>5.2}%)");
        }

        println!("top {TOP_N} CIDs by nnz:");
        for &(nnz, cid) in &self.top {
            println!("  cid={cid:>12} nnz={nnz}");
        }

        println!("cap exclusion table (cap -> fraction of rows above cap):");
        for &cap in &[64_u32, 96, 128, 192, 256, 384, 512, 768, 1024] {
            let pct = 100.0 * Self::fraction_above(samples, cap);
            println!("  cap={cap:>4}: {pct:>7.4}% rows exceed");
        }
    }
}

/// Streams every shard listed by the manifest, accumulates per-category
/// nnz stats, and emits four reports (encoder+target × positive+negative).
fn run(manifest_path: &Path) -> std::io::Result<()> {
    let manifest = ShardManifest::read_from_path(manifest_path).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("read manifest: {e}"),
        )
    })?;
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));

    println!(
        "manifest={} shards={} declared_rows={}",
        manifest_path.display(),
        manifest.shards().len(),
        manifest.row_count()
    );

    let mut encoder_pos = NnzStats::new();
    let mut encoder_neg = NnzStats::new();
    let mut target_pos = NnzStats::new();
    let mut target_neg = NnzStats::new();

    for (shard_idx, entry) in manifest.shards().iter().enumerate() {
        let shard_path = manifest_dir.join(entry.path());
        eprintln!(
            "[{:>2}/{:>2}] loading {} ({} rows declared)",
            shard_idx + 1,
            manifest.shards().len(),
            shard_path.display(),
            entry.row_count()
        );
        let shard = SparseMoleculeShard::read_from_path(&shard_path).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("read shard: {e}"))
        })?;
        for row_idx in 0..shard.len() {
            let row = shard.row(row_idx).expect("row in range");
            let cid = row.cid();
            let encoder_nnz = row.fingerprint_indices().len() as u32;
            let target_nnz = row.target_fingerprint_indices().len() as u32;
            if row.is_negative() {
                encoder_neg.observe(encoder_nnz, cid);
                target_neg.observe(target_nnz, cid);
            } else {
                encoder_pos.observe(encoder_nnz, cid);
                target_pos.observe(target_nnz, cid);
            }
        }
        // Explicit drop frees ~2-3 GB before the next shard loads.
        drop(shard);
    }

    encoder_pos.report("encoder input, POSITIVE rows");
    encoder_neg.report("encoder input, NEGATIVE rows");
    target_pos.report("decoder target, POSITIVE rows");
    target_neg.report("decoder target, NEGATIVE rows");

    Ok(())
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(manifest) = args.next() else {
        eprintln!("usage: analyze_nnz <path/to/manifest.json>");
        return ExitCode::from(2);
    };
    match run(Path::new(&manifest)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
