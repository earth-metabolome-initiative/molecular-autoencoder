//! Burn TUI / indicatif training progress reporter.
//!
//! Picks Burn's TUI renderer when stdout is a terminal, otherwise falls back
//! to indicatif progress bars writing to stderr.

use std::{io::IsTerminal, time::Duration};

use burn::{
    data::dataloader::Progress,
    train::{
        Interrupter,
        metric::{
            MetricDefinition, MetricEntry, MetricId, NumericAttributes, NumericEntry,
            SerializedEntry, format_float,
        },
        renderer::{
            MetricState, MetricsRenderer, ProgressType, TrainingProgress,
            tui::TuiMetricsRendererWrapper,
        },
    },
};
use indicatif::{ProgressBar, ProgressStyle};

use crate::{AppResult, dataloader::BatchControl, metrics::BatchLossMetrics};

/// Bundles the Burn TUI renderer with an indicatif fallback so the same call
/// sites work in both interactive and CI/log environments.
pub struct TrainingReporter {
    /// Burn TUI renderer when stdout is a terminal.
    renderer: Option<Box<dyn MetricsRenderer>>,
    /// Indicatif progress bars used when stdout is not a terminal.
    bars: Option<IndicatifTrainingBars>,
    /// Burn-side cancellation handle propagated to renderer + run loop.
    interrupter: Interrupter,
    /// Pre-allocated metric IDs for each named gauge.
    metric_ids: ReporterMetricIds,
    /// Total training rows expected this run.
    train_total: usize,
    /// Total validation rows expected this run.
    valid_total: usize,
    /// Cumulative training rows processed so far (within the active
    /// epoch).
    train_processed: usize,
    /// Cumulative validation rows processed within the active epoch.
    valid_processed: usize,
    /// Currently active training epoch, used to detect epoch
    /// boundaries.
    train_epoch: Option<usize>,
    /// Currently active validation epoch, used to detect epoch
    /// boundaries.
    valid_epoch: Option<usize>,
}

impl TrainingReporter {
    /// Builds a reporter sized for the upcoming run.
    pub fn new(
        row_count: usize,
        validation_per_mille: u16,
        batch_size: usize,
        max_train_batches: Option<usize>,
        max_valid_batches: Option<usize>,
        checkpoint: Option<usize>,
    ) -> Self {
        let interrupter = Interrupter::new();
        let use_tui = std::io::stdout().is_terminal();
        let mut renderer = use_tui.then(|| {
            Box::new(TuiMetricsRendererWrapper::new(
                interrupter.clone(),
                checkpoint,
            )) as Box<dyn MetricsRenderer>
        });
        let bars = (!use_tui).then(IndicatifTrainingBars::new);
        let metric_ids = ReporterMetricIds::new();

        if let Some(renderer) = renderer.as_mut() {
            for metric in ReporterMetric::ALL {
                renderer.register_metric(metric.definition(metric_ids.id(metric)));
            }
        }

        Self {
            renderer,
            bars,
            interrupter,
            metric_ids,
            train_total: progress_total(
                train_row_estimate(row_count, validation_per_mille),
                batch_size,
                max_train_batches,
            ),
            valid_total: progress_total(
                valid_row_estimate(row_count, validation_per_mille),
                batch_size,
                max_valid_batches,
            ),
            train_processed: 0,
            valid_processed: 0,
            train_epoch: None,
            valid_epoch: None,
        }
    }

    /// Returns whether the Burn TUI renderer is the active surface.
    pub fn is_active(&self) -> bool {
        self.renderer.is_some()
    }

    /// Returns whether the user has requested cancellation (Ctrl-C
    /// caught by the renderer's interrupter).
    pub fn should_stop(&self) -> bool {
        self.interrupter.should_stop()
    }

    /// Records one training batch on the renderer and indicatif
    /// fallback. Returns whether the loop should continue or stop.
    #[allow(clippy::too_many_arguments)]
    pub fn train_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        iteration: usize,
        examples: usize,
        metrics: Option<BatchLossMetrics>,
        data_time: Duration,
        step_time: Duration,
    ) -> BatchControl {
        if self.train_epoch != Some(epoch) {
            self.train_epoch = Some(epoch);
            self.train_processed = 0;
        }
        self.train_processed = self
            .train_processed
            .saturating_add(examples)
            .min(self.train_total);

        if let Some(renderer) = self.renderer.as_mut() {
            if let Some(metrics) = metrics {
                renderer.update_train(metric_state(
                    self.metric_ids.loss.clone(),
                    metrics.loss,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.reconstruction.clone(),
                    metrics.reconstruction,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.reconstruction_bce.clone(),
                    metrics.reconstruction_bce,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.descriptors.clone(),
                    metrics.descriptors,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.tanimoto_ranking.clone(),
                    metrics.tanimoto_ranking,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.tanimoto_ranking_accuracy.clone(),
                    metrics.tanimoto_ranking_accuracy,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.tanimoto_ranking_mrr.clone(),
                    metrics.tanimoto_ranking_mrr,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.tanimoto_ranking_pearson.clone(),
                    metrics.tanimoto_ranking_pearson,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.tanimoto_ranking_spearman.clone(),
                    metrics.tanimoto_ranking_spearman,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.count_tanimoto.clone(),
                    metrics.count_tanimoto,
                    examples,
                ));
                renderer.update_train(metric_state(
                    self.metric_ids.binary_tanimoto.clone(),
                    metrics.binary_tanimoto,
                    examples,
                ));
            }
            renderer.render_train(
                training_progress(
                    self.train_processed,
                    self.train_total,
                    epoch,
                    epoch_total,
                    iteration,
                ),
                progress_indicators(
                    self.train_processed,
                    self.train_total,
                    epoch,
                    epoch_total,
                    iteration,
                ),
            );
        }
        if let Some(bars) = self.bars.as_mut() {
            bars.train_batch(
                epoch,
                epoch_total,
                self.train_processed,
                self.train_total,
                metrics,
                data_time,
                step_time,
            );
        }

        if self.should_stop() {
            BatchControl::Stop
        } else {
            BatchControl::Continue
        }
    }

    /// Records one validation batch on the renderer and indicatif
    /// fallback. Returns whether the loop should continue or stop.
    #[allow(clippy::too_many_arguments)]
    pub fn valid_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        iteration: usize,
        examples: usize,
        metrics: Option<BatchLossMetrics>,
        count_tanimoto: f32,
        binary_tanimoto: f32,
    ) -> BatchControl {
        if self.valid_epoch != Some(epoch) {
            self.valid_epoch = Some(epoch);
            self.valid_processed = 0;
        }
        self.valid_processed = self
            .valid_processed
            .saturating_add(examples)
            .min(self.valid_total);

        if let Some(renderer) = self.renderer.as_mut() {
            if let Some(metrics) = metrics {
                renderer.update_valid(metric_state(
                    self.metric_ids.loss.clone(),
                    metrics.loss,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.reconstruction.clone(),
                    metrics.reconstruction,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.reconstruction_bce.clone(),
                    metrics.reconstruction_bce,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.descriptors.clone(),
                    metrics.descriptors,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.tanimoto_ranking.clone(),
                    metrics.tanimoto_ranking,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.tanimoto_ranking_accuracy.clone(),
                    metrics.tanimoto_ranking_accuracy,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.tanimoto_ranking_mrr.clone(),
                    metrics.tanimoto_ranking_mrr,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.tanimoto_ranking_pearson.clone(),
                    metrics.tanimoto_ranking_pearson,
                    examples,
                ));
                renderer.update_valid(metric_state(
                    self.metric_ids.tanimoto_ranking_spearman.clone(),
                    metrics.tanimoto_ranking_spearman,
                    examples,
                ));
            }
            renderer.update_valid(metric_state(
                self.metric_ids.count_tanimoto.clone(),
                count_tanimoto,
                examples,
            ));
            renderer.update_valid(metric_state(
                self.metric_ids.binary_tanimoto.clone(),
                binary_tanimoto,
                examples,
            ));
            renderer.render_valid(
                training_progress(
                    self.valid_processed,
                    self.valid_total,
                    epoch,
                    epoch_total,
                    iteration,
                ),
                progress_indicators(
                    self.valid_processed,
                    self.valid_total,
                    epoch,
                    epoch_total,
                    iteration,
                ),
            );
        }
        if let Some(bars) = self.bars.as_mut() {
            bars.valid_batch(
                epoch,
                epoch_total,
                self.valid_processed,
                self.valid_total,
                metrics,
                count_tanimoto,
                binary_tanimoto,
            );
        }

        if self.should_stop() {
            BatchControl::Stop
        } else {
            BatchControl::Continue
        }
    }

    /// Tells the renderer and indicatif bars that the run is over so
    /// they can clear their on-screen state.
    pub fn finish(&mut self) -> AppResult<()> {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.on_train_end(None)?;
        }
        if let Some(bars) = self.bars.as_mut() {
            bars.finish();
        }
        Ok(())
    }
}

/// Indicatif-based renderer fallback used when stdout is not a TTY.
struct IndicatifTrainingBars {
    /// Active progress bar; recreated whenever the
    /// (phase, epoch) pair changes.
    bar: Option<ProgressBar>,
    /// Phase tag associated with `bar`; `None` when no bar is active.
    phase: Option<ProgressPhase>,
}

/// Lightweight tag identifying which (phase, epoch) the current bar
/// belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProgressPhase {
    /// Train or validation phase tag.
    kind: ProgressKind,
    /// Epoch index (1-based).
    epoch: usize,
}

/// Phase identifier for a single indicatif progress bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressKind {
    /// Training phase.
    Train,
    /// Validation phase.
    Valid,
}

impl IndicatifTrainingBars {
    /// Constructs an empty progress-bar group; the first batch call
    /// will allocate a fresh bar.
    fn new() -> Self {
        Self {
            bar: None,
            phase: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Renders one training batch on the indicatif bar.
    fn train_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        processed: usize,
        total: usize,
        metrics: Option<BatchLossMetrics>,
        data_time: Duration,
        step_time: Duration,
    ) {
        let bar = self.phase_bar(ProgressKind::Train, epoch, epoch_total, total);
        bar.set_position(processed as u64);
        let data_ms = data_time.as_secs_f64() * 1000.0;
        let step_ms = step_time.as_secs_f64() * 1000.0;
        match metrics {
            Some(metrics) => bar.set_message(format!(
                "loss={:.4} recon={:.4} bce={:.4} desc={:.4} tanrank={:.4} acc={:.3} mrr={:.3} pearson={:.3} spearman={:.3} count_tan={:.4} bin_tan={:.4} data_ms={data_ms:.1} step_ms={step_ms:.1}",
                metrics.loss,
                metrics.reconstruction,
                metrics.reconstruction_bce,
                metrics.descriptors,
                metrics.tanimoto_ranking,
                metrics.tanimoto_ranking_accuracy,
                metrics.tanimoto_ranking_mrr,
                metrics.tanimoto_ranking_pearson,
                metrics.tanimoto_ranking_spearman,
                metrics.count_tanimoto,
                metrics.binary_tanimoto,
            )),
            None => bar.set_message(format!("data_ms={data_ms:.1} step_ms={step_ms:.1}")),
        }
    }

    /// Renders one validation batch on the indicatif bar.
    #[allow(clippy::too_many_arguments)]
    fn valid_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        processed: usize,
        total: usize,
        metrics: Option<BatchLossMetrics>,
        count_tanimoto: f32,
        binary_tanimoto: f32,
    ) {
        let bar = self.phase_bar(ProgressKind::Valid, epoch, epoch_total, total);
        bar.set_position(processed as u64);
        match metrics {
            Some(metrics) => bar.set_message(format!(
                "loss={:.4} recon={:.4} bce={:.4} desc={:.4} tanrank={:.4} acc={:.3} mrr={:.3} pearson={:.3} spearman={:.3} count_tan={count_tanimoto:.4} bin_tan={binary_tanimoto:.4}",
                metrics.loss,
                metrics.reconstruction,
                metrics.reconstruction_bce,
                metrics.descriptors,
                metrics.tanimoto_ranking,
                metrics.tanimoto_ranking_accuracy,
                metrics.tanimoto_ranking_mrr,
                metrics.tanimoto_ranking_pearson,
                metrics.tanimoto_ranking_spearman,
            )),
            None => {
                bar.set_message(format!(
                    "count_tan={count_tanimoto:.4} bin_tan={binary_tanimoto:.4}"
                ));
            }
        }
    }

    /// Finalises whichever bar is currently active and clears the
    /// phase tracker.
    fn finish(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
        self.phase = None;
    }

    /// Returns the active progress bar, allocating a fresh one when
    /// the requested phase / epoch differs from the current one.
    fn phase_bar(
        &mut self,
        kind: ProgressKind,
        epoch: usize,
        epoch_total: usize,
        total: usize,
    ) -> &ProgressBar {
        let phase = ProgressPhase { kind, epoch };
        if self.phase != Some(phase) {
            self.finish();
            let bar = ProgressBar::new(total as u64);
            bar.set_style(bar_style());
            bar.set_prefix(match kind {
                ProgressKind::Train => format!("train {epoch}/{epoch_total}"),
                ProgressKind::Valid => format!("valid {epoch}/{epoch_total}"),
            });
            self.bar = Some(bar);
            self.phase = Some(phase);
        }

        let Some(bar) = self.bar.as_ref() else {
            panic!("phase bar is initialized before use");
        };
        bar
    }
}

/// Named gauges surfaced on the renderer / indicatif fallback.
#[derive(Debug, Clone, Copy)]
enum ReporterMetric {
    /// Total weighted loss.
    Loss,
    /// Counted ECFP reconstruction Huber loss.
    Reconstruction,
    /// Bit-presence BCE auxiliary loss.
    ReconstructionBce,
    /// Descriptor regression side-task loss.
    Descriptors,
    /// Latent Tanimoto geometry softmax CE loss.
    TanimotoRanking,
    /// Best-candidate accuracy for the geometry head.
    TanimotoRankingAccuracy,
    /// Mean reciprocal rank for the geometry head.
    TanimotoRankingMrr,
    /// Per-anchor Pearson correlation for the geometry head.
    TanimotoRankingPearson,
    /// Per-anchor Spearman rank correlation for the geometry head.
    TanimotoRankingSpearman,
    /// Counted-Tanimoto reconstruction quality metric.
    CountTanimoto,
    /// Binary-Tanimoto reconstruction quality metric.
    BinaryTanimoto,
}

impl ReporterMetric {
    /// All gauges in the canonical order the reporter renders them.
    const ALL: [Self; 11] = [
        Self::Loss,
        Self::Reconstruction,
        Self::ReconstructionBce,
        Self::Descriptors,
        Self::TanimotoRanking,
        Self::TanimotoRankingAccuracy,
        Self::TanimotoRankingMrr,
        Self::TanimotoRankingPearson,
        Self::TanimotoRankingSpearman,
        Self::CountTanimoto,
        Self::BinaryTanimoto,
    ];

    /// Short human-readable label for the gauge.
    const fn name(self) -> &'static str {
        match self {
            Self::Loss => "Loss",
            Self::Reconstruction => "Reconstruction Loss",
            Self::ReconstructionBce => "Reconstruction BCE",
            Self::Descriptors => "Descriptor Loss",
            Self::TanimotoRanking => "Tanimoto Geometry Loss",
            Self::TanimotoRankingAccuracy => "Tanimoto Geometry Accuracy",
            Self::TanimotoRankingMrr => "Tanimoto Ranking MRR",
            Self::TanimotoRankingPearson => "Tanimoto Ranking Pearson",
            Self::TanimotoRankingSpearman => "Tanimoto Ranking Spearman",
            Self::CountTanimoto => "Count Tanimoto",
            Self::BinaryTanimoto => "Binary Tanimoto",
        }
    }

    /// Long-form description shown in the TUI side panel.
    const fn description(self) -> &'static str {
        match self {
            Self::Loss => "weighted total loss",
            Self::Reconstruction => "counted ECFP reconstruction Huber loss",
            Self::ReconstructionBce => "bit-presence BCE-with-logits auxiliary loss",
            Self::Descriptors => "descriptor side-task loss",
            Self::TanimotoRanking => "latent counted-Tanimoto softmax CE loss",
            Self::TanimotoRankingAccuracy => {
                "latent best-candidate accuracy for sampled Tanimoto sets"
            }
            Self::TanimotoRankingMrr => {
                "mean reciprocal rank of the true best candidate in the latent-sorted order"
            }
            Self::TanimotoRankingPearson => {
                "per-anchor Pearson correlation between latent cosine logits and Tanimoto scores"
            }
            Self::TanimotoRankingSpearman => {
                "per-anchor Spearman rank correlation between latent cosine logits and Tanimoto scores"
            }
            Self::CountTanimoto => "counted fingerprint Tanimoto reconstruction metric",
            Self::BinaryTanimoto => "binary fingerprint Tanimoto reconstruction metric",
        }
    }

    /// Whether higher values indicate better quality (used by Burn's
    /// renderer to colour gauges).
    const fn higher_is_better(self) -> bool {
        matches!(
            self,
            Self::CountTanimoto
                | Self::BinaryTanimoto
                | Self::TanimotoRankingAccuracy
                | Self::TanimotoRankingMrr
                | Self::TanimotoRankingPearson
                | Self::TanimotoRankingSpearman
        )
    }

    /// Builds the Burn `MetricDefinition` for this gauge from its
    /// name, description, and polarity.
    fn definition(self, metric_id: MetricId) -> MetricDefinition {
        MetricDefinition {
            metric_id,
            name: self.name().to_string(),
            description: Some(self.description().to_string()),
            attributes: NumericAttributes {
                unit: None,
                higher_is_better: self.higher_is_better(),
            }
            .into(),
        }
    }
}

/// Pre-allocated [`MetricId`] handles for every gauge so the renderer
/// can update them without cloning the underlying `Arc<String>`.
struct ReporterMetricIds {
    /// Metric ID for the weighted total loss.
    loss: MetricId,
    /// Metric ID for the reconstruction Huber loss.
    reconstruction: MetricId,
    /// Metric ID for the bit-presence BCE loss.
    reconstruction_bce: MetricId,
    /// Metric ID for the descriptor regression loss.
    descriptors: MetricId,
    /// Metric ID for the Tanimoto geometry softmax CE loss.
    tanimoto_ranking: MetricId,
    /// Metric ID for the Tanimoto geometry accuracy gauge.
    tanimoto_ranking_accuracy: MetricId,
    /// Metric ID for the Tanimoto ranking MRR gauge.
    tanimoto_ranking_mrr: MetricId,
    /// Metric ID for the Tanimoto ranking Pearson gauge.
    tanimoto_ranking_pearson: MetricId,
    /// Metric ID for the Tanimoto ranking Spearman gauge.
    tanimoto_ranking_spearman: MetricId,
    /// Metric ID for the counted-Tanimoto reconstruction gauge.
    count_tanimoto: MetricId,
    /// Metric ID for the binary-Tanimoto reconstruction gauge.
    binary_tanimoto: MetricId,
}

impl ReporterMetricIds {
    /// Constructs a fresh ID for every gauge.
    fn new() -> Self {
        Self {
            loss: metric_id(ReporterMetric::Loss),
            reconstruction: metric_id(ReporterMetric::Reconstruction),
            reconstruction_bce: metric_id(ReporterMetric::ReconstructionBce),
            descriptors: metric_id(ReporterMetric::Descriptors),
            tanimoto_ranking: metric_id(ReporterMetric::TanimotoRanking),
            tanimoto_ranking_accuracy: metric_id(ReporterMetric::TanimotoRankingAccuracy),
            tanimoto_ranking_mrr: metric_id(ReporterMetric::TanimotoRankingMrr),
            tanimoto_ranking_pearson: metric_id(ReporterMetric::TanimotoRankingPearson),
            tanimoto_ranking_spearman: metric_id(ReporterMetric::TanimotoRankingSpearman),
            count_tanimoto: metric_id(ReporterMetric::CountTanimoto),
            binary_tanimoto: metric_id(ReporterMetric::BinaryTanimoto),
        }
    }

    /// Maps an enum variant to its pre-allocated metric ID.
    fn id(&self, metric: ReporterMetric) -> MetricId {
        match metric {
            ReporterMetric::Loss => self.loss.clone(),
            ReporterMetric::Reconstruction => self.reconstruction.clone(),
            ReporterMetric::ReconstructionBce => self.reconstruction_bce.clone(),
            ReporterMetric::Descriptors => self.descriptors.clone(),
            ReporterMetric::TanimotoRanking => self.tanimoto_ranking.clone(),
            ReporterMetric::TanimotoRankingAccuracy => self.tanimoto_ranking_accuracy.clone(),
            ReporterMetric::TanimotoRankingMrr => self.tanimoto_ranking_mrr.clone(),
            ReporterMetric::TanimotoRankingPearson => self.tanimoto_ranking_pearson.clone(),
            ReporterMetric::TanimotoRankingSpearman => self.tanimoto_ranking_spearman.clone(),
            ReporterMetric::CountTanimoto => self.count_tanimoto.clone(),
            ReporterMetric::BinaryTanimoto => self.binary_tanimoto.clone(),
        }
    }
}

/// Allocates a Burn [`MetricId`] keyed on the gauge's display name.
fn metric_id(metric: ReporterMetric) -> MetricId {
    MetricId::new(std::sync::Arc::new(metric.name().to_string()))
}

/// Wraps an `f32` / `f64` scalar plus its example count into the
/// Burn renderer's `MetricState` shape.
fn metric_state(metric_id: MetricId, value: impl Into<f64>, count: usize) -> MetricState {
    let value = value.into();
    let numeric = NumericEntry::Aggregated {
        aggregated_value: value,
        count,
    };
    let serialized = SerializedEntry::new(format_float(value, 4), numeric.serialize());
    MetricState::Numeric(MetricEntry::new(metric_id, serialized), numeric)
}

/// Builds the Burn renderer's `TrainingProgress` payload — local
/// batch progress, global epoch progress, and the current iteration.
fn training_progress(
    processed: usize,
    total: usize,
    epoch: usize,
    epoch_total: usize,
    iteration: usize,
) -> TrainingProgress {
    TrainingProgress {
        progress: Some(Progress {
            items_processed: processed,
            items_total: total,
        }),
        global_progress: Progress {
            items_processed: epoch,
            items_total: epoch_total,
        },
        iteration: Some(iteration),
    }
}

/// Builds the indicator tags (items / epoch / iteration) the TUI
/// renders alongside the main progress bar.
fn progress_indicators(
    processed: usize,
    total: usize,
    epoch: usize,
    epoch_total: usize,
    iteration: usize,
) -> Vec<ProgressType> {
    vec![
        ProgressType::Detailed {
            tag: "Items".to_string(),
            progress: Progress {
                items_processed: processed,
                items_total: total,
            },
        },
        ProgressType::Detailed {
            tag: "Epoch".to_string(),
            progress: Progress {
                items_processed: epoch,
                items_total: epoch_total,
            },
        },
        ProgressType::Value {
            tag: "Iteration".to_string(),
            value: iteration,
        },
    ]
}

/// Returns the bar `len` target: `max_batches * batch_size` when set, else
/// `row_count`, clamped to at least one so empty datasets still render.
fn progress_total(row_count: usize, batch_size: usize, max_batches: Option<usize>) -> usize {
    max_batches
        .map_or(row_count, |batches| batches.saturating_mul(batch_size))
        .max(1)
}

/// Estimated number of validation rows for a split-per-mille setting.
fn valid_row_estimate(row_count: usize, validation_per_mille: u16) -> usize {
    row_count.saturating_mul(usize::from(validation_per_mille.min(1000))) / 1000
}

/// Estimated number of training rows after subtracting the validation split.
fn train_row_estimate(row_count: usize, validation_per_mille: u16) -> usize {
    row_count.saturating_sub(valid_row_estimate(row_count, validation_per_mille))
}

/// Shared `indicatif` template used by every progress bar in the fallback
/// reporter.
fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {wide_msg}",
    )
    .map_or_else(
        |_| ProgressStyle::default_bar(),
        |style| style.progress_chars("=> "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_metrics() -> BatchLossMetrics {
        BatchLossMetrics {
            loss: 0.5,
            reconstruction: 0.3,
            reconstruction_bce: 0.1,
            descriptors: 0.05,
            tanimoto_ranking: 0.02,
            tanimoto_ranking_accuracy: 0.85,
            tanimoto_ranking_mrr: 0.6,
            tanimoto_ranking_pearson: 0.4,
            tanimoto_ranking_spearman: 0.45,
            count_tanimoto: 0.75,
            binary_tanimoto: 0.80,
        }
    }

    #[test]
    fn progress_total_clamps_to_one_when_empty() {
        assert_eq!(progress_total(0, 32, None), 1);
        assert_eq!(progress_total(0, 32, Some(0)), 1);
    }

    #[test]
    fn progress_total_uses_row_count_when_max_batches_is_unset() {
        assert_eq!(progress_total(1024, 32, None), 1024);
    }

    #[test]
    fn progress_total_uses_max_batches_times_batch_size_when_capped() {
        assert_eq!(progress_total(1024, 32, Some(4)), 128);
    }

    #[test]
    fn valid_and_train_row_estimates_partition_row_count() {
        let row_count = 1_000_000;
        let permille = 50; // 5 %
        let valid = valid_row_estimate(row_count, permille);
        let train = train_row_estimate(row_count, permille);
        assert_eq!(valid, 50_000);
        assert_eq!(valid + train, row_count);
    }

    #[test]
    fn valid_row_estimate_clamps_per_mille_to_thousand() {
        // Per-mille above 1000 is impossible but the helper saturates safely.
        assert_eq!(valid_row_estimate(1000, 2000), 1000);
    }

    #[test]
    fn reporter_metric_definitions_cover_every_variant() {
        for metric in ReporterMetric::ALL {
            assert!(!metric.name().is_empty());
            assert!(!metric.description().is_empty());
            let definition = metric.definition(metric_id(metric));
            assert_eq!(definition.name, metric.name());
        }
    }

    #[test]
    fn reporter_metric_higher_is_better_flags_only_accuracy_and_tanimoto() {
        assert!(!ReporterMetric::Loss.higher_is_better());
        assert!(!ReporterMetric::Reconstruction.higher_is_better());
        assert!(!ReporterMetric::ReconstructionBce.higher_is_better());
        assert!(!ReporterMetric::Descriptors.higher_is_better());
        assert!(!ReporterMetric::TanimotoRanking.higher_is_better());
        assert!(ReporterMetric::TanimotoRankingAccuracy.higher_is_better());
        assert!(ReporterMetric::CountTanimoto.higher_is_better());
        assert!(ReporterMetric::BinaryTanimoto.higher_is_better());
    }

    #[test]
    fn metric_state_wraps_value_and_count() {
        let id = metric_id(ReporterMetric::Loss);
        let state = metric_state(id, 0.42_f64, 8);
        match state {
            MetricState::Numeric(
                _,
                NumericEntry::Aggregated {
                    aggregated_value,
                    count,
                },
            ) => {
                assert!((aggregated_value - 0.42).abs() < 1.0e-9);
                assert_eq!(count, 8);
            }
            MetricState::Numeric(_, _) => panic!("expected an aggregated numeric entry"),
            MetricState::Generic(_) => panic!("expected a numeric entry"),
        }
    }

    #[test]
    fn training_progress_threads_processed_and_iteration() {
        let progress = training_progress(50, 200, 1, 10, 7);
        let detailed = progress.progress.expect("processed progress present");
        assert_eq!(detailed.items_processed, 50);
        assert_eq!(detailed.items_total, 200);
        assert_eq!(progress.global_progress.items_processed, 1);
        assert_eq!(progress.global_progress.items_total, 10);
        assert_eq!(progress.iteration, Some(7));
    }

    #[test]
    fn progress_indicators_emit_items_epoch_and_iteration_tags() {
        let indicators = progress_indicators(10, 100, 2, 5, 42);
        assert_eq!(indicators.len(), 3);
        // Tags are part of the public Burn API; assert by destructuring.
        let mut items = false;
        let mut epoch = false;
        let mut iteration = false;
        for indicator in indicators {
            match indicator {
                ProgressType::Detailed { tag, progress } if tag == "Items" => {
                    assert_eq!(progress.items_processed, 10);
                    assert_eq!(progress.items_total, 100);
                    items = true;
                }
                ProgressType::Detailed { tag, progress } if tag == "Epoch" => {
                    assert_eq!(progress.items_processed, 2);
                    assert_eq!(progress.items_total, 5);
                    epoch = true;
                }
                ProgressType::Value { tag, value } if tag == "Iteration" => {
                    assert_eq!(value, 42);
                    iteration = true;
                }
                _ => panic!("unexpected progress indicator variant"),
            }
        }
        assert!(items && epoch && iteration);
    }

    #[test]
    fn training_reporter_under_tests_falls_back_to_indicatif() {
        // `cargo test` redirects stdout, so the renderer must be inactive.
        let reporter = TrainingReporter::new(1024, 100, 32, Some(4), Some(2), None);
        assert!(!reporter.is_active());
        assert!(!reporter.should_stop());
        // Totals come from the helper functions above.
        assert_eq!(
            reporter.train_total,
            progress_total(1024 - 102, 32, Some(4))
        );
        assert_eq!(reporter.valid_total, progress_total(102, 32, Some(2)));
    }

    #[test]
    fn training_reporter_train_batch_advances_processed_counter() {
        let mut reporter = TrainingReporter::new(1000, 0, 32, Some(2), Some(1), None);
        let control = reporter.train_batch(
            1,
            5,
            1,
            32,
            Some(fixture_metrics()),
            Duration::from_millis(1),
            Duration::from_millis(2),
        );
        assert!(matches!(control, BatchControl::Continue));
        assert_eq!(reporter.train_epoch, Some(1));
        assert_eq!(reporter.train_processed, 32);
    }

    #[test]
    fn training_reporter_train_batch_resets_on_new_epoch() {
        let mut reporter = TrainingReporter::new(1000, 0, 32, Some(4), Some(1), None);
        reporter.train_batch(1, 5, 1, 32, None, Duration::ZERO, Duration::ZERO);
        reporter.train_batch(2, 5, 1, 16, None, Duration::ZERO, Duration::ZERO);
        assert_eq!(reporter.train_epoch, Some(2));
        assert_eq!(reporter.train_processed, 16);
    }

    #[test]
    fn indicatif_train_and_valid_bars_construct_and_finish_without_panic() {
        let mut bars = IndicatifTrainingBars::new();
        bars.train_batch(
            1,
            5,
            16,
            64,
            Some(fixture_metrics()),
            Duration::from_millis(1),
            Duration::from_millis(2),
        );
        // Repeating the same epoch reuses the bar.
        bars.train_batch(1, 5, 32, 64, None, Duration::ZERO, Duration::ZERO);
        // Switching kinds finishes the train bar and starts the valid one.
        bars.valid_batch(1, 5, 8, 32, Some(fixture_metrics()), 0.5, 0.6);
        bars.valid_batch(1, 5, 16, 32, None, 0.5, 0.6);
        bars.finish();
        // A second finish is a no-op.
        bars.finish();
    }
}
