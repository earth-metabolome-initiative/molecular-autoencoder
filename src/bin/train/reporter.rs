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
    renderer: Option<Box<dyn MetricsRenderer>>,
    bars: Option<IndicatifTrainingBars>,
    interrupter: Interrupter,
    metric_ids: ReporterMetricIds,
    train_total: usize,
    valid_total: usize,
    train_processed: usize,
    valid_processed: usize,
    train_epoch: Option<usize>,
    valid_epoch: Option<usize>,
}

impl TrainingReporter {
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

    pub fn is_active(&self) -> bool {
        self.renderer.is_some()
    }

    pub fn should_stop(&self) -> bool {
        self.interrupter.should_stop()
    }

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
                    self.metric_ids.tanimoto.clone(),
                    metrics.count_tanimoto,
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

    pub fn valid_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        iteration: usize,
        examples: usize,
        metrics: Option<BatchLossMetrics>,
        tanimoto: f32,
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
            }
            renderer.update_valid(metric_state(
                self.metric_ids.tanimoto.clone(),
                tanimoto,
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
                tanimoto,
            );
        }

        if self.should_stop() {
            BatchControl::Stop
        } else {
            BatchControl::Continue
        }
    }

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

struct IndicatifTrainingBars {
    bar: Option<ProgressBar>,
    phase: Option<ProgressPhase>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProgressPhase {
    kind: ProgressKind,
    epoch: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressKind {
    Train,
    Valid,
}

impl IndicatifTrainingBars {
    fn new() -> Self {
        Self {
            bar: None,
            phase: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
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
                "loss={:.4} recon={:.4} desc={:.4} tanrank={:.4} acc={:.3} tanimoto={:.4} data_ms={data_ms:.1} step_ms={step_ms:.1}",
                metrics.loss,
                metrics.reconstruction,
                metrics.descriptors,
                metrics.tanimoto_ranking,
                metrics.tanimoto_ranking_accuracy,
                metrics.count_tanimoto,
            )),
            None => bar.set_message(format!("data_ms={data_ms:.1} step_ms={step_ms:.1}")),
        }
    }

    fn valid_batch(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        processed: usize,
        total: usize,
        metrics: Option<BatchLossMetrics>,
        tanimoto: f32,
    ) {
        let bar = self.phase_bar(ProgressKind::Valid, epoch, epoch_total, total);
        bar.set_position(processed as u64);
        match metrics {
            Some(metrics) => bar.set_message(format!(
                "loss={:.4} recon={:.4} desc={:.4} tanrank={:.4} acc={:.3} tanimoto={tanimoto:.4}",
                metrics.loss,
                metrics.reconstruction,
                metrics.descriptors,
                metrics.tanimoto_ranking,
                metrics.tanimoto_ranking_accuracy,
            )),
            None => bar.set_message(format!("tanimoto={tanimoto:.4}")),
        }
    }

    fn finish(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
        self.phase = None;
    }

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

#[derive(Debug, Clone, Copy)]
enum ReporterMetric {
    Loss,
    Reconstruction,
    Descriptors,
    TanimotoRanking,
    TanimotoRankingAccuracy,
    Tanimoto,
}

impl ReporterMetric {
    const ALL: [Self; 6] = [
        Self::Loss,
        Self::Reconstruction,
        Self::Descriptors,
        Self::TanimotoRanking,
        Self::TanimotoRankingAccuracy,
        Self::Tanimoto,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Loss => "Loss",
            Self::Reconstruction => "Reconstruction Loss",
            Self::Descriptors => "Descriptor Loss",
            Self::TanimotoRanking => "Tanimoto Geometry Loss",
            Self::TanimotoRankingAccuracy => "Tanimoto Geometry Accuracy",
            Self::Tanimoto => "Count Tanimoto",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Loss => "weighted total loss",
            Self::Reconstruction => "counted ECFP reconstruction loss",
            Self::Descriptors => "descriptor side-task loss",
            Self::TanimotoRanking => "latent counted-Tanimoto softmax CE loss",
            Self::TanimotoRankingAccuracy => {
                "latent best-candidate accuracy for sampled Tanimoto sets"
            }
            Self::Tanimoto => "counted fingerprint Tanimoto reconstruction metric",
        }
    }

    const fn higher_is_better(self) -> bool {
        matches!(self, Self::Tanimoto | Self::TanimotoRankingAccuracy)
    }

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

struct ReporterMetricIds {
    loss: MetricId,
    reconstruction: MetricId,
    descriptors: MetricId,
    tanimoto_ranking: MetricId,
    tanimoto_ranking_accuracy: MetricId,
    tanimoto: MetricId,
}

impl ReporterMetricIds {
    fn new() -> Self {
        Self {
            loss: metric_id(ReporterMetric::Loss),
            reconstruction: metric_id(ReporterMetric::Reconstruction),
            descriptors: metric_id(ReporterMetric::Descriptors),
            tanimoto_ranking: metric_id(ReporterMetric::TanimotoRanking),
            tanimoto_ranking_accuracy: metric_id(ReporterMetric::TanimotoRankingAccuracy),
            tanimoto: metric_id(ReporterMetric::Tanimoto),
        }
    }

    fn id(&self, metric: ReporterMetric) -> MetricId {
        match metric {
            ReporterMetric::Loss => self.loss.clone(),
            ReporterMetric::Reconstruction => self.reconstruction.clone(),
            ReporterMetric::Descriptors => self.descriptors.clone(),
            ReporterMetric::TanimotoRanking => self.tanimoto_ranking.clone(),
            ReporterMetric::TanimotoRankingAccuracy => self.tanimoto_ranking_accuracy.clone(),
            ReporterMetric::Tanimoto => self.tanimoto.clone(),
        }
    }
}

fn metric_id(metric: ReporterMetric) -> MetricId {
    MetricId::new(std::sync::Arc::new(metric.name().to_string()))
}

fn metric_state(metric_id: MetricId, value: impl Into<f64>, count: usize) -> MetricState {
    let value = value.into();
    let numeric = NumericEntry::Aggregated {
        aggregated_value: value,
        count,
    };
    let serialized = SerializedEntry::new(format_float(value, 4), numeric.serialize());
    MetricState::Numeric(MetricEntry::new(metric_id, serialized), numeric)
}

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

fn progress_total(row_count: usize, batch_size: usize, max_batches: Option<usize>) -> usize {
    max_batches
        .map_or(row_count, |batches| batches.saturating_mul(batch_size))
        .max(1)
}

fn valid_row_estimate(row_count: usize, validation_per_mille: u16) -> usize {
    row_count.saturating_mul(usize::from(validation_per_mille.min(1000))) / 1000
}

fn train_row_estimate(row_count: usize, validation_per_mille: u16) -> usize {
    row_count.saturating_sub(valid_row_estimate(row_count, validation_per_mille))
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {wide_msg}",
    )
    .map_or_else(
        |_| ProgressStyle::default_bar(),
        |style| style.progress_chars("=> "),
    )
}
