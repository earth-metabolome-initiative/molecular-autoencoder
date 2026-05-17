//! Burn training helpers.

use burn::{backend::NdArray, prelude::*, tensor::Transaction, train::metric::ItemLazy};

use crate::model::MoleculeLossBreakdown;

/// Autoencoder output adapted for Burn training loops.
pub struct MoleculeAutoencoderTrainingOutput<B: Backend> {
    /// Weighted total loss.
    pub loss: Tensor<B, 1>,
    /// Reconstructed log-count fingerprint.
    pub reconstruction: Tensor<B, 2>,
    /// Weighted component losses.
    pub losses: MoleculeLossBreakdown<B>,
}

impl<B: Backend> MoleculeAutoencoderTrainingOutput<B> {
    /// Creates a training output from component losses.
    #[must_use]
    pub fn new(reconstruction: Tensor<B, 2>, losses: MoleculeLossBreakdown<B>) -> Self {
        let loss = losses.total();
        Self {
            loss,
            reconstruction,
            losses,
        }
    }
}

impl<B: Backend> ItemLazy for MoleculeAutoencoderTrainingOutput<B> {
    type ItemSync = MoleculeAutoencoderTrainingOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [
            loss,
            reconstruction,
            reconstruction_bce,
            descriptors,
            tanimoto_ranking,
            tanimoto_accuracy,
            tanimoto_pairs,
        ] = Transaction::default()
            .register(self.loss)
            .register(self.losses.reconstruction)
            .register(self.losses.reconstruction_bce)
            .register(self.losses.descriptors)
            .register(self.losses.tanimoto_ranking)
            .register(self.losses.tanimoto_ranking_accuracy)
            .register(self.losses.tanimoto_ranking_pairs)
            .execute()
            .try_into()
            .expect("correct number of tensor data items");

        let device = &Default::default();
        MoleculeAutoencoderTrainingOutput {
            loss: Tensor::from_data(loss, device),
            reconstruction: Tensor::zeros([1, 1], device),
            losses: MoleculeLossBreakdown {
                reconstruction: Tensor::from_data(reconstruction, device),
                reconstruction_bce: Tensor::from_data(reconstruction_bce, device),
                descriptors: Tensor::from_data(descriptors, device),
                tanimoto_ranking: Tensor::from_data(tanimoto_ranking, device),
                tanimoto_ranking_accuracy: Tensor::from_data(tanimoto_accuracy, device),
                tanimoto_ranking_pairs: Tensor::from_data(tanimoto_pairs, device),
            },
        }
    }
}

/// Convenience accessors for training outputs.
pub trait MoleculeTrainingMetricsExt<B: Backend> {
    /// Returns the weighted total loss tensor.
    fn total_loss(&self) -> Tensor<B, 1>;
}

impl<B: Backend> MoleculeTrainingMetricsExt<B> for MoleculeAutoencoderTrainingOutput<B> {
    fn total_loss(&self) -> Tensor<B, 1> {
        self.loss.clone()
    }
}

mod train_impl {
    use burn::{
        tensor::backend::AutodiffBackend,
        train::{InferenceStep, TrainOutput, TrainStep},
    };

    use crate::{
        batch::MoleculeAutoencoderBatch, model::MoleculeAutoencoder,
        training::MoleculeAutoencoderTrainingOutput,
    };

    impl<B: AutodiffBackend> TrainStep for MoleculeAutoencoder<B> {
        type Input = MoleculeAutoencoderBatch<B>;
        type Output = MoleculeAutoencoderTrainingOutput<B>;

        fn step(&self, batch: MoleculeAutoencoderBatch<B>) -> TrainOutput<Self::Output> {
            let crate::batch::MoleculeAutoencoderBatch {
                fingerprints,
                descriptor_targets,
                tanimoto_ranking,
                ..
            } = batch;
            let output = self.forward_with_decoder_latent_noise(&fingerprints, true);
            let reconstruction = output.reconstructed_log_counts.clone();
            let losses =
                self.loss_from_output(output, fingerprints, descriptor_targets, tanimoto_ranking);
            let item = MoleculeAutoencoderTrainingOutput::new(reconstruction, losses);

            TrainOutput::new(self, item.loss.clone().backward(), item)
        }
    }

    impl<B: burn::prelude::Backend> InferenceStep for MoleculeAutoencoder<B> {
        type Input = MoleculeAutoencoderBatch<B>;
        type Output = MoleculeAutoencoderTrainingOutput<B>;

        fn step(&self, batch: MoleculeAutoencoderBatch<B>) -> Self::Output {
            let crate::batch::MoleculeAutoencoderBatch {
                fingerprints,
                descriptor_targets,
                tanimoto_ranking,
                ..
            } = batch;
            let output = self.forward(&fingerprints);
            let reconstruction = output.reconstructed_log_counts.clone();
            let losses =
                self.loss_from_output(output, fingerprints, descriptor_targets, tanimoto_ranking);

            MoleculeAutoencoderTrainingOutput::new(reconstruction, losses)
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        data::dataloader::batcher::Batcher,
        tensor::TensorData,
        train::{InferenceStep, TrainStep, metric::ItemLazy},
    };

    use super::*;
    use crate::{
        batch::{MoleculeAutoencoderBatch, MoleculeAutoencoderBatcher, MoleculeAutoencoderSample},
        features::REGRESSION_TARGET_WIDTH,
        model::{MoleculeAutoencoderConfig, MoleculeLossBreakdown},
    };

    fn sample() -> MoleculeAutoencoderSample {
        MoleculeAutoencoderSample {
            cid: 1,
            fingerprint_indices: vec![1, 3],
            fingerprint_counts: vec![2, 1],
            descriptor_targets: vec![0.0; REGRESSION_TARGET_WIDTH],
        }
    }

    fn scalar<B: Backend>(value: f32, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_data(TensorData::new(vec![value], [1]), device)
    }

    #[test]
    fn training_output_new_and_total_loss_helper_use_weighted_sum() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let reconstruction = Tensor::<B, 2>::zeros([1, 4], &device);
        let losses = MoleculeLossBreakdown {
            reconstruction: scalar::<B>(1.0, &device),
            reconstruction_bce: scalar::<B>(0.3, &device),
            descriptors: scalar::<B>(0.2, &device),
            tanimoto_ranking: scalar::<B>(0.4, &device),
            tanimoto_ranking_accuracy: scalar::<B>(0.5, &device),
            tanimoto_ranking_pairs: scalar::<B>(2.0, &device),
        };

        let output = MoleculeAutoencoderTrainingOutput::new(reconstruction, losses);

        assert!((output.loss.clone().into_scalar() - 1.9).abs() < 1.0e-6);
        assert!((output.total_loss().into_scalar() - 1.9).abs() < 1.0e-6);
    }

    #[test]
    fn training_output_sync_preserves_loss_components() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let output = MoleculeAutoencoderTrainingOutput::new(
            Tensor::<B, 2>::zeros([1, 4], &device),
            MoleculeLossBreakdown {
                reconstruction: scalar::<B>(1.0, &device),
                reconstruction_bce: scalar::<B>(0.3, &device),
                descriptors: scalar::<B>(0.2, &device),
                tanimoto_ranking: scalar::<B>(0.4, &device),
                tanimoto_ranking_accuracy: scalar::<B>(0.5, &device),
                tanimoto_ranking_pairs: scalar::<B>(2.0, &device),
            },
        )
        .sync();

        assert!((output.loss.into_scalar() - 1.9).abs() < 1.0e-6);
        assert!((output.losses.reconstruction.into_scalar() - 1.0).abs() < 1.0e-6);
        assert!((output.losses.reconstruction_bce.into_scalar() - 0.3).abs() < 1.0e-6);
        assert!((output.losses.descriptors.into_scalar() - 0.2).abs() < 1.0e-6);
        assert!((output.losses.tanimoto_ranking.into_scalar() - 0.4).abs() < 1.0e-6);
        assert!((output.losses.tanimoto_ranking_accuracy.into_scalar() - 0.5).abs() < 1.0e-6);
        assert!((output.losses.tanimoto_ranking_pairs.into_scalar() - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn inference_step_returns_finite_loss_and_expected_shapes() {
        type B = burn::backend::NdArray<f32, i64>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(32, REGRESSION_TARGET_WIDTH);
        let batch: MoleculeAutoencoderBatch<B> = batcher.batch(vec![sample()], &device);
        let model = MoleculeAutoencoderConfig::symmetric(32, 8, vec![16]).init::<B>(&device);

        let output = InferenceStep::step(&model, batch);

        assert!(output.loss.clone().into_scalar().is_finite());
        assert_eq!(output.reconstruction.dims(), [1, 32]);
    }

    #[test]
    fn train_step_returns_finite_loss_for_autodiff_ndarray() {
        type B = burn::backend::Autodiff<burn::backend::NdArray<f32, i64>>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = MoleculeAutoencoderBatcher::new(32, REGRESSION_TARGET_WIDTH);
        let batch: MoleculeAutoencoderBatch<B> = batcher.batch(vec![sample()], &device);
        let model = MoleculeAutoencoderConfig::symmetric(32, 8, vec![16]).init::<B>(&device);

        let output = TrainStep::step(&model, batch);
        let loss_data = output.item.loss.to_data();
        let loss = loss_data.as_slice::<f32>().expect("f32 loss")[0];

        assert!(loss.is_finite());
        assert_eq!(output.item.reconstruction.dims(), [1, 32]);
    }
}
