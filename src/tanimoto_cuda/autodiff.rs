use super::api::{CountedTanimotoKernelBackend, CountedTanimotoRankingKernelConfig};
use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::ops::{FloatTensor, IntTensor};

#[cfg(feature = "train")]
impl<B, C> CountedTanimotoKernelBackend for Autodiff<B, C>
where
    B: CountedTanimotoKernelBackend,
    C: CheckpointStrategy,
{
    fn counted_tanimoto_similarity_ranking_kernel(
        indices: IntTensor<Self>,
        counts: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        config: CountedTanimotoRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct NoGradientBackward;

        impl<B> Backward<B, 2> for NoGradientBackward
        where
            B: CountedTanimotoKernelBackend,
        {
            type State = ();

            fn backward(
                self,
                _ops: Ops<Self::State, 2>,
                _grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
            }
        }

        match NoGradientBackward
            .prepare::<C>([counts.node.clone(), mask.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (candidate_index, best_candidate_position, top2_gap) =
                    B::counted_tanimoto_similarity_ranking_kernel(
                        indices.clone(),
                        counts.primitive.clone(),
                        mask.primitive.clone(),
                        config,
                    );

                (
                    candidate_index,
                    best_candidate_position,
                    prep.finish((), top2_gap),
                )
            }
            OpsKind::UnTracked(prep) => {
                let (candidate_index, best_candidate_position, top2_gap) =
                    B::counted_tanimoto_similarity_ranking_kernel(
                        indices,
                        counts.primitive,
                        mask.primitive,
                        config,
                    );

                (
                    candidate_index,
                    best_candidate_position,
                    prep.finish(top2_gap),
                )
            }
        }
    }
}
