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
    ) -> (
        IntTensor<Self>,
        IntTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
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

        // The two float outputs (top2_gap and candidate_scores) are
        // metric-side artifacts: gradients should not flow through them
        // because they're derived from the metric fingerprints rather
        // than the latent embeddings the autodiff graph cares about.
        // We wrap both as autodiff-tracked nodes with a no-op backward
        // so callers can still hand them to downstream tensor ops
        // without losing the graph; their gradients are simply dropped.
        let prep_a = NoGradientBackward
            .prepare::<C>([counts.node.clone(), mask.node.clone()])
            .compute_bound()
            .stateful();
        let prep_b = NoGradientBackward
            .prepare::<C>([counts.node.clone(), mask.node.clone()])
            .compute_bound()
            .stateful();

        let (candidate_index, best_candidate_position, top2_gap, candidate_scores) =
            B::counted_tanimoto_similarity_ranking_kernel(
                indices,
                counts.primitive,
                mask.primitive,
                config,
            );

        let top2_gap = match prep_a {
            OpsKind::Tracked(prep) => prep.finish((), top2_gap),
            OpsKind::UnTracked(prep) => prep.finish(top2_gap),
        };
        let candidate_scores = match prep_b {
            OpsKind::Tracked(prep) => prep.finish((), candidate_scores),
            OpsKind::UnTracked(prep) => prep.finish(candidate_scores),
        };

        (
            candidate_index,
            best_candidate_position,
            top2_gap,
            candidate_scores,
        )
    }
}
