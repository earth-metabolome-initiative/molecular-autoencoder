use super::api::SparseInputLinearKernelBackend;
use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorMetadata, TensorPrimitive};

/// Hand-written backward for the forward EmbeddingBag-style kernel.
///
/// The forward saves nothing (no `[batch, max_nnz, hidden]`
/// intermediate); the backward re-materialises the dense input via the
/// standard `scatter`-add and routes `grad_W` through the inner
/// backend's dense matmul. This wastes the would-be matmul of a
/// custom backward kernel but keeps the change reversible and avoids
/// the `atomic<f32>` cubecl-cpu gap. `grad_b` is just
/// `grad_out.sum_dim(0)` when bias is present.
#[derive(Debug)]
struct SparseInputLinearBackward<B>
where
    B: SparseInputLinearKernelBackend,
{
    /// Padded sparse indices, shape `[batch, max_nnz]`.
    indices: IntTensor<B>,
    /// Padded sparse values, shape `[batch, max_nnz]`.
    values: FloatTensor<B>,
    /// Padding mask (0.0 / 1.0), shape `[batch, max_nnz]`.
    mask: FloatTensor<B>,
    /// Dense vocab dimension of the weight matrix.
    vocab: usize,
    /// Whether the upstream call passed a bias (and therefore expects
    /// `grad_b` to be registered as the second parent's gradient).
    has_bias: bool,
}

impl<B> Backward<B, 2> for SparseInputLinearBackward<B>
where
    B: SparseInputLinearKernelBackend,
{
    type State = ();

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_out = grads.consume::<B>(&ops.node);
        let grad_out_t = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(grad_out.clone()));

        let weight_parent = ops.parents[0].as_ref();
        let bias_parent = if self.has_bias {
            ops.parents[1].as_ref()
        } else {
            None
        };

        if let Some(weight_parent) = weight_parent {
            let indices_t = Tensor::<B, 2, Int>::from_primitive(self.indices);
            let values_t = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(self.values));
            let mask_t = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(self.mask));
            let [batch, _max_nnz] = indices_t.dims();
            let device = grad_out_t.device();
            let dense = Tensor::<B, 2>::zeros([batch, self.vocab], &device).scatter(
                1,
                indices_t,
                values_t * mask_t,
                IndexingUpdateOp::Add,
            );
            let grad_w = dense.transpose().matmul(grad_out_t.clone());
            grads.register::<B>(weight_parent.id, grad_w.into_primitive().tensor());
        }

        if let Some(bias_parent) = bias_parent {
            let grad_b = grad_out_t.sum_dim(0).squeeze_dim::<1>(0);
            grads.register::<B>(bias_parent.id, grad_b.into_primitive().tensor());
        }
    }
}

#[cfg(feature = "train")]
impl<B, C> SparseInputLinearKernelBackend for Autodiff<B, C>
where
    B: SparseInputLinearKernelBackend,
    C: CheckpointStrategy,
{
    fn sparse_input_linear_forward(
        weight: FloatTensor<Self>,
        indices: IntTensor<Self>,
        values: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        let [vocab, _hidden] = weight.primitive.shape().dims();
        let weight_node = weight.node.clone();
        let bias_node = bias.as_ref().map(|b| b.node.clone());
        let has_bias = bias.is_some();

        // Two parent slots: weight, then optionally bias. When bias is
        // None we pass weight twice to satisfy the arity; backward only
        // consults `parents[1]` when `has_bias` is true.
        let parents = bias_node.as_ref().map_or_else(
            || [weight_node.clone(), weight_node.clone()],
            |node| [weight_node.clone(), node.clone()],
        );

        let backward = SparseInputLinearBackward::<B> {
            indices: indices.clone(),
            values: values.primitive.clone(),
            mask: mask.primitive.clone(),
            vocab,
            has_bias,
        };
        let prep = backward.prepare::<C>(parents).compute_bound().stateful();

        let bias_primitive = bias.map(|b| b.primitive);
        // Autodiff training path: always route through the custom
        // kernel since autodiff blocks the cubecl-fusion fold of the
        // composed `scatter + matmul` pattern. Inference paths bypass
        // this impl entirely (B::sparse_input_linear_forward, the
        // composed default, runs for `MoleculeEncoder` and validation).
        let output = B::sparse_input_linear_forward_kernel(
            weight.primitive,
            indices,
            values.primitive,
            mask.primitive,
            bias_primitive,
        );

        match prep {
            OpsKind::Tracked(prep) => prep.finish((), output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}
