use burn::tensor::backend::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{
    IndexingUpdateOp, Int as TensorInt, Tensor as BurnTensor, TensorMetadata, TensorPrimitive,
};

/// Backend extension that implements the EmbeddingBag-style sparse-input
/// linear projection
/// `out[b, h] = sum_k W[indices[b, k], h] * values[b, k] * mask[b, k] + bias[h]`.
///
/// Two entry points so training and inference take different paths:
///
/// - [`Self::sparse_input_linear_forward`] uses the composed
///   `scatter -> matmul (+ bias)` pattern. cubecl-fusion folds the
///   scatter into the matmul on the inference path (no autodiff graph
///   to block it), so the dense formulation is the fast option for
///   `MoleculeEncoder` and validation batches.
/// - [`Self::sparse_input_linear_forward_kernel`] dispatches to the
///   custom `#[cube]` kernel that never materialises the dense
///   `[batch, fingerprint_size]` intermediate. Training cannot fuse the
///   scatter under autodiff (the original `select_assign_kernel` cost
///   nsys flagged at 22%), so the autodiff forward routes here instead.
///
/// Both methods have default composed-ops bodies, so a backend that
/// has no kernel implementation (NdArray) just needs an empty `impl`.
pub trait SparseInputLinearKernelBackend: Backend {
    /// Composed-ops forward. cubecl-fusion folds `scatter + matmul` on
    /// the inference path; this is the inference-friendly default.
    fn sparse_input_linear_forward(
        weight: FloatTensor<Self>,
        indices: IntTensor<Self>,
        values: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        let [vocab, _hidden] = weight.shape().dims();
        let weight_t = BurnTensor::<Self, 2>::from_primitive(TensorPrimitive::Float(weight));
        let indices_t = BurnTensor::<Self, 2, TensorInt>::from_primitive(indices);
        let values_t = BurnTensor::<Self, 2>::from_primitive(TensorPrimitive::Float(values));
        let mask_t = BurnTensor::<Self, 2>::from_primitive(TensorPrimitive::Float(mask));
        let [batch, _max_nnz] = indices_t.dims();
        let device = values_t.device();
        let dense = BurnTensor::<Self, 2>::zeros([batch, vocab], &device).scatter(
            1,
            indices_t,
            values_t * mask_t,
            IndexingUpdateOp::Add,
        );
        let proj = dense.matmul(weight_t);
        let out = match bias {
            Some(b) => {
                let b_t = BurnTensor::<Self, 1>::from_primitive(TensorPrimitive::Float(b));
                proj + b_t.unsqueeze_dim::<2>(0)
            }
            None => proj,
        };
        out.into_primitive().tensor()
    }

    /// Kernel forward. Default delegates to
    /// [`Self::sparse_input_linear_forward`] so non-kernel backends
    /// just inherit composed ops; CubeBackend overrides this with the
    /// `#[cube]` launch.
    fn sparse_input_linear_forward_kernel(
        weight: FloatTensor<Self>,
        indices: IntTensor<Self>,
        values: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        Self::sparse_input_linear_forward(weight, indices, values, mask, bias)
    }

    /// Weight-gradient backward for the kernel forward. `vocab` is the
    /// dense weight first-dim, needed to allocate `grad_W` since the
    /// indices tensor does not encode it.
    ///
    /// Default body reconstructs the dense input via `scatter` and uses
    /// `dense.transpose().matmul(grad_out)` to produce the gradient.
    /// CubeBackend overrides with the atomic scatter-add kernel that
    /// never materialises the dense intermediate. ulp-nondeterministic
    /// in the kernel path; acceptable for ML training.
    fn sparse_input_linear_backward_weight(
        grad_out: FloatTensor<Self>,
        indices: IntTensor<Self>,
        values: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        vocab: usize,
    ) -> FloatTensor<Self> {
        let grad_out_t = BurnTensor::<Self, 2>::from_primitive(TensorPrimitive::Float(grad_out));
        let indices_t = BurnTensor::<Self, 2, TensorInt>::from_primitive(indices);
        let values_t = BurnTensor::<Self, 2>::from_primitive(TensorPrimitive::Float(values));
        let mask_t = BurnTensor::<Self, 2>::from_primitive(TensorPrimitive::Float(mask));
        let [batch, _max_nnz] = indices_t.dims();
        let device = values_t.device();
        let dense = BurnTensor::<Self, 2>::zeros([batch, vocab], &device).scatter(
            1,
            indices_t,
            values_t * mask_t,
            IndexingUpdateOp::Add,
        );
        let grad_w = dense.transpose().matmul(grad_out_t);
        grad_w.into_primitive().tensor()
    }
}

/// Free-function entry that hides the trait machinery from call sites.
/// Returns the projected `[batch, hidden]` Burn tensor with full
/// autograd routing through the backend implementation.
pub fn sparse_input_linear_forward<B: SparseInputLinearKernelBackend>(
    weight: BurnTensor<B, 2>,
    indices: BurnTensor<B, 2, TensorInt>,
    values: BurnTensor<B, 2>,
    mask: BurnTensor<B, 2>,
    bias: Option<BurnTensor<B, 1>>,
) -> BurnTensor<B, 2> {
    let bias_primitive = bias.map(|b| b.into_primitive().tensor());
    let out = B::sparse_input_linear_forward(
        weight.into_primitive().tensor(),
        indices.into_primitive(),
        values.into_primitive().tensor(),
        mask.into_primitive().tensor(),
        bias_primitive,
    );
    BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(out))
}
