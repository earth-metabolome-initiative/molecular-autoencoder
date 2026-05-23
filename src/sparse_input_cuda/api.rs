use burn::tensor::backend::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{Int as TensorInt, Tensor as BurnTensor, TensorPrimitive};

/// Backend extension that implements the EmbeddingBag-style sparse-input
/// linear projection
/// `out[b, h] = sum_k W[indices[b, k], h] * values[b, k] * mask[b, k] + bias[h]`.
///
/// Forward streams over the K active indices per `(b, h)` cell with no
/// `[batch, max_nnz, hidden]` intermediate, eliminating the 22% scatter
/// cost nsys flagged. The backward path runs entirely through Burn's
/// composed dense ops (scatter to reconstruct the dense input, then
/// dense matmul backward); since `cubecl-cpu` does not implement
/// `atomic<f32>` and the cubecl `FloatAtomicAddFamily` boilerplate is
/// substantial, the win we currently capture is forward-only. A custom
/// backward kernel can slot in behind this same trait later without
/// touching callers.
pub trait SparseInputLinearKernelBackend: Backend {
    /// Forward kernel. `weight` is `[vocab, hidden]`, `indices` is
    /// `[batch, max_nnz]` (padded), `values` and `mask` mirror
    /// `indices`. `bias` is `[hidden]` or `None`. Output is
    /// `[batch, hidden]`.
    fn sparse_input_linear_forward(
        weight: FloatTensor<Self>,
        indices: IntTensor<Self>,
        values: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self>;
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
