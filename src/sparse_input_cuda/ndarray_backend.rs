//! NdArray fallback that implements the trait using Burn's composed dense
//! ops (`scatter` -> `matmul` + optional bias). Preserves the pre-kernel
//! behaviour so ndarray-only builds compile and tests stay valid; the
//! cubecl kernel is the path that captures the actual perf win.

use super::api::SparseInputLinearKernelBackend;
use burn::backend::ndarray::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SharedArray,
};
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorMetadata, TensorPrimitive};

impl<E, I, Q> SparseInputLinearKernelBackend for NdArray<E, I, Q>
where
    E: FloatNdArrayElement,
    I: IntNdArrayElement,
    Q: QuantElement,
    NdArrayTensor: From<SharedArray<E>> + From<SharedArray<I>>,
{
    fn sparse_input_linear_forward(
        weight: FloatTensor<Self>,
        indices: IntTensor<Self>,
        values: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        let [vocab, _hidden] = weight.shape().dims();
        let weight_t = Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(weight));
        let indices_t = Tensor::<Self, 2, Int>::from_primitive(indices);
        let values_t = Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(values));
        let mask_t = Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(mask));
        let [batch, _max_nnz] = indices_t.dims();
        let device = values_t.device();
        let dense = Tensor::<Self, 2>::zeros([batch, vocab], &device).scatter(
            1,
            indices_t,
            values_t * mask_t,
            IndexingUpdateOp::Add,
        );
        let proj = dense.matmul(weight_t);
        let out = match bias {
            Some(b) => {
                let b_t = Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(b));
                proj + b_t.unsqueeze_dim::<2>(0)
            }
            None => proj,
        };
        out.into_primitive().tensor()
    }
}
