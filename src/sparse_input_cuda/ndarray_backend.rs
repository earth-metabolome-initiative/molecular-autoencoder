//! NdArray inherits the trait's default composed-ops forward; there is
//! no custom kernel on this backend. The empty impl exists so the
//! model's `B: SparseInputLinearKernelBackend` bound is satisfied for
//! ndarray-only builds (tests, wasm inference).

use super::api::SparseInputLinearKernelBackend;
use burn::backend::ndarray::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SharedArray,
};

impl<E, I, Q> SparseInputLinearKernelBackend for NdArray<E, I, Q>
where
    E: FloatNdArrayElement,
    I: IntNdArrayElement,
    Q: QuantElement,
    NdArrayTensor: From<SharedArray<E>> + From<SharedArray<I>>,
{
}
