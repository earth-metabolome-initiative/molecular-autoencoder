//! Fusion-aware forwarder so `Fusion<B>` inherits the trait by recording
//! the op in the fusion graph and replaying it on the inner backend.

use super::api::SparseInputLinearKernelBackend;
use burn::tensor::Element as _;
use burn::tensor::Shape;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_fusion::{
    Fusion, FusionBackend,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, OperationIr, TensorIr};
use core::marker::PhantomData;

#[cfg(feature = "cuda-fusion")]
impl<B> SparseInputLinearKernelBackend for Fusion<B>
where
    B: FusionBackend + SparseInputLinearKernelBackend + Send + Sync,
{
    // `sparse_input_linear_forward` inherits the composed-ops default
    // (cubecl-fusion folds it naturally). Only the kernel variant
    // needs explicit recording via CustomOpIr.

    fn sparse_input_linear_forward_kernel(
        weight: FloatTensor<Self>,
        indices: IntTensor<Self>,
        values: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        let [_vocab, hidden] = weight.shape.dims();
        let [batch, _max_nnz] = indices.shape.dims();
        let dtype = weight.dtype;
        let output_shape = Shape::new([batch, hidden]);
        let client = weight.client.clone();
        let output = TensorIr::uninit(client.create_empty_handle(), output_shape, dtype);

        let streams = bias.as_ref().map_or_else(
            || OperationStreams::with_inputs([&weight, &indices, &values, &mask]),
            |b| OperationStreams::with_inputs([&weight, &indices, &values, &mask, b]),
        );

        let inputs: Vec<TensorIr> = bias.as_ref().map_or_else(
            || {
                vec![
                    weight.clone().into_ir(),
                    indices.clone().into_ir(),
                    values.clone().into_ir(),
                    mask.clone().into_ir(),
                ]
            },
            |b| {
                vec![
                    weight.clone().into_ir(),
                    indices.clone().into_ir(),
                    values.clone().into_ir(),
                    mask.clone().into_ir(),
                    b.clone().into_ir(),
                ]
            },
        );

        let desc = CustomOpIr::new(
            "sparse_input_linear_forward",
            &inputs,
            std::slice::from_ref(&output),
        );

        let mut outputs = client.register(
            streams,
            OperationIr::Custom(desc.clone()),
            SparseInputLinearForwardFusion::<B> {
                desc,
                has_bias: bias.is_some(),
                backend: PhantomData,
            },
        );

        outputs
            .pop()
            .expect("sparse-input-linear custom op has one output")
    }
}

/// Fusion operation shim that replays the kernel on the inner backend
/// when the fusion runtime flushes recorded ops.
#[cfg(feature = "cuda-fusion")]
#[derive(Debug)]
struct SparseInputLinearForwardFusion<B: FusionBackend> {
    /// Fusion-IR descriptor binding the inputs/outputs to handles.
    desc: CustomOpIr,
    /// Whether the recorded call passed a bias tensor.
    has_bias: bool,
    /// Phantom marker for the inner fusion backend.
    backend: PhantomData<B>,
}

#[cfg(feature = "cuda-fusion")]
impl<B> Operation<B::FusionRuntime> for SparseInputLinearForwardFusion<B>
where
    B: FusionBackend + SparseInputLinearKernelBackend + Send + Sync,
{
    fn execute(&self, handles: &mut burn_ir::HandleContainer<B::Handle>) {
        let inputs = self.desc.inputs.as_slice();
        let weight = handles.get_float_tensor::<B>(&inputs[0]);
        let indices = handles.get_int_tensor::<B>(&inputs[1]);
        let values = handles.get_float_tensor::<B>(&inputs[2]);
        let mask = handles.get_float_tensor::<B>(&inputs[3]);
        let bias = if self.has_bias {
            Some(handles.get_float_tensor::<B>(&inputs[4]))
        } else {
            None
        };

        let output = B::sparse_input_linear_forward_kernel(weight, indices, values, mask, bias);
        let _ = B::FloatElem::dtype();
        handles.register_float_tensor::<B>(&self.desc.outputs[0].id, output);
    }
}
