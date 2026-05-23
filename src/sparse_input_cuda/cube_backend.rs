use super::api::SparseInputLinearKernelBackend;
use super::kernels::{
    IntrinsicFloatAtomicAddFamily, sparse_input_linear_backward_weight_kernel,
    sparse_input_linear_forward_kernel,
};
use burn::tensor::Shape;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_cubecl::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn_cubecl::ops::numeric::{empty_device_dtype, zeros_client};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

impl<R, F, I, BT> SparseInputLinearKernelBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    // `sparse_input_linear_forward` inherits the trait default
    // (composed `scatter -> matmul + bias`), which cubecl-fusion folds
    // into one fused kernel on inference paths. The custom kernel below
    // is the autodiff-training path where fusion cannot help.

    fn sparse_input_linear_forward_kernel(
        weight: FloatTensor<Self>,
        indices: IntTensor<Self>,
        values: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        indices.assert_is_on_same_device(&values);
        indices.assert_is_on_same_device(&mask);
        indices.assert_is_on_same_device(&weight);
        if let Some(bias) = bias.as_ref() {
            indices.assert_is_on_same_device(bias);
        }

        let [batch, max_nnz] = indices.meta.shape().dims();
        let [value_batch, value_nnz] = values.meta.shape().dims();
        let [mask_batch, mask_nnz] = mask.meta.shape().dims();
        assert_eq!(
            [batch, max_nnz],
            [value_batch, value_nnz],
            "sparse-input forward: indices and values must share a shape"
        );
        assert_eq!(
            [batch, max_nnz],
            [mask_batch, mask_nnz],
            "sparse-input forward: indices and mask must share a shape"
        );
        let [_vocab, hidden] = weight.meta.shape().dims();
        if let Some(bias) = bias.as_ref() {
            let [bias_len] = bias.meta.shape().dims();
            assert_eq!(
                bias_len, hidden,
                "sparse-input forward: bias length must equal hidden width"
            );
        }

        let output_shape = Shape::new([batch, hidden]);
        let output = empty_device_dtype(
            weight.client.clone(),
            weight.device.clone(),
            output_shape,
            weight.dtype,
        );

        let total_elems = batch * hidden;
        let cube_dim = CubeDim::new(&weight.client, total_elems);
        let cube_count = calculate_cube_count_elemwise(&weight.client, total_elems, cube_dim);

        // Without a bias the kernel still needs a stand-in tensor. A
        // size-1 zeroed tensor satisfies the type signature and is
        // never read because `has_bias = 0`.
        let has_bias = u32::from(bias.is_some());
        let bias_tensor = bias.unwrap_or_else(|| {
            zeros_client::<R>(
                weight.client.clone(),
                weight.device.clone(),
                Shape::new([1]),
                weight.dtype,
            )
        });

        let client = weight.client.clone();
        sparse_input_linear_forward_kernel::launch::<F, I, R>(
            &client,
            cube_count,
            cube_dim,
            weight.into_tensor_arg(),
            indices.into_tensor_arg(),
            values.into_tensor_arg(),
            mask.into_tensor_arg(),
            bias_tensor.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            max_nnz as u32,
            hidden as u32,
            has_bias,
        );

        output
    }

    fn sparse_input_linear_backward_weight(
        grad_out: FloatTensor<Self>,
        indices: IntTensor<Self>,
        values: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        vocab: usize,
    ) -> FloatTensor<Self> {
        indices.assert_is_on_same_device(&grad_out);
        indices.assert_is_on_same_device(&values);
        indices.assert_is_on_same_device(&mask);

        let client = grad_out.client.clone();

        // CUDA (sm_60+) has intrinsic `atomicAdd(float*)` in hardware,
        // but cubecl-cuda 0.10 does not register scalar `Atomic<F32>`
        // Add in its runtime.rs (only F64 at sm_60+, F16 at sm_70+,
        // F32x2/F32x4 at sm_90+; see lines 177-258). So the standard
        // `client.properties().atomic_type_usage(...)` check returns
        // false even on a 4090 where the op runs fine. We gate on the
        // `cuda` feature instead: production training has it on and
        // takes the kernel path; cubecl-cpu test builds do not and
        // fall back to composed ops (cubecl-cpu has neither intrinsic
        // atomic f32 nor a working CAS-on-u32 fallback in 0.10).
        //
        // Cleanup path: a cubecl-cuda PR adding the missing scalar
        // `Atomic<F32>` Add registration at the `arch_version >= 60`
        // block lets us swap the cfg gate back for the runtime probe.
        if !cfg!(feature = "cuda") {
            return default_backward_weight::<Self>(grad_out, indices, values, mask, vocab);
        }

        let device = grad_out.device.clone();
        let dtype = grad_out.dtype;

        let [batch, max_nnz] = indices.meta.shape().dims();
        let [_grad_batch, hidden] = grad_out.meta.shape().dims();

        let grad_w = zeros_client::<R>(
            client.clone(),
            device.clone(),
            Shape::new([vocab, hidden]),
            dtype,
        );

        let total = batch * max_nnz * hidden;
        let cube_dim = CubeDim::new(&client, total);
        let cube_count = calculate_cube_count_elemwise(&client, total, cube_dim);

        // Inlined equivalent of the (crate-private) `address_type!`
        // macro: the kernel's dynamic-address resolution wants the max
        // address-type across all tensor args.
        let address_type = [
            grad_out.required_address_type(),
            indices.required_address_type(),
            values.required_address_type(),
            mask.required_address_type(),
            grad_w.required_address_type(),
        ]
        .into_iter()
        .max()
        .unwrap_or_default();

        sparse_input_linear_backward_weight_kernel::launch::<IntrinsicFloatAtomicAddFamily, I, R>(
            &client,
            cube_count,
            cube_dim,
            address_type,
            grad_out.into_tensor_arg(),
            indices.into_tensor_arg(),
            values.into_tensor_arg(),
            mask.into_tensor_arg(),
            grad_w.clone().into_tensor_arg(),
            max_nnz as u32,
            hidden as u32,
            [dtype.into(), dtype.into()],
        );

        grad_w
    }
}

/// Composed-ops backward used as the fallback when the runtime cannot
/// support the intrinsic atomic-f32-add kernel. Replicates the trait's
/// default impl as a free function so the trait override can delegate
/// to it without re-entering its own trait method.
fn default_backward_weight<B: SparseInputLinearKernelBackend>(
    grad_out: FloatTensor<B>,
    indices: IntTensor<B>,
    values: FloatTensor<B>,
    mask: FloatTensor<B>,
    vocab: usize,
) -> FloatTensor<B> {
    use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorPrimitive};
    let grad_out_t = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(grad_out));
    let indices_t = Tensor::<B, 2, Int>::from_primitive(indices);
    let values_t = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(values));
    let mask_t = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(mask));
    let [batch, _max_nnz] = indices_t.dims();
    let device = values_t.device();
    let dense = Tensor::<B, 2>::zeros([batch, vocab], &device).scatter(
        1,
        indices_t,
        values_t * mask_t,
        IndexingUpdateOp::Add,
    );
    let grad_w = dense.transpose().matmul(grad_out_t);
    grad_w.into_primitive().tensor()
}
