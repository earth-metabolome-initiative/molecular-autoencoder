use super::api::SparseInputLinearKernelBackend;
use super::kernels::sparse_input_linear_forward_kernel;
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
    fn sparse_input_linear_forward(
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
}
