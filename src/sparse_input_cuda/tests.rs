use super::api::SparseInputLinearKernelBackend;
use super::*;
use burn::backend::Autodiff;
use burn::tensor::{IndexingUpdateOp, Tensor as BurnTensor, TensorData, TensorPrimitive};

#[cfg(feature = "cuda")]
mod backend_alias {
    pub use burn::backend::Cuda as TestInner;
    pub use burn::backend::cuda::CudaDevice as TestDevice;
}

#[cfg(all(not(feature = "cuda"), feature = "cubecl-cpu"))]
mod backend_alias {
    pub use burn_cpu::Cpu as TestInner;
    pub use burn_cpu::CpuDevice as TestDevice;
}

use backend_alias::{TestDevice, TestInner};

type TestBackend = TestInner<f32, i32>;
type AutodiffTestBackend = Autodiff<TestBackend>;

/// Builds a fixed sparse fingerprint fixture used by every forward test:
/// `batch=4`, `max_nnz=3`, `vocab=16`, `hidden=8`.
fn fixture(
    device: &burn::tensor::Device<TestBackend>,
) -> (
    BurnTensor<TestBackend, 2>,                    // weight [vocab, hidden]
    BurnTensor<TestBackend, 2, burn::tensor::Int>, // indices [batch, max_nnz]
    BurnTensor<TestBackend, 2>,                    // values [batch, max_nnz]
    BurnTensor<TestBackend, 2>,                    // mask
) {
    let vocab = 16usize;
    let hidden = 8usize;

    // Deterministic small weights.
    let weight_data: Vec<f32> = (0..vocab * hidden).map(|i| 0.01 * i as f32 - 0.5).collect();
    let weight = BurnTensor::<TestBackend, 2>::from_data(
        TensorData::new(weight_data, [vocab, hidden]),
        device,
    );

    let indices = BurnTensor::<TestBackend, 2, burn::tensor::Int>::from_data(
        TensorData::new(
            vec![
                0_i32, 5, 11, // row 0: bins 0, 5, 11 active
                3, 7, 0, // row 1: bins 3, 7 active; last slot padded
                15, 0, 0, // row 2: only bin 15 active
                1, 2, 9, // row 3: bins 1, 2, 9 active
            ],
            [4, 3],
        ),
        device,
    );
    let values = BurnTensor::<TestBackend, 2>::from_data(
        TensorData::new(
            vec![
                0.7_f32, 1.1, 0.5, 1.4, 0.3, 0.0, 0.9, 0.0, 0.0, 0.6, 0.8, 1.2,
            ],
            [4, 3],
        ),
        device,
    );
    let mask = BurnTensor::<TestBackend, 2>::from_data(
        TensorData::new(
            vec![
                1.0_f32, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            ],
            [4, 3],
        ),
        device,
    );

    (weight, indices, values, mask)
}

/// Reference forward via the dense scatter-then-matmul path.
fn dense_reference(
    weight: BurnTensor<TestBackend, 2>,
    indices: BurnTensor<TestBackend, 2, burn::tensor::Int>,
    values: BurnTensor<TestBackend, 2>,
    mask: BurnTensor<TestBackend, 2>,
    bias: Option<BurnTensor<TestBackend, 1>>,
    vocab: usize,
) -> BurnTensor<TestBackend, 2> {
    let device = values.device();
    let batch = indices.dims()[0];
    let dense = BurnTensor::<TestBackend, 2>::zeros([batch, vocab], &device).scatter(
        1,
        indices,
        values * mask,
        IndexingUpdateOp::Add,
    );
    let proj = dense.matmul(weight);
    match bias {
        Some(b) => proj + b.unsqueeze_dim::<2>(0),
        None => proj,
    }
}

#[test]
#[allow(
    clippy::default_constructed_unit_structs,
    reason = "TestDevice is unit-struct under cubecl-cpu and tuple-struct under cuda; \
              Default::default keeps one expression across features."
)]
fn forward_kernel_matches_dense_reference_no_bias() {
    let device = TestDevice::default();
    let (weight, indices, values, mask) = fixture(&device);
    let vocab = weight.dims()[0];

    let kernel_out = sparse_input_linear_forward(
        weight.clone(),
        indices.clone(),
        values.clone(),
        mask.clone(),
        None,
    )
    .into_data()
    .to_vec::<f32>()
    .expect("f32 kernel output");

    let dense_out = dense_reference(weight, indices, values, mask, None, vocab)
        .into_data()
        .to_vec::<f32>()
        .expect("f32 dense output");

    assert_eq!(kernel_out.len(), dense_out.len());
    for (i, (k, d)) in kernel_out.iter().zip(&dense_out).enumerate() {
        assert!(
            (k - d).abs() < 1.0e-5,
            "mismatch at index {i}: kernel={k} dense={d}"
        );
    }
}

#[test]
#[allow(
    clippy::default_constructed_unit_structs,
    reason = "TestDevice cross-feature alias requires Default::default()."
)]
fn explicit_kernel_method_matches_dense_reference() {
    // Exercises `B::sparse_input_linear_forward_kernel` directly,
    // bypassing the autodiff path. Without this we'd only validate the
    // composed-ops default; the kernel itself would only ever run via
    // the autodiff backward test.
    let device = TestDevice::default();
    let (weight, indices, values, mask) = fixture(&device);
    let vocab = weight.dims()[0];

    let kernel_primitive =
        <TestBackend as SparseInputLinearKernelBackend>::sparse_input_linear_forward_kernel(
            weight.clone().into_primitive().tensor(),
            indices.clone().into_primitive(),
            values.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            None,
        );
    let kernel_out: BurnTensor<TestBackend, 2> =
        BurnTensor::<TestBackend, 2>::from_primitive(TensorPrimitive::Float(kernel_primitive));
    let kernel_vec = kernel_out
        .into_data()
        .to_vec::<f32>()
        .expect("f32 kernel output");
    let dense_vec = dense_reference(weight, indices, values, mask, None, vocab)
        .into_data()
        .to_vec::<f32>()
        .expect("f32 dense output");

    assert_eq!(kernel_vec.len(), dense_vec.len());
    for (i, (k, d)) in kernel_vec.iter().zip(&dense_vec).enumerate() {
        assert!(
            (k - d).abs() < 1.0e-5,
            "kernel mismatch at index {i}: kernel={k} dense={d}"
        );
    }
}

#[test]
#[allow(
    clippy::default_constructed_unit_structs,
    reason = "TestDevice cross-feature alias requires Default::default()."
)]
fn inference_path_uses_composed_default_not_kernel() {
    // Sanity: the high-level free function on a non-autodiff backend
    // routes through `sparse_input_linear_forward` (composed ops), not
    // `sparse_input_linear_forward_kernel`. We can't directly observe
    // the kernel pick without an instrumentation hook, but we can
    // confirm equivalence with the explicit composed-ops invocation.
    let device = TestDevice::default();
    let (weight, indices, values, mask) = fixture(&device);
    let vocab = weight.dims()[0];

    let free_fn_out = sparse_input_linear_forward(
        weight.clone(),
        indices.clone(),
        values.clone(),
        mask.clone(),
        None,
    )
    .into_data()
    .to_vec::<f32>()
    .expect("f32");

    let composed_primitive =
        <TestBackend as SparseInputLinearKernelBackend>::sparse_input_linear_forward(
            weight.clone().into_primitive().tensor(),
            indices.clone().into_primitive(),
            values.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            None,
        );
    let composed_vec: Vec<f32> =
        BurnTensor::<TestBackend, 2>::from_primitive(TensorPrimitive::Float(composed_primitive))
            .into_data()
            .to_vec::<f32>()
            .expect("f32");
    let dense_vec = dense_reference(weight, indices, values, mask, None, vocab)
        .into_data()
        .to_vec::<f32>()
        .expect("f32");

    for (i, ((free, comp), dense)) in free_fn_out
        .iter()
        .zip(&composed_vec)
        .zip(&dense_vec)
        .enumerate()
    {
        assert!(
            (free - dense).abs() < 1.0e-5,
            "free-fn vs dense mismatch at {i}"
        );
        assert!(
            (comp - dense).abs() < 1.0e-5,
            "composed-default vs dense mismatch at {i}"
        );
    }
}

/// Builds the same fixture on the autodiff backend so we can take a
/// backward pass through `weight`.
fn autodiff_fixture(
    device: &burn::tensor::Device<AutodiffTestBackend>,
) -> (
    BurnTensor<AutodiffTestBackend, 2>,
    BurnTensor<AutodiffTestBackend, 2, burn::tensor::Int>,
    BurnTensor<AutodiffTestBackend, 2>,
    BurnTensor<AutodiffTestBackend, 2>,
) {
    let vocab = 16usize;
    let hidden = 8usize;
    let weight_data: Vec<f32> = (0..vocab * hidden).map(|i| 0.01 * i as f32 - 0.5).collect();
    let weight = BurnTensor::<AutodiffTestBackend, 2>::from_data(
        TensorData::new(weight_data, [vocab, hidden]),
        device,
    )
    .require_grad();
    let indices = BurnTensor::<AutodiffTestBackend, 2, burn::tensor::Int>::from_data(
        TensorData::new(vec![0_i32, 5, 11, 3, 7, 0, 15, 0, 0, 1, 2, 9], [4, 3]),
        device,
    );
    let values = BurnTensor::<AutodiffTestBackend, 2>::from_data(
        TensorData::new(
            vec![
                0.7_f32, 1.1, 0.5, 1.4, 0.3, 0.0, 0.9, 0.0, 0.0, 0.6, 0.8, 1.2,
            ],
            [4, 3],
        ),
        device,
    );
    let mask = BurnTensor::<AutodiffTestBackend, 2>::from_data(
        TensorData::new(
            vec![
                1.0_f32, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            ],
            [4, 3],
        ),
        device,
    );
    (weight, indices, values, mask)
}

#[test]
#[allow(
    clippy::default_constructed_unit_structs,
    reason = "TestDevice cross-feature alias requires Default::default()."
)]
fn backward_kernel_weight_grad_matches_dense_reference() {
    let device = TestDevice::default();

    // Kernel path.
    let (weight_k, indices_k, values_k, mask_k) = autodiff_fixture(&device);
    let out_k = sparse_input_linear_forward(weight_k.clone(), indices_k, values_k, mask_k, None);
    let loss_k = out_k.sum();
    let grads_k = loss_k.backward();
    let grad_w_k = weight_k
        .grad(&grads_k)
        .expect("weight grad via kernel")
        .into_data()
        .to_vec::<f32>()
        .expect("f32 grad");

    // Reference path: dense scatter-then-matmul via Burn composed ops.
    let (weight_r, indices_r, values_r, mask_r) = autodiff_fixture(&device);
    let dense = BurnTensor::<AutodiffTestBackend, 2>::zeros([4, 16], &device).scatter(
        1,
        indices_r,
        values_r * mask_r,
        IndexingUpdateOp::Add,
    );
    let out_r = dense.matmul(weight_r.clone());
    let loss_r = out_r.sum();
    let grads_r = loss_r.backward();
    let grad_w_r = weight_r
        .grad(&grads_r)
        .expect("weight grad via reference")
        .into_data()
        .to_vec::<f32>()
        .expect("f32 grad");

    assert_eq!(grad_w_k.len(), grad_w_r.len());
    for (i, (k, r)) in grad_w_k.iter().zip(&grad_w_r).enumerate() {
        assert!(
            (k - r).abs() < 1.0e-4,
            "mismatch at flat idx {i}: kernel={k} reference={r}"
        );
    }
}

#[test]
#[allow(
    clippy::default_constructed_unit_structs,
    reason = "TestDevice cross-feature alias requires Default::default()."
)]
fn forward_kernel_matches_dense_reference_with_bias() {
    let device = TestDevice::default();
    let (weight, indices, values, mask) = fixture(&device);
    let vocab = weight.dims()[0];
    let hidden = weight.dims()[1];
    let bias = BurnTensor::<TestBackend, 1>::from_data(
        TensorData::new(
            (0..hidden)
                .map(|i| 0.1 * i as f32 - 0.3)
                .collect::<Vec<_>>(),
            [hidden],
        ),
        &device,
    );

    let kernel_out = sparse_input_linear_forward(
        weight.clone(),
        indices.clone(),
        values.clone(),
        mask.clone(),
        Some(bias.clone()),
    )
    .into_data()
    .to_vec::<f32>()
    .expect("f32 kernel output");

    let dense_out = dense_reference(weight, indices, values, mask, Some(bias), vocab)
        .into_data()
        .to_vec::<f32>()
        .expect("f32 dense output");

    assert_eq!(kernel_out.len(), dense_out.len());
    for (i, (k, d)) in kernel_out.iter().zip(&dense_out).enumerate() {
        assert!(
            (k - d).abs() < 1.0e-5,
            "mismatch at index {i}: kernel={k} dense={d}"
        );
    }
}
