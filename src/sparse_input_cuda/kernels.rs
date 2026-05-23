use burn_cubecl::cubecl::prelude::*;

/// EmbeddingBag-style forward.
///
/// One thread per `(batch, hidden)` output cell streams the row's K
/// active indices, accumulates `W[idx[k]] * value[k]` in a register,
/// adds the optional bias, writes once. No `[batch, K, hidden]`
/// intermediate is ever materialised.
#[cube(launch)]
pub(super) fn sparse_input_linear_forward_kernel<F: Float, I: Int>(
    weight: &Tensor<F>,
    indices: &Tensor<I>,
    values: &Tensor<F>,
    mask: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
    max_nnz: u32,
    hidden: u32,
    has_bias: u32,
) {
    let total = output.len();
    if ABSOLUTE_POS >= total {
        terminate!();
    }
    let pos = ABSOLUTE_POS;
    let h = pos % (hidden as usize);
    let b = pos / (hidden as usize);

    let zero = F::new(0.0_f32);
    let mut acc = zero;
    for k in 0..max_nnz as usize {
        let mask_offset = b * mask.stride(0) + k * mask.stride(1);
        if mask[mask_offset] > zero {
            let idx_offset = b * indices.stride(0) + k * indices.stride(1);
            let idx = usize::cast_from(indices[idx_offset]);
            let value_offset = b * values.stride(0) + k * values.stride(1);
            let weight_offset = idx * weight.stride(0) + h * weight.stride(1);
            acc += weight[weight_offset] * values[value_offset];
        }
    }

    if has_bias == 1u32 {
        acc += bias[h * bias.stride(0)];
    }

    output[b * output.stride(0) + h * output.stride(1)] = acc;
}
