use burn_cubecl::cubecl::prelude::*;
use core::marker::PhantomData;

/// Family of `f32 atomic add` strategies. Two impls ship: an intrinsic
/// path that uses hardware `atomicAdd(float*)` (CUDA sm_60+), and a
/// compare-exchange fallback that bitcasts f32 ↔ u32 and uses integer
/// CAS, which works on any backend that exposes `Atomic<u32>` (incl.
/// cubecl-cpu). Lifted from `burn-cubecl-0.21/src/kernel/conv/
/// deform_conv_transpose2d.rs`.
#[cube]
pub(super) trait FloatAtomicAddFamily: Send + Sync + 'static {
    /// Per-element strategy parameterised on the proxy float type.
    type Op<ProxyType: Float>: FloatAtomicAdd;
}

/// Per-element atomic-add strategy. `ProxyType` is the type actually
/// stored in the gradient buffer (matches `F` for intrinsic, `u32` for
/// CAS-on-bitcast).
#[cube]
pub(super) trait FloatAtomicAdd: Send + Sync + 'static {
    /// Storage type of the atomic.
    type ProxyType: Numeric;
    /// Atomically add `value` into `ptr`.
    fn float_atomic_add<F: Float>(ptr: &mut Atomic<Self::ProxyType>, value: F);
}

/// Resolved storage proxy type for the kernel signature.
pub(super) type ProxyType<FADF, FP> =
    <<FADF as FloatAtomicAddFamily>::Op<FP> as FloatAtomicAdd>::ProxyType;

#[derive(CubeType)]
pub(super) struct IntrinsicFloatAtomicAdd<F: Float> {
    #[cube(comptime)]
    _ty: PhantomData<F>,
}

#[derive(CubeType)]
pub(super) struct CASFloatAtomicAdd;

pub(super) struct IntrinsicFloatAtomicAddFamily;

impl FloatAtomicAddFamily for IntrinsicFloatAtomicAddFamily {
    type Op<ProxyType: Float> = IntrinsicFloatAtomicAdd<ProxyType>;
}

impl FloatAtomicAddFamily for CASFloatAtomicAdd {
    type Op<ProxyType: Float> = Self;
}

#[cube]
impl<FAdd: Float> FloatAtomicAdd for IntrinsicFloatAtomicAdd<FAdd> {
    type ProxyType = FAdd;

    fn float_atomic_add<F: Float>(ptr: &mut Atomic<FAdd>, value: F) {
        let value = FAdd::cast_from(value);
        ptr.fetch_add(value);
    }
}

#[cube]
impl FloatAtomicAdd for CASFloatAtomicAdd {
    type ProxyType = u32;

    fn float_atomic_add<F: Float>(ptr: &mut Atomic<Self::ProxyType>, value: F) {
        let value = f32::cast_from(value);
        if value != 0.0 {
            let mut v = ptr.load();
            loop {
                let prev = v;
                let v_float = f32::from_bits(v);
                let new = (v_float + value).to_bits();
                v = ptr.compare_exchange_weak(v, new);
                if prev == v {
                    break;
                }
            }
        }
    }
}

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

/// Backward kernel for the weight gradient. One thread per `(b, k, h)`
/// triple atomically adds `grad_out[b, h] * values[b, k] * mask[b, k]`
/// into `grad_W[indices[b, k], h]`. `grad_w` must be pre-zeroed
/// (dispatcher allocates via `zeros_client`). ulp-nondeterministic
/// across runs because the atomic ordering depends on warp scheduling;
/// acceptable for ML training.
#[cube(launch, address_type = "dynamic")]
pub(super) fn sparse_input_linear_backward_weight_kernel<
    F: Float,
    FP: Float,
    FAdd: FloatAtomicAddFamily,
    I: Int,
>(
    grad_out: &Tensor<F>,
    indices: &Tensor<I>,
    values: &Tensor<F>,
    mask: &Tensor<F>,
    grad_w: &mut Tensor<Atomic<ProxyType<FAdd, FP>>>,
    max_nnz: u32,
    hidden: u32,
    #[define(F, FP)] _dtypes: [StorageType; 2],
) {
    let total = grad_out.shape(0) * (max_nnz as usize) * (hidden as usize);
    if ABSOLUTE_POS >= total {
        terminate!();
    }
    let pos = ABSOLUTE_POS;
    let h = pos % (hidden as usize);
    let after_h = pos / (hidden as usize);
    let k = after_h % (max_nnz as usize);
    let b = after_h / (max_nnz as usize);

    let zero = F::new(0.0_f32);
    let mask_offset = b * mask.stride(0) + k * mask.stride(1);
    let m = mask[mask_offset];
    if m <= zero {
        terminate!();
    }

    let idx_offset = b * indices.stride(0) + k * indices.stride(1);
    let idx = usize::cast_from(indices[idx_offset]);
    let value_offset = b * values.stride(0) + k * values.stride(1);
    let grad_offset = b * grad_out.stride(0) + h * grad_out.stride(1);
    let contribution = grad_out[grad_offset] * values[value_offset];

    let target_offset = idx * grad_w.stride(0) + h * grad_w.stride(1);
    FAdd::Op::<FP>::float_atomic_add::<F>(&mut grad_w[target_offset], contribution);
}
