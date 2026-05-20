//! `CountedTanimotoKernelBackend` for [`burn_ndarray::NdArray`].
//!
//! The trait is a "GPU kernel" abstraction in name only — what it actually
//! exposes is the per-anchor ranking algorithm. The CPU has no kernel to
//! dispatch, so the impl just calls into [`super::cpu::ranking_reference`]
//! one row at a time and assembles the output tensors host-side.
//!
//! With this in place, `Autodiff<NdArray>` (and bare `NdArray`) become
//! valid `TanimotoMetricBackend`s under the `cuda` feature gate too, so
//! the train binary's inline `train_epoch` / `evaluate` tests typecheck
//! under every feature combo CI exercises.

use burn::backend::ndarray::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayTensor, QuantElement, SharedArray,
};
use burn::tensor::{
    Int, Tensor, TensorData, TensorPrimitive,
    ops::{FloatTensor, IntTensor},
};

use super::api::{CountedTanimotoKernelBackend, CountedTanimotoRankingKernelConfig};
use super::cpu::{SparseRows, ranking_reference};

impl<E, I, Q> CountedTanimotoKernelBackend for NdArray<E, I, Q>
where
    E: FloatNdArrayElement,
    I: IntNdArrayElement,
    Q: QuantElement,
    NdArrayTensor: From<SharedArray<E>> + From<SharedArray<I>>,
{
    fn counted_tanimoto_similarity_ranking_kernel(
        indices: IntTensor<Self>,
        counts: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        config: CountedTanimotoRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
        let indices_tensor: Tensor<Self, 2, Int> = Tensor::from_primitive(indices);
        let counts_tensor: Tensor<Self, 2> = Tensor::from_primitive(TensorPrimitive::Float(counts));
        let mask_tensor: Tensor<Self, 2> = Tensor::from_primitive(TensorPrimitive::Float(mask));

        let [row_count, width] = counts_tensor.shape().dims();
        assert_eq!(
            indices_tensor.shape().dims::<2>(),
            [row_count, width],
            "Tanimoto geometry indices and counts must share a shape"
        );
        assert_eq!(
            mask_tensor.shape().dims::<2>(),
            [row_count, width],
            "Tanimoto geometry indices and mask must share a shape"
        );
        assert!(
            config.batch_items() <= row_count,
            "Tanimoto geometry batch_items exceeds the sparse batch row count"
        );

        let device = counts_tensor.device();

        // Materialise the sparse batch as host buffers in the canonical
        // layout the CPU helper consumes. `to_vec::<i64>()` works for both
        // `i32` and `i64` backends because Burn promotes ints to the
        // largest signed integer on conversion.
        let indices_host: Vec<i32> = indices_tensor
            .into_data()
            .iter::<i64>()
            .map(|value| i32::try_from(value).expect("fingerprint index fits in i32"))
            .collect();
        let counts_host: Vec<f32> = counts_tensor.into_data().iter::<f32>().collect();
        let mask_host: Vec<f32> = mask_tensor.into_data().iter::<f32>().collect();

        let rows = SparseRows {
            indices: &indices_host,
            counts: &counts_host,
            mask: &mask_host,
            width,
        };

        let candidate_count = config.effective_candidates_per_anchor();
        let mut candidate_index_buf = Vec::with_capacity(config.batch_items() * candidate_count);
        let mut best_position_buf = Vec::with_capacity(config.batch_items());
        let mut top2_gap_buf = Vec::with_capacity(config.batch_items());

        for anchor in 0..config.batch_items() {
            let result = ranking_reference(&rows, anchor, config);
            candidate_index_buf.extend(result.candidate_indices.into_iter().map(i64::from));
            best_position_buf.push(
                i64::try_from(result.best_candidate_position)
                    .expect("best candidate position fits in i64"),
            );
            top2_gap_buf.push(result.top2_gap);
        }

        let candidate_index = Tensor::<Self, 2, Int>::from_data(
            TensorData::new(candidate_index_buf, [config.batch_items(), candidate_count]),
            &device,
        )
        .into_primitive();
        let best_candidate_position = Tensor::<Self, 1, Int>::from_data(
            TensorData::new(best_position_buf, [config.batch_items()]),
            &device,
        )
        .into_primitive();
        let top2_gap = Tensor::<Self, 1>::from_data(
            TensorData::new(top2_gap_buf, [config.batch_items()]),
            &device,
        )
        .into_primitive()
        .tensor();

        (candidate_index, best_candidate_position, top2_gap)
    }
}
