use super::api::{CountedTanimotoKernelBackend, CountedTanimotoRankingKernelConfig};
use super::kernels::counted_tanimoto_similarity_ranking_forward;
use burn::tensor::Shape;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_cubecl::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

impl<R, F, I, BT> CountedTanimotoKernelBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn counted_tanimoto_similarity_ranking_kernel(
        indices: IntTensor<Self>,
        counts: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        config: CountedTanimotoRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
        indices.assert_is_on_same_device(&counts);
        indices.assert_is_on_same_device(&mask);

        let [index_rows, index_width] = indices.meta.shape().dims();
        let [count_rows, count_width] = counts.meta.shape().dims();
        let [mask_rows, mask_width] = mask.meta.shape().dims();
        assert_eq!(
            [index_rows, index_width],
            [count_rows, count_width],
            "Tanimoto geometry indices and counts must share a shape"
        );
        assert_eq!(
            [index_rows, index_width],
            [mask_rows, mask_width],
            "Tanimoto geometry indices and mask must share a shape"
        );
        assert!(
            config.batch_items <= index_rows,
            "Tanimoto geometry batch_items exceeds the sparse batch row count"
        );

        let candidate_count = config.effective_candidates_per_anchor();
        let candidate_shape = Shape::new([config.batch_items, candidate_count]);
        let index_shape = Shape::new([config.batch_items]);
        let gap_shape = Shape::new([config.batch_items]);
        let candidate_index = empty_device_dtype(
            counts.client.clone(),
            counts.device.clone(),
            candidate_shape,
            I::dtype(),
        );
        let best_candidate_position = empty_device_dtype(
            counts.client.clone(),
            counts.device.clone(),
            index_shape,
            I::dtype(),
        );
        let top2_gap = empty_device_dtype(
            counts.client.clone(),
            counts.device.clone(),
            gap_shape,
            counts.dtype,
        );

        let total_elem = config.batch_items;
        let cube_dim = CubeDim::new(&counts.client, total_elem);
        let cube_count = calculate_cube_count_elemwise(&counts.client, total_elem, cube_dim);

        let client = counts.client.clone();
        counted_tanimoto_similarity_ranking_forward::launch::<F, I, R>(
            &client,
            cube_count,
            cube_dim,
            indices.into_tensor_arg(),
            counts.into_tensor_arg(),
            mask.into_tensor_arg(),
            candidate_index.clone().into_tensor_arg(),
            best_candidate_position.clone().into_tensor_arg(),
            top2_gap.clone().into_tensor_arg(),
            config.batch_items as u32,
            config.candidates_per_anchor as u32,
            config.seed as u32,
            config.epsilon as f32,
        );

        (candidate_index, best_candidate_position, top2_gap)
    }
}
