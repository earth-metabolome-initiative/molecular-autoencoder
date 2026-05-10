use super::api::{CountedTanimotoKernelBackend, CountedTanimotoRankingKernelConfig};
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
impl<B> CountedTanimotoKernelBackend for Fusion<B>
where
    B: FusionBackend + CountedTanimotoKernelBackend + Send + Sync,
{
    fn counted_tanimoto_similarity_ranking_kernel(
        indices: IntTensor<Self>,
        counts: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        config: CountedTanimotoRankingKernelConfig,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
        let [index_rows, index_width] = indices.shape.dims();
        let [count_rows, count_width] = counts.shape.dims();
        let [mask_rows, mask_width] = mask.shape.dims();
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

        let streams = OperationStreams::with_inputs([&indices, &counts, &mask]);
        let client = counts.client.clone();
        let index_shape = Shape::new([config.batch_items]);
        let delta_shape = Shape::new([config.batch_items]);
        let partner_a = TensorIr::uninit(
            client.create_empty_handle(),
            index_shape.clone(),
            B::IntElem::dtype(),
        );
        let partner_b = TensorIr::uninit(
            client.create_empty_handle(),
            index_shape,
            B::IntElem::dtype(),
        );
        let target_delta =
            TensorIr::uninit(client.create_empty_handle(), delta_shape, counts.dtype);
        let desc = CustomOpIr::new(
            "counted_tanimoto_similarity_ranking_forward",
            &[indices.into_ir(), counts.into_ir(), mask.into_ir()],
            &[partner_a, partner_b, target_delta],
        );

        let mut outputs = client.register(
            streams,
            OperationIr::Custom(desc.clone()),
            CountedTanimotoRankingFusionForward::<B> {
                desc,
                config,
                backend: PhantomData,
            },
        );
        let target_delta = outputs
            .pop()
            .expect("Tanimoto geometry custom op has delta output");
        let partner_b = outputs
            .pop()
            .expect("Tanimoto geometry custom op has second partner output");
        let partner_a = outputs
            .pop()
            .expect("Tanimoto geometry custom op has first partner output");

        (partner_a, partner_b, target_delta)
    }
}

#[cfg(feature = "cuda-fusion")]
#[derive(Debug)]
struct CountedTanimotoRankingFusionForward<B: FusionBackend> {
    desc: CustomOpIr,
    config: CountedTanimotoRankingKernelConfig,
    backend: PhantomData<B>,
}

#[cfg(feature = "cuda-fusion")]
impl<B> Operation<B::FusionRuntime> for CountedTanimotoRankingFusionForward<B>
where
    B: FusionBackend + CountedTanimotoKernelBackend + Send + Sync,
{
    fn execute(&self, handles: &mut burn_ir::HandleContainer<B::Handle>) {
        let (inputs, outputs) = self.desc.as_fixed::<3, 3>();
        let (partner_a, partner_b, target_delta) = B::counted_tanimoto_similarity_ranking_kernel(
            handles.get_int_tensor::<B>(&inputs[0]),
            handles.get_float_tensor::<B>(&inputs[1]),
            handles.get_float_tensor::<B>(&inputs[2]),
            self.config,
        );

        handles.register_int_tensor::<B>(&outputs[0].id, partner_a);
        handles.register_int_tensor::<B>(&outputs[1].id, partner_b);
        handles.register_float_tensor::<B>(&outputs[2].id, target_delta);
    }
}
