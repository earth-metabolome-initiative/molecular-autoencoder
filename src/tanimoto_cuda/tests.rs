use super::cpu::{SparseRows, ranking_reference};
use super::*;

use burn::{
    backend::Autodiff,
    tensor::{Tensor as BurnTensor, TensorData},
};

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

type TestBackend = Autodiff<TestInner<f32, i32>>;

#[test]
#[allow(
    clippy::default_constructed_unit_structs,
    reason = "TestDevice is a unit struct under cubecl-cpu and a tuple struct under cuda; \
              the cross-feature alias uses Default::default to keep one expression."
)]
fn ranking_kernel_matches_cpu_counted_tanimoto_reference() {
    let device = TestDevice::default();
    let indices_data: Vec<i32> = vec![
        1, 3, 0, //
        1, 3, 0, //
        2, 0, 0, //
        1, 4, 0, //
        3, 4, 0, //
    ];
    let counts_data: Vec<f32> = vec![
        2.0, 1.0, 0.0, //
        2.0, 1.0, 0.0, //
        5.0, 0.0, 0.0, //
        1.0, 1.0, 0.0, //
        1.0, 2.0, 0.0, //
    ];
    let mask_data: Vec<f32> = vec![
        1.0, 1.0, 0.0, //
        1.0, 1.0, 0.0, //
        1.0, 0.0, 0.0, //
        1.0, 1.0, 0.0, //
        1.0, 1.0, 0.0, //
    ];
    let row_count = 5;
    let width = 3;
    let rows = SparseRows {
        indices: &indices_data,
        counts: &counts_data,
        mask: &mask_data,
        width,
    };
    let config = CountedTanimotoRankingKernelConfig::builder()
        .batch_items(row_count)
        .candidates_per_anchor(4)
        .seed(12_345)
        .epsilon(1.0e-8)
        .build()
        .expect("kernel config is valid");

    let (candidate_index, best_candidate_position, top2_gap) =
        counted_tanimoto_similarity_ranking_kernel(
            BurnTensor::<TestBackend, 2, burn::tensor::Int>::from_data(
                TensorData::new(indices_data.clone(), [row_count, width]),
                &device,
            ),
            BurnTensor::<TestBackend, 2>::from_data(
                TensorData::new(counts_data.clone(), [row_count, width]),
                &device,
            ),
            BurnTensor::<TestBackend, 2>::from_data(
                TensorData::new(mask_data.clone(), [row_count, width]),
                &device,
            ),
            config,
        );

    let candidate_index = candidate_index
        .into_data()
        .to_vec::<i32>()
        .expect("candidate indices should be i32");
    let best_candidate_position = best_candidate_position
        .into_data()
        .to_vec::<i32>()
        .expect("best candidate positions should be i32");
    let top2_gap = top2_gap
        .into_data()
        .to_vec::<f32>()
        .expect("top-2 gaps should be f32");

    let candidate_count = config.effective_candidates_per_anchor();
    for anchor in 0..row_count {
        let expected = ranking_reference(&rows, anchor, config);
        let start = anchor * candidate_count;
        let actual_candidates = &candidate_index[start..start + candidate_count];
        assert_eq!(
            actual_candidates,
            expected.candidate_indices.as_slice(),
            "anchor {anchor}"
        );
        assert_eq!(
            usize::try_from(best_candidate_position[anchor])
                .expect("best candidate position is non-negative"),
            expected.best_candidate_position,
            "anchor {anchor}"
        );
        assert!(
            (top2_gap[anchor] - expected.top2_gap).abs() < 1.0e-5,
            "anchor {anchor}: cuda={} cpu={}",
            top2_gap[anchor],
            expected.top2_gap,
        );
        let anchor_i32 = i32::try_from(anchor).expect("anchor index fits in i32");
        assert!(!actual_candidates.contains(&anchor_i32));
        for (left, left_value) in actual_candidates.iter().enumerate() {
            for right_value in actual_candidates.iter().skip(left + 1) {
                assert_ne!(
                    left_value, right_value,
                    "anchor {anchor}: candidates must be unique"
                );
            }
        }
    }
}
