use super::*;

use burn::{
    backend::{Autodiff, Cuda, cuda::CudaDevice},
    tensor::{Tensor as BurnTensor, TensorData},
};

type TestBackend = Autodiff<Cuda<f32, i32>>;

#[test]
fn ranking_kernel_matches_cpu_counted_tanimoto_reference() {
    let device = CudaDevice::default();
    let rows = SparseRows {
        indices: vec![
            1, 3, 0, //
            1, 3, 0, //
            2, 0, 0, //
            1, 4, 0, //
            3, 4, 0, //
        ],
        counts: vec![
            2.0, 1.0, 0.0, //
            2.0, 1.0, 0.0, //
            5.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            1.0, 2.0, 0.0, //
        ],
        mask: vec![
            1.0, 1.0, 0.0, //
            1.0, 1.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            1.0, 1.0, 0.0, //
        ],
        row_count: 5,
        width: 3,
    };
    let config = CountedTanimotoRankingKernelConfig {
        batch_items: rows.row_count,
        candidates_per_anchor: 4,
        seed: 12_345,
        epsilon: 1.0e-8,
    };

    let (partner_a, partner_b, target_delta) = counted_tanimoto_similarity_ranking_kernel(
        BurnTensor::<TestBackend, 2, burn::tensor::Int>::from_data(
            TensorData::new(rows.indices.clone(), [rows.row_count, rows.width]),
            &device,
        ),
        BurnTensor::<TestBackend, 2>::from_data(
            TensorData::new(rows.counts.clone(), [rows.row_count, rows.width]),
            &device,
        ),
        BurnTensor::<TestBackend, 2>::from_data(
            TensorData::new(rows.mask.clone(), [rows.row_count, rows.width]),
            &device,
        ),
        config,
    );

    let partner_a = partner_a
        .into_data()
        .to_vec::<i32>()
        .expect("partner A indices should be i32");
    let partner_b = partner_b
        .into_data()
        .to_vec::<i32>()
        .expect("partner B indices should be i32");
    let target_delta = target_delta
        .into_data()
        .to_vec::<f32>()
        .expect("target deltas should be f32");

    for anchor in 0..rows.row_count {
        let (expected_a, expected_b, expected_delta) = ranking_reference(&rows, anchor, config);
        assert_eq!(partner_a[anchor] as usize, expected_a, "anchor {anchor}");
        assert_eq!(partner_b[anchor] as usize, expected_b, "anchor {anchor}");
        assert!(
            (target_delta[anchor] - expected_delta).abs() < 1.0e-5,
            "anchor {anchor}: cuda={} cpu={expected_delta}",
            target_delta[anchor],
        );
    }
}

#[derive(Clone)]
struct SparseRows {
    indices: Vec<i32>,
    counts: Vec<f32>,
    mask: Vec<f32>,
    row_count: usize,
    width: usize,
}

fn ranking_reference(
    rows: &SparseRows,
    anchor: usize,
    config: CountedTanimotoRankingKernelConfig,
) -> (usize, usize, f32) {
    let mut state = config.seed as u32 ^ (((anchor as u32) + 1) * 40503);
    if state == 0 {
        state = 0x6d2b_79f5;
    }
    let mut best_index = anchor;
    let mut worst_index = anchor;
    let mut best_score = f32::NEG_INFINITY;
    let mut worst_score = f32::INFINITY;
    let candidates = config
        .candidates_per_anchor
        .max(2)
        .min(config.batch_items.saturating_sub(1));

    for _ in 0..candidates {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let mut local_partner = (state % ((config.batch_items - 1) as u32)) as usize;
        if local_partner >= anchor {
            local_partner += 1;
        }
        let score = counted_tanimoto(rows, anchor, local_partner);
        if score > best_score {
            best_score = score;
            best_index = local_partner;
        }
        if score < worst_score {
            worst_score = score;
            worst_index = local_partner;
        }
    }

    (best_index, worst_index, (best_score - worst_score).max(0.0))
}

fn counted_tanimoto(rows: &SparseRows, left: usize, right: usize) -> f32 {
    let mut left_total = 0.0;
    let mut right_total = 0.0;
    let mut intersection = 0.0;
    for column in 0..rows.width {
        let left_offset = left * rows.width + column;
        let right_offset = right * rows.width + column;
        if rows.mask[left_offset] > 0.0 {
            left_total += rows.counts[left_offset];
        }
        if rows.mask[right_offset] > 0.0 {
            right_total += rows.counts[right_offset];
        }
    }

    let mut left_cursor = 0;
    let mut right_cursor = 0;
    while left_cursor < rows.width && right_cursor < rows.width {
        let left_offset = left * rows.width + left_cursor;
        let right_offset = right * rows.width + right_cursor;
        if rows.mask[left_offset] <= 0.0 {
            left_cursor += 1;
        } else if rows.mask[right_offset] <= 0.0 {
            right_cursor += 1;
        } else {
            let left_index = rows.indices[left_offset];
            let right_index = rows.indices[right_offset];
            if left_index == right_index {
                intersection += rows.counts[left_offset].min(rows.counts[right_offset]);
                left_cursor += 1;
                right_cursor += 1;
            } else if left_index < right_index {
                left_cursor += 1;
            } else {
                right_cursor += 1;
            }
        }
    }

    let denominator = left_total + right_total - intersection;
    if denominator <= 1.0e-8 {
        1.0
    } else {
        intersection / denominator
    }
}
