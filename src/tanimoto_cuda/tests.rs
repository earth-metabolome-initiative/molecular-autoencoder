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

    let (candidate_index, best_candidate_position, top2_gap) =
        counted_tanimoto_similarity_ranking_kernel(
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
    for anchor in 0..rows.row_count {
        let expected = ranking_reference(&rows, anchor, config);
        let start = anchor * candidate_count;
        let actual_candidates = &candidate_index[start..start + candidate_count];
        assert_eq!(
            actual_candidates,
            expected.candidate_indices.as_slice(),
            "anchor {anchor}"
        );
        assert_eq!(
            best_candidate_position[anchor] as usize, expected.best_candidate_position,
            "anchor {anchor}"
        );
        assert!(
            (top2_gap[anchor] - expected.top2_gap).abs() < 1.0e-5,
            "anchor {anchor}: cuda={} cpu={}",
            top2_gap[anchor],
            expected.top2_gap,
        );
        assert!(!actual_candidates.contains(&(anchor as i32)));
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
) -> RankingReference {
    let mut state = config.seed as u32 ^ (((anchor as u32) + 1) * 40503);
    if state == 0 {
        state = 0x6d2b_79f5;
    }
    let partner_slots = (config.batch_items - 1) as u32;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let offset = state % partner_slots;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let mut stride = (state % partner_slots) + 1;
    while gcd(stride, partner_slots) != 1 {
        stride += 1;
        if stride > partner_slots {
            stride = 1;
        }
    }
    let mut candidate_indices = Vec::new();
    let mut best_candidate_position = 0;
    let mut best_score = f32::NEG_INFINITY;
    let mut second_best_score = f32::NEG_INFINITY;
    let candidates = config.effective_candidates_per_anchor();

    for candidate_position in 0..candidates {
        let mut local_partner =
            ((offset + (candidate_position as u32) * stride) % partner_slots) as usize;
        if local_partner >= anchor {
            local_partner += 1;
        }
        candidate_indices.push(local_partner as i32);
        let score = counted_tanimoto(rows, anchor, local_partner);
        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_candidate_position = candidate_position;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    RankingReference {
        candidate_indices,
        best_candidate_position,
        top2_gap: (best_score - second_best_score).max(0.0).min(1.0),
    }
}

struct RankingReference {
    candidate_indices: Vec<i32>,
    best_candidate_position: usize,
    top2_gap: f32,
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
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
