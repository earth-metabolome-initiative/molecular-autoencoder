use burn_cubecl::cubecl::prelude::*;

#[allow(clippy::comparison_chain)]
#[cube(launch)]
pub(super) fn counted_tanimoto_similarity_ranking_forward<F: Float, I: Int>(
    indices: &Tensor<I>,
    counts: &Tensor<F>,
    mask: &Tensor<F>,
    partner_a: &mut Tensor<I>,
    partner_b: &mut Tensor<I>,
    target_delta: &mut Tensor<F>,
    batch_items: u32,
    candidates_per_anchor: u32,
    seed: u32,
    epsilon: f32,
) {
    if ABSOLUTE_POS >= partner_a.len() {
        terminate!();
    }

    let anchor = ABSOLUTE_POS;
    let zero = F::new(0.0_f32);
    partner_a[anchor] = I::cast_from(0u32);
    partner_b[anchor] = I::cast_from(0u32);
    target_delta[anchor] = zero;

    if anchor >= batch_items as usize || batch_items < 3 {
        terminate!();
    }

    let candidate_count = candidates_per_anchor.max(2).min(batch_items - 1);
    let eps = F::cast_from(epsilon);
    let mut state = seed ^ (((anchor as u32) + 1u32) * 40503u32);
    if state == 0u32 {
        state = 0x6d2b_79f5u32;
    }

    let mut best_local = anchor;
    let mut worst_local = anchor;
    let mut best_score = F::new(-1.0_f32);
    let mut worst_score = F::new(2.0_f32);

    for _candidate in 0..candidate_count {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let mut local_partner = (state % (batch_items - 1)) as usize;
        if local_partner >= anchor {
            local_partner += 1;
        }

        let score =
            sparse_count_tanimoto_score_rows(indices, counts, mask, anchor, local_partner, eps);
        if score > best_score {
            best_score = score;
            best_local = local_partner;
        }
        if score < worst_score {
            worst_score = score;
            worst_local = local_partner;
        }
    }

    partner_a[anchor] = I::cast_from(best_local as u32);
    partner_b[anchor] = I::cast_from(worst_local as u32);
    target_delta[anchor] = (best_score - worst_score).max(zero);
}

#[cube]
fn sparse_count_tanimoto_score_rows<F: Float, I: Int>(
    indices: &Tensor<I>,
    counts: &Tensor<F>,
    mask: &Tensor<F>,
    left_row: usize,
    right_row: usize,
    eps: F,
) -> F {
    let width = indices.shape(1);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);

    let mut left_total = zero;
    let mut right_total = zero;
    for column in 0..width {
        let left_offset = left_row * counts.stride(0) + column * counts.stride(1);
        let right_offset = right_row * counts.stride(0) + column * counts.stride(1);
        let left_active = mask[left_row * mask.stride(0) + column * mask.stride(1)] > zero;
        let right_active = mask[right_row * mask.stride(0) + column * mask.stride(1)] > zero;
        if left_active {
            left_total += counts[left_offset].max(zero);
        }
        if right_active {
            right_total += counts[right_offset].max(zero);
        }
    }

    let mut left_cursor = 0usize;
    let mut right_cursor = 0usize;
    let mut intersection = zero;
    while left_cursor < width && right_cursor < width {
        let left_mask = mask[left_row * mask.stride(0) + left_cursor * mask.stride(1)];
        let right_mask = mask[right_row * mask.stride(0) + right_cursor * mask.stride(1)];
        if left_mask <= zero {
            left_cursor += 1;
        } else if right_mask <= zero {
            right_cursor += 1;
        } else {
            let left_index =
                indices[left_row * indices.stride(0) + left_cursor * indices.stride(1)];
            let right_index =
                indices[right_row * indices.stride(0) + right_cursor * indices.stride(1)];
            if left_index == right_index {
                let left_count =
                    counts[left_row * counts.stride(0) + left_cursor * counts.stride(1)].max(zero);
                let right_count = counts
                    [right_row * counts.stride(0) + right_cursor * counts.stride(1)]
                .max(zero);
                intersection += left_count.min(right_count);
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
    if denominator <= eps {
        one
    } else {
        (intersection / denominator).max(zero).min(one)
    }
}
