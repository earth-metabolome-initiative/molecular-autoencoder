use burn_cubecl::cubecl::prelude::*;

#[allow(clippy::comparison_chain)]
#[cube(launch)]
pub(super) fn counted_tanimoto_similarity_ranking_forward<F: Float, I: Int>(
    indices: &Tensor<I>,
    counts: &Tensor<F>,
    mask: &Tensor<F>,
    candidate_index: &mut Tensor<I>,
    best_candidate_position: &mut Tensor<I>,
    top2_gap: &mut Tensor<F>,
    batch_items: u32,
    candidates_per_anchor: u32,
    seed: u32,
    epsilon: f32,
) {
    if ABSOLUTE_POS >= best_candidate_position.len() {
        terminate!();
    }

    let anchor = ABSOLUTE_POS;
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    best_candidate_position[anchor] = I::cast_from(0u32);
    top2_gap[anchor] = zero;

    if anchor >= batch_items as usize || batch_items < 3 {
        terminate!();
    }

    let candidate_count = candidates_per_anchor.max(2).min(batch_items - 1) as usize;
    let eps = F::cast_from(epsilon);
    let mut state = seed ^ (((anchor as u32) + 1u32) * 40503u32);
    if state == 0u32 {
        state = 0x6d2b_79f5u32;
    }

    let partner_slots = batch_items - 1;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let offset = state % partner_slots;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let mut stride = (state % partner_slots) + 1u32;
    let mut coprime = false;
    while !coprime {
        let mut left = stride;
        let mut right = partner_slots;
        while right != 0u32 {
            let remainder = left % right;
            left = right;
            right = remainder;
        }
        coprime = left == 1u32;
        if !coprime {
            stride += 1u32;
            if stride > partner_slots {
                stride = 1u32;
            }
        }
    }

    let mut best_score = F::new(-1.0_f32);
    let mut second_best_score = F::new(-1.0_f32);
    let mut best_position = 0usize;

    for candidate_position in 0..candidate_count {
        let mut local_partner =
            ((offset + (candidate_position as u32) * stride) % partner_slots) as usize;
        if local_partner >= anchor {
            local_partner += 1;
        }
        candidate_index
            [anchor * candidate_index.stride(0) + candidate_position * candidate_index.stride(1)] =
            I::cast_from(local_partner as u32);

        let score =
            sparse_count_tanimoto_score_rows(indices, counts, mask, anchor, local_partner, eps);
        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_position = candidate_position;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    best_candidate_position[anchor] = I::cast_from(best_position as u32);
    top2_gap[anchor] = (best_score - second_best_score).max(zero).min(one);
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
