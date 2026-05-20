//! Pure-Rust counted-Tanimoto ranking algorithm.
//!
//! Mirrors `counted_tanimoto_similarity_ranking_forward` (the CUDA / CubeCL
//! kernel under `kernels.rs`) one anchor at a time. Two consumers:
//!
//! 1. The test suite uses it as the reference oracle against which the
//!    `CubeBackend` impl is validated (`tests.rs::ranking_kernel_*`).
//! 2. The `NdArray` impl of [`super::api::CountedTanimotoKernelBackend`]
//!    runs it for real — there is no GPU dispatch on the CPU backend, so
//!    the trait just calls into this module.
//!
//! Both call sites share the same code so they cannot drift.
//!
//! All indices, counts, and masks are dense host buffers laid out
//! `[row_count, width]` row-major, the same shape the kernel reads on the
//! GPU.
//!
//! The per-anchor seed schedule, partner stride, and best/second-best
//! tracking are kept bit-identical with the kernel; the only difference is
//! the host-side `xorshift32` state, which the kernel implements with the
//! same constants.

use super::api::CountedTanimotoRankingKernelConfig;

/// Dense view of a sparse counted-fingerprint batch suitable for the CPU
/// ranking helper. Storage is row-major: row `r` occupies
/// `[r * width, (r + 1) * width)` in each of `indices`, `counts`, `mask`.
pub struct SparseRows<'a> {
    pub indices: &'a [i32],
    pub counts: &'a [f32],
    pub mask: &'a [f32],
    pub width: usize,
}

/// Per-anchor result: the sampled candidate row indices, the position of
/// the best-scoring candidate within that list, and the top-1/top-2 score
/// gap clamped to `[0, 1]`.
pub struct RankingReference {
    pub candidate_indices: Vec<i32>,
    pub best_candidate_position: usize,
    pub top2_gap: f32,
}

/// Computes the per-anchor ranking labels.
///
/// `anchor` is the row index in `[0, rows.row_count)`. Returns
/// [`config.effective_candidates_per_anchor()`] unique candidate row
/// indices, the position of the row whose counted Tanimoto score with
/// `anchor` is the highest, and the score gap to the runner-up.
pub fn ranking_reference(
    rows: &SparseRows<'_>,
    anchor: usize,
    config: CountedTanimotoRankingKernelConfig,
) -> RankingReference {
    let mut state = config.seed() as u32 ^ (((anchor as u32) + 1) * 40_503);
    if state == 0 {
        state = 0x6d2b_79f5;
    }
    let partner_slots = (config.batch_items() - 1) as u32;
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
        candidate_indices.push(i32::try_from(local_partner).expect("partner index fits in i32"));
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
        top2_gap: (best_score - second_best_score).clamp(0.0, 1.0),
    }
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn counted_tanimoto(rows: &SparseRows<'_>, left: usize, right: usize) -> f32 {
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
