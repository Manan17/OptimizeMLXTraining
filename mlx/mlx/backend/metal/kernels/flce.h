// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/utils.h"

// FLCE (Fused Linear Cross Entropy) reduction kernels
// These perform online softmax statistics computation over vocabulary chunks

// Fused chunk forward: compute max, sum(exp), and extract target logit
// Uses online merge algorithm for numerical stability
template <typename T, int N_READS = 4>
[[kernel]] void flce_chunk_forward(
    const device T* logits_chunk [[buffer(0)]],     // [N, chunk_size]
    const device float* prev_max [[buffer(1)]],     // [N]
    const device float* prev_sum [[buffer(2)]],     // [N]
    const device float* prev_target [[buffer(3)]],  // [N]
    const device int32_t* targets [[buffer(4)]],    // [N]
    device float* new_max [[buffer(5)]],            // [N]
    device float* new_sum [[buffer(6)]],            // [N]
    device float* new_target [[buffer(7)]],         // [N]
    constant uint& N [[buffer(8)]],
    constant uint& chunk_size [[buffer(9)]],
    constant int& v_start [[buffer(10)]],
    constant int& v_end [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

    // Each threadgroup processes one row
    uint row = gid;
    if (row >= N) return;

    constexpr int SIMD_SIZE = 32;
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_sum[SIMD_SIZE];

    // Initialize shared memory
    if (simd_group_id == 0) {
        local_max[simd_lane_id] = Limits<float>::finite_min;
        local_sum[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load previous running values
    float old_max = prev_max[row];
    float old_sum = prev_sum[row];
    float old_target = prev_target[row];
    int target = targets[row];

    // Row offset in logits_chunk
    size_t row_offset = row * size_t(chunk_size);

    // Step 1: Find chunk max using parallel reduction with N_READS
    float thread_max = Limits<float>::finite_min;
    for (uint i = lid * N_READS; i < chunk_size; i += lsize * N_READS) {
        if (i + N_READS <= chunk_size) {
            for (int r = 0; r < N_READS; r++) {
                float val = float(logits_chunk[row_offset + i + r]);
                thread_max = max(thread_max, val);
            }
        } else {
            for (int r = 0; r < N_READS && (i + r) < chunk_size; r++) {
                float val = float(logits_chunk[row_offset + i + r]);
                thread_max = max(thread_max, val);
            }
        }
    }

    // Reduce max within simdgroup
    thread_max = simd_max(thread_max);
    if (simd_lane_id == 0) {
        local_max[simd_group_id] = thread_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final max reduction
    float chunk_max;
    if (simd_group_id == 0) {
        chunk_max = simd_max(local_max[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_max[0] = chunk_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    chunk_max = local_max[0];

    // Step 2: Compute sum(exp(logits - chunk_max))
    float thread_sum = 0.0f;
    for (uint i = lid * N_READS; i < chunk_size; i += lsize * N_READS) {
        if (i + N_READS <= chunk_size) {
            for (int r = 0; r < N_READS; r++) {
                float val = float(logits_chunk[row_offset + i + r]);
                thread_sum += fast::exp(val - chunk_max);
            }
        } else {
            for (int r = 0; r < N_READS && (i + r) < chunk_size; r++) {
                float val = float(logits_chunk[row_offset + i + r]);
                thread_sum += fast::exp(val - chunk_max);
            }
        }
    }

    // Reduce sum within simdgroup
    thread_sum = simd_sum(thread_sum);
    if (simd_lane_id == 0) {
        local_sum[simd_group_id] = thread_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final sum reduction
    float chunk_sum;
    if (simd_group_id == 0) {
        chunk_sum = simd_sum(local_sum[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_sum[0] = chunk_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    chunk_sum = local_sum[0];

    // Only thread 0 writes outputs
    if (lid == 0) {
        // Online merge: new_max = max(old_max, chunk_max)
        // new_sum = old_sum * exp(old_max - new_max) + chunk_sum * exp(chunk_max - new_max)
        float merged_max = max(old_max, chunk_max);
        float merged_sum = old_sum * fast::exp(old_max - merged_max) +
                          chunk_sum * fast::exp(chunk_max - merged_max);

        new_max[row] = merged_max;
        new_sum[row] = merged_sum;

        // Extract target logit if target is in this chunk
        float merged_target = old_target;
        if (target >= v_start && target < v_end) {
            int local_idx = target - v_start;
            merged_target = float(logits_chunk[row_offset + local_idx]);
        }
        new_target[row] = merged_target;
    }
}

// Backward pass: compute grad_logits = (softmax - onehot) * scale
template <typename T, int N_READS = 4>
[[kernel]] void flce_chunk_backward(
    const device T* logits_chunk [[buffer(0)]],     // [N, chunk_size]
    const device float* logsumexp [[buffer(1)]],    // [N]
    const device int32_t* targets [[buffer(2)]],    // [N]
    device T* grad_logits [[buffer(3)]],            // [N, chunk_size]
    constant uint& N [[buffer(4)]],
    constant uint& chunk_size [[buffer(5)]],
    constant int& v_start [[buffer(6)]],
    constant int& v_end [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant int& ignore_index [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {

    uint row = gid;
    if (row >= N) return;

    int target = targets[row];
    float lse = logsumexp[row];

    // Check if target is in this chunk
    int local_target = -1;
    if (target >= v_start && target < v_end) {
        local_target = target - v_start;
    }

    // Check if this row should be ignored
    bool is_valid = (target != ignore_index);

    // Row offset
    size_t row_offset = row * size_t(chunk_size);

    // Compute softmax and gradient: grad = (softmax - onehot) * scale
    for (uint i = lid * N_READS; i < chunk_size; i += lsize * N_READS) {
        if (i + N_READS <= chunk_size) {
            for (int r = 0; r < N_READS; r++) {
                float grad = 0.0f;
                if (is_valid) {
                    float logit = float(logits_chunk[row_offset + i + r]);
                    float softmax_val = fast::exp(logit - lse);
                    float onehot = (int(i + r) == local_target) ? 1.0f : 0.0f;
                    grad = (softmax_val - onehot) * scale;
                }
                grad_logits[row_offset + i + r] = T(grad);
            }
        } else {
            for (int r = 0; r < N_READS && (i + r) < chunk_size; r++) {
                float grad = 0.0f;
                if (is_valid) {
                    float logit = float(logits_chunk[row_offset + i + r]);
                    float softmax_val = fast::exp(logit - lse);
                    float onehot = (int(i + r) == local_target) ? 1.0f : 0.0f;
                    grad = (softmax_val - onehot) * scale;
                }
                grad_logits[row_offset + i + r] = T(grad);
            }
        }
    }
}
