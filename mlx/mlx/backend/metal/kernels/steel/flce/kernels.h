// Copyright © 2025 Apple Inc.
//
// FLCE (Fused Linear Cross Entropy) Metal Kernels
// SIMD-optimized: 32 threads collaborate on each row

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/steel/flce/params.h"

using namespace metal;

namespace mlx {
namespace steel {

// =============================================================================
// SIMD Forward Kernel
//
// Each simdgroup (32 threads) handles one row.
// Each thread holds H/32 elements of hidden and computes partial dot products.
// simd_sum() reduces across all 32 threads to get full dot product.
// =============================================================================

template <typename T>
[[kernel]] void flce_simd_forward(
    const device T* hidden [[buffer(0)]],        // [N, H]
    const device T* weight [[buffer(1)]],        // [V, H]
    const device int32_t* targets [[buffer(2)]], // [N]
    device float* loss [[buffer(3)]],            // [1]
    device float* logsumexp_out [[buffer(4)]],   // [N]
    constant FLCEParams& params [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    const int N = params.N;
    const int H = params.H;
    const int V = params.V;

    // Each simdgroup processes one row
    // With 4 simdgroups per threadgroup, we process 4 rows per threadgroup
    constexpr int SIMD_SIZE = 32;
    constexpr int ROWS_PER_TG = 4;

    const int row = tid.x * ROWS_PER_TG + simd_gid;
    if (row >= N) return;

    const int target = targets[row];

    // Calculate how many elements each thread handles
    const int elems_per_thread = (H + SIMD_SIZE - 1) / SIMD_SIZE;
    const int h_start = simd_lid * elems_per_thread;

    // Load this thread's portion of hidden[row] into registers
    float h_local[64];  // Max elements per thread (supports H up to 2048)
    for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
        h_local[i] = float(hidden[row * H + h_start + i]);
    }

    // Online softmax state
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float target_logit = 0.0f;

    // Process each vocabulary position
    for (int v = 0; v < V; v++) {
        // Compute partial dot product (this thread's contribution)
        float partial = 0.0f;
        for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
            partial += h_local[i] * float(weight[v * H + h_start + i]);
        }

        // Reduce across simdgroup to get full dot product
        float logit = simd_sum(partial);

        // Track target logit (all threads have same value after simd_sum)
        if (v == target) {
            target_logit = logit;
        }

        // Online softmax update
        float new_max = max(running_max, logit);
        float scale_old = exp(running_max - new_max);
        float scale_new = exp(logit - new_max);
        running_sum = running_sum * scale_old + scale_new;
        running_max = new_max;
    }

    // Only thread 0 of each simdgroup writes output
    if (simd_lid == 0) {
        float lse = running_max + log(running_sum + 1e-9f);
        logsumexp_out[row] = lse;

        if (target != params.ignore_index) {
            float row_loss = (lse - target_logit) * params.scale;
            atomic_fetch_add_explicit(
                (device atomic<float>*)loss,
                row_loss,
                memory_order_relaxed);
        }
    }
}

// =============================================================================
// SIMD Backward Kernel
//
// Each simdgroup (32 threads) handles one row.
// Each thread computes its portion of grad_hidden.
// No atomics needed - each thread writes to different positions.
// =============================================================================

template <typename T>
[[kernel]] void flce_simd_backward(
    const device T* hidden [[buffer(0)]],        // [N, H]
    const device T* weight [[buffer(1)]],        // [V, H]
    const device int32_t* targets [[buffer(2)]], // [N]
    const device float* logsumexp [[buffer(3)]], // [N]
    const device float* grad_out [[buffer(4)]],  // [1]
    device float* grad_hidden [[buffer(5)]],     // [N, H]
    constant FLCEParams& params [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    const int N = params.N;
    const int H = params.H;
    const int V = params.V;

    constexpr int SIMD_SIZE = 32;
    constexpr int ROWS_PER_TG = 4;

    const int row = tid.x * ROWS_PER_TG + simd_gid;
    if (row >= N) return;

    const int target = targets[row];
    const float lse = logsumexp[row];
    const float upstream_grad = grad_out[0];

    // Calculate thread's portion
    const int elems_per_thread = (H + SIMD_SIZE - 1) / SIMD_SIZE;
    const int h_start = simd_lid * elems_per_thread;

    // Handle ignored targets - zero out this thread's portion
    if (target == params.ignore_index) {
        for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
            grad_hidden[row * H + h_start + i] = 0.0f;
        }
        return;
    }

    // Load this thread's portion of hidden[row]
    float h_local[64];
    for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
        h_local[i] = float(hidden[row * H + h_start + i]);
    }

    // Initialize gradient accumulator for this thread's portion
    float grad_h_local[64];
    for (int i = 0; i < elems_per_thread; i++) {
        grad_h_local[i] = 0.0f;
    }

    // Process each vocabulary position
    for (int v = 0; v < V; v++) {
        // Recompute logit via simd reduction
        float partial = 0.0f;
        for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
            partial += h_local[i] * float(weight[v * H + h_start + i]);
        }
        float logit = simd_sum(partial);

        // Compute gradient: (softmax - onehot) * upstream * scale
        float prob = exp(logit - lse);
        float grad_logit = prob;
        if (v == target) {
            grad_logit -= 1.0f;
        }
        grad_logit *= upstream_grad * params.scale;

        // Accumulate this thread's portion of grad_hidden
        // grad_hidden[h] += grad_logit * weight[v, h]
        for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
            grad_h_local[i] += grad_logit * float(weight[v * H + h_start + i]);
        }
    }

    // Write this thread's portion of grad_hidden (no atomics needed!)
    for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
        grad_hidden[row * H + h_start + i] = grad_h_local[i];
    }
}

} // namespace steel
} // namespace mlx
