// Copyright © 2025 Apple Inc.
//
// CCE (Cut Cross-Entropy) Metal Kernels
// Vocabulary-tiled implementation following Steel patterns
// Uses steel_matmul with chunked processing for best performance.

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
#include "mlx/backend/metal/kernels/bf16.h"

using namespace metal;

namespace mlx {
namespace steel {

// =============================================================================
// Helper Functions
// =============================================================================

// Fast exp with overflow protection for online softmax
template <typename T>
METAL_FUNC float safe_exp(T x) {
  float fx = float(x);
  fx = min(fx, 88.0f);  // exp(88) ~ 1.6e38, close to float max
  return fast::exp(fx);
}

// Safe exp(a - b) for online logsumexp
// Handles edge case where a = -INFINITY (padding for small vocab)
// Returns 0 when a = -INFINITY since exp(-inf) = 0
// Avoids NaN from -inf - (-inf) = indeterminate
METAL_FUNC float safe_exp_diff(float a, float b) {
  if (a <= -INFINITY) return 0.0f;
  return fast::exp(a - b);
}

// =============================================================================
// CCE Compute d_logits Kernel (for chunked backward with steel_matmul)
// Computes d_logits = (softmax - one_hot) * grad_output for a vocab chunk
// Vectorized: N_READS elements per thread for better memory coalescing
// =============================================================================

template <typename T, int N_READS = 4>
[[kernel]] void cce_compute_d_logits(
    const device T* logits [[buffer(0)]],            // [N, chunk_V]
    const device float* lse [[buffer(1)]],           // [N]
    const device int32_t* targets [[buffer(2)]],     // [N]
    const device float* grad_output [[buffer(3)]],   // [N]
    device T* d_logits [[buffer(4)]],                // [N, chunk_V] - same type as input
    constant int& N [[buffer(5)]],
    constant int& chunk_V [[buffer(6)]],
    constant int& v_start [[buffer(7)]],             // Start index in full vocab
    constant int& V [[buffer(8)]],                   // Full vocab size
    constant float& scale [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {

  const int base_idx = tid * N_READS;
  const int total_elements = N * chunk_V;

  if (base_idx >= total_elements) return;

  #pragma unroll
  for (int i = 0; i < N_READS; i++) {
    const int idx = base_idx + i;

    if (idx >= total_elements) {
      continue;
    }

    const int row = idx / chunk_V;
    const int col = idx % chunk_V;
    const int global_v = v_start + col;

    if (global_v >= V) {
      d_logits[idx] = T(0.0f);
      continue;
    }

    const float token_lse = lse[row];
    const int target = targets[row];
    const float grad_scale = grad_output[row] * scale;

    float logit = float(logits[idx]);
    float prob = safe_exp(logit - token_lse);
    prob = clamp(prob, 0.0f, 1.0f);

    // Branchless: subtract 1.0 if this is the target
    float d_logit = prob - float(global_v == target);
    d_logit *= grad_scale;

    d_logits[idx] = T(d_logit);
  }
}

// =============================================================================
// CCE Chunk LogSumExp Kernel (Single-Pass Online Algorithm)
// Updates running max and sum_exp for online logsumexp computation
// One threadgroup per row, threads cooperate on reduction
// =============================================================================

template <typename T, int N_READS = 4>
[[kernel]] void cce_chunk_logsumexp(
    const device T* logits [[buffer(0)]],       // [N, chunk_V]
    const device int32_t* targets [[buffer(1)]],// [N]
    device float* running_max [[buffer(2)]],    // [N]
    device float* running_sum_exp [[buffer(3)]],// [N]
    device float* target_logit [[buffer(4)]],   // [N]
    constant int& N [[buffer(5)]],
    constant int& chunk_V [[buffer(6)]],
    constant int& v_start [[buffer(7)]],
    constant int& V [[buffer(8)]],
    threadgroup float* smem [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {

  constexpr int THREADS_PER_TG = 256;
  constexpr int SIMD_SIZE = 32;
  constexpr int NUM_SIMDGROUPS = THREADS_PER_TG / SIMD_SIZE;

  const int row = tgid.x;
  if (row >= N) return;

  const int target = targets[row];
  const device T* row_logits = logits + row * chunk_V;

  threadgroup float* smem_max = smem;
  threadgroup float* smem_sum = smem + NUM_SIMDGROUPS;

  if (lid < NUM_SIMDGROUPS) {
    smem_max[lid] = -INFINITY;
    smem_sum[lid] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int valid_chunk_v = min(chunk_V, V - v_start);
  const int iterations = (valid_chunk_v + THREADS_PER_TG * N_READS - 1) / (THREADS_PER_TG * N_READS);

  float prevmax;
  float maxval = -INFINITY;
  float normalizer = 0.0f;
  float local_target = 0.0f;
  bool found_target = false;

  for (int r = 0; r < iterations; r++) {
    int offset = r * THREADS_PER_TG * N_READS + lid * N_READS;
    float vals[N_READS];

    if (offset + N_READS <= valid_chunk_v) {
      if constexpr (N_READS == 4) {
        vec<T, 4> packed = *reinterpret_cast<const device vec<T, 4>*>(row_logits + offset);
        vals[0] = float(packed[0]);
        vals[1] = float(packed[1]);
        vals[2] = float(packed[2]);
        vals[3] = float(packed[3]);
      } else {
        #pragma unroll
        for (int i = 0; i < N_READS; i++) {
          vals[i] = float(row_logits[offset + i]);
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < N_READS; i++) {
        vals[i] = (offset + i < valid_chunk_v) ? float(row_logits[offset + i]) : -INFINITY;
      }
    }

    #pragma unroll
    for (int i = 0; i < N_READS; i++) {
      int global_v = v_start + offset + i;
      if (global_v == target && offset + i < valid_chunk_v) {
        local_target = vals[i];
        found_target = true;
      }
    }

    prevmax = maxval;
    #pragma unroll
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < vals[i]) ? vals[i] : maxval;
    }

    normalizer *= safe_exp_diff(prevmax, maxval);

    #pragma unroll
    for (int i = 0; i < N_READS; i++) {
      normalizer += safe_exp_diff(vals[i], maxval);
    }
  }

  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= safe_exp_diff(prevmax, maxval);
  normalizer = simd_sum(normalizer);

  if (simd_lid == 0) {
    smem_max[simd_gid] = maxval;
    smem_sum[simd_gid] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float chunk_max = smem_max[0];
  float chunk_sum_exp = smem_sum[0];

  if (simd_gid == 0 && simd_lid == 0) {
    for (int i = 1; i < NUM_SIMDGROUPS; i++) {
      float sg_max = smem_max[i];
      float sg_sum = smem_sum[i];

      float new_max = max(chunk_max, sg_max);
      chunk_sum_exp = chunk_sum_exp * safe_exp_diff(chunk_max, new_max) +
                      sg_sum * safe_exp_diff(sg_max, new_max);
      chunk_max = new_max;
    }
  }

  if (lid == 0) {
    float old_max = running_max[row];
    float old_sum_exp = running_sum_exp[row];

    float new_max = max(old_max, chunk_max);
    float new_sum_exp = old_sum_exp * safe_exp_diff(old_max, new_max) +
                        chunk_sum_exp * safe_exp_diff(chunk_max, new_max);

    running_max[row] = new_max;
    running_sum_exp[row] = new_sum_exp;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float simd_target_val = simd_sum(local_target);
  bool any_found = simd_any(found_target);
  if (simd_lid == 0) {
    smem_max[simd_gid] = simd_target_val;
    smem_sum[simd_gid] = any_found ? 1.0f : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid == 0) {
    for (int i = 0; i < NUM_SIMDGROUPS; i++) {
      if (smem_sum[i] != 0.0f) {
        target_logit[row] = smem_max[i];
        break;
      }
    }
  }
}

// =============================================================================
// Helper functions for finalize kernels
// =============================================================================

METAL_FUNC float cce_compute_lse(float running_max_val, float running_sum_exp_val) {
  return running_max_val + log(running_sum_exp_val + 1e-9f);
}

METAL_FUNC float cce_compute_loss(float lse, float target_logit_val, float scale) {
  return (lse - target_logit_val) * scale;
}

// =============================================================================
// CCE Finalize LSE Kernel - Computes logsumexp from running max and sum_exp
// =============================================================================

[[host_name("cce_finalize_lse")]]
[[kernel]] void cce_finalize_lse(
    const device float* running_max [[buffer(0)]],
    const device float* running_sum_exp [[buffer(1)]],
    device float* logsumexp [[buffer(2)]],
    constant int& N [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {

  if (tid >= uint(N)) return;
  logsumexp[tid] = cce_compute_lse(running_max[tid], running_sum_exp[tid]);
}

// =============================================================================
// CCE Init Running Values Kernel - Initializes running state in one dispatch
// =============================================================================

[[host_name("cce_init_running_values")]]
[[kernel]] void cce_init_running_values(
    device float* running_max [[buffer(0)]],
    device float* running_sum_exp [[buffer(1)]],
    device float* target_logit [[buffer(2)]],
    constant int& N [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {

  if (tid >= uint(N)) return;

  running_max[tid] = -INFINITY;
  running_sum_exp[tid] = 0.0f;
  target_logit[tid] = 0.0f;
}

// =============================================================================
// CCE Finalize Loss Kernel - Computes final loss from running state
// =============================================================================

[[host_name("cce_finalize_loss")]]
[[kernel]] void cce_finalize_loss(
    const device float* running_max [[buffer(0)]],
    const device float* running_sum_exp [[buffer(1)]],
    const device float* target_logit [[buffer(2)]],
    const device int32_t* targets [[buffer(3)]],
    device float* loss [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& ignore_index [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {

  if (tid >= uint(N)) return;

  int target = targets[tid];
  if (target == ignore_index) {
    loss[tid] = 0.0f;
    return;
  }

  float lse = cce_compute_lse(running_max[tid], running_sum_exp[tid]);
  loss[tid] = cce_compute_loss(lse, target_logit[tid], scale);
}

// =============================================================================
// CCE Finalize Loss With LSE Kernel - Also outputs logsumexp for backward pass
// =============================================================================

[[host_name("cce_finalize_loss_with_lse")]]
[[kernel]] void cce_finalize_loss_with_lse(
    const device float* running_max [[buffer(0)]],
    const device float* running_sum_exp [[buffer(1)]],
    const device float* target_logit [[buffer(2)]],
    const device int32_t* targets [[buffer(3)]],
    device float* loss [[buffer(4)]],
    device float* logsumexp_out [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& ignore_index [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {

  if (tid >= uint(N)) return;

  float lse = cce_compute_lse(running_max[tid], running_sum_exp[tid]);
  logsumexp_out[tid] = lse;

  int target = targets[tid];
  if (target == ignore_index) {
    loss[tid] = 0.0f;
    return;
  }

  loss[tid] = cce_compute_loss(lse, target_logit[tid], scale);
}

} // namespace steel
} // namespace mlx
