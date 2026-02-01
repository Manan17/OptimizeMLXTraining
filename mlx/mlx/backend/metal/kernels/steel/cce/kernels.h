// Copyright © 2025 Apple Inc.
//
// CCE (Cut Cross-Entropy) Metal Kernels
// Vocabulary-tiled implementation following Steel patterns
//
// Current implementation uses steel_matmul with chunked processing for best performance.
// Legacy SIMD kernels kept for small problem sizes where dispatch overhead dominates.

#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/steel/cce/params.h"

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

// =============================================================================
// CCE SIMD Backward - Legacy kernel for small problems
// Used when V <= 2000 or N < 256 where dispatch overhead dominates
// =============================================================================

template <typename T>
[[kernel]] void cce_simd_backward(
    const device T* hidden [[buffer(0)]],         // [N, H]
    const device T* weight [[buffer(1)]],         // [V, H]
    const device int32_t* targets [[buffer(2)]],  // [N]
    const device float* logsumexp [[buffer(3)]],  // [N]
    const device float* grad_output [[buffer(4)]],// [N] per-token gradients
    device float* grad_hidden [[buffer(5)]],      // [N, H]
    device float* grad_weight [[buffer(6)]],      // [V, H] - accumulated with atomics
    constant CCEParams& params [[buffer(7)]],
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
  const float upstream_grad = grad_output[row];

  const int elems_per_thread = (H + SIMD_SIZE - 1) / SIMD_SIZE;
  const int h_start = simd_lid * elems_per_thread;

  // Handle ignored tokens
  if (target == params.ignore_index || target < 0 || target >= V) {
    for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
      grad_hidden[row * H + h_start + i] = 0.0f;
    }
    return;
  }

  // Load hidden into registers
  float h_local[64];
  for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
    h_local[i] = float(hidden[row * H + h_start + i]);
  }

  // Initialize gradient accumulator
  float grad_h_local[64];
  for (int i = 0; i < elems_per_thread; i++) {
    grad_h_local[i] = 0.0f;
  }

  // Process all vocabulary entries
  for (int v = 0; v < V; v++) {
    // Compute logit
    float partial = 0.0f;
    for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
      partial += h_local[i] * float(weight[v * H + h_start + i]);
    }
    float logit = simd_sum(partial);

    // Compute gradient: (softmax - onehot) * upstream
    float prob = safe_exp(logit - lse);
    prob = clamp(prob, 0.0f, 1.0f);
    float grad_logit = prob;
    if (v == target) {
      grad_logit -= 1.0f;
    }
    grad_logit *= upstream_grad * params.scale;

    // Accumulate grad_hidden
    for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
      grad_h_local[i] += grad_logit * float(weight[v * H + h_start + i]);
    }

    // Accumulate grad_weight using atomics
    for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
      atomic_fetch_add_explicit(
          (device atomic<float>*)&grad_weight[v * H + h_start + i],
          grad_logit * h_local[i],
          memory_order_relaxed);
    }
  }

  // Write output
  for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
    grad_hidden[row * H + h_start + i] = grad_h_local[i];
  }
}

// =============================================================================
// CCE SIMD Forward - Legacy kernel for small problems
// Used when V <= 2000 or N < 256 where dispatch overhead dominates
// =============================================================================

template <typename T>
[[kernel]] void cce_simd_forward(
    const device T* hidden [[buffer(0)]],
    const device T* weight [[buffer(1)]],
    const device int32_t* targets [[buffer(2)]],
    device float* loss [[buffer(3)]],
    device float* logsumexp_out [[buffer(4)]],
    constant CCEParams& params [[buffer(5)]],
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

  const int elems_per_thread = (H + SIMD_SIZE - 1) / SIMD_SIZE;
  const int h_start = simd_lid * elems_per_thread;

  // Load hidden into registers
  float h_local[64];
  for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
    h_local[i] = float(hidden[row * H + h_start + i]);
  }

  // Online softmax state
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float target_logit = 0.0f;

  // Process vocabulary
  for (int v = 0; v < V; v++) {
    float partial = 0.0f;
    for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
      partial += h_local[i] * float(weight[v * H + h_start + i]);
    }
    float logit = simd_sum(partial);

    if (v == target) {
      target_logit = logit;
    }

    // Online softmax update
    float new_max = max(running_max, logit);
    float scale_old = safe_exp(running_max - new_max);
    float scale_new = safe_exp(logit - new_max);
    running_sum = running_sum * scale_old + scale_new;
    running_max = new_max;
  }

  // Output
  if (simd_lid == 0) {
    float lse = running_max + log(running_sum + 1e-9f);
    logsumexp_out[row] = lse;

    if (target >= 0 && target < V && target != params.ignore_index) {
      loss[row] = (lse - target_logit) * params.scale;
    } else {
      loss[row] = 0.0f;
    }
  }
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

    normalizer *= fast::exp(prevmax - maxval);

    #pragma unroll
    for (int i = 0; i < N_READS; i++) {
      normalizer += fast::exp(vals[i] - maxval);
    }
  }

  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= fast::exp(prevmax - maxval);
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
      chunk_sum_exp = chunk_sum_exp * fast::exp(chunk_max - new_max) +
                      sg_sum * fast::exp(sg_max - new_max);
      chunk_max = new_max;
    }
  }

  if (lid == 0) {
    float old_max = running_max[row];
    float old_sum_exp = running_sum_exp[row];

    float new_max = max(old_max, chunk_max);
    float new_sum_exp = old_sum_exp * fast::exp(old_max - new_max) +
                        chunk_sum_exp * fast::exp(chunk_max - new_max);

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
