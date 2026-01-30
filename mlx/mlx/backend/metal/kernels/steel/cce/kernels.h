// Copyright © 2025 Apple Inc.
//
// CCE (Cut Cross-Entropy) Metal Kernels
// Vocabulary-tiled implementation following Steel patterns
//
// Key optimizations:
// 1. Tiles over vocabulary dimension (not tokens)
// 2. Uses threadgroup memory for weight tiles
// 3. Online logsumexp computation
// 4. Sparsity exploitation in backward pass

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

// Fast exp using exp2 for better precision in online softmax
template <typename T>
METAL_FUNC float safe_exp(T x) {
  // Clamp to avoid overflow
  float fx = float(x);
  fx = min(fx, 88.0f);  // exp(88) ~ 1.6e38, close to float max
  return fast::exp(fx);
}

// Online logsumexp update
// Given current (max, sum) and new value x:
// new_max = max(max, x)
// new_sum = sum * exp(max - new_max) + exp(x - new_max)
struct OnlineSoftmax {
  float running_max;
  float running_sum;

  METAL_FUNC OnlineSoftmax() : running_max(-INFINITY), running_sum(0.0f) {}

  METAL_FUNC void update(float x) {
    float new_max = max(running_max, x);
    float scale_old = safe_exp(running_max - new_max);
    float scale_new = safe_exp(x - new_max);
    running_sum = running_sum * scale_old + scale_new;
    running_max = new_max;
  }

  METAL_FUNC void merge(float other_max, float other_sum) {
    if (other_sum <= 0.0f) return;
    float new_max = max(running_max, other_max);
    float scale_self = safe_exp(running_max - new_max);
    float scale_other = safe_exp(other_max - new_max);
    running_sum = running_sum * scale_self + other_sum * scale_other;
    running_max = new_max;
  }

  METAL_FUNC float logsumexp() const {
    return running_max + log(running_sum + 1e-9f);
  }
};

// =============================================================================
// CCE Forward Kernel
//
// Strategy:
// - Each threadgroup processes TOKENS_PER_TG tokens
// - Each simdgroup handles one token
// - All threads cooperatively load vocabulary tiles into threadgroup memory
// - Each thread computes partial dot products, simd_sum reduces
// - Online logsumexp tracks max/sum across vocabulary
// =============================================================================

template <typename T, int VOCAB_TILE = 512, int HIDDEN_TILE = 64>
[[kernel]] void cce_forward(
    const device T* hidden [[buffer(0)]],         // [N, H]
    const device T* weight [[buffer(1)]],         // [V, H]
    const device int32_t* targets [[buffer(2)]],  // [N]
    device float* loss [[buffer(3)]],             // [N]
    device float* logsumexp_out [[buffer(4)]],    // [N]
    constant CCEParams& params [[buffer(5)]],
    threadgroup T* shared_mem [[threadgroup(0)]], // For weight tiles
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint thread_idx [[thread_position_in_threadgroup]]) {

  const int N = params.N;
  const int H = params.H;
  const int V = params.V;

  constexpr int SIMD_SIZE = 32;
  constexpr int TOKENS_PER_TG = 4;
  constexpr int THREADS_PER_TG = SIMD_SIZE * TOKENS_PER_TG;

  // This token's index
  const int token_idx = tid.x * TOKENS_PER_TG + simd_gid;
  if (token_idx >= N) return;

  const int target = targets[token_idx];

  // Partition hidden dimension across simd lanes
  // Each thread handles H/32 elements
  const int h_per_thread = (H + SIMD_SIZE - 1) / SIMD_SIZE;
  const int h_start = simd_lid * h_per_thread;
  const int h_end = min(h_start + h_per_thread, H);

  // Load this token's hidden state into registers (one-time cost)
  float h_local[128];  // Supports H up to 4096 (128 * 32)
  for (int i = 0; i < h_per_thread && h_start + i < H; i++) {
    h_local[i] = float(hidden[token_idx * H + h_start + i]);
  }

  // Online softmax state
  OnlineSoftmax softmax_state;
  float target_logit = 0.0f;

  // Process all vocabulary items for this token
  for (int v = 0; v < V; v++) {
    // Compute partial dot product
    float partial_dot = 0.0f;
    for (int i = 0; i < h_per_thread && h_start + i < H; i++) {
      partial_dot += h_local[i] * float(weight[v * H + h_start + i]);
    }

    // Reduce across simdgroup
    float logit = simd_sum(partial_dot);

    // Update online softmax
    softmax_state.update(logit);

    // Track target
    if (v == target) {
      target_logit = logit;
    }
  }

  // Only lane 0 writes output
  if (simd_lid == 0) {
    float lse = softmax_state.logsumexp();
    logsumexp_out[token_idx] = lse;

    if (target >= 0 && target < V && target != params.ignore_index) {
      loss[token_idx] = (lse - target_logit) * params.scale;
    } else {
      loss[token_idx] = 0.0f;
    }
  }
}

// =============================================================================
// CCE Forward Kernel - Optimized with SIMD parallelism
//
// Strategy: Each threadgroup processes ONE token
// Each simdgroup handles different vocab entries in parallel
// No cross-simdgroup barriers during main loop (much faster!)
// =============================================================================

template <typename T, int BV = 256, int BH = 64>
[[kernel]] void cce_forward_tiled(
    const device T* hidden [[buffer(0)]],         // [N, H]
    const device T* weight [[buffer(1)]],         // [V, H]
    const device int32_t* targets [[buffer(2)]],  // [N]
    device float* loss [[buffer(3)]],             // [N]
    device float* logsumexp_out [[buffer(4)]],    // [N]
    constant CCEParams& params [[buffer(5)]],
    threadgroup float* smem [[threadgroup(0)]],   // Shared memory
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  const int N = params.N;
  const int H = params.H;
  const int V = params.V;

  constexpr int SIMD_SIZE = 32;
  constexpr int NUM_SIMDGROUPS = 4;

  // Each threadgroup processes one token
  const int token_idx = tid.x;
  if (token_idx >= N) return;

  const int target = targets[token_idx];

  // Each thread handles H/32 hidden elements (within simdgroup)
  const int h_per_thread = (H + SIMD_SIZE - 1) / SIMD_SIZE;
  const int h_start = simd_lid * h_per_thread;

  // Load this thread's portion of hidden into registers
  float h_local[64];  // Supports H up to 2048
  for (int i = 0; i < h_per_thread && h_start + i < H; i++) {
    h_local[i] = float(hidden[token_idx * H + h_start + i]);
  }

  // Per-simdgroup online softmax state
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float target_logit = 0.0f;
  bool found_target = false;

  // Each simdgroup processes vocab entries: simd_gid, simd_gid + NUM_SIMDGROUPS, ...
  // This gives us 4x parallelism with no barriers!
  for (int v = simd_gid; v < V; v += NUM_SIMDGROUPS) {
    // Each thread computes partial dot product for its H portion
    float partial = 0.0f;
    for (int i = 0; i < h_per_thread && h_start + i < H; i++) {
      partial += h_local[i] * float(weight[v * H + h_start + i]);
    }

    // Reduce within simdgroup to get full logit
    float logit = simd_sum(partial);

    // Only lane 0 updates softmax state
    if (simd_lid == 0) {
      // Online softmax update
      float new_max = max(running_max, logit);
      float scale_old = safe_exp(running_max - new_max);
      float scale_new = safe_exp(logit - new_max);
      running_sum = running_sum * scale_old + scale_new;
      running_max = new_max;

      // Track target logit
      if (v == target) {
        target_logit = logit;
        found_target = true;
      }
    }
  }

  // Reduce softmax states across simdgroups
  // Use shared memory for final reduction
  // Shared memory: [0-3] = max, [4-7] = sum, [8-11] = target_logit, [12-15] = found_target
  threadgroup float* tg_max = smem;
  threadgroup float* tg_sum = smem + NUM_SIMDGROUPS;
  threadgroup float* tg_target_logits = smem + NUM_SIMDGROUPS * 2;
  threadgroup int* tg_found = (threadgroup int*)(smem + NUM_SIMDGROUPS * 3);

  // Initialize shared memory
  if (simd_lid == 0) {
    tg_max[simd_gid] = running_max;
    tg_sum[simd_gid] = running_sum;
    tg_target_logits[simd_gid] = target_logit;
    tg_found[simd_gid] = found_target ? 1 : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Thread 0 does final reduction
  if (simd_gid == 0 && simd_lid == 0) {
    float final_max = tg_max[0];
    float final_sum = tg_sum[0];

    for (int i = 1; i < NUM_SIMDGROUPS; i++) {
      float other_max = tg_max[i];
      float other_sum = tg_sum[i];

      float new_max = max(final_max, other_max);
      final_sum = final_sum * safe_exp(final_max - new_max) +
                  other_sum * safe_exp(other_max - new_max);
      final_max = new_max;
    }

    // Find target logit
    float final_target_logit = 0.0f;
    bool any_found = false;
    for (int i = 0; i < NUM_SIMDGROUPS; i++) {
      if (tg_found[i]) {
        final_target_logit = tg_target_logits[i];
        any_found = true;
        break;
      }
    }

    float lse = final_max + log(final_sum + 1e-9f);
    logsumexp_out[token_idx] = lse;

    if (target >= 0 && target < V && target != params.ignore_index && any_found) {
      loss[token_idx] = (lse - final_target_logit) * params.scale;
    } else {
      loss[token_idx] = 0.0f;
    }
  }
}

// =============================================================================
// CCE Backward Kernel with Sparsity
//
// Key optimization: Skip vocabulary tiles where max(softmax) < threshold
// For typical LLMs, this skips ~99.98% of computation
//
// Strategy: Each thread handles a subset of hidden dimensions
// and accumulates gradients in registers, avoiding threadgroup atomics
// =============================================================================

template <typename T, int BV = 256, int BH = 64>
[[kernel]] void cce_backward(
    const device T* hidden [[buffer(0)]],         // [N, H]
    const device T* weight [[buffer(1)]],         // [V, H]
    const device int32_t* targets [[buffer(2)]],  // [N]
    const device float* logsumexp [[buffer(3)]],  // [N]
    const device float* grad_output [[buffer(4)]],// [N] or [1]
    device float* grad_hidden [[buffer(5)]],      // [N, H]
    device float* grad_weight [[buffer(6)]],      // [V, H] (optional, can be nullptr)
    constant CCEParams& params [[buffer(7)]],
    threadgroup float* smem [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  const uint thread_idx = lid.x;

  const int N = params.N;
  const int H = params.H;
  const int V = params.V;

  constexpr int SIMD_SIZE = 32;
  constexpr int NUM_SIMDGROUPS = 4;
  constexpr int THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS;
  constexpr float SPARSITY_THRESHOLD = CCE_SPARSITY_THRESHOLD;

  const int token_idx = tid.x;
  if (token_idx >= N) return;

  const int target = targets[token_idx];
  const float lse = logsumexp[token_idx];
  const float grad_out = grad_output[token_idx] * params.scale;

  // Each thread handles a portion of the hidden dimension
  const int h_per_thread = (H + THREADS_PER_TG - 1) / THREADS_PER_TG;
  const int h_start = thread_idx * h_per_thread;
  const int h_end = min(h_start + h_per_thread, H);

  // Handle ignored tokens
  if (target == params.ignore_index || target < 0 || target >= V) {
    for (int h = h_start; h < h_end; h++) {
      grad_hidden[token_idx * H + h] = 0.0f;
    }
    return;
  }

  // Load hidden values into registers for this thread's portion
  float h_local[32];  // Max h_per_thread elements
  for (int i = 0; i < h_per_thread && h_start + i < H; i++) {
    h_local[i] = float(hidden[token_idx * H + h_start + i]);
  }

  // Per-thread gradient accumulator for hidden
  float grad_h_local[32];
  for (int i = 0; i < h_per_thread; i++) {
    grad_h_local[i] = 0.0f;
  }

  const int num_v_tiles = (V + BV - 1) / BV;

  // Process vocabulary in tiles
  for (int vt = 0; vt < num_v_tiles; vt++) {
    const int v_start = vt * BV;
    const int v_end = min(v_start + BV, V);
    const int v_tile_size = v_end - v_start;

    // SPARSITY CHECK DISABLED FOR DEBUGGING
    // The coarse sampling (every 16th entry) can cause numerical issues
    // TODO: Re-enable with denser sampling once NaN issue is resolved

    // Process this tile - all threads collaborate on each vocab entry
    // Each thread handles h_per_thread elements of H
    for (int local_v = 0; local_v < v_tile_size; local_v++) {
      int global_v = v_start + local_v;
      if (global_v >= V) continue;

      // Compute logit: each thread computes partial, then reduce across all threads
      float partial = 0.0f;
      for (int i = 0; i < h_per_thread && h_start + i < H; i++) {
        partial += h_local[i] * float(weight[global_v * H + h_start + i]);
      }

      // Sum partials across simd lanes first
      float simd_partial = simd_sum(partial);

      // Reduce across simdgroups
      if (simd_lid == 0) {
        smem[simd_gid] = simd_partial;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float logit = 0.0f;
      if (thread_idx == 0) {
        for (int sg = 0; sg < NUM_SIMDGROUPS; sg++) {
          logit += smem[sg];
        }
        smem[0] = logit;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      logit = smem[0];

      // Compute gradient: (softmax - onehot) * upstream
      float prob = safe_exp(logit - lse);
      float grad_logit = prob;
      if (global_v == target) {
        grad_logit -= 1.0f;
      }
      grad_logit *= grad_out;

      // Fine-grained sparsity check disabled for debugging
      // if (abs(grad_logit) < 1e-8f && global_v != target) {
      //   continue;
      // }

      // Accumulate grad_hidden: grad_h += grad_logit * weight[v]
      // Each thread handles its portion of H
      for (int i = 0; i < h_per_thread && h_start + i < H; i++) {
        float w_val = float(weight[global_v * H + h_start + i]);
        grad_h_local[i] += grad_logit * w_val;
      }

      // Accumulate grad_weight if needed: grad_w[v] += grad_logit * hidden
      if (grad_weight != nullptr) {
        for (int i = 0; i < h_per_thread && h_start + i < H; i++) {
          atomic_fetch_add_explicit(
              (device atomic<float>*)&grad_weight[global_v * H + h_start + i],
              grad_logit * h_local[i],
              memory_order_relaxed);
        }
      }
    }
  }

  // Write grad_hidden from registers to global memory
  for (int i = 0; i < h_per_thread && h_start + i < H; i++) {
    grad_hidden[token_idx * H + h_start + i] = grad_h_local[i];
  }
}

// =============================================================================
// Simplified CCE Backward - SIMD-based like original but with sparsity
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
  constexpr float SPARSITY_THRESHOLD = CCE_SPARSITY_THRESHOLD;

  const int row = tid.x * ROWS_PER_TG + simd_gid;
  if (row >= N) return;

  const int target = targets[row];
  const float lse = logsumexp[row];
  const float upstream_grad = grad_output[row];  // Per-token gradient

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

  // Process ALL vocabulary entries (sparsity disabled for debugging)
  for (int v = 0; v < V; v++) {
    // Compute logit
    float partial = 0.0f;
    for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
      partial += h_local[i] * float(weight[v * H + h_start + i]);
    }
    float logit = simd_sum(partial);

    // Compute gradient: (softmax - onehot) * upstream
    float prob = safe_exp(logit - lse);
    prob = clamp(prob, 0.0f, 1.0f);  // Probabilities must be in [0, 1]
    float grad_logit = prob;
    if (v == target) {
      grad_logit -= 1.0f;
    }
    grad_logit *= upstream_grad * params.scale;

    // Accumulate grad_hidden
    for (int i = 0; i < elems_per_thread && h_start + i < H; i++) {
      grad_h_local[i] += grad_logit * float(weight[v * H + h_start + i]);
    }

    // Accumulate grad_weight using atomics (multiple tokens contribute)
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
// Simple CCE Forward - Direct port with online softmax
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
// Optimized CCE Forward Kernel - 2D Tiled with Lock-based LSE
//
// Following Apple's ml-cross-entropy approach:
// - 2D grid: (B_tiles, V_tiles) for massive parallelism
// - Each threadgroup handles BLOCK_B tokens × BLOCK_V vocab entries
// - Lock-based atomic LSE combination across vocab tiles
// - Tiled matrix multiply for efficient computation
// =============================================================================

// Lock-free atomic logaddexp using compare-and-swap
// Computes log(exp(old) + exp(new)) atomically
METAL_FUNC void atomic_logaddexp(
    device atomic<uint>* lse_ptr_uint,  // Reinterpret float* as uint* for CAS
    float new_lse) {

  uint old_bits, new_bits;
  float old_lse, combined;

  // Compare-and-swap loop
  do {
    old_bits = atomic_load_explicit(lse_ptr_uint, memory_order_relaxed);
    old_lse = as_type<float>(old_bits);

    // Compute logaddexp: log(exp(old) + exp(new))
    if (isinf(old_lse) && old_lse < 0) {
      // old_lse is -inf, just use new_lse
      combined = new_lse;
    } else {
      float max_val = max(old_lse, new_lse);
      combined = max_val + log(exp(old_lse - max_val) + exp(new_lse - max_val));
    }

    new_bits = as_type<uint>(combined);
  } while (!atomic_compare_exchange_weak_explicit(
      lse_ptr_uint, &old_bits, new_bits,
      memory_order_relaxed, memory_order_relaxed));
}

template <typename T, int BLOCK_B = 32, int BLOCK_V = 128, int BLOCK_D = 32>
[[kernel]] void cce_forward_2d(
    const device T* hidden [[buffer(0)]],            // [N, H]
    const device T* weight [[buffer(1)]],            // [V, H]
    const device int32_t* targets [[buffer(2)]],     // [N]
    device float* lse_out [[buffer(3)]],             // [N] - logsumexp (init to -inf)
    device float* neg_target_logit [[buffer(4)]],    // [N] - negative target logit (init to 0)
    constant CCEParams& params [[buffer(5)]],
    threadgroup float* smem [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],      // (b_tile, v_tile, 0)
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  const int N = params.N;
  const int H = params.H;
  const int V = params.V;

  constexpr int SIMD_SIZE = 32;
  constexpr int NUM_SIMDGROUPS = 4;
  constexpr int THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS;  // 128 threads

  const uint thread_idx = lid.x;

  // This threadgroup handles tokens [b_start, b_end) and vocab [v_start, v_end)
  const int b_tile = tid.x;
  const int v_tile = tid.y;

  const int b_start = b_tile * BLOCK_B;
  const int b_end = min(b_start + BLOCK_B, N);
  const int v_start = v_tile * BLOCK_V;
  const int v_end = min(v_start + BLOCK_V, V);

  if (b_start >= N || v_start >= V) return;

  const int b_size = b_end - b_start;
  const int v_size = v_end - v_start;

  // Shared memory layout:
  // [0, BLOCK_B * BLOCK_D): Hidden tile
  // [BLOCK_B * BLOCK_D, BLOCK_B * BLOCK_D + BLOCK_V * BLOCK_D): Weight tile
  // [after tiles, ...): Logits accumulator [BLOCK_B, BLOCK_V]
  threadgroup float* H_smem = smem;
  threadgroup float* W_smem = smem + BLOCK_B * BLOCK_D;
  threadgroup float* logits_smem = smem + BLOCK_B * BLOCK_D + BLOCK_V * BLOCK_D;

  // Initialize logits to 0
  for (int i = thread_idx; i < BLOCK_B * BLOCK_V; i += THREADS_PER_TG) {
    logits_smem[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Tile over hidden dimension
  const int num_d_tiles = (H + BLOCK_D - 1) / BLOCK_D;

  for (int d_tile = 0; d_tile < num_d_tiles; d_tile++) {
    const int d_start = d_tile * BLOCK_D;
    const int d_end = min(d_start + BLOCK_D, H);
    const int d_size = d_end - d_start;

    // Cooperative load: Hidden tile [BLOCK_B, BLOCK_D]
    for (int i = thread_idx; i < BLOCK_B * BLOCK_D; i += THREADS_PER_TG) {
      int local_b = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_b = b_start + local_b;
      int global_d = d_start + local_d;

      if (global_b < N && global_d < H) {
        H_smem[local_b * BLOCK_D + local_d] = float(hidden[global_b * H + global_d]);
      } else {
        H_smem[local_b * BLOCK_D + local_d] = 0.0f;
      }
    }

    // Cooperative load: Weight tile [BLOCK_V, BLOCK_D] stored as [BLOCK_D, BLOCK_V] for better access
    for (int i = thread_idx; i < BLOCK_V * BLOCK_D; i += THREADS_PER_TG) {
      int local_d = i / BLOCK_V;
      int local_v = i % BLOCK_V;
      int global_v = v_start + local_v;
      int global_d = d_start + local_d;

      if (global_v < V && global_d < H) {
        W_smem[local_d * BLOCK_V + local_v] = float(weight[global_v * H + global_d]);
      } else {
        W_smem[local_d * BLOCK_V + local_v] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Matrix multiply: logits += H_tile @ W_tile^T
    // Each thread handles some elements of the output
    // logits[b, v] += sum_d(H[b, d] * W[v, d])
    for (int i = thread_idx; i < BLOCK_B * BLOCK_V; i += THREADS_PER_TG) {
      int local_b = i / BLOCK_V;
      int local_v = i % BLOCK_V;

      float dot = 0.0f;
      for (int local_d = 0; local_d < d_size; local_d++) {
        dot += H_smem[local_b * BLOCK_D + local_d] * W_smem[local_d * BLOCK_V + local_v];
      }
      logits_smem[local_b * BLOCK_V + local_v] += dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Now logits_smem[b, v] contains the full logits for this tile
  // Compute partial LSE for each token and combine atomically

  // Each simdgroup handles a subset of tokens
  for (int local_b = simd_gid; local_b < b_size; local_b += NUM_SIMDGROUPS) {
    int global_b = b_start + local_b;
    if (global_b >= N) continue;

    int target = targets[global_b];

    // Compute max and sum for this token's vocab tile
    // Distribute across simd lanes
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    float target_logit_val = 0.0f;
    bool found_target = false;

    for (int local_v = simd_lid; local_v < v_size; local_v += SIMD_SIZE) {
      int global_v = v_start + local_v;
      float logit = logits_smem[local_b * BLOCK_V + local_v];

      // Update online softmax
      float new_max = max(thread_max, logit);
      thread_sum = thread_sum * safe_exp(thread_max - new_max) + safe_exp(logit - new_max);
      thread_max = new_max;

      // Check for target
      if (global_v == target) {
        target_logit_val = logit;
        found_target = true;
      }
    }

    // Reduce across simd lanes
    float simd_max_val = simd_max(thread_max);
    float scaled_sum = thread_sum * safe_exp(thread_max - simd_max_val);
    float simd_sum_result = simd_sum(scaled_sum);

    // Reduce target logit (only one lane has it)
    float final_target_logit = simd_sum(found_target ? target_logit_val : 0.0f);
    int any_found_target = simd_or(int(found_target));

    // Lane 0 does the atomic update
    if (simd_lid == 0) {
      float tile_lse = simd_max_val + log(simd_sum_result + 1e-9f);

      // Atomic logaddexp update using CAS (lock-free)
      device atomic<uint>* lse_ptr_uint = (device atomic<uint>*)(lse_out + global_b);
      atomic_logaddexp(lse_ptr_uint, tile_lse);

      // Store negative target logit if found in this tile
      // Use atomic store to ensure visibility (only one tile should find the target)
      if (any_found_target != 0) {
        atomic_store_explicit(
            (device atomic<float>*)&neg_target_logit[global_b],
            -final_target_logit,
            memory_order_relaxed);
      }
    }
  }
}

// =============================================================================
// Optimized CCE Backward Kernel - 2D Tiled with Sparsity Filtering
// =============================================================================

template <typename T, int BLOCK_B = 32, int BLOCK_V = 128, int BLOCK_D = 32>
[[kernel]] void cce_backward_2d(
    const device T* hidden [[buffer(0)]],            // [N, H]
    const device T* weight [[buffer(1)]],            // [V, H]
    const device int32_t* targets [[buffer(2)]],     // [N]
    const device float* lse [[buffer(3)]],           // [N]
    const device float* grad_output [[buffer(4)]],   // [N]
    device float* grad_hidden [[buffer(5)]],         // [N, H] - accumulated with atomics
    device float* grad_weight [[buffer(6)]],         // [V, H] - accumulated with atomics
    constant CCEParams& params [[buffer(7)]],        // CHANGED: was buffer(9)
    threadgroup float* smem [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  const int N = params.N;
  const int H = params.H;
  const int V = params.V;

  constexpr int SIMD_SIZE = 32;
  constexpr int NUM_SIMDGROUPS = 4;
  constexpr int THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS;
  constexpr float FILTER_EPS = 1e-6f;  // Sparsity threshold

  const uint thread_idx = lid.x;

  const int b_tile = tid.x;
  const int v_tile = tid.y;

  const int b_start = b_tile * BLOCK_B;
  const int b_end = min(b_start + BLOCK_B, N);
  const int v_start = v_tile * BLOCK_V;
  const int v_end = min(v_start + BLOCK_V, V);

  if (b_start >= N || v_start >= V) return;

  const int b_size = b_end - b_start;
  const int v_size = v_end - v_start;

  // Shared memory layout
  threadgroup float* H_smem = smem;
  threadgroup float* W_smem = smem + BLOCK_B * BLOCK_D;
  threadgroup float* logits_smem = smem + BLOCK_B * BLOCK_D + BLOCK_V * BLOCK_D;
  threadgroup float* d_logits_smem = smem + BLOCK_B * BLOCK_D + BLOCK_V * BLOCK_D + BLOCK_B * BLOCK_V;

  // Initialize logits to 0
  for (int i = thread_idx; i < BLOCK_B * BLOCK_V; i += THREADS_PER_TG) {
    logits_smem[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Step 1: Recompute logits (same as forward)
  const int num_d_tiles = (H + BLOCK_D - 1) / BLOCK_D;

  for (int d_tile = 0; d_tile < num_d_tiles; d_tile++) {
    const int d_start = d_tile * BLOCK_D;
    const int d_end = min(d_start + BLOCK_D, H);
    const int d_size = d_end - d_start;

    // Load hidden tile
    for (int i = thread_idx; i < BLOCK_B * BLOCK_D; i += THREADS_PER_TG) {
      int local_b = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_b = b_start + local_b;
      int global_d = d_start + local_d;

      if (global_b < N && global_d < H) {
        H_smem[local_b * BLOCK_D + local_d] = float(hidden[global_b * H + global_d]);
      } else {
        H_smem[local_b * BLOCK_D + local_d] = 0.0f;
      }
    }

    // Load weight tile
    for (int i = thread_idx; i < BLOCK_V * BLOCK_D; i += THREADS_PER_TG) {
      int local_d = i / BLOCK_V;
      int local_v = i % BLOCK_V;
      int global_v = v_start + local_v;
      int global_d = d_start + local_d;

      if (global_v < V && global_d < H) {
        W_smem[local_d * BLOCK_V + local_v] = float(weight[global_v * H + global_d]);
      } else {
        W_smem[local_d * BLOCK_V + local_v] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute logits
    for (int i = thread_idx; i < BLOCK_B * BLOCK_V; i += THREADS_PER_TG) {
      int local_b = i / BLOCK_V;
      int local_v = i % BLOCK_V;

      float dot = 0.0f;
      for (int local_d = 0; local_d < d_size; local_d++) {
        dot += H_smem[local_b * BLOCK_D + local_d] * W_smem[local_d * BLOCK_V + local_v];
      }
      logits_smem[local_b * BLOCK_V + local_v] += dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Step 2: Compute d_logits = softmax(logits) - one_hot(target)
  // d_logits[b, v] = exp(logits[b, v] - lse[b]) - (v == target[b] ? 1 : 0)
  for (int i = thread_idx; i < BLOCK_B * BLOCK_V; i += THREADS_PER_TG) {
    int local_b = i / BLOCK_V;
    int local_v = i % BLOCK_V;
    int global_b = b_start + local_b;
    int global_v = v_start + local_v;

    if (global_b < N && global_v < V) {
      float logit = logits_smem[local_b * BLOCK_V + local_v];
      float token_lse = lse[global_b];
      float prob = safe_exp(logit - token_lse);
      prob = clamp(prob, 0.0f, 1.0f);  // Prevent NaN with float16 inputs

      int target = targets[global_b];
      float d_logit = prob;
      if (global_v == target) {
        d_logit -= 1.0f;
      }

      // Scale by grad_output
      d_logit *= grad_output[global_b] * params.scale;
      d_logits_smem[local_b * BLOCK_V + local_v] = d_logit;
    } else {
      d_logits_smem[local_b * BLOCK_V + local_v] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Sparsity check: skip if all d_logits are below threshold
  float max_abs_d_logit = 0.0f;
  for (int i = thread_idx; i < BLOCK_B * BLOCK_V; i += THREADS_PER_TG) {
    max_abs_d_logit = max(max_abs_d_logit, abs(d_logits_smem[i]));
  }
  max_abs_d_logit = simd_max(max_abs_d_logit);

  // Reduce across simdgroups
  threadgroup float tg_max_abs[NUM_SIMDGROUPS];
  if (simd_lid == 0) {
    tg_max_abs[simd_gid] = max_abs_d_logit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (thread_idx == 0) {
    float final_max = 0.0f;
    for (int i = 0; i < NUM_SIMDGROUPS; i++) {
      final_max = max(final_max, tg_max_abs[i]);
    }
    tg_max_abs[0] = final_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Skip if all gradients are negligible
  if (tg_max_abs[0] < FILTER_EPS) {
    return;
  }

  // Step 3: Compute gradients
  // grad_hidden[b] += d_logits[b] @ weight
  // grad_weight[v] += d_logits[:, v]^T @ hidden

  for (int d_tile = 0; d_tile < num_d_tiles; d_tile++) {
    const int d_start = d_tile * BLOCK_D;
    const int d_end = min(d_start + BLOCK_D, H);
    const int d_size = d_end - d_start;

    // Reload weight tile for this d_tile
    for (int i = thread_idx; i < BLOCK_V * BLOCK_D; i += THREADS_PER_TG) {
      int local_d = i / BLOCK_V;
      int local_v = i % BLOCK_V;
      int global_v = v_start + local_v;
      int global_d = d_start + local_d;

      if (global_v < V && global_d < H) {
        W_smem[local_d * BLOCK_V + local_v] = float(weight[global_v * H + global_d]);
      } else {
        W_smem[local_d * BLOCK_V + local_v] = 0.0f;
      }
    }

    // Reload hidden tile
    for (int i = thread_idx; i < BLOCK_B * BLOCK_D; i += THREADS_PER_TG) {
      int local_b = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_b = b_start + local_b;
      int global_d = d_start + local_d;

      if (global_b < N && global_d < H) {
        H_smem[local_b * BLOCK_D + local_d] = float(hidden[global_b * H + global_d]);
      } else {
        H_smem[local_b * BLOCK_D + local_d] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute grad_hidden contribution: d_logits @ W
    // grad_hidden[b, d] += sum_v(d_logits[b, v] * W[v, d])
    for (int i = thread_idx; i < BLOCK_B * BLOCK_D; i += THREADS_PER_TG) {
      int local_b = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_b = b_start + local_b;
      int global_d = d_start + local_d;

      if (global_b < N && global_d < H) {
        float grad_h = 0.0f;
        for (int local_v = 0; local_v < v_size; local_v++) {
          grad_h += d_logits_smem[local_b * BLOCK_V + local_v] * W_smem[local_d * BLOCK_V + local_v];
        }

        // Atomic add to grad_hidden
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_hidden[global_b * H + global_d],
            grad_h,
            memory_order_relaxed);
      }
    }

    // Compute grad_weight contribution: d_logits^T @ H
    // grad_weight[v, d] += sum_b(d_logits[b, v] * H[b, d])
    for (int i = thread_idx; i < BLOCK_V * BLOCK_D; i += THREADS_PER_TG) {
      int local_v = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_v = v_start + local_v;
      int global_d = d_start + local_d;

      if (global_v < V && global_d < H) {
        float grad_w = 0.0f;
        for (int local_b = 0; local_b < b_size; local_b++) {
          grad_w += d_logits_smem[local_b * BLOCK_V + local_v] * H_smem[local_b * BLOCK_D + local_d];
        }

        // Atomic add to grad_weight
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_weight[global_v * H + global_d],
            grad_w,
            memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// =============================================================================
// CCE Forward Kernel with simdgroup_matrix (MMA) optimization
//
// Uses Apple's hardware-accelerated simdgroup_matrix for 10-16x faster matmul
// Each simdgroup computes 8x8 output tiles using simdgroup_multiply_accumulate
// =============================================================================

template <typename T, int BLOCK_B = 32, int BLOCK_V = 64, int BLOCK_D = 32>
[[kernel]] void cce_forward_mma(
    const device T* hidden [[buffer(0)]],            // [N, H]
    const device T* weight [[buffer(1)]],            // [V, H]
    const device int32_t* targets [[buffer(2)]],     // [N]
    device float* lse_out [[buffer(3)]],             // [N] - logsumexp (init to -inf)
    device float* neg_target_logit [[buffer(4)]],    // [N] - negative target logit (init to 0)
    constant CCEParams& params [[buffer(5)]],
    threadgroup float* smem [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],      // (b_tile, v_tile, 0)
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  const int N = params.N;
  const int H = params.H;
  const int V = params.V;

  constexpr int SIMD_SIZE = 32;
  constexpr int NUM_SIMDGROUPS = 4;
  constexpr int THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS;  // 128 threads
  constexpr int FRAG_SIZE = 8;  // simdgroup_matrix is 8x8

  // Number of 8x8 output tiles: BLOCK_B/8 x BLOCK_V/8
  constexpr int TILES_B = BLOCK_B / FRAG_SIZE;  // 4 for BLOCK_B=32
  constexpr int TILES_V = BLOCK_V / FRAG_SIZE;  // 8 for BLOCK_V=64
  constexpr int TILES_D = BLOCK_D / FRAG_SIZE;  // 4 for BLOCK_D=32

  const uint thread_idx = lid.x;

  // This threadgroup handles tokens [b_start, b_end) and vocab [v_start, v_end)
  const int b_tile_idx = tid.x;
  const int v_tile_idx = tid.y;

  const int b_start = b_tile_idx * BLOCK_B;
  const int b_end = min(b_start + BLOCK_B, N);
  const int v_start = v_tile_idx * BLOCK_V;
  const int v_end = min(v_start + BLOCK_V, V);

  if (b_start >= N || v_start >= V) return;

  const int b_size = b_end - b_start;
  const int v_size = v_end - v_start;

  // Shared memory layout (all float for accumulation precision):
  // [0, BLOCK_B * BLOCK_D): Hidden tile [BLOCK_B, BLOCK_D] row-major
  // [BLOCK_B * BLOCK_D, +BLOCK_D * BLOCK_V): Weight tile [BLOCK_D, BLOCK_V] row-major
  // [after tiles, +BLOCK_B * BLOCK_V): Logits accumulator [BLOCK_B, BLOCK_V]
  threadgroup float* H_smem = smem;  // [BLOCK_B, BLOCK_D]
  threadgroup float* W_smem = smem + BLOCK_B * BLOCK_D;  // [BLOCK_D, BLOCK_V]
  threadgroup float* logits_smem = smem + BLOCK_B * BLOCK_D + BLOCK_D * BLOCK_V;

  // Initialize logits to 0
  for (int i = thread_idx; i < BLOCK_B * BLOCK_V; i += THREADS_PER_TG) {
    logits_smem[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Each simdgroup handles a subset of output tiles
  // Total output tiles: TILES_B * TILES_V = 4*16 = 64
  // With 4 simdgroups, each handles 16 tiles
  constexpr int TOTAL_OUTPUT_TILES = TILES_B * TILES_V;
  constexpr int TILES_PER_SIMDGROUP = (TOTAL_OUTPUT_TILES + NUM_SIMDGROUPS - 1) / NUM_SIMDGROUPS;

  // Accumulator registers for this simdgroup's tiles
  simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE> C_frag[TILES_PER_SIMDGROUP];

  // Initialize accumulators to zero
  for (int t = 0; t < TILES_PER_SIMDGROUP; t++) {
    C_frag[t] = simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE>(0.0f);
  }

  // Tile over hidden dimension
  const int num_d_tiles = (H + BLOCK_D - 1) / BLOCK_D;

  for (int d_tile = 0; d_tile < num_d_tiles; d_tile++) {
    const int d_start = d_tile * BLOCK_D;
    const int d_end = min(d_start + BLOCK_D, H);
    const int d_size = d_end - d_start;

    // Cooperative load: Hidden tile [BLOCK_B, BLOCK_D]
    for (int i = thread_idx; i < BLOCK_B * BLOCK_D; i += THREADS_PER_TG) {
      int local_b = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_b = b_start + local_b;
      int global_d = d_start + local_d;

      if (global_b < N && global_d < H) {
        H_smem[local_b * BLOCK_D + local_d] = float(hidden[global_b * H + global_d]);
      } else {
        H_smem[local_b * BLOCK_D + local_d] = 0.0f;
      }
    }

    // Cooperative load: Weight tile [BLOCK_D, BLOCK_V]
    // Weight matrix is [V, H], we need W^T which is [H, V]
    // So we load weight[v, d] into W_smem[d, v]
    for (int i = thread_idx; i < BLOCK_D * BLOCK_V; i += THREADS_PER_TG) {
      int local_d = i / BLOCK_V;
      int local_v = i % BLOCK_V;
      int global_v = v_start + local_v;
      int global_d = d_start + local_d;

      if (global_v < V && global_d < H) {
        W_smem[local_d * BLOCK_V + local_v] = float(weight[global_v * H + global_d]);
      } else {
        W_smem[local_d * BLOCK_V + local_v] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // MMA: logits += H @ W^T using simdgroup_matrix
    // Each simdgroup processes its assigned output tiles
    for (int tile_idx = simd_gid; tile_idx < TOTAL_OUTPUT_TILES; tile_idx += NUM_SIMDGROUPS) {
      int tb = tile_idx / TILES_V;  // Which 8-row block
      int tv = tile_idx % TILES_V;  // Which 8-col block
      int local_tile = tile_idx / NUM_SIMDGROUPS;  // Index into our accumulator array

      // For each k tile (along hidden dimension)
      for (int tk = 0; tk < TILES_D && (tk * FRAG_SIZE) < d_size; tk++) {
        // Load A fragment: H_smem[tb*8:(tb+1)*8, tk*8:(tk+1)*8]
        simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE> A_frag;
        simdgroup_load(A_frag, H_smem + tb * FRAG_SIZE * BLOCK_D + tk * FRAG_SIZE, BLOCK_D);

        // Load B fragment: W_smem[tk*8:(tk+1)*8, tv*8:(tv+1)*8]
        simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE> B_frag;
        simdgroup_load(B_frag, W_smem + tk * FRAG_SIZE * BLOCK_V + tv * FRAG_SIZE, BLOCK_V);

        // Accumulate: C += A @ B
        simdgroup_multiply_accumulate(C_frag[local_tile], A_frag, B_frag, C_frag[local_tile]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Store accumulated results to shared memory
  for (int tile_idx = simd_gid; tile_idx < TOTAL_OUTPUT_TILES; tile_idx += NUM_SIMDGROUPS) {
    int tb = tile_idx / TILES_V;
    int tv = tile_idx % TILES_V;
    int local_tile = tile_idx / NUM_SIMDGROUPS;

    // Store to logits_smem[tb*8:(tb+1)*8, tv*8:(tv+1)*8]
    simdgroup_store(C_frag[local_tile], logits_smem + tb * FRAG_SIZE * BLOCK_V + tv * FRAG_SIZE, BLOCK_V);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Now logits_smem contains the full logits for this tile
  // Compute partial LSE for each token and combine atomically
  // (Same as cce_forward_2d from here)

  // Each simdgroup handles a subset of tokens
  for (int local_b = simd_gid; local_b < b_size; local_b += NUM_SIMDGROUPS) {
    int global_b = b_start + local_b;
    if (global_b >= N) continue;

    int target = targets[global_b];

    // Compute max and sum for this token's vocab tile
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    float target_logit_val = 0.0f;
    bool found_target = false;

    for (int local_v = simd_lid; local_v < v_size; local_v += SIMD_SIZE) {
      int global_v = v_start + local_v;
      float logit = logits_smem[local_b * BLOCK_V + local_v];

      // Update online softmax
      float new_max = max(thread_max, logit);
      thread_sum = thread_sum * safe_exp(thread_max - new_max) + safe_exp(logit - new_max);
      thread_max = new_max;

      // Check for target
      if (global_v == target) {
        target_logit_val = logit;
        found_target = true;
      }
    }

    // Reduce across simd lanes
    float simd_max_val = simd_max(thread_max);
    float scaled_sum = thread_sum * safe_exp(thread_max - simd_max_val);
    float simd_sum_result = simd_sum(scaled_sum);

    // Reduce target logit (only one lane has it)
    float final_target_logit = simd_sum(found_target ? target_logit_val : 0.0f);
    int any_found_target = simd_or(int(found_target));

    // Lane 0 does the atomic update
    if (simd_lid == 0) {
      float tile_lse = simd_max_val + log(simd_sum_result + 1e-9f);

      // Atomic logaddexp update using CAS (lock-free)
      device atomic<uint>* lse_ptr_uint = (device atomic<uint>*)(lse_out + global_b);
      atomic_logaddexp(lse_ptr_uint, tile_lse);

      // Store negative target logit if found in this tile
      if (any_found_target != 0) {
        atomic_store_explicit(
            (device atomic<float>*)&neg_target_logit[global_b],
            -final_target_logit,
            memory_order_relaxed);
      }
    }
  }
}

// =============================================================================
// CCE Backward Kernel with simdgroup_matrix (MMA) optimization
// =============================================================================

template <typename T, int BLOCK_B = 32, int BLOCK_V = 64, int BLOCK_D = 32>
[[kernel]] void cce_backward_mma(
    const device T* hidden [[buffer(0)]],            // [N, H]
    const device T* weight [[buffer(1)]],            // [V, H]
    const device int32_t* targets [[buffer(2)]],     // [N]
    const device float* lse [[buffer(3)]],           // [N]
    const device float* grad_output [[buffer(4)]],   // [N]
    device float* grad_hidden [[buffer(5)]],         // [N, H] - accumulated with atomics
    device float* grad_weight [[buffer(6)]],         // [V, H] - accumulated with atomics
    constant CCEParams& params [[buffer(7)]],
    threadgroup float* smem [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  const int N = params.N;
  const int H = params.H;
  const int V = params.V;

  constexpr int SIMD_SIZE = 32;
  constexpr int NUM_SIMDGROUPS = 4;
  constexpr int THREADS_PER_TG = SIMD_SIZE * NUM_SIMDGROUPS;
  constexpr int FRAG_SIZE = 8;
  constexpr float FILTER_EPS = 1e-6f;

  constexpr int TILES_B = BLOCK_B / FRAG_SIZE;
  constexpr int TILES_V = BLOCK_V / FRAG_SIZE;  // 8 for BLOCK_V=64
  constexpr int TILES_D = BLOCK_D / FRAG_SIZE;
  constexpr int TOTAL_OUTPUT_TILES = TILES_B * TILES_V;
  constexpr int TILES_PER_SIMDGROUP = (TOTAL_OUTPUT_TILES + NUM_SIMDGROUPS - 1) / NUM_SIMDGROUPS;

  const uint thread_idx = lid.x;

  const int b_tile_idx = tid.x;
  const int v_tile_idx = tid.y;

  const int b_start = b_tile_idx * BLOCK_B;
  const int b_end = min(b_start + BLOCK_B, N);
  const int v_start = v_tile_idx * BLOCK_V;
  const int v_end = min(v_start + BLOCK_V, V);

  if (b_start >= N || v_start >= V) return;

  const int b_size = b_end - b_start;
  const int v_size = v_end - v_start;

  // Shared memory layout
  threadgroup float* H_smem = smem;  // [BLOCK_B, BLOCK_D]
  threadgroup float* W_smem = smem + BLOCK_B * BLOCK_D;  // [BLOCK_D, BLOCK_V]
  threadgroup float* logits_smem = smem + BLOCK_B * BLOCK_D + BLOCK_D * BLOCK_V;  // [BLOCK_B, BLOCK_V]
  threadgroup float* d_logits_smem = smem + BLOCK_B * BLOCK_D + BLOCK_D * BLOCK_V + BLOCK_B * BLOCK_V;  // [BLOCK_B, BLOCK_V]

  // Initialize logits to 0
  for (int i = thread_idx; i < BLOCK_B * BLOCK_V; i += THREADS_PER_TG) {
    logits_smem[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulators for logits computation
  simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE> C_frag[TILES_PER_SIMDGROUP];
  for (int t = 0; t < TILES_PER_SIMDGROUP; t++) {
    C_frag[t] = simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE>(0.0f);
  }

  // Step 1: Recompute logits using MMA
  const int num_d_tiles = (H + BLOCK_D - 1) / BLOCK_D;

  for (int d_tile = 0; d_tile < num_d_tiles; d_tile++) {
    const int d_start = d_tile * BLOCK_D;
    const int d_end = min(d_start + BLOCK_D, H);
    const int d_size = d_end - d_start;

    // Load hidden tile
    for (int i = thread_idx; i < BLOCK_B * BLOCK_D; i += THREADS_PER_TG) {
      int local_b = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_b = b_start + local_b;
      int global_d = d_start + local_d;

      if (global_b < N && global_d < H) {
        H_smem[local_b * BLOCK_D + local_d] = float(hidden[global_b * H + global_d]);
      } else {
        H_smem[local_b * BLOCK_D + local_d] = 0.0f;
      }
    }

    // Load weight tile
    for (int i = thread_idx; i < BLOCK_D * BLOCK_V; i += THREADS_PER_TG) {
      int local_d = i / BLOCK_V;
      int local_v = i % BLOCK_V;
      int global_v = v_start + local_v;
      int global_d = d_start + local_d;

      if (global_v < V && global_d < H) {
        W_smem[local_d * BLOCK_V + local_v] = float(weight[global_v * H + global_d]);
      } else {
        W_smem[local_d * BLOCK_V + local_v] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // MMA for logits (removed redundant check)
    for (int tile_idx = simd_gid; tile_idx < TOTAL_OUTPUT_TILES; tile_idx += NUM_SIMDGROUPS) {
      int tb = tile_idx / TILES_V;
      int tv = tile_idx % TILES_V;
      int local_tile = tile_idx / NUM_SIMDGROUPS;

      for (int tk = 0; tk < TILES_D && (tk * FRAG_SIZE) < d_size; tk++) {
        simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE> A_frag;
        simdgroup_load(A_frag, H_smem + tb * FRAG_SIZE * BLOCK_D + tk * FRAG_SIZE, BLOCK_D);

        simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE> B_frag;
        simdgroup_load(B_frag, W_smem + tk * FRAG_SIZE * BLOCK_V + tv * FRAG_SIZE, BLOCK_V);

        simdgroup_multiply_accumulate(C_frag[local_tile], A_frag, B_frag, C_frag[local_tile]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Store logits (removed redundant check)
  for (int tile_idx = simd_gid; tile_idx < TOTAL_OUTPUT_TILES; tile_idx += NUM_SIMDGROUPS) {
    int tb = tile_idx / TILES_V;
    int tv = tile_idx % TILES_V;
    int local_tile = tile_idx / NUM_SIMDGROUPS;

    simdgroup_store(C_frag[local_tile], logits_smem + tb * FRAG_SIZE * BLOCK_V + tv * FRAG_SIZE, BLOCK_V);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Step 2: Compute d_logits = softmax(logits) - one_hot(target)
  for (int i = thread_idx; i < BLOCK_B * BLOCK_V; i += THREADS_PER_TG) {
    int local_b = i / BLOCK_V;
    int local_v = i % BLOCK_V;
    int global_b = b_start + local_b;
    int global_v = v_start + local_v;

    if (global_b < N && global_v < V) {
      float logit = logits_smem[local_b * BLOCK_V + local_v];
      float token_lse = lse[global_b];
      float prob = safe_exp(logit - token_lse);
      prob = clamp(prob, 0.0f, 1.0f);

      int target = targets[global_b];
      float d_logit = prob;
      if (global_v == target) {
        d_logit -= 1.0f;
      }

      d_logit *= grad_output[global_b] * params.scale;
      d_logits_smem[local_b * BLOCK_V + local_v] = d_logit;
    } else {
      d_logits_smem[local_b * BLOCK_V + local_v] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Sparsity check
  float max_abs_d_logit = 0.0f;
  for (int i = thread_idx; i < BLOCK_B * BLOCK_V; i += THREADS_PER_TG) {
    max_abs_d_logit = max(max_abs_d_logit, abs(d_logits_smem[i]));
  }
  max_abs_d_logit = simd_max(max_abs_d_logit);

  threadgroup float tg_max_abs[NUM_SIMDGROUPS];
  if (simd_lid == 0) {
    tg_max_abs[simd_gid] = max_abs_d_logit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (thread_idx == 0) {
    float final_max = 0.0f;
    for (int i = 0; i < NUM_SIMDGROUPS; i++) {
      final_max = max(final_max, tg_max_abs[i]);
    }
    tg_max_abs[0] = final_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tg_max_abs[0] < FILTER_EPS) {
    return;
  }

  // Step 3: Compute gradients
  // grad_hidden[b, d] += sum_v(d_logits[b, v] * W[v, d])  -> d_logits @ W
  // grad_weight[v, d] += sum_b(d_logits[b, v] * H[b, d])  -> d_logits_T @ H

  for (int d_tile = 0; d_tile < num_d_tiles; d_tile++) {
    const int d_start = d_tile * BLOCK_D;
    const int d_end = min(d_start + BLOCK_D, H);
    const int d_size = d_end - d_start;

    // Reload weight tile for grad_hidden
    for (int i = thread_idx; i < BLOCK_V * BLOCK_D; i += THREADS_PER_TG) {
      int local_v = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_v = v_start + local_v;
      int global_d = d_start + local_d;

      // W_smem for grad_hidden: need W[v, d] stored as [BLOCK_V, BLOCK_D]
      if (global_v < V && global_d < H) {
        W_smem[local_v * BLOCK_D + local_d] = float(weight[global_v * H + global_d]);
      } else {
        W_smem[local_v * BLOCK_D + local_d] = 0.0f;
      }
    }

    // Reload hidden tile
    for (int i = thread_idx; i < BLOCK_B * BLOCK_D; i += THREADS_PER_TG) {
      int local_b = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_b = b_start + local_b;
      int global_d = d_start + local_d;

      if (global_b < N && global_d < H) {
        H_smem[local_b * BLOCK_D + local_d] = float(hidden[global_b * H + global_d]);
      } else {
        H_smem[local_b * BLOCK_D + local_d] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute grad_hidden contribution using MMA: d_logits @ W
    // d_logits: [BLOCK_B, BLOCK_V], W: [BLOCK_V, BLOCK_D] -> result: [BLOCK_B, BLOCK_D]
    constexpr int TILES_GH_B = TILES_B;  // BLOCK_B / 8
    constexpr int TILES_GH_D = TILES_D;  // BLOCK_D / 8
    constexpr int TILES_GH_V = TILES_V;  // BLOCK_V / 8 (reduction dim)
    constexpr int TOTAL_GH_TILES = TILES_GH_B * TILES_GH_D;

    // Accumulator for grad_hidden tile
    simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE> GH_frag[4];  // Max 4 tiles per simdgroup
    for (int t = 0; t < 4; t++) {
      GH_frag[t] = simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE>(0.0f);
    }

    for (int tile_idx = simd_gid; tile_idx < TOTAL_GH_TILES; tile_idx += NUM_SIMDGROUPS) {
      int tb = tile_idx / TILES_GH_D;
      int td = tile_idx % TILES_GH_D;
      int local_tile = tile_idx / NUM_SIMDGROUPS;

      // Reduction over vocab dimension
      for (int tv = 0; tv < TILES_GH_V && (tv * FRAG_SIZE) < v_size; tv++) {
        simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE> DL_frag;
        simdgroup_load(DL_frag, d_logits_smem + tb * FRAG_SIZE * BLOCK_V + tv * FRAG_SIZE, BLOCK_V);

        simdgroup_matrix<float, FRAG_SIZE, FRAG_SIZE> W_frag;
        simdgroup_load(W_frag, W_smem + tv * FRAG_SIZE * BLOCK_D + td * FRAG_SIZE, BLOCK_D);

        simdgroup_multiply_accumulate(GH_frag[local_tile], DL_frag, W_frag, GH_frag[local_tile]);
      }
    }
    simdgroup_barrier(mem_flags::mem_none);

    // Store grad_hidden to threadgroup memory, then atomic add to global
    // Use W_smem for temp storage since W is no longer needed after grad_hidden MMA
    threadgroup float* gh_temp = W_smem;  // Reuse W_smem (no longer needed)
    for (int tile_idx = simd_gid; tile_idx < TOTAL_GH_TILES; tile_idx += NUM_SIMDGROUPS) {
      int tb = tile_idx / TILES_GH_D;
      int td = tile_idx % TILES_GH_D;
      int local_tile = tile_idx / NUM_SIMDGROUPS;

      simdgroup_store(GH_frag[local_tile], gh_temp + tb * FRAG_SIZE * BLOCK_D + td * FRAG_SIZE, BLOCK_D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Atomic add to global grad_hidden
    for (int i = thread_idx; i < BLOCK_B * BLOCK_D; i += THREADS_PER_TG) {
      int local_b = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_b = b_start + local_b;
      int global_d = d_start + local_d;

      if (global_b < N && global_d < H) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_hidden[global_b * H + global_d],
            gh_temp[local_b * BLOCK_D + local_d],
            memory_order_relaxed);
      }
    }

    // Compute grad_weight: GW[v, d] = sum_b(d_logits[b, v] * H[b, d])
    // Using scalar computation (MMA had overhead from accumulator setup)
    for (int i = thread_idx; i < BLOCK_V * BLOCK_D; i += THREADS_PER_TG) {
      int local_v = i / BLOCK_D;
      int local_d = i % BLOCK_D;
      int global_v = v_start + local_v;
      int global_d = d_start + local_d;

      if (global_v < V && global_d < H) {
        float grad_w = 0.0f;
        for (int local_b = 0; local_b < b_size; local_b++) {
          grad_w += d_logits_smem[local_b * BLOCK_V + local_v] * H_smem[local_b * BLOCK_D + local_d];
        }

        atomic_fetch_add_explicit(
            (device atomic<float>*)&grad_weight[global_v * H + global_d],
            grad_w,
            memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// =============================================================================
// CCE Compute d_logits Kernel (for chunked backward)
// Computes d_logits = (softmax - one_hot) * grad_output for a vocab chunk
// OPTIMIZED:
// - Vectorized: N_READS elements per thread (4x fewer threads, better coalescing)
// - Branchless target subtraction (no warp divergence)
// - Cached row-level values (lse, target, grad read once per thread batch)
// =============================================================================

// Native dtype version: reads T, computes in FP32, outputs T
// Follows MLX pattern: BF16 in registers → FP32 compute → BF16 out
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

  // Each thread processes N_READS consecutive elements
  const int base_idx = tid * N_READS;
  const int total_elements = N * chunk_V;

  // Early exit if completely out of bounds
  if (base_idx >= total_elements) return;

  // Process N_READS elements
  // NOTE: Elements may cross row boundaries, so we must compute row per-element
  #pragma unroll
  for (int i = 0; i < N_READS; i++) {
    const int idx = base_idx + i;

    // Bounds check
    if (idx >= total_elements) {
      continue;
    }

    // Compute row and column for THIS element (handles row boundary crossing)
    const int row = idx / chunk_V;
    const int col = idx % chunk_V;
    const int global_v = v_start + col;

    // Check vocab bounds
    if (global_v >= V) {
      d_logits[idx] = T(0.0f);
      continue;
    }

    // Read row-level values for this element's actual row
    const float token_lse = lse[row];
    const int target = targets[row];
    const float grad_scale = grad_output[row] * scale;

    // Read logit in native dtype, convert to FP32
    float logit = float(logits[idx]);

    // Compute softmax probability
    float prob = safe_exp(logit - token_lse);
    prob = clamp(prob, 0.0f, 1.0f);

    // Branchless: subtract 1.0 if this is the target (no warp divergence)
    float d_logit = prob - float(global_v == target);

    // Scale by upstream gradient
    d_logit *= grad_scale;

    // Convert back to native dtype and write
    d_logits[idx] = T(d_logit);
  }
}

// =============================================================================
// CCE Chunk LogSumExp Kernel (Single-Pass Online Algorithm)
// Updates running max, sum_exp, and target_logit for online logsumexp
// Uses single-pass online algorithm (2x less memory bandwidth vs two-pass)
// Coalesced memory access pattern with N_READS=4 vectorization
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

  // Shared memory for simdgroup reductions
  threadgroup float* smem_max = smem;                    // [NUM_SIMDGROUPS]
  threadgroup float* smem_sum = smem + NUM_SIMDGROUPS;   // [NUM_SIMDGROUPS]

  // Initialize shared memory to identity values (like MLX softmax does)
  // This ensures deterministic behavior even if some slots aren't written
  if (lid < NUM_SIMDGROUPS) {
    smem_max[lid] = -INFINITY;  // Identity for max
    smem_sum[lid] = 0.0f;       // Identity for sum
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int valid_chunk_v = min(chunk_V, V - v_start);
  const int iterations = (valid_chunk_v + THREADS_PER_TG * N_READS - 1) / (THREADS_PER_TG * N_READS);

  // Single-pass online algorithm: compute max and normalizer together
  // This halves memory bandwidth compared to two-pass approach
  float prevmax;
  float maxval = -INFINITY;
  float normalizer = 0.0f;
  float local_target = 0.0f;
  bool found_target = false;

  for (int r = 0; r < iterations; r++) {
    // Coalesced memory access: consecutive threads read consecutive elements
    int offset = r * THREADS_PER_TG * N_READS + lid * N_READS;
    float vals[N_READS];

    // Load N_READS elements with vectorized access when possible
    // float4/half4 loads are more efficient than scalar loops
    if (offset + N_READS <= valid_chunk_v) {
      // Vectorized load path - use packed vector loads for better memory throughput
      if constexpr (N_READS == 4) {
        // Cast to vec4 pointer for single wide load instruction
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
      // Scalar fallback for boundary conditions
      #pragma unroll
      for (int i = 0; i < N_READS; i++) {
        vals[i] = (offset + i < valid_chunk_v) ? float(row_logits[offset + i]) : -INFINITY;
      }
    }

    // Check for target in this batch
    #pragma unroll
    for (int i = 0; i < N_READS; i++) {
      int global_v = v_start + offset + i;
      if (global_v == target && offset + i < valid_chunk_v) {
        local_target = vals[i];
        found_target = true;
      }
    }

    // Online logsumexp: update max and rescale normalizer
    prevmax = maxval;
    #pragma unroll
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < vals[i]) ? vals[i] : maxval;
    }

    // Rescale existing normalizer to new max
    normalizer *= fast::exp(prevmax - maxval);

    // Add new values with scaling
    #pragma unroll
    for (int i = 0; i < N_READS; i++) {
      normalizer += fast::exp(vals[i] - maxval);
    }
  }

  // SIMD reduction for max
  prevmax = maxval;
  maxval = simd_max(maxval);
  // Rescale normalizer after SIMD max reduction
  normalizer *= fast::exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);

  // Cross-simdgroup reduction (simplified for correctness)
  // Store SIMD results to shared memory
  if (simd_lid == 0) {
    smem_max[simd_gid] = maxval;
    smem_sum[simd_gid] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Only first simdgroup does the final reduction (thread 0 will use the result)
  float chunk_max = smem_max[0];
  float chunk_sum_exp = smem_sum[0];

  if (simd_gid == 0 && simd_lid == 0) {
    // Sequential reduction for determinism
    for (int i = 1; i < NUM_SIMDGROUPS; i++) {
      float sg_max = smem_max[i];
      float sg_sum = smem_sum[i];

      float new_max = max(chunk_max, sg_max);
      chunk_sum_exp = chunk_sum_exp * fast::exp(chunk_max - new_max) +
                      sg_sum * fast::exp(sg_max - new_max);
      chunk_max = new_max;
    }
  }

  // Thread 0 updates running values
  if (lid == 0) {
    // Online combination with running values
    float old_max = running_max[row];
    float old_sum_exp = running_sum_exp[row];

    float new_max = max(old_max, chunk_max);
    float new_sum_exp = old_sum_exp * fast::exp(old_max - new_max) +
                        chunk_sum_exp * fast::exp(chunk_max - new_max);

    running_max[row] = new_max;
    running_sum_exp[row] = new_sum_exp;
  }

  // Accumulate target logit (use SIMD reduction)
  float simd_target_val = simd_sum(local_target);

  // Use reduction for found_target
  // Store both the value and a found flag to handle target_logit == 0.0 correctly
  // Reuse smem_sum (no longer needed) for found flags
  bool any_found = simd_any(found_target);
  if (simd_lid == 0) {
    smem_max[simd_gid] = simd_target_val;           // Always store the value
    smem_sum[simd_gid] = any_found ? 1.0f : 0.0f;   // Store found flag
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid == 0) {
    for (int i = 0; i < NUM_SIMDGROUPS; i++) {
      if (smem_sum[i] != 0.0f) {  // Check found flag, not value
        target_logit[row] = smem_max[i];
        break;
      }
    }
  }
}

// =============================================================================
// CCE Finalize Helper (Unified Logic)
// Common computation for finalize kernels - reduces code duplication
// =============================================================================

METAL_FUNC float cce_compute_lse(float running_max_val, float running_sum_exp_val) {
  // Add epsilon for numerical stability (consistent with other logsumexp computations)
  return running_max_val + log(running_sum_exp_val + 1e-9f);
}

METAL_FUNC float cce_compute_loss(float lse, float target_logit_val, float scale) {
  return (lse - target_logit_val) * scale;
}

// =============================================================================
// CCE Finalize LSE Kernel
// Computes final logsumexp from running max and sum_exp
// logsumexp = max + log(sum_exp)
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
// CCE Init Running Values Kernel
// Initializes running_max=-inf, running_sum_exp=0, target_logit=0 in one dispatch
// Reduces 3 separate fill operations to 1 kernel launch
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
// CCE Finalize Loss Kernel
// Computes final loss from running max, sum_exp, and target_logit
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
// CCE Finalize Loss With LSE Kernel
// Computes final loss AND outputs logsumexp for backward pass
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
  logsumexp_out[tid] = lse;  // Always output for backward pass

  int target = targets[tid];
  if (target == ignore_index) {
    loss[tid] = 0.0f;
    return;
  }

  loss[tid] = cce_compute_loss(lse, target_logit[tid], scale);
}

// =============================================================================
// CCE Combine Loss Kernel
// Combines LSE and neg_target_logit into final loss after 2D forward pass
// loss = (lse + neg_target_logit) * scale = (lse - target_logit) * scale
// =============================================================================

[[host_name("cce_combine_loss")]]
[[kernel]] void cce_combine_loss(
    const device float* lse [[buffer(0)]],
    const device float* neg_target_logit [[buffer(1)]],
    device float* loss [[buffer(2)]],
    constant CCEParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {

  if (tid >= uint(params.N)) return;

  // loss = (lse - target_logit) * scale
  // neg_target_logit stores -target_logit, so: loss = (lse + neg_target_logit) * scale
  loss[tid] = (lse[tid] + neg_target_logit[tid]) * params.scale;
}

} // namespace steel
} // namespace mlx
