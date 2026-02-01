// Copyright © 2025 Apple Inc.
//
// CCE (Cut Cross-Entropy) Kernel Instantiations
//
// Current implementation uses steel_matmul with chunked processing for best performance.
// Legacy SIMD kernels kept for small problem sizes where dispatch overhead dominates.

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/steel/cce/kernels.h"

// =============================================================================
// SIMD Forward Kernels (legacy, for small problems)
// =============================================================================

#define instantiate_cce_simd_forward(name, type)                             \
  template [[host_name("cce_simd_forward_" #name)]]                          \
  [[kernel]] void mlx::steel::cce_simd_forward<type>(                        \
      const device type* hidden [[buffer(0)]],                               \
      const device type* weight [[buffer(1)]],                               \
      const device int32_t* targets [[buffer(2)]],                           \
      device float* loss [[buffer(3)]],                                      \
      device float* logsumexp_out [[buffer(4)]],                             \
      constant mlx::steel::CCEParams& params [[buffer(5)]],                  \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);

// =============================================================================
// SIMD Backward Kernels (legacy, for small problems)
// =============================================================================

#define instantiate_cce_simd_backward(name, type)                            \
  template [[host_name("cce_simd_backward_" #name)]]                         \
  [[kernel]] void mlx::steel::cce_simd_backward<type>(                       \
      const device type* hidden [[buffer(0)]],                               \
      const device type* weight [[buffer(1)]],                               \
      const device int32_t* targets [[buffer(2)]],                           \
      const device float* logsumexp [[buffer(3)]],                           \
      const device float* grad_output [[buffer(4)]],                         \
      device float* grad_hidden [[buffer(5)]],                               \
      device float* grad_weight [[buffer(6)]],                               \
      constant mlx::steel::CCEParams& params [[buffer(7)]],                  \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);

// =============================================================================
// Compute d_logits Kernel (used in chunked backward with steel_matmul)
// =============================================================================

#define instantiate_cce_compute_d_logits(name, type)                             \
  template [[host_name("cce_compute_d_logits_" #name)]]                           \
  [[kernel]] void mlx::steel::cce_compute_d_logits<type>(                         \
      const device type* logits [[buffer(0)]],                                    \
      const device float* lse [[buffer(1)]],                                      \
      const device int32_t* targets [[buffer(2)]],                                \
      const device float* grad_output [[buffer(3)]],                              \
      device type* d_logits [[buffer(4)]],                                        \
      constant int& N [[buffer(5)]],                                              \
      constant int& chunk_V [[buffer(6)]],                                        \
      constant int& v_start [[buffer(7)]],                                        \
      constant int& V [[buffer(8)]],                                              \
      constant float& scale [[buffer(9)]],                                        \
      uint tid [[thread_position_in_grid]]);

// =============================================================================
// Chunk LogSumExp Kernel (used in chunked forward for online logsumexp)
// =============================================================================

#define instantiate_cce_chunk_logsumexp(name, type)                              \
  template [[host_name("cce_chunk_logsumexp_" #name)]]                           \
  [[kernel]] void mlx::steel::cce_chunk_logsumexp<type>(                         \
      const device type* logits [[buffer(0)]],                                   \
      const device int32_t* targets [[buffer(1)]],                               \
      device float* running_max [[buffer(2)]],                                   \
      device float* running_sum_exp [[buffer(3)]],                               \
      device float* target_logit [[buffer(4)]],                                  \
      constant int& N [[buffer(5)]],                                             \
      constant int& chunk_V [[buffer(6)]],                                       \
      constant int& v_start [[buffer(7)]],                                       \
      constant int& V [[buffer(8)]],                                             \
      threadgroup float* smem [[threadgroup(0)]],                                \
      uint3 tgid [[threadgroup_position_in_grid]],                               \
      uint lid [[thread_index_in_threadgroup]],                                  \
      uint simd_lid [[thread_index_in_simdgroup]],                               \
      uint simd_gid [[simdgroup_index_in_threadgroup]]);

// =============================================================================
// Instantiate for all supported types
// =============================================================================

// SIMD versions (legacy, for small problems where dispatch overhead dominates)
instantiate_cce_simd_forward(float32, float)
instantiate_cce_simd_forward(float16, half)
instantiate_cce_simd_forward(bfloat16, bfloat16_t)

instantiate_cce_simd_backward(float32, float)
instantiate_cce_simd_backward(float16, half)
instantiate_cce_simd_backward(bfloat16, bfloat16_t)

// Compute d_logits (used in chunked backward with steel_matmul)
instantiate_cce_compute_d_logits(float32, float)
instantiate_cce_compute_d_logits(float16, half)
instantiate_cce_compute_d_logits(bfloat16, bfloat16_t)

// Chunk logsumexp (used in chunked forward for online logsumexp)
instantiate_cce_chunk_logsumexp(float32, float)
instantiate_cce_chunk_logsumexp(float16, half)
instantiate_cce_chunk_logsumexp(bfloat16, bfloat16_t)

// Non-template kernels (cce_finalize_lse, cce_init_running_values,
// cce_finalize_loss, cce_finalize_loss_with_lse) are declared with
// [[host_name(...)]] directly in kernels.h and don't need instantiation here.
