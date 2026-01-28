// Copyright © 2025 Apple Inc.
//
// CCE (Cut Cross-Entropy) Kernel Instantiations

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/steel/cce/kernels.h"

// =============================================================================
// SIMD Forward Kernels (simple version)
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
// SIMD Backward Kernels (with sparsity)
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
// Tiled Forward Kernels (optimized version)
// =============================================================================

#define instantiate_cce_tiled_forward(name, type, bv, bh)                    \
  template [[host_name("cce_tiled_forward_" #name "_bv" #bv "_bh" #bh)]]     \
  [[kernel]] void mlx::steel::cce_forward_tiled<type, bv, bh>(               \
      const device type* hidden [[buffer(0)]],                               \
      const device type* weight [[buffer(1)]],                               \
      const device int32_t* targets [[buffer(2)]],                           \
      device float* loss [[buffer(3)]],                                      \
      device float* logsumexp_out [[buffer(4)]],                             \
      constant mlx::steel::CCEParams& params [[buffer(5)]],                  \
      threadgroup float* smem [[threadgroup(0)]],                            \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);

// =============================================================================
// Tiled Backward Kernels (with sparsity, optimized)
// =============================================================================

#define instantiate_cce_tiled_backward(name, type, bv, bh)                   \
  template [[host_name("cce_tiled_backward_" #name "_bv" #bv "_bh" #bh)]]    \
  [[kernel]] void mlx::steel::cce_backward<type, bv, bh>(                    \
      const device type* hidden [[buffer(0)]],                               \
      const device type* weight [[buffer(1)]],                               \
      const device int32_t* targets [[buffer(2)]],                           \
      const device float* logsumexp [[buffer(3)]],                           \
      const device float* grad_output [[buffer(4)]],                         \
      device float* grad_hidden [[buffer(5)]],                               \
      device float* grad_weight [[buffer(6)]],                               \
      constant mlx::steel::CCEParams& params [[buffer(7)]],                  \
      threadgroup float* smem [[threadgroup(0)]],                            \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);

// =============================================================================
// Instantiate for all supported types
// =============================================================================

// SIMD versions (simple, works everywhere)
instantiate_cce_simd_forward(float32, float)
instantiate_cce_simd_forward(float16, half)
instantiate_cce_simd_forward(bfloat16, bfloat16_t)

instantiate_cce_simd_backward(float32, float)
instantiate_cce_simd_backward(float16, half)
instantiate_cce_simd_backward(bfloat16, bfloat16_t)

// Tiled versions with different tile sizes
// BV=256, BH=64: Good balance for most cases
instantiate_cce_tiled_forward(float32, float, 256, 64)
instantiate_cce_tiled_forward(float16, half, 256, 64)
instantiate_cce_tiled_forward(bfloat16, bfloat16_t, 256, 64)

instantiate_cce_tiled_backward(float32, float, 256, 64)
instantiate_cce_tiled_backward(float16, half, 256, 64)
instantiate_cce_tiled_backward(bfloat16, bfloat16_t, 256, 64)

// BV=512, BH=64: Larger vocab tiles for bigger vocabs
instantiate_cce_tiled_forward(float32, float, 512, 64)
instantiate_cce_tiled_forward(float16, half, 512, 64)
instantiate_cce_tiled_forward(bfloat16, bfloat16_t, 512, 64)

instantiate_cce_tiled_backward(float32, float, 512, 64)
instantiate_cce_tiled_backward(float16, half, 512, 64)
instantiate_cce_tiled_backward(bfloat16, bfloat16_t, 512, 64)

// =============================================================================
// 2D Tiled Kernels (optimized with lock-free atomic combination)
// =============================================================================

#define instantiate_cce_forward_2d(name, type, bb, bv, bd)                     \
  template [[host_name("cce_forward_2d_" #name "_bb" #bb "_bv" #bv "_bd" #bd)]] \
  [[kernel]] void mlx::steel::cce_forward_2d<type, bb, bv, bd>(                \
      const device type* hidden [[buffer(0)]],                                 \
      const device type* weight [[buffer(1)]],                                 \
      const device int32_t* targets [[buffer(2)]],                             \
      device float* lse_out [[buffer(3)]],                                     \
      device float* neg_target_logit [[buffer(4)]],                            \
      constant mlx::steel::CCEParams& params [[buffer(5)]],                    \
      threadgroup float* smem [[threadgroup(0)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint3 lid [[thread_position_in_threadgroup]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_cce_backward_2d(name, type, bb, bv, bd)                    \
  template [[host_name("cce_backward_2d_" #name "_bb" #bb "_bv" #bv "_bd" #bd)]] \
  [[kernel]] void mlx::steel::cce_backward_2d<type, bb, bv, bd>(               \
      const device type* hidden [[buffer(0)]],                                 \
      const device type* weight [[buffer(1)]],                                 \
      const device int32_t* targets [[buffer(2)]],                             \
      const device float* lse [[buffer(3)]],                                   \
      const device float* grad_output [[buffer(4)]],                           \
      device float* grad_hidden [[buffer(5)]],                                 \
      device float* grad_weight [[buffer(6)]],                                 \
      constant mlx::steel::CCEParams& params [[buffer(7)]],                    \
      threadgroup float* smem [[threadgroup(0)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint3 lid [[thread_position_in_threadgroup]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]);

// Instantiate 2D tiled kernels: BLOCK_B=32, BLOCK_V=128, BLOCK_D=32
instantiate_cce_forward_2d(float32, float, 32, 128, 32)
instantiate_cce_forward_2d(float16, half, 32, 128, 32)
instantiate_cce_forward_2d(bfloat16, bfloat16_t, 32, 128, 32)

instantiate_cce_backward_2d(float32, float, 32, 128, 32)
instantiate_cce_backward_2d(float16, half, 32, 128, 32)
instantiate_cce_backward_2d(bfloat16, bfloat16_t, 32, 128, 32)

// =============================================================================
// MMA-Optimized Kernels (simdgroup_matrix for hardware acceleration)
// =============================================================================

#define instantiate_cce_forward_mma(name, type, bb, bv, bd)                     \
  template [[host_name("cce_forward_mma_" #name "_bb" #bb "_bv" #bv "_bd" #bd)]] \
  [[kernel]] void mlx::steel::cce_forward_mma<type, bb, bv, bd>(                \
      const device type* hidden [[buffer(0)]],                                  \
      const device type* weight [[buffer(1)]],                                  \
      const device int32_t* targets [[buffer(2)]],                              \
      device float* lse_out [[buffer(3)]],                                      \
      device float* neg_target_logit [[buffer(4)]],                             \
      constant mlx::steel::CCEParams& params [[buffer(5)]],                     \
      threadgroup float* smem [[threadgroup(0)]],                               \
      uint3 tid [[threadgroup_position_in_grid]],                               \
      uint3 lid [[thread_position_in_threadgroup]],                             \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                         \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_cce_backward_mma(name, type, bb, bv, bd)                    \
  template [[host_name("cce_backward_mma_" #name "_bb" #bb "_bv" #bv "_bd" #bd)]] \
  [[kernel]] void mlx::steel::cce_backward_mma<type, bb, bv, bd>(               \
      const device type* hidden [[buffer(0)]],                                  \
      const device type* weight [[buffer(1)]],                                  \
      const device int32_t* targets [[buffer(2)]],                              \
      const device float* lse [[buffer(3)]],                                    \
      const device float* grad_output [[buffer(4)]],                            \
      device float* grad_hidden [[buffer(5)]],                                  \
      device float* grad_weight [[buffer(6)]],                                  \
      constant mlx::steel::CCEParams& params [[buffer(7)]],                     \
      threadgroup float* smem [[threadgroup(0)]],                               \
      uint3 tid [[threadgroup_position_in_grid]],                               \
      uint3 lid [[thread_position_in_threadgroup]],                             \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                         \
      uint simd_lid [[thread_index_in_simdgroup]]);

// MMA kernels: BLOCK_B=32, BLOCK_V=64, BLOCK_D=32 (optimized for 8x8 fragments)
instantiate_cce_forward_mma(float32, float, 32, 64, 32)
instantiate_cce_forward_mma(float16, half, 32, 64, 32)
instantiate_cce_forward_mma(bfloat16, bfloat16_t, 32, 64, 32)

instantiate_cce_backward_mma(float32, float, 32, 64, 32)
instantiate_cce_backward_mma(float16, half, 32, 64, 32)
instantiate_cce_backward_mma(bfloat16, bfloat16_t, 32, 64, 32)

// =============================================================================
// Compute d_logits Kernel (used in chunked backward)
// =============================================================================

#define instantiate_cce_compute_d_logits(name, type)                             \
  template [[host_name("cce_compute_d_logits_" #name)]]                           \
  [[kernel]] void mlx::steel::cce_compute_d_logits<type>(                         \
      const device type* logits [[buffer(0)]],                                    \
      const device float* lse [[buffer(1)]],                                      \
      const device int32_t* targets [[buffer(2)]],                                \
      const device float* grad_output [[buffer(3)]],                              \
      device float* d_logits [[buffer(4)]],                                       \
      constant int& N [[buffer(5)]],                                              \
      constant int& chunk_V [[buffer(6)]],                                        \
      constant int& v_start [[buffer(7)]],                                        \
      constant int& V [[buffer(8)]],                                              \
      constant float& scale [[buffer(9)]],                                        \
      uint tid [[thread_position_in_grid]]);

instantiate_cce_compute_d_logits(float32, float)
instantiate_cce_compute_d_logits(float16, half)
instantiate_cce_compute_d_logits(bfloat16, bfloat16_t)

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

instantiate_cce_chunk_logsumexp(float32, float)
instantiate_cce_chunk_logsumexp(float16, half)
instantiate_cce_chunk_logsumexp(bfloat16, bfloat16_t)

// =============================================================================
// Combine Loss Kernel (used after 2D forward to compute final loss)
// =============================================================================

// Note: cce_combine_loss and cce_finalize_loss are non-template functions, declared with host_name in kernels.h
