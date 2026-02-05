// Copyright © 2025 Apple Inc.

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/steel/cce/kernels.h"

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
      constant float& softcap [[buffer(10)]],                                     \
      uint tid [[thread_position_in_grid]]);

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
      constant float& softcap [[buffer(9)]],                                     \
      threadgroup float* smem [[threadgroup(0)]],                                \
      uint3 tgid [[threadgroup_position_in_grid]],                               \
      uint lid [[thread_index_in_threadgroup]],                                  \
      uint simd_lid [[thread_index_in_simdgroup]],                               \
      uint simd_gid [[simdgroup_index_in_threadgroup]]);

instantiate_cce_compute_d_logits(float32, float)
instantiate_cce_compute_d_logits(float16, half)
instantiate_cce_compute_d_logits(bfloat16, bfloat16_t)

instantiate_cce_chunk_logsumexp(float32, float)
instantiate_cce_chunk_logsumexp(float16, half)
instantiate_cce_chunk_logsumexp(bfloat16, bfloat16_t)
