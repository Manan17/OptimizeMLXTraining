// Copyright © 2025 Apple Inc.

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/steel/flce/kernels.h"

// Instantiate SIMD forward kernels
#define instantiate_flce_simd_forward(name, type)                         \
  template [[host_name("flce_simd_forward_" #name)]]                      \
  [[kernel]] void mlx::steel::flce_simd_forward<type>(                    \
      const device type* hidden [[buffer(0)]],                            \
      const device type* weight [[buffer(1)]],                            \
      const device int32_t* targets [[buffer(2)]],                        \
      device float* loss [[buffer(3)]],                                   \
      device float* logsumexp_out [[buffer(4)]],                          \
      constant mlx::steel::FLCEParams& params [[buffer(5)]],              \
      uint3 tid [[threadgroup_position_in_grid]],                         \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
      uint simd_lid [[thread_index_in_simdgroup]]);

// Instantiate SIMD backward kernels
#define instantiate_flce_simd_backward(name, type)                        \
  template [[host_name("flce_simd_backward_" #name)]]                     \
  [[kernel]] void mlx::steel::flce_simd_backward<type>(                   \
      const device type* hidden [[buffer(0)]],                            \
      const device type* weight [[buffer(1)]],                            \
      const device int32_t* targets [[buffer(2)]],                        \
      const device float* logsumexp [[buffer(3)]],                        \
      const device float* grad_out [[buffer(4)]],                         \
      device float* grad_hidden [[buffer(5)]],                            \
      constant mlx::steel::FLCEParams& params [[buffer(6)]],              \
      uint3 tid [[threadgroup_position_in_grid]],                         \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
      uint simd_lid [[thread_index_in_simdgroup]]);

// Instantiate for all supported types
instantiate_flce_simd_forward(float32, float)
instantiate_flce_simd_forward(float16, half)
instantiate_flce_simd_forward(bfloat16, bfloat16_t)

instantiate_flce_simd_backward(float32, float)
instantiate_flce_simd_backward(float16, half)
instantiate_flce_simd_backward(bfloat16, bfloat16_t)
