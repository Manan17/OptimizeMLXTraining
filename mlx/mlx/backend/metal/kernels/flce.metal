// Copyright © 2025 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/flce.h"

// Instantiate forward kernels
#define instantiate_flce_forward(name, type) \
  instantiate_kernel("flce_chunk_forward_" #name, flce_chunk_forward, type)

// Instantiate backward kernels
#define instantiate_flce_backward(name, type) \
  instantiate_kernel("flce_chunk_backward_" #name, flce_chunk_backward, type)

instantiate_flce_forward(float32, float)
instantiate_flce_forward(float16, half)
instantiate_flce_forward(bfloat16, bfloat16_t)

instantiate_flce_backward(float32, float)
instantiate_flce_backward(float16, half)
instantiate_flce_backward(bfloat16, bfloat16_t)
// clang-format on
