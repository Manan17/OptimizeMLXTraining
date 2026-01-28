// Copyright © 2025 Apple Inc.
//
// CCE (Cut Cross-Entropy) Parameters
// Memory-efficient cross-entropy that tiles over vocabulary

#pragma once

namespace mlx {
namespace steel {

struct CCEParams {
  int N;              // Number of tokens (batch * seq_len)
  int H;              // Hidden dimension
  int V;              // Vocabulary size
  int ignore_index;   // Index to ignore in loss computation (-100 typically)
  float scale;        // Loss scaling factor (1.0 / num_valid_tokens)
};

// Compile-time constants - defined as macros for Metal compatibility
#define CCE_VOCAB_TILE 1024       // Vocab tile size
#define CCE_HIDDEN_TILE 128       // Hidden tile size for loading
#define CCE_TOKENS_PER_TG 4       // Tokens per threadgroup
#define CCE_SIMD_SIZE 32
#define CCE_NUM_SIMDGROUPS 4
#define CCE_THREADS_PER_TG (CCE_SIMD_SIZE * CCE_NUM_SIMDGROUPS)

// Sparsity threshold for backward pass (2^-12 in bfloat16 precision)
#define CCE_SPARSITY_THRESHOLD 2.44140625e-4f

} // namespace steel
} // namespace mlx
