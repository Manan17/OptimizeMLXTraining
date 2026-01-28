// Copyright © 2025 Apple Inc.

#pragma once

namespace mlx {
namespace steel {

struct FLCEParams {
  int N;             // Number of rows (batch * seq_len)
  int H;             // Hidden dimension
  int V;             // Vocabulary size
  int ignore_index;  // Index to ignore in loss computation
  float scale;       // 1.0 / num_valid_tokens
};

} // namespace steel
} // namespace mlx
