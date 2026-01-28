// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <variant>

#include "mlx/api.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    StreamOrDevice s = {});

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false);

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

/** Computes fused linear cross entropy loss efficiently
 *
 * This function computes cross entropy loss without materializing the full
 * [N, V] logits tensor by processing vocabulary in chunks using online softmax.
 *
 * Args:
 *   hidden: Input hidden states [N, H] or [B, T, H]
 *   weight: Language model head weight [V, H]
 *   targets: Target indices [N] or [B, T]
 *   chunk_size: Vocabulary chunk size for memory-efficient computation
 *   ignore_index: Target index to ignore in loss computation
 *
 * Returns:
 *   Scalar loss value
 */
MLX_API array flce_loss(
    const array& hidden,
    const array& weight,
    const array& targets,
    int chunk_size = 4096,
    int ignore_index = -100,
    StreamOrDevice s = {});

/** Computes Cut Cross-Entropy loss efficiently
 *
 * CCE computes cross-entropy loss without materializing the full logits tensor
 * by tiling over the vocabulary dimension and using online logsumexp.
 *
 * Key features:
 * - Tiles vocabulary (not tokens) for optimal memory efficiency
 * - Uses threadgroup memory for weight tiles
 * - Exploits softmax sparsity in backward pass (~99% compute savings)
 * - Properly computes grad_weight for training
 *
 * Based on Apple's "Cut Your Losses" paper.
 *
 * Args:
 *   hidden: Input hidden states [N, H] or [B, T, H]
 *   weight: Language model head weight [V, H]
 *   targets: Target indices [N] or [B, T]
 *   ignore_index: Target index to ignore in loss computation
 *
 * Returns:
 *   Per-token loss values [N] or [B, T]
 */
MLX_API array cce_loss(
    const array& hidden,
    const array& weight,
    const array& targets,
    int ignore_index = -100,
    StreamOrDevice s = {});

} // namespace mlx::core::fast
