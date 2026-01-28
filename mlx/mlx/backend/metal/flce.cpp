// Copyright © 2025 Apple Inc.
//
// FLCE (Fused Linear Cross Entropy) GPU implementation
// SIMD-optimized: 32 threads per row, 4 rows per threadgroup

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/flce/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

// Dispatch constants
constexpr int SIMD_SIZE = 32;
constexpr int ROWS_PER_TG = 4;
constexpr int THREADS_PER_TG = SIMD_SIZE * ROWS_PER_TG;  // 128 threads

void FLCELoss::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& hidden_in = inputs[0];
  const array& weight_in = inputs[1];
  const array& targets_in = inputs[2];

  // Compute dimensions
  int N, H, V;
  if (hidden_in.ndim() == 3) {
    N = hidden_in.shape(0) * hidden_in.shape(1);
    H = hidden_in.shape(2);
  } else {
    N = hidden_in.shape(0);
    H = hidden_in.shape(1);
  }
  V = weight_in.shape(0);

  // Ensure contiguous
  std::vector<array> copies;
  auto ensure_contiguous = [&copies, &s](const array& arr) -> const array& {
    if (arr.flags().row_contiguous) {
      return arr;
    }
    array arr_copy = contiguous_copy_gpu(arr, s);
    copies.push_back(std::move(arr_copy));
    return copies.back();
  };

  const array& h = ensure_contiguous(hidden_in);
  const array& w = ensure_contiguous(weight_in);
  const array& t = ensure_contiguous(targets_in);

  // Allocate output
  array& loss = outputs[0];
  loss.set_data(allocator::malloc(loss.nbytes()));

  // Allocate logsumexp for backward
  array logsumexp({N}, float32, nullptr, {});
  logsumexp.set_data(allocator::malloc(logsumexp.nbytes()));

  // Initialize loss to zero
  array zero_val = array(0.0f, float32);
  fill_gpu(zero_val, loss, s);

  // Parameters
  float scale = 1.0f / static_cast<float>(N);
  mlx::steel::FLCEParams params{N, H, V, ignore_index_, scale};

  // Kernel dispatch
  std::string kernel_name = "flce_simd_forward_" + type_to_name(h.dtype());

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kernel_name);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(h, 0);
  compute_encoder.set_input_array(w, 1);
  compute_encoder.set_input_array(t, 2);
  compute_encoder.set_output_array(loss, 3);
  compute_encoder.set_output_array(logsumexp, 4);
  compute_encoder.set_bytes(params, 5);

  // Dispatch: 4 rows per threadgroup, 128 threads per threadgroup
  int num_tgs = (N + ROWS_PER_TG - 1) / ROWS_PER_TG;
  MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
  MTL::Size group_dims = MTL::Size(THREADS_PER_TG, 1, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Cleanup
  d.add_temporary(logsumexp, s.index);
  d.add_temporary(std::move(zero_val), s.index);
  d.add_temporaries(std::move(copies), s.index);
}

void FLCELossVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& hidden_in = inputs[0];
  const array& weight_in = inputs[1];
  const array& targets_in = inputs[2];
  const array& grad_output = inputs[3];

  // Compute dimensions
  int N, H, V;
  if (hidden_in.ndim() == 3) {
    N = hidden_in.shape(0) * hidden_in.shape(1);
    H = hidden_in.shape(2);
  } else {
    N = hidden_in.shape(0);
    H = hidden_in.shape(1);
  }
  V = weight_in.shape(0);

  // Ensure contiguous
  std::vector<array> copies;
  auto ensure_contiguous = [&copies, &s](const array& arr) -> const array& {
    if (arr.flags().row_contiguous) {
      return arr;
    }
    array arr_copy = contiguous_copy_gpu(arr, s);
    copies.push_back(std::move(arr_copy));
    return copies.back();
  };

  const array& h = ensure_contiguous(hidden_in);
  const array& w = ensure_contiguous(weight_in);
  const array& t = ensure_contiguous(targets_in);
  const array& g_out = ensure_contiguous(grad_output);

  array& grad_hidden = outputs[0];
  array& grad_weight = outputs[1];

  // Allocate outputs
  grad_hidden.set_data(allocator::malloc(grad_hidden.nbytes()));
  grad_weight.set_data(allocator::malloc(grad_weight.nbytes()));

  // Zero grad_weight (not computed in this kernel)
  array zero_val = array(0.0f, float32);
  fill_gpu(zero_val, grad_weight, s);

  // Allocate logsumexp
  array logsumexp({N}, float32, nullptr, {});
  logsumexp.set_data(allocator::malloc(logsumexp.nbytes()));

  // Dummy loss for forward pass
  array dummy_loss({1}, float32, nullptr, {});
  dummy_loss.set_data(allocator::malloc(dummy_loss.nbytes()));
  fill_gpu(zero_val, dummy_loss, s);

  float scale = 1.0f / static_cast<float>(N);
  mlx::steel::FLCEParams params{N, H, V, ignore_index_, scale};

  int num_tgs = (N + ROWS_PER_TG - 1) / ROWS_PER_TG;
  MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
  MTL::Size group_dims = MTL::Size(THREADS_PER_TG, 1, 1);

  auto& compute_encoder = d.get_command_encoder(s.index);

  // Forward pass to compute logsumexp
  {
    std::string kernel_name = "flce_simd_forward_" + type_to_name(h.dtype());
    auto kernel = d.get_kernel(kernel_name);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(h, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(t, 2);
    compute_encoder.set_output_array(dummy_loss, 3);
    compute_encoder.set_output_array(logsumexp, 4);
    compute_encoder.set_bytes(params, 5);

    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Backward pass
  {
    std::string kernel_name = "flce_simd_backward_" + type_to_name(h.dtype());
    auto kernel = d.get_kernel(kernel_name);
    compute_encoder.set_compute_pipeline_state(kernel);

    // Use float32 buffer for gradient computation
    array grad_hidden_f32 = grad_hidden;
    if (grad_hidden.dtype() != float32) {
      grad_hidden_f32 = array({N, H}, float32, nullptr, {});
      grad_hidden_f32.set_data(allocator::malloc(grad_hidden_f32.nbytes()));
    }

    compute_encoder.set_input_array(h, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(t, 2);
    compute_encoder.set_input_array(logsumexp, 3);
    compute_encoder.set_input_array(g_out, 4);
    compute_encoder.set_output_array(grad_hidden_f32, 5);
    compute_encoder.set_bytes(params, 6);

    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

    // Cast back if needed
    if (grad_hidden.dtype() != float32) {
      copy_gpu(grad_hidden_f32, grad_hidden, CopyType::General, s);
      d.add_temporary(grad_hidden_f32, s.index);
    }
  }

  // Cleanup
  d.add_temporary(dummy_loss, s.index);
  d.add_temporary(logsumexp, s.index);
  d.add_temporary(std::move(zero_val), s.index);
  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
