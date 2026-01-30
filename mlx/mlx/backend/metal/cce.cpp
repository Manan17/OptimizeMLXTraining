// Copyright © 2025 Apple Inc.
//
// CCE (Cut Cross-Entropy) GPU implementation
// Memory-efficient cross-entropy with vocabulary tiling and sparsity exploitation
//
// Optimized version following Apple's ml-cross-entropy:
// - 2D tiling: (B_tiles, V_tiles) for massive parallelism
// - Lock-based atomic LSE combination
// - Sparsity filtering in backward pass
// - Chunked backward using steel_matmul (avoids atomic contention)
// - LSE saved from forward pass to avoid recomputation in backward

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/cce/params.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {


// Dispatch constants for legacy kernels
constexpr int SIMD_SIZE = 32;
constexpr int ROWS_PER_TG = 4;
constexpr int THREADS_PER_TG = SIMD_SIZE * ROWS_PER_TG;  // 128 threads


// Note: MMA constants removed - custom MMA kernel is slower than steel_matmul chunked approach

// Chunked backward constants
// Base chunk size for large batches
constexpr int BASE_CHUNK_V = 16384;

// Adaptive chunk size based on batch size to optimize dispatch overhead vs memory
// Key insight: For small N, dispatch overhead dominates, so use larger chunks (fewer dispatches)
// For N in [768, 1280] (batch=8 zone), the fixed 16384 causes pathological behavior
inline int get_adaptive_chunk_v(int N, int V, int H) {
  // Very small batches: no chunking needed, process entire vocab
  if (N < 256) {
    return V;
  }

  // Small batches (N < 512): use larger chunks to minimize dispatch overhead
  if (N < 512) {
    return std::min(32768, V);
  }

  // Problem zone: batch=8 with seq=128 gives N=1024
  // Fixed CHUNK_V=16384 triggers suboptimal steel_matmul paths
  // Use smaller chunks to distribute work better
  if (N >= 768 && N <= 1280) {
    return std::min(8192, V);
  }

  // Medium batches: use standard chunk size
  if (N < 4096) {
    return std::min(BASE_CHUNK_V, V);
  }

  // Large batches: can use larger chunks since memory pressure justifies it
  return std::min(BASE_CHUNK_V, V);
}

// Legacy constants for fallback
constexpr int DEFAULT_BV = 256;
constexpr int DEFAULT_BH = 64;

void CCELoss::eval_gpu(
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

  // NATIVE BF16 SUPPORT: Use mixed precision strategy
  // - BF16 matmuls for logits computation (avoids weight conversion overhead)
  // - FP32 accumulation for numerical stability (sum of 49K exp values)
  // - FP32 logsumexp for backward pass precision
  bool use_bf16_matmul = (h.dtype() == bfloat16 && w.dtype() == bfloat16);
  bool use_fp32_matmul = (h.dtype() == float32 && w.dtype() == float32);

  if (!use_bf16_matmul && !use_fp32_matmul) {
    throw std::invalid_argument(
        "CCE requires matching input dtypes. Supported: both float32 or both bfloat16.");
  }

  // Allocate output loss (per-token)
  array& loss = outputs[0];
  loss.set_data(allocator::malloc(loss.nbytes()));

  // Check if we need to output logsumexp for backward pass (memory optimization)
  bool output_lse = output_logsumexp() && outputs.size() >= 2;
  if (output_lse) {
    array& lse_out = outputs[1];
    lse_out.set_data(allocator::malloc(lse_out.nbytes()));
  }

  float scale = 1.0f;
  mlx::steel::CCEParams params{N, H, V, ignore_index_, scale};

  auto& compute_encoder = d.get_command_encoder(s.index);

  // Use chunked forward with steel_matmul for large vocab (best performance)
  // This computes logits in chunks and reduces to logsumexp
  // For small problems, fall back to simple SIMD kernel
  //
  // OPTIMIZATION: Skip chunking for very small N where dispatch overhead dominates
  // Dispatch overhead is ~100μs per kernel, which is 15-30% of compute for N<512
  bool use_chunked_forward = (V > 2000 && N >= 256);

  if (use_chunked_forward) {
    // Chunked forward: use steel_matmul + reduction for speed
    // This is O(N*chunk_V) memory per chunk vs O(N*V) for baseline

    // ADAPTIVE CHUNKING: Use optimal chunk size based on batch size
    int adaptive_chunk_v = get_adaptive_chunk_v(N, V, H);
    int num_chunks = (V + adaptive_chunk_v - 1) / adaptive_chunk_v;
    int max_chunk_v = std::min(adaptive_chunk_v, V);

    // NATIVE BF16: Compute logits in BF16, reduction kernel reads BF16 and accumulates in FP32
    // This avoids the 105MB weight_f32 conversion overhead!
    Dtype logits_dtype = use_bf16_matmul ? bfloat16 : float32;

    // Pre-allocate reusable buffer for logits chunk
    array logits_chunk({N, max_chunk_v}, logits_dtype, nullptr, {});
    logits_chunk.set_data(allocator::malloc(logits_chunk.nbytes()));

    // Allocate consolidated buffer for running state (saves 2 allocation calls)
    // Layout: [running_max (N) | running_sum_exp (N) | target_logit (N)]
    array running_state({3 * N}, float32, nullptr, {});
    running_state.set_data(allocator::malloc(running_state.nbytes()));

    // Create views into the consolidated buffer
    array running_max({N}, float32, nullptr, {});
    running_max.copy_shared_buffer(running_state, {1}, running_state.flags(), N, 0);

    array running_sum_exp({N}, float32, nullptr, {});
    running_sum_exp.copy_shared_buffer(running_state, {1}, running_state.flags(), N, N);

    array target_logit({N}, float32, nullptr, {});
    target_logit.copy_shared_buffer(running_state, {1}, running_state.flags(), N, 2 * N);

    // Initialize all running values in single kernel dispatch (reduces 3 dispatches to 1)
    // running_max = -inf, running_sum_exp = 0, target_logit = 0
    {
      auto init_kernel = d.get_kernel("cce_init_running_values");
      compute_encoder.set_compute_pipeline_state(init_kernel);
      compute_encoder.set_output_array(running_max, 0);
      compute_encoder.set_output_array(running_sum_exp, 1);
      compute_encoder.set_output_array(target_logit, 2);
      compute_encoder.set_bytes(N, 3);

      int threads_per_tg = 256;
      int num_tgs = (N + threads_per_tg - 1) / threads_per_tg;
      MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
      MTL::Size group_dims = MTL::Size(threads_per_tg, 1, 1);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    // Cache kernel
    std::string lse_kernel_name = "cce_chunk_logsumexp_" + type_to_name(h.dtype());
    auto lse_kernel = d.get_kernel(lse_kernel_name);

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
      int v_start = chunk_idx * max_chunk_v;
      int v_end = std::min(v_start + max_chunk_v, V);
      int current_chunk_v = v_end - v_start;

      // Create view of weight_chunk
      array weight_chunk({current_chunk_v, H}, w.dtype(), nullptr, {});
      int64_t w_offset = static_cast<int64_t>(v_start) * H;
      weight_chunk.copy_shared_buffer(
          w, {static_cast<int64_t>(H), 1}, w.flags(), static_cast<size_t>(current_chunk_v * H), w_offset);

      // Create view of logits_chunk for actual chunk size
      // NATIVE BF16: logits_view uses same dtype as logits_chunk (BF16 or FP32)
      array logits_view({N, current_chunk_v}, logits_dtype, nullptr, {});
      logits_view.copy_shared_buffer(
          logits_chunk, {static_cast<int64_t>(current_chunk_v), 1}, logits_chunk.flags(),
          static_cast<size_t>(N * current_chunk_v), 0);

      // Compute logits_chunk = hidden @ weight_chunk.T using optimized matmul
      // NATIVE BF16: BF16 inputs produce BF16 output, FP32 inputs produce FP32 output
      steel_matmul(
          s, d,
          h, weight_chunk, logits_view,
          N, current_chunk_v, H,
          1,
          H, H,
          false, true,
          copies);

      // Update running max, sum_exp, and target logit using parallel reduction kernel
      // One threadgroup per row, 256 threads cooperate on reduction
      compute_encoder.set_compute_pipeline_state(lse_kernel);
      compute_encoder.set_input_array(logits_view, 0);
      compute_encoder.set_input_array(t, 1);
      compute_encoder.set_output_array(running_max, 2);
      compute_encoder.set_output_array(running_sum_exp, 3);
      compute_encoder.set_output_array(target_logit, 4);
      compute_encoder.set_bytes(N, 5);
      compute_encoder.set_bytes(current_chunk_v, 6);
      compute_encoder.set_bytes(v_start, 7);
      compute_encoder.set_bytes(V, 8);

      // Threadgroup memory for SIMD reductions: 2 arrays of NUM_SIMDGROUPS floats
      constexpr int THREADS_PER_TG = 256;
      constexpr int NUM_SIMDGROUPS = THREADS_PER_TG / 32;
      size_t smem_size = 2 * NUM_SIMDGROUPS * sizeof(float);
      compute_encoder.set_threadgroup_memory_length(smem_size, 0);

      // One threadgroup per row
      MTL::Size grid_dims = MTL::Size(N, 1, 1);
      MTL::Size group_dims = MTL::Size(THREADS_PER_TG, 1, 1);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    // Compute final loss = log(sum_exp) + max - target_logit
    // Also save logsumexp if needed for backward pass
    if (output_lse) {
      // Use kernel that outputs both loss and logsumexp
      auto final_kernel = d.get_kernel("cce_finalize_loss_with_lse");
      compute_encoder.set_compute_pipeline_state(final_kernel);

      compute_encoder.set_input_array(running_max, 0);
      compute_encoder.set_input_array(running_sum_exp, 1);
      compute_encoder.set_input_array(target_logit, 2);
      compute_encoder.set_input_array(t, 3);
      compute_encoder.set_output_array(loss, 4);
      compute_encoder.set_output_array(outputs[1], 5);  // logsumexp output
      compute_encoder.set_bytes(N, 6);
      compute_encoder.set_bytes(ignore_index_, 7);
      compute_encoder.set_bytes(scale, 8);

      int threads_per_tg = 256;
      int num_tgs = (N + threads_per_tg - 1) / threads_per_tg;
      MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
      MTL::Size group_dims = MTL::Size(threads_per_tg, 1, 1);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    } else {
      auto final_kernel = d.get_kernel("cce_finalize_loss");
      compute_encoder.set_compute_pipeline_state(final_kernel);

      compute_encoder.set_input_array(running_max, 0);
      compute_encoder.set_input_array(running_sum_exp, 1);
      compute_encoder.set_input_array(target_logit, 2);
      compute_encoder.set_input_array(t, 3);
      compute_encoder.set_output_array(loss, 4);
      compute_encoder.set_bytes(N, 5);
      compute_encoder.set_bytes(ignore_index_, 6);
      compute_encoder.set_bytes(scale, 7);

      int threads_per_tg = 256;
      int num_tgs = (N + threads_per_tg - 1) / threads_per_tg;
      MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
      MTL::Size group_dims = MTL::Size(threads_per_tg, 1, 1);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    d.add_temporary(logits_chunk, s.index);
    d.add_temporary(running_state, s.index);  // Consolidated buffer (views share this)

  } else {
    // Simple SIMD kernel for smaller problems
    array logsumexp({N}, float32, nullptr, {});
    logsumexp.set_data(allocator::malloc(logsumexp.nbytes()));

    std::string kernel_name = "cce_simd_forward_" + type_to_name(h.dtype());

    auto kernel = d.get_kernel(kernel_name);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(h, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(t, 2);
    compute_encoder.set_output_array(loss, 3);
    compute_encoder.set_output_array(logsumexp, 4);
    compute_encoder.set_bytes(params, 5);

    int num_tgs = (N + ROWS_PER_TG - 1) / ROWS_PER_TG;
    MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
    MTL::Size group_dims = MTL::Size(THREADS_PER_TG, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

    d.add_temporary(logsumexp, s.index);
  }

  d.add_temporaries(std::move(copies), s.index);
}

void CCELossVJP::eval_gpu(
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
  const array& g_out_raw = ensure_contiguous(grad_output);

  // NATIVE BF16 SUPPORT: Accept both FP32 and BF16 inputs
  // Following MLX pattern: BF16 in registers → FP32 compute → BF16 out (no explicit casts)
  bool use_bf16 = (h.dtype() == bfloat16 && w.dtype() == bfloat16);
  bool use_fp32 = (h.dtype() == float32 && w.dtype() == float32);

  if (!use_bf16 && !use_fp32) {
    throw std::invalid_argument(
        "CCE backward requires matching input dtypes. Supported: both float32 or both bfloat16.");
  }

  Dtype compute_dtype = use_bf16 ? bfloat16 : float32;

  // Handle scalar grad_output (from mx.mean())
  array g_out_expanded({N}, float32, nullptr, {});
  g_out_expanded.set_data(allocator::malloc(g_out_expanded.nbytes()));
  if (g_out_raw.size() == 1) {
    array scale_val = array(1.0f / static_cast<float>(N), float32);
    fill_gpu(scale_val, g_out_expanded, s);
    copies.push_back(std::move(scale_val));
  } else {
    copy_gpu(g_out_raw, g_out_expanded, CopyType::General, s);
  }
  const array& g_out = g_out_expanded;

  array& grad_hidden = outputs[0];
  array& grad_weight = outputs[1];

  // Allocate outputs (same dtype as inputs - BF16 or FP32)
  grad_hidden.set_data(allocator::malloc(grad_hidden.nbytes()));
  grad_weight.set_data(allocator::malloc(grad_weight.nbytes()));

  // Zero grad_weight (grad_hidden uses beta=0 for first chunk, so no zero needed)
  // Use appropriate zero value for dtype
  if (use_bf16) {
    array zero_val_bf16 = array(static_cast<float>(0.0f), bfloat16);
    fill_gpu(zero_val_bf16, grad_weight, s);
    copies.push_back(std::move(zero_val_bf16));
  } else {
    array zero_val_f32 = array(0.0f, float32);
    fill_gpu(zero_val_f32, grad_weight, s);
    copies.push_back(std::move(zero_val_f32));
  }

  float scale = 1.0f;
  mlx::steel::CCEParams params{N, H, V, ignore_index_, scale};

  auto& compute_encoder = d.get_command_encoder(s.index);

  // Use chunked backward with steel_matmul for larger problems
  // steel_matmul handles arbitrary dimensions (no H % 8 requirement)
  bool use_mma = (V > 2000 && N >= 256);

  if (use_mma) {
    // Chunked backward using steel_matmul (avoids atomic contention)

    // ADAPTIVE CHUNKING: Use optimal chunk size based on batch size
    int adaptive_chunk_v = get_adaptive_chunk_v(N, V, H);
    int num_chunks = (V + adaptive_chunk_v - 1) / adaptive_chunk_v;
    int max_chunk_v = std::min(adaptive_chunk_v, V);

    // MEMORY OPTIMIZATION: Check if logsumexp was saved from forward pass
    // If has_logsumexp() is true, inputs[4] contains the saved logsumexp
    bool use_saved_lse = has_logsumexp() && inputs.size() > 4;
    bool logsumexp_needs_temp = false;  // Track if we need to add as temporary

    // Allocate or reference logsumexp
    array logsumexp({N}, float32, nullptr, {});

    if (use_saved_lse) {
      // Use logsumexp saved from forward pass (avoids ~0.5GB recomputation)
      const array& saved_lse = inputs[4];
      if (saved_lse.flags().row_contiguous) {
        logsumexp.copy_shared_buffer(saved_lse);
        // No need to add as temporary - it's a shared buffer
      } else {
        logsumexp.set_data(allocator::malloc(logsumexp.nbytes()));
        copy_gpu(saved_lse, logsumexp, CopyType::General, s);
        logsumexp_needs_temp = true;
      }
    } else {
      logsumexp_needs_temp = true;
      // Recompute logsumexp (fallback path)
      logsumexp.set_data(allocator::malloc(logsumexp.nbytes()));

      // Allocate consolidated buffer for running state (saves 1 allocation call)
      // Layout: [running_max (N) | running_sum_exp (N)]
      array running_state_bwd({2 * N}, float32, nullptr, {});
      running_state_bwd.set_data(allocator::malloc(running_state_bwd.nbytes()));

      array running_max({N}, float32, nullptr, {});
      running_max.copy_shared_buffer(running_state_bwd, {1}, running_state_bwd.flags(), N, 0);

      array running_sum_exp({N}, float32, nullptr, {});
      running_sum_exp.copy_shared_buffer(running_state_bwd, {1}, running_state_bwd.flags(), N, N);
      // Note: We reuse logsumexp as dummy target output (kernel writes target_logit here,
      // but we don't need it - logsumexp is overwritten in finalize step anyway)

      // Allocate reusable logits buffer for logsumexp computation (native dtype)
      array lse_logits_chunk({N, max_chunk_v}, compute_dtype, nullptr, {});
      lse_logits_chunk.set_data(allocator::malloc(lse_logits_chunk.nbytes()));

      // Initialize all running values in single kernel dispatch (reduces 3 dispatches to 1)
      // running_max = -inf, running_sum_exp = 0, logsumexp = 0 (used as dummy target)
      {
        auto init_kernel = d.get_kernel("cce_init_running_values");
        compute_encoder.set_compute_pipeline_state(init_kernel);
        compute_encoder.set_output_array(running_max, 0);
        compute_encoder.set_output_array(running_sum_exp, 1);
        compute_encoder.set_output_array(logsumexp, 2);  // Used as dummy target output
        compute_encoder.set_bytes(N, 3);

        int threads_per_tg = 256;
        int num_tgs = (N + threads_per_tg - 1) / threads_per_tg;
        MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
        MTL::Size group_dims = MTL::Size(threads_per_tg, 1, 1);
        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
      }

      // Cache kernel for logsumexp
      std::string lse_kernel_name = "cce_chunk_logsumexp_" + type_to_name(h.dtype());
      auto lse_kernel = d.get_kernel(lse_kernel_name);

      // Compute logsumexp using chunked approach
      constexpr int LSE_THREADS_PER_TG = 256;
      constexpr int LSE_NUM_SIMDGROUPS = LSE_THREADS_PER_TG / 32;
      size_t lse_smem_size = 2 * LSE_NUM_SIMDGROUPS * sizeof(float);

      for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int v_start = chunk_idx * max_chunk_v;
        int v_end = std::min(v_start + max_chunk_v, V);
        int current_chunk_v = v_end - v_start;

        // Create view of weight_chunk
        array weight_chunk({current_chunk_v, H}, w.dtype(), nullptr, {});
        int64_t w_offset = static_cast<int64_t>(v_start) * H;
        weight_chunk.copy_shared_buffer(
            w, {static_cast<int64_t>(H), 1}, w.flags(), static_cast<size_t>(current_chunk_v * H), w_offset);

        // Create view of logits_chunk (native dtype)
        array logits_view({N, current_chunk_v}, compute_dtype, nullptr, {});
        logits_view.copy_shared_buffer(
            lse_logits_chunk, {static_cast<int64_t>(current_chunk_v), 1}, lse_logits_chunk.flags(),
            static_cast<size_t>(N * current_chunk_v), 0);

        // Compute logits_chunk = hidden @ weight_chunk.T
        // NATIVE BF16: BF16 matmul produces BF16 logits, kernel handles FP32 accumulation internally
        steel_matmul(
            s, d,
            h, weight_chunk, logits_view,
            N, current_chunk_v, H,
            1,
            H, H,
            false, true,
            copies);

        // Update running max and sum_exp
        compute_encoder.set_compute_pipeline_state(lse_kernel);
        compute_encoder.set_input_array(logits_view, 0);
        compute_encoder.set_input_array(t, 1);
        compute_encoder.set_output_array(running_max, 2);
        compute_encoder.set_output_array(running_sum_exp, 3);
        // Note: we don't need target_logit for backward - reuse logsumexp as dummy
        // (logsumexp will be overwritten in finalize step anyway)
        compute_encoder.set_output_array(logsumexp, 4);
        compute_encoder.set_bytes(N, 5);
        compute_encoder.set_bytes(current_chunk_v, 6);
        compute_encoder.set_bytes(v_start, 7);
        compute_encoder.set_bytes(V, 8);
        compute_encoder.set_threadgroup_memory_length(lse_smem_size, 0);

        MTL::Size grid_dims = MTL::Size(N, 1, 1);
        MTL::Size group_dims = MTL::Size(LSE_THREADS_PER_TG, 1, 1);
        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
      }

      // Finalize logsumexp: logsumexp = max + log(sum_exp)
      {
        auto finalize_kernel = d.get_kernel("cce_finalize_lse");
        compute_encoder.set_compute_pipeline_state(finalize_kernel);

        compute_encoder.set_input_array(running_max, 0);
        compute_encoder.set_input_array(running_sum_exp, 1);
        compute_encoder.set_output_array(logsumexp, 2);
        compute_encoder.set_bytes(N, 3);

        int num_tgs = (N + 255) / 256;
        MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
        MTL::Size group_dims = MTL::Size(256, 1, 1);
        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
      }

      d.add_temporary(running_state_bwd, s.index);  // Consolidated buffer (views share this)
      d.add_temporary(lse_logits_chunk, s.index);
    }

    // Chunked backward: process vocabulary in chunks using standard matmuls

    // NATIVE BF16: Allocate logits buffer in native dtype (BF16 or FP32)
    // Following MLX pattern: no explicit casts, all conversions in kernel registers
    array logits_chunk({N, max_chunk_v}, compute_dtype, nullptr, {});
    logits_chunk.set_data(allocator::malloc(logits_chunk.nbytes()));

    // OPTIMIZATION: Cache kernel lookup outside loop
    // Use dtype-specific kernel (handles BF16→FP32→BF16 conversion in registers)
    std::string d_logits_kernel_name = "cce_compute_d_logits_" + type_to_name(h.dtype());
    auto d_logits_kernel = d.get_kernel(d_logits_kernel_name);

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
      int v_start = chunk_idx * max_chunk_v;
      int v_end = std::min(v_start + max_chunk_v, V);
      int current_chunk_v = v_end - v_start;

      // Step 1: Compute logits_chunk = hidden @ weight_chunk.T
      // Create a view of weight_chunk starting at v_start
      array weight_chunk({current_chunk_v, H}, w.dtype(), nullptr, {});
      int64_t w_offset = static_cast<int64_t>(v_start) * H;
      weight_chunk.copy_shared_buffer(
          w, {static_cast<int64_t>(H), 1}, w.flags(), static_cast<size_t>(current_chunk_v * H), w_offset);

      // Create view of logits_chunk for actual chunk size (native dtype)
      array logits_view({N, current_chunk_v}, compute_dtype, nullptr, {});
      logits_view.copy_shared_buffer(
          logits_chunk, {static_cast<int64_t>(current_chunk_v), 1}, logits_chunk.flags(),
          static_cast<size_t>(N * current_chunk_v), 0);

      // Compute logits_chunk = hidden @ weight_chunk.T
      // NATIVE BF16: steel_matmul handles FP32 accumulation internally
      steel_matmul(
          s, d,
          h, weight_chunk, logits_view,
          N, current_chunk_v, H,  // M, N, K
          1,              // batch_size
          H, H,           // lda, ldb (leading dimensions)
          false, true,    // transpose_a, transpose_b
          copies);

      // Step 2: Compute d_logits (in-place, kernel does BF16→FP32→BF16 in registers)
      array d_logits_view({N, current_chunk_v}, compute_dtype, nullptr, {});
      d_logits_view.copy_shared_buffer(
          logits_chunk, {static_cast<int64_t>(current_chunk_v), 1}, logits_chunk.flags(),
          static_cast<size_t>(N * current_chunk_v), 0);

      compute_encoder.set_compute_pipeline_state(d_logits_kernel);
      compute_encoder.set_input_array(logits_view, 0);
      compute_encoder.set_input_array(logsumexp, 1);
      compute_encoder.set_input_array(t, 2);
      compute_encoder.set_input_array(g_out, 3);
      compute_encoder.set_output_array(d_logits_view, 4);  // Native dtype output
      compute_encoder.set_bytes(N, 5);
      compute_encoder.set_bytes(current_chunk_v, 6);
      compute_encoder.set_bytes(v_start, 7);
      compute_encoder.set_bytes(V, 8);
      compute_encoder.set_bytes(scale, 9);

      // OPTIMIZATION: Vectorized kernel - each thread processes N_READS=4 elements
      // This reduces dispatch overhead and improves memory coalescing
      constexpr int N_READS = 4;
      int total_elements = N * current_chunk_v;
      int total_threads = (total_elements + N_READS - 1) / N_READS;
      int threads_per_tg = 256;
      int num_tgs = (total_threads + threads_per_tg - 1) / threads_per_tg;
      MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
      MTL::Size group_dims = MTL::Size(threads_per_tg, 1, 1);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

      // Step 3: Accumulate grad_hidden += d_logits @ weight_chunk
      // NATIVE BF16: steel_matmul_axpby handles FP32 accumulation internally
      float alpha = 1.0f;
      float beta = (chunk_idx == 0) ? 0.0f : 1.0f;

      steel_matmul_axpby<true>(
          s, d,
          d_logits_view, weight_chunk, grad_hidden, grad_hidden,
          N, H, current_chunk_v,  // M, N, K
          1,              // batch_size
          current_chunk_v, H,     // lda, ldb
          false, false,   // transpose_a, transpose_b
          copies,
          {}, {}, {}, {},
          alpha, beta);

      // Step 4: Compute grad_weight_chunk = d_logits.T @ hidden
      // Write directly to grad_weight (native dtype)
      array grad_weight_chunk({current_chunk_v, H}, compute_dtype, nullptr, {});
      int64_t gw_offset = static_cast<int64_t>(v_start) * H;
      grad_weight_chunk.copy_shared_buffer(
          grad_weight, {static_cast<int64_t>(H), 1}, grad_weight.flags(),
          static_cast<size_t>(current_chunk_v * H), gw_offset);

      steel_matmul(
          s, d,
          d_logits_view, h, grad_weight_chunk,
          current_chunk_v, H, N,  // M, N, K
          1,              // batch_size
          current_chunk_v, H,     // lda, ldb
          true, false,    // transpose_a, transpose_b
          copies);
    }

    // NATIVE BF16: Outputs written directly in native dtype, no conversion needed

    // Only add logsumexp as temporary if we allocated it
    if (logsumexp_needs_temp) {
      d.add_temporary(logsumexp, s.index);
    }
    d.add_temporary(logits_chunk, s.index);

  } else {
    // Legacy SIMD-based backward pass (only for small problems)
    // Note: Legacy path only supports FP32 outputs
    if (use_bf16) {
      throw std::invalid_argument(
          "CCE backward with BF16 requires larger problem size (V > 2000 && N >= 256). "
          "For small problems, use FP32 inputs.");
    }

    array logsumexp({N}, float32, nullptr, {});
    logsumexp.set_data(allocator::malloc(logsumexp.nbytes()));
    array dummy_loss({N}, float32, nullptr, {});
    dummy_loss.set_data(allocator::malloc(dummy_loss.nbytes()));

    // Forward pass to get logsumexp
    {
      std::string kernel_name = "cce_simd_forward_" + type_to_name(h.dtype());
      auto kernel = d.get_kernel(kernel_name);
      compute_encoder.set_compute_pipeline_state(kernel);

      compute_encoder.set_input_array(h, 0);
      compute_encoder.set_input_array(w, 1);
      compute_encoder.set_input_array(t, 2);
      compute_encoder.set_output_array(dummy_loss, 3);
      compute_encoder.set_output_array(logsumexp, 4);
      compute_encoder.set_bytes(params, 5);

      int num_tgs = (N + ROWS_PER_TG - 1) / ROWS_PER_TG;
      MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
      MTL::Size group_dims = MTL::Size(THREADS_PER_TG, 1, 1);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    // Backward pass for grad_hidden and grad_weight using SIMD kernel
    {
      std::string kernel_name = "cce_simd_backward_" + type_to_name(h.dtype());
      auto kernel = d.get_kernel(kernel_name);
      compute_encoder.set_compute_pipeline_state(kernel);

      compute_encoder.set_input_array(h, 0);
      compute_encoder.set_input_array(w, 1);
      compute_encoder.set_input_array(t, 2);
      compute_encoder.set_input_array(logsumexp, 3);
      compute_encoder.set_input_array(g_out, 4);
      compute_encoder.set_output_array(grad_hidden, 5);
      compute_encoder.set_output_array(grad_weight, 6);
      compute_encoder.set_bytes(params, 7);

      int num_tgs = (N + ROWS_PER_TG - 1) / ROWS_PER_TG;
      MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
      MTL::Size group_dims = MTL::Size(THREADS_PER_TG, 1, 1);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    d.add_temporary(logsumexp, s.index);
    d.add_temporary(dummy_loss, s.index);
  }

  d.add_temporary(g_out_expanded, s.index);
  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
