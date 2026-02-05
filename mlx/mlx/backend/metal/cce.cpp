// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace mlx::core::fast {

constexpr int MAX_CHUNK_V = 16384;
constexpr int MIN_CHUNK_V = 1024;

inline int get_adaptive_chunk_v(int N, int V, int H) {
  if (N <= 2048) {
    return std::min(MAX_CHUNK_V, V);
  }

  constexpr size_t BYTES_PER_ELEMENT = 4;

  size_t system_memory = 64ULL * 1024 * 1024 * 1024;
#if defined(__APPLE__)
  size_t size = sizeof(system_memory);
  if (sysctlbyname("hw.memsize", &system_memory, &size, nullptr, 0) != 0) {
    system_memory = 64ULL * 1024 * 1024 * 1024;
  }
#endif

  size_t chunk_budget = system_memory / 200;
  int max_chunk_from_memory = static_cast<int>(chunk_budget / (static_cast<size_t>(N) * BYTES_PER_ELEMENT));
  int chunk_v = std::min({MAX_CHUNK_V, V, std::max(MIN_CHUNK_V, max_chunk_from_memory)});
  chunk_v = (chunk_v / 256) * 256;

  if (chunk_v < MIN_CHUNK_V) {
    chunk_v = std::min(MIN_CHUNK_V, V);
  }

  return chunk_v;
}

void CCELoss::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& hidden_in = inputs[0];
  const array& weight_in = inputs[1];
  const array& targets_in = inputs[2];

  int N, H, V;
  if (hidden_in.ndim() == 3) {
    N = hidden_in.shape(0) * hidden_in.shape(1);
    H = hidden_in.shape(2);
  } else {
    N = hidden_in.shape(0);
    H = hidden_in.shape(1);
  }
  V = weight_in.shape(0);

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

  bool use_bf16_matmul = (h.dtype() == bfloat16 && w.dtype() == bfloat16);
  bool use_fp16_matmul = (h.dtype() == float16 && w.dtype() == float16);
  bool use_fp32_matmul = (h.dtype() == float32 && w.dtype() == float32);

  if (!use_bf16_matmul && !use_fp16_matmul && !use_fp32_matmul) {
    throw std::invalid_argument(
        "CCE requires matching input dtypes. Supported: both float32, both bfloat16, or both float16.");
  }

  array& loss = outputs[0];
  loss.set_data(allocator::malloc(loss.nbytes()));

  bool output_lse = output_logsumexp() && outputs.size() >= 2;
  if (output_lse) {
    array& lse_out = outputs[1];
    lse_out.set_data(allocator::malloc(lse_out.nbytes()));
  }

  float scale = 1.0f;
  auto& compute_encoder = d.get_command_encoder(s.index);

  {
    int adaptive_chunk_v = get_adaptive_chunk_v(N, V, H);
    int num_chunks = (V + adaptive_chunk_v - 1) / adaptive_chunk_v;
    int max_chunk_v = std::min(adaptive_chunk_v, V);

    Dtype logits_dtype = use_bf16_matmul ? bfloat16 : (use_fp16_matmul ? float16 : float32);

    array logits_chunk({N, max_chunk_v}, logits_dtype, nullptr, {});
    logits_chunk.set_data(allocator::malloc(logits_chunk.nbytes()));

    array running_state({3 * N}, float32, nullptr, {});
    running_state.set_data(allocator::malloc(running_state.nbytes()));

    array running_max({N}, float32, nullptr, {});
    running_max.copy_shared_buffer(running_state, {1}, running_state.flags(), N, 0);

    array running_sum_exp({N}, float32, nullptr, {});
    running_sum_exp.copy_shared_buffer(running_state, {1}, running_state.flags(), N, N);

    array target_logit({N}, float32, nullptr, {});
    target_logit.copy_shared_buffer(running_state, {1}, running_state.flags(), N, 2 * N);

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

    std::string lse_kernel_name = "cce_chunk_logsumexp_" + type_to_name(h.dtype());
    auto lse_kernel = d.get_kernel(lse_kernel_name);

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
      int v_start = chunk_idx * max_chunk_v;
      int v_end = std::min(v_start + max_chunk_v, V);
      int current_chunk_v = v_end - v_start;

      array weight_chunk({current_chunk_v, H}, w.dtype(), nullptr, {});
      int64_t w_offset = static_cast<int64_t>(v_start) * H;
      weight_chunk.copy_shared_buffer(
          w, {static_cast<int64_t>(H), 1}, w.flags(), static_cast<size_t>(current_chunk_v * H), w_offset);

      array logits_view({N, current_chunk_v}, logits_dtype, nullptr, {});
      logits_view.copy_shared_buffer(
          logits_chunk, {static_cast<int64_t>(current_chunk_v), 1}, logits_chunk.flags(),
          static_cast<size_t>(N * current_chunk_v), 0);

      steel_matmul(
          s, d,
          h, weight_chunk, logits_view,
          N, current_chunk_v, H,
          1,
          H, H,
          false, true,
          copies);

      float softcap = logit_softcap();
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
      compute_encoder.set_bytes(softcap, 9);

      constexpr int THREADS_PER_TG = 256;
      constexpr int NUM_SIMDGROUPS = THREADS_PER_TG / 32;
      size_t smem_size = 2 * NUM_SIMDGROUPS * sizeof(float);
      compute_encoder.set_threadgroup_memory_length(smem_size, 0);

      MTL::Size grid_dims = MTL::Size(N, 1, 1);
      MTL::Size group_dims = MTL::Size(THREADS_PER_TG, 1, 1);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    if (output_lse) {
      auto final_kernel = d.get_kernel("cce_finalize_loss_with_lse");
      compute_encoder.set_compute_pipeline_state(final_kernel);

      compute_encoder.set_input_array(running_max, 0);
      compute_encoder.set_input_array(running_sum_exp, 1);
      compute_encoder.set_input_array(target_logit, 2);
      compute_encoder.set_input_array(t, 3);
      compute_encoder.set_output_array(loss, 4);
      compute_encoder.set_output_array(outputs[1], 5);
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
    d.add_temporary(running_state, s.index);
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

  int N, H, V;
  if (hidden_in.ndim() == 3) {
    N = hidden_in.shape(0) * hidden_in.shape(1);
    H = hidden_in.shape(2);
  } else {
    N = hidden_in.shape(0);
    H = hidden_in.shape(1);
  }
  V = weight_in.shape(0);

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

  bool use_bf16 = (h.dtype() == bfloat16 && w.dtype() == bfloat16);
  bool use_fp16 = (h.dtype() == float16 && w.dtype() == float16);
  bool use_fp32 = (h.dtype() == float32 && w.dtype() == float32);

  if (!use_bf16 && !use_fp16 && !use_fp32) {
    throw std::invalid_argument(
        "CCE backward requires matching input dtypes. Supported: both float32, both bfloat16, or both float16.");
  }

  Dtype compute_dtype = use_bf16 ? bfloat16 : (use_fp16 ? float16 : float32);

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

  grad_hidden.set_data(allocator::malloc(grad_hidden.nbytes()));
  grad_weight.set_data(allocator::malloc(grad_weight.nbytes()));

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
  auto& compute_encoder = d.get_command_encoder(s.index);

  {
    int adaptive_chunk_v = get_adaptive_chunk_v(N, V, H);
    int num_chunks = (V + adaptive_chunk_v - 1) / adaptive_chunk_v;
    int max_chunk_v = std::min(adaptive_chunk_v, V);

    bool use_saved_lse = has_logsumexp() && inputs.size() > 4;
    bool logsumexp_needs_temp = false;

    array logsumexp({N}, float32, nullptr, {});

    if (use_saved_lse) {
      const array& saved_lse = inputs[4];
      logsumexp.set_data(allocator::malloc(logsumexp.nbytes()));
      copy_gpu(saved_lse, logsumexp, CopyType::General, s);
      logsumexp_needs_temp = true;
    } else {
      logsumexp_needs_temp = true;
      logsumexp.set_data(allocator::malloc(logsumexp.nbytes()));

      array running_state_bwd({2 * N}, float32, nullptr, {});
      running_state_bwd.set_data(allocator::malloc(running_state_bwd.nbytes()));

      array running_max({N}, float32, nullptr, {});
      running_max.copy_shared_buffer(running_state_bwd, {1}, running_state_bwd.flags(), N, 0);

      array running_sum_exp({N}, float32, nullptr, {});
      running_sum_exp.copy_shared_buffer(running_state_bwd, {1}, running_state_bwd.flags(), N, N);

      array lse_logits_chunk({N, max_chunk_v}, compute_dtype, nullptr, {});
      lse_logits_chunk.set_data(allocator::malloc(lse_logits_chunk.nbytes()));

      {
        auto init_kernel = d.get_kernel("cce_init_running_values");
        compute_encoder.set_compute_pipeline_state(init_kernel);
        compute_encoder.set_output_array(running_max, 0);
        compute_encoder.set_output_array(running_sum_exp, 1);
        compute_encoder.set_output_array(logsumexp, 2);
        compute_encoder.set_bytes(N, 3);

        int threads_per_tg = 256;
        int num_tgs = (N + threads_per_tg - 1) / threads_per_tg;
        MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
        MTL::Size group_dims = MTL::Size(threads_per_tg, 1, 1);
        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
      }

      std::string lse_kernel_name = "cce_chunk_logsumexp_" + type_to_name(h.dtype());
      auto lse_kernel = d.get_kernel(lse_kernel_name);

      constexpr int LSE_THREADS_PER_TG = 256;
      constexpr int LSE_NUM_SIMDGROUPS = LSE_THREADS_PER_TG / 32;
      size_t lse_smem_size = 2 * LSE_NUM_SIMDGROUPS * sizeof(float);

      for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int v_start = chunk_idx * max_chunk_v;
        int v_end = std::min(v_start + max_chunk_v, V);
        int current_chunk_v = v_end - v_start;

        array weight_chunk({current_chunk_v, H}, w.dtype(), nullptr, {});
        int64_t w_offset = static_cast<int64_t>(v_start) * H;
        weight_chunk.copy_shared_buffer(
            w, {static_cast<int64_t>(H), 1}, w.flags(), static_cast<size_t>(current_chunk_v * H), w_offset);

        array logits_view({N, current_chunk_v}, compute_dtype, nullptr, {});
        logits_view.copy_shared_buffer(
            lse_logits_chunk, {static_cast<int64_t>(current_chunk_v), 1}, lse_logits_chunk.flags(),
            static_cast<size_t>(N * current_chunk_v), 0);

        steel_matmul(
            s, d,
            h, weight_chunk, logits_view,
            N, current_chunk_v, H,
            1,
            H, H,
            false, true,
            copies);

        float softcap = logit_softcap();
        compute_encoder.set_compute_pipeline_state(lse_kernel);
        compute_encoder.set_input_array(logits_view, 0);
        compute_encoder.set_input_array(t, 1);
        compute_encoder.set_output_array(running_max, 2);
        compute_encoder.set_output_array(running_sum_exp, 3);
        compute_encoder.set_output_array(logsumexp, 4);
        compute_encoder.set_bytes(N, 5);
        compute_encoder.set_bytes(current_chunk_v, 6);
        compute_encoder.set_bytes(v_start, 7);
        compute_encoder.set_bytes(V, 8);
        compute_encoder.set_bytes(softcap, 9);
        compute_encoder.set_threadgroup_memory_length(lse_smem_size, 0);

        MTL::Size grid_dims = MTL::Size(N, 1, 1);
        MTL::Size group_dims = MTL::Size(LSE_THREADS_PER_TG, 1, 1);
        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
      }

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

      d.add_temporary(running_state_bwd, s.index);
      d.add_temporary(lse_logits_chunk, s.index);
    }

    array logits_chunk({N, max_chunk_v}, compute_dtype, nullptr, {});
    logits_chunk.set_data(allocator::malloc(logits_chunk.nbytes()));

    const bool needs_separate_d_logits = (use_bf16 || use_fp16) && !metal::is_nax_available();
    array d_logits_chunk({N, max_chunk_v}, compute_dtype, nullptr, {});
    if (needs_separate_d_logits) {
      d_logits_chunk.set_data(allocator::malloc(d_logits_chunk.nbytes()));
    }

    std::string d_logits_kernel_name = "cce_compute_d_logits_" + type_to_name(h.dtype());
    auto d_logits_kernel = d.get_kernel(d_logits_kernel_name);

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
      int v_start = chunk_idx * max_chunk_v;
      int v_end = std::min(v_start + max_chunk_v, V);
      int current_chunk_v = v_end - v_start;

      array weight_chunk({current_chunk_v, H}, w.dtype(), nullptr, {});
      int64_t w_offset = static_cast<int64_t>(v_start) * H;
      weight_chunk.copy_shared_buffer(
          w, {static_cast<int64_t>(H), 1}, w.flags(), static_cast<size_t>(current_chunk_v * H), w_offset);

      array logits_view({N, current_chunk_v}, compute_dtype, nullptr, {});
      logits_view.copy_shared_buffer(
          logits_chunk, {static_cast<int64_t>(current_chunk_v), 1}, logits_chunk.flags(),
          static_cast<size_t>(N * current_chunk_v), 0);

      steel_matmul(
          s, d,
          h, weight_chunk, logits_view,
          N, current_chunk_v, H,
          1,
          H, H,
          false, true,
          copies);

      array d_logits_view({N, current_chunk_v}, compute_dtype, nullptr, {});
      if (needs_separate_d_logits) {
        d_logits_view.copy_shared_buffer(
            d_logits_chunk, {static_cast<int64_t>(current_chunk_v), 1}, d_logits_chunk.flags(),
            static_cast<size_t>(N * current_chunk_v), 0);
      } else {
        d_logits_view.copy_shared_buffer(
            logits_chunk, {static_cast<int64_t>(current_chunk_v), 1}, logits_chunk.flags(),
            static_cast<size_t>(N * current_chunk_v), 0);
      }

      float softcap_bwd = logit_softcap();
      compute_encoder.set_compute_pipeline_state(d_logits_kernel);
      compute_encoder.set_input_array(logits_view, 0);
      compute_encoder.set_input_array(logsumexp, 1);
      compute_encoder.set_input_array(t, 2);
      compute_encoder.set_input_array(g_out, 3);
      compute_encoder.set_output_array(d_logits_view, 4);
      compute_encoder.set_bytes(N, 5);
      compute_encoder.set_bytes(current_chunk_v, 6);
      compute_encoder.set_bytes(v_start, 7);
      compute_encoder.set_bytes(V, 8);
      compute_encoder.set_bytes(scale, 9);
      compute_encoder.set_bytes(softcap_bwd, 10);

      constexpr int N_READS = 4;
      int total_elements = N * current_chunk_v;
      int total_threads = (total_elements + N_READS - 1) / N_READS;
      int threads_per_tg = 256;
      int num_tgs = (total_threads + threads_per_tg - 1) / threads_per_tg;
      MTL::Size grid_dims = MTL::Size(num_tgs, 1, 1);
      MTL::Size group_dims = MTL::Size(threads_per_tg, 1, 1);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

      float alpha = 1.0f;
      float beta = (chunk_idx == 0) ? 0.0f : 1.0f;

      steel_matmul_axpby<true>(
          s, d,
          d_logits_view, weight_chunk, grad_hidden, grad_hidden,
          N, H, current_chunk_v,
          1,
          current_chunk_v, H,
          false, false,
          copies,
          {}, {}, {}, {},
          alpha, beta);

      array grad_weight_chunk({current_chunk_v, H}, compute_dtype, nullptr, {});
      int64_t gw_offset = static_cast<int64_t>(v_start) * H;
      grad_weight_chunk.copy_shared_buffer(
          grad_weight, {static_cast<int64_t>(H), 1}, grad_weight.flags(),
          static_cast<size_t>(current_chunk_v * H), gw_offset);

      steel_matmul(
          s, d,
          d_logits_view, h, grad_weight_chunk,
          current_chunk_v, H, N,
          1,
          current_chunk_v, H,
          true, false,
          copies);
    }

    if (logsumexp_needs_temp) {
      d.add_temporary(logsumexp, s.index);
    }
    d.add_temporary(logits_chunk, s.index);
    if (needs_separate_d_logits) {
      d.add_temporary(d_logits_chunk, s.index);
    }
  }

  d.add_temporary(g_out_expanded, s.index);
  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
