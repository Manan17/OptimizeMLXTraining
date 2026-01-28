// Copyright © 2023-2024 Apple Inc.
#include <cassert>
#include <numeric>

#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core::fast {

std::vector<array> Custom::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto [_, vjps] = mlx::core::vjp(fallback_, primals, cotangents);
  std::vector<array> vjp_outs;
  for (int i = 0, j = 0; i < vjps.size(); ++i) {
    if (j < argnums.size() && i == argnums[j]) {
      vjp_outs.push_back(vjps[i]);
      j++;
    }
  }
  return vjp_outs;
}

std::vector<array> Custom::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  std::vector<array> all_tangents;
  for (int i = 0, j = 0; i < primals.size(); i++) {
    if (j < argnums.size() && i == argnums[j]) {
      all_tangents.emplace_back(tangents[j++]);
    } else {
      all_tangents.emplace_back(zeros_like(primals[i]));
    }
  }
  auto [_, jvps] = mlx::core::jvp(fallback_, primals, all_tangents);
  return jvps;
}

std::pair<std::vector<array>, std::vector<int>> Custom::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto outputs = mlx::core::vmap(fallback_, axes)(inputs);
  auto out_axes = std::vector<int>(outputs.size(), 0);
  return {outputs, out_axes};
}

array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  bool has_weight = weight.has_value();

  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[rms_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (has_weight) {
    if ((*weight).ndim() != 1) {
      std::ostringstream msg;
      msg << "[rms_norm] (*weight) must have 1 dimension but has "
          << (*weight).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*weight).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[rms_norm] (*weight) must have the same size as the last dimension of"
             " x but has "
          << (*weight).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }

  auto out_type = (weight.has_value()) ? result_type(x, (*weight)) : x.dtype();
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[rms_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto fallback =
      [has_weight, eps, out_type, s](const std::vector<array>& inputs) {
        auto x = astype(inputs[0], float32, s);
        x = multiply(
            x,
            rsqrt(
                add(mean(square(x, s), -1, /* keepdims */ true, s),
                    array(eps, float32),
                    s),
                s),
            s);
        x = astype(x, out_type, s);

        if (has_weight) {
          x = multiply(x, inputs[1], s);
        }

        return std::vector<array>{x};
      };

  auto passed_weight =
      (has_weight) ? astype(*weight, out_type, s) : array(1, out_type);

  if (!RMSNorm::use_fallback(s)) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<RMSNorm>(s, fallback, eps),
        {astype(x, out_type, s), passed_weight});
  }
  return fallback({x, passed_weight})[0];
}

std::vector<array> RMSNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 2);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& g = inputs[2];

    std::vector<array> vjps;

    auto n = rsqrt(
        add(mean(square(x, s), /* axis= */ -1, /* keepdims= */ true, s),
            array(eps, x.dtype()),
            s),
        s);
    auto n3 = power(n, array(3, x.dtype()), s);

    // df/dx
    auto gw = multiply(g, w, s);
    auto t = mean(multiply(gw, x, s), /* axis= */ -1, /* keepdims= */ true, s);
    t = multiply(multiply(x, t, s), n3, s);
    vjps.push_back(subtract(multiply(gw, n, s), t, s));

    // df/dw
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    if (w.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(
          multiply(g, multiply(x, n, s), s), axes, /* keepdims= */ false, s));
    }

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape()},
      {primals[0].dtype(), primals[1].dtype()},
      std::make_shared<RMSNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool RMSNorm::is_equivalent(const Primitive& other) const {
  const RMSNorm& a_other = static_cast<const RMSNorm&>(other);
  return eps_ == a_other.eps_;
}

bool RMSNormVJP::is_equivalent(const Primitive& other) const {
  const RMSNormVJP& a_other = static_cast<const RMSNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  bool has_weight = weight.has_value();
  bool has_bias = bias.has_value();

  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[layer_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (has_weight) {
    if ((*weight).ndim() != 1) {
      std::ostringstream msg;
      msg << "[layer_norm] weight must have 1 dimension but has "
          << (*weight).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*weight).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[layer_norm] weight must have the same size as the last dimension of"
             " x but has "
          << (*weight).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }
  if (has_bias) {
    if ((*bias).ndim() != 1) {
      std::ostringstream msg;
      msg << "[layer_norm] bias must have 1 dimension but has "
          << (*bias).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*bias).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[layer_norm] bias must have the same size as the last dimension of"
             " x but has "
          << (*bias).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }

  auto out_type = (has_weight)
      ? ((has_bias) ? result_type(x, *weight, *bias) : result_type(x, *weight))
      : x.dtype();
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[layer_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto fallback = [has_weight, has_bias, eps, out_type, s](
                      const std::vector<array>& inputs) {
    auto x = astype(inputs[0], float32, s);

    auto mu = mean(x, /* axis= */ -1, /* keepdims= */ true, s);
    auto xc = subtract(x, mu, s);
    auto v = mean(square(xc, s), /* axis= */ -1, /* keepdims= */ true, s);

    x = multiply(xc, rsqrt(add(v, array(eps, float32), s), s));
    x = astype(x, out_type, s);

    // If the LN is affine then transform x according to the weight and bias
    if (has_weight) {
      x = multiply(x, inputs[1], s);
    }
    if (has_bias) {
      x = add(x, inputs[2], s);
    }

    return std::vector<array>{x};
  };

  auto passed_weight =
      (has_weight) ? astype(*weight, out_type, s) : array(1, out_type);
  auto passed_bias =
      (has_bias) ? astype(*bias, out_type, s) : array(0, out_type);

  if (!LayerNorm::use_fallback(s)) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<LayerNorm>(s, fallback, eps),
        {astype(x, out_type, s), passed_weight, passed_bias});
  }
  return fallback({x, passed_weight, passed_bias})[0];
}

std::vector<array> LayerNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 3);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& b = inputs[2];
    auto& g = inputs[3];

    std::vector<array> vjps;

    auto norm = number_of_elements(x, {-1}, true, x.dtype(), s);
    auto sumx = sum(x, /* axis= */ -1, /* keepdims= */ true, s);
    auto sumx2 = sum(square(x, s), /* axis= */ -1, /* keepdims= */ true, s);
    auto mu = multiply(sumx, norm, s);
    auto mu2 = multiply(sumx2, norm, s);
    auto var = subtract(mu2, square(mu, s), s);
    auto n = rsqrt(add(var, array(eps, x.dtype()), s));
    auto n3 = power(n, array(3, x.dtype()), s);
    auto x_c = subtract(x, mu, s);

    // df/dx
    auto wg = multiply(w, g, s);
    auto sumwg =
        multiply(sum(wg, /* axis= */ -1, /* keepdims= */ true, s), norm, s);
    auto sumwgxc = multiply(
        sum(multiply(wg, x_c, s), /* axis= */ -1, /* keepdims= */ true, s),
        norm,
        s);
    auto t1 = multiply(multiply(x_c, sumwgxc, s), n3, s);
    auto t2 = multiply(subtract(wg, sumwg, s), n, s);
    vjps.push_back(subtract(t2, t1, s));

    // df/dw
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    if (w.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(
          multiply(g, multiply(x_c, n, s), s), axes, /* keepdims= */ false, s));
    }

    // df/db
    if (b.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(g, axes, /* keepdims= */ false, s));
    }

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape(), primals[2].shape()},
      {primals[0].dtype(), primals[1].dtype(), primals[2].dtype()},
      std::make_shared<LayerNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], primals[2], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool LayerNorm::is_equivalent(const Primitive& other) const {
  const LayerNorm& a_other = static_cast<const LayerNorm&>(other);
  return eps_ == a_other.eps_;
}

bool LayerNormVJP::is_equivalent(const Primitive& other) const {
  const LayerNormVJP& a_other = static_cast<const LayerNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array rope(
    std::vector<array> inputs,
    int dims,
    bool traditional,
    float base,
    float scale,
    bool forward,
    StreamOrDevice s) {
  auto& x = inputs[0];
  auto& offset = inputs[1];
  if (x.ndim() < 3) {
    std::ostringstream msg;
    msg << "[rope] Input must have at least 3 dimensions but got input with "
        << x.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(x.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[rope] Input must be a floating type but got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.ndim() > 1) {
    std::ostringstream msg;
    msg << "[rope] offset must have at most one dimension but has shape "
        << offset.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.size() != 1 && offset.size() != x.shape(0)) {
    std::ostringstream msg;
    msg << "[rope] offset must be a scalar or vector with " << x.shape(0)
        << " elements but has shape " << offset.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(offset.dtype(), integer)) {
    std::ostringstream msg;
    msg << "[rope] offset must be an integer but got type " << offset.dtype()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.dtype().size() != 4) {
    inputs[1] = astype(offset, int32, s);
  }
  if (inputs.size() == 3 &&
      (inputs[2].ndim() != 1 || inputs[2].shape(0) != dims / 2)) {
    std::ostringstream msg;
    msg << "[rope] freqs must be one dimensional with size " << dims / 2
        << " but got shape " << inputs[2].shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto fallback = [dims, traditional, base, scale, forward, s](
                      std::vector<array> inputs) {
    auto x = inputs[0];
    auto shape = x.shape();
    if (x.ndim() == 3) {
      x = expand_dims(x, 1, s);
    } else if (x.ndim() > 4) {
      x = flatten(x, 1, 1 + (x.ndim() - 4), s);
    }

    auto B = x.shape(0);
    auto N = x.shape(1);
    auto T = x.shape(2);
    auto t = x.dtype();
    // Compute sines and cosines
    auto half_dims = dims / 2;
    auto offset = inputs[1];
    if (offset.size() > 1) {
      offset = expand_dims(offset, {-1, -2}, s);
    }
    auto positions = multiply(
        add(arange(x.shape(2), float32, s), offset, s),
        array(scale, float32),
        s);

    auto default_inv_freqs = [&s, base, half_dims]() {
      return exp(
          multiply(
              arange(0, -half_dims, -1, float32, s),
              array(std::log(base) / half_dims, float32),
              s),
          s);
    };

    auto inv_freqs =
        inputs.size() == 3 ? reciprocal(inputs[2], s) : default_inv_freqs();
    auto theta = multiply(expand_dims(positions, -1, s), inv_freqs, s);
    auto coss = astype(cos(theta, s), t, s);
    auto sins = astype(sin(theta, s), t, s);

    auto apply_rope = [forward, s](
                          const array& x1,
                          const array& x2,
                          const array& coss,
                          const array& sins) {
      std::vector<array> outs;
      if (forward) {
        outs.push_back(
            subtract(multiply(x1, coss, s), multiply(x2, sins, s), s));
        outs.push_back(add(multiply(x1, sins, s), multiply(x2, coss, s), s));
      } else {
        outs.push_back(add(multiply(x2, sins, s), multiply(x1, coss, s), s));
        outs.push_back(
            subtract(multiply(x2, coss, s), multiply(x1, sins, s), s));
      }
      return outs;
    };

    if (traditional) {
      auto x1 = slice(x, {0, 0, 0, 0}, {B, N, T, dims}, {1, 1, 1, 2}, s);
      auto x2 = slice(x, {0, 0, 0, 1}, {B, N, T, dims}, {1, 1, 1, 2}, s);
      auto outs = apply_rope(x1, x2, coss, sins);
      for (auto& o : outs) {
        o = expand_dims(o, -1, s);
      }
      auto out = reshape(concatenate(outs, -1, s), {B, N, T, dims}, s);
      if (dims < x.shape(-1)) {
        out =
            concatenate({out, slice(x, {0, 0, 0, dims}, x.shape(), s)}, -1, s);
      }
      return std::vector<array>{reshape(out, shape, s)};
    } else {
      auto out_s = x.shape();
      out_s.back() = half_dims;
      auto x1 = slice(x, {0, 0, 0, 0}, out_s, s);
      out_s.back() = dims;
      auto x2 = slice(x, {0, 0, 0, half_dims}, out_s, s);

      auto outs = apply_rope(x1, x2, coss, sins);
      if (dims < x.shape(-1)) {
        outs.push_back(slice(x, {0, 0, 0, dims}, x.shape(), s));
      }
      return std::vector<array>{reshape(concatenate(outs, -1, s), shape, s)};
    }
  };
  auto stream = to_stream(s);
  if (!RoPE::use_fallback(stream)) {
    return array(
        x.shape(),
        x.dtype(),
        std::make_shared<RoPE>(
            stream, fallback, dims, traditional, base, scale, forward),
        std::move(inputs));
  }
  return fallback(std::move(inputs))[0];
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  std::vector<array> inputs = {x, offset};
  if (freqs) {
    inputs.push_back(astype(*freqs, float32, s));
    if (base) {
      throw std::invalid_argument(
          "[rope] Only one of base or freqs can have a value.");
    }
  } else if (!base) {
    throw std::invalid_argument("[rope] Neither base nor freqs has a value.");
  }
  return rope(
      std::move(inputs),
      dims,
      traditional,
      base.has_value() ? *base : 1.0,
      scale,
      true,
      s);
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  return rope(
      x, dims, traditional, base, scale, array(offset, int32), freqs, s);
}

std::vector<array> RoPE::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto s = stream();
  auto fallback = [dims = dims_,
                   traditional = traditional_,
                   base = base_,
                   scale = scale_,
                   forward = forward_,
                   s](std::vector<array> inputs) {
    return std::vector<array>{
        rope(std::move(inputs), dims, traditional, base, scale, !forward, s)};
  };
  if (argnums.size() > 1 || argnums[0] != 0) {
    throw std::invalid_argument(
        "[RoPE::vjp] vjp for offset or frequencies not supported");
  }
  auto inputs = std::vector<array>{cotangents[0], primals[1]};
  if (primals.size() == 3) {
    inputs.push_back(primals[2]);
  }
  return {array(
      cotangents[0].shape(),
      cotangents[0].dtype(),
      std::make_shared<RoPE>(
          s, fallback, dims_, traditional_, base_, scale_, !forward_),
      std::move(inputs))};
}

bool RoPE::is_equivalent(const Primitive& other) const {
  const RoPE& a_other = static_cast<const RoPE&>(other);
  return (
      dims_ == a_other.dims_ && base_ == a_other.base_ &&
      scale_ == a_other.scale_ && traditional_ == a_other.traditional_ &&
      forward_ == a_other.forward_);
}

/** Computes: O = softmax(Q @ K.T) @ V **/
array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode /* = "" */,
    std::optional<array> mask_arr /* = {} */,
    const std::optional<array>& sinks /* = {} */,
    StreamOrDevice s /* = {}*/) {
  for (const auto& tensor : {queries, keys, values}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] input with shape "
          << tensor.shape() << " expected to be rank 4";
      throw std::invalid_argument(msg.str());
    }
  }
  // Check valid mask
  if (mask_mode != "" && mask_mode != "causal" && mask_mode != "array") {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Invalid mask_mode " << mask_mode
        << ". mask_mode must be 'causal', 'array' or ''.";
    throw std::invalid_argument(msg.str());
  }

  bool do_causal = false;
  bool has_mask = false;
  bool has_arr_mask = false;
  bool has_bool_mask = false;

  if (mask_mode == "causal") {
    has_mask = true;
    do_causal = true;

    if (mask_arr) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Invalid mask_arr for mask_mode "
          << "'casusal'. No array mask should be passed.";
      throw std::invalid_argument(msg.str());
    }
  } else if (mask_arr) {
    has_mask = true;
    has_arr_mask = true;
    has_bool_mask = mask_arr->dtype() == bool_;
  }

  if (has_arr_mask && mask_arr->ndim() > 4) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] the mask with shape "
        << mask_arr->shape() << " expected to have at most rank 4.";
    throw std::invalid_argument(msg.str());
  }

  const size_t batch_dim = queries.shape(0);
  for (const auto& tensor : {keys, values}) {
    if (tensor.shape(0) != batch_dim) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] mismatching batch dimension for input with shape "
          << tensor.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }

  // Q, K must have matching last dims (d_k aka 'head_dim');
  if (queries.shape(-1) != keys.shape(-1)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] query, keys expected to have matching last dimension; found query shape "
        << queries.shape() << " for keys shape " << keys.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // K, V must have matching number of heads (n_kv_heads);
  auto n_q_heads = queries.shape(-3);
  auto n_kv_heads = keys.shape(-3);

  if (keys.shape(-3) != values.shape(-3)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] keys, values expected to have matching n_kv_heads; found keys with n_heads "
        << keys.shape(-3) << " for values with n_heads " << values.shape(-3)
        << ".";
    throw std::invalid_argument(msg.str());
  }

  // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
  if (n_q_heads % n_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] n_heads must be a multiple of n_kv_heads, found n_heads "
        << n_q_heads << " for n_kv_heads " << n_kv_heads << ".";
    throw std::invalid_argument(msg.str());
  }

  auto final_type = result_type(queries, keys, values);
  if (!issubdtype(final_type, floating)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Received unsupported type "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  bool has_sinks = sinks.has_value();

  auto q = astype(queries, final_type, s);
  auto k = astype(keys, final_type, s);
  auto v = astype(values, final_type, s);

  auto fallback = [scale,
                   n_q_heads,
                   n_kv_heads,
                   do_causal,
                   has_sinks,
                   has_arr_mask,
                   s](const std::vector<array>& inputs) {
    auto q = multiply(array(scale, inputs[0].dtype()), inputs[0], s);
    int n_repeats = n_q_heads / n_kv_heads;
    auto k = inputs[1];
    auto v = inputs[2];
    if (n_repeats > 1) {
      q = unflatten(q, 1, {n_kv_heads, n_repeats}, s);
      k = expand_dims(k, 2, s);
      v = expand_dims(v, 2, s);
    }
    auto scores = matmul(q, swapaxes(k, -1, -2, s), s);
    if (has_arr_mask || do_causal) {
      // Mask must be broadcast-compatible with [B, n_q_heads, L_q, L_kv]
      auto make_or_fetch_mask = [&]() {
        if (do_causal) {
          int kL = k.shape(-2);
          int qL = q.shape(-2);
          int q_off = (kL - qL) < 0 ? 0 : (kL - qL);
          auto q_idx = arange(q_off, q_off + qL, s);
          auto k_idx = arange(0, kL, s);
          q_idx = expand_dims(q_idx, 1, s);
          k_idx = expand_dims(k_idx, 0, s);
          return greater_equal(q_idx, k_idx, s);
        }
        return inputs[3];
      };
      auto mask = make_or_fetch_mask();

      if (n_repeats > 1 && mask.ndim() >= 3) {
        if (mask.shape(-3) == 1) {
          mask = expand_dims(mask, -3, s);
        } else {
          mask = unflatten(mask, -3, {n_kv_heads, n_repeats}, s);
        }
      }
      if (mask.dtype() == bool_) {
        scores = where(
            mask, scores, array(finfo(scores.dtype()).min, scores.dtype()), s);
      } else {
        scores = add(scores, mask, s);
      }
    }
    if (has_sinks) {
      auto sinks = inputs.back();
      // scores has shape B N_q N_k L_q L_k
      sinks = expand_dims(sinks, {0, 2, 3}, s);
      if (scores.ndim() == 5) {
        sinks = unflatten(sinks, 1, {n_kv_heads, n_repeats}, s);
      }
      auto bsx_shape = scores.shape();
      bsx_shape.back() = 1;
      scores = concatenate({broadcast_to(sinks, bsx_shape, s), scores}, -1, s);
    }
    scores = softmax(scores, std::vector<int>{-1}, true, s);
    if (has_sinks) {
      // Slice off scores
      auto start = Shape(scores.ndim(), 0);
      start.back() = 1;
      auto stop = scores.shape();
      scores = slice(scores, std::move(start), std::move(stop), s);
    }
    auto out = matmul(scores, v, s);
    if (n_repeats > 1) {
      out = flatten(out, 1, 2, s);
    }
    return std::vector<array>{out};
  };

  auto stream = to_stream(s);
  std::vector<array> inputs = {q, k, v};
  if (has_arr_mask) {
    // Check type
    has_bool_mask = mask_arr->dtype() == bool_;
    if (promote_types(mask_arr->dtype(), final_type) != final_type) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Mask type must promote to output type "
          << final_type << ".";
      throw std::invalid_argument(msg.str());
    } else if (!has_bool_mask) {
      mask_arr = astype(*mask_arr, final_type, stream);
    }
    // Broadcast mask
    auto mask_shape = queries.shape();
    mask_shape.back() = keys.shape(-2);
    inputs.push_back(broadcast_to(*mask_arr, mask_shape, stream));
  }
  if (has_sinks) {
    if (promote_types(sinks->dtype(), final_type) != final_type) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Type of sinks must promote to output type "
          << final_type << ".";
      throw std::invalid_argument(msg.str());
    }
    if (sinks->ndim() != 1 || sinks->shape(0) != n_q_heads) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Received invalid shape for sinks "
          << sinks->shape() << ".";
      throw std::invalid_argument(msg.str());
    }
    inputs.push_back(astype(*sinks, final_type, stream));
  }

  bool is_training = detail::in_grad_tracing();
  bool has_fast_vjp = !ScaledDotProductAttentionVJP::use_fallback(q, stream);
  bool output_logsumexp = is_training && has_fast_vjp;
  if (!ScaledDotProductAttention::use_fallback(
          q,
          k,
          v,
          has_mask,
          has_arr_mask,
          do_causal,
          is_training,
          output_logsumexp,
          stream)) {
    if (has_bool_mask && !ScaledDotProductAttention::supports_bool_mask()) {
      // Convert bool mask to additive mask.
      float inf = std::numeric_limits<float>::infinity();
      array& mask = inputs[3];
      mask = where(
          mask,
          full_like(mask, 0, final_type, s),
          full_like(mask, -inf, final_type, s));
    }
    Shape out_shape{q.shape(0), q.shape(1), q.shape(2), v.shape(-1)};
    auto primitive = std::make_shared<ScaledDotProductAttention>(
        stream, fallback, scale, do_causal, has_sinks, output_logsumexp);
    if (output_logsumexp) {
      return array::make_arrays(
          {std::move(out_shape), Shape{q.shape(0), q.shape(1), q.shape(2), 1}},
          {final_type, float32},
          primitive,
          std::move(inputs))[0];
    } else {
      return array(
          std::move(out_shape), final_type, primitive, std::move(inputs));
    }
  }
  return fallback(std::move(inputs))[0];
}

std::vector<array> ScaledDotProductAttention::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() >= 3);
  assert(cotangents.size() == outputs.size());

  auto s = stream();
  if (ScaledDotProductAttentionVJP::use_fallback(primals[0], s)) {
    assert(outputs.size() == 1);
    return Custom::vjp(primals, cotangents, argnums, outputs);
  }

  auto fallback = [sdpa = fallback_, s](const std::vector<array>& inputs) {
    std::vector<array> primals(inputs.begin(), std::prev(inputs.end()));
    auto [_, vjps] = mlx::core::vjp(sdpa, primals, {inputs.back()});
    return vjps;
  };

  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes;
  for (int i = 0; i < /* outputs size */ 3; ++i) {
    shapes.push_back(primals[i].shape());
    dtypes.push_back(primals[i].dtype());
  }
  auto primitive = std::make_shared<ScaledDotProductAttentionVJP>(
      s, fallback, scale_, do_causal_, has_sinks_);
  std::vector<array> inputs = primals;
  inputs.push_back(outputs[0]);
  inputs.push_back(outputs[1]);
  inputs.push_back(cotangents[0]);
  auto vjps = array::make_arrays(std::move(shapes), dtypes, primitive, inputs);

  std::vector<array> returned_vjps;
  for (int arg : argnums) {
    if (arg >= 3) {
      throw std::invalid_argument(
          "[scale_dot_product_attention] Does not support VJP with respect "
          " to mask or attention sinks.");
    }
    returned_vjps.push_back(std::move(vjps[arg]));
  }
  return returned_vjps;
}

bool ScaledDotProductAttention::is_equivalent(const Primitive& other) const {
  const ScaledDotProductAttention& a_other =
      static_cast<const ScaledDotProductAttention&>(other);
  return scale_ == a_other.scale_ && do_causal_ == a_other.do_causal_ &&
      has_sinks_ == a_other.has_sinks_ &&
      output_logsumexp_ == a_other.output_logsumexp_;
}

bool ScaledDotProductAttentionVJP::is_equivalent(const Primitive& other) const {
  const ScaledDotProductAttentionVJP& a_other =
      static_cast<const ScaledDotProductAttentionVJP&>(other);
  return scale_ == a_other.scale_ && do_causal_ == a_other.do_causal_ &&
      has_sinks_ == a_other.has_sinks_;
}

bool Quantize::is_equivalent(const Primitive& other) const {
  const Quantize& p_other = static_cast<const Quantize&>(other);
  return (
      p_other.group_size_ == group_size_ && p_other.bits_ == bits_ &&
      p_other.mode_ == mode_ && p_other.dequantize_ == dequantize_);
}

std::vector<Shape> Quantize::output_shapes(const std::vector<array>& inputs) {
  auto& w = inputs[0];
  if (dequantize_) {
    auto out_size = w.shape(-1) * 32 / bits_;
    auto out_shape = w.shape();
    out_shape.back() = out_size;
    return {std::move(out_shape)};
  } else {
    auto wq_shape = w.shape();
    wq_shape.back() = w.shape(-1) * bits_ / 32;
    auto sshape = w.shape();
    sshape.back() = w.shape(-1) / group_size_;
    if (inputs.size() == 2) {
      return {std::move(wq_shape), std::move(sshape)};
    } else {
      auto bshape = sshape;
      return {std::move(wq_shape), std::move(sshape), std::move(bshape)};
    }
  }
}

bool ConvertFP8::is_equivalent(const Primitive& other) const {
  const ConvertFP8& a_other = static_cast<const ConvertFP8&>(other);
  return to_fp8_ == a_other.to_fp8_;
}

array flce_loss(
    const array& hidden,
    const array& weight,
    const array& targets,
    int chunk_size /* = 4096 */,
    int ignore_index /* = -100 */,
    StreamOrDevice s_ /* = {} */) {
  // Validate inputs
  if (hidden.ndim() < 2) {
    std::ostringstream msg;
    msg << "[flce_loss] hidden must have at least 2 dimensions but got "
        << hidden.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (weight.ndim() != 2) {
    std::ostringstream msg;
    msg << "[flce_loss] weight must have 2 dimensions but got " << weight.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (targets.ndim() < 1) {
    std::ostringstream msg;
    msg << "[flce_loss] targets must have at least 1 dimension but got "
        << targets.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  // Check dtype compatibility
  auto compute_type = result_type(hidden, weight);
  if (!issubdtype(compute_type, floating)) {
    std::ostringstream msg;
    msg << "[flce_loss] Received unsupported type " << compute_type << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(targets.dtype(), integer)) {
    std::ostringstream msg;
    msg << "[flce_loss] targets must be integer type but got "
        << targets.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Flatten batch dimensions: [B, T, H] -> [N, H]
  int H = hidden.shape(-1);
  int V = weight.shape(0);
  int N = hidden.size() / H;

  if (weight.shape(1) != H) {
    std::ostringstream msg;
    msg << "[flce_loss] hidden dim (" << H << ") must match weight dim ("
        << weight.shape(1) << ").";
    throw std::invalid_argument(msg.str());
  }
  if (targets.size() != N) {
    std::ostringstream msg;
    msg << "[flce_loss] targets size (" << targets.size()
        << ") must match N (" << N << ").";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);

  // Fallback implementation using standard MLX ops (for CPU)
  auto fallback = [chunk_size, ignore_index, H, V, N, compute_type, s](
                      const std::vector<array>& inputs) {
    auto h = reshape(inputs[0], {N, H}, s);
    auto w = inputs[1];
    auto t = reshape(inputs[2], {N}, s);
    t = astype(t, int32, s);

    auto valid_mask = not_equal(t, array(ignore_index, int32), s);
    auto num_valid = sum(astype(valid_mask, float32, s), s);
    num_valid = maximum(num_valid, array(1.0f, float32), s);

    int num_chunks = (V + chunk_size - 1) / chunk_size;
    array running_max = full({N}, -std::numeric_limits<float>::infinity(), float32, s);
    array running_sum = zeros({N}, float32, s);
    array target_logits = zeros({N}, float32, s);

    for (int c = 0; c < num_chunks; c++) {
      int v_start = c * chunk_size;
      int v_end = std::min(v_start + chunk_size, V);
      int actual_chunk = v_end - v_start;

      auto w_chunk = slice(w, {v_start, 0}, {v_end, H}, s);
      auto logits_chunk = matmul(astype(h, compute_type, s),
                                 transpose(astype(w_chunk, compute_type, s), s), s);
      logits_chunk = astype(logits_chunk, float32, s);

      auto chunk_max = max(logits_chunk, -1, true, s);
      auto chunk_exp = exp(subtract(logits_chunk, chunk_max, s), s);
      auto chunk_sum = sum(chunk_exp, -1, false, s);
      chunk_max = squeeze(chunk_max, -1, s);

      auto new_max = maximum(running_max, chunk_max, s);
      auto scale_old = exp(subtract(running_max, new_max, s), s);
      auto scale_new = exp(subtract(chunk_max, new_max, s), s);
      running_sum = add(multiply(running_sum, scale_old, s),
                        multiply(chunk_sum, scale_new, s), s);
      running_max = new_max;

      auto in_chunk = logical_and(
          greater_equal(t, array(v_start, int32), s),
          less(t, array(v_end, int32), s), s);
      auto local_idx = subtract(t, array(v_start, int32), s);
      local_idx = clip(local_idx, array(0, int32), array(actual_chunk - 1, int32), s);

      auto gathered = take_along_axis(logits_chunk, expand_dims(local_idx, -1, s), 1, s);
      gathered = squeeze(gathered, -1, s);
      target_logits = where(in_chunk, gathered, target_logits, s);
    }

    auto logsumexp = add(running_max, log(add(running_sum, array(1e-9f, float32), s), s), s);
    auto token_losses = subtract(logsumexp, target_logits, s);
    token_losses = where(valid_mask, token_losses, zeros_like(token_losses, s), s);
    auto loss = divide(sum(token_losses, s), num_valid, s);

    return std::vector<array>{loss};
  };

  // Prepare inputs (flatten if needed)
  auto h_flat = reshape(astype(hidden, compute_type, s), {N, H}, s);
  auto w_typed = astype(weight, compute_type, s);
  auto t_flat = reshape(astype(targets, int32, s), {N}, s);

  if (!FLCELoss::use_fallback(s)) {
    return array(
        {},
        float32,
        std::make_shared<FLCELoss>(s, fallback, chunk_size, ignore_index),
        {h_flat, w_typed, t_flat});
  }
  return fallback({h_flat, w_typed, t_flat})[0];
}

std::vector<array> FLCELoss::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 3);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto& hidden = primals[0];
  auto& weight = primals[1];
  auto& targets = primals[2];
  auto& cotan = cotangents[0];

  int H = hidden.shape(-1);
  int V = weight.shape(0);
  int N = hidden.size() / H;

  // Fallback for VJP computation
  auto fallback = [chunk_size = chunk_size_,
                   ignore_index = ignore_index_,
                   H, V, N, s](const std::vector<array>& inputs) {
    auto& h = inputs[0];
    auto& w = inputs[1];
    auto& t = inputs[2];
    auto& cotan = inputs[3];

    auto compute_type = h.dtype();

    // Create valid mask and compute scale
    auto valid_mask = not_equal(t, array(ignore_index, int32), s);
    auto num_valid = sum(astype(valid_mask, float32, s), s);
    auto scale = divide(cotan, maximum(num_valid, array(1.0f, float32), s), s);

    // First pass: compute logsumexp and target logits
    array running_max = full({N}, -std::numeric_limits<float>::infinity(), float32, s);
    array running_sum = zeros({N}, float32, s);
    array target_logits = zeros({N}, float32, s);

    int num_chunks = (V + chunk_size - 1) / chunk_size;

    for (int c = 0; c < num_chunks; c++) {
      int v_start = c * chunk_size;
      int v_end = std::min(v_start + chunk_size, V);
      int actual_chunk = v_end - v_start;

      auto w_chunk = slice(w, {v_start, 0}, {v_end, H}, s);
      auto logits_chunk = matmul(astype(h, compute_type, s),
                                 transpose(astype(w_chunk, compute_type, s), s), s);
      logits_chunk = astype(logits_chunk, float32, s);

      auto chunk_max = squeeze(max(logits_chunk, -1, true, s), -1, s);
      auto chunk_exp = exp(subtract(logits_chunk, expand_dims(chunk_max, -1, s), s), s);
      auto chunk_sum = sum(chunk_exp, -1, false, s);

      auto new_max = maximum(running_max, chunk_max, s);
      auto scale_old = exp(subtract(running_max, new_max, s), s);
      auto scale_new = exp(subtract(chunk_max, new_max, s), s);
      running_sum = add(multiply(running_sum, scale_old, s),
                        multiply(chunk_sum, scale_new, s), s);
      running_max = new_max;

      auto in_chunk = logical_and(
          greater_equal(t, array(v_start, int32), s),
          less(t, array(v_end, int32), s), s);
      auto local_idx = clip(subtract(t, array(v_start, int32), s),
                           array(0, int32), array(actual_chunk - 1, int32), s);
      // Gather target logits: get logits_chunk[i, local_idx[i]] for each row i
      auto gathered = take_along_axis(logits_chunk, expand_dims(local_idx, -1, s), 1, s);
      gathered = squeeze(gathered, -1, s);  // [N]
      target_logits = where(in_chunk, gathered, target_logits, s);

      // Periodic eval to prevent graph explosion
      if ((c + 1) % 4 == 0 || c == num_chunks - 1) {
        eval({running_max, running_sum, target_logits});
      }
    }

    auto logsumexp = add(running_max, log(add(running_sum, array(1e-9f, float32), s), s), s);
    eval({logsumexp});  // Ensure logsumexp is computed before second pass

    // Second pass: compute gradients
    array grad_hidden = zeros({N, H}, compute_type, s);
    std::vector<array> grad_weight_chunks;

    for (int c = 0; c < num_chunks; c++) {
      int v_start = c * chunk_size;
      int v_end = std::min(v_start + chunk_size, V);

      auto w_chunk = slice(w, {v_start, 0}, {v_end, H}, s);
      auto logits_chunk = matmul(astype(h, compute_type, s),
                                 transpose(astype(w_chunk, compute_type, s), s), s);
      logits_chunk = astype(logits_chunk, float32, s);

      // Compute softmax probabilities
      auto softmax_probs = exp(subtract(logits_chunk,
                                        expand_dims(logsumexp, -1, s), s), s);

      // Create one-hot for targets in this chunk
      auto in_chunk = logical_and(
          greater_equal(t, array(v_start, int32), s),
          less(t, array(v_end, int32), s), s);
      auto local_idx = subtract(t, array(v_start, int32), s);
      local_idx = clip(local_idx, array(0, int32), array(v_end - v_start - 1, int32), s);

      // grad_logits = (softmax - onehot) * scale
      // Manual one-hot: compare each column index to the target index
      int this_chunk_size = v_end - v_start;
      auto col_indices = arange(this_chunk_size, int32, s);  // [chunk_size]
      auto onehot = astype(equal(expand_dims(col_indices, 0, s),
                                  expand_dims(local_idx, -1, s), s), float32, s);
      onehot = where(expand_dims(in_chunk, -1, s), onehot, zeros_like(onehot, s), s);

      auto grad_logits = multiply(subtract(softmax_probs, onehot, s),
                                  expand_dims(scale, -1, s), s);
      grad_logits = where(expand_dims(valid_mask, -1, s), grad_logits,
                          zeros_like(grad_logits, s), s);
      grad_logits = astype(grad_logits, compute_type, s);

      // grad_hidden += grad_logits @ w_chunk
      grad_hidden = add(grad_hidden,
                       matmul(grad_logits, astype(w_chunk, compute_type, s), s), s);

      // grad_w_chunk = grad_logits.T @ h
      auto grad_w_chunk = matmul(transpose(grad_logits, s),
                                 astype(h, compute_type, s), s);
      grad_weight_chunks.push_back(grad_w_chunk);

      // Periodic eval to prevent graph explosion
      if ((c + 1) % 2 == 0) {
        eval({grad_hidden, grad_w_chunk});
      }
    }

    auto grad_weight = concatenate(grad_weight_chunks, 0, s);

    return std::vector<array>{grad_hidden, grad_weight};
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape()},
      {primals[0].dtype(), primals[1].dtype()},
      std::make_shared<FLCELossVJP>(s, fallback, chunk_size_, ignore_index_),
      {primals[0], primals[1], primals[2], cotangents[0]});

  std::vector<array> returned_vjps;
  for (int arg : argnums) {
    if (arg >= 2) {
      // No gradient for targets
      returned_vjps.push_back(zeros_like(primals[arg], s));
    } else {
      returned_vjps.push_back(std::move(vjps[arg]));
    }
  }
  return returned_vjps;
}

bool FLCELoss::is_equivalent(const Primitive& other) const {
  const FLCELoss& a_other = static_cast<const FLCELoss&>(other);
  return chunk_size_ == a_other.chunk_size_ &&
         ignore_index_ == a_other.ignore_index_;
}

bool FLCELossVJP::is_equivalent(const Primitive& other) const {
  const FLCELossVJP& a_other = static_cast<const FLCELossVJP&>(other);
  return chunk_size_ == a_other.chunk_size_ &&
         ignore_index_ == a_other.ignore_index_;
}

// =============================================================================
// CCE (Cut Cross-Entropy) Implementation
// Memory-efficient cross-entropy with vocabulary tiling and sparsity
// =============================================================================

array cce_loss(
    const array& hidden,
    const array& weight,
    const array& targets,
    int ignore_index /* = -100 */,
    StreamOrDevice s_ /* = {} */) {
  // Validate inputs
  if (hidden.ndim() < 2) {
    std::ostringstream msg;
    msg << "[cce_loss] hidden must have at least 2 dimensions but got "
        << hidden.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (weight.ndim() != 2) {
    std::ostringstream msg;
    msg << "[cce_loss] weight must have 2 dimensions but got " << weight.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (targets.ndim() < 1) {
    std::ostringstream msg;
    msg << "[cce_loss] targets must have at least 1 dimension but got "
        << targets.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  // Check dtype compatibility
  auto compute_type = result_type(hidden, weight);
  if (!issubdtype(compute_type, floating)) {
    std::ostringstream msg;
    msg << "[cce_loss] Received unsupported type " << compute_type << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(targets.dtype(), integer)) {
    std::ostringstream msg;
    msg << "[cce_loss] targets must be integer type but got "
        << targets.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Flatten batch dimensions: [B, T, H] -> [N, H]
  int H = hidden.shape(-1);
  int V = weight.shape(0);
  int N = hidden.size() / H;

  if (weight.shape(1) != H) {
    std::ostringstream msg;
    msg << "[cce_loss] hidden dim (" << H << ") must match weight dim ("
        << weight.shape(1) << ").";
    throw std::invalid_argument(msg.str());
  }
  if (targets.size() != N) {
    std::ostringstream msg;
    msg << "[cce_loss] targets size (" << targets.size()
        << ") must match N (" << N << ").";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);

  // Fallback implementation using standard MLX ops (for CPU)
  // Uses online logsumexp for memory efficiency
  auto fallback = [ignore_index, H, V, N, compute_type, s](
                      const std::vector<array>& inputs) {
    auto h = reshape(inputs[0], {N, H}, s);
    auto w = inputs[1];
    auto t = reshape(inputs[2], {N}, s);
    t = astype(t, int32, s);

    auto valid_mask = not_equal(t, array(ignore_index, int32), s);

    // Compute full logits and use standard cross-entropy for fallback
    // This is not memory efficient but correct
    auto logits = matmul(astype(h, compute_type, s),
                         transpose(astype(w, compute_type, s), s), s);
    logits = astype(logits, float32, s);

    // Compute logsumexp
    auto lse = logsumexp(logits, -1, false, s);

    // Gather target logits
    auto target_logits = take_along_axis(
        logits, expand_dims(t, -1, s), 1, s);
    target_logits = squeeze(target_logits, -1, s);

    // Per-token losses
    auto token_losses = subtract(lse, target_logits, s);
    token_losses = where(valid_mask, token_losses, zeros_like(token_losses, s), s);

    return std::vector<array>{token_losses};
  };

  // Prepare inputs (flatten if needed)
  auto h_flat = reshape(astype(hidden, compute_type, s), {N, H}, s);
  auto w_typed = astype(weight, compute_type, s);
  auto t_flat = reshape(astype(targets, int32, s), {N}, s);

  if (!CCELoss::use_fallback(s)) {
    // During training, output logsumexp for efficient backward pass
    bool is_training = detail::in_grad_tracing();
    bool output_logsumexp = is_training;

    auto primitive = std::make_shared<CCELoss>(s, fallback, ignore_index, output_logsumexp);

    if (output_logsumexp) {
      // Return both loss and logsumexp (logsumexp used in backward)
      return array::make_arrays(
          {{N}, {N}},
          {float32, float32},
          primitive,
          {h_flat, w_typed, t_flat})[0];
    } else {
      return array(
          {N},  // Output is per-token loss
          float32,
          primitive,
          {h_flat, w_typed, t_flat});
    }
  }
  return fallback({h_flat, w_typed, t_flat})[0];
}

std::vector<array> CCELoss::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 3);
  assert(cotangents.size() == outputs.size());

  auto s = stream();
  auto& hidden = primals[0];
  auto& weight = primals[1];
  auto& targets = primals[2];
  auto& cotan = cotangents[0];

  int H = hidden.shape(-1);
  int V = weight.shape(0);
  int N = hidden.size() / H;

  // Check if we have logsumexp saved from forward
  bool has_logsumexp = output_logsumexp_ && outputs.size() >= 2;

  // Fallback VJP implementation (used when logsumexp not provided)
  auto fallback = [ignore_index = ignore_index_, has_logsumexp,
                   H, V, N, s](const std::vector<array>& inputs) {
    auto& h = inputs[0];
    auto& w = inputs[1];
    auto& t = inputs[2];
    auto& cotan = inputs[3];
    // inputs[4] is logsumexp if has_logsumexp is true

    auto compute_type = h.dtype();

    // Compute full logits for softmax (required for gradient computation)
    auto logits = matmul(astype(h, compute_type, s),
                         transpose(astype(w, compute_type, s), s), s);
    logits = astype(logits, float32, s);

    // Use saved logsumexp if available, otherwise compute it
    auto lse = (has_logsumexp && inputs.size() > 4)
        ? expand_dims(inputs[4], -1, s)
        : logsumexp(logits, -1, true, s);

    // Compute softmax using saved/computed lse
    auto softmax_probs = exp(subtract(logits, lse, s), s);

    // Create one-hot for targets
    auto one_hot = zeros({N, V}, float32, s);
    // Scatter 1.0 at target positions
    auto valid_mask = logical_and(
        greater_equal(t, array(0, int32), s),
        less(t, array(V, int32), s), s);
    valid_mask = logical_and(
        valid_mask,
        not_equal(t, array(ignore_index, int32), s), s);

    // For each valid target, subtract 1 from softmax
    auto t_clamped = clip(t, array(0, int32), array(V - 1, int32), s);
    auto row_idx = expand_dims(arange(N, int32, s), -1, s);  // [N, 1]
    auto col_idx = expand_dims(t_clamped, -1, s);             // [N, 1]
    auto updates = expand_dims(where(valid_mask, ones({N}, float32, s),
                                     zeros({N}, float32, s), s), -1, s);  // [N, 1]
    one_hot = scatter(one_hot, {row_idx, col_idx}, updates, {0, 1}, s);

    // grad_logits = (softmax - one_hot) * upstream_grad
    auto grad_logits = subtract(softmax_probs, one_hot, s);
    grad_logits = multiply(grad_logits, expand_dims(cotan, -1, s), s);

    // Apply ignore mask
    auto ignore_mask = equal(t, array(ignore_index, int32), s);
    grad_logits = where(
        expand_dims(ignore_mask, -1, s),
        zeros_like(grad_logits, s),
        grad_logits, s);

    // grad_hidden = grad_logits @ weight
    auto grad_hidden = matmul(astype(grad_logits, compute_type, s),
                              astype(w, compute_type, s), s);

    // grad_weight = grad_logits.T @ hidden
    auto grad_weight = matmul(transpose(astype(grad_logits, compute_type, s), s),
                              astype(h, compute_type, s), s);

    return std::vector<array>{grad_hidden, grad_weight};
  };

  // Build inputs for VJP primitive
  std::vector<array> vjp_inputs = {primals[0], primals[1], primals[2], cotangents[0]};
  if (has_logsumexp) {
    vjp_inputs.push_back(outputs[1]);  // logsumexp from forward
  }

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape()},
      {primals[0].dtype(), primals[1].dtype()},
      std::make_shared<CCELossVJP>(s, fallback, ignore_index_, has_logsumexp),
      std::move(vjp_inputs));

  std::vector<array> returned_vjps;
  for (int arg : argnums) {
    if (arg >= 2) {
      // No gradient for targets
      returned_vjps.push_back(zeros_like(primals[arg], s));
    } else {
      returned_vjps.push_back(std::move(vjps[arg]));
    }
  }
  return returned_vjps;
}

bool CCELoss::is_equivalent(const Primitive& other) const {
  const CCELoss& a_other = static_cast<const CCELoss&>(other);
  return ignore_index_ == a_other.ignore_index_ &&
         output_logsumexp_ == a_other.output_logsumexp_;
}

bool CCELossVJP::is_equivalent(const Primitive& other) const {
  const CCELossVJP& a_other = static_cast<const CCELossVJP&>(other);
  return ignore_index_ == a_other.ignore_index_ &&
         has_logsumexp_ == a_other.has_logsumexp_;
}

} // namespace mlx::core::fast
