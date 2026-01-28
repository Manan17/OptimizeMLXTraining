# FLCE (Fused Linear Cross Entropy) Implementation Knowledge

## Goal
Create a Metal kernel that fuses the linear projection with cross-entropy loss computation to avoid materializing the full [N, V] logits tensor during LLM training on Mac.

## Mathematical Foundation

### Forward Pass
```
logits[n, v] = hidden[n] @ weight[v]  (dot product for each vocab term)
loss = mean(logsumexp(logits) - logits[target])
```

Using online softmax algorithm to avoid storing all logits:
```
for each vocab v:
    logit = hidden @ weight[v]
    new_max = max(running_max, logit)
    running_sum = running_sum * exp(running_max - new_max) + exp(logit - new_max)
    running_max = new_max
logsumexp = running_max + log(running_sum)
```

### Backward Pass
```
grad_logits[n, v] = (softmax[n, v] - onehot[n, v]) / N
grad_hidden[n, h] = sum_v(grad_logits[n, v] * weight[v, h])
grad_weight[v, h] = sum_n(grad_logits[n, v] * hidden[n, h])  # Optional for LoRA
```

## MLX Metal Kernel Patterns (from studying the codebase)

### 1. Two-Tier VJP Architecture
- Forward primitive: `FLCELoss` with `eval_gpu()`
- Backward primitive: `FLCELossVJP` with `eval_gpu()`
- Forward's `vjp()` method creates the backward primitive

### 2. Buffer Allocation in eval_gpu
```cpp
// Always allocate fresh buffers for outputs
output.set_data(allocator::malloc(output.nbytes()));

// For temporaries
array temp({N}, float32, nullptr, {});
temp.set_data(allocator::malloc(temp.nbytes()));
d.add_temporary(temp, s.index);  // Auto-freed after command buffer
```

### 3. Contiguity Guarantee
```cpp
auto ensure_contiguous = [&copies, &s](const array& arr) -> const array& {
    if (arr.flags().row_contiguous) return arr;
    array arr_copy = contiguous_copy_gpu(arr, s);
    copies.push_back(std::move(arr_copy));
    return copies.back();
};
```

### 4. Kernel Dispatch Pattern
```cpp
std::string kernel_name = "kernel_name_" + type_to_name(dtype);
auto kernel = d.get_kernel(kernel_name);
compute_encoder.set_compute_pipeline_state(kernel);
compute_encoder.set_input_array(input, 0);
compute_encoder.set_output_array(output, 1);
compute_encoder.set_bytes(params, 2);
compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
```

### 5. SDPA-Style Parallelism
All 32 SIMD lanes collaborate on ONE dot product:
- Each lane holds H/32 elements
- `simd_sum()` reduces partial sums across lanes
- Gives coalesced memory access

## Key Files Needed

1. **Header with kernel**: `mlx/backend/metal/kernels/steel/flce/flce.h`
2. **Metal instantiation**: `mlx/backend/metal/kernels/steel/flce/flce.metal`
3. **Params struct**: `mlx/backend/metal/kernels/steel/flce/params.h`
4. **C++ dispatch**: `mlx/backend/metal/flce.cpp`
5. **Primitive class**: Already exists in `mlx/fast_primitives.h`
6. **Python binding**: Already exists in `mlx/fast.cpp`

## Lessons Learned from Previous Attempt

### What Worked
- Forward pass loss computation matched baseline perfectly
- Online softmax algorithm was correct
- SDPA-style parallelism pattern is appropriate

### What Failed
- Backward kernel produced wrong gradients (all same value)
- Buffer aliasing issues
- Complexity grew too fast without verification at each step

### Root Causes to Investigate
1. **Stride calculation**: Need to verify h_offset and actual_stride are correct
2. **Buffer binding**: Ensure all buffers are correctly bound to kernel
3. **Kernel selection**: Verify correct kernel variant is dispatched
4. **Value propagation**: Ensure simd_sum broadcasts correctly to all lanes

## Recommended Fresh Approach

### Step 1: Simple CPU-like GPU Kernel
- One thread per row
- Loop over all vocab sequentially
- No SIMD tricks
- Verify gradients match

### Step 2: Add SIMD Parallelism
- Multiple threads per row
- Use simd_sum for dot products
- Verify gradients still match

### Step 3: Optimize
- Tune thread group sizes
- Consider chunking for large vocab
- Profile and optimize memory access

## Test Strategy
```python
# Small test case for debugging
N, H, V = 2, 8, 10  # Tiny dimensions
hidden = mx.random.normal((N, H))
weight = mx.random.normal((V, H))
targets = mx.array([0, 5])

# Compare forward
loss_baseline = cross_entropy(matmul(hidden, weight.T), targets)
loss_flce = mx.fast.flce_loss(hidden, weight, targets)

# Compare gradients
grad_baseline = mx.grad(baseline_fn)(hidden)
grad_flce = mx.grad(flce_fn)(hidden)
```

## Reference Implementations
- `mlx/backend/metal/normalization.cpp` - RMSNormVJP pattern
- `mlx/backend/metal/scaled_dot_product_attention.cpp` - SDPA pattern
- `mlx/fast.cpp` lines 941-1220 - FLCE fallback implementation

## File Locations
- MLX source: `/Users/mananshah/Desktop/MetalLora/mlx/`
- Kernels: `mlx/backend/metal/kernels/`
- Steel kernels: `mlx/backend/metal/kernels/steel/`
