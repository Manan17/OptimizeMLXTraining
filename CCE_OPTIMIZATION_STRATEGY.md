# CCE Optimization Strategy for Apple Silicon

## Document Info
- **Created:** 2026-01-27
- **Status:** Research Complete, Implementation Pending
- **Target:** Optimize CCE training performance over MLX baseline

---

## Executive Summary

After thorough analysis of the current CCE implementation, reference implementations (Liger FLCE, Unsloth CE), MLX optimized kernels, and Apple Silicon GPU architecture, this document outlines a prioritized optimization strategy.

**Key Finding:** The major memory issues identified in `APPLE_SILICON_CCE_DESIGN.md` have already been fixed. The remaining opportunities are **speed optimizations**, particularly for small batches.

---

## Part 1: Current State Analysis

### 1.1 Issues Already Fixed (Code Verified)

| Issue | Original Impact | Status | Evidence |
|-------|-----------------|--------|----------|
| FP32 grad_hidden_f32 buffer | +0.067 GB | ✅ FIXED | `cce.cpp:481` - Now alias |
| FP32 grad_weight_f32 buffer | +0.524 GB | ✅ FIXED | `cce.cpp:482` - Now alias |
| LSE recomputation | +0.524 GB | ✅ FIXED | `cce.cpp:503-524` - Saved from forward |
| d_logits buffer | +0.524 GB | ✅ FIXED | `cce.cpp:668-672` - In-place via shared buffer |
| **Total** | **1.94 GB** | **~0 GB** | All major memory issues resolved |

### 1.2 Remaining Issues

| Issue | Impact | Status |
|-------|--------|--------|
| Batch=8 anomaly (N=1024) | 3.6x slower than baseline | ⚠️ Unresolved |
| Small batch overhead | +6% memory, no speedup | ⚠️ Unresolved |
| Dispatch overhead for small N | ~6 dispatches per forward | ⚠️ Unresolved |

### 1.3 Current Benchmark Results

```
+---------+------+------------+----------+------------+---------+
| Batch   | N    | Base Mem   | CCE Mem  | Mem Ratio  | Speedup |
+---------+------+------------+----------+------------+---------+
| 4       | 512  | 1.46 GB    | 1.56 GB  | 0.94x      | 0.98x   |
| 8       | 1024 | 1.97 GB    | 2.09 GB  | 0.94x      | 0.28x*  | ← ANOMALY
| 16      | 2048 | 3.00 GB    | 3.00 GB  | 1.00x      | 1.00x   |
| 32      | 4096 | 4.81 GB    | 4.54 GB  | 1.06x      | 1.23x   | ← CCE wins here
+---------+------+------------+----------+------------+---------+
```

**Goal:** Achieve speedup across ALL batch sizes, not just large ones.

---

## Part 2: Apple Silicon Architecture Insights

### 2.1 Key Hardware Characteristics

| Spec | M1/M2/M3 | Optimization Implication |
|------|----------|--------------------------|
| L1 Cache | 8 KB | Very small - working set must fit |
| L2 Cache | 256-768 KB | Varies wildly across chip variants |
| SLC | Massive | Low-latency compensates for small L2 |
| SIMD Width | 32 threads | Same as CUDA warp |
| Global Atomics | **Expensive** | Use threadgroup atomics or SIMD reductions |
| Memory Barriers | **Flush tile memory** | Minimize at all costs |
| Unified Memory | Zero-copy | No separate buffer allocations needed |

### 2.2 Critical Optimization Rules

1. **Never use `threadgroup_barrier` inside loops** - Flushes tile memory
2. **Use `simd_sum/simd_max` over threadgroup reductions** - 10-100x cheaper
3. **Minimize global atomic operations** - They serialize across all threads
4. **Tile size must fit in L1** - 8 KB = ~2000 floats max per working set
5. **Vectorize memory access** - Use float4 for 4x bandwidth

### 2.3 References

- [metal-benchmarks - Apple GPU microarchitecture](https://github.com/philipturner/metal-benchmarks)
- [Apple vs. Oranges: M-Series HPC Performance (Feb 2025)](https://arxiv.org/html/2502.05317v1)
- [Profiling Apple Silicon for ML Training](https://arxiv.org/pdf/2501.14925)
- [WWDC24: Train ML models on Apple GPUs](https://developer.apple.com/videos/play/wwdc2024/10160/)

---

## Part 3: Root Cause Analysis

### 3.1 Why Small Batches Are Slow

Current chunked forward path (`cce.cpp:120-208`):

```cpp
bool use_chunked_forward = (V > 2000 && N >= 8);  // Triggers for all practical cases

for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {  // 3 chunks for V=49152
    // 1. Create weight_chunk view
    // 2. steel_matmul: hidden @ weight_chunk.T  → logits_chunk
    // 3. cce_chunk_logsumexp kernel: update running_max, running_sum_exp
}
// 4. cce_finalize_loss kernel
```

**Problem:** For V=49152, CHUNK_V=16384 → 3 chunks × 2 kernels = **6 dispatches per forward**

Dispatch overhead per kernel:
- Metal command buffer encoding: ~10-50 μs
- GPU kernel launch: ~20-100 μs
- Synchronization: ~10-50 μs

For small N (512-1024), compute is fast but **dispatch overhead dominates**.

### 3.2 The Batch=8 Anomaly

At N=1024, something pathological happens. Hypotheses:

1. **Memory alignment** - N=1024 hits a bad stride pattern
2. **steel_matmul splitk path** - May trigger inefficient code path for M=1024, N=16384, K=576
3. **Cache thrashing** - Working set exceeds L2 at this specific size

### 3.3 Barrier Overhead in Backward Pass

Current backward kernel (`kernels.h:422-449`) has O(V) barriers:

```metal
for (int local_v = 0; local_v < v_tile_size; local_v++) {
    // ... compute partial
    threadgroup_barrier(mem_flags::mem_threadgroup);  // BARRIER 1
    // ... reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);  // BARRIER 2
}
```

For V=49152, this is ~100,000 barriers per token!

---

## Part 4: Optimization Strategies

### Strategy 1: Single-Dispatch Fused Kernel for Small Batches

**Target:** N < 2048 (batch=4, 8, 16 with seq=128)

**Current:** 6+ dispatches per forward pass
**Proposed:** 1 dispatch per forward pass

**Design:**

```metal
// New kernel: cce_forward_fused_small
// Processes entire vocabulary in one kernel launch
// Each threadgroup handles one token completely

template <typename T>
[[kernel]] void cce_forward_fused_small(
    const device T* hidden,      // [N, H]
    const device T* weight,      // [V, H]
    const device int32_t* targets,
    device float* loss,
    device float* logsumexp_out,
    constant CCEParams& params,
    uint tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    const int token_idx = tid;
    if (token_idx >= params.N) return;

    // Load hidden state once into registers
    const int h_per_thread = (params.H + 31) / 32;
    float h_local[128];  // Up to H=4096
    for (int i = 0; i < h_per_thread; i++) {
        int h_idx = simd_lid * h_per_thread + i;
        if (h_idx < params.H) {
            h_local[i] = float(hidden[token_idx * params.H + h_idx]);
        }
    }

    // Online softmax over ALL vocabulary (no chunking!)
    OnlineSoftmax state;
    float target_logit = 0.0f;

    // Each simdgroup processes interleaved vocab entries
    // 4 simdgroups = 4x parallelism, NO barriers in main loop
    for (int v = simd_gid; v < params.V; v += 4) {
        float partial = 0.0f;
        for (int i = 0; i < h_per_thread; i++) {
            int h_idx = simd_lid * h_per_thread + i;
            if (h_idx < params.H) {
                partial += h_local[i] * float(weight[v * params.H + h_idx]);
            }
        }
        float logit = simd_sum(partial);

        if (simd_lid == 0) {
            state.update(logit);
            if (v == targets[token_idx]) {
                target_logit = logit;
            }
        }
    }

    // Single threadgroup barrier to combine 4 simdgroup states
    // ... (same as existing cce_forward_tiled)
}
```

**Key Differences from Current:**
1. **No matmul dispatch** - Dot products computed inline
2. **No intermediate buffers** - No logits_chunk allocation
3. **Single dispatch** - One kernel launch processes entire forward

**Dispatch Logic:**
```cpp
// In cce.cpp
if (N < 2048 && H <= 4096) {
    // Use fused small-batch kernel
    dispatch_cce_forward_fused_small(...);
} else {
    // Use chunked approach for large batches
    dispatch_chunked_forward(...);
}
```

**Expected Impact:**
- Batch=4: 2-3x speedup
- Batch=8: Fix anomaly, achieve 1.0x+ baseline
- Batch=16: 1.3-1.5x speedup

---

### Strategy 2: Fix Batch=8 Anomaly via Profiling

**Action Plan:**

1. **Profile with Metal System Trace:**
   ```bash
   # Use Xcode Instruments with Metal System Trace
   # Look for:
   # - GPU idle time between dispatches
   # - Memory bandwidth utilization
   # - Occupancy issues
   ```

2. **Check steel_matmul code path** for M=1024, N=16384, K=576

3. **Test alternative CHUNK_V configurations:**
   ```cpp
   int get_optimal_chunk_v(int N, int V, int H) {
       if (N <= 512) return std::min(8192, V);
       if (N <= 1024) return std::min(12288, V);  // May help batch=8
       return std::min(16384, V);
   }
   ```

---

### Strategy 3: SIMD-First Reduction Architecture

**Current Problem:** O(V) barriers per token in backward

**Proposed: Batch Reduction Pattern:**

```metal
// New: Accumulate in registers, barrier only once at end
float partial_max = -INFINITY;
float partial_sum = 0.0f;
float partial_target_logit = 0.0f;

// Each simdgroup processes V/4 vocab entries independently
for (int v = simd_gid; v < V; v += 4) {
    float partial = 0.0f;
    for (int i = 0; i < h_per_thread; i++) {
        partial += h_local[i] * weight_val;
    }
    float logit = simd_sum(partial);  // SIMD reduction (free!)

    if (simd_lid == 0) {
        // Update local online softmax state (no barrier!)
        float new_max = max(partial_max, logit);
        partial_sum = partial_sum * safe_exp(partial_max - new_max) + safe_exp(logit - new_max);
        partial_max = new_max;
        if (v == target) partial_target_logit = logit;
    }
}

// SINGLE barrier to combine 4 simdgroup states
threadgroup_barrier(mem_flags::mem_threadgroup);
// ... combine states (same as existing)
```

**Reduction:** From O(V) barriers → O(1) barriers

**Expected Impact:** 2-5x speedup for backward pass

---

### Strategy 4: Vectorized Memory Access

**Current (`kernels.h:149-151`):**
```metal
for (int i = 0; i < h_per_thread && h_start + i < H; i++) {
    partial_dot += h_local[i] * float(weight[global_v * H + h_start + i]);
}
```

**Problem:** Scalar loads, one element at a time.

**Proposed: Use float4 Vectorization:**
```metal
// Load 4 elements at once (requires H % 4 == 0)
for (int i = 0; i < h_per_thread; i += 4) {
    int h_idx = h_start + i;
    if (h_idx + 3 < H) {
        float4 h_vec = float4(h_local[i], h_local[i+1], h_local[i+2], h_local[i+3]);
        float4 w_vec = *reinterpret_cast<const device float4*>(&weight[global_v * H + h_idx]);
        partial_dot += dot(h_vec, w_vec);  // Single instruction!
    }
}
```

**Benefits:**
- 4x fewer memory transactions
- Better memory coalescing
- `dot(float4, float4)` is a single GPU instruction

**Expected Impact:** 1.5-2x speedup for compute-bound cases

---

### Strategy 5: Forward-Computed Gradients (Liger-style)

**Current Flow:**
```
Forward:  H @ W.T → logits → LSE, loss  (save LSE)
Backward: H @ W.T → logits → softmax → d_logits → gradients  (recompute logits!)
```

**The logits are computed TWICE.**

**Proposed: Compute d_logits During Forward:**

```metal
kernel void cce_forward_with_grads(
    // ... same inputs ...
    device T* d_logits_out,  // NEW: store gradients for backward
) {
    // ... compute logits and online softmax ...

    // After computing LSE, compute d_logits in same pass
    for (int v = simd_gid; v < V; v += 4) {
        float logit = /* already computed */;
        float prob = safe_exp(logit - lse);
        float d_logit = prob - (v == target ? 1.0f : 0.0f);

        if (simd_lid == 0) {
            d_logits_out[token_idx * V + v] = T(d_logit);
        }
    }
}
```

**Trade-off:**
- **Pro:** Backward no longer needs to recompute logits (saves O(N×V×H) FLOPs)
- **Con:** Must store d_logits (N×V×sizeof(T) bytes) between forward and backward

**When Beneficial:**
- When backward is slower than forward
- When memory is not the primary constraint

---

### Strategy 6: Kernel Specialization via Function Constants

**Current:** Runtime branching for ignore_index, scale, etc.

**Proposed:**
```metal
constant bool HAS_IGNORE_INDEX [[function_constant(0)]];
constant int VOCAB_SIZE [[function_constant(1)]];
constant int HIDDEN_SIZE [[function_constant(2)]];

kernel void cce_forward_specialized(...) {
    // Compile-time constants allow loop unrolling
    constexpr int H_PER_THREAD = (HIDDEN_SIZE + 31) / 32;

    // Branch eliminated at compile time
    if (HAS_IGNORE_INDEX) {
        if (target == params.ignore_index) return;
    }
}
```

**Benefits:**
- Eliminates runtime branching
- Enables loop unrolling
- Reduces register pressure

**Expected Impact:** 5-15% speedup

---

## Part 5: Implementation Roadmap

### Phase 1: Quick Wins (2-3 days)

| Task | Expected Impact | Complexity |
|------|----------------|------------|
| Add small-batch fused kernel | 2x for N<1024 | Medium |
| Profile and fix batch=8 anomaly | Fix 3.6x regression | Low |
| Add adaptive CHUNK_V for small N | 10-20% for small batches | Low |

### Phase 2: SIMD Optimizations (1 week)

| Task | Expected Impact | Complexity |
|------|----------------|------------|
| SIMD-first reduction (minimize barriers) | 2-3x for backward | Medium |
| float4 vectorized loads | 1.5x for compute-bound | Low |
| Function constants specialization | 5-15% overall | Medium |

### Phase 3: Advanced Fusion (2 weeks)

| Task | Expected Impact | Complexity |
|------|----------------|------------|
| Forward-computed gradients | 1.3x for backward | High |
| Chunked forward-backward fusion | 1.5x overall | High |
| simdgroup_matrix for large tiles | Variable | High |

---

## Part 6: Success Metrics

| Metric | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| Batch=4 Speed | 0.98x | 1.5x | 2.0x | 2.5x |
| Batch=8 Speed | 0.28x | 1.0x | 1.3x | 1.5x |
| Batch=16 Speed | 1.00x | 1.2x | 1.4x | 1.6x |
| Batch=32 Speed | 1.23x | 1.3x | 1.5x | 2.0x |
| Memory (N=4K) | 4.54 GB | 4.5 GB | 4.5 GB | 4.0 GB |

---

## Part 7: Key Metal Patterns to Apply

### Pattern 1: Online Softmax with SIMD Merge

```metal
struct OnlineSoftmax {
    float max_val;
    float sum_exp;

    METAL_FUNC void update(float x) {
        float new_max = metal::max(max_val, x);
        sum_exp = sum_exp * fast::exp(max_val - new_max) + fast::exp(x - new_max);
        max_val = new_max;
    }

    METAL_FUNC void simd_reduce() {
        // Reduce across 32 lanes without threadgroup memory
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = simd_shuffle_down(max_val, offset);
            float other_sum = simd_shuffle_down(sum_exp, offset);
            merge(other_max, other_sum);
        }
    }
};
```

### Pattern 2: Safe/Fast Path Separation

```metal
// From MLX BlockLoader pattern
if (is_full_tile) {
    load_unsafe();  // No bounds checking - fast path
} else {
    load_safe(tile_dim);  // Bounds checking for edge tiles
}
```

### Pattern 3: Vectorized Memory Access

```metal
// Load 4 elements at once for better bandwidth
float4 h_vec = *reinterpret_cast<const device float4*>(hidden + offset);
float4 w_vec = *reinterpret_cast<const device float4*>(weight + offset);
float dot_result = dot(h_vec, w_vec);
```

---

## Appendix: Reference Implementations Analyzed

### Liger Kernel FLCE
- **Location:** `Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py`
- **Key Technique:** Forward computes gradients, stores in-place
- **Memory:** Chunks over batch dimension, not vocabulary

### Unsloth CE
- **Location:** `unsloth/unsloth/kernels/cross_entropy_loss.py`
- **Key Technique:** Chunked logsumexp for large vocabularies, LSE caching
- **Memory:** Pre-computed LSE saves backward recomputation

### MLX Optimized Kernels
- **SDPA:** `mlx/mlx/backend/metal/kernels/sdpa_vector.h`
- **Softmax:** `mlx/mlx/backend/metal/kernels/softmax.h`
- **Key Patterns:** Online logsumexp, SIMD reductions, two-pass for large data

---

## Conclusion

The CCE implementation has evolved significantly - the major memory issues are resolved. The path to beating MLX baseline across all batch sizes requires:

1. **Eliminating dispatch overhead** for small batches (fused kernel)
2. **Fixing the batch=8 anomaly** (profiling + adaptive chunking)
3. **Minimizing synchronization** (SIMD-first reductions)
4. **Maximizing memory bandwidth** (vectorized access)

With these optimizations, CCE should achieve 1.5-2x speedup over baseline for all batch sizes while maintaining memory efficiency for large batches.

---

## Part 8: Why Memory Savings Are Only 6-8% (Not 80%)

### 8.1 The Core Issue: MLX Baseline is Already Optimized

**Liger/Unsloth compare against naive PyTorch:**
```python
# Naive PyTorch (what Liger compares against)
logits = hidden @ weight.T           # Allocates full [N, V] tensor
loss = F.cross_entropy(logits, targets)  # PyTorch autograd saves logits for backward
# Peak memory: 2-3 × N × V × sizeof(float)
```

**MLX's baseline is NOT naive:**
```python
# MLX's cross_entropy (from mlx/python/mlx/nn/losses.py:23-117)
def cross_entropy(logits, targets, ...):
    # 1. Extract only the target class score - O(N) memory, not O(N×V)
    score = mx.take_along_axis(logits, targets, axis)

    # 2. Compute logsumexp with streaming/fused kernel - O(1) extra memory
    logsumexp_logits = mx.logsumexp(logits, axis=axis)

    # 3. Loss = logsumexp - score
    loss = logsumexp_logits - score
```

**MLX NEVER materializes the full softmax(logits) tensor!**

The `logsumexp` kernel (from `mlx/backend/metal/kernels/logsumexp.h`) uses:
- Running max and normalizer (2 floats per row)
- Streams through vocabulary dimension
- No intermediate N×V allocation

### 8.2 Memory Comparison: What's Actually Happening

For SmolLM2-135M training with N=4096, V=49152:

| Component | Naive PyTorch | MLX Baseline | Our CCE |
|-----------|---------------|--------------|---------|
| **Full logits [N×V]** | 805 MB (materialized) | 805 MB (computed but not stored separately) | Chunked: 268 MB |
| **Softmax [N×V]** | 805 MB (materialized) | **0** (never created) | **0** |
| **Grad logits [N×V]** | 805 MB (for backward) | Recomputed on-the-fly | 268 MB chunk |
| **Logsumexp [N]** | Part of softmax | 16 KB | 16 KB |
| **CE peak memory** | ~2.4 GB | ~800 MB | ~536 MB |
| **CE memory savings** | baseline | 70% vs naive | 33% vs MLX |

**The 80% savings Liger claims is vs. naive PyTorch, not vs. an already-optimized framework.**

### 8.3 Why MLX Baseline Uses ~800 MB (Not 2.4 GB)

1. **`logsumexp()` is a fused kernel** - streams through data, stores only max + sum
2. **`take_along_axis()` gathers only target indices** - O(N) not O(N×V)
3. **No explicit softmax tensor** - gradients computed via chain rule
4. **Lazy evaluation** - MLX doesn't materialize intermediate tensors unnecessarily

### 8.4 What Our CCE Actually Saves

Looking at the benchmark: **Baseline 4.81 GB → CCE 4.54 GB = 270 MB savings**

This 270 MB is approximately:
- Full logits chunk: `N × CHUNK_V × 4 = 4096 × 16384 × 4 = 268 MB`

We're saving by not needing the full logits tensor during backward! But:

1. **MLX baseline already avoids most of the "naive" overhead**
2. **Total training memory is dominated by other things:**
   - Model weights: ~270 MB (135M params × 2 bytes for bf16)
   - Optimizer states (AdamW): ~540 MB (2× weights)
   - Activations for all layers: ~2-3 GB
   - Cross-entropy is only ~800 MB of the 4.81 GB total

**270 MB / 4810 MB = 5.6%** ← This matches our benchmark exactly!

### 8.5 The Real Comparison

| Comparison | Savings |
|------------|---------|
| CCE vs. Naive PyTorch | ~70-80% (what Liger claims) |
| CCE vs. MLX Baseline | ~33% of CE memory, ~6% of total memory |
| MLX Baseline vs. Naive | Already ~65% savings built-in |

### 8.6 Unified Memory Considerations

**The M2's unified memory changes the equation:**
- On discrete GPUs: Saving 800 MB on a 24 GB GPU matters (can fit larger batch)
- On M2 unified: Memory is shared with CPU, system can swap gracefully
- The "pressure" from memory usage is different

### 8.7 How to Get Bigger Savings

To see larger **percentage** savings, you would need:

1. **Larger vocab size** (V > 100K like Gemma's 256K):
   - CE memory grows as O(N × V)
   - Model weights grow as O(V × H) but not as fast

2. **Smaller model** (so CE is larger fraction of total):
   - With a tiny model, CE could be 50% of total memory

3. **Compare against truly naive baseline**:
   - Force MLX to materialize full logits: `logits = hidden @ weight.T; mx.eval(logits)`
   - Then compute loss separately

4. **Fusion with linear layer** (FLCE approach):
   - Never creates the logits tensor at all
   - This is where the biggest savings come from

### 8.8 Key Insight

**Our CCE provides value in different ways than Liger:**

| Aspect | Liger (vs PyTorch) | Our CCE (vs MLX) |
|--------|-------------------|------------------|
| Memory savings | 80% (huge win) | 6-33% (modest) |
| Speed at small batch | Similar | Currently slower |
| Speed at large batch | Similar | 1.23x faster |
| Primary benefit | Memory | Speed under memory pressure |

**Our optimization strategy should focus on SPEED**, since MLX already handles memory well.

---

## Part 9: FLCE (Fused Linear Cross-Entropy) Architecture

### 9.1 What Gets Fused

Standard LLM training does:
```
hidden [N, H] → Linear (LM Head) → logits [N, V] → CrossEntropy → loss
                    ↓
              weight [V, H]
```

**The problem:** `logits [N, V]` is huge (e.g., 4096 × 49152 × 4 = 805 MB)

**FLCE fuses these two operations** so logits never exist as a full tensor:
```
hidden [N, H] + weight [V, H] + targets [N] → FLCE Kernel → loss, gradients
                                                  ↓
                                        (logits computed but never stored)
```

### 9.2 Three Fundamental Approaches

#### Approach A: Chunk Over Batch Dimension (Liger's Approach)

```
For each batch_chunk of size B_chunk:
    1. logits_chunk = hidden_chunk @ weight.T    # [B_chunk, V] - FULL vocab
    2. CE(logits_chunk, targets_chunk)           # Standard CE on chunk
    3. Gradients computed, logits_chunk overwritten with grad_logits
    4. grad_weight += grad_logits.T @ hidden_chunk
```

**Memory:** `O(B_chunk × V)` - still large if V is big

**Visualization:**
```
hidden [N, H]     weight [V, H]
┌─────────┐       ┌─────────┐
│ chunk 1 │       │         │
├─────────┤   @   │  full   │  →  logits_chunk [B_chunk, V]
│ chunk 2 │       │  vocab  │
├─────────┤       │         │
│ chunk 3 │       └─────────┘
└─────────┘
```

#### Approach B: Chunk Over Vocab Dimension (Our Current CCE)

```
For each vocab_chunk of size V_chunk:
    1. logits_chunk = hidden @ weight_chunk.T    # [N, V_chunk] - FULL batch
    2. Update online_logsumexp with logits_chunk
    3. (Later) Combine logsumexp across chunks for final loss
```

**Memory:** `O(N × V_chunk)` - better for very large V

**Visualization:**
```
hidden [N, H]     weight [V, H]
┌─────────┐       ┌─────────┐
│         │       │ chunk 1 │
│  full   │   @   ├─────────┤  →  logits_chunk [N, V_chunk]
│  batch  │       │ chunk 2 │
│         │       ├─────────┤
└─────────┘       │ chunk 3 │
                  └─────────┘
```

#### Approach C: True Kernel-Level Fusion (Optimal for Apple Silicon)

**No intermediate logits buffer at all.** Single kernel that:
1. Each threadgroup handles one token
2. Streams through ALL vocab entries
3. Computes dot product → immediately updates online softmax
4. Logits live only in registers, never written to memory

```
For each token n (parallel across threadgroups):
    load hidden[n] into registers
    online_softmax_state = init()

    for v in 0..V:
        logit = dot(hidden[n], weight[v])  # In registers!
        online_softmax_state.update(logit)
        if v == target[n]: target_logit = logit

    loss[n] = online_softmax_state.logsumexp() - target_logit
```

**Memory:** `O(H)` per token - just the hidden state in registers!

### 9.3 Recommended FLCE Architecture for Apple Silicon

Given Apple Silicon's characteristics, we recommend a **hybrid** that combines:
- **Approach B (vocab chunking)** for large batches
- **Approach C (kernel fusion)** for small batches

#### Why This Combination?

1. **Apple Silicon has small L1 (8KB)** - Can't fit full vocab weights
2. **SIMD groups are 32 threads** - Natural for H-dimension parallelism
3. **Unified memory** - Can stream through vocab without discrete GPU penalties
4. **Our CCE infrastructure already exists** - Build on top, don't replace

### 9.4 FLCE Implementation Levels

#### Level 1: FLCE as a Wrapper (Easiest - Already Done!)

```python
class FusedLinearCrossEntropy(nn.Module):
    def __init__(self, in_features, vocab_size):
        self.weight = mx.zeros((vocab_size, in_features))  # LM head weight

    def __call__(self, hidden, targets):
        # Don't compute: logits = hidden @ self.weight.T
        # Instead, pass hidden and weight directly to CCE
        return mx.fast.cce_loss(hidden, self.weight, targets)
```

**This is what we already have!** Our CCE takes `hidden` and `weight` separately.

**Memory savings:** Same as current CCE (~6% of total)

#### Level 2: Fused Forward Kernel (Medium Complexity)

Create a single Metal kernel that fuses matmul + logsumexp:

```metal
// Instead of: steel_matmul() then cce_chunk_logsumexp()
// Single kernel that does both

template <typename T>
[[kernel]] void flce_forward_fused(
    const device T* hidden,         // [N, H]
    const device T* weight,         // [V, H]
    const device int32_t* targets,  // [N]
    device float* loss,             // [N]
    device float* logsumexp_out,    // [N]
    constant FLCEParams& params,
    uint tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    const int token_idx = tid;
    if (token_idx >= params.N) return;

    // Load hidden[token] into registers ONCE
    const int h_per_thread = (params.H + 31) / 32;
    float h_local[128];
    for (int i = 0; i < h_per_thread; i++) {
        int h_idx = simd_lid * h_per_thread + i;
        if (h_idx < params.H) {
            h_local[i] = float(hidden[token_idx * params.H + h_idx]);
        }
    }

    // Online softmax - stream through ALL vocab
    OnlineSoftmax state;
    float target_logit = 0.0f;
    const int target = targets[token_idx];

    // Each simdgroup handles interleaved vocab entries
    for (int v = simd_gid; v < params.V; v += NUM_SIMDGROUPS) {
        // Compute logit = dot(hidden, weight[v]) - FUSED!
        float partial = 0.0f;
        for (int i = 0; i < h_per_thread; i++) {
            int h_idx = simd_lid * h_per_thread + i;
            if (h_idx < params.H) {
                partial += h_local[i] * float(weight[v * params.H + h_idx]);
            }
        }
        float logit = simd_sum(partial);  // Reduce across SIMD lanes

        // Immediately update online softmax - logit stays in registers!
        if (simd_lid == 0) {
            state.update(logit);
            if (v == target) target_logit = logit;
        }
    }

    // Combine simdgroup states (single barrier)
    // ... reduction code ...

    // Write final output
    if (simd_lid == 0 && simd_gid == 0) {
        float lse = final_state.logsumexp();
        logsumexp_out[token_idx] = lse;
        loss[token_idx] = lse - target_logit;
    }
}
```

**Key difference from current CCE:**
- No `steel_matmul` dispatch
- No `logits_chunk` buffer allocation
- Dot product computed inline, result stays in registers

**Memory savings:** Eliminates `logits_chunk` buffer entirely
- Current: `N × CHUNK_V × 4 = 268 MB` for N=4096
- FLCE: `0 MB` intermediate (just registers)

#### Level 3: Fused Forward + Backward (Advanced)

Compute gradients during forward pass (Liger-style):

```metal
[[kernel]] void flce_forward_backward_fused(
    const device T* hidden,
    const device T* weight,
    const device int32_t* targets,
    device float* loss,
    device float* logsumexp_out,
    device float* grad_hidden,          // Output: [N, H]
    device atomic<float>* grad_weight,  // Output: [V, H] with atomics
    constant FLCEParams& params,
    ...) {

    const int token_idx = tid;

    // === FORWARD PASS ===
    // Load hidden, compute online softmax (same as Level 2)
    float h_local[128];
    // ... load hidden ...

    OnlineSoftmax state;
    // ... compute logsumexp ...
    float lse = state.logsumexp();

    // === BACKWARD PASS (fused!) ===
    // Now we have LSE, compute gradients in same kernel

    float grad_h_local[128] = {0};  // Accumulate grad_hidden in registers

    for (int v = simd_gid; v < params.V; v += NUM_SIMDGROUPS) {
        // Recompute logit (or cache in threadgroup memory)
        float logit = compute_dot_product(h_local, weight[v]);

        // Compute d_logit = softmax - onehot
        float prob = safe_exp(logit - lse);
        float d_logit = prob - (v == target ? 1.0f : 0.0f);
        d_logit *= grad_output;  // Chain rule

        // Accumulate grad_hidden += d_logit * weight[v]
        for (int i = 0; i < h_per_thread; i++) {
            grad_h_local[i] += d_logit * weight[v * H + h_idx];
        }

        // Accumulate grad_weight[v] += d_logit * hidden (atomic)
        for (int i = 0; i < h_per_thread; i++) {
            atomic_fetch_add(&grad_weight[v * H + h_idx], d_logit * h_local[i]);
        }
    }

    // Write grad_hidden
    for (int i = 0; i < h_per_thread; i++) {
        grad_hidden[token_idx * H + h_idx] = grad_h_local[i];
    }
}
```

**Trade-offs:**
- Pro: Single kernel for forward + backward
- Pro: No intermediate storage between forward/backward
- Con: Recomputes logits (or needs to cache them)
- Con: Atomic contention on grad_weight

### 9.5 FLCE Architecture Decision Tree

```
                        FLCE Implementation Choice
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
            Small Batch (N < 2K)           Large Batch (N ≥ 2K)
                    │                              │
                    ▼                              ▼
         Level 2: Fused Forward           Current CCE + optimizations
         (single kernel, no chunks)       (chunked approach scales better)
                    │
                    ▼
              Works well?
              ┌────┴────┐
              ▼         ▼
             Yes        No
              │         │
              ▼         ▼
         Level 3:    Optimize Level 2
         Fused F+B   (memory layout, etc.)
```

### 9.6 Implementation Relationship to Current CCE

**Recommended: FLCE Extends CCE**

```
Current Architecture:
┌─────────────────────────────────────────────┐
│ mx.fast.cce_loss(hidden, weight, targets)   │
│     └─► CCELoss primitive                   │
│           └─► Chunked forward (steel_matmul + logsumexp kernel) │
│           └─► Chunked backward (steel_matmul + d_logits kernel) │
└─────────────────────────────────────────────┘

Proposed FLCE Architecture:
┌─────────────────────────────────────────────┐
│ mx.fast.flce_loss(hidden, weight, targets)  │  ← New entry point (or same)
│     └─► FLCELoss primitive                  │
│           │                                  │
│           ├─► Small batch path (N < 2K):    │
│           │     └─► flce_forward_fused      │  ← New kernel (Level 2)
│           │     └─► flce_backward_fused     │  ← New kernel
│           │                                  │
│           └─► Large batch path (N ≥ 2K):    │
│                 └─► Existing CCE chunked    │  ← Reuse current code!
└─────────────────────────────────────────────┘
```

**Key insight:** For large batches, our current chunked CCE is already good. FLCE primarily helps small batches by eliminating dispatch overhead.

### 9.7 FLCE Memory Savings Analysis

#### Current CCE Memory (N=4096, V=49152, H=576)

```
Forward:
- logits_chunk: 4096 × 16384 × 4 = 268 MB
- running_max/sum: 4096 × 4 × 2 = 32 KB
- Total forward intermediate: ~268 MB

Backward:
- logits_chunk: 268 MB (recomputed)
- d_logits: shares buffer with logits
- Total backward intermediate: ~268 MB

Peak: ~268 MB intermediate
```

#### FLCE Level 2 Memory (Same dimensions)

```
Forward:
- logits_chunk: 0 MB (computed in registers!)
- running_max/sum: 32 KB
- Total forward intermediate: ~32 KB

Backward:
- logits recomputed in registers
- grad_h_local in registers
- Total backward intermediate: ~0 MB (registers only)

Peak: ~32 KB intermediate (just logsumexp state)
```

#### Savings Comparison

| Approach | CE Intermediate Memory | Total Training Memory | Savings vs MLX |
|----------|------------------------|----------------------|----------------|
| MLX Baseline | ~800 MB | 4.81 GB | - |
| Current CCE | ~268 MB | 4.54 GB | 6% |
| FLCE Level 2 | ~0.03 MB | ~4.27 GB | 11% |
| FLCE Level 3 | ~0.03 MB | ~4.27 GB | 11% + speed |

**The additional savings are modest** (~250 MB) because MLX baseline is already efficient. But the **speed gains** from eliminating dispatch overhead are significant.

### 9.8 Concrete FLCE Implementation Steps

#### Step 1: Create FLCE Primitive (C++)

```cpp
// In mlx/fast_primitives.h
class FLCELoss : public Primitive {
public:
    void eval_gpu(const std::vector<array>& inputs,
                  std::vector<array>& outputs) override;

private:
    int ignore_index_;
    bool use_fused_kernel_;  // Toggle between fused and chunked
};
```

#### Step 2: Add Fused Forward Kernel (Metal)

```cpp
// In cce.cpp, add new dispatch path
if (N < 2048 && H <= 4096) {
    // Use fused kernel - no intermediate buffer!
    auto kernel = d.get_kernel("flce_forward_fused_" + type_to_name(h.dtype()));

    compute_encoder.set_input_array(h, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(t, 2);
    compute_encoder.set_output_array(loss, 3);
    compute_encoder.set_output_array(lse_out, 4);

    // One threadgroup per token
    MTL::Size grid = MTL::Size(N, 1, 1);
    MTL::Size group = MTL::Size(128, 1, 1);  // 4 simdgroups
    compute_encoder.dispatch_threadgroups(grid, group);

    // NO logits_chunk allocation!
} else {
    // Fall back to existing chunked approach
    dispatch_chunked_forward(...);
}
```

#### Step 3: Add Fused Backward Kernel (Metal)

Similar structure, but computes gradients without intermediate logits buffer.

#### Step 4: Python Binding

```python
# In mlx/fast.py (or use existing cce_loss with automatic path selection)
def flce_loss(hidden, weight, targets, ignore_index=-100):
    """Fused Linear Cross-Entropy Loss.

    Memory-efficient loss computation that never materializes
    the full [N, V] logits tensor.
    """
    return FLCELoss(ignore_index=ignore_index).apply(hidden, weight, targets)
```

### 9.9 FLCE Summary

| Question | Answer |
|----------|--------|
| Build on top of CCE? | **Yes** - reuse chunked path for large batches |
| New kernels needed? | **Yes** - `flce_forward_fused`, `flce_backward_fused` |
| Memory savings? | **Modest** (~5% additional) because MLX baseline is already good |
| Speed savings? | **Significant** for small batches (2-3x) by eliminating dispatch overhead |
| Complexity? | **Medium** - new kernels but similar structure to existing |

**The main value of FLCE on Apple Silicon is SPEED, not memory.** The fused kernel eliminates:
1. `steel_matmul` dispatch overhead
2. `logits_chunk` buffer allocation
3. Separate logsumexp kernel dispatch

This directly addresses the small-batch performance problem identified in Part 3.

---

## Part 10: Updated Implementation Roadmap

Given the analysis in Parts 8 and 9, here's the revised prioritized roadmap:

### Phase 1: Quick Wins (2-3 days)

| Task | Expected Impact | Complexity |
|------|----------------|------------|
| Add `flce_forward_fused` kernel for N < 2K | 2-3x speedup for small batches | Medium |
| Profile and fix batch=8 anomaly | Fix 3.6x regression | Low |
| Add adaptive CHUNK_V for edge cases | 10-20% for edge cases | Low |

### Phase 2: Complete FLCE (1 week)

| Task | Expected Impact | Complexity |
|------|----------------|------------|
| Add `flce_backward_fused` kernel | 2x speedup for backward | Medium |
| SIMD-first reduction (minimize barriers) | Additional 1.5x for backward | Medium |
| float4 vectorized loads in fused kernels | 1.3x for compute-bound | Low |

### Phase 3: Advanced Optimizations (2 weeks)

| Task | Expected Impact | Complexity |
|------|----------------|------------|
| Level 3: Combined forward+backward kernel | 1.3x overall | High |
| Function constants specialization | 5-15% overall | Medium |
| simdgroup_matrix for large H dimensions | Variable | High |

### Revised Success Metrics

| Metric | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| Batch=4 Speed | 0.98x | 2.0x | 2.5x | 3.0x |
| Batch=8 Speed | 0.28x | 1.5x | 2.0x | 2.5x |
| Batch=16 Speed | 1.00x | 1.5x | 1.8x | 2.0x |
| Batch=32 Speed | 1.23x | 1.3x | 1.5x | 1.8x |
| Memory (N=4K) | 4.54 GB | 4.3 GB | 4.3 GB | 4.3 GB |

---

## Final Conclusion

The CCE/FLCE optimization strategy for Apple Silicon differs fundamentally from CUDA-focused implementations:

1. **Memory savings are modest** because MLX baseline is already well-optimized
2. **Speed is the primary optimization target**, especially for small batches
3. **Dispatch overhead is the main bottleneck** for small N, not memory
4. **Fused kernels (FLCE)** eliminate dispatch overhead and provide the biggest wins
5. **Chunked approach remains valuable** for large batches where it prevents memory pressure

The path forward is clear: implement fused forward/backward kernels for small batches while keeping the chunked approach for large batches. This hybrid strategy leverages the strengths of both approaches on Apple Silicon's unique architecture.
