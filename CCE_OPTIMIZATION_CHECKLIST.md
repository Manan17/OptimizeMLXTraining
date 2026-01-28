# CCE/FLCE Optimization Checklist for Apple Silicon

## Document Info
- **Created:** 2026-01-27
- **Purpose:** Complete inventory of optimizations (current, planned, potential)
- **Target:** All M-series Macs (M1, M2, M3, M4 and future)

---

## Important: simdgroup_matrix Compatibility

### Question: Is simdgroup_matrix only for M4 and above?

**Answer: NO.** `simdgroup_matrix` is available on **ALL M-series chips**.

| GPU Family | Chips | simdgroup_matrix | Metal Version |
|------------|-------|------------------|---------------|
| Apple7 | A14, **M1** | ✅ Supported | Metal 3 |
| Apple8 | A15, A16, **M2** | ✅ Supported | Metal 3 |
| Apple9 | A17, **M3, M4** | ✅ Supported | Metal 3 |
| Apple10 | **M5** (future) | ✅ Supported | Metal 4 |

**Source:** [Apple Developer - Discover Metal enhancements for A14 Bionic](https://developer.apple.com/videos/play/tech-talks/10858/)

> "The A14 introduces a brand new set of SIMD scope instructions which allows you to implement large matrix multiplies very efficiently."

> "If you are making apps targeting the A14 on iOS, you will find those same advanced features on Apple M1."

**Conclusion:** All optimizations using `simdgroup_matrix` will work on M1 and later.

---

## Part 1: Current Optimizations (Already Implemented)

### 1.1 Memory Management ✅

| Optimization | File Location | Description |
|--------------|---------------|-------------|
| LSE Caching | `cce.cpp:503-524` | Saves logsumexp from forward, reuses in backward (~0.5GB saved) |
| FP32 Buffer Aliasing | `cce.cpp:479-482` | grad_hidden_f32/grad_weight_f32 are aliases, not separate buffers |
| In-place d_logits | `cce.cpp:668-672` | d_logits shares buffer with logits_chunk via `copy_shared_buffer()` |
| Shared Buffer Views | `cce.cpp:162-172` | Weight chunks are views, not copies |
| Contiguity Check | `cce.cpp:71-82` | Avoids unnecessary copies of row-contiguous arrays |
| Temporary Tracking | `cce.cpp:252-257` | `add_temporary()` ensures proper GPU memory lifetime |

### 1.2 Numerical Stability ✅

| Optimization | File Location | Description |
|--------------|---------------|-------------|
| Online Softmax | `kernels.h:41-67` | Welford-style online logsumexp (never stores full exp array) |
| Safe Exponential | `kernels.h:29-35` | Clamps input to 88.0f to prevent overflow |
| FP32 Accumulation | `cce.cpp:84-94` | Uses FP32 for reductions even with BF16 inputs |
| Probability Clamping | `kernels.h` | `clamp(prob, 0.0f, 1.0f)` after softmax |
| Log Safety | `kernels.h:64-66` | `log(sum + 1e-9f)` prevents log(0) |

### 1.3 Parallelization ✅

| Optimization | File Location | Description |
|--------------|---------------|-------------|
| 2D Tiling | `kernels.h:697-862` | Grid of (B_tiles, V_tiles) for massive parallelism |
| SIMD Reductions | `kernels.h` | `simd_sum()`, `simd_max()` for lane-parallel reduction |
| 4 Simdgroups | `params.h:26` | 128 threads = 4 × 32 simdgroups per threadgroup |
| Interleaved Vocab | `kernels.h:258` | Each simdgroup handles vocab entries: v, v+4, v+8, ... |
| Online State Merge | `kernels.h:55-62` | Combines partial softmax states from simdgroups |

### 1.4 Kernel Variants ✅

| Variant | Condition | File Location |
|---------|-----------|---------------|
| Chunked Forward | V > 2000 && N ≥ 8 | `cce.cpp:120-208` |
| 2D Tiled Forward | Fallback from chunked | `cce.cpp:321-385` |
| MMA Forward | H % 8 == 0 | `cce.cpp:259-319` |
| SIMD Forward | Small problems | `cce.cpp:386-409` |
| Chunked Backward | Large batches | `cce.cpp:626-726` |
| 2D Tiled Backward | With sparsity | `cce.cpp:737-795` |
| SIMD Backward | Fallback | `cce.cpp:797-850` |

### 1.5 Hardware Acceleration ✅

| Optimization | File Location | Description |
|--------------|---------------|-------------|
| simdgroup_matrix (MMA) | `kernels.h:1120-1325` | 8×8 matrix fragments for hardware matmul |
| Lock-free Atomic LSE | `kernels.h:668-695` | Compare-and-swap for atomic logaddexp |
| Atomic Gradient Accum | `kernels.h:474-477` | `atomic_fetch_add_explicit` with relaxed ordering |

### 1.6 Sparsity Exploitation ✅

| Optimization | File Location | Description |
|--------------|---------------|-------------|
| Sparsity Threshold | `params.h:29` | CCE_SPARSITY_THRESHOLD = 2.44e-4 |
| Tile Skip | `kernels.h:1000-1026` | Skip vocab tiles if max(d_logits) < threshold |
| ~99.98% Skip Rate | Backward pass | Most gradients are near-zero in typical LLMs |

### 1.7 Compile-Time Specialization ✅

| Optimization | File Location | Description |
|--------------|---------------|-------------|
| 24 Kernel Variants | `steel_cce.metal:86-205` | 3 dtypes × 8 configurations |
| Tile Size Constants | `params.h:19-28` | BLOCK_B=32, BLOCK_V=128, BLOCK_D=32 |
| Type Templates | `kernels.h` | `template <typename T, int BV, int BH>` |

---

## Part 2: Optimizations To Implement (Planned)

### 2.1 Fused Kernel for Small Batches (HIGH PRIORITY)

**Problem:** For N < 2048, dispatch overhead dominates (6+ dispatches per forward)

**Solution:** Single-dispatch kernel that fuses matmul + logsumexp

```metal
[[kernel]] void flce_forward_fused(
    const device T* hidden,      // [N, H]
    const device T* weight,      // [V, H]
    const device int32_t* targets,
    device float* loss,
    device float* logsumexp_out,
    ...) {

    // Load hidden into registers ONCE
    float h_local[128];

    // Stream through ALL vocab in single kernel
    // No intermediate logits buffer!
    for (int v = simd_gid; v < V; v += 4) {
        float logit = dot(h_local, weight[v]);  // In registers
        online_softmax.update(logit);
    }
}
```

**Expected Impact:** 2-3x speedup for small batches

**Applies to:** All M-series (M1+)

### 2.2 N_READS Vectorization (HIGH PRIORITY)

**Problem:** Sequential vocab processing, one element at a time

**Solution:** Batch-load vocab entries into thread-local arrays

```metal
template <int N_READS = 4>
[[kernel]] void cce_forward_vectorized(...) {
    float logits_local[N_READS];

    for (int v_base = simd_gid * N_READS; v_base < V; v_base += 4 * N_READS) {
        // Load N_READS logits at once
        for (int i = 0; i < N_READS; i++) {
            logits_local[i] = compute_logit(v_base + i);
        }
        // Process all N_READS
        for (int i = 0; i < N_READS; i++) {
            online_softmax.update(logits_local[i]);
        }
    }
}
```

**Source:** `mlx/backend/metal/kernels/logsumexp.h`, `softmax.h`

**Expected Impact:** 1.5-2x improvement in main loop

**Applies to:** All M-series

### 2.3 Adaptive Tile Sizes (MEDIUM PRIORITY)

**Problem:** Fixed BLOCK sizes don't suit all problem sizes

**Solution:** Size-based tile selection

```cpp
// In cce.cpp dispatch logic
int block_v, block_b;
if (V < 1024) {
    block_v = 64; block_b = 16;
} else if (V < 8192) {
    block_v = 128; block_b = 32;
} else {
    block_v = 256; block_b = 32;
}

if (N < 256) {
    block_b = 16;  // Smaller batch tiles for small N
}
```

**Source:** `mlx/backend/metal/matmul.cpp:544-551, 1796-1805`

**Expected Impact:** 15-20% improvement at size boundaries

**Applies to:** All M-series

### 2.4 Safe/Unsafe Code Path Split (MEDIUM PRIORITY)

**Problem:** Bounds checking on every access, even when unnecessary

**Solution:** Function constants for alignment-based specialization

```metal
constant bool V_ALIGNED [[function_constant(0)]];  // V % BLOCK_V == 0
constant bool H_ALIGNED [[function_constant(1)]];  // H % 32 == 0

[[kernel]] void cce_forward_specialized(...) {
    if (V_ALIGNED) {
        load_unsafe();  // No bounds check
    } else {
        load_safe(bounds);
    }
}
```

**Source:** GEMM kernels with `align_M`, `align_N`, `align_K`

**Expected Impact:** 5-10% for aligned cases

**Applies to:** All M-series

### 2.5 float4 Vectorized Memory Access (MEDIUM PRIORITY)

**Problem:** Scalar loads waste memory bandwidth

**Solution:** Use vector types for coalesced access

```metal
// Instead of:
for (int i = 0; i < 4; i++) {
    partial += h[i] * w[i];
}

// Use:
float4 h_vec = *reinterpret_cast<const device float4*>(&hidden[offset]);
float4 w_vec = *reinterpret_cast<const device float4*>(&weight[offset]);
partial += dot(h_vec, w_vec);
```

**Expected Impact:** 1.3-1.5x for memory-bound cases

**Applies to:** All M-series (requires H % 4 == 0)

### 2.6 Adaptive CHUNK_V for Small Batches (LOW PRIORITY)

**Problem:** Fixed CHUNK_V=16384 creates overhead for small N

**Solution:** Dynamic chunk sizing

```cpp
int get_chunk_v(int N, int V) {
    if (N <= 512) return std::min(8192, V);
    if (N <= 1024) return std::min(12288, V);
    if (N <= 2048) return std::min(16384, V);
    return std::min(32768, V);
}
```

**Note:** Previous attempt caused regressions - needs careful testing

**Expected Impact:** 10-20% for small batches

**Applies to:** All M-series

---

## Part 3: Additional Optimization Opportunities

### 3.1 Barrier Optimization

**Current:** May be using `mem_threadgroup` barriers where `mem_none` suffices

**Optimization:**
```metal
// Current (heavy):
threadgroup_barrier(mem_flags::mem_threadgroup);

// When only sync needed (light):
threadgroup_barrier(mem_flags::mem_none);

// Within simdgroup (free):
simdgroup_barrier(mem_flags::mem_none);
```

**Expected Impact:** 5-10% reduction in barrier stalls

### 3.2 Register Pressure Optimization

**Current:** Large register arrays for hidden state

**Optimization:** Reduce `h_local[128]` when H is small

```metal
// Adaptive register allocation
constexpr int MAX_H_PER_THREAD = 64;  // vs 128
// Or use constexpr based on H:
constexpr int H_LOCAL_SIZE = (HIDDEN_SIZE + 31) / 32;
```

**Expected Impact:** Better occupancy for small H

### 3.3 Warp-Level Primitives

**Optimization:** Use `simd_shuffle` for faster data exchange

```metal
// Instead of threadgroup memory for small reductions:
float val = simd_shuffle(my_val, source_lane);

// Broadcast from lane 0:
float broadcast = simd_broadcast(val, 0);
```

**Expected Impact:** Reduced threadgroup memory pressure

### 3.4 Prefetching Weight Data

**Optimization:** Software prefetch for weight tiles

```metal
// Prefetch next weight tile while processing current
device const T* next_weight = weight + (v + 4) * H;
// Use __builtin_prefetch or async copy if available
```

**Expected Impact:** Hides memory latency (5-15%)

### 3.5 Loop Unrolling with Pragma

**Optimization:** Explicit unroll for hot loops

```metal
#pragma unroll 4
for (int i = 0; i < h_per_thread; i++) {
    partial += h_local[i] * weight_val;
}
```

**Expected Impact:** Better instruction scheduling (5-10%)

### 3.6 Separate Kernels for Training vs Inference

**Optimization:** Training doesn't need some inference optimizations and vice versa

```cpp
if (is_training) {
    dispatch_cce_forward_training(...);  // Saves LSE, prepares for backward
} else {
    dispatch_cce_forward_inference(...);  // Minimal outputs
}
```

**Expected Impact:** Reduced memory writes for inference

### 3.7 Fused Forward+Backward Kernel (Advanced)

**Optimization:** Compute gradients during forward pass (Liger-style)

```metal
[[kernel]] void flce_forward_backward_fused(...) {
    // Forward: compute LSE
    // Immediately compute d_logits while data is in registers
    // Accumulate gradients
}
```

**Trade-off:** Requires storing d_logits OR recomputing logits

**Expected Impact:** 1.3x overall when backward-bound

### 3.8 BF16 Throughout (No FP32 Accumulation)

**Optimization:** For memory-bound cases, skip FP32 conversion

```cpp
// Current: BF16 input → FP32 accumulation → BF16 output
// Proposed: BF16 input → BF16 accumulation → BF16 output
// With Kahan summation for precision
```

**Trade-off:** Numerical precision vs memory bandwidth

**Expected Impact:** 1.5x memory throughput

---

## Part 4: Optimization Priority Matrix

### By Impact × Effort

| Priority | Optimization | Impact | Effort | Batch Size Benefit |
|----------|-------------|--------|--------|-------------------|
| 🔴 P0 | Fused kernel for small batches | 2-3x | Medium | N < 2K |
| 🔴 P0 | Fix batch=8 anomaly | 3.6x fix | Low | N = 1024 |
| 🟡 P1 | N_READS vectorization | 1.5-2x | Low | All |
| 🟡 P1 | float4 memory access | 1.3-1.5x | Low | All |
| 🟡 P1 | Adaptive tile sizes | 15-20% | Medium | Edge cases |
| 🟢 P2 | Safe/unsafe code paths | 5-10% | Medium | Aligned sizes |
| 🟢 P2 | Barrier optimization | 5-10% | Low | All |
| 🟢 P2 | Loop unrolling | 5-10% | Low | All |
| 🔵 P3 | Fused forward+backward | 1.3x | High | All |
| 🔵 P3 | Adaptive CHUNK_V | 10-20% | Medium | Small batches |
| 🔵 P3 | Prefetching | 5-15% | Medium | Large V |

### By Hardware Compatibility

| Optimization | M1 | M2 | M3 | M4 | Notes |
|--------------|----|----|----|----|-------|
| simdgroup_matrix | ✅ | ✅ | ✅ | ✅ | Apple7+ |
| simd_shuffle | ✅ | ✅ | ✅ | ✅ | All Metal |
| float4 vectorization | ✅ | ✅ | ✅ | ✅ | All Metal |
| Function constants | ✅ | ✅ | ✅ | ✅ | All Metal |
| Atomic float | ✅ | ✅ | ✅ | ✅ | Via CAS |
| BF16 native | ✅ | ✅ | ✅ | ✅ | All M-series |

**All proposed optimizations work on M1 and later.**

---

## Part 5: Implementation Checklist

### Phase 1: Quick Wins (Week 1)

- [ ] **P0:** Create `flce_forward_fused` kernel for N < 2048
- [ ] **P0:** Profile batch=8 anomaly with Metal System Trace
- [ ] **P0:** Test CHUNK_V=12288 for N=1024 case
- [ ] **P1:** Add N_READS=4 vectorization to forward kernel
- [ ] **P1:** Add float4 loads for H dimension

### Phase 2: Performance Polish (Week 2)

- [ ] **P1:** Implement adaptive tile size selection
- [ ] **P2:** Add function constants for alignment specialization
- [ ] **P2:** Replace `mem_threadgroup` barriers with `mem_none` where safe
- [ ] **P2:** Add `#pragma unroll` to hot loops
- [ ] Create `flce_backward_fused` kernel

### Phase 3: Advanced (Week 3+)

- [ ] **P3:** Implement fused forward+backward kernel
- [ ] **P3:** Add software prefetching for weight tiles
- [ ] **P3:** Separate training vs inference kernel paths
- [ ] Benchmark all optimizations across M1/M2/M3/M4

---

## Part 6: Testing Requirements

### Correctness Tests

1. **Numerical accuracy:** Loss values match baseline within 1e-5
2. **Gradient accuracy:** Gradients match autograd within 1e-4
3. **Edge cases:** V=1, V=49152, N=1, N=8192, H=64, H=4096
4. **Ignore index:** Proper handling of -100 targets
5. **BF16 precision:** No NaN/Inf in mixed precision

### Performance Tests

1. **Small batch:** N=512, 1024, 2048 (target: 2x speedup)
2. **Large batch:** N=4096, 8192 (target: 1.5x speedup)
3. **Memory:** Peak GPU memory vs baseline
4. **Dispatch count:** Verify single dispatch for small batches

### Hardware Tests

1. **M1:** Base compatibility
2. **M2:** Performance scaling
3. **M3:** GPU core scaling
4. **M4:** (if available) Latest features

---

## References

- [Apple Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Discover Metal enhancements for A14 Bionic](https://developer.apple.com/videos/play/tech-talks/10858/)
- [Tailor your Metal apps for Apple M1](https://developer.apple.com/videos/play/tech-talks/10859/)
- [metal-benchmarks - Apple GPU microarchitecture](https://github.com/philipturner/metal-benchmarks)
- [Apple vs. Oranges: M-Series HPC Performance](https://arxiv.org/html/2502.05317v1)
