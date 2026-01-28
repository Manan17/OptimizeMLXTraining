# MetalLoRA: Hardware-Optimized LoRA for Apple Silicon

## Abstract

MetalLoRA is a high-performance implementation of Low-Rank Adaptation (LoRA) for Apple Silicon, leveraging Metal's GPU compute capabilities. This document describes the key architectural decisions and optimization techniques that enable significant performance improvements over naive implementations.

## 1. Motivation

Standard LoRA implementations on Apple Silicon suffer from three primary bottlenecks:

| Bottleneck             | Cause                                | Impact                 |
| ---------------------- | ------------------------------------ | ---------------------- |
| Dispatch overhead      | Separate kernel launch per operation | 10-50μs per dispatch   |
| Memory bandwidth       | Redundant data movement              | Limited by 400GB/s UMA |
| Underutilized hardware | Scalar ALU operations                | <10% of peak FLOPS     |

MetalLoRA addresses these through operation fusion, hardware-specific optimizations, and memory-aware scheduling.

## 2. Core Optimizations

### 2.1 Operation Fusion

The LoRA forward pass computes:

```
h = W₀x + (α/r) · B(Ax)
```

A naive implementation requires four separate dispatches. MetalLoRA fuses these into a single kernel, eliminating dispatch overhead and enabling data reuse across operations.

### 2.2 simdgroup_matrix Acceleration

Apple Silicon GPUs contain dedicated matrix units accessed via the `simdgroup_matrix` API. These units compute 8×8 matrix multiplications in a single cycle, providing up to 16× throughput compared to scalar operations.

```metal
simdgroup_matrix<float, 8, 8> acc;
simdgroup_multiply_accumulate(acc, W_tile, x_tile, acc);
```

### 2.3 TBDR-Aware Memory Access

Apple GPUs use Tile-Based Deferred Rendering (TBDR), which maintains fast on-chip tile memory. MetalLoRA structures computation to:

1. Load input data into threadgroup memory once
2. Perform all intermediate computations without eviction
3. Write final results to device memory

This reduces effective memory bandwidth requirements by approximately 3×.

### 2.4 Mixed Precision Computation

MetalLoRA uses FP16 for storage and memory operations, with FP32 accumulators for numerical stability:

- **Storage**: FP16 (2× memory bandwidth efficiency)
- **Accumulation**: FP32 (maintains precision)
- **Output**: FP16 (reduced memory footprint)

### 2.5 Quantized Inference (QLoRA)

For memory-constrained deployments, MetalLoRA supports 4-bit quantized base weights using the NF4 (Normal Float 4) format from the QLoRA paper. This provides:

- 75% reduction in base weight memory
- On-the-fly dequantization with minimal overhead
- Full precision LoRA adapters preserved

## 3. Advanced Features

### 3.1 Multi-Adapter Serving

MetalLoRA supports batched inference with per-sample adapter selection:

```python
manager = MultiAdapterManager(base_model)
manager.add_adapter("user_1", weights_1)
manager.add_adapter("user_2", weights_2)
output = manager.forward_batched(x, adapter_ids=[0, 1, 0, 1])
```

This enables efficient multi-tenant serving without model duplication.

### 3.2 Speculative Decoding

For autoregressive generation, MetalLoRA implements speculative decoding:

1. Draft model proposes K tokens
2. Target model verifies all K tokens in parallel
3. Accept matching tokens, resample on mismatch

Expected speedup: 2-5× depending on draft model quality.

### 3.3 Incremental KV-Cache

LoRA-aware KV-cache updates fuse the projection and cache update operations, reducing redundant computation during incremental decoding.

## 4. Architecture

```
MetalLoRA/
├── kernels/
│   ├── lora_kernels.metal    # Fused forward, simdgroup, multi-adapter
│   ├── lora_train.metal      # Training kernels with gradient fusion
│   └── lora_quantized.metal  # INT4/NF4 quantized inference
├── python/metal_lora/
│   ├── layers.py             # LoRALinear, LoRAEmbedding
│   ├── trainer.py            # Training infrastructure
│   └── optimizations.py      # Compression, pooling, speculative
└── TECHNICAL.md
```

## 5. Expected Performance

| Configuration        | Metric             | Target       |
| -------------------- | ------------------ | ------------ |
| Fused vs unfused     | Dispatch reduction | 4×           |
| simdgroup vs scalar  | FLOPS utilization  | 16×          |
| FP16 vs FP32         | Memory bandwidth   | 2×           |
| QLoRA vs FP16        | Memory footprint   | 4× reduction |
| Speculative decoding | Token generation   | 2-5×         |

## 6. Limitations

- Requires Apple Silicon (M1 or later)
- macOS 13+ for full Metal 3 support
- Quantized kernels optimized for inference only

## References

1. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021
2. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," 2023
3. Apple, "Metal Best Practices Guide," 2024
4. Apple, "Optimizing Machine Learning on Apple Silicon," WWDC 2023
