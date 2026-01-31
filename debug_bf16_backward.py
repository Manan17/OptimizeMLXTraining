#!/usr/bin/env python3
"""Isolate the BF16 backward issue"""
import mlx.core as mx
import mlx.nn as nn

# Test with smaller vocab to simplify
mx.random.seed(42)

for N in [360, 720, 1440, 2048]:
    print(f"\n{'='*60}")
    print(f"N = {N}")
    print(f"{'='*60}")

    H = 1024
    V = 151936  # Full vocab

    # Create random inputs in BF16
    h_bf16 = mx.random.normal((N, H)).astype(mx.bfloat16)
    w_bf16 = mx.random.normal((V, H)).astype(mx.bfloat16)
    t = mx.random.randint(0, V, (N,))
    mx.eval(h_bf16, w_bf16, t)

    # Also create FP32 versions
    h_fp32 = h_bf16.astype(mx.float32)
    w_fp32 = w_bf16.astype(mx.float32)
    mx.eval(h_fp32, w_fp32)

    print(f"Inputs: h={h_bf16.shape}, w={w_bf16.shape}, t={t.shape}")

    # Test 1: Forward only (both should work)
    print("\n--- Forward ---")
    cce_bf16 = mx.fast.cce_loss(h_bf16, w_bf16, t)
    cce_fp32 = mx.fast.cce_loss(h_fp32, w_fp32, t)
    mx.eval(cce_bf16, cce_fp32)
    print(f"CCE BF16 loss mean: {cce_bf16.mean().item():.6f}")
    print(f"CCE FP32 loss mean: {cce_fp32.mean().item():.6f}")
    print(f"Forward diff: {abs(cce_bf16.mean().item() - cce_fp32.mean().item()):.6f}")

    # Test 2: Backward with BF16 vs FP32
    print("\n--- Backward (CCE) ---")
    def cce_fn_bf16(h):
        return mx.fast.cce_loss(h, w_bf16, t).mean()

    def cce_fn_fp32(h):
        return mx.fast.cce_loss(h, w_fp32, t).mean()

    _, grad_bf16 = mx.value_and_grad(cce_fn_bf16)(h_bf16)
    _, grad_fp32 = mx.value_and_grad(cce_fn_fp32)(h_fp32)
    mx.eval(grad_bf16, grad_fp32)

    print(f"BF16 grad: mean={grad_bf16.astype(mx.float32).mean().item():.10f}, max={grad_bf16.astype(mx.float32).abs().max().item():.6f}")
    print(f"FP32 grad: mean={grad_fp32.mean().item():.10f}, max={grad_fp32.abs().max().item():.6f}")
    grad_ratio = (grad_bf16.astype(mx.float32).abs().max() / (grad_fp32.abs().max() + 1e-10)).item()
    print(f"Grad max ratio (BF16/FP32): {grad_ratio:.2f}x")

    # Test 3: Baseline (h @ w.T -> cross_entropy) as reference
    print("\n--- Baseline (matmul + CE) ---")
    def baseline_bf16(h):
        logits = h @ w_bf16.T
        return nn.losses.cross_entropy(logits, t).mean()

    def baseline_fp32(h):
        logits = h @ w_fp32.T
        return nn.losses.cross_entropy(logits, t).mean()

    _, grad_base_bf16 = mx.value_and_grad(baseline_bf16)(h_bf16)
    _, grad_base_fp32 = mx.value_and_grad(baseline_fp32)(h_fp32)
    mx.eval(grad_base_bf16, grad_base_fp32)

    print(f"Baseline BF16 grad: mean={grad_base_bf16.astype(mx.float32).mean().item():.10f}, max={grad_base_bf16.astype(mx.float32).abs().max().item():.6f}")
    print(f"Baseline FP32 grad: mean={grad_base_fp32.mean().item():.10f}, max={grad_base_fp32.abs().max().item():.6f}")

    # Compare CCE to baseline
    print("\n--- CCE vs Baseline ---")
    cce_vs_base_bf16 = (grad_bf16.astype(mx.float32).abs().max() / (grad_base_bf16.astype(mx.float32).abs().max() + 1e-10)).item()
    cce_vs_base_fp32 = (grad_fp32.abs().max() / (grad_base_fp32.abs().max() + 1e-10)).item()
    print(f"CCE/Baseline ratio (BF16): {cce_vs_base_bf16:.2f}x")
    print(f"CCE/Baseline ratio (FP32): {cce_vs_base_fp32:.2f}x")

    if cce_vs_base_bf16 > 10:
        print("*** BF16 CCE BACKWARD BUG! ***")
    elif cce_vs_base_fp32 > 10:
        print("*** FP32 CCE BACKWARD BUG! ***")
    else:
        print("OK")
