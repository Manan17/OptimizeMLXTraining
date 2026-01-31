#!/usr/bin/env python3
"""Debug logsumexp in backward - check if it's being passed correctly"""
import mlx.core as mx
import mlx.nn as nn

mx.random.seed(42)

for N in [360, 720, 2048]:
    print(f"\n{'='*60}")
    print(f"N = {N}")
    print(f"{'='*60}")

    H = 1024
    V = 151936

    # Create inputs
    h = mx.random.normal((N, H)).astype(mx.bfloat16)
    w = mx.random.normal((V, H)).astype(mx.bfloat16)
    t = mx.random.randint(0, V, (N,))
    mx.eval(h, w, t)

    # Compute baseline logsumexp for reference
    print("\n--- Baseline LogSumExp ---")
    h_fp32 = h.astype(mx.float32)
    w_fp32 = w.astype(mx.float32)
    logits = h_fp32 @ w_fp32.T
    mx.eval(logits)
    baseline_lse = mx.logsumexp(logits, axis=1)
    mx.eval(baseline_lse)
    print(f"Baseline LSE: mean={baseline_lse.mean().item():.2f}, min={baseline_lse.min().item():.2f}, max={baseline_lse.max().item():.2f}")
    del logits  # Free memory

    # Check forward CCE loss
    print("\n--- Forward CCE ---")
    cce_loss = mx.fast.cce_loss(h, w, t)
    mx.eval(cce_loss)
    print(f"CCE loss mean: {cce_loss.mean().item():.6f}")

    # Compare loss to expected: loss = logsumexp - target_logit
    # So logsumexp ≈ loss + target_logit

    # Check backward with simple gradient
    print("\n--- Backward Check ---")
    def simple_fn(h):
        loss = mx.fast.cce_loss(h, w, t)
        return loss.sum()  # Use sum instead of mean to avoid 1/N scaling

    total_loss, grad = mx.value_and_grad(simple_fn)(h)
    mx.eval(total_loss, grad)

    grad_fp32 = grad.astype(mx.float32)
    print(f"Total loss: {total_loss.item():.2f}")
    print(f"Grad: mean={grad_fp32.mean().item():.10f}, max={grad_fp32.abs().max().item():.6f}")

    # Check for NaN/Inf
    has_nan = mx.isnan(grad_fp32).any().item()
    has_inf = mx.isinf(grad_fp32).any().item()
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")

    # Compare to baseline gradient
    print("\n--- Baseline Gradient ---")
    def baseline_fn(h):
        logits = h.astype(mx.float32) @ w_fp32.T
        return nn.losses.cross_entropy(logits, t).sum()

    _, grad_base = mx.value_and_grad(baseline_fn)(h)
    mx.eval(grad_base)
    grad_base_fp32 = grad_base.astype(mx.float32)
    print(f"Baseline grad: mean={grad_base_fp32.mean().item():.10f}, max={grad_base_fp32.abs().max().item():.6f}")

    # Check ratio
    ratio = (grad_fp32.abs().max() / (grad_base_fp32.abs().max() + 1e-10)).item()
    print(f"Ratio (CCE/Baseline): {ratio:.2f}x")
