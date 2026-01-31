#!/usr/bin/env python3
"""Test if forcing fallback path gives correct gradients"""
import mlx.core as mx
import mlx.nn as nn

mx.random.seed(42)
N = 360
H = 1024
V = 4000  # Needs to be > 2000 for BF16 support

# Create inputs
h = mx.random.normal((N, H)).astype(mx.bfloat16)
w = mx.random.normal((V, H)).astype(mx.bfloat16)
t = mx.random.randint(0, V, (N,))
mx.eval(h, w, t)

print(f"N={N}, H={H}, V={V}")

# Test 1: GPU CCE
print("\n--- GPU CCE ---")
def cce_fn(h):
    return mx.fast.cce_loss(h, w, t).mean()

loss, grad = mx.value_and_grad(cce_fn)(h)
mx.eval(loss, grad)
print(f"Loss: {loss.item():.6f}")
print(f"Grad: mean={grad.astype(mx.float32).mean().item():.10f}, max={grad.astype(mx.float32).abs().max().item():.6f}")

# Test 2: Baseline (manual computation)
print("\n--- Baseline (manual) ---")
def baseline_fn(h):
    logits = h @ w.T
    return nn.losses.cross_entropy(logits, t).mean()

loss_base, grad_base = mx.value_and_grad(baseline_fn)(h)
mx.eval(loss_base, grad_base)
print(f"Loss: {loss_base.item():.6f}")
print(f"Grad: mean={grad_base.astype(mx.float32).mean().item():.10f}, max={grad_base.astype(mx.float32).abs().max().item():.6f}")

# Compare
print("\n--- Comparison ---")
grad_ratio = (grad.astype(mx.float32).abs().max() / (grad_base.astype(mx.float32).abs().max() + 1e-10)).item()
print(f"Grad ratio (CCE/Baseline): {grad_ratio:.2f}x")

if grad_ratio > 10:
    print("*** BUG: CCE gradient is wrong ***")
else:
    print("OK: Gradients match")

# Also check with FP32
print("\n--- FP32 Check ---")
h_fp32 = h.astype(mx.float32)
w_fp32 = w.astype(mx.float32)

def cce_fn_fp32(h):
    return mx.fast.cce_loss(h, w_fp32, t).mean()

def baseline_fn_fp32(h):
    logits = h @ w_fp32.T
    return nn.losses.cross_entropy(logits, t).mean()

_, grad_cce_fp32 = mx.value_and_grad(cce_fn_fp32)(h_fp32)
_, grad_base_fp32 = mx.value_and_grad(baseline_fn_fp32)(h_fp32)
mx.eval(grad_cce_fp32, grad_base_fp32)

grad_ratio_fp32 = (grad_cce_fp32.abs().max() / (grad_base_fp32.abs().max() + 1e-10)).item()
print(f"FP32 Grad ratio (CCE/Baseline): {grad_ratio_fp32:.2f}x")
