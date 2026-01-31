#!/usr/bin/env python3
"""Debug gradient flow through CCE vs baseline"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx.utils import tree_map

# Load model
print("Loading model...")
model, tok = load('mlx-community/Qwen3-0.6B-bf16')
mx.eval(model.parameters())
print(f"Model dtype: {model.model.embed_tokens.weight.dtype}")

# Create batch (same as test_cce_correctness.py)
batch_size = 8
seq_length = 128
tokens = tok.encode('Hello world. ' * 30)[:seq_length]
batch = mx.array([tokens] * batch_size, dtype=mx.int32)
lengths = mx.array([[0, len(tokens)-1]] * batch_size, dtype=mx.int32)

print(f"Batch: {batch_size} x {seq_length}")
print(f"N = {batch_size * (seq_length - 1)}")
print()

# Compute inputs/targets
inputs = batch[:, :-1]
targets = batch[:, 1:]

# Get hidden states and weight
hidden = model.model(inputs)
weight = model.model.embed_tokens.weight
mx.eval(hidden, weight)

B, S, H = hidden.shape
print(f"Hidden shape: {hidden.shape}, dtype: {hidden.dtype}")
print(f"Weight shape: {weight.shape}, dtype: {weight.dtype}")

# Test 1: Simple mean (like test_bf16_simple.py)
print("\n=== Test 1: CCE with .mean() ===")
def cce_simple(h, w, t):
    return mx.fast.cce_loss(h.reshape(B*S, H), w, t.reshape(B*S)).mean()

loss1, grad1 = mx.value_and_grad(cce_simple)(hidden, weight, targets)
mx.eval(loss1, grad1)
print(f"Loss: {loss1.item():.6f}")
print(f"Grad hidden: mean={grad1.mean().item():.10f}, std={grad1.std().item():.10f}")
print(f"Grad hidden sum: {grad1.sum().item():.6f}")

# Test 2: Masked reduction (like test_cce_correctness.py)
print("\n=== Test 2: CCE with masked FP32 reduction ===")
def cce_masked(h, w, t):
    ce = mx.fast.cce_loss(h.reshape(B*S, H), w, t.reshape(B*S))
    steps = mx.arange(1, S + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ntoks = mask.sum()
    ce = (ce * mask.reshape(B*S)).astype(mx.float32).sum() / ntoks
    return ce

loss2, grad2 = mx.value_and_grad(cce_masked)(hidden, weight, targets)
mx.eval(loss2, grad2)
print(f"Loss: {loss2.item():.6f}")
print(f"Grad hidden: mean={grad2.mean().item():.10f}, std={grad2.std().item():.10f}")
print(f"Grad hidden sum: {grad2.sum().item():.6f}")

# Test 3: Without mask but with FP32 reduction
print("\n=== Test 3: CCE with FP32 reduction (no mask) ===")
def cce_fp32_reduce(h, w, t):
    ce = mx.fast.cce_loss(h.reshape(B*S, H), w, t.reshape(B*S))
    return ce.astype(mx.float32).sum() / (B * S)

loss3, grad3 = mx.value_and_grad(cce_fp32_reduce)(hidden, weight, targets)
mx.eval(loss3, grad3)
print(f"Loss: {loss3.item():.6f}")
print(f"Grad hidden: mean={grad3.mean().item():.10f}, std={grad3.std().item():.10f}")
print(f"Grad hidden sum: {grad3.sum().item():.6f}")

# Test 4: Baseline cross_entropy with same masked reduction
print("\n=== Test 4: Baseline with masked FP32 reduction ===")
def baseline_masked(h, w, t):
    logits = h @ w.T  # [B*S, V]
    ce = nn.losses.cross_entropy(logits.reshape(B, S, -1), t)  # [B, S]
    steps = mx.arange(1, S + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ntoks = mask.sum()
    ce = (ce * mask).astype(mx.float32).sum() / ntoks
    return ce

loss4, grad4 = mx.value_and_grad(baseline_masked)(hidden, weight, targets)
mx.eval(loss4, grad4)
print(f"Loss: {loss4.item():.6f}")
print(f"Grad hidden: mean={grad4.mean().item():.10f}, std={grad4.std().item():.10f}")
print(f"Grad hidden sum: {grad4.sum().item():.6f}")

# Compare gradients
print("\n=== Gradient Comparison ===")
print(f"Test 1 vs Test 2: grad diff = {(grad1 - grad2).abs().max().item():.10f}")
print(f"Test 1 vs Test 3: grad diff = {(grad1 - grad3).abs().max().item():.10f}")
print(f"Test 1 vs Test 4: grad diff = {(grad1 - grad4).abs().max().item():.10f}")
print(f"Test 2 vs Test 4: grad diff = {(grad2 - grad4).abs().max().item():.10f}")

# Check if gradients have same sign
def sign_match(a, b):
    return ((a * b) > 0).astype(mx.float32).mean().item()

print(f"\nSign agreement (should be ~1.0):")
print(f"Test 1 vs Test 2: {sign_match(grad1, grad2):.4f}")
print(f"Test 1 vs Test 4: {sign_match(grad1, grad4):.4f}")
print(f"Test 2 vs Test 4: {sign_match(grad2, grad4):.4f}")
