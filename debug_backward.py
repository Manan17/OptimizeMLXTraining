#!/usr/bin/env python3
"""Debug the backward pass specifically"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

# Load model
print("Loading model...")
model, tok = load('mlx-community/Qwen3-0.6B-bf16')
mx.eval(model.parameters())

for batch_size in [4, 8, 16, 32]:
    print(f"\n{'='*60}")
    print(f"BATCH SIZE = {batch_size}")
    print(f"{'='*60}")

    # Create batch with fixed token count
    seq_length = 128
    # Use a longer string to get more tokens
    tokens = tok.encode('The quick brown fox jumps over the lazy dog. ' * 20)[:seq_length]
    batch = mx.array([tokens] * batch_size, dtype=mx.int32)

    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    hidden = model.model(inputs)
    weight = model.model.embed_tokens.weight
    mx.eval(hidden, weight)

    B, S, H = hidden.shape
    N = B * S
    V = weight.shape[0]
    print(f"N = {N}, V = {V}, H = {H}")
    print(f"N >= 256: {N >= 256}, V > 2000: {V > 2000}")

    h_flat = hidden.reshape(N, H)
    t_flat = targets.reshape(N)

    # Test gradient with different approaches
    print("\n--- Test 1: Direct CCE backward ---")
    def cce_fn1(h):
        return mx.fast.cce_loss(h, weight, t_flat).mean()

    loss1, grad1 = mx.value_and_grad(cce_fn1)(h_flat)
    mx.eval(loss1, grad1)
    print(f"Loss: {loss1.item():.6f}")
    print(f"Grad: mean={grad1.mean().item():.10f}, max={grad1.abs().max().item():.6f}, sum={grad1.sum().item():.6f}")

    # Test 2: Baseline
    print("\n--- Test 2: Baseline (h @ w.T -> cross_entropy) ---")
    def baseline_fn(h):
        logits = h @ weight.T
        return nn.losses.cross_entropy(logits, t_flat).mean()

    loss2, grad2 = mx.value_and_grad(baseline_fn)(h_flat)
    mx.eval(loss2, grad2)
    print(f"Loss: {loss2.item():.6f}")
    print(f"Grad: mean={grad2.mean().item():.10f}, max={grad2.abs().max().item():.6f}, sum={grad2.sum().item():.6f}")

    # Compare
    print(f"\n--- Comparison ---")
    print(f"Loss diff: {abs(loss1.item() - loss2.item()):.6f}")
    grad_ratio = (grad1.abs().max() / (grad2.abs().max() + 1e-10)).item()
    print(f"Grad max ratio (CCE/baseline): {grad_ratio:.2f}x")

    if grad_ratio > 10:
        print("*** GRADIENT BUG DETECTED! ***")
    else:
        print("Gradients OK")
