#!/usr/bin/env python3
"""Debug logsumexp values in CCE forward/backward"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

# Load model
print("Loading model...")
model, tok = load('mlx-community/Qwen3-0.6B-bf16')
mx.eval(model.parameters())

for batch_size in [4, 8, 16]:
    print(f"\n{'='*60}")
    print(f"BATCH SIZE = {batch_size}")
    print(f"{'='*60}")

    # Create batch
    seq_length = 128
    tokens = tok.encode('Hello world. ' * 30)[:seq_length]
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

    # Compute logits for first few rows
    h_flat = hidden.reshape(N, H)

    # Baseline logsumexp (using MLX)
    print("\n--- Baseline LogSumExp (first 5 rows) ---")
    logits = h_flat @ weight.T  # [N, V] - this is expensive!
    mx.eval(logits)
    baseline_lse = mx.logsumexp(logits, axis=1)  # [N]
    mx.eval(baseline_lse)
    print(f"LSE values: {baseline_lse[:5].tolist()}")
    print(f"LSE mean: {baseline_lse.mean().item():.6f}")
    print(f"LSE min: {baseline_lse.min().item():.6f}")
    print(f"LSE max: {baseline_lse.max().item():.6f}")

    # CCE forward (which computes logsumexp internally)
    print("\n--- CCE Loss (should match baseline CE) ---")
    cce_loss = mx.fast.cce_loss(h_flat, weight, targets.reshape(N))
    mx.eval(cce_loss)
    print(f"CCE loss mean: {cce_loss.mean().item():.6f}")

    # Baseline CE for comparison
    baseline_ce = nn.losses.cross_entropy(logits, targets.reshape(N))
    mx.eval(baseline_ce)
    print(f"Baseline CE mean: {baseline_ce.mean().item():.6f}")
    print(f"Diff: {abs(cce_loss.mean().item() - baseline_ce.mean().item()):.10f}")

    # Check gradient magnitude
    print("\n--- Gradient Check ---")
    def cce_fn(h):
        return mx.fast.cce_loss(h.reshape(N, H), weight, targets.reshape(N)).mean()

    def baseline_fn(h):
        logits = h.reshape(N, H) @ weight.T
        return nn.losses.cross_entropy(logits, targets.reshape(N)).mean()

    _, cce_grad = mx.value_and_grad(cce_fn)(hidden)
    _, baseline_grad = mx.value_and_grad(baseline_fn)(hidden)
    mx.eval(cce_grad, baseline_grad)

    print(f"CCE grad: mean={cce_grad.mean().item():.10f}, max={cce_grad.abs().max().item():.6f}")
    print(f"Baseline grad: mean={baseline_grad.mean().item():.10f}, max={baseline_grad.abs().max().item():.6f}")
    print(f"Grad ratio (CCE/baseline): {(cce_grad.abs().max() / baseline_grad.abs().max()).item():.2f}x")

    # Clean up logits to free memory
    del logits, baseline_lse, cce_loss, baseline_ce
    mx.eval([])
