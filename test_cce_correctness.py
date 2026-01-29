#!/usr/bin/env python3
"""Test CCE correctness with stable training settings"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx.utils import tree_map

def to_fp32(x):
    if isinstance(x, mx.array) and x.dtype in [mx.bfloat16, mx.float16]:
        return x.astype(mx.float32)
    return x

# Load model
print("Loading model...")
model, tok = load('mlx-community/Qwen3-0.6B-bf16')
model.update(tree_map(to_fp32, model.parameters()))
mx.eval(model.parameters())

# Smaller batch for stability
batch_size = 16
seq_length = 128
tokens = tok.encode('Hello world. ' * 30)[:seq_length]
batch = mx.array([tokens] * batch_size, dtype=mx.int32)
lengths = mx.array([[0, len(tokens)-1]] * batch_size, dtype=mx.int32)

print(f"Batch: {batch_size} x {seq_length}")
print()

# Loss functions
def baseline_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    return ce.sum() / mask.sum(), mask.sum()

def cce_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    hidden = model.model(inputs)
    weight = model.model.embed_tokens.weight
    B, S, H = hidden.shape
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ce = mx.fast.cce_loss(hidden.reshape(B*S, H), weight, targets.reshape(B*S))
    return (ce * mask.reshape(B*S)).sum() / mask.sum(), mask.sum()

# Save initial weights
initial_weights = tree_map(lambda x: mx.array(x) if isinstance(x, mx.array) else x, model.parameters())
mx.eval(initial_weights)

# STABLE TRAINING SETTINGS
lr = 1e-6  # Much lower learning rate
num_steps = 20

print(f"Settings: lr={lr}, steps={num_steps}")
print()

# Test baseline
print("=== BASELINE ===")
model.update(initial_weights)
mx.eval(model.parameters())
opt = optim.Adam(learning_rate=lr)
lvg = nn.value_and_grad(model, baseline_loss)

baseline_losses = []
for i in range(num_steps):
    (l, _), g = lvg(model, batch, lengths)
    # Gradient clipping
    g = tree_map(lambda x: mx.clip(x, -1.0, 1.0) if isinstance(x, mx.array) else x, g)
    opt.update(model, g)
    mx.eval(model.parameters(), opt.state, l)
    baseline_losses.append(l.item())
    if (i + 1) % 5 == 0:
        print(f"  Step {i+1}: loss = {l.item():.6f}")

# Test CCE
print()
print("=== CCE ===")
model.update(initial_weights)
mx.eval(model.parameters())
opt = optim.Adam(learning_rate=lr)
lvg = nn.value_and_grad(model, cce_loss)

cce_losses = []
for i in range(num_steps):
    (l, _), g = lvg(model, batch, lengths)
    # Gradient clipping
    g = tree_map(lambda x: mx.clip(x, -1.0, 1.0) if isinstance(x, mx.array) else x, g)
    opt.update(model, g)
    mx.eval(model.parameters(), opt.state, l)
    cce_losses.append(l.item())
    if (i + 1) % 5 == 0:
        print(f"  Step {i+1}: loss = {l.item():.6f}")

# Compare
print()
print("=== COMPARISON ===")
max_diff = 0
for i in range(num_steps):
    diff = abs(baseline_losses[i] - cce_losses[i])
    rel = diff / (baseline_losses[i] + 1e-10) * 100
    max_diff = max(max_diff, rel)
    if (i + 1) % 5 == 0:
        print(f"  Step {i+1}: baseline={baseline_losses[i]:.6f}, cce={cce_losses[i]:.6f}, diff={rel:.4f}%")

print()
print(f"Max relative difference: {max_diff:.4f}%")
if max_diff < 1.0:
    print("✓ CCE is working correctly (< 1% difference)")
else:
    print("✗ CCE may have issues (> 1% difference)")
