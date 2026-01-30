#!/usr/bin/env python3
"""Test if CCE vs baseline diverge over multiple training steps - supports BF16 and FP32"""
import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx.utils import tree_map

def to_fp32(x):
    if isinstance(x, mx.array) and x.dtype in [mx.bfloat16, mx.float16]:
        return x.astype(mx.float32)
    return x

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mlx-community/Qwen3-0.6B-bf16")
parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seq-length", type=int, default=128)
parser.add_argument("--steps", type=int, default=10)
args = parser.parse_args()

# Load model
print("Loading model...")
model, tok = load(args.model)

if args.dtype == "fp32":
    model.update(tree_map(to_fp32, model.parameters()))

mx.eval(model.parameters())
print(f"Model dtype: {model.model.embed_tokens.weight.dtype}")

# Create batch
tokens = tok.encode('Hello world. ' * 50)[:args.seq_length]
batch = mx.array([tokens] * args.batch_size, dtype=mx.int32)
lengths = mx.array([[0, len(tokens)-1]] * args.batch_size, dtype=mx.int32)

print(f"Batch: {args.batch_size} x {args.seq_length}")
print()

# Baseline loss function
def baseline_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    return ce.sum() / mask.sum(), mask.sum()

# CCE loss function
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

# Save initial weights (MLX uses mx.array() to copy)
initial_weights = tree_map(lambda x: mx.array(x) if isinstance(x, mx.array) else x, model.parameters())
mx.eval(initial_weights)

# Test baseline
print("=== BASELINE ===")
model.update(initial_weights)
mx.eval(model.parameters())
opt = optim.Adam(learning_rate=1e-4)
lvg = nn.value_and_grad(model, baseline_loss)

baseline_losses = []
for i in range(args.steps):
    (l, _), g = lvg(model, batch, lengths)
    opt.update(model, g)
    mx.eval(model.parameters(), opt.state, l)
    baseline_losses.append(l.item())
    print(f"  Step {i+1}: loss = {l.item():.6f}")

# Test CCE
print()
print("=== CCE ===")
model.update(initial_weights)
mx.eval(model.parameters())
opt = optim.Adam(learning_rate=1e-4)
lvg = nn.value_and_grad(model, cce_loss)

cce_losses = []
for i in range(args.steps):
    (l, _), g = lvg(model, batch, lengths)
    opt.update(model, g)
    mx.eval(model.parameters(), opt.state, l)
    cce_losses.append(l.item())
    print(f"  Step {i+1}: loss = {l.item():.6f}")

# Compare
print()
print("=== COMPARISON ===")
for i in range(args.steps):
    diff = abs(baseline_losses[i] - cce_losses[i])
    rel = diff / baseline_losses[i] * 100
    print(f"  Step {i+1}: baseline={baseline_losses[i]:.6f}, cce={cce_losses[i]:.6f}, diff={diff:.6f} ({rel:.2f}%)")
