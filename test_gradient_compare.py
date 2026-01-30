#!/usr/bin/env python3
"""Compare CCE vs baseline gradients directly - supports both BF16 and FP32"""
import argparse
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx.utils import tree_map, tree_flatten

def to_fp32(x):
    if isinstance(x, mx.array) and x.dtype in [mx.bfloat16, mx.float16]:
        return x.astype(mx.float32)
    return x

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mlx-community/Qwen3-0.6B-bf16")
parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seq-length", type=int, default=128)
args = parser.parse_args()

# Load model
model, tok = load(args.model)

if args.dtype == "fp32":
    model.update(tree_map(to_fp32, model.parameters()))

mx.eval(model.parameters())
print(f"Model dtype: {model.model.embed_tokens.weight.dtype}")

# Create batch
tokens = tok.encode('Hello world. ' * 20)[:args.seq_length]
batch = mx.array([tokens] * args.batch_size, dtype=mx.int32)
lengths = mx.array([[0, len(tokens)-1]] * args.batch_size, dtype=mx.int32)

print(f"Batch: {args.batch_size} x {args.seq_length}")
print(f"Vocab size: {model.model.embed_tokens.weight.shape[0]}")
print()

# Baseline loss function (from mlx_lm)
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

# Get value and grad functions
baseline_lvg = nn.value_and_grad(model, baseline_loss)
cce_lvg = nn.value_and_grad(model, cce_loss)

# Compute losses and gradients
print("Computing baseline...")
(baseline_l, _), baseline_grads = baseline_lvg(model, batch, lengths)
mx.eval(baseline_l, baseline_grads)

print("Computing CCE...")
(cce_l, _), cce_grads = cce_lvg(model, batch, lengths)
mx.eval(cce_l, cce_grads)

print()
print(f"Baseline loss: {baseline_l.item():.6f}")
print(f"CCE loss:      {cce_l.item():.6f}")
print(f"Difference:    {abs(baseline_l.item() - cce_l.item()):.6f}")
print()

# Flatten gradients and compare
baseline_flat = tree_flatten(baseline_grads)
cce_flat = tree_flatten(cce_grads)

print("Gradient comparison by layer:")
total_diff_norm = 0.0
total_baseline_norm = 0.0

for (name, bg), (_, cg) in zip(baseline_flat, cce_flat):
    # Convert to FP32 for comparison if needed
    bg_f32 = bg.astype(mx.float32) if bg.dtype != mx.float32 else bg
    cg_f32 = cg.astype(mx.float32) if cg.dtype != mx.float32 else cg

    bg_norm = mx.sqrt((bg_f32 ** 2).sum()).item()
    cg_norm = mx.sqrt((cg_f32 ** 2).sum()).item()
    diff = bg_f32 - cg_f32
    diff_norm = mx.sqrt((diff ** 2).sum()).item()

    total_diff_norm += diff_norm ** 2
    total_baseline_norm += bg_norm ** 2

    if diff_norm > 1e-4:  # Only print layers with significant differences
        rel_err = diff_norm / (bg_norm + 1e-10)
        print(f"  {name}: baseline={bg_norm:.4f}, cce={cg_norm:.4f}, diff={diff_norm:.4f}, rel={rel_err:.4f}")

total_diff_norm = total_diff_norm ** 0.5
total_baseline_norm = total_baseline_norm ** 0.5
print()
print(f"Total gradient diff norm: {total_diff_norm:.6f}")
print(f"Total baseline norm:      {total_baseline_norm:.6f}")
print(f"Total relative error:     {total_diff_norm / total_baseline_norm:.6f}")

# Check gradient dtypes
print()
print("Gradient dtypes:")
for name, g in cce_flat[:3]:
    print(f"  {name}: {g.dtype}")
print("  ...")
