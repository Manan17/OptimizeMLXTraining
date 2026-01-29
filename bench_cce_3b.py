#!/usr/bin/env python3
"""CCE benchmark for Llama 3.2 3B - mx.fast.cce_loss"""
import time, gc, argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx.utils import tree_map

def to_fp32(x):
    if isinstance(x, mx.array) and x.dtype in [mx.bfloat16, mx.float16]:
        return x.astype(mx.float32)
    return x

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

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-bf16")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--seq-length", type=int, default=128)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--iters", type=int, default=7)
args = parser.parse_args()

print(f"CCE (mx.fast.cce_loss) - 3B Model")
print(f"Model: {args.model}")
print(f"Batch: {args.batch_size} x {args.seq_length}")
print(f"Warmup: {args.warmup}, Iters: {args.iters}")
print()

# Setup
model, tok = load(args.model)

# Get vocab size and hidden dim
vocab_size = model.model.embed_tokens.weight.shape[0]
hidden_dim = model.model.embed_tokens.weight.shape[1]
print(f"Vocab size: {vocab_size}, Hidden dim: {hidden_dim}")

# Convert to fp32 (CCE backward requires fp32)
model.update(tree_map(to_fp32, model.parameters()))
mx.eval(model.parameters())
print(f"Model dtype: {model.model.embed_tokens.weight.dtype}")

# Create batch
tokens = tok.encode('Hello world. ' * 50)[:args.seq_length]
batch = mx.array([tokens] * args.batch_size, dtype=mx.int32)
lengths = mx.array([[0, len(tokens)-1]] * args.batch_size, dtype=mx.int32)

opt = optim.Adam(learning_rate=1e-4)
lvg = nn.value_and_grad(model, cce_loss)

# Warmup
print(f"Warmup...")
for _ in range(args.warmup):
    (l,_),g = lvg(model, batch, lengths)
    opt.update(model, g)
    mx.eval(model.parameters(), opt.state, l)

gc.collect()
mx.synchronize()
mx.reset_peak_memory()

# Benchmark
print(f"Running {args.iters} iterations...")
mx.synchronize()
t0 = time.perf_counter()
for _ in range(args.iters):
    (l,_),g = lvg(model, batch, lengths)
    opt.update(model, g)
    mx.eval(model.parameters(), opt.state, l)
mx.synchronize()
total_time = time.perf_counter() - t0

print()
print(f"RESULTS:")
print(f"  Time/iter: {total_time/args.iters*1000:.1f} ms")
print(f"  Peak mem:  {mx.get_peak_memory()/1e9:.3f} GB")
