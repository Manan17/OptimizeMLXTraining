#!/usr/bin/env python3
"""Baseline benchmark for Llama 3.2 3B - mlx_lm default_loss"""
import time, gc, argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.trainer import default_loss
from mlx.utils import tree_map

def to_fp32(x):
    if isinstance(x, mx.array) and x.dtype in [mx.bfloat16, mx.float16]:
        return x.astype(mx.float32)
    return x

def to_bf16(x):
    if isinstance(x, mx.array) and x.dtype in [mx.float32, mx.float16]:
        return x.astype(mx.bfloat16)
    return x

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-bf16")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--seq-length", type=int, default=128)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--iters", type=int, default=15)
parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp32"],
                    help="Data type for training (auto=keep native, bf16, fp32)")
args = parser.parse_args()

print(f"BASELINE (default_loss) - 3B Model")
print(f"Model: {args.model}")
print(f"Batch: {args.batch_size} x {args.seq_length}")
print(f"Dtype: {args.dtype}")
print(f"Warmup: {args.warmup}, Iters: {args.iters}")
print()

# Setup
model, tok = load(args.model)

# Get vocab size and hidden dim
vocab_size = model.model.embed_tokens.weight.shape[0]
hidden_dim = model.model.embed_tokens.weight.shape[1]
print(f"Vocab size: {vocab_size}, Hidden dim: {hidden_dim}")

# Convert dtype if needed
native_dtype = model.model.embed_tokens.weight.dtype
if args.dtype == "fp32":
    model.update(tree_map(to_fp32, model.parameters()))
elif args.dtype == "bf16":
    model.update(tree_map(to_bf16, model.parameters()))
elif args.dtype == "auto":
    # FP16 is unstable for training - convert to BF16
    if native_dtype == mx.float16:
        print(f"Note: Converting FP16 model to BF16 for training stability")
        model.update(tree_map(to_bf16, model.parameters()))

mx.eval(model.parameters())
print(f"Model dtype: {model.model.embed_tokens.weight.dtype}")

# Create batch
tokens = tok.encode('Hello world. ' * 50)[:args.seq_length]
batch = mx.array([tokens] * args.batch_size, dtype=mx.int32)
lengths = mx.array([[0, len(tokens)-1]] * args.batch_size, dtype=mx.int32)

opt = optim.AdamW(learning_rate=5e-6)
lvg = nn.value_and_grad(model, default_loss)

# Check initial loss (before any training)
(init_l, _), _ = lvg(model, batch, lengths)
mx.eval(init_l)
print(f"Initial loss: {init_l.item()}")

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
print(f"  Final loss: {l.item()}")
