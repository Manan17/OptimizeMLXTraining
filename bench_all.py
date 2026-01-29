#!/usr/bin/env python3
"""Run all CCE vs Baseline benchmarks."""
import time, gc
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

def run_bench(loss_fn, batch_size, tokens, warmup=3, iters=7):
    batch = mx.array([tokens]*batch_size, dtype=mx.int32)
    lengths = mx.array([[0,len(tokens)-1]]*batch_size, dtype=mx.int32)

    m,_ = load('HuggingFaceTB/SmolLM2-135M')
    m.update(tree_map(to_fp32, m.parameters()))
    mx.eval(m.parameters())

    opt = optim.Adam(learning_rate=1e-4)
    lvg = nn.value_and_grad(m, loss_fn)

    for _ in range(warmup):
        (l,_),g = lvg(m,batch,lengths); opt.update(m,g); mx.eval(m.parameters(),opt.state,l)

    gc.collect(); mx.synchronize(); mx.reset_peak_memory()
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        (l,_),g = lvg(m,batch,lengths); opt.update(m,g); mx.eval(m.parameters(),opt.state,l)
    mx.synchronize(); t = time.perf_counter() - t0

    return t/iters*1000, mx.get_peak_memory()/1e9

print("="*60)
print("CCE vs BASELINE BENCHMARK")
print("="*60)
print(f"MLX: {mx.__version__}")
print(f"Model: SmolLM2-135M (fp32)")
print(f"Seq length: 128, Warmup: 3, Iters: 7")
print()

_,tok = load('HuggingFaceTB/SmolLM2-135M')
tokens = tok.encode('Hello world. '*30)[:128]

results = []
for bs in [4, 8, 16, 32]:
    print(f"Batch size {bs}...")
    base_t, base_m = run_bench(default_loss, bs, tokens)
    gc.collect(); mx.synchronize()
    cce_t, cce_m = run_bench(cce_loss, bs, tokens)
    gc.collect(); mx.synchronize()
    results.append((bs, base_t, base_m, cce_t, cce_m))
    print(f"  Baseline: {base_t:.1f} ms, {base_m:.3f} GB")
    print(f"  CCE:      {cce_t:.1f} ms, {cce_m:.3f} GB")

print()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"{'Batch':<8} {'Base(ms)':<12} {'CCE(ms)':<12} {'Speedup':<10} {'Base Mem':<10} {'CCE Mem':<10} {'Mem Ratio':<10}")
print("-"*72)
for bs, bt, bm, ct, cm in results:
    print(f"{bs:<8} {bt:<12.1f} {ct:<12.1f} {bt/ct:<10.2f}x {bm:<10.3f} {cm:<10.3f} {bm/cm:<10.2f}x")
