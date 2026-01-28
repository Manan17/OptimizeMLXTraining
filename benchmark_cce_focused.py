#!/usr/bin/env python3
"""
Focused CCE Benchmark - Single model, multiple batch sizes
Tests: Correctness (loss matching), Speed, Memory

Based on train_cce_comparison.py approach which works correctly.
"""

import time
import argparse
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from datasets import load_dataset as hf_load_dataset


@dataclass
class TestResult:
    batch_size: int
    seq_length: int
    N: int  # batch_size * seq_length

    # Baseline metrics
    baseline_memory_gb: float
    baseline_time_per_step_ms: float
    baseline_total_time_s: float
    baseline_final_loss: float
    baseline_losses: list

    # CCE metrics
    cce_memory_gb: float
    cce_time_per_step_ms: float
    cce_total_time_s: float
    cce_final_loss: float
    cce_losses: list

    # Comparisons
    memory_ratio: float  # baseline/cce (>1 = CCE saves memory)
    speedup: float  # baseline/cce time (>1 = CCE faster)
    loss_diff: float  # |baseline - cce| final loss
    losses_match: bool  # losses within tolerance

    status: str
    notes: str = ""


def get_lm_head_weight(model):
    """Get the language model head weight matrix."""
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight
    else:
        raise ValueError("Cannot find lm_head weight in model")


def get_hidden_states(model, inputs):
    """Get hidden states before lm_head projection."""
    if hasattr(model, 'model'):
        return model.model(inputs)
    else:
        raise ValueError("Cannot find transformer model")


def baseline_loss_fn(model, batch, lengths):
    """Standard cross-entropy loss (baseline) - same as train_cce_comparison.py."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / ntoks

    return ce, ntoks


def cce_loss_fn(model, batch, lengths):
    """CCE loss - same as train_cce_comparison.py."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    hidden = get_hidden_states(model, inputs)
    weight = get_lm_head_weight(model)

    # CCE kernel requires float32
    if hidden.dtype != mx.float32:
        hidden = hidden.astype(mx.float32)
    if weight.dtype != mx.float32:
        weight = weight.astype(mx.float32)

    B, S, H = hidden.shape
    hidden_flat = hidden.reshape(B * S, H)
    targets_flat = targets.reshape(B * S).astype(mx.int32)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    mask_flat = mask.reshape(B * S).astype(mx.float32)

    ce = mx.fast.cce_loss(hidden_flat, weight, targets_flat)
    ce = ce * mask_flat
    ntoks = mask_flat.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


def create_batch_iterator(dataset, tokenizer, batch_size, max_seq_length):
    """Create batches from dataset - same as train_cce_comparison.py."""
    all_tokens = []
    for item in dataset:
        tokens = tokenizer.encode(item["text"])
        if len(tokens) > 2:
            all_tokens.append(tokens[:max_seq_length])

    all_tokens.sort(key=len)

    while True:
        indices = np.random.permutation(len(all_tokens))
        for i in range(0, len(indices) - batch_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_tokens = [all_tokens[j] for j in batch_indices]

            max_len = min(max(len(t) for t in batch_tokens), max_seq_length)
            padded = []
            lengths = []

            for tokens in batch_tokens:
                pad_len = max_len - len(tokens)
                padded.append(tokens + [tokenizer.pad_token_id or 0] * pad_len)
                lengths.append([0, len(tokens) - 1])

            yield mx.array(padded), mx.array(lengths)


def run_training(model, tokenizer, train_data, batch_size, seq_length, num_steps, warmup_steps, use_cce):
    """Run training and return metrics."""

    optimizer = optim.Adam(learning_rate=1e-4)
    loss_fn = cce_loss_fn if use_cce else baseline_loss_fn
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    batch_iter = create_batch_iterator(train_data, tokenizer, batch_size, seq_length)

    # Warmup
    for _ in range(warmup_steps):
        batch, lengths = next(batch_iter)
        (loss, ntoks), grads = loss_value_and_grad(model, batch, lengths)
        optimizer.update(model, grads)
        mx.eval(loss)

    # Reset memory tracking
    mx.clear_cache()
    mx.synchronize()
    mx.reset_peak_memory()

    # Training loop
    losses = []
    start_time = time.perf_counter()

    for step in range(num_steps):
        batch, lengths = next(batch_iter)
        (loss, ntoks), grads = loss_value_and_grad(model, batch, lengths)
        optimizer.update(model, grads)
        mx.eval(loss)
        losses.append(float(loss))

    mx.synchronize()
    total_time = time.perf_counter() - start_time
    peak_memory = mx.get_peak_memory() / (1024**3)

    return {
        'memory_gb': peak_memory,
        'time_per_step_ms': (total_time / num_steps) * 1000,
        'total_time_s': total_time,
        'final_loss': losses[-1],
        'losses': losses
    }


def run_test(model_name: str, batch_size: int, seq_length: int, num_steps: int = 10) -> TestResult:
    """Run a complete test for one batch size."""

    N = batch_size * seq_length
    print(f"\n{'='*60}")
    print(f"Testing batch_size={batch_size}, seq_length={seq_length}, N={N}")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    model, tokenizer = load(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    weight = get_lm_head_weight(model)
    V, H = weight.shape
    print(f"Vocab: {V}, Hidden: {H}")

    # Load dataset
    print("Loading dataset...")
    raw_train = hf_load_dataset("roneneldan/TinyStories", split="train[:500]")
    train_data = [{"text": ex["text"]} for ex in raw_train]

    result = TestResult(
        batch_size=batch_size,
        seq_length=seq_length,
        N=N,
        baseline_memory_gb=0,
        baseline_time_per_step_ms=0,
        baseline_total_time_s=0,
        baseline_final_loss=0,
        baseline_losses=[],
        cce_memory_gb=0,
        cce_time_per_step_ms=0,
        cce_total_time_s=0,
        cce_final_loss=0,
        cce_losses=[],
        memory_ratio=0,
        speedup=0,
        loss_diff=0,
        losses_match=False,
        status="pending"
    )

    # Run baseline
    print(f"\nRunning BASELINE training ({num_steps} steps)...")
    try:
        # Reload model for fresh state
        model, tokenizer = load(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        baseline = run_training(model, tokenizer, train_data, batch_size, seq_length,
                               num_steps, warmup_steps=2, use_cce=False)

        result.baseline_memory_gb = baseline['memory_gb']
        result.baseline_time_per_step_ms = baseline['time_per_step_ms']
        result.baseline_total_time_s = baseline['total_time_s']
        result.baseline_final_loss = baseline['final_loss']
        result.baseline_losses = baseline['losses']

        print(f"  Memory: {baseline['memory_gb']:.2f} GB")
        print(f"  Time/step: {baseline['time_per_step_ms']:.0f} ms")
        print(f"  Final loss: {baseline['final_loss']:.4f}")

    except Exception as e:
        print(f"  BASELINE FAILED: {e}")
        result.status = "baseline_error"
        result.notes = str(e)
        return result

    mx.clear_cache()
    mx.synchronize()

    # Run CCE
    print(f"\nRunning CCE training ({num_steps} steps)...")
    try:
        # Reload model for fresh state
        model, tokenizer = load(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        cce = run_training(model, tokenizer, train_data, batch_size, seq_length,
                          num_steps, warmup_steps=2, use_cce=True)

        result.cce_memory_gb = cce['memory_gb']
        result.cce_time_per_step_ms = cce['time_per_step_ms']
        result.cce_total_time_s = cce['total_time_s']
        result.cce_final_loss = cce['final_loss']
        result.cce_losses = cce['losses']

        print(f"  Memory: {cce['memory_gb']:.2f} GB")
        print(f"  Time/step: {cce['time_per_step_ms']:.0f} ms")
        print(f"  Final loss: {cce['final_loss']:.4f}")

    except Exception as e:
        print(f"  CCE FAILED: {e}")
        result.status = "cce_error"
        result.notes = str(e)
        return result

    # Calculate comparisons
    result.memory_ratio = result.baseline_memory_gb / result.cce_memory_gb
    result.speedup = result.baseline_time_per_step_ms / result.cce_time_per_step_ms
    result.loss_diff = abs(result.baseline_final_loss - result.cce_final_loss)
    result.losses_match = result.loss_diff < 0.5  # Within 0.5 tolerance
    result.status = "success"

    print(f"\nRESULTS:")
    print(f"  Memory ratio: {result.memory_ratio:.2f}x {'(CCE saves)' if result.memory_ratio > 1 else '(Baseline better)'}")
    print(f"  Speedup: {result.speedup:.2f}x {'(CCE faster)' if result.speedup > 1 else '(Baseline faster)'}")
    print(f"  Loss diff: {result.loss_diff:.4f} {'(MATCH)' if result.losses_match else '(MISMATCH)'}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Focused CCE Benchmark")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M",
                        help="Model to test")
    parser.add_argument("--seq-length", type=int, default=128,
                        help="Sequence length")
    parser.add_argument("--steps", type=int, default=10,
                        help="Training steps per test")
    parser.add_argument("--output", type=str, default="cce_benchmark_results.txt",
                        help="Output file")
    args = parser.parse_args()

    batch_sizes = [4, 8, 16, 32]

    print("="*70)
    print("FOCUSED CCE BENCHMARK")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Steps per test: {args.steps}")
    print(f"Batch sizes: {batch_sizes}")
    print("="*70)

    results = []

    for batch_size in batch_sizes:
        result = run_test(args.model, batch_size, args.seq_length, args.steps)
        results.append(result)
        mx.clear_cache()
        mx.synchronize()

    # Write results to file
    output = []
    output.append("="*80)
    output.append("CCE BENCHMARK RESULTS")
    output.append(f"Model: {args.model}")
    output.append(f"Sequence Length: {args.seq_length}")
    output.append(f"Steps: {args.steps}")
    output.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("="*80)

    # Summary table
    output.append("\n" + "="*80)
    output.append("SUMMARY TABLE")
    output.append("="*80)
    output.append(f"{'Batch':<8} {'N':<8} {'Base Mem':<12} {'CCE Mem':<12} {'Mem Ratio':<12} {'Speedup':<10} {'Loss Match':<12} {'Status':<10}")
    output.append("-"*80)

    for r in results:
        loss_match = "YES" if r.losses_match else "NO"
        output.append(
            f"{r.batch_size:<8} {r.N:<8} {r.baseline_memory_gb:<12.2f} {r.cce_memory_gb:<12.2f} "
            f"{r.memory_ratio:<12.2f} {r.speedup:<10.2f} {loss_match:<12} {r.status:<10}"
        )

    # Detailed timing
    output.append("\n" + "="*80)
    output.append("TIMING DETAILS (ms/step)")
    output.append("="*80)
    output.append(f"{'Batch':<8} {'Baseline':<15} {'CCE':<15} {'Speedup':<10}")
    output.append("-"*50)
    for r in results:
        output.append(f"{r.batch_size:<8} {r.baseline_time_per_step_ms:<15.0f} {r.cce_time_per_step_ms:<15.0f} {r.speedup:<10.2f}x")

    # Loss comparison
    output.append("\n" + "="*80)
    output.append("LOSS COMPARISON (Correctness Check)")
    output.append("="*80)
    output.append(f"{'Batch':<8} {'Baseline Loss':<15} {'CCE Loss':<15} {'Diff':<10} {'Match':<8}")
    output.append("-"*60)
    for r in results:
        match = "YES" if r.losses_match else "NO"
        output.append(f"{r.batch_size:<8} {r.baseline_final_loss:<15.4f} {r.cce_final_loss:<15.4f} {r.loss_diff:<10.4f} {match:<8}")

    # Bottleneck analysis
    output.append("\n" + "="*80)
    output.append("BOTTLENECK ANALYSIS")
    output.append("="*80)

    for r in results:
        if r.status != "success":
            output.append(f"\nBatch {r.batch_size}: {r.status} - {r.notes}")
            continue

        issues = []
        if r.memory_ratio < 1.0:
            output.append(f"\nBatch {r.batch_size} (N={r.N}):")
            output.append(f"  - CCE uses MORE memory ({r.memory_ratio:.2f}x)")
            if r.N < 2048:
                output.append(f"  - Small N={r.N}: CCE chunk overhead dominates")

        if r.speedup < 1.0:
            if r.memory_ratio >= 1.0:
                output.append(f"\nBatch {r.batch_size} (N={r.N}):")
            output.append(f"  - CCE is SLOWER ({r.speedup:.2f}x)")
            output.append(f"  - Possible cause: kernel dispatch overhead for small batches")

        if not r.losses_match:
            output.append(f"\nBatch {r.batch_size}: CORRECTNESS ISSUE")
            output.append(f"  - Loss difference: {r.loss_diff:.4f}")

    # Recommendations
    output.append("\n" + "="*80)
    output.append("RECOMMENDATIONS")
    output.append("="*80)

    success = [r for r in results if r.status == "success"]
    if success:
        good_mem = [r for r in success if r.memory_ratio > 1.0]
        good_speed = [r for r in success if r.speedup > 1.0]

        if good_mem:
            best_mem = max(good_mem, key=lambda r: r.memory_ratio)
            output.append(f"Best memory savings: batch={best_mem.batch_size} ({best_mem.memory_ratio:.2f}x)")

        if good_speed:
            best_speed = max(good_speed, key=lambda r: r.speedup)
            output.append(f"Best speedup: batch={best_speed.batch_size} ({best_speed.speedup:.2f}x)")

        if good_mem:
            min_n = min(r.N for r in good_mem)
            output.append(f"\nCCE recommended for N >= {min_n}")

    output_text = "\n".join(output)

    with open(args.output, 'w') as f:
        f.write(output_text)

    print("\n" + output_text)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
