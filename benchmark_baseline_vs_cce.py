#!/usr/bin/env python3
"""
Comprehensive Benchmark: Baseline Cross-Entropy vs Two-Tier CCE
================================================================

Models: Gemma 2B/7B, Qwen3 4B/8B, Llama 3.2 3B, Phi-3 Medium (6 total)
Configs: batch [8,16,32] x seq [512,1024,2048] (9 per model = 54 total)
Metrics: Time, Memory, Correctness (step-by-step losses)

FAIRNESS PROTOCOL:
==================
1. Data: Pre-tokenized batches with fixed seed, identical for both methods
2. Model weights: Saved once, restored before each run
3. Optimizer: Fresh Adafactor with identical hyperparams
4. Warmup: 15 steps for JIT/kernel selection
5. Timing: mx.synchronize() barriers for accurate GPU timing
6. Memory: Peak memory reset after warmup

NOTE: No RNG reset needed - modern LLMs don't use dropout.
"""

import os
import sys
import gc
import json
import time
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx.utils import tree_map
from datasets import load_dataset


# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = [
    ("google/gemma-2b", "Gemma-2B"),
    ("google/gemma-7b", "Gemma-7B"),
    ("Qwen/Qwen3-4B", "Qwen3-4B"),
    ("Qwen/Qwen3-8B", "Qwen3-8B"),
    ("meta-llama/Llama-3.2-3B", "Llama-3.2-3B"),
    ("microsoft/Phi-3-medium-4k-instruct", "Phi-3-Medium"),
]

BATCH_SIZES = [8, 16, 32]
SEQ_LENGTHS = [512, 1024, 2048]

WARMUP_STEPS = 15
MEASURE_STEPS = 100
COOLDOWN_SECONDS = 30

DATASET_NAME = "tatsu-lab/alpaca"
DATASET_SPLIT = "train"
SEED = 42

# Adafactor with fixed LR for reproducibility
LEARNING_RATE = 1e-5

# Skip first N configs (already completed)
# Gemma-2B: 1-9, Gemma-7B: 10-18, Qwen3-4B: 19-20 done
SKIP_CONFIGS = 20

# Skip configs where N > MAX_N (causes OOM/swap thrash on 128GB)
# Model-size specific limits based on actual memory usage
MAX_N_2B = 16384     # 2B models: N=16K uses ~111GB
MAX_N_3B_4B = 8192   # 3B-4B models: N=8K uses ~87GB
MAX_N_7B_8B = 4096   # 7B-8B models: N=4K uses ~83GB


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_lm_weight(model):
    """Get the language model head weight (handles tied/untied embeddings)."""
    if hasattr(model, 'lm_head') and model.lm_head is not None and hasattr(model.lm_head, 'weight'):
        return model.lm_head.weight
    return model.model.embed_tokens.weight


def get_max_n_for_model(model_name):
    """Get appropriate MAX_N based on model size."""
    name = model_name.upper()
    # 7B and 8B models
    if "7B" in name or "8B" in name:
        return MAX_N_7B_8B
    # 3B and 4B models
    if "3B" in name or "4B" in name:
        return MAX_N_3B_4B
    # 2B models (and anything else)
    return MAX_N_2B


def format_sample(sample):
    """Format an Alpaca sample into a training string."""
    if sample['input']:
        text = f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
    else:
        text = f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""
    return text


def create_batches(dataset, tokenizer, seed, batch_size, seq_len, num_batches):
    """Create deterministic batches from dataset."""
    mx.random.seed(seed)
    indices = mx.random.permutation(len(dataset)).tolist()

    batches = []
    for b in range(num_batches):
        start_idx = b * batch_size
        batch_indices = indices[start_idx:start_idx + batch_size]

        if len(batch_indices) < batch_size:
            # Wrap around if needed
            batch_indices += indices[:batch_size - len(batch_indices)]

        batch_tokens = []
        for idx in batch_indices:
            text = format_sample(dataset[idx])
            tokens = tokenizer.encode(text)

            # Truncate or pad to seq_len
            if len(tokens) >= seq_len:
                tokens = tokens[:seq_len]
            else:
                pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                tokens = tokens + [pad_id] * (seq_len - len(tokens))

            batch_tokens.append(tokens)

        batch = mx.array(batch_tokens, dtype=mx.int32)
        batches.append(batch)

    return batches


def save_weights(model):
    """Save model weights."""
    weights = tree_map(lambda x: mx.array(x) if isinstance(x, mx.array) else x,
                       model.parameters())
    mx.eval(weights)
    return weights


def restore_weights(model, weights):
    """Restore model weights."""
    model.update(weights)
    mx.eval(model.parameters())


def delete_model_cache(model_repo):
    """Delete model from HuggingFace cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = "models--" + model_repo.replace("/", "--")
    model_path = cache_dir / model_dir_name

    if model_path.exists():
        print(f"    Deleting cache: {model_path}")
        shutil.rmtree(model_path)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def baseline_loss_fn(model, batch):
    """Standard cross-entropy loss."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)
    loss = nn.losses.cross_entropy(logits, targets)

    # FP32 reduction for stability
    loss = loss.astype(mx.float32).mean()
    return loss


def cce_loss_fn(model, weight, batch):
    """CCE (Cut Cross-Entropy) loss with two-tier chunking."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    hidden = model.model(inputs)
    B, S, H = hidden.shape

    loss = mx.fast.cce_loss(
        hidden.reshape(B * S, H),
        weight,
        targets.reshape(B * S)
    )

    # FP32 reduction for stability
    loss = loss.astype(mx.float32).mean()
    return loss


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def run_baseline(model, batches, initial_weights, warmup_steps, measure_steps):
    """Run baseline cross-entropy training."""
    # Restore initial weights
    restore_weights(model, initial_weights)

    # Fresh optimizer
    optimizer = optim.Adafactor(
        learning_rate=LEARNING_RATE,
        relative_step=False,
        scale_parameter=False,
    )

    # Loss and grad function
    def loss_fn(model, batch):
        return baseline_loss_fn(model, batch)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup (JIT compilation, kernel selection)
    batch_idx = 0
    for _ in range(warmup_steps):
        batch = batches[batch_idx % len(batches)]
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        batch_idx += 1

    # Prepare for measurement
    gc.collect()
    mx.synchronize()
    mx.reset_peak_memory()

    # Measure with sync barriers
    losses = []
    mx.synchronize()
    start_time = time.perf_counter()

    for step in range(measure_steps):
        batch = batches[batch_idx % len(batches)]
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)
        print(f"        [BASELINE] step {step+1:3d}/{measure_steps}: loss={loss_val:.6f}")
        batch_idx += 1

    mx.synchronize()
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    peak_memory_gb = mx.get_peak_memory() / 1e9

    return {
        "losses": losses,
        "time_ms": total_time_ms,
        "memory_gb": peak_memory_gb
    }


def run_cce(model, weight, batches, initial_weights, warmup_steps, measure_steps):
    """Run CCE training."""
    # Restore initial weights
    restore_weights(model, initial_weights)

    # Fresh optimizer (identical to baseline)
    optimizer = optim.Adafactor(
        learning_rate=LEARNING_RATE,
        relative_step=False,
        scale_parameter=False,
    )

    # Loss and grad function
    def loss_fn(model, batch):
        return cce_loss_fn(model, weight, batch)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup (JIT compilation, kernel selection)
    batch_idx = 0
    for _ in range(warmup_steps):
        batch = batches[batch_idx % len(batches)]
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        batch_idx += 1

    # Prepare for measurement
    gc.collect()
    mx.synchronize()
    mx.reset_peak_memory()

    # Measure with sync barriers
    losses = []
    mx.synchronize()
    start_time = time.perf_counter()

    for step in range(measure_steps):
        batch = batches[batch_idx % len(batches)]
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_val = loss.item()
        losses.append(loss_val)
        print(f"        [CCE] step {step+1:3d}/{measure_steps}: loss={loss_val:.6f}")
        batch_idx += 1

    mx.synchronize()
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    peak_memory_gb = mx.get_peak_memory() / 1e9

    return {
        "losses": losses,
        "time_ms": total_time_ms,
        "memory_gb": peak_memory_gb
    }


def compute_gradient_diff(model, weight, batch, initial_weights):
    """Compute gradient difference between baseline and CCE (single step, no optimizer)."""
    # Restore weights for baseline
    restore_weights(model, initial_weights)

    def baseline_fn(model, batch):
        return baseline_loss_fn(model, batch)

    _, baseline_grads = nn.value_and_grad(model, baseline_fn)(model, batch)
    mx.eval(baseline_grads)

    # Restore weights for CCE
    restore_weights(model, initial_weights)

    def cce_fn(model, batch):
        return cce_loss_fn(model, weight, batch)

    _, cce_grads = nn.value_and_grad(model, cce_fn)(model, batch)
    mx.eval(cce_grads)

    # Compare gradients
    max_diff = 0.0
    total_diff = 0.0
    count = 0

    def compare_grads(baseline, cce):
        nonlocal max_diff, total_diff, count
        if isinstance(baseline, mx.array) and isinstance(cce, mx.array):
            diff = mx.abs(baseline.astype(mx.float32) - cce.astype(mx.float32))
            max_diff = max(max_diff, diff.max().item())
            total_diff += diff.sum().item()
            count += diff.size

    tree_map(compare_grads, baseline_grads, cce_grads)

    mean_diff = total_diff / count if count > 0 else 0.0

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff
    }


def compare_results(baseline, cce):
    """Compare baseline and CCE results."""
    baseline_losses = baseline["losses"]
    cce_losses = cce["losses"]

    # Loss differences
    loss_diffs = [abs(b - c) for b, c in zip(baseline_losses, cce_losses)]
    loss_diffs_pct = [abs(b - c) / (abs(b) + 1e-10) * 100
                     for b, c in zip(baseline_losses, cce_losses)]

    # Speedup and memory savings
    speedup = baseline["time_ms"] / cce["time_ms"] if cce["time_ms"] > 0 else 0
    memory_savings_pct = ((baseline["memory_gb"] - cce["memory_gb"]) /
                          baseline["memory_gb"] * 100) if baseline["memory_gb"] > 0 else 0

    return {
        "loss_max_diff": max(loss_diffs),
        "loss_mean_diff": sum(loss_diffs) / len(loss_diffs),
        "loss_max_diff_pct": max(loss_diffs_pct),
        "loss_mean_diff_pct": sum(loss_diffs_pct) / len(loss_diffs_pct),
        "speedup": speedup,
        "memory_savings_pct": memory_savings_pct
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(output_file=None):
    """Run the full benchmark."""

    print("=" * 70)
    print("Benchmark: Baseline Cross-Entropy vs Two-Tier CCE")
    print("=" * 70)
    print(f"Models: {[m[1] for m in MODELS]}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Seq lengths: {SEQ_LENGTHS}")
    print(f"Warmup: {WARMUP_STEPS} steps, Measure: {MEASURE_STEPS} steps")
    print(f"Cooldown: {COOLDOWN_SECONDS}s between runs")
    print(f"Max N: {MAX_N_2B} (2B), {MAX_N_3B_4B} (3B-4B), {MAX_N_7B_8B} (7B-8B)")
    print(f"Optimizer: Adafactor (lr={LEARNING_RATE})")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Seed: {SEED}")
    print("=" * 70)

    # Load dataset once
    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Results storage
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "cooldown_seconds": COOLDOWN_SECONDS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adafactor",
            "dataset": DATASET_NAME,
            "seed": SEED,
            "models": [m[1] for m in MODELS],
            "batch_sizes": BATCH_SIZES,
            "seq_lengths": SEQ_LENGTHS
        },
        "results": [],
        "summary": {}
    }

    total_configs = len(MODELS) * len(BATCH_SIZES) * len(SEQ_LENGTHS)
    current_config = 0

    # Process each model
    for model_repo, model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name} ({model_repo})")
        print(f"{'='*70}")

        # Load model
        print(f"  Loading model...")
        try:
            model, tokenizer = load(model_repo)
            mx.eval(model.parameters())
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            # Record error for all configs of this model
            for batch_size in BATCH_SIZES:
                for seq_len in SEQ_LENGTHS:
                    current_config += 1
                    results["results"].append({
                        "model": model_name,
                        "model_repo": model_repo,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "N": batch_size * seq_len,
                        "status": "error",
                        "error": f"Model load failed: {str(e)}"
                    })
            continue

        # Get LM weight for CCE
        weight = get_lm_weight(model)
        print(f"  Model loaded. Weight shape: {weight.shape}, dtype: {weight.dtype}")

        # Save initial weights (before any training)
        print(f"  Saving initial weights...")
        initial_weights = save_weights(model)

        # Run all configs for this model
        for batch_size in BATCH_SIZES:
            for seq_len in SEQ_LENGTHS:
                current_config += 1
                N = batch_size * seq_len

                # Skip already completed configs
                if current_config <= SKIP_CONFIGS:
                    print(f"\n  [{current_config}/{total_configs}] "
                          f"batch={batch_size}, seq={seq_len}, N={N} - SKIPPED (already done)")
                    continue

                # Skip configs that would OOM/swap thrash
                max_n = get_max_n_for_model(model_name)
                if N > max_n:
                    print(f"\n  [{current_config}/{total_configs}] "
                          f"batch={batch_size}, seq={seq_len}, N={N} - SKIPPED (N > {max_n} for {model_name})")
                    continue

                print(f"\n  [{current_config}/{total_configs}] "
                      f"batch={batch_size}, seq={seq_len}, N={N}")

                # Create batches for this config
                num_batches_needed = WARMUP_STEPS + MEASURE_STEPS + 10  # Extra buffer
                batches = create_batches(
                    dataset, tokenizer, SEED,
                    batch_size, seq_len, num_batches_needed
                )
                print(f"    Created {len(batches)} batches of shape {batches[0].shape}")

                result = {
                    "model": model_name,
                    "model_repo": model_repo,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "N": N,
                    "weight_shape": list(weight.shape),
                    "status": "ok"
                }

                try:
                    # =========================================================
                    # BASELINE RUN
                    # =========================================================
                    print(f"    Running BASELINE...")
                    baseline_result = run_baseline(
                        model, batches, initial_weights,
                        WARMUP_STEPS, MEASURE_STEPS
                    )
                    result["baseline"] = baseline_result
                    print(f"      Time: {baseline_result['time_ms']:.2f}ms, "
                          f"Memory: {baseline_result['memory_gb']:.2f}GB, "
                          f"Final loss: {baseline_result['losses'][-1]:.4f}")

                    # Cooldown
                    print(f"    [Cooldown {COOLDOWN_SECONDS}s...]", end="", flush=True)
                    gc.collect()
                    mx.synchronize()
                    time.sleep(COOLDOWN_SECONDS)
                    print(" done")

                    # =========================================================
                    # CCE RUN
                    # =========================================================
                    print(f"    Running CCE...")
                    cce_result = run_cce(
                        model, weight, batches, initial_weights,
                        WARMUP_STEPS, MEASURE_STEPS
                    )
                    result["cce"] = cce_result
                    print(f"      Time: {cce_result['time_ms']:.2f}ms, "
                          f"Memory: {cce_result['memory_gb']:.2f}GB, "
                          f"Final loss: {cce_result['losses'][-1]:.4f}")

                    # =========================================================
                    # GRADIENT CHECK
                    # =========================================================
                    print(f"    Computing gradient diff...")
                    grad_diff = compute_gradient_diff(
                        model, weight, batches[0], initial_weights
                    )
                    result["gradient_diff"] = grad_diff
                    print(f"      Max: {grad_diff['max_diff']:.2e}, "
                          f"Mean: {grad_diff['mean_diff']:.2e}")

                    # =========================================================
                    # COMPARISON
                    # =========================================================
                    comparison = compare_results(baseline_result, cce_result)
                    comparison["grad_max_diff"] = grad_diff["max_diff"]
                    comparison["grad_mean_diff"] = grad_diff["mean_diff"]
                    result["comparison"] = comparison

                    print(f"    Results: "
                          f"Speedup={comparison['speedup']:.2f}x, "
                          f"MemSave={comparison['memory_savings_pct']:.1f}%, "
                          f"MaxLossDiff={comparison['loss_max_diff_pct']:.2f}%")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    result["status"] = "error"
                    result["error"] = str(e)

                results["results"].append(result)

                # Cooldown before next config
                if current_config < total_configs:
                    print(f"    [Cooldown {COOLDOWN_SECONDS}s...]", end="", flush=True)
                    gc.collect()
                    mx.synchronize()
                    time.sleep(COOLDOWN_SECONDS)
                    print(" done")

        # Cleanup model
        print(f"\n  Cleaning up {model_name}...")
        del model, tokenizer, weight, initial_weights
        gc.collect()
        mx.synchronize()

        # Delete from cache
        delete_model_cache(model_repo)
        print(f"  Model cache deleted.")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [r for r in results["results"] if r["status"] == "ok"]

    if successful:
        avg_speedup = sum(r["comparison"]["speedup"] for r in successful) / len(successful)
        avg_mem_save = sum(r["comparison"]["memory_savings_pct"] for r in successful) / len(successful)
        max_loss_diff = max(r["comparison"]["loss_max_diff_pct"] for r in successful)
        avg_loss_diff = sum(r["comparison"]["loss_mean_diff_pct"] for r in successful) / len(successful)
        max_grad_diff = max(r["comparison"]["grad_max_diff"] for r in successful)

        results["summary"] = {
            "total_configs": total_configs,
            "successful": len(successful),
            "failed": total_configs - len(successful),
            "avg_speedup": avg_speedup,
            "avg_memory_savings_pct": avg_mem_save,
            "max_loss_diff_pct": max_loss_diff,
            "avg_loss_diff_pct": avg_loss_diff,
            "max_grad_diff": max_grad_diff,
            "all_correct": max_loss_diff < 5.0  # 5% threshold for BF16
        }

        print(f"Successful runs: {len(successful)}/{total_configs}")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Average memory savings: {avg_mem_save:.1f}%")
        print(f"Max loss difference: {max_loss_diff:.2f}%")
        print(f"Average loss difference: {avg_loss_diff:.2f}%")
        print(f"Max gradient difference: {max_grad_diff:.2e}")
        print(f"All correct (< 5% loss diff): {results['summary']['all_correct']}")
    else:
        print("No successful runs!")
        results["summary"] = {
            "total_configs": total_configs,
            "successful": 0,
            "failed": total_configs,
            "error": "All runs failed"
        }

    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/Users/ec2-user/Desktop/OptimizeMLXTraining/benchmark_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\nBenchmark complete!")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Baseline vs CCE")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    # Clear any existing model cache to start fresh (except first model if resuming)
    print("Clearing model cache before starting...")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if SKIP_CONFIGS == 0:
        for model_repo, _ in MODELS:
            delete_model_cache(model_repo)
    else:
        print(f"  Resuming from config {SKIP_CONFIGS + 1}, keeping first model cache")

    run_benchmark(output_file=args.output)
