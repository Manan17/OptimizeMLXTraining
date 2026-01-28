#!/usr/bin/env python3
"""
Training comparison: Baseline Cross-Entropy vs CCE (Cut Cross-Entropy)

This script compares memory and speed between:
1. Baseline: Standard nn.losses.cross_entropy (computes full N×V logits)
2. CCE: mx.fast.cce_loss (chunked computation, memory efficient)
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
class TrainingConfig:
    batch_size: int = 4
    max_seq_length: int = 128
    learning_rate: float = 1e-4
    num_steps: int = 20
    warmup_steps: int = 2
    use_cce: bool = False


def get_lm_head_weight(model):
    """Get the language model head weight matrix."""
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # Tied embeddings case
        return model.model.embed_tokens.weight
    else:
        raise ValueError("Cannot find lm_head weight in model")


def get_hidden_states(model, inputs):
    """Get hidden states before lm_head projection."""
    # model.model is the transformer (LlamaModel), model.lm_head is the projection
    if hasattr(model, 'model'):
        return model.model(inputs)
    else:
        raise ValueError("Cannot find transformer model")


def baseline_loss_fn(model, batch, lengths):
    """Standard cross-entropy loss (baseline)."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    # Forward pass - computes full logits (N × V)
    logits = model(inputs)

    # Create mask for valid positions
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    # Standard cross-entropy
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / ntoks

    return ce, ntoks


def cce_loss_fn(model, batch, lengths):
    """CCE (Cut Cross-Entropy) loss - memory efficient."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    # Get hidden states BEFORE lm_head (avoids materializing full N×V logits)
    hidden = get_hidden_states(model, inputs)
    weight = get_lm_head_weight(model)

    # CCE backward requires FP32 - convert if needed (BF16 models)
    if hidden.dtype == mx.bfloat16:
        hidden = hidden.astype(mx.float32)
        weight = weight.astype(mx.float32)

    # Flatten for CCE: [batch, seq, hidden] -> [batch*seq, hidden]
    B, S, H = hidden.shape
    hidden_flat = hidden.reshape(B * S, H)
    targets_flat = targets.reshape(B * S)

    # Create mask for valid positions
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    mask_flat = mask.reshape(B * S)

    # CCE loss - processes in chunks, never materializes full N×V
    ce = mx.fast.cce_loss(hidden_flat, weight, targets_flat)

    # Apply mask and reduce
    ce = (ce * mask_flat).sum() / mask_flat.sum()

    return ce, mx.sum(mask_flat)


def create_batch_iterator(dataset, tokenizer, batch_size, max_seq_length):
    """Create batches from dataset."""
    all_tokens = []
    for item in dataset:
        tokens = tokenizer.encode(item["text"])
        if len(tokens) > 2:  # Skip very short sequences
            all_tokens.append(tokens[:max_seq_length])

    # Sort by length for efficient batching
    all_tokens.sort(key=len)

    while True:
        # Shuffle
        indices = np.random.permutation(len(all_tokens))

        for i in range(0, len(indices) - batch_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_tokens = [all_tokens[j] for j in batch_indices]

            # Pad to fixed length to avoid recompilation
            padded = []
            lengths = []

            for tokens in batch_tokens:
                pad_len = max_seq_length - len(tokens)
                padded.append(tokens + [tokenizer.pad_token_id or 0] * pad_len)
                lengths.append([0, len(tokens) - 1])  # [start, end] for mask

            yield mx.array(padded), mx.array(lengths)


def train_step(model, optimizer, batch, lengths, loss_fn, loss_value_and_grad):
    """Single training step."""
    (loss, ntoks), grads = loss_value_and_grad(model, batch, lengths)
    optimizer.update(model, grads)
    return loss


def run_training(config: TrainingConfig, model_path: str, dataset_name: str, use_float32: bool = False):
    """Run training with specified configuration."""
    print(f"\n{'='*60}")
    print(f"Training with {'CCE' if config.use_cce else 'Baseline'} loss")
    print(f"{'='*60}")

    # Load model
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    if use_float32:
        print("Converting model to float32")
        model.set_dtype(mx.float32)

    # Ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get vocab size for info
    weight = get_lm_head_weight(model)
    vocab_size = weight.shape[0]
    hidden_size = weight.shape[1]
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}")

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    raw_train = hf_load_dataset(dataset_name, split="train[:500]")
    train_data = [{"text": ex["text"]} for ex in raw_train]

    # Create optimizer
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Select loss function
    loss_fn = cce_loss_fn if config.use_cce else baseline_loss_fn

    # Create batch iterator
    batch_iter = create_batch_iterator(
        train_data, tokenizer, config.batch_size, config.max_seq_length
    )

    # Create value_and_grad function (like mlx_lm trainer does)
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup
    print(f"Warming up ({config.warmup_steps} steps)...")
    for _ in range(config.warmup_steps):
        batch, lengths = next(batch_iter)
        loss = train_step(model, optimizer, batch, lengths, loss_fn, loss_value_and_grad)
        mx.eval(loss)

    # Reset memory tracking
    mx.clear_cache()
    mx.synchronize()
    mx.reset_peak_memory()

    # Training loop
    print(f"Training ({config.num_steps} steps)...")
    losses = []
    start_time = time.perf_counter()

    for step in range(config.num_steps):
        batch, lengths = next(batch_iter)
        loss = train_step(model, optimizer, batch, lengths, loss_fn, loss_value_and_grad)
        mx.eval(loss)
        losses.append(float(loss))

        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: loss = {losses[-1]:.4f}")

    mx.synchronize()
    total_time = time.perf_counter() - start_time
    peak_memory = mx.get_peak_memory() / (1024**3)

    # Results
    avg_loss = np.mean(losses)
    avg_time_per_step = total_time / config.num_steps * 1000

    print(f"\nResults:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Peak memory: {peak_memory:.2f} GB")
    print(f"  Avg time/step: {avg_time_per_step:.0f} ms")
    print(f"  Total time: {total_time:.1f} s")

    return {
        "avg_loss": avg_loss,
        "peak_memory": peak_memory,
        "avg_time_per_step": avg_time_per_step,
        "total_time": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs CCE training")
    parser.add_argument("--model", type=str, default="mlx-community/SmolLM-135M-Instruct",
                        help="Model to use (default: SmolLM-135M)")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Dataset to use")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128,
                        help="Max sequence length")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of training steps")
    parser.add_argument("--mode", type=str, choices=["baseline", "cce", "both"], default="both",
                        help="Which mode to run")
    parser.add_argument("--float32", action="store_true",
                        help="Load model in float32 (avoids conversion overhead)")
    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        max_seq_length=args.seq_length,
        num_steps=args.steps,
    )

    results = {}

    if args.mode in ["baseline", "both"]:
        np.random.seed(42)  # Same seed for fair comparison
        config.use_cce = False
        results["baseline"] = run_training(config, args.model, args.dataset, use_float32=args.float32)

    if args.mode in ["cce", "both"]:
        # Clear caches between runs to avoid state accumulation
        import gc
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        np.random.seed(42)  # Same seed for fair comparison
        config.use_cce = True
        results["cce"] = run_training(config, args.model, args.dataset, use_float32=args.float32)

    if args.mode == "both":
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Baseline':>15} {'CCE':>15} {'Improvement':>15}")
        print("-" * 65)

        baseline = results["baseline"]
        cce = results["cce"]

        mem_improvement = baseline["peak_memory"] / cce["peak_memory"]
        speed_improvement = baseline["avg_time_per_step"] / cce["avg_time_per_step"]

        print(f"{'Peak Memory (GB)':<20} {baseline['peak_memory']:>15.2f} {cce['peak_memory']:>15.2f} {mem_improvement:>14.2f}x")
        print(f"{'Avg Time/Step (ms)':<20} {baseline['avg_time_per_step']:>15.0f} {cce['avg_time_per_step']:>15.0f} {speed_improvement:>14.2f}x")
        print(f"{'Final Loss':<20} {baseline['avg_loss']:>15.4f} {cce['avg_loss']:>15.4f} {'(should match)':>15}")


if __name__ == "__main__":
    main()
