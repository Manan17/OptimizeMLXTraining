#!/usr/bin/env python3
"""
Benchmark: Quantized LoRA Training - Baseline CE vs CCE
========================================================

Tests 4-bit, 8-bit quantized models and non-quantized (fp32/bf16/fp16)
with LoRA fine-tuning using real Alpaca data.

Metrics: Correctness (loss match), Memory, Speed
"""

import os
import gc
import csv
import time
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx.utils import tree_map
from datasets import load_dataset

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = [
    # (model_id, display_name, quant_type)
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", "Llama-3.2-1B", "4bit"),
    ("mlx-community/Llama-3.2-1B-Instruct-8bit", "Llama-3.2-1B", "8bit"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama-3.2-3B", "4bit"),
    ("mlx-community/gemma-2-2b-it-4bit", "Gemma-2-2B", "4bit"),
    ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "Qwen2.5-1.5B", "4bit"),
]

# Non-quantized model for fp32/bf16/fp16 correctness tests
FP_TEST_MODEL = ("mlx-community/Llama-3.2-1B-Instruct-bf16", "Llama-3.2-1B", "fp")

BATCH_SIZES = [8, 16]
SEQ_LEN = 1024
WARMUP_STEPS = 5
MEASURE_STEPS = 20

LEARNING_RATE = 1e-5
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LAYERS = 8

SEED = 42
DATASET_NAME = "tatsu-lab/alpaca"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


def create_batches(dataset, tokenizer, batch_size, seq_len, num_batches):
    """Create deterministic batches from dataset."""
    mx.random.seed(SEED)
    indices = mx.random.permutation(len(dataset)).tolist()

    batches = []
    for b in range(num_batches):
        start_idx = b * batch_size
        batch_indices = indices[start_idx:start_idx + batch_size]

        if len(batch_indices) < batch_size:
            batch_indices += indices[:batch_size - len(batch_indices)]

        batch_tokens = []
        for idx in batch_indices:
            text = format_sample(dataset[idx])
            tokens = tokenizer.encode(text)

            if len(tokens) >= seq_len:
                tokens = tokens[:seq_len]
            else:
                pad_id = tokenizer.eos_token_id or 0
                tokens = tokens + [pad_id] * (seq_len - len(tokens))

            batch_tokens.append(tokens)

        batch = mx.array(batch_tokens, dtype=mx.int32)
        batches.append(batch)

    mx.eval(batches)
    return batches


def get_lm_head_weight(model):
    """Get LM head weight, handling quantized models."""
    embed = model.model.embed_tokens
    if hasattr(embed, 'scales'):
        return mx.dequantize(
            embed.weight, embed.scales,
            getattr(embed, 'biases', None),
            group_size=getattr(embed, 'group_size', 64),
            bits=getattr(embed, 'bits', 4)
        )
    return embed.weight


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


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def run_training(model, weight, batches, use_cce, warmup_steps, measure_steps):
    """Run training and return metrics."""

    # Fresh optimizer
    optimizer = optim.Adafactor(
        learning_rate=LEARNING_RATE,
        relative_step=False,
        scale_parameter=False,
    )

    # Loss functions
    def baseline_loss_fn(model, batch):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        loss = nn.losses.cross_entropy(logits, targets)
        return loss.astype(mx.float32).mean()

    def cce_loss_fn(model, batch):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        hidden = model.model(inputs)
        B, S, H = hidden.shape
        loss = mx.fast.cce_loss(
            hidden.reshape(B * S, H),
            weight,
            targets.reshape(B * S)
        )
        return loss.astype(mx.float32).mean()

    loss_fn = cce_loss_fn if use_cce else baseline_loss_fn
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup
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

    # Measure
    losses = []
    mx.synchronize()
    start_time = time.perf_counter()

    for step in range(measure_steps):
        batch = batches[batch_idx % len(batches)]
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        batch_idx += 1

    mx.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    peak_memory_gb = mx.get_peak_memory() / 1e9

    return {
        'time_ms': elapsed_ms,
        'memory_gb': peak_memory_gb,
        'final_loss': losses[-1],
        'losses': losses,
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def main():
    print("=" * 80)
    print("Quantized LoRA Training Benchmark: Baseline CE vs CCE")
    print(f"Config: {MEASURE_STEPS} steps, {WARMUP_STEPS} warmup, seq_len={SEQ_LEN}")
    print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, layers={LORA_LAYERS}")
    print(f"Optimizer: Adafactor, lr={LEARNING_RATE}")
    print("=" * 80)

    # Load dataset
    print("\nLoading Alpaca dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Dataset loaded: {len(dataset)} samples")

    # Results storage
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"/Users/ec2-user/Desktop/OptimizeMLXTraining/benchmark_results_{timestamp}.csv"

    # Test all models
    all_models = MODELS + [FP_TEST_MODEL]

    for model_id, model_name, quant_type in all_models:
        print(f"\n{'='*80}")
        print(f"Model: {model_name} ({quant_type})")
        print("=" * 80)

        # Load model
        print(f"Loading {model_id}...")
        try:
            model, tokenizer = load(model_id)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        # Apply LoRA
        linear_to_lora_layers(
            model,
            num_layers=LORA_LAYERS,
            config={'rank': LORA_RANK, 'alpha': LORA_ALPHA, 'dropout': 0.0, 'scale': LORA_ALPHA / LORA_RANK}
        )
        model.freeze()
        model.unfreeze(keys=['lora_a', 'lora_b'], strict=False)

        # Get weight
        weight = get_lm_head_weight(model)
        mx.eval(weight)
        print(f"  Weight dtype: {weight.dtype}, shape: {weight.shape}")

        # Save initial weights
        initial_weights = save_weights(model)

        # Determine dtypes to test
        if quant_type == "fp":
            dtypes_to_test = [
                ("fp32", mx.float32),
                ("bf16", mx.bfloat16),
                ("fp16", mx.float16),
            ]
        else:
            dtypes_to_test = [(quant_type, None)]  # Quantized models use their native dtype

        for dtype_name, dtype in dtypes_to_test:
            # Convert weight if needed
            if dtype is not None:
                test_weight = weight.astype(dtype)
                mx.eval(test_weight)
            else:
                test_weight = weight

            for batch_size in BATCH_SIZES:
                print(f"\n  [{dtype_name}] Batch size {batch_size}:")

                # Create batches
                num_batches = WARMUP_STEPS + MEASURE_STEPS + 10
                batches = create_batches(dataset, tokenizer, batch_size, SEQ_LEN, num_batches)

                # Run baseline
                restore_weights(model, initial_weights)
                print(f"    Running Baseline CE...")
                result_baseline = run_training(
                    model, test_weight, batches,
                    use_cce=False,
                    warmup_steps=WARMUP_STEPS,
                    measure_steps=MEASURE_STEPS
                )

                # Run CCE
                restore_weights(model, initial_weights)
                print(f"    Running CCE...")
                result_cce = run_training(
                    model, test_weight, batches,
                    use_cce=True,
                    warmup_steps=WARMUP_STEPS,
                    measure_steps=MEASURE_STEPS
                )

                # Calculate metrics
                speedup = result_baseline['time_ms'] / result_cce['time_ms']
                mem_saved = (result_baseline['memory_gb'] - result_cce['memory_gb']) / result_baseline['memory_gb'] * 100
                loss_diff = abs(result_baseline['final_loss'] - result_cce['final_loss'])
                loss_match = loss_diff < 0.1

                # Print results
                print(f"    Baseline: {result_baseline['time_ms']:7.0f}ms | {result_baseline['memory_gb']:.2f}GB | loss={result_baseline['final_loss']:.4f}")
                print(f"    CCE:      {result_cce['time_ms']:7.0f}ms | {result_cce['memory_gb']:.2f}GB | loss={result_cce['final_loss']:.4f}")
                print(f"    Speedup: {speedup:.2f}x | Mem saved: {mem_saved:.1f}% | Loss match: {'✓' if loss_match else '✗'}")

                # Store results
                results.append({
                    'model': model_name,
                    'quant': dtype_name,
                    'batch_size': batch_size,
                    'seq_len': SEQ_LEN,
                    'baseline_time_ms': result_baseline['time_ms'],
                    'baseline_memory_gb': result_baseline['memory_gb'],
                    'baseline_loss': result_baseline['final_loss'],
                    'cce_time_ms': result_cce['time_ms'],
                    'cce_memory_gb': result_cce['memory_gb'],
                    'cce_loss': result_cce['final_loss'],
                    'speedup': speedup,
                    'mem_saved_pct': mem_saved,
                    'loss_match': loss_match,
                })

                # Clear memory
                del batches
                gc.collect()

        # Cleanup model
        del model, tokenizer, weight, initial_weights
        gc.collect()

    # Save results to CSV
    print(f"\n{'='*80}")
    print(f"Saving results to {csv_file}")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Quant':<6} {'Batch':<5} {'Speedup':<8} {'Mem Saved':<10} {'Loss Match'}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<15} {r['quant']:<6} {r['batch_size']:<5} {r['speedup']:.2f}x    {r['mem_saved_pct']:>6.1f}%     {'✓' if r['loss_match'] else '✗'}")

    print(f"\nResults saved to: {csv_file}")
    print("Done!")


if __name__ == "__main__":
    main()
