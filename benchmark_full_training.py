#!/usr/bin/env python3
"""
Full CCE Training Benchmark with Real Alpaca Data
Tests: Correctness, Memory Savings, Time Improvement
Models: Quantized (4bit, 8bit) and Full Precision (fp32, bf16, fp16)
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.utils import build_schedule
import time
import gc
from pathlib import Path

# CCE is in mx.fast (mlx.core.fast)

def get_alpaca_batch(dataset, tokenizer, batch_size, seq_length, start_idx=0):
    """Get a batch from Alpaca dataset with proper tokenization."""
    batch_texts = []
    for i in range(batch_size):
        idx = (start_idx + i) % len(dataset)
        item = dataset[idx]
        # Format as instruction-following
        if "instruction" in item and "output" in item:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        elif "text" in item:
            text = item["text"]
        else:
            text = str(item)
        batch_texts.append(text)

    # Tokenize
    input_ids = []
    for text in batch_texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        # Pad or truncate to seq_length + 1 (for labels)
        if len(tokens) < seq_length + 1:
            tokens = tokens + [tokenizer.pad_token_id or 0] * (seq_length + 1 - len(tokens))
        else:
            tokens = tokens[:seq_length + 1]
        input_ids.append(tokens)

    input_ids = mx.array(input_ids)
    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    return inputs, labels

def baseline_cross_entropy(model, hidden, targets, ignore_index=-100, logit_softcap=0.0):
    """Baseline: Full materialization cross-entropy."""
    logits = hidden @ model.lm_head.weight.T

    # Apply softcap if needed (only for models without built-in softcap)
    if logit_softcap > 0.0:
        logits = logit_softcap * mx.tanh(logits / logit_softcap)

    # Compute cross-entropy
    N, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    # Create mask for ignore_index
    mask = (targets_flat != ignore_index).astype(mx.float32)

    # Compute log_softmax and gather
    log_probs = mx.log_softmax(logits_flat, axis=-1)

    # Clamp targets for gathering (ignore_index tokens will be masked out)
    safe_targets = mx.where(targets_flat == ignore_index, mx.zeros_like(targets_flat), targets_flat)

    # Gather target log probs
    target_log_probs = mx.take_along_axis(log_probs, safe_targets[:, None], axis=-1).squeeze(-1)

    # Apply mask and compute mean
    loss = -target_log_probs * mask
    loss = loss.sum() / (mask.sum() + 1e-9)

    return loss

def cce_cross_entropy(model, hidden, targets, ignore_index=-100, logit_softcap=0.0):
    """CCE: Memory-efficient cross-entropy."""
    loss = mx.fast.cce_loss(
        hidden,
        model.lm_head.weight,
        targets,
        ignore_index=ignore_index,
        logit_softcap=logit_softcap
    )
    return loss.mean()

def run_training_step(model, inputs, labels, loss_fn, optimizer, logit_softcap=0.0):
    """Run a single training step."""
    def forward_and_loss(model, inputs, labels):
        outputs = model(inputs)
        # Get hidden states (before lm_head for CCE, or full logits for baseline)
        if hasattr(outputs, 'hidden_states'):
            hidden = outputs.hidden_states
        else:
            # For most MLX models, we need to get hidden states before lm_head
            # Re-run model backbone only
            hidden = model.model(inputs)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        return loss_fn(model, hidden, labels, ignore_index=-100, logit_softcap=logit_softcap)

    loss, grads = nn.value_and_grad(model, forward_and_loss)(model, inputs, labels)
    optimizer.update(model, grads)
    return loss

def benchmark_model(model_path, model_name, quant_type, logit_softcap, batch_sizes, seq_length, dataset, num_steps=20):
    """Benchmark a single model configuration."""
    print(f"\n{'='*80}")
    print(f"Model: {model_name} ({quant_type})")
    print(f"Softcap: {logit_softcap}")
    print(f"{'='*80}")

    results = {}

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size}, Seq Length: {seq_length} ---")

        # Load model fresh for each test
        gc.collect()
        mx.metal.clear_cache()

        try:
            model, tokenizer = load(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue

        # Get a batch
        inputs, labels = get_alpaca_batch(dataset, tokenizer, batch_size, seq_length)

        # Ensure tokenizer has pad token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        # Check if model has built-in softcap (Gemma-2)
        has_builtin_softcap = hasattr(model, 'args') and hasattr(model.args, 'final_logit_softcapping')
        actual_softcap_for_baseline = 0.0 if has_builtin_softcap else logit_softcap

        # =========== CORRECTNESS TEST ===========
        print("\n[Correctness Test]")

        # Get hidden states
        hidden = model.model(inputs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        # Compute losses
        baseline_loss = baseline_cross_entropy(model, hidden, labels, logit_softcap=actual_softcap_for_baseline)
        cce_loss_val = cce_cross_entropy(model, hidden, labels, logit_softcap=logit_softcap)
        mx.eval(baseline_loss, cce_loss_val)

        baseline_loss_val = float(baseline_loss)
        cce_loss_float = float(cce_loss_val)
        loss_diff = abs(baseline_loss_val - cce_loss_float)

        print(f"  Baseline Loss: {baseline_loss_val:.6f}")
        print(f"  CCE Loss:      {cce_loss_float:.6f}")
        print(f"  Difference:    {loss_diff:.6f}")
        correctness_ok = loss_diff < 0.01
        print(f"  Match: {'✓ PASS' if correctness_ok else '✗ FAIL'}")

        # =========== BASELINE FULL TRAINING ===========
        print("\n[Baseline Training]")
        gc.collect()
        mx.metal.clear_cache()

        # Reload model
        model, tokenizer = load(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        optimizer = optim.Adam(learning_rate=1e-5)

        # Warmup
        for _ in range(2):
            inputs, labels = get_alpaca_batch(dataset, tokenizer, batch_size, seq_length, start_idx=0)
            hidden = model.model(inputs)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            loss = baseline_cross_entropy(model, hidden, labels, logit_softcap=actual_softcap_for_baseline)
            loss.backward()
            mx.eval(loss)

        gc.collect()
        mx.metal.clear_cache()
        mx.reset_peak_memory()

        # Timed run
        start_time = time.perf_counter()
        total_loss = 0.0

        for step in range(num_steps):
            inputs, labels = get_alpaca_batch(dataset, tokenizer, batch_size, seq_length, start_idx=step * batch_size)

            def baseline_loss_fn(model):
                hidden = model.model(inputs)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                return baseline_cross_entropy(model, hidden, labels, logit_softcap=actual_softcap_for_baseline)

            loss, grads = nn.value_and_grad(model, baseline_loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters())
            total_loss += float(loss)

        baseline_time = time.perf_counter() - start_time
        baseline_memory = mx.metal.get_peak_memory() / (1024**3)
        baseline_avg_loss = total_loss / num_steps

        print(f"  Time: {baseline_time:.2f}s ({baseline_time/num_steps*1000:.1f}ms/step)")
        print(f"  Peak Memory: {baseline_memory:.2f} GB")
        print(f"  Avg Loss: {baseline_avg_loss:.4f}")

        # =========== CCE FULL TRAINING ===========
        print("\n[CCE Training]")
        gc.collect()
        mx.metal.clear_cache()

        # Reload model
        model, tokenizer = load(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        optimizer = optim.Adam(learning_rate=1e-5)

        # Warmup
        for _ in range(2):
            inputs, labels = get_alpaca_batch(dataset, tokenizer, batch_size, seq_length, start_idx=0)
            hidden = model.model(inputs)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            loss = cce_cross_entropy(model, hidden, labels, logit_softcap=logit_softcap)
            loss.backward()
            mx.eval(loss)

        gc.collect()
        mx.metal.clear_cache()
        mx.reset_peak_memory()

        # Timed run
        start_time = time.perf_counter()
        total_loss = 0.0

        for step in range(num_steps):
            inputs, labels = get_alpaca_batch(dataset, tokenizer, batch_size, seq_length, start_idx=step * batch_size)

            def cce_loss_fn(model):
                hidden = model.model(inputs)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                return cce_cross_entropy(model, hidden, labels, logit_softcap=logit_softcap)

            loss, grads = nn.value_and_grad(model, cce_loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters())
            total_loss += float(loss)

        cce_time = time.perf_counter() - start_time
        cce_memory = mx.metal.get_peak_memory() / (1024**3)
        cce_avg_loss = total_loss / num_steps

        print(f"  Time: {cce_time:.2f}s ({cce_time/num_steps*1000:.1f}ms/step)")
        print(f"  Peak Memory: {cce_memory:.2f} GB")
        print(f"  Avg Loss: {cce_avg_loss:.4f}")

        # =========== SUMMARY ===========
        print(f"\n[Summary for batch_size={batch_size}]")
        memory_saved = baseline_memory - cce_memory
        memory_pct = (memory_saved / baseline_memory) * 100 if baseline_memory > 0 else 0
        time_saved = baseline_time - cce_time
        speedup = baseline_time / cce_time if cce_time > 0 else 0

        print(f"  Memory: {baseline_memory:.2f} GB -> {cce_memory:.2f} GB (saved {memory_saved:.2f} GB, {memory_pct:.1f}%)")
        print(f"  Time: {baseline_time:.2f}s -> {cce_time:.2f}s (speedup: {speedup:.2f}x)")
        print(f"  Correctness: {'✓ PASS' if correctness_ok else '✗ FAIL'} (diff={loss_diff:.6f})")

        results[batch_size] = {
            'correctness': correctness_ok,
            'loss_diff': loss_diff,
            'baseline_memory': baseline_memory,
            'cce_memory': cce_memory,
            'memory_saved_gb': memory_saved,
            'memory_saved_pct': memory_pct,
            'baseline_time': baseline_time,
            'cce_time': cce_time,
            'speedup': speedup,
        }

        # Clean up
        del model, tokenizer
        gc.collect()
        mx.metal.clear_cache()

    return results

def main():
    print("=" * 80)
    print("CCE Full Training Benchmark with Real Alpaca Data")
    print("=" * 80)

    # Load Alpaca dataset
    print("\nLoading Alpaca dataset...")
    try:
        from datasets import load_dataset as hf_load_dataset
        dataset = hf_load_dataset("tatsu-lab/alpaca", split="train")
        print(f"Loaded {len(dataset)} examples from Alpaca dataset")
    except Exception as e:
        print(f"Failed to load Alpaca: {e}")
        print("Falling back to synthetic data...")
        dataset = [{"instruction": f"Question {i}", "output": f"Answer {i} " * 100} for i in range(10000)]

    # Configuration
    batch_sizes = [8, 16]
    seq_length = 1024
    num_steps = 20

    # Models to test
    # Format: (model_path, display_name, quant_type, logit_softcap)
    models = [
        # Quantized models
        ("mlx-community/Llama-3.2-1B-Instruct-4bit", "Llama-3.2-1B", "4bit", 0.0),
        ("mlx-community/Llama-3.2-1B-Instruct-8bit", "Llama-3.2-1B", "8bit", 0.0),
        ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama-3.2-3B", "4bit", 0.0),
        ("mlx-community/gemma-2-2b-it-4bit", "Gemma-2-2B", "4bit", 30.0),
        ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "Qwen2.5-1.5B", "4bit", 0.0),
        # Full precision model (fp32/bf16/fp16)
        ("mlx-community/Llama-3.2-1B-Instruct", "Llama-3.2-1B", "fp", 0.0),
    ]

    all_results = {}

    for model_path, model_name, quant_type, logit_softcap in models:
        key = f"{model_name}_{quant_type}"
        try:
            results = benchmark_model(
                model_path, model_name, quant_type, logit_softcap,
                batch_sizes, seq_length, dataset, num_steps
            )
            all_results[key] = results
        except Exception as e:
            print(f"\nFailed to benchmark {model_name} ({quant_type}): {e}")
            import traceback
            traceback.print_exc()

    # Print final summary table
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"{'Model':<25} {'Batch':<6} {'Correct':<8} {'Baseline Mem':<12} {'CCE Mem':<10} {'Saved':<10} {'Speedup':<8}")
    print("-" * 100)

    for model_key, results in all_results.items():
        for batch_size, data in results.items():
            correct = "✓" if data['correctness'] else "✗"
            print(f"{model_key:<25} {batch_size:<6} {correct:<8} "
                  f"{data['baseline_memory']:.2f} GB{'':<4} {data['cce_memory']:.2f} GB{'':<2} "
                  f"{data['memory_saved_pct']:.1f}%{'':<4} {data['speedup']:.2f}x")

    print("=" * 100)

if __name__ == "__main__":
    main()
