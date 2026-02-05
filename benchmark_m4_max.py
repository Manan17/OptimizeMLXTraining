#!/usr/bin/env python3
"""
Benchmark: Baseline Cross-Entropy vs CCE for M4 Max 36GB
=========================================================

Models:
- Full Training (2): Llama-3.2-1B, Gemma-3-1B
- Quantized 4-bit (3): Llama-3.2-3B, Qwen2.5-3B, Gemma-2-2B
- BF16 + LoRA (2): Llama-3.2-3B, Qwen2.5-3B

Configs: batch [4, 8] x seq [512, 1024]
Dataset: tatsu-lab/alpaca
Optimizer: Adafactor (lr=1e-5)

FAIRNESS PROTOCOL:
1. Same pre-tokenized batches (fixed seed) for baseline and CCE
2. Model weights saved and restored before each run
3. Fresh optimizer with identical hyperparams
4. Warmup steps for JIT/kernel selection
5. mx.synchronize() barriers for accurate GPU timing
6. Peak memory reset after warmup
"""

import os
import sys
import gc
import json
import time
import shutil
import subprocess
import tempfile
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Full Training Models (bf16, all params trained)
FULL_TRAINING_MODELS = [
    ("mlx-community/Llama-3.2-1B-Instruct-bf16", "Llama-1B-full", None),
    ("mlx-community/gemma-3-1b-it-bf16", "Gemma3-1B-full", None),
]

# Quantized Models (4-bit)
QUANTIZED_MODELS = [
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama-3B-4bit", None),
    ("mlx-community/Qwen2.5-3B-Instruct-4bit", "Qwen2.5-3B-4bit", None),
    ("mlx-community/gemma-2-2b-it-4bit", "Gemma2-2B-4bit", 30.0),  # softcap=30
]

# BF16 + LoRA Models
LORA_MODELS = [
    ("mlx-community/Llama-3.2-3B-Instruct-bf16", "Llama-3B-LoRA", None),
    ("mlx-community/Qwen2.5-3B-Instruct-bf16", "Qwen2.5-3B-LoRA", None),
]

CONFIGS = [
    (4, 512),
    (4, 1024),
    (8, 512),
    (8, 1024),
]

WARMUP_STEPS = 5
MEASURE_STEPS = 10
COOLDOWN_SECONDS = 10

DATASET_NAME = "tatsu-lab/alpaca"
SEED = 42
LEARNING_RATE = 1e-5

# LoRA config
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LAYERS = 8  # Number of layers to apply LoRA

# =============================================================================
# WORKER CODE (runs in subprocess)
# =============================================================================

WORKER_CODE = '''
import os
import sys
import gc
import json
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx.utils import tree_map, tree_flatten
from datasets import load_dataset

def format_sample(sample):
    if sample['input']:
        return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
    else:
        return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

def create_batches(dataset, tokenizer, seed, batch_size, seq_len, num_batches):
    mx.random.seed(seed)
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
                pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                tokens = tokens + [pad_id] * (seq_len - len(tokens))
            batch_tokens.append(tokens)
        batch = mx.array(batch_tokens, dtype=mx.int32)
        batches.append(batch)
    return batches

def get_lm_weight(model, is_quantized=False):
    # Check for untied lm_head FIRST (Qwen, some other models have this)
    if hasattr(model, 'lm_head') and model.lm_head is not None and hasattr(model.lm_head, 'weight'):
        lm_head = model.lm_head
        if hasattr(lm_head, 'scales'):
            return mx.dequantize(
                lm_head.weight, lm_head.scales,
                getattr(lm_head, 'biases', None),
                group_size=getattr(lm_head, 'group_size', 64),
                bits=getattr(lm_head, 'bits', 4)
            )
        return lm_head.weight
    # Fall back to tied embeddings
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
    weights = tree_map(lambda x: mx.array(x) if isinstance(x, mx.array) else x, model.parameters())
    mx.eval(weights)
    return weights

def restore_weights(model, weights):
    model.update(weights)
    mx.eval(model.parameters())

def apply_lora(model, rank, alpha, num_layers):
    """Apply LoRA using mlx_lm's built-in function."""
    from mlx_lm.tuner.utils import linear_to_lora_layers
    linear_to_lora_layers(
        model,
        num_layers=num_layers,
        config={'rank': rank, 'alpha': alpha, 'dropout': 0.0, 'scale': alpha / rank}
    )
    model.freeze()
    model.unfreeze(keys=['lora_a', 'lora_b'], strict=False)

def baseline_loss_fn(model, batch):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    loss = nn.losses.cross_entropy(logits, targets)
    return loss.astype(mx.float32).mean()

def cce_loss_fn(model, batch, softcap, is_quantized):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    # model.model() already applies final norm internally for most models (Llama, Gemma, etc.)
    hidden = model.model(inputs)
    # Get weight INSIDE the loss function for proper gradient computation
    weight = get_lm_weight(model, is_quantized)
    B, S, H = hidden.shape
    loss = mx.fast.cce_loss(
        hidden.reshape(B * S, H),
        weight,
        targets.reshape(B * S),
        logit_softcap=softcap if softcap else 0.0
    )
    return loss.astype(mx.float32).mean()

def run_method(model, batches, initial_weights, warmup_steps, measure_steps, method, lr, softcap, is_quantized):
    restore_weights(model, initial_weights)
    optimizer = optim.Adafactor(learning_rate=lr, relative_step=False, scale_parameter=False)

    if method == "baseline":
        def loss_fn(model, batch):
            return baseline_loss_fn(model, batch)
    else:
        def loss_fn(model, batch):
            return cce_loss_fn(model, batch, softcap, is_quantized)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup
    batch_idx = 0
    for _ in range(warmup_steps):
        batch = batches[batch_idx % len(batches)]
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        batch_idx += 1

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
    end_time = time.perf_counter()

    return {
        "losses": losses,
        "time_ms": (end_time - start_time) * 1000,
        "memory_gb": mx.get_peak_memory() / 1e9
    }

if __name__ == "__main__":
    args = json.loads(sys.argv[1])

    model_repo = args["model_repo"]
    batch_size = args["batch_size"]
    seq_len = args["seq_len"]
    method = args["method"]
    warmup_steps = args["warmup_steps"]
    measure_steps = args["measure_steps"]
    seed = args["seed"]
    lr = args["learning_rate"]
    softcap = args.get("softcap")
    use_lora = args.get("use_lora", False)
    is_quantized = args.get("is_quantized", False)
    lora_rank = args.get("lora_rank", 8)
    lora_alpha = args.get("lora_alpha", 16)
    lora_layers = args.get("lora_layers", 8)

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    model, tokenizer = load(model_repo)
    mx.eval(model.parameters())

    if use_lora:
        # Set seed BEFORE applying LoRA to ensure same random initialization
        # in both CCE and baseline subprocesses
        mx.random.seed(seed)
        apply_lora(model, lora_rank, lora_alpha, lora_layers)
        mx.eval(model.parameters())

    initial_weights = save_weights(model)

    num_batches = warmup_steps + measure_steps + 5
    batches = create_batches(dataset, tokenizer, seed, batch_size, seq_len, num_batches)

    result = run_method(model, batches, initial_weights, warmup_steps, measure_steps, method, lr, softcap, is_quantized)
    print("RESULT_JSON:" + json.dumps(result))
'''

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def delete_model_cache(model_repo):
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = "models--" + model_repo.replace("/", "--")
    model_path = cache_dir / model_dir_name
    if model_path.exists():
        logger.info(f"  Deleting cache: {model_path}")
        shutil.rmtree(model_path)

def run_subprocess(model_repo, batch_size, seq_len, method, warmup_steps, measure_steps,
                   seed, lr, softcap, use_lora, is_quantized, lora_rank, lora_alpha, lora_layers, timeout=None):
    args = {
        "model_repo": model_repo,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "method": method,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "seed": seed,
        "learning_rate": lr,
        "softcap": softcap,
        "use_lora": use_lora,
        "is_quantized": is_quantized,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_layers": lora_layers,
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(WORKER_CODE)
        worker_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, worker_path, json.dumps(args)],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        for line in result.stdout.split('\n'):
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[12:])

        if result.returncode != 0:
            error_msg = result.stderr[-1000:] if result.stderr else result.stdout[-1000:]
            return {"error": f"Process failed: {error_msg}", "crashed": True}

        return {"error": "No result found in output", "crashed": True}

    except subprocess.TimeoutExpired:
        return {"error": "timeout", "crashed": True}
    except Exception as e:
        return {"error": str(e), "crashed": True}
    finally:
        os.unlink(worker_path)

def compare_results(baseline, cce):
    baseline_losses = baseline["losses"]
    cce_losses = cce["losses"]

    loss_diffs = [abs(b - c) for b, c in zip(baseline_losses, cce_losses)]
    loss_diffs_pct = [abs(b - c) / (abs(b) + 1e-10) * 100 for b, c in zip(baseline_losses, cce_losses)]

    speedup = baseline["time_ms"] / cce["time_ms"] if cce["time_ms"] > 0 else 0
    memory_savings_pct = ((baseline["memory_gb"] - cce["memory_gb"]) / baseline["memory_gb"] * 100) if baseline["memory_gb"] > 0 else 0

    return {
        "speedup": speedup,
        "memory_savings_pct": memory_savings_pct,
        "loss_max_diff": max(loss_diffs),
        "loss_mean_diff": sum(loss_diffs) / len(loss_diffs),
        "loss_max_diff_pct": max(loss_diffs_pct),
        "loss_mean_diff_pct": sum(loss_diffs_pct) / len(loss_diffs_pct),
        "correct": max(loss_diffs) < 0.05  # <0.05 absolute difference considered correct
    }

def print_result_summary(model_name, batch_size, seq_len, baseline, cce, comparison):
    print("\n" + "=" * 70)
    print(f"RESULT: {model_name} | batch={batch_size}, seq={seq_len}")
    print("=" * 70)
    print(f"  Baseline: {baseline['time_ms']:.1f}ms | {baseline['memory_gb']:.2f}GB | final_loss={baseline['losses'][-1]:.4f}")
    print(f"  CCE:      {cce['time_ms']:.1f}ms | {cce['memory_gb']:.2f}GB | final_loss={cce['losses'][-1]:.4f}")
    print("-" * 70)
    print(f"  Speedup:        {comparison['speedup']:.2f}x")
    print(f"  Memory Saved:   {comparison['memory_savings_pct']:.1f}%")
    print(f"  Max Loss Diff:  {comparison['loss_max_diff']:.4f} (threshold: 0.05)")
    print(f"  Correct:        {'YES' if comparison['correct'] else 'NO'}")
    print("=" * 70 + "\n")

# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark():
    output_dir = "/Users/ec2-user/Desktop/OptimizeMLXTraining"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"{output_dir}/benchmark_m4max_{timestamp}.csv"
    json_file = f"{output_dir}/benchmark_m4max_{timestamp}.json"

    logger.info("=" * 70)
    logger.info("Benchmark: Baseline CE vs CCE for M4 Max 36GB")
    logger.info("=" * 70)
    logger.info(f"Full Training: {[m[1] for m in FULL_TRAINING_MODELS]}")
    logger.info(f"Quantized: {[m[1] for m in QUANTIZED_MODELS]}")
    logger.info(f"LoRA: {[m[1] for m in LORA_MODELS]}")
    logger.info(f"Configs: {CONFIGS}")
    logger.info(f"Warmup: {WARMUP_STEPS}, Measure: {MEASURE_STEPS}")
    logger.info(f"Dataset: {DATASET_NAME}, Seed: {SEED}")
    logger.info(f"Optimizer: Adafactor (lr={LEARNING_RATE})")
    logger.info("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "learning_rate": LEARNING_RATE,
            "dataset": DATASET_NAME,
            "seed": SEED,
            "configs": CONFIGS,
        },
        "results": []
    }

    with open(csv_file, 'w') as f:
        f.write("model,type,batch,seq,baseline_time_ms,baseline_mem_gb,baseline_loss,"
                "cce_time_ms,cce_mem_gb,cce_loss,speedup,mem_save_pct,correct\n")

    all_models = [
        (repo, name, softcap, "full", False) for repo, name, softcap in FULL_TRAINING_MODELS
    ] + [
        (repo, name, softcap, "quantized", True) for repo, name, softcap in QUANTIZED_MODELS  # Quantized needs LoRA
    ] + [
        (repo, name, softcap, "lora", True) for repo, name, softcap in LORA_MODELS
    ]

    total_runs = len(all_models) * len(CONFIGS)
    current_run = 0

    for model_repo, model_name, softcap, model_type, use_lora in all_models:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"MODEL: {model_name} ({model_type})")
        logger.info(f"Repo: {model_repo}")
        if softcap:
            logger.info(f"Softcap: {softcap}")
        if use_lora:
            logger.info(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, layers={LORA_LAYERS}")
        logger.info("=" * 70)

        for batch_size, seq_len in CONFIGS:
            current_run += 1
            logger.info(f"\n[{current_run}/{total_runs}] batch={batch_size}, seq={seq_len}")

            is_quantized = model_type == "quantized"

            result = {
                "model": model_name,
                "model_repo": model_repo,
                "model_type": model_type,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "softcap": softcap,
                "use_lora": use_lora,
                "is_quantized": is_quantized,
            }

            # Run CCE FIRST (to avoid any thermal/cache advantage)
            logger.info("  Running CCE...")
            cce = run_subprocess(
                model_repo, batch_size, seq_len, "cce",
                WARMUP_STEPS, MEASURE_STEPS, SEED, LEARNING_RATE,
                softcap, use_lora, is_quantized, LORA_RANK, LORA_ALPHA, LORA_LAYERS
            )

            if "error" in cce:
                logger.error(f"  CCE CRASHED: {cce['error'][:200]}")
                result["cce"] = {"status": "crashed", "error": cce["error"]}
            else:
                logger.info(f"  CCE: {cce['time_ms']:.1f}ms, {cce['memory_gb']:.2f}GB, loss={cce['losses'][-1]:.4f}")
                result["cce"] = cce

            # Cooldown
            logger.info(f"  Cooldown {COOLDOWN_SECONDS}s...")
            gc.collect()
            time.sleep(COOLDOWN_SECONDS)

            # Run BASELINE SECOND
            logger.info("  Running BASELINE...")
            baseline = run_subprocess(
                model_repo, batch_size, seq_len, "baseline",
                WARMUP_STEPS, MEASURE_STEPS, SEED, LEARNING_RATE,
                softcap, use_lora, is_quantized, LORA_RANK, LORA_ALPHA, LORA_LAYERS
            )

            if "error" in baseline:
                logger.error(f"  BASELINE CRASHED: {baseline['error'][:200]}")
                result["baseline"] = {"status": "crashed", "error": baseline["error"]}
            else:
                logger.info(f"  BASELINE: {baseline['time_ms']:.1f}ms, {baseline['memory_gb']:.2f}GB, loss={baseline['losses'][-1]:.4f}")
                result["baseline"] = baseline

            # Compare results
            baseline_ok = "losses" in result.get("baseline", {})
            cce_ok = "losses" in result.get("cce", {})

            if baseline_ok and cce_ok:
                comparison = compare_results(baseline, cce)
                result["comparison"] = comparison
                print_result_summary(model_name, batch_size, seq_len, baseline, cce, comparison)

                with open(csv_file, 'a') as f:
                    f.write(f"{model_name},{model_type},{batch_size},{seq_len},"
                            f"{baseline['time_ms']:.2f},{baseline['memory_gb']:.2f},{baseline['losses'][-1]:.6f},"
                            f"{cce['time_ms']:.2f},{cce['memory_gb']:.2f},{cce['losses'][-1]:.6f},"
                            f"{comparison['speedup']:.2f},{comparison['memory_savings_pct']:.1f},"
                            f"{'YES' if comparison['correct'] else 'NO'}\n")
            elif cce_ok and not baseline_ok:
                print("\n" + "!" * 70)
                print(f"BASELINE CRASHED - CCE WORKED: {model_name} | batch={batch_size}, seq={seq_len}")
                print(f"  CCE: {cce['time_ms']:.1f}ms | {cce['memory_gb']:.2f}GB | loss={cce['losses'][-1]:.4f}")
                print("!" * 70 + "\n")
                with open(csv_file, 'a') as f:
                    f.write(f"{model_name},{model_type},{batch_size},{seq_len},"
                            f"CRASHED,CRASHED,CRASHED,"
                            f"{cce['time_ms']:.2f},{cce['memory_gb']:.2f},{cce['losses'][-1]:.6f},"
                            f"inf,inf,CCE_ONLY\n")
            elif baseline_ok and not cce_ok:
                print("\n" + "!" * 70)
                print(f"CCE CRASHED - BASELINE WORKED: {model_name} | batch={batch_size}, seq={seq_len}")
                print(f"  Baseline: {baseline['time_ms']:.1f}ms | {baseline['memory_gb']:.2f}GB | loss={baseline['losses'][-1]:.4f}")
                print("!" * 70 + "\n")
                with open(csv_file, 'a') as f:
                    f.write(f"{model_name},{model_type},{batch_size},{seq_len},"
                            f"{baseline['time_ms']:.2f},{baseline['memory_gb']:.2f},{baseline['losses'][-1]:.6f},"
                            f"CRASHED,CRASHED,CRASHED,"
                            f"N/A,N/A,BASELINE_ONLY\n")
            else:
                print("\n" + "X" * 70)
                print(f"BOTH CRASHED: {model_name} | batch={batch_size}, seq={seq_len}")
                print(f"  CCE error: {result.get('cce', {}).get('error', 'unknown')[:100]}")
                print(f"  Baseline error: {result.get('baseline', {}).get('error', 'unknown')[:100]}")
                print("X" * 70 + "\n")
                with open(csv_file, 'a') as f:
                    f.write(f"{model_name},{model_type},{batch_size},{seq_len},"
                            f"CRASHED,CRASHED,CRASHED,"
                            f"CRASHED,CRASHED,CRASHED,"
                            f"N/A,N/A,BOTH_CRASHED\n")

            results["results"].append(result)

            # Cooldown before next
            if current_run < total_runs:
                gc.collect()
                time.sleep(COOLDOWN_SECONDS)

        # Delete model cache after all configs for this model
        delete_model_cache(model_repo)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    successful = [r for r in results["results"] if "comparison" in r]
    if successful:
        by_type = {}
        for r in successful:
            t = r["model_type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(r)

        for model_type, runs in by_type.items():
            avg_speedup = sum(r["comparison"]["speedup"] for r in runs) / len(runs)
            avg_mem_save = sum(r["comparison"]["memory_savings_pct"] for r in runs) / len(runs)
            all_correct = all(r["comparison"]["correct"] for r in runs)
            logger.info(f"{model_type.upper():12} | Speedup: {avg_speedup:.2f}x | MemSave: {avg_mem_save:.1f}% | Correct: {'ALL' if all_correct else 'SOME FAILED'}")

    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {csv_file}")
    logger.info(f"Full results: {json_file}")
    logger.info("Benchmark complete!")

    return results

if __name__ == "__main__":
    run_benchmark()
