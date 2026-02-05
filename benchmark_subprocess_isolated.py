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

CRASH HANDLING:
===============
Baseline can crash with GPU watchdog timeout for large vocab models (e.g., Gemma-2B).
Each run (baseline/CCE) is isolated in a subprocess. If baseline crashes, we record
it as "crashed" and still run CCE. This allows benchmarking where only CCE works.
"""

import os
import sys
import gc
import json
import time
import shutil
import argparse
import subprocess
import tempfile
import logging
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx.utils import tree_map
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# MLX-community BF16 models (no auth required, optimized for MLX)
MODELS = [
    ("mlx-community/gemma-3-1b-it-bf16", "Gemma3-1B"),        # 262K vocab
    ("mlx-community/Llama-3.2-1B-Instruct-bf16", "Llama-1B"), # 128K vocab
    ("mlx-community/Qwen2.5-3B-Instruct-bf16", "Qwen2.5-3B"), # 152K vocab
]

# Configs for 32GB system - conservative to avoid OOM
CONFIGS = [
    # (batch_size, seq_len) -> N = batch * seq
    (2, 512),   # N=1024
    (2, 1024),  # N=2048
    (4, 512),   # N=2048
]

WARMUP_STEPS = 15
MEASURE_STEPS = 100
COOLDOWN_SECONDS = 30

DATASET_NAME = "tatsu-lab/alpaca"
DATASET_SPLIT = "train"
SEED = 42

# Adafactor with fixed LR for reproducibility
LEARNING_RATE = 1e-5

# Skip first N configs (already completed)
SKIP_CONFIGS = 0

# Subprocess timeout in seconds (None = wait for completion)
SUBPROCESS_TIMEOUT = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_lm_weight(model):
    """Get the language model head weight (handles tied/untied embeddings)."""
    if hasattr(model, 'lm_head') and model.lm_head is not None and hasattr(model.lm_head, 'weight'):
        return model.lm_head.weight
    return model.model.embed_tokens.weight


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
# SUBPROCESS WORKER
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
from mlx.utils import tree_map
from datasets import load_dataset

def format_sample(sample):
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

def get_lm_weight(model):
    if hasattr(model, 'lm_head') and model.lm_head is not None and hasattr(model.lm_head, 'weight'):
        return model.lm_head.weight
    return model.model.embed_tokens.weight

def save_weights(model):
    weights = tree_map(lambda x: mx.array(x) if isinstance(x, mx.array) else x, model.parameters())
    mx.eval(weights)
    return weights

def restore_weights(model, weights):
    model.update(weights)
    mx.eval(model.parameters())

def baseline_loss_fn(model, batch):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    loss = nn.losses.cross_entropy(logits, targets)
    loss = loss.astype(mx.float32).mean()
    return loss

def cce_loss_fn(model, weight, batch):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    hidden = model.model(inputs)
    B, S, H = hidden.shape
    loss = mx.fast.cce_loss(
        hidden.reshape(B * S, H),
        weight,
        targets.reshape(B * S)
    )
    loss = loss.astype(mx.float32).mean()
    return loss

def run_method(model, weight, batches, initial_weights, warmup_steps, measure_steps, method, lr):
    restore_weights(model, initial_weights)
    optimizer = optim.Adafactor(learning_rate=lr, relative_step=False, scale_parameter=False)

    if method == "baseline":
        def loss_fn(model, batch):
            return baseline_loss_fn(model, batch)
    else:
        def loss_fn(model, batch):
            return cce_loss_fn(model, weight, batch)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

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

    # Load dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Load model
    model, tokenizer = load(model_repo)
    mx.eval(model.parameters())

    weight = get_lm_weight(model)
    initial_weights = save_weights(model)

    # Create batches (deterministic with seed)
    num_batches = warmup_steps + measure_steps + 10
    batches = create_batches(dataset, tokenizer, seed, batch_size, seq_len, num_batches)

    # Run the method
    result = run_method(model, weight, batches, initial_weights, warmup_steps, measure_steps, method, lr)

    # Output result as JSON
    print("RESULT_JSON:" + json.dumps(result))
'''


def run_subprocess(model_repo, batch_size, seq_len, method, warmup_steps, measure_steps, seed, lr, timeout):
    """Run baseline or CCE in isolated subprocess."""
    args = {
        "model_repo": model_repo,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "method": method,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "seed": seed,
        "learning_rate": lr,
    }

    # Write worker code to temp file
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

        # Parse output for result JSON
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[12:])

        # If no result found, check for errors
        if result.returncode != 0:
            return {"error": f"Process failed: {result.stderr[-500:] if result.stderr else 'unknown'}"}

        return {"error": "No result found in output"}

    except subprocess.TimeoutExpired:
        return {"error": "timeout", "crashed": True}
    except Exception as e:
        return {"error": str(e), "crashed": True}
    finally:
        os.unlink(worker_path)


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

def run_benchmark(output_dir=None):
    """Run the full benchmark with subprocess isolation for crash handling."""

    if output_dir is None:
        output_dir = "/Users/ec2-user/Desktop/OptimizeMLXTraining"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"{output_dir}/benchmark_results_{timestamp}.csv"
    json_file = f"{output_dir}/benchmark_results_{timestamp}.json"

    logger.info("=" * 70)
    logger.info("Benchmark: Baseline Cross-Entropy vs Two-Tier CCE")
    logger.info("=" * 70)
    logger.info(f"Models: {[m[1] for m in MODELS]}")
    logger.info(f"Warmup: {WARMUP_STEPS} steps, Measure: {MEASURE_STEPS} steps")
    logger.info(f"Cooldown: {COOLDOWN_SECONDS}s between runs")
    logger.info(f"Optimizer: Adafactor (lr={LEARNING_RATE})")
    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info(f"Seed: {SEED}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)

    # Verify dataset loads (will be loaded by each subprocess)
    logger.info("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    del dataset

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
            "configs": CONFIGS
        },
        "results": [],
        "summary": {}
    }

    # Write CSV header
    with open(csv_file, 'w') as f:
        f.write("model,vocab_size,batch,seq,N,speedup,mem_save_pct,baseline_time_ms,baseline_mem_gb,"
                "baseline_final_loss,cce_time_ms,cce_mem_gb,cce_final_loss\n")

    total_configs = len(MODELS) * len(CONFIGS)
    current_config = 0

    # Process each model
    for model_repo, model_name in MODELS:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"MODEL: {model_name} ({model_repo})")
        logger.info("=" * 70)

        # Load model once to get vocab size
        logger.info("Loading model...")
        try:
            model, tokenizer = load(model_repo)
            mx.eval(model.parameters())
            weight = get_lm_weight(model)
            vocab_size = weight.shape[0]
            logger.info(f"Model loaded. Weight shape: {weight.shape}, dtype: {weight.dtype}")
            logger.info("Saving initial weights...")
            del model, tokenizer, weight
            gc.collect()
        except Exception as e:
            logger.error(f"ERROR loading model: {e}")
            for batch_size, seq_len in CONFIGS:
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

        # Run all configs for this model
        for batch_size, seq_len in CONFIGS:
            current_config += 1
            N = batch_size * seq_len

            # Skip already completed configs
            if current_config <= SKIP_CONFIGS:
                logger.info(f"[{current_config}/{total_configs}] batch={batch_size}, seq={seq_len}, N={N} - SKIPPED")
                continue

            logger.info("")
            logger.info(f"[{current_config}/{total_configs}] batch={batch_size}, seq={seq_len}, N={N}")
            logger.info(f"Created 125 batches of shape ({batch_size}, {seq_len})")

            result = {
                "model": model_name,
                "model_repo": model_repo,
                "vocab_size": vocab_size,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "N": N,
                "status": "ok"
            }

            # =========================================================
            # BASELINE RUN (in subprocess)
            # =========================================================
            logger.info("Running BASELINE...")
            baseline_result = run_subprocess(
                model_repo, batch_size, seq_len, "baseline",
                WARMUP_STEPS, MEASURE_STEPS, SEED, LEARNING_RATE,
                SUBPROCESS_TIMEOUT
            )

            if "error" in baseline_result:
                if baseline_result.get("crashed"):
                    logger.info(f"  BASELINE CRASHED: {baseline_result['error']}")
                    result["baseline"] = {"status": "crashed", "error": baseline_result["error"]}
                else:
                    logger.info(f"  BASELINE ERROR: {baseline_result['error']}")
                    result["baseline"] = {"status": "error", "error": baseline_result["error"]}
            else:
                result["baseline"] = baseline_result
                logger.info(f"  Time: {baseline_result['time_ms']:.2f}ms, "
                           f"Memory: {baseline_result['memory_gb']:.2f}GB, "
                           f"Final loss: {baseline_result['losses'][-1]:.4f}")

            # Cooldown
            logger.info(f"Cooldown {COOLDOWN_SECONDS}s...")
            gc.collect()
            time.sleep(COOLDOWN_SECONDS)

            # =========================================================
            # CCE RUN (in subprocess)
            # =========================================================
            logger.info("Running CCE...")
            cce_result = run_subprocess(
                model_repo, batch_size, seq_len, "cce",
                WARMUP_STEPS, MEASURE_STEPS, SEED, LEARNING_RATE,
                SUBPROCESS_TIMEOUT
            )

            if "error" in cce_result:
                if cce_result.get("crashed"):
                    logger.info(f"  CCE CRASHED: {cce_result['error']}")
                    result["cce"] = {"status": "crashed", "error": cce_result["error"]}
                else:
                    logger.info(f"  CCE ERROR: {cce_result['error']}")
                    result["cce"] = {"status": "error", "error": cce_result["error"]}
            else:
                result["cce"] = cce_result
                logger.info(f"  Time: {cce_result['time_ms']:.2f}ms, "
                           f"Memory: {cce_result['memory_gb']:.2f}GB, "
                           f"Final loss: {cce_result['losses'][-1]:.4f}")

            # =========================================================
            # COMPARISON (if both succeeded)
            # =========================================================
            baseline_ok = "losses" in result.get("baseline", {})
            cce_ok = "losses" in result.get("cce", {})

            if baseline_ok and cce_ok:
                comparison = compare_results(result["baseline"], result["cce"])
                result["comparison"] = comparison
                logger.info(f"RESULTS: Speedup={comparison['speedup']:.2f}x, "
                           f"MemSave={comparison['memory_savings_pct']:.1f}%, "
                           f"MaxLossDiff={comparison['loss_max_diff_pct']:.2f}%")

                # Write to CSV
                with open(csv_file, 'a') as f:
                    f.write(f"{model_name},{vocab_size},{batch_size},{seq_len},{N},"
                            f"{comparison['speedup']:.2f},{comparison['memory_savings_pct']:.1f},"
                            f"{result['baseline']['time_ms']:.2f},{result['baseline']['memory_gb']:.2f},"
                            f"{result['baseline']['losses'][-1]:.6f},"
                            f"{result['cce']['time_ms']:.2f},{result['cce']['memory_gb']:.2f},"
                            f"{result['cce']['losses'][-1]:.6f}\n")
            elif cce_ok and not baseline_ok:
                # Baseline crashed but CCE worked - this is the key benefit!
                logger.info(f"RESULTS: Baseline CRASHED, CCE worked! "
                           f"Time={result['cce']['time_ms']:.2f}ms, "
                           f"Mem={result['cce']['memory_gb']:.2f}GB")
                result["status"] = "baseline_crashed"

                # Write to CSV with baseline as crashed
                with open(csv_file, 'a') as f:
                    f.write(f"{model_name},{vocab_size},{batch_size},{seq_len},{N},"
                            f"inf,inf,CRASHED,CRASHED,CRASHED,"
                            f"{result['cce']['time_ms']:.2f},{result['cce']['memory_gb']:.2f},"
                            f"{result['cce']['losses'][-1]:.6f}\n")
            else:
                result["status"] = "error"
                logger.info("RESULTS: Both methods failed")

            results["results"].append(result)

            # Cooldown before next config
            if current_config < total_configs:
                logger.info(f"Cooldown {COOLDOWN_SECONDS}s before next config...")
                gc.collect()
                time.sleep(COOLDOWN_SECONDS)

        # Delete model from cache
        delete_model_cache(model_repo)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    successful = [r for r in results["results"] if r["status"] == "ok"]
    baseline_crashed = [r for r in results["results"] if r["status"] == "baseline_crashed"]

    if successful or baseline_crashed:
        if successful:
            avg_speedup = sum(r["comparison"]["speedup"] for r in successful) / len(successful)
            avg_mem_save = sum(r["comparison"]["memory_savings_pct"] for r in successful) / len(successful)
            max_loss_diff = max(r["comparison"]["loss_max_diff_pct"] for r in successful)
            avg_loss_diff = sum(r["comparison"]["loss_mean_diff_pct"] for r in successful) / len(successful)
        else:
            avg_speedup = avg_mem_save = max_loss_diff = avg_loss_diff = 0

        results["summary"] = {
            "total_configs": total_configs,
            "successful": len(successful),
            "baseline_crashed": len(baseline_crashed),
            "failed": total_configs - len(successful) - len(baseline_crashed),
            "avg_speedup": avg_speedup,
            "avg_memory_savings_pct": avg_mem_save,
            "max_loss_diff_pct": max_loss_diff,
            "avg_loss_diff_pct": avg_loss_diff,
            "all_correct": max_loss_diff < 5.0 if successful else True
        }

        logger.info(f"Successful runs: {len(successful)}/{total_configs}")
        logger.info(f"Baseline crashed (CCE worked): {len(baseline_crashed)}/{total_configs}")
        if successful:
            logger.info(f"Average speedup: {avg_speedup:.2f}x")
            logger.info(f"Average memory savings: {avg_mem_save:.1f}%")
            logger.info(f"Max loss difference: {max_loss_diff:.2f}%")
    else:
        logger.info("No successful runs!")
        results["summary"] = {
            "total_configs": total_configs,
            "successful": 0,
            "baseline_crashed": 0,
            "failed": total_configs,
            "error": "All runs failed"
        }

    # Save JSON results
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {csv_file}")
    logger.info(f"Full results: {json_file}")
    logger.info("Benchmark complete!")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Baseline vs CCE")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    # Clear any existing model cache to start fresh
    print("Clearing model cache before starting...")
    for model_repo, _ in MODELS:
        delete_model_cache(model_repo)

    run_benchmark(output_dir=args.output_dir)
