#!/usr/bin/env python3
"""
Simple MLX Training Script with CCE (Chunked Cross-Entropy)
============================================================
Usage:
    pip install mlx mlx-lm datasets
    python train.py                     # Full fine-tuning with CCE
    python train.py --lora              # LoRA fine-tuning
    python train.py --no-cce            # Baseline cross-entropy (no CCE)
    python train.py --model mlx-community/Llama-3.2-1B-Instruct-4bit --lora
"""

import argparse
import gc
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx.utils import tree_flatten
from datasets import load_dataset


# =============================================================================
# Helpers
# =============================================================================

def format_sample(sample):
    """Format an Alpaca dataset sample into a training string."""
    if sample["input"]:
        return (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Response:\n{sample['output']}"
    )


def create_batches(dataset, tokenizer, batch_size, seq_len, num_batches, seed):
    """Pre-tokenize deterministic batches from the Alpaca dataset."""
    rng = mx.random.key(seed)
    indices = mx.random.permutation(len(dataset), key=rng).tolist()

    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    batches = []

    for b in range(num_batches):
        start = b * batch_size
        batch_idx = indices[start : start + batch_size]
        # Wrap around if we run out of data
        if len(batch_idx) < batch_size:
            batch_idx += indices[: batch_size - len(batch_idx)]

        batch_tokens = []
        for idx in batch_idx:
            tokens = tokenizer.encode(format_sample(dataset[idx]))
            if len(tokens) >= seq_len:
                tokens = tokens[:seq_len]
            else:
                tokens = tokens + [pad_id] * (seq_len - len(tokens))
            batch_tokens.append(tokens)

        batches.append(mx.array(batch_tokens, dtype=mx.int32))

    return batches


def get_lm_weight(model):
    """
    Get the LM head weight matrix, handling:
    - Untied lm_head (Qwen, etc.)
    - Tied embeddings (Llama, etc.)
    - Quantized layers (auto-dequantize)
    """
    # Check for untied lm_head first
    if (
        hasattr(model, "lm_head")
        and model.lm_head is not None
        and hasattr(model.lm_head, "weight")
    ):
        layer = model.lm_head
        if hasattr(layer, "scales"):
            return mx.dequantize(
                layer.weight,
                layer.scales,
                getattr(layer, "biases", None),
                group_size=getattr(layer, "group_size", 64),
                bits=getattr(layer, "bits", 4),
            )
        return layer.weight

    # Fall back to tied embed_tokens
    embed = model.model.embed_tokens
    if hasattr(embed, "scales"):
        return mx.dequantize(
            embed.weight,
            embed.scales,
            getattr(embed, "biases", None),
            group_size=getattr(embed, "group_size", 64),
            bits=getattr(embed, "bits", 4),
        )
    return embed.weight


def is_quantized(model):
    """Check if the model has quantized layers."""
    if hasattr(model, "lm_head") and model.lm_head is not None:
        if hasattr(model.lm_head, "scales"):
            return True
    if hasattr(model.model, "embed_tokens") and hasattr(model.model.embed_tokens, "scales"):
        return True
    return False


def apply_lora(model, num_layers, rank=8, alpha=16):
    """Apply LoRA adapters and freeze base weights."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    linear_to_lora_layers(
        model,
        num_layers=num_layers,
        config={"rank": rank, "alpha": alpha, "dropout": 0.0, "scale": alpha / rank},
    )
    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"], strict=False)


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    trainable = sum(
        p.size for _, p in tree_flatten(model.trainable_parameters())
    )
    return total, trainable


# =============================================================================
# Loss functions
# =============================================================================

def make_cce_loss(model):
    """Create a CCE loss function (memory-efficient, no full logit materialization)."""
    def loss_fn(model, batch):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        hidden = model.model(inputs)
        weight = get_lm_weight(model)
        B, S, H = hidden.shape
        loss = mx.fast.cce_loss(
            hidden.reshape(B * S, H),
            weight,
            targets.reshape(B * S),
        )
        return loss.astype(mx.float32).mean()

    return loss_fn


def make_baseline_loss():
    """Create a baseline cross-entropy loss (full logit materialization)."""
    def loss_fn(model, batch):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        loss = nn.losses.cross_entropy(logits, targets)
        return loss.astype(mx.float32).mean()

    return loss_fn


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simple MLX training with CCE")
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-1B-Instruct-bf16",
        help="HuggingFace model repo (default: mlx-community/Llama-3.2-1B-Instruct-bf16)",
    )
    parser.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning instead of full")
    parser.add_argument("--lora-layers", type=int, default=8, help="Number of LoRA layers (default: 8)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--steps", type=int, default=100, help="Training steps (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-cce", action="store_true", help="Disable CCE, use baseline cross-entropy")
    args = parser.parse_args()

    use_cce = not args.no_cce
    mx.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load model + tokenizer
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    quantized = is_quantized(model)
    if quantized:
        print("  Quantized model detected — will dequantize lm_head for CCE")

    # ------------------------------------------------------------------
    # 2. Apply LoRA if requested
    # ------------------------------------------------------------------
    if args.lora:
        mx.random.seed(args.seed)  # deterministic LoRA init
        apply_lora(model, num_layers=args.lora_layers)
        mx.eval(model.parameters())

    total_params, trainable_params = count_params(model)
    mode = "LoRA" if args.lora else "Full"
    loss_mode = "Baseline CE" if args.no_cce else "CCE"
    print(f"  Mode: {mode} fine-tuning | Loss: {loss_mode}")
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ------------------------------------------------------------------
    # 3. Load dataset + create batches
    # ------------------------------------------------------------------
    print("Loading dataset: tatsu-lab/alpaca")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  {len(dataset)} examples loaded")

    num_batches = args.steps + 5  # a few extra for safety
    batches = create_batches(
        dataset, tokenizer, args.batch_size, args.seq_len, num_batches, args.seed
    )
    print(f"  Created {len(batches)} batches (batch_size={args.batch_size}, seq_len={args.seq_len})")

    # ------------------------------------------------------------------
    # 4. Setup optimizer + loss
    # ------------------------------------------------------------------
    optimizer = optim.Adafactor(
        learning_rate=args.lr, relative_step=False, scale_parameter=False
    )

    if use_cce:
        loss_fn = make_cce_loss(model)
    else:
        loss_fn = make_baseline_loss()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.steps} steps...")
    print("-" * 60)

    gc.collect()
    mx.synchronize()
    mx.reset_peak_memory()

    start_time = time.perf_counter()
    losses = []

    for step in range(args.steps):
        batch = batches[step % len(batches)]
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.perf_counter() - start_time
            mem_gb = mx.get_peak_memory() / 1e9
            print(
                f"  Step {step + 1:4d}/{args.steps} | "
                f"Loss: {loss_val:.4f} | "
                f"Time: {elapsed:.1f}s | "
                f"Mem: {mem_gb:.2f} GB"
            )

    mx.synchronize()
    total_time = time.perf_counter() - start_time
    peak_mem = mx.get_peak_memory() / 1e9

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Training complete!")
    print(f"  Model:       {args.model}")
    print(f"  Mode:        {mode} | {loss_mode}")
    print(f"  Steps:       {args.steps}")
    print(f"  Final loss:  {losses[-1]:.4f}")
    print(f"  Total time:  {total_time:.1f}s ({total_time / args.steps * 1000:.0f} ms/step)")
    print(f"  Peak memory: {peak_mem:.2f} GB")


if __name__ == "__main__":
    main()
