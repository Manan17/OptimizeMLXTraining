#!/usr/bin/env python3
"""
Compare MLX Baseline vs CCE training losses over N steps.
Plots loss curves and difference.
"""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from datasets import load_dataset as hf_load_dataset
import matplotlib.pyplot as plt
import argparse

# Config
batch_size = 8
max_seq_length = 128
warmup_steps = 2

def get_lm_head_weight(model):
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight
    raise ValueError('Cannot find lm_head weight')

def get_hidden_states(model, inputs):
    if hasattr(model, 'model'):
        return model.model(inputs)
    raise ValueError('Cannot find transformer model')

def baseline_loss_fn(model, batch, lengths):
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
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    hidden = get_hidden_states(model, inputs)
    weight = get_lm_head_weight(model)
    if hidden.dtype == mx.bfloat16:
        hidden = hidden.astype(mx.float32)
        weight = weight.astype(mx.float32)
    B, S, H = hidden.shape
    hidden_flat = hidden.reshape(B * S, H)
    targets_flat = targets.reshape(B * S)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    mask_flat = mask.reshape(B * S)
    ce = mx.fast.cce_loss(hidden_flat, weight, targets_flat)
    ce = (ce * mask_flat).sum() / mask_flat.sum()
    return ce, mx.sum(mask_flat)

def create_batch_iterator(dataset, tokenizer, batch_size, max_seq_length):
    all_tokens = []
    for item in dataset:
        tokens = tokenizer.encode(item['text'])
        if len(tokens) > 2:
            all_tokens.append(tokens[:max_seq_length])
    all_tokens.sort(key=len)
    while True:
        indices = np.random.permutation(len(all_tokens))
        for i in range(0, len(indices) - batch_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_tokens = [all_tokens[j] for j in batch_indices]
            padded = []
            lengths = []
            for tokens in batch_tokens:
                pad_len = max_seq_length - len(tokens)
                padded.append(tokens + [tokenizer.pad_token_id or 0] * pad_len)
                lengths.append([0, len(tokens) - 1])
            yield mx.array(padded), mx.array(lengths)

def run_training(use_cce, model_path, dataset, num_steps):
    model, tokenizer = load(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    optimizer = optim.Adam(learning_rate=1e-4)
    loss_fn = cce_loss_fn if use_cce else baseline_loss_fn
    batch_iter = create_batch_iterator(dataset, tokenizer, batch_size, max_seq_length)
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup
    for _ in range(warmup_steps):
        batch, lengths = next(batch_iter)
        (loss, _), grads = loss_value_and_grad(model, batch, lengths)
        optimizer.update(model, grads)
        mx.eval(loss)

    # Training
    losses = []
    for step in range(num_steps):
        batch, lengths = next(batch_iter)
        (loss, _), grads = loss_value_and_grad(model, batch, lengths)
        optimizer.update(model, grads)
        mx.eval(loss)
        losses.append(float(loss))
        if (step + 1) % 20 == 0:
            print(f'  Step {step+1}: loss = {losses[-1]:.4f}')

    return losses


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline vs CCE training')
    parser.add_argument('--steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--model', type=str, default='HuggingFaceTB/SmolLM2-135M', help='Model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Load dataset once
    print('Loading dataset...')
    raw_train = hf_load_dataset('roneneldan/TinyStories', split='train[:1000]')
    dataset = [{'text': ex['text']} for ex in raw_train]

    # Run baseline
    print(f'Running baseline ({args.steps} steps)...')
    np.random.seed(args.seed)
    baseline_losses = run_training(use_cce=False, model_path=args.model, dataset=dataset, num_steps=args.steps)

    # Run CCE
    print(f'Running CCE ({args.steps} steps)...')
    np.random.seed(args.seed)
    cce_losses = run_training(use_cce=True, model_path=args.model, dataset=dataset, num_steps=args.steps)

    # Stats
    baseline_arr = np.array(baseline_losses)
    cce_arr = np.array(cce_losses)
    diff = baseline_arr - cce_arr

    print(f'\nComparison over {args.steps} steps:')
    print(f'max(abs(baseline - cce)) = {np.max(np.abs(diff)):.6f}')
    print(f'max(baseline - cce)      = {np.max(diff):.6f}')
    print(f'min(baseline - cce)      = {np.min(diff):.6f}')
    print(f'mean(abs(diff))          = {np.mean(np.abs(diff)):.6f}')

    # Plot
    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(baseline_losses, label='MLX Baseline', alpha=0.8)
    plt.plot(cce_losses, label='CCE', alpha=0.8)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Training Loss: MLX Baseline vs CCE ({args.steps} steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Difference
    plt.subplot(1, 2, 2)
    plt.plot(diff, color='purple', alpha=0.8)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Step')
    plt.ylabel('Baseline - CCE')
    plt.title('Loss Difference (Baseline - CCE)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f'baseline_vs_cce_{args.steps}steps.png'
    plt.savefig(output_path, dpi=150)
    print(f'\nPlot saved to: {output_path}')


if __name__ == '__main__':
    main()
