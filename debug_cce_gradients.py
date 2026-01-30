"""
Debug script to identify exactly where CCE gradients diverge from baseline.
Run on EC2 with: python3 debug_cce_gradients.py
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import numpy as np

def debug_gradients(batch_size=8, seq_len=128):
    print(f"Loading model...")
    model, tok = load('mlx-community/Qwen3-0.6B-bf16')
    mx.eval(model.parameters())

    # Create batch
    tokens = tok.encode('Hello world. ' * 100)[:seq_len]
    batch = mx.array([tokens] * batch_size, dtype=mx.int32)

    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    # Get hidden states and weight
    hidden = model.model(inputs)
    weight = model.model.embed_tokens.weight
    mx.eval(hidden, weight)

    B, S, H = hidden.shape
    V = weight.shape[0]
    N = B * S

    print(f"\n=== Problem Size ===")
    print(f"Batch: {B}, Seq: {S}, N: {N}, V: {V}, H: {H}")
    print(f"Hidden dtype: {hidden.dtype}, Weight dtype: {weight.dtype}")

    h_flat = hidden.reshape(N, H)
    t_flat = targets.reshape(N)

    # =========================================================================
    # Step 1: Compare forward pass (logsumexp)
    # =========================================================================
    print(f"\n=== Step 1: Forward Pass (LogSumExp) ===")

    # Baseline logsumexp
    logits = h_flat @ weight.T
    mx.eval(logits)

    # Compute baseline logsumexp manually
    logits_f32 = logits.astype(mx.float32)
    max_logits = mx.max(logits_f32, axis=1, keepdims=True)
    baseline_lse = (max_logits.squeeze() + mx.log(mx.sum(mx.exp(logits_f32 - max_logits), axis=1)))
    mx.eval(baseline_lse)

    # CCE forward - get logsumexp
    cce_loss = mx.fast.cce_loss(h_flat, weight, t_flat)
    mx.eval(cce_loss)

    # We need to extract logsumexp from CCE internals
    # For now, compare losses
    baseline_ce = nn.losses.cross_entropy(logits, t_flat, reduction='none')
    mx.eval(baseline_ce)

    print(f"Baseline CE mean: {baseline_ce.mean().item():.6f}")
    print(f"CCE loss mean: {cce_loss.mean().item():.6f}")
    print(f"Forward diff: {abs(baseline_ce.mean().item() - cce_loss.mean().item()):.6f}")

    # =========================================================================
    # Step 2: Compare gradients element-by-element
    # =========================================================================
    print(f"\n=== Step 2: Gradient Comparison ===")

    def cce_loss_fn(h, w, t):
        return mx.fast.cce_loss(h, w, t).mean()

    def baseline_loss_fn(h, w, t):
        logits = h @ w.T
        return nn.losses.cross_entropy(logits, t).mean()

    # Compute gradients
    cce_grad_h, cce_grad_w = mx.grad(cce_loss_fn, argnums=(0,1))(h_flat, weight, t_flat)
    baseline_grad_h, baseline_grad_w = mx.grad(baseline_loss_fn, argnums=(0,1))(h_flat, weight, t_flat)
    mx.eval(cce_grad_h, cce_grad_w, baseline_grad_h, baseline_grad_w)

    # Analyze grad_hidden
    diff_h = (baseline_grad_h - cce_grad_h).abs()
    rel_diff_h = diff_h / (baseline_grad_h.abs() + 1e-10)

    print(f"\ngrad_hidden analysis:")
    print(f"  Baseline range: [{baseline_grad_h.min().item():.6f}, {baseline_grad_h.max().item():.6f}]")
    print(f"  CCE range: [{cce_grad_h.min().item():.6f}, {cce_grad_h.max().item():.6f}]")
    print(f"  Abs diff max: {diff_h.max().item():.6f}")
    print(f"  Abs diff mean: {diff_h.mean().item():.6f}")
    print(f"  Rel diff max: {rel_diff_h.max().item():.2%}")
    print(f"  Rel diff mean: {rel_diff_h.mean().item():.2%}")

    # Find which rows have the largest errors
    row_errors = diff_h.sum(axis=1)
    mx.eval(row_errors)
    worst_rows = mx.argsort(row_errors)[-10:]
    mx.eval(worst_rows)
    print(f"  Worst 10 rows (by total error): {worst_rows.tolist()}")

    # Analyze grad_weight
    diff_w = (baseline_grad_w - cce_grad_w).abs()
    rel_diff_w = diff_w / (baseline_grad_w.abs() + 1e-10)

    print(f"\ngrad_weight analysis:")
    print(f"  Baseline range: [{baseline_grad_w.min().item():.6f}, {baseline_grad_w.max().item():.6f}]")
    print(f"  CCE range: [{cce_grad_w.min().item():.6f}, {cce_grad_w.max().item():.6f}]")
    print(f"  Abs diff max: {diff_w.max().item():.6f}")
    print(f"  Abs diff mean: {diff_w.mean().item():.6f}")
    print(f"  Rel diff max: {rel_diff_w.max().item():.2%}")
    print(f"  Rel diff mean: {rel_diff_w.mean().item():.2%}")

    # =========================================================================
    # Step 3: Check sign agreement (critical for training direction)
    # =========================================================================
    print(f"\n=== Step 3: Sign Agreement ===")

    # Check if gradients have the same sign
    sign_match_h = ((baseline_grad_h > 0) == (cce_grad_h > 0)).astype(mx.float32).mean()
    sign_match_w = ((baseline_grad_w > 0) == (cce_grad_w > 0)).astype(mx.float32).mean()
    mx.eval(sign_match_h, sign_match_w)

    print(f"  grad_hidden sign agreement: {sign_match_h.item():.2%}")
    print(f"  grad_weight sign agreement: {sign_match_w.item():.2%}")

    # =========================================================================
    # Step 4: Compute d_logits manually and compare
    # =========================================================================
    print(f"\n=== Step 4: d_logits Analysis ===")

    # Baseline d_logits = softmax(logits) - one_hot(targets)
    # For numerical stability, use log-softmax
    log_softmax = logits_f32 - baseline_lse.reshape(-1, 1)
    baseline_softmax = mx.exp(log_softmax)
    mx.eval(baseline_softmax)

    # Create one-hot targets
    one_hot = mx.zeros((N, V), dtype=mx.float32)
    # one_hot[i, t_flat[i]] = 1  (need to do this differently in MLX)

    # d_logits = softmax - one_hot, scaled by 1/N for mean reduction
    # For target positions: d_logit = softmax[target] - 1
    # For non-target positions: d_logit = softmax[non_target]

    print(f"  Baseline softmax range: [{baseline_softmax.min().item():.6f}, {baseline_softmax.max().item():.6f}]")
    print(f"  Baseline softmax sum (should be ~1): {baseline_softmax.sum(axis=1).mean().item():.6f}")

    # Check specific target positions
    target_probs = []
    for i in range(min(10, N)):
        target_idx = t_flat[i].item()
        prob = baseline_softmax[i, target_idx].item()
        target_probs.append(prob)
    print(f"  First 10 target probs: {[f'{p:.4f}' for p in target_probs]}")

    # =========================================================================
    # Step 5: Test with FP32 to isolate BF16 issues
    # =========================================================================
    print(f"\n=== Step 5: FP32 Comparison ===")

    h_f32 = h_flat.astype(mx.float32)
    w_f32 = weight.astype(mx.float32)
    mx.eval(h_f32, w_f32)

    def cce_loss_fn_f32(h, w, t):
        return mx.fast.cce_loss(h, w, t).mean()

    def baseline_loss_fn_f32(h, w, t):
        logits = h @ w.T
        return nn.losses.cross_entropy(logits, t).mean()

    cce_grad_h_f32, cce_grad_w_f32 = mx.grad(cce_loss_fn_f32, argnums=(0,1))(h_f32, w_f32, t_flat)
    baseline_grad_h_f32, baseline_grad_w_f32 = mx.grad(baseline_loss_fn_f32, argnums=(0,1))(h_f32, w_f32, t_flat)
    mx.eval(cce_grad_h_f32, cce_grad_w_f32, baseline_grad_h_f32, baseline_grad_w_f32)

    diff_h_f32 = (baseline_grad_h_f32 - cce_grad_h_f32).abs()
    diff_w_f32 = (baseline_grad_w_f32 - cce_grad_w_f32).abs()

    print(f"  FP32 grad_hidden rel diff max: {(diff_h_f32.max() / (baseline_grad_h_f32.abs().max() + 1e-10)).item():.2%}")
    print(f"  FP32 grad_weight rel diff max: {(diff_w_f32.max() / (baseline_grad_w_f32.abs().max() + 1e-10)).item():.2%}")

    # =========================================================================
    # Step 6: Test different batch sizes
    # =========================================================================
    print(f"\n=== Step 6: Batch Size Sweep ===")

    for bs in [4, 8, 12, 16, 24, 32]:
        try:
            batch_test = mx.array([tokens] * bs, dtype=mx.int32)
            inputs_test = batch_test[:, :-1]
            targets_test = batch_test[:, 1:]

            hidden_test = model.model(inputs_test)
            mx.eval(hidden_test)

            B_t, S_t, H_t = hidden_test.shape
            h_test = hidden_test.reshape(B_t * S_t, H_t)
            t_test = targets_test.reshape(B_t * S_t)

            cce_g_h, _ = mx.grad(cce_loss_fn, argnums=(0,1))(h_test, weight, t_test)
            base_g_h, _ = mx.grad(baseline_loss_fn, argnums=(0,1))(h_test, weight, t_test)
            mx.eval(cce_g_h, base_g_h)

            diff = (base_g_h - cce_g_h).abs()
            rel_err = (diff.max() / (base_g_h.abs().max() + 1e-10)).item()

            print(f"  Batch {bs:2d} (N={B_t*S_t:4d}): grad_hidden rel diff max = {rel_err:.2%}")
        except Exception as e:
            print(f"  Batch {bs:2d}: Error - {e}")

    print(f"\n=== Summary ===")
    print(f"If FP32 errors are small but BF16 errors are large, the issue is in BF16 handling.")
    print(f"If errors increase with batch size, the issue is in chunking logic.")
    print(f"If sign agreement is low (<95%), gradients point in wrong direction (critical bug).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=128)
    args = parser.parse_args()

    debug_gradients(args.batch_size, args.seq_len)
