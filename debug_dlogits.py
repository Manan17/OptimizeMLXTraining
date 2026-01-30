"""
Debug script to compare d_logits computation step by step.
This isolates the exact point where CCE diverges from baseline.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

def debug_dlogits(batch_size=8, seq_len=128):
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
    print(f"dtype: {hidden.dtype}")

    h_flat = hidden.reshape(N, H)
    t_flat = targets.reshape(N)

    # Use FP32 for debugging
    h_f32 = h_flat.astype(mx.float32)
    w_f32 = weight.astype(mx.float32)
    mx.eval(h_f32, w_f32)

    # =========================================================================
    # Step 1: Compute baseline logits and logsumexp
    # =========================================================================
    print(f"\n=== Step 1: Baseline Logits and LogSumExp ===")

    logits = h_f32 @ w_f32.T
    mx.eval(logits)

    max_logits = mx.max(logits, axis=1, keepdims=True)
    lse = max_logits.squeeze() + mx.log(mx.sum(mx.exp(logits - max_logits), axis=1))
    mx.eval(lse)

    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"LogSumExp range: [{lse.min().item():.4f}, {lse.max().item():.4f}]")

    # =========================================================================
    # Step 2: Compute baseline d_logits = softmax - one_hot
    # =========================================================================
    print(f"\n=== Step 2: Baseline d_logits ===")

    # softmax = exp(logits - lse)
    softmax = mx.exp(logits - lse.reshape(-1, 1))
    mx.eval(softmax)

    print(f"Softmax sum (should be 1): {softmax.sum(axis=1).mean().item():.6f}")

    # d_logits = softmax, then subtract 1 at target positions
    d_logits_baseline = softmax.astype(mx.float32)

    # Subtract 1 at target positions
    for i in range(N):
        target_idx = t_flat[i].item()
        d_logits_baseline[i, target_idx] = d_logits_baseline[i, target_idx] - 1.0
    mx.eval(d_logits_baseline)

    # Scale by 1/N (mean reduction)
    d_logits_baseline = d_logits_baseline / N
    mx.eval(d_logits_baseline)

    print(f"d_logits_baseline range: [{d_logits_baseline.min().item():.6f}, {d_logits_baseline.max().item():.6f}]")

    # =========================================================================
    # Step 3: Compute baseline gradients
    # =========================================================================
    print(f"\n=== Step 3: Baseline Gradients ===")

    # grad_hidden = d_logits @ weight
    grad_h_baseline = d_logits_baseline @ w_f32
    mx.eval(grad_h_baseline)

    # grad_weight = d_logits.T @ hidden
    grad_w_baseline = d_logits_baseline.T @ h_f32
    mx.eval(grad_w_baseline)

    print(f"grad_hidden_baseline range: [{grad_h_baseline.min().item():.6f}, {grad_h_baseline.max().item():.6f}]")
    print(f"grad_weight_baseline range: [{grad_w_baseline.min().item():.6f}, {grad_w_baseline.max().item():.6f}]")

    # =========================================================================
    # Step 4: Compute CCE gradients using mx.grad
    # =========================================================================
    print(f"\n=== Step 4: CCE Gradients ===")

    def cce_loss_fn(h, w, t):
        return mx.fast.cce_loss(h, w, t).mean()

    cce_grad_h, cce_grad_w = mx.grad(cce_loss_fn, argnums=(0,1))(h_f32, w_f32, t_flat)
    mx.eval(cce_grad_h, cce_grad_w)

    print(f"grad_hidden_cce range: [{cce_grad_h.min().item():.6f}, {cce_grad_h.max().item():.6f}]")
    print(f"grad_weight_cce range: [{cce_grad_w.min().item():.6f}, {cce_grad_w.max().item():.6f}]")

    # =========================================================================
    # Step 5: Compare
    # =========================================================================
    print(f"\n=== Step 5: Comparison ===")

    diff_h = (grad_h_baseline - cce_grad_h).abs()
    diff_w = (grad_w_baseline - cce_grad_w).abs()

    print(f"grad_hidden diff max: {diff_h.max().item():.6f}")
    print(f"grad_hidden diff mean: {diff_h.mean().item():.6f}")
    print(f"grad_hidden rel diff max: {(diff_h.max() / (grad_h_baseline.abs().max() + 1e-10)).item():.2%}")

    print(f"grad_weight diff max: {diff_w.max().item():.6f}")
    print(f"grad_weight diff mean: {diff_w.mean().item():.6f}")
    print(f"grad_weight rel diff max: {(diff_w.max() / (grad_w_baseline.abs().max() + 1e-10)).item():.2%}")

    # =========================================================================
    # Step 6: Analyze error distribution per row
    # =========================================================================
    print(f"\n=== Step 6: Error Distribution ===")

    row_errors = diff_h.sum(axis=1)
    mx.eval(row_errors)

    # Group errors by sequence position
    errors_by_pos = []
    for pos in range(S):
        pos_errors = []
        for b in range(B):
            row_idx = b * S + pos
            if row_idx < N:
                pos_errors.append(row_errors[row_idx].item())
        errors_by_pos.append(sum(pos_errors) / len(pos_errors) if pos_errors else 0)

    # Find positions with highest error
    sorted_pos = sorted(range(len(errors_by_pos)), key=lambda i: errors_by_pos[i], reverse=True)
    print(f"Top 10 error positions (by avg across batches):")
    for i in range(min(10, len(sorted_pos))):
        pos = sorted_pos[i]
        print(f"  Position {pos}: avg error = {errors_by_pos[pos]:.6f}")

    # =========================================================================
    # Step 7: Check specific rows with high error
    # =========================================================================
    print(f"\n=== Step 7: High Error Row Analysis ===")

    worst_rows = mx.argsort(row_errors)[-5:]
    mx.eval(worst_rows)

    for row_idx in worst_rows.tolist():
        target = t_flat[row_idx].item()
        baseline_row = grad_h_baseline[row_idx]
        cce_row = cce_grad_h[row_idx]
        diff_row = (baseline_row - cce_row).abs()

        print(f"\nRow {row_idx} (target={target}):")
        print(f"  baseline grad_h range: [{baseline_row.min().item():.6f}, {baseline_row.max().item():.6f}]")
        print(f"  cce grad_h range: [{cce_row.min().item():.6f}, {cce_row.max().item():.6f}]")
        print(f"  max diff at dim: {mx.argmax(diff_row).item()}")

    # =========================================================================
    # Step 8: Verify using nn.losses.cross_entropy gradient
    # =========================================================================
    print(f"\n=== Step 8: Verify with nn.losses.cross_entropy ===")

    def ce_loss_fn(h, w, t):
        logits = h @ w.T
        return nn.losses.cross_entropy(logits, t).mean()

    ce_grad_h, ce_grad_w = mx.grad(ce_loss_fn, argnums=(0,1))(h_f32, w_f32, t_flat)
    mx.eval(ce_grad_h, ce_grad_w)

    diff_manual_vs_ce = (grad_h_baseline - ce_grad_h).abs()
    print(f"Manual baseline vs nn.cross_entropy rel diff max: {(diff_manual_vs_ce.max() / (grad_h_baseline.abs().max() + 1e-10)).item():.4%}")

    diff_cce_vs_ce = (cce_grad_h - ce_grad_h).abs()
    print(f"CCE vs nn.cross_entropy rel diff max: {(diff_cce_vs_ce.max() / (ce_grad_h.abs().max() + 1e-10)).item():.4%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=128)
    args = parser.parse_args()

    debug_dlogits(args.batch_size, args.seq_len)
