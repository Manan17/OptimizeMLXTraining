#!/usr/bin/env python3
"""
Memory benchmark for Metal CCE kernel (mx.fast.cce_loss).

This tests the actual C++ Metal kernel, not pure Python implementations.
"""

import mlx.core as mx
import numpy as np
import time


def benchmark_memory(name, fn, hidden, weight, targets):
    """Benchmark memory usage of a function."""
    mx.clear_cache()
    mx.synchronize()
    mx.reset_peak_memory()

    result = fn(hidden, weight, targets)
    mx.eval(result)
    mx.synchronize()

    peak_mem = mx.get_peak_memory() / (1024**2)
    return peak_mem


def baseline_ce_forward(hidden, weight, targets):
    """Standard cross-entropy forward."""
    logits = hidden @ weight.T
    loss = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
        logits, targets[:, None], axis=1
    ).squeeze(-1)
    return loss


def baseline_ce_forward_backward(hidden, weight, targets):
    """Standard cross-entropy forward + backward."""
    def fn(h, w):
        logits = h @ w.T
        loss = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
            logits, targets[:, None], axis=1
        ).squeeze(-1)
        return mx.mean(loss)

    loss, grads = mx.value_and_grad(fn, argnums=(0, 1))(hidden, weight)
    return loss, grads[0], grads[1]


def cce_forward(hidden, weight, targets):
    """CCE kernel forward only."""
    return mx.fast.cce_loss(hidden, weight, targets)


def cce_forward_backward(hidden, weight, targets):
    """CCE kernel forward + backward."""
    def fn(h, w):
        return mx.mean(mx.fast.cce_loss(h, w, targets))

    loss, grads = mx.value_and_grad(fn, argnums=(0, 1))(hidden, weight)
    return loss, grads[0], grads[1]


def test_memory_comparison():
    """Compare memory usage between baseline and CCE kernel."""
    print("=" * 80)
    print("Metal CCE Kernel Memory Benchmark")
    print("=" * 80)

    configs = [
        (64, 2048, 128000),
        (128, 2048, 128000),
        (256, 2048, 128000),
        (512, 2048, 128000),
        (1024, 2048, 128000),
        (2048, 2048, 128000),
    ]

    print("\n--- FORWARD ONLY ---")
    print(f"{'Config':<30} {'Baseline':>12} {'CCE':>12} {'Savings':>10}")
    print("-" * 70)

    for N, H, V in configs:
        np.random.seed(42)
        hidden = mx.array(np.random.randn(N, H).astype(np.float32) * 0.1)
        weight = mx.array(np.random.randn(V, H).astype(np.float32) * 0.02)
        targets = mx.array(np.random.randint(0, V, N).astype(np.int32))
        mx.eval(hidden, weight, targets)

        # Baseline forward
        baseline_mem = benchmark_memory("baseline", baseline_ce_forward, hidden, weight, targets)

        # CCE forward
        cce_mem = benchmark_memory("cce", cce_forward, hidden, weight, targets)

        savings = baseline_mem / cce_mem if cce_mem > 0 else 0
        config_str = f"N={N}, H={H}, V={V}"
        print(f"{config_str:<30} {baseline_mem:>10.0f}MB {cce_mem:>10.0f}MB {savings:>9.2f}x")

    print("\n--- FORWARD + BACKWARD ---")
    print(f"{'Config':<30} {'Baseline':>12} {'CCE':>12} {'Savings':>10}")
    print("-" * 70)

    for N, H, V in configs:
        np.random.seed(42)
        hidden = mx.array(np.random.randn(N, H).astype(np.float32) * 0.1)
        weight = mx.array(np.random.randn(V, H).astype(np.float32) * 0.02)
        targets = mx.array(np.random.randint(0, V, N).astype(np.int32))
        mx.eval(hidden, weight, targets)

        try:
            # Baseline forward+backward
            baseline_mem = benchmark_memory("baseline", baseline_ce_forward_backward, hidden, weight, targets)

            # CCE forward+backward
            cce_mem = benchmark_memory("cce", cce_forward_backward, hidden, weight, targets)

            savings = baseline_mem / cce_mem if cce_mem > 0 else 0
            config_str = f"N={N}, H={H}, V={V}"
            print(f"{config_str:<30} {baseline_mem:>10.0f}MB {cce_mem:>10.0f}MB {savings:>9.2f}x")
        except Exception as e:
            config_str = f"N={N}, H={H}, V={V}"
            print(f"{config_str:<30} ERROR: {e}")


def test_correctness():
    """Test CCE kernel correctness."""
    print("\n" + "=" * 80)
    print("Correctness Test")
    print("=" * 80)

    N, H, V = 128, 512, 32000
    np.random.seed(42)
    hidden = mx.array(np.random.randn(N, H).astype(np.float32) * 0.1)
    weight = mx.array(np.random.randn(V, H).astype(np.float32) * 0.02)
    targets = mx.array(np.random.randint(0, V, N).astype(np.int32))

    # Baseline
    def baseline_fn(h, w):
        logits = h @ w.T
        loss = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
            logits, targets[:, None], axis=1
        ).squeeze(-1)
        return mx.mean(loss)

    loss_base, grads_base = mx.value_and_grad(baseline_fn, argnums=(0, 1))(hidden, weight)
    mx.eval(loss_base, grads_base)

    # CCE
    def cce_fn(h, w):
        return mx.mean(mx.fast.cce_loss(h, w, targets))

    loss_cce, grads_cce = mx.value_and_grad(cce_fn, argnums=(0, 1))(hidden, weight)
    mx.eval(loss_cce, grads_cce)

    loss_diff = abs(float(loss_base) - float(loss_cce))
    gh_diff = float(mx.max(mx.abs(grads_base[0] - grads_cce[0])))
    gw_diff = float(mx.max(mx.abs(grads_base[1] - grads_cce[1])))

    print(f"Loss diff: {loss_diff:.2e} {'PASS' if loss_diff < 1e-3 else 'FAIL'}")
    print(f"grad_h diff: {gh_diff:.2e} {'PASS' if gh_diff < 1e-3 else 'FAIL'}")
    print(f"grad_w diff: {gw_diff:.2e} {'PASS' if gw_diff < 1e-3 else 'FAIL'}")

    return loss_diff < 1e-3 and gh_diff < 1e-3 and gw_diff < 1e-3


if __name__ == "__main__":
    if test_correctness():
        test_memory_comparison()
    else:
        print("\nSkipping memory tests due to correctness failures")
