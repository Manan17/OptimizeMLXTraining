#!/usr/bin/env python3
"""Comprehensive CCE kernel test suite."""

import mlx.core as mx
import time

def test_forward_backward(N, H, V):
    """Test CCE forward and backward against baseline."""
    mx.random.seed(42)
    hidden = mx.random.normal([N, H])
    weight = mx.random.normal([V, H]) * 0.02
    targets = mx.random.randint(0, V, [N]).astype(mx.int32)

    # CCE
    def loss_fn_cce(h, w):
        loss = mx.fast.cce_loss(h, w, targets)
        return mx.mean(loss)

    loss_cce, grads_cce = mx.value_and_grad(loss_fn_cce, argnums=(0, 1))(hidden, weight)
    grad_h_cce, grad_w_cce = grads_cce
    mx.eval(loss_cce, grad_h_cce, grad_w_cce)

    # Baseline
    def loss_fn_baseline(h, w):
        logits = h @ w.T
        max_logits = mx.max(logits, axis=-1, keepdims=True)
        exp_logits = mx.exp(logits - max_logits)
        sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
        log_softmax = logits - max_logits - mx.log(sum_exp)
        batch_indices = mx.arange(N)
        target_log_probs = log_softmax[batch_indices, targets]
        loss = -target_log_probs
        return mx.mean(loss)

    loss_baseline, grads_baseline = mx.value_and_grad(loss_fn_baseline, argnums=(0, 1))(hidden, weight)
    grad_h_baseline, grad_w_baseline = grads_baseline
    mx.eval(loss_baseline, grad_h_baseline, grad_w_baseline)

    # Compare
    loss_diff = abs(float(loss_baseline) - float(loss_cce))
    grad_h_diff = float(mx.max(mx.abs(grad_h_baseline - grad_h_cce)))
    grad_w_diff = float(mx.max(mx.abs(grad_w_baseline - grad_w_cce)))

    passed = loss_diff < 1e-3 and grad_h_diff < 1e-3 and grad_w_diff < 1e-3
    status = "OK" if passed else "FAILED"

    return passed, status, loss_diff, grad_h_diff, grad_w_diff

def benchmark_cce(N, H, V, iterations=10):
    """Benchmark CCE vs baseline."""
    mx.random.seed(42)
    hidden = mx.random.normal([N, H])
    weight = mx.random.normal([V, H]) * 0.02
    targets = mx.random.randint(0, V, [N]).astype(mx.int32)

    # Warmup
    for _ in range(3):
        loss = mx.fast.cce_loss(hidden, weight, targets)
        mx.eval(loss)

    # CCE forward timing
    start = time.perf_counter()
    for _ in range(iterations):
        loss = mx.fast.cce_loss(hidden, weight, targets)
        mx.eval(loss)
    cce_forward_time = (time.perf_counter() - start) / iterations * 1000

    # CCE backward timing
    def loss_fn_cce(h, w):
        return mx.mean(mx.fast.cce_loss(h, w, targets))

    # Warmup
    for _ in range(3):
        loss, grads = mx.value_and_grad(loss_fn_cce, argnums=(0, 1))(hidden, weight)
        mx.eval(loss, grads)

    start = time.perf_counter()
    for _ in range(iterations):
        loss, grads = mx.value_and_grad(loss_fn_cce, argnums=(0, 1))(hidden, weight)
        mx.eval(loss, grads)
    cce_backward_time = (time.perf_counter() - start) / iterations * 1000

    # Baseline forward timing
    def forward_baseline(h, w):
        logits = h @ w.T
        max_logits = mx.max(logits, axis=-1, keepdims=True)
        exp_logits = mx.exp(logits - max_logits)
        sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
        log_softmax = logits - max_logits - mx.log(sum_exp)
        batch_indices = mx.arange(N)
        target_log_probs = log_softmax[batch_indices, targets]
        return -target_log_probs

    # Warmup
    for _ in range(3):
        loss = mx.mean(forward_baseline(hidden, weight))
        mx.eval(loss)

    start = time.perf_counter()
    for _ in range(iterations):
        loss = mx.mean(forward_baseline(hidden, weight))
        mx.eval(loss)
    baseline_forward_time = (time.perf_counter() - start) / iterations * 1000

    # Baseline backward timing
    def loss_fn_baseline(h, w):
        return mx.mean(forward_baseline(h, w))

    # Warmup
    for _ in range(3):
        loss, grads = mx.value_and_grad(loss_fn_baseline, argnums=(0, 1))(hidden, weight)
        mx.eval(loss, grads)

    start = time.perf_counter()
    for _ in range(iterations):
        loss, grads = mx.value_and_grad(loss_fn_baseline, argnums=(0, 1))(hidden, weight)
        mx.eval(loss, grads)
    baseline_backward_time = (time.perf_counter() - start) / iterations * 1000

    return {
        'cce_forward': cce_forward_time,
        'cce_backward': cce_backward_time,
        'baseline_forward': baseline_forward_time,
        'baseline_backward': baseline_backward_time,
    }

if __name__ == "__main__":
    print("=" * 70)
    print("CCE Kernel Correctness Tests")
    print("=" * 70)

    test_cases = [
        (4, 128, 1000),
        (16, 256, 2000),
        (32, 512, 4000),
        (64, 768, 8000),
        (128, 768, 16000),
    ]

    all_passed = True
    for N, H, V in test_cases:
        passed, status, loss_diff, grad_h_diff, grad_w_diff = test_forward_backward(N, H, V)
        print(f"N={N:4d}, H={H:4d}, V={V:5d}: {status:6s} "
              f"(loss={loss_diff:.2e}, grad_h={grad_h_diff:.2e}, grad_w={grad_w_diff:.2e})")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All correctness tests PASSED!")
    else:
        print("Some tests FAILED!")

    print("\n" + "=" * 70)
    print("CCE Kernel Benchmark")
    print("=" * 70)

    benchmark_cases = [
        (64, 768, 8000),
        (128, 768, 16000),
        (256, 768, 32000),
    ]

    for N, H, V in benchmark_cases:
        print(f"\nN={N}, H={H}, V={V}")
        times = benchmark_cce(N, H, V)
        print(f"  Forward:  CCE={times['cce_forward']:.2f}ms, Baseline={times['baseline_forward']:.2f}ms "
              f"(speedup: {times['baseline_forward']/times['cce_forward']:.2f}x)")
        print(f"  Backward: CCE={times['cce_backward']:.2f}ms, Baseline={times['baseline_backward']:.2f}ms "
              f"(speedup: {times['baseline_backward']/times['cce_backward']:.2f}x)")

    print("\n" + "=" * 70)
    print("Memory Analysis")
    print("=" * 70)
    N, H, V = 1024, 768, 128000
    logits_memory_gb = N * V * 4 / 1e9
    cce_memory_gb = N * 256 * 4 / 1e9  # Only vocab tile size
    print(f"For N={N}, H={H}, V={V}:")
    print(f"  Baseline logits memory: {logits_memory_gb:.2f} GB")
    print(f"  CCE working memory:     {cce_memory_gb:.4f} GB")
    print(f"  Memory savings:         {logits_memory_gb/cce_memory_gb:.0f}x")
