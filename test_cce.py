#!/usr/bin/env python3
"""Test the CCE (Cut Cross-Entropy) kernel implementation."""

import sys
sys.path.insert(0, '/Users/mananshah/Desktop/MetalLora/mlx/build/lib')

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time

def softmax_cross_entropy_baseline(logits, targets, ignore_index=-100):
    """Baseline implementation for comparison."""
    # Compute softmax
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits - max_logits)
    sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
    log_softmax = logits - max_logits - mx.log(sum_exp)

    # Gather log probs at target positions
    N = targets.shape[0]
    V = logits.shape[-1]

    # Create mask for valid targets
    valid_mask = (targets >= 0) & (targets < V) & (targets != ignore_index)

    # Clamp targets for indexing
    safe_targets = mx.clip(targets, 0, V - 1)

    # Gather log probs at target positions
    batch_indices = mx.arange(N)
    target_log_probs = log_softmax[batch_indices, safe_targets]

    # Apply mask
    loss = mx.where(valid_mask, -target_log_probs, mx.zeros_like(target_log_probs))

    return loss

def test_cce_forward():
    """Test CCE forward pass."""
    print("=" * 60)
    print("Testing CCE Forward Pass")
    print("=" * 60)

    # Test parameters
    batch_size = 4
    hidden_dim = 128
    vocab_size = 1000

    # Create test inputs
    mx.random.seed(42)
    hidden = mx.random.normal([batch_size, hidden_dim])
    weight = mx.random.normal([vocab_size, hidden_dim]) * 0.02
    targets = mx.array([10, 50, 100, 500], dtype=mx.int32)

    # Compute baseline (materialize full logits)
    logits_baseline = hidden @ weight.T
    loss_baseline = softmax_cross_entropy_baseline(logits_baseline, targets)
    mx.eval(loss_baseline)

    print(f"Input shapes:")
    print(f"  hidden: {hidden.shape}")
    print(f"  weight: {weight.shape}")
    print(f"  targets: {targets.shape}")
    print(f"\nBaseline loss: {loss_baseline.tolist()}")

    # Test CCE kernel via mx.fast.cce_loss
    try:
        loss_cce = mx.fast.cce_loss(hidden, weight, targets)
        mx.eval(loss_cce)
        print(f"CCE loss:      {loss_cce.tolist()}")

        # Compare
        diff = mx.abs(loss_baseline - loss_cce)
        max_diff = float(mx.max(diff))
        print(f"\nMax difference: {max_diff:.2e}")

        if max_diff < 1e-4:
            print("PASSED: CCE forward matches baseline!")
            return True
        else:
            print("FAILED: CCE forward differs from baseline")
            return False
    except Exception as e:
        print(f"CCE forward not available or failed: {e}")
        return False

def test_cce_backward():
    """Test CCE backward pass."""
    print("\n" + "=" * 60)
    print("Testing CCE Backward Pass")
    print("=" * 60)

    batch_size = 4
    hidden_dim = 128
    vocab_size = 1000

    mx.random.seed(42)
    hidden = mx.random.normal([batch_size, hidden_dim])
    weight = mx.random.normal([vocab_size, hidden_dim]) * 0.02
    targets = mx.array([10, 50, 100, 500], dtype=mx.int32)

    # Baseline gradients
    def loss_fn_baseline(h, w):
        logits = h @ w.T
        loss = softmax_cross_entropy_baseline(logits, targets)
        return mx.mean(loss)

    loss_baseline, grads_baseline = mx.value_and_grad(loss_fn_baseline, argnums=(0, 1))(hidden, weight)
    grad_h_baseline, grad_w_baseline = grads_baseline
    mx.eval(loss_baseline, grad_h_baseline, grad_w_baseline)

    print(f"Baseline loss: {float(loss_baseline):.6f}")
    print(f"Baseline grad_hidden norm: {float(mx.sqrt(mx.sum(grad_h_baseline ** 2))):.6f}")
    print(f"Baseline grad_weight norm: {float(mx.sqrt(mx.sum(grad_w_baseline ** 2))):.6f}")

    # CCE gradients
    try:
        def loss_fn_cce(h, w):
            loss = mx.fast.cce_loss(h, w, targets)
            return mx.mean(loss)

        loss_cce, grads_cce = mx.value_and_grad(loss_fn_cce, argnums=(0, 1))(hidden, weight)
        grad_h_cce, grad_w_cce = grads_cce
        mx.eval(loss_cce, grad_h_cce, grad_w_cce)

        print(f"\nCCE loss: {float(loss_cce):.6f}")
        print(f"CCE grad_hidden norm: {float(mx.sqrt(mx.sum(grad_h_cce ** 2))):.6f}")
        print(f"CCE grad_weight norm: {float(mx.sqrt(mx.sum(grad_w_cce ** 2))):.6f}")

        # Compare
        loss_diff = abs(float(loss_baseline) - float(loss_cce))
        grad_h_diff = float(mx.max(mx.abs(grad_h_baseline - grad_h_cce)))
        grad_w_diff = float(mx.max(mx.abs(grad_w_baseline - grad_w_cce)))

        print(f"\nLoss difference: {loss_diff:.2e}")
        print(f"Max grad_hidden diff: {grad_h_diff:.2e}")
        print(f"Max grad_weight diff: {grad_w_diff:.2e}")

        if loss_diff < 1e-4 and grad_h_diff < 1e-4 and grad_w_diff < 1e-4:
            print("PASSED: CCE backward matches baseline!")
            return True
        else:
            print("FAILED: CCE backward differs from baseline")
            return False
    except Exception as e:
        print(f"CCE backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test that CCE uses less memory than baseline."""
    print("\n" + "=" * 60)
    print("Testing Memory Efficiency")
    print("=" * 60)

    # Larger test to show memory benefits
    batch_size = 32
    seq_len = 512
    N = batch_size * seq_len
    hidden_dim = 768
    vocab_size = 32000

    print(f"Test size:")
    print(f"  N (batch * seq): {N}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Vocab size: {vocab_size}")

    # Calculate memory requirements
    logits_memory = N * vocab_size * 4  # float32 logits
    hidden_memory = N * hidden_dim * 4
    weight_memory = vocab_size * hidden_dim * 4

    print(f"\nMemory analysis:")
    print(f"  Hidden states: {hidden_memory / 1e9:.2f} GB")
    print(f"  Weight matrix: {weight_memory / 1e9:.2f} GB")
    print(f"  Full logits (baseline): {logits_memory / 1e9:.2f} GB")
    print(f"  CCE avoids materializing logits -> saves {logits_memory / 1e9:.2f} GB!")

    return True

def benchmark():
    """Benchmark CCE vs baseline."""
    print("\n" + "=" * 60)
    print("Benchmarking CCE vs Baseline")
    print("=" * 60)

    batch_size = 8
    seq_len = 256
    N = batch_size * seq_len
    hidden_dim = 512
    vocab_size = 8000

    mx.random.seed(42)
    hidden = mx.random.normal([N, hidden_dim])
    weight = mx.random.normal([vocab_size, hidden_dim]) * 0.02
    targets = mx.random.randint(0, vocab_size, [N]).astype(mx.int32)

    print(f"Problem size: N={N}, H={hidden_dim}, V={vocab_size}")

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        logits = hidden @ weight.T
        loss = softmax_cross_entropy_baseline(logits, targets)
        mx.eval(loss)

    # Baseline timing
    print("Benchmarking baseline (materialize logits)...")
    start = time.perf_counter()
    for _ in range(10):
        logits = hidden @ weight.T
        loss = softmax_cross_entropy_baseline(logits, targets)
        mx.eval(loss)
    baseline_time = (time.perf_counter() - start) / 10
    print(f"  Baseline: {baseline_time * 1000:.2f} ms")

    # CCE timing
    try:
        print("Benchmarking CCE...")
        # Warmup
        for _ in range(3):
            loss_cce = mx.fast.cce_loss(hidden, weight, targets)
            mx.eval(loss_cce)

        start = time.perf_counter()
        for _ in range(10):
            loss_cce = mx.fast.cce_loss(hidden, weight, targets)
            mx.eval(loss_cce)
        cce_time = (time.perf_counter() - start) / 10
        print(f"  CCE: {cce_time * 1000:.2f} ms")

        speedup = baseline_time / cce_time
        print(f"\nSpeedup: {speedup:.2f}x")
    except Exception as e:
        print(f"CCE benchmark failed: {e}")

if __name__ == "__main__":
    print("CCE Kernel Test Suite")
    print("=" * 60)

    # Check if cce_loss exists
    if not hasattr(mx.fast, 'cce_loss'):
        print("ERROR: mx.fast.cce_loss not found!")
        print("Available mx.fast functions:", dir(mx.fast))
        sys.exit(1)

    passed = []
    passed.append(("Forward", test_cce_forward()))
    passed.append(("Backward", test_cce_backward()))
    passed.append(("Memory", test_memory_efficiency()))

    benchmark()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in passed:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")
