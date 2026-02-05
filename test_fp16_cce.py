#!/usr/bin/env python3
"""
Test script to verify float16 CCE support.
This tests that CCE works with float16 inputs without requiring conversion to bfloat16.
"""

import mlx.core as mx
import mlx.nn as nn

def test_cce_dtypes():
    """Test CCE with different dtypes."""
    print("Testing CCE with different dtypes...")

    # Test parameters
    N = 512  # Must be >= 256 for GPU CCE
    H = 256
    V = 32000

    # Create test data
    mx.random.seed(42)

    for dtype_name, dtype in [("float32", mx.float32), ("bfloat16", mx.bfloat16), ("float16", mx.float16)]:
        print(f"\n--- Testing {dtype_name} ---")

        # Generate random data in float32 first for consistency
        hidden_f32 = mx.random.normal((N, H))
        weight_f32 = mx.random.normal((V, H)) * 0.01
        targets = mx.random.randint(0, V, (N,))

        # Convert to test dtype
        hidden = hidden_f32.astype(dtype)
        weight = weight_f32.astype(dtype)

        print(f"  hidden dtype: {hidden.dtype}, shape: {hidden.shape}")
        print(f"  weight dtype: {weight.dtype}, shape: {weight.shape}")

        try:
            # Test CCE
            loss = mx.fast.cce_loss(hidden, weight, targets)
            mx.eval(loss)
            print(f"  CCE loss: {loss.mean().item():.6f}")
            print(f"  SUCCESS: {dtype_name} CCE works!")
        except Exception as e:
            print(f"  FAILED: {e}")

        # Compare with baseline
        print(f"  Computing baseline for comparison...")
        logits_f32 = hidden_f32 @ weight_f32.T
        baseline_loss = nn.losses.cross_entropy(logits_f32, targets, reduction="none")
        mx.eval(baseline_loss)
        print(f"  Baseline loss: {baseline_loss.mean().item():.6f}")

def test_cce_backward():
    """Test CCE backward pass with different dtypes."""
    print("\n\nTesting CCE backward with different dtypes...")

    N = 512
    H = 256
    V = 32000

    mx.random.seed(42)

    for dtype_name, dtype in [("float32", mx.float32), ("bfloat16", mx.bfloat16), ("float16", mx.float16)]:
        print(f"\n--- Testing {dtype_name} backward ---")

        hidden = mx.random.normal((N, H)).astype(dtype)
        weight = mx.random.normal((V, H)).astype(dtype) * 0.01
        targets = mx.random.randint(0, V, (N,))

        def loss_fn(h, w):
            loss = mx.fast.cce_loss(h, w, targets)
            return loss.mean()

        try:
            # Compute gradients
            loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(hidden, weight)
            grad_hidden, grad_weight = grads
            mx.eval(loss, grad_hidden, grad_weight)

            print(f"  Loss: {loss.item():.6f}")
            print(f"  grad_hidden shape: {grad_hidden.shape}, dtype: {grad_hidden.dtype}")
            print(f"  grad_weight shape: {grad_weight.shape}, dtype: {grad_weight.dtype}")
            print(f"  grad_hidden norm: {mx.sqrt((grad_hidden ** 2).sum()).item():.6f}")
            print(f"  grad_weight norm: {mx.sqrt((grad_weight ** 2).sum()).item():.6f}")
            print(f"  SUCCESS: {dtype_name} backward works!")
        except Exception as e:
            print(f"  FAILED: {e}")

if __name__ == "__main__":
    test_cce_dtypes()
    test_cce_backward()
    print("\n\nAll tests completed!")
