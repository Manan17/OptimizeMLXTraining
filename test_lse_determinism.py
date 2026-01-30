"""Test if the logsumexp computation is deterministic"""
import mlx.core as mx
import mlx.nn as nn

# Create fixed FP32 inputs
mx.random.seed(42)
N, V, H = 720, 151936, 1024

h = mx.random.normal((N, H)).astype(mx.float32)
w = mx.random.normal((V, H)).astype(mx.float32)
t = mx.random.randint(0, V, (N,))
mx.eval(h, w, t)

print(f"Shape: N={N}, V={V}, H={H}")
print()

# Test CCE forward multiple times
print("=== CCE Forward (5 runs) ===")
losses = []
for i in range(5):
    loss = mx.fast.cce_loss(h, w, t).sum()
    mx.eval(loss)
    losses.append(loss.item())
    print(f"  Run {i+1}: {losses[-1]:.10f}")

if len(set(losses)) == 1:
    print("CCE Forward: DETERMINISTIC")
else:
    print(f"CCE Forward: NON-DETERMINISTIC (range: {max(losses) - min(losses):.10f})")

# Compare to baseline
print()
print("=== Baseline (nn.cross_entropy) ===")
logits = h @ w.T
mx.eval(logits)
baseline = nn.losses.cross_entropy(logits, t).sum()
mx.eval(baseline)
print(f"Baseline: {baseline.item():.10f}")
print(f"CCE mean: {sum(losses)/len(losses):.10f}")
print(f"Diff: {abs(baseline.item() - sum(losses)/len(losses)):.10f}")

# Test with explicit sync
print()
print("=== CCE with mx.synchronize() ===")
losses2 = []
for i in range(5):
    loss = mx.fast.cce_loss(h, w, t).sum()
    mx.eval(loss)
    mx.synchronize()  # Force GPU sync
    losses2.append(loss.item())
    print(f"  Run {i+1}: {losses2[-1]:.10f}")

if len(set(losses2)) == 1:
    print("With sync: DETERMINISTIC")
else:
    print(f"With sync: NON-DETERMINISTIC (range: {max(losses2) - min(losses2):.10f})")
