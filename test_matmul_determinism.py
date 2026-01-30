"""Test if steel_matmul is deterministic"""
import mlx.core as mx

mx.random.seed(42)
N, V, H = 720, 16384, 1024  # Single chunk size

h = mx.random.normal((N, H)).astype(mx.float32)
w = mx.random.normal((V, H)).astype(mx.float32)
mx.eval(h, w)

print(f"Shape: N={N}, V={V}, H={H}")
print()

# Test matmul determinism
print("=== Matmul (h @ w.T) Determinism ===")
sums = []
for i in range(5):
    logits = h @ w.T
    mx.eval(logits)
    sums.append(logits.sum().item())
    print(f"  Run {i+1}: sum = {sums[-1]:.10f}")

if len(set(sums)) == 1:
    print("Matmul: DETERMINISTIC")
else:
    print(f"Matmul: NON-DETERMINISTIC (range: {max(sums) - min(sums):.10f})")

# Test logsumexp determinism on same logits
print()
print("=== LogSumExp on Fixed Logits ===")
logits = h @ w.T
mx.eval(logits)

lse_results = []
for i in range(5):
    max_logits = mx.max(logits, axis=1, keepdims=True)
    lse = max_logits.squeeze() + mx.log(mx.sum(mx.exp(logits - max_logits), axis=1))
    mx.eval(lse)
    lse_results.append(lse.sum().item())
    print(f"  Run {i+1}: sum = {lse_results[-1]:.10f}")

if len(set(lse_results)) == 1:
    print("LogSumExp: DETERMINISTIC")
else:
    print(f"LogSumExp: NON-DETERMINISTIC (range: {max(lse_results) - min(lse_results):.10f})")

# Test CCE on same inputs
print()
print("=== CCE on Fixed Inputs ===")
t = mx.random.randint(0, V, (N,))
mx.eval(t)

cce_results = []
for i in range(5):
    loss = mx.fast.cce_loss(h, w, t).sum()
    mx.eval(loss)
    cce_results.append(loss.item())
    print(f"  Run {i+1}: sum = {cce_results[-1]:.10f}")

if len(set(cce_results)) == 1:
    print("CCE: DETERMINISTIC")
else:
    print(f"CCE: NON-DETERMINISTIC (range: {max(cce_results) - min(cce_results):.10f})")
