"""Test if CCE is deterministic - run same computation multiple times"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

print("Loading model...")
model, tok = load('mlx-community/Qwen3-0.6B-bf16')

# Convert to FP32 for testing
from mlx.utils import tree_map
def to_fp32(x):
    if isinstance(x, mx.array) and x.dtype in [mx.bfloat16, mx.float16]:
        return x.astype(mx.float32)
    return x
model.update(tree_map(to_fp32, model.parameters()))
mx.eval(model.parameters())

# Create fixed inputs
tokens = tok.encode('Hello world. ' * 30)[:128]
batch = mx.array([tokens] * 8, dtype=mx.int32)
inputs = batch[:, :-1]
targets = batch[:, 1:]

# Get hidden and weight
hidden = model.model(inputs)
weight = model.model.embed_tokens.weight
mx.eval(hidden, weight)

B, S, H = hidden.shape
h_flat = hidden.reshape(B*S, H)
t_flat = targets.reshape(B*S)

print(f"Shape: N={B*S}, V={weight.shape[0]}, H={H}")
print()

# Test forward determinism
print("=== Forward Determinism ===")
losses = []
for i in range(5):
    loss = mx.fast.cce_loss(h_flat, weight, t_flat)
    mx.eval(loss)
    losses.append(loss.mean().item())
    print(f"  Run {i+1}: loss mean = {losses[-1]:.10f}")

if all(l == losses[0] for l in losses):
    print("Forward: DETERMINISTIC")
else:
    print(f"Forward: NON-DETERMINISTIC (range: {min(losses):.10f} to {max(losses):.10f})")

# Test backward determinism
print()
print("=== Backward Determinism ===")

def cce_loss_fn(h, w, t):
    return mx.fast.cce_loss(h, w, t).mean()

grad_h_results = []
grad_w_results = []
for i in range(5):
    grad_h, grad_w = mx.grad(cce_loss_fn, argnums=(0,1))(h_flat, weight, t_flat)
    mx.eval(grad_h, grad_w)
    grad_h_results.append(grad_h.sum().item())
    grad_w_results.append(grad_w.sum().item())
    print(f"  Run {i+1}: grad_h sum = {grad_h_results[-1]:.10f}, grad_w sum = {grad_w_results[-1]:.10f}")

if all(g == grad_h_results[0] for g in grad_h_results):
    print("Backward grad_h: DETERMINISTIC")
else:
    print(f"Backward grad_h: NON-DETERMINISTIC (range: {min(grad_h_results):.10f} to {max(grad_h_results):.10f})")

if all(g == grad_w_results[0] for g in grad_w_results):
    print("Backward grad_w: DETERMINISTIC")
else:
    print(f"Backward grad_w: NON-DETERMINISTIC (range: {min(grad_w_results):.10f} to {max(grad_w_results):.10f})")

# Compare to baseline
print()
print("=== Comparison to Baseline ===")
def baseline_loss_fn(h, w, t):
    logits = h @ w.T
    return nn.losses.cross_entropy(logits, t).mean()

baseline_loss = baseline_loss_fn(h_flat, weight, t_flat)
cce_loss_val = cce_loss_fn(h_flat, weight, t_flat)
mx.eval(baseline_loss, cce_loss_val)
print(f"Baseline loss: {baseline_loss.item():.10f}")
print(f"CCE loss:      {cce_loss_val.item():.10f}")
print(f"Diff:          {abs(baseline_loss.item() - cce_loss_val.item()):.10f}")
