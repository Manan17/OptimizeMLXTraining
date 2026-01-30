"""Simple BF16 training test without masking"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx.utils import tree_map

# Load model (keep BF16)
print("Loading model...")
model, tok = load('mlx-community/Qwen3-0.6B-bf16')
mx.eval(model.parameters())
print(f'Model dtype: {model.model.embed_tokens.weight.dtype}')

# Create batch
tokens = tok.encode('Hello world. ' * 50)[:128]
batch = mx.array([tokens] * 8, dtype=mx.int32)
print(f'Batch shape: {batch.shape}')

inputs = batch[:, :-1]
targets = batch[:, 1:]
print(f'N = {inputs.shape[0] * inputs.shape[1]}')

# Save initial weights
initial_weights = tree_map(lambda x: mx.array(x), model.parameters())
mx.eval(initial_weights)

# Baseline loss (no mask)
def baseline_loss(model, inputs, targets):
    logits = model(inputs)
    return nn.losses.cross_entropy(logits, targets).mean()

# CCE loss (no mask)
def cce_loss(model, inputs, targets):
    hidden = model.model(inputs)
    weight = model.model.embed_tokens.weight
    B, S, H = hidden.shape
    return mx.fast.cce_loss(hidden.reshape(B*S, H), weight, targets.reshape(B*S)).mean()

# Train with baseline
print('\n=== BASELINE (10 steps) ===')
model.update(initial_weights)
mx.eval(model.parameters())
opt = optim.Adam(learning_rate=1e-5)
lvg = nn.value_and_grad(model, baseline_loss)
baseline_losses = []
for i in range(10):
    l, g = lvg(model, inputs, targets)
    g = tree_map(lambda x: mx.clip(x, -1.0, 1.0), g)
    opt.update(model, g)
    mx.eval(model.parameters(), opt.state, l)
    baseline_losses.append(l.item())
    print(f'  Step {i+1}: loss = {l.item():.6f}')

# Train with CCE
print('\n=== CCE (10 steps) ===')
model.update(initial_weights)
mx.eval(model.parameters())
opt = optim.Adam(learning_rate=1e-5)
lvg = nn.value_and_grad(model, cce_loss)
cce_losses = []
for i in range(10):
    l, g = lvg(model, inputs, targets)
    g = tree_map(lambda x: mx.clip(x, -1.0, 1.0), g)
    opt.update(model, g)
    mx.eval(model.parameters(), opt.state, l)
    cce_losses.append(l.item())
    print(f'  Step {i+1}: loss = {l.item():.6f}')

# Compare
print('\n=== COMPARISON ===')
for i in range(10):
    diff = abs(baseline_losses[i] - cce_losses[i]) / baseline_losses[i] * 100
    print(f'  Step {i+1}: baseline={baseline_losses[i]:.6f}, cce={cce_losses[i]:.6f}, diff={diff:.2f}%')

# Check if CCE is training in right direction
if cce_losses[-1] < cce_losses[0]:
    print('\nCCE: Loss DECREASING (good!)')
else:
    print('\nCCE: Loss INCREASING (bad - gradients wrong!)')
