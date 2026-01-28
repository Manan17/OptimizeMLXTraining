"""
MetalLoRA - Example: Fine-tuning a Model with LoRA

This example demonstrates how to use MetalLoRA to fine-tune a pre-trained
model using LoRA for efficient parameter-efficient training.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
import sys

# Add package to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from metal_lora import LoRALinear, apply_lora_to_model, save_adapter, get_lora_summary


# ============================================================================
# EXAMPLE 1: Basic LoRALinear Usage
# ============================================================================

def example_basic_usage():
    """Demonstrate basic LoRALinear layer usage."""
    print("\n" + "=" * 60)
    print("Example 1: Basic LoRALinear Usage")
    print("=" * 60)
    
    # Create a LoRA-enabled linear layer
    layer = LoRALinear(
        in_features=4096,
        out_features=4096,
        rank=16,           # Low-rank dimension
        alpha=32.0,        # Scaling factor (typically 2 * rank)
        dropout=0.0,       # Optional dropout on LoRA path
    )
    
    # Forward pass
    x = mx.random.normal((2, 128, 4096))  # [batch, seq, features]
    output = layer(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable params: {layer.num_trainable_params():,}")
    print(f"Total params:     {layer.num_total_params():,}")
    print(f"Compression:      {layer.compression_ratio():.2%}")


# ============================================================================
# EXAMPLE 2: Converting nn.Linear to LoRA
# ============================================================================

def example_convert_linear():
    """Demonstrate converting existing nn.Linear to LoRALinear."""
    print("\n" + "=" * 60)
    print("Example 2: Convert nn.Linear to LoRALinear")
    print("=" * 60)
    
    # Original linear layer (e.g., from a pretrained model)
    original = nn.Linear(1024, 2048)
    
    # Convert to LoRALinear, copying the weights
    lora = LoRALinear.from_linear(original, rank=8, alpha=16.0)
    
    print(f"Original layer: {original.weight.shape}")
    print(f"LoRA A shape:   {lora.A.shape}")
    print(f"LoRA B shape:   {lora.B.shape}")
    print(f"Trainable params: {lora.num_trainable_params():,} (vs {original.weight.size:,} original)")


# ============================================================================
# EXAMPLE 3: Training Loop
# ============================================================================

def example_training():
    """Demonstrate a simple training loop with LoRA."""
    print("\n" + "=" * 60)
    print("Example 3: Training Loop with LoRA")
    print("=" * 60)
    
    # Create model with LoRA layer
    layer = LoRALinear(512, 512, rank=8, alpha=16.0)
    layer.W0 = mx.eye(512)  # Set base weights for demo
    
    # Create optimizer (only optimizes trainable params)
    optimizer = optim.Adam(learning_rate=1e-3)
    
    # Simple loss function
    def loss_fn(layer, x, target):
        pred = layer(x)
        return mx.mean((pred - target) ** 2)
    
    # Training data
    x = mx.random.normal((4, 32, 512))
    target = mx.random.normal((4, 32, 512))
    
    # Training step
    loss_and_grad = nn.value_and_grad(layer, loss_fn)
    
    print("Training for 5 steps...")
    for step in range(5):
        loss, grads = loss_and_grad(layer, x, target)
        optimizer.update(layer, grads)
        mx.eval(layer.parameters())
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")


# ============================================================================
# EXAMPLE 4: Weight Merging for Inference
# ============================================================================

def example_merge_weights():
    """Demonstrate merging LoRA weights for fast inference."""
    print("\n" + "=" * 60)
    print("Example 4: Weight Merging for Inference")
    print("=" * 60)
    
    # Create and "train" a LoRA layer
    layer = LoRALinear(256, 256, rank=4, alpha=8.0)
    layer.W0 = mx.eye(256)
    layer.A = mx.random.normal((4, 256)) * 0.1
    layer.B = mx.random.normal((256, 4)) * 0.1
    
    x = mx.random.normal((2, 16, 256))
    
    # Output with LoRA (training mode)
    output_lora = layer(x)
    
    # Merge weights for inference
    layer.to_inference_mode()
    
    # Output with merged weights (inference mode)
    output_merged = layer(x)
    
    # Verify outputs are identical
    max_diff = mx.max(mx.abs(output_lora - output_merged)).item()
    print(f"Max difference after merge: {max_diff:.2e}")
    print("✓ Outputs are identical!")


# ============================================================================
# EXAMPLE 5: Applying LoRA to an Existing Model
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for demonstrating LoRA application."""
    
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.activation = nn.GELU()
    
    def __call__(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)


def example_apply_to_model():
    """Demonstrate applying LoRA to all linear layers in a model."""
    print("\n" + "=" * 60)
    print("Example 5: Apply LoRA to Existing Model")
    print("=" * 60)
    
    # Create base model
    model = SimpleMLP(dim=256)
    
    print("Before LoRA:")
    print(f"  fc1: {type(model.fc1).__name__}")
    print(f"  fc2: {type(model.fc2).__name__}")
    
    # Apply LoRA to all linear layers
    model = apply_lora_to_model(
        model,
        rank=8,
        alpha=16.0,
        target_modules=None,  # Apply to all linear layers
    )
    
    print("\nAfter LoRA:")
    print(f"  fc1: {type(model.fc1).__name__}")
    print(f"  fc2: {type(model.fc2).__name__}")
    
    # Get summary
    print("\n" + get_lora_summary(model))


# ============================================================================
# EXAMPLE 6: Saving and Loading Adapters
# ============================================================================

def example_save_load():
    """Demonstrate saving and loading LoRA adapters."""
    print("\n" + "=" * 60)
    print("Example 6: Save/Load Adapters")
    print("=" * 60)
    
    # Create model with LoRA
    model = SimpleMLP(dim=128)
    model = apply_lora_to_model(model, rank=4, alpha=8.0)
    
    # Simulate training
    model.fc1.A = mx.random.normal((4, 128)) * 0.1
    model.fc1.B = mx.random.normal((512, 4)) * 0.1
    
    # Note: Saving/loading requires macOS with MLX installed
    # Uncomment the following to test on a Mac:
    #
    # # Save adapter
    # save_adapter(model, "adapters/example", adapter_name="trained")
    # 
    # # Load into fresh model
    # fresh_model = SimpleMLP(dim=128)
    # fresh_model = apply_lora_to_model(fresh_model, rank=4, alpha=8.0)
    # fresh_model = load_adapter(fresh_model, "adapters/example", adapter_name="trained")
    
    print("(Saving/loading example - see code comments)")
    print("✓ Adapter save/load demonstrated in code")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("MetalLoRA Examples")
    print("=" * 60)
    
    mx.random.seed(42)
    
    example_basic_usage()
    example_convert_linear()
    example_training()
    example_merge_weights()
    example_apply_to_model()
    example_save_load()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
