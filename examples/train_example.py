#!/usr/bin/env python3
"""
MetalLoRA Training Example

This script demonstrates how to fine-tune a model using MetalLoRA on Apple Silicon.
It's a complete, copy-paste ready example.

Usage:
    python train_example.py --model meta-llama/Llama-3.2-1B --rank 16 --epochs 3
"""

import argparse
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

import mlx.core as mx
import mlx.nn as nn

from metal_lora.trainer import LoRATrainer, TrainingConfig, train_lora


# ============================================================================
# EXAMPLE 1: Simple Training with Trainer Class
# ============================================================================

def example_trainer_class():
    """
    Full example using the Trainer class.
    """
    print("\n" + "=" * 60)
    print("Example 1: Training with LoRATrainer")
    print("=" * 60)
    
    # Create a simple model for demonstration
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=1000, dim=256, num_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.layers = [
                TransformerBlock(dim) for _ in range(num_layers)
            ]
            self.head = nn.Linear(dim, vocab_size)
        
        def __call__(self, x):
            h = self.embed(x)
            for layer in self.layers:
                h = layer(h)
            return self.head(h)
    
    class TransformerBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = SimpleAttention(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = SimpleMLP(dim)
        
        def __call__(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x
    
    class SimpleAttention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.q_proj = nn.Linear(dim, dim)  # Will be LoRA-adapted
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)  # Will be LoRA-adapted
            self.o_proj = nn.Linear(dim, dim)
        
        def __call__(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            # Simple attention (no masking for demo)
            scores = mx.matmul(q, k.transpose(0, 2, 1)) / (q.shape[-1] ** 0.5)
            attn = mx.softmax(scores, axis=-1)
            out = mx.matmul(attn, v)
            return self.o_proj(out)
    
    class SimpleMLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.up = nn.Linear(dim, dim * 4)
            self.down = nn.Linear(dim * 4, dim)
            self.act = nn.GELU()
        
        def __call__(self, x):
            return self.down(self.act(self.up(x)))
    
    # Create model
    model = SimpleTransformer(vocab_size=1000, dim=256, num_layers=2)
    
    # Create training config
    config = TrainingConfig(
        rank=8,
        alpha=16.0,
        target_modules=["q_proj", "v_proj"],  # Only adapt attention Q and V
        learning_rate=1e-4,
        batch_size=2,
        log_interval=5,
        output_dir="./lora_demo_output",
    )
    
    # Create trainer
    trainer = LoRATrainer(model, config)
    
    print(f"Model trainable params: {trainer.model.num_trainable_params():,}")
    print(f"Model total params: {trainer.model.num_total_params():,}")
    print(f"Reduction: {100 * (1 - trainer.model.num_trainable_params() / trainer.model.num_total_params()):.1f}%")
    
    # Create dummy training data
    def generate_dummy_data(num_batches=20):
        for _ in range(num_batches):
            batch_size = 2
            seq_len = 32
            yield {
                "input_ids": mx.random.randint(0, 999, (batch_size, seq_len)),
                "labels": mx.random.randint(0, 999, (batch_size, seq_len)),
            }
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        train_data=generate_dummy_data(num_batches=20),
        num_epochs=1,
    )
    
    print(f"\nFinal loss: {history['loss'][-1]:.4f}")
    print("Training complete!")


# ============================================================================
# EXAMPLE 2: One-Liner Training
# ============================================================================

def example_one_liner():
    """
    Minimal one-liner training example.
    """
    print("\n" + "=" * 60)
    print("Example 2: One-Liner Training")
    print("=" * 60)
    
    # Simple linear model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.GELU(),
        nn.Linear(128, 64),
    )
    
    # Generate dummy data
    def data_gen():
        for _ in range(10):
            x = mx.random.normal((4, 16, 64))
            yield {"input_ids": x, "target_output": x}  # Auto-encoder style
    
    print("Training with train_lora() one-liner...")
    
    # One-liner training - not running to avoid file creation
    print("(Skipping actual training to avoid file output)")
    print("Usage: trainer = train_lora(model, data, rank=8, num_epochs=3)")


# ============================================================================
# EXAMPLE 3: Custom Training Loop
# ============================================================================

def example_custom_loop():
    """
    Custom training loop for maximum control.
    """
    print("\n" + "=" * 60)
    print("Example 3: Custom Training Loop")
    print("=" * 60)
    
    from metal_lora import LoRALinear
    import mlx.optimizers as optim
    
    # Create model with LoRA layers
    class LoRAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = LoRALinear(64, 128, rank=8)
            self.layer2 = LoRALinear(128, 64, rank=8)
            self.act = nn.GELU()
        
        def __call__(self, x):
            x = self.act(self.layer1(x))
            return self.layer2(x)
    
    model = LoRAModel()
    optimizer = optim.AdamW(learning_rate=1e-3)
    
    def loss_fn(model, x, target):
        pred = model(x)
        return mx.mean((pred - target) ** 2)
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    print("Running custom training loop...")
    
    for step in range(5):
        # Generate batch
        x = mx.random.normal((4, 16, 64))
        target = mx.random.normal((4, 16, 64))
        
        # Forward + backward
        loss, grads = loss_and_grad(model, x, target)
        
        # Update (only LoRA params are trainable)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        
        print(f"  Step {step + 1}: Loss = {loss.item():.4f}")
    
    print("Custom training complete!")
    
    # Merge weights for inference
    print("\nMerging LoRA weights for inference...")
    model.layer1.to_inference_mode()
    model.layer2.to_inference_mode()
    print("Weights merged!")


# ============================================================================
# EXAMPLE 4: Training with Gradient Accumulation
# ============================================================================

def example_gradient_accumulation():
    """
    Training with gradient accumulation for larger effective batch size.
    """
    print("\n" + "=" * 60)
    print("Example 4: Gradient Accumulation")
    print("=" * 60)
    
    from metal_lora import LoRALinear
    import mlx.optimizers as optim
    
    model = nn.Sequential(
        LoRALinear(64, 64, rank=4),
    )
    
    optimizer = optim.AdamW(learning_rate=1e-3)
    
    accumulation_steps = 4
    
    print(f"Effective batch size: {2 * accumulation_steps} = 8")
    print("Running with gradient accumulation...")
    
    accumulated_grads = None
    
    for step in range(8):
        x = mx.random.normal((2, 16, 64))  # Small batch
        target = mx.random.normal((2, 16, 64))
        
        def loss_fn(model):
            pred = model(x)
            return mx.mean((pred - target) ** 2)
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        
        # Accumulate gradients
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = mx.tree_map(
                lambda a, g: a + g,
                accumulated_grads,
                grads
            )
        
        # Update every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            # Average gradients
            avg_grads = mx.tree_map(
                lambda g: g / accumulation_steps,
                accumulated_grads
            )
            
            optimizer.update(model, avg_grads)
            mx.eval(model.parameters())
            
            print(f"  Update at step {step + 1}: Loss = {loss.item():.4f}")
            accumulated_grads = None
    
    print("Gradient accumulation complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MetalLoRA Training Examples")
    parser.add_argument("--example", type=int, default=0,
                        help="Run specific example (1-4), 0 for all")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MetalLoRA Training Examples")
    print("=" * 60)
    
    mx.random.seed(42)
    
    examples = [
        example_trainer_class,
        example_one_liner,
        example_custom_loop,
        example_gradient_accumulation,
    ]
    
    if args.example == 0:
        for example in examples:
            try:
                example()
            except Exception as e:
                print(f"Error in example: {e}")
    else:
        if 1 <= args.example <= len(examples):
            examples[args.example - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
