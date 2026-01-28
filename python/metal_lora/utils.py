"""Utility functions for LoRA adapters."""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Optional, List, Dict
import json


def save_adapter(
    model: nn.Module,
    path: str,
    adapter_name: str = "default",
) -> None:
    """Save LoRA adapter weights."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    lora_params = {}
    for name, module in model.named_modules():
        if hasattr(module, "A") and hasattr(module, "B"):
            lora_params[f"{name}.A"] = module.A
            lora_params[f"{name}.B"] = module.B
    
    mx.save(str(path / f"{adapter_name}.safetensors"), lora_params)
    
    config = {
        "adapter_name": adapter_name,
        "num_params": sum(p.size for p in lora_params.values()),
    }
    with open(path / f"{adapter_name}_config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_adapter(
    model: nn.Module,
    path: str,
    adapter_name: str = "default",
) -> nn.Module:
    """Load LoRA adapter weights into model."""
    path = Path(path)
    lora_params = mx.load(str(path / f"{adapter_name}.safetensors"))
    
    for name, module in model.named_modules():
        a_key = f"{name}.A"
        b_key = f"{name}.B"
        if a_key in lora_params and b_key in lora_params:
            module.A = lora_params[a_key]
            module.B = lora_params[b_key]
    
    return model


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Replace target linear layers with LoRA layers."""
    from .layers import LoRALinear
    
    target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def should_apply(name: str) -> bool:
        return any(target in name for target in target_modules)
    
    def replace_linear(module: nn.Module, prefix: str = ""):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            
            if isinstance(child, nn.Linear) and should_apply(full_name):
                lora_layer = LoRALinear.from_linear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
            else:
                replace_linear(child, full_name)
    
    replace_linear(model)
    return model


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    trainable = 0
    total = 0
    lora_layers = 0
    
    for name, module in model.named_modules():
        if hasattr(module, "num_trainable_params"):
            trainable += module.num_trainable_params()
            total += module.num_total_params()
            lora_layers += 1
    
    return {
        "trainable": trainable,
        "total": total,
        "lora_layers": lora_layers,
        "ratio": trainable / total if total > 0 else 0,
    }


def merge_all_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into base weights."""
    for name, module in model.named_modules():
        if hasattr(module, "to_inference_mode"):
            module.to_inference_mode()
    return model


def get_lora_summary(model: nn.Module) -> str:
    """Get summary of LoRA configuration."""
    stats = count_lora_parameters(model)
    lines = [
        f"LoRA layers: {stats['lora_layers']}",
        f"Trainable: {stats['trainable']:,}",
        f"Total: {stats['total']:,}",
        f"Ratio: {stats['ratio']:.2%}",
    ]
    return "\n".join(lines)
