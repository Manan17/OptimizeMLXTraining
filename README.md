# MetalLoRA

Optimized LoRA kernels for MLX on Apple Silicon.

## Requirements

- macOS 13+
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX

## Installation

```bash
pip install -e .
```

## Usage

### Training

```python
from metal_lora import LoRATrainer, TrainingConfig

trainer = LoRATrainer(model, config=TrainingConfig(rank=16))
trainer.train(train_data, num_epochs=3)
trainer.save("./adapter")
```

### Inference

```python
from metal_lora import LoRALinear

layer = LoRALinear(4096, 4096, rank=16, alpha=32)
output = layer(x)

# Merge for deployment
layer.to_inference_mode()
```

### Apply to Existing Model

```python
from metal_lora import apply_lora_to_model

model = apply_lora_to_model(base_model, rank=16, target_modules=["q_proj", "v_proj"])
```

## API

| Class            | Description                       |
| ---------------- | --------------------------------- |
| `LoRALinear`     | Linear layer with LoRA adaptation |
| `LoRATrainer`    | Training wrapper with optimizer   |
| `TrainingConfig` | Training hyperparameters          |

| Function              | Description                  |
| --------------------- | ---------------------------- |
| `apply_lora_to_model` | Add LoRA to existing model   |
| `save_adapter`        | Save LoRA weights            |
| `load_adapter`        | Load LoRA weights            |
| `merge_lora_weights`  | Merge LoRA into base weights |

## Kernels

| File                   | Contents                            |
| ---------------------- | ----------------------------------- |
| `lora_kernels.metal`   | Forward, backward, merge operations |
| `lora_train.metal`     | Fused training kernels              |
| `lora_quantized.metal` | 4-bit/8-bit quantized inference     |

## License

MIT
