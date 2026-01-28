"""LoRA training infrastructure."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Optional, Dict, List, Iterator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import json


@dataclass
class TrainingConfig:
    """Training configuration."""
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"
    
    use_fp16: bool = True
    gradient_checkpointing: bool = False
    
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    output_dir: str = "./lora_output"


@dataclass
class TrainingState:
    """Training state tracker."""
    global_step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    train_loss: float = 0.0
    learning_rate: float = 0.0
    tokens_seen: int = 0
    start_time: float = 0.0


class LoRATrainableModule(nn.Module):
    """Wrapper that makes a model LoRA-trainable."""
    
    def __init__(
        self,
        base_model: nn.Module,
        rank: int = 16,
        alpha: float = 32.0,
        target_modules: Optional[List[str]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        self.lora_A: Dict[str, mx.array] = {}
        self.lora_B: Dict[str, mx.array] = {}
        
        self._apply_lora()
    
    def _apply_lora(self):
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]
                    
                    self.lora_A[name] = mx.random.normal((self.rank, in_features)).astype(mx.float16) * 0.02
                    self.lora_B[name] = mx.zeros((out_features, self.rank)).astype(mx.float16)
                    module.freeze()
    
    def get_lora_params(self) -> Dict[str, mx.array]:
        params = {}
        for name in self.lora_A:
            params[f"{name}.lora_A"] = self.lora_A[name]
            params[f"{name}.lora_B"] = self.lora_B[name]
        return params
    
    def set_lora_params(self, params: Dict[str, mx.array]):
        for key, value in params.items():
            if key.endswith(".lora_A"):
                self.lora_A[key[:-7]] = value
            elif key.endswith(".lora_B"):
                self.lora_B[key[:-7]] = value
    
    def num_trainable_params(self) -> int:
        return sum(a.size + self.lora_B[name].size for name, a in self.lora_A.items())
    
    def num_total_params(self) -> int:
        return sum(p.size for p in self.base_model.parameters()) + self.num_trainable_params()
    
    def __call__(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


class LoRATrainer:
    """LoRA trainer for Apple Silicon."""
    
    def __init__(self, model: nn.Module, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        
        self.model = LoRATrainableModule(
            model,
            rank=self.config.rank,
            alpha=self.config.alpha,
            target_modules=self.config.target_modules,
            dropout=self.config.dropout,
        )
        
        self.optimizer = self._create_optimizer()
        self.state = TrainingState()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        if self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                learning_rate=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adam":
            return optim.Adam(
                learning_rate=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
        else:
            return optim.SGD(learning_rate=self.config.learning_rate)
    
    def _compute_loss(self, model: LoRATrainableModule, batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict[str, float]]:
        outputs = model(batch["input_ids"])
        
        if "labels" in batch:
            logits = outputs[:, :-1, :]
            labels = batch["labels"][:, 1:]
            loss = mx.mean(nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)))
        else:
            loss = mx.mean((outputs - batch.get("target_output", outputs)) ** 2)
        
        return loss, {"loss": float(loss)}
    
    def _training_step(self, batch: Dict[str, mx.array]) -> Dict[str, float]:
        def loss_fn(params):
            self.model.set_lora_params(params)
            loss, _ = self._compute_loss(self.model, batch)
            return loss
        
        params = self.model.get_lora_params()
        loss, grads = mx.value_and_grad(loss_fn)(params)
        
        if self.config.max_grad_norm > 0:
            grad_norm = mx.sqrt(sum(mx.sum(g ** 2) for g in grads.values()))
            scale = mx.minimum(1.0, self.config.max_grad_norm / (grad_norm + 1e-6))
            grads = {k: g * scale for k, g in grads.items()}
        
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())
        self.state.global_step += 1
        
        return {"loss": float(loss)}
    
    def train(
        self,
        train_data: Iterator[Dict[str, mx.array]],
        num_epochs: int = 1,
        num_steps: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        self.state.start_time = time.time()
        history = {"loss": [], "step": []}
        
        print(f"Trainable: {self.model.num_trainable_params():,} | Total: {self.model.num_total_params():,}")
        
        step = 0
        for epoch in range(num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch in train_data:
                metrics = self._training_step(batch)
                epoch_loss += metrics["loss"]
                epoch_steps += 1
                step += 1
                
                if step % self.config.log_interval == 0:
                    print(f"Step {step} | Loss: {metrics['loss']:.4f}")
                    history["loss"].append(metrics["loss"])
                    history["step"].append(step)
                
                if step % self.config.save_interval == 0:
                    self.save(f"{self.config.output_dir}/checkpoint-{step}")
                
                if num_steps and step >= num_steps:
                    break
            
            if num_steps and step >= num_steps:
                break
        
        self.save(f"{self.config.output_dir}/final")
        return history
    
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        mx.save(str(path / "adapter.safetensors"), self.model.get_lora_params())
        
        with open(path / "config.json", "w") as f:
            json.dump({"rank": self.config.rank, "alpha": self.config.alpha}, f)
    
    def load(self, path: str):
        path = Path(path)
        params = mx.load(str(path / "adapter.safetensors"))
        self.model.set_lora_params(params)


def train_lora(
    model: nn.Module,
    train_data: Iterator[Dict[str, mx.array]],
    rank: int = 16,
    alpha: float = 32.0,
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    output_dir: str = "./lora_output",
) -> LoRATrainer:
    """One-liner LoRA training."""
    config = TrainingConfig(rank=rank, alpha=alpha, learning_rate=learning_rate, output_dir=output_dir)
    trainer = LoRATrainer(model, config)
    trainer.train(train_data, num_epochs=num_epochs)
    return trainer


def quick_finetune(model: nn.Module, texts: List[str], tokenizer, rank: int = 8, num_steps: int = 100) -> LoRATrainer:
    """Quick fine-tuning on text data."""
    def data_gen():
        for i in range(0, len(texts), 4):
            tokens = tokenizer(texts[i:i+4], padding=True, return_tensors="np")
            yield {"input_ids": mx.array(tokens["input_ids"]), "labels": mx.array(tokens["input_ids"])}
    
    config = TrainingConfig(rank=rank)
    trainer = LoRATrainer(model, config)
    trainer.train(data_gen(), num_steps=num_steps)
    return trainer
