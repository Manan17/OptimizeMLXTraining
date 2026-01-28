"""LoRA layers for MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict
import math

from .ops import lora_forward, lora_backward_efficient, merge_lora_weights
from .exceptions import validate_rank, validate_alpha, validate_probability, ConfigurationError
from .logging import logger


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    h = W0 @ x + (alpha/rank) * B @ A @ x

    Parameters
    ----------
    in_features : int
        Input dimension.
    out_features : int
        Output dimension.
    rank : int
        LoRA rank (1-256).
    alpha : float
        Scaling factor.
    dropout : float
        Dropout probability (0-1).
    use_bias : bool
        Include bias term.
    freeze_base : bool
        Freeze base weights.

    Raises
    ------
    ConfigurationError
        If parameters are invalid.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_bias: bool = False,
        freeze_base: bool = True,
    ):
        super().__init__()

        # Validation
        if in_features <= 0:
            raise ConfigurationError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ConfigurationError(f"out_features must be positive, got {out_features}")
        validate_rank(rank)
        validate_alpha(alpha)
        validate_probability(dropout, "dropout")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.freeze_base = freeze_base

        self.W0 = mx.zeros((out_features, in_features))

        bound = 1.0 / math.sqrt(in_features)
        self.A = mx.random.uniform(-bound, bound, (rank, in_features))
        self.B = mx.zeros((out_features, rank))

        self.bias = mx.zeros((out_features,)) if use_bias else None

        if freeze_base:
            self.W0 = mx.stop_gradient(self.W0)

        logger.debug(f"LoRALinear: {in_features}→{out_features}, rank={rank}")
    
    def __call__(self, x: mx.array) -> mx.array:
        original_shape = x.shape
        
        if x.ndim == 2:
            x = x[None, :, :]
        elif x.ndim > 3:
            batch_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
        
        out = lora_forward(
            x=x, W0=self.W0, A=self.A, B=self.B,
            alpha=self.alpha, dropout=self.dropout, training=self.training,
        )
        
        if self.bias is not None:
            out = out + self.bias
        
        if len(original_shape) == 2:
            out = out.squeeze(0)
        elif len(original_shape) > 3:
            out = out.reshape(*batch_dims, out.shape[-1])
        
        return out
    
    def merge_weights(self) -> mx.array:
        """Merge LoRA into base weights."""
        return merge_lora_weights(self.W0, self.A, self.B, self.alpha)
    
    def to_inference_mode(self) -> "LoRALinear":
        """Merge weights for inference."""
        self.W0 = self.merge_weights()
        self.A = mx.zeros_like(self.A)
        self.B = mx.zeros_like(self.B)
        return self
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> "LoRALinear":
        """Create from existing nn.Linear."""
        in_features = linear.weight.shape[1]
        out_features = linear.weight.shape[0]
        
        lora = cls(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            use_bias=linear.bias is not None,
            freeze_base=True,
        )
        
        lora.W0 = linear.weight
        if linear.bias is not None:
            lora.bias = linear.bias
        
        return lora
    
    def trainable_parameters(self) -> dict:
        """Get trainable LoRA parameters."""
        params = {"A": self.A, "B": self.B}
        if self.bias is not None and not self.freeze_base:
            params["bias"] = self.bias
        return params
    
    def num_trainable_params(self) -> int:
        count = self.A.size + self.B.size
        if self.bias is not None and not self.freeze_base:
            count += self.bias.size
        return count
    
    def num_total_params(self) -> int:
        count = self.W0.size + self.A.size + self.B.size
        if self.bias is not None:
            count += self.bias.size
        return count


class LoRAEmbedding(nn.Module):
    """Embedding layer with LoRA adaptation."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.alpha = alpha
        
        self.embeddings = mx.random.normal((num_embeddings, embedding_dim)) * 0.02
        
        bound = 1.0 / math.sqrt(embedding_dim)
        self.A = mx.random.uniform(-bound, bound, (rank, embedding_dim))
        self.B = mx.zeros((num_embeddings, rank))
        
        if freeze_base:
            self.embeddings = mx.stop_gradient(self.embeddings)
    
    def __call__(self, indices: mx.array) -> mx.array:
        base_emb = self.embeddings[indices]
        scale = self.alpha / self.rank
        B_selected = self.B[indices]
        lora_emb = mx.matmul(B_selected, self.A)
        return base_emb + scale * lora_emb
    
    def merge_weights(self) -> mx.array:
        scale = self.alpha / self.rank
        return self.embeddings + scale * mx.matmul(self.B, self.A)
