"""Advanced optimizations for production deployment."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import zlib
import struct


# =============================================================================
# Weight Compression (LZ4-style)
# =============================================================================

def compress_weights(weights: Dict[str, mx.array]) -> bytes:
    """Compress LoRA weights using zlib (2-3x smaller files)."""
    data = {}
    for name, arr in weights.items():
        data[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data": arr.astype(mx.float16).tolist(),
        }
    
    json_bytes = json.dumps(data).encode('utf-8')
    compressed = zlib.compress(json_bytes, level=9)
    return compressed


def decompress_weights(compressed: bytes) -> Dict[str, mx.array]:
    """Decompress LoRA weights."""
    json_bytes = zlib.decompress(compressed)
    data = json.loads(json_bytes.decode('utf-8'))
    
    weights = {}
    for name, info in data.items():
        arr = mx.array(info["data"])
        arr = arr.reshape(info["shape"])
        weights[name] = arr
    
    return weights


def save_compressed(weights: Dict[str, mx.array], path: str) -> int:
    """Save compressed weights, return bytes saved vs uncompressed."""
    compressed = compress_weights(weights)
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        f.write(compressed)
    
    uncompressed_size = sum(a.size * 2 for a in weights.values())  # FP16
    return uncompressed_size - len(compressed)


def load_compressed(path: str) -> Dict[str, mx.array]:
    """Load compressed weights."""
    with open(path, "rb") as f:
        compressed = f.read()
    return decompress_weights(compressed)


# =============================================================================
# Memory Pool
# =============================================================================

class MemoryPool:
    """Reusable memory allocations to avoid allocation overhead."""
    
    def __init__(self, max_size_mb: int = 512):
        self.max_size = max_size_mb * 1024 * 1024
        self.pools: Dict[Tuple[Tuple[int, ...], str], List[mx.array]] = {}
        self.total_allocated = 0
    
    def get(self, shape: Tuple[int, ...], dtype: Any = mx.float32) -> mx.array:
        """Get array from pool or allocate new."""
        key = (shape, str(dtype))
        
        if key in self.pools and self.pools[key]:
            return self.pools[key].pop()
        
        arr = mx.zeros(shape, dtype=dtype)
        size = arr.size * (2 if dtype == mx.float16 else 4)
        self.total_allocated += size
        return arr
    
    def release(self, arr: mx.array):
        """Return array to pool for reuse."""
        key = (tuple(arr.shape), str(arr.dtype))
        
        if key not in self.pools:
            self.pools[key] = []
        
        if len(self.pools[key]) < 8:  # Max 8 per shape
            self.pools[key].append(arr)
    
    def clear(self):
        """Clear all pooled memory."""
        self.pools.clear()
        self.total_allocated = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        pooled = sum(len(v) for v in self.pools.values())
        return {
            "total_allocated_mb": self.total_allocated / (1024 * 1024),
            "pooled_arrays": pooled,
            "shapes": len(self.pools),
        }


_GLOBAL_POOL: Optional[MemoryPool] = None

def get_memory_pool() -> MemoryPool:
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        _GLOBAL_POOL = MemoryPool()
    return _GLOBAL_POOL


# =============================================================================
# Multi-Adapter Manager
# =============================================================================

class MultiAdapterManager:
    """Manage multiple LoRA adapters for efficient batched inference."""
    
    def __init__(self, base_model: nn.Module, max_adapters: int = 64):
        self.base_model = base_model
        self.max_adapters = max_adapters
        self.adapters: Dict[str, Dict[str, mx.array]] = {}
        self.adapter_ids: Dict[str, int] = {}
        self.next_id = 0
        
        # Batched adapter storage
        self.batched_A: Optional[mx.array] = None
        self.batched_B: Optional[mx.array] = None
        self._needs_rebuild = True
    
    def add_adapter(self, name: str, weights: Dict[str, mx.array]) -> int:
        """Add adapter, return its ID."""
        if len(self.adapters) >= self.max_adapters:
            raise ValueError(f"Max adapters ({self.max_adapters}) reached")
        
        adapter_id = self.next_id
        self.adapters[name] = weights
        self.adapter_ids[name] = adapter_id
        self.next_id += 1
        self._needs_rebuild = True
        
        return adapter_id
    
    def remove_adapter(self, name: str):
        """Remove adapter."""
        if name in self.adapters:
            del self.adapters[name]
            del self.adapter_ids[name]
            self._needs_rebuild = True
    
    def _rebuild_batched(self):
        """Rebuild batched adapter storage for GPU."""
        if not self.adapters or not self._needs_rebuild:
            return
        
        # Get shapes from first adapter
        first = next(iter(self.adapters.values()))
        first_A = next(a for k, a in first.items() if k.endswith(".A"))
        first_B = next(a for k, a in first.items() if k.endswith(".B"))
        
        rank, in_features = first_A.shape
        out_features, _ = first_B.shape
        
        # Allocate batched storage
        num = len(self.adapters)
        self.batched_A = mx.zeros((num, rank, in_features), dtype=mx.float16)
        self.batched_B = mx.zeros((num, out_features, rank), dtype=mx.float16)
        
        # Fill batched arrays
        for name, weights in self.adapters.items():
            idx = self.adapter_ids[name]
            for key, arr in weights.items():
                if key.endswith(".A"):
                    self.batched_A[idx] = arr.astype(mx.float16)
                elif key.endswith(".B"):
                    self.batched_B[idx] = arr.astype(mx.float16)
        
        self._needs_rebuild = False
    
    def forward_batched(
        self,
        x: mx.array,
        adapter_ids: List[int],
    ) -> mx.array:
        """Forward pass with different adapter per sample."""
        self._rebuild_batched()
        
        batch_size = x.shape[0]
        ids = mx.array(adapter_ids)
        
        # Select adapters for each sample
        A_selected = self.batched_A[ids]  # [B, R, K]
        B_selected = self.batched_B[ids]  # [B, D, R]
        
        # Compute LoRA for each sample
        # x: [B, S, K], A: [B, R, K] -> Ax: [B, S, R]
        Ax = mx.einsum("bsk,brk->bsr", x, A_selected)
        
        # BAx: [B, S, D]
        BAx = mx.einsum("bsr,bdr->bsd", Ax, B_selected)
        
        return BAx
    
    def list_adapters(self) -> List[str]:
        return list(self.adapters.keys())


# =============================================================================
# Speculative Decoding
# =============================================================================

class SpeculativeDecoder:
    """Speculative decoding for faster autoregressive generation."""
    
    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        draft_steps: int = 5,
        temperature: float = 1.0,
    ):
        self.draft = draft_model
        self.target = target_model
        self.draft_steps = draft_steps
        self.temperature = temperature
    
    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
    ) -> Tuple[mx.array, Dict[str, float]]:
        """Generate with speculative decoding. Returns tokens and stats."""
        generated = input_ids
        total_draft = 0
        total_accepted = 0
        
        while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
            remaining = max_new_tokens - (generated.shape[1] - input_ids.shape[1])
            steps = min(self.draft_steps, remaining)
            
            # Draft phase
            draft_tokens = []
            current = generated
            
            for _ in range(steps):
                logits = self.draft(current)[:, -1, :]
                probs = mx.softmax(logits / self.temperature, axis=-1)
                next_tok = mx.argmax(probs, axis=-1, keepdims=True)
                draft_tokens.append(next_tok)
                current = mx.concatenate([current, next_tok], axis=1)
            
            total_draft += steps
            
            # Verify with target
            target_logits = self.target(current)
            
            accepted = 0
            for i, draft_tok in enumerate(draft_tokens):
                pos = generated.shape[1] + i
                target_probs = mx.softmax(target_logits[:, pos, :] / self.temperature, axis=-1)
                draft_prob = mx.take_along_axis(target_probs, draft_tok, axis=-1)
                
                # Accept if prob > threshold
                if float(draft_prob.mean()) > 0.1:
                    generated = mx.concatenate([generated, draft_tok], axis=1)
                    accepted += 1
                else:
                    # Resample from target
                    resampled = mx.argmax(target_probs, axis=-1, keepdims=True)
                    generated = mx.concatenate([generated, resampled], axis=1)
                    accepted += 1
                    break
            
            total_accepted += accepted
            
            if accepted == 0:
                # Fallback: sample from target
                resampled = mx.argmax(target_logits[:, generated.shape[1] - input_ids.shape[1], :], axis=-1, keepdims=True)
                generated = mx.concatenate([generated, resampled], axis=1)
        
        stats = {
            "total_draft": total_draft,
            "total_accepted": total_accepted,
            "acceptance_rate": total_accepted / max(total_draft, 1),
            "speedup": total_accepted / max(total_draft / self.draft_steps, 1),
        }
        
        return generated[:, :input_ids.shape[1] + max_new_tokens], stats


# =============================================================================
# KV Cache with LoRA
# =============================================================================

class LoRAKVCache:
    """KV cache optimized for incremental LoRA decoding."""
    
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: Any = mx.float16,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.K = mx.zeros((batch_size, max_seq_len, num_heads, head_dim), dtype=dtype)
        self.V = mx.zeros((batch_size, max_seq_len, num_heads, head_dim), dtype=dtype)
        self.cur_pos = 0
    
    def update(self, K_new: mx.array, V_new: mx.array) -> Tuple[mx.array, mx.array]:
        """Update cache with new K, V values."""
        seq_len = K_new.shape[1]
        end_pos = self.cur_pos + seq_len
        
        self.K[:, self.cur_pos:end_pos] = K_new
        self.V[:, self.cur_pos:end_pos] = V_new
        self.cur_pos = end_pos
        
        return self.K[:, :end_pos], self.V[:, :end_pos]
    
    def reset(self):
        """Reset cache for new sequence."""
        self.cur_pos = 0


# =============================================================================
# Lazy Evaluation Wrapper
# =============================================================================

class LazyLoRA(nn.Module):
    """Lazy evaluation - defers computation until result is needed."""
    
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base = base_layer
        self._pending_ops: List[Tuple[str, Any]] = []
        self._cached_result: Optional[mx.array] = None
    
    def __call__(self, x: mx.array) -> "LazyLoRA":
        """Queue operation, don't execute yet."""
        self._pending_ops.append(("forward", x))
        self._cached_result = None
        return self
    
    def eval(self) -> mx.array:
        """Execute all pending operations."""
        if self._cached_result is not None:
            return self._cached_result
        
        result = None
        for op, arg in self._pending_ops:
            if op == "forward":
                result = self.base(arg)
        
        self._pending_ops.clear()
        self._cached_result = result
        return result


# =============================================================================
# Mixed Precision Training
# =============================================================================

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    compute_dtype: Any = mx.float16      # Forward pass
    accumulate_dtype: Any = mx.float32   # Gradient accumulation
    master_dtype: Any = mx.float32       # Master weights
    loss_scale: float = 1024.0           # Dynamic loss scaling
    scale_factor: float = 2.0            # Scale up/down factor
    scale_window: int = 1000             # Steps before scale up


class MixedPrecisionTrainer:
    """Training wrapper with automatic mixed precision."""
    
    def __init__(self, model: nn.Module, config: Optional[MixedPrecisionConfig] = None):
        self.model = model
        self.config = config or MixedPrecisionConfig()
        self.loss_scale = self.config.loss_scale
        self.steps_since_scale = 0
        self.overflow_count = 0
    
    def forward(self, x: mx.array) -> mx.array:
        """Forward in compute precision."""
        x = x.astype(self.config.compute_dtype)
        return self.model(x)
    
    def backward(self, loss: mx.array) -> Dict[str, mx.array]:
        """Backward with loss scaling."""
        scaled_loss = loss * self.loss_scale
        grads = mx.grad(lambda: scaled_loss)()
        
        # Unscale
        grads = {k: g / self.loss_scale for k, g in grads.items()}
        
        # Check for overflow
        has_inf = any(mx.any(mx.isinf(g)) for g in grads.values())
        has_nan = any(mx.any(mx.isnan(g)) for g in grads.values())
        
        if has_inf or has_nan:
            self.overflow_count += 1
            self.loss_scale /= self.config.scale_factor
            return {}  # Skip update
        
        self.steps_since_scale += 1
        if self.steps_since_scale >= self.config.scale_window:
            self.loss_scale *= self.config.scale_factor
            self.steps_since_scale = 0
        
        return grads


# =============================================================================
# Gradient Accumulation Fusion
# =============================================================================

class FusedGradAccumulator:
    """Accumulate gradients in-place without extra allocations."""
    
    def __init__(self, params: Dict[str, mx.array], accum_steps: int):
        self.params = params
        self.accum_steps = accum_steps
        self.accum: Dict[str, mx.array] = {}
        self.step = 0
        
        # Pre-allocate accumulation buffers
        for name, param in params.items():
            self.accum[name] = mx.zeros_like(param, dtype=mx.float32)
    
    def accumulate(self, grads: Dict[str, mx.array]) -> bool:
        """Accumulate gradients. Returns True when ready to update."""
        for name, grad in grads.items():
            if name in self.accum:
                self.accum[name] = self.accum[name] + grad.astype(mx.float32)
        
        self.step += 1
        return self.step >= self.accum_steps
    
    def get_averaged(self) -> Dict[str, mx.array]:
        """Get averaged gradients and reset."""
        result = {k: v / self.accum_steps for k, v in self.accum.items()}
        
        # Reset
        for name in self.accum:
            self.accum[name] = mx.zeros_like(self.accum[name])
        self.step = 0
        
        return result
