"""Low-level LoRA operations."""

import mlx.core as mx
import mlx.core.fast as fast
from pathlib import Path
from typing import Optional, Tuple
import struct


def _get_kernel_source() -> str:
    """Load Metal kernel sources."""
    kernel_dir = Path(__file__).parent.parent.parent / "kernels"
    sources = []
    for filename in ["lora_kernels.metal", "lora_train.metal", "lora_quantized.metal"]:
        kernel_path = kernel_dir / filename
        if kernel_path.exists():
            sources.append(kernel_path.read_text())
    return "\n".join(sources)


_KERNEL_SOURCE: Optional[str] = None
_LORA_FORWARD_KERNEL = None
_LORA_BACKWARD_KERNEL = None


def get_kernel_source() -> str:
    global _KERNEL_SOURCE
    if _KERNEL_SOURCE is None:
        _KERNEL_SOURCE = _get_kernel_source()
    return _KERNEL_SOURCE


def _get_lora_forward_kernel():
    """Get or compile the LoRA forward kernel."""
    global _LORA_FORWARD_KERNEL
    if _LORA_FORWARD_KERNEL is not None:
        return _LORA_FORWARD_KERNEL
    
    # Kernel header with necessary Metal includes for simdgroup
    kernel_header = """
#include <metal_simdgroup>
using namespace metal;
"""
    
    # Kernel body - EXACT match to lora_forward_simd_fp16 from lora_kernels.metal
    # MLX's metal_kernel generates the signature automatically
    # Pass alpha as an additional input array (single float)
    kernel_source = """
        // Extract thread indices from Metal attributes (matching original)
        uint batch_idx = thread_position_in_grid.z;
        uint seq_idx = thread_position_in_grid.y;
        uint lid = thread_index_in_threadgroup;
        uint simd_lane_id = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;
        uint d = thread_position_in_grid.x * 256 + lid;
        
        // Get dimensions from shapes (MLX provides these when ensure_row_contiguous=False)
        uint batch_size = x_shape[0];
        uint seq_len = x_shape[1];
        uint in_features = x_shape[2];
        uint out_features = W0_shape[0];
        uint rank = A_shape[0];
        float alpha_val = alpha[0];  // alpha is passed as single-element array
        
        if (batch_idx >= batch_size || seq_idx >= seq_len || d >= out_features) return;
        
        float scale = alpha_val / float(rank);
        uint x_offset = (batch_idx * seq_len + seq_idx) * in_features;
        uint out_offset = (batch_idx * seq_len + seq_idx) * out_features;
        
        // Threadgroup memory (matching original)
        threadgroup T tg_x[4096];
        threadgroup float tg_Ax[128];
        
        // Load x into threadgroup memory (EXACT match to original lines 162-165)
        for (uint k = lid; k < in_features; k += 256) {
            tg_x[k] = x[x_offset + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute Ax with simd reduction (EXACT match to original lines 167-179)
        for (uint r = simd_group_id; r < rank; r += 4) {
            float ax = 0.0f;
            for (uint k = simd_lane_id; k < in_features; k += 32) {
                ax += float(A[r * in_features + k]) * float(tg_x[k]);
            }
            // simd reduction
            ax = simd_sum(ax);
            if (simd_lane_id == 0) {
                tg_Ax[r] = ax;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute output with simd reduction for W0 @ x (EXACT match to original lines 181-186)
        float h = 0.0f;
        for (uint k = simd_lane_id; k < in_features; k += 32) {
            h += float(W0[d * in_features + k]) * float(tg_x[k]);
        }
        h = simd_sum(h);
        
        // Compute LoRA contribution: B @ Ax (EXACT match to original lines 188-191)
        float lora = 0.0f;
        for (uint r = 0; r < rank; ++r) {
            lora += float(B[d * rank + r]) * tg_Ax[r];
        }
        
        // Write output (EXACT match to original lines 193-195)
        if (simd_lane_id == 0) {
            out[out_offset + d] = T(h + scale * lora);
        }
    """
    
    _LORA_FORWARD_KERNEL = fast.metal_kernel(
        name="lora_forward_fused",
        input_names=["x", "W0", "A", "B", "alpha"],
        output_names=["out"],
        source=kernel_source,
        header=kernel_header,
        ensure_row_contiguous=False,  # Need shapes/strides for indexing
    )
    
    return _LORA_FORWARD_KERNEL


def _dispatch_lora_forward_kernel(
    x: mx.array,
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float,
) -> mx.array:
    """Dispatch the custom Metal kernel for LoRA forward pass."""
    kernel = _get_lora_forward_kernel()
    
    batch_size, seq_len, in_features = x.shape
    out_features, rank = B.shape
    
    # Ensure FP16
    x = x.astype(mx.float16)
    W0 = W0.astype(mx.float16)
    A = A.astype(mx.float16)
    B = B.astype(mx.float16)
    
    # Pass alpha as single-element array
    alpha_arr = mx.array([alpha], dtype=mx.float32)
    
    # Calculate grid and threadgroup sizes
    grid_size = ((out_features + 255) // 256, seq_len, batch_size)
    threadgroup_size = (256, 1, 1)
    
    # Dispatch kernel with zero initialization (important: kernel writes all elements now)
    outputs = kernel(
        inputs=[x, W0, A, B, alpha_arr],
        template=[("T", mx.float16)],
        grid=grid_size,
        threadgroup=threadgroup_size,
        output_shapes=[(batch_size, seq_len, out_features)],
        output_dtypes=[mx.float16],
        init_value=0.0,  # Initialize to zero
    )
    
    return outputs[0]


def _get_lora_backward_kernel():
    """Get or compile the LoRA backward kernel."""
    global _LORA_BACKWARD_KERNEL
    if _LORA_BACKWARD_KERNEL is not None:
        return _LORA_BACKWARD_KERNEL
    
    # Kernel header
    kernel_header = """
using namespace metal;
"""
    
    # Kernel body adapted from lora_train_backward
    # Computes Ax on the fly (no cache needed for benchmark)
    kernel_source = """
        // Extract thread indices
        uint batch_idx = thread_position_in_grid.z;
        uint seq_idx = thread_position_in_grid.y;
        uint lid = thread_index_in_threadgroup;
        
        // Get dimensions from shapes
        uint batch_size = grad_output_shape[0];
        uint seq_len = grad_output_shape[1];
        uint out_features = grad_output_shape[2];
        uint rank = A_shape[0];
        uint in_features = A_shape[1];
        float alpha_val = alpha[0];
        float clip_val = grad_clip[0];
        
        if (batch_idx >= batch_size || seq_idx >= seq_len) return;
        
        float scale = alpha_val / float(rank);
        uint x_offset = (batch_idx * seq_len + seq_idx) * in_features;
        uint grad_offset = (batch_idx * seq_len + seq_idx) * out_features;
        
        // Threadgroup memory
        threadgroup T tg_x[4096];
        threadgroup T tg_grad[4096];
        threadgroup float tg_Ax[128];
        threadgroup float tg_Bt_grad[128];
        
        // Load x and grad_output into threadgroup memory
        for (uint k = lid; k < in_features; k += 256) {
            tg_x[k] = x[x_offset + k];
        }
        for (uint d = lid; d < out_features; d += 256) {
            tg_grad[d] = grad_output[grad_offset + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute Ax on the fly (no cache)
        for (uint r = lid; r < rank; r += 256) {
            float ax = 0.0f;
            for (uint k = 0; k < in_features; ++k) {
                ax += float(A[r * in_features + k]) * float(tg_x[k]);
            }
            tg_Ax[r] = ax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute Bt_grad = B^T @ grad_output
        for (uint r = lid; r < rank; r += 256) {
            float acc = 0.0f;
            for (uint d = 0; d < out_features; ++d) {
                acc += float(B[d * rank + r]) * float(tg_grad[d]);
            }
            tg_Bt_grad[r] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute grad_B: scale * grad_output @ Ax (with atomic accumulation)
        for (uint dr = lid; dr < out_features * rank; dr += 256) {
            uint d = dr / rank;
            uint r = dr % rank;
            if (d < out_features && r < rank) {
                float grad_val = scale * float(tg_grad[d]) * tg_Ax[r];
                grad_val = clamp(grad_val, -clip_val, clip_val);
                atomic_fetch_add_explicit(&grad_B[d * rank + r], grad_val, memory_order_relaxed);
            }
        }
        
        // Compute grad_A: scale * Bt_grad @ x (with atomic accumulation)
        for (uint rk = lid; rk < rank * in_features; rk += 256) {
            uint r = rk / in_features;
            uint k = rk % in_features;
            if (r < rank && k < in_features) {
                float grad_val = scale * tg_Bt_grad[r] * float(tg_x[k]);
                grad_val = clamp(grad_val, -clip_val, clip_val);
                atomic_fetch_add_explicit(&grad_A[r * in_features + k], grad_val, memory_order_relaxed);
            }
        }
    """
    
    _LORA_BACKWARD_KERNEL = fast.metal_kernel(
        name="lora_backward_fused",
        input_names=["grad_output", "x", "A", "B", "alpha", "grad_clip"],
        output_names=["grad_A", "grad_B"],
        source=kernel_source,
        header=kernel_header,
        ensure_row_contiguous=False,
        atomic_outputs=True,  # Required for atomic operations
    )
    
    return _LORA_BACKWARD_KERNEL


def _dispatch_lora_backward_kernel(
    grad_output: mx.array,
    x: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float,
    clip_value: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """Dispatch the custom Metal kernel for LoRA backward pass."""
    kernel = _get_lora_backward_kernel()
    
    batch_size, seq_len, out_features = grad_output.shape
    rank, in_features = A.shape
    
    # Ensure FP16
    grad_output = grad_output.astype(mx.float16)
    x = x.astype(mx.float16)
    A = A.astype(mx.float16)
    B = B.astype(mx.float16)
    
    # Pass alpha and clip_value as arrays
    alpha_arr = mx.array([alpha], dtype=mx.float32)
    clip_arr = mx.array([clip_value], dtype=mx.float32)
    
    # Initialize output gradients to zero
    grad_A = mx.zeros((rank, in_features), dtype=mx.float32)
    grad_B = mx.zeros((out_features, rank), dtype=mx.float32)
    
    # Calculate grid and threadgroup sizes
    # Process all batch/seq combinations - gradients accumulate atomically
    grid_size = (1, seq_len, batch_size)
    threadgroup_size = (256, 1, 1)
    
    # Dispatch kernel
    # Note: init_value must be a single float, not a list (MLX API limitation)
    # Since we're using atomic_outputs=True, outputs are initialized to 0 by default
    outputs = kernel(
        inputs=[grad_output, x, A, B, alpha_arr, clip_arr],
        template=[("T", mx.float16)],
        grid=grid_size,
        threadgroup=threadgroup_size,
        output_shapes=[(rank, in_features), (out_features, rank)],
        output_dtypes=[mx.float32, mx.float32],
        # init_value not needed for atomic outputs - they initialize to 0
    )
    
    return outputs[0], outputs[1]


def lora_forward(
    x: mx.array,
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
    dropout: float = 0.0,
    training: bool = False,
    use_custom_kernel: bool = True,
) -> mx.array:
    """LoRA forward: h = W0 @ x + (alpha/rank) * B @ A @ x
    
    Args:
        use_custom_kernel: If True, use custom Metal kernel. If False, use mx.matmul().
    """
    if x.ndim == 2:
        x = x[None, :, :]
    
    batch_size, seq_len, in_features = x.shape
    out_features, rank = B.shape
    
    # Use custom Metal kernel if requested
    if use_custom_kernel:
        try:
            output = _dispatch_lora_forward_kernel(x, W0, A, B, alpha)
            
            # Check for NaN/Inf in output
            if not mx.all(mx.isfinite(output)):
                nan_count = mx.sum(~mx.isfinite(output))
                import warnings
                warnings.warn(f"Custom kernel produced {nan_count} NaN/Inf values, falling back to mx.matmul()")
                use_custom_kernel = False
            
            # Handle dropout if needed (fallback to MLX ops for now)
            if training and dropout > 0:
                # For dropout, we'd need to modify the kernel or use MLX ops
                # For now, compute BAx separately for dropout
                BAx = mx.matmul(mx.matmul(x, A.T), B.T)
                W0x = mx.matmul(x, W0.T)
                scale = alpha / rank
                mask = mx.random.bernoulli(1 - dropout, BAx.shape)
                dropout_scale = 1.0 / (1.0 - dropout)
                output = W0x + scale * BAx * mask * dropout_scale
            
            if use_custom_kernel:
                return output
        except Exception as e:
            # Fallback to MLX if kernel fails
            import warnings
            warnings.warn(f"Custom kernel failed, falling back to mx.matmul(): {e}")
            use_custom_kernel = False
    
    # Fallback: Standard MLX implementation
    if not use_custom_kernel:
        scale = alpha / rank
        
        Ax = mx.matmul(x, A.T)
        BAx = mx.matmul(Ax, B.T)
        W0x = mx.matmul(x, W0.T)
        
        output = W0x + scale * BAx
        
        if training and dropout > 0:
            mask = mx.random.bernoulli(1 - dropout, BAx.shape)
            dropout_scale = 1.0 / (1.0 - dropout)
            output = W0x + scale * BAx * mask * dropout_scale
        
        return output


def lora_forward_inference(x: mx.array, W_merged: mx.array) -> mx.array:
    """Inference forward with merged weights."""
    if x.ndim == 2:
        x = x[None, :, :]
    return mx.matmul(x, W_merged.T)


def lora_backward(
    grad_output: mx.array,
    x: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
    clip_value: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """Compute gradients for A and B."""
    batch_size, seq_len, out_features = grad_output.shape
    rank, in_features = A.shape
    scale = alpha / rank
    
    Ax = mx.matmul(x, A.T)
    Bt_grad = mx.matmul(grad_output, B)
    
    grad_B = mx.zeros((out_features, rank))
    grad_A = mx.zeros((rank, in_features))
    
    for b in range(batch_size):
        grad_B = grad_B + mx.matmul(grad_output[b].T, Ax[b])
        grad_A = grad_A + mx.matmul(Bt_grad[b].T, x[b])
    
    grad_B = mx.clip(scale * grad_B, -clip_value, clip_value)
    grad_A = mx.clip(scale * grad_A, -clip_value, clip_value)
    
    return grad_A, grad_B


def lora_backward_efficient(
    grad_output: mx.array,
    x: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
    clip_value: float = 1.0,
    use_custom_kernel: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Memory-efficient backward using batched matmul or custom Metal kernel.
    
    Args:
        use_custom_kernel: If True, use custom Metal kernel. If False, use mx.matmul().
    """
    # Use custom Metal kernel if requested
    if use_custom_kernel:
        try:
            return _dispatch_lora_backward_kernel(grad_output, x, A, B, alpha, clip_value)
        except Exception as e:
            # Fallback to MLX if kernel fails
            import warnings
            warnings.warn(f"Custom backward kernel failed, falling back to mx.matmul(): {e}")
            use_custom_kernel = False
    
    # Fallback: Standard MLX implementation
    if not use_custom_kernel:
        batch_size, seq_len, out_features = grad_output.shape
        rank, in_features = A.shape
        scale = alpha / rank
        
        Ax = mx.matmul(x, A.T)
        grad_output_t = mx.transpose(grad_output, (0, 2, 1))
        grad_B_batched = mx.matmul(grad_output_t, Ax)
        grad_B = scale * mx.sum(grad_B_batched, axis=0)
        
        Bt_grad = mx.matmul(grad_output, B)
        Bt_grad_t = mx.transpose(Bt_grad, (0, 2, 1))
        grad_A_batched = mx.matmul(Bt_grad_t, x)
        grad_A = scale * mx.sum(grad_A_batched, axis=0)
        
        grad_A = mx.clip(grad_A, -clip_value, clip_value)
        grad_B = mx.clip(grad_B, -clip_value, clip_value)
        
        return grad_A, grad_B
    
    # Should not reach here, but return fallback just in case
    return lora_backward_efficient(grad_output, x, A, B, alpha, clip_value, use_custom_kernel=False)


def merge_lora_weights(
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
) -> mx.array:
    """Merge LoRA into base weights: W' = W0 + (alpha/rank) * B @ A"""
    rank = A.shape[0]
    scale = alpha / rank
    BA = mx.matmul(B, A)
    return W0 + scale * BA


def unmerge_lora_weights(
    W_merged: mx.array,
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    alpha: float = 16.0,
) -> Tuple[mx.array, mx.array]:
    """Recover LoRA from merged weights using SVD."""
    rank = A.shape[0]
    scale = alpha / rank
    delta = (W_merged - W0) / scale
    
    U, S, Vt = mx.linalg.svd(delta)
    sqrt_S = mx.sqrt(S[:rank])
    
    A_recovered = mx.diag(sqrt_S) @ Vt[:rank, :]
    B_recovered = U[:, :rank] @ mx.diag(sqrt_S)
    
    return A_recovered, B_recovered
