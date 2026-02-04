"""
CCE (Cut Cross-Entropy) Loss wrapper with automatic fallback for small batch sizes.

Usage:
    from cce_loss import cce_loss

    # Works for any N (batch_size * seq_len)
    loss = cce_loss(hidden_states, lm_head_weight, targets)
"""

import warnings
import mlx.core as mx
import mlx.nn as nn

# Minimum batch size for GPU CCE kernels
MIN_N_FOR_CCE = 256

_warning_shown = False


def cce_loss(
    hidden: mx.array,
    weight: mx.array,
    targets: mx.array,
    ignore_index: int = -100,
    reduction: str = "none"
) -> mx.array:
    """
    Memory-efficient cross-entropy loss for large vocabularies.

    For N >= 256: Uses optimized GPU CCE kernels (memory efficient)
    For N < 256: Falls back to baseline cross-entropy (full logits materialization)

    Args:
        hidden: Hidden states [N, H] or [batch, seq, H]
        weight: LM head weight matrix [V, H]
        targets: Target token IDs [N] or [batch, seq]
        ignore_index: Target value to ignore (default: -100)
        reduction: 'none', 'mean', or 'sum' (default: 'none')

    Returns:
        Per-token losses [N] if reduction='none', scalar otherwise
    """
    global _warning_shown

    # Flatten inputs if needed
    if hidden.ndim == 3:
        batch, seq, H = hidden.shape
        N = batch * seq
        hidden = hidden.reshape(N, H)
        targets = targets.reshape(N)
    else:
        N = hidden.shape[0]

    # Check if we need to use baseline fallback
    if N < MIN_N_FOR_CCE:
        if not _warning_shown:
            warnings.warn(
                f"CCE: N={N} < {MIN_N_FOR_CCE}, using baseline cross-entropy. "
                f"For memory-efficient CCE, use batch_size * seq_len >= {MIN_N_FOR_CCE}.",
                UserWarning
            )
            _warning_shown = True

        # Baseline: compute full logits and use standard cross-entropy
        logits = hidden @ weight.T
        if hidden.dtype == mx.bfloat16:
            logits = logits.astype(mx.float32)

        loss = nn.losses.cross_entropy(
            logits, targets,
            reduction="none"
        )

        # Handle ignore_index
        if ignore_index >= 0:
            valid_mask = targets != ignore_index
            loss = mx.where(valid_mask, loss, mx.zeros_like(loss))
    else:
        # Use optimized GPU CCE
        loss = mx.fast.cce_loss(hidden, weight, targets, ignore_index=ignore_index)

    # Apply reduction
    if reduction == "mean":
        if ignore_index >= 0:
            valid_mask = targets != ignore_index
            return loss.sum() / valid_mask.sum()
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def cce_loss_with_z_loss(
    hidden: mx.array,
    weight: mx.array,
    targets: mx.array,
    z_loss_weight: float = 1e-4,
    ignore_index: int = -100,
    reduction: str = "mean"
) -> tuple:
    """
    CCE loss with auxiliary z-loss for training stability.

    Z-loss encourages the logits to stay small, which helps with numerical stability
    during training, especially with large vocabularies.

    Args:
        hidden: Hidden states [N, H] or [batch, seq, H]
        weight: LM head weight matrix [V, H]
        targets: Target token IDs [N] or [batch, seq]
        z_loss_weight: Weight for z-loss term (default: 1e-4)
        ignore_index: Target value to ignore (default: -100)
        reduction: 'mean' or 'sum' (default: 'mean')

    Returns:
        Tuple of (total_loss, ce_loss, z_loss)
    """
    # Flatten inputs if needed
    if hidden.ndim == 3:
        batch, seq, H = hidden.shape
        N = batch * seq
        hidden_flat = hidden.reshape(N, H)
        targets_flat = targets.reshape(N)
    else:
        N = hidden.shape[0]
        hidden_flat = hidden
        targets_flat = targets

    # Compute CE loss
    ce_loss = cce_loss(hidden_flat, weight, targets_flat, ignore_index, reduction="none")

    # Compute z-loss (requires logits)
    logits = hidden_flat @ weight.T
    if hidden_flat.dtype == mx.bfloat16:
        logits = logits.astype(mx.float32)
    lse = mx.logsumexp(logits, axis=-1)

    z_loss = lse ** 2

    # Handle ignore_index
    if ignore_index >= 0:
        valid_mask = targets_flat != ignore_index
        ce_loss = mx.where(valid_mask, ce_loss, mx.zeros_like(ce_loss))
        z_loss = mx.where(valid_mask, z_loss, mx.zeros_like(z_loss))
        num_valid = valid_mask.sum()
    else:
        num_valid = N

    # Apply reduction
    if reduction == "mean":
        ce_loss_reduced = ce_loss.sum() / num_valid
        z_loss_reduced = z_loss.sum() / num_valid
    else:  # sum
        ce_loss_reduced = ce_loss.sum()
        z_loss_reduced = z_loss.sum()

    total_loss = ce_loss_reduced + z_loss_weight * z_loss_reduced

    return total_loss, ce_loss_reduced, z_loss_reduced
