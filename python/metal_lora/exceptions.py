"""Custom exceptions for MetalLoRA."""


class MetalLoRAError(Exception):
    """Base exception for MetalLoRA."""
    pass


class ConfigurationError(MetalLoRAError):
    """Invalid configuration."""
    pass


class ShapeError(MetalLoRAError):
    """Tensor shape mismatch."""
    pass


class DeviceError(MetalLoRAError):
    """Device not supported."""
    pass


class AdapterError(MetalLoRAError):
    """Adapter-related error."""
    pass


class QuantizationError(MetalLoRAError):
    """Quantization error."""
    pass


def validate_rank(rank: int, min_rank: int = 1, max_rank: int = 256) -> None:
    """Validate LoRA rank."""
    if not isinstance(rank, int):
        raise ConfigurationError(f"rank must be int, got {type(rank).__name__}")
    if rank < min_rank or rank > max_rank:
        raise ConfigurationError(f"rank must be in [{min_rank}, {max_rank}], got {rank}")


def validate_alpha(alpha: float) -> None:
    """Validate LoRA alpha."""
    if not isinstance(alpha, (int, float)):
        raise ConfigurationError(f"alpha must be numeric, got {type(alpha).__name__}")
    if alpha <= 0:
        raise ConfigurationError(f"alpha must be positive, got {alpha}")


def validate_shape(tensor, expected_shape: tuple, name: str = "tensor") -> None:
    """Validate tensor shape."""
    if tensor.shape != expected_shape:
        raise ShapeError(f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}")


def validate_dtype(tensor, expected_dtype, name: str = "tensor") -> None:
    """Validate tensor dtype."""
    if tensor.dtype != expected_dtype:
        raise ShapeError(f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")


def validate_positive(value: float, name: str = "value") -> None:
    """Validate positive value."""
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")


def validate_probability(value: float, name: str = "probability") -> None:
    """Validate probability in [0, 1]."""
    if not 0 <= value <= 1:
        raise ConfigurationError(f"{name} must be in [0, 1], got {value}")
