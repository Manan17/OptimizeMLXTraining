"""Unit tests for MetalLoRA."""

import pytest
import mlx.core as mx

from metal_lora import (
    LoRALinear,
    LoRAEmbedding,
    lora_forward,
    merge_lora_weights,
    ConfigurationError,
    validate_rank,
    validate_alpha,
)


class TestLoRALinear:
    """Tests for LoRALinear layer."""

    def test_init_valid(self):
        """Test valid initialization."""
        layer = LoRALinear(64, 128, rank=8, alpha=16.0)
        assert layer.in_features == 64
        assert layer.out_features == 128
        assert layer.rank == 8
        assert layer.alpha == 16.0

    def test_init_invalid_rank(self):
        """Test invalid rank raises error."""
        with pytest.raises(ConfigurationError):
            LoRALinear(64, 128, rank=0)
        with pytest.raises(ConfigurationError):
            LoRALinear(64, 128, rank=300)

    def test_init_invalid_alpha(self):
        """Test invalid alpha raises error."""
        with pytest.raises(ConfigurationError):
            LoRALinear(64, 128, alpha=-1.0)

    def test_init_invalid_dropout(self):
        """Test invalid dropout raises error."""
        with pytest.raises(ConfigurationError):
            LoRALinear(64, 128, dropout=1.5)

    def test_forward_2d(self):
        """Test forward with 2D input."""
        layer = LoRALinear(64, 128, rank=8)
        x = mx.random.normal((16, 64))
        out = layer(x)
        assert out.shape == (16, 128)

    def test_forward_3d(self):
        """Test forward with 3D input."""
        layer = LoRALinear(64, 128, rank=8)
        x = mx.random.normal((4, 16, 64))
        out = layer(x)
        assert out.shape == (4, 16, 128)

    def test_merge_weights(self):
        """Test weight merging."""
        layer = LoRALinear(64, 128, rank=8)
        merged = layer.merge_weights()
        assert merged.shape == (128, 64)

    def test_trainable_params(self):
        """Test trainable parameter counting."""
        layer = LoRALinear(64, 128, rank=8)
        params = layer.trainable_parameters()
        assert "A" in params
        assert "B" in params
        assert layer.num_trainable_params() == 8 * 64 + 128 * 8


class TestLoRAEmbedding:
    """Tests for LoRAEmbedding layer."""

    def test_init(self):
        """Test initialization."""
        layer = LoRAEmbedding(1000, 256, rank=8)
        assert layer.num_embeddings == 1000
        assert layer.embedding_dim == 256

    def test_forward(self):
        """Test forward pass."""
        layer = LoRAEmbedding(1000, 256, rank=8)
        indices = mx.array([1, 5, 10, 100])
        out = layer(indices)
        assert out.shape == (4, 256)


class TestValidation:
    """Tests for validation functions."""

    def test_validate_rank_valid(self):
        """Test valid rank."""
        validate_rank(8)
        validate_rank(1)
        validate_rank(256)

    def test_validate_rank_invalid(self):
        """Test invalid rank."""
        with pytest.raises(ConfigurationError):
            validate_rank(0)
        with pytest.raises(ConfigurationError):
            validate_rank(300)
        with pytest.raises(ConfigurationError):
            validate_rank("8")

    def test_validate_alpha_valid(self):
        """Test valid alpha."""
        validate_alpha(16.0)
        validate_alpha(1)

    def test_validate_alpha_invalid(self):
        """Test invalid alpha."""
        with pytest.raises(ConfigurationError):
            validate_alpha(-1.0)
        with pytest.raises(ConfigurationError):
            validate_alpha(0)


class TestOps:
    """Tests for low-level operations."""

    def test_lora_forward_shape(self):
        """Test forward output shape."""
        x = mx.random.normal((4, 16, 64))
        W0 = mx.random.normal((128, 64))
        A = mx.random.normal((8, 64))
        B = mx.random.normal((128, 8))
        out = lora_forward(x, W0, A, B)
        assert out.shape == (4, 16, 128)

    def test_merge_weights_shape(self):
        """Test merge output shape."""
        W0 = mx.random.normal((128, 64))
        A = mx.random.normal((8, 64))
        B = mx.random.normal((128, 8))
        merged = merge_lora_weights(W0, A, B)
        assert merged.shape == (128, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
