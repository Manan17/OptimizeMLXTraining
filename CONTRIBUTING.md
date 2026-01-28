# Contributing to MetalLoRA

## Development Setup

```bash
# Clone
git clone https://github.com/yourusername/MetalLoRA.git
cd MetalLoRA

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- Python: Formatted with ruff
- Metal: 4-space indentation, 100 char line limit
- Commits: Conventional commits (`feat:`, `fix:`, `docs:`, etc.)

## Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feat/my-feature`)
3. Make changes
4. Run tests (`pytest tests/`)
5. Run linter (`ruff check python/`)
6. Commit with conventional format
7. Push and create PR

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_kernels.py -v

# Run with coverage
pytest tests/ --cov=metal_lora
```

## Adding New Kernels

1. Add kernel to `kernels/lora_kernels.metal`
2. Add Python wrapper in `python/metal_lora/ops.py`
3. Add tests in `tests/test_kernels.py`
4. Update documentation

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.1.0`
4. Push: `git push origin v0.1.0`
