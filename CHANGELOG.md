# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-18

### Added
- Initial release
- Core LoRA layers: `LoRALinear`, `LoRAEmbedding`
- Metal kernels: fused forward/backward, simdgroup optimization
- Quantized inference: INT4, NF4, INT8
- Training infrastructure: `LoRATrainer`, `TrainingConfig`
- Advanced optimizations:
  - simdgroup_matrix acceleration
  - Tile memory persistence
  - Cooperative threadgroups
  - Dynamic kernel dispatch
  - Multi-adapter batched inference
  - KV-cache with LoRA
  - Speculative decoding
- Utilities:
  - Weight compression
  - Memory pooling
  - Mixed precision training
  - Gradient accumulation fusion

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A
