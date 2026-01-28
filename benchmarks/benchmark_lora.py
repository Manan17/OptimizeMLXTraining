"""
MetalLoRA - Benchmark Suite

Compares MetalLoRA performance against baseline MLX LoRA implementation.
Measures forward/backward pass latency and memory usage across various
configurations.
"""

import mlx.core as mx
import mlx.nn as nn
import time
import argparse
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from metal_lora import LoRALinear
from metal_lora.ops import lora_forward, lora_backward_efficient


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    batch_size: int
    seq_len: int
    in_features: int
    out_features: int
    rank: int
    alpha: float = 16.0
    warmup_iters: int = 10
    bench_iters: int = 100


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_mb: float
    throughput_samples_per_sec: float


# ============================================================================
# BASELINE IMPLEMENTATION (Standard MLX)
# ============================================================================

class BaselineLoRALinear(nn.Module):
    """Standard LoRA implementation using vanilla MLX ops."""
    
    def __init__(self, in_features, out_features, rank, alpha=16.0, use_fp16=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.use_fp16 = use_fp16
        
        dtype = mx.float16 if use_fp16 else mx.float32
        self.W0 = (mx.random.normal((out_features, in_features)) * 0.02).astype(dtype)
        self.A = (mx.random.normal((rank, in_features)) * 0.02).astype(dtype)
        self.B = mx.zeros((out_features, rank), dtype=dtype)
    
    def __call__(self, x):
        # Ensure same dtype as weights for fair comparison
        if self.use_fp16:
            x = x.astype(mx.float16)
        
        # Standard separate operations
        W0x = mx.matmul(x, self.W0.T)
        Ax = mx.matmul(x, self.A.T)
        BAx = mx.matmul(Ax, self.B.T)
        scale = self.alpha / self.rank
        return W0x + scale * BAx


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_forward(layer: nn.Module, x: mx.array, config: BenchmarkConfig) -> float:
    """Benchmark forward pass latency."""
    # Warmup
    for _ in range(config.warmup_iters):
        out = layer(x)
        mx.eval(out)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        out = layer(x)
        mx.eval(out)
    end = time.perf_counter()
    
    return (end - start) / config.bench_iters * 1000  # ms


def benchmark_backward(
    x: mx.array,
    W0: mx.array,
    A: mx.array,
    B: mx.array,
    config: BenchmarkConfig,
    use_optimized: bool = True,
    use_custom_kernel: bool = True,
) -> float:
    """Benchmark backward pass latency."""
    # Create grad_output - use FP16 to match forward pass
    grad_h = mx.random.normal((config.batch_size, config.seq_len, config.out_features)).astype(mx.float16)
    
    # Warmup
    for _ in range(config.warmup_iters):
        if use_optimized:
            grad_A, grad_B = lora_backward_efficient(
                grad_h, x, A, B, config.alpha, use_custom_kernel=use_custom_kernel
            )
        else:
            # Baseline: separate operations
            scale = config.alpha / config.rank
            Ax = mx.matmul(x, A.T)
            Bt_grad = mx.matmul(grad_h, B)
            grad_B = scale * mx.sum(mx.matmul(grad_h.transpose(0, 2, 1), Ax), axis=0)
            grad_A = scale * mx.sum(mx.matmul(Bt_grad.transpose(0, 2, 1), x), axis=0)
        mx.eval(grad_A, grad_B)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        if use_optimized:
            grad_A, grad_B = lora_backward_efficient(
                grad_h, x, A, B, config.alpha, use_custom_kernel=use_custom_kernel
            )
        else:
            scale = config.alpha / config.rank
            Ax = mx.matmul(x, A.T)
            Bt_grad = mx.matmul(grad_h, B)
            grad_B = scale * mx.sum(mx.matmul(grad_h.transpose(0, 2, 1), Ax), axis=0)
            grad_A = scale * mx.sum(mx.matmul(Bt_grad.transpose(0, 2, 1), x), axis=0)
        mx.eval(grad_A, grad_B)
    end = time.perf_counter()
    
    return (end - start) / config.bench_iters * 1000  # ms


def check_accuracy(config: BenchmarkConfig):
    """Check numerical accuracy between MetalLoRA custom kernels and baseline MLX implementation."""
    mx.random.seed(123)

    # Shared weights and inputs (FP16) so we're only measuring kernel differences
    x = mx.random.normal((config.batch_size, config.seq_len, config.in_features)).astype(mx.float16)
    W0 = (mx.random.normal((config.out_features, config.in_features)) * 0.02).astype(mx.float16)
    A = (mx.random.normal((config.rank, config.in_features)) * 0.02).astype(mx.float16)
    B = mx.zeros((config.out_features, config.rank), dtype=mx.float16)
    mx.eval(x, W0, A, B)

    # Forward: MetalLoRA custom kernel vs BaselineLoRALinear
    # Create MetalLoRA layer with custom kernels
    metal_layer = LoRALinear(
        in_features=config.in_features,
        out_features=config.out_features,
        rank=config.rank,
        alpha=config.alpha,
    )
    # Set weights to match shared weights
    metal_layer.W0 = W0
    metal_layer.A = A
    metal_layer.B = B
    y_kernel = metal_layer(x)
    
    # Create BaselineLoRALinear layer
    baseline_layer = BaselineLoRALinear(
        in_features=config.in_features,
        out_features=config.out_features,
        rank=config.rank,
        alpha=config.alpha,
        use_fp16=True,
    )
    # Set weights to match shared weights
    baseline_layer.W0 = W0
    baseline_layer.A = A
    baseline_layer.B = B
    y_baseline = baseline_layer(x)
    
    mx.eval(y_kernel, y_baseline)

    diff_fwd = (y_kernel.astype(mx.float32) - y_baseline.astype(mx.float32))
    fwd_max_err = float(mx.max(mx.abs(diff_fwd)))
    fwd_mean_err = float(mx.mean(mx.abs(diff_fwd)))

    # Backward: custom Metal kernel vs baseline MLX matmul path
    grad_h = mx.random.normal((config.batch_size, config.seq_len, config.out_features)).astype(mx.float16)

    # Custom kernel backward
    grad_A_kernel, grad_B_kernel = lora_backward_efficient(
        grad_h, x, A, B, config.alpha, use_custom_kernel=True,
    )
    
    # Baseline backward: use the same function but with use_custom_kernel=False
    # (this calls the pure-MLX matmul implementation inside metal_lora.ops)
    grad_A_baseline, grad_B_baseline = lora_backward_efficient(
        grad_h, x, A, B, config.alpha, use_custom_kernel=False,
    )
    
    mx.eval(grad_A_kernel, grad_B_kernel, grad_A_baseline, grad_B_baseline)

    diff_grad_A = grad_A_kernel.astype(mx.float32) - grad_A_baseline.astype(mx.float32)
    diff_grad_B = grad_B_kernel.astype(mx.float32) - grad_B_baseline.astype(mx.float32)

    bwd_max_err_A = float(mx.max(mx.abs(diff_grad_A)))
    bwd_mean_err_A = float(mx.mean(mx.abs(diff_grad_A)))
    bwd_max_err_B = float(mx.max(mx.abs(diff_grad_B)))
    bwd_mean_err_B = float(mx.mean(mx.abs(diff_grad_B)))

    print("\nAccuracy check (MetalLoRA custom kernel vs Baseline MLX):")
    print(f"Forward  max |Δ|: {fwd_max_err:.3e}, mean |Δ|: {fwd_mean_err:.3e}")
    print(f"Backward grad_A max |Δ|: {bwd_max_err_A:.3e}, mean |Δ|: {bwd_mean_err_A:.3e}")
    print(f"Backward grad_B max |Δ|: {bwd_max_err_B:.3e}, mean |Δ|: {bwd_mean_err_B:.3e}")


def run_benchmark(config: BenchmarkConfig, compare: bool = True) -> Dict[str, BenchmarkResult]:
    """Run benchmark for both MetalLoRA and baseline."""
    mx.random.seed(42)
    
    # Create input - use FP16 to match MetalLoRA custom kernel for fair comparison
    x = mx.random.normal((config.batch_size, config.seq_len, config.in_features)).astype(mx.float16)
    mx.eval(x)
    
    results = {}
    
    # MetalLoRA - ensure FP16 for fair comparison
    metal_layer = LoRALinear(
        config.in_features,
        config.out_features,
        rank=config.rank,
        alpha=config.alpha,
    )
    metal_layer.W0 = (mx.random.normal((config.out_features, config.in_features)) * 0.02).astype(mx.float16)
    metal_layer.A = metal_layer.A.astype(mx.float16)
    metal_layer.B = metal_layer.B.astype(mx.float16)
    mx.eval(metal_layer.W0, metal_layer.A, metal_layer.B)
    
    fwd_time = benchmark_forward(metal_layer, x, config)
    bwd_time = benchmark_backward(x, metal_layer.W0, metal_layer.A, metal_layer.B, config, use_optimized=True, use_custom_kernel=True)
    
    total_samples = config.batch_size * config.seq_len * config.bench_iters
    throughput = total_samples / ((fwd_time + bwd_time) * config.bench_iters / 1000)
    
    results["metal_lora"] = BenchmarkResult(
        config=config,
        forward_time_ms=fwd_time,
        backward_time_ms=bwd_time,
        total_time_ms=fwd_time + bwd_time,
        memory_mb=0,  # TODO: implement memory tracking
        throughput_samples_per_sec=throughput,
    )
    
    if compare:
        # Option 1: Baseline - naive MLX implementation (separate mx.matmul calls)
        baseline_layer = BaselineLoRALinear(
            config.in_features,
            config.out_features,
            rank=config.rank,
            alpha=config.alpha,
            use_fp16=True,  # Match MetalLoRA's FP16 usage
        )
        mx.eval(baseline_layer.W0, baseline_layer.A, baseline_layer.B)
        
        fwd_time = benchmark_forward(baseline_layer, x, config)
        bwd_time = benchmark_backward(x, baseline_layer.W0, baseline_layer.A, baseline_layer.B, config, use_optimized=False, use_custom_kernel=False)
        
        throughput = total_samples / ((fwd_time + bwd_time) * config.bench_iters / 1000)
        
        results["baseline"] = BenchmarkResult(
            config=config,
            forward_time_ms=fwd_time,
            backward_time_ms=bwd_time,
            total_time_ms=fwd_time + bwd_time,
            memory_mb=0,
            throughput_samples_per_sec=throughput,
        )
        
        # Option 2: MetalLoRA MLX fallback (same codebase, no custom kernel)
        # This shows pure kernel optimization benefit
        try:
            from metal_lora.ops import lora_forward
            
            # Benchmark forward with MLX fallback
            def mlx_forward_fn(x_in):
                return lora_forward(x_in, metal_layer.W0, metal_layer.A, metal_layer.B, 
                                  alpha=config.alpha, use_custom_kernel=False)
            
            # Warmup
            for _ in range(config.warmup_iters):
                out = mlx_forward_fn(x)
                mx.eval(out)
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(config.bench_iters):
                out = mlx_forward_fn(x)
                mx.eval(out)
            end = time.perf_counter()
            fwd_time_mlx = (end - start) / config.bench_iters * 1000
            
            # Backward with MLX fallback (no custom kernel)
            bwd_time_mlx = benchmark_backward(x, metal_layer.W0, metal_layer.A, metal_layer.B, 
                                             config, use_optimized=True, use_custom_kernel=False)
            
            throughput_mlx = total_samples / ((fwd_time_mlx + bwd_time_mlx) * config.bench_iters / 1000)
            
            results["metal_lora_mlx"] = BenchmarkResult(
                config=config,
                forward_time_ms=fwd_time_mlx,
                backward_time_ms=bwd_time_mlx,
                total_time_ms=fwd_time_mlx + bwd_time_mlx,
                memory_mb=0,
                throughput_samples_per_sec=throughput_mlx,
            )
        except Exception as e:
            print(f"Warning: Could not benchmark MetalLoRA MLX fallback: {e}")
    
    return results


def print_results(results: Dict[str, BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    config = results["metal_lora"].config
    
    print("\n" + "=" * 80)
    print(f"Configuration: batch={config.batch_size}, seq={config.seq_len}, "
          f"dim={config.in_features}, rank={config.rank}")
    print("=" * 80)
    print(f"{'Implementation':<20} {'Forward (ms)':<15} {'Backward (ms)':<15} "
          f"{'Total (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    baseline_total = results.get("baseline", results["metal_lora"]).total_time_ms
    
    for name, result in results.items():
        speedup = baseline_total / result.total_time_ms if result.total_time_ms > 0 else 0
        speedup_str = f"{speedup:.2f}x" if name != "baseline" else "1.00x"
        
        print(f"{name:<20} {result.forward_time_ms:<15.3f} {result.backward_time_ms:<15.3f} "
              f"{result.total_time_ms:<15.3f} {speedup_str:<10}")
    
    print("=" * 80)


def run_benchmark_suite():
    """Run full benchmark suite with various configurations."""
    configs = [
        # Small model (testing)
        BenchmarkConfig(batch_size=1, seq_len=128, in_features=768, out_features=768, rank=8),
        BenchmarkConfig(batch_size=1, seq_len=128, in_features=768, out_features=768, rank=16),
        
        # Medium model (7B scale projections)
        BenchmarkConfig(batch_size=4, seq_len=512, in_features=4096, out_features=4096, rank=8),
        BenchmarkConfig(batch_size=4, seq_len=512, in_features=4096, out_features=4096, rank=16),
        BenchmarkConfig(batch_size=4, seq_len=512, in_features=4096, out_features=4096, rank=32),
        BenchmarkConfig(batch_size=4, seq_len=512, in_features=4096, out_features=4096, rank=64),
        
        # Large batch
        BenchmarkConfig(batch_size=8, seq_len=512, in_features=4096, out_features=4096, rank=16),
        
        # Long sequence
        BenchmarkConfig(batch_size=2, seq_len=2048, in_features=4096, out_features=4096, rank=16),
    ]
    
    print("\n" + "=" * 80)
    print("MetalLoRA Benchmark Suite")
    print("=" * 80)
    
    all_results = []
    
    for config in configs:
        try:
            results = run_benchmark(config, compare=True)
            print_results(results)
            all_results.append(results)
        except Exception as e:
            print(f"Error benchmarking config {config}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_speedups = []
    for results in all_results:
        if "baseline" in results:
            speedup = results["baseline"].total_time_ms / results["metal_lora"].total_time_ms
            total_speedups.append(speedup)
    
    if total_speedups:
        avg_speedup = sum(total_speedups) / len(total_speedups)
        max_speedup = max(total_speedups)
        min_speedup = min(total_speedups)
        
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Max Speedup:     {max_speedup:.2f}x")
        print(f"Min Speedup:     {min_speedup:.2f}x")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="MetalLoRA Benchmark Suite")
    parser.add_argument("--compare-baseline", action="store_true", default=True,
                        help="Compare against baseline implementation")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for single benchmark")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for single benchmark")
    parser.add_argument("--dim", type=int, default=4096,
                        help="Model dimension for single benchmark")
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank for single benchmark")
    parser.add_argument("--full-suite", action="store_true",
                        help="Run full benchmark suite")
    
    args = parser.parse_args()
    
    if args.full_suite:
        run_benchmark_suite()
    else:
        config = BenchmarkConfig(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            in_features=args.dim,
            out_features=args.dim,
            rank=args.rank,
        )
        results = run_benchmark(config, compare=args.compare_baseline)
        print_results(results)


if __name__ == "__main__":
    main()
