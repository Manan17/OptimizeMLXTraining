#!/usr/bin/env python3
"""Plot baseline vs CCE loss curves from benchmark data."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Load the JSON data
    json_path = Path(__file__).parent / 'benchmark_step_losses.json'
    with open(json_path, 'r') as f:
        data = json.load(f)

    runs = data['runs']
    n_runs = len(runs)

    # Create a figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, run in enumerate(runs):
        ax = axes[idx]
        steps = np.arange(1, len(run['baseline_losses']) + 1)

        ax.plot(steps, run['baseline_losses'], label='Baseline', alpha=0.8, linewidth=1.5)
        ax.plot(steps, run['cce_losses'], label='CCE', alpha=0.8, linewidth=1.5)

        ax.set_title(f"{run['model']}\nbatch={run['batch']}, seq={run['seq']}, N={run['N']}", fontsize=10)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Hide any unused subplots
    for idx in range(n_runs, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Baseline vs CCE Training Loss Curves\n(100 steps each, Adafactor lr=1e-05, Alpaca dataset)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save the figure
    output_path = Path(__file__).parent / 'loss_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Also create a zoomed version showing the difference more clearly
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
    axes2 = axes2.flatten()

    for idx, run in enumerate(runs):
        ax = axes2[idx]
        steps = np.arange(1, len(run['baseline_losses']) + 1)

        baseline = np.array(run['baseline_losses'])
        cce = np.array(run['cce_losses'])
        diff = cce - baseline

        ax.plot(steps, diff, color='purple', alpha=0.8, linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.fill_between(steps, diff, 0, alpha=0.3, color='purple')

        mean_diff = np.mean(diff)
        ax.axhline(y=mean_diff, color='red', linestyle=':', alpha=0.7, label=f'Mean: {mean_diff:.4f}')

        ax.set_title(f"{run['model']}\nbatch={run['batch']}, seq={run['seq']}, N={run['N']}", fontsize=10)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('CCE - Baseline Loss', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for idx in range(n_runs, len(axes2)):
        axes2[idx].set_visible(False)

    plt.suptitle('Loss Difference (CCE - Baseline)\nPositive = CCE higher loss, Negative = CCE lower loss',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path2 = Path(__file__).parent / 'loss_difference.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved difference plot to {output_path2}")


if __name__ == '__main__':
    main()
