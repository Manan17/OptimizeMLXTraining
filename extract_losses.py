#!/usr/bin/env python3
"""Extract step-by-step losses from benchmark logs into JSON."""

import json
import re
from pathlib import Path

def parse_log_file(filepath):
    """Parse a benchmark log file and extract step-by-step losses."""
    results = []
    current_model = None
    current_config = None
    current_baseline_losses = []
    current_cce_losses = []
    in_baseline = False
    in_cce = False

    with open(filepath, 'r') as f:
        for line in f:
            # Match model header
            model_match = re.match(r'^MODEL: (.+) \((.+)\)', line)
            if model_match:
                current_model = model_match.group(1)
                continue

            # Match config line (not skipped)
            config_match = re.match(r'^\s+\[(\d+)/54\] batch=(\d+), seq=(\d+), N=(\d+)\s*$', line)
            if config_match:
                # Save previous config if we have data
                if current_config and (current_baseline_losses or current_cce_losses):
                    results.append({
                        'model': current_config['model'],
                        'config_num': current_config['config_num'],
                        'batch': current_config['batch'],
                        'seq': current_config['seq'],
                        'N': current_config['N'],
                        'baseline_losses': current_baseline_losses,
                        'cce_losses': current_cce_losses
                    })

                current_config = {
                    'model': current_model,
                    'config_num': int(config_match.group(1)),
                    'batch': int(config_match.group(2)),
                    'seq': int(config_match.group(3)),
                    'N': int(config_match.group(4))
                }
                current_baseline_losses = []
                current_cce_losses = []
                in_baseline = False
                in_cce = False
                continue

            # Check for baseline/cce section start
            if 'Running BASELINE...' in line:
                in_baseline = True
                in_cce = False
                continue
            if 'Running CCE...' in line:
                in_baseline = False
                in_cce = True
                continue

            # Match step losses
            baseline_step = re.match(r'.*\[BASELINE\] step\s+(\d+)/100: loss=(\d+\.\d+)', line)
            if baseline_step and in_baseline:
                current_baseline_losses.append(float(baseline_step.group(2)))
                continue

            cce_step = re.match(r'.*\[CCE\] step\s+(\d+)/100: loss=(\d+\.\d+)', line)
            if cce_step and in_cce:
                current_cce_losses.append(float(cce_step.group(2)))
                continue

            # End of CCE section
            if 'Computing gradient diff' in line:
                in_cce = False

    # Don't forget the last config
    if current_config and (current_baseline_losses or current_cce_losses):
        results.append({
            'model': current_config['model'],
            'config_num': current_config['config_num'],
            'batch': current_config['batch'],
            'seq': current_config['seq'],
            'N': current_config['N'],
            'baseline_losses': current_baseline_losses,
            'cce_losses': current_cce_losses
        })

    return results


def main():
    log_files = [
        'benchmark_resumed3.log',
        'benchmark_resumed4.log'
    ]

    all_results = []
    seen_configs = set()

    for log_file in log_files:
        filepath = Path(__file__).parent / log_file
        if filepath.exists():
            results = parse_log_file(filepath)
            for r in results:
                # Only add if we have both baseline and CCE losses
                config_key = (r['model'], r['config_num'])
                if config_key not in seen_configs and r['baseline_losses'] and r['cce_losses']:
                    seen_configs.add(config_key)
                    all_results.append(r)

    # Sort by config number
    all_results.sort(key=lambda x: x['config_num'])

    # Create output structure
    output = {
        'description': 'Step-by-step training losses for Baseline vs CCE benchmark',
        'config': {
            'dataset': 'tatsu-lab/alpaca',
            'optimizer': 'Adafactor (lr=1e-05)',
            'warmup_steps': 15,
            'measure_steps': 100,
            'seed': 42,
            'dtype': 'bfloat16'
        },
        'runs': all_results
    }

    output_path = Path(__file__).parent / 'benchmark_step_losses.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Extracted {len(all_results)} configs with step-by-step losses")
    for r in all_results:
        print(f"  Config {r['config_num']}: {r['model']} batch={r['batch']} seq={r['seq']} N={r['N']}")
        print(f"    Baseline: {len(r['baseline_losses'])} steps, CCE: {len(r['cce_losses'])} steps")


if __name__ == '__main__':
    main()
