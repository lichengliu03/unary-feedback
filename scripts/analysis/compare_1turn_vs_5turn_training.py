#!/usr/bin/env python3
"""Compare 1-turn vs 5-turn training at specific steps."""

import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def plot_comparison(json_1turn, json_5turn, steps, output_file):
    """Plot comparison of 1-turn vs 5-turn at specific steps."""

    # Load data
    with open(json_1turn, 'r') as f:
        data_1turn = json.load(f)

    with open(json_5turn, 'r') as f:
        data_5turn = json.load(f)

    # Create 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # First, collect all accuracies to determine global y-axis range
    all_accuracies = []
    for step in steps:
        step_key = f"step_{step}"

        if step_key in data_1turn and 'pass_at_k' in data_1turn[step_key]:
            for k_str, acc in data_1turn[step_key]['pass_at_k'].items():
                all_accuracies.append(acc * 100)

        if step_key in data_5turn and 'pass_at_k' in data_5turn[step_key]:
            for k_str, acc in data_5turn[step_key]['pass_at_k'].items():
                all_accuracies.append(acc * 100)

    # Calculate global y-axis range
    if all_accuracies:
        y_min = min(all_accuracies)
        y_max = max(all_accuracies)
        y_range = y_max - y_min
        margin = max(2, y_range * 0.1)
        global_ylim = [y_min - margin, y_max + margin]
    else:
        global_ylim = [0, 100]

    # Plot each step in a separate subplot
    for idx, step in enumerate(steps):
        ax = axes[idx]
        step_key = f"step_{step}"

        # Plot 1-turn
        if step_key in data_1turn and 'pass_at_k' in data_1turn[step_key]:
            k_values = []
            accuracies = []
            for k_str, acc in sorted(data_1turn[step_key]['pass_at_k'].items(),
                                    key=lambda x: int(x[0].split('@')[1])):
                k = int(k_str.split('@')[1])
                k_values.append(k)
                accuracies.append(acc * 100)

            ax.plot(k_values, accuracies, marker='o', label='1-Turn',
                    linewidth=2, markersize=8, linestyle='--', color='#1f77b4')

        # Plot 5-turn
        if step_key in data_5turn and 'pass_at_k' in data_5turn[step_key]:
            k_values = []
            accuracies = []
            for k_str, acc in sorted(data_5turn[step_key]['pass_at_k'].items(),
                                    key=lambda x: int(x[0].split('@')[1])):
                k = int(k_str.split('@')[1])
                k_values.append(k)
                accuracies.append(acc * 100)

            ax.plot(k_values, accuracies, marker='s', label='5-Turn',
                    linewidth=2, markersize=8, linestyle='-', color='#d62728')

        # Set common properties
        ax.set_ylim(global_ylim)
        ax.set_xlabel('Number of Turns (k)', fontsize=11)
        ax.set_ylabel('Pass@k Accuracy (%)', fontsize=11)
        ax.set_title(f'Step {step}', fontsize=12, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('1-Turn vs 5-Turn Training Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_comparison.py <1turn_json> <5turn_json> [steps...]")
        print("Example: python plot_comparison.py results/qwen_1turn_new.json results/qwen_5turn_new.json 100 200")
        sys.exit(1)

    json_1turn = sys.argv[1]
    json_5turn = sys.argv[2]

    # Default to steps 100 and 200 if not specified
    if len(sys.argv) > 3:
        steps = [int(s) for s in sys.argv[3:]]
    else:
        steps = [100, 200]

    # Check files exist
    if not Path(json_1turn).exists():
        print(f"Error: File not found: {json_1turn}")
        sys.exit(1)
    if not Path(json_5turn).exists():
        print(f"Error: File not found: {json_5turn}")
        sys.exit(1)

    # Create output directory
    output_dir = Path('eval_results')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'comparison_1turn_vs_5turn.png'
    plot_comparison(json_1turn, json_5turn, steps, output_file)

    # Print summary
    print("\nComparison Summary:")
    print("-" * 80)

    with open(json_1turn, 'r') as f:
        data_1turn = json.load(f)
    with open(json_5turn, 'r') as f:
        data_5turn = json.load(f)

    for step in steps:
        step_key = f"step_{step}"
        print(f"\nStep {step}:")

        if step_key in data_1turn:
            print(f"  1-Turn: {data_1turn[step_key]['avg_success']:.2%} success")
            if 'pass_at_k' in data_1turn[step_key]:
                for k in [1, 2, 3, 4, 5]:
                    k_str = f'pass@{k}'
                    if k_str in data_1turn[step_key]['pass_at_k']:
                        print(f"    {k_str}: {data_1turn[step_key]['pass_at_k'][k_str]:.2%}")

        if step_key in data_5turn:
            print(f"  5-Turn: {data_5turn[step_key]['avg_success']:.2%} success")
            if 'pass_at_k' in data_5turn[step_key]:
                for k in [1, 2, 3, 4, 5]:
                    k_str = f'pass@{k}'
                    if k_str in data_5turn[step_key]['pass_at_k']:
                        print(f"    {k_str}: {data_5turn[step_key]['pass_at_k'][k_str]:.2%}")

if __name__ == "__main__":
    main()
