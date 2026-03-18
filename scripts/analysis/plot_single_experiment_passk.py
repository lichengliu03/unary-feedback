#!/usr/bin/env python3
"""Plot pass@k metrics from evaluation results."""

import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Define consistent colors for each step
STEP_COLORS = {
    50: '#1f77b4',   # blue
    100: '#ff7f0e',  # orange
    150: '#2ca02c',  # green
    200: '#d62728',  # red
}

def plot_single_experiment(json_file, output_file, title):
    """Plot pass@k metrics for a single experiment."""

    with open(json_file, 'r') as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all accuracies for y-axis range calculation
    all_accuracies = []

    # Plot each checkpoint
    for step_key in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
        step_data = data[step_key]

        if 'pass_at_k' not in step_data:
            print(f"Warning: No pass@k data in {step_key}")
            continue

        # Extract k values and accuracies
        k_values = []
        accuracies = []
        for k_str, acc in sorted(step_data['pass_at_k'].items(),
                                key=lambda x: int(x[0].split('@')[1])):
            k = int(k_str.split('@')[1])
            k_values.append(k)
            accuracies.append(acc * 100)  # Convert to percentage

        all_accuracies.extend(accuracies)

        step_num = int(step_key.split('_')[1])
        color = STEP_COLORS.get(step_num, None)
        label = f"Step {step_num}"
        ax.plot(k_values, accuracies, marker='o', label=label,
                color=color, linewidth=2, markersize=8)

    # Set dynamic y-axis range with margin
    if all_accuracies:
        y_min = min(all_accuracies)
        y_max = max(all_accuracies)
        y_range = y_max - y_min
        margin = max(2, y_range * 0.1)  # At least 2%, or 10% of range
        ax.set_ylim([y_min - margin, y_max + margin])

    ax.set_xlabel('Number of Turns (k)', fontsize=12)
    ax.set_ylabel('Pass@k Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_results.py <1turn_json> <5turn_json>")
        print("Example: python plot_results.py results/qwen_1turn_new.json results/qwen_5turn_new.json")
        sys.exit(1)

    json_1turn = sys.argv[1]
    json_5turn = sys.argv[2]

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

    # Generate two separate plots
    plot_single_experiment(json_1turn, output_dir / 'qwen_1turn_plot.png',
                          'Qwen2.5-3B 1-Turn Training: Pass@k Accuracy')
    plot_single_experiment(json_5turn, output_dir / 'qwen_5turn_plot.png',
                          'Qwen2.5-3B 5-Turn Training: Pass@k Accuracy')

if __name__ == "__main__":
    main()
