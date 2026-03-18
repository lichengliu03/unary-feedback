#!/usr/bin/env python3
"""Plot conditional success rates."""

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


def plot_conditional_success(json_file, output_file):
    """Plot conditional success rates."""

    with open(json_file, 'r') as f:
        data = json.load(f)

    data_1turn = data['1turn']
    data_5turn = data['5turn']
    data_base = data.get('base', None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = ['Succ@2|fail@1', 'Succ@3|fail@2', 'Succ@4|fail@3', 'Succ@5|fail@4']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Plot each step
        steps = [50, 100, 150, 200]
        x_positions = ['Base'] + [f'Step {s}' for s in steps]
        x_pos = range(len(x_positions))

        # Bar width
        width = 0.25

        # Plot base model first if available
        if data_base and 'base_model' in data_base:
            val_base = data_base['base_model']['conditional_success'].get(metric)
            if val_base is not None:
                ax.bar(0, val_base * 100, width*2,
                       color='gray', alpha=0.7, label='Base Model' if idx == 0 else '')

        # Plot training steps
        for step in steps:
            step_key = f"step_{step}"

            if step_key in data_1turn and step_key in data_5turn:
                val_1turn = data_1turn[step_key]['conditional_success'].get(metric)
                val_5turn = data_5turn[step_key]['conditional_success'].get(metric)

                if val_1turn is not None and val_5turn is not None:
                    # Get position for this step (offset by 1 for base model)
                    step_idx = steps.index(step) + 1
                    x = step_idx

                    # Plot bars side by side
                    ax.bar(x - width/2, val_1turn * 100, width,
                           color=STEP_COLORS[step], alpha=0.5,
                           label=f'1-Turn Step {step}' if idx == 0 else '')
                    ax.bar(x + width/2, val_5turn * 100, width,
                           color=STEP_COLORS[step], alpha=1.0,
                           label=f'5-Turn Step {step}' if idx == 0 else '')

        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Success Rate (%)', fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_positions, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Only show legend on the first subplot
        if idx == 0:
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = []
            if data_base:
                legend_elements.append(Patch(facecolor='gray', alpha=0.7, label='Base Model'))
            for step in steps:
                legend_elements.append(Patch(facecolor=STEP_COLORS[step],
                                            alpha=0.5, label=f'1-Turn Step {step}'))
            for step in steps:
                legend_elements.append(Patch(facecolor=STEP_COLORS[step],
                                            alpha=1.0, label=f'5-Turn Step {step}'))
            ax.legend(handles=legend_elements, loc='upper left', fontsize=7, ncol=2)

    plt.suptitle('Conditional Success Rates: Base Model vs 1-Turn vs 5-Turn Training',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")


def plot_conditional_success_comparison(json_file, output_file):
    """Plot conditional success rate comparison as line plots."""

    with open(json_file, 'r') as f:
        data = json.load(f)

    data_1turn = data['1turn']
    data_5turn = data['5turn']
    data_base = data.get('base', None)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    steps = [50, 100, 150, 200]
    metrics = ['Succ@2|fail@1', 'Succ@3|fail@2', 'Succ@4|fail@3', 'Succ@5|fail@4']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        vals_1turn = []
        vals_5turn = []

        for step in steps:
            step_key = f"step_{step}"
            if step_key in data_1turn and step_key in data_5turn:
                val_1turn = data_1turn[step_key]['conditional_success'].get(metric)
                val_5turn = data_5turn[step_key]['conditional_success'].get(metric)

                if val_1turn is not None and val_5turn is not None:
                    vals_1turn.append(val_1turn * 100)
                    vals_5turn.append(val_5turn * 100)
                else:
                    vals_1turn.append(0)
                    vals_5turn.append(0)

        # Plot base model as horizontal line if available
        if data_base and 'base_model' in data_base:
            val_base = data_base['base_model']['conditional_success'].get(metric)
            if val_base is not None:
                ax.axhline(y=val_base * 100, color='gray', linestyle=':',
                          linewidth=2, label='Base Model', alpha=0.8)

        ax.plot(steps, vals_1turn, marker='o', label='1-Turn',
                linewidth=2, markersize=8, linestyle='--', color='#1f77b4')
        ax.plot(steps, vals_5turn, marker='s', label='5-Turn',
                linewidth=2, markersize=8, linestyle='-', color='#d62728')

        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel('Success Rate (%)', fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(steps)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Conditional Success Rates: Reflection Ability Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_conditional_success.py <conditional_success_json>")
        print("Example: python plot_conditional_success.py results/conditional_success_rates.json")
        sys.exit(1)

    json_file = sys.argv[1]

    # Check file exists
    if not Path(json_file).exists():
        print(f"Error: File not found: {json_file}")
        sys.exit(1)

    # Create output directory
    output_dir = Path('eval_results')
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_conditional_success(json_file, output_dir / 'conditional_success_bars.png')
    plot_conditional_success_comparison(json_file, output_dir / 'conditional_success_lines.png')


if __name__ == '__main__':
    main()
