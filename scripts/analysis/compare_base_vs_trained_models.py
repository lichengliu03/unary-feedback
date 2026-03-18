#!/usr/bin/env python3
"""
Plot final comparison: Base Model vs 1-Turn Step 200 vs 5-Turn Step 200
"""

import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path


def plot_final_comparison(json_1turn, json_5turn, json_base, output_file):
    """
    Plot comparison of base model vs trained models at step 200.
    Shows pass@k metrics for all three models.
    """

    # Load data
    with open(json_1turn, 'r') as f:
        data_1turn = json.load(f)

    with open(json_5turn, 'r') as f:
        data_5turn = json.load(f)

    with open(json_base, 'r') as f:
        data_base = json.load(f)

    # Extract step 200 data
    step_200_1turn = data_1turn['step_200']
    step_200_5turn = data_5turn['step_200']
    base_model = data_base['base_model']

    # Extract pass@k values
    k_values = [1, 2, 3, 4, 5]

    # Handle both formats: 'pass@1' and '1'
    def get_pass_k_value(data, k):
        pass_k_dict = data['pass@k'] if 'pass@k' in data else data['pass_at_k']
        key = f'pass@{k}' if f'pass@{k}' in pass_k_dict else str(k)
        return pass_k_dict[key] * 100

    base_values = [get_pass_k_value(base_model, k) for k in k_values]
    turn1_values = [get_pass_k_value(step_200_1turn, k) for k in k_values]
    turn5_values = [get_pass_k_value(step_200_5turn, k) for k in k_values]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines
    ax.plot(k_values, base_values, marker='o', label='Base Model (Qwen2.5-3B)',
            color='#808080', linewidth=2.5, markersize=8, linestyle=':')
    ax.plot(k_values, turn1_values, marker='s', label='1-Turn Training (Step 200)',
            color='#1f77b4', linewidth=2.5, markersize=8, linestyle='--')
    ax.plot(k_values, turn5_values, marker='^', label='5-Turn Training (Step 200)',
            color='#d62728', linewidth=2.5, markersize=8, linestyle='-')

    # Add value labels on points
    for i, k in enumerate(k_values):
        ax.text(k, base_values[i] - 3, f'{base_values[i]:.1f}%',
               ha='center', va='top', fontsize=9, color='#808080')
        ax.text(k, turn1_values[i] + 2, f'{turn1_values[i]:.1f}%',
               ha='center', va='bottom', fontsize=9, color='#1f77b4')
        ax.text(k, turn5_values[i] + 2, f'{turn5_values[i]:.1f}%',
               ha='center', va='bottom', fontsize=9, color='#d62728')

    # Customize plot
    ax.set_xlabel('Number of Attempts (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Base Model vs 1-Turn vs 5-Turn Training (Step 200)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(k_values)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(25, 100)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Final comparison plot saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("Final Model Comparison Summary (Step 200)")
    print("="*80)
    print(f"\n{'Metric':<12} {'Base Model':<15} {'1-Turn':<15} {'5-Turn':<15} {'5T vs Base':<15}")
    print("-"*80)

    for i, k in enumerate(k_values):
        improvement = turn5_values[i] - base_values[i]
        print(f"pass@{k:<7} {base_values[i]:>6.2f}%{'':<7} "
              f"{turn1_values[i]:>6.2f}%{'':<7} "
              f"{turn5_values[i]:>6.2f}%{'':<7} "
              f"{improvement:>+6.2f}pp")

    print("\nKey Insights:")
    print(f"  • Base model pass@1: {base_values[0]:.2f}%")
    print(f"  • 1-turn training improves pass@1 to {turn1_values[0]:.2f}% (+{turn1_values[0]-base_values[0]:.2f}pp)")
    print(f"  • 5-turn training improves pass@1 to {turn5_values[0]:.2f}% (+{turn5_values[0]-base_values[0]:.2f}pp)")
    print(f"  • 5-turn training achieves {turn5_values[-1]:.2f}% success with 5 attempts")
    print(f"  • Improvement from base to 5-turn at pass@5: +{turn5_values[-1]-base_values[-1]:.2f}pp")


def main():
    if len(sys.argv) < 4:
        print("Usage: python plot_final_comparison.py <1turn_json> <5turn_json> <base_json>")
        print("Example: python plot_final_comparison.py results/qwen_1turn_new.json results/qwen_5turn_new.json results/qwen25_3b_base.json")
        sys.exit(1)

    json_1turn = sys.argv[1]
    json_5turn = sys.argv[2]
    json_base = sys.argv[3]

    # Check files exist
    for f in [json_1turn, json_5turn, json_base]:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)

    # Create output directory
    output_dir = Path('eval_results')
    output_dir.mkdir(exist_ok=True)

    # Generate plot
    output_file = output_dir / 'final_comparison_base_vs_trained.png'
    plot_final_comparison(json_1turn, json_5turn, json_base, output_file)


if __name__ == '__main__':
    main()
