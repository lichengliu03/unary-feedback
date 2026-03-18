#!/usr/bin/env python3
"""
Comprehensive comparison of different feedback types: Normal, Critique, and No Feedback.

This script analyzes and visualizes:
- Absolute success rates (Pass@k)
- Marginal gains per attempt
- Expected number of attempts to succeed
- Cumulative improvement over baseline
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


def compute_metrics(pass_at_k):
    """
    Compute various metrics from pass@k values.

    Returns:
        - conditional_success: Succ@k|fail@(k-1)
        - marginal_gain: pass@k - pass@(k-1)
        - expected_attempts: E[number of attempts needed to succeed]
    """
    conditional = {}
    marginal = {}

    for k in [2, 3, 4, 5]:
        pass_k = pass_at_k.get(f'pass@{k}', None)
        pass_k_minus_1 = pass_at_k.get(f'pass@{k-1}', None)

        if pass_k is not None and pass_k_minus_1 is not None:
            # Marginal gain
            marginal[k] = pass_k - pass_k_minus_1

            # Conditional success
            fail_k_minus_1 = 1 - pass_k_minus_1
            if fail_k_minus_1 > 0:
                conditional[k] = (pass_k - pass_k_minus_1) / fail_k_minus_1

    # Expected number of attempts
    # E[attempts] = sum(k * P(succeed at exactly turn k)) + 5 * P(fail all 5 attempts)
    expected_attempts = 0

    # Turn 1: succeed at first try
    prob_succeed_at_1 = pass_at_k['pass@1']
    expected_attempts += 1 * prob_succeed_at_1

    # Turns 2-5: succeed at turn k (but not before)
    for k in [2, 3, 4, 5]:
        prob_succeed_at_k = pass_at_k[f'pass@{k}'] - pass_at_k[f'pass@{k-1}']
        expected_attempts += k * prob_succeed_at_k

    # Failed all 5 attempts (count as 5 attempts used)
    prob_fail_all = 1 - pass_at_k['pass@5']
    expected_attempts += 5 * prob_fail_all

    return conditional, marginal, expected_attempts


def plot_comprehensive_comparison(normal_passk, critique_passk, no_feedback_passk, output_dir):
    """
    Create a comprehensive comparison with four key metrics.
    """
    # Compute metrics
    normal_cond, normal_marg, normal_expected = compute_metrics(normal_passk)
    critique_cond, critique_marg, critique_expected = compute_metrics(critique_passk)
    no_feedback_cond, no_feedback_marg, no_feedback_expected = compute_metrics(no_feedback_passk)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comprehensive Feedback Mechanism Comparison\nQwen2.5-3B on MetaMathQA',
                 fontsize=16, fontweight='bold', y=0.995)

    k_values = [1, 2, 3, 4, 5]
    k_values_cond = [2, 3, 4, 5]

    colors = {
        'normal': '#2E86AB',
        'critique': '#A23B72',
        'no_feedback': '#F18F01'
    }

    # Subplot A: Absolute Pass@k
    ax1 = axes[0, 0]
    normal_pass = [normal_passk[f'pass@{k}'] for k in k_values]
    critique_pass = [critique_passk[f'pass@{k}'] for k in k_values]
    no_feedback_pass = [no_feedback_passk[f'pass@{k}'] for k in k_values]

    ax1.plot(k_values, normal_pass, marker='o', linewidth=2.5, markersize=8,
             label='Normal Feedback', color=colors['normal'])
    ax1.plot(k_values, critique_pass, marker='s', linewidth=2.5, markersize=8,
             label='Critique Feedback', color=colors['critique'])
    ax1.plot(k_values, no_feedback_pass, marker='^', linewidth=2.5, markersize=8,
             label='No Feedback', color=colors['no_feedback'])

    ax1.set_xlabel('Number of Attempts (k)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Pass@k (Cumulative Success Rate)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Absolute Success Rate Over Attempts', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax1.set_xticks(k_values)
    ax1.set_ylim(0.2, 0.7)

    # Subplot B: Marginal Gain
    ax2 = axes[0, 1]
    normal_marg_vals = [normal_marg.get(k, 0) for k in k_values_cond]
    critique_marg_vals = [critique_marg.get(k, 0) for k in k_values_cond]
    no_feedback_marg_vals = [no_feedback_marg.get(k, 0) for k in k_values_cond]

    x = np.arange(len(k_values_cond))
    width = 0.25

    ax2.bar(x - width, normal_marg_vals, width, label='Normal Feedback',
            color=colors['normal'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.bar(x, critique_marg_vals, width, label='Critique Feedback',
            color=colors['critique'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.bar(x + width, no_feedback_marg_vals, width, label='No Feedback',
            color=colors['no_feedback'], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Attempt Number (k)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Marginal Gain\n(pass@k - pass@k-1)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Marginal Gain per Additional Attempt', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{k}' for k in k_values_cond])
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Subplot C: Expected Number of Attempts
    ax3 = axes[1, 0]

    methods = ['Normal\nFeedback', 'Critique\nFeedback', 'No\nFeedback']
    expected_values = [normal_expected, critique_expected, no_feedback_expected]
    bar_colors = [colors['normal'], colors['critique'], colors['no_feedback']]

    bars = ax3.bar(methods, expected_values, color=bar_colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for bar, val in zip(bars, expected_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax3.set_ylabel('Expected Number of Attempts', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Expected Attempts Until Success\n(Lower is Better)',
                  fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax3.set_ylim(0, max(expected_values) * 1.15)

    # Subplot D: Cumulative Improvement from pass@1
    ax4 = axes[1, 1]
    normal_cumul = [(normal_passk[f'pass@{k}'] - normal_passk['pass@1']) / normal_passk['pass@1'] * 100
                    for k in k_values]
    critique_cumul = [(critique_passk[f'pass@{k}'] - critique_passk['pass@1']) / critique_passk['pass@1'] * 100
                      for k in k_values]
    no_feedback_cumul = [(no_feedback_passk[f'pass@{k}'] - no_feedback_passk['pass@1']) / no_feedback_passk['pass@1'] * 100
                         for k in k_values]

    ax4.plot(k_values, normal_cumul, marker='o', linewidth=2.5, markersize=8,
             label='Normal Feedback', color=colors['normal'])
    ax4.plot(k_values, critique_cumul, marker='s', linewidth=2.5, markersize=8,
             label='Critique Feedback', color=colors['critique'])
    ax4.plot(k_values, no_feedback_cumul, marker='^', linewidth=2.5, markersize=8,
             label='No Feedback', color=colors['no_feedback'])

    ax4.set_xlabel('Number of Attempts (k)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Relative Improvement from pass@1 (%)', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Cumulative Improvement Over Single Attempt', fontsize=12, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax4.set_xticks(k_values)

    plt.tight_layout()

    output_file = output_dir / 'feedback_comparison_comprehensive_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_file}")

    # Print detailed statistics
    print("\n" + "="*95)
    print(" "*25 + "COMPREHENSIVE COMPARISON STATISTICS")
    print("="*95)

    print("\n1. ABSOLUTE SUCCESS RATES (Pass@k)")
    print("-" * 95)
    print(f"{'Metric':<15} {'Normal':>18} {'Critique':>18} {'No Feedback':>18} {'Winner':>18}")
    print("-" * 95)
    for k in k_values:
        metric = f'pass@{k}'
        vals = {
            'normal': normal_passk[metric],
            'critique': critique_passk[metric],
            'no_feedback': no_feedback_passk[metric]
        }
        best = max(vals, key=vals.get)
        winner_display = best.replace('_', ' ').title()
        print(f"{metric:<15} {vals['normal']:>17.4f}  {vals['critique']:>17.4f}  {vals['no_feedback']:>17.4f}  {winner_display:>18}")

    print("\n2. MARGINAL GAINS (pass@k - pass@k-1)")
    print("-" * 95)
    print(f"{'Turn k':<15} {'Normal':>18} {'Critique':>18} {'No Feedback':>18} {'Winner':>18}")
    print("-" * 95)
    for k in k_values_cond:
        vals = {
            'normal': normal_marg.get(k, 0),
            'critique': critique_marg.get(k, 0),
            'no_feedback': no_feedback_marg.get(k, 0)
        }
        best = max(vals, key=vals.get)
        winner_display = best.replace('_', ' ').title()
        print(f"Turn {k:<10} {vals['normal']:>17.4f}  {vals['critique']:>17.4f}  {vals['no_feedback']:>17.4f}  {winner_display:>18}")

    print("\n3. EXPECTED NUMBER OF ATTEMPTS")
    print("-" * 95)
    print(f"{'Method':<15} {'Expected Attempts':>25} {'Interpretation':>50}")
    print("-" * 95)
    print(f"{'Normal':<15} {normal_expected:>24.2f}  {'Average tries until success':>50}")
    print(f"{'Critique':<15} {critique_expected:>24.2f}  {'Average tries until success':>50}")
    print(f"{'No Feedback':<15} {no_feedback_expected:>24.2f}  {'Average tries until success':>50}")

    # Determine winner (lower is better)
    expected_dict = {
        'Normal': normal_expected,
        'Critique': critique_expected,
        'No Feedback': no_feedback_expected
    }
    best_method = min(expected_dict, key=expected_dict.get)
    print(f"\n{'Winner':<15} {best_method:>24}  {'(Requires fewest attempts)':>50}")

    print("\n4. TOTAL IMPROVEMENT (pass@5 - pass@1)")
    print("-" * 95)
    normal_total = normal_passk['pass@5'] - normal_passk['pass@1']
    critique_total = critique_passk['pass@5'] - critique_passk['pass@1']
    no_feedback_total = no_feedback_passk['pass@5'] - no_feedback_passk['pass@1']

    print(f"Normal Feedback:    +{normal_total:.4f} absolute ({normal_total/normal_passk['pass@1']*100:>5.1f}% relative improvement)")
    print(f"Critique Feedback:  +{critique_total:.4f} absolute ({critique_total/critique_passk['pass@1']*100:>5.1f}% relative improvement)")
    print(f"No Feedback:        +{no_feedback_total:.4f} absolute ({no_feedback_total/no_feedback_passk['pass@1']*100:>5.1f}% relative improvement)")

    print("\n5. KEY INSIGHTS")
    print("-" * 95)
    print(f"• Normal Feedback excels in early attempts (pass@2: {normal_passk['pass@2']:.4f})")
    print(f"• Critique Feedback shows strongest sustained improvement (best pass@5: {critique_passk['pass@5']:.4f})")
    print(f"• No Feedback demonstrates limited improvement capacity")
    print(f"• Critique crosses over Normal at k=4, indicating better handling of difficult cases")

    print("\n" + "="*95 + "\n")

    # Save all data to JSON
    data_file = output_dir / 'feedback_comparison_detailed_metrics.json'
    data = {
        'metadata': {
            'model': 'Qwen2.5-3B-Instruct',
            'dataset': 'MetaMathQA',
            'evaluation_date': '2026-01-15',
            'max_attempts': 5
        },
        'normal_feedback': {
            'pass_at_k': normal_passk,
            'conditional_success': normal_cond,
            'marginal_gain': normal_marg,
            'expected_attempts': normal_expected
        },
        'critique_feedback': {
            'pass_at_k': critique_passk,
            'conditional_success': critique_cond,
            'marginal_gain': critique_marg,
            'expected_attempts': critique_expected
        },
        'no_feedback': {
            'pass_at_k': no_feedback_passk,
            'conditional_success': no_feedback_cond,
            'marginal_gain': no_feedback_marg,
            'expected_attempts': no_feedback_expected
        },
        'summary': {
            'best_final_performance': 'Critique Feedback',
            'best_early_performance': 'Normal Feedback',
            'lowest_expected_attempts': best_method,
            'total_improvements': {
                'normal': normal_total,
                'critique': critique_total,
                'no_feedback': no_feedback_total
            }
        }
    }

    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Data saved to: {data_file}\n")


def main():
    """Main function."""

    # Data from evaluations
    normal_passk = {
        'pass@1': 0.3125,
        'pass@2': 0.5537,
        'pass@3': 0.5996,
        'pass@4': 0.6123,
        'pass@5': 0.6152
    }

    critique_passk = {
        'pass@1': 0.3066,
        'pass@2': 0.4961,
        'pass@3': 0.5879,
        'pass@4': 0.6152,
        'pass@5': 0.6250
    }

    no_feedback_passk = {
        'pass@1': 0.2910,
        'pass@2': 0.4316,
        'pass@3': 0.5137,
        'pass@4': 0.5352,
        'pass@5': 0.5391
    }

    # Output directory
    output_dir = Path('/u/lliu22/unary-feedback/results/feedback_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_comprehensive_comparison(normal_passk, critique_passk, no_feedback_passk, output_dir)


if __name__ == '__main__':
    main()
