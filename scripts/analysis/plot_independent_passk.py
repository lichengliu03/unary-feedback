#!/usr/bin/env python3
"""
Plot independent pass@k results for 1turn and 5turn models.

Shows how pass@k evolves across training steps for both 1turn and 5turn models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
from typing import Dict, List, Tuple


def extract_model_info(result_dir: Path) -> Tuple[str, str, int]:
    """
    Extract model type (1turn/5turn), checkpoint step, and whether it's base model.

    Returns:
        (model_type, step_name, step_num) where:
        - model_type: '1turn' or '5turn' or 'base'
        - step_name: 'base' or 'step_50' etc
        - step_num: -1 for base, else the step number
    """
    dir_name = result_dir.name

    # Check if base model
    if 'base_model' in dir_name or 'Qwen2.5-3B-Instruct' in dir_name:
        return 'base', 'base', -1

    # Extract turn type
    if '1turn' in dir_name:
        model_type = '1turn'
    elif '5turn' in dir_name:
        model_type = '5turn'
    else:
        return None, None, None

    # Extract step number from directory name
    # Pattern: eval_independent_passk_qwen25_3b_Xturn_200steps_TIMESTAMP
    # or from the results file which should have checkpoint info

    # Try to find step from results file
    results_file = result_dir / 'independent_passk_results.json'
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            model_path = data.get('model', '')
            # Extract step from path like: .../global_step_50/...
            step_match = re.search(r'global_step_(\d+)', model_path)
            if step_match:
                step_num = int(step_match.group(1))
                return model_type, f'step_{step_num}', step_num

    # If can't determine, return None
    return model_type, 'unknown', -2


def load_passk_results(results_dir: Path) -> Dict:
    """Load all pass@k results from the evaluation output directory."""
    all_results = {
        '1turn': {},
        '5turn': {},
        'base': {}
    }

    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return all_results

    # Find all result subdirectories
    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        results_file = subdir / 'independent_passk_results.json'
        if not results_file.exists():
            continue

        # Extract model info
        model_type, step_name, step_num = extract_model_info(subdir)
        if model_type is None:
            print(f"[WARN] Could not parse model info from: {subdir.name}")
            continue

        # Load results
        with open(results_file) as f:
            data = json.load(f)

        pass_at_k = data.get('pass_at_k', {})
        if not pass_at_k:
            print(f"[WARN] No pass@k data in: {results_file}")
            continue

        # Store results
        if model_type == 'base':
            all_results['base'][step_name] = {
                'pass_at_k': pass_at_k,
                'step_num': step_num,
                'model_path': data.get('model', ''),
                'num_samples': data.get('num_samples_per_problem', 0)
            }
        else:
            all_results[model_type][step_name] = {
                'pass_at_k': pass_at_k,
                'step_num': step_num,
                'model_path': data.get('model', ''),
                'num_samples': data.get('num_samples_per_problem', 0)
            }

        print(f"[INFO] Loaded {model_type} {step_name}: {len(pass_at_k)} k-values")

    return all_results


def plot_passk_comparison(results: Dict, output_dir: Path):
    """Plot pass@k curves for 1turn and 5turn models."""

    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors for different steps
    colors = {
        'base': '#808080',  # Gray
        'step_50': '#e74c3c',  # Red
        'step_100': '#f39c12',  # Orange
        'step_150': '#2ecc71',  # Green
        'step_200': '#3498db'   # Blue
    }

    markers = {
        'base': 's',
        'step_50': 'o',
        'step_100': '^',
        'step_150': 'v',
        'step_200': 'D'
    }

    # Plot for each model type
    for idx, (model_type, ax) in enumerate(zip(['1turn', '5turn'], axes)):
        model_results = results[model_type]
        base_results = results.get('base', {})

        # Combine base and model results
        all_model_results = {**base_results, **model_results}

        if not all_model_results:
            ax.text(0.5, 0.5, f'No data for {model_type}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Sort by step number
        sorted_steps = sorted(all_model_results.items(),
                            key=lambda x: x[1]['step_num'])

        # Plot each step
        for step_name, step_data in sorted_steps:
            pass_at_k = step_data['pass_at_k']

            # Extract k values and success rates
            k_values = []
            success_rates = []
            for key, value in sorted(pass_at_k.items(),
                                    key=lambda x: int(x[0].split('@')[1])):
                k = int(key.split('@')[1])
                k_values.append(k)
                success_rates.append(value * 100)  # Convert to percentage

            # Plot
            label = 'Base Model' if step_name == 'base' else f'Step {step_data["step_num"]}'
            ax.plot(k_values, success_rates,
                   marker=markers.get(step_name, 'o'),
                   color=colors.get(step_name, '#000000'),
                   linewidth=2, markersize=8,
                   label=label, alpha=0.8)

        # Format plot
        ax.set_xlabel('k (number of samples)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pass@k Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_type.upper()} Model - Independent Pass@k',
                    fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(bottom=0, top=100)

        # Set x-axis ticks to be at the actual k values
        if k_values:
            ax.set_xticks(k_values)
            ax.set_xticklabels([str(k) for k in k_values], rotation=45)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / 'independent_passk_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved plot to: {output_file}")

    # Also save as PDF
    output_pdf = output_dir / 'independent_passk_comparison.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"[INFO] Saved PDF to: {output_pdf}")

    plt.close()


def plot_passk_individual(results: Dict, output_dir: Path):
    """Plot separate pass@k curves for 1turn and 5turn models."""

    for model_type in ['1turn', '5turn']:
        model_results = results[model_type]
        base_results = results.get('base', {})

        # Combine base and model results
        all_model_results = {**base_results, **model_results}

        if not all_model_results:
            print(f"[WARN] No data for {model_type}, skipping individual plot")
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define colors and markers
        colors = {
            'base': '#808080',
            'step_50': '#e74c3c',
            'step_100': '#f39c12',
            'step_150': '#2ecc71',
            'step_200': '#3498db'
        }

        markers = {
            'base': 's',
            'step_50': 'o',
            'step_100': '^',
            'step_150': 'v',
            'step_200': 'D'
        }

        # Sort by step number
        sorted_steps = sorted(all_model_results.items(),
                            key=lambda x: x[1]['step_num'])

        # Plot each step
        for step_name, step_data in sorted_steps:
            pass_at_k = step_data['pass_at_k']

            # Extract k values and success rates
            k_values = []
            success_rates = []
            for key, value in sorted(pass_at_k.items(),
                                    key=lambda x: int(x[0].split('@')[1])):
                k = int(key.split('@')[1])
                k_values.append(k)
                success_rates.append(value * 100)

            # Plot
            label = 'Base Model' if step_name == 'base' else f'Step {step_data["step_num"]}'
            ax.plot(k_values, success_rates,
                   marker=markers.get(step_name, 'o'),
                   color=colors.get(step_name, '#000000'),
                   linewidth=2.5, markersize=9,
                   label=label, alpha=0.85)

        # Format plot
        ax.set_xlabel('k (number of samples)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Pass@k Success Rate (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'{model_type.upper()} Model - Independent Pass@k Performance',
                    fontsize=15, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.set_ylim(bottom=0, top=100)

        # Set x-axis ticks
        if k_values:
            ax.set_xticks(k_values)
            ax.set_xticklabels([str(k) for k in k_values], rotation=45)

        plt.tight_layout()

        # Save figure
        output_file = output_dir / f'independent_passk_{model_type}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved {model_type} plot to: {output_file}")

        # Also save as PDF
        output_pdf = output_dir / f'independent_passk_{model_type}.pdf'
        plt.savefig(output_pdf, bbox_inches='tight')
        print(f"[INFO] Saved {model_type} PDF to: {output_pdf}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot independent pass@k results for 1turn and 5turn models'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='/u/lliu22/unary-feedback/results/independent_passk',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as results_dir)'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Also create combined plot with both 1turn and 5turn'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Plotting Independent Pass@k Results")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load results
    print("[INFO] Loading results...")
    results = load_passk_results(results_dir)

    # Print summary
    print("\n" + "="*80)
    print("Summary of loaded results:")
    print("="*80)
    for model_type in ['base', '1turn', '5turn']:
        if results[model_type]:
            print(f"\n{model_type.upper()}:")
            for step_name, step_data in sorted(results[model_type].items(),
                                              key=lambda x: x[1]['step_num']):
                num_k = len(step_data['pass_at_k'])
                num_samples = step_data['num_samples']
                print(f"  {step_name}: {num_k} k-values, {num_samples} samples/problem")
    print()

    # Check if we have data
    if not any(results[t] for t in ['1turn', '5turn']):
        print("[ERROR] No valid results found!")
        return

    # Generate plots
    print("[INFO] Generating plots...")

    # Individual plots for 1turn and 5turn
    plot_passk_individual(results, output_dir)

    # Combined plot
    if args.combined:
        plot_passk_comparison(results, output_dir)

    print("\n" + "="*80)
    print("Done! Plots saved to:", output_dir)
    print("="*80)


if __name__ == '__main__':
    main()
