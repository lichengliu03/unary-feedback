#!/usr/bin/env python3
"""
Aggregate independent pass@k results from multiple evaluation runs.

This script collects results from all checkpoint evaluations and creates
a single JSON file with the aggregated data.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def extract_model_info(result_dir: Path) -> Tuple[str, str, int]:
    """
    Extract model type (1turn/5turn/base), checkpoint step from directory name.

    Returns:
        (model_type, step_name, step_num) where:
        - model_type: '1turn' or '5turn' or 'base'
        - step_name: 'base' or 'step_50' etc
        - step_num: -1 for base, else the step number
    """
    dir_name = result_dir.name

    # Check if base model
    if 'base_model' in dir_name:
        return 'base', 'base_model', -1

    # Extract turn type
    if '1turn' in dir_name:
        model_type = '1turn'
    elif '5turn' in dir_name:
        model_type = '5turn'
    else:
        return None, None, None

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

    return model_type, 'unknown', -2


def aggregate_results(results_dir: Path, output_file: Path):
    """Aggregate all independent pass@k results into a single JSON file."""

    aggregated = {
        '1turn': {},
        '5turn': {},
        'base': {}
    }

    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return

    # Find all result subdirectories
    for subdir in sorted(results_dir.iterdir()):
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

        # Convert pass@k keys to integers for consistency
        pass_at_k_int = {}
        for key, value in pass_at_k.items():
            k = int(key.split('@')[1])
            pass_at_k_int[k] = value

        # Store results
        result_entry = {
            'step': step_num,
            'pass_at_k': pass_at_k_int,
            'num_samples': data.get('num_samples_per_problem', 0),
            'num_problems': data.get('num_problems', 0),
            'temperature': data.get('temperature', 0),
            'top_p': data.get('top_p', 0),
            'model_path': data.get('model', '')
        }

        if model_type == 'base':
            aggregated['base'][step_name] = result_entry
        else:
            aggregated[model_type][step_name] = result_entry

        print(f"[INFO] Loaded {model_type} {step_name}: {len(pass_at_k_int)} k-values")

    # Save aggregated results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n[INFO] Aggregated results saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    for model_type in ['base', '1turn', '5turn']:
        if aggregated[model_type]:
            print(f"\n{model_type.upper()}:")
            for step_name in sorted(aggregated[model_type].keys(),
                                   key=lambda x: aggregated[model_type][x]['step']):
                step_data = aggregated[model_type][step_name]
                num_k = len(step_data['pass_at_k'])
                num_samples = step_data['num_samples']
                num_problems = step_data['num_problems']
                print(f"  {step_name}: {num_k} k-values, {num_samples} samples/problem, {num_problems} problems")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate independent pass@k results from multiple evaluation runs'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='/u/lliu22/unary-feedback/results/independent_passk',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='/u/lliu22/unary-feedback/results/independent_passk_aggregated.json',
        help='Output JSON file path'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output_file)

    print("="*80)
    print("Aggregating Independent Pass@k Results")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output file: {output_file}")
    print()

    aggregate_results(results_dir, output_file)


if __name__ == '__main__':
    main()
