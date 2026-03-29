#!/usr/bin/env python3
"""
Compute conditional success rates: Succ@k | fail@(k-1)

Supports both:
1. New summary JSONs with a top-level ``base_model`` entry.
2. Old checkpoint JSONs with top-level ``step_<n>`` entries.

Formula:
    Succ@k | fail@(k-1) = (pass@k - pass@(k-1)) / (1 - pass@(k-1))
"""

import json
import sys
from pathlib import Path


PASS_KEYS = [1, 2, 3, 4, 5]
COND_KEYS = [2, 3, 4, 5]


def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _extract_ordered_entries(data):
    if 'base_model' in data:
        return [('base_model', data['base_model'])]

    step_entries = []
    for key, value in data.items():
        if key.startswith('step_'):
            try:
                step_num = int(key.split('_', 1)[1])
            except ValueError:
                continue
            step_entries.append((key, step_num, value))
    step_entries.sort(key=lambda item: item[1])
    return [(key, value) for key, _, value in step_entries]


def _compute_single_entry(label, entry):
    if 'pass_at_k' not in entry:
        return None

    pass_at_k = entry['pass_at_k']
    pass_values = {}
    for k in PASS_KEYS:
        key = f'pass@{k}'
        if key in pass_at_k:
            pass_values[k] = pass_at_k[key]

    conditional_success = {}
    for k in COND_KEYS:
        if k not in pass_values or (k - 1) not in pass_values:
            continue
        pass_k = pass_values[k]
        pass_prev = pass_values[k - 1]
        fail_prev = 1 - pass_prev
        if fail_prev > 0:
            conditional_success[f'Succ@{k}|fail@{k-1}'] = (pass_k - pass_prev) / fail_prev
        else:
            conditional_success[f'Succ@{k}|fail@{k-1}'] = None

    step_or_base = 0 if label == 'base_model' else int(label.split('_', 1)[1])
    return {
        'step': step_or_base,
        'pass_at_k': pass_values,
        'conditional_success': conditional_success,
    }


def compute_conditional_success(results_json):
    data = _load_json(results_json)
    results = {}
    for label, entry in _extract_ordered_entries(data):
        computed = _compute_single_entry(label, entry)
        if computed is not None:
            results[label] = computed
    return results


def print_results(results, experiment_name):
    print(f"\n{'=' * 80}")
    print(f"Conditional Success Rates: {experiment_name}")
    print(f"{'=' * 80}\n")

    def _sort_key(item_key):
        return -1 if item_key == 'base_model' else int(item_key.split('_', 1)[1])

    for label in sorted(results.keys(), key=_sort_key):
        data = results[label]
        title = "Base Model" if label == 'base_model' else f"Step {data['step']}"
        print(f"{title}:")
        print("  Pass@k:")
        for k in PASS_KEYS:
            if k in data['pass_at_k']:
                value = data['pass_at_k'][k]
                print(f"    pass@{k}: {value:.4f} ({value * 100:.2f}%)")

        print("  Conditional Success (Succ@k | fail@(k-1)):")
        for key in ['Succ@2|fail@1', 'Succ@3|fail@2', 'Succ@4|fail@3', 'Succ@5|fail@4']:
            if key not in data['conditional_success']:
                continue
            value = data['conditional_success'][key]
            if value is None:
                print(f"    {key}: N/A (all succeeded)")
            else:
                print(f"    {key}: {value:.4f} ({value * 100:.2f}%)")
        print()


def compare_results(lhs_results, rhs_results, lhs_name, rhs_name):
    print(f"\n{'=' * 80}")
    print(f"Comparison: {rhs_name} vs {lhs_name}")
    print(f"{'=' * 80}\n")

    shared_labels = [label for label in lhs_results.keys() if label in rhs_results]
    def _sort_key(item_key):
        return -1 if item_key == 'base_model' else int(item_key.split('_', 1)[1])

    for label in sorted(shared_labels, key=_sort_key):
        title = "Base Model" if label == 'base_model' else f"Step {lhs_results[label]['step']}"
        print(f"{title}:")
        for metric in ['Succ@2|fail@1', 'Succ@3|fail@2', 'Succ@4|fail@3', 'Succ@5|fail@4']:
            lhs_val = lhs_results[label]['conditional_success'].get(metric)
            rhs_val = rhs_results[label]['conditional_success'].get(metric)
            if lhs_val is None or rhs_val is None:
                continue
            improvement = rhs_val - lhs_val
            improvement_pct = (rhs_val / lhs_val - 1) * 100 if lhs_val > 0 else float('inf')
            print(f"  {metric}:")
            print(f"    {lhs_name}: {lhs_val:.4f} ({lhs_val * 100:.2f}%)")
            print(f"    {rhs_name}: {rhs_val:.4f} ({rhs_val * 100:.2f}%)")
            print(f"    Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        print()


def _default_output_path(input_path):
    input_path = Path(input_path)
    if input_path.name.endswith('.summary.json'):
        return input_path.with_name(input_path.name.replace('.summary.json', '.conditional_success.json'))
    return input_path.with_suffix('.conditional_success.json')


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_conditional_success.py <results_json> [comparison_json] [output_json]")
        print("Example: python compute_conditional_success.py eval_results/run.summary.json")
        print("Example: python compute_conditional_success.py one_turn.json five_turn.json eval_results/conditional_success.json")
        sys.exit(1)

    primary_json = Path(sys.argv[1])
    comparison_json = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].endswith('.json') else None
    output_json = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    if not primary_json.exists():
        print(f"Error: File not found: {primary_json}")
        sys.exit(1)

    primary_results = compute_conditional_success(primary_json)
    if not primary_results:
        print(f"Error: No pass@k data found in {primary_json}")
        sys.exit(1)

    print_results(primary_results, primary_json.name)

    output_data = {
        'primary': {
            'input_file': str(primary_json),
            'results': primary_results,
        }
    }

    if comparison_json is not None:
        if not comparison_json.exists():
            print(f"Error: File not found: {comparison_json}")
            sys.exit(1)
        comparison_results = compute_conditional_success(comparison_json)
        if not comparison_results:
            print(f"Error: No pass@k data found in {comparison_json}")
            sys.exit(1)
        print_results(comparison_results, comparison_json.name)
        compare_results(primary_results, comparison_results, primary_json.name, comparison_json.name)
        output_data['comparison'] = {
            'input_file': str(comparison_json),
            'results': comparison_results,
        }

    if output_json is None:
        output_json = _default_output_path(primary_json)

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_json}")


if __name__ == '__main__':
    main()
