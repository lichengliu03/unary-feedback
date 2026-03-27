#!/usr/bin/env python3
"""
Compute conditional success rates: Succ@k | fail@(k-1)
"""

import json
import sys
from pathlib import Path


def compute_conditional_success(results_json):
    """
    Compute conditional success rates from pass@k metrics.

    Succ@k | fail@(k-1) = P(success at turn k | failed at turn k-1)
                         = (pass@k - pass@(k-1)) / (1 - pass@(k-1))
    """
    with open(results_json, 'r') as f:
        data = json.load(f)

    all_results = {}

    for step_key in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
        step_data = data[step_key]
        step_num = int(step_key.split('_')[1])

        if 'pass_at_k' not in step_data:
            continue

        pass_at_k = step_data['pass_at_k']

        # Extract pass@k values for k=1,2,3,4,5
        pass_values = {}
        for k in [1, 2, 3, 4, 5]:
            key = f'pass@{k}'
            if key in pass_at_k:
                pass_values[k] = pass_at_k[key]

        # Compute conditional success rates
        conditional_success = {}

        for k in [2, 3, 4, 5]:
            if k in pass_values and (k-1) in pass_values:
                pass_k = pass_values[k]
                pass_k_minus_1 = pass_values[k-1]

                # Probability of failing at turn k-1
                fail_k_minus_1 = 1 - pass_k_minus_1

                if fail_k_minus_1 > 0:
                    # Probability of succeeding at turn k given failed at k-1
                    succ_k_given_fail_k_minus_1 = (pass_k - pass_k_minus_1) / fail_k_minus_1
                    conditional_success[f'Succ@{k}|fail@{k-1}'] = succ_k_given_fail_k_minus_1
                else:
                    # Everyone succeeded by turn k-1, no data for this condition
                    conditional_success[f'Succ@{k}|fail@{k-1}'] = None

        all_results[step_key] = {
            'step': step_num,
            'pass_at_k': pass_values,
            'conditional_success': conditional_success
        }

    return all_results


def print_results(results, experiment_name):
    """Print conditional success rates in a nice format."""
    print(f"\n{'='*80}")
    print(f"Conditional Success Rates: {experiment_name}")
    print(f"{'='*80}\n")

    for step_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        data = results[step_key]
        step = data['step']

        print(f"Step {step}:")
        print(f"  Pass@k:")
        for k in [1, 2, 3, 4, 5]:
            if k in data['pass_at_k']:
                print(f"    pass@{k}: {data['pass_at_k'][k]:.4f} ({data['pass_at_k'][k]*100:.2f}%)")

        print(f"  Conditional Success (Succ@k | fail@(k-1)):")
        for key in ['Succ@2|fail@1', 'Succ@3|fail@2', 'Succ@4|fail@3', 'Succ@5|fail@4']:
            if key in data['conditional_success']:
                val = data['conditional_success'][key]
                if val is not None:
                    print(f"    {key}: {val:.4f} ({val*100:.2f}%)")
                else:
                    print(f"    {key}: N/A (all succeeded)")
        print()


def compute_conditional_success_base(results_json):
    """
    Compute conditional success rates for base model.
    Base model JSON has different structure with 'base_model' key.
    """
    with open(results_json, 'r') as f:
        data = json.load(f)

    if 'base_model' not in data:
        return None

    base_data = data['base_model']

    if 'pass_at_k' not in base_data:
        return None

    pass_at_k = base_data['pass_at_k']

    # Extract pass@k values for k=1,2,3,4,5
    pass_values = {}
    for k in [1, 2, 3, 4, 5]:
        key = f'pass@{k}'
        if key in pass_at_k:
            pass_values[k] = pass_at_k[key]

    # Compute conditional success rates
    conditional_success = {}

    for k in [2, 3, 4, 5]:
        if k in pass_values and (k-1) in pass_values:
            pass_k = pass_values[k]
            pass_k_minus_1 = pass_values[k-1]

            # Probability of failing at turn k-1
            fail_k_minus_1 = 1 - pass_k_minus_1

            if fail_k_minus_1 > 0:
                # Probability of succeeding at turn k given failed at k-1
                succ_k_given_fail_k_minus_1 = (pass_k - pass_k_minus_1) / fail_k_minus_1
                conditional_success[f'Succ@{k}|fail@{k-1}'] = succ_k_given_fail_k_minus_1
            else:
                # Everyone succeeded by turn k-1, no data for this condition
                conditional_success[f'Succ@{k}|fail@{k-1}'] = None

    return {
        'base_model': {
            'step': 0,
            'pass_at_k': pass_values,
            'conditional_success': conditional_success
        }
    }


def print_results_base(results):
    """Print conditional success rates for base model."""
    print(f"\n{'='*80}")
    print(f"Conditional Success Rates: Base Model")
    print(f"{'='*80}\n")

    data = results['base_model']

    print(f"Base Model (Before Training):")
    print(f"  Pass@k:")
    for k in [1, 2, 3, 4, 5]:
        if k in data['pass_at_k']:
            print(f"    pass@{k}: {data['pass_at_k'][k]:.4f} ({data['pass_at_k'][k]*100:.2f}%)")

    print(f"  Conditional Success (Succ@k | fail@(k-1)):")
    for key in ['Succ@2|fail@1', 'Succ@3|fail@2', 'Succ@4|fail@3', 'Succ@5|fail@4']:
        if key in data['conditional_success']:
            val = data['conditional_success'][key]
            if val is not None:
                print(f"    {key}: {val:.4f} ({val*100:.2f}%)")
            else:
                print(f"    {key}: N/A (all succeeded)")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_conditional_success.py <1turn_json> <5turn_json> [base_model_json]")
        print("Example: python compute_conditional_success.py qwen_1turn_new.json qwen_5turn_new.json qwen25_3b_base.json")
        sys.exit(1)

    json_1turn = sys.argv[1]
    json_5turn = sys.argv[2] if len(sys.argv) > 2 else None
    json_base = sys.argv[3] if len(sys.argv) > 3 else None

    # Check files exist
    if not Path(json_1turn).exists():
        print(f"Error: File not found: {json_1turn}")
        sys.exit(1)

    # Compute for 1-turn
    results_1turn = compute_conditional_success(json_1turn)
    print_results(results_1turn, "1-Turn Training")

    # Compute for 5-turn if provided
    results_5turn = None
    if json_5turn:
        if not Path(json_5turn).exists():
            print(f"Error: File not found: {json_5turn}")
            sys.exit(1)

        results_5turn = compute_conditional_success(json_5turn)
        print_results(results_5turn, "5-Turn Training")

        # Print comparison
        print(f"\n{'='*80}")
        print("Comparison: 5-Turn vs 1-Turn Improvement")
        print(f"{'='*80}\n")

        for step_key in sorted(results_1turn.keys(), key=lambda x: int(x.split('_')[1])):
            step = results_1turn[step_key]['step']
            print(f"Step {step}:")

            for metric in ['Succ@2|fail@1', 'Succ@3|fail@2', 'Succ@4|fail@3', 'Succ@5|fail@4']:
                if step_key in results_5turn:
                    val_1turn = results_1turn[step_key]['conditional_success'].get(metric)
                    val_5turn = results_5turn[step_key]['conditional_success'].get(metric)

                    if val_1turn is not None and val_5turn is not None:
                        improvement = val_5turn - val_1turn
                        improvement_pct = (val_5turn / val_1turn - 1) * 100 if val_1turn > 0 else float('inf')
                        print(f"  {metric}:")
                        print(f"    1-Turn: {val_1turn:.4f} ({val_1turn*100:.2f}%)")
                        print(f"    5-Turn: {val_5turn:.4f} ({val_5turn*100:.2f}%)")
                        print(f"    Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
            print()

    # Compute for base model if provided
    results_base = None
    if json_base:
        if not Path(json_base).exists():
            print(f"Error: File not found: {json_base}")
            sys.exit(1)

        results_base = compute_conditional_success_base(json_base)
        if results_base:
            print_results_base(results_base)

    # Save to JSON
    output_dir = Path('eval_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'conditional_success_rates.json'

    output_data = {
        '1turn': results_1turn
    }
    if results_5turn:
        output_data['5turn'] = results_5turn
    if results_base:
        output_data['base'] = results_base

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
