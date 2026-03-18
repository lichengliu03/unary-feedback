#!/usr/bin/env python3
"""Convert SLURM .out files to structured JSON format with pass@k metrics."""

import re
import json
import sys
from pathlib import Path
from collections import defaultdict

def extract_checkpoint_results(out_file):
    """Extract evaluation results per checkpoint from .out file."""

    with open(out_file, 'r') as f:
        content = f.read()

    # Find all sections by looking for model initialization messages
    model_pattern = r"model='([^']*global_step_(\d+)[^']*)'"

    # Find all model initializations and their positions
    model_matches = list(re.finditer(model_pattern, content))

    results = defaultdict(lambda: {
        'batches': [],
        'success_rates': [],
        'rewards': [],
        'num_actions': [],
        'response_lengths': [],
        'pass_at_k': defaultdict(list)
    })

    # Process each section
    for i, match in enumerate(model_matches):
        step = int(match.group(2))
        start_pos = match.end()

        # Find the end of this section
        if i + 1 < len(model_matches):
            end_pos = model_matches[i + 1].start()
        else:
            end_pos = len(content)

        section = content[start_pos:end_pos]

        # Extract metrics from this section
        metrics_pattern = r'rollout time:.*?MetamathQA/success:\s+([\d.]+).*?MetamathQA/num_actions:\s+([\d.]+).*?MetamathQA/final_total_reward:\s+([\d.]+).*?response_length:\s+([\d.]+)'

        for metrics_match in re.finditer(metrics_pattern, section, re.DOTALL):
            success = float(metrics_match.group(1))
            num_actions = float(metrics_match.group(2))
            reward = float(metrics_match.group(3))
            response_len = float(metrics_match.group(4))

            results[step]['batches'].append({
                'success': success,
                'num_actions': num_actions,
                'reward': reward,
                'response_length': response_len
            })
            results[step]['success_rates'].append(success)
            results[step]['rewards'].append(reward)
            results[step]['num_actions'].append(num_actions)
            results[step]['response_lengths'].append(response_len)

        # Extract pass@k metrics if available
        pass_at_k_pattern = r'pass@k metrics:.*?(?:pass@(\d+):\s+([\d.]+))'
        pass_at_k_section = re.search(r'pass@k metrics:(.*?)(?=\n\n|\nrollout time:|\Z)', section, re.DOTALL)

        if pass_at_k_section:
            pass_at_k_text = pass_at_k_section.group(1)
            for k_match in re.finditer(r'pass@(\d+):\s+([\d.]+)', pass_at_k_text):
                k = int(k_match.group(1))
                value = float(k_match.group(2))
                results[step]['pass_at_k'][k].append(value)

    # Calculate aggregates for each checkpoint
    final_results = {}
    for step, data in results.items():
        if data['success_rates']:
            step_result = {
                'num_batches': len(data['batches']),
                'avg_success': sum(data['success_rates']) / len(data['success_rates']),
                'avg_reward': sum(data['rewards']) / len(data['rewards']),
                'avg_num_actions': sum(data['num_actions']) / len(data['num_actions']),
                'avg_response_length': sum(data['response_lengths']) / len(data['response_lengths']),
                'min_success': min(data['success_rates']),
                'max_success': max(data['success_rates']),
                'batches': data['batches']
            }

            # Add pass@k metrics if available
            if data['pass_at_k']:
                pass_at_k_avg = {}
                for k, values in data['pass_at_k'].items():
                    pass_at_k_avg[f'pass@{k}'] = sum(values) / len(values) if values else 0.0
                step_result['pass_at_k'] = pass_at_k_avg

            final_results[f"step_{step}"] = step_result

    return final_results

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_out_to_json.py <out_file> [output_json]")
        print("Example: python convert_out_to_json.py slurm-eval-15496823.out results/results.json")
        sys.exit(1)

    out_file = sys.argv[1]

    # Default output to eval_results directory
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # Create eval_results directory if it doesn't exist
        output_dir = Path('eval_results')
        output_dir.mkdir(exist_ok=True)

        # Generate output filename based on input
        base_name = Path(out_file).stem.replace('.out', '')
        output_file = output_dir / f"{base_name}.json"

    if not Path(out_file).exists():
        print(f"Error: File not found: {out_file}")
        sys.exit(1)

    print(f"Parsing {out_file}...")
    results = extract_checkpoint_results(out_file)

    if not results:
        print("Warning: No results found in file")
        sys.exit(1)

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\nSummary:")
    print("-" * 80)

    for step_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        data = results[step_key]
        step_num = step_key.split('_')[1]
        print(f"\nStep {step_num:>3}:")
        print(f"  Overall success: {data['avg_success']:>6.2%} (batches: {data['num_batches']})")

        if 'pass_at_k' in data:
            print(f"  Pass@k metrics:")
            for k in sorted([int(k.split('@')[1]) for k in data['pass_at_k'].keys()]):
                print(f"    pass@{k:>2}: {data['pass_at_k'][f'pass@{k}']:>6.2%}")

    print("-" * 80)

if __name__ == "__main__":
    main()
