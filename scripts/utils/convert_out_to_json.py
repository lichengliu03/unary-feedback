#!/usr/bin/env python3
"""Convert SLURM .out files to structured JSON format with pass@k metrics."""

import ast
import re
import json
import sys
from pathlib import Path
from collections import defaultdict

ANSI_ESCAPE_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
NUMERIC_RE = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'


def _make_result_bucket():
    return {
        'batches': [],
        'success_rates': [],
        'rewards': [],
        'num_actions': [],
        'response_lengths': [],
        'pass_at_k': defaultdict(list),
    }


def _strip_ansi(text):
    return ANSI_ESCAPE_RE.sub('', text)


def _extract_metric(section, patterns):
    for pattern in patterns:
        match = re.search(pattern, section, re.DOTALL)
        if match:
            return float(match.group(1))
    return None


def _extract_metric_from_snapshot(snapshot, key_patterns):
    for pattern in key_patterns:
        regex = re.compile(pattern)
        for key, value in snapshot.items():
            if regex.fullmatch(key):
                return float(value)
    return None


def _extract_pass_at_k(section):
    pass_at_k = defaultdict(list)
    pass_at_k_section = re.search(r'pass@k metrics:(.*?)(?=\n\n|\nmodel=|\Z)', section, re.DOTALL)
    if pass_at_k_section:
        pass_at_k_text = pass_at_k_section.group(1)
        for match in re.finditer(r'pass@(\d+):\s+([\d.]+)', pass_at_k_text):
            pass_at_k[int(match.group(1))].append(float(match.group(2)))

    for match in re.finditer(rf'(?:^|\n).*?(?:val/)?pass@(\d+):\s*({NUMERIC_RE})', section, re.DOTALL):
        pass_at_k[int(match.group(1))].append(float(match.group(2)))
    return pass_at_k


def _parse_step_metric_line(line):
    metrics = {}
    for segment in line.split(' - '):
        if ':' not in segment:
            continue
        key, value = segment.split(':', 1)
        key = key.strip()
        value = value.strip().rstrip(',')
        if key == 'step':
            continue
        if re.fullmatch(NUMERIC_RE, value):
            metrics[key] = float(value)
    return metrics


def _extract_step_metric_snapshots(section):
    snapshots = []
    for raw_line in section.splitlines():
        line = _strip_ansi(raw_line).strip()
        if 'step:' not in line or 'val/' not in line:
            continue
        metrics = _parse_step_metric_line(line)
        if metrics:
            snapshots.append(metrics)
    return snapshots


def _extract_eval_metrics_dict_snapshots(section):
    snapshots = []
    lines = [_strip_ansi(line).strip() for line in section.splitlines()]
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if 'Evaluation metrics:' not in line:
            idx += 1
            continue

        block = line
        idx += 1
        while idx < len(lines) and '}' not in block:
            block += '\n' + lines[idx]
            idx += 1

        if 'Evaluation metrics:' in block:
            try:
                literal = block.split('Evaluation metrics:', 1)[1].strip()
                if literal.startswith('(') and literal.endswith(')'):
                    literal = ast.literal_eval(literal)
                if isinstance(literal, str):
                    literal = ast.literal_eval(literal)
                if isinstance(literal, dict):
                    snapshots.append({k: float(v) for k, v in literal.items() if isinstance(v, (int, float))})
            except Exception:
                pass
        idx += 1
    return snapshots


def _collect_snapshot_metrics(results_bucket, snapshot):
    success = _extract_metric_from_snapshot(snapshot, [r'(?:val/)?[^/]+/success', r'(?:val/)?success'])
    num_actions = _extract_metric_from_snapshot(snapshot, [r'(?:val/)?[^/]+/num_actions', r'(?:val/)?num_actions'])
    reward = _extract_metric_from_snapshot(snapshot, [
        r'(?:val/)?[^/]+/final_total_reward',
        r'(?:val/)?final_total_reward',
        r'val/test_score(?:/[^/]+)?',
        r'test_score(?:/[^/]+)?',
    ])
    response_length = _extract_metric_from_snapshot(snapshot, [r'(?:val/)?response_length'])

    if success is None:
        return

    batch = {'success': success}
    results_bucket['success_rates'].append(success)
    if num_actions is not None:
        batch['num_actions'] = num_actions
        results_bucket['num_actions'].append(num_actions)
    if reward is not None:
        batch['reward'] = reward
        results_bucket['rewards'].append(reward)
    if response_length is not None:
        batch['response_length'] = response_length
        results_bucket['response_lengths'].append(response_length)
    results_bucket['batches'].append(batch)

    for key, value in snapshot.items():
        pass_match = re.fullmatch(r'(?:val/)?pass@(\d+)', key)
        if pass_match:
            results_bucket['pass_at_k'][int(pass_match.group(1))].append(float(value))


def _collect_section_metrics(results_bucket, section):
    snapshots = _extract_step_metric_snapshots(section)
    if not snapshots:
        snapshots = _extract_eval_metrics_dict_snapshots(section)

    if snapshots:
        for snapshot in snapshots:
            _collect_snapshot_metrics(results_bucket, snapshot)
        return

    success = _extract_metric(section, [rf'(?:^|\n)(?:val/)?[^:\n]+/success:\s*({NUMERIC_RE})'])
    num_actions = _extract_metric(section, [rf'(?:^|\n)(?:val/)?[^:\n]+/num_actions:\s*({NUMERIC_RE})'])
    reward = _extract_metric(section, [
        rf'(?:^|\n)(?:val/)?[^:\n]+/final_total_reward:\s*({NUMERIC_RE})',
        rf'(?:^|\n)val/test_score(?:/[^:\n]+)?:\s*({NUMERIC_RE})',
    ])
    response_length = _extract_metric(section, [rf'(?:^|\n)(?:val/)?response_length:\s*({NUMERIC_RE})'])

    if success is not None:
        batch = {'success': success}
        results_bucket['success_rates'].append(success)
        if num_actions is not None:
            batch['num_actions'] = num_actions
            results_bucket['num_actions'].append(num_actions)
        if reward is not None:
            batch['reward'] = reward
            results_bucket['rewards'].append(reward)
        if response_length is not None:
            batch['response_length'] = response_length
            results_bucket['response_lengths'].append(response_length)
        results_bucket['batches'].append(batch)

    extracted_pass_at_k = _extract_pass_at_k(section)
    for k, values in extracted_pass_at_k.items():
        results_bucket['pass_at_k'][k].extend(values)


def _finalize_results(results):
    final_results = {}
    for step, data in results.items():
        if data['success_rates']:
            step_result = {
                'num_batches': len(data['batches']),
                'avg_success': sum(data['success_rates']) / len(data['success_rates']),
                'min_success': min(data['success_rates']),
                'max_success': max(data['success_rates']),
                'batches': data['batches']
            }
            if data['rewards']:
                step_result['avg_reward'] = sum(data['rewards']) / len(data['rewards'])
            if data['num_actions']:
                step_result['avg_num_actions'] = sum(data['num_actions']) / len(data['num_actions'])
            if data['response_lengths']:
                step_result['avg_response_length'] = sum(data['response_lengths']) / len(data['response_lengths'])

            if data['pass_at_k']:
                pass_at_k_avg = {}
                for k, values in data['pass_at_k'].items():
                    pass_at_k_avg[f'pass@{k}'] = sum(values) / len(values) if values else 0.0
                step_result['pass_at_k'] = pass_at_k_avg

            final_results[step] = step_result
    return final_results

def extract_checkpoint_results(out_file):
    """Extract evaluation results per checkpoint from .out file."""

    with open(out_file, 'r') as f:
        content = _strip_ansi(f.read())

    # Find all sections by looking for model initialization messages
    model_pattern = r"(?:model=|model_path['\"]?\s*:\s*|Model:\s*)(?:'|\")?([^'\"\n]*global_step_(\d+)[^'\"\n]*)"

    # Find all model initializations and their positions
    model_matches = list(re.finditer(model_pattern, content))

    results = defaultdict(_make_result_bucket)

    if not model_matches:
        base_results = {'base_model': _make_result_bucket()}
        _collect_section_metrics(base_results['base_model'], content)
        return _finalize_results(base_results)

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
        _collect_section_metrics(results[f'step_{step}'], section)

    return _finalize_results(results)

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

    def _result_sort_key(key):
        if key == 'base_model':
            return -1
        return int(key.split('_')[1])

    for step_key in sorted(results.keys(), key=_result_sort_key):
        data = results[step_key]
        if step_key == 'base_model':
            print("\nBase Model:")
        else:
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
