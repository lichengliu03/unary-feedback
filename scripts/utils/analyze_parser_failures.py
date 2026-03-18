#!/usr/bin/env python3
"""
Analyze answer extraction success rates between 1-turn and 5-turn models.
Check if \boxed{} formatting causes parser failures.
"""

import re
import json
from pathlib import Path
from collections import defaultdict

def extract_answer_from_response(response):
    """Extract answer from <answer>...</answer> tags."""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        answer_content = match.group(1).strip()
        # Remove special tokens
        special_tokens = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
        for token in special_tokens:
            answer_content = answer_content.replace(token, "")
        return answer_content.strip()
    return None

def extract_numeric_answer(answer_text):
    """Extract numeric answer (AIME24 scoring function)."""
    if answer_text is None:
        return None
    # Extract first numeric value
    match = re.search(r'(\d+(?:\.\d+)?)', answer_text)
    if match:
        return match.group(0)
    return None

def count_boxed_usage(response):
    """Count \boxed{} occurrences in response."""
    return len(re.findall(r'\\boxed\{', response))

def analyze_model_outputs(slurm_file, label="model"):
    """Analyze model outputs from slurm file."""
    print(f"\n{'='*80}")
    print(f"Analyzing {label}")
    print(f"{'='*80}")

    with open(slurm_file, 'r') as f:
        content = f.read()

    # Split by sample markers
    samples = re.split(r'Sample \d+ - Problem \d+', content)

    stats = {
        'total_responses': 0,
        'answer_tag_found': 0,
        'answer_tag_missing': 0,
        'numeric_extracted': 0,
        'numeric_failed': 0,
        'uses_boxed': 0,
        'boxed_count': 0,
        'correct': 0,
        'incorrect': 0,
    }

    examples = {
        'missing_answer_tag': [],
        'failed_numeric_extraction': [],
        'high_boxed_count': [],
    }

    for sample in samples[1:]:  # Skip first empty split
        if 'Model full trajectory' not in sample:
            continue

        stats['total_responses'] += 1

        # Extract the assistant response
        match = re.search(r'assistant\n(.*?)\n\nSuccess at turn:', sample, re.DOTALL)
        if not match:
            continue

        response = match.group(1)

        # Check if correct
        if 'CORRECT' in sample:
            stats['correct'] += 1
        elif 'INCORRECT' in sample:
            stats['incorrect'] += 1

        # Check answer tag extraction
        answer_content = extract_answer_from_response(response)
        if answer_content is not None:
            stats['answer_tag_found'] += 1

            # Check numeric extraction
            numeric_answer = extract_numeric_answer(answer_content)
            if numeric_answer is not None:
                stats['numeric_extracted'] += 1
            else:
                stats['numeric_failed'] += 1
                if len(examples['failed_numeric_extraction']) < 5:
                    examples['failed_numeric_extraction'].append({
                        'answer_content': answer_content[:200],
                        'is_correct': 'CORRECT' in sample
                    })
        else:
            stats['answer_tag_missing'] += 1
            if len(examples['missing_answer_tag']) < 5:
                examples['missing_answer_tag'].append(response[:300])

        # Check \boxed{} usage
        boxed_count = count_boxed_usage(response)
        stats['boxed_count'] += boxed_count
        if boxed_count > 0:
            stats['uses_boxed'] += 1
            if boxed_count >= 3 and len(examples['high_boxed_count']) < 5:
                examples['high_boxed_count'].append({
                    'count': boxed_count,
                    'response': response[:300],
                    'is_correct': 'CORRECT' in sample
                })

    # Print statistics
    print(f"\nTotal responses analyzed: {stats['total_responses']}")
    print(f"\nAnswer Tag Extraction:")
    print(f"  Found: {stats['answer_tag_found']} ({stats['answer_tag_found']/stats['total_responses']*100:.1f}%)")
    print(f"  Missing: {stats['answer_tag_missing']} ({stats['answer_tag_missing']/stats['total_responses']*100:.1f}%)")

    print(f"\nNumeric Extraction (from found answers):")
    if stats['answer_tag_found'] > 0:
        print(f"  Extracted: {stats['numeric_extracted']} ({stats['numeric_extracted']/stats['answer_tag_found']*100:.1f}%)")
        print(f"  Failed: {stats['numeric_failed']} ({stats['numeric_failed']/stats['answer_tag_found']*100:.1f}%)")

    print(f"\n\\boxed{{}} Usage:")
    print(f"  Responses using \\boxed: {stats['uses_boxed']} ({stats['uses_boxed']/stats['total_responses']*100:.1f}%)")
    print(f"  Total \\boxed occurrences: {stats['boxed_count']}")
    print(f"  Average \\boxed per response: {stats['boxed_count']/stats['total_responses']:.2f}")

    print(f"\nCorrectness:")
    print(f"  Correct: {stats['correct']} ({stats['correct']/stats['total_responses']*100:.1f}%)")
    print(f"  Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total_responses']*100:.1f}%)")

    # Print examples
    if examples['missing_answer_tag']:
        print(f"\n{'='*60}")
        print("Examples of Missing Answer Tags:")
        for i, ex in enumerate(examples['missing_answer_tag'][:3]):
            print(f"\nExample {i+1}:")
            print(ex)

    if examples['failed_numeric_extraction']:
        print(f"\n{'='*60}")
        print("Examples of Failed Numeric Extraction:")
        for i, ex in enumerate(examples['failed_numeric_extraction'][:3]):
            print(f"\nExample {i+1} (correct={ex['is_correct']}):")
            print(ex['answer_content'])

    return stats

if __name__ == "__main__":
    slurm_dir = Path("/u/lliu22/unary-feedback")

    # Analyze both models
    stats_1turn = analyze_model_outputs(
        slurm_dir / "slurm-eval_independent_passk-15567604.out",
        label="1-turn (200 steps)"
    )

    stats_5turn = analyze_model_outputs(
        slurm_dir / "slurm-eval_independent_passk-15567605.out",
        label="5-turn (200 steps)"
    )

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    print(f"\nParser Failure Rate (missing answer tag or failed numeric extraction):")
    failure_rate_1turn = (stats_1turn['answer_tag_missing'] + stats_1turn['numeric_failed']) / stats_1turn['total_responses'] * 100
    failure_rate_5turn = (stats_5turn['answer_tag_missing'] + stats_5turn['numeric_failed']) / stats_5turn['total_responses'] * 100
    print(f"  1-turn: {failure_rate_1turn:.2f}%")
    print(f"  5-turn: {failure_rate_5turn:.2f}%")
    print(f"  Difference: {failure_rate_5turn - failure_rate_1turn:+.2f}%")

    print(f"\n\\boxed{{}} Usage:")
    print(f"  1-turn: {stats_1turn['uses_boxed']/stats_1turn['total_responses']*100:.1f}%")
    print(f"  5-turn: {stats_5turn['uses_boxed']/stats_5turn['total_responses']*100:.1f}%")
    print(f"  5-turn uses \\boxed {stats_5turn['boxed_count']/stats_1turn['boxed_count']:.1f}x more")
