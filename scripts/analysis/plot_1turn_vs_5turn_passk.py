#!/usr/bin/env python3
"""
Plot pass@k curves comparing 1turn vs 5turn training on AIME24.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths to result files
RESULTS_DIR = Path("/u/lliu22/unary-feedback/results/independent_passk")
OUTPUT_DIR = Path("/u/lliu22/unary-feedback/results/independent_passk")

# Load results
results_1turn = json.load(open(RESULTS_DIR / "eval_independent_passk_qwen25_3b_1turn_200steps_20260118_035903" / "independent_passk_results.json"))
results_5turn = json.load(open(RESULTS_DIR / "eval_independent_passk_qwen25_3b_5turn_200steps_20260118_041456" / "independent_passk_results.json"))

# Extract pass@k data
k_values = results_1turn['k_values_computed']
passk_1turn = [results_1turn['pass_at_k'][f'pass@{k}'] * 100 for k in k_values]
passk_5turn = [results_5turn['pass_at_k'][f'pass@{k}'] * 100 for k in k_values]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot curves
ax.plot(k_values, passk_1turn, marker='o', linewidth=2, markersize=8, label='1-turn Training (200 steps)', color='#2E86AB')
ax.plot(k_values, passk_5turn, marker='s', linewidth=2, markersize=8, label='5-turn Training (200 steps)', color='#A23B72')

# Formatting
ax.set_xscale('log', base=2)
ax.set_xlabel('k (number of samples)', fontsize=12, fontweight='bold')
ax.set_ylabel('Pass@k (%)', fontsize=12, fontweight='bold')
ax.set_title('Independent Pass@k on AIME24: 1-turn vs 5-turn Training\nQwen2.5-3B-Instruct, 200 Training Steps',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='lower right')

# Set x-axis ticks
ax.set_xticks(k_values)
ax.set_xticklabels([str(k) for k in k_values], rotation=45)

# Add value annotations for key points
for k in [1, 64, 512]:
    idx = k_values.index(k)
    ax.annotate(f"{passk_1turn[idx]:.1f}%",
                xy=(k, passk_1turn[idx]),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color='#2E86AB',
                fontweight='bold')
    ax.annotate(f"{passk_5turn[idx]:.1f}%",
                xy=(k, passk_5turn[idx]),
                xytext=(0, -15),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color='#A23B72',
                fontweight='bold')

plt.tight_layout()

# Save figure
output_path = OUTPUT_DIR / "1turn_vs_5turn_passk_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot to: {output_path}")

# Print summary statistics
print("\n" + "="*60)
print("Pass@k Summary Statistics")
print("="*60)
print(f"Model: Qwen2.5-3B-Instruct, 200 steps")
print(f"Dataset: AIME24 ({results_1turn['num_problems']} problems)")
print(f"Samples per problem: {results_1turn['num_samples_per_problem']}")
print()
print(f"{'k':<10} {'1-turn':<15} {'5-turn':<15} {'Improvement'}")
print("-" * 60)
for k in k_values:
    p1 = results_1turn['pass_at_k'][f'pass@{k}'] * 100
    p5 = results_5turn['pass_at_k'][f'pass@{k}'] * 100
    improvement = p5 - p1
    print(f"{k:<10} {p1:<15.2f}% {p5:<15.2f}% {improvement:+.2f}%")

print("="*60)
