#!/usr/bin/env python3
"""Test pass@k calculation logic"""

import numpy as np

def compute_pass_at_k(success_at_turn, max_turns):
    """Original implementation from agent_proxy.py"""
    success_at_turn = np.array(success_at_turn)
    total = len(success_at_turn)

    pass_at_k = {}
    for k in range(1, max_turns + 1):
        # Count how many succeeded within first k turns
        succeeded = np.sum(success_at_turn < k)
        pass_at_k[f'pass@{k}'] = succeeded / total if total > 0 else 0.0

    return pass_at_k

# Test case 1: All succeed on first turn
print("Test 1: All succeed on turn 0 (first turn)")
success_at_turn = [0, 0, 0, 0, 0]  # All succeeded on turn 0
result = compute_pass_at_k(success_at_turn, 5)
print(f"  success_at_turn: {success_at_turn}")
print(f"  pass@1: {result['pass@1']:.2%} (expected: 100%)")
print(f"  Overall success: {np.sum(np.array(success_at_turn) >= 0) / len(success_at_turn):.2%}")
print()

# Test case 2: Mixed success
print("Test 2: Mixed success")
success_at_turn = [0, 1, 2, -1, -1]  # 3 succeeded (turns 0,1,2), 2 failed
result = compute_pass_at_k(success_at_turn, 5)
print(f"  success_at_turn: {success_at_turn}")
print(f"  pass@1: {result['pass@1']:.2%} (expected: 20% - only turn 0)")
print(f"  pass@2: {result['pass@2']:.2%} (expected: 40% - turns 0,1)")
print(f"  pass@3: {result['pass@3']:.2%} (expected: 60% - turns 0,1,2)")
print(f"  Overall success: {np.sum(np.array(success_at_turn) >= 0) / len(success_at_turn):.2%} (expected: 60%)")
print()

# Test case 3: The problematic case
print("Test 3: Simulating the actual data")
# If pass@1 is 84% but overall success is 66%, what would success_at_turn look like?
# This is impossible! If 84% succeeded on turn 0, overall success should be at least 84%
print("  If pass@1 = 84% and overall success = 66%, this is IMPOSSIBLE")
print("  pass@1 should always be ≤ overall success")
print()

# Test case 4: What if we're counting wrong?
print("Test 4: What if success_at_turn uses 1-indexing instead of 0-indexing?")
success_at_turn = [1, 2, 3, -1, -1]  # If 1 means first turn, 2 means second turn
result = compute_pass_at_k(success_at_turn, 5)
print(f"  success_at_turn: {success_at_turn} (1-indexed)")
print(f"  pass@1: {result['pass@1']:.2%} (counts success_at_turn < 1, i.e., 0 items)")
print(f"  pass@2: {result['pass@2']:.2%} (counts success_at_turn < 2, i.e., 1 item = 20%)")
print(f"  This would explain why pass@1 is wrong!")
