"""
Test script for the self-critique feedback environment.

This script demonstrates how the critique environment provides structured
feedback instead of simple "try again" messages.
"""

import sys
sys.path.insert(0, '/u/lliu22/unary-feedback')

from ufb.env.metamathqa.env_critique import MetaMathQAEnvCritique
from ufb.env.metamathqa.config import MetaMathQAEnvConfig

def test_critique_feedback():
    """Test the critique environment with multiple incorrect attempts."""

    print("="*80)
    print("Testing Self-Critique Feedback Environment")
    print("="*80)

    # Create environment
    config = MetaMathQAEnvConfig(
        dataset_path="meta-math/MetaMathQA",
        cache_dir="./data",
        split="train"
    )

    env = MetaMathQAEnvCritique(config)

    # Reset environment
    print("\n[QUESTION]")
    question = env.reset(seed=42)
    print(question)
    print(f"\n[CORRECT ANSWER (for testing)]: {env.correct_answer}")

    # Test with multiple incorrect attempts to see different critique prompts
    test_answers = [
        "100",  # First attempt
        "200",  # Second attempt
        "300",  # Third attempt
        "400",  # Fourth attempt
    ]

    for i, answer in enumerate(test_answers):
        print(f"\n{'='*80}")
        print(f"ATTEMPT {i+1}: {answer}")
        print('='*80)

        obs, reward, done, info = env.step(answer)

        print("\n[FEEDBACK FROM ENVIRONMENT]")
        print(obs)
        print(f"\n[REWARD]: {reward}")
        print(f"[DONE]: {done}")
        print(f"[INFO]: {info}")

        if done:
            print("\n[EPISODE ENDED]")
            break

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)

def compare_environments():
    """Compare original vs critique environment feedback."""

    from ufb.env.metamathqa.env import MetaMathQAEnv

    print("\n" + "="*80)
    print("Comparing Original vs Critique Feedback")
    print("="*80)

    config = MetaMathQAEnvConfig(
        dataset_path="meta-math/MetaMathQA",
        cache_dir="./data",
        split="train"
    )

    # Test original environment
    print("\n[ORIGINAL ENVIRONMENT]")
    print("-"*80)
    env_original = MetaMathQAEnv(config)
    question = env_original.reset(seed=42)
    print(f"Question: {question}")
    obs, reward, done, info = env_original.step("wrong_answer")
    print(f"Feedback: {obs}")

    # Test critique environment
    print("\n[CRITIQUE ENVIRONMENT]")
    print("-"*80)
    env_critique = MetaMathQAEnvCritique(config)
    question = env_critique.reset(seed=42)
    print(f"Question: {question}")
    obs, reward, done, info = env_critique.step("wrong_answer")
    print(f"Feedback:\n{obs}")

    print("\n" + "="*80)

if __name__ == "__main__":
    # Run tests
    test_critique_feedback()
    print("\n\n")
    compare_environments()
