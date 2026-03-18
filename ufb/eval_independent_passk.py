#!/usr/bin/env python3
"""
Evaluate pass@k with independent sampling (not sequential multi-turn).

This evaluates the standard pass@k metric where k samples are generated
INDEPENDENTLY for each problem, as opposed to passSeq@k where the model
tries k times sequentially with feedback.

Key differences from sequential evaluation:
- max_turn = 1 (single turn generation)
- do_sample = True, temperature > 0 (stochastic sampling)
- Generate k independent samples per problem
- Compute pass@k = percentage of problems with at least 1 correct answer in k samples
"""

import os
import sys
import json
import hydra
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer
from verl import DataProto

from ufb.llm_agent.agent_proxy import VllmWrapperWg, LLMAgentProxy
from ufb.llm_agent.es_manager import EnvStateManager


def compute_pass_at_k(results: List[Dict], k_values: List[int]) -> Dict[str, float]:
    """
    Compute pass@k metrics using unbiased estimator from HumanEval/Codex paper.

    For each problem with n samples and c correct samples:
    pass@k = 1 - C(n-c, k) / C(n, k)

    Numerically stable implementation:
    pass@k = 1 - prod_{i=1}^{k} (n-c-k+i) / (n-k+i)

    Args:
        results: List of dicts with 'problem_id', 'sample_id', 'correct'
        k_values: List of k values to compute pass@k for

    Returns:
        Dictionary mapping 'pass@k' to success rate

    Reference:
        Chen et al. "Evaluating Large Language Models Trained on Code" (Codex paper)
        https://arxiv.org/abs/2107.03374
    """
    # Group by problem
    problems = {}
    for result in results:
        pid = result['problem_id']
        if pid not in problems:
            problems[pid] = []
        problems[pid].append(result['correct'])

    # Compute pass@k for each k
    pass_at_k = {}

    for k in k_values:
        total_problems = 0
        total_pass_at_k = 0.0

        for pid, corrects in problems.items():
            n = len(corrects)  # total samples for this problem
            c = sum(corrects)   # number of correct samples

            if n < k:
                # Not enough samples, skip this problem for this k
                continue

            if k > n - c:
                # Not enough incorrect samples to draw k, guaranteed ≥1 correct
                prob_pass = 1.0
            elif c == 0:
                # No correct samples, pass@k = 0.0
                prob_pass = 0.0
            else:
                # Use unbiased estimator: pass@k = 1 - C(n-c, k) / C(n, k)
                # Numerically stable: pass@k = 1 - prod((n-c-i)/(n-i) for i in range(k))
                prob_pass = 1.0
                for i in range(k):
                    prob_pass *= (n - c - i) / (n - i)
                prob_pass = 1.0 - prob_pass

            total_pass_at_k += prob_pass
            total_problems += 1

        if total_problems > 0:
            pass_at_k[f'pass@{k}'] = total_pass_at_k / total_problems

    return pass_at_k


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config):
    """Main evaluation function for independent pass@k."""

    # Override config for independent sampling
    config.agent_proxy.max_turn = 1  # Single turn only
    config.val_agent_proxy.max_turn = 1

    # Enable stochastic sampling
    if not hasattr(config.actor_rollout_ref.rollout, 'val_kwargs'):
        config.actor_rollout_ref.rollout.val_kwargs = {}

    # Get sampling parameters from command line or use defaults
    temperature = config.actor_rollout_ref.rollout.get('temperature', 0.8)
    top_p = config.actor_rollout_ref.rollout.get('top_p', 0.95)
    num_samples = config.get('num_samples_per_problem', 512)  # Total samples to generate

    # K values to compute (can be specified in config or use defaults)
    if hasattr(config, 'k_values_to_compute') and config.k_values_to_compute is not None:
        k_values_to_compute = list(config.k_values_to_compute)
    else:
        # Default: exponential scale up to num_samples
        k_values_to_compute = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        k_values_to_compute = [k for k in k_values_to_compute if k <= num_samples]

    config.actor_rollout_ref.rollout.val_kwargs.do_sample = True
    config.actor_rollout_ref.rollout.val_kwargs.temperature = temperature
    config.actor_rollout_ref.rollout.val_kwargs.top_p = top_p

    print("="*80)
    print("Independent Pass@k Evaluation")
    print("="*80)
    print(f"Model: {config.actor_rollout_ref.model.path}")
    print(f"Number of samples per problem: {num_samples}")
    print(f"K values to compute: {k_values_to_compute}")
    print(f"Temperature: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"Max turn: {config.agent_proxy.max_turn} (single turn)")
    print("="*80)

    # Initialize
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)

    # Load tokenizer from base model (checkpoints may not have tokenizer files)
    tokenizer_path = config.actor_rollout_ref.model.path
    if 'global_step' in tokenizer_path:
        # This is a checkpoint path, use base model for tokenizer
        tokenizer_path = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    actor_wg = VllmWrapperWg(config, tokenizer)
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)

    # Get environment manager to know total problems
    es_manager = EnvStateManager(config, mode="val")
    val_envs = es_manager.envs
    num_problems = len(val_envs)

    # CRITICAL: Reset environments to initialize questions and answers
    # Without this, current_question and correct_answer are None
    print("\n[SETUP] Initializing environments...")
    es_manager.reset(seed=42)  # Use fixed seed for reproducibility
    print("Environments initialized.")

    # CRITICAL: Save original questions and answers BEFORE any rollout
    # This is necessary because StaticEnv resets change the question on each rollout
    print("\n[SETUP] Saving original questions and ground truth answers...")
    original_data = []
    for prob_id, env_entry in enumerate(val_envs):
        env_obj = env_entry['env']
        original_data.append({
            'problem_id': prob_id,
            'question': env_obj.current_question,
            'answer': env_obj.correct_answer,
            'seed': env_entry['status'].seed if hasattr(env_entry['status'], 'seed') else None
        })
    print(f"Saved {len(original_data)} problem questions and answers.")

    print(f"\nEvaluating {num_problems} problems with {num_samples} samples each...")
    print(f"Total generations: {num_problems * num_samples}")
    print(f"\n[IMPORTANT] Ensuring correct pass@k evaluation:")
    print(f"  1. max_turn=1 ensures single-turn generation only")
    print(f"  2. Environment resets each iteration (no prompt carryover)")
    print(f"  3. pass@k computed with unbiased estimator (HumanEval method)")
    print()

    all_results = []

    # Generate k independent samples for each problem
    for sample_idx in range(num_samples):
        print(f"\n{'='*80}")
        print(f"Generating sample {sample_idx + 1}/{num_samples}")
        print(f"{'='*80}")

        import time
        start_time = time.time()

        # Reset environments for each sample
        rollouts = proxy.rollout(
            DataProto(
                batch=None,
                non_tensor_batch=None,
                meta_info={
                    'eos_token_id': tokenizer.eos_token_id,
                    'pad_token_id': tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': True,
                    'validate': True,
                    'sample_id': sample_idx
                }
            ),
            val=True,
            seed=42  # Use fixed seed to ensure all samples see the same questions
        )

        end_time = time.time()
        print(f"Sample {sample_idx + 1} generation time: {end_time - start_time:.2f} seconds")

        # Extract results
        metrics = rollouts.meta_info.get('metrics', {})

        # Get per-problem correctness from success_at_turn
        # success_at_turn[i] >= 0 means problem i succeeded, -1 means never succeeded
        if 'success_at_turn' in rollouts.meta_info:
            success_at_turn = rollouts.meta_info['success_at_turn']

            # Output debug info for ALL problems in ALL samples
            print(f"\n{'='*80}")
            print(f"DEBUG: Sample {sample_idx + 1}/{num_samples} - Showing all {num_problems} problems")
            print(f"{'='*80}")

            # Decode trajectories from input_ids
            if 'input_ids' in rollouts.batch:
                trajectories = tokenizer.batch_decode(rollouts.batch['input_ids'], skip_special_tokens=True)
            else:
                trajectories = []

            rewards = rollouts.meta_info.get('rewards', [])

            for prob_id in range(num_problems):  # Changed from min(3, num_problems) to num_problems
                print(f"\n{'*'*60}")
                print(f"Sample {sample_idx + 1} - Problem {prob_id}")
                print(f"{'*'*60}")

                # Get question and ground truth from saved original data
                orig_data = original_data[prob_id]
                question_text = orig_data['question']
                ground_truth = orig_data['answer']

                # Print question preview
                if question_text is not None:
                    question_preview = question_text[:300] if len(question_text) > 300 else question_text
                    print(f"Question: {question_preview}...")
                else:
                    print(f"Question: [Not available]")

                print(f"Ground truth: {ground_truth}")

                # Show trajectory
                if prob_id < len(trajectories):
                    traj = trajectories[prob_id]
                    print(f"\nModel full trajectory (length={len(traj)}):")
                    print(f"{traj}")

                # Show reward if available
                if prob_id < len(rewards):
                    print(f"\nReward: {rewards[prob_id]}")

                # Show success status
                success_turn = success_at_turn[prob_id]
                print(f"\nSuccess at turn: {success_turn}")
                print(f"Judged as: {'CORRECT' if success_turn >= 0 else 'INCORRECT'}")

                # Also show env status
                if prob_id < len(es_manager.envs):
                    env_entry = es_manager.envs[prob_id]
                    print(f"Env status - terminated: {env_entry['status'].terminated}, truncated: {env_entry['status'].truncated}")
                print(f"{'*'*60}")

                print(f"\n{'='*80}\n")

            for prob_id, turn in enumerate(success_at_turn):
                all_results.append({
                    'problem_id': prob_id,
                    'sample_id': sample_idx,
                    'correct': turn >= 0  # True if succeeded at any turn
                })
        else:
            raise ValueError("No success_at_turn in rollouts.meta_info - evaluation cannot proceed")

    # Compute pass@k metrics
    print(f"\n{'='*80}")
    print("Computing pass@k metrics...")
    print(f"{'='*80}")

    pass_at_k = compute_pass_at_k(all_results, k_values_to_compute)

    print("\n" + "="*80)
    print("RESULTS: Independent Pass@k")
    print("="*80)
    for k in k_values_to_compute:
        key = f'pass@{k}'
        if key in pass_at_k:
            print(f"{key}: {pass_at_k[key]:.4f} ({pass_at_k[key]*100:.2f}%)")
    print("="*80)

    # Save results
    output_dir = Path(config.get('output', {}).get('dir', './results/independent_passk'))
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'independent_passk_results.json'
    output_data = {
        'model': config.actor_rollout_ref.model.path,
        'num_samples_per_problem': num_samples,
        'k_values_computed': k_values_to_compute,
        'num_problems': num_problems,
        'temperature': temperature,
        'top_p': top_p,
        'pass_at_k': pass_at_k,
        'all_results': all_results
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return pass_at_k


if __name__ == "__main__":
    main()
