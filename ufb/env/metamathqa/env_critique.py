import gym
from gym import spaces
import numpy as np
from datasets import load_dataset
import re
import random
from ufb.env.base import BaseLanguageBasedEnv
from ufb.utils import all_seed
from .config import MetaMathQAEnvConfig
from collections import defaultdict

class MetaMathQAEnvCritique(BaseLanguageBasedEnv):
    """
    MetaMathQA Environment with Self-Critique Feedback

    Instead of simple "Incorrect. Please think again.", this environment provides
    structured critique prompts that encourage the model to:
    1. Identify potential errors in their reasoning
    2. Reflect on common mistakes for this type of problem
    3. Consider alternative approaches
    """

    def __init__(self, config: MetaMathQAEnvConfig):
        super(MetaMathQAEnvCritique, self).__init__()

        self.config = config
        self.dataset = load_dataset(path=self.config.dataset_path, cache_dir=self.config.cache_dir)
        self.dataset = self.dataset[self.config.split].filter(
            lambda example: example['type'].startswith('MATH_')
        )
        self.current_question_idx = None
        self.current_question = None
        self.correct_answer = None
        self.step_num = None
        self.render_cache = None
        self.unique_answers_count = defaultdict(int)
        self.total_valid_answers = 0
        self._step_rewards = []
        self._previous_answers = []  # Track previous attempts for critique
        self.penalty_lambda = 0.5
        self.max_steps = 5

    def _extract_answer(self, response):
        match = re.search(r"The answer is: (.*?)$", response, re.DOTALL)
        print(response)
        if match:
            return match.group(1).strip()
        return None

    def _generate_critique_prompt(self, user_answer, step_num):
        """
        Generate a structured critique prompt based on the attempt number
        and previous answers.
        """
        critique_prompts = []

        # Base critique structure
        critique_prompts.append("Your answer is incorrect.")

        # Step-specific guidance
        if step_num == 0:
            # First attempt - encourage careful review
            critique_prompts.append(
                "\nBefore trying again, please critique your approach:"
                "\n1. Did you correctly identify what the problem is asking?"
                "\n2. Are there any calculation errors in your work?"
                "\n3. Did you use the right formula or method?"
            )
        elif step_num == 1:
            # Second attempt - compare with previous
            critique_prompts.append(
                f"\nYour previous answer was: {self._previous_answers[-1]}"
                "\nReflect on what might have gone wrong:"
                "\n1. What assumptions did you make that might be incorrect?"
                "\n2. Did you misinterpret any part of the problem?"
                "\n3. Are there alternative approaches you should consider?"
            )
        elif step_num == 2:
            # Third attempt - deeper analysis
            critique_prompts.append(
                f"\nYour previous attempts: {', '.join(self._previous_answers)}"
                "\nYou've tried multiple times. Consider:"
                "\n1. What pattern do you see in your errors?"
                "\n2. Are you making the same type of mistake repeatedly?"
                "\n3. Should you try a completely different approach?"
                "\n4. Break down the problem into smaller steps and verify each one."
            )
        else:
            # Later attempts - encourage systematic checking
            critique_prompts.append(
                f"\nYou've made {step_num} attempts. Previous answers: {', '.join(self._previous_answers)}"
                "\nSystematically review your solution:"
                "\n1. Write out each step of your calculation explicitly"
                "\n2. Check each intermediate result"
                "\n3. Verify your final answer makes sense in context"
                "\n4. Consider if there are any edge cases or special conditions"
            )

        # Add encouragement
        remaining_attempts = self.max_steps - step_num - 1
        if remaining_attempts > 0:
            critique_prompts.append(
                f"\nYou have {remaining_attempts} attempt(s) remaining. Take your time to think carefully."
            )
        else:
            critique_prompts.append(
                "\nThis is your last attempt. Review everything carefully before answering."
            )

        return "\n".join(critique_prompts)

    def reset(self, seed=None):
        dataset = self.dataset
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset) - 1)
        question_data = dataset[self.current_question_idx]
        self.current_question = question_data['query']
        self.correct_answer = self._extract_answer(question_data['response'])
        self.step_num = 0
        self.render_cache = self.current_question

        # Reset tracking variables
        self.unique_answers_count = defaultdict(int)
        self.total_valid_answers = 0
        self._step_rewards = []
        self._previous_answers = []

        return self.render_cache

    def step(self, action):
        is_correct, is_valid = self._check_answer(action)
        reward = 1.0 / (2 ** self.step_num) if is_correct else 0.0

        if is_valid:
            minimal_normalized_action = self._minimal_normalize_answer(action)
            self.unique_answers_count[minimal_normalized_action] += 1
            self.total_valid_answers += 1
            self._step_rewards.append(reward)
            self._previous_answers.append(action)  # Track this attempt

            unique_answers_proportion = 0.0
            if self.total_valid_answers > 0:
                unique_answers_proportion = len(self.unique_answers_count) / self.total_valid_answers

            info = {
                "action_is_valid": is_valid,
                "success": is_correct,
                "per_question_unique_answers_ratio": unique_answers_proportion
            }

            self.step_num += 1
        else:
            # Invalid answer - provide basic feedback
            info = {
                "action_is_valid": is_valid,
                "success": False,
                "per_question_unique_answers_ratio": 0.0
            }

        if is_correct or self.step_num >= self.max_steps:
            # Episode done
            T = self.total_valid_answers
            E = len(self.unique_answers_count)
            penalty = self.penalty_lambda * (1 - (E / T)) if T > 0 else 0.0
            total_reward = sum(self._step_rewards) - penalty
            info["global_repetition_penalty"] = penalty
            info["final_total_reward"] = total_reward

            if is_correct:
                observation = "Correct!"
            else:
                observation = f"Maximum attempts reached. The correct answer was: {self.correct_answer}"

            done = True
            self.render_cache = observation
            return self.render_cache, total_reward, done, info
        else:
            # Generate critique prompt instead of simple "Incorrect"
            observation = self._generate_critique_prompt(action, self.step_num - 1)
            done = False
            self.render_cache = observation
            return self.render_cache, reward, done, info

    def _minimal_normalize_answer(self, answer):
        """Minimally normalize answer for unique counting - preserves different forms"""
        if answer is None:
            return ""
        return re.sub(r'\s+', '', answer.strip().lower())

    def _normalize_answer(self, answer):
        """Normalize the answer for consistent counting."""
        if answer is None:
            return ""
        return re.sub(r'\s+', '', answer.strip().lower())

    def _check_answer(self, user_answer):
        """Check if the user's answer matches the correct answer."""
        user_answer = user_answer.strip()
        normalized_answer = self._normalize_answer(user_answer)
        if self.correct_answer:
            normalized_label = self._normalize_answer(self.correct_answer)
            is_correct = normalized_answer == normalized_label
        else:
            is_correct = False
        is_valid = normalized_answer != ""
        return is_correct, is_valid

    def render(self):
        return self.render_cache


if __name__ == "__main__":
    # Test the critique environment
    config = MetaMathQAEnvConfig(
        dataset_path="meta-math/MetaMathQA",
        cache_dir="./data",
        split="train"
    )

    env = MetaMathQAEnvCritique(config)

    # Reset and test
    print("Question:")
    question = env.reset(seed=42)
    print(question)
    print("\nCorrect answer (for testing):")
    print(env.correct_answer)

    # Simulate multiple incorrect attempts to see critique prompts
    test_answers = ["10", "20", "30", "40"]

    for i, answer in enumerate(test_answers):
        print(f"\n{'='*60}")
        print(f"Attempt {i+1}: {answer}")
        print('='*60)

        obs, reward, done, info = env.step(answer)

        print("\nFeedback:")
        print(obs)
        print(f"\nReward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")

        if done:
            break
