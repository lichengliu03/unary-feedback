import gym
from gym import spaces
import numpy as np
from datasets import load_dataset
import re
import random
from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed
from .config import MetaMathQAEnvConfig
from collections import defaultdict

class MetaMathQAEnv(BaseLanguageBasedEnv):
    def __init__(self, config: MetaMathQAEnvConfig):
        super(MetaMathQAEnv, self).__init__()
        
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
        self._step_rewards = []  # Store per-step rewards for global penalty
        self.penalty_lambda = 0.5
        self.max_steps = 5
        
    def _extract_answer(self, response):
        match = re.search(r"The answer is: (.*?)$", response, re.DOTALL)
        print(response)
        if match:
            return match.group(1).strip()
        return None
        
    def reset(self,seed=None):
        # dataset = self.dataset[self.config.split]
        dataset = self.dataset 
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset) - 1)
        question_data = dataset[self.current_question_idx]
        self.current_question = question_data['query']
        self.correct_answer = self._extract_answer(question_data['response'])
        self.step_num = 0
        self.render_cache = self.current_question
        
        # Reset unique answer counters for the new question (per-question tracking)
        self.unique_answers_count = defaultdict(int)
        self.total_valid_answers = 0
        self._step_rewards = []
        
        return self.render_cache
        
    def step(self, action):
        is_correct, is_valid = self._check_answer(action)
        reward = 1.0 / (2 ** self.step_num) if is_correct else 0.0
        if is_valid:
            minimal_normalized_action = self._minimal_normalize_answer(action)
            self.unique_answers_count[minimal_normalized_action] += 1
            self.total_valid_answers += 1
            self._step_rewards.append(reward)
            unique_answers_proportion = 0.0
            if self.total_valid_answers > 0:
                unique_answers_proportion = len(self.unique_answers_count) / self.total_valid_answers
            self.step_num += 1
            info = {
                "action_is_valid": is_valid, 
                "success": is_correct, 
                "per_question_unique_answers_ratio": unique_answers_proportion
            }
        
        if is_correct or self.step_num >= self.max_steps:
            T = self.total_valid_answers
            E = len(self.unique_answers_count)
            penalty = self.penalty_lambda * (1 - (E / T)) if T > 0 else 0.0
            total_reward = sum(self._step_rewards) - penalty
            info["global_repetition_penalty"] = penalty
            info["final_total_reward"] = total_reward
            observation = "Correct!"
            done = True
            self.render_cache = observation
            return self.render_cache, total_reward, done, info
        else:
            observation = "Incorrect. Please think again."
            done = False
            self.render_cache = observation
            return self.render_cache, reward, done, info


    def _minimal_normalize_answer(self, answer):
        """Minimally normalize answer for unique counting - preserves different forms"""
        if answer is None:
            return ""
        # Only remove whitespace and convert to lowercase, preserving the representation form
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
    # Create the environment configuration
    config = MetaMathQAEnvConfig(
        dataset_path="meta-math/MetaMathQA",
        cache_dir="./data",
        split="train"
    )

    # Initialize the environment
    env = MetaMathQAEnv(config)

    # Reset the environment to get the first question
    print("Question:")
    question = env.reset(seed=42)
    print(question)
    print("\nCorrect answer (for testing purposes):")
    print(env.correct_answer)

    # Interactive loop for testing
    while True:
        user_answer = input("\nEnter your answer (or 'q' to quit): ")
        if user_answer.lower() == 'q':
            break

        # Take a step in the environment with the user's answer
        #breakpoint()
        obs, reward, done, info = env.step(user_answer)


        # Print the results
        print("\nFeedback:", obs)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)

        # If the episode is done, reset the environment for a new question
        if done:
            print("\n--- New Question ---")
            question = env.reset()
            print(question)
            print("\nCorrect answer (for testing purposes):")
            print(env.correct_answer)
            print(f"Proportion of unique answers so far: {info['per_question_unique_answers_ratio']:.2%}")
