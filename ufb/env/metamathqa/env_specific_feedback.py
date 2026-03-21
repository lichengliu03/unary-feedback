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


class MetaMathQAEnvSpecificFeedback(BaseLanguageBasedEnv):
    """
    MetaMathQA Environment with Specific Feedback (answer-directed hint).

    Instead of one-bit "Incorrect", this environment provides a directional
    hint based on the ground truth: "The correct answer should be larger/smaller."
    This is Level 3 in the feedback information scaling experiment.
    """

    def __init__(self, config: MetaMathQAEnvConfig):
        super(MetaMathQAEnvSpecificFeedback, self).__init__()

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
        self.penalty_lambda = 0.5
        self.max_steps = 5

    def _extract_answer(self, response):
        match = re.search(r"The answer is: (.*?)$", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _try_parse_number(self, answer_str):
        """Try to parse a string as a number for comparison."""
        if answer_str is None:
            return None
        # Remove common formatting
        cleaned = answer_str.strip().replace(',', '').replace('$', '').replace('%', '')
        # Try to extract number
        match = re.search(r'-?\d+\.?\d*', cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

    def _generate_specific_feedback(self, user_answer):
        """Generate feedback with directional hint based on ground truth."""
        user_num = self._try_parse_number(user_answer)
        correct_num = self._try_parse_number(self.correct_answer)

        if user_num is not None and correct_num is not None:
            if user_num > correct_num:
                return "Incorrect. The correct answer should be smaller."
            elif user_num < correct_num:
                return "Incorrect. The correct answer should be larger."
            else:
                # Numbers are equal but string comparison failed (formatting difference)
                return "Incorrect. Your answer is close but not in the correct format."
        else:
            # Can't compare numerically, fall back to generic
            return "Incorrect. Your answer does not match the expected answer."

    def reset(self, seed=None):
        dataset = self.dataset
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset) - 1)
        question_data = dataset[self.current_question_idx]
        self.current_question = question_data['query']
        self.correct_answer = self._extract_answer(question_data['response'])
        self.step_num = 0
        self.render_cache = self.current_question

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
            observation = self._generate_specific_feedback(action)
            done = False
            self.render_cache = observation
            return self.render_cache, reward, done, info

    def _minimal_normalize_answer(self, answer):
        if answer is None:
            return ""
        return re.sub(r'\s+', '', answer.strip().lower())

    def _normalize_answer(self, answer):
        if answer is None:
            return ""
        return re.sub(r'\s+', '', answer.strip().lower())

    def _check_answer(self, user_answer):
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
