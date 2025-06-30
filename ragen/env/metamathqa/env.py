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
        
        self.train_incorrect_feedback = "Incorrect. Please try again."
        self.val_incorrect_feedback = "Incorrect. Please try again."
        
        # Reward function type setting, default is exponential decay
        # reward_type options:
        # - "exponential_decay": Exponential decay reward 1.0/(2**step_num)
        # - "constant": Constant reward, always 1.0
        # - "linear_decay": Linear decay reward, starts at 1.0, decreases by 0.2 each step
        self.reward_type = getattr(config, "reward_type", "exponential_decay")

        self.is_validation = self.config.split.lower() in ["val", "validation", "test", "dev"]
        print(f"Environment initialized in {'validation' if self.is_validation else 'training'} mode (split: {self.config.split})")
        
    def _extract_answer(self, response):
        match = re.search(r"The answer is: (.*?)$", response, re.DOTALL)
        print(response)
        if match:
            return match.group(1).strip()
        return None
        
    def reset(self, seed=None, validation_mode=None):
        if validation_mode is not None:
            self.is_validation = validation_mode

        # dataset = self.dataset[self.config.split]
        dataset = self.dataset 
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset) - 1)
        question_data = dataset[self.current_question_idx]
        self.current_question = question_data['query']
        self.correct_answer = self._extract_answer(question_data['response'])
        self.step_num = 0
        self.render_cache = self.current_question
        
        # Reset unique answer counters for the new question
        self.unique_answers_count = defaultdict(int)
        self.total_valid_answers = 0
        
        return self.render_cache
        
    def step(self, action):
        is_correct, is_valid = self._check_answer(action)
        
        # Calculate reward based on different reward types
        if is_correct:
            if self.reward_type == "constant":
                # Constant reward: always 1.0 regardless of step number
                reward = 1.0
            elif self.reward_type == "linear_decay":
                # Linear decay: starts at 1.0, decreases by 0.2 each step, minimum 0.0
                reward = max(0.0, 1.0 - 0.2 * self.step_num)
            else:  # Default is exponential_decay
                # Exponential decay: 1.0/(2**step_num)
                reward = 1.0 / (2 ** self.step_num)
        else:
            reward = 0.0
        
        # === Repetition penalty ===
        # T: current number of steps (including this step)
        T = self.step_num + 1  # step_num is not incremented yet, so +1
        E_tau = len(self.unique_answers_count)  # number of unique (effective) answers
        if T > 0:
            penalty = 1.0 * (1 - (E_tau / T))
        else:
            penalty = 0.0
        reward = reward - penalty
        # =========================
            
        if is_correct:
            observation = "Correct!"
            done = True
        else:
            if self.is_validation:
                observation = self.val_incorrect_feedback
            else:
                observation = self.train_incorrect_feedback
            done = False
        
        unique_answers_proportion = 0.0
        if is_valid:
            # Use minimal normalization to preserve different forms for unique answer counting
            # Note: unique_answers_count is now reset per question, so this ratio is per-question only
            minimal_normalized_action = self._minimal_normalize_answer(action)
            self.unique_answers_count[minimal_normalized_action] += 1
            self.total_valid_answers += 1
        
        if self.total_valid_answers > 0:
            unique_answers_proportion = len(self.unique_answers_count) / self.total_valid_answers
            
        self.step_num += 1
        info = {
            "action_is_valid": is_valid, 
            "success": is_correct, 
            "per_question_unique_answers_ratio": unique_answers_proportion,
            "reward_type": self.reward_type,
            "step_num": self.step_num - 1  # Current step (before increment)
        }
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
        # Remove all whitespace characters
        normalized = re.sub(r'\s+', '', answer.strip().lower())
        
        # Try to convert LaTeX fraction format \frac{a}{b} to a/b
        frac_pattern = r'\\frac{(\d+)}{(\d+)}'
        if re.search(frac_pattern, normalized):
            normalized = re.sub(frac_pattern, r'\1/\2', normalized)
        
        # Convert fraction to decimal for comparison
        if '/' in normalized:
            try:
                parts = normalized.split('/')
                if len(parts) == 2 and all(part.isdigit() for part in parts):
                    num = int(parts[0])
                    denom = int(parts[1])
                    if denom != 0:  # Avoid division by zero
                        # Convert to simplified fraction representation as "num/denom"
                        from math import gcd
                        g = gcd(num, denom)
                        simplified = f"{num//g}/{denom//g}"
                        return simplified
            except:
                pass
                
        # Normalize decimal points (ensure 5 and 5.0 are treated the same)
        try:
            if normalized.replace('.', '', 1).isdigit():
                value = float(normalized)
                # Check if it's a simple fraction
                if value.is_integer():
                    return str(int(value))
                else:
                    # Try to convert to simplified fraction
                    from fractions import Fraction
                    frac = Fraction(value).limit_denominator(1000)
                    return f"{frac.numerator}/{frac.denominator}"
        except:
            pass
            
        return normalized

    def _check_answer(self, user_answer):
        """Check if the user's answer matches the correct answer."""
        user_answer = user_answer.strip()
        normalized_answer = self._normalize_answer(user_answer)
        
        is_correct = False
        if self.correct_answer:
            normalized_label = self._normalize_answer(self.correct_answer)
            
            # Exact matching (now works with normalized fractions)
            if normalized_answer == normalized_label:
                is_correct = True
            
            # Handle mixed formats (one answer might be kept as decimal, the other as fraction)
            elif '/' in normalized_answer or '/' in normalized_label:
                try:
                    # Convert both to decimal for comparison
                    user_decimal = self._fraction_to_decimal(normalized_answer)
                    correct_decimal = self._fraction_to_decimal(normalized_label)
                    
                    if user_decimal is not None and correct_decimal is not None:
                        if abs(user_decimal - correct_decimal) < 1e-6:
                            is_correct = True
                except:
                    pass
                    
        is_valid = normalized_answer != ""
        return is_correct, is_valid
        
    def _is_numeric(self, text):
        """Check if text is numeric (integer or float)"""
        try:
            float(text)
            return True
        except:
            return False
            
    def _fraction_to_decimal(self, text):
        """Convert a fraction string to decimal value"""
        if '/' in text:
            try:
                num, denom = map(int, text.split('/'))
                if denom != 0:
                    return num / denom
                return None
            except:
                return None
        else:
            try:
                return float(text)
            except:
                return None

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
