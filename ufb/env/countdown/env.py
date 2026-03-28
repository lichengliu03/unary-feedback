import gymnasium as gym
from ufb.env.base import BaseLanguageBasedEnv
import datasets
import re
import random
import itertools
from collections import defaultdict
from .config import CountdownEnvConfig


def check_format(equation, nums):
    try:
        nums_in_eq = [int(n) for n in re.findall(r'\d+', equation)]
        return sorted(nums_in_eq) == sorted(nums)
    except:
        return False

def check_correctness(equation_str, target):
    try:
        result = eval(equation_str, {"__builtins__": None}, {})
        return abs(result - target) < 1e-5
    except:
        return False

def has_solution(nums, target):
    """Check if there is a valid equation using each number exactly once."""
    length = 4
    nums = nums + [0] * (length - len(nums))
    combinations = list(itertools.product([1, -1], repeat=length))
    for combination in combinations:
        if sum(combination[i] * nums[i] for i in range(length)) == target:
            return True
    return False


class CountdownEnv(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config=None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else CountdownEnvConfig()
        self.data = self._get_data_from_parquet(self.config.train_path)
        self.index = None
        self.render_cache = None
        self.render_mode = self.config.render_mode
        self.step_num = 0
        self.max_steps = 5
        self.unique_answers_count = defaultdict(int)
        self.total_valid_answers = 0
        self._step_rewards = []
        self.penalty_lambda = 0.5
        assert self.render_mode == 'text'

    def _get_data_from_parquet(self, path):
        try:
            df = datasets.load_dataset("parquet", data_files=path)['train'].select(range(self.config.max_instances))
        except FileNotFoundError:
            print(f"Local file {path} not found, downloading from HuggingFace...")
            df = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train").select(range(self.config.max_instances))
        df = df.filter(lambda x: has_solution(x['nums'], x['target']))
        return df

    def reset(self, seed=None, mode=None):
        gym.Env.reset(self, seed=seed)
        self.index = seed % len(self.data)
        data = self.data[self.index]
        self.render_cache = f"Target: {data['target']}, nums: {data['nums']}"
        self.step_num = 0
        self.unique_answers_count = defaultdict(int)
        self.total_valid_answers = 0
        self._step_rewards = []
        return self.render_cache

    def step(self, action):
        ground_truth = self.data[self.index]
        is_correct, is_format_valid = self._check_answer(action, ground_truth)
        reward = (1.0 / (2 ** self.step_num)) * self.config.score if is_correct else 0.0

        info = {
            "action_is_valid": True,
            "success": is_correct,
            "per_question_unique_answers_ratio": 0.0,
        }

        # Track unique answers
        normalized = action.strip().lower()
        self.unique_answers_count[normalized] += 1
        self.total_valid_answers += 1
        self._step_rewards.append(reward)
        self.step_num += 1

        if self.total_valid_answers > 0:
            info["per_question_unique_answers_ratio"] = len(self.unique_answers_count) / self.total_valid_answers

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
                target = ground_truth['target']
                observation = f"Maximum attempts reached. The target was: {target}"

            done = True
            self.render_cache = observation
            return self.render_cache, total_reward, done, info
        else:
            # Retry with feedback
            if self.config.randomize_feedback:
                observation = random.choice(self.config.feedback_pool)
            else:
                observation = self.config.fixed_feedback
            done = False
            self.render_cache = observation
            return self.render_cache, reward, done, info

    def _check_answer(self, action, ground_truth):
        """Check format and correctness. Returns (is_correct, is_format_valid)."""
        target = ground_truth['target']
        nums = ground_truth['nums']
        is_format_valid = check_format(action, nums)
        if not is_format_valid:
            return False, False
        is_correct = check_correctness(action, target)
        return is_correct, True

    def compute_reward(self, action, ground_truth):
        """Score the countdown task solution (kept for compatibility)."""
        target = ground_truth['target']
        nums = ground_truth['nums']
        if not check_format(action, nums):
            return 0
        if not check_correctness(action, target):
            return self.config.format_score
        else:
            return self.config.score

    def render(self):
        return self.render_cache

    def close(self):
        pass


if __name__ == "__main__":
    config = CountdownEnvConfig()
    env = CountdownEnv(config)

    obs = env.reset(seed=43)
    print(f"Question: {obs}")
    data = env.data[env.index]
    print(f"Target: {data['target']}, Nums: {data['nums']}")

    # Simulate multiple wrong attempts then correct
    test_answers = ["1 + 2 + 3", "10 + 20", f"{data['nums'][0]} + {data['nums'][1]}"]
    for i, ans in enumerate(test_answers):
        obs, reward, done, info = env.step(ans)
        print(f"\nAttempt {i+1}: {ans}")
        print(f"  obs={obs}, reward={reward}, done={done}, info={info}")
        if done:
            break