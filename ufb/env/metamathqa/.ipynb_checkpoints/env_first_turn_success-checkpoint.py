import random

from .env import MetaMathQAEnv


class MetaMathQAEnvFirstTurnSuccess(MetaMathQAEnv):
    """
    MetaMathQA environment where only a first-turn correct answer gets reward 1.

    Later correct answers still terminate the episode successfully, but receive 0
    step reward. The existing repetition accounting and final penalty remain
    unchanged so this stays comparable to the other MetaMathQA ablations.
    """

    def step(self, action):
        is_correct, is_valid = self._check_answer(action)
        reward = 1.0 if is_correct and self.step_num == 0 else 0.0
        info = {
            "action_is_valid": is_valid,
            "success": is_correct,
            "first_turn_success": bool(is_correct and self.step_num == 0),
            "per_question_unique_answers_ratio": 0.0,
        }

        if is_valid:
            minimal_normalized_action = self._minimal_normalize_answer(action)
            self.unique_answers_count[minimal_normalized_action] += 1
            self.total_valid_answers += 1
            self._step_rewards.append(reward)
            unique_answers_proportion = 0.0
            if self.total_valid_answers > 0:
                unique_answers_proportion = len(self.unique_answers_count) / self.total_valid_answers
            self.step_num += 1
            info["per_question_unique_answers_ratio"] = unique_answers_proportion

        if is_correct or self.step_num >= self.max_steps:
            total_valid_answers = self.total_valid_answers
            unique_answers = len(self.unique_answers_count)
            penalty = self.penalty_lambda * (1 - (unique_answers / total_valid_answers)) if total_valid_answers > 0 else 0.0
            total_reward = sum(self._step_rewards) - penalty
            info["global_repetition_penalty"] = penalty
            info["final_total_reward"] = total_reward
            observation = "Correct!" if is_correct else f"Maximum attempts reached. The correct answer was: {self.correct_answer}"
            done = True
            self.render_cache = observation
            return self.render_cache, total_reward, done, info

        if self.config.randomize_feedback:
            observation = random.choice(self.config.feedback_pool)
        else:
            observation = self.config.fixed_feedback

        done = False
        self.render_cache = observation
        return self.render_cache, reward, done, info