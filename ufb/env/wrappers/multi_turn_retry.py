"""
Multi-turn retry wrapper for interactive environments (Sokoban, FrozenLake, etc.).

Wraps a multi-turn env to add attempt-level retry on failure.
When the inner env fails (done=True + success=False), the per-attempt turn
budget is exhausted, or the per-attempt action budget is exhausted, the wrapper
resets the inner env to its initial state, returns retry feedback + fresh
observation, and signals done=False so the framework continues generating turns.

Turn boundaries are notified by es_manager calling `notify_turn_end()` after
each turn's actions are executed.

This reuses the exact same done=False pattern that single-turn retry
(MetaMathQA, Countdown) uses.
"""

import random
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MULTI_TURN_RETRY_POOL = [
    "You failed to solve the task in the previous attempt. The environment has been reset to its initial state. Try again.",
    "The previous attempt was unsuccessful. The environment has been reset. Try again.",
    "You did not complete the task. The environment is back to its initial state. Try again.",
    "That attempt didn't work. The environment has been reset to the start. Try again.",
    "The task was not solved. The environment has returned to its initial state. Try again.",
    "Your previous attempt failed. The environment has been reset. Try again.",
    "You were unable to solve the task. The environment is back to the beginning. Try again.",
    "The last attempt was not successful. The environment has been reset to its initial state. Try again.",
]
DEFAULT_MULTI_TURN_FIXED_FEEDBACK = DEFAULT_MULTI_TURN_RETRY_POOL[0]


class MultiTurnRetryWrapper:
    """Wraps a multi-turn env to add attempt-level retry on failure.

    Each attempt has two budgets:
    - max_turns_per_attempt: max LLM generation turns per attempt
    - max_actions_per_attempt: max env actions per attempt

    Either budget being exhausted (or inner env failure) triggers a retry.
    Both counters reset on retry.

    Args:
        inner_env: The environment to wrap (e.g. SokobanEnv, FrozenLakeEnv).
        max_turns_per_attempt: Max turns allowed per single attempt.
        max_actions_per_attempt: Max actions allowed per single attempt.
        max_retry_attempts: Total number of attempts (including the first).
        randomize_feedback: Whether to sample retry feedback from the pool.
        retry_feedback_pool: List of feedback strings to randomly sample from.
        fixed_feedback: Retry feedback string used when randomize_feedback is False.
            Set to "" for no-feedback mode.
        reward_decay_base: Base for exponential reward decay across attempts.
            Attempt k rewards are scaled by 1/(base^k). Default 2.0.
    """

    def __init__(
        self,
        inner_env,
        max_turns_per_attempt: int,
        max_actions_per_attempt: int,
        max_retry_attempts: int,
        randomize_feedback: bool = True,
        retry_feedback_pool: Optional[List[str]] = None,
        fixed_feedback: Optional[str] = None,
        reward_decay_base: float = 2.0,
    ):
        self.inner_env = inner_env
        self.config = inner_env.config
        self.max_turns_per_attempt = max_turns_per_attempt
        self.max_actions_per_attempt = max_actions_per_attempt
        self.max_retry_attempts = max_retry_attempts
        self.randomize_feedback = randomize_feedback
        self.retry_feedback_pool = retry_feedback_pool or list(DEFAULT_MULTI_TURN_RETRY_POOL)
        self.fixed_feedback = (
            DEFAULT_MULTI_TURN_FIXED_FEEDBACK
            if fixed_feedback is None
            else fixed_feedback
        )
        self.reward_decay_base = reward_decay_base

        # State tracking
        self.attempt_num: int = 0
        self.turns_in_attempt: int = 0
        self.actions_in_attempt: int = 0
        self.initial_seed: Optional[int] = None
        self.render_cache: Optional[str] = None

    def reset(self, seed=None, **kwargs) -> Any:
        self.initial_seed = seed
        self.attempt_num = 0
        self.turns_in_attempt = 0
        self.actions_in_attempt = 0
        obs = self.inner_env.reset(seed=seed, **kwargs)
        self.render_cache = obs
        return obs

    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        obs, reward, done, info = self.inner_env.step(action)
        self.actions_in_attempt += 1

        # Apply reward decay based on current attempt
        decay = 1.0 / (self.reward_decay_base ** self.attempt_num)
        reward = reward * decay

        # Any terminal env step ends the current attempt. Whether we continue
        # to a new attempt depends only on the success flag of this attempt.
        if done:
            return self._finish_attempt(
                obs=obs,
                reward=reward,
                info=info,
                success=bool(info.get("success", False)),
            )

        # Budget exhaustion ends the attempt as unsuccessful.
        if self.actions_in_attempt >= self.max_actions_per_attempt:
            return self._finish_attempt(
                obs=obs,
                reward=reward,
                info=info,
                success=False,
            )

        # Normal step within an attempt
        self.render_cache = obs
        return obs, reward, False, info

    def notify_turn_end(self) -> Optional[Tuple[Any, float, bool, Dict]]:
        """Called by es_manager after each turn's actions are executed.

        Increments the turn counter. If the per-attempt turn budget is
        exhausted, triggers a retry (or final failure).

        Returns:
            None if no retry needed (normal continuation).
            (obs, reward, done, info) tuple if retry triggered — the caller
            should use this as the turn's result instead.
        """
        self.turns_in_attempt += 1

        if self.turns_in_attempt >= self.max_turns_per_attempt:
            return self._finish_attempt(
                obs=self.render_cache,
                reward=0.0,
                info={"success": False},
                success=False,
            )

        return None

    def _finish_attempt(
        self,
        obs: Any,
        reward: float,
        info: Optional[Dict],
        success: bool,
    ) -> Tuple[Any, float, bool, Dict]:
        """Finish the current attempt and decide whether to retry.

        Retry decisions are based solely on whether the just-finished attempt
        succeeded. Successful attempts terminate immediately; unsuccessful
        attempts start a new attempt when retries remain.
        """
        info = dict(info or {})
        info["success"] = bool(success)

        if success:
            self.render_cache = obs
            return obs, reward, True, info

        if self.attempt_num + 1 < self.max_retry_attempts:
            return self._do_retry(reward, info)

        self.render_cache = obs
        return obs, reward, True, info

    def _do_retry(self, reward: float, info: Dict) -> Tuple[Any, float, bool, Dict]:
        """Reset inner env and return retry feedback + fresh initial state."""
        self.attempt_num += 1
        self.turns_in_attempt = 0
        self.actions_in_attempt = 0
        self.inner_env.reset(seed=self.initial_seed)
        initial_state = self.inner_env.render()
        if self.randomize_feedback:
            feedback = random.choice(self.retry_feedback_pool)
        else:
            feedback = self.fixed_feedback

        if feedback:
            retry_obs = f"{feedback}\n{initial_state}"
        else:
            retry_obs = initial_state
        self.render_cache = retry_obs

        info["retry"] = True
        info["attempt_num"] = self.attempt_num
        info["success"] = False
        return retry_obs, reward, False, info

    def render(self, *args, **kwargs) -> Any:
        return self.render_cache

    def get_all_actions(self):
        return self.inner_env.get_all_actions()

    def close(self):
        return self.inner_env.close()

    def __getattr__(self, name):
        """Delegate any other attribute access to the inner env."""
        return getattr(self.inner_env, name)
