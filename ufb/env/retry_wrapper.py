"""
RetryWrapper: wraps any multi-turn environment with UFO-style retry feedback.

When the inner environment finishes an episode unsuccessfully, the wrapper
resets it to the same puzzle (same seed) and returns a feedback observation
instead of terminating.  This turns a single episode into up to
`max_attempts` attempts at the same task, matching the MetaMathQA pattern
but for inherently multi-turn environments like Sokoban.

Integration with es_manager:
  - On retry, step() returns done=True (so _execute_actions breaks the
    action loop) but sets info["retry"]=True.
  - es_manager._log_env_state checks info["retry"] and skips marking the
    env as terminated, so it continues to the next LLM turn.
  - render() prepends the feedback text to the fresh puzzle state on the
    first call after a retry.

Turn budget interaction (important for configuration):
  Each retry consumes LLM turns from agent_proxy.max_turn. A single
  "attempt" at a multi-turn env may take multiple turns (e.g. Sokoban
  needs several turns of actions per attempt). So the total turns needed
  is roughly:

      required max_turn >= max_attempts * (avg turns per attempt)

  For example, if Sokoban needs ~3 turns per attempt and max_attempts=5,
  set agent_proxy.max_turn >= 15. If max_turn is too small, later
  attempts will never be reached.

  For single-turn envs (e.g. Countdown with max_actions_per_traj=1),
  each attempt costs exactly 1 turn, so max_turn >= max_attempts suffices.
"""

import random
from typing import List, Optional

from ufb.env.base import BaseEnv

# Default pool of 1-bit retry feedback prompts.
DEFAULT_RETRY_FEEDBACK_POOL = [
    "You failed to complete the task. Try again.",
    "That attempt was unsuccessful. Please try again.",
    "Task not completed. Give it another try.",
    "You didn't succeed. Try a different approach.",
    "Unsuccessful attempt. Please try again.",
    "That didn't work. Try again.",
    "Not solved. Think carefully and try again.",
    "You ran out of steps. Try again from the start.",
]


class RetryWrapper(BaseEnv):
    """Wrap a multi-turn env with retry-on-failure and feedback injection.

    Parameters
    ----------
    inner_env : BaseEnv
        The environment to wrap (e.g. SokobanEnv).
    max_attempts : int
        Maximum number of attempts at the same puzzle. Note that each
        attempt may consume one or more LLM turns from agent_proxy.max_turn.
        Ensure max_turn is large enough to accommodate all attempts.
    randomize_feedback : bool
        If True, randomly sample from feedback_pool; otherwise use fixed_feedback.
    feedback_pool : list[str] | None
        Pool of feedback strings to sample from.
    fixed_feedback : str
        Feedback string used when randomize_feedback is False.
    reward_decay_base : float
        Reward for attempt k is multiplied by (1 / decay_base^k).
        Set to 1.0 to disable decay.
    """

    def __init__(
        self,
        inner_env: BaseEnv,
        max_attempts: int = 5,
        randomize_feedback: bool = True,
        feedback_pool: Optional[List[str]] = None,
        fixed_feedback: str = "You failed to complete the task. Try again.",
        reward_decay_base: float = 2.0,
    ):
        super().__init__()
        self.inner_env = inner_env
        self.max_attempts = max_attempts
        self.randomize_feedback = randomize_feedback
        self.feedback_pool = feedback_pool or list(DEFAULT_RETRY_FEEDBACK_POOL)
        self.fixed_feedback = fixed_feedback
        self.reward_decay_base = reward_decay_base

        self._seed = None
        self._attempt = 0          # current attempt index (0-based)
        self._feedback_pending = False
        self._feedback_text = ""

    # ------------------------------------------------------------------
    # Proxy attributes so es_manager can access inner env's config, etc.
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self.inner_env.config

    def __getattr__(self, name):
        return getattr(self.inner_env, name)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, **kwargs):
        self._seed = seed
        self._attempt = 0
        self._feedback_pending = False
        self._feedback_text = ""
        return self.inner_env.reset(seed=seed, **kwargs)

    def step(self, action):
        obs, reward, done, info = self.inner_env.step(action)

        if done and not info.get("success", False):
            # Inner episode failed (truncated / max_steps exceeded).
            self._attempt += 1
            if self._attempt < self.max_attempts:
                # Still have attempts left — reset to same puzzle.
                self.inner_env.reset(seed=self._seed)
                reward = reward / (self.reward_decay_base ** self._attempt)

                if self.randomize_feedback:
                    self._feedback_text = random.choice(self.feedback_pool)
                else:
                    self._feedback_text = self.fixed_feedback
                self._feedback_pending = True

                info["attempt"] = self._attempt
                info["max_attempts"] = self.max_attempts
                # Signal es_manager: break the action loop (done=True) but
                # do NOT mark the env as terminated (retry=True).
                info["retry"] = True
                return obs, reward, True, info

            # All attempts exhausted — truly done.
            info["attempt"] = self._attempt
            info["max_attempts"] = self.max_attempts
            return obs, reward, True, info

        if done and info.get("success", False):
            # Succeeded — scale reward by attempt number.
            reward = reward / (self.reward_decay_base ** self._attempt)
            info["attempt"] = self._attempt
            info["max_attempts"] = self.max_attempts

        return obs, reward, done, info

    def render(self, **kwargs):
        inner_render = self.inner_env.render(**kwargs)
        if self._feedback_pending:
            self._feedback_pending = False
            if isinstance(inner_render, str):
                # Text mode: prepend feedback to the fresh puzzle state.
                return f"{self._feedback_text}\n\n{inner_render}"
            # Multimodal (images): cannot prepend text into image frames.
            # Feedback is still recorded in the history via info dict.
            return inner_render
        return inner_render

    def get_all_actions(self):
        return self.inner_env.get_all_actions()

    def close(self):
        self.inner_env.close()
