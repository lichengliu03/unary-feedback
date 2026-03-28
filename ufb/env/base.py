from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict

DEFAULT_FEEDBACK_POOL = [
    "Incorrect. Please think again.",
    "That's wrong, try again.",
    "Not quite right. Please try again.",
    "Your answer is incorrect.",
    "Wrong answer. Give it another try.",
    "That is not correct. Try again.",
    "Incorrect. Think carefully and try again.",
    "No, that's not right. Please reconsider.",
]


class BaseEnv(ABC):
    """
    Abstract base class for all environments.
    The class needs to handle text-based input, input may be invalid
        - Environment will track the total reward for the trajectory

    """
    def __init__(self):
        pass

    @abstractmethod
    def reset(self, seed=None, **kwargs) -> Any:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Returns:
            rendered environment
        """
        pass

    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass

    # below are optional methods

    def render(self, mode: str = 'text') -> Any:
        """Render the environment. Optional method."""
        pass

    def compute_reward(self, action, **kwargs) -> float:
        """Compute reward for the action."""
        pass

    def close(self):
        """Close the environment."""
        pass


class BaseDiscreteActionEnv(BaseEnv, ABC):
    """
    Abstract base class for environments with discrete action spaces.
    """

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        Args:
            action: Action to take, must be in action space, or default invalid action
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass

    @abstractmethod
    def get_all_actions(self) -> List[int]:
        """Get list of all valid actions."""
        pass


class BaseLanguageBasedEnv(BaseEnv, ABC):
    """
    Abstract base class for environments with language-based action space.
    """

    @abstractmethod
    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        Args:
            action: Action to take, must be in action space, or default invalid action
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass


class BaseEnvConfig(ABC):
    """
    Abstract base class for environment configurations.
    """
    def __init__(self):
        self.invalid_act = ""
        self.invalid_act_score = 0
