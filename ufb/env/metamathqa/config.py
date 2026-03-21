from typing import Optional, List, Dict
from dataclasses import dataclass, field

# Default pool of one-bit feedback prompts.
# All convey the same 1-bit signal (incorrect) with different wording.
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

@dataclass
class MetaMathQAEnvConfig:
    """Configuration for MetaMathQA environment"""
    # Map config
    dataset_path: str = field(default="meta-math/MetaMathQA")
    cache_dir:str = field(default="./data")
    split: str = field(default="train")
    # Feedback config
    randomize_feedback: bool = field(default=True)
    feedback_pool: List[str] = field(default_factory=lambda: list(DEFAULT_FEEDBACK_POOL))
    fixed_feedback: str = field(default="Incorrect. Please think again.")
