from typing import List
from ufb.env.base import BaseEnvConfig
from dataclasses import dataclass, field

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
class CountdownEnvConfig:
    train_path: str = "data/countdown/train.parquet"
    max_instances: int = 20000
    render_mode: str = "text"
    score: float = 1.0
    format_score: float = 0.1
    # Feedback config
    randomize_feedback: bool = True
    feedback_pool: List[str] = field(default_factory=lambda: list(DEFAULT_FEEDBACK_POOL))
    fixed_feedback: str = "Incorrect. Please think again."