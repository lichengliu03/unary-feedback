from typing import List
from ufb.env.base import BaseEnvConfig, DEFAULT_FEEDBACK_POOL
from dataclasses import dataclass, field

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