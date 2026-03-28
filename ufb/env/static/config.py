from typing import Optional, List, Dict
from dataclasses import dataclass, field
from ufb.env.base import DEFAULT_FEEDBACK_POOL

@dataclass
class StaticEnvConfig:
    """Configuration for StaticEnv environment"""
    # Dataset config
    dataset_name: str = field(default="metamathqa") #metamathqa, gsm8k,theoremqa,mmlu
    cache_dir: str = field(default="./data")
    split: Optional[str] = field(default=None)
    # Feedback config
    randomize_feedback: bool = field(default=True)
    feedback_pool: List[str] = field(default_factory=lambda: list(DEFAULT_FEEDBACK_POOL))
    fixed_feedback: str = field(default="Incorrect. Please think again.")