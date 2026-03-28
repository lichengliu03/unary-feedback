from typing import Optional, List, Dict
from dataclasses import dataclass, field
from ufb.env.base import DEFAULT_FEEDBACK_POOL

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
