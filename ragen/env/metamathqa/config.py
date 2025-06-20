from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class MetaMathQAEnvConfig:
    """Configuration for FrozenLake environment"""
    # Map config
    dataset_path: str = field(default="meta-math/MetaMathQA")
    cache_dir:str = field(default="./data")
    split: str = field(default="train")
    # Reward type configuration: "exponential_decay", "constant", or "linear_decay"
    reward_type: str = field(default="exponential_decay")
