from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class StaticEnvConfig:
    """Configuration for StaticEnv environment"""
    # Dataset config
    dataset_name: str = field(default="metamathqa") #metamathqa, gsm8k,theoremqa,mmlu
    cache_dir: str = field(default="./data")
    split: Optional[str] = field(default=None)