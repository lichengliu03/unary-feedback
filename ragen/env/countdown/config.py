
from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass

@dataclass
class CountdownEnvConfig:
    train_path: str = "data/countdown/train.parquet"
    max_instances: int = 20000
    render_mode: str = "text"
    score = 1
    format_score = 0.1