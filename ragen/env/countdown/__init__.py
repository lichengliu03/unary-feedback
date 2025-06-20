"""
Adapted from the nicely written code from TinyZero and veRL
We plan to generalize this environment to support any sort of static problem sets
"""

from .env import CountdownEnv
from .config import CountdownEnvConfig

__all__ = ["CountdownEnv", "CountdownEnvConfig"]
