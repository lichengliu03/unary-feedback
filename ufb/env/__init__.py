import logging

logger = logging.getLogger(__name__)

# === Always available environments ===
from .metamathqa.env import MetaMathQAEnv
from .metamathqa.env_critique import MetaMathQAEnvCritique
from .metamathqa.env_no_feedback import MetaMathQAEnvNoFeedback
from .metamathqa.config import MetaMathQAEnvConfig
from .static.env import StaticEnv
from .static.config import StaticEnvConfig
from .sokoban.env import SokobanEnv
from .sokoban.config import SokobanEnvConfig
from .bandit.env import BanditEnv
from .bandit.config import BanditEnvConfig
from .countdown.env import CountdownEnv
from .countdown.config import CountdownEnvConfig
from .frozen_lake.env import FrozenLakeEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .sudoku.env import SudokuEnv
from .sudoku.config import SudokuEnvConfig

REGISTERED_ENVS = {
    'metamathqa': MetaMathQAEnv,
    'metamathqa_critique': MetaMathQAEnvCritique,
    'metamathqa_no_feedback': MetaMathQAEnvNoFeedback,
    'static': StaticEnv,
    'sokoban': SokobanEnv,
    'bandit': BanditEnv,
    'countdown': CountdownEnv,
    'frozen_lake': FrozenLakeEnv,
    'sudoku': SudokuEnv,
}

REGISTERED_ENV_CONFIGS = {
    'metamathqa': MetaMathQAEnvConfig,
    'metamathqa_critique': MetaMathQAEnvConfig,
    'metamathqa_no_feedback': MetaMathQAEnvConfig,
    'static': StaticEnvConfig,
    'sokoban': SokobanEnvConfig,
    'bandit': BanditEnvConfig,
    'countdown': CountdownEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
    'sudoku': SudokuEnvConfig,
}

# === Environments with heavy external dependencies (lazy registration) ===

def _try_register(name, env_module, env_cls, config_module, config_cls):
    try:
        mod = __import__(f'ufb.env.{env_module}', fromlist=[env_cls])
        cfg = __import__(f'ufb.env.{config_module}', fromlist=[config_cls])
        REGISTERED_ENVS[name] = getattr(mod, env_cls)
        REGISTERED_ENV_CONFIGS[name] = getattr(cfg, config_cls)
    except ImportError as e:
        logger.debug(f"Environment '{name}' not available: {e}")

_try_register('lean', 'lean.env', 'LeanEnv', 'lean.config', 'LeanEnvConfig')
_try_register('search', 'search.env', 'SearchEnv', 'search.config', 'SearchEnvConfig')
_try_register('webshop', 'webshop.env', 'WebShopEnv', 'webshop.config', 'WebShopEnvConfig')
_try_register('alfworld', 'alfworld.env', 'AlfredTXTEnv', 'alfworld.config', 'AlfredEnvConfig')
_try_register('spatial', 'spatial.env', 'SpatialGym', 'spatial.config', 'SpatialGymConfig')
