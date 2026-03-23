import logging

logger = logging.getLogger(__name__)

REGISTERED_ENVS = {}
REGISTERED_ENV_CONFIGS = {}


def _make_lazy_env(name, env_module, env_cls):
    class _LazyEnv:
        def __init__(self, *args, **kwargs):
            try:
                mod = __import__(f'ufb.env.{env_module}', fromlist=[env_cls])
                env_type = getattr(mod, env_cls)
            except Exception as e:
                raise ImportError(f"Environment '{name}' is unavailable: {e}") from e
            self._env = env_type(*args, **kwargs)

        def __getattr__(self, attr):
            return getattr(self._env, attr)

    _LazyEnv.__name__ = f"Lazy{name.title().replace('_', '')}Env"
    return _LazyEnv


def _make_lazy_config(name, config_module, config_cls):
    class _LazyConfig:
        def __new__(cls, *args, **kwargs):
            try:
                mod = __import__(f'ufb.env.{config_module}', fromlist=[config_cls])
                config_type = getattr(mod, config_cls)
            except Exception as e:
                raise ImportError(f"Environment config '{name}' is unavailable: {e}") from e
            return config_type(*args, **kwargs)

    _LazyConfig.__name__ = f"Lazy{name.title().replace('_', '')}Config"
    return _LazyConfig


def _register(name, env_module, env_cls, config_module, config_cls, optional=False):
    if optional:
        REGISTERED_ENV_CONFIGS[name] = _make_lazy_config(name, config_module, config_cls)
        REGISTERED_ENVS[name] = _make_lazy_env(name, env_module, env_cls)
        return

    try:
        cfg = __import__(f'ufb.env.{config_module}', fromlist=[config_cls])
        REGISTERED_ENV_CONFIGS[name] = getattr(cfg, config_cls)
    except Exception as e:
        raise

    try:
        mod = __import__(f'ufb.env.{env_module}', fromlist=[env_cls])
    except Exception as e:
        if not optional:
            raise
        logger.debug(f"Environment '{name}' not available: {e}")
        REGISTERED_ENVS[name] = _make_missing_env(name, e)
        return

    REGISTERED_ENVS[name] = getattr(mod, env_cls)

# === Core environments used by common training runs ===
_register('metamathqa', 'metamathqa.env', 'MetaMathQAEnv', 'metamathqa.config', 'MetaMathQAEnvConfig')
_register('metamathqa_generic_feedback', 'metamathqa.env_generic_feedback', 'MetaMathQAEnvGenericFeedback', 'metamathqa.config', 'MetaMathQAEnvConfig')
_register('metamathqa_no_feedback', 'metamathqa.env_no_feedback', 'MetaMathQAEnvNoFeedback', 'metamathqa.config', 'MetaMathQAEnvConfig')
_register('metamathqa_specific_feedback', 'metamathqa.env_specific_feedback', 'MetaMathQAEnvSpecificFeedback', 'metamathqa.config', 'MetaMathQAEnvConfig')
_register('static', 'static.env', 'StaticEnv', 'static.config', 'StaticEnvConfig')

# === Optional environments with extra dependencies ===
_register('sokoban', 'sokoban.env', 'SokobanEnv', 'sokoban.config', 'SokobanEnvConfig', optional=True)
_register('bandit', 'bandit.env', 'BanditEnv', 'bandit.config', 'BanditEnvConfig', optional=True)
_register('countdown', 'countdown.env', 'CountdownEnv', 'countdown.config', 'CountdownEnvConfig', optional=True)
_register('frozen_lake', 'frozen_lake.env', 'FrozenLakeEnv', 'frozen_lake.config', 'FrozenLakeEnvConfig', optional=True)
_register('sudoku', 'sudoku.env', 'SudokuEnv', 'sudoku.config', 'SudokuEnvConfig', optional=True)
_register('lean', 'lean.env', 'LeanEnv', 'lean.config', 'LeanEnvConfig', optional=True)
_register('search', 'search.env', 'SearchEnv', 'search.config', 'SearchEnvConfig', optional=True)
_register('webshop', 'webshop.env', 'WebShopEnv', 'webshop.config', 'WebShopEnvConfig', optional=True)
_register('alfworld', 'alfworld.env', 'AlfredTXTEnv', 'alfworld.config', 'AlfredEnvConfig', optional=True)
_register('spatial', 'spatial.env', 'SpatialGym', 'spatial.config', 'SpatialGymConfig', optional=True)
