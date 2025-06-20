# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
from .metamathqa.env import MetaMathQAEnv
from .metamathqa.config import MetaMathQAEnvConfig


REGISTERED_ENVS = {
    # 'alfworld': AlfredTXTEnv,
    'metamathqa': MetaMathQAEnv,
}

REGISTERED_ENV_CONFIGS = {
    # 'alfworld': AlfredEnvConfig,
    'metamathqa': MetaMathQAEnvConfig,
}

try:
    from .webshop.env import WebShopEnv
    from .webshop.config import WebShopEnvConfig
    REGISTERED_ENVS['webshop'] = WebShopEnv
    REGISTERED_ENV_CONFIGS['webshop'] = WebShopEnvConfig
except ImportError:
    pass
