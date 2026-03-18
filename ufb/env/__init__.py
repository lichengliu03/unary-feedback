from .metamathqa.env import MetaMathQAEnv
from .metamathqa.env_critique import MetaMathQAEnvCritique
from .metamathqa.env_no_feedback import MetaMathQAEnvNoFeedback
from .metamathqa.config import MetaMathQAEnvConfig
from .static.env import StaticEnv
from .static.config import StaticEnvConfig


REGISTERED_ENVS = {
    'metamathqa': MetaMathQAEnv,
    'metamathqa_critique': MetaMathQAEnvCritique,
    'metamathqa_no_feedback': MetaMathQAEnvNoFeedback,
    'static': StaticEnv,
}

REGISTERED_ENV_CONFIGS = {
    'metamathqa': MetaMathQAEnvConfig,
    'metamathqa_critique': MetaMathQAEnvConfig,
    'metamathqa_no_feedback': MetaMathQAEnvConfig,
    'static': StaticEnvConfig,
}
