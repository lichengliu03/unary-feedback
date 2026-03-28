from ufb.env.base import BaseLanguageBasedEnv
from ufb.env.webshop.config import WebShopEnvConfig
from webshop_minimal import WebAgentTextEnv, init_basedir
from webshop_minimal.env import SimServer
from webshop_minimal.utils import get_file_path
from typing import Optional, Union
from ufb.utils import all_seed
import random
import string
import uuid
import numpy as np


# RENDER_INSTRUCTIONS moved to config or removed for alignment with verl-agent
RENDER_INSTRUCTIONS = [] 


class WebShopEnv(BaseLanguageBasedEnv, WebAgentTextEnv):
    def __init__(self, config: Optional[WebShopEnvConfig] = None, **kwargs: any) -> None:
        """
        Adapter for WebAgentTextEnv to conform to the BaseLanguageBasedEnv interface.
        """
        self.config = config or WebShopEnvConfig()
        self.observation_mode = self.config.observation_mode
        self.file_path = self.config.file_path
        self.server = self.config.server
        self.filter_goals = self.config.filter_goals
        self.limit_goals = self.config.limit_goals
        self.num_products = self.config.num_products
        self.human_goals = self.config.human_goals
        self.show_attrs = self.config.show_attrs
        self.render_cache = None
        self.task_desc = None # Cache task description to avoid AttributeError during step
        if self.config.dataset:
            init_basedir(self.config.dataset)

        BaseLanguageBasedEnv.__init__(self)

        self._seed = 0 # Match verl-agent env.seed=0
        random.seed(self._seed)
        np.random.seed(self._seed)
        
        self.server = SimServer(
            base_url='http://127.0.0.1:3000',
            file_path=get_file_path(),
            filter_goals=self.config.filter_goals,
            limit_goals=self.config.limit_goals,
            num_products=self.config.num_products,
            human_goals=self.config.human_goals,
            show_attrs=self.config.show_attrs,
        ) if self.config.server is None else self.config.server

        WebAgentTextEnv.__init__(
            self,
            observation_mode=self.observation_mode,
            file_path=self.file_path,
            server=self.server, # Pass the initialized server
            filter_goals=self.filter_goals,
            limit_goals=self.limit_goals,
            num_products=self.num_products,
            human_goals=self.human_goals,
            show_attrs=self.show_attrs,
            session_prefix=str(uuid.uuid4().hex), # we use a random session prefix to avoid collision
            **kwargs
        )

    def _get_permuted_index(self, idx, seed=42):
        """Map index to a deterministically permuted index in the same range.
        
        Args:
            idx: The original index
            seed: Random seed to ensure deterministic permutation
            
        Returns:
            int: The permuted index
        """
        # Create a cache key based on goals length and seed
        cache_key = f"perm_{len(self.server.goals)}_{seed}"
        
        # Create or retrieve the permutation map
        if not hasattr(self, cache_key):
            # Initialize with fixed seed
            rng = random.Random(seed)
            
            # Generate the full permutation
            indices = list(range(len(self.server.goals)))
            rng.shuffle(indices)
            
            # Store the permutation as an instance attribute
            setattr(self, cache_key, indices)
        
        # Look up the permuted index
        permutation = getattr(self, cache_key)
        return permutation[idx]

    def reset(self, seed: int = None, mode: str = "train", session: Optional[Union[str, int]] = None, **kwargs) -> str:
        """
        Reset the environment and return the full rendered prompt.
        """
        # use seed if provided, else use self._seed
        seed = seed if seed is not None else self._seed
        random.seed(seed)
        np.random.seed(seed)

        if session is not None:
            goal_idx = session
        else:
            # Handle RAGEN's train/val/test splits
            if mode in ["test", "val"]:
                goal_idx = seed % 500
            elif mode == "train":
                goal_idx = seed % (len(self.server.goals) - 500) + 500
            else:
                goal_idx = seed
            
            # optional permutation
            goal_idx = self._get_permuted_index(goal_idx)
            
        # Call base reset. It returns (obs, info) from our modified webshop-minimal
        obs, _ = WebAgentTextEnv.reset(self, session=goal_idx)
        
        # Extract and cache task description exactly like verl-agent
        parts = obs.split(" [SEP] ")
        if len(parts) > 2 and parts[1].strip() == 'Instruction:':
            self.task_desc = parts[2].strip()
        else:
            try:
                self.task_desc = self.get_instruction_text().replace("Instruction:", "").strip()
            except Exception:
                # Fallback if parsing fails
                self.task_desc = "Find products as requested."

        self.prepare_render_cache(obs)
        return self.render()

    def step(self, action: str):
        """
        Execute one step and return (rendered_prompt, reward, done, info).
        """
        last_observation = self.observation
        
        # Call base step
        obs, raw_reward, done, info = WebAgentTextEnv.step(self, action)
        
        # Identify if action worked
        action_is_valid = True # WebAgentTextEnv handles invalid actions by doing nothing
        
        # Align reward function (binarized {0, 10})
        # info['won'] is True if reward == 1.0 (success)
        if done and raw_reward == 1.0:
            reward = 10.0
            won = True
        else:
            reward = 0.0
            won = False
            
        info = (info or {}).copy()
        info.update({
            "reward": reward,
            "raw_reward": raw_reward,
            "action_is_effective": self.observation != last_observation,
            "action_is_valid": action_is_valid,
            "success": 1 if won else 0,
            "won": won,
            "success_purchase": 1 if done else 0,
            "success_find": 1 if won else 0,
        })
        
        self.prepare_render_cache(obs)
        return self.render(), reward, done, info

    def render(self, mode=None):
        """
        Render the environment.
        """
        return self.render_cache

    def close(self):
        """
        Close the environment.
        """
        WebAgentTextEnv.close(self)

    def format_obs(self, text_obs: str) -> str:
        """
        Align with verl-agent's format_obs logic in WebshopEnvironmentManager.
        Strips before task and quotes segments.
        """
        parts = text_obs.split(" [SEP] ")
        # In WebShop, parts[1] is 'Instruction:', parts[2] is the actual task
        try:
            task = parts[2]
            index = 2
            reformatted_obs = " [SEP] ".join(f"'{p.strip()}'" for p in parts[index+1:])
        except:
            reformatted_obs = text_obs
        return reformatted_obs

    def prepare_render_cache(self, observation: str):
        """
        Prepare the render cache for the environment to match verl-agent's structure.
        """
        available_actions = self.get_available_actions()
        
        # Use cached task description
        if self.task_desc is None:
            parts = observation.split(" [SEP] ")
            if len(parts) > 2 and parts[1].strip() == 'Instruction:':
                self.task_desc = parts[2].strip()
            else:
                try:
                    self.task_desc = self.get_instruction_text().replace("Instruction:", "").strip()
                except Exception:
                    self.task_desc = "Find products as requested."
        
        task_desc = self.task_desc
        formatted_obs = self.format_obs(observation)
        
        # Quote actions and add trailing commas to match verl-agent's EnvManager
        reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)
        
        # Match WEBSHOP_TEMPLATE_NO_HIS exactly
        render = f"Your task is to: {task_desc}.\n"
        render += f"Your current observation is: {formatted_obs}.\n"
        render += "Your admissible actions of the current situation are: \n[\n"
        render += reformatted_available_actions
        render += "\n]."
        self.render_cache = render

    def get_available_actions(self):
        """
        Parse the available actions in the environment to a list of strings.
        Matches verl-agent's format_avail_actions logic.
        """
        orig_available_actions = WebAgentTextEnv.get_available_actions(self)
        available_actions = []

        if orig_available_actions['has_search_bar']:
            # verl-agent uses <your query>
            available_actions.append('search[<your query>]')

        for clickable in orig_available_actions['clickables']:
            if clickable != 'search':
                available_actions.append(f'click[{clickable}]')
        # TODO: we may need to purge the case when available_actions == ['click[back to search]', 'click[< prev]', 'click[next >]']
        is_end_of_page = tuple(available_actions) == ('click[back to search]', 'click[< prev]', 'click[next >]')
        if is_end_of_page:
            available_actions.remove('click[next >]')
        return available_actions

if __name__ == '__main__':
    from ufb.env.webshop.config import WebShopEnvConfig
    config = WebShopEnvConfig()
    env = WebShopEnv(config)
    obs = env.reset(seed=1500, mode="train")
    print(f"Initial Observation:\n{obs}")
    print(f"Render output:\n{env.render()}")
    
    while True:
        # print(env.observation)
        # print(env.server.user_sessions[env.session]['goal']['asin'])
        available_actions = env.get_available_actions()
        print(f"Available actions: {available_actions}")
        action = input("Enter action: ")
        if action == 'q':
            break
        obs, reward, done, info = env.step(action)
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        if done:
            print("Episode ended.")
            break
    env.close()
