"""
Configuration example for training with self-critique feedback.

This shows how to use the critique environment in your training config.
"""

# In your config YAML file (e.g., config/base.yaml), modify the custom_envs section:

# Original configuration (simple feedback):
"""
custom_envs:
  MetamathQA:
    env_type: 'metamathqa'
    max_actions_per_traj: 5
    env_instruction: 'You are solving Math problems. Only give the final answer between <answer> and </answer>.'
    max_tokens: 1000
    env_config: null
"""

# New configuration (critique feedback):
"""
custom_envs:
  MetamathQACritique:
    env_type: 'metamathqa_critique'  # Use the critique version
    max_actions_per_traj: 5
    env_instruction: 'You are solving Math problems. Only give the final answer between <answer> and </answer>.'
    max_tokens: 1000
    env_config: null
"""

# Then update the es_manager section to use the new environment:
"""
es_manager:
  format_penalty: -0.1
  train:
    env_groups: 8
    group_size: 16
    env_configs:
      tags: ['MetamathQACritique']  # Changed from 'MetamathQA'
      n_groups: [8]
  val:
    env_groups: 256
    group_size: 1
    env_configs:
      tags: ['MetamathQACritique']  # Changed from 'MetamathQA'
      n_groups: [256]
"""
