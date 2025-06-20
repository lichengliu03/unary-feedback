<h1 align="center"> <em>Let's Try Again</em>: Eliciting Multi-Turn Reasoning in Language Models via Simplistic Feedback </h1>



## Overview

<!--
Reinforcement Learning (RL) with rule-based rewards has shown promise in enhancing reasoning capabilities of large language models (LLMs). However, existing approaches have primarily focused on static, single-turn tasks like math reasoning and coding. Extending these methods to agent scenarios introduces two fundamental challenges:

1. **Multi-turn Interactions**: Agents must perform sequential decision-making and react to environment feedback
2. **Stochastic Environments**: Uncertainty where identical actions can lead to different outcomes

RAGEN addresses these challenges through:
- A Markov Decision Process (MDP) formulation for agent tasks
- State-Thinking-Actions-Reward Policy Optimization (StarPO) algorithm that optimizes entire trajectory distributions
- Progressive reward normalization strategies to handle diverse, complex environments
-->

Reinforcement Learning (RL) with rule-based rewards has shown promise in enhancing reasoning capabilities of large language models (LLMs). However, existing approaches have primarily focused on static, single-turn tasks like math reasoning and coding. Extending these methods to agent scenarios introduces two fundamental challenges:

1. **Multi-turn Interactions**: Agents must perform sequential decision-making and react to environment feedback
2. **Stochastic Environments**: Uncertainty where identical actions can lead to different outcomes

To address these challenges, we propose a general RL framework: **StarPO** (**S**tate-**T**hinking-**A**ctions-**R**eward **P**olicy **O**ptimization), a comprehensive RL framework that provides a unified approach for training multi-turn, trajectory-level agents with flexible control over reasoning processes, reward assignment mechanisms, and prompt-rollout structures. 
Building upon StarPO, we introduce **RAGEN**, a modular agent training and evaluation system that implements the complete training loop, including rollout generation, reward calculation, and trajectory optimization. RAGEN serves as a robust research infrastructure for systematically analyzing LLM agent training dynamics in multi-turn and stochastic environments.

## Algorithm

RAGEN introduces a reinforcement learning framework to train reasoning-capable LLM agents that can operate in interactive, stochastic environments. 

<p align="center"><img src="public/starpo_logo.png" width="800px" alt="StarPO Framework" /></p>
<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
The StarPO (State-Thinking-Action-Reward Policy Optimization) framework with two interleaved stages: <b>rollout stage</b> and <b>update stage</b>. LLM iteratively generates reasoning-guided actions to interact with the environment to obtain trajectory-level rewards for LLM update to jointly   optimize reasoning and action strategies.
</p>

The framework consists of two key components:

### > MDP Formulation 
We formulate agent-environment interactions as Markov Decision Processes (MDPs) where states and actions are token sequences, allowing LLMs to reason over environment dynamics. At time t, state $s_t$ transitions to the next state through action $a_t$ following a transition function. The policy generates actions given the trajectory history. The objective is to maximize expected cumulative rewards across multiple interaction turns.

### > StarPO: Reinforcing Reasoning via Trajectory-Level Optimization
StarPO is a general RL framework for optimizing entire multi-turn interaction trajectories for LLM agents.
The algorithm alternates between two phases:

#### Rollout Stage: Reasoning-Interaction Trajectories
Given an initial state, the LLM generates multiple trajectories. At each step, the model receives the trajectory history and generates a reasoning-guided action: `<think>...</think><ans> action </ans>`. The environment receives the action and returns feedback (reward and next state).

#### Update Stage: Multi-turn Trajectory Optimization 
After generating trajectories, we train LLMs to optimize expected rewards. Instead of step-by-step optimization, StarPO optimizes entire trajectories using importance sampling. This approach enables long-horizon reasoning while maintaining computational efficiency. 
StarPO supports multiple optimization strategies: 
- PPO: We estimate token-level advantages using a value function over trajectories
- GRPO: We assign normalized reward to the full trajectory

Rollout and update stages interleave in StarPO, enabling both online and offline learning.

<!--
### > Reward Normalization Strategies 
We implement three progressive normalization strategies to stabilize training: 
1. **ARPO**: Preserves raw rewards directly 
2. **BRPO**: Normalizes rewards across each training batch using batch statistics
3. **GRPO**: Normalizes within prompt groups to balance learning across varying task difficulties
-->

## Environment Setup
For detailed setup instructions, please check our [documentation](https://ragen-tutorial.readthedocs.io/). Here's a quick start guide:

```bash
# Setup environment and download data (2.7MB)
bash scripts/setup_ragen.sh
```

If this fails, you can follow the manual setup instructions in `scripts/setup_ragen.md`.

## Training Models
Here's how to train models with RAGEN:

### Export variables and train
We provide default configuration in `config/base.yaml`. This file includes symbolic links to:
- `config/ppo_trainer.yaml` 
- `config/envs.yaml`

The base configuration automatically inherits all contents from these two config files, creating a unified configuration system.

To train:

```bash
python train.py --config-name base
```

### Parameter efficient training with LoRA
We provide a default configuration with LoRA enabled in `config/base-lora.yaml`. To customize the LoRA settings, see the the `lora` section at the top of the configuration file.

To train with LoRA:

```bash
python train.py --config-name base-lora
```

<!--
## Supervised Finetuning (Optional)
For supervised finetuning with LoRA:

1. Create supervised finetuning data:
```bash
bash sft/generate_data.sh <env_type>
```

2. Finetune the model:
```bash
bash sft/finetune_lora.sh <env_type> <num_gpus> <save_path>
```

3. Merge LoRA weights with the base model:
```bash
python sft/utils/merge_lora.py \
    --base_model_name <base_model_name> \
    --lora_model_path <lora_model_path> \
    --output_path <output_path>
```
-->

## Visualization
Check `val/generations` in wandb


## Performance

We evaluate RAGEN across multiple environments. Below are results Qwen-2.5-0.5B-Instruct on Sokoban, Frozenlake, and Bandit. 
- No KL loss or KL penalty was applied during training
- We selectively retained only the top 25% of trajectories that successfully completed their respective tasks

<p align="center" style="display: flex; justify-content: center; align-items: center; flex-direction: column; gap: 20px; max-width: 500px; margin: 0 auto;">
    <img src="public/exp1.png" width="250px" alt="Bandit" />
    <img src="public/exp2.png" width="250px"  alt="Simple Sokoban" />
    <img src="public/exp3.png" width="250px"  alt="Frozen lake" />
</p>

We demonstrate RAGEN's robust generalization ability by training on simple Sokoban environments (6×6 with 1 box) and successfully evaluating performance on:
- Larger Sokoban environments (8×8 with 2 boxes)
- Simple Sokoban with alternative grid vocabulary representations
- FrozenLake environments

<p align="center" style="display: flex; justify-content: center; align-items: center; flex-direction: column; gap: 20px; max-width: 500px; margin: 0 auto;">
    <img src="public/exp4.png" width="250px" alt="Larger Sokoban" />
    <img src="public/exp5.png" width="250px"  alt="Sokoban with Different Grid Vocabulary" />
    <img src="public/exp6.png" width="250px"  alt="Frozen lake" />
</p>

Key observations:
- By using no KL and filtering out failed trajectories, we can achieve better and stable performance
- Generalization results highlight RAGEN's capacity to transfer learned policies across varying environment complexities, representations, and domains.


## Modular System Design of RAGEN

We implement RAGEN as a modular system: there are three main modules: **Environment State Manager** (`ragen/llm_agent/es_manager.py`), **Context Manager** (`ragen/llm_agent/ctx_manager.py`), and **Agent Proxy** (`ragen/llm_agent/agent_proxy.py`).

- Environment State Manager (**es_manager**):
  - Supports multiple environments (different environments, same environment different seeds, same environment same seed)
  - Records states of each environment during rollout
  - Processes actions from **ctx_manager**, executes step, and returns action results (observations) to **ctx_manager** in a batch-wise manner
- Context Manager (**ctx_manager**):
  - Parses raw agent tokens into structured actions for the **es_manager**
  - Formats observation from **es_manager**, parses and formulates them for following rollout of agent.
  - Gathers final rollout trajectories and compiles them into tokens, attention masks, reward scores, and loss masks for llm updating.
- Agent Proxy (**agent_proxy**): Serves as the interface for executing single or multi-round rollouts

## Adding Custom Environments

To add a new environment to our framework:

1. Implement an OpenAI Gym-compatible environment in `ragen/env/new_env/env.py` with these required methods:
   - `step(action)`: Process actions and return next state
   - `reset(seed)`: Initialize environment with new seed
   - `render()`: Return current state observation
   - `close()`: Clean up resources

2. Define environment configuration in `ragen/env/new_env/config.py`

3. Register your environment in `config/envs.yaml`:
   ```yaml
   custom_envs:
     - NewEnvironment # Tag
       - env_type: new_env  # Must match environment class name
       - max_actions_per_traj: 50  # Example value
       - env_instruction: "Your environment instructions here"
       - env_config: {}  # Configuration options from config.py
   ```

4. Add the environment tag to the `es_manager` section in `config/base.yaml`

## Evaluation
RAGEN provides a easy way to evaluate a model:
```bash
python -m ragen.llm_agent.agent_proxy --config-name <eval_config>
```
You only need to set model and environment to evaluate in `config/<eval_config>.yaml`
