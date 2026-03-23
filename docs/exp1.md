# Experiment 1: Unary Feedback Training Across Environments

## Goal

Train LLMs with RL using **unary feedback** (minimal "try again" signals) across both single-turn and multi-turn environments. The model learns to retry and improve its answers/strategies based on simple failure feedback, without receiving detailed corrections.

## Environments

| Environment | Type | Retry Mechanism |
|-------------|------|-----------------|
| **MetamathQA** | Single-turn | Wrong answer → "Incorrect. Try again." → retry same question |
| **Countdown** | Single-turn | Wrong equation → "Incorrect. Try again." → retry same target |
| **SimpleSokoban** | Multi-turn | Failed attempt (5 turns) → "You failed. Environment reset." → retry same puzzle |
| **FrozenLake** | Multi-turn | Failed attempt (5 turns or fell in hole) → "You failed. Environment reset." → retry same puzzle |

### Single-turn retry (MetamathQA, Countdown)
- Model gives one answer per turn
- If wrong: env returns feedback from a randomized pool + `done=False`
- Model retries on the same question with the full history visible
- Reward decays exponentially: attempt k gets `1/(2^k)` reward

### Multi-turn retry (SimpleSokoban, FrozenLake)
- Model takes multiple actions across turns (move up/down/left/right)
- Each attempt has a budget of 5 turns and 10 actions
- If the model fails (budget exhausted or env failure like falling in a hole), the environment resets to its initial state
- The model sees its full failed trajectory + retry feedback + fresh initial state
- Reward decays exponentially across attempts

## How to Run

### Local (interactive)

```bash
# Default: MetamathQA, 2 GPUs, 200 steps
bash scripts/exp1_train.sh

# Choose environment and GPU count
ENV_TAG=Countdown NGPUS=2 bash scripts/exp1_train.sh
ENV_TAG=SimpleSokoban NGPUS=1 bash scripts/exp1_train.sh
ENV_TAG=FrozenLake NGPUS=2 bash scripts/exp1_train.sh

# Override other parameters
STEPS=100 MODEL_PATH=Qwen/Qwen2.5-7B-Instruct ENV_TAG=MetamathQA bash scripts/exp1_train.sh
```

### SLURM (Delta cluster)

```bash
sbatch scripts/delta/exp1/metamathqa.slurm
sbatch scripts/delta/exp1/countdown.slurm
sbatch scripts/delta/exp1/simple_sokoban.slurm
sbatch scripts/delta/exp1/frozenlake.slurm
```

## Configuration

Key parameters in `scripts/exp1_train.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENV_TAG` | MetamathQA | Environment to train on |
| `NGPUS` | 2 | Number of GPUs (1 or 2) |
| `STEPS` | 200 | Total training steps |
| `MAX_TURN` | auto | Max turns per rollout (5 for single-turn, 15 for multi-turn) |
| `MODEL_PATH` | Qwen/Qwen2.5-3B-Instruct | Model to train |

Multi-turn retry config is defined per-environment in `configs/envs.yaml` under the `retry:` block:

```yaml
SimpleSokoban:
  retry:
    max_turns_per_attempt: 5    # max turns per attempt
    max_actions_per_attempt: 10  # max actions per attempt (resets on retry)
    max_retry_attempts: 3        # total attempts including the first
    reward_decay_base: 2.0       # reward multiplied by 1/(base^attempt_num)
```

## Outputs

- **Checkpoints**: `outputs/checkpoints/exp1_<ENV_TAG>/`
- **Logs**: `outputs/logs/exp1_<ENV_TAG>.log`
- **W&B**: project `ufb_train`, experiment `exp1_<ENV_TAG>`
