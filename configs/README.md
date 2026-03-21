# Configurations

Hydra-based configuration files. All per-environment configs inherit from `base.yaml`, which itself inherits from `ppo_trainer.yaml` and `envs.yaml`.

## Structure

```
configs/
├── base.yaml                       # Base training config (randomized one-bit feedback, default)
├── train_generic_feedback.yaml     # Generic feedback training (fixed self-check template)
├── train_specific_feedback.yaml    # Specific feedback training (answer-directed hints)
├── train_no_feedback.yaml          # No-feedback training (ablation)
├── ppo_trainer.yaml                # PPO algorithm hyperparameters
├── envs.yaml                # All environment definitions (33 tags)
└── envs/                    # Per-environment configs (28 environments)
    ├── metamathqa.yaml
    ├── sokoban.yaml
    ├── countdown.yaml
    ├── frozen_lake.yaml
    ├── sudoku.yaml
    ├── bandit.yaml
    ├── gsm8k.yaml
    ├── math.yaml
    ├── aime24.yaml
    ├── hotpotqa.yaml
    ├── webshop.yaml
    ├── alfworld.yaml
    └── ...
```

## How it works

Each `configs/envs/<env>.yaml` inherits from `base.yaml` and overrides environment-specific parameters (env tag, response length, max turns, batch size). This means you only need to specify the environment name — all hyperparameters are pre-configured.

## Usage

```bash
# Training — just pick an environment
ENV=sokoban sbatch scripts/train.sh
ENV=metamathqa sbatch scripts/train.sh

# Feedback variants
ENV=metamathqa CONFIG=train_generic_feedback sbatch scripts/train.sh
ENV=metamathqa CONFIG=train_specific_feedback sbatch scripts/train.sh
ENV=metamathqa CONFIG=train_no_feedback sbatch scripts/train.sh

# Evaluation
ENV=sokoban MODEL=Qwen/Qwen2.5-3B-Instruct sbatch scripts/eval.sh
```
