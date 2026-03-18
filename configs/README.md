# Configurations

Hydra-based configuration files. Training configs inherit from `base.yaml`, which itself inherits from `ppo_trainer.yaml` and `envs.yaml`.

## Config Files

| File | Purpose |
|---|---|
| **Training** | |
| `base.yaml` | Base training config — normal "Try Again" feedback (MetaMathQA) |
| `train_critique.yaml` | Training with detailed critique feedback (MetaMathQACritique) |
| `train_no_feedback.yaml` | Training without feedback — ablation (MetaMathQANoFeedback) |
| **Evaluation** | |
| `eval.yaml` | General evaluation config (inherits from base, 0 training steps) |
| **Shared** | |
| `ppo_trainer.yaml` | PPO algorithm settings (lr, clip ratio, GAE, etc.) |
| `envs.yaml` | All environment definitions |

## Usage

```bash
# Training
python train.py --config-name base                 # normal feedback
python train.py --config-name train_critique        # critique feedback
python train.py --config-name train_no_feedback     # no feedback (ablation)

# Evaluation
python train.py --config-name eval model_path=<checkpoint_path>

# Override parameters
python train.py --config-name base \
    model_path=Qwen/Qwen2.5-3B-Instruct \
    trainer.total_training_steps=200 \
    agent_proxy.max_turn=5
```
