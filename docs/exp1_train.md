# Experiment 1 Training

## Goal

Train LLMs with RL using unary feedback across both single-turn and multi-turn environments. The model learns to retry after minimal failure feedback without seeing detailed corrections.

Related docs:

- [exp1_download.md](./exp1_download.md)
- [exp1_eval.md](./exp1_eval.md)

## Recommended Workflow

1. Run `scripts/quick_test.sh` first.
2. Confirm that the environment can finish 10 training steps cleanly.
3. Check the quick-test summary log for startup cost, validation cost, and estimated per-step training time.
4. Launch the full run with `scripts/exp1_train.sh` only after the quick test looks healthy.

For new environments, the quick test is the recommended first checkpoint before spending time on a longer Experiment 1 run.

## Environments

| Environment | Type | Retry Mechanism |
|---|---|---|
| `MetamathQA` | Single-turn | Wrong answer -> unary feedback -> retry same question |
| `Countdown` | Single-turn | Wrong equation -> unary feedback -> retry same target |
| `SimpleSokoban` | Multi-turn | Failed attempt -> environment reset -> retry same puzzle |
| `FrozenLake` | Multi-turn | Failed attempt -> environment reset -> retry same puzzle |

### Single-turn Retry

- One model answer is produced per turn.
- If the answer is wrong, the environment returns randomized one-bit feedback and keeps `done=False`.
- The model retries on the same task with the full history visible.
- Reward decays exponentially across attempts.

### Multi-turn Retry

- The model acts across multiple turns inside one attempt.
- Each attempt has its own turn and action budget.
- If the attempt fails, the environment resets to the same initial state and injects retry feedback.
- The model sees the failed history plus the fresh reset state.
- Reward decays exponentially across attempts.

## Scripts

### Quick Test

`scripts/quick_test.sh` is the lightweight entry point for:

- sanity-checking a new environment before a full run
- checking that 10 training steps complete without crashing
- getting a coarse profiling signal for startup cost, validation cost, and per-step training time

Examples:

```bash
# Quick test on one environment
ENV_TAGS_STR="MetamathQA" NGPUS=1 bash scripts/quick_test.sh

# Quick test on multiple environments
ENV_TAGS_STR="Countdown SimpleSokoban FrozenLake MetamathQA" NGPUS=2 bash scripts/quick_test.sh
```

### Main Training Script

`scripts/exp1_train.sh` is the main Experiment 1 launcher.

Examples:

```bash
# Default: MetamathQA, 2 GPUs, 200 steps
bash scripts/exp1_train.sh

# Choose environment and GPU count
ENV_TAG=Countdown NGPUS=2 bash scripts/exp1_train.sh
ENV_TAG=SimpleSokoban NGPUS=1 bash scripts/exp1_train.sh
ENV_TAG=FrozenLake NGPUS=2 bash scripts/exp1_train.sh

# Override other parameters
ENV_TAG=MetamathQA NGPUS=1 STEPS=100 MODEL_PATH=Qwen/Qwen2.5-7B-Instruct bash scripts/exp1_train.sh

# Disable checkpoint saving for short debug runs
ENV_TAG=Countdown NGPUS=1 STEPS=10 SAVE_FREQ=-1 bash scripts/exp1_train.sh
```

What `exp1_train.sh` sets for you:

- `MAX_TURN=5` for single-turn environments such as `MetamathQA` and `Countdown`
- `MAX_TURN=15` for multi-turn environments such as `SimpleSokoban` and `FrozenLake`
- the base PPO config plus Experiment 1 overrides
- logs at `outputs/logs/exp1_<ENV_TAG>.log`
- checkpoints at `outputs/checkpoints/exp1_<ENV_TAG>/`

## Configuration

Key environment variables in `scripts/exp1_train.sh`:

| Parameter | Default | Description |
|---|---|---|
| `ENV_TAG` | `MetamathQA` | Environment to train on |
| `NGPUS` | `2` | Number of GPUs |
| `STEPS` | `200` | Total training steps |
| `SAVE_FREQ` | `50` | Checkpoint frequency; set `-1` to disable saves |
| `MAX_TURN` | auto | `5` for single-turn envs, `15` for multi-turn envs |
| `MODEL_PATH` | `Qwen/Qwen2.5-3B-Instruct` | Model to train |

Multi-turn retry config is defined per environment in `configs/envs.yaml`:

```yaml
SimpleSokoban:
  retry:
    max_turns_per_attempt: 5
    max_actions_per_attempt: 10
    max_retry_attempts: 3
    reward_decay_base: 2.0
```

## Outputs

- Training checkpoints: `outputs/checkpoints/exp1_<ENV_TAG>/`
- Training logs: `outputs/logs/exp1_<ENV_TAG>.log`
- Quick-test summary logs: `outputs/logs/quick_test_summary_<NGPUS>gpu.log`
- W&B project: `ufb_train`
- W&B experiment: `exp1_<ENV_TAG>`
