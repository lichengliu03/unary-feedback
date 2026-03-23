# Experiment 1: Unary Feedback Training Across Environments

## Goal

Train LLMs with RL using **unary feedback** (minimal "try again" signals) across both single-turn and multi-turn environments. The model learns to retry and improve its answers/strategies based on simple failure feedback, without receiving detailed corrections.

## Recommended workflow

1. Run `scripts/quick_test.sh` first.
2. Confirm that the environment can finish 10 training steps cleanly.
3. Check the quick-test summary log for startup cost, validation cost, and estimated per-step training time.
4. Launch the full run with `scripts/exp1_train.sh` only after the quick test looks healthy.

For new environments, `Quick test` is the recommended first checkpoint before spending time on a longer Experiment 1 run.

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

## Scripts

### Quick test

`scripts/quick_test.sh` is the lightweight entry point for:

- sanity-checking a new environment before a full run
- checking that 10 training steps complete without crashing
- getting a coarse profiling signal for startup cost, validation cost, and per-step training time

Example usage:

```bash
# Quick test on one environment
ENV_TAGS_STR="MetamathQA" NGPUS=1 bash scripts/quick_test.sh

# Quick test on multiple environments
ENV_TAGS_STR="Countdown SimpleSokoban FrozenLake MetamathQA" NGPUS=2 bash scripts/quick_test.sh
```

### Experiment 1 train

`scripts/exp1_train.sh` is the main Experiment 1 launcher. It runs the unary-feedback PPO setup used in this document.

Example usage:

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

- chooses `MAX_TURN=5` for single-turn environments such as `MetamathQA` and `Countdown`
- chooses `MAX_TURN=15` for multi-turn environments such as `SimpleSokoban` and `FrozenLake`
- uses the base PPO config and applies the Experiment 1 overrides
- writes logs to `outputs/logs/exp1_<ENV_TAG>.log`
- writes checkpoints to `outputs/checkpoints/exp1_<ENV_TAG>/`


## Configuration

Key environment variables in `scripts/exp1_train.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENV_TAG` | MetamathQA | Environment to train on |
| `NGPUS` | 2 | Number of GPUs (1 or 2) |
| `STEPS` | 200 | Total training steps |
| `SAVE_FREQ` | 50 | Checkpoint frequency; set `-1` to disable saves |
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

- **Training checkpoints**: `outputs/checkpoints/exp1_<ENV_TAG>/`
- **Training logs**: `outputs/logs/exp1_<ENV_TAG>.log`
- **Quick-test summary logs**: `outputs/logs/quick_test_summary_<NGPUS>gpu.log`
- **W&B**: project `ufb_train`, experiment `exp1_<ENV_TAG>`
