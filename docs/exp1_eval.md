# Experiment 1 Evaluation

## Goal

Evaluate Experiment 1 models across environments without repeatedly reloading the same model. The main launcher loads one model, runs all selected evaluation environments, saves the outputs, and then moves to the next model.

Related docs:

- [exp1_train.md](./exp1_train.md)
- [exp1_download.md](./exp1_download.md)

## Current Eval Scope

`scripts/exp1_eval.sh` currently evaluates these four environments:

- `MetamathQA`
- `Countdown`
- `SimpleSokoban`
- `FrozenLake`

The default evaluation protocol uses:

- one-bit feedback
- `MAX_ATTEMPTS=5`
- `VAL_GROUPS_PER_ENV=1024`
- `CUDA_DEVICES=0`
- eval progress logging enabled by default

## Environment Semantics

The evaluation script is aligned with the training environment semantics, with the main difference that the attempt budget is unified to `5`.

| Environment | Attempt Definition | Default Eval Budget |
|---|---|---|
| `MetamathQA` | each turn is one attempt | `5` turns total |
| `Countdown` | each turn is one attempt | `5` turns total |
| `SimpleSokoban` | retry wrapper attempt | `5` attempts x `5` turns/attempt |
| `FrozenLake` | retry wrapper attempt | `5` attempts x `5` turns/attempt |

This means:

- `MetamathQA` and `Countdown` stay as native training-style single-turn retry environments.
- `SimpleSokoban` and `FrozenLake` keep their retry-wrapper structure, but `max_retry_attempts` is overridden to `5`.

## Main Script

The main evaluation launcher is:

```bash
bash scripts/exp1_eval.sh
```

### Model Loop Behavior

If you pass multiple models, the script runs them sequentially:

1. load one model
2. evaluate all selected environments
3. save outputs
4. move to the next model

So the script does not reload the same model once per environment.

### Progress Logging

`scripts/exp1_eval.sh` now prints validation progress by turn during each model run.

- environments are evaluated together in one rollout, not one-by-one
- each progress line shows total completed episodes and a per-environment breakdown
- when one environment finishes all its assigned samples, the log prints an explicit completion line

Example knobs:

```bash
SHOW_EVAL_PROGRESS=1 EVAL_PROGRESS_INTERVAL=1 bash scripts/exp1_eval.sh
```

Set `SHOW_EVAL_PROGRESS=0` to silence the turn-by-turn progress lines.

## Common Usage

Evaluate one model on the default four environments:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate ragen && \
MODEL=loaded_checkpoints/exp1/metamathqa \
bash scripts/exp1_eval.sh
```

Evaluate two models sequentially:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate ragen && \
MODELS_STR="loaded_checkpoints/exp1/metamathqa loaded_checkpoints/exp1/countdown" \
bash scripts/exp1_eval.sh
```

Evaluate models listed in a file:

```bash
MODELS_FILE=scripts/exp1_models.txt bash scripts/exp1_eval.sh
```

Run into a named output subdirectory:

```bash
RUN_NAME=my_eval bash scripts/exp1_eval.sh
```

Override sample count or attempt budget:

```bash
VAL_GROUPS_PER_ENV=128 MAX_ATTEMPTS=5 bash scripts/exp1_eval.sh
```

## Model Selection

The script resolves models in this order:

1. `MODEL`
2. `MODELS_STR`
3. `MODELS_FILE`
4. auto-discovery under `loaded_checkpoints/exp1/`

The current auto-discovery list includes:

- `metamathqa`
- `countdown`
- `simplesokoban`
- `hotpotqa`
- `frozenlake`
- `webshop`

Only directories that actually exist are used.

## Metrics

The summary pipeline reports attempt-level metrics.

- `pass@k` means cumulative success within the first `k` attempts
- `Succ@k | fail@(k-1)` means conditional success at attempt `k` given failure through attempt `k-1`

Attempt source depends on the environment:

- `MetamathQA` and `Countdown`: derived from turn index
- `SimpleSokoban` and `FrozenLake`: derived from retry `attempt_num`

## Output Layout

By default, the script does not add a timestamped run folder.

- results default to `eval_results/exp1_generalization/`
- logs default to `logs/exp1_eval/`

If `RUN_NAME` is set, outputs go under:

- `eval_results/exp1_generalization/<RUN_NAME>/`
- `logs/exp1_eval/<RUN_NAME>/`

Per-model result structure:

```text
eval_results/exp1_generalization/
├── results.tsv
├── <model_slug>/
│   ├── combined.json
│   ├── combined.params.json
│   ├── combined.summary.json
│   ├── combined.conditional_success.json
│   └── by_env/
│       ├── MetamathQA.summary.json
│       ├── MetamathQA.conditional_success.json
│       ├── Countdown.summary.json
│       ├── Countdown.conditional_success.json
│       ├── SimpleSokoban.summary.json
│       ├── SimpleSokoban.conditional_success.json
│       ├── FrozenLake.summary.json
│       └── FrozenLake.conditional_success.json
```

The top-level `results.tsv` flattens one row per model-environment pair for quick inspection.
