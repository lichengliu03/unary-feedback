# Experiment 1 Model Download

## Goal

Download the released Experiment 1 models into the local `loaded_checkpoints/exp1/` layout expected by the training and evaluation scripts.

Related docs:

- [exp1_train.md](./exp1_train.md)
- [exp1_eval.md](./exp1_eval.md)

## Main Script

The shell entry point is:

```bash
bash scripts/download_exp1_models.sh
```

That script simply calls:

```bash
python scripts/utils/download_exp1_models.py
```

## Default Model Set

By default, the downloader fetches the full released Exp1 set.

### Direct HF models (per-step checkpoints)

Keys follow the pattern `<task>_<step>` for steps 50, 100, 150, 200:

| Model Key | HF Repo | Local Output |
|---|---|---|
| `metamathqa_50` | `ZihanWang314/exp1_MetamathQA_global_step_50` | `loaded_checkpoints/exp1/metamathqa_50/` |
| `metamathqa_100` | `ZihanWang314/exp1_MetamathQA_global_step_100` | `loaded_checkpoints/exp1/metamathqa_100/` |
| `metamathqa_150` | `ZihanWang314/exp1_MetamathQA_global_step_150` | `loaded_checkpoints/exp1/metamathqa_150/` |
| `metamathqa_200` | `ZihanWang314/exp1_MetamathQA_global_step_200` | `loaded_checkpoints/exp1/metamathqa_200/` |
| `countdown_50` | `ZihanWang314/exp1_Countdown_global_step_50` | `loaded_checkpoints/exp1/countdown_50/` |
| `countdown_100` | `ZihanWang314/exp1_Countdown_global_step_100` | `loaded_checkpoints/exp1/countdown_100/` |
| `countdown_150` | `ZihanWang314/exp1_Countdown_global_step_150` | `loaded_checkpoints/exp1/countdown_150/` |
| `countdown_200` | `ZihanWang314/exp1_Countdown_global_step_200` | `loaded_checkpoints/exp1/countdown_200/` |
| `simplesokoban_50` | `ZihanWang314/exp1_SimpleSokoban_global_step_50` | `loaded_checkpoints/exp1/simplesokoban_50/` |
| `simplesokoban_100` | `ZihanWang314/exp1_SimpleSokoban_global_step_100` | `loaded_checkpoints/exp1/simplesokoban_100/` |
| `simplesokoban_150` | `ZihanWang314/exp1_SimpleSokoban_global_step_150` | `loaded_checkpoints/exp1/simplesokoban_150/` |
| `simplesokoban_200` | `ZihanWang314/exp1_SimpleSokoban_global_step_200` | `loaded_checkpoints/exp1/simplesokoban_200/` |
| `frozenlake_50` | `ZihanWang314/exp1_Frozenlake_global_step_50` | `loaded_checkpoints/exp1/frozenlake_50/` |
| `frozenlake_100` | `ZihanWang314/exp1_Frozenlake_global_step_100` | `loaded_checkpoints/exp1/frozenlake_100/` |
| `frozenlake_150` | `ZihanWang314/exp1_Frozenlake_global_step_150` | `loaded_checkpoints/exp1/frozenlake_150/` |
| `frozenlake_200` | `ZihanWang314/exp1_Frozenlake_global_step_200` | `loaded_checkpoints/exp1/frozenlake_200/` |

### Checkpoint-backed models (convert from raw)

| Model Key | Source Type | Local Output |
|---|---|---|
| `hotpotqa` | raw checkpoint download + convert | `loaded_checkpoints/exp1/hotpotqa/` |
| `webshop` | raw checkpoint download + convert | `loaded_checkpoints/exp1/webshop/` |

For checkpoint-backed models, the raw download is also kept locally:

- `loaded_checkpoints/exp1/hotpotqa_raw/`
- `loaded_checkpoints/exp1/webshop_raw/`

## Common Usage

Download the whole set:

```bash
bash scripts/download_exp1_models.sh
```

Download only selected models (use `<task>_<step>` keys):

```bash
python scripts/utils/download_exp1_models.py --models metamathqa_50 metamathqa_200 countdown_100
```

Download raw checkpoint-backed artifacts without converting:

```bash
python scripts/utils/download_exp1_models.py --models hotpotqa webshop --skip-convert
```

Force a fresh download:

```bash
python scripts/utils/download_exp1_models.py --force
```

Fail immediately if any requested model is missing:

```bash
python scripts/utils/download_exp1_models.py --strict
```

## What The Downloader Does

- Direct models are downloaded as ready-to-evaluate Hugging Face model folders.
- Checkpoint-backed models are downloaded first as raw training artifacts.
- For checkpoint-backed models, the latest `global_step_*` actor checkpoint is converted into a standard HF folder unless `--skip-convert` is set.

## Output Layout

A typical local layout looks like:

```text
loaded_checkpoints/exp1/
├── countdown_50/
├── countdown_100/
├── countdown_150/
├── countdown_200/
├── frozenlake_50/
├── frozenlake_100/
├── frozenlake_150/
├── frozenlake_200/
├── hotpotqa/
├── hotpotqa_raw/
├── metamathqa_50/
├── metamathqa_100/
├── metamathqa_150/
├── metamathqa_200/
├── simplesokoban_50/
├── simplesokoban_100/
├── simplesokoban_150/
├── simplesokoban_200/
├── webshop/
└── webshop_raw/
```

The evaluation scripts auto-discover model directories from this root unless you pass explicit model paths.
