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

By default, the downloader fetches the full released Exp1 set:

| Model Key | Source Type | Local Output |
|---|---|---|
| `metamathqa` | direct HF model | `loaded_checkpoints/exp1/metamathqa/` |
| `countdown` | direct HF model | `loaded_checkpoints/exp1/countdown/` |
| `simplesokoban` | direct HF model | `loaded_checkpoints/exp1/simplesokoban/` |
| `frozenlake` | direct HF model | `loaded_checkpoints/exp1/frozenlake/` |
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

Download only selected models:

```bash
python scripts/utils/download_exp1_models.py --models metamathqa countdown
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
├── countdown/
├── frozenlake/
├── hotpotqa/
├── hotpotqa_raw/
├── metamathqa/
├── simplesokoban/
├── webshop/
└── webshop_raw/
```

The evaluation scripts auto-discover model directories from this root unless you pass explicit model paths.
