# Scripts

## Directory Layout

| Subdirectory | Purpose |
|---|---|
| `training/` | SLURM job scripts for PPO training (normal, critique, no-feedback) |
| `evaluation/` | SLURM job scripts for model evaluation |
| `analysis/` | Plotting and visualization (feedback comparison, Pass@k curves, etc.) |
| `utils/` | Checkpoint conversion (`convert_fsdp_to_hf.py`), data processing, tests |
| `docs/` | Additional documentation on critique feedback, no-feedback experiments |

## Key Scripts

**Setup**: `bash scripts/setup_ufb.sh` — creates conda env, installs dependencies, downloads data.

**Training** (submit via `sbatch`):
- `training/submit_base_train.sh` — normal "Try Again" feedback
- `training/submit_critique_train.sh` — detailed critique feedback
- `training/submit_no_feedback_train.sh` — no feedback (ablation)

**Evaluation**: see `evaluation/` for SLURM submission scripts.

**Checkpoint conversion**: `python scripts/utils/convert_fsdp_to_hf.py --fsdp_checkpoint_path <path> --output_path <out>`

## Environment Variables

Training scripts accept these overrides:
- `MODEL_PATH` — base model (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `TRAIN_MAX_TURN` / `EVAL_MAX_TURN` — max interaction turns (default: 5)
- `HF_TOKEN` — HuggingFace token for gated models
