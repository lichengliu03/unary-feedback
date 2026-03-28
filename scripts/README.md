# Scripts

## Core Scripts

| Script | Usage |
|---|---|
| `train.sh` | `ENV=sokoban sbatch scripts/train.sh` |
| `eval.sh` | `ENV=sokoban MODEL=<path> sbatch scripts/eval.sh` |
| `download_data.py` | `python scripts/download_data.py` |
| `setup_ufb.sh` | `bash scripts/setup_ufb.sh` |

## Utilities (`utils/`)

| Script | Purpose |
|---|---|
| `convert_fsdp_to_hf.py` | Convert FSDP checkpoints to HuggingFace format |
| `convert_and_upload_checkpoints_to_hf.py` | Convert `global_step_*/actor` checkpoints, upload to HF, and delete temp HF dirs |
| `convert_out_to_json.py` | Parse SLURM `.out` logs into structured JSON |
| `upload_hf_models_in_dir.py` | Upload every HF model subdirectory under a parent folder |
| `compute_conditional_success.py` | Compute conditional success rates from results |

HF upload notes: see `scripts/utils/HF_UPLOAD.md`.

## Training

```bash
# Pick any environment
ENV=metamathqa sbatch scripts/train.sh
ENV=sokoban sbatch scripts/train.sh

# Optional overrides
ENV=sokoban MODEL_PATH=Qwen/Qwen2.5-7B-Instruct sbatch scripts/train.sh
ENV=metamathqa STEPS=100 sbatch scripts/train.sh

# Feedback variants
ENV=metamathqa CONFIG=train_critique sbatch scripts/train.sh
```

## Evaluation

```bash
# Evaluate base model
ENV=metamathqa MODEL=Qwen/Qwen2.5-3B-Instruct sbatch scripts/eval.sh

# Evaluate checkpoint
ENV=metamathqa MODEL=/path/to/checkpoint sbatch scripts/eval.sh

# Override eval turns
ENV=metamathqa MODEL=/path/to/checkpoint EVAL_TURN=10 sbatch scripts/eval.sh
```

## Workflow

```
1. Setup:     bash scripts/setup_ufb.sh
2. Data:      python scripts/download_data.py
3. Train:     ENV=metamathqa sbatch scripts/train.sh
4. Evaluate:  ENV=metamathqa MODEL=<ckpt> sbatch scripts/eval.sh
5. Convert:   python scripts/utils/convert_fsdp_to_hf.py <ckpt> <output>
6. Analyze:   python scripts/utils/convert_out_to_json.py <slurm.out> <result.json>
```
