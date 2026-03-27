# Scripts

## Core Scripts

| Script | Usage |
|---|---|
| `train.sh` | `ENV=sokoban sbatch scripts/train.sh` |
| `eval.sh` | `ENV=sokoban MODEL=<path> sbatch scripts/eval.sh` |
| `transfer_to_hf.sh` | `CKPT_DIR=... MODEL_PATH=... bash scripts/transfer_to_hf.sh` |
| `download_data.py` | `python scripts/download_data.py` |
| `setup_ufb.sh` | `bash scripts/setup_ufb.sh` |

## Utilities (`utils/`)

| Script | Purpose |
|---|---|
| `convert_fsdp_to_hf.py` | Export FSDP checkpoints to HuggingFace format and optionally upload to the Hub |
| `convert_out_to_json.py` | Parse SLURM `.out` logs into structured JSON |
| `compute_conditional_success.py` | Compute conditional success rates from results |

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
5. Transfer:  CKPT_DIR=<ckpt_or_exp_dir> MODEL_PATH=<base_model> bash scripts/transfer_to_hf.sh
6. Upload:    HF_REPO_ID=<user/repo> CKPT_DIR=<ckpt_or_exp_dir> MODEL_PATH=<base_model> bash scripts/transfer_to_hf.sh
7. Analyze:   python scripts/utils/convert_out_to_json.py <slurm.out> <result.json>
```

## Hugging Face export

```bash
# Export the latest checkpoint under an experiment directory
CKPT_DIR=outputs/checkpoints/exp1_MetamathQA \
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
OUTPUT_DIR=outputs/hf/exp1_MetamathQA \
bash scripts/transfer_to_hf.sh

# Export and upload to the Hub
HF_TOKEN=hf_xxx \
HF_REPO_ID=your-name/exp1-metamathqa \
CKPT_DIR=outputs/checkpoints/exp1_MetamathQA \
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
bash scripts/transfer_to_hf.sh

# If needed, you can still force the shard count explicitly
WORLD_SIZE=1 \
CKPT_DIR=outputs/checkpoints/exp1_MetamathQA/top_k/global_step_10 \
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
bash scripts/transfer_to_hf.sh
```
