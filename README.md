<h1 align="center"> <em>UFO</em>: A Simple "Try Again" Can Elicit Multi-Turn LLM Reasoning </h1>

<p align="center">
  <a href="https://huggingface.co/LichengLiu03/Qwen2.5-3B-UFO">
    <img src="https://img.shields.io/badge/View_on-HuggingFace-yellow?logo=huggingface&style=for-the-badge" alt="View on Hugging Face"/>
  </a>
  &nbsp;
  <a href="https://unary-feedback.github.io/">
    <img src="https://img.shields.io/badge/Project-Website-blue?logo=googlechrome&style=for-the-badge" alt="Project Homepage"/>
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2507.14295">
    <img src="https://img.shields.io/badge/View_on-arXiv-B31B1B?logo=arxiv&style=for-the-badge" alt="View on arXiv"/>
  </a>
</p>

## Overview

**"Let's Try Again"** addresses a critical gap in language model training: while single-turn reinforcement learning (RL) improves reasoning, these models fail in **multi-turn interactive scenarios**, often repeating the same wrong answers despite feedback.

### Key Problem
Single-turn RL models lose the ability to revise reasoning across multiple turns. In 70% of failure cases, they produce identical answers across 5 interaction rounds, unable to incorporate simple feedback like "try again."

### Solution: UFO Framework
**Unary Feedback as Observation (UFO)** transforms static datasets into multi-turn training by:
- Using only minimal feedback signals ("Try Again")
- Treating failure feedback as part of the observation
- Enabling models to learn from historical mistakes

### Results
- **14% improvement** in multi-turn success rates
- **10% reduction** in average interaction turns
- Better performance even in single-turn scenarios
- **90% non-repetitive answers** (vs 80% baseline)

## Repository Structure

```
unary-feedback/
├── ufb/                        # Core UFO framework
│   ├── env/                    #   13 environment types (metamathqa, sokoban, sudoku, ...)
│   ├── llm_agent/              #   Agent proxy, context & episode-state management
│   ├── trainer/                #   PPO trainer adapted for multi-turn episodes
│   ├── workers/                #   Distributed FSDP actor, critic, rollout workers
│   ├── eval.py                 #   Multi-turn evaluation with feedback (Succ@k)
│   ├── eval_api.py             #   API-based evaluation (OpenAI, Anthropic, etc.)
│   └── utils.py                #   Shared utilities
│
├── verl/                       # veRL distributed RL infrastructure (vendored)
│
├── configs/
│   ├── base.yaml               #   Base training config (normal feedback)
│   ├── train_generic_feedback.yaml     #   Generic feedback training
│   ├── train_no_feedback.yaml  #   No-feedback training (ablation)
│   ├── envs/                   #   Per-environment configs (28 environments)
│   ├── envs.yaml               #   Environment definitions (33 tags)
│   └── ppo_trainer.yaml        #   PPO algorithm hyperparameters
│
├── scripts/
│   ├── train.sh                #   Training (ENV=sokoban sbatch scripts/train.sh)
│   ├── eval.sh                 #   Evaluation (ENV=sokoban MODEL=... sbatch scripts/eval.sh)
│   ├── download_data.py        #   Dataset download
│   ├── setup_ufb.sh            #   Environment setup
│   └── utils/                  #   Checkpoint conversion, data processing tools
│
├── external/                   # External dependencies (webshop-minimal)
├── train.py                    # Main training entry point
├── setup.py                    # Package installation (pip install -e .)
├── requirements.txt            # Python dependencies
├── LICENSE                     # Apache 2.0
└── README.md
```

## Setup

```bash
# Clone and setup
git clone https://github.com/lichengliu03/unary-feedback.git
cd unary-feedback
bash scripts/setup_ufb.sh
```

For manual setup, see `scripts/setup_ufb.md`.

## Training

Each environment has a pre-configured YAML in `configs/envs/` with appropriate hyperparameters (response length, max turns, batch size, etc.). Just specify the environment name:

```bash
ENV=metamathqa  sbatch scripts/train.sh
ENV=sokoban     sbatch scripts/train.sh
ENV=countdown   sbatch scripts/train.sh
ENV=frozen_lake sbatch scripts/train.sh
ENV=sudoku      sbatch scripts/train.sh
ENV=bandit      sbatch scripts/train.sh
ENV=gsm8k       sbatch scripts/train.sh
ENV=hotpotqa    sbatch scripts/train.sh
ENV=webshop     sbatch scripts/train.sh
ENV=alfworld    sbatch scripts/train.sh
# ... see configs/envs/ for all 28 environments
```

Optional overrides:

```bash
# Different model
ENV=sokoban MODEL_PATH=Qwen/Qwen2.5-7B-Instruct sbatch scripts/train.sh

# Different number of training steps
ENV=metamathqa STEPS=100 sbatch scripts/train.sh
```

Rollout filtering presets are documented in `docs/rollout_filtering.md`.

### Feedback variants

By default, one-bit feedback is randomly sampled from a prompt pool (e.g. "Incorrect.", "That's wrong, try again.", "Not quite right.", etc.) to prevent overfitting to specific wording. To use a fixed feedback prompt instead, set `randomize_feedback: false` in the env config.

```bash
# Normal feedback with randomized prompt pool (default)
ENV=metamathqa sbatch scripts/train.sh

# Fixed feedback (ablation: single prompt "Incorrect. Please think again.")
ENV=metamathqa sbatch scripts/train.sh  # override env_config.randomize_feedback=false

# Generic feedback
ENV=metamathqa CONFIG=train_generic_feedback sbatch scripts/train.sh

# Specific feedback (answer-directed hint)
ENV=metamathqa CONFIG=train_specific_feedback sbatch scripts/train.sh

# No feedback (ablation)
ENV=metamathqa CONFIG=train_no_feedback sbatch scripts/train.sh
```

### Available environments

| Category | Environments |
|----------|-------------|
| Math | `metamathqa`, `gsm8k`, `math`, `aime24`, `countdown`, `theoremqa` |
| QA / Reasoning | `hotpotqa`, `concurrentqa`, `musique`, `gpqa` |
| General Knowledge | `mmlu`, `mmlu_pro`, `mmlu_stem`, `mmlu_redux` |
| Code | `humaneval`, `mbpp`, `multiple`, `livecodebench`, `livebench` |
| Planning | `sokoban`, `frozen_lake`, `sudoku`, `spatial` |
| Interactive | `webshop`, `alfworld`, `bandit` |
| Formal | `lean`, `search` |

## Evaluation

```bash
# Evaluate base model (no training)
ENV=metamathqa MODEL=Qwen/Qwen2.5-3B-Instruct sbatch scripts/eval.sh

# Evaluate a trained checkpoint
ENV=metamathqa MODEL=/path/to/checkpoint/global_step_200 sbatch scripts/eval.sh

# Evaluate on a different environment
ENV=sokoban MODEL=/path/to/checkpoint sbatch scripts/eval.sh
```

## Visualization
Check `val/generations` in wandb.

## Key Results

### Multi-Turn Reasoning Performance

We compare our multi-turn UFO model against a strong single-turn PPO baseline. For a fair comparison, the baseline is evaluated on 5 independent samples (Pass@5), while our model uses 5 sequential attempts with feedback (Succ@5).

**Key Findings:**
- **+14% success rate** over single-turn PPO baseline
- Benefits generalize to both multi-turn and single-turn inference
- Best results with 5-turn training; more turns yield diminishing returns

### Effectiveness of Unary Feedback

- Feedback in both training and validation is crucial for improvement
- Feedback only in training phase does **not** help at inference

### Reward Design Impact

- **Exponential Reward Decay**: Decreases average actions required by ~10%
- **Answer Diversity**: Non-repetitive answer ratio increases from 79.7% to 92.8%

## Citation

```bibtex
@article{liu2025ufo,
  title={UFO: A Simple "Try Again" Can Elicit Multi-Turn LLM Reasoning},
  author={Liu, Licheng and others},
  journal={arXiv preprint arXiv:2507.14295},
  year={2025}
}
```

## Acknowledgements

We thank the [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) team for providing the DeepSeek-R1 model and early conceptual inspirations. We are grateful to the [veRL](https://github.com/volcengine/verl) team for their infrastructure support and the [RAGEN](https://github.com/RAGEN-AI/RAGEN) team for their multi-turn RL framework.
