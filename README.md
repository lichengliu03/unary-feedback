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
│   ├── env/                    #   Environment definitions
│   │   ├── metamathqa/         #     MetaMathQA (normal / critique / no-feedback)
│   │   └── static/             #     Static benchmark environments (GSM8k, MATH, AIME, etc.)
│   ├── llm_agent/              #   Agent proxy, context & episode-state management
│   ├── trainer/                #   PPO trainer adapted for multi-turn episodes
│   ├── workers/                #   Distributed FSDP actor, critic, rollout workers
│   ├── eval.py                 #   Multi-turn evaluation with feedback (Succ@k)
│   ├── eval_api.py             #   API-based evaluation (OpenAI, Anthropic, etc.)
│   ├── eval_independent_passk.py  # Single-turn independent Pass@k evaluation
│   └── utils.py                #   Shared utilities
│
├── verl/                       # veRL distributed RL infrastructure (vendored)
│
├── configs/                    # Hydra configuration
│   ├── base.yaml               #   Default training config (normal feedback)
│   ├── base_critique_fixed.yaml#   Critique feedback training
│   ├── no_feedback_eval.yaml   #   No-feedback evaluation (ablation)
│   ├── eval.yaml               #   General evaluation config
│   ├── envs.yaml               #   All environment definitions
│   └── ppo_trainer.yaml        #   PPO algorithm hyperparameters
│
├── scripts/
│   ├── training/               #   SLURM job scripts for training
│   ├── evaluation/             #   SLURM job scripts for evaluation
│   ├── analysis/               #   Plotting & visualization scripts
│   ├── utils/                  #   Checkpoint conversion, data processing tools
│   ├── docs/                   #   Additional documentation (critique, no-feedback, etc.)
│   ├── setup_ufb.sh            #   Automated environment setup
│   └── download_data.py        #   Dataset download script
│
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

We provide default configuration in `configs/base.yaml`, which automatically inherits from `configs/ppo_trainer.yaml` and `configs/envs.yaml`.

### Quick start

```bash
python train.py --config-name base
```

### SLURM cluster

```bash
# Normal feedback training
sbatch scripts/training/submit_base_train.sh

# With critique feedback
sbatch scripts/training/submit_critique_train.sh

# No feedback (ablation)
sbatch scripts/training/submit_no_feedback_train.sh
```

### Configuration overrides

```bash
python train.py --config-name base \
    model_path=Qwen/Qwen2.5-3B-Instruct \
    trainer.total_training_steps=200 \
    agent_proxy.max_turn=5
```

## Evaluation

```bash
python -m ufb.eval --config-name eval
```

You only need to set model and environment in `configs/eval.yaml`.

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
