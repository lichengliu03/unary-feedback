# UFB — Core Implementation

The `ufb` package implements the UFO (Unary Feedback as Observation) framework for multi-turn RL training of language models.

## Modules

| Module | Purpose |
|---|---|
| `env/metamathqa/` | MetaMathQA environment with normal, critique, and no-feedback variants |
| `env/static/` | Static benchmark environments (GSM8k, MATH, AIME24, HotpotQA, etc.) |
| `env/base.py` | Base environment interface |
| `llm_agent/agent_proxy.py` | Manages multi-turn rollout episodes |
| `llm_agent/ctx_manager.py` | Prompt context construction |
| `llm_agent/es_manager.py` | Episode-state manager (env sampling, reward tracking) |
| `trainer/agent_trainer.py` | PPO trainer adapted for multi-turn episodes (extends veRL) |
| `workers/` | Distributed FSDP workers (actor, critic, rollout, sharding) |

## Evaluation

| Script | Metric | Description |
|---|---|---|
| `eval.py` | Succ@k | Multi-turn evaluation with feedback |
| `eval_api.py` | Succ@k | Same, but calls external APIs (GPT-4, Claude, etc.) |
| `eval_independent_passk.py` | Pass@k | Single-turn, k independent samples, no feedback |
