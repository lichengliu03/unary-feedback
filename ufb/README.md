# UFB — Core Implementation

The `ufb` package implements the UFO (Unary Feedback as Observation) framework for multi-turn RL training of language models.

## Modules

| Module | Purpose |
|---|---|
| `env/` | 13 environment types (see below) |
| `env/base.py` | Base environment interface |
| `llm_agent/agent_proxy.py` | Manages multi-turn rollout episodes |
| `llm_agent/ctx_manager.py` | Prompt context construction |
| `llm_agent/es_manager.py` | Episode-state manager (env sampling, reward tracking) |
| `trainer/agent_trainer.py` | PPO trainer adapted for multi-turn episodes (extends veRL) |
| `workers/` | Distributed FSDP workers (actor, critic, rollout, sharding) |
| `eval.py` | Multi-turn evaluation with feedback (Succ@k) |
| `eval_api.py` | API-based evaluation (OpenAI, Anthropic, etc.) |

## Environments (`env/`)

| Environment | Type | Category |
|---|---|---|
| `metamathqa/` | Math reasoning (normal, critique, no-feedback variants) | Math |
| `static/` | Static benchmarks (GSM8k, MATH, AIME24, HotpotQA, MMLU, etc.) | Multi-domain |
| `sokoban/` | Box-pushing puzzle (spatial planning) | Planning |
| `frozen_lake/` | Grid navigation with stochastic transitions | Planning |
| `sudoku/` | Constraint satisfaction | Reasoning |
| `countdown/` | Combinatorial number game | Math |
| `bandit/` | Multi-armed bandit (exploration/exploitation) | Decision |
| `webshop/` | Online shopping agent | Interactive |
| `alfworld/` | Embodied household tasks | Interactive |
| `lean/` | Theorem proving (requires Lean server) | Formal |
| `search/` | RAG-based information retrieval | Interactive |
| `spatial/` | Room navigation and spatial reasoning | Planning |
