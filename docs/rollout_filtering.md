# Rollout Filtering Config Guide

This document lists the three rollout-filtering presets currently used in `ufb` and shows exactly how to write the config for each one.

The rollout filter is applied on the per-group reward standard deviation. In the current codebase, the common setting is:

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_type: std
```

This means we keep groups with larger in-group reward std. The three presets below differ only in how many groups are kept.


## Recommended Presets

If you only want the three standard presets, use exactly these:

### No filtering

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_k
    rollout_filter_value: 1.0
    rollout_filter_type: std
    rollout_filter_include_zero: True
```

### Top-k filtering

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_k
    rollout_filter_value: 0.25
    rollout_filter_type: std
    rollout_filter_include_zero: True
```

### Top-p linear filtering

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_p
    rollout_filter_value: 0.9
    rollout_filter_type: std
    rollout_filter_top_p_prob_mode: linear
    rollout_filter_include_zero: False
    rollout_filter_selection_eps: 0.01
```

## Sweep Script

If you want to run multiple filter presets sequentially without modifying `scripts/exp1_train.sh`, use:

```bash
bash scripts/filtering_sweep.sh
```

You can also choose a subset:

```bash
FILTERS=nofilter bash scripts/filtering_sweep.sh
FILTERS=top_k,top_p bash scripts/filtering_sweep.sh
```

The supported values are:

- `nofilter`
- `top_k`
- `top_p`
- `all`


## Config Keys

These are the main rollout-filter keys:

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_k | top_p
    rollout_filter_value: 0.25
    rollout_filter_type: std
    rollout_filter_top_p_prob_mode: linear
    rollout_filter_include_zero: True
    rollout_filter_selection_eps: 0.01
```

Notes:

- `rollout_filter_type: std` means keep groups with larger in-group reward std.
- `rollout_filter_strategy: top_k` means select a fraction of groups.
- `rollout_filter_strategy: top_p` means select the smallest prefix of sorted groups whose mass reaches `rollout_filter_value`.
- `rollout_filter_top_p_prob_mode: linear` is the linear top-p rule.
- `rollout_filter_include_zero: False` removes zero-std groups before top-p or top-k selection.

## 1. No Filtering

Use this when you want to keep all groups.

Required setting:

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_k
    rollout_filter_value: 1.0
    rollout_filter_type: std
    rollout_filter_include_zero: True
```

Behavior:

- All groups are kept.
- `include_zero=True` means zero-std groups are also kept.

CLI override example:

```bash
python train.py \
  --config-name=envs/sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=top_k \
  actor_rollout_ref.rollout.rollout_filter_value=1.0 \
  actor_rollout_ref.rollout.rollout_filter_type=std \
  actor_rollout_ref.rollout.rollout_filter_include_zero=True
```

## 2. Top-k Filtering

Use this when you want to keep the top `k%` groups by in-group reward std.

Required setting:

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_k
    rollout_filter_value: 0.25
    rollout_filter_type: std
    rollout_filter_include_zero: True
```

Behavior:

- `rollout_filter_value` is a percentage written as a fraction.
- `0.25` means keep the top 25% groups.
- `0.50` means keep the top 50% groups.
- `include_zero=True` means zero-std groups still participate in ranking.

CLI override example:

```bash
python train.py \
  --config-name=envs/sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=top_k \
  actor_rollout_ref.rollout.rollout_filter_value=0.25 \
  actor_rollout_ref.rollout.rollout_filter_type=std \
  actor_rollout_ref.rollout.rollout_filter_include_zero=True
```

## 3. Top-p Linear Filtering

Use this when you want linear top-p filtering and want to drop zero-std groups before selection.

Required setting:

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_p
    rollout_filter_value: 0.9
    rollout_filter_type: std
    rollout_filter_top_p_prob_mode: linear
    rollout_filter_include_zero: False
    rollout_filter_selection_eps: 0.01
```

Behavior:

- Groups are sorted by in-group reward std from large to small.
- We keep the smallest prefix whose cumulative linear mass reaches `rollout_filter_value`.
- `rollout_filter_include_zero=False` removes zero-std groups before selection.
- `rollout_filter_selection_eps=0.01` is the epsilon used by the linear top-p threshold check.

CLI override example:

```bash
python train.py \
  --config-name=envs/sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=top_p \
  actor_rollout_ref.rollout.rollout_filter_value=0.9 \
  actor_rollout_ref.rollout.rollout_filter_type=std \
  actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=linear \
  actor_rollout_ref.rollout.rollout_filter_include_zero=False \
  actor_rollout_ref.rollout.rollout_filter_selection_eps=0.01
```

## Sweep Script

To run these presets sequentially without changing the original `scripts/exp1_train.sh`, use:

```bash
bash scripts/filtering_sweep.sh
```

You can also select a subset:

```bash
FILTERS=nofilter bash scripts/filtering_sweep.sh
FILTERS=top_k,top_p bash scripts/filtering_sweep.sh
```

Supported values:

- `nofilter`
- `top_k`
- `top_p`
- `all`
