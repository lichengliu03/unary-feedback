"""Utilities for filtering rollout trajectories before PPO updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from verl import DataProto


_FILTER_TYPE_ALIASES = {
    "largest": "largest",
    "smallest": "smallest",
    "std": "largest",
    "std_rev": "smallest",
}


@dataclass
class RolloutFilterConfig:
    value: float
    strategy: str = "top_k"
    filter_type: str = "std"
    include_zero: bool = True
    top_p_prob_mode: str = "linear"
    selection_eps: float = 0.01


def build_rollout_filter_config(rollout_cfg: Any) -> RolloutFilterConfig:
    return RolloutFilterConfig(
        value=float(getattr(rollout_cfg, "rollout_filter_value", getattr(rollout_cfg, "rollout_filter_ratio", 0.25))),
        strategy=getattr(rollout_cfg, "rollout_filter_strategy", "top_k"),
        filter_type=getattr(rollout_cfg, "rollout_filter_type", "std"),
        include_zero=getattr(rollout_cfg, "rollout_filter_include_zero", True),
        top_p_prob_mode=getattr(rollout_cfg, "rollout_filter_top_p_prob_mode", "linear"),
        selection_eps=float(getattr(rollout_cfg, "rollout_filter_selection_eps", 0.01)),
    )


def _normalize_filter_type(filter_type: str) -> str:
    try:
        return _FILTER_TYPE_ALIASES[filter_type]
    except KeyError as exc:
        valid = ", ".join(sorted(_FILTER_TYPE_ALIASES))
        raise ValueError(f"Invalid rollout filter type: {filter_type}. Expected one of {{{valid}}}.") from exc


def _selected_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    if selected.numel() == 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return values[selected].mean()


def select_rollout_groups(scores: torch.Tensor, num_groups: int, config: RolloutFilterConfig) -> torch.Tensor:
    scores = scores.float()
    if scores.numel() != num_groups:
        raise ValueError(f"Expected {num_groups} group scores, got {scores.numel()}.")

    indices = torch.arange(num_groups, device=scores.device)
    filter_type = _normalize_filter_type(config.filter_type)

    if not config.include_zero:
        non_zero_mask = torch.abs(scores) > 1e-10
        scores = scores[non_zero_mask]
        indices = indices[non_zero_mask]
        if indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=indices.device)

    if config.strategy == "top_p":
        if config.value >= 1.0:
            return indices

        if config.top_p_prob_mode == "softmax":
            logits = scores if filter_type == "largest" else -scores
            probs = torch.softmax(logits, dim=0)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            cutoff_index = torch.searchsorted(cumulative_probs, config.value).item()
            k = min(cutoff_index + 1, indices.numel())
            return indices[sorted_indices[:k]]

        if config.top_p_prob_mode != "linear":
            raise ValueError(
                f"Unknown top_p_prob_mode: {config.top_p_prob_mode}. Expected one of {{'linear', 'softmax'}}."
            )

        descending = filter_type == "largest"
        sorted_scores, sorted_indices = torch.sort(scores, descending=descending)
        threshold = config.value * scores.sum() - config.selection_eps
        cumulative_score = 0.0
        selected_count = 0

        for score in sorted_scores:
            if cumulative_score >= threshold:
                break
            if score.item() <= 0:
                break
            cumulative_score += score.item()
            selected_count += 1

        if cumulative_score >= threshold:
            return indices[sorted_indices[:selected_count]]
        return torch.empty(0, dtype=torch.long, device=indices.device)

    if config.strategy == "top_k":
        if config.value >= 1.0:
            return indices

        k = int(config.value * num_groups)
        k = min(k, indices.numel())
        k = max(k, 1)
        local_indices = scores.topk(k).indices if filter_type == "largest" else (-scores).topk(k).indices
        return indices[local_indices]

    if config.strategy == "top_k_abs":
        k = int(config.value)
        k = min(k, indices.numel())
        k = max(k, 1)
        local_indices = scores.topk(k).indices if filter_type == "largest" else (-scores).topk(k).indices
        return indices[local_indices]

    raise ValueError(f"Unknown rollout filter strategy: {config.strategy}")


def _groups_to_mask(top_groups: torch.Tensor, num_groups: int, group_size: int) -> torch.Tensor:
    mask = torch.zeros(num_groups, dtype=torch.bool, device=top_groups.device)
    if top_groups.numel() > 0:
        mask[top_groups] = True
    return mask.unsqueeze(1).expand(-1, group_size).reshape(-1).cpu()


def apply_rollout_filter(
    batch: "DataProto",
    num_groups: int,
    group_size: int,
    config: RolloutFilterConfig,
) -> Tuple["DataProto", Dict[str, torch.Tensor]]:
    rm_scores = batch.batch["original_rm_scores"].sum(dim=-1).view(num_groups, group_size)
    in_group_std = rm_scores.std(dim=-1)
    in_group_max = rm_scores.max(dim=-1).values
    in_group_mean = rm_scores.mean(dim=-1)

    top_groups = select_rollout_groups(in_group_std, num_groups=num_groups, config=config)
    mask = _groups_to_mask(top_groups, num_groups=num_groups, group_size=group_size)

    batch.batch = batch.batch[mask]
    if batch.non_tensor_batch is not None:
        np_mask = mask.numpy()
        for key, value in batch.non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                batch.non_tensor_batch[key] = value[np_mask]
            else:
                batch.non_tensor_batch[key] = [v for v, keep in zip(value, np_mask) if keep]

    metrics = {
        "rollout/in_group_std": in_group_std.mean(),
        "rollout/in_group_max": in_group_max.mean(),
        "rollout/in_group_mean": in_group_mean.mean(),
        "rollout/chosen_in_group_std": _selected_mean(in_group_std, top_groups),
        "rollout/chosen_in_group_max": _selected_mean(in_group_max, top_groups),
        "rollout/chosen_in_group_mean": _selected_mean(in_group_mean, top_groups),
        "rollout/filter_kept_ratio": torch.tensor(
            top_groups.numel() / max(num_groups, 1),
            device=in_group_std.device,
            dtype=in_group_std.dtype,
        ),
    }
    return batch, metrics
