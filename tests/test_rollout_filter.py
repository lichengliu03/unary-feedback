from types import SimpleNamespace

import numpy as np
import torch

from ufb.trainer.rollout_filter import (
    RolloutFilterConfig,
    apply_rollout_filter,
    build_rollout_filter_config,
    select_rollout_groups,
)


class FakeTensorBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.data[item]
        return FakeTensorBatch({key: value[item] for key, value in self.data.items()})


def test_top_k_matches_existing_largest_std_behavior():
    config = RolloutFilterConfig(value=0.5, strategy="top_k", filter_type="std")
    selected = select_rollout_groups(torch.tensor([1.0, 4.0, 3.0, 2.0]), num_groups=4, config=config)
    assert selected.tolist() == [1, 2]


def test_top_k_supports_legacy_smallest_alias():
    config = RolloutFilterConfig(value=0.5, strategy="top_k", filter_type="std_rev")
    selected = select_rollout_groups(torch.tensor([1.0, 4.0, 3.0, 2.0]), num_groups=4, config=config)
    assert selected.tolist() == [0, 3]


def test_linear_top_p_keeps_min_prefix_covering_target_mass():
    config = RolloutFilterConfig(
        value=0.7,
        strategy="top_p",
        filter_type="std",
        top_p_prob_mode="linear",
    )
    selected = select_rollout_groups(torch.tensor([4.0, 3.0, 2.0, 1.0]), num_groups=4, config=config)
    assert selected.tolist() == [0, 1]


def test_linear_top_p_returns_empty_when_mass_is_non_positive():
    config = RolloutFilterConfig(
        value=0.9,
        strategy="top_p",
        filter_type="std",
        top_p_prob_mode="linear",
        include_zero=False,
    )
    selected = select_rollout_groups(torch.tensor([0.0, 0.0, 0.0]), num_groups=3, config=config)
    assert selected.numel() == 0


def test_apply_rollout_filter_updates_tensor_and_non_tensor_batches():
    batch = SimpleNamespace(
        batch=FakeTensorBatch(
            {
                "original_rm_scores": torch.tensor(
                    [[1.0], [5.0], [2.0], [2.0], [3.0], [4.0], [0.0], [0.0]]
                ),
                "other": torch.arange(8),
            }
        ),
        non_tensor_batch={"uid": np.array(list("abcdefgh"), dtype=object)},
    )
    config = RolloutFilterConfig(
        value=0.75,
        strategy="top_p",
        filter_type="largest",
        top_p_prob_mode="linear",
        include_zero=False,
    )

    filtered_batch, metrics = apply_rollout_filter(batch, num_groups=4, group_size=2, config=config)

    assert filtered_batch.batch["other"].tolist() == [0, 1]
    assert filtered_batch.non_tensor_batch["uid"].tolist() == ["a", "b"]
    assert metrics["rollout/filter_kept_ratio"].item() == 0.25
    assert metrics["rollout/chosen_in_group_std"].item() > 2.8


def test_build_rollout_filter_config_supports_new_and_legacy_fields():
    rollout_cfg = SimpleNamespace(
        rollout_filter_ratio=0.25,
        rollout_filter_value=0.9,
        rollout_filter_strategy="top_p",
        rollout_filter_type="std",
        rollout_filter_top_p_prob_mode="linear",
        rollout_filter_include_zero=False,
        rollout_filter_selection_eps=0.02,
    )
    config = build_rollout_filter_config(rollout_cfg)

    assert config.value == 0.9
    assert config.strategy == "top_p"
    assert config.include_zero is False
    assert config.selection_eps == 0.02
