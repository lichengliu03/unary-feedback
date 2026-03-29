#!/usr/bin/env python3
"""Download the released Exp1 models into loaded_checkpoints/exp1."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from convert_fsdp_to_hf import convert_fsdp_to_hf


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP1_ROOT = REPO_ROOT / "loaded_checkpoints" / "exp1"

MODEL_SPECS = {
    "hotpotqa": {
        "kind": "checkpoint",
        "repo_id": "zihanwang314/unary-feedback-checkpoints",
        "repo_type": "model",
        "local_dir": EXP1_ROOT / "hotpotqa_raw",
        "allow_patterns": ["ufb_exp/hotpotqa_qwen25_3b_one_bit_5attempts_1turn/**"],
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "output_dir": EXP1_ROOT / "hotpotqa",
    },
    "webshop": {
        "kind": "checkpoint",
        "repo_id": "zihanwang314/unary-feedback",
        "repo_type": "model",
        "local_dir": EXP1_ROOT / "webshop_raw",
        "allow_patterns": ["ufb_exp/webshop_qwen25_3b_one_bit_2attempts_7turns/**"],
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "output_dir": EXP1_ROOT / "webshop",
    },
    "metamathqa": {
        "kind": "direct",
        "repo_id": "ZihanWang314/exp1_MetamathQA_global_step_200",
        "repo_type": "model",
        "local_dir": EXP1_ROOT / "metamathqa",
    },
    "countdown": {
        "kind": "direct",
        "repo_id": "ZihanWang314/exp1_Countdown_global_step_200",
        "repo_type": "model",
        "local_dir": EXP1_ROOT / "countdown",
    },
    "simplesokoban": {
        "kind": "direct",
        "repo_id": "ZihanWang314/exp1_SimpleSokoban_global_step_200",
        "repo_type": "model",
        "local_dir": EXP1_ROOT / "simplesokoban",
    },
    "frozenlake": {
        "kind": "direct",
        "repo_id": "ZihanWang314/exp1_Frozenlake_global_step_200",
        "repo_type": "model",
        "local_dir": EXP1_ROOT / "frozenlake",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_SPECS),
        help="Optional explicit model keys to download. By default, download the full Exp1 set.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="For checkpoint-backed models, download raw checkpoints without converting the latest step.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force fresh downloads even if files already exist locally.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately if any requested model is missing or fails.",
    )
    return parser.parse_args()


def select_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        return args.models
    return list(MODEL_SPECS)


def ensure_checkpoint_pattern_exists(spec: dict) -> None:
    api = HfApi()
    prefix = spec["allow_patterns"][0].split("/**", 1)[0]
    files = api.list_repo_files(spec["repo_id"], repo_type=spec["repo_type"])
    if not any(path.startswith(prefix + "/") for path in files):
        raise FileNotFoundError(
            f"No remote files matched prefix '{prefix}' in {spec['repo_id']}. "
            "Double-check the experiment name in allow_patterns."
        )


def download_spec(name: str, spec: dict, force: bool) -> Path:
    local_dir = Path(spec["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": spec["repo_id"],
        "repo_type": spec["repo_type"],
        "local_dir": str(local_dir),
        "force_download": force,
    }
    if "allow_patterns" in spec:
        ensure_checkpoint_pattern_exists(spec)
        kwargs["allow_patterns"] = spec["allow_patterns"]

    print(f"[DOWNLOAD] {name}: {spec['repo_id']} -> {local_dir}")
    result = snapshot_download(**kwargs)
    print(f"[DONE] {name}: downloaded to {result}")
    return local_dir


def find_latest_actor_dir(local_dir: Path) -> Path:
    candidates = []
    for actor_dir in local_dir.glob("ufb_exp/*/global_step_*/actor"):
        match = re.search(r"global_step_(\d+)", str(actor_dir))
        if match:
            candidates.append((int(match.group(1)), actor_dir))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint actor directories found under {local_dir}")

    candidates.sort(key=lambda item: item[0])
    latest_step, latest_actor_dir = candidates[-1]
    print(f"[CHECKPOINT] Using latest step {latest_step}: {latest_actor_dir}")
    return latest_actor_dir


def convert_latest_checkpoint(name: str, spec: dict, raw_dir: Path) -> Path:
    actor_dir = find_latest_actor_dir(raw_dir)
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONVERT] {name}: {actor_dir} -> {output_dir}")
    success = convert_fsdp_to_hf(
        checkpoint_dir=str(actor_dir),
        output_dir=str(output_dir),
        base_model=spec["base_model"],
        world_size=2,
    )
    if not success:
        raise RuntimeError(f"Conversion failed for {name}")

    print(f"[DONE] {name}: converted HF model at {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    selected = select_models(args)

    for name in selected:
        spec = MODEL_SPECS[name]
        try:
            raw_or_model_dir = download_spec(name=name, spec=spec, force=args.force)

            if spec["kind"] == "checkpoint" and not args.skip_convert:
                convert_latest_checkpoint(name=name, spec=spec, raw_dir=raw_or_model_dir)
        except Exception as exc:  # pragma: no cover - CLI flow
            if args.strict:
                raise
            print(f"[SKIP] {name}: {exc}")


if __name__ == "__main__":
    main()
