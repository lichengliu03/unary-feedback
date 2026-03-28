#!/usr/bin/env python3
"""Export UFO FSDP checkpoints to Hugging Face format and optionally upload them."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


WEIGHT_FILES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


def _print(message: str) -> None:
    print(f"[HF-EXPORT] {message}")


def _error(message: str) -> "NoReturn":
    raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a sharded FSDP checkpoint into a Hugging Face model folder and "
            "optionally upload it to the Hub."
        )
    )
    parser.add_argument("legacy_input_dir", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_output_dir", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_base_model", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_world_size", nargs="?", type=int, help=argparse.SUPPRESS)

    parser.add_argument(
        "--input-dir",
        help=(
            "Experiment root, global_step directory, or actor checkpoint directory. "
            "Legacy positional usage is also supported."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for the exported Hugging Face model. If omitted and the checkpoint "
            "already contains full HF weights, that folder is reused."
        ),
    )
    parser.add_argument(
        "--base-model",
        help="Base model name/path used to rebuild a full HF checkpoint from FSDP shards.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of FSDP model shards. If omitted, it is inferred from checkpoint files.",
    )
    parser.add_argument(
        "--step",
        default="latest",
        help="Checkpoint step to export when --input-dir points at an experiment root.",
    )
    parser.add_argument("--repo-id", help="Optional Hugging Face repo id, for example user/model-name.")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private if needed.")
    parser.add_argument(
        "--upload-mode",
        choices=("auto", "api", "cli"),
        default="auto",
        help="Upload implementation. 'auto' prefers the hf CLI when available.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Custom commit message for the Hub upload.",
    )
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading the base model/tokenizer.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Torch dtype used when loading the base model for conversion.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="10GB",
        help="Shard size used when saving the merged HF model.",
    )
    parser.add_argument(
        "--no-model-card",
        action="store_true",
        help="Skip generating a lightweight README.md in the export directory.",
    )
    parser.add_argument(
        "--force-reconvert",
        action="store_true",
        help="Ignore existing HF weights under actor/huggingface and rebuild from FSDP shards.",
    )
    args = parser.parse_args()

    # Backward compatible mode:
    #   python convert_fsdp_to_hf.py <checkpoint_dir> <output_dir> <base_model> [world_size]
    if args.legacy_input_dir and not args.input_dir:
        args.input_dir = args.legacy_input_dir
    if args.legacy_output_dir and not args.output_dir:
        args.output_dir = args.legacy_output_dir
    if args.legacy_base_model and not args.base_model:
        args.base_model = args.legacy_base_model
    if args.legacy_world_size is not None and args.world_size is None:
        args.world_size = args.legacy_world_size

    if not args.input_dir:
        parser.error("missing input directory")

    return args


def has_hf_weights(hf_dir: Path) -> bool:
    if not hf_dir.is_dir():
        return False
    return any((hf_dir / name).exists() for name in WEIGHT_FILES)


def list_available_steps(experiment_dir: Path) -> list[int]:
    steps: list[int] = []
    for child in experiment_dir.glob("global_step_*"):
        if not child.is_dir():
            continue
        suffix = child.name.removeprefix("global_step_")
        if suffix.isdigit() and (child / "actor").is_dir():
            steps.append(int(suffix))
    return sorted(steps)


def resolve_actor_checkpoint(input_dir: Path, step: str) -> tuple[Path, int | None]:
    input_dir = input_dir.resolve()

    if (input_dir / "actor").is_dir() and input_dir.name.startswith("global_step_"):
        suffix = input_dir.name.removeprefix("global_step_")
        return (input_dir / "actor", int(suffix) if suffix.isdigit() else None)

    if any(input_dir.glob("model_world_size_*_rank_*.pt")) or (input_dir / "huggingface").is_dir():
        return input_dir, None

    steps = list_available_steps(input_dir)
    if not steps:
        _error(
            f"Could not find an actor checkpoint under {input_dir}. "
            "Expected an experiment dir, a global_step dir, or an actor dir."
        )

    if step == "latest":
        latest_file = input_dir / "latest_checkpointed_iteration.txt"
        if latest_file.exists():
            latest_text = latest_file.read_text(encoding="utf-8").strip()
            if latest_text.isdigit() and (input_dir / f"global_step_{latest_text}" / "actor").is_dir():
                selected_step = int(latest_text)
            else:
                selected_step = steps[-1]
        else:
            selected_step = steps[-1]
    else:
        if not str(step).isdigit():
            _error(f"--step must be an integer or 'latest', got: {step}")
        selected_step = int(step)
        if selected_step not in steps:
            _error(
                f"Checkpoint step {selected_step} does not exist under {input_dir}. "
                f"Available steps: {', '.join(str(s) for s in steps)}"
            )

    return input_dir / f"global_step_{selected_step}" / "actor", selected_step


def infer_world_size(checkpoint_dir: Path, requested_world_size: int | None) -> int:
    if requested_world_size is not None:
        return requested_world_size

    candidates = []
    for path in checkpoint_dir.glob("model_world_size_*_rank_*.pt"):
        parts = path.stem.split("_")
        try:
            world_size = int(parts[3])
        except (IndexError, ValueError):
            continue
        candidates.append(world_size)

    if not candidates:
        _error(
            "Could not infer world size from checkpoint files. "
            "Please pass --world-size explicitly."
        )

    unique = sorted(set(candidates))
    if len(unique) != 1:
        _error(f"Found inconsistent world sizes in checkpoint files: {unique}")
    return unique[0]


def pick_output_dir(actor_dir: Path, output_dir: str | None, force_reconvert: bool) -> Path:
    hf_dir = actor_dir / "huggingface"
    if output_dir:
        return Path(output_dir).resolve()
    if has_hf_weights(hf_dir) and not force_reconvert:
        return hf_dir.resolve()
    return (actor_dir / "huggingface_export").resolve()


def _to_cpu_tensor(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    _error(f"Unsupported plain tensor value type: {type(value)!r}")


def _extract_dtensor_info(value):
    placements = list(getattr(value, "placements", getattr(getattr(value, "_spec", None), "placements", [])))
    shard_dims = [getattr(placement, "dim", None) for placement in placements if placement.__class__.__name__ == "Shard"]

    if hasattr(value, "to_local"):
        local_tensor = value.to_local().detach().cpu()
    elif hasattr(value, "_local_tensor"):
        local_tensor = value._local_tensor.detach().cpu()
    else:
        _error(f"DTensor-like object {type(value)!r} is missing a local tensor accessor")

    if not shard_dims:
        return {
            "kind": "replicated",
            "tensor": local_tensor,
        }

    if len(shard_dims) != 1:
        _error(f"Only single-dimension DTensor sharding is supported, got placements={placements}")

    return {
        "kind": "dtensor_shard",
        "tensor": local_tensor,
        "dim": shard_dims[0],
        "global_shape": tuple(value.shape),
    }


def _merge_sharded_tensor_values(values):
    import torch

    merged = None
    expected_shape = None

    for value in values:
        metadata = value.metadata()
        global_shape = tuple(metadata.size)
        local_shards = value.local_shards()
        if not local_shards:
            continue

        if merged is None:
            merged = torch.empty(global_shape, dtype=local_shards[0].tensor.dtype)
            expected_shape = global_shape

        for local_shard in local_shards:
            offsets = tuple(local_shard.metadata.shard_offsets)
            sizes = tuple(local_shard.metadata.shard_sizes)
            slices = tuple(slice(offset, offset + size) for offset, size in zip(offsets, sizes))
            merged[slices] = local_shard.tensor.detach().cpu()

    if merged is None or expected_shape is None:
        _error("Found an empty ShardedTensor while rebuilding the checkpoint")

    return merged


def _extract_value_info(value):
    if hasattr(value, "local_shards") and hasattr(value, "metadata"):
        return {"kind": "sharded_tensor", "value": value}

    if hasattr(value, "to_local") or hasattr(value, "_local_tensor"):
        return _extract_dtensor_info(value)

    return {"kind": "tensor", "tensor": _to_cpu_tensor(value)}


def _merge_values_for_key(key: str, values: Iterable[object]):
    import torch

    infos = [_extract_value_info(value) for value in values]
    kinds = {info["kind"] for info in infos}

    if kinds == {"sharded_tensor"}:
        return _merge_sharded_tensor_values([info["value"] for info in infos])

    if kinds <= {"tensor", "replicated"}:
        tensors = [info["tensor"] for info in infos]
        if len(tensors) == 1:
            return tensors[0]

        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) == 1:
            return tensors[0]

        reference = tensors[0]
        candidate_dims = []
        for dim in range(reference.ndim):
            if any(tensor.ndim != reference.ndim for tensor in tensors):
                break
            if all(
                tensor.shape[other_dim] == reference.shape[other_dim]
                for tensor in tensors
                for other_dim in range(reference.ndim)
                if other_dim != dim
            ):
                candidate_dims.append(dim)

        if len(candidate_dims) == 1:
            return torch.cat(tensors, dim=candidate_dims[0])

        return infos[0]["tensor"]

    if kinds == {"dtensor_shard"}:
        shard_dims = {info["dim"] for info in infos}
        if len(shard_dims) != 1:
            _error(f"Key {key} was sharded across multiple dimensions: {sorted(shard_dims)}")

        shard_dim = next(iter(shard_dims))
        merged = torch.cat([info["tensor"] for info in infos], dim=shard_dim)
        global_shape = infos[0]["global_shape"]
        if tuple(merged.shape) != tuple(global_shape):
            slices = [slice(None)] * merged.ndim
            slices[shard_dim] = slice(0, global_shape[shard_dim])
            merged = merged[tuple(slices)]
        return merged

    _error(f"Unsupported state dict layout for key {key}: {sorted(kinds)}")


def copy_supporting_hf_files(source_hf_dir: Path, output_dir: Path) -> None:
    if not source_hf_dir.is_dir():
        return

    skip_names = WEIGHT_FILES | {
        ".gitattributes",
    }
    for child in source_hf_dir.iterdir():
        if child.name in skip_names:
            continue
        target = output_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def resolve_torch_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def convert_fsdp_to_hf(
    checkpoint_dir: Path,
    output_dir: Path,
    base_model: str,
    world_size: int,
    *,
    dtype: str,
    trust_remote_code: bool,
    max_shard_size: str,
) -> Path:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        _error(
            "convert_fsdp_to_hf.py needs torch and transformers in the active environment. "
            f"Original import error: {exc}"
        )

    shard_files = [
        checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
    ]
    missing_files = [path for path in shard_files if not path.exists()]
    if missing_files:
        _error(f"Missing checkpoint shard files: {missing_files}")

    model_dtype = resolve_torch_dtype(dtype)
    _print(f"Loading base model from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=model_dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )

    _print(f"Loading {world_size} FSDP shards from {checkpoint_dir}")
    shard_state_dicts = []
    for rank, shard_file in enumerate(shard_files):
        _print(f"Loading shard {rank + 1}/{world_size}: {shard_file.name}")
        shard_state_dict = torch.load(shard_file, map_location="cpu", weights_only=False)
        shard_state_dicts.append(shard_state_dict)

    all_keys: set[str] = set()
    for shard_state_dict in shard_state_dicts:
        all_keys.update(shard_state_dict.keys())

    merged_state_dict = {}
    for key in sorted(all_keys):
        values = [shard_state_dict[key] for shard_state_dict in shard_state_dicts if key in shard_state_dict]
        merged_state_dict[key] = _merge_values_for_key(key, values)

    cleaned_state_dict = {
        key.replace("_fsdp_wrapped_module.", "", 1): value
        for key, value in merged_state_dict.items()
    }

    _print("Loading merged weights into the base model")
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing_keys:
        _print(f"Missing keys while loading merged weights: {len(missing_keys)}")
    if unexpected_keys:
        _print(f"Unexpected keys while loading merged weights: {len(unexpected_keys)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _print(f"Saving merged Hugging Face model to {output_dir}")
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    source_hf_dir = checkpoint_dir / "huggingface"
    copy_supporting_hf_files(source_hf_dir, output_dir)

    if not (output_dir / "tokenizer_config.json").exists():
        _print("Tokenizer files were not found in the checkpoint, pulling them from the base model")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        tokenizer.save_pretrained(str(output_dir))

    return output_dir


def ensure_export_dir(
    actor_dir: Path,
    output_dir: Path,
    *,
    base_model: str | None,
    world_size: int,
    dtype: str,
    trust_remote_code: bool,
    max_shard_size: str,
    force_reconvert: bool,
) -> Path:
    source_hf_dir = actor_dir / "huggingface"

    if has_hf_weights(output_dir) and not force_reconvert:
        _print(f"Reusing existing HF export in {output_dir}")
        return output_dir

    if has_hf_weights(source_hf_dir) and not force_reconvert:
        if source_hf_dir.resolve() == output_dir.resolve():
            _print(f"Reusing existing HF export in {output_dir}")
            return output_dir

        _print(f"Copying existing HF weights from {source_hf_dir} to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_hf_dir, output_dir, dirs_exist_ok=True)
        return output_dir

    if not base_model:
        _error(
            "This checkpoint only contains FSDP shards, so --base-model is required "
            "to rebuild a Hugging Face checkpoint."
        )

    return convert_fsdp_to_hf(
        checkpoint_dir=actor_dir,
        output_dir=output_dir,
        base_model=base_model,
        world_size=world_size,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        max_shard_size=max_shard_size,
    )


def write_model_card(
    export_dir: Path,
    *,
    repo_id: str | None,
    base_model: str | None,
    actor_dir: Path,
    step: int | None,
) -> None:
    readme_path = export_dir / "README.md"
    if readme_path.exists():
        return

    title = repo_id or export_dir.name
    source_step = f"global_step_{step}" if step is not None else actor_dir.parent.name
    card = f"""---
library_name: transformers
base_model: {base_model or "unknown"}
tags:
- ufb
- reinforcement-learning
- fsdp
---

# {title}

This model was exported from a UFO training checkpoint.

- Base model: `{base_model or "unknown"}`
- Source checkpoint: `{actor_dir}`
- Exported step: `{source_step}`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{repo_id or export_dir}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
```
"""
    readme_path.write_text(card, encoding="utf-8")


def upload_with_cli(
    export_dir: Path,
    repo_id: str,
    *,
    private: bool,
    commit_message: str,
    token: str | None,
) -> None:
    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token

    create_cmd = ["hf", "repos", "create", repo_id, "--type", "model", "--exist-ok"]
    if private:
        create_cmd.append("--private")
    subprocess.run(create_cmd, check=True, env=env)

    upload_cmd = [
        "hf",
        "upload-large-folder",
        repo_id,
        str(export_dir),
        "--type",
        "model",
    ]
    subprocess.run(upload_cmd, check=True, env=env)

    _print(f"Uploaded with hf CLI: https://huggingface.co/{repo_id}")


def upload_with_api(
    export_dir: Path,
    repo_id: str,
    *,
    private: bool,
    commit_message: str,
    token: str | None,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        _error(
            "Upload requested, but huggingface_hub is not available and hf CLI is not installed. "
            f"Original import error: {exc}"
        )

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    _print(f"Uploaded with huggingface_hub: https://huggingface.co/{repo_id}")


def upload_export(
    export_dir: Path,
    repo_id: str,
    *,
    private: bool,
    upload_mode: str,
    commit_message: str,
    token: str | None,
) -> None:
    has_hf_cli = shutil.which("hf") is not None

    if upload_mode in {"auto", "cli"} and has_hf_cli:
        upload_with_cli(
            export_dir,
            repo_id,
            private=private,
            commit_message=commit_message,
            token=token,
        )
        return

    if upload_mode == "cli" and not has_hf_cli:
        _error("Upload mode 'cli' was requested, but the `hf` command is not installed")

    upload_with_api(
        export_dir,
        repo_id,
        private=private,
        commit_message=commit_message,
        token=token,
    )


def main() -> int:
    args = parse_args()

    actor_dir, step = resolve_actor_checkpoint(Path(args.input_dir), args.step)
    output_dir = pick_output_dir(actor_dir, args.output_dir, args.force_reconvert)
    needs_merge = args.force_reconvert or not has_hf_weights(actor_dir / "huggingface")
    world_size = (
        infer_world_size(actor_dir, args.world_size)
        if needs_merge or args.world_size is not None
        else None
    )

    _print(f"Actor checkpoint: {actor_dir}")
    if step is not None:
        _print(f"Selected step: {step}")
    _print(f"World size: {world_size if world_size is not None else 'n/a'}")
    _print(f"Export directory: {output_dir}")

    export_dir = ensure_export_dir(
        actor_dir=actor_dir,
        output_dir=output_dir,
        base_model=args.base_model,
        world_size=world_size or 0,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_shard_size=args.max_shard_size,
        force_reconvert=args.force_reconvert,
    )

    export_metadata = {
        "actor_dir": str(actor_dir),
        "step": step,
        "base_model": args.base_model,
        "world_size": world_size,
    }
    (export_dir / "ufb_export_metadata.json").write_text(
        json.dumps(export_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not args.no_model_card:
        write_model_card(
            export_dir,
            repo_id=args.repo_id,
            base_model=args.base_model,
            actor_dir=actor_dir,
            step=step,
        )

    if args.repo_id:
        commit_message = args.commit_message or (
            f"Upload UFB checkpoint from step {step}" if step is not None else "Upload UFB checkpoint"
        )
        upload_export(
            export_dir=export_dir,
            repo_id=args.repo_id,
            private=args.private,
            upload_mode=args.upload_mode,
            commit_message=commit_message,
            token=os.environ.get("HF_TOKEN"),
        )

    _print("Done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[HF-EXPORT][ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
