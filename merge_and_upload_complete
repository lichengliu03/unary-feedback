#!/usr/bin/env python3
"""
upload_checkpoint_to_hf.py
==========================
Utility script to upload local checkpoints (single `model.safetensors` or a sharded
directory) together with tokenizer / config files to the Hugging Face Hub.

Example (bash):
    python scripts/upload_checkpoint_to_hf.py \
        --model-path merged_actor_model_correct/model.safetensors \
        --config-dir checkpoints/ufo/test/global_step_200/actor/huggingface \
        --repo-name LichengLiu03/qwen2.5-3b-UFO-1turn \
        --token $HF_TOKEN

Arguments:
    --model-path   Required. Path to `model.safetensors` OR a directory that contains
                   shards (`model-xxxx-of-xxxx.safetensors` + `index.json`).
    --config-dir   Optional. Directory that contains `tokenizer.json`, `config.json`,
                   etc. If omitted and `--model-path` is a directory, the same
                   directory will be used.
    --repo-name    Target HF repository (username/model-name).
    --token        Hugging Face access token.
    --private      Create a private repo (flag).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, upload_file, upload_folder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload checkpoint & configs to HF Hub")
    p.add_argument("--model-path", required=True, help="Path to model.safetensors or shard directory")
    p.add_argument("--config-dir", help="Directory containing tokenizer/config files")
    p.add_argument("--repo-name", required=True, help="HF repo (username/model)")
    p.add_argument("--token", required=True, help="HF token (env var or string)")
    p.add_argument("--private", action="store_true", help="Create private repo")
    return p.parse_args()


def gather_config_files(config_dir: Path) -> List[Path]:
    patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ]
    files: List[Path] = []
    for name in patterns:
        f = config_dir / name
        if f.exists():
            files.append(f)
    return files


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        sys.exit(f"❌ model-path does not exist: {model_path}")

    # infer config directory
    config_dir = Path(args.config_dir).expanduser().resolve() if args.config_dir else None
    if config_dir is None:
        if model_path.is_dir():
            # fall back to the same directory as model_path
            config_dir = model_path
        else:
            sys.exit("❌ --config-dir is missing and --model-path is not a directory, tokenizer/config cannot be located")
    else:
        if not config_dir.exists():
            sys.exit(f"❌ config-dir does not exist: {config_dir}")

    print(f"Model path : {model_path}")
    print(f"Config dir : {config_dir}")
    print(f"Repo name  : {args.repo_name}")

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_name,
        repo_type="model",
        exist_ok=True,
        private=args.private,
    )

    # upload model weights
    if model_path.is_dir():
        print("Uploading weight directory (may take a while)…")
        upload_folder(
            folder_path=str(model_path),
            repo_id=args.repo_name,
            repo_type="model",
            token=args.token,
            commit_message="Upload model weights directory",
        )
    else:
        print("Uploading single model.safetensors file…")
        upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="model.safetensors",
            repo_id=args.repo_name,
            repo_type="model",
            token=args.token,
            commit_message="Upload single model.safetensors",
        )

    # upload config / tokenizer
    cfg_files = gather_config_files(config_dir)
    if cfg_files:
        print(f"Uploading {len(cfg_files)} tokenizer/config files…")
        for f in cfg_files:
            upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=args.repo_name,
                repo_type="model",
                token=args.token,
                commit_message=f"Add {f.name}",
            )
    else:
        print("⚠️ No tokenizer/config files found, skipped.")

    print("✅ Upload finished!")


if __name__ == "__main__":
    main() 
