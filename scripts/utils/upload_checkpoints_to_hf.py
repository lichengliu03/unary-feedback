#!/usr/bin/env python3
"""Upload a local checkpoint directory to the Hugging Face Hub.

Example:
    export HF_UPLOAD_TOKEN=hf_xxx
    hf auth login
    export EXPERIMENT=my_run
    python public/upload_checkpoints_to_hf.py \
        --checkpoint-dir /workspace/ufb_exp/${EXPERIMENT} \
        --path-prefix ufb_exp/${EXPERIMENT}

Resolution order:
1. Token: ``--token`` -> ``HF_UPLOAD_TOKEN`` -> ``HF_TOKEN`` -> cached hf auth login token
2. Repo id: ``--repo-id`` -> ``HF_UPLOAD_REPO`` -> ``HF_REPO_ID`` ->
   ``<logged_in_username>/unary-feedback``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


DEFAULT_REPO_NAME = "unary-feedback"
DEFAULT_IGNORE_PATTERNS = [
    ".git/*",
    "**/.git/*",
    "**/__pycache__/*",
    "**/*.pyc",
    "**/*.pyo",
    "**/.DS_Store",
    "**/Thumbs.db",
    "**/wandb/*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a checkpoint directory to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Local experiment/checkpoint directory to upload.",
    )
    parser.add_argument(
        "--path-prefix",
        required=True,
        help="Destination prefix inside the repo, e.g. ufb_exp/my_experiment.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Target Hugging Face repo id. Optional if HF_UPLOAD_REPO/HF_REPO_ID is set.",
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default=os.getenv("HF_UPLOAD_REPO_TYPE", "model"),
        help="Target repo type. Defaults to model.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. Optional if HF_UPLOAD_TOKEN/HF_TOKEN or hf auth login is available.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch or revision to upload to.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional custom commit message.",
    )
    parser.add_argument(
        "--default-repo-name",
        default=DEFAULT_REPO_NAME,
        help="Repo name used when repo id is auto-derived from the logged-in username.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Additional glob pattern to ignore. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings and file stats without uploading.",
    )
    return parser.parse_args()


def resolve_token(cli_token: str | None) -> str | None:
    return cli_token or os.getenv("HF_UPLOAD_TOKEN") or os.getenv("HF_TOKEN")


def normalize_path_prefix(path_prefix: str) -> str:
    normalized = path_prefix.replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("--path-prefix must not be empty")
    return normalized


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def summarize_directory(root: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for path in iter_files(root):
        file_count += 1
        total_bytes += path.stat().st_size
    return file_count, total_bytes


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def resolve_repo_id(
    api: HfApi,
    repo_id: str | None,
    repo_type: str,
    token: str | None,
    default_repo_name: str,
) -> str:
    if repo_id:
        return repo_id

    env_repo_id = os.getenv("HF_UPLOAD_REPO") or os.getenv("HF_REPO_ID")
    if env_repo_id:
        return env_repo_id

    whoami = api.whoami(token=token)
    username = whoami["name"]
    return f"{username}/{default_repo_name}"


def main() -> int:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        print(f"[ERROR] Checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        return 1
    if not checkpoint_dir.is_dir():
        print(f"[ERROR] Checkpoint path is not a directory: {checkpoint_dir}", file=sys.stderr)
        return 1

    token = resolve_token(args.token)
    api = HfApi(token=token)

    try:
        repo_id = resolve_repo_id(
            api=api,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            token=token,
            default_repo_name=args.default_repo_name,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to resolve repo id: {exc}", file=sys.stderr)
        return 1

    try:
        path_prefix = normalize_path_prefix(args.path_prefix)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)
    ignore_patterns.extend(args.ignore_pattern)

    file_count, total_bytes = summarize_directory(checkpoint_dir)
    commit_message = args.commit_message or f"Upload {checkpoint_dir.name} to {path_prefix}"

    print(f"[INFO] checkpoint_dir: {checkpoint_dir}")
    print(f"[INFO] path_prefix:    {path_prefix}")
    print(f"[INFO] repo_id:        {repo_id}")
    print(f"[INFO] repo_type:      {args.repo_type}")
    print(f"[INFO] revision:       {args.revision}")
    print(f"[INFO] files:          {file_count}")
    print(f"[INFO] size:           {format_size(total_bytes)}")
    print(f"[INFO] commit_message: {commit_message}")

    if args.dry_run:
        print("[INFO] Dry run complete. No upload performed.")
        return 0

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
            token=token,
        )
        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type=args.repo_type,
            folder_path=str(checkpoint_dir),
            path_in_repo=path_prefix,
            revision=args.revision,
            commit_message=commit_message,
            ignore_patterns=ignore_patterns,
            token=token,
        )
    except Exception as exc:
        print(f"[ERROR] Upload failed: {exc}", file=sys.stderr)
        return 1

    print("[INFO] Upload completed.")
    print(f"[INFO] Commit: {commit_info.oid}")
    print(f"[INFO] Repo:   {repo_id}")
    print(f"[INFO] Path:   {path_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
