#!/usr/bin/env python3
"""Batch upload Hugging Face model directories from a parent folder.

Example:
    export HF_TOKEN=hf_xxx
    python scripts/utils/upload_hf_models_in_dir.py \
        --input-dir outputs/hf \
        --delete-after-upload \
        --remove-input-dir-if-empty

Each immediate child directory under ``--input-dir`` is treated as one model repo.
By default repos are created as public and named ``<username>/<child_dir_name>``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi


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

WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.ckpt",
)


@dataclass(frozen=True)
class UploadTarget:
    local_dir: Path
    repo_id: str
    file_count: int
    total_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload every HF model subdirectory under a parent directory.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Parent directory whose immediate child directories are HF model folders.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="HF namespace/user/org. Defaults to the currently logged-in user.",
    )
    parser.add_argument(
        "--repo-prefix",
        default="",
        help="Optional prefix added before each repo name, e.g. exp1_.",
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="model",
        help="Repo type to create on the Hub. Defaults to model.",
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
        "--commit-message-template",
        default="Upload {model_name}",
        help="Commit message template. Available fields: model_name, repo_id.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repos as private. Public is the default.",
    )
    parser.add_argument(
        "--delete-after-upload",
        action="store_true",
        help="Delete each local model directory after a successful upload.",
    )
    parser.add_argument(
        "--remove-input-dir-if-empty",
        action="store_true",
        help="Remove --input-dir after uploads if it becomes empty.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue uploading remaining directories after a failure.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Upload all child directories without checking whether they look like HF model dirs.",
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
        help="Print discovered repos and exit without uploading.",
    )
    return parser.parse_args()


def resolve_token(cli_token: str | None) -> str | None:
    return cli_token or os.getenv("HF_UPLOAD_TOKEN") or os.getenv("HF_TOKEN")


def resolve_namespace(api: HfApi, namespace: str | None, token: str | None) -> str:
    if namespace:
        return namespace
    whoami = api.whoami(token=token)
    return whoami["name"]


def looks_like_hf_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    return any(any(path.glob(pattern)) for pattern in WEIGHT_PATTERNS)


def summarize_directory(root: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
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


def discover_targets(
    input_dir: Path,
    namespace: str,
    repo_prefix: str,
    skip_validation: bool,
) -> tuple[list[UploadTarget], list[Path]]:
    targets: list[UploadTarget] = []
    skipped: list[Path] = []

    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        if not skip_validation and not looks_like_hf_model_dir(child):
            skipped.append(child)
            continue

        repo_name = f"{repo_prefix}{child.name}"
        file_count, total_bytes = summarize_directory(child)
        targets.append(
            UploadTarget(
                local_dir=child,
                repo_id=f"{namespace}/{repo_name}",
                file_count=file_count,
                total_bytes=total_bytes,
            )
        )

    return targets, skipped


def upload_target(
    api: HfApi,
    target: UploadTarget,
    token: str | None,
    repo_type: str,
    revision: str,
    private: bool,
    commit_message_template: str,
    ignore_patterns: list[str],
) -> str:
    commit_message = commit_message_template.format(
        model_name=target.local_dir.name,
        repo_id=target.repo_id,
    )
    api.create_repo(
        repo_id=target.repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
        token=token,
    )
    commit_info = api.upload_folder(
        repo_id=target.repo_id,
        repo_type=repo_type,
        folder_path=str(target.local_dir),
        revision=revision,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
        token=token,
    )
    return commit_info.oid


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"[ERROR] Input path is not a directory: {input_dir}", file=sys.stderr)
        return 1

    token = resolve_token(args.token)
    api = HfApi(token=token)

    try:
        namespace = resolve_namespace(api=api, namespace=args.namespace, token=token)
    except Exception as exc:
        print(f"[ERROR] Failed to resolve HF namespace: {exc}", file=sys.stderr)
        return 1

    ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)
    ignore_patterns.extend(args.ignore_pattern)

    try:
        targets, skipped = discover_targets(
            input_dir=input_dir,
            namespace=namespace,
            repo_prefix=args.repo_prefix,
            skip_validation=args.skip_validation,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to scan input directory: {exc}", file=sys.stderr)
        return 1

    if skipped:
        print("[INFO] Skipping directories that do not look like complete HF model folders:")
        for path in skipped:
            print(f"  - {path}")

    if not targets:
        print("[ERROR] No uploadable model directories found.", file=sys.stderr)
        return 1

    print(f"[INFO] input_dir: {input_dir}")
    print(f"[INFO] namespace: {namespace}")
    print(f"[INFO] repo_type: {args.repo_type}")
    print(f"[INFO] public:    {not args.private}")
    print(f"[INFO] targets:   {len(targets)}")
    for target in targets:
        print(
            f"[PLAN] {target.local_dir.name} -> {target.repo_id} "
            f"({target.file_count} files, {format_size(target.total_bytes)})"
        )

    if args.dry_run:
        print("[INFO] Dry run complete. No upload performed.")
        return 0

    successes: list[tuple[UploadTarget, str]] = []
    failures: list[tuple[UploadTarget, str]] = []

    for index, target in enumerate(targets, start=1):
        print(f"[INFO] ({index}/{len(targets)}) Uploading {target.local_dir} -> {target.repo_id}")
        try:
            commit_oid = upload_target(
                api=api,
                target=target,
                token=token,
                repo_type=args.repo_type,
                revision=args.revision,
                private=args.private,
                commit_message_template=args.commit_message_template,
                ignore_patterns=ignore_patterns,
            )
            print(f"[INFO] Uploaded {target.repo_id} at commit {commit_oid}")
            successes.append((target, commit_oid))

            if args.delete_after_upload:
                shutil.rmtree(target.local_dir)
                print(f"[INFO] Deleted local directory {target.local_dir}")
        except Exception as exc:
            print(f"[ERROR] Failed to upload {target.local_dir}: {exc}", file=sys.stderr)
            failures.append((target, str(exc)))
            if not args.continue_on_error:
                break

    if args.delete_after_upload and args.remove_input_dir_if_empty:
        try:
            if not any(input_dir.iterdir()):
                input_dir.rmdir()
                print(f"[INFO] Removed empty input directory {input_dir}")
        except Exception as exc:
            print(f"[WARN] Failed to remove empty input directory {input_dir}: {exc}", file=sys.stderr)

    print("[INFO] Upload summary:")
    print(f"[INFO]   success: {len(successes)}")
    print(f"[INFO]   failure: {len(failures)}")

    for target, commit_oid in successes:
        print(f"[DONE] {target.repo_id} commit={commit_oid}")

    for target, error_message in failures:
        print(f"[FAIL] {target.local_dir} -> {target.repo_id}: {error_message}", file=sys.stderr)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
