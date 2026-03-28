#!/usr/bin/env python3
"""Convert actor checkpoints to temporary HF models, upload them, then clean up.

Example:
    export HF_TOKEN=hf_xxx
    python scripts/utils/convert_and_upload_checkpoints_to_hf.py \
        --checkpoint-parent-dir outputs/checkpoints/exp1_MetamathQA \
        --base-model Qwen/Qwen2.5-3B-Instruct

The script scans ``global_step_*/actor`` directories under ``--checkpoint-parent-dir``,
converts each checkpoint to a temporary HF model directory, uploads it as a public
model repo by default, and removes the temporary HF directory after a successful upload.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi

from convert_fsdp_to_hf import convert_fsdp_to_hf
from upload_hf_models_in_dir import (
    DEFAULT_IGNORE_PATTERNS,
    format_size,
    resolve_namespace,
    resolve_token,
    summarize_directory,
)


STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")
SHARD_FILE_RE = re.compile(r"^model_world_size_(\d+)_rank_(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointTarget:
    step_name: str
    step_number: int
    actor_dir: Path
    world_size: int
    repo_name: str
    repo_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert actor checkpoints to HF format, upload them, and clean up temp dirs.",
    )
    parser.add_argument(
        "--checkpoint-parent-dir",
        required=True,
        help="Directory containing global_step_*/actor checkpoint subdirectories.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model used to reconstruct the HF model before loading checkpoint shards.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="HF namespace/user/org. Defaults to the currently logged-in user.",
    )
    parser.add_argument(
        "--repo-prefix",
        default="",
        help="Optional prefix added before each generated repo name.",
    )
    parser.add_argument(
        "--repo-name-template",
        default="{experiment_name}_{step_name}",
        help=(
            "Template used to build repo names. Available fields: "
            "experiment_name, step_name, step_number."
        ),
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
        default="Upload {repo_name}",
        help=(
            "Commit message template. Available fields: "
            "repo_name, repo_id, experiment_name, step_name, step_number."
        ),
    )
    parser.add_argument(
        "--temp-root",
        default=None,
        help=(
            "Parent directory for temporary HF model folders. "
            "Defaults to <checkpoint-parent-dir>/.hf_upload_tmp."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repos as private. Public is the default.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing later checkpoints after a failure.",
    )
    parser.add_argument(
        "--keep-temp-on-error",
        action="store_true",
        help="Keep the temporary HF directory for failed checkpoints.",
    )
    parser.add_argument(
        "--remove-temp-root-if-empty",
        action="store_true",
        help="Remove the temp root directory after the run if it becomes empty.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Additional glob pattern to ignore during upload. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered checkpoints and target repos without converting or uploading.",
    )
    return parser.parse_args()


def infer_world_size(actor_dir: Path) -> int:
    world_sizes: set[int] = set()
    ranks: set[int] = set()

    for path in actor_dir.iterdir():
        if not path.is_file():
            continue
        match = SHARD_FILE_RE.match(path.name)
        if match is None:
            continue
        world_size = int(match.group(1))
        rank = int(match.group(2))
        world_sizes.add(world_size)
        ranks.add(rank)

    if not world_sizes:
        raise ValueError(f"No FSDP model shard files found under {actor_dir}")
    if len(world_sizes) != 1:
        raise ValueError(f"Found inconsistent world sizes under {actor_dir}: {sorted(world_sizes)}")

    world_size = next(iter(world_sizes))
    expected_ranks = set(range(world_size))
    if ranks != expected_ranks:
        raise ValueError(
            f"Missing shard ranks under {actor_dir}: expected {sorted(expected_ranks)}, got {sorted(ranks)}"
        )
    return world_size


def discover_targets(
    checkpoint_parent_dir: Path,
    namespace: str,
    repo_prefix: str,
    repo_name_template: str,
) -> tuple[list[CheckpointTarget], list[Path]]:
    experiment_name = checkpoint_parent_dir.name
    targets: list[CheckpointTarget] = []
    skipped: list[Path] = []

    for child in sorted(checkpoint_parent_dir.iterdir()):
        if not child.is_dir():
            continue
        match = STEP_DIR_RE.match(child.name)
        if match is None:
            continue

        step_number = int(match.group(1))
        actor_dir = child / "actor"
        if not actor_dir.is_dir():
            skipped.append(child)
            continue

        try:
            world_size = infer_world_size(actor_dir)
        except ValueError:
            skipped.append(actor_dir)
            continue

        repo_name = repo_prefix + repo_name_template.format(
            experiment_name=experiment_name,
            step_name=child.name,
            step_number=step_number,
        )
        targets.append(
            CheckpointTarget(
                step_name=child.name,
                step_number=step_number,
                actor_dir=actor_dir,
                world_size=world_size,
                repo_name=repo_name,
                repo_id=f"{namespace}/{repo_name}",
            )
        )

    targets.sort(key=lambda target: target.step_number)
    return targets, skipped


def upload_directory(
    api: HfApi,
    local_dir: Path,
    repo_id: str,
    token: str | None,
    revision: str,
    private: bool,
    commit_message: str,
    ignore_patterns: list[str],
) -> str:
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        revision=revision,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
        token=token,
    )
    return commit_info.oid


def remove_dir_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def main() -> int:
    args = parse_args()

    checkpoint_parent_dir = Path(args.checkpoint_parent_dir).expanduser().resolve()
    if not checkpoint_parent_dir.exists():
        print(f"[ERROR] Checkpoint parent directory not found: {checkpoint_parent_dir}", file=sys.stderr)
        return 1
    if not checkpoint_parent_dir.is_dir():
        print(f"[ERROR] Checkpoint parent path is not a directory: {checkpoint_parent_dir}", file=sys.stderr)
        return 1

    if not args.dry_run and not args.base_model:
        print(
            "[ERROR] --base-model is required for conversion. "
            "The checkpoint shards do not reliably preserve the original model path.",
            file=sys.stderr,
        )
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
            checkpoint_parent_dir=checkpoint_parent_dir,
            namespace=namespace,
            repo_prefix=args.repo_prefix,
            repo_name_template=args.repo_name_template,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to scan checkpoint directory: {exc}", file=sys.stderr)
        return 1

    if skipped:
        print("[INFO] Skipping paths that do not have complete actor shard checkpoints:")
        for path in skipped:
            print(f"  - {path}")

    if not targets:
        print("[ERROR] No uploadable actor checkpoints found.", file=sys.stderr)
        return 1

    temp_root = (
        Path(args.temp_root).expanduser().resolve()
        if args.temp_root
        else checkpoint_parent_dir / ".hf_upload_tmp"
    )

    print(f"[INFO] checkpoint_parent_dir: {checkpoint_parent_dir}")
    print(f"[INFO] namespace:             {namespace}")
    print(f"[INFO] temp_root:             {temp_root}")
    print(f"[INFO] public:                {not args.private}")
    print(f"[INFO] targets:               {len(targets)}")
    for target in targets:
        print(
            f"[PLAN] {target.actor_dir} -> {target.repo_id} "
            f"(world_size={target.world_size})"
        )

    if args.dry_run:
        print("[INFO] Dry run complete. No conversion or upload performed.")
        return 0

    temp_root.mkdir(parents=True, exist_ok=True)

    successes: list[tuple[CheckpointTarget, str]] = []
    failures: list[tuple[CheckpointTarget, str]] = []

    for index, target in enumerate(targets, start=1):
        temp_run_dir = Path(tempfile.mkdtemp(prefix=f"{target.step_name}_", dir=str(temp_root)))
        temp_model_dir = temp_run_dir / target.repo_name
        commit_message = args.commit_message_template.format(
            repo_name=target.repo_name,
            repo_id=target.repo_id,
            experiment_name=checkpoint_parent_dir.name,
            step_name=target.step_name,
            step_number=target.step_number,
        )

        print(f"[INFO] ({index}/{len(targets)}) Converting {target.actor_dir} -> {temp_model_dir}")
        try:
            success = convert_fsdp_to_hf(
                checkpoint_dir=str(target.actor_dir),
                output_dir=str(temp_model_dir),
                base_model=args.base_model,
                world_size=target.world_size,
            )
            if not success:
                raise RuntimeError("Conversion returned False")

            file_count, total_bytes = summarize_directory(temp_model_dir)
            print(
                f"[INFO] Converted {target.step_name}: "
                f"{file_count} files, {format_size(total_bytes)}"
            )

            print(f"[INFO] Uploading {temp_model_dir} -> {target.repo_id}")
            commit_oid = upload_directory(
                api=api,
                local_dir=temp_model_dir,
                repo_id=target.repo_id,
                token=token,
                revision=args.revision,
                private=args.private,
                commit_message=commit_message,
                ignore_patterns=ignore_patterns,
            )
            print(f"[INFO] Uploaded {target.repo_id} at commit {commit_oid}")
            successes.append((target, commit_oid))

            remove_dir_if_exists(temp_run_dir)
            print(f"[INFO] Deleted temporary HF directory {temp_run_dir}")
        except Exception as exc:
            print(f"[ERROR] Failed for {target.actor_dir}: {exc}", file=sys.stderr)
            failures.append((target, str(exc)))
            if not args.keep_temp_on_error:
                remove_dir_if_exists(temp_run_dir)
            else:
                print(f"[WARN] Kept temporary directory for debugging: {temp_run_dir}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if args.remove_temp_root_if_empty:
        try:
            if temp_root.exists() and not any(temp_root.iterdir()):
                temp_root.rmdir()
                print(f"[INFO] Removed empty temp root {temp_root}")
        except Exception as exc:
            print(f"[WARN] Failed to remove temp root {temp_root}: {exc}", file=sys.stderr)

    print("[INFO] Upload summary:")
    print(f"[INFO]   success: {len(successes)}")
    print(f"[INFO]   failure: {len(failures)}")

    for target, commit_oid in successes:
        print(f"[DONE] {target.repo_id} commit={commit_oid}")

    for target, error_message in failures:
        print(f"[FAIL] {target.actor_dir} -> {target.repo_id}: {error_message}", file=sys.stderr)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
