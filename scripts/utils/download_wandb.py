import os
import json
from pathlib import Path

import pandas as pd
import wandb


ENTITY = "yuxianglin2025-northwestern-university"        # 例如你的用户名或 team 名
PROJECT = "ufb_train_batch"      # project 名
OUTDIR = Path("/workspace/unary-feedback/Trained_wandb_datasets")

# 是否下载 run files
DOWNLOAD_RUN_FILES = True

# 是否下载 artifacts（checkpoint / dataset 等）
DOWNLOAD_ARTIFACTS = True

# 是否尝试下载 parquet history 导出
TRY_HISTORY_EXPORT = True


def safe_name(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "@"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def export_run_history_csv(run, run_dir: Path):
    """
    用 scan_history 导出全量 history，而不是采样后的 history()
    """
    rows = list(run.scan_history())
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "history_full.csv", index=False)


def export_run_files(run, run_dir: Path):
    files_dir = run_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    for f in run.files():
        try:
            # root 指定下载根目录，replace=True 覆盖已有文件
            f.download(root=str(files_dir), replace=True)
        except Exception as e:
            print(f"[WARN] failed to download file {f.name} for {run.path}: {e}")


def export_history_parquet_if_possible(run, run_dir: Path):
    """
    官方文档中的 download_history_exports()
    """
    try:
        result = run.download_history_exports(download_dir=run_dir / "history_exports")
        print(f"[OK] parquet history exported for {run.path}: {result}")
    except Exception as e:
        print(f"[WARN] history export parquet failed for {run.path}: {e}")


def export_artifacts(run, run_dir: Path):
    art_root = run_dir / "artifacts"
    art_root.mkdir(parents=True, exist_ok=True)

    # 该 run 输出的 artifacts
    logged_dir = art_root / "logged"
    logged_dir.mkdir(parents=True, exist_ok=True)
    try:
        for art in run.logged_artifacts():
            target = logged_dir / safe_name(art.name.replace(":", "__"))
            try:
                art.download(root=str(target))
            except Exception as e:
                print(f"[WARN] failed logged artifact {art.name} for {run.path}: {e}")
    except Exception as e:
        print(f"[WARN] cannot list logged artifacts for {run.path}: {e}")

    # 该 run 使用过的 artifacts
    used_dir = art_root / "used"
    used_dir.mkdir(parents=True, exist_ok=True)
    try:
        for art in run.used_artifacts():
            target = used_dir / safe_name(art.name.replace(":", "__"))
            try:
                art.download(root=str(target))
            except Exception as e:
                print(f"[WARN] failed used artifact {art.name} for {run.path}: {e}")
    except Exception as e:
        print(f"[WARN] cannot list used artifacts for {run.path}: {e}")


def main():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    project_index = []

    for i, run in enumerate(runs):
        run_name = safe_name(run.name or "unnamed")
        run_id = run.id
        run_dir = OUTDIR / f"{i:04d}_{run_name}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Exporting {run.path} -> {run_dir} ===")

        # 基本信息
        meta = {
            "id": run.id,
            "name": run.name,
            "path": "/".join(run.path) if isinstance(run.path, (list, tuple)) else str(run.path),
            "state": getattr(run, "state", None),
            "url": getattr(run, "url", None),
            "project": getattr(run, "project", None),
            "entity": getattr(run, "entity", None),
            "group": getattr(run, "group", None),
            "job_type": getattr(run, "job_type", None),
            "created_at": getattr(run, "created_at", None),
            "heartbeat_at": getattr(run, "heartbeat_at", None),
            "commit": getattr(run, "commit", None),
            "tags": getattr(run, "tags", None),
        }
        save_json(run_dir / "run_meta.json", meta)
        save_json(run_dir / "config.json", dict(run.config))
        save_json(run_dir / "summary.json", dict(run.summary))

        # 全项目索引
        project_index.append({
            "run_id": run.id,
            "run_name": run.name,
            "state": getattr(run, "state", None),
            "dir": str(run_dir),
            "url": getattr(run, "url", None),
        })

        # 全量 history
        try:
            export_run_history_csv(run, run_dir)
        except Exception as e:
            print(f"[WARN] failed scan_history for {run.path}: {e}")

        # parquet history export
        if TRY_HISTORY_EXPORT:
            export_history_parquet_if_possible(run, run_dir)

        # run files
        if DOWNLOAD_RUN_FILES:
            export_run_files(run, run_dir)

        # artifacts
        if DOWNLOAD_ARTIFACTS:
            export_artifacts(run, run_dir)

    pd.DataFrame(project_index).to_csv(OUTDIR / "project_runs_index.csv", index=False)
    print(f"\nDone. Exported to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()