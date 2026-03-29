#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split large eval JSON files into smaller parts by dividing the episodes list."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more eval JSON files to split.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=2,
        help="Number of parts to split into. Default: 2.",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete the original file after successful splitting.",
    )
    return parser


def split_indices(total: int, parts: int) -> list[tuple[int, int]]:
    base = total // parts
    extra = total % parts
    spans = []
    start = 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        end = start + size
        spans.append((start, end))
        start = end
    return spans


def build_part_payload(data: dict, episodes: list, part_index: int, num_parts: int, start: int, end: int) -> dict:
    payload = dict(data)
    payload["episodes"] = episodes[start:end]

    metadata = dict(payload.get("metadata", {}))
    metadata["split_part"] = part_index
    metadata["split_parts_total"] = num_parts
    metadata["split_episode_start"] = start
    metadata["split_episode_end"] = end
    metadata["split_episode_count"] = end - start
    metadata["original_episode_count"] = len(episodes)
    payload["metadata"] = metadata

    summary = dict(payload.get("summary", {}))
    if "num_episodes" in summary:
        summary["num_episodes"] = end - start
    if "num_success" in summary:
        summary["num_success"] = sum(1 for episode in payload["episodes"] if episode.get("success"))
    if "num_failure" in summary:
        summary["num_failure"] = (end - start) - summary.get("num_success", 0)
    payload["summary"] = summary

    return payload


def split_file(path: Path, parts: int, delete_original: bool) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("episodes"), list):
        raise ValueError(f"{path} does not look like an eval JSON with a top-level episodes list")

    episodes = data["episodes"]
    if len(episodes) < parts:
        raise ValueError(f"{path} has only {len(episodes)} episodes, cannot split into {parts} parts")

    written_paths = []
    for part_index, (start, end) in enumerate(split_indices(len(episodes), parts), start=1):
        out_path = path.with_name(f"{path.stem}.part{part_index}{path.suffix}")
        payload = build_part_payload(data, episodes, part_index, parts, start, end)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        written_paths.append(out_path)
        print(f"[OK] wrote {out_path} with {end - start} episodes")

    if delete_original:
        path.unlink()
        print(f"[OK] deleted original {path}")


def main() -> None:
    args = build_parser().parse_args()
    for raw_path in args.paths:
        split_file(Path(raw_path), args.parts, args.delete_original)


if __name__ == "__main__":
    main()