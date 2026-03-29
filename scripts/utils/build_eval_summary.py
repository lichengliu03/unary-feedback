#!/usr/bin/env python3
"""Build combined and per-environment summary JSONs from a detailed eval payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detailed_json", type=Path, help="Path to the detailed eval JSON payload.")
    parser.add_argument("summary_json", type=Path, help="Output path for the combined summary JSON.")
    parser.add_argument(
        "--per-env-summary-dir",
        type=Path,
        default=None,
        help="Optional directory to write one summary JSON per environment tag.",
    )
    parser.add_argument(
        "--default-max-attempt",
        type=int,
        default=None,
        help="Optional default max attempt count used for attempt-level pass@k.",
    )
    parser.add_argument(
        "--env-max-attempt",
        action="append",
        default=[],
        help="Optional per-env max attempt override in the form ENV_TAG=MAX_ATTEMPT.",
    )
    parser.add_argument(
        "--env-attempt-source",
        action="append",
        default=[],
        help="Optional per-env attempt source in the form ENV_TAG=turn or ENV_TAG=attempt_num.",
    )
    parser.add_argument(
        "--env-max-turn",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def parse_env_int_overrides(items: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid ENV_TAG=INT override value: {item}")
        tag, value = item.split("=", 1)
        tag = tag.strip()
        value = value.strip()
        if not tag:
            raise ValueError(f"Invalid ENV_TAG=INT override value: {item}")
        overrides[tag] = max(1, int(value))
    return overrides


def parse_env_str_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid ENV_TAG=VALUE override value: {item}")
        tag, value = item.split("=", 1)
        tag = tag.strip()
        value = value.strip()
        if not tag or not value:
            raise ValueError(f"Invalid ENV_TAG=VALUE override value: {item}")
        overrides[tag] = value
    return overrides


def normalize_attempt_source(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"turn", "attempt_num"}:
        raise ValueError(f"Unsupported attempt source: {value}")
    return normalized


def resolve_attempt_source(
    tag: str | None,
    attempt_source_overrides: dict[str, str],
    default_attempt_source: str,
) -> str:
    if tag is not None and tag in attempt_source_overrides:
        return attempt_source_overrides[tag]
    return default_attempt_source


def extract_attempt_num_from_turn(turn: dict[str, Any]) -> int | None:
    attempt_num = turn.get("attempt_num")
    if is_number(attempt_num):
        return int(attempt_num)

    info = turn.get("info") or {}
    attempt_num = info.get("attempt_num")
    if is_number(attempt_num):
        attempt_num = int(attempt_num)
        # Retry info is logged as the next attempt number (1-based after first retry).
        return max(0, attempt_num)

    return None


def extract_success_at_attempt(record: dict[str, Any], *, attempt_source: str) -> int:
    if not record.get("success"):
        return -1

    turns = record.get("turns") or []
    success_at_turn = record.get("success_at_turn")

    if attempt_source == "turn":
        if is_number(success_at_turn) and int(success_at_turn) >= 0:
            return int(success_at_turn)
        for index, turn in enumerate(turns):
            info = turn.get("info") or {}
            if info.get("success") is True:
                return index
        if turns:
            return max(0, len(turns) - 1)
        return 0

    if is_number(success_at_turn):
        turn_index = int(success_at_turn)
        if 0 <= turn_index < len(turns):
            attempt_num = extract_attempt_num_from_turn(turns[turn_index])
            if attempt_num is not None:
                return attempt_num

    for turn in turns:
        info = turn.get("info") or {}
        if info.get("success") is True:
            attempt_num = extract_attempt_num_from_turn(turn)
            if attempt_num is not None:
                return attempt_num

    attempt_nums = [
        attempt_num
        for attempt_num in (extract_attempt_num_from_turn(turn) for turn in turns)
        if attempt_num is not None
    ]
    if attempt_nums:
        return max(attempt_nums)

    final_info = record.get("final_info") or {}
    attempt_num = final_info.get("attempt_num", final_info.get("attempt"))
    if is_number(attempt_num):
        attempt_num = int(attempt_num)
        return max(0, attempt_num)

    return 0


def infer_attempt_count(record: dict[str, Any], *, attempt_source: str) -> int:
    turns = record.get("turns") or []

    if attempt_source == "turn":
        if turns:
            return len(turns)
        turns_taken = record.get("turns_taken")
        if is_number(turns_taken):
            return max(1, int(turns_taken))
        success_at_turn = record.get("success_at_turn")
        if is_number(success_at_turn) and int(success_at_turn) >= 0:
            return int(success_at_turn) + 1
        return 1

    attempt_nums = [
        attempt_num
        for attempt_num in (extract_attempt_num_from_turn(turn) for turn in turns)
        if attempt_num is not None
    ]
    if attempt_nums:
        return max(attempt_nums) + 1

    final_info = record.get("final_info") or {}
    max_attempts = final_info.get("max_attempts")
    if is_number(max_attempts):
        return max(1, int(max_attempts))

    attempt_num = final_info.get("attempt_num", final_info.get("attempt"))
    if is_number(attempt_num):
        return max(1, int(attempt_num) + 1)

    if record.get("success"):
        return max(1, extract_success_at_attempt(record, attempt_source=attempt_source) + 1)

    return 1


def infer_max_attempt(
    records: list[dict[str, Any]],
    *,
    attempt_source_overrides: dict[str, str],
    default_attempt_source: str,
) -> int:
    observed = []
    for record in records:
        attempt_source = resolve_attempt_source(record.get("tag"), attempt_source_overrides, default_attempt_source)
        observed.append(infer_attempt_count(record, attempt_source=attempt_source))

    if observed:
        return max(observed)
    return 5


def compute_pass_at_k(
    records: list[dict[str, Any]],
    *,
    max_attempt: int,
    attempt_source_overrides: dict[str, str],
    default_attempt_source: str,
) -> dict[str, float]:
    total = len(records)
    if total == 0:
        return {}

    success_attempts = [
        extract_success_at_attempt(
            record,
            attempt_source=resolve_attempt_source(record.get("tag"), attempt_source_overrides, default_attempt_source),
        )
        for record in records
    ]

    pass_at_k = {}
    for k in range(1, max_attempt + 1):
        successes = sum(1 for attempt in success_attempts if 0 <= attempt < k)
        pass_at_k[f"pass@{k}"] = successes / total
    return pass_at_k


def summarize_attempt_source(
    records: list[dict[str, Any]],
    *,
    attempt_source_overrides: dict[str, str],
    default_attempt_source: str,
) -> str:
    sources = {
        resolve_attempt_source(record.get("tag"), attempt_source_overrides, default_attempt_source)
        for record in records
    }
    if not sources:
        return default_attempt_source
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


def build_entry(
    records: list[dict[str, Any]],
    payload_summary: dict[str, Any],
    *,
    tag: str | None,
    max_attempt: int,
    attempt_source_overrides: dict[str, str],
    default_attempt_source: str,
) -> dict[str, Any]:
    total = len(records)
    success_count = sum(1 for record in records if record.get("success"))
    avg_success = (success_count / total) if total else 0.0

    reward_values = [float(record["score"]) for record in records if is_number(record.get("score"))]
    action_values = [float(record["num_actions"]) for record in records if is_number(record.get("num_actions"))]

    batch = {"success": avg_success}
    avg_reward = mean(reward_values)
    if avg_reward is not None:
        batch["reward"] = avg_reward

    avg_num_actions = mean(action_values)
    if avg_num_actions is not None:
        batch["num_actions"] = avg_num_actions

    response_length = payload_summary.get("val/response_length")
    if is_number(response_length):
        batch["response_length"] = float(response_length)

    entry = {
        "num_batches": 1,
        "avg_success": avg_success,
        "min_success": avg_success,
        "max_success": avg_success,
        "num_episodes": total,
        "num_success": success_count,
        "num_failure": total - success_count,
        "batches": [batch],
    }

    if avg_reward is not None:
        entry["avg_reward"] = avg_reward
    if avg_num_actions is not None:
        entry["avg_num_actions"] = avg_num_actions
    if is_number(response_length):
        entry["avg_response_length"] = float(response_length)

    pass_at_k = compute_pass_at_k(
        records,
        max_attempt=max_attempt,
        attempt_source_overrides=attempt_source_overrides,
        default_attempt_source=default_attempt_source,
    )
    if pass_at_k:
        entry["pass_at_k"] = pass_at_k
        entry["pass_at_k_unit"] = "attempt"
        entry["attempt_source"] = summarize_attempt_source(
            records,
            attempt_source_overrides=attempt_source_overrides,
            default_attempt_source=default_attempt_source,
        )
        entry["max_attempts"] = max_attempt

    if tag is not None:
        entry["env_tag"] = tag

    return entry


def sanitize_metadata(payload: dict[str, Any], *, env_tag: str | None = None) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    metadata["source_format_version"] = payload.get("format_version")
    metadata["pass_at_k_unit"] = "attempt"
    if env_tag is not None:
        metadata["env_tag"] = env_tag
        metadata["env_tags"] = [env_tag]
    return metadata


def main() -> None:
    args = parse_args()
    payload = json.loads(args.detailed_json.read_text(encoding="utf-8"))
    env_max_attempt_overrides = parse_env_int_overrides(args.env_max_attempt)
    env_attempt_source_overrides = {
        tag: normalize_attempt_source(value)
        for tag, value in parse_env_str_overrides(args.env_attempt_source).items()
    }
    default_attempt_source = "turn"

    records = payload.get("episodes") or []
    summary = payload.get("summary") or {}
    inferred_max_attempt = infer_max_attempt(
        records,
        attempt_source_overrides=env_attempt_source_overrides,
        default_attempt_source=default_attempt_source,
    )
    if args.default_max_attempt is not None:
        default_max_attempt = max(1, int(args.default_max_attempt))
    else:
        default_max_attempt = max(
            [value for value in [inferred_max_attempt, *env_max_attempt_overrides.values()] if value is not None],
            default=5,
        )

    by_env_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        tag = record.get("tag", "unknown")
        by_env_records.setdefault(tag, []).append(record)

    overall_entry = build_entry(
        records,
        summary,
        tag=None,
        max_attempt=default_max_attempt,
        attempt_source_overrides=env_attempt_source_overrides,
        default_attempt_source=default_attempt_source,
    )
    by_env_entries = {
        tag: build_entry(
            tag_records,
            summary,
            tag=tag,
            max_attempt=env_max_attempt_overrides.get(tag, default_max_attempt),
            attempt_source_overrides=env_attempt_source_overrides,
            default_attempt_source=default_attempt_source,
        )
        for tag, tag_records in sorted(by_env_records.items())
    }

    combined_output = {
        "base_model": overall_entry,
        "by_env": by_env_entries,
        "metadata": sanitize_metadata(payload),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(combined_output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Wrote combined summary JSON: {args.summary_json}")

    if args.per_env_summary_dir is None:
        return

    args.per_env_summary_dir.mkdir(parents=True, exist_ok=True)
    for tag, entry in by_env_entries.items():
        output_path = args.per_env_summary_dir / f"{tag}.summary.json"
        output_payload = {
            "base_model": entry,
            "metadata": sanitize_metadata(payload, env_tag=tag),
        }
        output_path.write_text(
            json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] Wrote per-env summary JSON: {output_path}")


if __name__ == "__main__":
    main()
