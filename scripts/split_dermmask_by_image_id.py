#!/usr/bin/env python3
"""Create a reproducible image-disjoint split from DermMask MCQA rows.

The existing DermMask Qwen3.5 train/test files are split at the sample level.
This script recombines those rows, groups them by image_id, and assigns whole
image groups to train or test so that no image_id appears in both outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_INPUT = PROJECT_ROOT / "data/dermmask_qwen35_train_normalized.json"
DEFAULT_TEST_INPUT = PROJECT_ROOT / "data/dermmask_qwen35_test_normalized.json"
DEFAULT_TRAIN_OUTPUT = PROJECT_ROOT / "data/dermmask_qwen35_train_image_disjoint.json"
DEFAULT_TEST_OUTPUT = PROJECT_ROOT / "data/dermmask_qwen35_test_image_disjoint.json"
DEFAULT_METADATA_OUTPUT = PROJECT_ROOT / "data/dermmask_qwen35_image_disjoint_split.json"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_id_for(row: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    if isinstance(provenance, dict) and provenance.get("image_id"):
        return str(provenance["image_id"])
    if row.get("image_id"):
        return str(row["image_id"])
    raise ValueError(f"row has no image_id: {row.get('id')}")


def validate_input_rows(rows: list[dict[str, Any]]) -> None:
    ids = [str(row.get("id")) for row in rows]
    if any(row_id in {"None", ""} for row_id in ids):
        raise ValueError("input contains a row without id")
    if len(ids) != len(set(ids)):
        raise ValueError("input contains duplicate row ids")
    for row in rows:
        image_id_for(row)


def choose_test_groups(
    groups: dict[str, list[dict[str, Any]]],
    target_rows: int,
    seed: int,
) -> tuple[set[str], str]:
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    sizes = {len(rows) for rows in groups.values()}

    if len(sizes) == 1:
        group_size = next(iter(sizes))
        if target_rows % group_size == 0:
            group_count = target_rows // group_size
            return set(keys[:group_count]), "shuffled_whole_groups_exact"

    selected: set[str] = set()
    selected_rows = 0
    for key in keys:
        size = len(groups[key])
        if selected_rows + size <= target_rows:
            selected.add(key)
            selected_rows += size
        if selected_rows == target_rows:
            break
    return selected, "shuffled_whole_groups_greedy"


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--test-rows",
        type=int,
        default=None,
        help="Target test row count; defaults to the current test input row count.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_input = args.train_input.resolve()
    test_input = args.test_input.resolve()
    train_output = args.train_output.resolve()
    test_output = args.test_output.resolve()
    metadata_output = args.metadata_output.resolve()

    old_train = load_rows(train_input)
    old_test = load_rows(test_input)
    all_rows = old_train + old_test
    validate_input_rows(all_rows)

    target_test_rows = args.test_rows if args.test_rows is not None else len(old_test)
    if not 0 < target_test_rows < len(all_rows):
        raise ValueError(f"test row target must be between 1 and {len(all_rows) - 1}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[image_id_for(row)].append(row)

    test_image_ids, selection_method = choose_test_groups(groups, target_test_rows, args.seed)
    test_rows = [row for row in all_rows if image_id_for(row) in test_image_ids]
    train_rows = [row for row in all_rows if image_id_for(row) not in test_image_ids]

    train_image_ids = {image_id_for(row) for row in train_rows}
    test_image_ids_actual = {image_id_for(row) for row in test_rows}
    overlap = train_image_ids & test_image_ids_actual
    if overlap:
        raise RuntimeError(f"image-disjoint split failed; overlap examples: {sorted(overlap)[:5]}")

    output_ids = [str(row.get("id")) for row in train_rows + test_rows]
    if len(output_ids) != len(set(output_ids)):
        raise RuntimeError("output contains duplicate row ids")
    if len(train_rows) + len(test_rows) != len(all_rows):
        raise RuntimeError("output row count does not match input row count")

    write_json(train_output, train_rows)
    write_json(test_output, test_rows)

    metadata = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_policy": "whole image_id groups assigned to one split",
        "group_key": "provenance.image_id",
        "seed": args.seed,
        "selection_method": selection_method,
        "source": {
            "train_input": str(train_input),
            "test_input": str(test_input),
            "train_sha256": sha256_file(train_input),
            "test_sha256": sha256_file(test_input),
            "input_rows": len(all_rows),
            "input_unique_images": len(groups),
        },
        "outputs": {
            "train": str(train_output),
            "test": str(test_output),
            "metadata": str(metadata_output),
        },
        "counts": {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_unique_images": len(train_image_ids),
            "test_unique_images": len(test_image_ids_actual),
            "image_overlap": len(overlap),
            "target_test_rows": target_test_rows,
            "target_test_images": len(test_image_ids),
        },
        "validation": {
            "row_count_preserved": len(train_rows) + len(test_rows) == len(all_rows),
            "unique_row_ids": len(output_ids) == len(set(output_ids)),
            "image_disjoint": not overlap,
        },
    }
    write_json(metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
