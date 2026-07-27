#!/usr/bin/env python3
"""Extract strict A-D four-choice DermMask manifests from normalized data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_IN = PROJECT_ROOT / "data/dermmask_qwen35_train_normalized.json"
DEFAULT_TEST_IN = PROJECT_ROOT / "data/dermmask_qwen35_test_normalized.json"
DEFAULT_TRAIN_OUT = PROJECT_ROOT / "data/dermmask_qwen35_4choice_train.json"
DEFAULT_TEST_OUT = PROJECT_ROOT / "data/dermmask_qwen35_4choice_test.json"
DEFAULT_READY_OUT = PROJECT_ROOT / "data/dermmask_qwen35_4choice_ready.json"
STRICT_LABELS = ["A", "B", "C", "D"]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{path} must be a JSON array")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_abs_path_text(value: str) -> bool:
    if value.startswith("zip://"):
        archive = value[len("zip://") :].split("!", 1)[0]
        return archive.startswith("/")
    return value.startswith("/")


def collect_absolute_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(collect_absolute_paths(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            found.extend(collect_absolute_paths(item, f"{prefix}[{idx}]"))
    elif isinstance(value, str) and is_abs_path_text(value):
        found.append(prefix)
    return found


def is_strict_four_choice(row: dict[str, Any]) -> bool:
    options = row.get("options")
    if not isinstance(options, list) or len(options) != 4:
        return False
    labels = [opt.get("label") for opt in options if isinstance(opt, dict)]
    if labels != STRICT_LABELS:
        return False
    answer = row.get("gold_label") or row.get("answer")
    if answer not in STRICT_LABELS:
        return False
    return sum(1 for opt in options if opt.get("label") == answer) == 1


def extract_split(source: Path, target: Path) -> dict[str, Any]:
    rows = load_rows(source)
    ids_seen: set[str] = set()
    output: list[dict[str, Any]] = []
    source_ids: list[str] = []
    for row in rows:
        row_id = row.get("id")
        if not row_id:
            raise ValueError(f"{source} contains row without id")
        if row_id in ids_seen:
            raise ValueError(f"{source} contains duplicate id {row_id}")
        ids_seen.add(row_id)
        if is_strict_four_choice(row):
            output.append(row)
            source_ids.append(str(row_id))

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    absolute_paths = collect_absolute_paths(output)
    if absolute_paths:
        raise ValueError(f"{target} contains absolute path fields: {absolute_paths[:10]}")
    return {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "output": str(target.relative_to(PROJECT_ROOT)),
        "source_count": len(rows),
        "output_count": len(output),
        "filtered_count": len(rows) - len(output),
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(target),
        "first_id": source_ids[0] if source_ids else None,
        "last_id": source_ids[-1] if source_ids else None,
    }


def write_ready(args: argparse.Namespace, train: dict[str, Any], test: dict[str, Any]) -> None:
    ready = {
        "status": "PASS",
        "path_format": "relative",
        "filter_rule": {
            "source_fields": ["options", "gold_label", "answer"],
            "require_options_count": 4,
            "require_labels_exact_order": STRICT_LABELS,
            "require_answer_in": STRICT_LABELS,
            "note": "Filtering is based on recovered normalized options, not raw manifest options.",
        },
        "splits": {
            "train": train,
            "test": test,
        },
        "source": {
            "train": str(Path(args.train_in).resolve().relative_to(PROJECT_ROOT)),
            "test": str(Path(args.test_in).resolve().relative_to(PROJECT_ROOT)),
        },
        "outputs": {
            "train": str(Path(args.train_out).resolve().relative_to(PROJECT_ROOT)),
            "test": str(Path(args.test_out).resolve().relative_to(PROJECT_ROOT)),
        },
        "script": "scripts/extract_dermmask_4choice.py",
    }
    ready_path = Path(args.ready_out).resolve()
    ready_path.write_text(
        json.dumps(ready, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-in", default=str(DEFAULT_TRAIN_IN))
    parser.add_argument("--test-in", default=str(DEFAULT_TEST_IN))
    parser.add_argument("--train-out", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--test-out", default=str(DEFAULT_TEST_OUT))
    parser.add_argument("--ready-out", default=str(DEFAULT_READY_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = extract_split(Path(args.train_in).resolve(), Path(args.train_out).resolve())
    test = extract_split(Path(args.test_in).resolve(), Path(args.test_out).resolve())
    write_ready(args, train, test)
    print(json.dumps({"train": train, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
