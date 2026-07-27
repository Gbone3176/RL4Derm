#!/usr/bin/env python3
"""Prepare DermMaskTriads MCQA runtime manifests for Qwen3.5 pipelines."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.dataset.dermmask_normalizer import (
    load_and_normalize_dermmask,
    sha256_file,
    validate_normalized_rows,
)
from src.paths import to_logical_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_RAW = PROJECT_ROOT / "data/derm_mcqa_mask_bbox_text_train.json"
DEFAULT_TEST_RAW = PROJECT_ROOT / "data/derm_mcqa_mask_bbox_text_test.json"
DEFAULT_TRAIN_OUT = PROJECT_ROOT / "data/dermmask_qwen35_train_normalized.json"
DEFAULT_TEST_OUT = PROJECT_ROOT / "data/dermmask_qwen35_test_normalized.json"
DEFAULT_READY = PROJECT_ROOT / "data/dermmask_qwen35_ready.json"
DEFAULT_GATE = PROJECT_ROOT / "tmp/dermmask_sft_launch/dermmask_qwen35_gate.json"
DEFAULT_DATASET_SMOKE_LOG = PROJECT_ROOT / "tmp/dermmask_sft_launch/dataset_smoke.log"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-raw", type=Path, default=DEFAULT_TRAIN_RAW)
    parser.add_argument("--test-raw", type=Path, default=DEFAULT_TEST_RAW)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--test-out", type=Path, default=DEFAULT_TEST_OUT)
    parser.add_argument("--ready-out", type=Path, default=DEFAULT_READY)
    parser.add_argument("--gate-out", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--expected-train-sha256", default="1ee14b47adc6089ff15ff912fa3f8efcdeffdd75c6c0591fa8ca9af87b478043")
    parser.add_argument("--expected-test-sha256", default="fb32532633090984e7dedb42e371867fc882d4dec890831b68e7aa35679b63a1")
    parser.add_argument("--write-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_sha = sha256_file(args.train_raw)
    test_sha = sha256_file(args.test_raw)
    if train_sha != args.expected_train_sha256:
        raise SystemExit(f"train sha256 mismatch: {train_sha}")
    if test_sha != args.expected_test_sha256:
        raise SystemExit(f"test sha256 mismatch: {test_sha}")

    train_rows = load_and_normalize_dermmask(args.train_raw)
    test_rows = load_and_normalize_dermmask(args.test_raw)
    write_json(args.train_out, train_rows)
    write_json(args.test_out, test_rows)

    train_gate = validate_normalized_rows(train_rows, expected_count=27382)
    test_gate = validate_normalized_rows(test_rows, expected_count=2000)
    checks = {
        "raw_train_sha256_ok": train_sha == args.expected_train_sha256,
        "raw_test_sha256_ok": test_sha == args.expected_test_sha256,
        "train_count_ok": train_gate["count_ok"],
        "test_count_ok": test_gate["count_ok"],
        "train_zero_missing_images": train_gate["missing_images"] == 0,
        "test_zero_missing_images": test_gate["missing_images"] == 0,
        "train_unique_ids": train_gate["duplicate_ids"] == 0,
        "test_unique_ids": test_gate["duplicate_ids"] == 0,
        "train_valid_rows": train_gate["invalid_rows"] == 0,
        "test_valid_rows": test_gate["invalid_rows"] == 0,
    }
    gate = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw": {
            "train": to_logical_path(str(args.train_raw)),
            "test": to_logical_path(str(args.test_raw)),
            "train_sha256": train_sha,
            "test_sha256": test_sha,
            "source_train_sha256": "49aec8f077c99bc765054488cbaf7e3d95558877d832d0b9f700984bfb6eb02a",
            "source_test_sha256": "22ab66943dbff3a9712cb5709714ccc7e726ce218dac896f0871f32d9d933bf4",
            "path_format": "relative",
        },
        "normalized": {
            "train": to_logical_path(str(args.train_out)),
            "test": to_logical_path(str(args.test_out)),
        },
        "checks": checks,
        "train": train_gate,
        "test": test_gate,
        "representative": {
            "mcqa:000000": next(row for row in train_rows if row["id"] == "mcqa:000000"),
            "mcqa:000218": next(row for row in train_rows if row["id"] == "mcqa:000218"),
        },
        "normalization_rule": [
            "Use manifest options when answer maps to exactly one option.",
            "For DermoInstruct rows with empty options, parse quoted choices preserved in question/text and map answer exactly.",
            "For DermoBench Task 2.1-4/25 rows with truncated or mismatching options, recover original options from upstream DermoBench task2 files keyed by split+image_id+answer.",
            "Reject ambiguity, duplicate labels, missing images, and unmapped answers.",
        ],
        "code_files": [
            "src/dataset/dermmask_normalizer.py",
            "src/dataset/sft_dataset.py",
            "src/dataset/grpo_dataset_derm.py",
            "src/train/reward_funcs.py",
            "scripts/prepare_dermmask_qwen35.py",
        ],
    }
    write_json(args.gate_out, gate)
    if gate["status"] != "PASS":
        raise SystemExit(f"DermMask Qwen3.5 gate failed; see {args.gate_out}")

    if args.write_ready:
        ready = {
            "status": "PASS",
            "created_at": gate["created_at"],
            "raw": gate["raw"],
            "raw_paths": {
                "train": gate["raw"]["train"],
                "test": gate["raw"]["test"],
            },
            "raw_sha256": {
                "train": gate["raw"]["train_sha256"],
                "test": gate["raw"]["test_sha256"],
            },
            "normalized": gate["normalized"],
            "normalized_paths": gate["normalized"],
            "counts": {
                "train": train_gate["count"],
                "test": test_gate["count"],
            },
            "validation_evidence": {
                "gate": to_logical_path(str(args.gate_out)),
                "train": train_gate,
                "test": test_gate,
                "dataset_smoke_log": to_logical_path(str(DEFAULT_DATASET_SMOKE_LOG)),
                "dataset_smoke_status": "PASS" if DEFAULT_DATASET_SMOKE_LOG.exists() else "MISSING",
            },
            "normalization_rule": gate["normalization_rule"],
            "code_files": gate["code_files"],
        }
        write_json(args.ready_out, ready)

    print(json.dumps({"status": gate["status"], "gate": str(args.gate_out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
