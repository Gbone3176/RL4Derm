"""DermMaskTriads MCQA runtime normalization.

The copied DermMaskTriads manifests are preserved as provenance records. This
module converts them into the repository's existing LLaVA/Qwen-VL MCQA schema
only when loading or generating derived runtime manifests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from paths import configured_dataset_path, resolve_dataset_path, to_logical_path
except ImportError:  # pragma: no cover - supports src.* imports.
    from src.paths import configured_dataset_path, resolve_dataset_path, to_logical_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERMOBENCH_ROOT = configured_dataset_path("dermobench_root")
DERMOBENCH_TASK_FILES = {
    "Task 2.1-4": DERMOBENCH_ROOT / "task2/2_1_mcq/4_choices/task2.1_test_2k_non_uniform_sample_final.json",
    "Task 2.1-25": DERMOBENCH_ROOT / "task2/2_1_mcq/25_choices/task2.1_25choices_test_2k_non_uniform_sample_final.json",
}


class DermMaskNormalizationError(ValueError):
    """Raised when a DermMask row cannot be normalized without ambiguity."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_dermmask_manifest(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("samples"), list)
        and isinstance(payload.get("metadata"), dict)
        and payload.get("metadata", {}).get("task_type") == "mcqa"
    )


def _normalize_text(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"^[a-z]\s*[\).:-]\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .;:")


def _lettered_options(text: Any) -> list[dict[str, str]]:
    value = str(text or "").replace("<image>", "").strip()
    matches = list(re.finditer(r"(?:^|\n|\s)([A-Z])\)\s*", value))
    options: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        option_text = re.sub(r"\s+", " ", value[start:end]).strip()
        if option_text:
            options.append({"label": match.group(1), "text": option_text})
    return options if len(options) >= 2 else []


def _quoted_options(text: Any) -> list[dict[str, str]]:
    values = [item.strip() for item in re.findall(r'"([^"]+)"', str(text or "")) if item.strip()]
    if len(values) < 2:
        return []
    normalized = [_normalize_text(item) for item in values]
    if len(set(normalized)) != len(normalized):
        return []
    return [{"label": chr(ord("A") + index), "text": value} for index, value in enumerate(values)]


def _option_rows(options: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(options, list):
        for index, item in enumerate(options):
            if isinstance(item, dict):
                label = str(item.get("label") or chr(ord("A") + index)).strip().upper()
                text = str(item.get("text") or "").strip()
            else:
                label = chr(ord("A") + index)
                text = str(item).strip()
            if label and text:
                rows.append({"label": label, "text": text})
    elif isinstance(options, dict):
        for label, text in options.items():
            rows.append({"label": str(label).strip().upper(), "text": str(text).strip()})
    return rows


def _answer_match(answer: Any, options: list[dict[str, str]]) -> tuple[str, str, str]:
    answer_text = str(answer or "").strip()
    explicit = re.match(r"^([A-Za-z])\s*[\).:-]?\s*(.*?)\s*$", answer_text)
    if explicit:
        label = explicit.group(1).upper()
        rest = explicit.group(2).strip()
        hits = [row for row in options if row["label"].upper() == label]
        if len(hits) == 1 and (
            not rest
            or _normalize_text(rest) == _normalize_text(hits[0]["text"])
            or _normalize_text(answer_text) == _normalize_text(hits[0]["text"])
        ):
            return hits[0]["label"], hits[0]["text"], "explicit_label_prefix"

    hits = [row for row in options if _normalize_text(answer_text) == _normalize_text(row["text"])]
    if len(hits) == 1:
        return hits[0]["label"], hits[0]["text"], "answer_text_exact_match"
    if len(hits) > 1:
        raise DermMaskNormalizationError(f"ambiguous answer={answer_text!r} matches {len(hits)} options")
    raise DermMaskNormalizationError(f"answer={answer_text!r} does not match options")


def _conversation_qa(record: dict[str, Any]) -> tuple[str, str]:
    question = ""
    answer = ""
    for turn in record.get("conversations", []):
        role = str(turn.get("from") or turn.get("role") or "").lower()
        value = str(turn.get("value") or turn.get("content") or "")
        if role in {"human", "user"} and not question:
            question = value
        elif role in {"gpt", "assistant", "model"} and not answer:
            answer = value
    return question, answer


def _image_id_from_record(record: dict[str, Any]) -> str | None:
    match = re.search(r"ISIC_\d+", json.dumps(record, ensure_ascii=False))
    return match.group(0) if match else None


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        rows = payload.get("samples") or payload.get("data") or payload.get("records")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise DermMaskNormalizationError(f"Unsupported JSON rows in {path}")
    return rows


def _load_dermobench_lookup() -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    collisions: set[tuple[str, str, str]] = set()
    for split, path in DERMOBENCH_TASK_FILES.items():
        for record in _load_json_rows(path):
            image_id = _image_id_from_record(record)
            question, answer = _conversation_qa(record)
            options = _lettered_options(question)
            if not image_id or not answer or not options:
                continue
            label, text, rule = _answer_match(answer, options)
            key = (split, image_id, _normalize_text(answer))
            value = {
                "question": question.replace("<image>", "").strip(),
                "answer": answer,
                "options": options,
                "label": label,
                "text": text,
                "rule": f"dermobench_original_{rule}",
                "source_file": to_logical_path(str(path)),
                "source_id": record.get("id"),
            }
            if key in lookup and lookup[key] != value:
                collisions.add(key)
            lookup[key] = value
    if collisions:
        raise DermMaskNormalizationError(f"DermoBench lookup collisions: {sorted(collisions)[:5]}")
    return lookup


_DERMOBENCH_LOOKUP: dict[tuple[str, str, str], dict[str, Any]] | None = None


def dermobench_lookup() -> dict[tuple[str, str, str], dict[str, Any]]:
    global _DERMOBENCH_LOOKUP
    if _DERMOBENCH_LOOKUP is None:
        _DERMOBENCH_LOOKUP = _load_dermobench_lookup()
    return _DERMOBENCH_LOOKUP


def _resolve_sample_options(sample: dict[str, Any]) -> tuple[list[dict[str, str]], str, str, str, dict[str, Any]]:
    manifest_options = _option_rows(sample.get("options"))
    answer = sample.get("answer")
    provenance: dict[str, Any] = {}
    if manifest_options:
        try:
            label, text, rule = _answer_match(answer, manifest_options)
            return manifest_options, label, text, f"manifest_options_{rule}", provenance
        except DermMaskNormalizationError:
            pass

    source = sample.get("source_text_dataset")
    if source == "DermoInstruct":
        parsed_options = _quoted_options(sample.get("question")) or _quoted_options(sample.get("text"))
        if not parsed_options:
            parsed_options = _lettered_options(sample.get("question")) or _lettered_options(sample.get("text"))
        label, text, rule = _answer_match(answer, parsed_options)
        return parsed_options, label, text, f"dermoinstruct_preserved_text_{rule}", provenance

    if source == "DermoBench":
        key = (str(sample.get("split")), str(sample.get("image_id")), _normalize_text(answer))
        record = dermobench_lookup().get(key)
        if record is None:
            raise DermMaskNormalizationError(f"no DermoBench upstream match for key={key!r}")
        provenance.update(
            upstream_options_source_file=record["source_file"],
            upstream_options_source_id=record["source_id"],
            upstream_answer=record["answer"],
        )
        return record["options"], record["label"], record["text"], record["rule"], provenance

    raise DermMaskNormalizationError(f"unsupported source_text_dataset={source!r}")


def normalize_dermmask_sample(sample: dict[str, Any]) -> dict[str, Any]:
    options, label, answer_text, rule, extra_provenance = _resolve_sample_options(sample)
    if len(options) < 2:
        raise DermMaskNormalizationError(f"sample {sample.get('sample_id')} has fewer than two options")
    labels = [row["label"] for row in options]
    if len(set(labels)) != len(labels):
        raise DermMaskNormalizationError(f"sample {sample.get('sample_id')} has duplicate option labels")
    if label not in labels:
        raise DermMaskNormalizationError(f"sample {sample.get('sample_id')} target label not in options")
    image_path = str(sample.get("image_path") or "")
    if not image_path:
        raise DermMaskNormalizationError(f"sample {sample.get('sample_id')} missing image_path")
    resolved_image_path = resolve_dataset_path(image_path)
    if not resolved_image_path.exists():
        raise DermMaskNormalizationError(f"sample {sample.get('sample_id')} missing image: {image_path}")
    logical_image_path = to_logical_path(image_path)

    normalized = {
        "id": str(sample.get("sample_id") or sample.get("triad_id")),
        "image": logical_image_path,
        "path_format": "relative",
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{str(sample.get('question') or '').strip()}",
            },
            {
                "from": "gpt",
                "value": label,
            },
        ],
        "options": options,
        "gold_label": label,
        "gold_text": answer_text,
        "answer": label,
        "ground_truth": {
            "task": "task2_mcqa",
            "modality": "dermmask_mcqa_bbox_mask_text",
            "answer": label,
            "answer_letter": label,
            "answer_text": answer_text,
            "source_answer": sample.get("answer"),
            "options": options,
            "normalization_rule": rule,
        },
        "provenance": {
            "dataset": "DermMaskTriads",
            "source_sample_id": sample.get("sample_id"),
            "source_triad_id": sample.get("source_triad_id") or sample.get("triad_id"),
            "source_text_dataset": sample.get("source_text_dataset"),
            "source_mask_dataset": sample.get("source_mask_dataset"),
            "source_split": sample.get("split"),
            "dataset_split": sample.get("dataset_split"),
            "image_id": sample.get("image_id"),
            "mask_path": to_logical_path(sample.get("mask_path")),
            "bbox_xyxy": sample.get("bbox_xyxy"),
            "bbox_xywh": sample.get("bbox_xywh"),
            "mcqa_sample_index": sample.get("mcqa_sample_index"),
            "normalization_rule": rule,
            **extra_provenance,
        },
    }
    return normalized


def normalize_dermmask_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not is_dermmask_manifest(payload):
        raise DermMaskNormalizationError("payload is not a DermMaskTriads MCQA manifest")
    normalized = []
    errors = []
    for index, sample in enumerate(payload["samples"]):
        try:
            normalized.append(normalize_dermmask_sample(sample))
        except Exception as exc:
            errors.append({"index": index, "sample_id": sample.get("sample_id"), "error": f"{type(exc).__name__}: {exc}"})
    if errors:
        raise DermMaskNormalizationError(f"{len(errors)} DermMask rows failed normalization; first={errors[0]}")
    return normalized


def load_and_normalize_dermmask(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if is_dermmask_manifest(payload):
        return normalize_dermmask_payload(payload)
    if isinstance(payload, list):
        return payload
    raise DermMaskNormalizationError(f"Unsupported dataset payload: {path}")


def validate_normalized_rows(rows: list[dict[str, Any]], expected_count: int | None = None) -> dict[str, Any]:
    ids = [row.get("id") for row in rows]
    duplicate_ids = len(ids) - len(set(ids))
    missing_images = [
        row.get("id")
        for row in rows
        if not resolve_dataset_path(str(row.get("image") or "")).exists()
    ]
    invalid = []
    rule_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    max_label = "A"
    for row in rows:
        options = row.get("options") or []
        label = row.get("gold_label")
        rule_counts[row.get("ground_truth", {}).get("normalization_rule", "unknown")] += 1
        source_counts[row.get("provenance", {}).get("source_text_dataset", "unknown")] += 1
        if label:
            max_label = max(max_label, str(label))
        if len(options) < 2:
            invalid.append({"id": row.get("id"), "error": "options_lt_2"})
        elif sum(1 for item in options if item.get("label") == label) != 1:
            invalid.append({"id": row.get("id"), "error": "target_label_not_unique"})
        if not row.get("conversations") or len(row["conversations"]) != 2:
            invalid.append({"id": row.get("id"), "error": "invalid_conversations"})
    return {
        "count": len(rows),
        "expected_count": expected_count,
        "count_ok": expected_count is None or len(rows) == expected_count,
        "duplicate_ids": duplicate_ids,
        "missing_images": len(missing_images),
        "invalid_rows": len(invalid),
        "max_label": max_label,
        "rule_counts": dict(rule_counts),
        "source_counts": dict(source_counts),
        "examples_invalid": invalid[:10],
        "examples_missing_images": missing_images[:10],
    }
