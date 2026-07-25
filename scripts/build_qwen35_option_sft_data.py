#!/usr/bin/env python3
"""Build option-only Qwen3.5 MCQA SFT JSON from DermoInstruct MCQA data."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PROMPT_SUFFIX = "Answer with only the single correct option letter."
LETTER_RE = re.compile(r"^\s*([A-Da-d])\s*(?:[\).:-]|$)")
STRICT_LETTER_RE = re.compile(r"^[A-D]$")


def extract_letter(answer: Any) -> str:
    match = LETTER_RE.match(str(answer))
    if not match:
        raise ValueError(f"cannot extract A-D answer from {answer!r}")
    return match.group(1).upper()


def human_and_gpt(sample: dict[str, Any]) -> tuple[str, str]:
    conversations = sample.get("conversations") or []
    human = next((msg for msg in conversations if msg.get("from") == "human"), None)
    gpt = next((msg for msg in conversations if msg.get("from") == "gpt"), None)
    if not human or not gpt:
        raise ValueError(f"sample {sample.get('id', '<no id>')} must contain human and gpt messages")
    return str(human.get("value", "")).rstrip(), str(gpt.get("value", ""))


def build_record(sample: dict[str, Any]) -> dict[str, Any]:
    human_text, gpt_text = human_and_gpt(sample)
    letter = extract_letter(gpt_text)
    record = {key: value for key, value in sample.items() if key != "conversations"}
    record["conversations"] = [
        {"from": "human", "value": f"{human_text}\n\n{PROMPT_SUFFIX}"},
        {"from": "gpt", "value": letter},
    ]
    return record


def validate(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {letter: 0 for letter in "ABCD"}
    bad: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        gpt = next((msg for msg in record.get("conversations", []) if msg.get("from") == "gpt"), {})
        value = gpt.get("value")
        if not isinstance(value, str) or not STRICT_LETTER_RE.fullmatch(value):
            bad.append({"index": index, "id": record.get("id"), "gpt": value})
        else:
            counts[value] += 1
    return {"rows": len(records), "bad_gpt_rows": bad, "letter_counts": counts}


def append_manifest(manifest_path: Path, source: Path, output: Path, validation: dict[str, Any]) -> None:
    entry = f"""

## {output.name}
- Created: {datetime.now().isoformat(timespec='seconds')}
- Source: `{source}`
- Rows: {validation['rows']}
- Purpose: Qwen3.5-4B dense LoRA SFT MCQA clean-base control; option-only target for direct answer training.
- User content: original DermoInstruct MCQA question/options plus only `{PROMPT_SUFFIX}`.
- Assistant content: one uppercase option letter only, matching `^[A-D]$`; no tags, explanation, whitespace, newline, or punctuation.
- Letter counts: A={validation['letter_counts']['A']}, B={validation['letter_counts']['B']}, C={validation['letter_counts']['C']}, D={validation['letter_counts']['D']}.
- Constraints: preserves `image` and original sample metadata outside `conversations`, derived under `data/` only, does not touch `dataset_final/benchmark/`.
"""
    prior = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else "# data manifest\n"
    if f"## {output.name}" not in prior:
        manifest_path.write_text(prior.rstrip() + entry + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/dermoinstruct_mcqa_train_10k.json")
    parser.add_argument("--output", default="data/qwen35_mcqa_option_sft_train_10k.json")
    parser.add_argument("--manifest", default="data/MANIFEST.md")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    data = json.loads(source.read_text(encoding="utf-8"))
    records = [build_record(sample) for sample in data]
    validation = validate(records)
    if validation["bad_gpt_rows"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_manifest(Path(args.manifest), source, output, validation)
    print(json.dumps({"input": str(source), "output": str(output), **validation}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
