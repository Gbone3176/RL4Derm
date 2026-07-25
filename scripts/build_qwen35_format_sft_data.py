#!/usr/bin/env python3
"""Build Qwen3.5 MCQA format-calibration SFT JSON from DermoInstruct MCQA data."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

STANDARD_PROMPT_SUFFIX = """Your task:
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
3. No extra information or text outside of these tags."""

FORMAT_THINK = "简短浅思考，说明比较图像证据与选项，并准备选择最匹配选项。"


def _extract_letter(answer: str) -> str:
    match = re.match(r"\s*([A-Za-z])\s*(?:[\).:-]|$)", str(answer))
    if not match:
        raise ValueError(f"cannot extract single-letter answer from {answer!r}")
    return match.group(1).upper()


def _human_and_gpt(conversations: list[dict[str, Any]]) -> tuple[str, str]:
    human = next((m for m in conversations if m.get("from") == "human"), None)
    gpt = next((m for m in conversations if m.get("from") == "gpt"), None)
    if not human or not gpt:
        raise ValueError("sample must contain human and gpt messages")
    return str(human.get("value", "")).rstrip(), str(gpt.get("value", ""))


def build_record(sample: dict[str, Any]) -> dict[str, Any]:
    human_text, gpt_text = _human_and_gpt(sample.get("conversations") or [])
    letter = _extract_letter(gpt_text)
    record = {k: v for k, v in sample.items() if k != "conversations"}
    record["conversations"] = [
        {"from": "human", "value": f"{human_text}\n\n{STANDARD_PROMPT_SUFFIX}"},
        {"from": "gpt", "value": f"<think>{FORMAT_THINK}</think>\n<answer>{letter}</answer>"},
    ]
    return record


def append_manifest(manifest_path: Path, source: Path, output: Path, rows: int) -> None:
    entry = f"""

## {output.name}
- Created: {datetime.now().isoformat(timespec='seconds')}
- Source: `{source}`
- Rows: {rows}
- Purpose: Qwen3.5-4B MCQA format-calibration SFT data before GRPO.
- User content: original DermoInstruct MCQA question/options plus the standard prompt suffix, byte-preserved in `scripts/build_qwen35_format_sft_data.py`.
- Assistant content: strict `<think>...</think>\n<answer>X</answer>` with a short format-calibration rationale and a single-letter answer.
- Constraints: no system role, preserves `image` and original sample metadata outside `conversations`, derived under `data/` only.
"""
    prior = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else "# Data Manifest\n"
    if f"## {output.name}" not in prior:
        manifest_path.write_text(prior.rstrip() + entry + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/dermoinstruct_mcqa_train_10k.json")
    parser.add_argument("--output", default="data/qwen35_mcqa_format_sft_train_10k.json")
    parser.add_argument("--manifest", default="data/MANIFEST.md")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    data = json.loads(source.read_text(encoding="utf-8"))
    built = [build_record(sample) for sample in data]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(built, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_manifest(Path(args.manifest), source, output, len(built))
    print(json.dumps({"input": str(source), "output": str(output), "rows": len(built)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
