#!/usr/bin/env python3
"""Evaluate a Qwen3.5 dense LoRA adapter on closed-set VQA MCQA data."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.dataset.data_utils import get_image_info, llava_to_openai
from src.train.qwen35_dense_utils import (
    apply_qwen35_no_default_thinking_prefill,
    git_commit,
    patch_qwen35_processor_apply_chat_template,
)
from src.utils import disable_torch_init, get_model_name_from_path, load_pretrained_model


PROMPT_SUFFIX = "Answer with only the single correct option letter."
ASSISTANT_HEADER = "<|im_start|>assistant\n"
CLOSED_THINK_PREFILL = "<think>\n\n</think>\n\n"
GOLD_PREFIX_RE = re.compile(r"^\s*([A-Da-d])\s*(?:\)|$)")
PRED_PREFIX_RE = re.compile(r"^\s*([A-Da-d])\s*(?:[\).:\-]|$)")
STRICT_PRED_RE = re.compile(r"^[A-D]$")
STANDALONE_PRED_RE = re.compile(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])")
RISK_PATTERNS = (
    "checkpoint seem corrupted",
    "checkpoint seems corrupted",
    "newly initialized",
    "were not initialized",
    "missing key",
    "missing keys",
    "weights missing",
    "corrupted",
)
ADAPTER_REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")
NON_LORA_NAME = "non_lora_state_dict.bin"
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
MIN_FINAL_NON_LORA_SIZE_RATIO = 0.8


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def shell_json(command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, check=False, text=True, capture_output=True)
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def has_adapter_files(path: Path) -> bool:
    return all((path / filename).exists() for filename in ADAPTER_REQUIRED_FILES)


def non_lora_bytes(path: Path) -> int | None:
    non_lora = path / NON_LORA_NAME
    return non_lora.stat().st_size if non_lora.exists() else None


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else None


def adapter_candidate(path: Path) -> dict[str, Any]:
    size = non_lora_bytes(path)
    return {
        "path": str(path),
        "name": path.name,
        "step": checkpoint_step(path),
        "has_adapter_files": has_adapter_files(path),
        "non_lora_exists": size is not None,
        "non_lora_bytes": size,
    }


def numeric_checkpoint_candidates(adapter_run: Path) -> list[Path]:
    checkpoints = [
        path
        for path in adapter_run.iterdir()
        if path.is_dir() and checkpoint_step(path) is not None and has_adapter_files(path)
    ]
    return sorted(checkpoints, key=lambda path: checkpoint_step(path) or -1)


def resolve_adapter(adapter_run: Path) -> tuple[Path | None, str, dict[str, Any]]:
    resolution: dict[str, Any] = {
        "requested_adapter_run": str(adapter_run),
        "selected_adapter_path": None,
        "adapter_source": "missing",
        "selection_reason": None,
        "direct_checkpoint": checkpoint_step(adapter_run) is not None,
        "final": adapter_candidate(adapter_run),
        "checkpoint_candidates": [],
        "size_ratio_rule": {
            "min_final_to_checkpoint_ratio": MIN_FINAL_NON_LORA_SIZE_RATIO,
            "note": "Qwen3.5 dense LoRA loader still enforces normalized non_lora key compatibility.",
        },
    }

    if resolution["direct_checkpoint"]:
        if has_adapter_files(adapter_run):
            resolution.update(
                {
                    "selected_adapter_path": str(adapter_run),
                    "adapter_source": "final",
                    "selection_reason": "adapter_run is already a checkpoint-* directory; preserving direct checkpoint behavior",
                }
            )
            return adapter_run, "final", resolution
        resolution["selection_reason"] = "direct checkpoint path is missing adapter files"
        return None, "missing", resolution

    checkpoints = numeric_checkpoint_candidates(adapter_run) if adapter_run.exists() else []
    resolution["checkpoint_candidates"] = [adapter_candidate(path) for path in checkpoints]
    latest_checkpoint = checkpoints[-1] if checkpoints else None
    latest_checkpoint_info = adapter_candidate(latest_checkpoint) if latest_checkpoint else None

    final_has_adapter = has_adapter_files(adapter_run)
    final_size = non_lora_bytes(adapter_run)
    checkpoint_size = latest_checkpoint_info["non_lora_bytes"] if latest_checkpoint_info else None

    if latest_checkpoint is not None:
        if not final_has_adapter:
            source = latest_checkpoint.name
            reason = "run root has no final adapter files; selected latest numeric checkpoint"
            resolution.update({"selected_adapter_path": str(latest_checkpoint), "adapter_source": source, "selection_reason": reason})
            return latest_checkpoint, source, resolution
        if final_size is None:
            source = latest_checkpoint.name
            reason = "run root final adapter is missing non_lora_state_dict.bin; selected latest numeric checkpoint"
            resolution.update({"selected_adapter_path": str(latest_checkpoint), "adapter_source": source, "selection_reason": reason})
            return latest_checkpoint, source, resolution
        if checkpoint_size is not None and final_size < int(checkpoint_size * MIN_FINAL_NON_LORA_SIZE_RATIO):
            source = latest_checkpoint.name
            reason = (
                "run root final non_lora_state_dict.bin is much smaller than latest checkpoint "
                f"({final_size} < {MIN_FINAL_NON_LORA_SIZE_RATIO:.0%} of {checkpoint_size}); "
                "selected latest numeric checkpoint"
            )
            resolution.update({"selected_adapter_path": str(latest_checkpoint), "adapter_source": source, "selection_reason": reason})
            return latest_checkpoint, source, resolution

    if final_has_adapter:
        resolution.update(
            {
                "selected_adapter_path": str(adapter_run),
                "adapter_source": "final",
                "selection_reason": "run root final adapter passed resolver completeness heuristics",
            }
        )
        return adapter_run, "final", resolution

    resolution["selection_reason"] = "no adapter files found in run root or numeric checkpoints"
    return None, "missing", resolution


def first_turn(sample: dict[str, Any], role: str) -> str | None:
    for turn in sample.get("conversations") or []:
        if turn.get("from") == role:
            return str(turn.get("value", ""))
    return None


def parse_gold(text: str | None) -> str | None:
    if text is None:
        return None
    match = GOLD_PREFIX_RE.match(text)
    return match.group(1).upper() if match else None


def parse_prediction(raw_output: str) -> tuple[str | None, str]:
    stripped = raw_output.strip()
    upper = stripped.upper()
    if STRICT_PRED_RE.fullmatch(upper):
        return upper, "strict_single_letter"

    prefix = PRED_PREFIX_RE.match(stripped)
    if prefix:
        return prefix.group(1).upper(), "option_prefix"

    standalone = [match.group(1).upper() for match in STANDALONE_PRED_RE.finditer(stripped)]
    unique = sorted(set(standalone))
    if len(unique) == 1:
        return unique[0], "single_standalone_letter"
    if len(unique) > 1:
        return None, "multiple_candidate_letters"
    return None, "no_parse"


def build_user_text(sample: dict[str, Any]) -> str:
    human = first_turn(sample, "human")
    if human is None:
        raise ValueError(f"sample {sample.get('id', '<no id>')} has no human turn")
    question = human.replace("<image>", "").strip()
    return f"{question}\n\n{PROMPT_SUFFIX}"


def with_prompt_suffix(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith(PROMPT_SUFFIX):
        return stripped
    return f"{stripped}\n\n{PROMPT_SUFFIX}"


def build_sft_manual_prompt(sample: dict[str, Any], *, append_suffix: bool) -> str:
    conversations = sample.get("conversations") or []
    if len(conversations) < 2:
        raise ValueError(f"sample {sample.get('id', '<no id>')} has fewer than two conversation turns")
    human_turn = dict(conversations[0])
    if human_turn.get("from") != "human":
        raise ValueError(f"sample {sample.get('id', '<no id>')} first turn is not human")
    if append_suffix:
        human_turn["value"] = with_prompt_suffix(str(human_turn.get("value", "")))
    transformed = llava_to_openai([human_turn, conversations[1]], is_video=False)
    user_input = transformed[0]
    gpt_response = transformed[1]
    return (
        f"<|im_start|>{user_input['role']}\n"
        f"{user_input['content']}<|im_end|>\n"
        f"<|im_start|>{gpt_response['role']}\n"
    )


def sft_template_checks(prompt: str) -> dict[str, Any]:
    return {
        "ends_with_assistant_header": prompt.endswith(ASSISTANT_HEADER),
        "has_no_system_role": "<|im_start|>system\n" not in prompt,
        "has_no_closed_think_prefill": CLOSED_THINK_PREFILL not in prompt,
        "has_image_pad_wrapped": "<|vision_start|><|image_pad|><|vision_end|>" in prompt,
        "matches_dataset_prefix_shape": prompt.startswith("<|im_start|>user\n"),
    }


def build_messages(image_path: Path, text: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                    "min_pixels": args.image_min_pixels,
                    "max_pixels": args.image_max_pixels,
                },
                {"type": "text", "text": text},
            ],
        }
    ]


def check_gpu_status() -> dict[str, Any]:
    gpu_query = shell_json(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    proc_query = shell_json(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if gpu_query.get("returncode") == 0:
        for line in gpu_query.get("stdout", "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                used = int(parts[2])
                total = int(parts[3])
                gpus.append(
                    {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mib": used,
                    "memory_total_mib": total,
                    "memory_free_mib": total - used,
                    "utilization_gpu_percent": int(parts[4]),
                    "meets_eval_memory_rule": (total - used) >= 35000,
                    "meets_eval_util_rule": int(parts[4]) < 20,
                    }
                )
    return {
        "availability_rule": "launch GPU must have >=35000 MiB free and utilization below 20% on two consecutive checks",
        "gpus": gpus,
        "gpu_query": gpu_query,
        "compute_apps_query": proc_query,
    }


def write_template_match_preflight(args: argparse.Namespace, test_data: list[dict[str, Any]]) -> dict[str, Any]:
    train_path = Path(args.train_data)
    train_data = read_json(train_path)
    train_samples = train_data[:3] if isinstance(train_data, list) else []
    test_samples = test_data[:3]

    def render_rows(samples: list[dict[str, Any]], *, split: str, append_suffix: bool) -> list[dict[str, Any]]:
        rows = []
        for index, sample in enumerate(samples):
            rendered = build_sft_manual_prompt(sample, append_suffix=append_suffix)
            checks = sft_template_checks(rendered)
            rows.append(
                {
                    "split": split,
                    "index": index,
                    "id": sample.get("id"),
                    "image": sample.get("image"),
                    "rendered_prompt": rendered,
                    "rendered_tail": rendered[-360:],
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
        return rows

    adapter_run_path = Path(args.adapter_run)
    run34_args_path = adapter_run_path / "args_snapshot.json"
    if not run34_args_path.exists() and adapter_run_path.name.startswith("checkpoint-"):
        run34_args_path = adapter_run_path.parent / "args_snapshot.json"
    run34_args = read_json(run34_args_path) if run34_args_path.exists() else {}
    run34_data_path = (run34_args.get("training_args") or {}).get("data_path") or (run34_args.get("args") or {}).get("data_path")
    expected_train_data = str(train_path)
    rows = render_rows(train_samples, split="train", append_suffix=False) + render_rows(
        test_samples,
        split="test",
        append_suffix=True,
    )
    checks = {
        "train_rows_sampled": len(train_samples) == 3,
        "test_rows_sampled": len(test_samples) == 3,
        "all_rendered_prompts_pass": all(row["passed"] for row in rows),
        "run34_train_data_matches": run34_data_path == expected_train_data,
        "source_structure": "src/dataset/sft_dataset.py:131-159 llava_to_openai + '<|im_start|>{role}\\n{content}<|im_end|>\\n<|im_start|>{role}\\n'",
        "prompt_mode": args.prompt_mode,
    }
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_data": expected_train_data,
        "run34_args_snapshot": str(run34_args_path),
        "run34_train_data_path": run34_data_path,
        "suffix": PROMPT_SUFFIX,
        "template": "<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n",
        "replacement": {"<image>": "<|vision_start|><|image_pad|><|vision_end|>"},
        "checks": checks,
        "samples": rows,
        "passed": bool(
            checks["train_rows_sampled"]
            and checks["test_rows_sampled"]
            and checks["all_rendered_prompts_pass"]
            and checks["run34_train_data_matches"]
        ),
    }
    path = Path(args.exp_dir) / "preflight" / "template_match.json"
    write_json(path, payload)
    write_json(Path(args.exp_dir) / "template_preflight.json", payload)
    return payload


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    exp_dir = Path(args.exp_dir)
    preflight_dir = exp_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    test_path = Path(args.test_json)
    image_root = Path(args.image_root)
    adapter_run = Path(args.adapter_run)
    base_model = Path(args.base_model)
    data = read_json(test_path)
    eval_data = data[: args.limit] if isinstance(data, list) and args.limit is not None else data
    rows_ok = isinstance(eval_data, list) and len(eval_data) == args.expected_total

    missing_images = []
    malformed_rows = []
    gold_counts = Counter()
    for index, item in enumerate(eval_data if isinstance(eval_data, list) else []):
        image = item.get("image")
        if not image or not (image_root / image).exists():
            missing_images.append({"index": index, "id": item.get("id"), "image": image})
        gold = parse_gold(first_turn(item, "gpt"))
        if gold is None:
            malformed_rows.append({"index": index, "id": item.get("id"), "gpt": first_turn(item, "gpt")})
        else:
            gold_counts[gold] += 1

    adapter_path, adapter_source, adapter_resolution = resolve_adapter(adapter_run)
    adapter_files = {
        "adapter_config": str(adapter_path / "adapter_config.json") if adapter_path else None,
        "adapter_model": str(adapter_path / "adapter_model.safetensors") if adapter_path else None,
        "non_lora_state_dict": str(adapter_path / "non_lora_state_dict.bin") if adapter_path and (adapter_path / "non_lora_state_dict.bin").exists() else None,
    }
    template_match = write_template_match_preflight(args, eval_data if isinstance(eval_data, list) else [])
    gpu = check_gpu_status()
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo": str(Path.cwd()),
        "test_json": str(test_path),
        "expected_total": args.expected_total,
        "actual_total": len(data) if isinstance(data, list) else None,
        "eval_total": len(eval_data) if isinstance(eval_data, list) else None,
        "limit": args.limit,
        "rows_ok": rows_ok,
        "image_root": str(image_root),
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:20],
        "gold_parse_fail_count": len(malformed_rows),
        "gold_parse_fail_sample": malformed_rows[:20],
        "gold_counts": dict(gold_counts),
        "base_model": str(base_model),
        "base_model_exists": base_model.exists(),
        "adapter_run": str(adapter_run),
        "adapter_source": adapter_source,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_selection_reason": adapter_resolution.get("selection_reason"),
        "adapter_resolution": adapter_resolution,
        "adapter_files": adapter_files,
        "adapter_ok": adapter_path is not None,
        "gpu": gpu,
        "template_match_path": str(preflight_dir / "template_match.json"),
        "template_match_passed": template_match["passed"],
        "passed": bool(
            rows_ok
            and len(missing_images) == 0
            and len(malformed_rows) == 0
            and base_model.exists()
            and adapter_path is not None
            and template_match["passed"]
        ),
    }
    write_json(preflight_dir / "preflight.json", result)
    write_json(preflight_dir / "gpu_preflight.json", gpu)
    return result


def write_chat_template_preflight(
    output_dir: Path,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    thinking_prefill: str,
    strategy: dict[str, Any] | None,
) -> None:
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assistant_start = "<|im_start|>assistant\n"
    closed_think = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    checks = {
        "has_no_system_role": not any(message.get("role") == "system" for message in messages),
        "has_generation_prompt": assistant_start in rendered,
        "mode_closed_prefill_has_empty_think": thinking_prefill != "closed" or closed_think in rendered,
        "mode_no_default_ends_with_assistant_header": thinking_prefill != "no_default" or rendered.endswith(assistant_start),
    }
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "thinking_prefill": thinking_prefill,
        "strategy": strategy,
        "prompt": messages,
        "rendered": rendered,
        "rendered_tail": rendered[-360:],
        "checks": checks,
        "passed": all(checks.values()),
    }
    path = output_dir / "preflight" / "chat_template_thinking_mode.json"
    write_json(path, payload)
    if not payload["passed"]:
        raise RuntimeError(f"chat template thinking-mode preflight failed; see {path}")
    print(f"chat_template_thinking_mode_preflight={path}", flush=True)
    print(f"chat_template_tail={rendered[-160:]!r}", flush=True)


def load_model(args: argparse.Namespace, adapter_path: Path):
    disable_torch_init()
    model_name = get_model_name_from_path(str(adapter_path))
    processor, model = load_pretrained_model(
        model_path=str(adapter_path),
        model_base=str(args.base_model),
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="cuda:0",
        device="cuda",
        use_flash_attn=not args.disable_flash_attention,
    )
    strategy = None
    if args.prompt_mode == "chat_template":
        patch_qwen35_processor_apply_chat_template(processor)
        if args.thinking_prefill == "no_default":
            strategy = apply_qwen35_no_default_thinking_prefill(processor)
    diagnostics = getattr(model, "_dermogpt_loader_diagnostics", None) or getattr(
        processor, "_dermogpt_loader_diagnostics", None
    )
    if diagnostics is not None:
        write_json(Path(args.exp_dir) / "loader_diagnostics.json", diagnostics)
    return processor, model.eval(), strategy


def im_end_token_id(tokenizer: Any) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if token_id is None or token_id == tokenizer.unk_token_id:
        encoded = tokenizer("<|im_end|>", add_special_tokens=False).get("input_ids", [])
        if len(encoded) == 1:
            token_id = encoded[0]
        else:
            token_id = None
    return token_id


def model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0")


def run_one(
    sample: dict[str, Any],
    *,
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    device: torch.device,
) -> dict[str, Any]:
    image_rel = sample.get("image")
    image_path = Path(args.image_root) / str(image_rel)
    text = build_user_text(sample)
    messages = build_messages(image_path, text, args)
    if args.prompt_mode == "sft_dataset_manual":
        prompt = build_sft_manual_prompt(sample, append_suffix=True)
        prompt_checks = sft_template_checks(prompt)
        if not all(prompt_checks.values()):
            raise RuntimeError(f"SFT manual prompt failed checks for {sample.get('id')}: {prompt_checks}")
    else:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    image = get_image_info(
        str(image_path),
        args.image_min_pixels,
        args.image_max_pixels,
        None,
        None,
        16,
    )
    inputs = processor(
        text=[prompt],
        images=[image],
        videos=None,
        padding=False if args.prompt_mode == "sft_dataset_manual" else True,
        do_resize=False,
        return_tensors="pt",
    ).to(device)
    generation_kwargs = {
        **inputs,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    generation_kwargs["pad_token_id"] = processor.tokenizer.pad_token_id
    end_token_id = im_end_token_id(processor.tokenizer)
    if end_token_id is not None:
        generation_kwargs["eos_token_id"] = end_token_id

    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    response_tokens = outputs[0][input_len:]
    raw_output = processor.tokenizer.decode(
        response_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()
    gold_raw = first_turn(sample, "gpt") or ""
    gold_letter = parse_gold(gold_raw)
    parse_text = raw_output.replace("<|im_end|>", "").strip()
    pred_letter, parse_status = parse_prediction(parse_text)
    return {
        "id": sample.get("id"),
        "image": image_rel,
        "gold_raw": gold_raw,
        "gold_letter": gold_letter,
        "raw_output": raw_output,
        "pred_letter": pred_letter,
        "correct": bool(gold_letter is not None and pred_letter == gold_letter),
        "parse_status": parse_status,
    }


def summarize(predictions_path: Path) -> dict[str, Any]:
    rows = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    invalid = sum(1 for row in rows if not row.get("pred_letter"))
    strict = sum(1 for row in rows if row.get("parse_status") == "strict_single_letter")
    parse_status = Counter(row.get("parse_status") for row in rows)
    per_gold_pred: dict[str, dict[str, int]] = {gold: {} for gold in "ABCD"}
    for row in rows:
        gold = row.get("gold_letter") or "NA"
        pred = row.get("pred_letter") or "INVALID"
        per_gold_pred.setdefault(gold, {})
        per_gold_pred[gold][pred] = per_gold_pred[gold].get(pred, 0) + 1

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "correct": correct,
        "acc": correct / total if total else None,
        "invalid": invalid,
        "strict_single_letter_rate": strict / total if total else None,
        "parse_status_counts": dict(parse_status),
        "confusion_matrix": {"per_gold_pred": per_gold_pred},
    }


def write_args_snapshot(
    args: argparse.Namespace,
    adapter_path: Path,
    adapter_source: str,
    adapter_resolution: dict[str, Any] | None = None,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "src/eval/qwen35_lora_closed_vqa_acc.py",
        "git_commit": git_commit(),
        "args": vars(args),
        "adapter_path": str(adapter_path),
        "adapter_source": adapter_source,
        "adapter_selection_reason": (adapter_resolution or {}).get("selection_reason"),
        "adapter_resolution": adapter_resolution,
        "prompt_suffix": PROMPT_SUFFIX,
        "prompt_mode": args.prompt_mode,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "system_prompt": None,
            "thinking_prefill": args.thinking_prefill,
            "eos_token": "<|im_end|>",
        },
        "parse_policy": {
            "gold": "start A-D followed by ')' or single letter",
            "prediction": "strict single A-D first; otherwise option prefix; otherwise exactly one standalone A-D",
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "MODELSCOPE_CACHE": os.environ.get("MODELSCOPE_CACHE"),
        },
    }
    write_json(Path(args.exp_dir) / "eval_args_snapshot.json", payload)


def run_eval(args: argparse.Namespace) -> int:
    exp_dir = Path(args.exp_dir)
    predictions_path = exp_dir / "predictions.jsonl"
    metrics_path = exp_dir / "metrics.json"
    status_path = exp_dir / "eval_status.json"
    exp_dir.mkdir(parents=True, exist_ok=True)

    preflight_result = read_json(exp_dir / "preflight" / "preflight.json")
    if not preflight_result.get("passed"):
        write_json(status_path, {"status": "FAIL", "reason": "preflight_failed", "preflight": preflight_result})
        return 2

    adapter_path = Path(preflight_result["adapter_path"])
    adapter_source = preflight_result["adapter_source"]
    write_args_snapshot(args, adapter_path, adapter_source, preflight_result.get("adapter_resolution"))

    data = read_json(Path(args.test_json))
    if args.limit is not None:
        data = data[: args.limit]
    if predictions_path.exists() and not args.resume:
        predictions_path.unlink()

    processed = set()
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    processed.add(json.loads(line).get("id"))

    print(f"adapter_source={adapter_source} adapter_path={adapter_path}", flush=True)
    processor, model, no_think_strategy = load_model(args, adapter_path)
    device = model_device(model)
    print(f"model_device={device}", flush=True)
    end_token_id = im_end_token_id(processor.tokenizer)
    print(
        f"prompt_mode={args.prompt_mode} thinking_prefill={args.thinking_prefill} "
        f"strategy={json.dumps(no_think_strategy, ensure_ascii=False, sort_keys=True)} "
        f"im_end_token_id={end_token_id}",
        flush=True,
    )
    if args.prompt_mode == "chat_template":
        write_chat_template_preflight(
            exp_dir,
            processor,
            build_messages(Path(args.image_root) / data[0]["image"], build_user_text(data[0]), args),
            thinking_prefill=args.thinking_prefill,
            strategy=no_think_strategy,
        )

    completed = 0
    with predictions_path.open("a", encoding="utf-8") as handle:
        for sample in tqdm(data, desc="closed-vqa eval", file=sys.stdout):
            if sample.get("id") in processed:
                continue
            row = run_one(sample, args=args, processor=processor, model=model, device=device)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            if completed % args.metrics_every == 0:
                write_json(metrics_path, summarize(predictions_path))

    metrics = summarize(predictions_path)
    write_json(metrics_path, metrics)
    write_json(
        status_path,
        {
            "status": "PASS" if metrics["total"] == args.expected_total else "PARTIAL",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": metrics,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if metrics["total"] == args.expected_total else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-run", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--expected-total", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--image-max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--metrics-every", type=int, default=50)
    parser.add_argument("--disable-flash-attention", action="store_true")
    parser.add_argument("--prompt-mode", choices=["chat_template", "sft_dataset_manual"], default="chat_template")
    parser.add_argument("--thinking-prefill", choices=["closed", "no_default"], default="closed")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = preflight(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if not result["passed"] or args.preflight_only:
            return 0 if result["passed"] else 2
        return run_eval(args)
    except torch.cuda.OutOfMemoryError:
        print("CUDA_OOM", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 75
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
