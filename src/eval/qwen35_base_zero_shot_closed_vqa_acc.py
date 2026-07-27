#!/usr/bin/env python3
"""Evaluate Qwen3.5-4B base model zero-shot on closed-set VQA MCQA data."""
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

from src.dataset.data_utils import get_image_info
from src.paths import resolve_dataset_path, resolve_model_path, resolve_project_path
from src.train.qwen35_dense_utils import git_commit
from src.utils import disable_torch_init, get_model_name_from_path, load_pretrained_model


PROMPT_SUFFIX = "Answer with only the single correct option letter."
GOLD_PREFIX_RE = re.compile(r"^\s*([A-Da-d])\s*(?:\)|$)")
PRED_PREFIX_RE = re.compile(r"^\s*([A-Da-d])\s*(?:[\).:\-]|$)")
STRICT_PRED_RE = re.compile(r"^[A-D]$")
STANDALONE_PRED_RE = re.compile(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])")


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
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


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
    question = str(human).replace("<image>", "").strip()
    return f"{question}\n\n{PROMPT_SUFFIX}"


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


def resolve_image_path(image_root: str, image_rel: str) -> Path:
    image_path = Path(image_rel)
    if image_path.is_absolute():
        return image_path
    configured = resolve_dataset_path(image_rel)
    if configured.exists():
        return configured
    return resolve_dataset_path(image_root) / image_rel


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
                    }
                )
    return {
        "availability_rule": "GPU3 sharing allowed by conductor; no kill, no pause, no GPU switch",
        "gpus": gpus,
        "gpu_query": gpu_query,
        "compute_apps_query": proc_query,
    }


def write_prompt_preflight(args: argparse.Namespace, samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for index, sample in enumerate(samples[:3]):
        image_path = resolve_image_path(args.image_root, str(sample.get("image")))
        text = build_user_text(sample)
        messages = build_messages(image_path, text, args)
        rows.append(
            {
                "index": index,
                "id": sample.get("id"),
                "image": sample.get("image"),
                "messages": messages,
                "checks": {
                    "has_no_system_prompt": all(message.get("role") != "system" for message in messages),
                    "uses_raw_test_user_turn": (first_turn(sample, "human") or "").replace("<image>", "").strip()
                    in text,
                    "has_single_letter_constraint": PROMPT_SUFFIX in text,
                    "does_not_use_sft_manual_template": "<|im_start|>user\n" not in text,
                },
            }
        )
    for row in rows:
        row["passed"] = all(row["checks"].values())

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_mode": args.prompt_mode,
        "prompt_suffix": PROMPT_SUFFIX,
        "template_source": "base model chat template over raw test MCQA user turn; no system prompt",
        "samples": rows,
        "passed": len(rows) == min(3, len(samples)) and all(row["passed"] for row in rows),
    }
    write_json(Path(args.exp_dir) / "preflight" / "prompt_preflight.json", payload)
    return payload


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    exp_dir = Path(args.exp_dir)
    preflight_dir = exp_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    test_path = resolve_project_path(args.test_json)
    image_root = resolve_dataset_path(args.image_root)
    base_model = resolve_model_path(args.base_model)
    data = read_json(test_path)
    eval_data = data[: args.limit] if isinstance(data, list) and args.limit is not None else data
    rows_ok = isinstance(eval_data, list) and len(eval_data) == args.expected_total

    missing_images = []
    malformed_rows = []
    gold_counts = Counter()
    for index, item in enumerate(eval_data if isinstance(eval_data, list) else []):
        image = item.get("image")
        if not image or not resolve_image_path(args.image_root, str(image)).exists():
            missing_images.append({"index": index, "id": item.get("id"), "image": image})
        gold = parse_gold(first_turn(item, "gpt"))
        if gold is None:
            malformed_rows.append({"index": index, "id": item.get("id"), "gpt": first_turn(item, "gpt")})
        else:
            gold_counts[gold] += 1

    prompt_preflight = write_prompt_preflight(args, eval_data if isinstance(eval_data, list) else [])
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
        "zero_shot": True,
        "adapter_path": None,
        "adapter_run": None,
        "adapter_loaded": False,
        "adapter_ok": False,
        "gpu": gpu,
        "prompt_preflight_path": str(preflight_dir / "prompt_preflight.json"),
        "prompt_preflight_passed": prompt_preflight["passed"],
        "passed": bool(
            rows_ok
            and len(missing_images) == 0
            and len(malformed_rows) == 0
            and base_model.exists()
            and prompt_preflight["passed"]
        ),
    }
    write_json(preflight_dir / "preflight.json", result)
    write_json(preflight_dir / "gpu_preflight.json", gpu)
    return result


def write_loader_diagnostics(exp_dir: Path, processor: Any, model: Any, args: argparse.Namespace) -> None:
    diagnostics = getattr(model, "_dermogpt_loader_diagnostics", None) or getattr(
        processor, "_dermogpt_loader_diagnostics", None
    )
    diagnostics = dict(diagnostics or {})
    diagnostics.update(
        {
            "zero_shot": True,
            "adapter_path": None,
            "adapter_run": None,
            "adapter_loaded": False,
            "base_model_only": True,
            "model_path": str(args.base_model),
            "model_base": None,
        }
    )
    write_json(exp_dir / "loader_diagnostics.json", diagnostics)


def load_base_model(args: argparse.Namespace):
    disable_torch_init()
    processor, model = load_pretrained_model(
        model_path=str(resolve_model_path(args.base_model)),
        model_base=None,
        model_name=get_model_name_from_path(str(args.base_model)),
        load_8bit=False,
        load_4bit=False,
        device_map="cuda:0",
        device="cuda",
        use_flash_attn=not args.disable_flash_attention,
    )
    write_loader_diagnostics(Path(args.exp_dir), processor, model, args)
    return processor, model.eval()


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
    image_path = resolve_image_path(args.image_root, str(image_rel))
    text = build_user_text(sample)
    messages = build_messages(image_path, text, args)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "<|im_start|>system" in prompt:
        raise RuntimeError(f"base zero-shot prompt unexpectedly contains system role for {sample.get('id')}")

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
        padding=True,
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


def write_args_snapshot(args: argparse.Namespace) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "src/eval/qwen35_base_zero_shot_closed_vqa_acc.py",
        "git_commit": git_commit(),
        "args": vars(args),
        "base_model": args.base_model,
        "test_json": args.test_json,
        "image_root": args.image_root,
        "exp_dir": args.exp_dir,
        "expected_total": args.expected_total,
        "limit": args.limit,
        "zero_shot": True,
        "adapter_path": None,
        "adapter_run": None,
        "adapter_loaded": False,
        "base_model_only": True,
        "prompt_suffix": PROMPT_SUFFIX,
        "prompt_mode": args.prompt_mode,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "system_prompt": None,
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


def write_rollouts_and_summary(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    exp_dir = Path(args.exp_dir)
    predictions_path = exp_dir / "predictions.jsonl"
    rollout_path = exp_dir / "rollouts" / "rollouts_2000.jsonl"
    loader_path = exp_dir / "loader_diagnostics.json"
    rows = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    with rollout_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "id": row.get("id"),
                        "image": row.get("image"),
                        "gold_letter": row.get("gold_letter"),
                        "gold_raw": row.get("gold_raw"),
                        "pred_letter": row.get("pred_letter"),
                        "raw_output": row.get("raw_output"),
                        "parse_status": row.get("parse_status"),
                        "correct": row.get("correct"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    loader = read_json(loader_path) if loader_path.exists() else {}
    summary = {
        "exp_dir": str(exp_dir),
        "total": len(rows),
        "acc": metrics.get("acc"),
        "correct": metrics.get("correct"),
        "invalid": metrics.get("invalid"),
        "strict_single_letter_rate": metrics.get("strict_single_letter_rate"),
        "metrics": metrics,
        "parse_status_counts": metrics.get("parse_status_counts", {}),
        "zero_shot": True,
        "adapter_path": None,
        "adapter_loaded": False,
        "base_model_only": True,
        "prompt_mode": args.prompt_mode,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "raw_output_samples": [
            {
                "id": row.get("id"),
                "gold_letter": row.get("gold_letter"),
                "gold_raw": row.get("gold_raw"),
                "pred_letter": row.get("pred_letter"),
                "raw_output": row.get("raw_output"),
                "parse_status": row.get("parse_status"),
                "correct": row.get("correct"),
            }
            for row in rows[:5]
        ],
        "loader_flash_attention": {
            "requested_attn_implementation": loader.get("requested_attn_implementation"),
            "from_pretrained_attn_key": loader.get("from_pretrained_attn_key"),
            "config_attn_implementation": loader.get("config_attn_implementation"),
            "forbidden_internal_attn_key_present": loader.get("forbidden_internal_attn_key_present"),
            "model_class": loader.get("model_class"),
            "processor_class": loader.get("processor_class"),
        },
        "rollout_path": str(rollout_path),
    }
    write_json(exp_dir / "summary.json", summary)
    (exp_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Ts49 Qwen3.5-4B Base Zero-Shot Closed VQA",
                "",
                f"Status: {'PASS' if metrics.get('total') == args.expected_total else 'PARTIAL'}",
                f"Total: {metrics.get('total')}",
                f"Acc: {metrics.get('acc')}",
                f"Correct: {metrics.get('correct')}",
                f"Invalid: {metrics.get('invalid')}",
                f"Strict single-letter rate: {metrics.get('strict_single_letter_rate')}",
                f"Parse status counts: {json.dumps(metrics.get('parse_status_counts', {}), ensure_ascii=False, sort_keys=True)}",
                "Zero-shot/base-only: adapter_path=null, adapter_loaded=false",
                f"Rollouts: {rollout_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )


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

    write_args_snapshot(args)
    data = read_json(resolve_project_path(args.test_json))
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

    print(f"zero_shot=true base_model={args.base_model} adapter_path=null adapter_loaded=false", flush=True)
    processor, model = load_base_model(args)
    device = model_device(model)
    end_token_id = im_end_token_id(processor.tokenizer)
    print(
        f"model_device={device} prompt_mode={args.prompt_mode} "
        f"im_end_token_id={end_token_id} adapter_loaded=false",
        flush=True,
    )

    completed = 0
    with predictions_path.open("a", encoding="utf-8") as handle:
        for sample in tqdm(data, desc="base zero-shot closed-vqa eval", file=sys.stdout):
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
    write_rollouts_and_summary(args, metrics)
    status = {
        "status": "PASS" if metrics["total"] == args.expected_total else "PARTIAL",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total": metrics.get("total"),
        "acc": metrics.get("acc"),
        "correct": metrics.get("correct"),
        "invalid": metrics.get("invalid"),
        "strict_single_letter_rate": metrics.get("strict_single_letter_rate"),
        "parse_status_counts": metrics.get("parse_status_counts", {}),
        "metrics": metrics,
        "zero_shot": True,
        "adapter_path": None,
        "adapter_loaded": False,
        "base_model_only": True,
        "prompt_mode": args.prompt_mode,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "rollout_path": str(exp_dir / "rollouts" / "rollouts_2000.jsonl"),
        "summary_path": str(exp_dir / "summary.json"),
    }
    write_json(status_path, status)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if metrics["total"] == args.expected_total else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", required=True)
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
    parser.add_argument("--prompt-mode", choices=["raw_user_chat_template"], default="raw_user_chat_template")
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
        status_path = Path(getattr(args, "exp_dir", ".")) / "eval_status.json"
        write_json(
            status_path,
            {
                "status": "FAIL",
                "reason": "CUDA_OOM",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "zero_shot": True,
                "adapter_path": None,
                "adapter_loaded": False,
            },
        )
        print("CUDA_OOM", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 75
    except Exception:
        status_path = Path(getattr(args, "exp_dir", ".")) / "eval_status.json"
        write_json(
            status_path,
            {
                "status": "FAIL",
                "reason": "exception",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "traceback": traceback.format_exc(),
                "zero_shot": True,
                "adapter_path": None,
                "adapter_loaded": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
