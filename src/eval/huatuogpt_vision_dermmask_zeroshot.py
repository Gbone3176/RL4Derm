#!/usr/bin/env python3
"""DermMask MCQA evaluation for HuatuoGPT-Vision-7B.

The entrypoint is intentionally strict: it records dataset/model preflight without
loading weights, then at runtime performs a one-sample model smoke before full eval.
"""
from __future__ import annotations

import argparse
import hashlib
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

from PIL import Image


DEFAULT_PROMPT_SUFFIX = "Answer with only the single correct option letter."
IMAGE_TOKEN_INDEX = -200
PRED_PREFIX_RE = re.compile(r"^\s*([A-Za-z])\s*(?:[\).:\-]|$)")
STRICT_RE = re.compile(r"^[A-Za-z]$")
STANDALONE_RE = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")
LOCAL_CLIP_REPO_DIR = "models--openai--clip-vit-large-patch14-336"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shell_json(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def has_clip_vision_files(path: Path) -> bool:
    return bool(
        path.exists()
        and (path / "config.json").exists()
        and (path / "preprocessor_config.json").exists()
        and ((path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists())
    )


def local_clip_cache_roots() -> list[Path]:
    roots = []
    for env_name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_name)
        if not value:
            continue
        root = Path(value)
        roots.append(root / "hub" if env_name == "HF_HOME" else root)
    roots.extend(
        [
            Path("/mnt/nas1/disk06/bowenguo/hf_cache/hub"),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
    )
    unique = []
    seen = set()
    for root in roots:
        resolved = str(root)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return unique


def resolve_local_vision_tower(config: Any, model_path: Path, requested: str | None = None) -> tuple[Path | None, list[dict[str, Any]]]:
    declared = requested or getattr(config, "mm_vision_tower", None) or getattr(config, "vision_tower", None)
    candidates: list[Path] = []
    if declared:
        declared_path = Path(str(declared)).expanduser()
        candidates.append(declared_path)
        if not declared_path.is_absolute():
            candidates.append(model_path / declared_path)
    for root in local_clip_cache_roots():
        repo_dir = root / LOCAL_CLIP_REPO_DIR / "snapshots"
        if repo_dir.exists():
            candidates.extend(sorted(repo_dir.iterdir()))

    checked = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        complete = has_clip_vision_files(candidate)
        checked.append(
            {
                "path": key,
                "exists": candidate.exists(),
                "has_config": (candidate / "config.json").exists(),
                "has_preprocessor": (candidate / "preprocessor_config.json").exists(),
                "has_pytorch_bin": (candidate / "pytorch_model.bin").exists(),
                "has_safetensors": (candidate / "model.safetensors").exists(),
                "complete": complete,
            }
        )
        if complete:
            return candidate, checked
    return None, checked


def first_turn(sample: dict[str, Any], role: str) -> str | None:
    for turn in sample.get("conversations") or []:
        if turn.get("from") == role:
            return str(turn.get("value", ""))
    return None


def option_labels(sample: dict[str, Any]) -> list[str]:
    labels = []
    for option in sample.get("options") or []:
        label = str(option.get("label", "")).strip().upper()
        if label:
            labels.append(label)
    return labels


def gold_label(sample: dict[str, Any]) -> str | None:
    for key in ("gold_label", "answer"):
        value = sample.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z]", value.strip()):
            return value.strip().upper()
    text = first_turn(sample, "gpt")
    if text:
        match = re.match(r"^\s*([A-Za-z])\s*(?:[\).]|$)", text)
        if match:
            return match.group(1).upper()
    return None


def parse_prediction(raw_output: str, allowed_labels: set[str]) -> tuple[str | None, str]:
    stripped = raw_output.strip()
    upper = stripped.upper()
    if STRICT_RE.fullmatch(upper) and upper in allowed_labels:
        return upper, "strict_single_letter"
    prefix = PRED_PREFIX_RE.match(stripped)
    if prefix:
        label = prefix.group(1).upper()
        return (label, "option_prefix") if label in allowed_labels else (None, "prefix_not_allowed")
    standalone = [m.group(1).upper() for m in STANDALONE_RE.finditer(stripped)]
    candidates = sorted({label for label in standalone if label in allowed_labels})
    if len(candidates) == 1:
        return candidates[0], "single_standalone_allowed_letter"
    if len(candidates) > 1:
        return None, "multiple_candidate_letters"
    return None, "no_parse"


def select_completion_ids(input_ids: Any, outputs: Any) -> tuple[Any, str, list[int], list[int], list[int], list[int]]:
    """Select completion tokens from generate output without assuming prompt echo."""
    input_row = input_ids[0].detach().cpu().tolist()
    output_row = outputs[0].detach().cpu().tolist()
    prefix_len = len(input_row)
    if len(output_row) >= prefix_len and output_row[:prefix_len] == input_row:
        completion = outputs[0, prefix_len:]
        mode = "prompt_prefix_stripped"
    else:
        completion = outputs[0]
        mode = "outputs_as_completion_no_prompt_prefix"
    completion_row = completion.detach().cpu().tolist()
    return completion, mode, input_row, output_row, completion_row, list(outputs.shape)


def resolve_image_path(image_root: Path, image_value: str) -> Path:
    path = Path(image_value)
    if path.is_absolute():
        return path
    return image_root / path


def user_prompt(sample: dict[str, Any]) -> str:
    human = first_turn(sample, "human")
    if human:
        question = human.replace("<image>", "").strip()
    else:
        question = str(sample.get("question", "")).replace("<image>", "").strip()
    return f"{question}\n\n{DEFAULT_PROMPT_SUFFIX}"


def tr57_sft_prompt(sample: dict[str, Any]) -> str:
    human = first_turn(sample, "human")
    if human is None:
        human = "<image>\n" + str(sample.get("question", "")).strip()
    return f"<|im_start|>user\n{human}<|im_end|>\n<|im_start|>assistant\n"


def import_huatuo_loader() -> None:
    """Register upstream Huatuo/LLaVA-Qwen2 classes with Transformers."""
    import llava.model.language_model.llava_qwen2  # noqa: F401


def tokenizer_image_token(tokenizer: Any, prompt: str, return_tensors: str | None = None) -> Any:
    import torch

    prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(parts: list[list[int]], sep: list[int]) -> list[list[int]]:
        return [item for pair in zip(parts, [sep] * len(parts)) for item in pair][:-1]

    input_ids: list[int] = []
    offset = 0
    if prompt_chunks and prompt_chunks[0] and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for chunk in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
        input_ids.extend(chunk[offset:])
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    if return_tensors is not None:
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    exp_dir = Path(args.exp_dir)
    preflight_dir = exp_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    test_path = Path(args.test_json)
    image_root = Path(args.image_root)
    data = read_json(test_path)
    rows = data if isinstance(data, list) else data.get("samples", [])
    eval_rows = rows[: args.limit] if args.limit else rows

    missing_images = []
    malformed = []
    label_counts = Counter()
    option_count_counts = Counter()
    max_label = None
    for index, sample in enumerate(eval_rows):
        labels = option_labels(sample)
        gold = gold_label(sample)
        option_count_counts[len(labels)] += 1
        if labels:
            max_label = max(max_label or labels[-1], labels[-1])
        if gold:
            label_counts[gold] += 1
        if not labels or gold not in labels:
            malformed.append({"index": index, "id": sample.get("id"), "gold": gold, "labels": labels})
        image_value = sample.get("image") or sample.get("image_path")
        if not image_value or not resolve_image_path(image_root, str(image_value)).exists():
            missing_images.append({"index": index, "id": sample.get("id"), "image": image_value})

    config = {}
    tokenizer_config = {}
    for filename, target in (("config.json", config), ("tokenizer_config.json", tokenizer_config)):
        path = model_path / filename
        if path.exists():
            target.update(read_json(path))
    vision_tower_path = None
    vision_tower_candidates: list[dict[str, Any]] = []
    if config:
        class StaticConfig:
            pass

        static_config = StaticConfig()
        for key, value in config.items():
            setattr(static_config, key, value)
        vision_tower_path, vision_tower_candidates = resolve_local_vision_tower(static_config, model_path, args.vision_tower_path)

    loader_smoke_code = (
        "import llava.model.language_model.llava_qwen2; "
        "from transformers import AutoConfig, AutoModelForCausalLM; "
        f"p={str(model_path)!r}; "
        "cfg=AutoConfig.from_pretrained(p, local_files_only=True, trust_remote_code=False); "
        "print(type(cfg).__name__, cfg.model_type, AutoModelForCausalLM._model_mapping[type(cfg)].__name__)"
    )
    loader_smoke = shell_json([args.python, "-c", loader_smoke_code])
    loader_smoke_passed = loader_smoke.get("returncode") == 0

    interface = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "tokenizer_class": tokenizer_config.get("tokenizer_class"),
        "mm_vision_tower": config.get("mm_vision_tower"),
        "resolved_local_vision_tower": str(vision_tower_path) if vision_tower_path else None,
        "vision_tower_candidates": vision_tower_candidates,
        "loader_static_smoke_passed": loader_smoke_passed,
        "loader_static_smoke_policy": "repo-local upstream llava_qwen2 registration; no model weights loaded",
        "static_loader_risk": (
            "Repo-local upstream Huatuo/LLaVA-Qwen2 loader registration is statically available."
            if loader_smoke_passed
            else "Current environment still cannot register llava_qwen2 statically; runtime model-load smoke must fail fast."
        ),
    }
    smoke = shell_json(
        [
            args.python,
            "-c",
            (
                "from transformers import AutoTokenizer; "
                f"p={str(model_path)!r}; "
                "tok=AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=False); "
                "print(type(tok).__name__)"
            ),
        ]
    )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "src/eval/huatuogpt_vision_dermmask_zeroshot.py",
        "git_commit": git_commit(),
        "model_path": str(model_path),
        "adapter_path": args.adapter_path,
        "adapter_path_exists": bool(args.adapter_path and Path(args.adapter_path).exists()),
        "model_path_exists": model_path.exists(),
        "model_files": sorted(p.name for p in model_path.iterdir()) if model_path.exists() else [],
        "interface": interface,
        "test_json": str(test_path),
        "test_sha256": sha256_file(test_path),
        "image_root": str(image_root),
        "actual_total": len(rows),
        "eval_total": len(eval_rows),
        "expected_total": args.expected_total,
        "limit": args.limit,
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:20],
        "malformed_count": len(malformed),
        "malformed_sample": malformed[:20],
        "label_counts": dict(label_counts),
        "option_count_counts": dict(option_count_counts),
        "max_label_seen": max_label,
        "tokenizer_static_smoke": smoke,
        "loader_static_smoke": loader_smoke,
        "prompt_contract": {
            "prompt_mode": args.prompt_mode,
            "system_prompt": None,
            "chat_template": False,
            "thinking_prefill": None,
            "tr57_prompt_template": "<|im_start|>user\\n{human}<|im_end|>\\n<|im_start|>assistant\\n",
            "target_semantics": "{gpt}<|im_end|>\\n",
            "image_token_rule": "literal <image> in human prompt is converted to IMAGE_TOKEN_INDEX=-200; images tensor passed separately",
        },
        "gpu": {
            "query": shell_json(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "compute_apps": shell_json(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            ),
        },
    }
    payload["passed"] = bool(
        model_path.exists()
        and len(rows) == args.expected_total
        and len(eval_rows) == (args.limit or args.expected_total)
        and len(missing_images) == 0
        and len(malformed) == 0
        and smoke.get("returncode") == 0
        and loader_smoke_passed
        and vision_tower_path is not None
        and (not args.adapter_path or Path(args.adapter_path).exists())
    )
    write_json(preflight_dir / "preflight.json", payload)
    write_json(exp_dir / "status.json", {"status": "PREFLIGHT_PASS" if payload["passed"] else "FAIL", "preflight": payload})
    return payload


def load_huatuo(args: argparse.Namespace):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model_path)
    import_huatuo_loader()
    try:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise RuntimeError(
            "HuatuoGPT-Vision-7B config is not loadable by current Transformers without a compatible "
            f"LLaVA-Qwen2 loader: {type(exc).__name__}: {exc}"
        ) from exc

    vision_tower_path, vision_tower_candidates = resolve_local_vision_tower(config, model_path, args.vision_tower_path)
    if vision_tower_path is None:
        write_json(
            Path(args.exp_dir) / "loader_diagnostics.json",
            {
                "status": "FAIL",
                "reason": "local_vision_tower_not_found",
                "declared_mm_vision_tower": getattr(config, "mm_vision_tower", None),
                "searched_candidates": vision_tower_candidates,
            },
        )
        raise RuntimeError(
            "No complete local CLIP vision tower found. Required files: config.json, "
            "preprocessor_config.json, and pytorch_model.bin or model.safetensors."
        )
    config.mm_vision_tower = str(vision_tower_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        init_vision_encoder_from_ckpt=True,
        output_loading_info=True,
    )
    model.eval()
    model = model.to("cuda:0")
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
        vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(model_path)
    vision_tower.to(dtype=torch.bfloat16, device=model.device)
    image_processor = vision_tower.image_processor
    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    adapter_loaded = False
    if adapter_path is not None:
        from peft import PeftModel

        if not (adapter_path / "adapter_config.json").exists() or not (adapter_path / "adapter_model.safetensors").exists():
            raise RuntimeError(f"Invalid PEFT adapter path: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.eval()
        model = model.to("cuda:0")
        adapter_loaded = True
    missing_keys = loading_info.get("missing_keys", [])
    unexpected_keys = loading_info.get("unexpected_keys", [])
    missing_vision_keys = [key for key in missing_keys if "vision_tower" in key or "mm_projector" in key]
    diagnostics = {
        "status": "PASS" if not missing_vision_keys else "FAIL",
        "model_class": type(model).__name__,
        "base_model_class": type(getattr(model, "base_model", model)).__name__,
        "config_class": type(config).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_loaded": adapter_loaded,
        "adapter_config_exists": bool(adapter_path and (adapter_path / "adapter_config.json").exists()),
        "adapter_model_exists": bool(adapter_path and (adapter_path / "adapter_model.safetensors").exists()),
        "declared_mm_vision_tower": getattr(AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False), "mm_vision_tower", None),
        "resolved_local_vision_tower": str(vision_tower_path),
        "vision_tower_candidates": vision_tower_candidates,
        "loading_missing_keys_count": len(missing_keys),
        "loading_unexpected_keys_count": len(unexpected_keys),
        "missing_vision_keys_count": len(missing_vision_keys),
        "missing_vision_keys_sample": missing_vision_keys[:50],
        "unexpected_keys_all_vision_tower": all("vision_tower" in key for key in unexpected_keys),
        "vision_tower_loaded": bool(vision_tower.is_loaded),
        "vision_tower_name": getattr(vision_tower, "vision_tower_name", None),
        "vision_tower_class": type(getattr(vision_tower, "vision_tower", None)).__name__,
        "image_processor_class": type(image_processor).__name__,
    }
    write_json(Path(args.exp_dir) / "loader_diagnostics.json", diagnostics)
    if missing_vision_keys:
        raise RuntimeError(f"Missing vision weights after Huatuo load: {missing_vision_keys[:5]}")
    return tokenizer, model, image_processor


def run_one(sample: dict[str, Any], args: argparse.Namespace, tokenizer: Any, model: Any, image_processor: Any) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    image_value = str(sample.get("image") or sample.get("image_path"))
    image_path = resolve_image_path(Path(args.image_root), image_value)
    image = Image.open(image_path).convert("RGB")
    image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device=device)

    if args.prompt_mode == "tr57_sft":
        text = tr57_sft_prompt(sample)
    else:
        prompt = user_prompt(sample)
        text = f"<|user|>\n<image>\n{prompt}\n<|assistant|>\n"
    input_ids = tokenizer_image_token(tokenizer, text, return_tensors="pt").unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion_ids, completion_decode_mode, input_token_ids, generated_token_ids, completion_token_ids, output_shape = (
        select_completion_ids(input_ids, outputs)
    )
    raw_full_skip_special = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    raw_full_with_special = tokenizer.decode(outputs[0], skip_special_tokens=False).strip()
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    raw_with_special = tokenizer.decode(completion_ids, skip_special_tokens=False).strip()
    labels = set(option_labels(sample))
    gold = gold_label(sample)
    pred, parse_status = parse_prediction(raw, labels)
    return {
        "id": sample.get("id"),
        "prompt": text,
        "prompt_mode": args.prompt_mode,
        "input_image_token_count": int((input_ids == IMAGE_TOKEN_INDEX).sum().item()),
        "input_len": int(input_ids.shape[1]),
        "output_shape": output_shape,
        "input_token_ids": input_token_ids,
        "generated_token_ids": generated_token_ids,
        "completion_token_ids": completion_token_ids,
        "completion_decode_mode": completion_decode_mode,
        "raw_output_full_skip_special": raw_full_skip_special,
        "raw_output_full_with_special": raw_full_with_special,
        "raw_output_with_special": raw_with_special,
        "image": image_value,
        "gold_label": gold,
        "allowed_labels": sorted(labels),
        "raw_output": raw,
        "pred_label": pred,
        "parse_status": parse_status,
        "correct": bool(gold and pred == gold),
    }


def summarize(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    invalid = sum(1 for row in rows if not row.get("pred_label"))
    strict = sum(1 for row in rows if row.get("parse_status") == "strict_single_letter")
    parse_status_counts = dict(Counter(row.get("parse_status") for row in rows))
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "correct": correct,
        "acc": correct / total if total else None,
        "accuracy": correct / total if total else None,
        "invalid": invalid,
        "invalid_rate": invalid / total if total else None,
        "strict_single_letter": strict,
        "strict_single_letter_rate": strict / total if total else None,
        "parse_status_counts": parse_status_counts,
    }


def write_args_snapshot(args: argparse.Namespace, gpu_id: str | None) -> None:
    write_json(
        Path(args.exp_dir) / "args_snapshot.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "script": "src/eval/huatuogpt_vision_dermmask_zeroshot.py",
            "git_commit": git_commit(),
            "args": vars(args),
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "selected_gpu": gpu_id,
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
                "HF_HOME": os.environ.get("HF_HOME"),
            },
            "prompt_contract": {
                "prompt_mode": args.prompt_mode,
                "system_prompt": None,
                "chat_template": False,
                "thinking_prefill": None,
                "tr57_prompt_template": "<|im_start|>user\\n{human}<|im_end|>\\n<|im_start|>assistant\\n",
                "target_semantics": "{gpt}<|im_end|>\\n",
                "image_token_rule": "literal <image> in human prompt is converted to IMAGE_TOKEN_INDEX=-200; images tensor passed separately",
            },
            "checkpoint_source": {
                "base_model": args.model_path,
                "adapter_path": args.adapter_path,
                "vision_tower_path": args.vision_tower_path,
            },
            "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens},
            "parse_policy": "strict/option-prefix/single-standalone letter constrained to per-sample option labels",
        },
    )


class SwanEvalTracker:
    def __init__(self, args: argparse.Namespace):
        self.run = None
        self.status_path = Path(args.exp_dir) / "swanlog" / "tracking_status.json"
        self.local_log = Path(args.exp_dir) / "swanlog" / "local_tracking.jsonl"
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import swanlab

            self.swanlab = swanlab
            self.run = swanlab.init(
                mode=args.swanlab_mode,
                log_dir=str(Path(args.exp_dir) / "swanlog"),
                project=args.swanlab_project,
                name=args.swanlab_run_name or Path(args.exp_dir).name,
                config=vars(args),
            )
            write_json(
                self.status_path,
                {
                    "status": "PASS",
                    "backend": "swanlab",
                    "mode": args.swanlab_mode,
                    "run_id": getattr(self.run, "id", None),
                    "log_dir": str(Path(args.exp_dir) / "swanlog"),
                },
            )
        except Exception as exc:
            write_json(
                self.status_path,
                {
                    "status": "FAIL",
                    "backend": "swanlab",
                    "mode": args.swanlab_mode,
                    "error": f"{type(exc).__name__}: {exc}",
                    "log_dir": str(Path(args.exp_dir) / "swanlog"),
                },
            )

    def ok(self) -> bool:
        return self.run is not None and read_json(self.status_path).get("status") == "PASS"

    def log(self, payload: dict[str, Any], step: int) -> None:
        payload = dict(payload)
        payload["step"] = step
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")
        append_jsonl(self.local_log, payload)
        if self.run is not None:
            metrics = {key: value for key, value in payload.items() if isinstance(value, (int, float, bool))}
            self.swanlab.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.swanlab.finish()
            except Exception:
                pass


def run_eval(args: argparse.Namespace) -> int:
    exp_dir = Path(args.exp_dir)
    predictions = exp_dir / "predictions.jsonl"
    raw_outputs = exp_dir / "rollouts" / "raw_outputs.jsonl"
    metrics_path = exp_dir / "metrics.json"
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_args_snapshot(args, os.environ.get("CUDA_VISIBLE_DEVICES"))

    preflight_result = preflight(args)
    if not preflight_result.get("passed"):
        write_json(exp_dir / "status.json", {"status": "FAIL", "reason": "preflight_failed", "preflight": preflight_result})
        return 2
    tracker = SwanEvalTracker(args)
    if not tracker.ok():
        write_json(
            exp_dir / "status.json",
            {"status": "FAIL", "reason": "swanlab_tracking_failed", "tracking_status": str(exp_dir / "swanlog" / "tracking_status.json")},
        )
        return 12

    data = read_json(Path(args.test_json))
    rows = data[: args.limit] if args.limit else data
    try:
        tokenizer, model, image_processor = load_huatuo(args)
    except Exception:
        write_json(
            exp_dir / "status.json",
            {
                "status": "FAIL",
                "reason": "model_load_smoke_failed",
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 10

    try:
        smoke_row = run_one(rows[0], args, tokenizer, model, image_processor)
    except Exception:
        write_json(
            exp_dir / "status.json",
            {
                "status": "FAIL",
                "reason": "one_sample_smoke_failed",
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 11
    write_json(exp_dir / "preflight" / "one_sample_model_smoke.json", {"status": "PASS", "row": smoke_row})

    if predictions.exists() and not args.resume:
        write_json(exp_dir / "status.json", {"status": "FAIL", "reason": "predictions_exists_without_resume", "path": str(predictions)})
        return 13
    if raw_outputs.exists() and not args.resume:
        write_json(exp_dir / "status.json", {"status": "FAIL", "reason": "raw_outputs_exists_without_resume", "path": str(raw_outputs)})
        return 13
    processed = set()
    if predictions.exists():
        processed = {json.loads(line).get("id") for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()}

    for index, sample in enumerate(rows, start=1):
        if sample.get("id") in processed:
            continue
        row = run_one(sample, args, tokenizer, model, image_processor)
        row["index"] = index
        append_jsonl(predictions, row)
        append_jsonl(raw_outputs, {"index": index, "id": row.get("id"), "raw_output": row.get("raw_output")})
        if index % args.metrics_every == 0:
            partial_metrics = summarize(predictions)
            write_json(metrics_path, partial_metrics)
            tracker.log({f"eval/{key}": value for key, value in partial_metrics.items() if isinstance(value, (int, float))}, index)

    metrics = summarize(predictions)
    write_json(metrics_path, metrics)
    tracker.log({f"eval/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))}, metrics.get("total", 0))
    tracker.finish()
    expected_eval_total = args.limit or args.expected_total
    status = "PASS" if metrics.get("total") == expected_eval_total else "PARTIAL"
    write_json(
        exp_dir / "status.json",
        {"status": status, "metrics": metrics, "expected_eval_total": expected_eval_total, "tracking_status": str(exp_dir / "swanlog" / "tracking_status.json")},
    )
    write_json(
        exp_dir / "summary.json",
        {
            "status": status,
            "metrics": metrics,
            "expected_eval_total": expected_eval_total,
            "predictions": str(predictions),
            "raw_outputs": str(raw_outputs),
            "tracking_status": str(exp_dir / "swanlog" / "tracking_status.json"),
        },
    )
    (exp_dir / "summary.md").write_text(
        "\n".join(
            [
                "# HuatuoGPT-Vision-7B DermMask Image-Disjoint Zero-Shot",
                "",
                f"Status: {status}",
                f"Total: {metrics.get('total')}",
                f"Accuracy: {metrics.get('accuracy')}",
                f"Correct: {metrics.get('correct')}",
                f"Invalid: {metrics.get('invalid')}",
                f"Invalid rate: {metrics.get('invalid_rate')}",
                f"Strict single-letter rate: {metrics.get('strict_single_letter_rate')}",
                f"Parse status counts: {json.dumps(metrics.get('parse_status_counts', {}), ensure_ascii=False, sort_keys=True)}",
                f"Tracking status: {exp_dir / 'swanlog' / 'tracking_status.json'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--vision-tower-path", default=None)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--expected-total", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--metrics-every", type=int, default=50)
    parser.add_argument("--prompt-mode", choices=["legacy_zeroshot", "tr57_sft"], default="legacy_zeroshot")
    parser.add_argument("--swanlab-project", default="DermoGPT-HuatuoGPT-Vision-7B")
    parser.add_argument("--swanlab-run-name", default=None)
    parser.add_argument("--swanlab-mode", choices=["online", "local", "offline"], default="online")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--load-smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.preflight_only:
            result = preflight(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            return 0 if result.get("passed") else 2
        if args.load_smoke_only:
            write_args_snapshot(args, os.environ.get("CUDA_VISIBLE_DEVICES"))
            preflight_result = preflight(args)
            if not preflight_result.get("passed"):
                return 2
            try:
                load_huatuo(args)
            except Exception:
                write_json(
                    Path(args.exp_dir) / "status.json",
                    {
                        "status": "FAIL",
                        "reason": "model_load_smoke_failed",
                        "traceback": traceback.format_exc(),
                    },
                )
                traceback.print_exc()
                return 10
            write_json(Path(args.exp_dir) / "status.json", {"status": "PASS", "reason": "loader_smoke_passed"})
            print(json.dumps(read_json(Path(args.exp_dir) / "loader_diagnostics.json"), ensure_ascii=False, sort_keys=True), flush=True)
            return 0
        return run_eval(args)
    except Exception:
        exp_dir = Path(getattr(args, "exp_dir", "."))
        write_json(exp_dir / "status.json", {"status": "FAIL", "reason": "exception", "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
