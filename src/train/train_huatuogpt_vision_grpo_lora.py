#!/usr/bin/env python3
"""HuatuoGPT-Vision-7B DermoMask GRPO LoRA entrypoint.

The implementation is separate from the Qwen3.5 GRPO trainer because Huatuo
uses the repo-local LLaVA-Qwen2 path: literal ``<image>`` tokens become
``IMAGE_TOKEN_INDEX=-200`` and image tensors are passed through ``images``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.dataset.grpo_dataset_derm import MEDVLMR1_RL_PROMPT_SUFFIX
from src.train import train_huatuogpt_vision_sft_lora as sft
from src.train.reward_funcs import accuracy_reward, analyze_format_completion


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
PROMPT_PREFIX = "<|im_start|>user\n"
PROMPT_MIDDLE = "<|im_end|>\n<|im_start|>assistant\n"
POLICY_ADAPTER_NAME = "policy"
REFERENCE_ADAPTER_NAME = "reference"
TARGET_LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FROZEN_MODULE_KEYWORDS = ("vision_tower", "mm_projector", "lm_head", "embed_tokens")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def shell_json(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\nTimeoutExpired after {timeout}s",
        }
    return {"command": command, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_status_short() -> str:
    proc = subprocess.run(["git", "status", "--short"], text=True, capture_output=True, check=False)
    return proc.stdout if proc.returncode == 0 else proc.stderr


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = sft.read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array at {path}")
    return rows


def build_user_text(sample: dict[str, Any]) -> str:
    human = sft.first_turn(sample, "human")
    if human is None:
        raise ValueError(f"Missing human turn for id={sample.get('id')}")
    return f"{human.rstrip()}\n\n{MEDVLMR1_RL_PROMPT_SUFFIX}"


def build_prompt(sample: dict[str, Any]) -> str:
    return f"{PROMPT_PREFIX}{build_user_text(sample)}{PROMPT_MIDDLE}"


def select_completion_ids(input_ids: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, str]:
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)
    input_len = int(input_ids.shape[1])
    if outputs.shape[1] >= input_len and torch.equal(outputs[0, :input_len].detach().cpu(), input_ids[0].detach().cpu()):
        return outputs[0, input_len:], "strip_prompt_prefix"
    return outputs[0], "outputs_as_completion_no_prompt_prefix"


def option_labels(sample: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for option in sample.get("options") or []:
        label = option.get("label") if isinstance(option, dict) else None
        if label:
            labels.append(str(label).strip().upper())
    if labels:
        return labels
    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class HuatuoGrpoDataset:
    def __init__(self, data_path: Path, image_root: Path, tokenizer: Any, image_processor: Any, limit: int = 0, max_prompt_length: int = 512):
        rows = read_rows(data_path)
        self.rows = rows[:limit] if limit else rows
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sft.ensure_train_imports()
        sample = self.rows[index]
        prompt = build_prompt(sample)
        prompt_ids = sft.tokenizer_image_token(self.tokenizer, prompt, return_tensors="pt")
        if self.max_prompt_length > 0 and prompt_ids.numel() > self.max_prompt_length:
            raise ValueError(
                f"Prompt length {prompt_ids.numel()} exceeds max_prompt_length={self.max_prompt_length} "
                f"for index={index} id={sample.get('id')}"
            )
        image_value = str(sample.get("image") or sample.get("image_path"))
        image_path = sft.resolve_image_path(self.image_root, image_value)
        image = sft.Image.open(image_path).convert("RGB")
        image = sft.expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        assistant = sft.first_turn(sample, "gpt")
        if assistant is None:
            raise ValueError(f"Missing gpt turn for id={sample.get('id')}")
        return {
            "id": sample.get("id"),
            "index": index,
            "image": image_value,
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "images": image_tensor,
            "assistant": assistant,
            "gold_label": str(sample.get("gold_label") or sample.get("answer") or assistant).strip().upper(),
            "allowed_labels": option_labels(sample),
            "image_token_count": int((prompt_ids == IMAGE_TOKEN_INDEX).sum().item()),
        }


def validate_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(Path(args.data_path))
    image_root = Path(args.image_root)
    missing_images = []
    malformed = []
    image_values = set()
    option_counts: dict[int, int] = {}
    image_token_counts: dict[int, int] = {}
    for index, sample in enumerate(rows):
        human = sft.first_turn(sample, "human")
        target = sft.first_turn(sample, "gpt")
        options = sample.get("options") or []
        option_counts[len(options)] = option_counts.get(len(options), 0) + 1
        image_value = sample.get("image") or sample.get("image_path")
        if image_value:
            image_values.add(str(image_value))
            if args.image_existence_check_limit < 0 or index < args.image_existence_check_limit:
                if not sft.resolve_image_path(image_root, str(image_value)).exists() and len(missing_images) < 20:
                    missing_images.append({"index": index, "id": sample.get("id"), "image": image_value})
        else:
            missing_images.append({"index": index, "id": sample.get("id"), "image": image_value})
        if human:
            image_token_counts[human.count("<image>")] = image_token_counts.get(human.count("<image>"), 0) + 1
        if human is None or target is None or not options:
            malformed.append(
                {
                    "index": index,
                    "id": sample.get("id"),
                    "has_human": human is not None,
                    "has_gpt": target is not None,
                    "option_count": len(options),
                }
            )
    return {
        "data_path": args.data_path,
        "data_sha256": sft.sha256_file(Path(args.data_path)),
        "row_count": len(rows),
        "unique_images": len(image_values),
        "duplicate_image_rows": len(rows) - len(image_values),
        "option_count_distribution": dict(sorted(option_counts.items())),
        "image_token_count_distribution": dict(sorted(image_token_counts.items())),
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images,
        "malformed_count": len(malformed),
        "malformed_sample": malformed[:20],
        "prompt_suffix": MEDVLMR1_RL_PROMPT_SUFFIX,
        "first_prompt_sample": build_prompt(rows[0])[:1200] if rows else None,
        "first_target": sft.first_turn(rows[0], "gpt") if rows else None,
    }


def strict_reward_gate_probe() -> dict[str, Any]:
    probes = {
        "strict_valid": "<think>reason</think><answer>A</answer>",
        "empty_think_prefix": "<think></think><answer>A</answer>",
        "outside_text": "x<think>reason</think><answer>A</answer>",
    }
    results = {name: analyze_format_completion(text) for name, text in probes.items()}
    passed = (
        results["strict_valid"]["format_reward"] == 1.0
        and results["empty_think_prefix"]["format_reward"] == 0.0
        and results["outside_text"]["format_reward"] == 0.0
        and results["empty_think_prefix"]["empty_think_prefix"]
        and results["outside_text"]["outside_text_nonspace_chars"] > 0
    )
    return {"status": "PASS" if passed else "FAIL", "results": results}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_code = (
        "import llava.model.language_model.llava_qwen2; "
        "from transformers import AutoConfig, AutoTokenizer; "
        f"p={str(Path(args.model_path))!r}; "
        "cfg=AutoConfig.from_pretrained(p, local_files_only=True, trust_remote_code=False); "
        "tok=AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=False); "
        "print(type(cfg).__name__, cfg.model_type, type(tok).__name__)"
    )
    adapter_path = Path(args.init_adapter_path)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "src/train/train_huatuogpt_vision_grpo_lora.py",
        "git_commit": git_commit(),
        "git_status_short": git_status_short(),
        "model_path": args.model_path,
        "vision_tower_path": args.vision_tower_path,
        "init_adapter_path": args.init_adapter_path,
        "adapter_files": {
            "adapter_config": str(adapter_path / "adapter_config.json"),
            "adapter_model": str(adapter_path / "adapter_model.safetensors"),
            "adapter_config_exists": (adapter_path / "adapter_config.json").exists(),
            "adapter_model_exists": (adapter_path / "adapter_model.safetensors").exists(),
        },
        "manifest": validate_manifest(args),
        "reward_gate_probe": strict_reward_gate_probe(),
        "static_loader_smoke": shell_json([sys.executable, "-c", smoke_code], timeout=args.static_smoke_timeout),
        "gpu_query": shell_json(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
        ),
        "compute_apps": shell_json(
            ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
        ),
    }
    payload["passed"] = bool(
        payload["adapter_files"]["adapter_config_exists"]
        and payload["adapter_files"]["adapter_model_exists"]
        and payload["manifest"]["row_count"] > 0
        and payload["manifest"]["missing_images_count"] == 0
        and payload["manifest"]["malformed_count"] == 0
        and payload["reward_gate_probe"]["status"] == "PASS"
        and payload["static_loader_smoke"]["returncode"] == 0
    )
    write_json(output_dir / "preflight" / "preflight.json", payload)
    write_json(output_dir / "status.json", {"status": "PREFLIGHT_PASS" if payload["passed"] else "FAIL", "preflight": str(output_dir / "preflight" / "preflight.json")})
    return payload


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def adapter_names(model: torch.nn.Module) -> list[str]:
    base = unwrap_model(model)
    return sorted(getattr(base, "peft_config", {}).keys())


def active_adapter_name(model: torch.nn.Module) -> str | None:
    active = getattr(unwrap_model(model), "active_adapter", None)
    if isinstance(active, (list, tuple)):
        return ",".join(str(item) for item in active)
    return str(active) if active is not None else None


def is_blocked_module_param(name: str) -> bool:
    return any(blocked in name for blocked in FROZEN_MODULE_KEYWORDS)


def is_adapter_lora_param(name: str, adapter_name: str) -> bool:
    return "lora_" in name and f".{adapter_name}." in name


def is_policy_language_lora_param(name: str) -> bool:
    return (
        is_adapter_lora_param(name, POLICY_ADAPTER_NAME)
        and ".model.layers." in name
        and not is_blocked_module_param(name)
        and any(f".{suffix}.lora_" in name for suffix in TARGET_LORA_SUFFIXES)
    )


def reset_policy_trainable_gate(model: torch.nn.Module) -> dict[str, Any]:
    base = unwrap_model(model)
    policy_lora_names = []
    reference_lora_names = []
    target_hits = {suffix: 0 for suffix in TARGET_LORA_SUFFIXES}
    for name, param in base.named_parameters():
        is_policy_trainable = is_policy_language_lora_param(name)
        param.requires_grad_(is_policy_trainable)
        if is_adapter_lora_param(name, POLICY_ADAPTER_NAME):
            policy_lora_names.append(name)
            for suffix in TARGET_LORA_SUFFIXES:
                if f".{suffix}.lora_" in name:
                    target_hits[suffix] += 1
        if is_adapter_lora_param(name, REFERENCE_ADAPTER_NAME):
            reference_lora_names.append(name)

    trainable = [(name, param) for name, param in base.named_parameters() if param.requires_grad]
    non_policy_trainable = [name for name, _ in trainable if not is_policy_language_lora_param(name)]
    blocked_trainable = [name for name, _ in trainable if is_blocked_module_param(name)]
    policy_trainable = [name for name, _ in trainable if is_policy_language_lora_param(name)]
    trainable_count = sum(param.numel() for _, param in trainable)
    total_count = sum(param.numel() for param in base.parameters())
    missing_suffixes = [suffix for suffix, count in target_hits.items() if count <= 0]
    status = (
        "PASS"
        if policy_trainable
        and reference_lora_names
        and not non_policy_trainable
        and not blocked_trainable
        and not missing_suffixes
        else "FAIL"
    )
    return {
        "status": status,
        "shared_base": True,
        "adapter_names": adapter_names(base),
        "active_adapter": active_adapter_name(base),
        "policy_adapter": POLICY_ADAPTER_NAME,
        "reference_adapter": REFERENCE_ADAPTER_NAME,
        "policy_lora_tensor_count": len(policy_lora_names),
        "reference_lora_tensor_count": len(reference_lora_names),
        "policy_trainable_tensor_count": len(policy_trainable),
        "policy_trainable_param_count": trainable_count,
        "total_param_count": total_count,
        "trainable_ratio": trainable_count / total_count if total_count else None,
        "target_hits": dict(sorted(target_hits.items())),
        "missing_target_suffixes": missing_suffixes,
        "non_policy_trainable_count": len(non_policy_trainable),
        "blocked_trainable_count": len(blocked_trainable),
        "policy_trainable_names_sample": policy_trainable[:120],
        "non_policy_trainable_names_sample": non_policy_trainable[:80],
        "blocked_trainable_names_sample": blocked_trainable[:80],
    }


def set_adapter_for_forward(model: torch.nn.Module, adapter_name: str) -> dict[str, Any]:
    base = unwrap_model(model)
    if adapter_name not in getattr(base, "peft_config", {}):
        raise RuntimeError(f"Adapter {adapter_name!r} is not loaded; available={adapter_names(base)}")
    base.set_adapter(adapter_name)
    gate = reset_policy_trainable_gate(base)
    if active_adapter_name(base) != adapter_name:
        raise RuntimeError(f"Failed to activate adapter={adapter_name}; active={active_adapter_name(base)}")
    if gate["status"] != "PASS":
        raise RuntimeError(f"Shared-base policy trainable gate failed after set_adapter({adapter_name!r}): {gate}")
    return gate


def policy_optimizer_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    base = unwrap_model(model)
    gate = reset_policy_trainable_gate(base)
    params = [param for name, param in base.named_parameters() if param.requires_grad and is_policy_language_lora_param(name)]
    if gate["status"] != "PASS" or not params:
        raise RuntimeError(f"Optimizer policy parameter gate failed: {gate}")
    return params


def load_policy_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    from peft import PeftModel

    tokenizer, base_model, image_processor, loader_diag = sft.load_model_tokenizer_processor(args)
    adapter_path = Path(args.init_adapter_path)
    if not (adapter_path / "adapter_config.json").exists() or not (adapter_path / "adapter_model.safetensors").exists():
        raise RuntimeError(f"Required Tr57 PEFT adapter is missing or incomplete: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        adapter_name=POLICY_ADAPTER_NAME,
        is_trainable=True,
    )
    model.load_adapter(
        adapter_path,
        adapter_name=REFERENCE_ADAPTER_NAME,
        is_trainable=False,
    )
    set_adapter_for_forward(model, POLICY_ADAPTER_NAME)
    model.config.use_cache = False
    model.to(device=device, dtype=dtype)
    trainable_gate = set_adapter_for_forward(model, POLICY_ADAPTER_NAME)
    lora_diag = {
        "status": trainable_gate["status"],
        "shared_base": True,
        "init_adapter_path": str(adapter_path),
        "selection_policy": "single Huatuo base with policy/reference PEFT adapters; only policy language LoRA params require grad",
        "required_target_suffixes": list(TARGET_LORA_SUFFIXES),
        "frozen": list(FROZEN_MODULE_KEYWORDS) + ["reference adapter", "base model"],
        "peft_init": {
            "from_pretrained_adapter_name": POLICY_ADAPTER_NAME,
            "from_pretrained_is_trainable": True,
            "load_adapter_adapter_name": REFERENCE_ADAPTER_NAME,
            "load_adapter_is_trainable": False,
        },
        "trainable_gate": trainable_gate,
        "matched_target_param_counts": trainable_gate["target_hits"],
        "adapter_names": trainable_gate["adapter_names"],
        "active_adapter": trainable_gate["active_adapter"],
        "policy_trainable_tensor_count": trainable_gate["policy_trainable_tensor_count"],
        "policy_lora_tensor_count": trainable_gate["policy_lora_tensor_count"],
        "reference_lora_tensor_count": trainable_gate["reference_lora_tensor_count"],
        "blocked_trainables": trainable_gate["blocked_trainable_count"],
        "non_policy_trainables": trainable_gate["non_policy_trainable_count"],
        "trainable_param_count": trainable_gate["policy_trainable_param_count"],
        "total_param_count": trainable_gate["total_param_count"],
        "trainable_ratio": trainable_gate["trainable_ratio"],
        "trainable_param_names_sample": trainable_gate["policy_trainable_names_sample"],
        "blocked_trainable_names": trainable_gate["blocked_trainable_names_sample"],
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    if lora_diag["status"] != "PASS":
        raise RuntimeError(f"LoRA initialization failed gate: {lora_diag}")
    return tokenizer, model, image_processor, loader_diag, lora_diag


def write_args_snapshot(args: argparse.Namespace, extra: dict[str, Any]) -> None:
    write_json(
        Path(args.output_dir) / "args_snapshot.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "script": "src/train/train_huatuogpt_vision_grpo_lora.py",
            "git_commit": git_commit(),
            "git_status_short": git_status_short(),
            "args": vars(args),
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
                "RANK": os.environ.get("RANK"),
                "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
                "HF_HOME": os.environ.get("HF_HOME"),
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
                "SWANLAB_MODE": os.environ.get("SWANLAB_MODE"),
                "SWANLAB_LOG_DIR": os.environ.get("SWANLAB_LOG_DIR"),
                "http_proxy": os.environ.get("http_proxy"),
                "https_proxy": os.environ.get("https_proxy"),
                "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
                "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
            },
            "prompt_contract": {
                "system_prompt": None,
                "chat_template": False,
                "thinking_prefill": None,
                "prompt_template": "<|im_start|>user\\n{human_plus_MEDVLMR1_RL_PROMPT_SUFFIX}<|im_end|>\\n<|im_start|>assistant\\n",
                "medvlmr1_suffix": MEDVLMR1_RL_PROMPT_SUFFIX,
                "completion_contract": "<think>...</think><answer>single-letter</answer>",
                "image_token_rule": "literal <image> is converted to IMAGE_TOKEN_INDEX=-200; images tensor is passed separately",
            },
            "lora_contract": {
                "shared_base": True,
                "adapter_names": [POLICY_ADAPTER_NAME, REFERENCE_ADAPTER_NAME],
                "init": "Tr57 checkpoint-3422 adapter, no clean-base fallback",
                "target_language_suffixes": list(TARGET_LORA_SUFFIXES),
                "frozen": list(FROZEN_MODULE_KEYWORDS) + [REFERENCE_ADAPTER_NAME, "base"],
                "r_alpha_dropout": [args.lora_rank, args.lora_alpha, args.lora_dropout],
            },
            "extra": extra,
        },
    )


class SwanTracker:
    def __init__(self, args: argparse.Namespace, rank: int):
        self.rank = rank
        self.run = None
        self.swanlab = None
        self.local_log = Path(args.output_dir) / "swanlog" / "local_tracking.jsonl"
        self.status_path = Path(args.output_dir) / "swanlog" / "tracking_status.json"
        self.mode = os.environ.get("SWANLAB_MODE", args.swanlab_mode)
        if rank != 0:
            return
        try:
            import swanlab

            self.swanlab = swanlab
            self.run = swanlab.init(
                mode=self.mode,
                log_dir=str(Path(args.output_dir) / "swanlog"),
                project=args.swanlab_project,
                name=args.swanlab_run_name or Path(args.output_dir).name,
                config=vars(args),
            )
            write_json(
                self.status_path,
                {
                    "status": "PASS",
                    "backend": "swanlab",
                    "mode": self.mode,
                    "run_id": getattr(self.run, "id", None),
                    "log_dir": str(Path(args.output_dir) / "swanlog"),
                },
            )
        except Exception as exc:
            write_json(
                self.status_path,
                {
                    "status": "FAIL",
                    "backend": "swanlab",
                    "mode": self.mode,
                    "error": f"{type(exc).__name__}: {exc}",
                    "log_dir": str(Path(args.output_dir) / "swanlog"),
                },
            )
            raise

    def log(self, payload: dict[str, Any], step: int) -> None:
        if self.rank != 0:
            return
        payload = dict(payload)
        payload["step"] = step
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")
        append_jsonl(self.local_log, payload)
        if self.run is not None and self.swanlab is not None:
            metrics = {key: value for key, value in payload.items() if isinstance(value, (int, float, bool))}
            self.swanlab.log(metrics, step=step)

    def finish(self) -> None:
        if self.rank == 0 and self.run is not None and self.swanlab is not None:
            try:
                self.swanlab.finish()
            except Exception:
                pass


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def model_for_generation(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def make_dataset(args: argparse.Namespace, tokenizer: Any, image_processor: Any) -> HuatuoGrpoDataset:
    return HuatuoGrpoDataset(Path(args.data_path), Path(args.image_root), tokenizer, image_processor, args.limit, args.max_prompt_length)


def decode_completion(tokenizer: Any, ids: torch.Tensor) -> tuple[str, str]:
    return (
        tokenizer.decode(ids.detach().cpu().tolist(), skip_special_tokens=True).strip(),
        tokenizer.decode(ids.detach().cpu().tolist(), skip_special_tokens=False).strip(),
    )


def generate_rollouts(
    args: argparse.Namespace,
    policy_model: torch.nn.Module,
    tokenizer: Any,
    sample: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    prompt_ids = sample["prompt_ids"].unsqueeze(0).to(device)
    images = sample["images"].unsqueeze(0).to(device=device, dtype=dtype)
    generator = model_for_generation(policy_model)
    set_adapter_for_forward(generator, POLICY_ADAPTER_NAME)
    generator.eval()
    rollouts = []
    with torch.inference_mode():
        for rank in range(args.num_generations):
            outputs = generator.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids, dtype=torch.bool),
                images=images,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=1,
                max_new_tokens=args.max_completion_length,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            completion_ids, decode_mode = select_completion_ids(prompt_ids, outputs)
            completion_text, completion_with_special = decode_completion(tokenizer, completion_ids)
            terminated = bool((completion_ids == tokenizer.eos_token_id).any().item()) if completion_ids.numel() else False
            clipped = bool(completion_ids.numel() >= args.max_completion_length and not terminated)
            rollouts.append(
                {
                    "rank": rank,
                    "output_shape": list(outputs.shape),
                    "input_len": int(prompt_ids.shape[1]),
                    "generated_token_ids": outputs[0].detach().cpu().tolist(),
                    "completion_token_ids": completion_ids.detach().cpu().tolist(),
                    "completion_len": int(completion_ids.numel()),
                    "completion_decode_mode": decode_mode,
                    "completion": completion_text,
                    "completion_with_special": completion_with_special,
                    "terminated": terminated,
                    "clipped": clipped,
                }
            )
    generator.train()
    set_adapter_for_forward(generator, POLICY_ADAPTER_NAME)
    return rollouts


def sequence_logprob(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    image: torch.Tensor,
    dtype: torch.dtype,
    adapter_name: str,
) -> torch.Tensor:
    if adapter_name not in (POLICY_ADAPTER_NAME, REFERENCE_ADAPTER_NAME):
        raise RuntimeError(f"Unsupported adapter_name={adapter_name!r}")
    forward_model = model if adapter_name == POLICY_ADAPTER_NAME else unwrap_model(model)
    set_adapter_for_forward(model, adapter_name)
    if completion_ids.numel() == 0:
        if adapter_name == REFERENCE_ADAPTER_NAME:
            set_adapter_for_forward(model, POLICY_ADAPTER_NAME)
            return torch.zeros((), device=prompt_ids.device, dtype=torch.float32)
        return torch.zeros((), device=prompt_ids.device, dtype=torch.float32, requires_grad=True)
    input_ids = torch.cat([prompt_ids, completion_ids.to(prompt_ids.device)], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    images = image.unsqueeze(0).to(device=prompt_ids.device, dtype=dtype)
    context = torch.no_grad() if adapter_name == REFERENCE_ADAPTER_NAME else nullcontext()
    try:
        with context:
            outputs = forward_model(input_ids=input_ids, attention_mask=attention_mask, images=images, use_cache=False)
            logits = outputs.logits.float()
            completion_len = int(completion_ids.numel())
            expanded_completion_start = int(logits.shape[1]) - completion_len
            if expanded_completion_start <= 0:
                raise RuntimeError(
                    f"Invalid Huatuo expanded completion start: logits_len={int(logits.shape[1])}, completion_len={completion_len}"
                )
            token_logits = logits[:, expanded_completion_start - 1 : int(logits.shape[1]) - 1, :]
            if int(token_logits.shape[1]) != completion_len:
                raise RuntimeError(
                    f"Huatuo logprob alignment failed: token_logits_len={int(token_logits.shape[1])}, "
                    f"completion_len={completion_len}, logits_len={int(logits.shape[1])}"
                )
            log_probs = torch.log_softmax(token_logits, dim=-1)
            target = completion_ids.to(prompt_ids.device).view(1, -1)
            gathered = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
            return gathered.sum()
    finally:
        if adapter_name == REFERENCE_ADAPTER_NAME:
            set_adapter_for_forward(model, POLICY_ADAPTER_NAME)


def rollout_rewards(completion: str, assistant: str) -> dict[str, Any]:
    format_diag = analyze_format_completion(completion)
    accuracy = float(accuracy_reward([completion], [assistant])[0])
    return {
        "format_reward": float(format_diag.get("format_reward", 0.0)),
        "accuracy_reward": accuracy,
        "reward": float(format_diag.get("format_reward", 0.0)) + accuracy,
        "format_diagnostics": format_diag,
        "accuracy_diagnostics": {
            "assistant": assistant,
            "accuracy_reward": accuracy,
        },
    }


def grad_coverage(model: torch.nn.Module) -> dict[str, Any]:
    base = model.module if hasattr(model, "module") else model
    trainable = [(name, param) for name, param in base.named_parameters() if param.requires_grad]
    missing = [name for name, param in trainable if param.grad is None]
    zero = [
        name
        for name, param in trainable
        if param.grad is not None and torch.isfinite(param.grad).all().item() and float(param.grad.detach().float().abs().sum().item()) == 0.0
    ]
    nonfinite = [name for name, param in trainable if param.grad is not None and not torch.isfinite(param.grad).all().item()]
    return {
        "trainable_param_tensors": len(trainable),
        "grad_present_count": len(trainable) - len(missing),
        "grad_missing_count": len(missing),
        "grad_zero_count": len(zero),
        "grad_nonfinite_count": len(nonfinite),
        "grad_missing_sample": missing[:80],
        "grad_zero_sample": zero[:80],
        "grad_nonfinite_sample": nonfinite[:80],
        "all_trainable_params_have_grad": len(missing) == 0,
    }


def save_checkpoint(args: argparse.Namespace, model: torch.nn.Module, optimizer: torch.optim.Optimizer, state: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / f"checkpoint-{state['global_step']}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    adapter_dir = ckpt_dir / "adapter"
    model_to_save.save_pretrained(adapter_dir)
    training_state_path = ckpt_dir / "training_state.pt"
    torch.save({"optimizer": optimizer.state_dict(), "state": state}, training_state_path)
    entry = {
        "global_step": state["global_step"],
        "path": str(ckpt_dir),
        "adapter_path": str(adapter_dir),
        "training_state_path": str(training_state_path),
        "resumable": training_state_path.exists(),
        "weight_only": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = output_dir / "checkpoints" / "manifest.json"
    old = []
    if manifest_path.exists():
        old = sft.read_json(manifest_path).get("checkpoints", [])
    old.append(entry)
    if args.save_total_limit > 0 and len(old) > args.save_total_limit:
        stale = old[:-args.save_total_limit]
        old = old[-args.save_total_limit :]
        for item in stale:
            shutil.rmtree(item["path"], ignore_errors=True)
    write_json(manifest_path, {"checkpoints": old, "latest": old[-1] if old else None})
    return entry


def lr_for_step(args: argparse.Namespace, step: int) -> float:
    return args.learning_rate


def run_train(args: argparse.Namespace, *, debug_smoke: bool) -> int:
    sft.ensure_train_imports()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    output_dir = Path(args.output_dir)
    try:
        if rank == 0:
            preflight = run_preflight(args)
        else:
            preflight = None
        if distributed:
            obj = [preflight]
            torch.distributed.broadcast_object_list(obj, src=0)
            preflight = obj[0]
        if not preflight or not preflight.get("passed"):
            return 2

        tokenizer, policy_model, image_processor, loader_diag, lora_diag = load_policy_model(args, device, dtype)
        if rank == 0:
            write_json(output_dir / "loader_diagnostics.json", loader_diag)
            write_json(output_dir / "lora_diagnostics.json", lora_diag)
        if loader_diag["loading_missing_keys_count"] or loader_diag["loading_unexpected_keys_count"] or loader_diag["missing_vision_keys_count"]:
            if rank == 0:
                write_json(output_dir / "status.json", {"status": "FAIL", "reason": "loader_key_gate_failed", "loader_diagnostics": loader_diag})
            return 10

        if args.gradient_checkpointing:
            policy_model.gradient_checkpointing_enable()
            policy_model.enable_input_require_grads()
        set_adapter_for_forward(policy_model, POLICY_ADAPTER_NAME)
        policy_model.train()
        if distributed:
            policy_model = torch.nn.parallel.DistributedDataParallel(
                policy_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        trainable_gate = reset_policy_trainable_gate(policy_model)
        if trainable_gate["status"] != "PASS":
            raise RuntimeError(f"Policy trainable gate failed before optimizer: {trainable_gate}")

        dataset = make_dataset(args, tokenizer, image_processor)
        if len(dataset) == 0:
            raise RuntimeError("GRPO dataset is empty.")
        tracker = SwanTracker(args, rank)
        write_args_snapshot(
            args,
            {
                "preflight": str(output_dir / "preflight" / "preflight.json"),
                "loader_diagnostics": str(output_dir / "loader_diagnostics.json"),
                "lora_diagnostics": str(output_dir / "lora_diagnostics.json"),
                "world_size": world_size,
                "debug_smoke": debug_smoke,
                "shared_base": True,
                "adapter_names": adapter_names(policy_model),
                "active_adapter": active_adapter_name(policy_model),
                "trainable_gate": trainable_gate,
            },
        )

        optimizer = torch.optim.AdamW(policy_optimizer_parameters(policy_model), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        global_step = 0
        micro_step = 0
        first_step_payload = None
        rollout_path = output_dir / "rollouts" / f"rollouts_rank{rank}.jsonl"
        while global_step < args.max_steps:
            accumulated_metrics: list[dict[str, Any]] = []
            for _ in range(args.gradient_accumulation_steps):
                sample_index = (micro_step * world_size + rank + args.seed) % len(dataset)
                sample = dataset[sample_index]
                if sample["image_token_count"] <= 0:
                    raise RuntimeError(f"Sample has no -200 image token: index={sample_index} id={sample['id']}")
                rollouts = generate_rollouts(args, policy_model, tokenizer, sample, device, dtype)
                if not any(item["completion"] for item in rollouts):
                    raise RuntimeError(f"All generated completions are empty at micro_step={micro_step} sample={sample['id']}")
                prompt_ids = sample["prompt_ids"].to(device)
                image = sample["images"].to(device=device, dtype=dtype)
                rewards = []
                rollout_records = []
                for rollout in rollouts:
                    reward_payload = rollout_rewards(rollout["completion"], sample["assistant"])
                    rewards.append(reward_payload["reward"])
                    record = {
                        "step": global_step + 1,
                        "micro_step": micro_step,
                        "rank": rank,
                        "rollout_rank": rollout["rank"],
                        "sample_index": sample_index,
                        "id": sample["id"],
                        "image": sample["image"],
                        "gold_label": sample["gold_label"],
                        "allowed_labels": sample["allowed_labels"],
                        "prompt": sample["prompt"],
                        **rollout,
                        **reward_payload,
                    }
                    rollout_records.append(record)
                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                if reward_tensor.numel() > 1 and torch.std(reward_tensor, unbiased=False) > 1e-6:
                    advantages = (reward_tensor - reward_tensor.mean()) / (torch.std(reward_tensor, unbiased=False) + 1e-6)
                else:
                    advantages = reward_tensor - reward_tensor.mean()
                policy_values = []
                ref_values = []
                old_values = []
                ratio_values = []
                loss_values = []
                pg_values = []
                kl_values = []
                for rollout_index, record in enumerate(rollout_records):
                    completion_ids = torch.tensor(record["completion_token_ids"], dtype=torch.long, device=device)
                    ref_logp = sequence_logprob(policy_model, prompt_ids, completion_ids, image, dtype, REFERENCE_ADAPTER_NAME)
                    if active_adapter_name(policy_model) != POLICY_ADAPTER_NAME:
                        raise RuntimeError(f"Reference logprob did not restore policy adapter; active={active_adapter_name(policy_model)}")
                    current_logp = sequence_logprob(policy_model, prompt_ids, completion_ids, image, dtype, POLICY_ADAPTER_NAME)
                    old_logp = current_logp.detach()
                    ratio = torch.exp(current_logp - old_logp)
                    pg_loss = -(ratio * advantages[rollout_index])
                    diff = ref_logp.detach() - current_logp
                    kl = torch.exp(diff) - diff - 1.0
                    loss = pg_loss + args.beta * kl
                    if not torch.isfinite(loss).item():
                        raise RuntimeError(f"Non-finite GRPO loss at micro_step={micro_step}: {float(loss.detach().float().item())}")
                    (loss / (args.gradient_accumulation_steps * max(1, len(rollout_records)))).backward()
                    record.update(
                        {
                            "policy_logp_sum": float(current_logp.detach().float().item()),
                            "old_logp_sum": float(old_logp.detach().float().item()),
                            "ref_logp_sum": float(ref_logp.detach().float().item()),
                            "ratio": float(ratio.detach().float().item()),
                            "advantage": float(advantages[rollout_index].detach().float().item()),
                            "pg_loss": float(pg_loss.detach().float().item()),
                            "kl": float(kl.detach().float().item()),
                            "loss": float(loss.detach().float().item()),
                        }
                    )
                    append_jsonl(rollout_path, record)
                    policy_values.append(record["policy_logp_sum"])
                    old_values.append(record["old_logp_sum"])
                    ref_values.append(record["ref_logp_sum"])
                    ratio_values.append(record["ratio"])
                    loss_values.append(record["loss"])
                    pg_values.append(record["pg_loss"])
                    kl_values.append(record["kl"])
                    del current_logp, ref_logp, old_logp, ratio, pg_loss, diff, kl, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                loss_mean = sum(loss_values) / len(loss_values)
                pg_mean = sum(pg_values) / len(pg_values)
                kl_mean = sum(kl_values) / len(kl_values)
                accumulated_metrics.append(
                    {
                        "loss": float(loss_mean),
                        "pg_loss": float(pg_mean),
                        "kl": float(kl_mean),
                        "reward_mean": float(reward_tensor.mean().detach().cpu().item()),
                        "reward_min": float(reward_tensor.min().detach().cpu().item()),
                        "reward_max": float(reward_tensor.max().detach().cpu().item()),
                        "format_reward_mean": float(sum(item["format_reward"] for item in rollout_records) / len(rollout_records)),
                        "accuracy_reward_mean": float(sum(item["accuracy_reward"] for item in rollout_records) / len(rollout_records)),
                        "completion_nonempty_count": int(sum(bool(item["completion"]) for item in rollout_records)),
                        "completion_count": len(rollout_records),
                        "policy_logps_finite": bool(all(math.isfinite(value) for value in policy_values)),
                        "old_logps_finite": bool(all(math.isfinite(value) for value in old_values)),
                        "ref_logps_finite": bool(all(math.isfinite(value) for value in ref_values)),
                        "ratios_finite": bool(all(math.isfinite(value) for value in ratio_values)),
                        "pg_losses_finite": bool(all(math.isfinite(value) for value in pg_values)),
                    }
                )
                micro_step += 1
            coverage = grad_coverage(policy_model)
            if coverage["grad_missing_count"] or coverage["grad_nonfinite_count"]:
                raise RuntimeError(f"LoRA gradient coverage failed: {coverage}")
            trainable_gate = reset_policy_trainable_gate(policy_model)
            if trainable_gate["status"] != "PASS":
                raise RuntimeError(f"Policy trainable gate failed before optimizer step: {trainable_gate}")
            lr = lr_for_step(args, global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_optimizer_parameters(policy_model), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            mean_metrics = {
                key: sum(item[key] for item in accumulated_metrics) / len(accumulated_metrics)
                for key in accumulated_metrics[0]
                if isinstance(accumulated_metrics[0][key], (int, float, bool))
            }
            payload = {
                "global_step": global_step,
                "micro_step": micro_step,
                "world_size": world_size,
                "rank": rank,
                "lr": lr,
                "grad_norm": float(grad_norm.detach().float().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                "finite_loss": bool(math.isfinite(mean_metrics["loss"])),
                "finite_old_logps": bool(mean_metrics["old_logps_finite"]),
                "finite_ref_logps": bool(mean_metrics["ref_logps_finite"]),
                "finite_policy_logps": bool(mean_metrics["policy_logps_finite"]),
                "finite_ratios": bool(mean_metrics["ratios_finite"]),
                "finite_pg_losses": bool(mean_metrics["pg_losses_finite"]),
                "lora_grad_coverage": coverage,
                "active_adapter": active_adapter_name(policy_model),
                "shared_base": True,
                "trainable_gate": trainable_gate,
                **mean_metrics,
            }
            if rank == 0:
                tracker.log(payload, global_step)
                print(json.dumps({"event": "grpo_step", **payload}, ensure_ascii=False), flush=True)
                if first_step_payload is None:
                    first_step_payload = payload
                    write_json(output_dir / "first_step.json", payload)
                if global_step % args.save_steps == 0 or debug_smoke:
                    save_checkpoint(args, policy_model, optimizer, {"global_step": global_step, "micro_step": micro_step, "debug_smoke": debug_smoke})
            if distributed:
                torch.distributed.barrier()
            if debug_smoke:
                break

        if rank == 0:
            if not debug_smoke and (global_step % args.save_steps != 0):
                save_checkpoint(args, policy_model, optimizer, {"global_step": global_step, "micro_step": micro_step, "debug_smoke": debug_smoke})
            status_name = "DEBUG_SMOKE_PASS" if debug_smoke else "RUNNING_COMPLETE"
            tracking_status_path = output_dir / "swanlog" / "tracking_status.json"
            tracking_status = sft.read_json(tracking_status_path) if tracking_status_path.exists() else {}
            status = {
                "status": status_name,
                "debug_smoke": debug_smoke,
                "global_step": global_step,
                "micro_step": micro_step,
                "world_size": world_size,
                "first_step": first_step_payload,
                "elapsed_seconds": time.time() - start_time,
                "rollout_paths": [str(path) for path in sorted((output_dir / "rollouts").glob("rollouts_rank*.jsonl"))],
                "tracking_status": str(tracking_status_path),
                "tracking_status_payload": tracking_status,
                "checkpoint_manifest": str(output_dir / "checkpoints" / "manifest.json"),
            }
            write_json(output_dir / "status.json", status)
            write_json(output_dir / "summary.json", status)
            (output_dir / "summary.md").write_text(
                "\n".join(
                    [
                        "# HuatuoGPT-Vision-7B DermoMask GRPO LoRA",
                        "",
                        f"Status: {status_name}",
                        f"Global step: {global_step}",
                        f"World size: {world_size}",
                        f"Rollouts: {', '.join(status['rollout_paths'])}",
                        f"Tracking: {tracking_status_path}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        tracker.finish()
        return 0
    except Exception:
        if rank == 0:
            write_json(output_dir / "status.json", {"status": "FAIL", "debug_smoke": debug_smoke, "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1
    finally:
        cleanup_distributed(distributed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower-path", required=True)
    parser.add_argument("--init-adapter-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--image-existence-check-limit", type=int, default=2000)
    parser.add_argument("--static-smoke-timeout", type=int, default=600)
    parser.add_argument("--swanlab-project", default="DermoGPT-HuatuoGPT-Vision-7B")
    parser.add_argument("--swanlab-run-name", default=None)
    parser.add_argument("--swanlab-mode", default="online", choices=["online", "local", "offline"])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--debug-smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.preflight_only:
        payload = run_preflight(args)
        write_args_snapshot(args, {"preflight": str(Path(args.output_dir) / "preflight" / "preflight.json")})
        return 0 if payload.get("passed") else 2
    return run_train(args, debug_smoke=args.debug_smoke)


if __name__ == "__main__":
    raise SystemExit(main())
