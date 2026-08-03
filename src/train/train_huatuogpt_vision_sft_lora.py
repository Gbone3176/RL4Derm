#!/usr/bin/env python3
"""HuatuoGPT-Vision-7B DermoMask SFT LoRA entrypoint.

This entrypoint is deliberately independent from the Qwen3.5 trainer because
HuatuoGPT-Vision uses the repo-local LLaVA-Qwen2 model class and CLIP image
processor. The prompt and label-mask contract mirrors the accepted Qwen3.5 SFT
dataset formatter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
QWEN35_SFT_PROMPT_PREFIX = "<|im_start|>user\n"
QWEN35_SFT_PROMPT_MIDDLE = "<|im_end|>\n<|im_start|>assistant\n"
QWEN35_SFT_TARGET_SUFFIX = "<|im_end|>\n"
LOCAL_CLIP_REPO_DIR = "models--openai--clip-vit-large-patch14-336"
torch = None
Image = None
DDP = None
DataLoader = None
DistributedSampler = None


def ensure_train_imports() -> None:
    global torch, Image, DDP, DataLoader, DistributedSampler
    if torch is not None:
        return
    import torch as torch_mod
    from PIL import Image as image_mod
    from torch.nn.parallel import DistributedDataParallel as ddp_mod
    from torch.utils.data import DataLoader as dataloader_mod
    from torch.utils.data import DistributedSampler as sampler_mod

    torch = torch_mod
    Image = image_mod
    DDP = ddp_mod
    DataLoader = dataloader_mod
    DistributedSampler = sampler_mod


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
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_status_short() -> str:
    proc = subprocess.run(["git", "status", "--short"], text=True, capture_output=True, check=False)
    return proc.stdout if proc.returncode == 0 else proc.stderr


def first_turn(sample: dict[str, Any], role: str) -> str | None:
    for turn in sample.get("conversations") or []:
        if turn.get("from") == role:
            return str(turn.get("value", ""))
    return None


def resolve_image_path(image_root: Path, image_value: str) -> Path:
    path = Path(image_value)
    if path.is_absolute():
        return path
    return image_root / path


def has_clip_vision_files(path: Path) -> bool:
    return bool(
        path.exists()
        and (path / "config.json").exists()
        and (path / "preprocessor_config.json").exists()
        and ((path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists())
    )


def local_clip_cache_roots() -> list[Path]:
    roots: list[Path] = []
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
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_local_vision_tower(config: Any, model_path: Path, requested: str | None) -> tuple[Path | None, list[dict[str, Any]]]:
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


def tokenizer_image_token(tokenizer: Any, prompt: str, return_tensors: str | None = None) -> Any:
    ensure_train_imports()
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


def build_qwen35_canonical_example(human_value: str, target_value: str) -> dict[str, str]:
    prompt = f"{QWEN35_SFT_PROMPT_PREFIX}{human_value}{QWEN35_SFT_PROMPT_MIDDLE}"
    target = f"{target_value}{QWEN35_SFT_TARGET_SUFFIX}"
    return {"prompt": prompt, "target": target, "full_text": prompt + target}


class DermoMaskHuatuoSftDataset:
    def __init__(self, data_path: Path, image_root: Path, tokenizer: Any, image_processor: Any, limit: int = 0):
        rows = read_json(data_path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON array at {data_path}")
        self.rows = rows[:limit] if limit else rows
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ensure_train_imports()
        sample = self.rows[index]
        human = first_turn(sample, "human")
        target = first_turn(sample, "gpt")
        if human is None or target is None:
            raise ValueError(f"Missing human/gpt conversation at index={index} id={sample.get('id')}")
        prompt_parts = build_qwen35_canonical_example(human, target)
        prompt_ids = tokenizer_image_token(self.tokenizer, prompt_parts["prompt"], return_tensors="pt")
        target_ids = torch.tensor(self.tokenizer(prompt_parts["target"], add_special_tokens=False).input_ids, dtype=torch.long)
        input_ids = torch.cat([prompt_ids, target_ids], dim=0)
        labels = torch.cat([torch.full_like(prompt_ids, IGNORE_INDEX), target_ids], dim=0)
        image_value = str(sample.get("image") or sample.get("image_path"))
        image_path = resolve_image_path(self.image_root, image_value)
        image = Image.open(image_path).convert("RGB")
        image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return {
            "id": sample.get("id"),
            "index": index,
            "image": image_value,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "images": image_tensor,
            "supervised_token_count": int((labels != IGNORE_INDEX).sum().item()),
            "image_token_count": int((input_ids == IMAGE_TOKEN_INDEX).sum().item()),
        }


class DataCollator:
    def __init__(self, tokenizer: Any):
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        ensure_train_imports()
        max_len = max(item["input_ids"].numel() for item in features)
        input_ids = torch.full((len(features), max_len), int(self.pad_token_id), dtype=torch.long)
        labels = torch.full((len(features), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.bool)
        for row, item in enumerate(features):
            length = item["input_ids"].numel()
            input_ids[row, :length] = item["input_ids"]
            labels[row, :length] = item["labels"]
            attention_mask[row, :length] = True
        return {
            "ids": [item["id"] for item in features],
            "indices": [item["index"] for item in features],
            "images": torch.stack([item["images"] for item in features]),
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "supervised_token_count": sum(item["supervised_token_count"] for item in features),
            "image_token_count": sum(item["image_token_count"] for item in features),
        }


def import_huatuo_loader() -> None:
    import llava.model.language_model.llava_qwen2  # noqa: F401


def load_model_tokenizer_processor(args: argparse.Namespace, *, load_weights: bool = True) -> tuple[Any, Any, Any, dict[str, Any]]:
    ensure_train_imports()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    import_huatuo_loader()
    model_path = Path(args.model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    original_vision_tower = getattr(config, "mm_vision_tower", None)
    vision_tower_path, vision_candidates = resolve_local_vision_tower(config, model_path, args.vision_tower_path)
    if vision_tower_path is None:
        raise RuntimeError("No complete local CLIP vision tower was found.")
    config.mm_vision_tower = str(vision_tower_path)
    config.tokenizer_padding_side = "right"
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if load_weights:
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
            init_vision_encoder_from_ckpt=True,
            output_loading_info=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False, init_vision_encoder_from_ckpt=True)
        loading_info = {"missing_keys": [], "unexpected_keys": []}
    model.tokenizer = tokenizer
    model.config.use_cache = False
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(load_weights=False)
    image_processor = vision_tower.image_processor
    missing_keys = sorted(loading_info.get("missing_keys", []))
    unexpected_keys = sorted(loading_info.get("unexpected_keys", []))
    missing_vision_keys = [key for key in missing_keys if "vision_tower" in key or "mm_projector" in key]
    diagnostics = {
        "status": "PASS" if not missing_keys and not unexpected_keys and not missing_vision_keys else "FAIL",
        "model_class": type(model).__name__,
        "config_class": type(config).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "declared_mm_vision_tower": original_vision_tower,
        "resolved_local_vision_tower": str(vision_tower_path),
        "vision_tower_candidates": vision_candidates,
        "vision_tower_loaded": bool(vision_tower.is_loaded),
        "vision_tower_class": type(getattr(vision_tower, "vision_tower", None)).__name__,
        "image_processor_class": type(image_processor).__name__,
        "loading_missing_keys_count": len(missing_keys),
        "loading_unexpected_keys_count": len(unexpected_keys),
        "missing_vision_keys_count": len(missing_vision_keys),
        "missing_keys_sample": missing_keys[:50],
        "unexpected_keys_sample": unexpected_keys[:50],
        "missing_vision_keys_sample": missing_vision_keys[:50],
    }
    return tokenizer, model, image_processor, diagnostics


def find_lora_target_suffixes(model: torch.nn.Module) -> tuple[list[str], list[str]]:
    ensure_train_imports()
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    matched_full_names: list[str] = []
    suffixes = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "vision_tower" in name or "mm_projector" in name or "lm_head" in name or "embed_tokens" in name:
            continue
        parts = name.split(".")
        if len(parts) >= 4 and parts[-1] in allowed and ".layers." in f".{name}.":
            matched_full_names.append(name)
            suffixes.add(parts[-1])
    return sorted(suffixes), matched_full_names


def apply_lora(args: argparse.Namespace, model: torch.nn.Module) -> tuple[torch.nn.Module, dict[str, Any]]:
    ensure_train_imports()
    from peft import LoraConfig, get_peft_model

    target_suffixes, matched_modules = find_lora_target_suffixes(model)
    if not target_suffixes or not matched_modules:
        raise RuntimeError("No Huatuo language Linear modules matched for LoRA.")
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_suffixes,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora_params = [name for name, param in model.named_parameters() if param.requires_grad]
    diagnostics = {
        "status": "PASS" if trainable > 0 else "FAIL",
        "selection_policy": "actual torch.nn.Linear modules under language model.layers; excludes vision_tower, mm_projector, lm_head, embed_tokens",
        "target_suffixes": target_suffixes,
        "matched_module_count": len(matched_modules),
        "matched_modules_sample": matched_modules[:80],
        "trainable_param_count": trainable,
        "total_param_count": total,
        "trainable_ratio": trainable / total if total else None,
        "trainable_param_names_sample": lora_params[:80],
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    if trainable <= 0:
        raise RuntimeError("LoRA trainable parameter count is zero.")
    return model, diagnostics


def validate_manifest(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data_path)
    image_root = Path(args.image_root)
    rows = read_json(data_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array at {data_path}")
    missing_images = []
    malformed = []
    option_counts = Counter()
    image_counts = Counter()
    answer_counts = Counter()
    image_token_counts = Counter()
    image_check_limit = args.image_existence_check_limit
    checked_images = 0
    for index, sample in enumerate(rows):
        options = sample.get("options") or []
        option_counts[len(options)] += 1
        image_value = sample.get("image") or sample.get("image_path")
        if image_value:
            image_counts[str(image_value)] += 1
            if image_check_limit < 0 or checked_images < image_check_limit:
                checked_images += 1
                if not resolve_image_path(image_root, str(image_value)).exists() and len(missing_images) < 20:
                    missing_images.append({"index": index, "id": sample.get("id"), "image": image_value})
        else:
            missing_images.append({"index": index, "id": sample.get("id"), "image": image_value})
        human = first_turn(sample, "human")
        target = first_turn(sample, "gpt")
        if target:
            answer_counts[target.strip()] += 1
        if human:
            image_token_counts[human.count("<image>")] += 1
        if human is None or target is None or not options:
            malformed.append({"index": index, "id": sample.get("id"), "has_human": human is not None, "has_gpt": target is not None, "option_count": len(options)})
    return {
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "row_count": len(rows),
        "unique_images": len(image_counts),
        "duplicate_image_rows": len(rows) - len(image_counts),
        "option_count_distribution": dict(sorted(option_counts.items())),
        "answer_count_sample": dict(answer_counts.most_common(30)),
        "image_token_count_distribution": dict(sorted(image_token_counts.items())),
        "image_existence_check_limit": image_check_limit,
        "image_existence_checked_rows": checked_images,
        "image_existence_full_check": image_check_limit < 0 or checked_images == len(rows),
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images,
        "malformed_count": len(malformed),
        "malformed_sample": malformed[:20],
        "first_id": rows[0].get("id") if rows else None,
        "first_image": rows[0].get("image") if rows else None,
        "first_human_sample": first_turn(rows[0], "human")[:500] if rows and first_turn(rows[0], "human") else None,
        "first_target": first_turn(rows[0], "gpt") if rows else None,
    }


def tracking_preflight(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import swanlab

        return {
            "required": True,
            "importable": True,
            "version": getattr(swanlab, "__version__", None),
            "mode": os.environ.get("SWANLAB_MODE", args.swanlab_mode),
            "log_dir": str(Path(args.output_dir) / "swanlog"),
            "status": "PASS",
        }
    except Exception as exc:
        return {
            "required": True,
            "importable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "mode": os.environ.get("SWANLAB_MODE", args.swanlab_mode),
            "log_dir": str(Path(args.output_dir) / "swanlog"),
            "status": "FAIL",
        }


def write_args_snapshot(args: argparse.Namespace, extra: dict[str, Any]) -> None:
    output_dir = Path(args.output_dir)
    write_json(
        output_dir / "args_snapshot.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "script": "src/train/train_huatuogpt_vision_sft_lora.py",
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
            },
            "canonical_prompt_output": {
                "source": "Tr53 args_snapshot extra.rollout + src/dataset/sft_dataset.py LLavaSFTDataset",
                "prompt_mode": "sft_dataset_manual",
                "system_prompt": None,
                "chat_template": False,
                "thinking_prefill": None,
                "prompt_template": "<|im_start|>user\\n{human}<|im_end|>\\n<|im_start|>assistant\\n",
                "target_template": "{gpt}<|im_end|>\\n",
                "label_mask_rule": "all prompt tokens are IGNORE_INDEX; assistant target plus <|im_end|>\\n are supervised",
                "image_token_rule": "literal <image> in the canonical human prompt is converted to IMAGE_TOKEN_INDEX=-200 for Huatuo/LLaVA multimodal insertion",
            },
            "compatibility_notes": {
                "qwen35_tr53_freeze_vision_tower": False,
                "huatuo_vision_tower_training": "frozen by repo-local CLIPVisionTower @torch.no_grad() and LoRA targets are language attention/MLP only",
                "qwen35_tr53_deepspeed": "scripts/zero3_offload.json",
                "huatuo_launcher": "torchrun DDP; compatible training hyperparameters inherited, DeepSpeed not reused because trainer is Huatuo-specific",
                "qwen35_image_min_max_pixels": [args.image_min_pixels, args.image_max_pixels],
                "huatuo_image_processor": "CLIPImageProcessor fixed 336 square preprocessing; Qwen pixel min/max recorded but not applied",
            },
            "extra": extra,
        },
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = validate_manifest(args)
    tracking = tracking_preflight(args)
    smoke_code = (
        "import llava.model.language_model.llava_qwen2; "
        "from transformers import AutoConfig, AutoTokenizer; "
        f"p={str(Path(args.model_path))!r}; "
        "cfg=AutoConfig.from_pretrained(p, local_files_only=True, trust_remote_code=False); "
        "tok=AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=False); "
        "print(type(cfg).__name__, cfg.model_type, type(tok).__name__)"
    )
    static_loader_smoke = shell_json([sys.executable, "-c", smoke_code], timeout=args.static_smoke_timeout)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "src/train/train_huatuogpt_vision_sft_lora.py",
        "git_commit": git_commit(),
        "model_path": args.model_path,
        "vision_tower_path": args.vision_tower_path,
        "manifest": manifest,
        "tracking": tracking,
        "static_loader_smoke": static_loader_smoke,
        "gpu_query": shell_json(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]),
        "compute_apps": shell_json(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]),
    }
    payload["passed"] = bool(
        manifest["row_count"] > 0
        and manifest["missing_images_count"] == 0
        and manifest["malformed_count"] == 0
        and static_loader_smoke["returncode"] == 0
        and tracking["status"] == "PASS"
    )
    write_json(output_dir / "preflight" / "preflight.json", payload)
    write_json(output_dir / "status.json", {"status": "PREFLIGHT_PASS" if payload["passed"] else "FAIL", "preflight": payload})
    return payload


def move_batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    ensure_train_imports()
    return {
        "input_ids": batch["input_ids"].to(device),
        "labels": batch["labels"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "images": batch["images"].to(device=device, dtype=dtype),
    }


def make_dataloader(args: argparse.Namespace, tokenizer: Any, image_processor: Any, *, distributed: bool) -> tuple[DataLoader, DistributedSampler | None]:
    ensure_train_imports()
    dataset = DermoMaskHuatuoSftDataset(Path(args.data_path), Path(args.image_root), tokenizer, image_processor, limit=args.limit)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=DataCollator(tokenizer),
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    return loader, sampler


def save_checkpoint(args: argparse.Namespace, model: torch.nn.Module, optimizer: torch.optim.Optimizer, state: dict[str, Any]) -> dict[str, Any]:
    ensure_train_imports()
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / f"checkpoint-{state['global_step']}"
    model_to_save = model.module if isinstance(model, DDP) else model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    model_to_save.save_pretrained(adapter_dir)
    training_state_path = ckpt_dir / "training_state.pt"
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "state": state,
            "rng_state_cpu": torch.get_rng_state(),
            "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        training_state_path,
    )
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
        old = read_json(manifest_path).get("checkpoints", [])
    old.append(entry)
    if args.save_total_limit > 0 and len(old) > args.save_total_limit:
        to_remove = old[:-args.save_total_limit]
        old = old[-args.save_total_limit :]
        for stale in to_remove:
            shutil.rmtree(stale["path"], ignore_errors=True)
    write_json(manifest_path, {"checkpoints": old, "latest": old[-1] if old else None})
    return entry


class SwanTracker:
    def __init__(self, args: argparse.Namespace, rank: int):
        self.rank = rank
        self.local_log = Path(args.output_dir) / "swanlog" / "local_tracking.jsonl"
        self.run = None
        self.mode = os.environ.get("SWANLAB_MODE", args.swanlab_mode)
        self.status_path = Path(args.output_dir) / "swanlog" / "tracking_status.json"
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
            self.run = None
            write_json(
                self.status_path,
                {
                    "status": "LOCAL_FALLBACK",
                    "backend": "jsonl",
                    "mode": self.mode,
                    "error": f"{type(exc).__name__}: {exc}",
                    "log_dir": str(Path(args.output_dir) / "swanlog"),
                    "sync_required": True,
                },
            )

    def log(self, payload: dict[str, Any], step: int) -> None:
        if self.rank != 0:
            return
        payload = dict(payload)
        payload["step"] = step
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")
        append_jsonl(self.local_log, payload)
        if self.run is not None:
            metric_payload = {
                key: value
                for key, value in payload.items()
                if isinstance(value, (int, float, bool)) and not isinstance(value, bool)
            }
            metric_payload.update({key: value for key, value in payload.items() if isinstance(value, bool)})
            self.swanlab.log(metric_payload, step=step)

    def finish(self) -> None:
        if self.rank == 0 and self.run is not None:
            try:
                self.swanlab.finish()
            except Exception:
                pass


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    ensure_train_imports()
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


def run_smoke_or_train(args: argparse.Namespace, *, dry_run: bool) -> int:
    ensure_train_imports()
    distributed, rank, local_rank, world_size, device = setup_distributed()
    output_dir = Path(args.output_dir)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    try:
        preflight_payload = run_preflight(args) if rank == 0 else None
        if distributed:
            obj = [preflight_payload]
            torch.distributed.broadcast_object_list(obj, src=0)
            preflight_payload = obj[0]
        if not preflight_payload or not preflight_payload.get("passed"):
            return 2

        tokenizer, model, image_processor, loader_diag = load_model_tokenizer_processor(args)
        if rank == 0:
            write_json(output_dir / "loader_diagnostics.json", loader_diag)
        if loader_diag["loading_missing_keys_count"] or loader_diag["loading_unexpected_keys_count"] or loader_diag["missing_vision_keys_count"]:
            if rank == 0:
                write_json(output_dir / "status.json", {"status": "FAIL", "reason": "loader_weight_key_gate_failed", "loader_diagnostics": loader_diag})
            return 10

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model, lora_diag = apply_lora(args, model)
        if rank == 0:
            write_json(output_dir / "lora_diagnostics.json", lora_diag)
            write_args_snapshot(args, {"preflight": str(output_dir / "preflight" / "preflight.json"), "loader_diagnostics": str(output_dir / "loader_diagnostics.json"), "lora_diagnostics": str(output_dir / "lora_diagnostics.json")})
        model.to(device=device, dtype=dtype)
        model.train()
        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        loader, sampler = make_dataloader(args, tokenizer, image_processor, distributed=distributed)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
        total_update_steps = args.max_steps if args.max_steps > 0 else math.ceil((len(loader) * args.num_train_epochs) / args.gradient_accumulation_steps)
        warmup_steps = int(total_update_steps * args.warmup_ratio)

        def lr_for_step(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return args.learning_rate * float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_update_steps - warmup_steps)
            return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        tracker = SwanTracker(args, rank)
        global_step = 0
        micro_step = 0
        optimizer.zero_grad(set_to_none=True)
        first_step_payload = None
        start_time = time.time()
        max_epochs = 1 if dry_run else int(math.ceil(args.num_train_epochs))
        for epoch in range(max_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch + args.seed)
            for batch in loader:
                moved = move_batch_to_device(batch, device, dtype)
                supervised = int((moved["labels"] != IGNORE_INDEX).sum().item())
                image_tokens = int((moved["input_ids"] == IMAGE_TOKEN_INDEX).sum().item())
                if supervised <= 0 or image_tokens <= 0:
                    raise RuntimeError(f"Invalid batch: supervised={supervised}, image_tokens={image_tokens}")
                outputs = model(**moved)
                loss = outputs.loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite loss: {loss.detach().float().item()}")
                (loss / args.gradient_accumulation_steps).backward()
                micro_step += 1
                if micro_step % args.gradient_accumulation_steps == 0:
                    lr = lr_for_step(global_step)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    payload = {
                        "loss": float(loss.detach().float().item()),
                        "lr": lr,
                        "epoch": epoch,
                        "global_step": global_step,
                        "micro_step": micro_step,
                        "grad_norm": float(grad_norm.detach().float().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                        "supervised_token_count": supervised,
                        "image_token_count": image_tokens,
                        "world_size": world_size,
                    }
                    tracker.log(payload, global_step)
                    if first_step_payload is None:
                        first_step_payload = payload
                        if rank == 0:
                            write_json(output_dir / "first_step.json", payload)
                    if rank == 0 and global_step % args.logging_steps == 0:
                        print(json.dumps({"event": "train_step", **payload}, ensure_ascii=False), flush=True)
                    if rank == 0 and global_step % args.save_steps == 0:
                        save_checkpoint(args, model, optimizer, {"global_step": global_step, "epoch": epoch, "micro_step": micro_step})
                    if dry_run or (args.max_steps > 0 and global_step >= args.max_steps):
                        break
            if dry_run or (args.max_steps > 0 and global_step >= args.max_steps):
                break

        if rank == 0:
            final_ckpt = save_checkpoint(args, model, optimizer, {"global_step": global_step, "epoch": max_epochs - 1, "micro_step": micro_step})
            status = {
                "status": "DRY_RUN_PASS" if dry_run else "RUNNING_COMPLETE",
                "dry_run": dry_run,
                "global_step": global_step,
                "micro_step": micro_step,
                "finite_loss": bool(first_step_payload and math.isfinite(first_step_payload["loss"])),
                "first_step": first_step_payload,
                "elapsed_seconds": time.time() - start_time,
                "latest_checkpoint": final_ckpt,
                "tracking_status": str(output_dir / "swanlog" / "tracking_status.json"),
                "checkpoint_manifest": str(output_dir / "checkpoints" / "manifest.json"),
            }
            write_json(output_dir / "status.json", status)
            write_json(output_dir / "summary.json", status)
            (output_dir / "summary.md").write_text(
                "\n".join(
                    [
                        "# HuatuoGPT-Vision-7B DermoMask SFT LoRA",
                        "",
                        f"Status: {status['status']}",
                        f"Global step: {global_step}",
                        f"Finite first loss: {status['finite_loss']}",
                        f"Checkpoint manifest: {status['checkpoint_manifest']}",
                        f"Tracking status: {status['tracking_status']}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        tracker.finish()
        return 0
    except Exception:
        if rank == 0:
            write_json(output_dir / "status.json", {"status": "FAIL", "dry_run": dry_run, "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1
    finally:
        cleanup_distributed(distributed)


def run_loader_lora_smoke(args: argparse.Namespace) -> int:
    ensure_train_imports()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        tokenizer, model, image_processor, loader_diag = load_model_tokenizer_processor(args)
        write_json(output_dir / "loader_diagnostics.json", loader_diag)
        loader_ok = bool(
            loader_diag["loading_missing_keys_count"] == 0
            and loader_diag["loading_unexpected_keys_count"] == 0
            and loader_diag["missing_vision_keys_count"] == 0
            and loader_diag["vision_tower_loaded"]
            and loader_diag["image_processor_class"]
            and loader_diag["model_class"] == "LlavaQwen2ForCausalLM"
            and loader_diag["tokenizer_class"] == "Qwen2Tokenizer"
        )
        if not loader_ok:
            write_json(output_dir / "status.json", {"status": "FAIL", "reason": "loader_diagnostics_failed", "loader_diagnostics": loader_diag})
            return 10
        model, lora_diag = apply_lora(args, model)
        write_json(output_dir / "lora_diagnostics.json", lora_diag)
        lora_ok = bool(lora_diag["trainable_param_count"] > 0 and lora_diag["matched_module_count"] > 0)
        status = {
            "status": "LOADER_LORA_SMOKE_PASS" if lora_ok else "FAIL",
            "model_class": loader_diag["model_class"],
            "config_class": loader_diag["config_class"],
            "tokenizer_class": loader_diag["tokenizer_class"],
            "image_processor_class": loader_diag["image_processor_class"],
            "vision_tower_loaded": loader_diag["vision_tower_loaded"],
            "loading_missing_keys_count": loader_diag["loading_missing_keys_count"],
            "loading_unexpected_keys_count": loader_diag["loading_unexpected_keys_count"],
            "missing_vision_keys_count": loader_diag["missing_vision_keys_count"],
            "lora_matched_module_count": lora_diag["matched_module_count"],
            "lora_trainable_param_count": lora_diag["trainable_param_count"],
            "lora_target_suffixes": lora_diag["target_suffixes"],
            "loader_diagnostics": str(output_dir / "loader_diagnostics.json"),
            "lora_diagnostics": str(output_dir / "lora_diagnostics.json"),
        }
        write_args_snapshot(
            args,
            {
                "loader_diagnostics": str(output_dir / "loader_diagnostics.json"),
                "lora_diagnostics": str(output_dir / "lora_diagnostics.json"),
                "image_processor_class": type(image_processor).__name__,
                "tokenizer_length": len(tokenizer),
            },
        )
        write_json(output_dir / "status.json", status)
        return 0 if lora_ok else 11
    except Exception:
        write_json(output_dir / "status.json", {"status": "FAIL", "reason": "loader_lora_smoke_exception", "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=1003520)
    parser.add_argument("--image-existence-check-limit", type=int, default=2000)
    parser.add_argument("--static-smoke-timeout", type=int, default=45)
    parser.add_argument("--swanlab-project", default="DermoGPT-HuatuoGPT-Vision-7B")
    parser.add_argument("--swanlab-run-name", default=None)
    parser.add_argument("--swanlab-mode", default="online", choices=["online", "local", "offline"])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--loader-lora-smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)
    if args.preflight_only:
        payload = run_preflight(args)
        write_args_snapshot(args, {"preflight": str(Path(args.output_dir) / "preflight" / "preflight.json")})
        return 0 if payload.get("passed") else 2
    if args.loader_lora_smoke_only:
        return run_loader_lora_smoke(args)
    ensure_train_imports()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    return run_smoke_or_train(args, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
