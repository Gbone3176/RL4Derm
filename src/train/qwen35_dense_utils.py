import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from importlib import metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SWANLAB_PROJECT_ENV_BACKUP = "DERMOGPT_ORIGINAL_SWANLAB_PROJECT"

try:
    from paths import configured_dataset_path, configured_model_path, resolve_project_path
except ImportError:  # pragma: no cover - supports src.* imports in eval scripts.
    from src.paths import configured_dataset_path, configured_model_path, resolve_project_path


DEFAULT_QWEN35_4B_PATH = str(configured_model_path("qwen35_4b"))
FALLBACK_QWEN35_4B_PATH = str(configured_model_path("qwen35_4b_fallback"))
DEFAULT_SFT_DATA_PATH = str(resolve_project_path("data/dermoinstruct_mcqa_train_10k.json"))
DEFAULT_DERMOINSTRUCT_ROOT = str(configured_dataset_path("dermoinstruct_root"))


class Qwen35DependencyError(RuntimeError):
    """Raised when the installed dependency set cannot safely load Qwen3.5 dense."""


@dataclass(frozen=True)
class Qwen35Compatibility:
    model_path: str
    transformers_version: str
    model_type: str | None
    architectures: list[str]
    processor_class: str | None
    model_loader: str | None


def existing_qwen35_4b_path() -> str:
    for candidate in (DEFAULT_QWEN35_4B_PATH, FALLBACK_QWEN35_4B_PATH):
        if Path(candidate).exists():
            return candidate
    return DEFAULT_QWEN35_4B_PATH


def is_qwen35_dense_identifier(model_id: str | os.PathLike[str] | None) -> bool:
    if model_id is None:
        return False
    text = str(model_id).lower()
    return any(token in text for token in ("qwen3.5", "qwen3___5", "qwen3_5", "qwen3-5"))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_qwen35_raw_metadata(model_id: str | os.PathLike[str]) -> dict[str, Any]:
    model_path = Path(model_id)
    config = _read_json(model_path / "config.json") if (model_path / "config.json").exists() else {}
    preprocessor = (
        _read_json(model_path / "preprocessor_config.json")
        if (model_path / "preprocessor_config.json").exists()
        else {}
    )
    return {
        "model_path": str(model_id),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "processor_class": preprocessor.get("processor_class"),
        "config": config,
        "preprocessor": preprocessor,
    }


def validate_qwen35_raw_config(model_id: str | os.PathLike[str]) -> dict[str, Any]:
    metadata_dict = read_qwen35_raw_metadata(model_id)
    model_type = metadata_dict.get("model_type")
    architectures = metadata_dict.get("architectures") or []
    if model_type != "qwen3_5" or "Qwen3_5ForConditionalGeneration" not in architectures:
        raise ValueError(
            "Qwen3.5 dense expected model_type='qwen3_5' and "
            "architectures containing 'Qwen3_5ForConditionalGeneration'; "
            f"got model_type={model_type!r}, architectures={architectures!r}"
        )
    return metadata_dict


def _transformers_version() -> str:
    try:
        return metadata.version("transformers")
    except metadata.PackageNotFoundError:
        return "MISSING"


def _qwen35_model_loader_name() -> str | None:
    try:
        from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401

        return "Qwen3_5ForConditionalGeneration"
    except Exception:
        pass

    try:
        from transformers import AutoModelForImageTextToText  # noqa: F401

        return "AutoModelForImageTextToText"
    except Exception:
        return None


def check_qwen35_compatibility(model_id: str | os.PathLike[str]) -> Qwen35Compatibility:
    metadata_dict = validate_qwen35_raw_config(model_id)
    model_loader = _qwen35_model_loader_name()

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            str(model_id),
            local_files_only=Path(model_id).exists(),
            trust_remote_code=False,
        )
    except Exception as exc:
        raise Qwen35DependencyError(
            "Installed transformers cannot read Qwen3.5 dense config "
            f"(transformers=={_transformers_version()}, model_type=qwen3_5). "
            "Minimum upgrade target for this repository is transformers>=5.4.0; "
            "rerun import/config/processor smoke after upgrade. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if getattr(cfg, "model_type", None) != "qwen3_5":
        raise Qwen35DependencyError(
            f"AutoConfig loaded unexpected model_type={getattr(cfg, 'model_type', None)!r}; "
            "refusing to route Qwen3.5 dense through a Qwen3-VL class."
        )
    if model_loader is None:
        raise Qwen35DependencyError(
            "Installed transformers exposes neither Qwen3_5ForConditionalGeneration nor "
            "AutoModelForImageTextToText. Upgrade transformers before Qwen3.5 dense training."
        )

    return Qwen35Compatibility(
        model_path=str(model_id),
        transformers_version=_transformers_version(),
        model_type=metadata_dict.get("model_type"),
        architectures=list(metadata_dict.get("architectures") or []),
        processor_class=metadata_dict.get("processor_class"),
        model_loader=model_loader,
    )


def load_qwen35_dense_model(model_id: str | os.PathLike[str], **from_pretrained_kwargs: Any):
    compatibility = check_qwen35_compatibility(model_id)
    if compatibility.model_loader == "Qwen3_5ForConditionalGeneration":
        from transformers import Qwen3_5ForConditionalGeneration

        model_cls = Qwen3_5ForConditionalGeneration
    elif compatibility.model_loader == "AutoModelForImageTextToText":
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    else:
        raise Qwen35DependencyError("No safe Qwen3.5 dense model loader is available.")

    return model_cls.from_pretrained(str(model_id), **from_pretrained_kwargs)


def load_qwen35_processor(model_id: str | os.PathLike[str], **from_pretrained_kwargs: Any):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_id), **from_pretrained_kwargs)
    return patch_qwen35_processor_apply_chat_template(processor)


def patch_qwen35_processor_apply_chat_template(processor: Any) -> Any:
    """Move tokenizer/processor kwargs into ``processor_kwargs`` for Qwen3.5.

    Transformers 5.x warns when ``apply_chat_template(tokenize=True, padding=..., return_tensors=...)``
    forwards processor call arguments through ``**kwargs``. TRL 0.28 still uses that older call shape in
    GRPOTrainer. Keep the compatibility patch local to processors loaded through the Qwen3.5 entrypoint.
    """
    if getattr(processor, "_dermogpt_qwen35_processor_kwargs_patch", False):
        return processor
    original = getattr(processor, "apply_chat_template", None)
    if original is None:
        return processor

    processor_kwarg_names = {
        "padding",
        "padding_side",
        "add_special_tokens",
        "truncation",
        "max_length",
        "return_attention_mask",
    }

    def apply_chat_template_with_processor_kwargs(*args: Any, **kwargs: Any):
        if kwargs.get("tokenize") is True:
            moved = {key: kwargs.pop(key) for key in list(kwargs) if key in processor_kwarg_names}
            if moved:
                processor_kwargs = dict(kwargs.pop("processor_kwargs", {}) or {})
                processor_kwargs.update(moved)
                kwargs["processor_kwargs"] = processor_kwargs
        if (
            kwargs.get("enable_thinking") is False
            and "enable_thinking" not in str(getattr(processor, "chat_template", "") or "")
        ):
            # After the no-default-thinking template patch, ``enable_thinking`` is no
            # longer a Jinja variable. Passing it through makes Transformers treat it
            # as a processor kwarg and warn, while the rendered prompt is unchanged.
            kwargs.pop("enable_thinking")
        return original(*args, **kwargs)

    processor.apply_chat_template = apply_chat_template_with_processor_kwargs
    processor._dermogpt_qwen35_processor_kwargs_patch = True
    return processor


def sanitize_swanlab_project_env() -> str | None:
    """Remove malformed-prone SWANLAB_PROJECT before importing swanlab.

    SwanLab/pydantic can parse this environment variable at import time. The harness passes project names
    explicitly to ``swanlab.init(project=...)`` instead, so an externally mis-set plain string must not poison import.
    """
    value = os.environ.pop("SWANLAB_PROJECT", None)
    if value is not None:
        os.environ[_SWANLAB_PROJECT_ENV_BACKUP] = value
    return value


_QWEN35_DEFAULT_THINKING_GENERATION_BLOCK = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}
{%- endif %}"""

_QWEN35_NO_THINKING_PREFILL_GENERATION_BLOCK = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""


def apply_qwen35_no_default_thinking_prefill(processor: Any) -> dict[str, Any]:
    """Patch Qwen3.5 chat template so generation starts at assistant header only.

    Qwen3.5's bundled template injects either ``<think>\n`` by default or
    ``<think>\n\n</think>\n\n`` when ``enable_thinking=False``. For GRPO we want
    those tags to be generated by the completion itself, not prefilled into the
    prompt.
    """
    original_template = getattr(processor, "chat_template", None) or getattr(
        getattr(processor, "tokenizer", None), "chat_template", None
    )
    if not isinstance(original_template, str):
        raise ValueError("processor/tokenizer does not expose a string chat_template")

    if _QWEN35_DEFAULT_THINKING_GENERATION_BLOCK in original_template:
        patched_template = original_template.replace(
            _QWEN35_DEFAULT_THINKING_GENERATION_BLOCK,
            _QWEN35_NO_THINKING_PREFILL_GENERATION_BLOCK,
        )
        strategy = "literal_generation_block_replace"
    else:
        patched_template, count = re.subn(
            r"\{%- if add_generation_prompt %\}\s*"
            r"\{\{- '<\|im_start\|>assistant\\n' \}\}\s*"
            r"\{%- if enable_thinking is defined and enable_thinking is false %\}\s*"
            r"\{\{- '<think>\\n\\n</think>\\n\\n' \}\}\s*"
            r"\{%- else %\}\s*"
            r"\{\{- '<think>\\n' \}\}\s*"
            r"\{%- endif %\}\s*"
            r"\{%- endif %\}",
            _QWEN35_NO_THINKING_PREFILL_GENERATION_BLOCK,
            original_template,
            count=1,
        )
        if count != 1:
            raise ValueError("could not identify Qwen3.5 generation thinking-prefill block in chat_template")
        strategy = "regex_generation_block_replace"

    processor.chat_template = patched_template
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.chat_template = patched_template
    return {
        "strategy": strategy,
        "original_template_len": len(original_template),
        "patched_template_len": len(patched_template),
        "removed_default_thinking_prefill": True,
    }


def write_no_default_thinking_prefill_preflight(
    output_dir: str | os.PathLike[str],
    processor: Any,
    prompt: list[dict[str, Any]],
    *,
    strategy: dict[str, Any],
) -> Path:
    preflight_dir = Path(output_dir) / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    rendered_default = processor.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    rendered_enable_false = processor.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    forbidden = {
        "default_think_prefill": "<|im_start|>assistant\n<think>\n",
        "empty_think_prefill": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    }
    assistant_start = "<|im_start|>assistant\n"
    checks = {
        "default_render_ends_with_assistant_header_only": rendered_default.endswith(assistant_start),
        "enable_false_render_ends_with_assistant_header_only": rendered_enable_false.endswith(assistant_start),
        "default_render_has_no_think_after_assistant_start": forbidden["default_think_prefill"] not in rendered_default,
        "enable_false_render_has_no_empty_think_prefill": forbidden["empty_think_prefill"] not in rendered_enable_false,
        "mcqa_prompt_has_no_system_role": not any(message.get("role") == "system" for message in prompt),
    }
    passed = all(checks.values())
    evidence = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": strategy,
        "prompt": prompt,
        "rendered_default": rendered_default,
        "rendered_enable_thinking_false": rendered_enable_false,
        "rendered_default_tail": rendered_default[-240:],
        "rendered_enable_thinking_false_tail": rendered_enable_false[-240:],
        "forbidden_prefills": forbidden,
        "checks": checks,
        "passed": passed,
    }
    evidence_path = preflight_dir / "chat_template_no_default_think.json"
    with evidence_path.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2, sort_keys=True)
    if not passed:
        raise RuntimeError(f"no-default-thinking-prefill preflight failed; see {evidence_path}")
    print(f"no_default_thinking_prefill_preflight={evidence_path}")
    print(f"no_default_thinking_prefill_tail={rendered_default[-120:]!r}")
    return evidence_path


def compatibility_smoke(model_id: str | os.PathLike[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"model_path": str(model_id)}
    result.update(read_qwen35_raw_metadata(model_id))
    result["transformers_version"] = _transformers_version()
    result["model_loader"] = _qwen35_model_loader_name()

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            str(model_id),
            local_files_only=Path(model_id).exists(),
            trust_remote_code=False,
        )
        result["auto_config"] = {
            "ok": True,
            "class": type(cfg).__name__,
            "model_type": getattr(cfg, "model_type", None),
            "architectures": getattr(cfg, "architectures", None),
        }
    except Exception as exc:
        result["auto_config"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        processor = load_qwen35_processor(
            model_id,
            local_files_only=Path(model_id).exists(),
            trust_remote_code=False,
        )
        result["auto_processor"] = {"ok": True, "class": type(processor).__name__}
    except Exception as exc:
        result["auto_processor"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return result


def compatibility_smoke_ok(result: dict[str, Any]) -> bool:
    return (
        result.get("model_type") == "qwen3_5"
        and "Qwen3_5ForConditionalGeneration" in (result.get("architectures") or [])
        and result.get("model_loader") is not None
        and result.get("auto_config", {}).get("ok") is True
        and result.get("auto_processor", {}).get("ok") is True
    )


def resolve_exp_dir(base_model_name: str, exp_id: str, exp_root: str | os.PathLike[str]) -> Path:
    explicit = os.environ.get("EXP_DIR")
    if explicit:
        return Path(explicit)
    return Path(exp_root) / base_model_name / exp_id


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_args_snapshot(output_dir: str | os.PathLike[str], args: argparse.Namespace, extra: dict[str, Any]) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(extra.get("script", "")).name,
        "git_commit": git_commit(),
        "args": vars(args),
        "env": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "DERMOGPT_LOG_ROLLOUTS": os.environ.get("DERMOGPT_LOG_ROLLOUTS"),
            "EXP_DIR": os.environ.get("EXP_DIR"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "MODELSCOPE_CACHE": os.environ.get("MODELSCOPE_CACHE"),
            "DERMOINSTRUCT_TRAIN_JSON": os.environ.get("DERMOINSTRUCT_TRAIN_JSON"),
            "SWANLAB_LOG_DIR": os.environ.get("SWANLAB_LOG_DIR"),
            "SWANLAB_MODE": os.environ.get("SWANLAB_MODE"),
        },
        "extra": extra,
    }
    snapshot_path = output_path / "args_snapshot.json"
    with snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return snapshot_path


def _report_to_requires_swanlab(report_to: Any) -> bool:
    if report_to is None:
        return False
    if isinstance(report_to, str):
        return report_to.lower() == "swanlab"
    try:
        return any(str(item).lower() == "swanlab" for item in report_to)
    except TypeError:
        return False


def _is_world_process_zero() -> bool:
    return os.environ.get("RANK", "0") in ("0", "-1")


def ensure_swanlab_active_run(training_args: Any, *, default_project: str = "DermoGPT-Qwen3.5") -> Path | None:
    """Create a real SwanLab run before Transformers' callback checks get_run()."""
    if not _report_to_requires_swanlab(getattr(training_args, "report_to", None)):
        return None
    if not _is_world_process_zero():
        return None

    mode = os.environ.get("SWANLAB_MODE") or "offline"
    if mode == "cloud":
        mode = "online"
    if mode == "disabled":
        raise RuntimeError(
            "SwanLab is required by the DermoGPT harness, but SWANLAB_MODE=disabled was set."
        )
    if mode not in {"online", "local", "offline"}:
        raise RuntimeError(
            "SwanLab mode must be one of online/local/offline for harness logging; "
            f"got SWANLAB_MODE={mode!r}."
        )

    output_dir = Path(getattr(training_args, "output_dir"))
    log_dir = Path(os.environ.get("SWANLAB_LOG_DIR") or output_dir / "swanlog")
    log_dir.mkdir(parents=True, exist_ok=True)

    removed_project = sanitize_swanlab_project_env()
    if removed_project is not None:
        print("swanlab_project_env_sanitized=1", flush=True)

    import swanlab

    try:
        run = swanlab.get_run()
    except RuntimeError:
        run = None
    if run is None:
        run = swanlab.init(
            mode=mode,
            log_dir=str(log_dir),
            project=default_project,
            name=getattr(training_args, "run_name", None),
            config={
                "output_dir": str(output_dir),
                "max_steps": getattr(training_args, "max_steps", None),
                "report_to": "swanlab",
            },
        )
    try:
        active_id = getattr(swanlab.get_run(), "id", None)
    except RuntimeError as exc:
        raise RuntimeError("SwanLab init returned but no active run is available.") from exc
    print(f"swanlab_active_run mode={mode} log_dir={log_dir} run_id={active_id}")
    return log_dir


def add_common_qwen35_args(parser: argparse.ArgumentParser, *, grpo: bool = False) -> None:
    default_model = existing_qwen35_4b_path()
    parser.add_argument("--model_id", default=default_model)
    parser.add_argument("--base_model_name", default="Qwen3.5-4B")
    parser.add_argument("--exp_id", default=("qwen35_grpo_lora_dryrun" if grpo else "qwen35_sft_lora_dryrun"))
    parser.add_argument("--exp_root", default=str(PROJECT_ROOT / "experiment"))
    parser.add_argument("--data_path", default=DEFAULT_SFT_DATA_PATH)
    parser.add_argument("--image_folder", "--image_root", dest="image_folder", default=DEFAULT_DERMOINSTRUCT_ROOT)
    parser.add_argument("--deepspeed", default="scripts/zero3_offload.json")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_lora_modules", type=int, default=-1)
    parser.add_argument("--lora_namespan_exclude", default="['lm_head', 'embed_tokens']")
    parser.add_argument("--freeze_llm", action="store_true", default=True)
    parser.add_argument("--unfreeze_llm", dest="freeze_llm", action="store_false")
    parser.add_argument("--freeze_vision_tower", action="store_true", default=False)
    parser.add_argument("--freeze_merger", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--disable_flash_attn2", action="store_true", default=False)
    parser.add_argument("--image_min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--image_max_pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--vision_lr", type=float, default=2e-6)
    parser.add_argument("--merger_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--swanlab_project", default="DermoGPT-Qwen3.5")
    parser.add_argument("--swanlab_run_name", default=None)
    parser.add_argument("--qwen35_no_default_thinking_prefill", action="store_true", default=True)
    parser.add_argument(
        "--allow_qwen35_default_thinking_prefill",
        dest="qwen35_no_default_thinking_prefill",
        action="store_false",
    )
    parser.add_argument("--dry_run_plan", action="store_true")
    parser.add_argument("--compatibility_smoke", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
