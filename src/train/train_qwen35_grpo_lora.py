import argparse
import json
import shlex
from pathlib import Path

from train.qwen35_dense_utils import (
    Qwen35DependencyError,
    add_common_qwen35_args,
    apply_qwen35_no_default_thinking_prefill,
    check_qwen35_compatibility,
    compatibility_smoke,
    compatibility_smoke_ok,
    ensure_swanlab_active_run,
    load_qwen35_dense_model,
    load_qwen35_processor,
    resolve_exp_dir,
    sanitize_swanlab_project_env,
    write_no_default_thinking_prefill_preflight,
    write_args_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5 dense GRPO LoRA entrypoint with dependency gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_qwen35_args(parser, grpo=True)
    parser.add_argument("--reward_module", default="train.reward_funcs")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument(
        "--init_lora_path",
        default=None,
        help="Optional PEFT LoRA adapter checkpoint/directory used to initialize trainable GRPO LoRA weights.",
    )
    return parser


def dry_run_command(args: argparse.Namespace, output_dir: Path) -> str:
    parts = [
        "python", "-m", "train.train_qwen35_grpo_lora",
        "--model_id", args.model_id,
        "--data_path", args.data_path,
        "--image_folder", args.image_folder,
        "--base_model_name", args.base_model_name,
        "--exp_id", args.exp_id,
        "--max_steps", str(args.max_steps),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--save_steps", str(args.save_steps),
        "--save_total_limit", str(args.save_total_limit),
    ]
    return (
        "PYTHONPATH=src:${PYTHONPATH:-} "
        + " ".join(shlex.quote(part) for part in parts)
        + f" 2>&1 | tee {shlex.quote(str(output_dir / 'train.log'))}"
    )


def print_dry_run_plan(args: argparse.Namespace, output_dir: Path) -> None:
    print(json.dumps(
        {
            "status": "dry_run_plan_only",
            "task": "qwen35_dense_grpo_lora",
            "model_id": args.model_id,
            "base_model_name": args.base_model_name,
            "exp_id": args.exp_id,
            "data_path": args.data_path,
            "image_folder": args.image_folder,
            "exp_dir": str(output_dir),
            "logs": {
                "train_log": str(output_dir / "train.log"),
                "stdout_stderr": str(output_dir / "train.log"),
            },
            "args_snapshot": str(output_dir / "args_snapshot.json"),
            "reward_module": args.reward_module,
            "init_lora_path": args.init_lora_path,
            "no_default_thinking_prefill": {
                "required": bool(args.qwen35_no_default_thinking_prefill),
                "policy": "assistant generation prompt must end at '<|im_start|>assistant\\n'; completion must generate <think>...</think><answer>...</answer> itself.",
            },
            "swanlab": {
                "required": True,
                "report_to": "swanlab",
                "project": args.swanlab_project,
                "run_name": args.swanlab_run_name or args.exp_id,
            },
            "checkpoint_policy": {
                "save_strategy": "steps",
                "save_steps": args.save_steps,
                "save_total_limit": args.save_total_limit,
            },
            "dependency_gate": "Qwen3.5 AutoConfig/model loader and TRL GRPOTrainer must import before model load.",
            "one_batch_dry_run_command": dry_run_command(args, output_dir),
            "note": "Submit this plan to conductor before launching. Do not run without explicit training authorization.",
        },
        ensure_ascii=False,
        indent=2,
    ))


def grpo_dependency_smoke() -> dict[str, object]:
    result: dict[str, object] = {}
    try:
        from src.trainer.grpo_trainer import patch_trl_optional_weave_gate

        patch_trl_optional_weave_gate()
        result["trl_optional_weave_gate"] = {"ok": True}
    except Exception as exc:
        result["trl_optional_weave_gate"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        from peft import LoraConfig, get_peft_model  # noqa: F401

        result["peft"] = {"ok": True}
    except Exception as exc:
        result["peft"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        from trl import GRPOConfig  # noqa: F401

        result["trl_grpo_config"] = {"ok": True}
    except Exception as exc:
        result["trl_grpo_config"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        from trl import GRPOTrainer  # noqa: F401

        result["trl_grpo_trainer"] = {"ok": True}
    except Exception as exc:
        result["trl_grpo_trainer"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return result


def grpo_dependency_smoke_ok(result: dict[str, object]) -> bool:
    return all(isinstance(value, dict) and value.get("ok") is True for value in result.values())


def build_grpo_config_kwargs(args: argparse.Namespace, output_dir: Path, grpo_config_cls) -> tuple[dict[str, object], list[str]]:
    requested_kwargs = {
        "output_dir": str(output_dir),
        "run_name": args.swanlab_run_name or args.exp_id,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "max_steps": args.max_steps,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": ["swanlab"],
        "seed": args.seed,
        "beta": args.beta,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    supported_fields = getattr(grpo_config_cls, "__dataclass_fields__", None)
    if supported_fields is None:
        raise TypeError("GRPOConfig does not expose __dataclass_fields__; cannot safely filter version-specific kwargs.")

    supported_keys = set(supported_fields)
    unsupported_keys = sorted(key for key in requested_kwargs if key not in supported_keys)
    filtered_kwargs = {key: value for key, value in requested_kwargs.items() if key in supported_keys}
    return filtered_kwargs, unsupported_keys


def main() -> int:
    sanitize_swanlab_project_env()
    parser = build_parser()
    args = parser.parse_args()
    output_dir = resolve_exp_dir(args.base_model_name, args.exp_id, args.exp_root)

    if args.dry_run_plan:
        print_dry_run_plan(args, output_dir)
        return 0

    if args.compatibility_smoke:
        result = compatibility_smoke(args.model_id)
        result["grpo_dependency_gate"] = grpo_dependency_smoke()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if compatibility_smoke_ok(result) and grpo_dependency_smoke_ok(result["grpo_dependency_gate"]) else 1

    try:
        compatibility = check_qwen35_compatibility(args.model_id)
    except Qwen35DependencyError as exc:
        raise SystemExit(f"Qwen3.5 dense dependency gate failed before GRPO model load: {exc}") from exc

    try:
        from src.trainer.grpo_trainer import patch_trl_optional_weave_gate

        patch_trl_optional_weave_gate()
        from peft import LoraConfig, PeftModel, get_peft_model
        from trl import GRPOConfig
        from src.dataset import make_grpo_data_module_derm
        from src.trainer import QwenGRPOTrainer
        from src.trainer.grpo_trainer import write_grpo_run_artifacts
        from utils import load_reward_funcs
    except Exception as exc:
        raise SystemExit(
            "GRPO dependency gate failed: TRL GRPOConfig/GRPOTrainer, PEFT, dataset, "
            f"or reward functions could not import cleanly. {type(exc).__name__}: {exc}"
        ) from exc

    grpo_config_kwargs, unsupported_grpo_config_keys = build_grpo_config_kwargs(args, output_dir, GRPOConfig)
    print(f"unsupported_grpo_config_keys={unsupported_grpo_config_keys}")

    processor = load_qwen35_processor(args.model_id)
    no_default_thinking_prefill = {
        "required": bool(args.qwen35_no_default_thinking_prefill),
        "enabled": False,
        "policy": (
            "Do not prefill Qwen3.5 default '<think>\\n' or "
            "enable_thinking=False empty '<think>\\n\\n</think>\\n\\n'; completion must generate tags itself."
        ),
    }
    if args.qwen35_no_default_thinking_prefill:
        no_default_thinking_prefill.update(apply_qwen35_no_default_thinking_prefill(processor))
        no_default_thinking_prefill["enabled"] = True

    data_args = argparse.Namespace(
        data_path=args.data_path,
        image_folder=args.image_folder,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        video_min_pixels=100352,
        video_max_pixels=602112,
        image_resized_width=None,
        image_resized_height=None,
        video_resized_width=None,
        video_resized_height=None,
        fps=None,
        nframes=None,
    )
    data_module = make_grpo_data_module_derm(args.model_id, processor, data_args)
    preflight_path = None
    if args.qwen35_no_default_thinking_prefill:
        first_prompt = data_module["train_dataset"][0]["prompt"]
        preflight_path = write_no_default_thinking_prefill_preflight(
            output_dir,
            processor,
            first_prompt,
            strategy=no_default_thinking_prefill,
        )
        no_default_thinking_prefill["preflight_evidence"] = str(preflight_path)

    snapshot_path = write_args_snapshot(
        output_dir,
        args,
        {
            "script": __file__,
            "pipeline": "qwen35_dense_grpo_lora",
            "qwen35_compatibility": compatibility.__dict__,
            "train_log": str(output_dir / "train.log"),
            "report_to": "swanlab",
            "init_lora_path": args.init_lora_path,
            "unsupported_grpo_config_keys": unsupported_grpo_config_keys,
            "grpo_config_kwargs_keys": sorted(grpo_config_kwargs),
            "grpo_config_intent": {
                "max_prompt_length": args.max_prompt_length,
                "warmup_ratio_cli": args.warmup_ratio,
                "warmup_steps_passed": grpo_config_kwargs.get("warmup_steps"),
            },
            "no_default_thinking_prefill": no_default_thinking_prefill,
            "preflight_chat_template_evidence": str(preflight_path) if preflight_path else None,
        },
    )
    print(f"args_snapshot={snapshot_path}")

    model = load_qwen35_dense_model(
        args.model_id,
        dtype="auto",
        attn_implementation="flash_attention_2" if not args.disable_flash_attn2 else "sdpa",
    )
    if not hasattr(model, "warnings_issued"):
        # TRL expects PreTrainedModel.warnings_issued; current Qwen3.5 dense loader
        # in this environment does not expose it before PEFT wrapping.
        model.warnings_issued = {}

    if args.init_lora_path:
        init_lora_path = Path(args.init_lora_path)
        if not init_lora_path.exists():
            raise SystemExit(f"--init_lora_path does not exist: {init_lora_path}")
        model = PeftModel.from_pretrained(model, str(init_lora_path), is_trainable=True)
        print(f"init_lora_path={init_lora_path} trainable_adapter=True")
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)
    reward_funcs = load_reward_funcs(args.reward_module)

    grpo_args = GRPOConfig(**grpo_config_kwargs)
    trainer = QwenGRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=reward_funcs,
        processing_class=processor,
        **data_module,
    )
    ensure_swanlab_active_run(grpo_args, default_project=args.swanlab_project)
    trainer.train()
    artifacts = write_grpo_run_artifacts(output_dir)
    print(f"grpo_run_artifacts={json.dumps(artifacts, ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
