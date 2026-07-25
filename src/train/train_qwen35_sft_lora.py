import argparse
import json
import shlex
import sys
from pathlib import Path

from train.qwen35_dense_utils import (
    add_common_qwen35_args,
    compatibility_smoke,
    compatibility_smoke_ok,
    resolve_exp_dir,
    write_args_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5 dense SFT LoRA entrypoint with DermoGPT harness checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_qwen35_args(parser, grpo=False)
    parser.add_argument("--lazy_preprocess", action="store_true", default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--sft_rollout_every_steps", type=int, default=0)
    parser.add_argument("--sft_rollout_max_new_tokens", type=int, default=64)
    return parser


def build_train_sft_argv(args: argparse.Namespace, output_dir: Path) -> list[str]:
    run_name = args.swanlab_run_name or args.exp_id
    argv = [
        "src/train/train_sft.py",
        "--use_liger", "False",
        "--lora_enable", "True",
        "--use_dora", "False",
        "--lora_namespan_exclude", args.lora_namespan_exclude,
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--num_lora_modules", str(args.num_lora_modules),
        "--model_id", args.model_id,
        "--data_path", args.data_path,
        "--image_folder", args.image_folder,
        "--remove_unused_columns", "False",
        "--freeze_vision_tower", str(args.freeze_vision_tower),
        "--freeze_llm", str(args.freeze_llm),
        "--freeze_merger", str(args.freeze_merger),
        "--bf16", str(args.bf16),
        "--fp16", str(args.fp16),
        "--disable_flash_attn2", str(args.disable_flash_attn2),
        "--output_dir", str(output_dir),
        "--run_name", run_name,
        "--num_train_epochs", str(args.num_train_epochs),
        "--max_steps", str(args.max_steps),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--image_min_pixels", str(args.image_min_pixels),
        "--image_max_pixels", str(args.image_max_pixels),
        "--learning_rate", str(args.learning_rate),
        "--merger_lr", str(args.merger_lr),
        "--vision_lr", str(args.vision_lr),
        "--weight_decay", str(args.weight_decay),
        "--warmup_ratio", str(args.warmup_ratio),
        "--lr_scheduler_type", "cosine",
        "--logging_steps", str(args.logging_steps),
        "--tf32", "True",
        "--gradient_checkpointing", "True",
        "--report_to", "swanlab",
        "--lazy_preprocess", str(args.lazy_preprocess),
        "--save_strategy", "steps",
        "--save_steps", str(args.save_steps),
        "--save_total_limit", str(args.save_total_limit),
        "--dataloader_num_workers", str(args.dataloader_num_workers),
        "--seed", str(args.seed),
        "--sft_rollout_every_steps", str(args.sft_rollout_every_steps),
        "--sft_rollout_max_new_tokens", str(args.sft_rollout_max_new_tokens),
    ]
    if args.deepspeed and str(args.deepspeed).lower() not in {"none", "null", "false", "0"}:
        model_id_index = argv.index("--model_id")
        argv[model_id_index:model_id_index] = ["--deepspeed", args.deepspeed]
    return argv


def print_dry_run_plan(args: argparse.Namespace, output_dir: Path) -> None:
    argv = build_train_sft_argv(args, output_dir)
    command = (
        "PYTHONPATH=src:${PYTHONPATH:-} deepspeed "
        + " ".join(shlex.quote(part) for part in argv)
        + f" 2>&1 | tee {shlex.quote(str(output_dir / 'train.log'))}"
    )
    print(json.dumps(
        {
            "status": "dry_run_plan_only",
            "task": "qwen35_dense_sft_lora",
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
            "one_batch_dry_run_command": command,
            "note": "Submit this plan to conductor before launching. Do not run without explicit training authorization.",
        },
        ensure_ascii=False,
        indent=2,
    ))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = resolve_exp_dir(args.base_model_name, args.exp_id, args.exp_root)

    if args.dry_run_plan:
        print_dry_run_plan(args, output_dir)
        return 0

    if args.compatibility_smoke:
        result = compatibility_smoke(args.model_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if compatibility_smoke_ok(result) else 1

    snapshot_path = write_args_snapshot(
        output_dir,
        args,
        {
            "script": __file__,
            "pipeline": "qwen35_dense_sft_lora",
            "data_format": "LLaVA/Qwen-VL JSON array",
            "train_log": str(output_dir / "train.log"),
            "report_to": "swanlab",
            "rollout": {
                "enabled": args.sft_rollout_every_steps > 0,
                "every_steps": args.sft_rollout_every_steps,
                "do_sample": False,
                "num_beams": 1,
                "eos_token": "<|im_end|>",
                "max_new_tokens": args.sft_rollout_max_new_tokens,
                "prompt_mode": "sft_dataset_manual",
                "system_prompt": None,
                "chat_template": False,
                "thinking_prefill": None,
            },
        },
    )
    print(f"args_snapshot={snapshot_path}")

    sys.argv = build_train_sft_argv(args, output_dir)
    from train.train_sft import train

    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
