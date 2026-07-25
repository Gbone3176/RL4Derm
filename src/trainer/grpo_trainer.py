import json
import os
from datetime import datetime
from pathlib import Path


def patch_trl_optional_weave_gate() -> None:
    try:
        import importlib.util
        import trl.import_utils as import_utils
    except Exception:
        return

    if importlib.util.find_spec("weave") is None:
        import_utils.is_weave_available = lambda: False
    if importlib.util.find_spec("mergekit") is None:
        import_utils.is_mergekit_available = lambda: False
    if importlib.util.find_spec("llm_blender") is None:
        import_utils.is_llm_blender_available = lambda: False


def get_qwen35_compatible_grpo_trainer_cls():
    patch_trl_optional_weave_gate()

    import torch
    from trl.data_utils import apply_chat_template, prepare_multimodal_messages
    from trl.trainer.grpo_trainer import (
        FSDP,
        GRPOTrainer,
        entropy_from_logits,
        is_conversational,
        nullcontext,
        profiling_context,
        selective_log_softmax,
        unwrap_model_for_generation,
    )

    class Qwen35CompatibleGRPOTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            if "mm_token_type_ids" not in output:
                mm_token_type_ids = self._build_mm_token_type_ids_for_loss(inputs, output["completion_ids"])
                if mm_token_type_ids is not None:
                    output["mm_token_type_ids"] = mm_token_type_ids
            self._write_rollout_debug(inputs, output)
            return output

        def _generate_single_turn(self, prompts):
            """TRL 0.28 regular generation path without deprecated ``disable_compile`` mix.

            Upstream passes ``generation_config=...`` together with ``disable_compile=True``, which
            Transformers 5.4 logs as a deprecation warning on every rollout. Qwen3.5 GRPO here uses
            the regular Transformers path, so keep upstream behavior except for omitting that extra
            generation kwarg. Non-regular paths fall back to upstream unchanged.
            """
            if getattr(self, "use_vllm", False) or getattr(self, "use_transformers_paged", False):
                return super()._generate_single_turn(prompts)

            device = self.accelerator.device
            if is_conversational({"prompt": prompts[0]}):
                generate_inputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    tools=self.tools,
                    chat_template=self.chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                generate_inputs = self.processing_class(
                    text=prompts, padding=True, padding_side="left", return_tensors="pt"
                )
            # ``GRPOTrainer._prepare_inputs`` is the batch-generation entrypoint in TRL:
            # calling it here recursively re-enters ``_generate_and_score_completions``.
            # We only need the base Trainer tensor/device move for processor outputs.
            generate_inputs = self._prepare_input(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config
                )

            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
            return prompt_ids, completion_ids, None, {}

        def _write_rollout_debug(self, inputs, output):
            if os.environ.get("DERMOGPT_LOG_ROLLOUTS") != "1":
                return
            rank = getattr(self.accelerator, "process_index", 0)
            if rank != 0:
                return

            output_dir = Path(self.args.output_dir)
            rollout_dir = output_dir / "rollouts"
            rollout_dir.mkdir(parents=True, exist_ok=True)
            rollout_path = rollout_dir / f"rollouts_rank{rank}.jsonl"

            completion_ids = output["completion_ids"]
            completion_mask = output.get("completion_mask")
            prompt_ids = output.get("prompt_ids")
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            prompts = (
                self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
                if prompt_ids is not None
                else [self._summarize_prompt(example.get("prompt")) for example in inputs]
            )
            advantages = output.get("advantages")
            mode = "train" if self.model.training else "eval"
            reward_metrics = {
                key: values[-1]
                for key, values in self._metrics.get(mode, {}).items()
                if values and (key == "reward" or key.startswith("rewards/"))
            }
            try:
                from train.reward_funcs import analyze_format_completion
            except Exception:
                analyze_format_completion = None

            num_generations = self.num_generations if mode == "train" else self.num_generations_eval
            eos_and_pad = {self.eos_token_id, self.pad_token_id}
            rows = []
            for sample_index, completion_text in enumerate(completions):
                mask_row = completion_mask[sample_index] if completion_mask is not None else None
                completion_len = int(mask_row.sum().item()) if mask_row is not None else int(completion_ids.size(1))
                if completion_len > 0:
                    last_token = int(completion_ids[sample_index, completion_len - 1].item())
                    completion_terminated = last_token in eos_and_pad
                else:
                    completion_terminated = False
                clipped = not completion_terminated
                input_index = min(sample_index // max(num_generations, 1), len(inputs) - 1)
                example = inputs[input_index]
                format_diagnostics = (
                    analyze_format_completion(completion_text) if analyze_format_completion is not None else {}
                )
                row = {
                    "step": int(getattr(self.state, "global_step", 0)),
                    "global_step": int(getattr(self.state, "global_step", 0)),
                    "rank": int(rank),
                    "process_index": int(rank),
                    "sample_index": int(sample_index),
                    "input_index": int(input_index),
                    "prompt_summary": self._shorten(prompts[sample_index] if sample_index < len(prompts) else ""),
                    "image": self._image_identifier(example),
                    "completion_text": completion_text,
                    "completion_len": completion_len,
                    "completion_terminated": completion_terminated,
                    "clipped": clipped,
                    "reward_metrics": reward_metrics,
                }
                row.update(format_diagnostics)
                if advantages is not None and sample_index < advantages.numel():
                    row["advantage"] = float(advantages.flatten()[sample_index].detach().cpu().item())
                rows.append(row)

            with rollout_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            for row in rows[:2]:
                print(
                    "ROLLOUT_DEBUG "
                    f"step={row['global_step']} sample={row['sample_index']} "
                    f"len={row['completion_len']} clipped={int(row['clipped'])} "
                    f"strict={int(bool(row.get('strict_format')))} "
                    f"fmt_reward={row.get('format_reward')} "
                    f"empty_think={int(bool(row.get('empty_think_prefix')))} "
                    f"outside={int(not bool(row.get('no_text_outside_tags')))} "
                    f"text={self._shorten(row['completion_text'], limit=300)}",
                    flush=True,
                )

        @staticmethod
        def _shorten(value, limit=500):
            text = str(value).replace("\n", "\\n")
            return text if len(text) <= limit else text[: limit - 3] + "..."

        def _summarize_prompt(self, prompt):
            if isinstance(prompt, list):
                parts = []
                for message in prompt:
                    if isinstance(message, dict):
                        parts.append(f"{message.get('role', '')}: {message.get('content', '')}")
                    else:
                        parts.append(str(message))
                return self._shorten(" | ".join(parts))
            return self._shorten(prompt)

        def _image_identifier(self, example):
            for key in ("image_path", "image", "images"):
                if key not in example:
                    continue
                value = example.get(key)
                if isinstance(value, (str, os.PathLike)):
                    return str(value)
                if isinstance(value, list):
                    identifiers = []
                    for item in value[:4]:
                        identifiers.append(getattr(item, "filename", None) or str(type(item).__name__))
                    return identifiers
                return getattr(value, "filename", None) or str(type(value).__name__)
            return None

        def _build_mm_token_type_ids_for_loss(self, inputs, completion_ids):
            prompts = [example["prompt"] for example in inputs]
            if "images" in inputs[0]:
                images = [example.get("images") for example in inputs]
            elif "image" in inputs[0]:
                images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
            else:
                return None
            if images is not None and all(img_list == [] for img_list in images):
                return None

            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            mm_token_type_ids = prompt_inputs.get("mm_token_type_ids")
            if mm_token_type_ids is None:
                return None
            mm_token_type_ids = mm_token_type_ids.to(completion_ids.device)
            return torch.cat(
                [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)],
                dim=1,
            )

        def _compute_loss(self, model, inputs):
            previous_mm_token_type_ids = getattr(self, "_qwen35_current_mm_token_type_ids", None)
            self._qwen35_current_mm_token_type_ids = inputs.get("mm_token_type_ids")
            try:
                return super()._compute_loss(model, inputs)
            finally:
                self._qwen35_current_mm_token_type_ids = previous_mm_token_type_ids

        def _get_per_token_logps_and_entropies(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=None,
            compute_entropy=False,
            pixel_values=None,
            image_grid_thw=None,
            num_images=None,
            pixel_attention_mask=None,
            image_sizes=None,
            token_type_ids=None,
            mm_token_type_ids=None,
        ):
            """Compute log-probs and entropies while preserving Qwen3.5 multimodal token types."""
            if mm_token_type_ids is None:
                mm_token_type_ids = getattr(self, "_qwen35_current_mm_token_type_ids", None)
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids.to(input_ids.device)
                if mm_token_type_ids.size(1) < input_ids.size(1):
                    pad_shape = (mm_token_type_ids.size(0), input_ids.size(1) - mm_token_type_ids.size(1))
                    padding = mm_token_type_ids.new_zeros(pad_shape)
                    mm_token_type_ids = torch.cat([mm_token_type_ids, padding], dim=1)
                elif mm_token_type_ids.size(1) > input_ids.size(1):
                    mm_token_type_ids = mm_token_type_ids[:, : input_ids.size(1)]

            batch_size = batch_size or input_ids.size(0)
            all_logps = []
            all_entropies = []
            for start in range(0, input_ids.size(0), batch_size):
                input_ids_batch = input_ids[start : start + batch_size]
                attention_mask_batch = attention_mask[start : start + batch_size]

                model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
                if image_grid_thw is not None and pixel_values is not None:
                    rows_per_image = image_grid_thw.prod(dim=-1)
                    rows_per_sample = torch.split(rows_per_image, num_images)
                    rows_per_sample = torch.stack([sample_rows.sum() for sample_rows in rows_per_sample])
                    cum_rows = torch.cat(
                        [torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)]
                    )
                    row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                    model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                    cum_imgs = torch.tensor([0] + num_images, device=image_grid_thw.device).cumsum(0)
                    img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                    model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
                elif pixel_values is not None:
                    model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
                if pixel_attention_mask is not None:
                    model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
                if image_sizes is not None:
                    model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
                if token_type_ids is not None:
                    model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]
                if mm_token_type_ids is not None:
                    model_inputs["mm_token_type_ids"] = mm_token_type_ids[start : start + batch_size]

                if "logits_to_keep" in self.model_kwarg_keys:
                    model_inputs["logits_to_keep"] = logits_to_keep + 1

                model_inputs["use_cache"] = False

                logits = model(**model_inputs).logits
                logits = logits[:, :-1, :]
                logits = logits[:, -logits_to_keep:, :]
                logits = logits / self.temperature
                completion_ids = input_ids_batch[:, -logits_to_keep:]
                logps = selective_log_softmax(logits, completion_ids)
                all_logps.append(logps)

                if compute_entropy:
                    with torch.no_grad():
                        entropies = entropy_from_logits(logits)
                    all_entropies.append(entropies)

            logps = torch.cat(all_logps, dim=0)
            entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
            return logps, entropies

    return Qwen35CompatibleGRPOTrainer


def write_grpo_run_artifacts(output_dir):
    """Write rollout summary and checkpoint manifest for harness evidence."""
    output_dir = Path(output_dir)
    manifest_dir = output_dir / "checkpoints"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dirs = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-", 1)[1]) if path.name.split("-", 1)[1].isdigit() else -1,
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "checkpoints": [
            {
                "name": path.name,
                "step": int(path.name.split("-", 1)[1]) if path.name.split("-", 1)[1].isdigit() else None,
                "path": str(path),
                "has_trainer_state": (path / "trainer_state.json").exists(),
                "has_optimizer": (path / "optimizer.pt").exists(),
                "has_scheduler": (path / "scheduler.pt").exists(),
                "has_rng_state": (path / "rng_state.pth").exists(),
            }
            for path in checkpoint_dirs
        ],
        "final": str(output_dir),
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    rollout_path = output_dir / "rollouts" / "rollouts_rank0.jsonl"
    summary_path = output_dir / "rollout_summary.json"
    if rollout_path.exists():
        rows = []
        with rollout_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        def mean_bool(key):
            return sum(1 for row in rows if bool(row.get(key))) / len(rows) if rows else 0.0
        def mean_num(key):
            vals = [float(row[key]) for row in rows if row.get(key) is not None]
            return sum(vals) / len(vals) if vals else 0.0
        last_metrics = rows[-1].get("reward_metrics", {}) if rows else {}
        lengths = [int(row.get("completion_len", 0)) for row in rows]
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "completed" if rows else "no-rollouts",
            "rollout_path": str(rollout_path),
            "rows": len(rows),
            "steps": sorted({int(row.get("global_step", row.get("step", 0))) for row in rows}),
            "completion_len": {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "mean": sum(lengths) / len(lengths) if lengths else None,
            },
            "clipped": sum(1 for row in rows if row.get("clipped")),
            "strict_format_rate": mean_bool("strict_format"),
            "strict_fullmatch_rate": mean_bool("strict_fullmatch"),
            "think_nonempty_rate": mean_bool("think_nonempty"),
            "no_text_outside_tags_rate": mean_bool("no_text_outside_tags"),
            "answer_single_letter_rate": mean_bool("answer_single_letter"),
            "empty_think_prefix_rate": mean_bool("empty_think_prefix"),
            "four_tag_presence_rate": mean_bool("four_tag_presence"),
            "duplicate_tag_rate": mean_bool("duplicate_tags"),
            "missing_tag_rate": mean_bool("missing_tags"),
            "format_reward_mean": mean_num("format_reward"),
            "last_step_metrics": last_metrics,
            "accuracy_reward_mean": float(last_metrics.get("rewards/accuracy_reward/mean", 0.0)) if last_metrics else None,
            "diagnostic_fields": [
                "strict_format", "strict_fullmatch", "think_nonempty", "no_text_outside_tags",
                "answer_single_letter", "empty_think_prefix", "duplicate_tags", "missing_tags", "format_reward"
            ],
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": str(manifest_path), "rollout_summary": str(summary_path) if summary_path.exists() else None}


class QwenGRPOTrainer:
    """Lazy wrapper around TRL GRPOTrainer.

    Importing TRL's GRPOTrainer can pull in torch distributed/FSDP modules. Keep
    that dependency gate at instantiation time so `import src.trainer` and
    `--help` paths stay lightweight and fail with a clear message only when GRPO
    training is actually requested.
    """

    def __new__(cls, *args, **kwargs):
        try:
            GRPOTrainer = get_qwen35_compatible_grpo_trainer_cls()
        except Exception as exc:
            raise RuntimeError(
                "GRPO is not available with the current TRL/torch dependency set. "
                "Install a TRL version whose GRPOTrainer imports cleanly before "
                "launching Qwen3.5 dense GRPO LoRA."
            ) from exc
        return GRPOTrainer(*args, **kwargs)
