import json
import os
import re
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig, 
    Qwen2VLForConditionalGeneration, 
    HfArgumentParser, 
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from transformers import TrainerCallback
from src.trainer import QwenSFTTrainer
from src.dataset import make_supervised_data_module
from src.dataset.data_utils import get_image_info, llava_to_openai
from src.params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from monkey_patch_forward import (
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward, 
    replace_qwen_2_with_mixed_modality_forward
)
from monkey_patch_vision import replace_qwen2_5_vision
from train.qwen35_dense_utils import (
    Qwen35DependencyError,
    ensure_swanlab_active_run,
    is_qwen35_dense_identifier,
    load_qwen35_dense_model,
)

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


def _rank0_env() -> bool:
    return os.environ.get("RANK", "0") in ("0", "-1")


def _im_end_token_id(tokenizer):
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        encoded = tokenizer("<|im_end|>", add_special_tokens=False).get("input_ids", [])
        token_id = encoded[0] if len(encoded) == 1 else None
    return token_id


def _parse_prediction(raw_output: str) -> tuple[str | None, str]:
    stripped = raw_output.replace("<|im_end|>", "").strip()
    if re.fullmatch(r"[A-D]", stripped):
        return stripped, "strict_single_letter"
    match = re.match(r"^\s*([A-Da-d])\s*[\).:-]", stripped)
    if match:
        return match.group(1).upper(), "option_prefix"
    standalone = re.findall(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", stripped)
    if len(standalone) == 1:
        return standalone[0].upper(), "single_standalone_letter"
    return None, "no_parse"


def _gold_letter(sample: dict) -> str | None:
    for message in sample.get("conversations", []):
        if message.get("from") == "gpt":
            value = str(message.get("value", "")).strip()
            if re.fullmatch(r"[A-D]", value):
                return value
    return None


def _render_sft_manual_prompt(sample: dict) -> str:
    conversations = sample.get("conversations") or []
    if len(conversations) < 2:
        raise ValueError(f"sample {sample.get('id', '<no id>')} has fewer than two turns")
    transformed = llava_to_openai(conversations[:2], is_video=False)
    user_input = transformed[0]
    gpt_response = transformed[1]
    return (
        f"<|im_start|>{user_input['role']}\n"
        f"{user_input['content']}<|im_end|>\n"
        f"<|im_start|>{gpt_response['role']}\n"
    )


class StepRolloutCallback(TrainerCallback):
    def __init__(self, *, processor, train_dataset, data_args, every_steps: int, max_new_tokens: int):
        self.processor = processor
        self.train_dataset = train_dataset
        self.data_args = data_args
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens
        self.last_step = 0
        self.rows = []

    def _output_paths(self, args):
        rollout_dir = pathlib.Path(args.output_dir) / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        return rollout_dir / "rollouts_rank0.jsonl", rollout_dir / "rollout_summary.json"

    def _write_summary(self, args):
        _, summary_path = self._output_paths(args)
        total = len(self.rows)
        strict = sum(1 for row in self.rows if row.get("strict_single_letter"))
        think = sum(1 for row in self.rows if "<think>" in row.get("raw_output", ""))
        outside = sum(
            1
            for row in self.rows
            if not row.get("strict_single_letter") and row.get("raw_output", "").replace("<|im_end|>", "").strip()
        )
        parse_counts = {}
        for row in self.rows:
            key = row.get("parse_status")
            parse_counts[key] = parse_counts.get(key, 0) + 1
        payload = {
            "total_rollouts": total,
            "strict_single_letter": strict,
            "strict_single_letter_rate": strict / total if total else None,
            "contains_think_count": think,
            "outside_single_letter_count": outside,
            "parse_status_counts": parse_counts,
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "eos_token": "<|im_end|>",
                "max_new_tokens": self.max_new_tokens,
                "prompt_mode": "sft_dataset_manual",
                "system_prompt": None,
                "chat_template": False,
                "thinking_prefill": None,
            },
        }
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _rollout(self, args, state, model):
        sample_index = (state.global_step - 1) % len(self.train_dataset.list_data_dict)
        sample = self.train_dataset.list_data_dict[sample_index]
        prompt = _render_sft_manual_prompt(sample)
        prompt_checks = {
            "starts_with_user": prompt.startswith("<|im_start|>user\n"),
            "ends_with_assistant_header": prompt.endswith("<|im_start|>assistant\n"),
            "has_no_system": "<|im_start|>system\n" not in prompt,
            "has_no_closed_thinking_prefill": "<|im_start|>assistant\n<think>" not in prompt,
        }
        if not all(prompt_checks.values()):
            raise RuntimeError(f"SFT rollout prompt preflight failed at step={state.global_step}: {prompt_checks}")

        image_rel = sample.get("image")
        image_path = image_rel
        images = None
        if image_rel:
            image_path = str(image_rel)
            if not os.path.exists(image_path) and not image_path.startswith("http"):
                image_path = os.path.join(self.data_args.image_folder, image_path)
            image_input = get_image_info(
                image_path,
                self.data_args.image_min_pixels,
                self.data_args.image_max_pixels,
                self.data_args.image_resized_width,
                self.data_args.image_resized_height,
                16,
            )
            images = [image_input]

        inputs = self.processor(
            text=[prompt],
            images=images,
            videos=None,
            padding=False,
            do_resize=False,
            return_tensors="pt",
        )
        device = args.device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        generation_model = model.module if hasattr(model, "module") and hasattr(model.module, "generate") else model
        was_training = generation_model.training
        generation_model.eval()
        config = getattr(generation_model, "config", None)
        old_use_cache = getattr(config, "use_cache", None) if config is not None else None
        if config is not None:
            config.use_cache = True
        generation_args = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        end_token_id = _im_end_token_id(tokenizer)
        if end_token_id is not None:
            generation_args["eos_token_id"] = end_token_id
        with torch.inference_mode():
            outputs = generation_model.generate(**inputs, **generation_args)
        if config is not None and old_use_cache is not None:
            config.use_cache = old_use_cache
        if was_training:
            generation_model.train()

        input_len = inputs["input_ids"].shape[1]
        raw_output = tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        pred_letter, parse_status = _parse_prediction(raw_output)
        gold = _gold_letter(sample)
        row = {
            "step": state.global_step,
            "global_step": state.global_step,
            "sample_index": sample_index,
            "id": sample.get("id"),
            "image": image_rel,
            "rendered_prompt": prompt,
            "prompt_summary": {
                "length_chars": len(prompt),
                "tail": prompt[-240:],
                "checks": prompt_checks,
            },
            "gold_answer": gold,
            "raw_output": raw_output,
            "parse_status": parse_status,
            "pred_letter": pred_letter,
            "strict_single_letter": parse_status == "strict_single_letter",
            "generation_args": {
                **generation_args,
                "eos_token": "<|im_end|>" if end_token_id is not None else None,
            },
            "checkpoint_adapter_context": {
                "training_output_dir": str(args.output_dir),
                "adapter": "in_memory_clean_base_lora",
                "checkpoint": f"checkpoint-{state.global_step}" if state.global_step % args.save_steps == 0 else None,
            },
        }
        jsonl_path, _ = self._output_paths(args)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.rows.append(row)
        raw_compact = raw_output.replace("\n", "\\n")[:180]
        print(
            "ROLLOUT_DEBUG "
            f"step={state.global_step} sample={sample_index} id={sample.get('id')} "
            f"gold={gold} pred={pred_letter} strict={row['strict_single_letter']} "
            f"parse={parse_status} raw={raw_compact!r}",
            flush=True,
        )
        self._write_summary(args)

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_steps <= 0 or not _rank0_env() or state.global_step <= 0:
            return control
        if state.global_step == self.last_step or state.global_step % self.every_steps != 0:
            return control
        self.last_step = state.global_step
        model = kwargs.get("model")
        self._rollout(args, state, model)
        return control

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def get_nested_attr(obj, path):
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current

def resolve_first_module(model, paths, module_name, *, required=True):
    for path in paths:
        module = get_nested_attr(model, path)
        if module is not None:
            return module
    if required:
        raise AttributeError(
            f"Could not find Qwen {module_name} module via paths: {', '.join(paths)}."
        )
    return None

def resolve_vision_tower(model, *, required=True):
    return resolve_first_module(
        model,
        ("visual", "model.visual"),
        "vision tower",
        required=required,
    )

def resolve_language_module(model, *, required=True):
    return resolve_first_module(
        model,
        ("language_model", "model.language_model", "model"),
        "language",
        required=required,
    )

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = resolve_vision_tower(model)
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    if hasattr(vision_tower, "merger"):
        merger_params = vision_tower.merger.parameters()
        set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(vision_tower, "deepstack_merger_list"):
        deepstack_merger_list_params = vision_tower.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = resolve_first_module(model, ("lm_head", "model.lm_head"), "lm_head", required=False)
    if lm_head is not None:
        set_requires_grad(lm_head.parameters(), not training_args.freeze_llm)

    llm_module = resolve_language_module(model)
    llm_params = llm_module.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    llm_module = resolve_language_module(model, required=False)
    if k_llm and llm_module is not None and hasattr(llm_module, "layers"):
        for layer in llm_module.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    vision_tower = resolve_vision_tower(model, required=False)
    if k_vis and vision_tower is not None and hasattr(vision_tower, "blocks"):
        for blk in vision_tower.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    use_liger = training_args.use_liger
    is_qwen35_dense = is_qwen35_dense_identifier(model_args.model_id)
    if is_qwen35_dense:
        if use_liger:
            raise ValueError("Liger is not enabled for Qwen3.5 dense until transformers support is verified.")
    elif "Qwen2.5" in model_args.model_id:
        # monkey patch the vision model
        replace_qwen2_5_vision()
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen2_5_with_mixed_modality_forward()
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_5_vl()

    elif "Qwen3" in model_args.model_id:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen3_with_mixed_modality_forward()
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            raise ValueError("Liger is not supported for Qwen3 model.")
    
    else:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward()
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_vl()
    
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    if is_qwen35_dense:
        try:
            model = load_qwen35_dense_model(
                model_args.model_id,
                dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )
        except Qwen35DependencyError as exc:
            raise RuntimeError(
                "Qwen3.5 dense is not supported by the current dependency set; "
                "refusing to route it through Qwen3VLForConditionalGeneration. "
                f"{exc}"
            ) from exc

    elif "Qwen2.5" in model_args.model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )

    elif "Qwen3" in model_args.model_id:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )
        
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
        model.enable_input_require_grads()

    if training_args.bits in [4,8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
    
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and merger freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )
    if training_args.sft_rollout_every_steps > 0:
        trainer.add_callback(
            StepRolloutCallback(
                processor=processor,
                train_dataset=data_module["train_dataset"],
                data_args=data_args,
                every_steps=training_args.sft_rollout_every_steps,
                max_new_tokens=training_args.sft_rollout_max_new_tokens,
            )
        )

    ensure_swanlab_active_run(training_args)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
