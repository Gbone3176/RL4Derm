from pathlib import Path
from peft import PeftModel
import torch
from transformers import (
    BitsAndBytesConfig, 
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    AutoConfig, 
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration
)
import warnings
import os
import json
import importlib
import inspect
from types import ModuleType
from typing import Callable, List
from train.qwen35_dense_utils import (
    Qwen35DependencyError,
    is_qwen35_dense_identifier,
    load_qwen35_dense_model,
    load_qwen35_processor,
)
try:
    from paths import resolve_model_path, resolve_project_path
except ImportError:  # pragma: no cover
    from src.paths import resolve_model_path, resolve_project_path

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def _resolve_model_or_existing(path_value):
    if path_value is None:
        return None
    path_text = str(path_value)
    path = Path(path_text).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)
    project_candidate = resolve_project_path(path_text)
    if project_candidate.exists():
        return str(project_candidate)
    model_candidate = resolve_model_path(path_text)
    if model_candidate.exists():
        return str(model_candidate)
    return path_text

# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    model_path = _resolve_model_or_existing(model_path)
    model_base = _resolve_model_or_existing(model_base)
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['_attn_implementation'] = 'flash_attention_2'

    if is_lora_model(model_path) and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if is_lora_model(model_path) and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        if is_qwen35_dense_identifier(model_base):
            processor, model, loader_diagnostics = _load_qwen35_dense_lora_model(
                model_path=model_path,
                model_base=model_base,
                base_kwargs=kwargs,
            )
        elif "Qwen3" in model_base:
            processor = AutoProcessor.from_pretrained(model_base)
            print('Loading Qwen3-VL from base model...')
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            loader_diagnostics = _loader_diagnostics(model_path, model_base, processor, model, "Qwen3-VL LoRA")
        elif "Qwen2.5" in model_base:
            processor = AutoProcessor.from_pretrained(model_base)
            print('Loading Qwen2.5-VL from base model...')
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            loader_diagnostics = _loader_diagnostics(model_path, model_base, processor, model, "Qwen2.5-VL LoRA")
        else:
            processor = AutoProcessor.from_pretrained(model_base)
            print('Loading Qwen2-VL from base model...')
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            loader_diagnostics = _loader_diagnostics(model_path, model_base, processor, model, "Qwen2-VL LoRA")
            
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        if not is_qwen35_dense_identifier(model_base):
            print('Loading additional Qwen-VL weights...')
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')
            non_lora_trainables = _normalize_non_lora_state_dict(non_lora_trainables, strip_peft_base_layer=False)
            model.load_state_dict(non_lora_trainables, strict=False)
    
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)

        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        loader_diagnostics = _loader_diagnostics(
            model_path,
            model_base,
            processor,
            model,
            loader_diagnostics.get("loader", "Qwen-VL LoRA"),
            extra={**loader_diagnostics, "merge_lora": True},
        )
        _attach_loader_diagnostics(processor, model, loader_diagnostics)

        print(
            "Model Loaded!!! "
            f"loader={loader_diagnostics['loader']} "
            f"model_class={loader_diagnostics['model_class']} "
            f"dtype={loader_diagnostics['dtype']} "
            f"processor_class={loader_diagnostics['processor_class']}"
        )

    else:
        print(f"Loading model from {model_path} as a standard model. Adapter files were not found, so it can't be merged")
        config_path = Path(model_path) / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        architecture = config.get("architectures", [None])[0]
        model_type = config.get("model_type")
        if model_type == "qwen3_5" or architecture == "Qwen3_5ForConditionalGeneration" or is_qwen35_dense_identifier(model_path):
            try:
                processor = load_qwen35_processor(model_path)
                qwen35_kwargs, attn_diagnostics = _prepare_qwen35_from_pretrained_kwargs(kwargs)
                if "load_in_8bit" not in qwen35_kwargs and "quantization_config" not in qwen35_kwargs:
                    qwen35_kwargs["torch_dtype"] = torch.bfloat16
                print("Loading Qwen3.5 dense standard model...")
                model = load_qwen35_dense_model(model_path, low_cpu_mem_usage=True, **qwen35_kwargs)
                _require_qwen35_flash_attention(model, attn_diagnostics)
                loader_diagnostics = _loader_diagnostics(
                    model_path,
                    None,
                    processor,
                    model,
                    "Qwen3.5 dense standard",
                    extra={
                        **attn_diagnostics,
                        "config_attn_implementation": _model_attn_implementation(model),
                    },
                )
                _attach_loader_diagnostics(processor, model, loader_diagnostics)
            except Qwen35DependencyError as exc:
                raise RuntimeError(
                    "Qwen3.5 dense checkpoint is unsupported by the current dependency set; "
                    "refusing to load it as Qwen3VLForConditionalGeneration. "
                    f"{exc}"
                ) from exc
        elif "Qwen3" in architecture:
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif "Qwen2_5" in architecture:
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        else:
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return processor, model


def _load_qwen35_dense_lora_model(model_path, model_base, base_kwargs):
    print("Loading Qwen3.5 dense LoRA from base model...")
    load_kwargs, attn_diagnostics = _prepare_qwen35_from_pretrained_kwargs(base_kwargs)
    if "load_in_8bit" not in load_kwargs and "quantization_config" not in load_kwargs:
        load_kwargs["torch_dtype"] = torch.bfloat16

    try:
        base_config = AutoConfig.from_pretrained(model_base)
        if hasattr(base_config, "quantization_config"):
            del base_config.quantization_config
        processor = load_qwen35_processor(model_base)
        model = load_qwen35_dense_model(model_base, low_cpu_mem_usage=True, config=base_config, **load_kwargs)
        _require_qwen35_flash_attention(model, attn_diagnostics)
    except Qwen35DependencyError as exc:
        raise RuntimeError(
            "Qwen3.5 dense LoRA base is unsupported by the current dependency set; "
            "refusing to load it as Qwen3VLForConditionalGeneration. "
            f"{exc}"
        ) from exc

    diagnostics = _loader_diagnostics(model_path, model_base, processor, model, "Qwen3.5 dense LoRA")
    diagnostics.update(attn_diagnostics)
    diagnostics["config_attn_implementation"] = _model_attn_implementation(model)
    diagnostics["requested_torch_dtype"] = str(load_kwargs.get("torch_dtype"))
    diagnostics["non_lora"] = _load_qwen35_non_lora_state_dict(model, model_path)
    diagnostics["loaded_non_lora"] = diagnostics["non_lora"]["loaded"]
    diagnostics["merge_lora"] = False
    return processor, model, diagnostics


def _prepare_qwen35_from_pretrained_kwargs(base_kwargs):
    """Convert legacy private attention kwargs to the public Transformers API for Qwen3.5.

    Older repository eval wrappers set ``_attn_implementation`` before the exact model class is known.
    Transformers 5.6.2's Qwen3.5 constructor rejects that private key, but its ``from_pretrained`` path
    accepts the public ``attn_implementation`` argument and applies it to the config.
    """
    load_kwargs = dict(base_kwargs)
    legacy_value = load_kwargs.pop("_attn_implementation", None)
    public_value = load_kwargs.get("attn_implementation")
    requested = public_value or legacy_value
    if legacy_value and public_value and legacy_value != public_value:
        raise ValueError(
            "Conflicting Qwen3.5 attention implementations: "
            f"_attn_implementation={legacy_value!r}, attn_implementation={public_value!r}"
        )
    if legacy_value and not public_value:
        load_kwargs["attn_implementation"] = legacy_value

    diagnostics = {
        "requested_attn_implementation": requested,
        "from_pretrained_attn_key": "attn_implementation" if requested else None,
        "had_internal_attn_key": legacy_value is not None,
        "forbidden_internal_attn_key_present": "_attn_implementation" in load_kwargs,
    }
    if diagnostics["forbidden_internal_attn_key_present"]:
        raise ValueError("Qwen3.5 dense loader must not pass _attn_implementation to from_pretrained.")
    return load_kwargs, diagnostics


def _model_attn_implementation(model):
    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "_attn_implementation", None) or getattr(config, "attn_implementation", None)


def _require_qwen35_flash_attention(model, attn_diagnostics):
    requested = attn_diagnostics.get("requested_attn_implementation")
    if requested != "flash_attention_2":
        return
    actual = _model_attn_implementation(model)
    if actual != "flash_attention_2":
        raise RuntimeError(
            "Qwen3.5 dense loader requested Flash Attention 2 but the loaded config did not keep it; "
            f"requested={requested!r}, actual={actual!r}. Refusing SDPA/eager fallback."
        )


def _normalize_non_lora_state_dict(state_dict, *, strip_peft_base_layer):
    normalized = {(k[11:] if k.startswith('base_model.') else k): v for k, v in state_dict.items()}
    if any(k.startswith('model.model.') for k in normalized):
        normalized = {(k[6:] if k.startswith('model.') else k): v for k, v in normalized.items()}
    if strip_peft_base_layer:
        normalized = {k.replace(".base_layer.", "."): v for k, v in normalized.items()}
    return normalized


def _load_qwen35_non_lora_state_dict(model, model_path):
    path = os.path.join(model_path, "non_lora_state_dict.bin")
    info = {
        "path": path,
        "exists": os.path.exists(path),
        "loaded": False,
        "raw_key_count": 0,
        "normalized_key_count": 0,
        "missing_count": None,
        "unexpected_count": None,
        "missing_sample": [],
        "unexpected_sample": [],
    }
    if not os.path.exists(path):
        print("Qwen3.5 dense LoRA non_lora_state_dict.bin not found; loading adapter only.")
        return info

    print(f"Loading Qwen3.5 dense non-LoRA weights from {path}...")
    non_lora_trainables = torch.load(path, map_location="cpu")
    info["raw_key_count"] = len(non_lora_trainables)
    info["raw_key_sample"] = list(non_lora_trainables)[:10]
    non_lora_trainables = _normalize_non_lora_state_dict(non_lora_trainables, strip_peft_base_layer=True)
    info["normalized_key_count"] = len(non_lora_trainables)
    info["normalized_key_sample"] = list(non_lora_trainables)[:10]
    incompatible = model.load_state_dict(non_lora_trainables, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    info.update(
        {
            "loaded": True,
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "missing_sample": missing[:20],
            "unexpected_sample": unexpected[:20],
        }
    )
    print(
        "Qwen3.5 dense non-LoRA load summary: "
        f"raw_keys={info['raw_key_count']} normalized_keys={info['normalized_key_count']} "
        f"missing={info['missing_count']} unexpected={info['unexpected_count']} "
        f"missing_sample={info['missing_sample']} unexpected_sample={info['unexpected_sample']}"
    )
    allowed_missing = {"lm_head.weight"}
    disallowed_missing = [key for key in missing if key not in allowed_missing]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Qwen3.5 dense non-LoRA weights did not match the dense base model after key normalization; "
            f"missing_count={len(missing)} unexpected_count={len(unexpected)} "
            f"disallowed_missing_sample={disallowed_missing[:20]} unexpected_sample={unexpected[:20]}"
        )
    return info


def _loader_diagnostics(model_path, model_base, processor, model, loader, extra=None):
    config = getattr(model, "config", None)
    tokenizer = getattr(processor, "tokenizer", None)
    try:
        dtype = str(next(model.parameters()).dtype)
    except StopIteration:
        dtype = None
    diagnostics = {
        "loader": loader,
        "model_path": str(model_path),
        "model_base": str(model_base),
        "model_class": type(model).__name__,
        "config_class": type(config).__name__ if config is not None else None,
        "config_model_type": getattr(config, "model_type", None),
        "config_architectures": getattr(config, "architectures", None),
        "processor_class": type(processor).__name__,
        "dtype": dtype,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    if extra:
        diagnostics.update(extra)
    return diagnostics


def _attach_loader_diagnostics(processor, model, diagnostics):
    setattr(processor, "_dermogpt_loader_diagnostics", diagnostics)
    setattr(model, "_dermogpt_loader_diagnostics", diagnostics)

def is_lora_model(model_path: str | Path) -> bool:
    """
    Check if a model directory contains LoRA adapter files.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        bool: True if the directory contains LoRA adapter files
    """
    model_dir = Path(model_path)
    return (model_dir / 'adapter_config.json').exists() and (model_dir / 'adapter_model.safetensors').exists()

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    
def load_reward_funcs(
    module_path: str = "train.reward_funcs",
    *,
    name_pred = lambda n: n.endswith("_reward"),
    obj_pred  = lambda o: callable(o),
    keep_order: bool = True
) -> List[Callable]:

    mod: ModuleType = importlib.import_module(module_path)
    
    members = inspect.getmembers(mod, predicate=obj_pred)

    reward_funcs = [(n, o) for n, o in members if name_pred(n)]

    if keep_order:
        reward_funcs.sort(key=lambda pair: inspect.getsourcelines(pair[1])[1])

    return [o for _, o in reward_funcs]
