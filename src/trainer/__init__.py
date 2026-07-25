_EXPORTS = {
    "QwenSFTTrainer": ".sft_trainer",
    "QwenDPOTrainer": ".dpo_trainer",
    "QwenGRPOTrainer": ".grpo_trainer",
    "QwenCLSTrainer": ".cls_trainer",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name], package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
