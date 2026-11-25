"""Decoupled-rank LoRA under a controlled parameter budget.

`drlora.budget` is pure-numpy and importable without torch, so allocations can
be designed and tested on a laptop; the torch-dependent pieces are imported
lazily on attribute access.
"""
from .budget import (ALLOCATORS, LoRASite, RankAllocation, allocate_ranks,
                     group_ratio_allocator, lora_param_cost, register_allocator,
                     sensitivity_allocator)

_LAZY = {
    "LoRALinear": "lora", "inject_lora": "lora", "lora_parameters": "lora",
    "head_parameters": "lora", "mark_only_lora_trainable": "lora",
    "count_parameters": "lora",
    "GroupSpec": "models", "classify_module": "models", "discover_sites": "models",
    "build_backbone": "models", "build_model": "models",
}


def __getattr__(name):
    import importlib
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)


__all__ = [
    "ALLOCATORS", "LoRASite", "RankAllocation", "allocate_ranks",
    "group_ratio_allocator", "lora_param_cost", "register_allocator",
    "sensitivity_allocator", *_LAZY,
]
