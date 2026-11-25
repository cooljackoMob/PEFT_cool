"""Forward/backward smoke test on random tensors (CPU, no dataset, no weights).

Covers what a --dry-run cannot: that the injected adapters actually run, that
only LoRA and head parameters receive gradients, and that the dense decoder
handles each backbone's feature layout (timm Swin emits NHWC, ViT NCHW).
"""
from __future__ import annotations

import pytest
import torch

from drlora.budget import allocate_ranks
from drlora.lora import (count_parameters, inject_lora, mark_only_lora_trainable)
from drlora.models import GroupSpec, build_model, discover_sites

CASES = [(bb, task) for bb in ("swin_tiny", "vit_b16")
         for task in ("classification", "segmentation")]


def _build(backbone: str, task: str, strategy: str = "qkv_heavy", base_rank: int = 4):
    n_cls = 10 if task == "classification" else 21
    model = build_model(task, backbone, n_cls, pretrained=False)
    sites = discover_sites(model, GroupSpec())
    ref = allocate_ranks(sites, "uniform", base_rank=base_rank)
    alloc = allocate_ranks(sites, strategy, base_rank=base_rank,
                           budget=ref.meta["budget"])
    inject_lora(model, alloc)
    mark_only_lora_trainable(model)
    return model, alloc, ref, n_cls


@pytest.mark.parametrize("backbone,task", CASES)
def test_forward_backward_and_grad_routing(backbone, task):
    torch.manual_seed(0)
    model, alloc, ref, n_cls = _build(backbone, task)
    assert abs(count_parameters(model)["lora"] - ref.total_params) \
        <= 0.005 * ref.total_params

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    if task == "classification":
        assert out.shape == (2, n_cls)
        loss = torch.nn.functional.cross_entropy(out, torch.randint(0, n_cls, (2,)))
    else:
        assert out.shape == (2, n_cls, 224, 224), out.shape
        loss = torch.nn.functional.cross_entropy(
            out, torch.randint(0, n_cls, (2, 224, 224)))
    loss.backward()

    for name, p in model.named_parameters():
        if "lora_A" in name:                     # A is non-zero at init
            assert p.grad is not None and p.requires_grad, name
        if not p.requires_grad:
            assert p.grad is None, f"frozen param got a gradient: {name}"


@pytest.mark.parametrize("backbone,task", CASES)
def test_adapter_is_identity_at_init(backbone, task):
    """B is zero-initialised, so an untrained adapter must not change outputs."""
    torch.manual_seed(0)
    model, _, _, _ = _build(backbone, task)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        adapted = model(x)
        for m in model.modules():
            if hasattr(m, "lora_B") and m.lora_B is not None:
                m.rank = 0                       # bypass the adapter branch
        plain = model(x)
    assert torch.allclose(adapted, plain, atol=1e-5)


def test_rank_zero_site_has_no_parameters():
    model, alloc, _, _ = _build("swin_tiny", "classification", strategy="qkv_only")
    off = [m for m in model.modules() if hasattr(m, "rank") and m.rank == 0]
    assert off and all(m.lora_A is None and m.n_lora_params == 0 for m in off)
