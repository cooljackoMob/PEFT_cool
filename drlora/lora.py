"""Decoupled-rank LoRA injection for timm ViT / Swin backbones."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

from .budget import RankAllocation


class LoRALinear(nn.Module):
    """nn.Linear wrapper with a per-site rank.

    Rank 0 is allowed and means "frozen, no adapter" -- so a strategy can
    switch a site off without changing the module graph.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None,
                 dropout: float = 0.0, init: str = "kaiming"):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = int(rank)
        # alpha tied to rank keeps the effective update scale comparable across
        # sites with different ranks -- essential when ranks are decoupled,
        # otherwise "more rank" also means "larger learning rate".
        self.alpha = float(alpha) if alpha is not None else float(max(rank, 1))
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.rank, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
            if init == "kaiming":
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            else:
                nn.init.normal_(self.lora_A, std=1.0 / self.rank)
            nn.init.zeros_(self.lora_B)  # adapter starts as identity
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def n_lora_params(self) -> int:
        if self.rank == 0:
            return 0
        return self.lora_A.numel() + self.lora_B.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.rank > 0:
            out = out + self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return out

    def merge(self) -> nn.Linear:
        """Fold the adapter into the base weight (for inference-time timing)."""
        if self.rank > 0:
            with torch.no_grad():
                self.base.weight += (self.lora_B @ self.lora_A) * self.scaling
        return self.base

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"


def _get_parent(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    mod = root
    for p in parts[:-1]:
        mod = getattr(mod, p) if not p.isdigit() else mod[int(p)]
    return mod, parts[-1]


def inject_lora(model: nn.Module, alloc: RankAllocation, alpha: float | None = None,
                dropout: float = 0.0, init: str = "kaiming") -> Dict[str, LoRALinear]:
    """Replace each allocated site with a LoRALinear of its own rank."""
    injected: Dict[str, LoRALinear] = {}
    for name, rank in alloc.ranks.items():
        parent, attr = _get_parent(model, name)
        base = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is {type(base).__name__}, not nn.Linear")
        mod = LoRALinear(base, rank, alpha=alpha, dropout=dropout, init=init)
        if attr.isdigit():
            parent[int(attr)] = mod
        else:
            setattr(parent, attr, mod)
        injected[name] = mod
    return injected


def mark_only_lora_trainable(model: nn.Module, train_norm: bool = False,
                             head_prefixes: Sequence[str] = ("head", "decoder", "heads")
                             ) -> None:
    """Freeze the backbone; train adapters, task heads and optionally norms.

    Heads are always trainable (a frozen random head is not a baseline) but
    their parameters are reported separately from the LoRA budget so the
    controlled variable stays clean.
    """
    for name, p in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_head = any(name.startswith(h + ".") or f".{h}." in name for h in head_prefixes)
        is_norm = train_norm and (".norm" in name or name.startswith("norm"))
        p.requires_grad_(bool(is_lora or is_head or is_norm))


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters()
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad]


def head_parameters(model: nn.Module,
                    head_prefixes: Sequence[str] = ("head", "decoder", "heads")
                    ) -> List[nn.Parameter]:
    out = []
    for n, p in model.named_parameters():
        if not p.requires_grad or "lora_" in n:
            continue
        if any(n.startswith(h + ".") or f".{h}." in n for h in head_prefixes):
            out.append(p)
    return out


def count_parameters(model: nn.Module) -> Dict[str, int]:
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"lora": lora, "trainable": trainable,
            "trainable_non_lora": trainable - lora, "total": total}
