"""Fixed backbones (ViT-B/16, Swin-Tiny) + classification and dense heads.

The backbone and task are held constant across experimental groups; only the
rank allocation varies.  Site discovery is backbone-aware so that `qkv`,
`proj`, `fc1`, `fc2` mean the same thing for ViT and Swin.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .budget import LoRASite

#: regexes mapping a module path to a logical group, in priority order.
GROUP_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("qkv", r"(^|\.)(qkv|q_proj|k_proj|v_proj|query|key|value)$"),
    ("proj", r"(^|\.)(attn|attention)\.(proj|out_proj|o_proj)$"),
    ("fc1", r"(^|\.)(mlp|ffn)\.(fc1|w1|up_proj|dense_h_to_4h)$"),
    ("fc2", r"(^|\.)(mlp|ffn)\.(fc2|w2|down_proj|dense_4h_to_h)$"),
)

GROUPS: Tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")


@dataclass
class GroupSpec:
    """Which groups may receive rank at all (set by the experiment config)."""
    include: Sequence[str] = GROUPS

    def allows(self, group: str) -> bool:
        return group in self.include


def classify_module(name: str) -> str | None:
    for group, pat in GROUP_PATTERNS:
        if re.search(pat, name):
            return group
    return None


def _depth_of(name: str) -> Tuple[int, int]:
    """(stage, block) indices parsed from the module path.

    ViT:  blocks.7.attn.qkv            -> (0, 7)
    Swin: layers.2.blocks.1.attn.qkv   -> (2, 1)
    """
    stage = 0
    m = re.search(r"layers\.(\d+)", name)
    if m:
        stage = int(m.group(1))
    m = re.search(r"blocks\.(\d+)", name)
    block = int(m.group(1)) if m else 0
    return stage, block


def discover_sites(model: nn.Module, groups: GroupSpec | None = None,
                   global_depth: bool = True) -> List[LoRASite]:
    """Find every injectable linear layer and tag it with group + depth.

    `global_depth=True` renumbers Swin's per-stage blocks into one monotone
    depth index, so depth-based strategies behave the same on both backbones.
    """
    groups = groups or GroupSpec()
    found: List[Tuple[str, nn.Linear, str, int, int]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        group = classify_module(name)
        if group is None or not groups.allows(group):
            continue
        stage, block = _depth_of(name)
        found.append((name, mod, group, stage, block))

    if not found:
        raise RuntimeError(
            "no LoRA sites found -- the backbone's module naming is not covered "
            "by GROUP_PATTERNS; add a pattern for it")

    # monotone global depth ordering
    keys = sorted({(s, b) for _, _, _, s, b in found})
    depth_of = {k: i for i, k in enumerate(keys)}

    sites = []
    for name, mod, group, stage, block in found:
        d = depth_of[(stage, block)] if global_depth else block
        sites.append(LoRASite(name=name, group=group, in_features=mod.in_features,
                              out_features=mod.out_features, depth=d, stage=stage))
    return sites


# --------------------------------------------------------------------------
# backbones
# --------------------------------------------------------------------------

BACKBONES = {
    "vit_b16": "vit_base_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


def build_backbone(name: str, pretrained: bool = True, img_size: int | None = None,
                   features_only: bool = False) -> nn.Module:
    import timm
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {list(BACKBONES)}")
    kwargs: Dict[str, object] = dict(pretrained=pretrained, num_classes=0)
    if features_only:
        kwargs = dict(pretrained=pretrained, features_only=True)
    if img_size is not None:
        kwargs["img_size"] = img_size
    return timm.create_model(BACKBONES[name], **kwargs)


class ClassificationModel(nn.Module):
    """Fixed backbone + linear head (task 1: image classification)."""

    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True,
                 img_size: int | None = None):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained, img_size)
        dim = self.backbone.num_features
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class DenseModel(nn.Module):
    """Fixed backbone + lightweight FPN-ish decoder (task 2: segmentation).

    The decoder is deliberately small and identical across groups: it is part
    of the constant, not the variable, so differences come from rank
    allocation rather than decoder capacity.
    """

    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True,
                 decoder_dim: int = 128, img_size: int | None = None):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained, img_size,
                                       features_only=True)
        chs = list(self.backbone.feature_info.channels())
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, decoder_dim, 1) for c in chs])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(chs), decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Conv2d(decoder_dim, num_classes, 1)

    @staticmethod
    def _to_nchw(f: torch.Tensor, channels: int) -> torch.Tensor:
        """timm Swin feature maps are NHWC; ViT/CNN ones are NCHW."""
        if f.shape[1] == channels:
            return f
        if f.shape[-1] == channels:
            return f.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"cannot locate channel dim {channels} in {tuple(f.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        ref = None
        outs = []
        for f, lat in zip(feats, self.laterals):
            y = lat(self._to_nchw(f, lat.in_channels))
            if ref is None:
                ref = y.shape[-2:]
            outs.append(F.interpolate(y, size=ref, mode="bilinear", align_corners=False))
        y = self.decoder(self.fuse(torch.cat(outs, dim=1)))
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(task: str, backbone: str, num_classes: int, pretrained: bool = True,
                img_size: int | None = None) -> nn.Module:
    if task == "classification":
        return ClassificationModel(backbone, num_classes, pretrained, img_size)
    if task == "segmentation":
        return DenseModel(backbone, num_classes, pretrained, img_size=img_size)
    raise KeyError(f"unknown task {task!r}")
