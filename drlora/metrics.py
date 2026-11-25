"""Evaluation metrics: top-1 accuracy and mean IoU."""
from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return (logits.argmax(1) == target).float().mean().item()


class ConfusionMatrix:
    """Streaming mIoU, ignoring `ignore_index` (255 for VOC)."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.n = num_classes
        self.ignore = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(1).flatten()
        tgt = target.flatten()
        keep = tgt != self.ignore
        pred, tgt = pred[keep], tgt[keep]
        idx = tgt * self.n + pred
        self.mat += torch.bincount(idx.cpu(), minlength=self.n ** 2).reshape(self.n, self.n)

    def miou(self) -> float:
        m = self.mat.float()
        inter = m.diag()
        union = m.sum(0) + m.sum(1) - inter
        iou = inter / union.clamp(min=1)
        valid = union > 0
        return iou[valid].mean().item() if valid.any() else 0.0

    def pixel_acc(self) -> float:
        m = self.mat.float()
        return (m.diag().sum() / m.sum().clamp(min=1)).item()
