#!/usr/bin/env python3
"""Measure per-site rank sensitivity *before* committing a budget.

Two probes, both cheap relative to a full run:

1. `grad`  -- mean gradient-norm of the frozen base weight over N batches,
              normalised by weight norm.  A proxy for how much a site "wants"
              to move; used to build a sensitivity-proportional allocation.
2. `ablate`-- train-free: measure the loss increase when a site's output is
              perturbed by a random rank-r update.  Closer to what rank buys.

Output JSON feeds `drlora.budget.sensitivity_allocator`, so a data-driven
allocation can be compared against uniform at the same parameter budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from drlora.models import GroupSpec, build_model, discover_sites


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="swin_tiny", choices=["swin_tiny", "vit_b16"])
    p.add_argument("--task", default="classification",
                   choices=["classification", "segmentation"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--probe", default="grad", choices=["grad", "ablate"])
    p.add_argument("--probe-rank", type=int, default=8, help="for --probe ablate")
    p.add_argument("--groups", default="qkv,proj,fc1,fc2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/sensitivity.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = args.dataset or ("cifar100" if args.task == "classification" else "voc")
    if args.task == "classification":
        from drlora.data import classification_loaders as loaders
    else:
        from drlora.data import segmentation_loaders as loaders
    loader, _, num_classes = loaders(args.data_root, dataset, args.img_size,
                                     args.batch_size, 4, args.seed)

    model = build_model(args.task, args.backbone, num_classes, img_size=args.img_size)
    sites = discover_sites(model, GroupSpec([g for g in args.groups.split(",") if g]))
    by_name = dict(model.named_modules())
    model.to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=255 if args.task == "segmentation" else -100)

    scores = {}
    if args.probe == "grad":
        for p in model.parameters():
            p.requires_grad_(False)
        for s in sites:
            by_name[s.name].weight.requires_grad_(True)
        model.train()
        acc = {s.name: 0.0 for s in sites}
        n = 0
        for i, (x, y) in enumerate(loader):
            if i >= args.batches:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            for s in sites:
                w = by_name[s.name].weight
                if w.grad is not None:
                    acc[s.name] += (w.grad.norm() / w.detach().norm().clamp(min=1e-8)).item()
            n += 1
        scores = {k: v / max(n, 1) for k, v in acc.items()}
    else:
        model.eval()
        batches = []
        for i, (x, y) in enumerate(loader):
            if i >= args.batches:
                break
            batches.append((x.to(device), y.to(device)))
        with torch.no_grad():
            base = sum(criterion(model(x), y).item() for x, y in batches) / len(batches)
            for s in sites:
                lin = by_name[s.name]
                w = lin.weight
                g = torch.Generator(device=w.device).manual_seed(args.seed)
                a = torch.randn(args.probe_rank, s.in_features, generator=g,
                                device=w.device, dtype=w.dtype)
                b = torch.randn(s.out_features, args.probe_rank, generator=g,
                                device=w.device, dtype=w.dtype)
                delta = (b @ a)
                delta *= 0.01 * w.norm() / delta.norm().clamp(min=1e-8)
                w.add_(delta)
                pert = sum(criterion(model(x), y).item() for x, y in batches) / len(batches)
                w.sub_(delta)
                scores[s.name] = max(pert - base, 0.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = {}
    for s in sites:
        groups.setdefault(s.group, []).append(scores[s.name])
    out.write_text(json.dumps({
        "probe": args.probe, "backbone": args.backbone, "task": args.task,
        "scores": scores,
        "group_mean": {g: sum(v) / len(v) for g, v in groups.items()},
    }, indent=2))
    print(json.dumps({g: round(sum(v) / len(v), 6) for g, v in groups.items()}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
