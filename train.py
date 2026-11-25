#!/usr/bin/env python3
"""Train one experimental group: one backbone, one task, one rank-allocation
strategy, under a strictly controlled trainable-parameter budget.

Everything except `--strategy` is held fixed across groups.  The realised
parameter budget is asserted against the uniform reference before training
starts, so a run that would not be a fair comparison fails fast instead of
producing a misleading number.

Example:
    python train.py --backbone swin_tiny --task classification \
        --strategy qkv_heavy --base-rank 8 --epochs 20 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from drlora.budget import ALLOCATORS, allocate_ranks, group_ratio_allocator
from drlora.lora import (count_parameters, head_parameters, inject_lora,
                         lora_parameters, mark_only_lora_trainable)
from drlora.metrics import ConfusionMatrix, accuracy
from drlora.models import GroupSpec, build_model, discover_sites


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # fixed baseline settings
    p.add_argument("--backbone", default="swin_tiny", choices=["swin_tiny", "vit_b16"])
    p.add_argument("--task", default="classification",
                   choices=["classification", "segmentation"])
    p.add_argument("--dataset", default=None, help="default: cifar100 / voc")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--no-pretrained", action="store_true")
    # the independent variable
    p.add_argument("--strategy", default="uniform",
                   help=f"one of {sorted(ALLOCATORS)}, or ratios like "
                        f"'qkv=4,proj=1,fc1=1,fc2=1'")
    p.add_argument("--base-rank", type=int, default=8,
                   help="rank of the uniform reference; defines the budget")
    p.add_argument("--budget-mode", default="params", choices=["params", "rank"])
    p.add_argument("--min-rank", type=int, default=0)
    p.add_argument("--budget-tolerance", type=float, default=0.005)
    p.add_argument("--groups", default="qkv,proj,fc1,fc2",
                   help="which groups may receive rank at all")
    p.add_argument("--lora-alpha", type=float, default=None,
                   help="fixed alpha; default alpha=rank, which keeps the "
                        "effective update scale equal across decoupled ranks")
    p.add_argument("--lora-dropout", type=float, default=0.0)
    # fixed optimisation
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head-lr-mult", type=float, default=10.0,
                   help="head LR = lr * this. The task head is trained from "
                        "scratch while the adapters only perturb pretrained "
                        "weights, so a head on the adapter LR learns too "
                        "slowly and early accuracy reflects head convergence "
                        "rather than rank allocation. Held fixed across "
                        "groups, so it is part of the controlled setup.")
    p.add_argument("--head-lr", type=float, default=None,
                   help="absolute head LR; overrides --head-lr-mult")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--train-norm", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    p.add_argument("--tag", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="allocate, report the budget, and exit without training")
    return p.parse_args()


def build_allocator(spec: str):
    if spec in ALLOCATORS:
        return spec
    if "=" in spec:
        ratios = {}
        for part in spec.split(","):
            k, v = part.split("=")
            ratios[k.strip()] = float(v)
        return group_ratio_allocator(ratios)
    raise SystemExit(f"unknown strategy {spec!r}; choose from {sorted(ALLOCATORS)} "
                     f"or pass ratios such as 'qkv=4,proj=1,fc1=1,fc2=1'")


def evaluate(model, loader, task, device, num_classes):
    model.eval()
    if task == "classification":
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.numel()
        return {"top1": correct / max(total, 1)}
    cm = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            cm.update(model(x), y)
    return {"miou": cm.miou(), "pixel_acc": cm.pixel_acc()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = args.dataset or ("cifar100" if args.task == "classification" else "voc")
    if args.dry_run:
        # allocation and the budget assertion do not depend on the data, so
        # skip the download: a dry run must be cheap enough to be habitual.
        train_loader = val_loader = None
        num_classes = 100 if args.task == "classification" else 21
    else:
        if args.task == "classification":
            from drlora.data import classification_loaders as loaders
        else:
            from drlora.data import segmentation_loaders as loaders
        train_loader, val_loader, num_classes = loaders(
            args.data_root, dataset, args.img_size, args.batch_size, args.workers,
            args.seed)

    model = build_model(args.task, args.backbone, num_classes,
                        pretrained=not args.no_pretrained, img_size=args.img_size)

    groups = GroupSpec(include=[g.strip() for g in args.groups.split(",") if g.strip()])
    sites = discover_sites(model, groups)

    # Reference budget = uniform allocation at base_rank over the same sites.
    reference = allocate_ranks(sites, "uniform", base_rank=args.base_rank,
                               budget_mode=args.budget_mode, min_rank=args.min_rank)
    alloc = allocate_ranks(sites, build_allocator(args.strategy),
                           base_rank=args.base_rank, budget_mode=args.budget_mode,
                           min_rank=args.min_rank, tolerance=args.budget_tolerance,
                           budget=reference.meta["budget"])
    alloc.strategy = args.strategy
    print(reference.summary())
    print(alloc.summary())

    inject_lora(model, alloc, alpha=args.lora_alpha, dropout=args.lora_dropout)
    mark_only_lora_trainable(model, train_norm=args.train_norm)
    counts = count_parameters(model)
    # hard guarantee of the controlled variable
    assert abs(counts["lora"] - reference.total_params) <= \
        args.budget_tolerance * reference.total_params, (
        f"LoRA budget drift: {counts['lora']:,} vs reference "
        f"{reference.total_params:,}")
    print(f"params: lora={counts['lora']:,} head/other_trainable="
          f"{counts['trainable_non_lora']:,} total={counts['total']:,}")

    tag = args.tag or f"{args.backbone}-{args.task}-{args.strategy}-r{args.base_rank}-s{args.seed}"
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "args": vars(args), "dataset": dataset, "device": device,
        "param_counts": counts, "allocation": alloc.ranks,
        "allocation_by_group": alloc.by_group(), "budget_meta": alloc.meta,
        "reference_params": reference.total_params,
        "head_lr": (args.head_lr if args.head_lr is not None
                    else args.lr * args.head_lr_mult),
    }
    (out_dir / "config.json").write_text(json.dumps(record, indent=2, default=str))
    if args.dry_run:
        print(f"dry run complete -> {out_dir/'config.json'}")
        return

    model.to(device)
    head_lr = (args.head_lr if args.head_lr is not None
               else args.lr * args.head_lr_mult)
    print(f"lr: lora={args.lr:g} head={head_lr:g} "
          f"(mult={args.head_lr_mult:g})")
    optim = torch.optim.AdamW([
        {"params": lora_parameters(model), "lr": args.lr},
        {"params": head_parameters(model), "lr": head_lr},
    ], weight_decay=args.weight_decay)
    steps = max(len(train_loader), 1)
    warmup = args.warmup_epochs * steps
    total_steps = args.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: (s + 1) / max(warmup, 1) if s < warmup else
        0.5 * (1 + np.cos(np.pi * (s - warmup) / max(total_steps - warmup, 1))))
    ignore = 255 if args.task == "segmentation" else -100
    criterion = nn.CrossEntropyLoss(ignore_index=ignore)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")

    history = []
    best = -1.0
    key = "top1" if args.task == "classification" else "miou"
    for epoch in range(args.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=args.amp and device == "cuda"):
                loss = criterion(model(x), y)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            run_loss += loss.item() * y.shape[0]
            seen += y.shape[0]
        metrics = evaluate(model, val_loader, args.task, device, num_classes)
        best = max(best, metrics[key])
        row = {"epoch": epoch, "train_loss": run_loss / max(seen, 1),
               "secs": round(time.time() - t0, 1), **metrics}
        history.append(row)
        print(json.dumps(row))
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else None
    (out_dir / "result.json").write_text(json.dumps({
        **record, "history": history, "final": history[-1] if history else {},
        f"best_{key}": best, "peak_mem_mb": peak,
    }, indent=2, default=str))
    print(f"best {key} = {best:.4f}  ->  {out_dir/'result.json'}")


if __name__ == "__main__":
    main()
