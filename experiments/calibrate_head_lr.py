#!/usr/bin/env python3
"""Calibrate --head-lr-mult before launching the matrix.

The head LR multiple is a *fixed* setting shared by every experimental group,
but it is not neutral: too low and early epochs measure head convergence
instead of rank allocation, too high and the head's own fitting can mask the
differences between allocations.  This sweeps it with the allocation held at
`uniform`, so the only thing varying is the multiple.

Pick the smallest multiple whose accuracy has plateaued -- that trains the head
fast enough to stop being the bottleneck without letting it dominate.

    python experiments/calibrate_head_lr.py --mults 1,10,30 \
        --subset 2000 --val-subset 1000 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mults", default="1,10,30")
    p.add_argument("--backbone", default="swin_tiny")
    p.add_argument("--dataset", default="hf_cifar100")
    p.add_argument("--strategy", default="uniform",
                   help="held fixed: this sweep is about the head, not allocation")
    p.add_argument("--base-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--subset", type=int, default=2000)
    p.add_argument("--val-subset", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="runs/calib")
    p.add_argument("--extra", default="")
    return p.parse_args()


def main():
    args = parse_args()
    mults = [float(m) for m in args.mults.split(",") if m]
    results = {}
    for m in mults:
        tag = f"calib-{args.backbone}-mult{m:g}"
        cmd = [sys.executable, "train.py", "--backbone", args.backbone,
               "--task", "classification", "--dataset", args.dataset,
               "--strategy", args.strategy, "--base-rank", str(args.base_rank),
               "--head-lr-mult", str(m), "--lr", str(args.lr),
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--img-size", str(args.img_size), "--subset", str(args.subset),
               "--val-subset", str(args.val_subset), "--workers", str(args.workers),
               "--seed", str(args.seed), "--out", args.out, "--tag", tag]
        if args.extra:
            cmd += args.extra.split()
        print("$", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"mult={m} FAILED")
            continue
        r = json.loads((Path(args.out) / tag / "result.json").read_text())
        results[m] = {"best_top1": r["best_top1"], "head_lr": r["head_lr"],
                      "per_epoch": [h["top1"] for h in r["history"]]}

    print("\n=== head-LR calibration (allocation fixed at "
          f"{args.strategy}, {args.subset} train / {args.val_subset} val) ===")
    print(f"{'mult':>6} {'head_lr':>9} {'best top1':>10}   per-epoch top1")
    for m, v in sorted(results.items()):
        per = " ".join(f"{t:.4f}" for t in v["per_epoch"])
        print(f"{m:>6g} {v['head_lr']:>9.2e} {v['best_top1']:>10.4f}   {per}")
    if results:
        best = max(results, key=lambda k: results[k]["best_top1"])
        print(f"\nhighest: mult={best:g}. Prefer the smallest multiple within "
              f"noise of it; a tiny subset exaggerates differences.")
    out = Path(args.out) / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
