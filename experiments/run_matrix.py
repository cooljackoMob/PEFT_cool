#!/usr/bin/env python3
"""Run the full controlled matrix: {backbone} x {task} x {strategy} x {seed}.

All groups share one parameter budget per (backbone, task, base_rank) cell,
so any difference in the reported metric is attributable to allocation.
Use --dry-run first: it prints each group's realised budget so you can confirm
the control holds before spending GPU time.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_STRATEGIES = [
    "uniform",            # reference (the MTLoRA assumption)
    "qkv_heavy",
    "proj_heavy",
    "mlp_heavy",
    "attn_only",
    "mlp_only",
    "depth_increasing",
    "depth_decreasing",
    "random_control",     # budget-matched null hypothesis
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbones", default="swin_tiny,vit_b16")
    p.add_argument("--tasks", default="classification,segmentation")
    p.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--base-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", default="runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", default="", help="extra flags passed to train.py")
    return p.parse_args()


def main():
    args = parse_args()
    grid = list(itertools.product(
        [b for b in args.backbones.split(",") if b],
        [t for t in args.tasks.split(",") if t],
        [s for s in args.strategies.split(",") if s],
        [int(s) for s in args.seeds.split(",") if s],
    ))
    print(f"{len(grid)} runs queued")
    failures = []
    for backbone, task, strategy, seed in grid:
        cmd = [sys.executable, "train.py", "--backbone", backbone, "--task", task,
               "--strategy", strategy, "--seed", str(seed),
               "--base-rank", str(args.base_rank), "--epochs", str(args.epochs),
               "--out", args.out]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.extra:
            cmd += args.extra.split()
        print("$", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0:
            failures.append((backbone, task, strategy, seed))
    if failures:
        print("FAILED:", json.dumps(failures))
        sys.exit(1)
    print("all runs finished")


if __name__ == "__main__":
    main()
