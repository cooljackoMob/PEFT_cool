#!/usr/bin/env python3
"""Aggregate runs/*/result.json into a comparison table.

Reports mean +/- std over seeds per (backbone, task, strategy), the delta
against the uniform reference, a paired t-test against uniform, and the
realised parameter budget of each group -- so a reader can verify the control
rather than trust it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(out_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(out_dir.glob("*/result.json")):
        r = json.loads(f.read_text())
        a = r["args"]
        key = "top1" if a["task"] == "classification" else "miou"
        rows.append({
            "backbone": a["backbone"], "task": a["task"], "strategy": a["strategy"],
            "base_rank": a["base_rank"], "seed": a["seed"], "metric": key,
            "score": r.get(f"best_{key}", r.get("final", {}).get(key)),
            "lora_params": r["param_counts"]["lora"],
            "reference_params": r["reference_params"],
            "budget_dev": abs(r["param_counts"]["lora"] - r["reference_params"])
            / max(r["reference_params"], 1),
            "run": f.parent.name,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    df = load(Path(args.out))
    if df.empty:
        raise SystemExit(f"no result.json found under {args.out}")

    bad = df[df.budget_dev > 0.005]
    if not bad.empty:
        print("WARNING: these runs broke the parameter control:")
        print(bad[["run", "lora_params", "reference_params", "budget_dev"]].to_string(index=False))

    g = df.groupby(["backbone", "task", "base_rank", "strategy"])
    agg = g["score"].agg(["mean", "std", "count"]).reset_index()
    agg["params"] = g["lora_params"].max().values

    out = []
    for (bb, task, rank), block in agg.groupby(["backbone", "task", "base_rank"]):
        ref = block[block.strategy == "uniform"]
        base = float(ref["mean"].iloc[0]) if not ref.empty else np.nan
        block = block.assign(delta_vs_uniform=block["mean"] - base)
        # paired t-test over shared seeds
        pvals = []
        for s in block.strategy:
            a = df[(df.backbone == bb) & (df.task == task) & (df.base_rank == rank)]
            x = a[a.strategy == s].set_index("seed")["score"]
            y = a[a.strategy == "uniform"].set_index("seed")["score"]
            common = x.index.intersection(y.index)
            if s == "uniform" or len(common) < 2:
                pvals.append(np.nan)
                continue
            d = (x[common] - y[common]).values
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = d.mean() / se if se > 0 else np.inf * np.sign(d.mean())
            try:
                from scipy import stats
                pvals.append(float(2 * (1 - stats.t.cdf(abs(t), len(d) - 1))))
            except ImportError:
                pvals.append(float("nan"))
        block = block.assign(p_vs_uniform=pvals)
        out.append(block.sort_values("mean", ascending=False))

    table = pd.concat(out)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
