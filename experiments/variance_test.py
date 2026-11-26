#!/usr/bin/env python3
"""Compare two strategies on both mean and *spread* across seeds.

`analyze.py` tests means.  A strategy can match uniform on average while being
markedly more (or less) reproducible run to run, and that is a separate claim
needing a separate test -- Levene (robust, no normality assumption) and the
F-test of variances, plus a paired t-test and Welch's t-test on the means, with
a bootstrap CI on the variance ratio since variance tests are weak at n~10.

    python experiments/variance_test.py --out runs/pilot_r2 \
        --a qkv_heavy --b uniform --base-rank 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(out_dir: Path, strategy: str, base_rank: int, task: str, backbone: str):
    scores = {}
    for f in sorted(out_dir.glob("*/result.json")):
        r = json.loads(f.read_text())
        a = r["args"]
        if (a["strategy"] == strategy and a["base_rank"] == base_rank
                and a["task"] == task and a["backbone"] == backbone):
            key = "top1" if task == "classification" else "miou"
            scores[a["seed"]] = r.get(f"best_{key}")
            params = r["param_counts"]["lora"]
    if not scores:
        raise SystemExit(f"no runs found for {strategy} at rank {base_rank}")
    return scores, params


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/pilot_r2")
    p.add_argument("--a", default="qkv_heavy")
    p.add_argument("--b", default="uniform")
    p.add_argument("--base-rank", type=int, default=2)
    p.add_argument("--task", default="classification")
    p.add_argument("--backbone", default="swin_tiny")
    p.add_argument("--boot", type=int, default=20000)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    from scipy import stats

    d = Path(args.out)
    sa, pa = load(d, args.a, args.base_rank, args.task, args.backbone)
    sb, pb = load(d, args.b, args.base_rank, args.task, args.backbone)
    common = sorted(set(sa) & set(sb))
    if len(common) < 3:
        raise SystemExit(f"only {len(common)} shared seeds; need more")
    x = np.array([sa[s] for s in common])
    y = np.array([sb[s] for s in common])

    print(f"{args.a} vs {args.b} @ rank {args.base_rank}, n={len(common)} shared seeds")
    print(f"  budget: {pa:,} vs {pb:,} params "
          f"({'matched' if abs(pa-pb) <= 0.005*pb else 'MISMATCHED'})")
    print(f"  {args.a:<15} mean={x.mean():.4f} sd={x.std(ddof=1):.4f} "
          f"min={x.min():.4f} max={x.max():.4f}")
    print(f"  {args.b:<15} mean={y.mean():.4f} sd={y.std(ddof=1):.4f} "
          f"min={y.min():.4f} max={y.max():.4f}")

    # means
    t_p = stats.ttest_rel(x, y)
    w_p = stats.ttest_ind(x, y, equal_var=False)
    print(f"\n  means:  paired t p={t_p.pvalue:.4f}   Welch p={w_p.pvalue:.4f}   "
          f"delta={100*(x.mean()-y.mean()):+.2f}pp")

    # spread
    lev = stats.levene(x, y, center="median")
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    ratio = vx / vy if vy > 0 else float("inf")
    n1 = n2 = len(common)
    f_p = 2 * min(stats.f.cdf(ratio, n1 - 1, n2 - 1),
                  1 - stats.f.cdf(ratio, n1 - 1, n2 - 1))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(args.boot):
        bx = rng.choice(x, len(x), replace=True)
        by = rng.choice(y, len(y), replace=True)
        if by.var(ddof=1) > 0:
            boots.append(bx.var(ddof=1) / by.var(ddof=1))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    def _fmt(v: float) -> str:
        # a resample can be near-constant, sending the ratio to ~0 or ~inf;
        # printing 1e26 implies precision that is not there
        if not np.isfinite(v) or v > 1e3:
            return ">1000"
        return f"{v:.2f}" if v >= 0.01 else "<0.01"

    print(f"  spread: Levene p={lev.pvalue:.4f}   F-test p={f_p:.4f}")
    print(f"          sd ratio {np.sqrt(ratio):.2f}x, variance ratio {ratio:.2f} "
          f"(95% bootstrap CI {_fmt(lo)}-{_fmt(hi)})")
    verdict = ("more reproducible" if ratio < 1 else "less reproducible")
    sig = "significant" if min(lev.pvalue, f_p) < 0.05 else "NOT significant"
    print(f"\n  {args.a} is {np.sqrt(1/ratio) if ratio else 0:.2f}x "
          f"{'tighter' if ratio < 1 else 'wider'} than {args.b} "
          f"-- {sig} at alpha=0.05.")
    if lo <= 1.0 <= hi:
        print("  The bootstrap CI contains 1.0, so equal spread is not excluded.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "a": args.a, "b": args.b, "base_rank": args.base_rank, "seeds": common,
            "scores_a": {str(s): sa[s] for s in common},
            "scores_b": {str(s): sb[s] for s in common},
            "mean_a": x.mean(), "mean_b": y.mean(),
            "sd_a": x.std(ddof=1), "sd_b": y.std(ddof=1),
            "paired_t_p": t_p.pvalue, "welch_p": w_p.pvalue,
            "levene_p": lev.pvalue, "f_test_p": f_p,
            "variance_ratio": ratio, "variance_ratio_ci95": [lo, hi],
            "params_a": pa, "params_b": pb,
        }, indent=2, default=float))
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
