"""Rank allocation under a strictly controlled trainable-parameter budget.

The core independent variable of the study is *how* a fixed rank/parameter
budget is distributed across module groups (qkv / proj / mlp.fc1 / mlp.fc2)
and across depth.  Every strategy here returns an allocation whose total
LoRA parameter count matches the reference (uniform) allocation to within
`tolerance` (default 0.5%), so groups are comparable by construction.

A LoRA pair (A: r x in, B: out x r) on a weight of shape (out, in) costs
    r * (in + out)
trainable parameters.  Costs therefore differ per site, which is exactly why
"equal total rank" is NOT the same as "equal parameters".  Both notions are
supported via `budget_mode`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class LoRASite:
    """One injectable linear layer."""
    name: str          # full module path, e.g. "blocks.3.attn.qkv"
    group: str         # "qkv" | "proj" | "fc1" | "fc2"
    in_features: int
    out_features: int
    depth: int         # block index, 0 = closest to input
    stage: int = 0     # hierarchical stage (Swin); 0 for plain ViT

    @property
    def unit_cost(self) -> int:
        """Trainable params added per unit of rank."""
        return self.in_features + self.out_features

    @property
    def max_rank(self) -> int:
        return min(self.in_features, self.out_features)


@dataclass
class RankAllocation:
    strategy: str
    ranks: Dict[str, int]
    sites: Sequence[LoRASite]
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def total_params(self) -> int:
        cost = {s.name: s.unit_cost for s in self.sites}
        return sum(r * cost[n] for n, r in self.ranks.items())

    @property
    def total_rank(self) -> int:
        return sum(self.ranks.values())

    def by_group(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        cost = {s.name: s.unit_cost for s in self.sites}
        grp = {s.name: s.group for s in self.sites}
        for n, r in self.ranks.items():
            if r == 0:          # site switched off by the strategy
                continue
            d = out.setdefault(grp[n], {"rank": 0, "params": 0, "sites": 0})
            d["rank"] += r
            d["params"] += r * cost[n]
            d["sites"] += 1
        return out

    def summary(self) -> str:
        lines = [f"[{self.strategy}] total_params={self.total_params:,} "
                 f"total_rank={self.total_rank}"]
        for g, d in sorted(self.by_group().items()):
            lines.append(f"  {g:>5}: rank={d['rank']:<5} params={d['params']:<10,} "
                         f"sites={d['sites']}")
        return "\n".join(lines)


def lora_param_cost(sites: Sequence[LoRASite], ranks: Dict[str, int]) -> int:
    cost = {s.name: s.unit_cost for s in sites}
    return sum(r * cost[n] for n, r in ranks.items())


# --------------------------------------------------------------------------
# weight profiles: each strategy produces a *relative* weight per site, which
# is then rescaled to hit the exact budget.
# --------------------------------------------------------------------------

WeightFn = Callable[[LoRASite, Sequence[LoRASite]], float]


def _uniform(site: LoRASite, sites: Sequence[LoRASite]) -> float:
    return 1.0


def _group_weighted(weights: Dict[str, float]) -> WeightFn:
    def fn(site: LoRASite, sites: Sequence[LoRASite]) -> float:
        return float(weights.get(site.group, 1.0))
    return fn


def _depth_ramp(lo: float, hi: float) -> WeightFn:
    def fn(site: LoRASite, sites: Sequence[LoRASite]) -> float:
        dmax = max(s.depth for s in sites) or 1
        t = site.depth / dmax
        return lo + (hi - lo) * t
    return fn


def _random_profile(seed: int, spread: float = 2.0) -> WeightFn:
    def fn(site: LoRASite, sites: Sequence[LoRASite]) -> float:
        rng = np.random.default_rng(abs(hash((seed, site.name))) % (2 ** 32))
        return float(np.exp(rng.normal(0.0, math.log(spread))))
    return fn


#: Named strategies. Each entry maps a site to a relative weight; the
#: allocator rescales so the realised budget matches the reference.
ALLOCATORS: Dict[str, WeightFn] = {
    # reference: every site gets the same rank (MTLoRA-style assumption)
    "uniform": _uniform,
    # attention-projection-centric variants
    "qkv_heavy": _group_weighted({"qkv": 3.0, "proj": 1.0, "fc1": 1.0, "fc2": 1.0}),
    "qkv_only": _group_weighted({"qkv": 1.0, "proj": 0.0, "fc1": 0.0, "fc2": 0.0}),
    "proj_heavy": _group_weighted({"qkv": 1.0, "proj": 3.0, "fc1": 1.0, "fc2": 1.0}),
    "attn_only": _group_weighted({"qkv": 1.0, "proj": 1.0, "fc1": 0.0, "fc2": 0.0}),
    "mlp_heavy": _group_weighted({"qkv": 1.0, "proj": 1.0, "fc1": 3.0, "fc2": 3.0}),
    "mlp_only": _group_weighted({"qkv": 0.0, "proj": 0.0, "fc1": 1.0, "fc2": 1.0}),
    # depth-decoupled variants (groups uniform, depth varies)
    "depth_increasing": _depth_ramp(0.5, 2.0),
    "depth_decreasing": _depth_ramp(2.0, 0.5),
    # control: a budget-matched but meaningless allocation.  If tuned
    # strategies do not beat this, the ranking signal is noise.
    "random_control": _random_profile(seed=0),
}


def register_allocator(name: str, fn: WeightFn) -> None:
    ALLOCATORS[name] = fn


def group_ratio_allocator(ratios: Dict[str, float]) -> WeightFn:
    """Build an allocator from explicit per-group ratios (for sweeps)."""
    return _group_weighted(ratios)


def sensitivity_allocator(scores: Dict[str, float], temperature: float = 1.0) -> WeightFn:
    """Allocate proportionally to measured per-site sensitivity.

    `scores` are typically gradient-norm or Fisher proxies collected by
    `experiments/probe_sensitivity.py`.  temperature < 1 flattens, > 1 sharpens.
    """
    if not scores:
        raise ValueError("empty sensitivity scores")
    mean = float(np.mean(list(scores.values()))) or 1.0

    def fn(site: LoRASite, sites: Sequence[LoRASite]) -> float:
        s = scores.get(site.name, mean) / mean
        return float(max(s, 1e-6) ** temperature)

    return fn


# --------------------------------------------------------------------------
# budget-exact allocation
# --------------------------------------------------------------------------

def allocate_ranks(
    sites: Sequence[LoRASite],
    strategy: str | WeightFn = "uniform",
    base_rank: int = 8,
    budget_mode: str = "params",
    min_rank: int = 0,
    tolerance: float = 0.005,
    budget: int | None = None,
) -> RankAllocation:
    """Allocate integer ranks so the realised budget matches the reference.

    Args:
        sites: injectable layers discovered on the backbone.
        strategy: a key of ALLOCATORS or a WeightFn.
        base_rank: rank of the *uniform reference*; defines the budget when
            `budget` is None.
        budget_mode: "params" equalises trainable parameters (recommended, it
            is what a reader will compare); "rank" equalises the summed rank.
        min_rank: floor for sites with non-zero weight (use 1 to keep every
            site adaptable, 0 to allow a strategy to switch sites off).
        tolerance: max allowed relative deviation from the reference budget.
        budget: override the reference budget explicitly.

    Returns:
        RankAllocation with integer ranks and bookkeeping in `.meta`.
    """
    sites = list(sites)
    if not sites:
        raise ValueError("no LoRA sites given")
    if budget_mode not in {"params", "rank"}:
        raise ValueError(f"budget_mode must be 'params' or 'rank', got {budget_mode}")

    name = strategy if isinstance(strategy, str) else getattr(strategy, "__name__", "custom")
    fn = ALLOCATORS[strategy] if isinstance(strategy, str) else strategy

    unit = {s.name: (s.unit_cost if budget_mode == "params" else 1) for s in sites}
    if budget is None:
        budget = base_rank * sum(unit.values())

    weights = {s.name: max(float(fn(s, sites)), 0.0) for s in sites}
    if sum(weights.values()) <= 0:
        raise ValueError(f"strategy '{name}' zeroed out every site")

    active = [s for s in sites if weights[s.name] > 0]
    caps = {s.name: s.max_rank for s in sites}

    # Water-filling: scale weights to the budget, respecting per-site caps and
    # the min_rank floor, then round with the largest-remainder method so the
    # integer total lands on the budget rather than drifting with each round.
    frac: Dict[str, float] = {}
    fixed: Dict[str, int] = {}
    pool = [s for s in active]
    remaining = float(budget - sum(min_rank * unit[s.name] for s in active))
    if remaining < 0:
        raise ValueError(
            f"min_rank={min_rank} already exceeds the budget for strategy '{name}'")

    for _ in range(len(active) + 1):
        # r_i = scale * w_i, so weights express *rank* shares while the budget
        # is spent in `unit` (params or rank). With equal weights this returns
        # exactly base_rank everywhere, which makes "uniform" the clean
        # reference the comparison needs.
        wcost = sum(weights[s.name] * unit[s.name] for s in pool)
        if not pool or wcost <= 0:
            break
        scale = remaining / wcost
        spilled = [s for s in pool if min_rank + scale * weights[s.name] > caps[s.name]]
        if not spilled:
            for s in pool:
                frac[s.name] = min_rank + scale * weights[s.name]
            break
        for s in spilled:
            fixed[s.name] = caps[s.name]
            remaining -= (caps[s.name] - min_rank) * unit[s.name]
            pool.remove(s)
        if remaining < 0:
            raise ValueError(f"budget infeasible under rank caps for '{name}'")

    ranks = {s.name: 0 for s in sites}
    ranks.update(fixed)
    for s in active:
        if s.name not in frac and s.name not in fixed:
            frac[s.name] = float(min_rank)

    # largest-remainder rounding: hand out the leftover budget to the sites
    # with the largest truncated fraction, cheapest-affordable first, so the
    # integer total lands on the budget instead of drifting below it.
    floors = {n: int(math.floor(v)) for n, v in frac.items()}
    ranks.update(floors)
    spent = sum(ranks[n] * unit[n] for n in ranks)
    order = sorted(frac, key=lambda n: (frac[n] - floors[n]), reverse=True)
    progress = True
    while progress and spent < budget:
        progress = False
        for n in order:
            if ranks[n] < caps[n] and spent + unit[n] <= budget:
                ranks[n] += 1
                spent += unit[n]
                progress = True

    realised = sum(ranks[n] * unit[n] for n in ranks)
    dev = abs(realised - budget) / max(budget, 1)
    if dev > tolerance:
        raise ValueError(
            f"strategy '{name}' missed the {budget_mode} budget by {dev:.2%} "
            f"(realised {realised:,} vs budget {budget:,}); relax `tolerance`, "
            f"lower `min_rank`, or raise `base_rank` so ranks are less granular")

    alloc = RankAllocation(strategy=name, ranks=ranks, sites=sites)
    alloc.meta.update(
        budget_mode=budget_mode, budget=budget, realised=realised,
        deviation=dev, base_rank=base_rank, min_rank=min_rank,
        active_sites=sum(1 for r in ranks.values() if r > 0), n_sites=len(sites),
    )
    return alloc
