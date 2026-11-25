"""Budget-control tests -- these are the guarantees the study depends on.

Run: python -m pytest tests/test_budget.py   (needs only numpy)
"""
from __future__ import annotations

import numpy as np
import pytest

from drlora.budget import (ALLOCATORS, LoRASite, allocate_ranks,
                           group_ratio_allocator, sensitivity_allocator)

TOL = 0.005


def swin_like_sites(depths=(2, 2, 6, 2), dim=96) -> list[LoRASite]:
    sites, d = [], 0
    for stage, n in enumerate(depths):
        c = dim * 2 ** stage
        for b in range(n):
            pre = f"layers.{stage}.blocks.{b}"
            sites += [
                LoRASite(f"{pre}.attn.qkv", "qkv", c, 3 * c, d, stage),
                LoRASite(f"{pre}.attn.proj", "proj", c, c, d, stage),
                LoRASite(f"{pre}.mlp.fc1", "fc1", c, 4 * c, d, stage),
                LoRASite(f"{pre}.mlp.fc2", "fc2", 4 * c, c, d, stage),
            ]
            d += 1
    return sites


@pytest.mark.parametrize("strategy", sorted(ALLOCATORS))
@pytest.mark.parametrize("mode", ["params", "rank"])
def test_every_strategy_hits_the_budget(strategy, mode):
    sites = swin_like_sites()
    ref = allocate_ranks(sites, "uniform", base_rank=8, budget_mode=mode)
    alloc = allocate_ranks(sites, strategy, base_rank=8, budget_mode=mode,
                           budget=ref.meta["budget"])
    realised = alloc.total_params if mode == "params" else alloc.total_rank
    expected = ref.total_params if mode == "params" else ref.total_rank
    assert abs(realised - expected) <= TOL * expected, (strategy, realised, expected)


def test_equal_rank_is_not_equal_params():
    """The reason decoupling needs care: a qkv site costs more per unit rank."""
    sites = swin_like_sites()
    by_rank = allocate_ranks(sites, "qkv_heavy", base_rank=8, budget_mode="rank",
                             budget=allocate_ranks(sites, "uniform", base_rank=8,
                                                   budget_mode="rank").total_rank)
    uniform = allocate_ranks(sites, "uniform", base_rank=8, budget_mode="rank")
    assert by_rank.total_rank == pytest.approx(uniform.total_rank, rel=TOL)
    assert by_rank.total_params != uniform.total_params


def test_ranks_are_non_negative_integers_within_caps():
    sites = swin_like_sites()
    alloc = allocate_ranks(sites, "qkv_heavy", base_rank=8)
    caps = {s.name: s.max_rank for s in sites}
    for n, r in alloc.ranks.items():
        assert isinstance(r, int) and 0 <= r <= caps[n]


def test_group_only_strategies_concentrate_the_whole_budget():
    sites = swin_like_sites()
    ref = allocate_ranks(sites, "uniform", base_rank=8)
    alloc = allocate_ranks(sites, "qkv_only", base_rank=8, budget=ref.meta["budget"])
    groups = alloc.by_group()
    assert set(groups) == {"qkv"}
    assert groups["qkv"]["params"] == pytest.approx(ref.total_params, rel=TOL)


def test_min_rank_keeps_every_site_alive():
    sites = swin_like_sites()
    alloc = allocate_ranks(sites, "qkv_heavy", base_rank=8, min_rank=1)
    assert all(r >= 1 for r in alloc.ranks.values())


def test_custom_ratios_and_sensitivity_are_budget_matched():
    sites = swin_like_sites()
    ref = allocate_ranks(sites, "uniform", base_rank=8)
    for fn in (group_ratio_allocator({"qkv": 4, "proj": 1, "fc1": 2, "fc2": 2}),
               sensitivity_allocator({s.name: 1.0 + 0.1 * s.depth for s in sites})):
        alloc = allocate_ranks(sites, fn, base_rank=8, budget=ref.meta["budget"])
        assert alloc.total_params == pytest.approx(ref.total_params, rel=TOL)


def test_budget_violation_raises():
    sites = swin_like_sites()
    with pytest.raises(ValueError):
        allocate_ranks(sites, "uniform", base_rank=8, min_rank=10_000)
