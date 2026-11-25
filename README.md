# Decoupled-Rank LoRA under a Controlled Parameter Budget

Does the MTLoRA finding — that removing the **QKV** low-rank modules hurts most
(+2.55 → +1.26 accuracy) while removing **proj** hurts less (+1.65) — still
hold once ranks are *decoupled* per module type instead of being uniform?

That ablation varied *presence* under a uniform-rank assumption, so "QKV
matters most" and "QKV should get the most rank" are not the same claim. This
repo isolates the second one.

## Experimental design

| role | setting |
|---|---|
| fixed | backbone (`swin_tiny` or `vit_b16`, pretrained), task + dataset + splits, augmentation, optimizer, schedule, epochs, seed set, head/decoder architecture, head LR multiple |
| **independent variable** | rank allocation across groups `qkv / proj / fc1 / fc2` and across depth |
| controlled | total trainable LoRA parameters (default) or total rank (`--budget-mode rank`), matched to the uniform reference within 0.5% |
| tasks | classification (top-1) **and** dense prediction (mIoU), to test task generality |
| measured | best metric, per-seed spread, paired t-test vs uniform, peak memory |

A LoRA pair on a `(out, in)` weight costs `r * (in + out)` parameters, so
**equal rank is not equal parameters** — a Swin `qkv` site costs 2× a `proj`
site per unit of rank. The allocator therefore solves for integer ranks by
water-filling against rank caps plus largest-remainder rounding, and
`train.py` *asserts* the realised LoRA parameter count against the uniform
reference before training starts. An unfair run fails rather than reporting a
number.

`random_control` is a budget-matched nonsense allocation: if tuned strategies
do not beat it, the ranking signal is noise, not insight.

The task head trains from scratch while the adapters only perturb pretrained
weights, so the head runs at `10x` the adapter LR by default
(`--head-lr-mult`, or `--head-lr` for an absolute value). On the adapter LR a
fresh head converges slowly enough that early epochs measure head convergence
instead of rank allocation. The multiple is identical across groups, so it is
part of the fixed setup, not the variable — but it is worth one calibration
run on your dataset before launching the matrix, and the realised value is
recorded in each run's `config.json`.

`alpha` defaults to `alpha = rank`, so the adapter scale `alpha/r` stays
constant across decoupled ranks. Without that, "more rank" silently also means
"larger effective learning rate" and the comparison measures the wrong thing.

## Usage

```bash
pip install -r requirements.txt
python -m pytest tests -q      # 35 tests, CPU, no dataset or weights needed

# 1. confirm the control holds before spending GPU time (no training)
python train.py --backbone swin_tiny --task classification \
  --strategy qkv_heavy --base-rank 8 --dry-run

# 2. one group
python train.py --backbone swin_tiny --task classification \
  --strategy qkv_heavy --base-rank 8 --epochs 20 --seed 0 --amp

# 3. the full matrix: backbones x tasks x strategies x seeds
python experiments/run_matrix.py --dry-run      # inspect budgets first
python experiments/run_matrix.py --epochs 20 --seeds 0,1,2 \
  --base-ranks 2,4,8 --extra "--amp"   # one controlled cell per rank

# 4. aggregate: mean +/- std, delta vs uniform, paired t-test, budget check
python experiments/analyze.py --out runs --csv results.csv
```

Custom ratios bypass the named strategies:

```bash
python train.py --strategy "qkv=4,proj=1,fc1=1,fc2=1" --base-rank 8
```

Data-driven allocation — probe sensitivity, then allocate proportionally at the
same budget:

```bash
python experiments/probe_sensitivity.py --probe grad --out runs/sens.json
# feed runs/sens.json to drlora.budget.sensitivity_allocator
```

## Strategies

`uniform` (reference) · `qkv_heavy` · `proj_heavy` · `mlp_heavy` ·
`qkv_only` · `attn_only` · `mlp_only` · `depth_increasing` ·
`depth_decreasing` · `random_control`. Register your own with
`drlora.budget.register_allocator`.

## What the results can and cannot say

- A win for `qkv_heavy` over `uniform` at equal parameters supports
  "QKV is rank-sensitive, not merely necessary".
- A win for `qkv_only` would be stronger still, but confounds *allocation* with
  *site coverage* — report it separately from the graded variants.
- If every strategy lands within the seed spread, the honest conclusion is that
  allocation does not matter at this budget. Sweep downward
  (`--base-ranks 2,4,8`) before concluding anything: at a generous rank every
  strategy has enough capacity everywhere, so a tie there says nothing about
  where rank should go when it is scarce. Each rank is its own self-contained
  comparison with its own re-derived budget, and `analyze.py` groups by
  `base_rank` accordingly.

## Layout

```
drlora/budget.py   allocation strategies + exact budget solver (pure numpy)
drlora/lora.py     per-site-rank LoRALinear, injection, freezing, counting
drlora/models.py   fixed backbones, site discovery/grouping, cls + dense heads
drlora/data.py     fixed loaders   drlora/metrics.py  top-1, mIoU
train.py           one experimental group     experiments/  matrix, probe, analysis
tests/             budget-control + forward/backward guarantees (pytest, CPU)
```
