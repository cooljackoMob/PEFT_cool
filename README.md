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

The task head runs at the **same LR as the adapters** by default
(`--head-lr-mult 1.0`; `--head-lr` sets an absolute value). A faster head is
the intuitive choice -- it trains from scratch while the adapters only perturb
pretrained weights -- but it measured worse here:

| head-LR mult | head LR | ep1 | ep2 | ep3 | best top-1 |
|---|---|---|---|---|---|
| **1x** | 1e-3 | 54.8% | 67.3% | **70.5%** | **0.7050** |
| 10x | 1e-2 | 59.7% | 66.7% | 68.9% | 0.6890 |
| 30x | 3e-2 | 54.6% | 64.7% | 66.9% | 0.6690 |

Swin-T, pretrained, `uniform` rank 8, 2000 train / 1000 val CIFAR-100, seed 0.
The 10x head led at epoch 1 and finished 1.6pp behind with the lowest train
loss of the three: it fits the head rather than teaching the adapters. One
seed on 20 images per class, and all three were still climbing at epoch 3, so
treat this as early-training behaviour and re-calibrate for your own dataset
and epoch budget. The multiple is identical across groups, so it is part of
the fixed setup, not the variable — but it is worth one calibration
run on your dataset before launching the matrix, and the realised value is
recorded in each run's `config.json`:

```bash
python experiments/calibrate_head_lr.py --mults 1,10,30 \
  --subset 2000 --val-subset 1000 --epochs 3
```

Take the smallest multiple whose accuracy has plateaued.

`alpha` defaults to `alpha = rank`, so the adapter scale `alpha/r` stays
constant across decoupled ranks. Without that, "more rank" silently also means
"larger effective learning rate" and the comparison measures the wrong thing.

## Usage

```bash
pip install -r requirements.txt
python -m pytest tests -q      # 35 tests, CPU, no dataset or weights needed

# `--dataset hf_cifar100` pulls CIFAR-100 from the Hugging Face hub instead of
# www.cs.toronto.edu, for networks where that host is unreachable.

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

## Measured result: no allocation effect on this task

Swin-T / CIFAR-100 (2000 train / 1000 val), 3 epochs, `head_lr_mult=1`,
every row at an identical parameter budget. **Three seeds at `--base-rank 2`
(141,312 LoRA params each):**

| strategy | s0 | s1 | s2 | mean | sd | paired delta vs uniform | p |
|---|---|---|---|---|---|---|---|
| mlp_heavy | .708 | .723 | .723 | 0.7180 | 0.0087 | +1.6 / +0.1 / 0.0 -> **+0.57pp** | 0.39 |
| random_control | .704 | .721 | .718 | 0.7143 | 0.0091 | +1.2 / -0.1 / -0.5 -> +0.20pp | 0.73 |
| qkv_heavy | .710 | .715 | .716 | 0.7137 | 0.0032 | +1.8 / -0.7 / -0.7 -> +0.13pp | 0.89 |
| proj_heavy | .701 | .720 | .716 | 0.7123 | 0.0100 | +0.9 / -0.2 / -0.7 -> +0.00pp | 1.00 |
| uniform | .692 | .722 | .723 | 0.7123 | 0.0176 | — | — |

**MTLoRA's QKV ordering does not transfer to decoupled rank allocation here.**
No p-value is below 0.39. `qkv_heavy` -- the strategy their ablation predicts
should win -- places third at p = 0.89, behind a budget-matched *random*
allocation, and `proj_heavy` places last.

Two things make this a usable null rather than a failed run:

1. **`random_control` earns its place in the strategy list.** At one seed every
   reallocation "beat" uniform by 0.9-1.8pp, random included. All of it came
   from `uniform`'s seed-0 draw (.692 against .722/.723 after; sd 0.0176, five
   times the spread between strategy means). Without a nonsense allocation in
   the table, that reads as a QKV effect.
2. **The budget never binds, so nothing can exploit it.** Holding the
   allocation at `uniform` and sweeping the budget 8x:

   | reference rank | LoRA params | top-1 (seed 0) |
   |---|---|---|
   | 1 | 70,656 | 0.7100 |
   | 2 | 141,312 | 0.6920 |
   | 8 | 565,248 | 0.7050 |

   Flat at ~71% across the range -- the differences are smaller than one seed
   of noise. Allocation strategy cannot matter where capacity is not scarce,
   which is the condition this task is in. Going *lower* does not help: rank 1
   is not binding either (`uniform` .710, `qkv_heavy` .705, `mlp_heavy` .710,
   `random_control` .694 at 70,656 params).

The distinction worth keeping: MTLoRA's ablation *removed* QKV adapters. That
shows a site must be adapted. It does not show the site deserves more rank
once every site is already adapted.

### 10 seeds: qkv_heavy vs uniform

Both arms at 141,312 params, `experiments/variance_test.py`:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| qkv_heavy | .710 | .715 | .716 | .711 | .707 | .708 | .702 | .726 | .725 | .716 |
| uniform | .692 | .722 | .723 | .698 | .718 | .705 | .708 | .716 | .703 | .710 |
| delta (pp) | +1.8 | -0.7 | -0.7 | +1.3 | -1.1 | +0.3 | -0.6 | +1.0 | +2.2 | +0.6 |

- means: 0.7136 +/- 0.0076 vs 0.7095 +/- 0.0103, delta **+0.41pp**, paired t
  **p=0.29**, Welch p=0.33, wins 6/10
- spread: sd ratio 0.74x, Levene **p=0.31**, F-test p=0.39, variance-ratio
  bootstrap CI **0.15-1.62** (spans 1.0)

**The reproducibility hypothesis died with more data.** At 3 seeds `qkv_heavy`
looked 5.5x tighter than `uniform`; the ratio fell monotonically to 2.2x,
1.4x and finally 1.35x as it drew .702, then .726 and .725 -- a textbook
small-sample variance underestimate. Do not read a variance claim off 3 seeds.

**The mean effect survives as a bound, not a finding.** +0.41pp at p=0.29 with
6/10 wins is not significance, but it is not nothing either. With
sd(diff) ~ 1.15pp, detecting +0.41pp at 80% power needs **~60 seeds per arm**
(~26 h of CPU here, under an hour on a GPU). So the defensible statement is
"no detectable benefit from QKV-heavy allocation, upper-bounded around +1pp"
rather than "no benefit" -- and that bound is the useful part: this axis is not
where the gains are, which is worth knowing before committing a GPU to the
full matrix.

Where to look for a real effect instead: a task where rank is actually scarce
-- dense prediction, multi-task, a harder/larger dataset, full-size training
rather than a 2000-image subset, or a backbone where the pretrained features
transfer less well. Raw per-run numbers for all 19 runs: `results/`.

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
