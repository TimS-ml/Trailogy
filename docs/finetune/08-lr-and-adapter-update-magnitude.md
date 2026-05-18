# 08 — LR, LoRA-scale, batch: per-step adapter update magnitude is the catastrophic-forgetting knob

## TL;DR

- This doc investigates why some LoRA finetunes preserve general ability while others drift or collapse.
- It introduces `S_step ~= (alpha/r) * lr * effective_batch_size` as a rough proxy for how hard each optimizer step pushes the adapter.
- Later audits show `S_step` is useful but incomplete: runs with the same proxy value can still produce very different general-benchmark scores.
- The takeaway is to reason jointly about LoRA scale, rank, learning rate, batch size, and total steps rather than relying on one scalar.

## Original Hypothesis

Define the **per-step adapter update magnitude**:

```
S_step  ≈  (α / r)  ×  η  ×  eff_bs
```

| run | α/r | η (lr) | eff_bs | **S_step** | mmlu | plant |
|---|---|---|---|---|---|---|
| H200 r8-a16-nokl (post-fb15d6b yaml)  | 2.0 | 3.0e-4 | 32 | **0.0192** | **0.260** (collapsed) | 0.02 |
| H200 r8-a8-nokl  (SOTA, inferred yaml) | 1.0 | 3.0e-4 | 32 | **0.0096** | healthy        | OK    |
| 4090 r8-a16-drop005-mix50k @ step 4k   | 2.0 | 2.0e-4 | 16 | **0.0064** | 0.560          | 0.14  |
| 4090 r8-a16-drop005-mix50k @ step 5k   | 2.0 | 2.0e-4 | 16 | **0.0064** | **0.620**      | 0.15  |
| 4090 r8-a16-drop005-mix50k @ step 6k   | 2.0 | 2.0e-4 | 16 | **0.0064** | 0.580          | 0.15  |

The ranking on `mmlu` decay tracks `S_step` monotonically. Total
samples seen (= step × eff_bs) does **not** explain the ordering:
the H200 SOTA run saw 147k samples and still held generality, while
the H200 r8-a16 run saw only 64k samples and crashed. Per-step push
on the base distribution is the discriminator, not cumulative
exposure.

## Why these three runs look the same on paper but aren't

| field | H200 r8-a16-nokl | H200 r8-a8-nokl (SOTA) | 4090 r8-a16-mix50k |
|---|---|---|---|
| config source | `configs/cloud_sweep/r8-a16-nokl.yaml` HEAD (post commit `fb15d6b`) | not in git — H200-only, assumed = r8-a16-nokl yaml with `lora_alpha: 8` | `configs/local_sweep/r8-a16-drop005-mix50k.yaml` HEAD |
| `lora.r` | 8 | 8 | 8 |
| `lora.lora_alpha` | **16** | **8** | 16 |
| **α / r (LoRA output scale)** | **2.0** | **1.0** | **2.0** |
| `lora.lora_dropout` | 0.05 | 0.05 | 0.05 |
| `lora.projector_learning_rate` | 1.4e-4 | 1.4e-4 (assumed) | 1.0e-4 |
| `training.per_device_train_batch_size` | 32 | 32 | 2 |
| `training.gradient_accumulation_steps` | 1 | 1 | 8 |
| **effective batch size** | **32** | **32** | **16** |
| `training.learning_rate` | **3.0e-4** | **3.0e-4** | **2.0e-4** |
| `training.lr_scheduler_type` | cosine | cosine | cosine |
| `training.warmup_steps` (warmup_ratio=0.03) | 30 of ~1000 | 30 of ~1000 | 30 of ~4000 |
| `regularization.kl_enabled` | false | false | false |
| `regularization.l2_enabled` | false | false | false |
| steps actually run | 2 000 | 4 600 | 6 000+ (ongoing) |
| samples seen (= step × eff_bs) | 64 000 | 147 200 | 96 000+ |

The two columns "H200 r8-a16-nokl" and "H200 r8-a8-nokl" are
**not** a clean α-only A/B — they're identical only if the SOTA
yaml on the H200 box matches the inferred row above. Until that
yaml is committed to `cloud_sweep/`, the analysis below treats
α/r as the **only** difference between the two H200 runs; if the
SOTA box actually had a different `learning_rate` or batch, the
attribution shifts.

### Source of truth for each row

- **H200 r8-a16-nokl** (the failure mode): config at
  `src/finetune/configs/cloud_sweep/r8-a16-nokl.yaml`
  after commit `fb15d6b "sweep: add r8-a16-nokl-vision config,
  fix queue typo, H200 batch sizing"`. That commit bumped
  `per_device_train_batch_size 2 → 32`, `gradient_accumulation_steps
  8 → 1`, `learning_rate 2.0e-4 → 3.0e-4`,
  `projector_learning_rate 1.0e-4 → 1.4e-4`,
  `per_device_eval_batch_size 4 → 32` — the exact knobs that
  produce `S_step = 0.0192`. There is an **earlier** H200 run
  (`r8-a16-kl000_20260517_031850`, train.log:30-36) that used
  the pre-`fb15d6b` settings (`bs=2 × ga=8 = 16`, `lr=2e-4`) and
  is therefore in the 4090 cell of the table, not the H200
  r8-a16 cell.

- **H200 r8-a8-nokl** (SOTA): no yaml in
  `src/finetune/configs/`. Either H200-local edit or
  unsynced branch. Assumed = the same post-`fb15d6b` yaml with
  `lora_alpha: 8` substituted. **Action item**: commit this yaml
  before next sweep so the SOTA recipe is reproducible from the
  repo alone.

- **4090 r8-a16-drop005-mix50k**: actual training log at
  `src/finetune/outputs/r8-a16-drop005-mix50k_20260517_134255/train.log:30-36`
  confirms `bs=2, ga=8 (eff_bs=16), lr=2e-4, lora_alpha=16,
  projector_lr=1e-4`. Eval results per checkpoint (Mac M5 Pro eval):

  | step | plant | mmlu | aime | llava | refusal | composite |
  |---|---|---|---|---|---|---|
  | 4000 | 0.140 | 0.560 | 0.150 | 0.143 (leak=0%) | 0.000 | 0.243 |
  | 5000 | 0.150 | **0.620** | 0.200 | 0.128 (leak=0%) | **1.000** | **0.380** |
  | 6000 | 0.150 | 0.580 | 0.200 | 0.150 (leak=0%) | 1.000 | 0.372 |

  Peak composite at step 5k. mmlu peaks at 5k then drops 4pp by 6k
  (mild overfitting signal). Refusal flips from 0% (4k) to 100%
  (5k/6k) — safety behavior recovered mid-training. plant plateaus
  at 0.15 from step 4k onward.

## Why `S_step = (α/r) × η × eff_bs` is the right object to compare

A LoRA-only forward pass yields, for a layer with frozen weight
`W` and adapter `BA`,

```
h = W x  +  (α / r) · (B A) x
```

so `(α / r)` is the linear gain on the adapter's contribution to
the activation. One Adam update with step `η` and gradient
estimated over `eff_bs` independent examples moves `(B, A)` by
roughly `η · g_hat`, where the gradient signal-to-noise of
`g_hat` scales **linearly** with `eff_bs` once the per-example
gradient is non-trivial (large-batch regime; cf. McCandlish et
al. 2018 "Empirical Model of Large-Batch Training"). The
adapter's output delta on a downstream token, in one optimizer
step, is therefore

```
Δh  ∝  (α / r)  ·  η  ·  eff_bs
```

This is the quantity that competes against the base model's
log-prob landscape for `mmlu` / `aime` / `llava`. Once this
per-step push exceeds the "anchoring radius" provided by KL,
L2, dropout, or the mix-50k anti-forgetting buckets, the LoRA
delta drifts enough per step to overwrite generality faster
than the anti-forgetting machinery can compensate.

The three runs above sit on three different sides of that
threshold:

- **S_step ≈ 0.019** (H200 r8-a16): above threshold → mmlu
  decays from 0.46 (base) → 0.26 in 2 000 optimizer steps.
- **S_step ≈ 0.010** (H200 r8-a8 SOTA): just below threshold →
  mmlu held at base levels even after 4 600 steps × 147 k
  samples, plant learning still progresses.
- **S_step ≈ 0.006** (4090 r8-a16-mix50k): well below threshold
  → mmlu actually **climbs** to 0.62 (above base 0.46) by step
  5 000, peaks there, then dips to 0.58 by step 6 000 (mild
  overfitting). Plant plateaus at 0.15 from step 4k onward.

## Trade space — three knobs, one threshold

Because S_step is the product of three terms, there's a family
of (α/r, η, eff_bs) tuples that all sit at the same generality-
safety budget. Two practical operating points emerge from the
data above:

**Operating point A — "high throughput, half scale" (SOTA recipe, H200)**
- `α/r = 1.0` (e.g. r=8, alpha=8 — what H200 SOTA used)
- `η = 3.0e-4` (cosine, warmup_ratio=0.03)
- `eff_bs = 32` (H200 fits per_device=32, ga=1)
- `S_step ≈ 0.0096`
- Wall-time: ~best throughput on H200 (large per-device batch).
- Reproducibility cost: needs r8-a8-nokl yaml committed.

**Operating point B — "standard scale, half LR, half batch" (4090)**
- `α/r = 2.0` (r=8, alpha=16 — the canonical LoRA scale)
- `η = 2.0e-4`
- `eff_bs = 16` (4090 24 GB ceiling at r=8 + projector + 960×672
  vision)
- `S_step ≈ 0.0064`
- Wall-time: ~best the 4090 can sustain without OOM.
- Reproducibility cost: yaml already exists in
  `local_sweep/r8-a16-drop005-mix50k.yaml`.

The two points differ in **plant learning speed**: A is at
1.5× the per-step push of B, so for a given step count A
learns plant faster. To match A's plant score on 4090 (B),
budget ~1.5× more steps (i.e. ~7 500 step instead of 4 600).

**Anti-pattern — H200 r8-a16-nokl (post-fb15d6b)**
- `α/r = 2.0`, `η = 3.0e-4`, `eff_bs = 32` → S_step ≈ 0.019.
- This stacks all three knobs at the high end simultaneously.
- Crashes mmlu by step 2 000. Do not run this configuration
  without an active KL anchor.

## Concrete recommendations

1. **Commit the H200 r8-a8-nokl yaml.** Without it the SOTA
   recipe is not reproducible from `<repo>/` alone.
   Suggested filename: `configs/cloud_sweep/r8-a8-nokl.yaml`,
   identical to `r8-a16-nokl.yaml` except `lora_alpha: 8`.

2. **Retire post-fb15d6b `r8-a16-nokl.yaml` from the live sweep
   queue, or pair it with KL ≥ 0.005.** With S_step ≈ 0.019 and
   no anchor it is a known mmlu-killer.

3. **For any future H200 batch-size bump, scale lr by the
   inverse-square-root rule, not the linear rule.** The pre-
   fb15d6b config (`bs=16, lr=2e-4`) → linear-scaled to `bs=32`
   would give `lr=4e-4`; sqrt-scaled gives `lr=2.8e-4`. The
   actually-shipped `lr=3e-4` is between the two, closer to
   sqrt — which is the right side, but at α/r = 2.0 even
   sqrt-scaled lr pushes S_step over the cliff. The lesson is
   that **when you raise eff_bs, drop α (or add KL) to keep
   S_step constant**, rather than relying on the lr schedule
   alone.

4. **For the 4090 local sweep, S_step is already conservative.**
   If we want faster plant learning without re-introducing
   forgetting, the cheapest move is `lr 2e-4 → 2.5e-4` (S_step
   → 0.008, still below the H200 SOTA point). Bumping
   `gradient_accumulation_steps 8 → 16` (eff_bs → 32) gives
   the same `S_step` ≈ 0.013 as the SOTA but doubles per-step
   wall time and risks OOM under projector + augmentation.

5. **Apples-to-apples α A/B requires fixed (lr, eff_bs).** The
   current "α=8 vs α=16" comparison between H200 SOTA and H200
   crash is confounded by the simultaneous lr+batch bump in
   `fb15d6b`. To attribute the SOTA win to α alone, run
   `r8-a8-nokl` and `r8-a16-nokl` on the same hardware **at
    the same lr and eff_bs** (either both at `bs=32, lr=3e-4`,
    or both at `bs=16, lr=2e-4`). The project timeline
   does not strictly require this for the final deliverable,
   but the writeup should note the confound.

## Reference numbers (where the table cells come from)

| cell | source path | line(s) |
|---|---|---|
| H200 r8-a16-nokl yaml (post-fb15d6b) | `src/finetune/configs/cloud_sweep/r8-a16-nokl.yaml` | 19-67 |
| H200 r8-a16-nokl yaml (pre-fb15d6b) | `git show fb15d6b^:finetune/configs/cloud_sweep/r8-a16-nokl.yaml` | — |
| H200 r8-a16-kl000 actual log (pre-fb15d6b config) | Cloud-side run log (not in public repo) | 30-36 |
| H200 batch-size bump commit | this repo's git | `fb15d6b "sweep: add r8-a16-nokl-vision config, fix queue typo, H200 batch sizing"` |
| 4090 r8-a16-mix50k yaml | `src/finetune/configs/local_sweep/r8-a16-drop005-mix50k.yaml` | 33-67 |
| 4090 r8-a16-mix50k actual log | `src/finetune/outputs/r8-a16-drop005-mix50k_20260517_134255/train.log` | 30-36 |
| 4090 step-4000/5000/6000 generality eval | Mac M5 Pro eval session 2026-05-17 | — |
| H200 r8-a8-nokl yaml | **MISSING from `<repo>/` — to be committed** | — |
| H200 r8-a8-nokl run log | **Missing from synced run logs** | — |

## Open items

- Sync `r8-a8-nokl_*` and `r8-a16-nokl_*` run logs
  from the H200 box so this doc can replace the "inferred yaml"
  row with an actual log-confirmed row.
- Once committed, add the matching plant / mmlu / aime numbers
  per checkpoint (step 1k / 2k / 3k / 4k / 4.6k for SOTA) so
  the S_step → metric curve can be plotted directly rather than
  argued by single-point comparison.
- Re-evaluate whether S_step's batch-size term should be
  `eff_bs` (linear, large-batch noise-floor regime) or
  `√eff_bs` (small-batch noise regime). The three current data
  points don't separate the two — both orderings give the same
  monotonic ranking on mmlu. A clean A/B at fixed (α, lr) with
  `eff_bs ∈ {16, 32, 64}` would settle this.

---

# Patch: actual H200 run data + corrections to the original TL;DR table

Added 2026-05-17 evening (H200-side supplement).

**Key finding**: the row in the original TL;DR table labelled
"H200 r8-a16-nokl" with config `bs=32, lr=3e-4, S_step=0.0192`
**was never actually run**. The H200 run that did execute,
`r8-a16-nokl_163521`, used the yaml from before commit `fb15d6b`
(`bs=2 × ga=8 = 16, lr=2e-4`), giving S_step = **0.0064** —
identical to the 4090 row — yet mmlu still collapsed. This pokes a
hole in the original "S_step single-variable explains everything"
narrative and forces a refinement.

## 1. Real H200 run config audit

### `r8-a8-nokl_20260517_193530` (SOTA, ran a full 3 epochs)

Resolved config dumped at `train.log:9-37` (no yaml exists for this
run in `src/finetune/configs/`; the train.log captures the full
resolved config):

```
lora.r = 8
lora.lora_alpha = 8                       → α/r = 1.0
lora.lora_dropout = 0.05
lora.projector_learning_rate = 1.4e-4
lora.tune_projector = True
lora.tune_last_n_vision_layers = 0
training.per_device_train_batch_size = 32
training.gradient_accumulation_steps = 1  → eff_bs = 32
training.num_train_epochs = 3
training.learning_rate = 3.0e-4
training.warmup_steps = 30
training.lr_scheduler_type = cosine
training.save_steps = 1000
regularization.kl_enabled = False
```

→ **S_step = 1.0 × 3.0e-4 × 32 = 0.0096** (matches the original
estimate).

The run started at 19:35 UTC; commit `fb15d6b` landed at 19:41 UTC.
The yaml for this run exists locally on the H200 box but was not
committed to this repo's git, so **the SOTA recipe still cannot be
reproduced from this repo alone**. The train.log is currently the
only source of truth.

### `r8-a16-nokl_20260517_163521` (mmlu collapsed, but not because S_step is high)

`train.log:9-37`:

```
lora.r = 8
lora.lora_alpha = 16                      → α/r = 2.0
lora.lora_dropout = 0.05
lora.projector_learning_rate = 1.0e-4     (note: not 1.4e-4)
training.per_device_train_batch_size = 2
training.gradient_accumulation_steps = 8  → eff_bs = 16
training.num_train_epochs = 3
training.learning_rate = 2.0e-4
regularization.kl_enabled = False
```

→ **S_step = 2.0 × 2.0e-4 × 16 = 0.0064**.

Run start: 16:35 UTC. Commit `f18eb35` (the initial yaml with
`bs=2 lr=2e-4`) landed at 16:23 UTC; commit `fb15d6b` (the
`bs=32 lr=3e-4` upgrade) didn't land until 19:41 UTC. So this run
used **the f18eb35 version** — same S_step, same α/r, same lr as
the 4090 `r8-a16-drop005-mix50k`. **In theory the mmlu trajectory
should match the 4090 run**.

### Yaml corresponding to the original TL;DR first row (`S_step=0.0192`)

= `src/finetune/configs/cloud_sweep/r8-a16-nokl.yaml` at HEAD
(post-`fb15d6b`), i.e. `bs=32, ga=1, lr=3e-4, lora_alpha=16`.
**We never actually ran this yaml**. The original row was strictly
a "what would happen if we ran it" prediction, not a measurement.

## 2. Real generality-eval data (produced in this session on H200)

`PLANT_IMAGE_ROOT=<workspace>/src/finetune/data/english-desc-v2/images_resized/test`,
`--skip_judge` (MMLU/AIME are multiple-choice with direct answer
comparison; plant uses species_match; llava uses ROUGE-L; no Qwen
judge needed):

| run / step | α/r | eff_bs | lr | S_step | samples | plant | mmlu | aime | composite |
|---|---|---|---|---|---|---|---|---|---|
| **base** (`unsloth/gemma-4-E2B-it`) | — | — | — | — | 0 | 0.000 | 0.460 | 0.100 | — |
| r4-a8-kl005 step 1000 | 2.0 | 16 | 2e-4 | 0.0064 | 16k | 0.000 | 0.540 | 0.150 | 0.280 |
| **r8-a8-nokl step 4647 (SOTA)** | **1.0** | 32 | 3e-4 | 0.0096 | 148.7k | **0.230** | **0.480** | **0.200** | **0.319** |
| r8-a16-nokl step 1000 | 2.0 | 16 | 2e-4 | 0.0064 | 16k | 0.040 | 0.280 | 0.200 | 0.169 |
| r8-a16-nokl step 2000 | 2.0 | 16 | 2e-4 | 0.0064 | 32k | 0.020 | 0.260 | 0.100 | 0.131 |
| r256-a256-kl005 step 1000 | 1.0 | 16 | 2e-4 | 0.0064 | 16k | 0.000 | **0.100** | 0.050 | 0.050 |

Training-loss trajectories not in the original table (from
`metrics.jsonl`):

```
r8-a8-nokl  step 1000 (0.65 epoch):  plant=0.700  nonplant=1.538
            step 2000 (1.29 epoch):  plant=0.494  nonplant=1.511
            step 3000 (1.94 epoch):  plant=0.394  nonplant=1.488
            step 4000 (2.58 epoch):  plant=0.360  nonplant=1.491
            step 4647 (3.00 epoch):  plant=0.357  nonplant=1.488

r8-a16-nokl step 1000 (0.32 epoch):  plant=1.004  nonplant=1.557
            step 2000 (0.65 epoch):  plant=0.666  nonplant=1.521
            (killed at step 2000)
```

## 3. S_step single-variable theory vs the actual measurements

The original §"Why S_step is the right object" claimed:

- S_step ≈ 0.006 → mmlu *holds or even rises* (the 4090's 0.62)
- S_step ≈ 0.010 → mmlu maintained (H200 SOTA)
- S_step ≈ 0.019 → mmlu collapses

Actual measurements:

| run | S_step | measured mmlu |
|---|---|---|
| 4090 r8-a16 step 5k | 0.0064 | 0.620 (as originally reported) |
| H200 r8-a16-nokl step 2k | 0.0064 | 0.260 |
| H200 r8-a8-nokl step 4.6k | 0.0096 | 0.480 |
| H200 r4-a8-kl005 step 1k | 0.0064 | 0.540 |
| H200 r256-a256-kl005 step 1k | 0.0064 | **0.100** |

Runs with the **same** S_step (0.0064) spread mmlu from 0.10 to
0.62 — a 6.2× range. S_step alone is clearly insufficient. At
least three variables the formula doesn't absorb:

1. **Rank itself (independent of α/r)**. r=256 with α/r=1.0 still
   collapses straight to mmlu=0.10. The higher the LoRA subspace
   dimensionality, the more "directions" the base model's logit
   manifold gets pushed away in, and the harder it is to recover
   even if each step is small.
2. **Sample-exposure pace (`step × eff_bs`)**. When the 4090 was
   evaluated at step 5000 it had probably traversed a "mmlu dips
   then recovers" U-shape; H200 r8-a16 was killed at step 2000,
   right at the bottom of that U-shape. The original claim that
   "total exposure doesn't explain things" only holds when
   comparing 4090 (80k) vs SOTA (148k); when comparing
   r8-a16-nokl (32k) vs 4090 (80k), the 2.5× gap in samples seen
   does account for some of the trajectory difference.
3. **The real non-linearity between rank and α**. Adam's second-
   moment estimate normalizes gradient magnitudes, so the
   linear product of lr and gradient size doesn't fully determine
   the actual LoRA-weight displacement. `α/r` controls both
   forward gain and backward-grad gain; **after Adam there's still
   a residual √2 effective-lr bias** (rough estimate, since Adam's
   `v` term partially cancels the gradient scale).

Revised working hypothesis:

S_step is the **lower bound on forgetting velocity**, but the
actual mmlu collapse couples (α/r, rank, cumulative steps). A run
at `α/r=2.0` is more prone to getting stuck at the bottom of the
mmlu dip than an `α/r=1.0` run at any S_step; r=256 wrecks the
anchor entirely, no matter how small each step is.

## 4. Revisions to the original "specific recommendations" list

**Rec 1 "commit r8-a8-nokl yaml"**: still valid. Reproduced verbatim
from train.log:

```yaml
# configs/cloud_sweep/r8-a8-nokl.yaml (pending commit)
model:
  base_model: "unsloth/gemma-4-E2B-it"
  max_seq_length: 1024
  dtype: bfloat16
lora:
  finetune_vision_layers: false
  finetune_audio_layers: false
  finetune_language_layers: true
  finetune_attention_modules: true
  finetune_mlp_modules: true
  r: 8
  lora_alpha: 8
  lora_dropout: 0.05
  bias: "none"
  random_state: 3407
  tune_projector: true
  projector_learning_rate: 1.4e-4
  tune_last_n_vision_layers: 0
  vision_layers_learning_rate: 1.0e-5
training:
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1
  num_train_epochs: 3
  max_steps: -1
  learning_rate: 3.0e-4
  warmup_steps: 30
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  weight_decay: 0.001
  optim: "adamw_torch_fused"
  save_steps: 1000
  save_total_limit: null
  eval_strategy: "steps"
  eval_steps: 1000
  per_device_eval_batch_size: 32
  modality_aware_sampler: true
  group_by_length: false
  tf32: true
data:
  train_file: "data/mix-50k/train.jsonl"
  val_files:
    plant:      "data/mix-50k/val_plant.jsonl"
    nonplant:   "data/mix-50k/val_nonplant.jsonl"
    negative:   "data/mix-50k/val_negative.jsonl"
    offline_qa: "data/mix-50k/val_offline_qa.jsonl"
  augmentation: true
  prompt_prefixes:
    camera_on:  "[camera=on] "
    camera_off: "[camera=off] "
eval:
  enabled: true
  max_eval_samples: 300
  max_new_tokens: 256
  use_unsloth: true
  batch_size: 1
regularization:
  kl_enabled: false
  kl_weight: 0.0
  l2_enabled: false
```

**Rec 2 "retire post-fb15d6b r8-a16-nokl.yaml"**: **not directly
measured**, but based on the known r256-a256-kl005 collapse (same
S_step=0.0064, only going up to 0.0192) and the r8-a16-nokl_163521
collapse (S_step=0.0064), it's reasonable to infer the post-fb15d6b
r8-a16-nokl (S_step=0.0192) would only do worse. **Not worth
running just to falsify**.

**Rec 3 "sqrt-scale lr when batch goes up"**: still valid, but only
under fixed α/r=1.0. At α/r=2.0 this recommendation can't save you.

**Rec 5 "apples-to-apples α A/B"**: **actually confounded**, because
H200 SOTA and H200 r8-a16-nokl differ not only in α but also in
**lr (3e-4 vs 2e-4) and eff_bs (32 vs 16)**. A clean A/B requires
a new H200 config: r=8, α=16, lr=3e-4, bs=32 (= the current
r8-a16-nokl.yaml at HEAD) compared against the r8-a8-nokl SOTA.
This recommendation is more important than the original text
indicated, **because the current SOTA win cannot be attributed to
α as a single variable**.

## 5. Additional H200 datapoints (not in the original text)

- **r4-a8-kl005 step 1000** is another strong "mmlu held" point
  (0.54), at S_step = 2.0 × 2e-4 × 16 = 0.0064. But plant=0 (rank
  too low to learn plant features). Shows that **small rank +
  standard KL preserves mmlu but doesn't learn plant**.
- **r4-a8-kl003 step 4000**: plant=0 again, mmlu=0.52. The KL
  0.05 → 0.03 difference is negligible — at r=4, KL doesn't help
  at all.
- **r256-a256-kl005 step 1000**: plant=0, mmlu=0.10, aime=0.05.
  Even though S_step matches r4-a8-kl005, rank=256 alone is
  enough to wreck mmlu. This is the strongest evidence yet that
  **rank itself is an independent collapse axis**.
- **r32-a32-kl005 step 3000**: plant=0, mmlu=0.52, aime=0.05.
  Compared to step 1000, aime fell from 0.15 to 0.05 — **further
  training doesn't fix plant=0; it overfits nonplant instead**.
  Without unfreezing the vision tower, language-side LoRA simply
  cannot learn plant.

## 6. Current production recipe (data as of 2026-05-17 21:30 UTC)

**The only empirically-verified config that produces both plant > 0
and mmlu ≥ base**:

```
r = 8
lora_alpha = 8           (α/r = 1.0 is a necessary condition)
lora_dropout = 0.05
projector_lr = 1.4e-4
tune_projector = True
tune_last_n_vision_layers = 0
training:
  effective batch = 32   (on H200: bs=32, ga=1)
  learning_rate = 3.0e-4
  num_train_epochs = 3   (= ~4647 steps)
  warmup_steps = 30
regularization:
  kl_enabled = False     (no KL needed; small rank self-regularizes)
```

vision_tower stays frozen the entire run. plant=0.23 comes entirely
from projector updates + language-side LoRA. To push plant further,
**the next thing to try is vision-tower last-N finetuning while
keeping α/r=1.0**, not enlarging LoRA capacity. The latter was
already disproved by r256-a256-kl005.

## 7. Open questions for the next sweep

- r8-a8-nokl plant / mmlu / aime trajectory at steps 1000 / 2000 /
  3000 / 4000. Currently only step 4647 is measured. Filling these
  in lets us draw a plant-vs-samples-seen curve and tell whether
  SOTA is monotonically rising or U-shaped. Checkpoints are still
  at `sft_out/r8-a8-nokl_20260517_193530/`; eval can be backfilled.
- If r8-a16-nokl is rerun to step 5000 (80k samples), does mmlu
  recover from 0.26 back toward the 4090's 0.62? If yes,
  **S_step theory holds approximately given enough samples**; if
  no, **α/r is an independent collapse axis**. The run was killed
  and isn't currently in the sweep queue — it needs to be re-added.
- r8-a8-nokl + vision-tower last-2 finetuning: can plant push from
  0.23 to 0.40+ while mmlu holds? `r8-a16-kl005-vision` died
  earlier (0 checkpoints), so this needs a restart.

---

# Patch 2: per-checkpoint val_loss trajectories from the 4090 local runs + a new vision-unfrozen 24-hour datapoint

Added 2026-05-17 evening, after re-trawling the local
`finetune/outputs/`.

**Filter** (per the project-timeline 05-17 directive): only use runs
from the past 24 h, on `data/mix-50k/`, with v4 camera-state prefix
(`[camera=on]/[camera=off]`); reject v3 `[task=...]` and pure-PlantNet
runs. Earlier `plantnet-50k-baseline-{lora,qlora}-r256-*`
(PlantNet-only / task-tag-prefix) runs are dropped.

Local runs that pass the filter:
- **`r8-a16-drop005-mix50k_20260517_134255`** (already in the
  original tables)
- **`plantnet-50k-mix-lora-r256+fullproj+vision2-lr5e5-modalityaware_20260516_114454`**
  — new here. This is the only `tune_last_n_vision_layers=2` local
  run with vision unfrozen on mix-50k + camera-prefix within the
  24 h window.

## A1. Cross-run val_loss trajectory (epoch-aligned, not step-aligned)

This is the comparison axis missing from the original H200 docs/08
tables. Every val_loss number comes from
`checkpoint-*/trainer_state.json:log_history`, i.e. data the trainer
itself reports — no eval rerun needed.

| epoch | 4090 r8-a16-mix50k (α/r=2.0, lr=2e-4, eff_bs=16) | H200 r8-a16-nokl (same config) | H200 r8-a8-nokl SOTA (α/r=1.0, lr=3e-4, eff_bs=32) | 4090 r256-vision2 (α/r=1.0, lr=2e-4, eff_bs=16, vision last-2) |
|---|---|---|---|---|
| 0.32 | **0.935** (step 1000) | 1.004 (step 1000) | — | **0.834** (step 1000) |
| 0.65 | **0.627** (step 2000) | 0.666 (step 2000) | 0.700 (step 1000) | 0.621 (step 2000) |
| 0.97 | 0.549 (step 3000) | — (killed) | — | — (killed) |
| 1.29 | 0.470 (step 4000) | — | 0.494 (step 2000) | — |
| 1.62 | 0.431 (step 5000) | — | — | — |
| 1.94 | — | — | 0.394 (step 3000) | — |
| 2.58 | — | — | 0.360 (step 4000) | — |
| 3.00 | — | — | 0.357 (step 4647) | — |

Each cell is `eval_plant_loss` (token-level CE on 300 val samples).

Data sources:
- 4090 r8-a16-mix50k:
  `src/finetune/outputs/r8-a16-drop005-mix50k_20260517_134255/checkpoint-5000/trainer_state.json:log_history`
- 4090 r256-vision2:
  `src/finetune/outputs/plantnet-50k-mix-lora-r256+fullproj+vision2-lr5e5-modalityaware_20260516_114454/checkpoint-2000/trainer_state.json:log_history`
- The two H200 rows: extracted from the project-timeline patch 1 §2
  `metrics.jsonl` snippet.

## A2. Three new observations (not in the original text or patch 1)

### A2.1 Cross-hardware noise floor at the same yaml is small

H200 `r8-a16-nokl` and 4090 `r8-a16-drop005-mix50k` use the **exact
same** config (same `α/r=2.0`, `lr=2e-4`, `eff_bs=16`, same mix-50k
data, same camera prefix, same `seed=3407`, same `lora_dropout=0.05`),
yet:

- epoch 0.32: H200 plant_loss = 1.004 vs 4090 = 0.935 (**Δ=0.069,
  4090 lower**)
- epoch 0.65: H200 = 0.666 vs 4090 = 0.627 (**Δ=0.039, 4090 lower**)

The 4090 fits plant_loss lower at both points. Likely causes (not
verified): non-deterministic cuBLAS, differing bf16 reduction order,
`dataloader_num_workers 8 → 4` (H200 yaml uses 4, 4090 yaml uses 8)
shuffling batch sequences differently. But the magnitude is small
(~5 %) and **does not explain** the ~4× generation gap between
H200 r8-a16-nokl's `plant_match=0.04` and the 4090 r8-a16's 0.15.
The generation gap comes from checkpoint selection and eval timing,
not from the training hardware.

(Implication: S_step theory says "same config, same step → similar
plant", which the ~5 % val_loss agreement supports; the 4×
plant_match gap contradicts S_step, but that's on the evaluation
side, not the training side — see §A2.3.)

### A2.2 4090 r8-a16: **nonplant + offline_qa val_loss falls monotonically**

This is direct training-side evidence for the docs/09 §3 claim that
"anti-forgetting force comes from the data mix". Pulled from
`trainer_state.json`:

| step | epoch | eval_plant | eval_nonplant | eval_offline_qa |
|---|---|---|---|---|
| 1000 | 0.32 | 0.935 | 1.544 | 1.541 |
| 2000 | 0.65 | 0.627 | 1.514 | 1.403 |
| 3000 | 0.97 | 0.549 | 1.496 | 1.316 |
| 4000 | 1.29 | 0.470 | 1.483 | 1.218 |
| 5000 | 1.62 | 0.431 | 1.476 | 1.220 |
| 6000 | 1.94 | 0.407 | 1.471 | 1.173 |

`eval_nonplant_loss` goes 1.544 → 1.476 (−4.4 %),
`eval_offline_qa_loss` goes 1.541 → 1.220 (−20.8 %),
**never rising at any step**. This is consistent with the
generation-level `mmlu = 0.62 (> base 0.46)` — token CE on non-plant
data is **genuinely improving**, not just random eval noise.

Contrast with docs/09 §3.b's r256-a256-kl005 collapse to mmlu=0.10:
for that run, `eval_nonplant_loss` should **rise** past step 1000
(pending the H200 trainer_state being synced over for verification).
If that rise can be confirmed, **monotonic `eval_nonplant_loss`** can
serve as a cheap in-training indicator for "hasn't collapsed yet" in
the next sweep — no need to wait for generation eval.

### A2.3 Val_loss convergence speed ≠ generation accuracy: α/r=2.0 is "willing to say but says it wrong"

Putting training-side (val_plant_loss) and eval-side (plant_match
species recognition rate) numbers side by side:

| run | epoch | val_plant_loss | plant_match | "match-pp per nat of CE" efficiency |
|---|---|---|---|---|
| 4090 r8-a16 (α/r=2.0) step 4k | 1.29 | 0.470 | 0.140 | ~0.30 (poor) |
| 4090 r8-a16 (α/r=2.0) step 5k | 1.62 | 0.431 | 0.150 | ~0.35 (poor) |
| 4090 r8-a16 (α/r=2.0) step 6k | 1.94 | 0.407 | 0.150 | ~0.37 (poor) |
| H200 r8-a8 SOTA (α/r=1.0) | 3.00 | 0.357 | 0.230 | ~0.64 (good) |
| H200 r8-a16-nokl (α/r=2.0) | 0.65 | 0.666 | 0.040 | ~0.06 (very poor) |
| 4090 r256-vision2 (α/r=1.0, vision last-2) | 0.65 | 0.621 | (not measured) | — |

The two α/r=2.0 runs have **faster token-CE convergence** (lower
plant_loss) but **much lower plant_match** than the α/r=1.0 SOTA.
This is direct evidence for the docs/08 patch 1 §3 hypothesis that
"α/r=2.0's forward gain pushes the logit toward sharper species-
token distributions" — teacher-forced CE benefits from "dare to
say", but in autoregressive sampling, "dare to say the wrong
species" and "hedge and not say" both count as 0 match, so the
generation side actually loses.

This adds a layer to the docs/08 §3 "anchoring radius" theory:
**α/r controls not only forgetting rate but also generation
calibration**. The α/r=1.0 SOTA isn't winning because it learns
slowly — **it's winning because it learns cautiously**.

### A2.4 4090 r256-vision2: the only local vision-unfrozen datapoint in the 24 h window

Within the mix-50k + camera-prefix-v4 + 24 h filter, this is the
**second qualifying local run** after r8-a16-mix50k. Its
distinguishing features:

- `tune_last_n_vision_layers: 2` (last 2 vision-tower layers
  unfrozen + lr=1e-5)
- `lora_alpha=256` (α/r=1.0, same scale as SOTA)
- `lora_dropout=0.05`, both KL and L2 disabled
- Ran 2000 steps (epoch 0.65) then was killed (reason not in the
  metrics; likely a manual kill to free GPU for r8-a16-mix50k)

Differences vs H200 SOTA:
- **rank 256 vs 8** (huge capacity gap)
- **vision unfrozen vs fully frozen**
- same α/r=1.0

At epoch 0.65, val_plant_loss = 0.621 — **lower than the H200 SOTA's
0.700 at the same epoch**. So **on plant_loss convergence speed
alone**, `r=256 + vision unfreeze` ≥ `r=8 + vision frozen`.
However, docs/09 §1 already showed that `r=256 + KL=0.05` collapses
mmlu to 0.10. KL is disabled here (matching SOTA), so the mmlu
behaviour is unknown — but docs/09 §2.b's argument ("KL is the
counter-force at small rank; rank itself is the independent
forgetting axis at high rank") has not yet been falsified or
confirmed by an mmlu number from this run.

**Recommendation**: on H200, run `r=256 + α=256 + nokl + vision
last-2 + lr=2e-4 + eff_bs=16` (the full local yaml replayed) out to
step 4000+ and do a generation eval. If mmlu < 0.40, docs/09 §2.b
stands (rank itself is an independent collapse axis, decoupled from
KL); if mmlu > 0.55, the claim that "KL=0.05 is the main cause of
the r=256 mmlu collapse" is overturned → the whole small-rank
preference needs re-examination.

## A3. One-paragraph patch summary

On training-side val_loss (which can be pulled for free from
`trainer_state.json`, no eval rerun needed), the 4090 r8-a16-mix50k
and H200 r8-a16-nokl have a cross-hardware noise floor of only
~5 %; within a run, nonplant + offline_qa val_loss both **fall
monotonically**, confirming the docs/09 §3 claim that "the data mix
provides a dense anti-forgetting signal." But the conversion rate
from `val_plant_loss` to `plant_match` is ~2× higher at α/r=1.0
than at α/r=2.0, so the SOTA's advantage isn't "it learns more" —
**it's "it learns carefully"**. The next sweep should add
r=256+vision-last2+nokl as the definitive test of "is rank itself
an independent collapse axis."

**Note on the HF Trainer resume LR-scheduler diagnostic**: we
suspected that resume would lock cosine total steps to the old
value and drive lr to zero prematurely. After checking the
transformers 5.8 source and measuring the actual lr trajectory, it
was determined to be **not a bug** — full diagnosis in
[`../general/15-postmortems.md`](../general/15-postmortems.md) §6.
