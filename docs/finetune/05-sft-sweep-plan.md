# SFT sweep plan — adaptive multi-stage exploration

## TL;DR

- This doc lays out a staged experiment plan for improving supervised finetuning accuracy before deployment quantization.
- The sweep starts with single-knob ablations, promotes promising settings into combinations, then runs full evaluation and quantization on the winners.
- The target was to beat the baseline-1 PlantNet score of 70.6% by a few percentage points while keeping the experiment interpretable.
- Early Stage 1 results found strong candidates after correcting an evaluation-reference mistake, with two variants improving the quick eval by more than 12 percentage points.

> last edit: 2026-05-16
> Strategic + execution plan for the pre-deadline SFT push.
> Companion to:
> - `01-pipeline.md` — baseline LoRA-only pipeline
> - `02-projector-mode.md` — `tune_projector` mode
> - `03-vision-mode.md` — `tune_last_n_vision_layers` mode
> - `../../quantization/A-baseline2-qlora-progress.md` — QLoRA route A

## Sweep Plan

We have one baseline (`baseline-1` bf16 LoRA r=256 + full projector +
data-aug-enwiki, 5 epoch). Full eval = 70.6 % PlantNet n=2,870.
Deadline is days, not weeks. Goal: push the SFT ceiling **+2-5 pp**
before quantization, in a way that the report can defend as principled.

Plan: rent **2× remote 5090 GPU (32 GB VRAM)** independent boxes. Run
SFT in **3 adaptive stages**:

| Stage | Configs | Wall (per machine) | Decision criterion to advance |
|---|---|---|---|
| **Stage 1** | 4 single-knob ablations on the existing code path (zero new code) | ~16-18 h | top-2 by PlantNet quick-eval (n=300) |
| **Stage 2** | 3-4 combos from Stage 1 winners + small enabler PR (DoRA / PiSSA / data-side) | ~16-20 h | top-1 by PlantNet quick-eval |
| **Stage 3** | 1-2 winners get full eval (n=2,870) + quantization pipeline | ~6-8 h (eval + quant) | n/a — terminal |

Total elapsed: **~2-3 days** assuming both machines run in parallel.

The two machines are **independent remote GPU hosts**; do not assume
shared filesystem. Each runs its own queue, results rsync back to the
local 4090 desktop for eval comparison.

**Stage 1 status (2026-05-16, evening — major correction):** baseline4,
baseline5, baseline6 finished. Initial reading compared them to an
**n=200** archived baseline-1 number (72 %) and concluded all three
were flat/negative; that comparison was apples-to-oranges. After
re-eval'ing baseline-1 at n=300 on the **same canonical val.jsonl**
the Stage 1 configs use, the corrected picture is the inverse:

| Config | n=300 species | Δ vs baseline-1 (n=300) | Verdict |
|---|---|---|---|
| baseline-1 (n=300, re-eval) | 58.00 % | — | reference |
| baseline-4 (`+drop005`) | 70.67 % | **+12.67 pp** | **Strong signal** ✅ |
| baseline-5 (`-a512`)     | 62.67 % | **+4.67 pp** | Signal ✅ |
| baseline-6 (`-projlr1e4`)| 71.67 % | **+13.67 pp** | **Strong signal** ✅ |

Stage 2 promotion candidates by the existing decision tree are
baseline4 and baseline6 (both ≥ +1.5 pp). baseline5's +4.67 pp is
weaker but real; the original "α=2r crashes training" hypothesis was
based on the wrong reference and is **withdrawn**. baseline7 still
queued. Full breakdown: §"Stage 1 results (2026-05-16 update)" below;
the diagnosis of the n=200-vs-n=300 reference mistake is in
§"baseline-1 reference re-aligned" inside that section.

> ⚠ **Caveat — stale val.jsonl on disk.** All four numbers in the table
> above were measured on the **current 5,000-row / 605-species
> val.jsonl**. That file under-represents the long tail (177 / 782
> species end up in train only, not val) because it was generated
> before the `class_stratified_split` fix landed. Status of the fix:
>
> - ✅ **Code fix already landed in `@6e49ec2`** ("fix(prepare):
>   per-species stratified train/val split"). `class_stratified_split`
>   is wired into both `prepare_plantnet.py` and
>   `prepare_plantnet_enriched.py`; the old `samples[:val_count]` slice
>   is gone.
> - ❌ **`val.jsonl` on disk has NOT been regenerated yet.** mtime
>   2026-05-15, predates the fix. `wc -l val.jsonl` → 5000, species
>   count 605 — those numbers will change to ~5000+ rows / 782 species
>   after a `bash src/finetune/scripts/run/prepare_plantnet_50k.sh`
>   on this checkout.
>
> Once val.jsonl is regenerated, baseline-1 / baseline-4 / baseline-5 /
> baseline-6 should all be re-eval'd on the new val. **Relative
> ordering (Δ between Stage 1 configs) should survive** since all four
> eval on the same stale val today; absolute numbers will shift. Stage 2
> launch is gated on val.jsonl regeneration + the n=300 re-eval pass —
> running Stage 2 on a known-stale val would compound the problem.

## Why this shape

1. **Single-knob ablations first.** Every Stage 1 config differs from
   baseline-1 by exactly one knob. This guarantees attribution: if
   `baseline7` ((+ vision_last_2)) wins by 3 pp, we know it's because
   of vision-layer tuning, not a side-effect of also bumping epochs.
2. **Code freeze during Stage 1.** Stage 1 uses ONLY the YAML knobs
   the existing `LoraConfig` dataclass exposes. No PR needed. Stage 2
   pulls in DoRA / PiSSA / class-balanced sampler — those need a
   ~20-30 LOC enabler PR, which lands in parallel with Stage 1
   training so it's ready when Stage 2 launches.
3. **Adaptive decision tree, not grid search.** Stage 2 picks combos
   based on Stage 1 results. See the decision tree under §"Decision
   framework" — and note in particular that baseline7 (vision tuning)
   is treated as a **default-negative** probe per the prior insight
   in the next section, not as a likely winner.
4. **Report angle**: hypothesis-driven sweep > grid search. The
   write-up tells the story "we identified 4 candidate axes from
   the baseline post-mortem and ran a principled 3-stage adaptive
   sweep" not "we tried 12 things".

## Key prior insight — vision side is the most fragile stack

Two facts that override the naive "vision is the bottleneck for
fine-grained species ID, so push capacity there" hypothesis:

1. **Gemma 4 ships with an undocumented vision decoder.** Only the last
   two transformer blocks of `vision_tower.encoder.layers` are
   safe to touch under any production training setup. Earlier layers
   feed into pretrained behavior the public Gemma 4 release does not
   reveal enough about to reason over; modifying them risks silent
   misalignment with whatever the decoder expects. This is why our
   existing `tune_last_n_vision_layers` mode (`03-vision-mode.md`) caps
   at `N=2` as a design constraint, not just as a default value.
2. **Even at N=2, the early test pointed *down*, not up.** The
   `02-projector-mode.md` overfit100 baseline at 100 % species_match
   degraded to 96 % when last-2 vision encoder layers were also
   unfrozen as full params (`03-vision-mode.md`, "Empirical results").
   `03-vision-mode.md` framed this as "n=100 is too small to
   demonstrate the benefit". The stronger reading — confirmed by
   subsequent investigation into the Gemma 4 vision-decoder gap — is
   that the −4 pp **is the signal**: any visual feature drift past
   the pretrained operating point hurts the cross-modal mapping the
   language LoRA built on top.

Consequences for this sweep:

- baseline7 is kept in Stage 1 to **settle the production-scale
  question** (50k samples might still flip the overfit100 reading per
  `03-vision-mode.md`'s "When vision-tower tuning should become
  valuable" prediction), but the **prior expectation is regression**,
  not a win. If it lands within ±0.5 pp of baseline-1 that's the most
  optimistic realistic outcome.
- Vision-side capacity expansion ideas in the C1-C20 backlog
  (especially C9/C16 LoRA-on-SigLIP and any "unfreeze more vision
  layers" follow-up) drop to P2/P3 in the appendix ranking. Stage 2
  does **not** double-down on vision tuning even if baseline7 narrowly
  beats baseline-1 (which would more likely indicate noise than a real
  vision-capacity win).
- Stage 2 enabler-PR priority goes to **language-side** mechanisms
  (DoRA, PiSSA, rsLoRA, class-balanced sampler, last-N LM-layer
  full-param) — the safe direction for capacity expansion.
- Quantization-side companion observation: bnb NF4 of the vision tower
  collapses PlantNet match to 0.1 % (`07-quantization/docs/B1-bnb-nf4-vision-collapse.md`).
  The vision tower is fragile in both directions — at training and at
  PTQ. Treating it as a fixed sensor that the language side has to
  adapt to is the consistent throughline.

## Resource & budget

| Resource | Quantity | Cost assumption |
|---|---|---|
| remote 5090 GPU 32G | 2 × independent hosts | Don't worry about $ |
| Stage 1 wall | ~16-18 h | Both machines parallel → 1 calendar day |
| Stage 2 wall | ~16-20 h | Both machines parallel → 1 calendar day |
| Stage 3 wall (eval + quant) | ~6-8 h | Local 4090 + Mac |

remote 5090 GPU vs local 4090 throughput: bf16 LoRA SFT is ~1.5-2× faster
on 5090 (32 GB vs 24 GB VRAM lifts batch ceiling; sm_120 + faster mem
bus). Baseline-1's 10 h on 4090 → ~6-7 h on 5090. baseline7
(`vision_last_2` + projector) adds ~2 h for the extra trainable params.

## Decision framework

After every stage, score each config on the **PlantNet val quick eval
(n=300, seed=0)** running locally on the 4090 desktop via the
`mlx-cuda` build (see
[`../general/11-cuda-vs-mlx-eval-parity.md`](../general/11-cuda-vs-mlx-eval-parity.md)
— aggregate noise floor ±0.33 pp on Linux at n=300; signal Δ ≥ 1.0 pp
is treated as real). Quick eval is enough to rank; full n=2,870 only
runs on Stage 3 winners.

### Signal thresholds

| Δ vs baseline-1 (n=300) | Interpretation |
|---|---|
| Δ ≥ +1.5 pp | **Signal** — promote to next stage |
| +0.5 pp ≤ Δ < +1.5 pp | **Weak signal** — promote only if the hypothesis is strong (e.g. vision LoRA even at +0.7 pp is a signal because the hypothesis is "vision features are the bottleneck") |
| -0.5 ≤ Δ < +0.5 pp | **No signal** — discard |
| Δ < -0.5 pp | **Negative signal** — kill, don't carry forward |

### Stage 2 decision tree

Based on Stage 1 quick-eval rankings. **Vision-side wins are not
expanded into a vision-tuning branch** — per §"Key prior insight",
any baseline7 win is more likely n=300 noise than a real vision
capacity gain, and even a real win caps at the last-2-layer ceiling.

```
                    Stage 1 results
                          │
        ┌─────────────────┼─────────────────┬──────────────────┐
        │                 │                 │                  │
   dropout (b4)      alpha=2r (b5)    projector_lr (b6)   vision_last_2 (b7)
   wins (≥+1.5)      wins (≥+1.5)    wins (≥+1.5)        wins (any margin)
        │                 │                 │                  │
        ▼                 ▼                 ▼                  ▼
   Stage 2:          Stage 2:         Stage 2:           Stage 2:
   regularization    LoRA-scale       projector-axis     CONSOLIDATE on
   axis combos      axis combos       combos              language-side, do
   - b4 + DoRA      - b5 + DoRA       - proj_lr ×{1,2,4} not double-down on
   - b4 + PiSSA     - b5 + PiSSA        sweep            vision (per
   - b4 + 7 epoch   - rsLoRA r=512    - + DoRA on top    "Key prior insight").
     + warmup 0.05    α≈√r scaled                        - b7 + DoRA only
   - b4 + rsLoRA r512                                    - b7 + PiSSA only
                                                          (no vision_last_4)
```

Any "vision_last_4" / "LoRA on SigLIP" follow-ups stay out of Stage 2
regardless of baseline7's Stage 1 number. The argument is asymmetric:
the upside (vision capacity) is bounded by the undocumented-decoder
constraint, the downside (cross-modal misalignment) is unbounded.

If NO Stage 1 config produces ≥ +1.5 pp, fall back to:

| Fallback | What it tests |
|---|---|
| `baseline-1 + DoRA` (enabler PR needed) | Pure DoRA vs LoRA effect |
| `baseline-1 + PiSSA init` (enabler PR needed) | Pure PiSSA effect |
| `baseline-1 + class-balanced sampler` (enabler PR needed) | Long-tail fix |
| Data ablation: `english, no wiki` (regen jsonl) | Per `01-pipeline.md` 90 % vs 93 % overfit100 result — production-scale data shape question |

### Stage 3 gate

Top 1-2 configs by Stage 2 quick-eval. Each gets:

1. **Full n=2,870 eval** on local 4090 (~1.5 h per variant).
2. **bf16 → MLX 4-bit affine quantization** via `mlx_vlm.convert`
   (~2.5 min) + full eval on Mac (~25 min).
3. **bf16 → GPTQModel w4g128 desc_act=True** on local 4090 (~18 min
   quant + ~3-5 h eval via Marlin/exllama backend).
4. Generate report-grade rows.

## Stage 1 — kickoff configs

All four configs are zero-code (yaml only). They live under
`src/finetune/configs/`. Naming continues the established
`baselineN-` series.

| # | Config | Filename suffix | Knob (vs baseline-1) | Hypothesis | Wall (5090 32G) | Status (2026-05-16, post-correction) |
|---|---|---|---|---|---|---|
| baseline4 | `lora_dropout=0.05` | `+drop005` | LoRA over-fits the small per-species sample count (~50 images/species after filter); a dropout of 0.05 should regularize without killing capacity | ~7-8 h | ✅ done — **strong signal** (local 70.67% sp, **+12.67 pp** vs n=300 baseline-1) |
| baseline5 | `lora_alpha=512` (r=256, scale 1.0 → 2.0) | `-a512` | The α=r=256 baseline gives scale 1.0; some literature recommends α=2r for stronger LoRA influence. If LoRA is undertrained at α/r=1, this lifts. | ~7-8 h | ✅ done — **signal** (local 62.67% sp, **+4.67 pp**); weakest of the three but no crash |
| baseline6 | `projector_learning_rate=1e-4` (was 5e-5) | `-projlr1e4` | The 1.18M-param projector is the bottleneck between frozen SigLIP and the LM; doubling its LR may unlock more cross-modal capacity. The validator allows it; the auto-default is LR/10 = 2e-5, baseline-1 pinned 5e-5 = LR/4. | ~7-8 h | ✅ done — **strong signal** (local 71.67% sp, **+13.67 pp**) |
| baseline7 | `tune_last_n_vision_layers=2` | `+vision2` | **Default-negative probe.** Settles whether the overfit100 −4 pp (96 % vs 100 % projector-only, `03-vision-mode.md`) was n=100 noise or a real signal that touching SigLIP late layers hurts cross-modal alignment. See §"Key prior insight" — Gemma 4's undocumented vision decoder caps safe blast radius at N=2 anyway. Prior expectation: regression, not a win. | ~9-10 h | ⏳ queued |

### Machine assignment

Goal: balance wall time across the two machines, mix one
"cheaper" + one "heavier" per machine so a single config OOM doesn't
strand a machine.

| Machine | Queue (sequential) | Total wall |
|---|---|---|
| **M1** | baseline4 → baseline7 | ~17-18 h |
| **M2** | baseline5 → baseline6 | ~14-16 h |

### Stage 1 results (2026-05-16 update)

Three of the four Stage 1 configs (baseline4, baseline5, baseline6)
have finished training on the rented 5090 hosts and have been
re-evaluated locally on the 4090 desktop at n=300. baseline7 is still
queued (not yet kicked off on M1).

All numbers below are PlantNet quick-eval (n=300, **val split**,
`eval.max_new_tokens=256`, bf16, `use_unsloth=true`). The "VM" column
is the in-VM 5090 eval written to `outputs/<run>/eval.log` by
`train.sh`; "Local" is the rerun on the 4090 desktop after rsync,
written to `outputs/<run>/eval_local_4090.log`. Baseline-1's local
number is taken from
`results/plantnet-50k-baseline-lora-r256+fullproj-lr5e5-data-aug-enwiki_20260513_031214_eval.json`.

> **Do not compare these numbers to B1 / B2 quantization-doc numbers.**
> Those use `test.jsonl` (paper-grade, head-of-tail-biased, n=300 draws
> are ~66 % from top-50 species); these use `val.jsonl` (in-distribution
> 10 % holdout, n=300 head-of-file is uniform across 605 / 782 species).
> The SAME baseline-1 checkpoint scores **58.0 % on the current
> canonical val.jsonl** (n=300, head-slice — see "baseline-1 reference
> re-aligned" below) and **86.7 % on test.jsonl** — that ~29 pp gap is
> purely the eval-set distribution shift, NOT a model property.

> **baseline-1 reference re-aligned (2026-05-16).** The archived
> baseline-1 result JSON
> (`results/plantnet-50k-baseline-lora-r256+fullproj-lr5e5-data-aug-enwiki_20260513_031214_eval.json`)
> contained **200 samples**, not 300 — the 2026-05-13 eval was kicked
> off with `--max_eval_samples 200` on the CLI before the project
> standardized on n=300 for Stage-1 quick-eval. **`evaluate.py:874`
> selects via `test_data[: args.max_eval_samples]` (head-of-file slice),
> not `random.Random(0).sample(records, N)`**, so n=200 is a strict
> prefix of n=300 — but the file itself has changed.
>
> The original 72.0 % was measured against the **stale 4090 val.jsonl
> from before the canonical `prepare_plantnet_50k.sh` flow landed**.
> That stale val.jsonl pointed at raw `PlantNet-300K-data-v2/test/`
> images and was head-of-tail-biased like `test.jsonl` is today. The
> current canonical val.jsonl is in-distribution-from-`train/` and
> covers 605/782 species uniformly; it is structurally a harder eval.
>
> Re-eval'ing the SAME baseline-1 checkpoint at n=300 on the **current
> canonical val.jsonl** gives **58.00 %** (174/300, ROUGE-L 0.6435,
> see `outputs/.../eval_local_4090_n300.log`). This — not the archived
> 72.0 % — is the apples-to-apples reference for baseline4/5/6.
>
> The archived 200-sample JSON has been overwritten by this re-eval
> (`results/` is gitignored, no recovery path). That number was never
> a defensible Stage-1 reference anyway because the eval split it ran
> on no longer exists on disk.

| Config | VM ROUGE-L / species | Local ROUGE-L / species (n=300, canonical val.jsonl) | Δ species vs baseline-1 (n=300) | Verdict |
|---|---|---|---|---|
| baseline-1 (re-eval) | n/a | 0.6435 / **58.00%** | — | reference |
| baseline4 (`+drop005`) | 0.7078 / 68.33% | 0.7257 / **70.67%** | **+12.67 pp** | **Strong signal** ✅ |
| baseline5 (`-a512`)    | 0.6748 / 63.33% | 0.6717 / **62.67%** | **+4.67 pp** | Signal ✅ (weakest) |
| baseline6 (`-projlr1e4`) | _truncated (eval log only [10..100/300] before remote-gpu box went down)_ | 0.7315 / **71.67%** | **+13.67 pp** | **Strong signal** ✅ |
| baseline7 (`+vision2`)   | _not started_ | _not started_ | — | — |

VM↔local agreement for the finished runs: baseline4 VM 68.33 % vs
local 70.67 % (+2.34 pp), baseline5 VM 63.33 % vs local 62.67 %
(−0.66 pp), baseline6 _VM number not available — the remote-gpu eval was
interrupted at sample [100/300] (`outputs/.../eval.log` only contains
the first 10 progress prints), and by the time we tried to re-fetch
the box the SSH endpoint was already refused. baseline6's local
number is the only reading on this run._

baseline4's gap nudges past the §"Per-config eval" "> 2 pp ⇒ suspect
adapter-load bug" threshold. Why this is **not** a serialization
regression:

- baseline5 (which exercises the same `save_pretrained` /
  `from_pretrained` code path) lands inside ±1 pp VM↔local, so any
  fleet-wide adapter-load bug would show up there too.
- A `save_pretrained` orphan-tensor regression produces an
  essentially-base-model adapter; that would drop baseline4 toward
  the un-SFT'd ~0 % floor, not lift it +2.34 pp.
- The gap is per-run generation-side noise from the eval not pinning
  `temperature=0` / `do_sample=False`. The eval driver currently
  samples; two independent generation passes on the same checkpoint
  can drift several pp on n=300, especially on rare-class samples
  whose conditional distributions are flat.

The "rank, not absolute number" rule still applies: all three Stage 1
configs sit comfortably above baseline-1 on local readings, so the
verdicts in the table above are robust to the ±2-3 pp generation-side
noise band.

**baseline6 ran from `checkpoint-14065/`, not `final-adapter/`.** When
the remote-gpu box went down before the post-train rsync finished, the
local mirror had `checkpoint-14065/` (the last-step checkpoint with
optimizer state) but no `final-adapter/` directory. Verification that
this is equivalent: `train.log` confirms the `final-adapter` save +
tripwire passed on the VM
("`Save tripwire passed: 411 tensors written to disk match in-memory
PEFT state byte-for-byte`" + "`Projector save tripwire passed`"); the
`checkpoint-14065/adapter_model.safetensors` byte size (1,548,546,864)
matches baseline4's `final-adapter/adapter_model.safetensors` exactly
(same `r=256` + projector module count → same on-disk footprint);
`adapter_config.json` differs only in `lora_dropout: 0.0` (baseline6
design) vs `0.05` (baseline4 design), as expected. With `save_steps=1000`
+ `num_train_epochs=5` + `save_total_limit=2`, the last checkpoint at
step 14065 IS the same parameter state that `final-adapter/` was
copied from. Evaluation used `--adapter_path
outputs/<run>/checkpoint-14065`, overriding the config's default
`final-adapter` lookup.

**Reading vs the original hypotheses (post-reference-correction):**

- baseline4 hypothesis was "+0.5-1.5 pp species via dropout
  regularization on the small per-species sample count". The
  measurement is **+12.67 pp**, ~10× the upper-end prediction. The
  hypothesis direction was correct (regularization helps); the
  magnitude is much larger than expected, which suggests baseline-1
  with `lora_dropout=0.0` was meaningfully over-fitting the
  ~57.5-image/species training distribution. Stage 2 candidate
  promotion: ✅ baseline4 + DoRA, ✅ baseline4 + rsLoRA r=512,
  ✅ baseline4 + 7 epoch with longer warmup.
- baseline5 hypothesis was "α=2r unlocks under-trained LoRA capacity".
  Original verdict (against the stale 72 % reference) was "−9 pp
  crash, dead config"; **withdrawn**. The corrected measurement is
  **+4.67 pp** — a real but small signal. α=2r doubles the effective
  LoRA learning rate (PEFT's α/r scale) and most of the +4.67 pp is
  probably the LR bump, not the scaling per se. baseline5 is the
  weakest of the three but not negative; it is a Stage 2 candidate only
  if the decision tree wants an α / rsLoRA-axis branch (rsLoRA r=512
  is the more principled high-rank LoRA scaling).
- baseline6 hypothesis was "doubling `projector_learning_rate`
  (5e-5 → 1e-4) unlocks more cross-modal capacity in the 1.18M-param
  projector". The measurement is **+13.67 pp**, the strongest of the
  three. This contradicts the §"Appendix" pre-stage rank-10/P2 framing
  ("marginal-impact-ceiling"): when the cross-modal binding between
  frozen SigLIP and the LM is the bottleneck — which on the current
  in-distribution-but-broad-coverage val.jsonl it appears to be — even
  a 2× projector LR pays meaningfully. Stage 2 candidate: ✅
  projector_lr sweep at {2e-5, 5e-5, 1e-4, 2e-4} to find the saturation
  point; ✅ baseline6 + DoRA on the language LoRA.
- Two of three Stage 1 configs cleared **+10 pp** (baseline4 +12.67,
  baseline6 +13.67) over the canonical-val.jsonl baseline-1. The
  decision tree's "≥+1.5 pp ⇒ promote" gate is comfortably exceeded;
  the original "NO signal fallback" framing is no longer the operative
  branch. Stage 2 enabler PR (DoRA / PiSSA / rsLoRA) is still desirable
  for the next-level Stage 2 candidates, but it's no longer a hard
  prerequisite — there are clear single-knob winners to combine.

**Open follow-up flagged by these results:**

- The +10-pp jumps on baseline4 and baseline6 suggest baseline-1 was
  meaningfully under-regularized AND under-using its projector capacity.
  Re-running baseline-1 itself to confirm reproducibility (different
  seed, otherwise identical config) would harden the reference: the
  58 % could in principle be a single-seed unlucky draw. Cheap (~25 min
  on 4090).
- Failed-sample alignment is **no longer a useful tripwire** for these
  three runs because the eval driver uses
  `test_data[: args.max_eval_samples]` (deterministic head-of-file
  slice), not `random.Random(0).sample`, so all four runs (b1 re-eval,
  b4, b5, b6) score on the **same 300 records in the same order**. The
  apples-to-apples comparison is purely on `species_match_rate`, not on
  per-sample failure-set IoU.
- Augmentation sanity check still worth doing: confirm
  `data.augmentation: true` is actually applied at collation time in
  baseline-1's training (`data.py`). If augmentation was a silent
  no-op, baseline-1 was even more over-fit than the +12.67 pp gap
  already implies. Add a unit test that the augmented-vs-clean image
  bytes differ at collation.

### Per-config eval

Each config's `eval.enabled: true` triggers an in-VM PlantNet quick
eval (n=300) immediately after training. That number is the **Stage 1
signal** for that config.

For Stage 2 promotion, we also rsync the `final-adapter/` back to the
local 4090 desktop and re-run the quick eval there. The two readings
let us catch noise:

- VM eval = "is the training-side accuracy reasonable?"
- Local eval = "does the adapter actually transfer?"

Both should be within ±1 pp of each other. Gap > 2 pp implies adapter
serialization/load issue (e.g. a regression of the PEFT orphan-tensor
bug from `01-pipeline.md` § Package version fix).

### Launcher

`src/finetune/scripts/run/stage1_sweep.sh` runs each
machine's queue. Usage on the rented box:

```bash
# Pre-flight check (config + data + env + dry-run for each in queue):
bash scripts/run/stage1_sweep.sh M1 --preflight-only

# Real run:
bash scripts/run/stage1_sweep.sh M1
```

The script:

1. Reads the machine identifier (`M1` or `M2`).
2. Walks the hardcoded queue for that machine.
3. For each config, runs `bash scripts/run/train.sh <config>` (which
   does training → auto-eval → log tee per `01-pipeline.md`).
4. On failure of a single config, logs the failure but **continues** to
   the next config (so a single OOM doesn't waste the whole queue).
5. At the end, prints a one-line summary per config with its
   PlantNet n=300 result for quick human inspection.

## Stage 2 — contingent combos (PR + 3-4 configs)

Launches after Stage 1 quick-eval results come in (1 calendar day
after Stage 1 launch). The exact configs depend on Stage 1 ranking —
see decision tree above.

### Enabler PR (in parallel with Stage 1 training)

Add to `LoraConfig`:

```python
use_dora: bool = False
init_lora_weights: Optional[str] = None  # e.g. "pissa"
```

Pass them through `FastModel.get_peft_model` in `finetune.py`. Add to
validator. ~20-30 LOC + unit test. Lands in
`@feature/quantization` before Stage 2 launches.

Class-balanced sampler is more invasive (~50-80 LOC, needs a custom
`Sampler` for the HF Trainer); deferred unless Stage 1 decision tree
picks it.

### Likely Stage 2 candidates

Suffix the Stage 2 winner as `baseline8`, `baseline9`, …

Examples (the exact 3-4 depend on Stage 1 ranking):

- `baseline8 = baseline7 + DoRA on language LoRA`
  (vision tuning + DoRA combo)
- `baseline9 = baseline4 + DoRA`
  (dropout + DoRA combo)
- `baseline10 = baseline7 + tune_last_n_vision_layers=4`
  (more vision layers tuned)
- `baseline11 = baseline4 + 7 epoch + warmup_ratio=0.05`
  (longer training with regularization)
- `baseline12 = baseline-1 + PiSSA init only` (PiSSA fallback if
  Stage 1 is flat)

### Stage 2 machine assignment

Same shape as Stage 1: 2 configs per machine, balanced.
Concrete queues to be filled in once Stage 1 results land.

## Stage 3 — full eval + quantization for top 1-2

Per the Stage 3 gate above. The top 1-2 winners (by Stage 2
quick-eval) go through:

1. Full PlantNet n=2,870 eval (bf16). Reported in
   `B1-sft-results.md` R-rows extension.
2. Quantization via the existing Route B.1 (HF GPTQModel w4g128
   desc_act=True) + Route B.2 (mlx_vlm.convert -q affine g64 / g128).
3. Full eval of each quantized variant.
4. Report-grade comparison table.

This is the SFT-side input to the deliverable. The quantization-side
plan is in `docs/quantization/00-quantization-roadmap.md`.

## Risks & failure modes

| Risk | Mitigation |
|---|---|
| remote 5090 GPU → mlx-cuda eval needs from-source mlx build (~1 h) | Don't eval on 5090 — push adapter, eval locally on 4090 |
| Single config crashes (OOM, NaN, mode collapse) | stage1_sweep.sh continues to next config; failed config logged |
| Stage 1 produces no signal (all Δ < +0.5 pp) | Decision tree's "no signal" fallback path — enabler-PR-dependent configs become Stage 2 |
| Enabler PR slips, blocks Stage 2 | Stage 2 falls back to multi-knob combos within the existing config schema (e.g. baseline4 × baseline7 multi-knob) |
| 5090 wall time is much slower than estimate (e.g. PCIe bottleneck on remote GPU) | Drop epoch count from 5 → 4 for any config that's > 9 h projected; document the deviation |
| Adapter-serialization regression (PEFT orphan tensors) | Per `01-pipeline.md` § Package version fix, `scripts/inspect/save_reload.py` runs as preflight on each rented box |
| 2 machines disagree on similar configs (cross-machine numerical drift) | Each machine runs DIFFERENT configs — disagree is impossible by construction. Cross-config rankings are what we compare, not absolute numbers |

## Open questions

1. **Hiking-QA mix.** Current data-aug-enwiki training feeds plant
   conversations + WikiText anchor. Should Stage 2 try a higher
   plant:wiki ratio (90:10 vs current ~80:20)? Need to read the data
   prep script first to confirm the actual ratio.
2. **Eval template stability.** baseline-1's response style is "This
   appears to be {common}. {latin} is a species of …". The
   `extract_species` regex handles both Title Case English and Latin
   binomials, but if Stage 1 alters response length significantly
   (e.g. baseline5 longer responses due to higher α), some hits may
   be lost to the 256-token eval cap. Verify on first Stage 1 eval.
3. **Multi-seed verification.** None of Stage 1/2 is multi-seed.
   If two configs tie within ±0.5 pp at quick-eval, do we re-run one
   with seed=1 to break the tie, or just full-eval both? Decision:
   full-eval both at n=2,870 — the n increases the signal more than a
   seed switch would.

## Appendix — full candidate ranking (C1-C20)

Before Stage 1 was defined we brainstormed ~20 candidate knobs / recipes.
This appendix records all of them, ranked by expected
PlantNet-match-impact ÷ engineering-cost, with a tag for each saying
whether it's already in a stage and how much codebase change it would
need.

### Priority ranking

Re-ranked 2026-05-15 after incorporating the "vision side is the most
fragile stack" insight (see §"Key prior insight"). All vision-side
expansion ideas demoted; all language-side capacity / data ideas
promoted.

| Rank | ID | Recipe | Priority | Rationale | In sweep? |
|---|---|---|---|---|---|
| 1 | C5 | DoRA on language LoRA + dropout | P0 | Drop-in PEFT flag; consistently +1-2 pp over LoRA in literature; stays on the safe language-side capacity axis | Stage 2 candidate (small enabler PR) |
| 2 | C7 | PiSSA init on language LoRA + dropout | P0 | Drop-in PEFT flag; non-zero init gives +0.5-1.5 pp; stacks with DoRA cleanly | Stage 2 candidate (small enabler PR) |
| 3 | C1 | LoRA dropout 0.05 (regularization only) | P0 | Cleanest single-knob baseline improvement; common ancestor for most Stage 2 combos | **Stage 1 — baseline4** ✅ |
| 4 | C13 | Class-balanced sampling | P0 | PlantNet has heavy long-tail; metric is exact species match → rare classes get little gradient signal under uniform sampling. **Data-side fix, not vision-side**, so safe direction | Stage 2 candidate (medium PR — conflicts with `group_by_length`) |
| 5 | C2′ | rsLoRA r=512 with α scaled by √r | P0 | The "correct" high-rank LoRA scaling; canonical fix for the "high r isn't helping" failure mode. Pure language-side capacity bump | Stage 2 candidate (small enabler PR) |
| 6 | C12 | Partial unfreeze last 2-4 LM layers (full-param) | P1 | Mirror of `tune_last_n_vision_layers` but on the SAFE side of the model. 5090 32G makes it tractable. ~80 LOC new mode | Backlog (medium PR) |
| 7 | C14 | PlantNet-only / reduced enwiki mix | P1 | Final metric is plant exact-match; less language anchor may help. Cheap to test once data builder gets a `--wiki_ratio` flag | Stage 2 fallback (data prep change) |
| 8 | C15 | Canonical answer format ("This appears to be {common}. ...") | P1 | Reduces extractor false-negatives without changing model capacity at all. Purely data-side metric hardening | Stage 2 fallback (data prep change) |
| 9 | — | `tune_last_n_vision_layers=2` (full-param last 2 SigLIP) | P1 | Settles whether overfit100's −4 pp generalizes to 50k. **Default-negative probe**, not a likely winner. Kept in Stage 1 only because (a) we want the data point and (b) M1's wall-time budget would otherwise sit idle | **Stage 1 — baseline7** ✅ |
| 10 | C19 | Projector LR sweep (1e-4 vs 5e-5) | P2 | Cheap, but projector is 1.18 M params — marginal impact ceiling | **Stage 1 — baseline6** ✅ |
| 11 | C8 | α=2r (vs current α=r) | P2 | Simple but no strong theory advantage over DoRA / PiSSA / rsLoRA | **Stage 1 — baseline5** ✅ |
| 12 | C18 | Conservative image augmentation | P2 | Existing pipeline has reasonable defaults; risk of color-clue damage on species ID is real | Backlog (small refactor) |
| 13 | C4 | Base-model attention/hidden dropout | P2 | Helps generalization but may interfere with frozen pretrained behavior; untested unsloth path | Backlog |
| 14 | C9 / C16 | **LoRA on vision tower (low LR)** | **P3** ⚠ | Was previously ranked P0 on the "vision is the bottleneck" hypothesis. **Demoted to P3** after the §"Key prior insight" review: undocumented Gemma 4 vision decoder caps safe touch to the last 2 layers, and the existing full-param last-2 test already pointed down (overfit100 96 % vs 100 %). LoRA on the same 2 layers is unlikely to beat the full-param test, which already lost. Only revisit if a published follow-up shows the undocumented-decoder constraint can be reasoned over | Out of scope for this deadline |
| 15 | C17 / C10 | High-res training (1280×896 or native) | P3 | Plant features live in fine detail, but this is a 3-file coordinated change (`prepare_plantnet.py`, `export_mlx.py`, `fetch-gemma.sh` all hold `TRAINED_SIZE` constants) AND lands on the fragile vision side. Risk/reward unfavorable pre-deadline | Out of scope (high PR) |
| 16 | C11 | Hard-negative curriculum / 2-stage training | P3 | Requires a baseline-1-inference-over-trainset pass + a second training stage; ~doubles experiment time | Out of scope (high PR) |
| 17 | C20 | Short-answer auxiliary mix | P3 | Promising but needs data-builder rewrite + multi-target loss handling | Out of scope (high PR) |

Notes:

- **C9/C16 vs baseline7 distinction stands**, even though both got
  demoted. C9/C16 = LoRA adapters on `vision_tower.encoder.layers.*`;
  baseline7 = full-param unfreezing of last 2 of the same layers.
  Different mechanisms; both are vision-side; both now treated as
  default-negative under §"Key prior insight".
- The ranking re-order does NOT change Stage 1 — those configs were
  picked under the zero-code constraint, not under the priority
  ranking. baseline7 is the lowest-prior-expectation config in Stage 1
  but stays because the data point itself is valuable.

### Codebase-change classification

What stands between an idea and a runnable config:

| Tier | What it takes | Members |
|---|---|---|
| 🟢 Zero code (YAML only) | Edit a config file; existing `LoraConfig` dataclass + `finetune.py` flow already handle it | C1 (baseline4), C8 (baseline5), C19 (baseline6), `tune_last_n_vision_layers` (baseline7), longer training, `tune_last_n_vision_layers=4` |
| 🟡 Small enabler PR (~10-30 LOC, PEFT flag pass-through) | Add field to `LoraConfig`, pass through `FastModel.get_peft_model`, update validator | C5 (DoRA), C7 (PiSSA), C2′ (rsLoRA) |
| 🟠 Medium PR (~30-80 LOC, new mode or new data pipeline) | Either new freeze-pass / `modules_to_save` path, or new flag in `prepare_plantnet_enriched.py` + data regen | C9/C16 (LoRA on vision tower — also needs `freeze.py` allowlist update), C13 (class-balanced sampler, conflicts with `group_by_length`), C14 (data prep), C15 (data prep), C12 (last-N LM full-param) |
| 🔴 Large PR (~100+ LOC, multi-file coordination) | Three-file sync (`prepare_plantnet.py`, `export_mlx.py`, `fetch-gemma.sh`), or multi-stage training loop, or multi-target loss | C10/C17 (high-res — three trained-size constants must move together), C11 (curriculum, doubles wall), C20 (short-answer aux loss) |

The Stage 2 enabler PR plan in §"Stage 2 — contingent combos" lands the
🟡 tier in parallel with Stage 1 training. After the §"Key prior
insight" re-ranking, the **single 🟡 enabler PR** carries the three
language-side P0s (C5 DoRA, C7 PiSSA, C2′ rsLoRA) — all three are
PEFT flag pass-throughs sharing the same `LoraConfig` + validator
touch-points. C9/C16 (LoRA on SigLIP) is NOT in this PR because it
dropped to P3.

🟠 ideas:
- **C13 class-balanced sampler** stays high (P0) but lands as its own
  PR — it conflicts with `group_by_length`, needs its own validator
  branch, and requires a custom Sampler injected into the unsloth
  `SFTTrainer` wrapper.
- **C14 / C15** (data-side reformulations) land as `prepare_plantnet_enriched.py`
  flags + data regen; no training-side code touched.
- **C12** (last-N LM-layer full-param) is the safe-side analogue of
  baseline7. Worth implementing if Stage 2 results show language-side
  capacity is the bottleneck — promotion path is "DoRA / PiSSA both
  cap out → escalate to LM-layer-unfreezing".

🔴 ideas stay out of scope for the deadline.
- **C10/C17 (high-res)** explicitly demoted to P3 because (a) it's a
  three-file coordinated change and (b) it lands on the fragile vision
  side. Even though "more pixels" is a different mechanism than
  "more vision capacity", both interventions ultimately rely on the
  vision tower behaving well under distributional shift, which it
  has not in our tests.
- If post-deadline work continues, the best vision-side investment is
  reproducing the public mlx-community 4-bit checkpoint's recipe
  (data-free affine, towers bf16, projector quantized) on our SFT'd
  merge — see `07-quantization/docs/00-quantization-roadmap.md`. That
  doesn't change the vision tower at training time; it studies how
  the language-side gains propagate through fixed-vision quantization.

## File pointers

| Concern | Path |
|---|---|
| Stage 1 configs | `src/finetune/configs/plantnet-50k-baseline{4,5,6,7}-*.yaml` |
| Stage 1 launcher | `src/finetune/scripts/run/stage1_sweep.sh` |
| Train + auto-eval | `src/finetune/scripts/run/train.sh` |
| Save/reload preflight | `src/finetune/scripts/inspect/save_reload.py` |
| Eval-side noise contract (Linux 4090 ±0.33 pp at n=300) | `docs/general/11-cuda-vs-mlx-eval-parity.md` |
| Baseline-1 progress / number | `docs/quantization/B1-sft-results.md` R0 |
| Projector mode mechanism | `docs/finetune/02-projector-mode.md` |
| Vision-mode mechanism | `docs/finetune/03-vision-mode.md` |
| Route-B.1 quant (deploy-time PTQ on bf16) | `docs/quantization/00-quantization-roadmap.md` |
