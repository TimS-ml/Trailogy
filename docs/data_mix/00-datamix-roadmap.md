# 05c — data_mix roadmap

## TL;DR

- Plant-only fine-tuning made the model answer as if every prompt was a plant task, even for text questions or non-plant images.
- The mixed corpus adds plant images, general vision examples, text-only chat, refusal examples, and offline persona QA so the model keeps broader behavior.
- The final prompt design uses `[camera=on]` and `[camera=off]` to tell the model whether an image is present during both training and app use.
- This document tracks how the mix changed from an early 20K version to the current 50K/100K production setup.

> last edit: 2026-05-16

Strategic overview for the anti-overtraining mixed SFT corpus.
Companion to the code in `src/data_mix/`. **Read this first.**

## What problem we are solving

Pure-PlantNet SFT runs over-fit hard. The r=256 LoRA + projector
adapter gets captured by the PlantNet distribution and the model
answers "plant" to almost any input — including text-only prompts
("what is the capital of France?") and non-plant images. The LoRA
delta is too big and the input distribution is too narrow; the
adapter has memorized "be plant" rather than "be plant when asked
about a plant".

Anti-overtraining is a three-pronged problem and the mix is the data
side of the answer:

| Lever | Where it lives | What it bounds |
|---|---|---|
| Distribution diversity in train data | `src/data_mix/` (this module) | The *coverage* of the LoRA delta — multi-domain training forces the adapter to encode less plant-specific structure. |
| KL + L2 regularizers | `src/finetune/src/regularization.py` | The *magnitude* of the LoRA delta drift (output-distribution KL and weight-space L2 anchor). |
| Camera-state prefix gate | `src/finetune/src/data.py::build_vision_messages` | The *input gate* — every prompt carries `[camera=on]` (image present) or `[camera=off]` (text-only), so the model conditions modality-specific behaviour on the marker rather than guessing from the prompt distribution. |

This doc covers the first lever (the mix). See
[`01-data-prefix.md`](01-data-prefix.md) for the prefix mechanism and
`docs/finetune/07-anti-forgetting-regularization.md`
for the full anti-forgetting design including KL + L2.

## Version evolution

| Version | Date | What changed | Status |
|---|---|---|---|
| v1 (20K) | 2026-05-15 | Initial 4-bucket design: Plant 45 / Cambrian 30 / smoltalk 15 / Negative 10. Cambrian-10M HF streaming blocker discovered. | **Superseded** — see [`A-mix-20k-v1.md`](A-mix-20k-v1.md) |
| v2 (50K, 100K) | 2026-05-16 | Cambrian replaced by LLaVA-mix (`f6d0c1f`); `image=None` text-only allowed (no more dummy image); multi-val output per source (plant / nonplant / negative). Per-species stratified train/val split. | **Production** — see [`B-mix-50k-v2.md`](B-mix-50k-v2.md) |
| v3 (50K, 100K) | 2026-05-16 | Same mix composition as v2; ADDS source-keyed task-tag prefix dispatch (`prompt_prefixes` with `plant`/`cambrian`/`negative` keys → `[task=plantnet]` etc.) + KL/L2 regularizers in the finetune pipeline; ADDS `offline_qa` persona bucket (~42 records, unprefixed). | **Superseded by v4** — pinned at git tag `v3-task-tag-prefix` for evaluating older models |
| v3.1 | 2026-05-16 | `data.default_source` loader hook lets the v3 prefix dispatch fire on legacy single-source JSONLs (e.g. `english-desc/train.jsonl`). | **Superseded by v4** — no longer needed under image-presence dispatch |
| v4 (50K, 100K) | 2026-05-16 | Replaces v3 source-keyed dispatch with image-presence dispatch: `prompt_prefixes` now uses two fixed keys `camera_on`/`camera_off`, dispatched on whether the record carries an image. Removes `data.default_source` (no source-field lookup → no fallback needed). The marker becomes a modality-state flag the on-device app can compute trivially from its existing `.text`/`.vlm` mode branch. | **Production** — `src/finetune/src/data.py` |

The mix file format hasn't changed across v2/v3/v4; the only thing
v3 added on the data side is the `offline_qa` persona bucket
(orchestrator appends it on top of the 45/30/15/10 ratio so the main
bucket math stays interpretable). v4 is a code/config-only change:
the JSONL records on disk are unchanged.

## What ships today

```
src/data_mix/
├── configs/
│   ├── mix-20k.yaml                # v1 historical (Cambrian — blocked, do not run)
│   ├── mix-200.yaml                # v1 smoke
│   ├── mix-200-llava.yaml          # v2 smoke (LLaVA — current canonical smoke)
│   ├── mix-50k.yaml                # v2/v3 PRODUCTION
│   └── mix-100k.yaml               # v2/v3 larger variant
├── src/
│   ├── schema.py                   # record validator (strict role alternation)
│   ├── dummy_image.py              # v1 only — superseded by image=None in v2
│   ├── image_resize.py             # 960×672 stretch (mirrors prepare_plantnet.py)
│   ├── plant_sampler.py            # per-class cap + 3 prompt variants + dual-source
│   ├── negative_builder.py         # fixed refusal template
│   ├── smoltalk_sampler.py         # text-only (image=None in v2; dummy image in v1)
│   ├── cambrian_sampler.py         # v1 only — HF streaming blocker, kept for reference
│   ├── llava_sampler.py            # v2+ — replacement for Cambrian (LLaVA-mix on HF)
│   ├── offline_qa_sampler.py       # v3 — persona corpus, ~42 records appended
│   ├── env_paths.py                # env-var-resolved storage roots
│   └── mix.py                      # orchestrator
├── scripts/
│   └── build_mix.sh                # driver with env preflight + diagnostics
└── tests/
    └── 96 tests, all green         # was 81 before offline_qa landed
```

Code lives on the public repo, branch
`feature/quantization` (co-located with EoRA quantization work for
historical reasons; the two efforts touch disjoint directories so the
co-location has not caused conflicts).

## Composition target (v2/v3)

| Bucket | Share | mix-50k train | mix-50k val | Source |
|---|---:|---:|---:|---|
| Plant | 44 % | 22,000 | 1,100 | PlantNet enriched train JSONL (per-class cap, 3 prompt variants) |
| LLaVA | 30 % | 15,000 | 750 | `liuhaotian/LLaVA-Instruct-150K` (and siblings) — general VQA |
| smoltalk | 15 % | 7,500 | 375 | `HuggingFaceTB/smol-smoltalk` — text-only, `image=None` |
| Negative | 10 % | 5,000 | 250 | non-plant LLaVA images + fixed refusal template |
| **offline_qa** | — | +~38 | +~4 | hand-curated persona corpus (UNPREFIXED, sits outside the ratio) |

Sub-totals: 50,038 train / 2,479 val for mix-50k (plant slightly below
45 % after the post-cap pool fit-down, see commit `875803c`). Same
pattern at mix-100k scale (~100,038 train).

The 100K mix is for runs where compute budget allows; the 50K mix is
the standard baseline. See [`B-mix-50k-v2.md`](B-mix-50k-v2.md) for
per-bucket implementation details.

## Reading order

For a new contributor:

1. This file — `00-datamix-roadmap.md` — what shipped + why.
2. [`01-data-prefix.md`](01-data-prefix.md) — the v4 camera-state
   prefix mechanism. Required reading for anyone touching the iOS
   side (deployed prompts must carry `[camera=on]` / `[camera=off]`
   based on whether an image is captured).
3. [`02-bucket-design.md`](02-bucket-design.md) — how each source
   bucket is built; record schema; the smoltalk text-only trick.
4. [`03-orchestrator-and-build.md`](03-orchestrator-and-build.md) —
   `build_mix.sh` + `mix.py` + env vars + idempotence + how to run.
5. [`B-mix-50k-v2.md`](B-mix-50k-v2.md) — the current production
   config in detail.
6. [`A-mix-20k-v1.md`](A-mix-20k-v1.md) — historical, only if you
   need context on why we switched from Cambrian to LLaVA.

For machine-specific run instructions (the dev box), see
[`RUN.md`](RUN.md).

## Related notes

| Location | Purpose |
|---|---|
| `src/data_mix/` | Code: builders + orchestrator + configs |
| `src/finetune/src/data.py` | Where the mix JSONL lands — `load_vision_dataset` + `build_vision_messages` + the prefix dispatch |
| `src/finetune/src/regularization.py` | KL + L2 implementation |
| `04-sft/docs/07-anti-forgetting-regularization.md` | Full design doc for the anti-forgetting stack (KL + L2 + camera-state prefix gate, with rationale) |
| `04-sft/docs/01-pipeline.md` | Where the bf16 SFT'd merged model comes from |
| `DEV_TIMELINE.md` (repo root) | Deadline + priorities |

## Out of scope for v3 (intentionally)

- No PlantNet sub-bucketing (strict / trait / coarse / uncertain).
- No dynamic `{brief_caption}` substitution in negative refusals.
- No two-stage curriculum — single SFT pass, single train.jsonl.
- No source-aware KL weighting (parked for v4; see
  `07-anti-forgetting-regularization.md` §"Open follow-ups").
- No mid-training capability probes beyond the multi-eval-dataset
  loss tracking the trainer already emits.
- No multi-language coverage beyond what each source provides natively.
