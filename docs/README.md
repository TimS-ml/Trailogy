# Docs

## TL;DR

- These docs explain how Trailogy's on-device Gemma 4 model was trained,
  evaluated, quantized, and shipped into the iOS app.
- `data_mix/`, `finetune/`, and `quantization/` mirror the three model
  workstreams: dataset construction, SFT experiments, and deploy-time
  compression.
- `general/` holds cross-cutting notes: architecture, iOS runtime,
  memory management, eval caveats, package bugs, postmortems, and final
  results.
- Start with the reading order below if you want the shortest path to
  "what shipped and why".

Engineering notes and experiment reports. The per-module docs mirror
the code under `src/`; `general/` carries cross-cutting docs
(architecture, dev timelines, eval setup, known bugs, postmortems).

```
docs/
├── general/          # architecture, timelines, eval, postmortems (cross-cutting)
├── data_mix/         # data prep, mix recipes, prompt-prefix gating
├── finetune/         # SFT pipeline, projector/vision modes, final recipe
└── quantization/     # post-training quantization (GPTQ, MLX, EoRA)
```

These docs trade some polish for completeness. They describe the
actual experiments that shipped, including failed paths and the
reasoning behind each design decision.

## First-Pass Reading Order

Read these if you have limited context and want the fastest route to the
technical story:

| Order | Read | Why it matters |
|---:|---|---|
| 1 | [`general/02-architecture-ios-app.md`](general/02-architecture-ios-app.md) | Explains the offline product, runtime stack, and why memory discipline matters. |
| 2 | [`general/01-architecture-model-pipeline.md`](general/01-architecture-model-pipeline.md) | Shows how data, SFT, quantization, and iOS deployment connect. |
| 3 | [`finetune/03-anti-forgetting-and-final-recipe.md`](finetune/03-anti-forgetting-and-final-recipe.md) | Explains the catastrophic-forgetting problem and the recipe that kept Gemma usable. |
| 4 | [`data_mix/B-mix-50k-v2.md`](data_mix/B-mix-50k-v2.md) | Explains the mixed corpus that prevented the model from becoming only a plant classifier. |
| 5 | [`quantization/00-quantization-report-pub.md`](quantization/00-quantization-report-pub.md) | Explains how the 9.5 GB bf16 model became an iOS-loadable artifact. |
| 6 | [`general/15-postmortems.md`](general/15-postmortems.md) | Summarizes the silent failures that could have invalidated the results. |

## What To Skip On First Review

The following files are useful for reproduction or debugging, but they are
not required to understand the project narrative:

- Timeline logs: `general/08-*`, `general/09-*`.
- Full sweep matrices: `finetune/05-*`, `quantization/B1-*`, `quantization/B2-*`.
- Tooling investigations and superseded designs: `finetune/04-*`, `data_mix/A-*`, `data_mix/C-*`, `quantization/00-quantization-roadmap.md`.

## Directory Map

```text
docs/
├── general/          # architecture, runtime, eval caveats, package bugs, postmortems
├── data_mix/         # dataset mix design and build notes
├── finetune/         # SFT pipeline and anti-forgetting recipe
└── quantization/     # deploy-time compression and evaluation
```

## Core Story

Trailogy had one constraint that drove the whole design: it must work on a
trail with no network access. That means local retrieval, local speech, local
vision, and an on-device Gemma 4 model small enough to fit mobile memory.

The project therefore split into four technical problems:

| Problem | Solution | Result |
|---|---|---|
| Local trail facts cannot be memorized reliably by a compact model. | Keep facts in a bundled RAG package; use Gemma for explanation and synthesis. | Trail answers stay grounded and updateable per trail. |
| Plant SFT made the assistant answer unrelated prompts as plant tasks. | Use a mixed corpus plus `[camera=on/off]` modality tags; keep KL/L2 as fallback designs. | Plant ability improved while general benchmarks stayed usable. |
| Model/library version mismatches silently invalidated adapters. | Add save/reload tensor tripwires and retrain under the correct Gemma 4 layout. | Evaluation measures the trained adapter, not a silently degraded reload. |
| The bf16 model was too large for iOS. | Quantize through the MLX/VLM deployment tree and recover quality with EoRA. | Current candidate: 3.6 GB, 88.0% PlantNet quick test, iOS-loadable. |

## Per-Module Entry Points

- [`general/README.md`](general/README.md): architecture, runtime, eval, package bugs, postmortems.
- [`data_mix/README.md`](data_mix/README.md): data-mix design and corpus build.
- [`finetune/README.md`](finetune/README.md): training pipeline and SFT recipes.
- [`quantization/README.md`](quantization/README.md): quantization routes and results.
