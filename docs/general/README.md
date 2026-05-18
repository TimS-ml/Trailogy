# general/ — cross-cutting docs

Architecture, dev timeline, runtime patterns, eval setup, and known-bug
postmortems that don't belong to a single module
(`data_mix/` / `finetune/` / `quantization/`).

These docs support the Trailogy writeup. Each writeup section has a
corresponding link target below.

## Reading order tied to the writeup

| Writeup section | Doc |
|---|---|
| Architecture (offline stack, ~3.2 GB bundle) | [`01-architecture-model-pipeline.md`](01-architecture-model-pipeline.md) + [`02-architecture-ios-app.md`](02-architecture-ios-app.md) |
| Technical Challenge #1 — Silent PEFT loading | [`14-package-versions-and-known-bugs.md`](14-package-versions-and-known-bugs.md) + [`12-mlx-vlm-vs-hf-kv-sharing.md`](12-mlx-vlm-vs-hf-kv-sharing.md) |
| Technical Challenge #2 — On-device quantization | [`13-mlx-vision-input-parity.md`](13-mlx-vision-input-parity.md) + [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md) + the per-method docs under [`../quantization/`](../quantization/) |
| Technical Challenge #3 — Catastrophic forgetting | [`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md) + [`../data_mix/02-bucket-design.md`](../data_mix/02-bucket-design.md) |
| Technical Challenge #4 — Offline honesty | [`../data_mix/01-data-prefix.md`](../data_mix/01-data-prefix.md) (offline_qa bucket) |
| Eval methodology | [`10-eval-setup.md`](10-eval-setup.md) |
| What broke and how we noticed | [`15-postmortems.md`](15-postmortems.md) |

## All docs in this directory

### Architecture

- [`01-architecture-model-pipeline.md`](01-architecture-model-pipeline.md) — model-side pipeline (data mixture, SFT, quantization tracks) and how the deploy artifact lands in the iOS bundle.
- [`02-architecture-ios-app.md`](02-architecture-ios-app.md) — iOS app architecture (views → services → frameworks), Ask pipelines, RAG path.
- [`03-memory-management.md`](03-memory-management.md) — on-device memory discipline, two-phase Kokoro unload, quantization memory math.
- [`04-xcode-build-and-deps.md`](04-xcode-build-and-deps.md) — Xcode build pipeline, model-variant switching, vendored SPM dependencies, Phase-7 build-setting fixes.

### Runtime patterns (iOS)

- [`05-rag-runtime.md`](05-rag-runtime.md) — on-device retrieval path, multi-subject active set, prompt injection.
- [`06-scenephase-metal-background.md`](06-scenephase-metal-background.md) — why Swift can't catch the Metal-backgrounding crash and the two scenePhase gates that prevent it.
- [`07-optimizations-and-future.md`](07-optimizations-and-future.md) — catalog of what shipped vs what's open.

### Development timelines

- [`08-dev-timeline-model.md`](08-dev-timeline-model.md) — SFT + data-mixture + quantization phases.
- [`09-dev-timeline-ios.md`](09-dev-timeline-ios.md) — iOS app phases.

### Eval and cross-platform parity audits

- [`10-eval-setup.md`](10-eval-setup.md) — what the eval driver actually computes, the 300-sample PlantNet slice, **and explicit caveats about benchmark drift across phases**.
- [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md) — Linux mlx-cuda vs Mac mlx-metal numerical contract; why pypi `mlx-cuda-12==0.31.1` produces ~7 % on Gemma 4 INT4 and the from-source fix.
- [`12-mlx-vlm-vs-hf-kv-sharing.md`](12-mlx-vlm-vs-hf-kv-sharing.md) — KV-shared layer parity audit between `transformers ≥ 5.8`, `mlx_vlm`, and `mlx-swift-lm`.
- [`13-mlx-vision-input-parity.md`](13-mlx-vision-input-parity.md) — Gemma 4 train↔deploy image preprocessing parity (the `processor_config.json` 224×224 bug and the `mlx-swift-lm` fixed-stretch fallback).

### Debugging knowledge

- [`14-package-versions-and-known-bugs.md`](14-package-versions-and-known-bugs.md) — tested package versions and the known bugs in the stack (the transformers 5.5→5.8 silent PEFT-loading bug, the peft + gptqmodel dispatcher patch, etc.).
- [`15-postmortems.md`](15-postmortems.md) — dated postmortems rolled up: overfit-memorization → save→reload, mlx quantization, mlx-on-Linux, train/eval data mismatch, eval-pipeline silent failures, HF trainer LR resume.

## Conventions

- **Benchmark drift**: eval surface changed across phases. `n` went 200 → 300 → 2870; subsets and metrics evolved. Numbers in different docs are not always apples-to-apples; cross-phase comparisons are documented explicitly in [`10-eval-setup.md`](10-eval-setup.md).
- **Hardware**: training on RTX 4090 (24 GB) + 4090 laptop (16 GB) + 2× A100 (40 GB) + H100/H200 cloud; MLX work on Apple Silicon (M-series). Specific machine identities are scrubbed; only the general class matters.
