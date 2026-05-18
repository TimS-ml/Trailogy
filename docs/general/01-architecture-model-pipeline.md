# Model-Side Architecture Overview

## TL;DR

- The model pipeline turns a stock Gemma 4 E2B checkpoint into a supervised-finetuned, quantized artifact for the iOS app bundle.
- Data mixture, fine-tuning, and quantization run as separate workstreams but share one evaluation surface.
- The deploy target is a small on-device model bundle that stays under mobile memory limits while preserving general reasoning and plant-identification quality.
- The iOS app consumes the final artifact from its bundled `Models/Gemma/` directory.

## The Question This Pipeline Answers

Trailogy needs a Gemma 4 model that can run offline on iOS, answer hiking
questions conversationally, and identify plants from photos. A stock compact
model is not enough: it does not know trail-local facts, it does not reliably
identify the plant species we care about, and its bf16 checkpoint is too large
for mobile deployment.

The model-side pipeline therefore keeps three concerns separate:

| Workstream | Problem | Output |
|---|---|---|
| `data_mix/` | Plant-only training makes the model treat every prompt as a plant prompt. | Mixed JSONL corpus with plant, general vision, text chat, refusal, and offline persona records. |
| `finetune/` | Gemma needs domain adaptation without losing general assistant behavior. | bf16 LoRA/adapter checkpoint and merged bf16 model. |
| `quantization/` | The merged bf16 model is too large for iOS. | MLX/VLM-format quantized artifact for the app bundle. |

The iOS app treats the final artifact as a local model under
`HikeCompanion/Resources/Models/Gemma/`.

## Data Flow

```text
PlantNet + general VQA + text chat + refusal + offline QA
        |
        v
data_mix builds a mixed training corpus
        |
        v
finetune trains Gemma 4 in bf16 with LoRA/projector options
        |
        v
export merges adapter into a bf16 multimodal checkpoint
        |
        v
quantization converts to an MLX/VLM iOS-loadable artifact
        |
        v
iOS bundle loads Gemma locally for text and image asks
```

Local trail facts are deliberately not baked into the model. They live in the
retrieval layer so trail packages can change without retraining Gemma. The
model is responsible for conversational synthesis, visual grounding, offline
honesty, and end-of-hike recaps.

## Shared Evaluation Surface

All model workstreams compare against the same broad questions:

- Does PlantNet species recognition improve?
- Does general assistant behavior survive?
- Does the exported artifact load through the same family of model classes as
  the iOS runtime?
- Does quantization preserve enough quality under the mobile size budget?

The exact datasets and sample counts changed during development, so cross-doc
number comparisons require care. See [`10-eval-setup.md`](10-eval-setup.md)
before comparing numbers from different phases.

## Key Design Decisions

| Decision | Why |
|---|---|
| Train in bf16 unless explicitly exploring QLoRA/QAT. | Keeps the SFT result interpretable and avoids hidden 4-bit/8-bit optimizer artifacts. |
| Use the multimodal loader and converter. | Language-only loaders silently drop vision weights. |
| Keep trail-local truth in RAG, not weights. | Trail facts must be updateable and grounded. |
| Use `[camera=on/off]` as the deployment-visible modality gate. | The iOS app can emit this reliably without a topic classifier. |
| Quantize through `mlx_vlm`/MLX-compatible paths. | iOS uses the MLX/VLM Gemma 4 tree; other model trees can produce misleading results. |

## What Shipped Conceptually

The shipped model story is not "we trained a plant classifier." It is:

1. Build a mixed corpus so plant examples are important but not dominant.
2. Fine-tune Gemma 4 for plant-focused vision while preserving general text
   behavior and offline persona responses.
3. Export through a multimodal-safe path with tripwires for dropped tensors.
4. Quantize into the MLX/VLM deployment format and verify PlantNet quality.
5. Let the iOS app combine Gemma with local retrieval, speech, and trail state.

## Main Supporting Docs

- iOS runtime architecture: [`02-architecture-ios-app.md`](02-architecture-ios-app.md)
- Memory constraints: [`03-memory-management.md`](03-memory-management.md)
- RAG runtime: [`05-rag-runtime.md`](05-rag-runtime.md)
- Final SFT recipe: [`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
- Quantization overview: [`../quantization/00-quantization-report-pub.md`](../quantization/00-quantization-report-pub.md)
- Silent failure postmortems: [`15-postmortems.md`](15-postmortems.md)
