# data_mix — Anti-overtraining SFT corpus for Gemma 4 E2B VLM

## TL;DR

- These docs explain the mixed training corpus used to stop a plant-focused model from answering every prompt as a plant task.
- Start with the camera-state prefix doc, then read the bucket design, build pipeline, and current 50K mix details.
- The corpus combines plant images, general vision examples, text-only chat, refusal examples, and offline persona QA.
- Historical notes explain why the first 20K mix and source-based prompt tags were replaced.
- The data-mix tests use mocked remote data streams, so they can run without network access.

Engineering notes for the mixed-source SFT corpus. Companion to the
code in `src/data_mix/`. The goal: break PlantNet's monopoly on the
LoRA subspace so the fine-tuned Gemma 4 E2B doesn't answer "plant"
to every prompt.

This is the **data side** of the anti-forgetting stack. The finetune-
side companion (KL output-distribution penalty + L2 weight anchor +
camera-state prefix wiring) is in
[`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md).

## Read First

| Read | Why |
|---|---|
| [`B-mix-50k-v2.md`](B-mix-50k-v2.md) | Current production corpus and why it exists. |
| [`02-bucket-design.md`](02-bucket-design.md) | What each bucket teaches the model. |
| [`01-data-prefix.md`](01-data-prefix.md) | How `[camera=on/off]` is attached to training and eval prompts. |
| [`03-orchestrator-and-build.md`](03-orchestrator-and-build.md) | How the JSONL files are built. |

## Historical Notes

- [`00-datamix-roadmap.md`](00-datamix-roadmap.md): original design roadmap.
- [`A-mix-20k-v1.md`](A-mix-20k-v1.md): first 20K mix, superseded.
- [`C-v3-task-tag-eval-checkpoint-2000.md`](C-v3-task-tag-eval-checkpoint-2000.md): source-keyed tags, replaced by camera-state tags.

## Related

- Final SFT recipe: [`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
- Finetune pipeline: [`../finetune/01-pipeline.md`](../finetune/01-pipeline.md)
