# 03 — Anti-forgetting and the Final SFT Recipe

## TL;DR

- This doc explains how the finetune tried to add plant knowledge without making the model forget general assistant skills.
- The recorded shipped recipe uses rank 8 LoRA, alpha 8, projector tuning, no KL penalty, no L2 anchor, and 3 epochs on a mixed dataset.
- On a 300-sample PlantNet validation slice, species match rose from 0.000 to 0.230 while general benchmarks stayed at or above the base model.
- Of the anti-forgetting ideas considered, the production recipe kept only the camera-state prompt tag; KL and L2 remained fallback designs.

How we kept Gemma 4 E2B from forgetting how to be a general assistant
while teaching it 782 plant species, and the small-rank recipe that
actually shipped.

## The Problem

The first useful plant-focused fine-tunes were too narrow. They improved plant
answers, but unrelated prompts started to receive plant-classification-style
responses. That failure is unacceptable for Trailogy because the product is an
offline hiking companion, not a plant classifier.

The goal was therefore two-sided:

| Need | What success means |
|---|---|
| Plant adaptation | The model can identify or describe plants from photos. |
| General behavior | Text questions, trail explanations, refusals, and offline persona responses still work. |

## Final Recipe

```yaml
lora:
  r: 8
  lora_alpha: 8
  lora_dropout: 0.05
  tune_projector: true
  projector_learning_rate: 1.4e-4
  tune_last_n_vision_layers: 0

training:
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1
  num_train_epochs: 3
  learning_rate: 3.0e-4
  warmup_steps: 30
  lr_scheduler_type: cosine

regularization:
  kl_enabled: false
  l2_enabled: false
```

Recorded 300-sample validation result:

| Metric | Base | Recipe |
|---|---:|---:|
| Plant species match | 0.000 | **0.230** |
| MMLU | 0.460 | **0.480** |
| AIME | 0.100 | **0.200** |

The result is not presented as a plant-classification ceiling. It is evidence
that the model learned a new visual domain without losing the general behavior
needed by the app.

## What Actually Prevented Forgetting

Three ideas were considered:

| Lever | Purpose | Shipped? |
|---|---|---|
| Mixed data | Keep plant examples below dominance and add general VQA, text chat, refusal, and offline persona records. | Yes |
| Camera-state prefix | Give the model an explicit `[camera=on]` or `[camera=off]` routing signal. | Yes |
| KL/L2 anchors | Penalize drift from the base model in output or weight space. | No, retained as fallback |

The production recipe relied on the mixed corpus and camera-state prefix. KL and
L2 were implemented and tested but not enabled because the small-rank recipe did
not need them.

## Camera-State Prefix

Every first user turn receives a modality marker:

```text
[camera=on]  <image-bearing prompt>
[camera=off] <text-only prompt>
```

This is not a topic label. It only tells the model whether a photo is present.
That makes the training contract deployable: the iOS app already knows whether
it is taking the text path or VLM path, so it can emit the same marker without
running a topic classifier.

The prefix replaced earlier source/task tags because task tags were brittle:
they depended on dataset bookkeeping and forced the app to guess the user's
intent at inference time.

## Why KL Was Not Shipped

KL against the base model sounds like a natural anti-forgetting tool, but it
conflicted with plant learning. The base model often refuses to name a precise
species; the SFT target requires a species name. A KL penalty therefore pushes
against the exact tokens the fine-tune needs to learn.

Observed pattern:

- KL at standard strength preserved generic behavior but blocked plant learning.
- Tiny KL behaved similarly to no KL while adding complexity.
- High-rank runs could still collapse even with KL.
- The mixed dataset provided a denser and more useful preservation signal.

## Why Small Rank Helped

The winning recipe used a small LoRA subspace and `alpha / r = 1.0`. That kept
the update expressive enough to learn the task but not so large that it pushed
the base model into a new narrow mode.

The practical rule from these sweeps:

```text
If plant learning improves but general behavior collapses, reduce update
capacity or strengthen the non-plant data signal before adding more regularizers.
```

## Offline Persona Bucket

The mix includes a small hand-authored offline QA bucket. It teaches responses
such as:

- The assistant runs on-device.
- It cannot search the web.
- It cannot know live weather, closures, or current alerts.
- It should answer from downloaded trail context when possible.

This is a product safety behavior: offline assistants must be honest about live
information they cannot access.

## Verification

The recipe is backed by:

- unit tests for prefix dispatch, KL/L2 wiring, and offline QA sampling;
- GPU smoke tests proving adapter-on versus adapter-off behavior differs;
- generality evals alongside PlantNet evals;
- save/reload tripwires described in [`../general/15-postmortems.md`](../general/15-postmortems.md).

## Related Docs

- Mixed corpus: [`../data_mix/B-mix-50k-v2.md`](../data_mix/B-mix-50k-v2.md)
- Prefix design: [`../data_mix/01-data-prefix.md`](../data_mix/01-data-prefix.md)
- Baseline pipeline: [`01-pipeline.md`](01-pipeline.md)
- Final model eval: [`../general/16-final-model-eval.md`](../general/16-final-model-eval.md)
