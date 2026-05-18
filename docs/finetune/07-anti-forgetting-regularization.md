# 07 — Anti-forgetting Regularization (v3 SFT)

## TL;DR

- This is the deep dive behind the regularization sentence in the writeup's catastrophic-forgetting section.
- It designs three complementary controls for a plant-identification finetune that still needs to act like a general assistant: KL against the base model, an L2 anchor toward initial weights, and camera-state prompt tags.
- Each mechanism targets a different failure mode: output-distribution drift, weight-space drift, or confusion about whether the current turn has an image.
- The final recipe later kept only the camera-state tag, making KL and L2 documented fallback options rather than default training behavior.

## Context

After the v2 mix (Plant 45 / LLaVA 30 / smoltalk 15 / Negative 10) the
model still showed signs of "domain collapse": on prompts outside the
plant distribution it produced plant-flavoured responses (e.g. naming
species in answers to "what is the capital of France?"). The mix
buckets help but don't fully constrain the LoRA delta — even on generic
inputs the trained model's output distribution drifts.

The anti-forgetting design uses two complementary regularizers plus a
third independent gate:

1. **KL penalty** against the base model's outputs (RLHF-style anti-drift
   applied at SFT time).
2. **L2 weight anchor** toward the parameter values at trainer init
   (the diagonal-ones approximation of EWC from the continual-learning
   literature).
3. **Conditional-FT camera-state tags** (v4; v3 was a source-keyed
   task-tag variant — see §3.1) — prepend a literal marker
   `[camera=on]` or `[camera=off]` to every training prompt, picked
   per record by image presence. The marker gives the model an
   explicit modality-state signal at the prompt level rather than
   forcing it to infer state from the rest of the input distribution.

None of these eliminate the catastrophic-forgetting risk on their own;
together they cover three different attack surfaces:

| Mechanism            | What it bounds                              | Where it acts          |
| -------------------- | ------------------------------------------- | ---------------------- |
| KL penalty           | per-token output **distribution** drift      | every supervised token |
| L2 weight anchor     | **weight-space** drift of every trainable θ | every optimizer step   |
| Camera-state tag     | **input gate**: explicit modality-state flag on every prompt | first user turn |

The first two are model-internal (orthogonal to data); the third is
data-side (orthogonal to model). They're stackable, and we're stacking
all three by default in the v3/v4 configs.

## 1. KL penalty against base

### Formula

For each training step with student logits ``s ∈ ℝ^{B×T×V}``, labels
``y`` (``-100`` outside supervised positions), and the same model run
under PEFT's ``model.disable_adapter()`` context as the teacher
``t ∈ ℝ^{B×T×V}``:

```
KL = T² · sum_{b,τ : y_{b,τ}≠-100} KL( softmax(s_{b,τ}/T) ‖ softmax(t_{b,τ}/T) )
       / max(1, |positions|)
```

(``T`` = temperature; Hinton-style ``T²`` scaling keeps the
kl_weight coefficient interpretable across temperatures.)

### Why ``disable_adapter()`` instead of a second model?

The "teacher" is just the base model — exactly what
``peft_model.disable_adapter()`` returns at forward time. Using the
context manager has three advantages over a second copy:

- **Zero extra GPU memory.** No teacher checkpoint to load — important
  on a 4090 24G where Gemma 4 E2B + LoRA r=256 + projector +
  last-2 vision layers already uses ~12 GB at batch_size 16.
- **Bit-identical base.** No risk of the teacher and student getting
  out of sync due to different load paths, different dtype promotions,
  different processor versions, etc.
- **Free vision-side teacher.** The vision tower and embed_vision are
  in `modules_to_save` so they're wrapped by PEFT
  `AuxiliaryTrainingWrapper`. `disable_adapter()` correctly unwinds
  both the LoRA delta AND the modules_to_save copy, restoring the
  pretrained base for the entire forward pass.

### Cost

~+1 forward pass per training step. With the model on GPU + activations
already materialised by the CE forward, the teacher forward reuses the
KV cache layout but produces detached logits (`torch.no_grad`). Net
overhead measured on the 4090 smoke test: ~30–40 % wall-clock per step
(forward is ~half of a step's cost; the teacher forward adds another
half-forward worth of compute). Acceptable for the v3 plantnet runs.

### Memory: no chunking (2026-05-16 simplification)

The implementation **does not chunk** the fp32 KL math along the
supervised-positions axis. An earlier draft chunked
``masked_student[start:start+64]`` to bound the per-step fp32
footprint to ~320 MB, but the analysis underlying ``chunk_size=64``
was overly conservative:

- After ``mask = labels != -100``, the KL operates on ``N`` rows
  where ``N`` is the count of supervised tokens in the micro-batch —
  typically 200–700 in SFT, not ``B × T``.
- At Gemma 4 vocab ``V ≈ 262 K``, one fp32 row is ~1 MB.
- The five concurrent fp32 buffers (``s_fp32``, ``t_fp32``,
  ``s_log``, ``t_log``, ``elementwise``) at a realistic ``N = 600``
  sum to ~3 GB peak — comfortably inside the 24 GB budget alongside
  bf16 weights, optimizer state, and the teacher's ``[B, T, V]`` bf16
  forward (which we explicitly ``del`` before the fp32 conversions).
- A ``chunk_size = 64`` default cost ~5–10× extra kernel launches
  per KL step without a measurable memory benefit at this ``N``.

Removed in this repo's commit `c652580` (2026-05-16). The
``__init__`` signature is now ``KLPenalty(temperature=1.0)`` only —
the previous ``chunk_size`` parameter is gone. If a future config
pushes ``N`` toward ~2 K supervised tokens (e.g. ``B`` raised on
larger GPUs or every position supervised), revisit: the naive fp32
peak scales linearly in ``N`` and would start to hurt around
``N ≈ 2000``.

### Production-config status (2026-05-16)

The 50k-mix r=256 modality-aware production run trains with KL
**disabled** (`regularization.kl_enabled: false`) and L2 **disabled**
(`regularization.l2_enabled: false`). The combined memory pressure
of LoRA r=256 + projector + last-2 vision layers + teacher forward
+ L2 snapshot was at the 4090 24 GB margin at the practical batch
size we want, and the v3 mix already constrains output drift
through the data side. The KL + L2 code path is wired and
unit-tested (22 tests green post-chunking-removal), so re-enabling
is a one-line config change once we have memory headroom (e.g. a
lower-rank ablation, or A100-class hardware).

### KL direction

We compute ``KL(student ‖ teacher)`` (not the reverse). Rationale:

- This is the **policy-vs-reference** convention from PPO / DPO / RLHF.
- Asymmetric: KL(student ‖ teacher) penalizes student putting mass
  WHERE the teacher does not — i.e. it forbids new outputs that the
  base would never have produced. The reverse direction
  (KL(teacher ‖ student)) would penalize student dropping mass the
  teacher had, which encourages mode-covering — wrong sign for
  anti-forgetting.

### Hyperparameters (production defaults)

```yaml
regularization:
  kl_enabled: true
  kl_weight: 0.05
  kl_temperature: 1.0
```

`kl_weight=0.05` is conservative. At typical CE losses in the 1–2 range,
the KL term contributes 5–10 % of total loss after a few hundred steps
of drift — visible in the trainer log via `reg_kl` (we hook
`trainer.log()` to drain a rolling-window average every
`logging_steps`).

## 2. L2 weight anchor (EWC-light)

### Formula

At trainer init, snapshot a detached copy of every trainable parameter:

```
θ₀ = { p_i.detach().clone() : p_i ∈ trainable_params }
```

During each `compute_loss` call, add:

```
L2 = sum_i ‖p_i - θ₀_i‖²_F
```

Then the total loss is:

```
total = CE + kl_weight · KL + l2_weight · L2
```

### Why anchor to "init" rather than "pretrained"?

For full-finetune the two are equivalent. For our LoRA setup they're
subtly different per param group:

| Param group                  | At init = θ₀          | Anchoring to θ₀ is...               |
| ---------------------------- | --------------------- | ----------------------------------- |
| LoRA-A                       | small random          | ≈ weight decay (toward random init) |
| LoRA-B                       | exactly 0             | exactly L2 toward 0                 |
| `modules_to_save` projector  | pretrained value      | **anchor to pretrained** (EWC use case) |
| `modules_to_save` vision layers (last-N) | pretrained value | **anchor to pretrained** (EWC use case) |

So the same L2-toward-init mechanism does the "right thing" for both
the LoRA delta (≈ standard weight decay, redundant with AdamW's
`weight_decay`) AND the full-rank `modules_to_save` params (genuine
EWC-style anchoring against drift, the headline use case).

We chose to anchor **all** trainable params uniformly rather than
gating to just `modules_to_save`:

- Symmetry / simplicity (one knob).
- AdamW's `weight_decay` is decoupled (applied to params directly in
  the optimizer step, not added to the loss) — having an in-loss L2
  on LoRA params is a non-redundant signal that participates in the
  gradient.

### Memory cost

Snapshot lives in the same dtype as the live param (bf16 for our
setup). For Gemma 4 E2B + LoRA r=256 + projector + last-2 vision
layers, the snapshot is ~50 M bf16 params = ~100 MB. Fits comfortably
in the 4090 24G budget without disturbing batch size.

### Default

```yaml
regularization:
  l2_enabled: true
  l2_weight: 1.0e-4
```

`l2_weight=1e-4` is a low-magnitude floor. Early in training when
params are still near θ₀ it contributes near zero; as the optimizer
pushes weights further from init it grows organically.

## 3. Conditional-FT camera-state tags (v4)

### Mechanism

For every training record we prepend a literal string to the FIRST user
turn's text. The prefix is dispatched on whether the record carries an
image:

```yaml
data:
  prompt_prefixes:
    camera_on:  "[camera=on] "
    camera_off: "[camera=off] "
```

So a plant record's user turn becomes:

```
[camera=on] What plant is this?
```

a smoltalk text-only turn becomes:

```
[camera=off] Hello, how are you?
```

and an image-bearing weather question:

```
[camera=on] How's the weather today?
```

The marker is a modality-state flag, not a topic classifier — it says
"look at the photo" or "no photo", independent of what the user is
asking about.

### What the model learns

The model learns to condition modality-specific behaviour on the marker:

- On `[camera=on] <image> <question>`, ground the answer in the image
  (plant-ID for plant images, VQA for general images, refusal for the
  image-grounded refusal records).
- On `[camera=off] <question>`, answer text-only (chat, persona,
  text-mode refusal).
- The marker becomes a hard observable the model can route on,
  rather than forcing it to infer modality from the rest of the
  prompt distribution.

This bounds the **input gate** of the fine-tune by giving the
deployment path (the iOS app) a trivial way to express the trained
contract: the same `imageInputs.isEmpty` branch that picks `.text` vs
`.vlm` model kind also picks the marker.

### Implementation

- **Code**: `src/data.py::build_vision_messages` accepts a
  `prompt_prefixes: Optional[Dict[str, str]]` parameter and dispatches
  on `record["image"]` truthiness (`"camera_on"` if image, else
  `"camera_off"`). Prepends to the first user turn after the legacy
  `<image>` placeholder strip. A record may also carry a
  `prefix_key` field that overrides the default dispatch — future-
  proofs the format for multi-axis tags (e.g. `camera_on_plant_true`)
  without dispatcher changes.
- **Plumbing**: `cfg.data.prompt_prefixes` → `load_vision_dataset` /
  `load_vision_dataset_dict` → `build_vision_messages`. No
  `default_source` field anymore — image presence is in every JSONL
  by construction so there's nothing to backfill.
- **Eval**: `src/evaluate.py::_build_eval_prompt` also accepts
  `prompt_prefixes`. When called via `evaluate.py --config <yaml>`
  the prefixes are auto-loaded from the same yaml; without
  `--config`, no prefix is injected (legitimate path for evaluating
  no-prefix-trained models such as pre-v3 baselines).
- **iOS** (`HikeCompanion/GemmaService.swift` —
  follow-up work): prepend `[camera=on] ` or `[camera=off] ` to
  every user prompt based on `imageInputs.isEmpty`. The branch is
  already there for `.text` vs `.vlm` dispatch, so no new runtime
  state is introduced.

### Interaction with KL

There is a **deliberate tension** between KL and the camera-state gate:

- KL pulls student → teacher on **every** training input.
- The gate wants student to drift FROM teacher on both branches, with
  different specializations on each side.

If `kl_weight` were too large, KL would dominate and the gate effect
would collapse — the model would learn to behave teacher-like in
both branches. We picked `kl_weight=0.05` to leave room for the gate
to assert itself while still bounding overall drift magnitude.

A clean alternative would be modality-aware KL: scale `kl_weight`
down on image batches (let vision specialization develop) and keep
full `kl_weight` on text-only batches (more aggressive anti-drift
where general chat / persona lives). The ModalityAware sampler
already routes by image presence, so wiring this in is no longer a
second-axis problem — parked for an empirical follow-up rather than
out of scope.

### Default

```yaml
data:
  prompt_prefixes:
    camera_on:  "[camera=on] "
    camera_off: "[camera=off] "
```

`prompt_prefixes: null` (or omitted) keeps v2 behaviour: no prefix
injected anywhere. Either key may also be omitted individually for
asymmetric ablations.

### 3.1 v3 history — source-keyed task tags (superseded)

v3 used a source-keyed dispatch:

```yaml
data:
  prompt_prefixes:
    plant:    "[task=plantnet] "
    cambrian: "[task=plantnet] "
    negative: "[task=refuse] "
  default_source: "plant"   # for legacy single-source JSONLs
```

The model learned `[task=plantnet]` → plant-ID, `[task=refuse]` →
refusal, no marker → base-like. Three issues drove the switch to v4:

1. **Coupling to dataset bookkeeping.** The dispatcher needed a
   `source` field on every record; legacy JSONLs without it required
   the `data.default_source` fallback.
2. **Topic / modality conflation.** `[task=plantnet]` meant both "this
   is from PlantNet" AND "answer with a plant ID." A perfectly
   reasonable plant image with a weather question (e.g. "is this a
   shade tree?") was forced onto the plant-ID manifold.
3. **iOS deployment.** Picking a task-tag at inference time required
   a topic classifier the on-device app doesn't have.

v3 is pinned at git tag `v3-task-tag-prefix` in this repo for
evaluating models trained under that mechanism. To migrate
v3-style baked-prefix JSONLs to v4, use
`finetune/scripts/repair/data_mix_tag_remap.py`.

## Architectural integration

```
finetune.py
   │
   ├─ build_regularizers(cfg.regularization, model)
   │   └─ snapshots trainable params for L2; constructs KLPenalty
   │
   ├─ make_modality_aware_sft_trainer_class(seed, regularization_state)
   │   └─ ModalityAwareSFTTrainer subclass
   │       ├─ get_train_dataloader: ModalityAwareBatchSampler (v2)
   │       ├─ compute_loss:
   │       │     ce_loss + extra_loss
   │       │     extra_loss = kl_weight·KL + l2_weight·L2
   │       └─ log: drain rolling-window reg_kl / reg_l2 averages
   │
   └─ load_vision_dataset(prompt_prefixes=cfg.data.prompt_prefixes)
       └─ build_vision_messages(record, prompt_prefixes=...)
           └─ prepends prefix to first user turn; dispatches on
              record["image"] presence (camera_on / camera_off)
```

The three regularizers are independently togglable:

```yaml
# Pure CE, v2 behaviour:
regularization:
  kl_enabled: false
  l2_enabled: false
data:
  prompt_prefixes: null

# v4 defaults (all three on):
regularization:
  kl_enabled: true
  kl_weight: 0.05
  l2_enabled: true
  l2_weight: 1.0e-4
data:
  prompt_prefixes:
    camera_on:  "[camera=on] "
    camera_off: "[camera=off] "
```

## Verification

- **Unit tests**: 38 KL + L2 tests
  (`finetune/tests/test_regularization.py`, `test_config_regularization.py`,
  `test_trainer_regularization.py`); 11 prefix tests
  (`test_data_prompt_prefix.py`, `test_config_prompt_prefix.py`).
  After the 2026-05-16 KL chunking removal (commit `c652580`) the
  KL + L2 subset (22 tests in `test_regularization.py` +
  `test_trainer_regularization.py`) is re-run green on CPU.
- **GPU smoke**: `tests/test_regularization_gpu_smoke.py` — proves
  `disable_adapter()` swap actually reaches Gemma 4's forward
  (logits delta > 1e-3 between adapter on/off), and the full
  `compute_loss + KL + L2 + backward` pipeline produces a finite
  loss with non-zero gradients on the LoRA params.
- **Generation-side prefix eval (v3 lineage)**: refusal gate showed
  100/100 exact-template match on `[task=refuse] ` at
  `checkpoint-2000`; full numbers in
  [`../data_mix/C-v3-task-tag-eval-checkpoint-2000.md`](../data_mix/C-v3-task-tag-eval-checkpoint-2000.md).
  v4 camera-state generation eval pending (no v4-trained checkpoint
  exists at this writing).

## 4. offline_qa persona bucket (added 2026-05-16)

After the v3 stack landed, we added a fourth mechanism that's conceptually
orthogonal to the previous three: a tiny (~42-entry) "persona corpus"
sourced from `assets/data_offline_qa/offline_qa.json` —
hand-curated `{question, answer}` pairs that teach the model the
"I'm an offline AI on this device" persona. Examples:

- "Are you ChatGPT?" → "No, I run on-device, not connected to ChatGPT."
- "Google this for me." → "I can't search online, I'm offline."
- "What's the weather?" → "I can't check live weather, please use a weather app."

### Why it's not just another negative-bucket

The `negative` bucket in v2 is **general** refusal data: model declines
non-plant prompts because they're off-task. The offline_qa bucket is
**specific** refusal-as-persona: model declines with the on-device-AI
character regardless of topic. Mixing them would confuse the signal — a
"are you online?" prompt should fire the persona, not a generic refusal.
Keeping them as separate sources lets the multi-eval-dataset track them
independently.

### Persona marker — `[camera=off]` (v4) vs UNPREFIXED (v3)

Under the v3 source-keyed dispatch, offline_qa was intentionally left
out of `prompt_prefixes` so the **unconditional** (no-gate) output
distribution learned the persona. The reasoning: when a user asks
"are you online?" they don't think to add a task tag, and we still
want the persona to fire.

Under the v4 camera-state dispatch, the iOS app prepends a marker on
every Ask (`[camera=off]` for text-only Asks, `[camera=on]` for image
Asks), so "unconditional" is no longer a runtime state at deploy
time. The offline_qa records are text-only (`image=None`), so they
land on the `[camera=off]` branch alongside smoltalk and the
text-only negative records. The persona is therefore learned as part
of the model's default text-only behaviour — what it does whenever
the iOS app is in `.text` mode. Same persona-shaping effect, now
expressed through the same gate the rest of the corpus uses.

### Why no oversampling

The corpus is tiny (42 entries). Repeating each entry N times to inflate
the bucket would teach the model the *exact phrasings* ("No, I run
on-device, not connected to ChatGPT.") rather than the persona. We
include each entry exactly once across train+val.

### Position in the mix budget

The bucket sits OUTSIDE the main 45/30/15/10 ratio — the orchestrator
appends ~38 train + 4 val records on top of the budget. mix-50k becomes
50,038 records (0.08 % drift); mix-100k becomes 100,038 (0.04 %). The
existing bucket ratios stay interpretable.

### Verification

15 new tests in `data_mix/tests/test_offline_qa_sampler.py` +
`test_mix_integration.py::test_mix_includes_offline_qa_when_path_set` /
`test_mix_skips_offline_qa_when_path_unset`. data_mix suite total:
81 → 96 / finetune suite unchanged.

### Mid-training visibility

Trainer logs gain a 4th eval bucket: `eval_offline_qa_loss` (alongside
`eval_plant_loss` / `eval_nonplant_loss` / `eval_negative_loss`). At
batch_size=4 the offline_qa val set is one batch (4 records), so the
signal is high-variance step-to-step but trend-meaningful across
checkpoints.

## Open follow-ups

1. **iOS GemmaService**: prepend `[camera=on] ` or `[camera=off] `
   per `imageInputs.isEmpty` in `streamResponse`. Without this, the
   deployed app misses the gate and scores will look artificially
   low on both paths (the model has never seen un-marked prompts).
2. **Modality-aware KL weighting**: scale `kl_weight` down on image
   batches if the camera-state gate is showing collapse in early
   runs (the ModalityAware sampler already routes by image presence,
   so this is single-axis now — easier than the v3 source-aware
   variant).
3. **Fisher-weighted EWC**: replace the diagonal-ones L2 with a
   diagonal Fisher matrix estimated from a pre-training calibration
   pass. More principled than the unit-weight L2 but adds a
   calibration step we don't currently need.
4. **Persona corpus expansion**: if the 4-record val signal proves
   too noisy to read after a few runs, expand the corpus by 2-3×
   with paraphrased entries (carefully — see "no oversampling"
   note above; paraphrases that change the wording are fine,
   literal duplicates are not).

## References

- Hinton, Vinyals, Dean (2015), "Distilling the Knowledge in a Neural Network" — T² distillation scaling.
- Kirkpatrick et al. (2017), "Overcoming catastrophic forgetting in neural networks" — EWC.
- Ouyang et al. (2022), "Training language models to follow instructions with human feedback" — RLHF KL.
- Raffel et al. (2020), "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" — T5 task prefixes.
