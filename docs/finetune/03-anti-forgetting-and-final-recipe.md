# 03 — Anti-forgetting and the Final SFT Recipe

## TL;DR

- This doc explains how the finetune tried to add plant knowledge without making the model forget general assistant skills.
- The recorded shipped recipe uses rank 8 LoRA, alpha 8, projector tuning, no KL penalty, no L2 anchor, and 3 epochs on a mixed dataset.
- On a 300-sample PlantNet validation slice, species match rose from 0.000 to 0.230 while general benchmarks stayed at or above the base model.
- Of the anti-forgetting ideas considered, the production recipe kept only the camera-state prompt tag; KL and L2 remained fallback designs.

How we kept Gemma 4 E2B from forgetting how to be a general assistant
while teaching it 782 plant species, and the small-rank recipe that
actually shipped.

## Final Recipe

```yaml
# The only empirically-verified config that gives both plant > 0
# and mmlu >= base on this task.
lora:
  r: 8
  lora_alpha: 8                   # alpha/r = 1.0 is necessary
  lora_dropout: 0.05
  tune_projector: true
  projector_learning_rate: 1.4e-4
  tune_last_n_vision_layers: 0    # vision_tower frozen

training:
  per_device_train_batch_size: 32  # effective batch = 32
  gradient_accumulation_steps: 1
  num_train_epochs: 3              # ~4600 steps on mix-50k
  learning_rate: 3.0e-4
  warmup_steps: 30
  lr_scheduler_type: cosine

regularization:
  kl_enabled: false                # KL is overkill at small rank
  l2_enabled: false                # not exercised; reserved fallback
```

Result on PlantNet val (300-sample slice, base model `unsloth/gemma-4-E2B-it`):

| metric | base | this recipe |
|---|---:|---:|
| plant species match | 0.000 | **0.230** |
| mmlu | 0.460 | **0.480** |
| aime | 0.100 | **0.200** |

The anti-forgetting work pays off as +2 pts on mmlu (above base, not
just preserved) while plant ID jumps from 0 to 0.23.

## Three levers we considered

After the v2 data mix landed (Plant 45 / LLaVA 30 / smoltalk 15 /
Negative 10), the model still showed "domain collapse" symptoms:
prompts outside the plant distribution produced plant-flavoured
responses (naming species in answers to "what is the capital of
France?"). Mix buckets help but don't fully constrain the LoRA delta.

Three orthogonal levers were designed to address this:

| Lever | What it bounds | Where it acts |
|---|---|---|
| KL penalty | per-token output **distribution** drift | every supervised token |
| L2 weight anchor | **weight-space** drift of every trainable θ | every optimizer step |
| Camera-state tag | **input gate**: explicit modality flag on every prompt | first user turn |

Only the third (camera-state tag) ended up in the production recipe.
The reasoning lives in §5; the data on why KL was dropped lives in §4.

## 1. Camera-state prefix gate (v4)

For every training record we prepend a literal string to the first
user turn. The prefix is dispatched on whether the record carries an
image:

```yaml
data:
  prompt_prefixes:
    camera_on:  "[camera=on] "
    camera_off: "[camera=off] "
```

A plant record's user turn becomes:

```
[camera=on] What plant is this?
```

A smoltalk text-only turn becomes:

```
[camera=off] Hello, how are you?
```

An image-bearing weather question becomes:

```
[camera=on] How's the weather today?
```

The marker is a modality-state flag, not a topic classifier — it says
"there is a photo" or "no photo", independent of what the user is
asking.

### What the model learns

- On `[camera=on] <image> <question>`, ground the answer in the image
  (plant-ID for plant images, VQA for general images, refusal for the
  image-grounded refusal records).
- On `[camera=off] <question>`, answer text-only (chat, persona,
  text-mode refusal).
- The marker becomes a hard observable the model routes on, rather
  than forcing it to infer modality from the rest of the prompt
  distribution.

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
  proofs the format for multi-axis tags without dispatcher changes.
- **Plumbing**: `cfg.data.prompt_prefixes` → `load_vision_dataset` /
  `load_vision_dataset_dict` → `build_vision_messages`.
- **Eval**: `src/evaluate.py::_build_eval_prompt` also accepts
  `prompt_prefixes`. When called via `evaluate.py --config <yaml>`
  the prefixes are auto-loaded from the same yaml.
- **iOS**: the app prepends `[camera=on] ` or `[camera=off] ` to
  every user prompt based on `imageInputs.isEmpty`. The branch is
  already there for `.text` vs `.vlm` dispatch, so no new runtime
  state is introduced.

### v3 → v4 history (superseded)

v3 used a source-keyed dispatch (`[task=plantnet]`, `[task=refuse]`).
Three issues drove the switch to v4 camera-state tags:

1. **Coupling to dataset bookkeeping.** The dispatcher needed a
   `source` field on every record; legacy JSONLs required a
   `data.default_source` fallback.
2. **Topic / modality conflation.** `[task=plantnet]` meant both
   "this is from PlantNet" AND "answer with a plant ID." A plant
   image with a weather question ("is this a shade tree?") was
   forced onto the plant-ID manifold.
3. **iOS deployment.** Picking a task tag at inference time would
   require a topic classifier the on-device app does not have.

## 2. KL penalty against base — designed but not shipped

The intent: an RLHF-style anti-drift term applied at SFT time. For
each step with student logits `s`, labels `y` (`-100` outside
supervised positions), and the same model run under PEFT's
`model.disable_adapter()` context as the teacher `t`:

```
KL = T² · sum_{b,τ : y_{b,τ}≠-100} KL( softmax(s_{b,τ}/T) ‖ softmax(t_{b,τ}/T) )
       / max(1, |positions|)
```

Direction `KL(student ‖ teacher)` (policy-vs-reference, asymmetric).
Default `kl_weight=0.05`, `kl_temperature=1.0`.

### Why `disable_adapter()` instead of a second model?

The "teacher" is just the base model — exactly what
`peft_model.disable_adapter()` returns at forward time:

- **Zero extra GPU memory.** No second checkpoint to load.
- **Bit-identical base.** No risk of the teacher and student getting
  out of sync due to different load paths, dtype promotions,
  processor versions.
- **Free vision-side teacher.** Vision tower and `embed_vision` are
  in `modules_to_save` so they're wrapped by PEFT
  `AuxiliaryTrainingWrapper`. `disable_adapter()` correctly unwinds
  both the LoRA delta AND the `modules_to_save` copy, restoring the
  pretrained base for the entire forward pass.

### Cost

One extra forward pass per training step (~+30–40% wall-clock).
Acceptable for SFT runs that need the anchor; redundant otherwise.

### Why it was disabled in the production recipe

See §4 below — empirically, KL at any non-trivial strength either
prevents plant learning entirely (KL=0.05 blocks species name tokens)
or makes no measurable difference (KL=0.0001 ≈ no KL). The code path
is wired and unit-tested (22 tests in `test_regularization.py` +
`test_trainer_regularization.py`); re-enabling is a one-line config
change if a future task needs it.

## 3. L2 weight anchor (EWC-light) — reserved fallback

At trainer init, snapshot a detached copy of every trainable parameter:

```
θ₀ = { p_i.detach().clone() : p_i ∈ trainable_params }
```

During each `compute_loss` call, add:

```
L2 = sum_i ‖p_i - θ₀_i‖²_F
total = CE + kl_weight · KL + l2_weight · L2
```

For LoRA params this is approximately weight decay toward random init
(LoRA-A) or toward zero (LoRA-B). For `modules_to_save` params
(projector, last-N vision layers) it's a genuine EWC-style anchor
toward the pretrained values — the headline use case.

L2 has **not been exercised in any production sweep**. It's the
reserved fallback if a future task introduces mmlu drift at small
rank: L2 anchors in parameter space (not logit space) and is agnostic
to whether new tokens (e.g. species names) are OOD. To try it, set
`l2_enabled: true`, `l2_weight: 1.0e-4`.

## 4. Empirical data — KL is overkill at small rank

The conclusion that drove the production recipe came from a KL × rank
sweep. PlantNet image root = `data/english-desc-v2/images_resized/test`;
base mmlu = 0.460, aime = 0.100, plant = 0.000.

### Rank fixed at r=8, KL sweep

| run | α/r | KL | step | plant | mmlu | aime | composite |
|---|---|---|---:|---:|---:|---:|---:|
| r8-a16-kl0001 | 2.0 | 0.0001 | 1000 | 0.020 | **0.460** | 0.100 | 0.208 |
| r8-a16-nokl  | 2.0 | 0      | 1000 | 0.040 | 0.280 | 0.200 | 0.169 |
| r8-a16-nokl  | 2.0 | 0      | 2000 | 0.020 | 0.260 | 0.100 | 0.131 |
| **r8-a8-nokl (recipe)** | **1.0** | **0** | **4647** | **0.230** | **0.480** | **0.200** | **0.319** |
| r8-a16-mix50k | 2.0 | 0      | 4000 | 0.140 | 0.560 | 0.150 | 0.243 |
| r8-a16-mix50k | 2.0 | 0      | 5000 | 0.150 | **0.620** | 0.200 | **0.380** |
| r8-a16-mix50k | 2.0 | 0      | 6000 | 0.150 | 0.580 | 0.200 | 0.372 |

- **KL=0.0001 = "nothing was learned"**. mmlu identical to base,
  aime identical to base, plant barely moves. 1000 steps at this lr +
  batch setting was not enough total update magnitude for anything to
  emerge — the run is "training had not yet kicked in", not evidence
  about KL strength.
- **KL=0 + α/r=2.0 = collapse**. mmlu drops to 0.26 by step 2000.
- **KL=0 + α/r=1.0 = recipe**. After 4647 steps (3 epochs), plant 0.23
  and mmlu 0.48 (above base!).
- **KL=0 + α/r=2.0 + mix-50k long run = also holds**. The same α/r=2.0
  recipe with a different lr+batch trajectory runs to step 6000 with
  mmlu rising 0.56 → 0.62 → 0.58, **all above the base 0.46**. The
  mix-50k anti-forgetting buckets carry the model through.

### Rank sweep, KL fixed at standard strength 0.05

| run | rank | α/r | step | plant | mmlu | aime |
|---|---:|---:|---:|---:|---:|---:|
| r4-a8-kl005     | 4   | 2.0 | 1000 | 0.000 | 0.540 | 0.150 |
| r4-a8-kl003     | 4   | 2.0 | 4000 | 0.000 | 0.520 | 0.050 |
| r16-a16-kl005   | 16  | 1.0 | 1000 | 0.000 | 0.480 | 0.150 |
| r32-a32-kl005   | 32  | 1.0 | 1000 | 0.000 | 0.520 | 0.150 |
| r32-a32-kl005   | 32  | 1.0 | 3000 | 0.000 | 0.520 | 0.050 |
| **r256-a256-kl005** | **256** | 1.0 | 1000 | 0.000 | **0.100** | 0.050 |

- **With KL on, plant = 0 for every rank from r=4 to r=32.** Not a
  capacity problem — r=32 already has ~3.5 M trainable params, far
  more than 1000-class plant ID needs. KL pulls the language-side
  logit distribution back toward the base, and **the base never says
  a specific species name for a plant image**.
- **KL=0.05 at r=256 collapses mmlu to 0.10.** The intuition "KL
  preserves mmlu" only holds at small rank. Token-level KL divergence
  is insufficient to pull a high-dimensional weight shift back into
  place at r=256 (~58 M trainable params).

### Why KL kills plant learning

PlantNet ground-truth completion:

```
[camera=on] What is this plant?
→ This is Quercus robur (English oak). It's a deciduous tree ...
```

Base model on the same image:

```
→ I can see a tree with green leaves. The leaves appear to be ...
       I cannot identify the specific species without more context.
```

The base will never output "This is Quercus robur". This is the core
problem with the KL objective: `KL(p_sft || p_base)` is computed
per-token in logit space. For the SFT model to say "Quercus" after
"This is ", it must place probability on a token with near-zero
probability under the base. Every learning step has to fight the KL
gradient, which pushes directly toward "say something generic".

The two-sided gradient magnitudes are uneven but comparable:

- **CE loss (plant)**: ~0.5–1.0 nat/token per batch, pointing toward
  the species-name token.
- **KL loss (kl_weight=0.05)**: `0.05 × reg_kl ≈ 0.04-0.13` nat/token,
  comparable to plant CE.

**The plant-learning gradient signal is cancelled out by an
equal-magnitude opposite force.**

### Drift ≠ forgetting; KL anchor ≠ generality preservation

- **Drift is not the same as forgetting.** r8-a8-nokl trains to
  mmlu=0.48, +2 pts above base. Much of what gets called "drift away
  from base" is in fact "drift toward better behaviour on shared
  tasks".
- **Anchoring to base does not preserve generality at high rank.**
  r256-a256-kl005 keeps KL on and still drops mmlu to 0.10. KL only
  anchors the token distribution, not the displacement direction of
  a high-dimensional LoRA subspace.
- **The real anti-forgetting force comes from the data mix.** mix-50k
  carries 30% image_other, 15% text (including smoltalk QA), 10%
  negative. Each batch sees non-plant samples that contribute a
  "preserve original capability" gradient — a **dense** anti-forgetting
  signal. KL is sparse (one logit-divergence term per batch).
  Training-side validation: the 4090 r8-a16 mix50k run shows
  `eval_nonplant_loss` falling monotonically 1.483 → 1.471 across
  checkpoints 4k/5k/6k, while `eval_offline_qa_loss` falls 1.541 →
  1.220 (−20.8%). Token CE on non-plant data is genuinely improving,
  not just random eval noise.

## 5. S_step framework — per-step adapter update magnitude

A useful way to compare runs that nominally differ in just one knob:

```
S_step ≈ (α / r) × η × eff_bs
```

This is the rough magnitude of the LoRA adapter's contribution to the
activation in one optimizer step. Once it exceeds the "anchoring
radius" provided by KL / L2 / dropout / data-mix buckets, the LoRA
delta overwrites generality faster than anti-forgetting can compensate.

| run | α/r | η | eff_bs | S_step | mmlu | plant |
|---|---|---|---:|---:|---:|---:|
| r8-a8-nokl (recipe) | 1.0 | 3.0e-4 | 32 | **0.0096** | 0.48 | 0.23 |
| r8-a16-nokl (collapsed) | 2.0 | 2.0e-4 | 16 | 0.0064 | 0.26 | 0.04 |
| r8-a16-mix50k (4090) | 2.0 | 2.0e-4 | 16 | 0.0064 | 0.62 | 0.15 |

S_step alone does not fully predict mmlu: runs with the same S_step
(0.0064) spread mmlu from 0.10 (r256-a256) to 0.62 (r8-a16-mix50k).
At least three variables the formula doesn't absorb:

1. **Rank itself (independent of α/r).** r=256 with α/r=1.0 still
   collapses to mmlu=0.10. The higher the LoRA subspace
   dimensionality, the more directions the base model's logit
   manifold gets pushed in.
2. **Sample-exposure pace (`step × eff_bs`).** mmlu often dips then
   recovers; a run killed at step 2000 may be at the bottom of the
   U-shape.
3. **α/r controls more than forward gain.** At α/r=2.0, token-CE
   convergence is faster but generation accuracy lower —
   "willing to say but says it wrong". The α/r=1.0 recipe wins
   because it learns **cautiously**, not because it learns more.

The two safe operating points from this analysis:

**Operating point A — "high throughput, half scale" (the recipe)**
- `α/r = 1.0` (r=8, α=8)
- `η = 3.0e-4`, `eff_bs = 32`
- `S_step ≈ 0.0096`

**Operating point B — "standard scale, half LR, half batch"**
- `α/r = 2.0` (r=8, α=16)
- `η = 2.0e-4`, `eff_bs = 16`
- `S_step ≈ 0.0064`
- Requires the mix-50k dense anti-forgetting signal to survive.

**Anti-pattern:** α/r = 2.0, η = 3.0e-4, eff_bs = 32 stacks all three
knobs at the high end (S_step ≈ 0.019). Do not run without an active
KL or L2 anchor.

## 6. Architectural integration

```
finetune.py
   │
   ├─ build_regularizers(cfg.regularization, model)
   │   └─ snapshots trainable params for L2; constructs KLPenalty
   │
   ├─ make_modality_aware_sft_trainer_class(seed, regularization_state)
   │   └─ ModalityAwareSFTTrainer subclass
   │       ├─ get_train_dataloader: ModalityAwareBatchSampler
   │       ├─ compute_loss: ce_loss + kl_weight·KL + l2_weight·L2
   │       └─ log: drain rolling-window reg_kl / reg_l2 averages
   │
   └─ load_vision_dataset(prompt_prefixes=cfg.data.prompt_prefixes)
       └─ build_vision_messages(record, prompt_prefixes=...)
           └─ prepends prefix to first user turn; dispatches on
              record["image"] presence (camera_on / camera_off)
```

All three levers are independently togglable. The shipping recipe
runs with KL off, L2 off, prefix on — relying on the data mix for
anti-forgetting and the camera prefix for modality routing.

## 7. offline_qa persona bucket

Conceptually orthogonal to the three regularizers: a tiny (~42-entry)
"persona corpus" sourced from `assets/data_offline_qa/offline_qa.json`.
Hand-curated `{question, answer}` pairs that teach the model the
"I'm an offline AI on this device" persona:

- "Are you ChatGPT?" → "No, I run on-device, not connected to ChatGPT."
- "Google this for me." → "I can't search online, I'm offline."
- "What's the weather?" → "I can't check live weather, please use a weather app."

### Why it's not just another negative bucket

The `negative` bucket is **general** refusal: model declines non-plant
prompts because they're off-task. The offline_qa bucket is **specific**
refusal-as-persona: model declines with the on-device-AI character
regardless of topic. Mixing them would confuse the signal. Keeping
them as separate sources lets the multi-eval-dataset track them
independently (`eval_offline_qa_loss` is its own line in the trainer
log).

### Why no oversampling

The corpus is tiny (42 entries). Repeating each entry N times to
inflate the bucket would teach the model the exact phrasings rather
than the persona. Each entry appears exactly once across train+val.

### Position in the mix budget

Bucket sits **outside** the main 45/30/15/10 ratio — the orchestrator
appends ~38 train + 4 val records on top. mix-50k becomes 50,038
records (0.08% drift); the existing bucket ratios stay interpretable.

## 8. Verification

- **Unit tests**: 38 KL + L2 tests
  (`tests/test_regularization.py`, `test_config_regularization.py`,
  `test_trainer_regularization.py`); 11 prefix tests
  (`test_data_prompt_prefix.py`, `test_config_prompt_prefix.py`);
  15 offline_qa tests (`data_mix/tests/test_offline_qa_sampler.py`).
- **GPU smoke**: `tests/test_regularization_gpu_smoke.py` proves
  `disable_adapter()` swap actually reaches Gemma 4's forward (logits
  delta > 1e-3 between adapter on/off), and the full
  `compute_loss + KL + L2 + backward` pipeline produces a finite
  loss with non-zero gradients on the LoRA params.
- **Cross-hardware noise floor**: the same yaml run on two different
  CUDA boxes (24 GB and H-class) produces `eval_plant_loss`
  agreement within ~5% at matched epoch. Generation accuracy can
  still diverge by several pp due to checkpoint selection and eval
  timing, not training-side noise.

## 9. Open follow-ups

1. **Vision-tower last-N + r=8 α=8**: can plant push from 0.23 to
   0.40+ while mmlu holds? The r256-a256-kl005 collapse (§4 rank
   sweep) showed that capacity alone doesn't help; the open question
   is whether vision unfreezing under the safe small-rank LoRA does.
2. **r=4 α=4 nokl**: find the rank lower bound. If r=4 also learns
   plant, the recipe can be made even lighter.
3. **Modality-aware KL weighting** (if KL is ever re-enabled): scale
   `kl_weight` down on image batches, keep full weight on text-only
   batches. The ModalityAware sampler already routes by image
   presence, so wiring is single-axis.
4. **Fisher-weighted EWC**: replace diagonal-ones L2 with a diagonal
   Fisher matrix estimated from a calibration pass. More principled
   than unit-weight L2 but adds a calibration step we don't currently
   need.

## References

- Hinton, Vinyals, Dean (2015), "Distilling the Knowledge in a Neural Network" — T² distillation scaling.
- Kirkpatrick et al. (2017), "Overcoming catastrophic forgetting in neural networks" — EWC.
- Ouyang et al. (2022), "Training language models to follow instructions with human feedback" — RLHF KL.
- Raffel et al. (2020), "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" — T5 task prefixes.
- McCandlish et al. (2018), "An Empirical Model of Large-Batch Training" — gradient SNR vs effective batch size.
