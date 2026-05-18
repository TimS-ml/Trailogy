# B — mix-50k v2 (current production canonical) + mix-100k sibling

## What this is

The 50K-row v2 mix that the v3 anti-forgetting stack consumes. Used by
`finetune/configs/plantnet-50k-mix-lora-r256+fullproj+vision2-lr5e5-modalityaware.yaml`
and any other finetune config pointing at `data/mix-50k/*.jsonl`.

A `mix-100k.yaml` sibling exists for runs with a larger compute budget
— identical bucket ratios, 2× the per-bucket sizes, different plant
per-class cap policy. See §"mix-100k sibling" below.

## What v2 changed (relative to v1)

The v2 mix is the cumulative result of four simultaneous changes
(commit `f6d0c1f` landed all four together):

1. **Image source swap.** `nyu-visionx/Cambrian-10M` (streaming-blocked
   on multi-part tarballs) → `HuggingFaceH4/llava-instruct-mix-vsft`
   (parquet-backed, streams cleanly; ~273 K rows = train 259 K +
   test 13.6 K).
2. **Two production sizes.** 100K + 50K instead of a single 20K.
3. **Mid-training multi-val + full checkpoint retention.** Per-source
   val splits, eval every save step, keep every checkpoint (drop the
   `save_total_limit: 2` ring-buffer).
4. **smoltalk text-only skip-vision.** Drop the v1 dummy-gray-image
   trick; let smoltalk records have `image: None` and route them
   through a vision-skip forward path via the modality-aware sampler.

## Config

Source: `src/data_mix/configs/mix-50k.yaml`.

```yaml
seed: 42
source: llava

plant:
  train: 22000              # ≤ 45 % after pool fit-down
  val:   1100
  per_class_cap: 50         # head classes compressed; long tail upweighted

llava:
  train: 15000              # 30 %  general VQA, multi-turn preserved
  val:   750

smoltalk:
  train: 7500               # 15 %  text-only chat (image=None)
  val:   375

negative:
  train: 5000               # 10 %  non-plant images + refusal
  val:   250

offline_qa:
  path: "assets/data_offline_qa/offline_qa.json"
  val_ratio: 0.1            # 38 train + 4 val on a 42-entry corpus
```

`plant.train=22000` leaves a ~438-record buffer under the post-cap
22,438 pool so the build is robust against small drift in the prep
step (commit `875803c`). Earlier `mix-50k.yaml` used `plant.train=22500`
which exactly matched the pool ceiling and broke when prep changed
the species filter slightly.

## Final composition on disk

| File | Bucket(s) | Records |
|---|---|---:|
| `train.jsonl` | mixed, shuffled | **50,038** |
| `val_plant.jsonl` | plant only | 1,100 |
| `val_nonplant.jsonl` | llava + smoltalk | 1,125 |
| `val_negative.jsonl` | negative only | 250 |
| `val_offline_qa.jsonl` | offline_qa persona | 4 |
| `build_report.json` | metadata | — |

50,038 = 50,000 (main mix) + 38 (offline_qa). The 0.08 % drift over
the headline 50K number is deliberately small enough to keep the
ratio interpretable.

## Per-bucket spec (current production)

### Plant — 22,000 train / 1,100 val

- Source: `finetune/data/english-desc-v2/train.jsonl` (the canonical
  PlantNet enriched JSONL produced by `prepare_plantnet_50k.sh`;
  ~45 K records, 782 species after the English-vernacular filter).
- Distribution (PlantNet train, per-species image count):
  `min=1, median=27, p90=139, max=146`. Long-tailed — a per-class cap
  is what controls the head/tail balance.
- Per-class cap: **50** (was 30 in v1 mix-20k; bumped for mix-50k so
  the larger train budget pulls more per-species variety). At cap=50,
  ~22,438 rows land in the train pool and 331 of 782 species hit the
  cap — head classes compressed, long tail preserved fully.
- Per-species stratified train/val split (commit `6e49ec2`): train
  and val come from disjoint per-species partitions, so the val set
  is a true held-out, not random-overlap with train.
- 3 prompt variants, uniform random. See
  [`02-bucket-design.md`](02-bucket-design.md) §"Plant".

### LLaVA — 15,000 train / 750 val

- Source: **`HuggingFaceH4/llava-instruct-mix-vsft`** (273 K rows,
  parquet-backed, streams cleanly via
  `load_dataset(..., streaming=True)`).
- `messages` length 2–14 (multi-turn is common, not rare).
- **Multi-turn PRESERVED** (not truncated to first turn). Rationale:
  iOS deploys 10-turn dialogue (`maxHistoryMessages = 20` in the iOS
  `GemmaService`); training on multi-turn aligns SFT distribution
  with deploy.
- **Content array flattened to string.** Each `message.content` in
  LLaVA-vsft is a list of typed segments (`{type:text, text:...}` and
  `{type:image, index:...}`). Text segments are joined with spaces;
  image segments are dropped (Gemma 4's chat template handles the
  single image slot from the JSONL `image` field separately).
- **Single-image invariant.** Only rows with `len(images) == 1` are
  kept (matches the plant / negative buckets; simplifies the
  collator).
- **No plant filter.** Per spec, LLaVA-mix has very few plant images
  (most are book covers, indoor scenes, charts, OCR). Letting any
  incidental plant images through is fine because the negative
  bucket's refusal template handles the false-context-collapse risk.
  (This is a change from v1, which filtered Cambrian.)
- Image: decode → resize to 960×672 → saved under
  `${DATA_MIX_IMAGE_ROOT}/llava/<rid>.jpg`. Skip-if-exists fast path.

### smoltalk — 7,500 train / 375 val

- Source: `HuggingFaceTB/smol-smoltalk`, streaming with
  `.shuffle(seed=42).take(7875)`.
- **`image = None`** (v2 replaces the v1 dummy gray image with native
  text-only routing via the modality-aware sampler — see commit
  `f6d0c1f` and [`02-bucket-design.md`](02-bucket-design.md) §"v1 vs
  v2 text-only handling"). Savings analysis in §"Skip-vision savings"
  below.
- Single-turn (user 0 → assistant 0).

### Negative — 5,000 train / 250 val

- Source: 5,250 LLaVA-vsft images sampled from the non-plant
  pool — same source as the LLaVA bucket, no plant filter (consistent
  with §"LLaVA — no plant filter" above).
- Fixed refusal template:
  ```
  I don't see an identifiable plant in this image. Please provide a
  clear image of a plant for identification.
  ```
- Prompt: `"What plant species is shown in this image?"` — identical
  to the Plant bucket's prompt variant 1. The contrast is by image,
  not by prompt — the model learns to refuse when the image isn't
  a plant.

### offline_qa — 38 train / 4 val (v3, OUTSIDE the ratio)

- Source: `assets/data_offline_qa/offline_qa.json` (42
  hand-curated `{question, answer}` persona pairs).
- **No image** (`image = None`).
- **Marker = `[camera=off]`** under the v4 dispatch (image=None →
  text-only branch). Under v3 the bucket was intentionally left out
  of `prompt_prefixes` so the persona fired on unconditional
  prompts; under v4 the iOS app always emits a marker so all
  text-only behaviour, persona included, lives on the
  `[camera=off]` path.
- **No oversampling** — each entry appears exactly once. See the
  rationale in
  [`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
  §7.

## How the finetune side consumes this

The production finetune config that uses this mix:

```yaml
# finetune/configs/plantnet-50k-mix-lora-r256+fullproj+vision2-lr5e5-modalityaware.yaml
data:
  train_file: "data/mix-50k/train.jsonl"
  val_files:
    plant:      "data/mix-50k/val_plant.jsonl"
    nonplant:   "data/mix-50k/val_nonplant.jsonl"
    negative:   "data/mix-50k/val_negative.jsonl"
    offline_qa: "data/mix-50k/val_offline_qa.jsonl"
  # v4 camera-state gate: dispatched on image presence, not on bucket.
  # plant, cambrian, llava → image-bearing → [camera=on]
  # smoltalk, offline_qa   → text-only    → [camera=off]
  # negative               → mostly text-only → [camera=off]
  #                          (image-grounded refusal records → [camera=on])
  prompt_prefixes:
    camera_on:  "[camera=on] "
    camera_off: "[camera=off] "

regularization:
  kl_enabled: true
  kl_weight: 0.05
  kl_temperature: 1.0
  l2_enabled: true
  l2_weight: 1.0e-4

training:
  modality_aware_sampler: true     # routes image / text-only batches
  eval_strategy: "steps"
  eval_steps: 1000                 # aligned with save_steps
  per_device_eval_batch_size: 4
  save_steps: 1000
  save_total_limit: null           # keep ALL ckpts (v2 change vs v1's 2)
  group_by_length: false           # conflicts with modality_aware sort
  ...
```

Key points:

- `val_files` is a dict (not `val_file: <str>`), routing each source
  to its own `eval_<source>_loss` line in the trainer log. See
  §"Mid-training multi-val eval" below for the rationale.
- `prompt_prefixes` declares two modality markers (`camera_on` /
  `camera_off`); the dispatcher picks one per record by image
  presence. Every record sees exactly one marker — the gate's job
  is to give the model a literal signal for which modality state
  it's in rather than to mark "which prompts the fine-tune
  activates on" (the v3 framing).
- `modality_aware_sampler: true` is required for the v3
  regularization (KL + L2) compute_loss override AND for the
  text-only routing that lets `image=None` records exist in the same
  training pass as image records.

## Mid-training multi-val eval — catastrophic-forget watch

v1 ran eval only at end-of-training on a single `val.jsonl`. v2 splits
val by source and runs eval every `save_steps` (1000 by default), so
each checkpoint produces:

```
eval_plant_loss
eval_nonplant_loss   # llava + smoltalk combined
eval_negative_loss
eval_offline_qa_loss
```

Why per-source instead of a single mean: **catastrophic forget shows
up as one bucket's loss falling while another rises**. A single mean
loss hides the failure mode. Concretely, the original PlantNet-only
overtraining symptom (model answers "plant" to everything) would
show as `eval_plant_loss` dropping while `eval_nonplant_loss` and
`eval_offline_qa_loss` rise — exactly the signal the multi-val split
makes visible.

HF `SFTTrainer.eval_dataset` natively accepts a dict; each key
produces an independent `eval_<key>_loss` / `eval_<key>_runtime` /
etc. in the metric log.

## Checkpoint retention — keep every one

v1 used `save_total_limit: 2` (rolling buffer of 2 ckpts) to save
disk. v2 changes this to `save_total_limit: null` (keep all) so we
can pick the best by mid-training eval after the fact.

Storage estimate (LoRA r=256 + projector full + last-2 vision layers):

| Mix | save_steps | total steps | ckpt count | per-ckpt | total disk |
|---|---:|---:|---:|---:|---:|
| 50K, 5 epoch, batch 16 | 1000 | 15,625 | ~16 | ~200 MB | **~3.2 GB** |
| 100K, 5 epoch, batch 16 | 1000 | 31,250 | ~31 | ~200 MB | **~6.3 GB** |

A 16 GB-laptop r=64 + projector run (no vision-2 layers) is
proportionally smaller — ~60 MB per ckpt, ~14 ckpts over 14,063
steps → ~840 MB total.

**Risk**: pushing `save_steps` down to 250 + epoch up to 10 + mix
size up to 100K runs to ~250 ckpts ≈ 50 GB. The `mix-100k.yaml`
docstring notes this but no automatic guardrail exists yet — a
sanity print at trainer init that estimates total ckpt disk would be
a cheap future addition.

## Skip-vision savings (smoltalk bucket)

v1 bound every smoltalk record to a shared 960×672 mid-gray JPEG to
satisfy `Gemma4Processor`'s `len(images) == len(text)` per-batch
assertion. Two costs:

1. **Compute**: vision tower forwards on a constant gray image,
   producing 280 useless pooled tokens. ~30–40 % of per-step
   compute is the vision tower.
2. **Token budget**: those 280 tokens occupy 280/1024 of
   `max_seq_length`. Smoltalk's text portion has **744 tokens
   usable** instead of the full 1024 — a **~27 % information density
   loss**.

v2's `image = None` + `ModalityAwareBatchSampler` routes text-only
batches through a no-vision-tower forward path. Effects:

| Lever | v1 dummy-image | v2 skip-vision | Change |
|---|---|---|---|
| Effective text token budget per smoltalk record | 744 | 1024 | **+37 %** |
| Per-step compute on text-only batch | full (vision + LM) | LM only | ~30–40 % faster |
| GPU peak memory on text-only step | full | ~50 % lower | activations drop |
| Wall-time savings across full run (smoltalk = 15 % of train) | — | ~4–6 % | marginal |

The **+37 % effective text budget** is the core win; the wall-time
saving is a side bonus. Memory savings on text-only steps are useful
on the 16 GB laptop budget (head-room for KL teacher forward).

## mix-100k sibling

`src/data_mix/configs/mix-100k.yaml` is a 2× scale-up with
two policy differences:

| Setting | mix-50k | mix-100k |
|---|---:|---:|
| Plant train | 22,000 | 45,000 |
| Plant val | 1,100 | 2,250 |
| **Plant per_class_cap** | **50** | **146** |
| LLaVA train / val | 15,000 / 750 | 30,000 / 1,500 |
| smoltalk train / val | 7,500 / 375 | 15,000 / 750 |
| Negative train / val | 5,000 / 250 | 10,000 / 500 |
| offline_qa | same (~38 train / 4 val) | same |
| **Train total** | 50,038 | 100,038 |

`per_class_cap: 146` on mix-100k is the **max class size** in the
PlantNet enriched train pool — effectively a no-op cap that consumes
all available plant rows. mix-100k cannot be made "more balanced"
without dropping plant rows; mix-50k's `cap=50` is the lever for that.

Build command and on-disk layout are identical to mix-50k, modulo the
target paths under `DATA_MIX_OUTPUT_ROOT`.

## Reproducing the build

```bash
# Set storage roots to wherever you keep the HF cache + working images
# (see env-var table in 03-orchestrator-and-build.md):
export HF_HOME=<your-hf-cache>
export DATA_MIX_IMAGE_ROOT=<your-image-root>/data_mix/images
export DATA_MIX_OUTPUT_ROOT=<repo>/src/finetune/data/mix-50k
# PLANTNET_JSONL falls back to finetune/data/english-desc-v2/train.jsonl

cd <repo>
export CONFIG=$PWD/src/data_mix/configs/mix-50k.yaml
nohup bash src/data_mix/scripts/build_mix.sh \
  > "$DATA_MIX_OUTPUT_ROOT"/mix-50k-build.log 2>&1 &
```

Or for the 100K variant:

```bash
export DATA_MIX_OUTPUT_ROOT=<repo>/src/finetune/data/mix-100k
export CONFIG=$PWD/src/data_mix/configs/mix-100k.yaml
# ... same nohup line
```

Both builds are streaming-sampling — LLaVA-vsft is the slowest step
(image resize CPU-bound). At ~11 GB LLaVA HF cache on the local
disk, 100K predicts ~30–60 min depending on CPU.

Determinism: `seed: 42` + same HF dataset versions → byte-identical
JSONL (pinned by `tests/test_mix_integration.py::test_mix_deterministic`).

## Results to date

Multiple SFT runs against this mix; results tracked under the
finetune-side experiment docs rather than here. Key files:

- `finetune/outputs/plantnet-50k-mix-lora-r256+fullproj+vision2-lr5e5-modalityaware_<timestamp>/`
  — r=256 production runs on 24 GB hardware.
- `finetune/outputs/plantnet-50k-baseline-v2_<timestamp>/`
  — r=64 + projector + full v3 regularization stack runs (16 GB
  laptop budget).

The corresponding eval numbers and `build_report.json` per build are
the source of truth for "what this mix actually produced."

## Cross-refs

- [`01-data-prefix.md`](01-data-prefix.md) — how the
  `prompt_prefixes` dispatch works under the hood.
- [`02-bucket-design.md`](02-bucket-design.md) — per-bucket
  implementation contracts.
- [`03-orchestrator-and-build.md`](03-orchestrator-and-build.md) —
  env-driven build pipeline + idempotence guarantees.
- [`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
  — finetune-side companion: KL + L2 + camera-state prefix design.
