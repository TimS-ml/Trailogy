# C — v3 task-tag prefix: eval @ checkpoint-2000

## TL;DR

- This document records a mid-training evaluation of the older source-based prompt tags.
- The refusal tag worked exactly on all 250 negative-image examples in the evaluation set.
- General non-plant behavior stayed healthy, which showed the mixed training data was helping prevent collapse.
- Plant responses learned the expected answer format, but species exact-match was still low at this checkpoint.
- The source-based tags were replaced by the simpler `[camera=on]` and `[camera=off]` design.

The v3 task-tag prefix dispatch (`[task=plantnet] ` / `[task=refuse] `
keyed by `record.source`) shipped on 2026-05-16 and was replaced the
same day by the v4 camera-state gate. Before the switch we ran a
mid-training eval against `checkpoint-2000` of the 50k mix run to
confirm the mechanism actually changed model behaviour. This doc
captures those numbers so the v3 design's empirical claims are
not lost.

## Setup

- **Model run**: `plantnet-50k-mix-lora-r256+fullproj+vision2-lr5e5-modalityaware_20260516_114454`
- **Checkpoint**: `checkpoint-2000` (step 2000 of 15485 → 12.9 % of
  the planned schedule, epoch 0.65)
- **Mid-training trainer eval losses at the same step**:
  - `eval_plant_loss: 0.6213`
  - `eval_nonplant_loss: 1.586`
  - `eval_negative_loss: 3.96e-05`
  - `eval_offline_qa_loss: 1.532`
- **Generation eval**: `python -m src.evaluate --config <yaml>
  --adapter_path .../checkpoint-2000 --test_file <val_*.jsonl>`
  with auto-injected training-time prefixes:
  ```
  {'plant':    '[task=plantnet] ',
   'cambrian': '[task=plantnet] ',
   'negative': '[task=refuse] '}
  ```
- **Sample budget**: 300 per bucket (caps to bucket size where smaller).

## Results

| bucket       |   N | ROUGE-L mean | ROUGE-L median | exact match | resp_len | s/sample |
|--------------|----:|-------------:|---------------:|------------:|---------:|---------:|
| `plant`      | 300 |        0.214 |          0.182 |       16.7 % |     71.6 |     1.33 |
| `negative`   | 250 |        1.000 |          1.000 |      100.0 % |    108.0 |     1.57 |
| `nonplant`   | 300 |        0.759 |          1.000 |       61.7 % |    245.7 |     3.43 |
| `offline_qa` |   — | eval pipeline `require_image=True` dropped all 4 text-only persona records; no data | | | | |

`exact match` for plant is species exact-match (case-insensitive).
For negative and nonplant the same column is `species_match_rate`
as emitted by `evaluate.py` but the semantics are different:
negative is "exact refusal template", nonplant is whatever phrase the
species extractor matched in the LLaVA / smoltalk reference — not a
species-ID claim. See "Caveats" below.

The corresponding mid-training trainer loss trajectory from step
1000 → 2000:

| metric                  | step 1000 | step 2000 | direction  |
|-------------------------|----------:|----------:|------------|
| `train.loss` (smoothed) |     0.58  |     0.13  | ↓ (good)   |
| `eval_plant_loss`       |     0.834 |     0.621 | ↓          |
| `eval_nonplant_loss`    |     1.581 |     1.586 | flat       |
| `eval_negative_loss`    |   1.8e-05 |   4.0e-05 | flat ~0    |
| `eval_offline_qa_loss`  |     1.670 |     1.532 | ↓          |

## What the v3 mechanism demonstrably did

1. **Prefix-gated refusal works perfectly.** 250/250 `[task=refuse] `
   prompts returned the exact-template refusal string — zero drift,
   zero hallucinated reasoning, exact match rate 100 %. The dispatch
   from `record.source == 'negative' → '[task=refuse] '` was visible
   and unambiguous in model output.
2. **Generic capability stayed up.** Unprefixed LLaVA/smoltalk
   records (`nonplant` bucket) reach ROUGE-L 0.76 and a median of
   1.0, which is the trainer-side anti-forgetting signal landing
   in generation. The "no prefix → stay close to base" property
   held empirically at this checkpoint.
3. **Format learning beat content learning on `plant`.** Every
   `[task=plantnet] ` response started with the trained answer
   template (`That's <Species>. <Latin name>, known as ...`), but
   species exact-match was only 16.7 % at this checkpoint —
   typical examples:
   - ref `Dutch Clover` → pred `Boar Thistle`
   - ref `European Royal Fern` → pred `Australian Blackwood`
   - ref `Field Marigold` → pred `Canary Island St.` (truncated)

   The trainer-side `eval_plant_loss` dropping 0.83 → 0.62 was
   mostly the template fitting; the vision-tower / projector path
   that maps pixels → species was still early in optimisation.

## What this doesn't say

- **Is v3 the reason `plant` is at 16.7 %?** No. The training
  schedule was 12.9 % complete. The number is a checkpoint-2000
  measurement, not the v3 design's final accuracy ceiling — that
  run was stopped before the v4 switch.
- **Does v4 camera-state perform better here?** Not measured yet
  by this doc. v4 changes the prefix from
  `[task=plantnet] / [task=refuse]` (two markers, source-keyed) to
  `[camera=on] / [camera=off]` (two markers, image-presence-keyed),
  so the *gate count* and *gate role* differ. A v4 run is required
  to make any comparison; the assumption is v4 ≥ v3 on capability
  because the gate carries strictly more information (image
  presence is observable at deploy; source identity is not).
- **`offline_qa` bucket says nothing.** `evaluate.py` defaulted to
  `require_image=True` and dropped all 4 text-only persona records.
  Fix is independent of the v3 → v4 switch; tracked separately.

## Caveats

- `nonplant` and `negative` `exact_match` rates are from
  `evaluate.py`'s species extractor, which is a regex-based
  best-effort phrase-finder tuned for plant Q&A. On non-plant
  references the extracted "species" is whatever short phrase the
  regex returned, so the 61.7 % nonplant column is a coincidence
  signal, not species ID. Use ROUGE-L as the primary metric on
  that bucket.
- `s/sample` includes model load + tokenizer + processor; the
  per-call generation overhead is closer to 0.2–1 s/sample for
  short replies (negative, smoltalk-like) and 3–10 s/sample for
  longer LLaVA-style answers.
- Eval used `use_unsloth: true` per config; behaviour matches
  training-time inference path (FastModel + PeftModel merge).

## Reproducibility

```bash
# In src/finetune/, with the model run still on disk:
python -m src.evaluate \
  --config configs/plantnet-50k-mix-lora-r256+fullproj+vision2-lr5e5-modalityaware.yaml \
  --adapter_path outputs/plantnet-50k-mix-lora-r256+fullproj+vision2-lr5e5-modalityaware_20260516_114454/checkpoint-2000 \
  --test_file data/mix-50k/val_plant.jsonl \
  --output_file results/checkpoint-2000_plant.json \
  --max_eval_samples 300
```

`evaluate.py` auto-injects `cfg.data.prompt_prefixes` from the same
config used to train. For a v3 reproduction you need the config at
the **pre-v4 commit**:

```bash
git checkout v3-task-tag-prefix -- finetune/configs/plantnet-50k-mix-lora-r256+fullproj+vision2-lr5e5-modalityaware.yaml
```

Otherwise the auto-injection will inject the v4 camera-state markers
the model was not trained against.

## Cross-refs

- [`01-data-prefix.md`](01-data-prefix.md) — current v4 camera-state
  gate (the design that replaced v3).
- [`B-mix-50k-v2.md`](B-mix-50k-v2.md) — the 50k mix the run was
  trained on. Bucket ratios and source breakdown live there.
- `../../finetune/07-anti-forgetting-regularization.md` — the
  v3 anti-forgetting stack overall (KL + L2 + task-tag prefix);
  has been amended to point at v4 for the prefix half.
