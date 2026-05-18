# 02 — Bucket design (record schema + per-source contracts)

## TL;DR

- All training examples use one JSONL shape with an optional `image`, a `source` label, and alternating user/assistant messages.
- The mix combines plant identification, general image questions, text-only chat, non-plant refusal examples, and offline persona QA.
- Plant examples stay below half of the corpus so the model learns plant skills without treating every prompt as plant-related.
- Text-only records now use `image: null` instead of a fake placeholder image.

How each source bucket is built into a unified JSONL record that
drops into `finetune/src/data.py::load_vision_dataset` without
modification. Versioned: v1 used Cambrian (blocked) + a dummy gray
image trick for text-only records; v2 replaces Cambrian with LLaVA
and allows `image=None` for text-only records natively.

## Why A Mixed Corpus Was Needed

PlantNet examples teach a valuable skill, but plant-only training creates a bad
assistant: it learns that most prompts should produce species-identification
answers. Trailogy needs the model to answer plant questions only when the user
is actually asking one, and otherwise remain a general offline companion.

The data mix attacks that failure by training multiple behaviors together.

## Unified Record Schema

```json
{
  "image": "/path/to/image.jpg or null",
  "source": "plant|llava|smoltalk|negative|offline_qa",
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

`image: null` is intentional. Text-only records no longer need fake gray images;
the modality-aware sampler routes text-only batches through a path that skips
the vision tower.

## Buckets

| Bucket | Share | Behavior it teaches |
|---|---:|---|
| Plant | about 44-45% | Identify and describe plant images. |
| LLaVA/general VQA | about 30% | Answer visual questions that are not PlantNet species classification. |
| smoltalk/text chat | about 15% | Preserve ordinary text-only assistant behavior. |
| Negative/non-plant refusal | about 10% | Refuse plant ID when the image is not an identifiable plant. |
| offline_qa | tiny add-on | State offline limitations honestly. |

The exact production composition is documented in [`B-mix-50k-v2.md`](B-mix-50k-v2.md).

## Most Important Design Choices

| Choice | Reason |
|---|---|
| Keep plant below half the corpus. | Avoid making plant classification the default behavior. |
| Preserve general image QA. | A camera question is not always a species-ID request. |
| Include text-only chat. | Trailogy also answers spoken text questions. |
| Use non-plant refusal records. | The model must not hallucinate a plant when the image is unrelated. |
| Add offline persona records once. | Repetition would teach exact phrasings instead of behavior. |

## v1 To v2 Change: Text-Only Records

The first mix used a dummy image for text-only records because the vision
collator expected image and text counts to match. That was wasteful: it spent
vision compute on meaningless gray images and reduced the useful text budget.

The current mix uses `image: null`. The trainer batches image and text-only
records separately, so text-only batches do not run the vision tower.

## How Prefixes Interact With Buckets

The `source` field is for bookkeeping and per-bucket evaluation. It is not the
runtime routing signal. The model receives `[camera=on]` when `image` is present
and `[camera=off]` when it is absent. See [`01-data-prefix.md`](01-data-prefix.md).

This distinction matters because the app can know whether a camera image exists,
but it should not have to infer whether the user's question is a plant task.

## Tests

The data-mix suite checks schema validity, deterministic builds, and drop-in
compatibility with the finetune loader. Tests use mocked remote streams, so they
can run without network access.

## Related Files

| File | Purpose |
|---|---|
| `src/data_mix/src/schema.py` | Record validation. |
| `src/data_mix/src/*_sampler.py` | Per-bucket builders. |
| `src/data_mix/src/mix.py` | Build orchestrator. |
| `src/finetune/src/data.py` | Converts records into Gemma chat messages. |
