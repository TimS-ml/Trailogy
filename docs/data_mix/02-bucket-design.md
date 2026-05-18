# 02 — Bucket design (record schema + per-source contracts)

How each source bucket is built into a unified JSONL record that
drops into `finetune/src/data.py::load_vision_dataset` without
modification. Versioned: v1 used Cambrian (blocked) + a dummy gray
image trick for text-only records; v2 replaces Cambrian with LLaVA
and allows `image=None` for text-only records natively.

## Unified record schema

```json
{
  "image": "<absolute_path_or_null>",
  "source": "plant|llava|smoltalk|negative|offline_qa",
  "conversations": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

- `image`: absolute path for image-bearing buckets (plant, llava,
  negative); `null` for text-only buckets (smoltalk, most of
  offline_qa). v1 used a shared dummy gray 960×672 JPEG for
  smoltalk to satisfy Gemma4Processor's mixed-batch assertion; v2
  unblocked this via the modality-aware sampler — see §"v1 vs v2
  text-only handling" below.
- `source`: tag for the trainer's multi-eval-dataset reporting
  (`eval_<source>_loss`) and for downstream telemetry / multi-val
  routing. Under the v3 design this field also drove prefix
  dispatch; under v4 the prefix dispatcher reads `image` presence
  instead, so the `source` field is informational only on the
  finetune side. See [`01-data-prefix.md`](01-data-prefix.md).
- `conversations`: exactly the format `build_vision_messages`
  already accepts. Strict user → assistant alternation. The schema
  validator (`src/data_mix/src/schema.py`) enforces this.

## Composition (v2/v3 production mix)

The 45 / 30 / 15 / 10 ratio is set per-config in
`src/data_mix/configs/mix-{50k,100k}.yaml`. Rationale per
ratio:

| Bucket | Share | Why this share |
|---|---:|---|
| Plant | ≤ 45 % | Kept under 50 % to avoid re-overtraining, ≥ 40 % to preserve PlantNet signal. After per-class cap and pool fit-down, lands ~44 % in mix-50k (commit `875803c`). |
| LLaVA | 30 % | Broad visual reasoning diversity — disjoint from PlantNet's species-ID distribution. Replaces Cambrian after the HF streaming blocker on multi-part tarballs. |
| smoltalk | 15 % | Linguistic diversity, text-only, multi-turn chat patterns. Keeps the language backbone limber. |
| Negative | 10 % | Refusal-style pattern on non-plant images. Teaches the model to NOT say "plant" when shown a non-plant image. |

`offline_qa` sits outside this ratio — see §"offline_qa persona
bucket" below.

## Per-bucket implementation contracts

### Plant — `plant_sampler.py`

- Input: `finetune/data/english-desc-v2/train.jsonl` (the canonical
  PlantNet enriched JSONL, ~45,000 records, 782 species after the
  English-vernacular filter — see
  [`../general/15-postmortems.md`](../general/15-postmortems.md) §4).
- Per-class cap: ≤ 30 images per species (configurable via
  `plant.cap_per_class`). Random within cap.
- Prompt variants (uniform random):
  1. `"What plant species is shown in this image?"`
     — response = concise 1-sentence common-name + summary.
  2. `"Identify this plant."`
     — response = same concise desc.
  3. `"Can you tell me what plant this is and describe its key features?"`
     — response = full desc including trait sentences.
- `image` reuses the absolute path of the resized PlantNet JPEG
  (no copy, no re-resize).
- `source = "plant"`.

Dual-source support landed in `6e49ec2` so the sampler can read
from either the v1 single-source pool or a stratified split that
pulls train/val from disjoint per-species partitions.

### LLaVA — `llava_sampler.py` (v2+)

- Source: `liuhaotian/LLaVA-Instruct-150K` family on HuggingFace,
  streaming mode.
- Filter: same `plant` / `flower` / `botan` word-boundary regex
  used by the negative bucket (`ddb836a`) to keep general-VQA
  distinct from negative-VQA.
- Image: decode → resize to 960×672 (`image_resize.py`) → save
  under `${DATA_MIX_IMAGE_ROOT}/llava/<uid>.jpg`. Skip-if-exists
  fast path (`6afa339`).
- Conversation: keep the source record's first user/assistant
  turn pair (LLaVA records are often multi-turn; we take turn 0
  for v2 simplicity).
- `source = "llava"`.

Replaces the v1 `cambrian_sampler.py` which was blocked on the
Cambrian-10M HF streaming tar bug. The v1 sampler is retained in
the historical notes for reference but its config is not shipped and does not
run end-to-end.

### smoltalk — `smoltalk_sampler.py`

- Source: `HuggingFaceTB/smol-smoltalk`, streaming,
  `.shuffle().take(N)`.
- v1 behaviour: bound to shared dummy gray 960×672 JPEG to satisfy
  Gemma4Processor's `len(images) == len(text)` per-batch assertion.
- v2 behaviour: `image = None`. The modality-aware sampler
  (`finetune/src/batch_sampler.py::ModalityAwareBatchSampler`) routes
  text-only records to all-text batches that skip the vision tower
  forward pass entirely. Faster + memory-cleaner than the dummy
  image trick.
- Conversation: turns 0–1 of the source record. Multi-turn
  flattening deferred (would help linguistic diversity, but the
  bucket is already pulling its weight at single-turn).
- `source = "smoltalk"`.

### Negative — `negative_builder.py`

- Source: 2 K (mix-50k) more LLaVA images sampled from the non-plant
  pool (same word-boundary regex as the general LLaVA bucket).
- Image: same resize-and-save pattern as LLaVA, under
  `images/negative/`.
- Prompt: uniformly `"What plant species is shown in this image?"`
  — intentionally identical to the Plant bucket's first prompt
  variant. We want the model to learn to refuse when asked this
  question on a non-plant image.
- Response (fixed template, no `{brief_caption}` substitution
  in v1/v2):
  ```
  I don't see an identifiable plant in this image. Please provide a
  clear image of a plant for identification.
  ```
- `source = "negative"`.

The negative-pool LLaVA images are returned as `List[Path]` (refactor
`0f051c3`) — the legacy dict-of-metadata shape was unused and added
API surface area.

### offline_qa persona bucket — `offline_qa_sampler.py` (v3+)

Added in commit `b1dd29d`. A small (~42-entry) "persona corpus"
sourced from `assets/data_offline_qa/offline_qa.json` —
hand-curated `{question, answer}` pairs that teach the model the
"I run on-device, not connected to ChatGPT" persona. Examples:

- "Are you ChatGPT?" → "No, I run on-device, not connected to ChatGPT."
- "Google this for me." → "I can't search online, I'm offline."
- "What's the weather?" → "I can't check live weather, please use a
  weather app."

Differences from the other buckets:

- **Image-less** (`image = None`).
- **Sits OUTSIDE the main 45/30/15/10 ratio.** The orchestrator
  appends ~38 train + 4 val records on top of the budget. mix-50k
  becomes 50,038 records (0.08 % drift); mix-100k becomes 100,038
  (0.04 %). The existing bucket ratios stay interpretable.
- **No oversampling.** The corpus is tiny (42 entries). Repeating
  each entry N times would teach the model the *exact phrasings*
  rather than the persona. Each entry appears exactly once across
  train + val.
- **Lives on the `[camera=off]` branch** under the v4 camera-state
  gate. `offline_qa` records are text-only (`image=None`) so the
  dispatcher tags them with `[camera=off]` alongside smoltalk and
  the text-only negative records. The persona is therefore taught
  as part of the text-only behaviour the model defaults to whenever
  the iOS app's `.text` mode is active. (Under the v3 design the
  bucket was intentionally UNPREFIXED so the unconditional output
  distribution learned the persona; under v4 the iOS app always
  emits a marker, so "unconditional" is no longer a deploy state
  and the `[camera=off]` branch carries the same role.) See
  [`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
  §7 for the design rationale.
- **`source = "offline_qa"`** — surfaces as `eval_offline_qa_loss`
  in the trainer's multi-eval-dataset log so the persona signal can
  be tracked across checkpoints.

## v1 vs v2 text-only handling

The single biggest schema change across versions.

### v1 — dummy gray image

`finetune/src/data.py` documents (and `finetune.py:445`
historically enforced via `require_image=True`) that
Gemma4Processor + UnslothVisionDataCollator cannot batch
image and text-only records together: the processor asserts
`len(images) == len(text)` per batch and raises `ValueError` the
first time the assertion fails.

v1 worked around this by binding every smoltalk record to a single
shared 960×672 mid-gray (RGB=128) JPEG. The visual gradient through a
constant image averages to ~0 over the bucket, so the visual
encoder doesn't learn from it, but the processor sees a
well-formed multimodal batch and never crashes.

Spec: `dummy_image.py` generates the file once if absent at
`${DATA_MIX_IMAGE_ROOT}/dummy_gray_960x672.jpg`. JPEG quality 90,
all 3 channels = 128.

### v2 — `image=None` + modality-aware sampler

Commit `155cfc9` added `ModalityAwareSFTTrainer` +
`ModalityAwareBatchSampler` which routes records by modality:
all-text batches go to a text-only forward path (no vision tower);
all-image batches go through the normal vision path. Mixed batches
are constructed never to exist.

v2 then dropped the dummy image — smoltalk and offline_qa records
have `image=None` and the trainer handles them natively. The
`dummy_image.py` module stays in the tree for v1 backward compat
but is not invoked by the v2 mix-{50k,100k} configs.

## Determinism + idempotence

Two invariants the test suite pins:

1. **Determinism** (`07f7e1c`): same seed produces byte-identical
   train.jsonl + val.jsonl across runs (modulo HF dataset version
   drift; the build report records the commit hash where available).
   Pinned by `tests/test_mix_integration.py::test_mix_deterministic`.

2. **finetune-side drop-in** (`7f02009`): every record in the mix
   output round-trips through
   `finetune.src.data.build_vision_messages` without raising and
   produces a non-empty messages list. Pinned by
   `tests/test_mix_integration.py::test_mix_output_loads_via_finetune_pipeline`.

## Test status

96 / 96 unit + integration tests pass (was 81 before the
offline_qa landing). Tests use mocked HF streams — no network
required to run the data_mix test suite.

## Related files

| File | Purpose |
|---|---|
| `src/data_mix/src/schema.py` | Record validator |
| `src/data_mix/src/{plant,llava,smoltalk,negative,offline_qa}_sampler.py` | Per-bucket builders |
| `src/data_mix/src/negative_builder.py` | Refusal template + non-plant pool sampler |
| `src/data_mix/src/image_resize.py` | 960×672 stretch (matches `finetune/src/prepare_plantnet.py`) |
| `src/data_mix/src/dummy_image.py` | v1-only — generates the shared gray placeholder |
| `src/data_mix/src/mix.py` | Orchestrator (see [`03-orchestrator-and-build.md`](03-orchestrator-and-build.md)) |
