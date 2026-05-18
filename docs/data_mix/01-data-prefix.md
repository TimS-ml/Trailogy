# 01 — Data prefix mechanism (v4 camera-state gate)

## TL;DR

- Every training and app prompt gets `[camera=on]` when an image is present or `[camera=off]` when the turn is text-only.
- The marker tells the model whether to use vision context without hard-coding the topic of the question.
- This replaced an older source-based tag system that depended on dataset labels such as plant or refusal examples.
- The app can emit the same marker from its existing text-versus-image branch, so no extra runtime state is needed.

How the conditional-FT input-gate prefix is plumbed from config to
the tokens the model actually sees. The mechanism is **input-gate**
regularization keyed on **image presence**: every prompt the model
sees at training time AND at deploy time carries one of two literal
markers:

    record has image (vision turn)  →  "[camera=on] "  + user text
    record has no image (text turn) →  "[camera=off] " + user text

This is one of three independently-toggled levers in the anti-forgetting
stack. The other two (KL output-distribution penalty, L2 weight anchor)
are described in
[`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md).
KL and L2 bound the *magnitude* of drift; the camera-state gate gives
the model a **literal observable signal** for which modality state it
is in, so vision-conditioned behaviour and text-only behaviour stay
on separate manifolds at the prompt level.

## Design rationale — why image-presence, not source

An earlier iteration (v3, deprecated) dispatched per `record.source`
with markers like `[task=plantnet]` / `[task=refuse]`. That coupled
the gate to dataset bookkeeping and forced a `default_source` config
field for legacy single-source JSONLs that didn't carry the field.
It also overloaded the marker with *topic* — `[task=plantnet]` meant
both "this is from PlantNet" and "answer with a plant ID."

The camera-state design strips that down to a two-state modality
flag the deployment path can compute trivially:

- The iOS app already branches on `imageInputs.isEmpty` to pick
  `.text` vs `.vlm` mode (`GemmaService.streamResponse`). Re-using
  that branch for the prefix means there is no extra runtime state
  to track.
- The on-disk JSONL already carries an `image` field; no
  schema change, no `source`-field backfill.
- The marker is **topic-agnostic**: `"[camera=on] What is this plant?"`
  and `"[camera=on] How's the weather today?"` both legitimately
  carry `[camera=on]` — the model is free to answer about the plant
  or the sky as appropriate. The gate says "look at the photo" not
  "answer with a plant ID."

## 1. Config layer — declaring the two markers

In a finetune YAML (`finetune/configs/*.yaml`):

```yaml
data:
  prompt_prefixes:
    camera_on:  "[camera=on] "
    camera_off: "[camera=off] "
```

Default value is `None` — backward-compatible with v2 configs that
don't mention this field. Either key may be omitted (= no prefix for
that branch); asymmetric configs are valid for ablations.

The dataclass lives in `src/finetune/src/config.py`:

```python
@dataclass
class DataConfig:
    ...
    prompt_prefixes: Optional[Dict[str, str]] = None
```

No `default_source` field — the v3.1 fallback existed only because
the dispatcher needed a `source` value to look up. The image-presence
dispatcher has no such dependency.

## 2. Plumbing layer — config → data loader

`finetune/src/finetune.py::load_datasets` reads the dict and threads
it through to the loader:

```python
prompt_prefixes = cfg.data.prompt_prefixes

train_records = load_vision_dataset(
    cfg.data.train_file,
    prompt_prefixes=prompt_prefixes,
    ...
)
val_records_dict = load_vision_dataset_dict(
    resolved_val_files,
    prompt_prefixes=prompt_prefixes,
    ...
)
```

`load_vision_dataset` (`finetune/src/data.py`) iterates the JSONL and
passes each parsed record through `build_vision_messages`:

```python
for raw in iter_jsonl(jsonl_path):
    records.append(build_vision_messages(raw, prompt_prefixes=prompt_prefixes))
```

## 3. Injection layer — `build_vision_messages`

Each JSONL record looks like (v2 schema):

```json
{
  "image": "/path/to/img.jpg",
  "conversations": [
    {"role": "user",      "content": "What plant is this?"},
    {"role": "assistant", "content": "Acer rubrum."}
  ]
}
```

`build_vision_messages` does three things:

```python
# (a) Resolve image_path at the top.
image_path = record.get("image") or None

# (b) Pick the prefix once per record by image-presence lookup.
#     Missing key in the dict (or empty string) = no prefix.
prefix = ""
if prompt_prefixes is not None:
    key = "camera_on" if image_path else "camera_off"
    prefix = prompt_prefixes.get(key, "")

# (c) Inject the prefix ONCE on the first user turn only.
first_user_prefixed = False
for turn in convos:
    text = _strip_image_placeholder(turn["content"])   # legacy <image>\n strip
    if role == "user" and prefix and not first_user_prefixed:
        text = prefix + text                            # ← injection point
        first_user_prefixed = True
    ...
```

## 4. What the model actually sees

After Gemma 4's chat template renders the messages, a plant record:

```
<bos><|turn|>user
[camera=on] What plant is this?<image_soft_token>...<image_soft_token><turn|>
<|turn|>model
Acer rubrum.<turn|>
```

A smoltalk (text-only) record:

```
<bos><|turn|>user
[camera=off] Tell me a joke.<turn|>
<|turn|>model
Why did the chicken...<turn|>
```

Same model, same fine-tune, two surface contracts the model can
condition on.

## 5. Design details worth knowing

1. **Only the first user turn gets the prefix.** The
   `first_user_prefixed` flag ensures multi-turn conversations don't
   accumulate multiple markers. The gate fires once per conversation.

2. **Strip-then-prefix ordering.** The legacy `<image>\n` placeholder
   strip happens *before* the prefix is prepended, so the result is
   `"[camera=on] What plant ..."` not
   `"[camera=on] <image>\nWhat plant ..."`. Pinned by
   `test_data_prompt_prefix.test_prefix_preserves_image_placeholder_strip`.

3. **Empty-string prefix is a no-op.** Setting
   `prompt_prefixes: {camera_off: ""}` is explicitly identical to
   omitting `camera_off` from the dict — no leading space, no garbage.
   Useful escape hatch for asymmetric ablations. Pinned by
   `test_data_prompt_prefix.test_empty_prefix_string_is_noop`.

4. **No system message.** Gemma 4's chat template does not support a
   system role, so the prefix lives at the start of the user content
   rather than as a separate system turn. This is why the gate must
   be a literal string in the prompt, not metadata.

5. **`record.source` is no longer the dispatch key.** Records may
   still carry a `source` field for multi-val routing and telemetry,
   but the prefix dispatcher ignores it entirely. Pinned by
   `test_data_prompt_prefix.test_source_field_does_not_affect_prefix_dispatch`.

6. **Missing `image` field = `camera_off`.** A record that omits the
   `image` key entirely (rather than setting it to `None`) still
   resolves to "no image" → `camera_off`. Falsy and missing are
   equivalent. Pinned by
   `test_data_prompt_prefix.test_missing_image_field_treated_as_camera_off`.

## 6. Where the marker comes from at deploy time

The deployment path (iOS `GemmaService.swift::streamResponse`) must
prepend the matching marker on every Ask:

```swift
let cameraTag = imageInputs.isEmpty ? "[camera=off] " : "[camera=on] "
let userMsg = cameraTag + originalUserPrompt
```

The branch is already there — `imageInputs.isEmpty` is the same
predicate the service uses to pick `.text` vs `.vlm` model kind, so
no new runtime state is introduced. Without this hook the deployed
model sees prompts that miss the gate and falls back toward
base-like behaviour; eval scores can look artificially low when this
hasn't shipped yet.

## 7. Eval path — same dispatch by construction

`finetune/src/evaluate.py::_build_eval_prompt` also accepts
`prompt_prefixes` and forwards to the same `build_vision_messages`,
ensuring eval prompts get the same marker the trained model expects:

```python
def _build_eval_prompt(sample, prompt_prefixes=None):
    record = {"conversations": prompt_conversations}
    if sample.get("image"):
        record["image"] = sample["image"]   # ← presence drives dispatch
    messages = build_vision_messages(record, prompt_prefixes=prompt_prefixes)
    ...
```

When `evaluate.py --config <yaml>` is used, the prefixes are auto-read
from `cfg.data.prompt_prefixes` so training-time and eval-time stay in
sync. A WARNING is logged if eval is run without `--config` — that's
the common footgun where eval scores look artificially low because
eval prompts miss the gate.

## 8. Composition with KL — the deliberate tension

KL pulls student → teacher on **every** training input. The camera-state
gate wants the student to drift FROM teacher inside both branches,
but with different specializations:

- `[camera=on]` records teach vision-grounded behaviour (PlantNet
  species ID, llava VQA, image-grounded refusal).
- `[camera=off]` records teach text-only behaviour (chat, persona,
  text refusal).

If `kl_weight` were too large, KL would dominate and the gate effect
would collapse — the model would just stay close to teacher in both
branches. We picked `kl_weight=0.05` to leave room for the gate to
assert itself while still bounding overall drift magnitude (see
[`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md) §2).

## 9. Bucket → marker assignment (mix configs)

For the v2 data-mix runs (`mix-50k`, `plantnet-50k-mix`,
`plantnet-100k-mix`), the bucket→marker mapping is implicit because
dispatch is on the `image` field, not the bucket name:

| Bucket      | `image` field         | Marker            |
|-------------|-----------------------|-------------------|
| plant       | image-bearing         | `[camera=on]`     |
| cambrian    | image-bearing         | `[camera=on]`     |
| llava       | image-bearing         | `[camera=on]`     |
| smoltalk    | text-only             | `[camera=off]`    |
| offline_qa  | text-only             | `[camera=off]`    |
| negative    | mostly text-only      | `[camera=off]` (image-grounded refusal records get `[camera=on]`) |

The previous design tried to put `negative` and `plant` on separate
markers so refusal wouldn't pollute generic Q&A. Under camera-state
dispatch the refusal data shares its marker with everything else in
its modality — KL + L2 are responsible for keeping refusal behaviour
contained to the actual refusal prompts. If empirical results show
that's not enough, a follow-up could introduce a second axis (e.g.
`[camera=on,intent=refuse]`) without changing the dispatcher.

## 10. One-line summary

The prefix is a literal string prepended to the first user turn of
each training record, dispatched by **image presence** (not by
`record.source`), configured by `data.prompt_prefixes` with up to
two keys (`camera_on` / `camera_off`). The iOS app prepends the
matching marker at inference time using the same image-presence
branch it already uses to pick `.text` vs `.vlm` model kind.
