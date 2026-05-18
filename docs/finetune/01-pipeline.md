> last edit: 2026-05-16 (metrics logging — `metrics.jsonl` + opt-in wandb)

# Finetune Pipeline — Baseline (LoRA-only) Mode

## TL;DR

- This doc records the baseline training pipeline for adapting a stock multimodal Gemma 4 model to PlantNet plant identification.
- The recipe prepares fixed-size plant images, trains a bf16 LoRA adapter, merges it back into the base model, and converts the result for MLX INT4 deployment.
- The baseline keeps the vision tower, audio tower, and vision projector frozen so the experiment isolates language-side LoRA behavior.
- The main takeaway is to guard against silent export failures: use a multimodal loader, use the VLM converter, and preserve the trained image size in the processor config.

How `src/finetune/` turns a stock multimodal Gemma 4 E2B
checkpoint and a stack of PlantNet-300K plant photos into a LoRA
adapter that can be merged + re-quantized into the same INT4 MLX shape
the iOS app already ships.

This doc covers the **baseline LoRA-only mode** (vision tower, audio
tower, and the `embed_vision` projector all frozen; LoRA only on the
language model). For the two opt-in modes that unfreeze additional
modules, see:

- [`02-projector-mode.md`](02-projector-mode.md) — adds full-param
  tuning of the `embed_vision` projector. **This is the current
  production baseline-1 recipe** (see `B1-sft-results.md` R0).
- [`03-vision-mode.md`](03-vision-mode.md) — additionally unfreezes the
  last N transformer blocks of the vision encoder. Works correctly
  post-package-fix but treated as a **default-negative probe** at
  production scale.

For the "should we add 4-bit at training time" decision (bnb QLoRA vs
torchao QAT), see [`06-bnb-vs-torchao-sft.md`](06-bnb-vs-torchao-sft.md).

It is the training-side mirror image of the inference-side
preprocessing traps documented in
[`../general/13-mlx-vision-input-parity.md`](../general/13-mlx-vision-input-parity.md);
that one explains the inference-side traps, this one explains the
training-side mirror.

---

## Pipeline Table

| Stage | Hardware | Tool | Dtype on disk | Dtype in memory |
|---|---|---|---|---|
| Data prep (PlantNet → JSONL) | Any | `prepare_plantnet.py` | JPEG, **pre-stretched 960×672** | n/a |
| LoRA training | NVIDIA (A100/4090) | unsloth `FastModel` + trl `SFTTrainer` | base = bf16 (full precision), adapter = bf16 | bf16 |
| Adapter merge | CPU (NVIDIA box or Mac) | `AutoModelForImageTextToText` + PEFT | bf16 safetensors | bf16 |
| MLX conversion | Apple Silicon Mac | `mlx_vlm.convert -q --q-bits 4` | INT4 group-quant | INT4 weights dequant'd to fp16 at matmul |
| Deploy | iPhone | `mlx-swift-lm` | INT4 + processor_config patched to 960×672 | INT4 |

Three things that are easy to get wrong, all guarded by tripwires in
the training and export code:

1. **Don't load with `AutoModelForCausalLM`** at merge time — it
   silently drops `vision_tower.*` / `embed_vision.*`.
   (`_model_has_vision_tower`)
2. **Don't convert with `mlx_lm.convert`** — same silent vision drop at
   sanitize time. (`_mlx_dir_has_vision_weights`)
3. **Don't ship without patching `processor_config.json` to 960×672** —
   mlx-swift-lm falls back to 800×800 and the trained pooler degenerates
   (full mechanism in
   [`../general/13-mlx-vision-input-parity.md`](../general/13-mlx-vision-input-parity.md)).

Three things that aren't tripwired but matter as much:

4. **Both vision AND audio towers must be frozen.** Unsloth has
   `finetune_vision_layers=False` but no audio equivalent — `freeze.py`
   does an explicit post-LoRA walk.
5. **Pre-resize all training images to 960×672**, because mlx-swift-lm
   does fixed-stretch resampling, not aspect-ratio-preserving resize.
6. **Don't apply `train_on_responses_only` to vision SFT** — the
   formatter can't traverse multimodal content blocks.

---

## Pipeline at a glance

```
PlantNet-300K (~1.4M images, 1081 species)
    │
    │  src/prepare_plantnet.py
    │    • stratified 50K-sample subset
    │    • pre-stretch every image to 960×672 (BICUBIC)
    │    • build {image, conversations[]} JSONL
    │    • randomized question/answer templates
    ▼
data/{train,val}.jsonl  + images_resized/<split>/<sid>/*.jpg
    │
    │  src/finetune.py        (NVIDIA only)
    │    • FastModel.from_pretrained (load_in_4bit=False, dtype=bfloat16)
    │    • get_peft_model (finetune_vision_layers=False)
    │    • freeze_vision_audio_towers (post-LoRA walker)
    │    • assert_frozen tripwire
    │    • SFTTrainer + UnslothVisionDataCollator
    ▼
outputs/<run>/final-adapter/  (LoRA weights, bf16, ~30 MB)
    │
    │  src/export_mlx.py      (CPU merge → Mac convert)
    │    • AutoModelForImageTextToText.from_pretrained (bf16, CPU)
    │    • PeftModel.from_pretrained → merge_and_unload
    │    • _model_has_vision_tower tripwire
    │    • mlx_vlm.convert -q --q-bits 4
    │    • _mlx_dir_has_vision_weights tripwire
    │    • patch_processor_config_for_mlx_swift (size: 960×672)
    │    • optional: strip_audio_tower
    ▼
exports/gemma4-mlx/mlx/  (INT4 MLX, ~2.8 GB, drop into iOS bundle)
```

The training side and the inference side meet at the **`size: 960×672`**
constant. Both `prepare_plantnet.py:DEFAULT_TRAINED_VISION_HW`,
`export_mlx.py:TRAINED_VISION_SIZE`, and
`scripts/fetch-gemma.sh:TRAINED_SIZE` must agree. If you change one,
change all three.

---

## Why this stack: unsloth + bf16 + mlx_vlm

The base model the iOS app loads is `mlx-community/gemma-4-e2b-it-4bit`
— already INT4-quantized for MLX. The training pipeline can't just
train against that checkpoint directly:

- It's MLX format, not PyTorch — needs HF/PyTorch to do gradient
  descent through PEFT/transformers.
- It's INT4 with mlx's group quantization scheme — PEFT can't backprop
  through that representation.

So the training base is `unsloth/gemma-4-E2B-it` — same architecture,
unquantized. We load it in **bf16 (full precision)** via
`load_in_4bit=False`. The LoRA adapter sits on top, also in **bf16**.
During `merge_and_unload()`, the LoRA delta is added to the bf16 base,
and the result is saved as bf16 safetensors. That bf16 checkpoint is
then re-quantized to MLX INT4 by `mlx_vlm.convert -q --q-bits 4` for
deploy.

So the weights pass through one quantization step:

```
HF bf16 (unsloth mirror)
  → bf16 (training-time, full precision throughout)
  → bf16 (after merge_and_unload, base + LoRA fused)
  → mlx INT4 group-64 (mlx_vlm.convert, deployed)
```

The only lossy quantization step is the final bf16 → INT4 conversion
for MLX deployment. No intermediate quantization noise.

`mlx_vlm` (not `mlx_lm`) is required because Gemma 4 E2B is
`Gemma4ForConditionalGeneration` — vision_tower + audio_tower +
language_model. `mlx_lm.convert` only reads the language sub-module and
silently drops the rest at sanitize time. Same trap as
`AutoModelForCausalLM`. Both are guarded with explicit safetensors-header
tripwires (`_model_has_vision_tower`,
`_mlx_dir_has_vision_weights`) so a regression aborts the export
instead of silently shipping a vision-blind bundle.

---

## Step 1 — Data prep: `prepare_plantnet.py`

Input: PlantNet-300K extracted as `<root>/{train,val,test}/<species_id>/*.jpg`.
Plus optional `plantnet300K_species_id_2_name.json` for scientific names.

Output:

```
data/
├── train.jsonl                        # ~45K lines for 50K cap (10% val split)
├── val.jsonl                          # ~5K lines
└── images_resized/
    ├── train/<species_id>/<img>.jpg   # 960×672 BICUBIC stretches
    └── val/<species_id>/<img>.jpg
```

Each JSONL line:

```json
{
  "image": "/abs/path/to/images_resized/train/1355937/abc123.jpg",
  "conversations": [
    {"role": "user",      "content": "What plant is this?"},
    {"role": "assistant", "content": "This is Quercus robur. Nice find!"}
  ]
}
```

Three things this script does that aren't obvious:

### Stratified round-robin sampling

PlantNet-300K is heavy-tailed: the long tail is hundreds of species
with a single image. A naive random sample biases toward the head.
`stratified_sample` (`prepare_plantnet.py:117-149`) does
class-by-class round-robin: pop one image from each class in sorted
order, repeat until the budget is exhausted or all classes are empty.
At 50K samples you get the full 1081 PlantNet classes seen at least
once, which keeps the LoRA from collapsing to the top-N.

### Pre-stretch to 960×672 (the core train/inference parity hack)

`resize_image_to_disk` (`prepare_plantnet.py:199-238`) reads each
PlantNet image with PIL, converts to RGB, and **stretches** (not
aspect-ratio-preserving) to 960×672 BICUBIC. The resized copies are
written to `images_resized/<split>/<sid>/<basename>` and the JSONL
points at those.

Why this is necessary: HF's `Gemma4ImageProcessor` (which unsloth uses
internally) does proper aspect-ratio-preserving resize. `mlx-swift-lm`'s
`Gemma4Processor.preprocess` does fixed-size `resampleBicubic` to
whatever's in `processor_config.json`. If the LoRA learns conditional on
aspect-preserved features and the iPhone sees aspect-stretched ones at
inference, the visual feature distributions diverge. Pre-stretching at
data-prep time is the cheapest way to force both sides into the same
distribution. See
[`../general/13-mlx-vision-input-parity.md`](../general/13-mlx-vision-input-parity.md)
"Bug B" for the full story.

This is opt-out: `--resize_to none` disables it.

### No `<image>` placeholder in the user text

`build_conversation` (`prepare_plantnet.py:241-259`) deliberately does
**not** prepend `<image>\n` to the user prompt. The image is conveyed
structurally as a content block by `build_vision_messages`
(`data.py:45-103`):

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "image", "image": "/abs/path/.../foo.jpg"},
    {"type": "text",  "text":  "What plant is this?"}
  ]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "This is Quercus robur. Nice find!"}
  ]}
]}
```

If a `<image>\n` literal also lived in the user text, Gemma 4's chat
template would expand both — double-reserving image soft tokens — and
the vision encoder would only fill one slot, leaving 280 zero vectors
in the prompt. `build_vision_messages._strip_image_placeholder`
defensively strips it if present.

---

## Step 2 — Training: `finetune.py`

```bash
python -m src.finetune --config configs/plantnet-50k.yaml
```

`real_train` (`finetune.py:236-362`) is the only function that touches
CUDA. The dry-run path (`finetune.py:168-197`) exercises everything
else on Mac/CPU.

### Loading — bf16 full precision

```python
# finetune.py:265-272
model, tokenizer = FastModel.from_pretrained(
    model_name=cfg.model.base_model,         # "unsloth/gemma-4-E2B-it"
    dtype=cfg.model.dtype,                   # None → unsloth auto-detects bf16
    max_seq_length=cfg.model.max_seq_length, # 1024
    load_in_4bit=cfg.model.load_in_4bit,     # False → full bf16 precision
    full_finetuning=cfg.model.full_finetuning,  # False
)
```

`max_seq_length=1024` is the cap that defines KV-cache memory at train
time. Push past 2048 and a single A100 throughput drops. Image soft
tokens count toward this — ~280 of the 1024 budget is the `<boi>...`
block, so usable text is ~744 tokens.

### LoRA injection — the explicit `finetune_vision_layers=False`

```python
# finetune.py:277-288
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,   # ← cfg.lora.finetune_vision_layers
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=256,
    lora_alpha=256,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
)
```

### LoRA scaling: why `alpha` must track `r`

LoRA's update is `delta_W = (alpha / r) * B @ A`. The ratio `alpha/r`
is the **effective scaling** of the low-rank update relative to the
pretrained weights. What this means in practice:

| `r` | `alpha` | `alpha/r` | Effective behavior |
|---|---|---|---|
| 8 | 8 | 1.0 | Unsloth default. Adapter has full unit-scale influence. LR controls step size. |
| 256 | 8 | 0.03125 | **Bug.** 256 rank of capacity but the update is multiplied by 0.03 — nearly zeroed out. The optimizer must push 32× harder per step to get the same effective weight change. |
| 256 | 256 | 1.0 | Correct for high-rank. Same effective scaling as r=8/alpha=8. |

The original LoRA-only reference config had `r=256,
alpha=8` — a typo inherited from the r=8 config where alpha=8 was
correct. At `alpha/r = 0.03125`, the adapter is paying VRAM for 256-
rank matrices but barely contributing to the forward pass. The LR
(2e-4) was tuned for `alpha/r = 1.0`; at 0.03 the effective LR drops
to ~6.25e-6, far below what cosine-schedule warmup expects.

**Rule of thumb**: when scaling `r`, always scale `alpha` proportionally.
`alpha = r` (scaling = 1.0) is the safe default. Some practitioners use
`alpha = 2*r` for stronger initial signal, but `alpha << r` is always
wrong — it wastes rank.

Trainable parameter count at `r=256` is ~405M params (~7.3% of the
~5.5 GB model). At `r=8` it's ~30 MB in bf16. The gradient-accumulation
budget on a 4090 stays the same regardless of rank because the LoRA
matrices are small compared to the frozen base.

The validator in `config.py:191-198` rejects `finetune_vision_layers:
true` or `finetune_audio_layers: true` outright — config drift can't
sneak past `validate_config()`.

### Belt-and-braces freeze pass — `freeze.py`

Unsloth's `finetune_vision_layers=False` blocks LoRA injection into
vision modules but **does not** flip `requires_grad` on the underlying
vision parameters. They're frozen by virtue of having no adapters
attached, but a stray `model.requires_grad_(True)` somewhere in the
training loop would re-enable them silently.

Worse, unsloth has no `finetune_audio_layers` parameter at all (per the
notebook commentary: "the audio part can also be finetuned — we're
working to make it selectable as well!"). So audio is the *real*
problem — without intervention, gradients flow through the audio tower
even if `finetune_vision_layers=False`.

`freeze_vision_audio_towers` (`freeze.py:92-162`) walks
`model.named_parameters()` and sets `requires_grad = False` on every
parameter whose dotted path contains one of:

```python
DEFAULT_FROZEN_TOKENS = (
    "audio_tower.",
    "embed_audio.",
    "vision_tower.",
    "embed_vision.",
    "multi_modal_projector.audio.",   # some VLM variants nest here
)
```

The match is **substring with trailing-dot anchoring**, not
`startswith(prefix)`. PEFT and HF wrap the underlying model with
arbitrary extra prefixes:

```
vision_tower.encoder...                       # raw HF
model.vision_tower.encoder...                 # HF Gemma4 wrapper
base_model.model.vision_tower.encoder...      # PEFT + single HF wrap
base_model.model.model.vision_tower.encoder...# PEFT + double HF wrap
```

A finite prefix list silently misses any wrapping depth we forget. The
substring match catches all of them. The trailing dot prevents
look-alikes like `vision_tower_descriptor.weight` from being matched.

`assert_frozen` (`freeze.py:195-211`) is the tripwire — runs
immediately after the freeze pass, raises if any `requires_grad=True`
parameter still matches a frozen token. Without this, a regression in
the freeze logic is invisible until the audio tower starts producing
NaN gradients hours into training. Unit tests in
`finetune/tests/test_freeze.py` parametrize over all four wrapping
variants to lock the behavior in.

### Chat template + vision data collator

```python
# finetune.py:297, 308
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
collator = UnslothVisionDataCollator(model, tokenizer)
```

The collator opens images at batch time (lazy), resizes them via HF's
`Gemma4ImageProcessor`, and produces the multimodal batches that
`SFTTrainer` consumes. Because `prepare_plantnet.py` already
pre-stretched the images to 960×672, HF's aspect-ratio-preserving
resize becomes a no-op — the input is already at the trained shape, so
the resulting patch grid is exactly 60×42=2520 (= `max_patches`) and
the kernel-3 pooler emits exactly 280 cleanly-pooled tokens. Same input
distribution the iPhone will see at inference.

### `train_on_responses_only` — enabled via the collator, NOT the top-level function

There are two separate `train_on_responses_only` mechanisms in unsloth:

1. **Top-level function** (`unsloth.chat_templates.train_on_responses_only`):
   wraps the trainer by pre-tokenizing the dataset with
   `_maybe_tokenize_dataset`, then masking labels. **Broken for vision**:
   `_maybe_tokenize_dataset` calls `tokenizer(texts, ...)` where `texts`
   is expected to be a list of strings, but vision datasets have
   structural content blocks (`[{"type": "image"}, {"type": "text"}]`).
   This is the one that was previously commented out with the note
   "can't handle multi-modal content blocks."

2. **Collator-level support** (`UnslothVisionDataCollator` constructor):
   accepts `train_on_responses_only=True` + `instruction_part` +
   `response_part` as kwargs. Applies the response-only masking
   **after** the collator has already tokenized and processed images.
   No pre-tokenization needed. Works correctly with multimodal batches.

```python
# finetune.py — current (fixed)
collator = UnslothVisionDataCollator(
    model, tokenizer,
    train_on_responses_only=True,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)
```

The Gemma 4 chat template delimits turns as:
```
<bos><|turn>user\n{user text}<turn|>\n<|turn>model\n{model text}<turn|>\n
```

With `train_on_responses_only=True`, labels are set to `-100`
(ignored) on all tokens from `<bos>` through `<|turn>model\n`, and
loss is computed only over the model's response tokens.

**Why this matters**: without response-only masking, the model trains
on the full sequence including the user prompt template. After 3
epochs of 45K examples, it memorizes the prompt patterns ("What plant
is this?", "Describe this plant.") and begins parroting species-ID
responses to *any* similarly-phrased question — even ones about
geology, weather, or trail conditions. This is a form of catastrophic
narrowing: the model hasn't forgotten language ability, it's
overfitted the *prompt pattern → species-ID response* mapping.

### The SFTTrainer entropy patch

```python
# finetune.py:205-233
def _patch_sft_trainer_entropy(cls):
    """Wrap SFTTrainer.compute_loss to skip the entropy metric.

    Unsloth wraps `outputs.logits` as a lazy callable, which breaks
    trl's `entropy_from_logits` (it receives a function instead of a
    tensor). We catch the TypeError and log NaN entropy so training can
    proceed.
    """
```

trl's `SFTTrainer` computes a per-batch entropy metric over the output
logits as a debugging aid. unsloth wraps `outputs.logits` in a lazy
callable for memory reasons (the full logits tensor is huge —
B × T × V floats). When trl's entropy code calls
`logits[..., :-1, :]` on the callable, it crashes with
`TypeError: 'function' object is not subscriptable`.

The patch wraps `compute_loss` to catch that specific TypeError and
fall back to the parent `Trainer.compute_loss`, which doesn't compute
entropy. Loss converges normally — only the entropy stat is dropped.
This is patched at module scope so it runs before `SFTTrainer` is
instantiated; otherwise the very first `training_step` would crash.

### Training hyperparameters

`configs/plantnet-50k.yaml` (the validated 50K reference run):

| Field | Value | Notes |
|---|---|---|
| `per_device_train_batch_size` | 8 | Fits in 24 GB VRAM on a 4090 with `max_seq_length=1024` |
| `gradient_accumulation_steps` | 1 | Effective batch = 8 |
| `num_train_epochs` | 3 | ~37.5K effective steps |
| `learning_rate` | 2e-4 | LoRA-typical |
| `lr_scheduler_type` | cosine | (default.yaml uses linear for short smoke runs) |
| `warmup_steps` | 50 | |
| `optim` | adamw_torch | Full-precision AdamW (no 8-bit quantized optimizers) |
| `weight_decay` | 0.001 | |
| `dataloader_num_workers` | 8 | PlantNet image decode is the throughput bottleneck |
| `dataloader_pin_memory` | true | |

`configs/default.yaml` is the much smaller smoke-test config: batch=1,
grad_accum=4 (effective batch 4), 1 epoch, no dataloader workers.
Fits on a 16 GB GPU but slower per token.

### Metrics logging — `{output_dir}/metrics.jsonl` + opt-in wandb

Until 2026-05-16, training metrics only went to two places: Python
stdout (one line every `logging_steps`) and HF's `trainer_state.json`
inside each checkpoint dir. Stdout disappears with the terminal, and
`trainer_state.json` only lands on disk at `save_steps` — so a
mid-training jetsam / OOM lost the final segment of the curve, and
there was no first-class "loss curve" file you could `pandas.read_*` to
plot a run.

Current pipeline writes two artifacts:

1. **`{output_dir}/metrics.jsonl`** — always on. One JSON line per HF
   trainer `log()` emit, fsync'd to disk each line so a hard kill loses
   at most the in-flight log.
2. **wandb** (or tensorboard) — opt-in, gated on `training.report_to`.
   Stacks cleanly with (1); the callback never mutates the logs dict
   that HF forwards to external trackers.

#### `JsonlMetricsCallback`

Source: `finetune/src/metrics_callback.py`. Wired into both
`ModalityAwareSFTTrainer` and plain `SFTTrainer` branches in
`finetune.py` immediately after trainer construction:

```python
trainer.add_callback(JsonlMetricsCallback(output_dir=cfg.training.output_dir))
```

Each line carries three metadata fields plus the verbatim HF logs dict:

```json
{"step": 100, "epoch": 0.42, "kind": "train",
 "loss": 1.23, "learning_rate": 1.2e-4, "grad_norm": 0.51,
 "reg_kl": 0.012}
{"step": 1000, "epoch": 4.2, "kind": "eval",
 "eval_plant_loss": 0.82, "eval_nonplant_loss": 1.15,
 "eval_negative_loss": 0.41, "eval_offline_qa_loss": 0.93,
 "eval_plant_runtime": 12.3, ...}
{"step": 15625, "epoch": 5.0, "kind": "other",
 "train_runtime": 7821.4, "train_samples_per_second": 6.4}
```

`kind` is a coarse classifier:

- `"train"` — `loss` + `learning_rate` present, no `eval_*` keys.
- `"eval"` — any `eval_*` key present. Multi-val-set runs emit one
  eval log per `eval_steps` with all `eval_<key>_loss` fields in the
  same dict.
- `"other"` — end-of-training summary (`train_runtime` etc).

`reg_kl` / `reg_l2` flow through verbatim — `ModalityAwareSFTTrainer.log`
already injects them into the logs dict the callback sees
(`trainer_modality.py:392`), so the v3 regularizer curves land in the
same JSONL as CE loss without any extra plumbing.

Plotting:

```python
import pandas as pd
df = pd.read_json("outputs/<run>/metrics.jsonl", lines=True)
df[df.kind=="train"].plot(x="step", y=["loss", "reg_kl"])
df[df.kind=="eval"].plot(
    x="step",
    y=[c for c in df.columns if c.endswith("_loss") and c.startswith("eval_")],
)
```

Append-on-resume: the file opens in append mode, so
`--resume_from_checkpoint` continues the curve rather than truncating
it. Resume continues past the last logged step so there's no duplicate
row in practice; if you ever need to dedupe, group by `step`.

Lazy open: a callback instantiated for a dry-run / no-train code path
never sees an `on_log` call, so no phantom 0-byte `metrics.jsonl` is
created.

Unit tests (`finetune/tests/test_metrics_callback.py`) cover the
classify-train/eval/other matrix, lazy open, append semantics,
multi-val-set eval keys, resume append, non-JSON value fallback (degrades
to `repr` rather than crashing the training loop), and the `reg_kl` /
`reg_l2` passthrough.

#### Wandb — opt-in, run name = output_dir basename

`training.report_to: "wandb"` (or `--report_to wandb`) flips on HF
Trainer's wandb integration. `finetune.py` primes a couple of env vars
in `real_train`:

- `WANDB_PROJECT` defaulted to `"hikecompanion-finetune"` if unset, so
  all runs land in one project unless overridden.
- `WANDB_MODE` left to the operator — set `WANDB_MODE=offline` on
  air-gapped boxes, then `wandb sync outputs/<run>/wandb/` later from a
  connected machine.

The wandb run name comes from `cfg.training.run_name`, which is the
basename of `output_dir` by construction (`_generate_run_name` in
`finetune.py` produces `<config_stem>_<timestamp>` if not explicitly
set). To override at launch time:

```bash
python -m src.finetune --config configs/<...>.yaml --run_name my_ablation_v1
```

The `run_name` value is also passed to `SFTConfig` directly so HF
Trainer's wandb integration uses our run name (without this passthrough
HF falls back to a synthesized name that drifts from the
`outputs/<run_name>/` directory layout).

Privacy note: wandb run metadata can carry absolute
local paths. `.gitignore` excludes `wandb/` and `**/wandb/` so the
artifacts never get committed.

### Dry-run path

`python -m src.finetune --config configs/default.yaml --dry-run` runs
`dry_run` (`finetune.py:168-197`) instead of `real_train`. It validates
the config, loads JSONL through `build_vision_messages`, prints stats,
and skips `FastModel.from_pretrained` + the training loop. CUDA-free,
runs on a Mac. Used in CI (`tests/test_dry_run.py`) and as a pre-flight
on the GPU box before kicking off real training.

---

## Step 3 — Export: `export_mlx.py`

```bash
# Mac with both torch and mlx (full pipeline):
python src/export_mlx.py \
    --base_model unsloth/gemma-4-E2B-it \
    --adapter_path outputs/plantnet-50k-lora/final-adapter \
    --output_dir exports/gemma4-mlx \
    --quantize_bits 4 \
    --strip_audio

# Or split: merge on NVIDIA box, convert on Mac
# (training box):
python src/export_mlx.py --adapter_path ... --merge_only
# (Mac):
python src/export_mlx.py --merged_dir exports/gemma4-merged/merged \
    --output_dir exports/gemma4-mlx --quantize_bits 4
```

Four sub-steps, each with explicit failure modes.

### 3a. Merge — `AutoModelForImageTextToText`, NOT `AutoModelForCausalLM`

```python
# export_mlx.py:225-231
model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cpu",
)
```

`AutoModelForImageTextToText` resolves to
`Gemma4ForConditionalGeneration` for Gemma 4, preserving
`language_model + vision_tower + embed_vision + audio_tower` as a
single top-level module. **`AutoModelForCausalLM` resolves to the
language sub-module only** — it loads only the `language_model.*`
weights and silently drops everything else. The resulting checkpoint
can answer text questions, but `vision_tower.weight is None` at
runtime and any attempt to scatter image tokens into the prompt
crashes the iOS app (or worse, returns garbage).

Tripwire: `_model_has_vision_tower` (`export_mlx.py:263-272`) walks
`named_parameters()` and aborts the merge if no `vision_tower.` /
`embed_vision.` keys are found.

`device_map="cpu"` avoids fitting the full bf16 model on a single GPU
during merge. The base is ~5 GB bf16; PEFT briefly holds adapter +
base + delta = ~10 GB during `merge_and_unload`. CPU RAM has no
ceiling, GPU does.

### 3b. MLX conversion — `mlx_vlm.convert`, NOT `mlx_lm.convert`

```python
# export_mlx.py:337-343
cmd = [
    sys.executable, "-m", "mlx_vlm.convert",
    "--hf-path", str(merged_dir),
    "--mlx-path", str(mlx_dir),
    "-q",
    "--q-bits", str(quantize_bits),  # 4
]
```

Same trap as 3a: `mlx_lm.convert` is language-only and silently drops
vision_tower at sanitize time. `mlx_vlm.convert` is the vision-aware
sibling.

The `-q --q-bits 4` flag triggers MLX's group-wise quantization.
Default group_size=64. Weights are stored as packed uint32,
scales/biases stored as fp16 sidecars. At inference the matmul kernel
dequantizes on-the-fly to fp16. The PPL hit is well under 2% on
Gemma-class models, so end-to-end the only "real" lossy quantization
step is the final bf16 → INT4 conversion for MLX deployment.

Tripwire: `_mlx_dir_has_vision_weights` (`export_mlx.py:379-400`)
reads the safetensors headers (without loading tensor data) of every
`*.safetensors` in `mlx_dir`, aborts if no `vision_tower.` /
`embed_vision.` key appears. Robust to sharded checkpoints.

Subprocess invocation rather than Python API because `mlx_vlm`'s Python
API is not version-stable across releases — the CLI flags are.

### 3c. processor_config.json patch

```python
# export_mlx.py:295
TRAINED_VISION_SIZE = {"height": 960, "width": 672}

# export_mlx.py:403-456
def patch_processor_config_for_mlx_swift(mlx_dir: Path) -> bool:
    """Hoist do_normalize / image_mean / image_std from nested
    image_processor block to top level, AND force size to 960×672."""
```

This is the export-side mirror of `scripts/fetch-gemma.sh:66-122`. It
does the same two transformations:

1. **Hoist** `do_normalize` / `image_mean` / `image_std` from the
   nested `image_processor` block to the top level — `mlx-swift-lm`'s
   `Gemma4ProcessorConfiguration` decoder reads them from the top
   level only.
2. **Force** `size: {height: 960, width: 672}` at both the top level
   and inside `image_processor`.

Without this patch, the iOS bundle would either fall back to 800×800
(no `size`) or to whatever `mlx_vlm.convert` happened to write through
from the upstream HF config (likely `224×224`). Either way the trained
kernel-3 vision pooler degenerates — full mechanism in
[`../general/13-mlx-vision-input-parity.md`](../general/13-mlx-vision-input-parity.md).

The architectural fields (`image_seq_length=280`, `pooling_kernel_size=3`,
`patch_size=16`, `default_output_length=280`) are deliberately left
alone. They are trained values.

### 3d. Optional audio strip

```python
# export_mlx.py:464-479
def strip_audio_from_mlx_dir(mlx_dir: Path) -> None:
```

Same logic as `scripts/strip-gemma-audio.py`: read each safetensors
header, drop any tensor whose key starts with `audio_tower` /
`embed_audio`, write a new safetensors with the kept tensors at
re-computed offsets. ~580 MB savings on disk. Zero functional impact
because `mlx-swift-lm`'s sanitize step filters audio anyway.

The MLX-converted INT4 checkpoint is much smaller than the raw bf16
(~2.8 GB vs ~5 GB), so the audio strip saves proportionally less in
INT4 space — but it's still ~150 MB at 4-bit, worth doing if the
target is iPhone storage.

---

## How the LoRA-only mode works (forward pass)

This setup looks counterintuitive at first: it's a **vision** LoRA SFT
(the training data has images, the model sees them at training time),
yet the entire vision tower AND the projector are frozen. So what
changes between a base checkpoint and a finetuned one?

### The forward pass, with frozen / trainable annotated

```
image (960×672)
   │
   ▼  vision_tower (SigLIP encoder)         [FROZEN]
patches → 2520 × 768 features
   │
   ▼  Gemma4VisionPooler (kernel-3 stride)  [FROZEN]
280 × 768 pooled features
   │
   ▼  embed_vision (projection → LM hidden) [FROZEN]
280 image soft tokens (each is an LM-hidden-dim vector)
   │
   │  scattered into the prompt:
   │  <bos>...<boi>[280 soft tokens]<eoi> What plant is this?<eos>
   ▼
language_model (decoder layers)              [LoRA-adapted ✓]
   │
   ▼
"This is Quercus robur. Nice find!"
```

For any given input image, the 280 soft tokens that come out of
`embed_vision` are **bit-identical** before and after fine-tuning. The
visual feature space is fixed. What changes is how the language model
attends to and interprets those soft tokens.

### Where LoRA actually intervenes

In each language-model attention layer, the 280 image soft tokens are
**re-projected** by that layer's `W_K` and `W_V` matrices to produce
keys and values for attention. LoRA injects a low-rank delta into
those projection matrices:

```
x_image (frozen, from vision tower)
   ▼  W_K + LoRA_K            ← LoRA changes here
K_image  (effectively re-keyed)
   ▼  W_V + LoRA_V            ← LoRA changes here
V_image  (effectively re-valued)
```

So even though the *source* image representation is frozen, LoRA can
**re-key** and **re-value** that representation per-layer. The vision
tower is a fixed sensor; LoRA teaches the LM how to interpret the
sensor's readout.

What each LoRA-adapted matrix is doing:

| LoRA-adapted matrix | Effect |
|---|---|
| `W_Q` (text positions) | Teaches text tokens how to "ask questions" of the image — e.g. when generating after `"This is"`, what to retrieve from the image soft tokens |
| `W_K` (image positions are re-projected through this) | Re-keys image soft tokens — promotes some visual dimensions to be more attendable, demotes others |
| `W_V` (image positions are re-projected through this) | Re-values image soft tokens — changes what information they actually contribute when attended to |
| `W_O` | Mixes the attended-to information back into the residual stream differently |
| MLP `up_proj` / `gate_proj` / `down_proj` | Improves the per-token mapping from "I attended to plant features" → "output the species-name token sequence" |

### Information-theoretic framing

SigLIP was pretrained on web-scale image-text pairs. Its 280-token
output already implicitly encodes which oak species this is, which
fern, which fungus — it has seen millions of plant photos with
captions during pretraining. That visual discrimination capability is
**already in the frozen tower**.

What the base Gemma 4 doesn't have is a precise mapping from
"SigLIP's visual signature for this oak" to "the string `Quercus
robur`". It knows the Latin name as a language fact (it was trained on
text mentioning *Quercus robur*), and it can produce vague answers
like "this looks like an oak" because SigLIP gives it enough signal —
but the precise cross-modal lookup table doesn't exist out of the box.

The LoRA learns exactly that lookup, distributed across all the
language-layer attention and MLP weights. In one sentence:

> **Frozen-vision LoRA SFT learns a new cross-modal mapping between
> a fixed visual feature space and a fixed language knowledge space.
> It does not teach the model to see, and does not teach the model new
> facts. It teaches the model: "when you see X, say Y".**

This is functionally a form of implicit knowledge distillation: the
PlantNet ground-truth labels (expert-annotated species IDs) get
distilled into the LM's attention pattern, in the form of "given
SigLIP's visual signature, prefer this species token over that one."

### The trade-off — catastrophic forgetting on the language side

Because LoRA only touches language layers, and the training data is
narrow (templated plant Q&A), the model can drift on **general
conversational ability**. A LoRA trained only on PlantNet starts to
answer "how's the weather?" with "This is some plant. Nice find!"
because the templated answer pattern bleeds into every response.

This is mitigated three ways in the current pipeline:

1. **Bounded LoRA influence via α/r scaling**: even at r=256 the
   effective update magnitude is `α/r = 1.0`, so the adapter's
   contribution is bounded relative to the pretrained weights.
   Trainable params land at ~7.3% of the base in the r=256 config, but
   the *effective magnitude* of the update is what governs forgetting,
   not the parameter count. Lower-r configs (r=8, α=8) sit at ~0.5%
   trainable with the same effective scaling, and are useful when
   forgetting symptoms appear.
2. **Short training**: 1–3 epochs over 50K samples. Long training
   amplifies drift.
3. **Mixed-task data**: `prepare_data.sh` step 3 merges
   `prepare_hiking_qa.py`'s synthetic hiking conversations into the
   training set alongside the PlantNet plant-ID data. The LM keeps
   seeing "general hiking knowledge" examples between plant-ID
   examples, which anchors its conversational distribution.

---

## Empirical results (2026-05-12; full-eval re-validation in progress 2026-05-15)

After fixing the PEFT orphan-tensor bug (see "Package version fix"
below), the pipeline works correctly. Results from the overfit100
diagnostic suite (100 samples, 30 epochs, same data for train+eval —
a **pure memorization ceiling test**, not a generalization measurement)
and the full plantnet-50k validation run.

> **Re-eval caveat (2026-05-15).** The prior eval pipeline had defects
> that biased species_match numbers **low**, especially on the full
> n=2,870 PlantNet validation. Re-eval is in progress. Numbers in
> §"Full plantnet-50k validation" below should be read as a lower
> bound, not as the canonical reference; the production-grade numbers
> will land in `../../quantization/B1-sft-results.md` and the
> `M*` rows of `../../quantization/B2-sft-results.md`. The
> overfit100 numbers in this doc are **not** affected — they are
> training-time memorization measurements, not held-out eval.

### Overfit100 memorization ceiling (100 samples, 30 epochs)

| Config | LoRA r | alpha | Extra modules | Final train loss | ROUGE-L | Species Match |
|---|---|---|---|---|---|---|
| lora-r8-a8-lr2e4 | 8 | 8 | none | 0.0005 | 0.073 | 0 % |
| lora-r256-a8-lr2e4 | 256 | 8 | none | 0.003 | 0.898 | 93 % |
| r256+fullproj-lr5e5 | 256 | 256 | projector | 0.00004 | 0.9998 | **100 %** |
| r256+fullproj+vision2-lr1e5 | 256 | 256 | projector + vision last 2 | 0.00007 | 0.960 | 96 % |
| classify-r8-a8-lr2e4 | 8 | 8 | none (classify) | 0.007 | 0.720 | 72 % |

**Mode ranking by species_match (overfit100 memorization)**:
**projector + LoRA (100 %) > projector + LoRA + vision-last-2 (96 %)
> LoRA r=256 (93 %) > LoRA r=8 (0 %)**.

Two readings of this ranking. (a) The projector bridge is the bottleneck
that the LoRA-only mode can't reach around — adding 1.18 M projector
params closes the last 7 % to a clean ceiling. (b) Adding vision-tower
capacity on top of projector tuning **regresses by 4 pp on the same
memorization test**, with twice the training time. The capacity is
there but the loss landscape gets worse, not better. See
`03-vision-mode.md` §"When vision-tower tuning should become valuable —
revised 2026-05-15" for the production-scale reading of this
regression.

Caveat: overfit100 measures *only* train-set memorization. A mode that
wins here may or may not win at production scale; a mode that loses
here is unlikely to be saved by more data unless there's a mechanistic
reason (e.g. "vision encoder needs more data to adapt without
forgetting"). The vision-mode regression at overfit100 is the
opening-rounds evidence that the SigLIP tower is fragile under
training-time updates — companion PTQ-side evidence is the bnb-NF4
0.1 % collapse in `../../quantization/B1-bnb-nf4-vision-collapse.md`.

### Overfit100 data-format ablation (r=256, α=8, language-only)

Same 100 V2 images, same LoRA config (`r=256, alpha=8, lr=2e-4`,
language-only), 30 epochs.  Only the target text format varies:

| Variant | Species name | Wiki desc | Final loss | ROUGE-L | Species Match | Avg resp len |
|---|---|---|---|---|---|---|
| **english + wiki** (baseline) | English common | Yes | 0.003 | **0.898** | **93 %** | 176 chars |
| **latin + wiki** | Latin scientific | Yes | 0.002 | 0.814 | 83 % | 330 chars |
| **english, no wiki** | English common | No | 0.002 | 0.699 | **90 %** | 30 chars |

**Data-format ranking by species_match (overfit100 memorization)**:
**english + wiki (93 %) > english only (90 %) > latin + wiki (83 %)**.
ROUGE-L orders differently because `english only` has a shorter target
(30 chars), so template-prefix variance dominates the ROUGE
denominator without hurting the species-id signal. For the deployed
metric (exact binomial match) the ranking that matters is
species_match, not ROUGE.

Configs:
- `plantnet-overfit100-lora-r256-a8-lr2e4.yaml` (english + wiki — the
  original baseline from the table above)
- `plantnet-overfit100-lora-r256-a8-lr2e4-latin.yaml`
- `plantnet-overfit100-lora-r256-a8-lr2e4-english-nodesc.yaml`

Data files (all generated from the same 100 V2 val-set images):
- `data/overfit100-v2-english.jsonl` — English common name + Wikipedia
  summary (e.g. "Looks like Coppertone Stonecrop to me. Sedum adolphi,
  the coppertone stonecrop or golden Sedum, is a species of …")
- `data/overfit100-v2-latin.jsonl` — Latin binomial + Wikipedia summary
  (e.g. "Looks like Sedum adolphi Raym.-Hamet to me. Sedum adolphi, the
  coppertone stonecrop or golden Sedum, is a species of …")
- `data/overfit100-v2-english-nodesc.jsonl` — English common name only
  (e.g. "Good eye — this is Coppertone Stonecrop.")

**Observations:**

1. **English + wiki is the best overall** (93% species match). The wiki
   description acts as additional grounding context — the model sees the
   species name repeated in the description body, reinforcing the
   image → name mapping. English common names are shorter and have
   higher pre-training frequency than Latin binomials, making them easier
   to memorize and reproduce.

2. **Latin + wiki drops to 83%.** Latin binomials are harder to memorize:
   they contain unusual tokens (author abbreviations like `"L."`,
   `"(Hook.f.) Baker"`), and the model has less pre-training exposure to
   exact taxonomic naming conventions. The average response is also longer
   (330 chars vs 176), meaning more tokens to get right per sample.

3. **English, no wiki hits 90% species match but lower ROUGE-L (0.699).**
   The model memorizes the short species names well (30 chars avg). The
   lower ROUGE-L is a template-matching artifact: the model sometimes
   picks a different template prefix ("Looks like X" vs "Good eye — this
   is X") than the reference. For the iOS app (which only cares about
   species identification, not exact template reproduction), this
   configuration delivers nearly the same accuracy as the full
   english + wiki variant but with a much shorter (and cheaper) target.

4. **Practical implication for production training:** English common
   names are the right default for the iOS app's use case. The wiki
   description provides a modest accuracy boost (+3% over no-desc) at the
   cost of longer sequences. For a 50k-sample production run where
   sequence length directly impacts throughput, the no-desc variant is
   worth evaluating as a faster alternative.

### Full plantnet-50k validation (n = 2,870)

> **Re-eval in progress (2026-05-15).** The numbers below were produced
> by an eval pipeline that has since been found to bias species_match
> low. They remain in this doc only to establish the qualitative
> conclusion (LoRA-only can learn species identification when packages
> are correct). The **canonical** production-grade numbers — including
> the projector-mode baseline-1 row — live in
> `../../quantization/B1-sft-results.md` (row R0 for the bf16
> reference) and `../../quantization/B2-sft-results.md` (rows
> M0-M3 for the MLX-INT4 deploy variants), and will be re-stated here
> after re-eval lands.

| Config | Dataset | Species Match | Status |
|---|---|---|---|
| LoRA r=8, bf16, plantnet-50k | 50k train / 2870 val | 46 % exact binomial match | low-biased; see caveat above |
| projector + LoRA r=256 + data-aug-enwiki, bf16 (baseline-1 recipe) | 50k train / 2870 val | tracked in `B1-sft-results.md` R0 | under re-eval; current row in B1 may also move |

What is robust regardless of re-eval:

- **LoRA-only at r=8 can learn species ID** when packages are correct.
  The earlier 0 % results were entirely caused by the PEFT
  orphan-tensor bug silently dropping 80 LoRA tensors on adapter
  reload (see §"Package version fix" below).
- **Production baseline-1 uses the projector + LoRA recipe**
  (`02-projector-mode.md`), not LoRA-only. The qualitative ranking
  established by overfit100 (`projector + LoRA > LoRA-only`) is the
  reason the production config is the projector-mode variant.
- **Mode and data-format rankings from overfit100 are robust** because
  they are train-loss / training-set match measurements, not
  held-out eval — they do not depend on the eval pipeline that is
  being revalidated.

### LoRA rank capacity insight

The overfit100 results reveal a key distinction between **training
convergence** and **inference recall**:

- **r=8, alpha=8 (scaling=1.0)**: training loss converges to near-zero
  (0.0005), proving the model has learned the mapping during training.
  But at inference (save → reload → generate), ROUGE-L=0.073 and
  species_match=0% — the model cannot reproduce its training-time
  knowledge. The 8-rank bottleneck is too narrow to store enough
  information for faithful generation across 100 distinct plant
  descriptions.

- **r=256, alpha=8 (scaling=0.03)**: despite the suppressed scaling,
  the raw rank capacity is enough — ROUGE-L=0.898, species_match=93%.
  The 256-rank matrices can encode enough distinct image→text mappings
  to survive the save/reload roundtrip.

- **r=256, alpha=256 (scaling=1.0) + projector**: near-perfect —
  ROUGE-L=0.9998, species_match=100%. Full effective scaling plus
  projector tuning gives both capacity and expressiveness.

The takeaway: LoRA rank determines how much information the adapter can
*store and recall* at inference, not just whether training loss
converges. A low-rank adapter can memorize during training (the
optimizer finds a solution in the training-time computational graph)
but fail to reproduce that knowledge through the saved low-rank
matrices alone. This is because training benefits from the full
forward-pass context (optimizer state, activation patterns) that
doesn't survive serialization into just the A and B matrices.

---

## Package version fix (PEFT orphan-tensor bug)

All results prior to 2026-05-12 showing 0% species_match were caused
by a PEFT/transformers version incompatibility, not a fundamental
limitation of LoRA-only training.

### Root cause

`transformers 5.8.0` restructured Gemma 4 E2B's hybrid attention.
The model has 35 language-model decoder layers. Under the **new**
transformers, layers 0-14 use global attention (with separate
`q_proj`, `k_proj`, `v_proj`, `o_proj` nn.Linear modules) while
layers 15-34 use sliding-window attention that does **not expose
standalone `k_proj`/`v_proj` modules**. The older transformers had
all four projections in every layer.

An adapter trained on the old layout had 245 wrapped modules (490
LoRA tensors). On reload with the new layout, PEFT's
`from_pretrained` found only 205 matching modules (410 tensors) and
**silently dropped the remaining 80** — no warning, no error. The
model lost the `k_proj`/`v_proj` LoRA adaptations for layers 15-34
and reverted to base behavior on those layers.

### Fix

Update packages in order:

1. `pip install -U unsloth unsloth_zoo` (accept the pin-warning about
   transformers/peft moving past unsloth's tested set)
2. `pip install -U transformers peft trl accelerate`
3. Re-run `scripts/smoke_save_reload.py` to verify zero orphan tensors
4. **Retrain** — adapters trained on the old layout are incompatible

### Verified working versions

```
unsloth       2026.5.2
unsloth_zoo   2026.5.1
transformers  5.8.0
peft          0.19.1
trl           1.4.0
accelerate    1.13.0
torch         2.10.0+cu130
```

See [`../general/15-postmortems.md`](../general/15-postmortems.md) §1
for the full diagnostic session and
[`../general/14-package-versions-and-known-bugs.md`](../general/14-package-versions-and-known-bugs.md)
for the complete version snapshot with system details.

---

## What's deliberately not in scope (for this baseline mode)

- **Audio tower fine-tuning.** We don't use audio at all on iOS —
  audio_tower weights are stripped at deploy time. Frozen training is
  just a courtesy to anyone who wants to use this LoRA in a different
  context.
- **LoRA on the projector.** A single Linear `768 → 2048` (~1.18 M
  params) is small enough that full-param tuning is both cheaper than
  LoRA *and* more expressive. LoRA's `r*(in+out)` param budget on this
  Linear caps below the full param count for any reasonable rank.
  Full-param projector tuning lives in
  [`02-projector-mode.md`](02-projector-mode.md).
- **`train_on_responses_only` as a top-level function.** Can't traverse
  multimodal content blocks. We use the collator-level variant
  (`UnslothVisionDataCollator(train_on_responses_only=True, ...)`)
  instead, which masks after tokenization.
- **Dynamic `aspect_ratio_preserving_resize` at training time.**
  Skipped because mlx-swift-lm doesn't do it at inference. The
  pre-stretch at data-prep time forces both sides to agree on 960×672.

---

## Verification recipe

End-to-end smoke test after any change to the baseline finetune stack:

```bash
# 1. Unit tests — runs on Mac without CUDA
cd finetune
pytest                        # expect all green; freeze + config + data tests

# 2. Dry run on the actual config — validates data + chat template.
python -m src.finetune --config configs/plantnet-50k-baseline-v2.yaml --dry-run

# 3. Tiny real run on the GPU box — 50 steps, verifies freeze + LoRA.
python -m src.finetune --config configs/default.yaml \
    --max_steps 50 --max_train_samples 200

# 3b. Verify metrics.jsonl was written and is parseable.
#     A 50-step run with logging_steps=10 should produce ~5 train lines.
python -c "
import json, pathlib
p = next(pathlib.Path('outputs').glob('*/metrics.jsonl'))
lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
assert any(r['kind']=='train' for r in lines), 'no train rows in metrics.jsonl'
print(f'{p}: {len(lines)} log rows, kinds={set(r[\"kind\"] for r in lines)}')
"

# 4. Export the resulting tiny adapter end-to-end
bash scripts/export.sh outputs/<run-name>/final-adapter exports/smoke 4

# 5. Check the export passed the three tripwires (look for these log lines):
#    "Patched processor_config.json for mlx-swift-lm: do_normalize, ..."
#    (and absence of "ERROR: Loaded base model has no vision_tower")
#    (and absence of "ERROR: Converted MLX directory contains NO vision_tower")
```

For the projector and vision-tower modes' verification recipes, see
their respective docs.

---

## File references — baseline pipeline

| Concern | Path |
|---|---|
| Training entry point | `src/finetune/src/finetune.py` |
| Config dataclasses + validator | `src/finetune/src/config.py` |
| JSONL → unsloth `messages` adapter | `src/finetune/src/data.py:45-103` |
| Vision/audio freeze walker (LoRA-only) | `src/finetune/src/freeze.py:92-162` |
| Freeze tripwire (`assert_frozen` + allowlist) | `src/finetune/src/freeze.py:245-285` |
| SFTTrainer entropy patch | `src/finetune/src/finetune.py:_patch_sft_trainer_entropy` |
| Per-run metrics callback (JSONL loss curve) | `src/finetune/src/metrics_callback.py` |
| Metrics callback unit tests | `src/finetune/tests/test_metrics_callback.py` |
| Wandb env priming (project + run_name) | `src/finetune/src/finetune.py` (in `real_train`) |
| PlantNet → JSONL converter | `src/finetune/src/prepare_plantnet.py` |
| Pre-resize logic | `src/finetune/src/prepare_plantnet.py:199-238` |
| Trained vision shape constant (data prep) | `src/finetune/src/prepare_plantnet.py:158` |
| Trained vision shape constant (export) | `src/finetune/src/export_mlx.py:295` |
| Trained vision shape constant (iOS fetch) | `scripts/fetch-gemma.sh:87` |
| Adapter merge | `src/finetune/src/export_mlx.py:193-260` |
| Merge tripwire (vision tower present) | `src/finetune/src/export_mlx.py:263-272` |
| MLX conversion | `src/finetune/src/export_mlx.py:298-376` |
| MLX conversion tripwire | `src/finetune/src/export_mlx.py:379-400` |
| processor_config.json patch | `src/finetune/src/export_mlx.py:403-456` |
| Audio strip | `src/finetune/src/export_mlx.py:87-185` |
| Smoke-test config | `src/finetune/configs/default.yaml` |
| 50K baseline config | `src/finetune/configs/plantnet-50k-baseline-v2.yaml` |
| Train launcher | `src/finetune/scripts/run/train.sh` |
| Export launcher | `src/finetune/scripts/run/export.sh` |
| End-to-end automation | `src/finetune/scripts/run_plantnet_50k.sh` |
| Freeze unit tests (PEFT wrapping variants) | `src/finetune/tests/test_freeze.py` |
| Save/reload smoke test | `src/finetune/scripts/inspect/save_reload.py` |
| Overfit debug log | [`../general/15-postmortems.md`](../general/15-postmortems.md) §1 |
| Tested package versions | [`../general/14-package-versions-and-known-bugs.md`](../general/14-package-versions-and-known-bugs.md) |
| Companion: inference-side input bug | [`../general/13-mlx-vision-input-parity.md`](../general/13-mlx-vision-input-parity.md) |

---

## Related upstream

- **unsloth**: a `finetune_audio_layers` flag would make the explicit
  freeze walker redundant for the audio side. Tracked informally in
  the notebook commentary; no upstream issue filed yet.
- **trl**: the entropy-from-logits crash on lazy-callable logits is a
  unsloth/trl interface mismatch. Either side could be patched —
  unsloth could materialize logits on `.__getitem__`, or trl could
  type-check before slicing. We patch trl-side via the
  `_patch_sft_trainer_entropy` shim because it's local to our code.
- **mlx_vlm**: the only blocker for "single-command export" is that
  `mlx_vlm.convert` doesn't auto-patch `processor_config.json` for
  `mlx-swift-lm`'s decoder shape.
- **peft**: `from_pretrained` silently drops tensors that don't match
  the current model architecture. This caused the orphan-tensor bug
  described above. A warning or error on unmatched saved tensors would
  have caught this immediately.
