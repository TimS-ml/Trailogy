# Calibration data design — text / mixed-text / multimodal

## Scope

Any calibration-driven PTQ (GPTQ, AWQ, dynamic_quant) needs a small
data set to estimate per-layer activation statistics. This doc
specifies what that data is, how it's sampled, and how it's fed into
the quant pipeline regardless of route.

Out of scope:

- Data-free PTQ (`mlx_vlm.convert -q` flat affine, NF4) — no
  calibration step.
- Training data design — see `src/finetune/`.

## Research questions

Three calibration choices we want to ablate. Eval metric = PlantNet
val n=300 (quick test from `02-methods-and-eval.md`).

- **Q1 — multimodal vs text-only**: how much does feeding image+text
  pairs through the FULL vision-tower → embed_vision → language_model
  forward during calibration lift PlantNet match, vs feeding the LM
  text-only?
- **Q2 — domain-matched text vs general**: how much does using
  PlantNet train text (botanical descriptions) as a calibration source
  lift PlantNet match, vs WikiText-only?
- **Q3 — MLX-native vs HF**: does the calibration setup transfer
  cleanly between B.1 (HF GPTQModel + CUDA forward) and B.2 (hybrid
  flow + MLX forward) at the same calibration scope?

Q3 is the cross-route consistency check. Q1 and Q2 are the
calibration-design ablation; they're the same questions for either
route.

## Calibration sources

### Text-only

`load_text_calibration(processor, wikitext_samples, plantnet_samples,
plantnet_jsonl, sequence_length=2048, seed=0)` returns a tensor of
shape `(total_samples, sequence_length)`:

```python
def load_text_calibration(
    processor,
    wikitext_samples: int,
    plantnet_samples: int,
    plantnet_jsonl: Path,
    sequence_length: int = 2048,
    seed: int = 0,
):
    """Concatenate WikiText train and PlantNet train text into fixed-
    length token sequences. PlantNet text = conversation `content`
    fields with image references stripped. Tokenize via
    processor.tokenizer, pad with eos.

    Guards: reject val.jsonl paths (eval-leak), reject overfit100*
    paths (memorization set). See §"Eval-leak guards" below."""
```

Use cases:

- `wikitext_samples=512, plantnet_samples=0` → general-language only
  (baseline calibration).
- `wikitext_samples=256, plantnet_samples=256` → mixed-text
  (domain-matched + general anchor).
- `wikitext_samples=0, plantnet_samples=512` → domain-only (compare
  against mixed for Q2).

### Multimodal (image + text)

`load_multimodal_calibration(processor, plantnet_samples,
plantnet_jsonl, image_field="image", sequence_length=2048, seed=0)`
returns a list of dicts `{"input_ids": ..., "pixel_values": ...}`:

```python
def load_multimodal_calibration(
    processor,
    plantnet_samples: int,
    plantnet_jsonl: Path,
    image_field: str = "image",
    sequence_length: int = 2048,
    seed: int = 0,
):
    """Paired (image, text) calibration items. For each PlantNet sample:
      1. Load image at training resolution (960×672 portrait).
      2. Build the same chat-template prompt the iOS app uses.
      3. Run processor to get input_ids + pixel_values.

    Returns dict per sample. Eval-leak guards as above."""
```

Calibration items go through the FULL multimodal forward
(vision_tower → embed_vision → scatter → language_model layers) so the
language_model's Hessian / sensitivity collection sees activations
that look like inference traffic.

### Mixed multimodal + text anchor

When using the multimodal route, we typically combine 256 multimodal
items with 256 WikiText text-only items (wrapped as
`{"input_ids": tokens, "pixel_values": None}` so the driver skips the
vision forward for those). This keeps the LM-only path warm during
calibration and matches A/B sample counts at 512 total.

## Combined-mode sample counts (default)

To make A/B/C ablation results comparable, fix the total at **512
samples** across modes:

| Mode | WikiText | PlantNet text | PlantNet multimodal |
|---|---|---|---|
| `text_only_wiki` | 512 | 0 | 0 |
| `text_only_mixed` | 256 | 256 | 0 |
| `multimodal_mixed` | 256 | 0 | 256 |

Other splits are fine when you're not trying to isolate Q1/Q2 cleanly
— but document the deviation in the variant name.

## Eval-leak guards

`_reject_calibration_leak` and `_reject_overfit100` are mandatory
pre-flight checks in any calibration-data loader. They hard-fail
loudly on:

- Any path containing the substring `val.jsonl` (the eval split). GPTQ
  calibrating on val would trivially preserve eval scores while
  learning nothing transferable.
- Any path containing `overfit100`. Those sets have `train == eval` by
  construction; calibrating or evaluating on them is meaningless.

Implemented in `src/quantization/src/methods/gptq.py` for the
B.1 (HF) side; same check ported to the B.2 (MLX) calibration loaders.
Both load functions in this doc MUST call the guards as the first
operation.

## Route-specific adapters

The calibration items themselves are route-agnostic. What differs is
how they're fed into the quantizer.

### B.1 — HF GPTQModel

```python
from gptqmodel import GPTQModel, QuantizeConfig
# (existing wrapper in src/quantization/src/methods/gptq.py)

calib_items = load_text_calibration(
    processor=processor,
    wikitext_samples=256,
    plantnet_samples=256,
    plantnet_jsonl=PLANTNET_TRAIN_JSONL,
)
# calib_items is a tensor; GPTQModel accepts a list of dicts with
# "input_ids" and optional "attention_mask" — wrap accordingly.
gptq_model.quantize(calib_items, batch_size=1)
```

For multimodal calibration on the HF side, GPTQModel needs to be
driven through `Gemma4ForConditionalGeneration.forward(**batch)` with
`pixel_values` so the SigLIP tower runs and the projector produces the
multimodal tokens. This is a small wrapper around GPTQModel's hook
infrastructure; spec lands in the B.1 bridge doc when written.

### B.2 — hybrid flow (mlx_vlm.load + mlx_lm.quant)

```python
from mlx_vlm import load
from mlx_lm.quant.gptq import gptq_quantize

model, processor = load(bf16_dir)
calib = load_text_calibration(processor, wikitext_samples=512, ...)
# Or for multimodal:
calib_mm = load_multimodal_calibration(processor, plantnet_samples=256, ...)
calib = calib_mm + load_text_calibration(processor, wikitext_samples=256, plantnet_samples=0)

# text_only: feed model.language_model directly
target = model.language_model

# multimodal: wrap so the calibration forward goes through the FULL
# vision_tower → embed_vision → language_model chain. mlx_lm.quant
# hooks the language_model leaves either way; the wrapper just makes
# sure the activations on those leaves are vision-conditioned.
class MultimodalCalibrationDriver:
    def __init__(self, outer_model):
        self.outer = outer_model
    def __call__(self, batch):
        return self.outer(**batch)
    def leaf_modules(self):
        return self.outer.language_model.leaf_modules()

target = MultimodalCalibrationDriver(model)  # for multimodal mode

gptq_quantize(target, calib, bits=4, group_size=64,
              fallback_bits=6, fallback_group_size=64)
```

The wrapper class makes the `mlx_lm.quant.*_quantize` core treat the
outer multimodal model as a callable taking a multimodal batch, while
still exposing only `language_model.leaf_modules()` as the quantization
targets. Vision/audio modules are never touched.

## Validation

Same n=300 PlantNet quick test from `02-methods-and-eval.md`.
Comparison structure:

- **Q1 result**: (multimodal_mixed match) − (text_only_mixed match).
  Positive delta means multimodal calibration helps; near-zero means
  the iOS forward pass isn't sensitive to vision-context activations
  during inference even though calibration saw them.
- **Q2 result**: (text_only_mixed match) − (text_only_wiki match).
  Positive means domain-matched text calibration helps.
- **Q3 result**: |B.1 result − B.2 result| at the same calibration
  scope. Small delta means the calibration design transfers cleanly
  across routes.

Per the §3.2 framing in the old 08, on R2/R3 reference (~68-69 % at
n=2,870 on the older SFT) the expected band at n=300 is 65-70 %. On
the data-aug-enwiki SFT the bf16 ceiling is **85.7 %** (`M0` in
`B2-sft-results.md`, paper-grade test/ n=300 seed=0, re-tested
2026-05-15). The absolute numbers are therefore HIGHER than the B1
reference — the data-aug-enwiki SFT is a stronger model. The deltas
Q1/Q2/Q3 remain model-invariant, which is what we care about.

## Open risks

1. **Multimodal calibration wall time**. SigLIP forward on 256 images
   at 960×672 is ~50-200 ms/image (un-measured on M5 Pro; rough
   estimate). 13-50 s total. Acceptable. If worse, drop multimodal
   count to 128 + WikiText 128, keeping total 256 (and adjust A/B
   accordingly for fairness).
2. **PlantNet train.jsonl size**. The text-only path concatenates
   conversation `content` fields with image references stripped. Need
   to confirm the stripping doesn't leave dangling soft-token slots
   (it shouldn't — text-only mode skips the image-feature scatter).
3. **Q3 consistency** assumes HF GPTQModel's forward through
   `Gemma4ForConditionalGeneration` is numerically close to mlx-vlm's
   forward through its Gemma 4 model class. They're SUPPOSED to be
   the same math (mlx-vlm's `language.py` follows HF transformers'
   reference). Confirm with a small bf16 forward parity test on 8
   PlantNet samples before drawing Q3 conclusions.

## File pointers

| Concept | Path |
|---|---|
| Eval-leak guards (HF side) | `src/quantization/src/methods/gptq.py:_reject_calibration_leak` |
| Eval-leak guards (overfit) | `src/quantization/src/methods/gptq.py:_reject_overfit100` |
| PlantNet train JSONL | `src/finetune/data/english-desc/train.jsonl` |
| MLX quick-eval harness | `src/quantization/src/eval/plantnet.py` |
