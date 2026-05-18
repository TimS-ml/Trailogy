# Why bnb NF4 is catastrophic on our SFT'd model

## TL;DR

`bnb_nf4` on the SFT'd Gemma 4 E2B drops PlantNet species_match from
**70.6 % → 0.1 %** (2 / 2,870 correct). WikiText PPL **2,873 → 4,086**
(1.42×, still under the 2× tripwire). The **language modeling barely
degrades, but visual recognition collapses completely**.

A three-variant ablation (`skip_ev` / `skip_vt` / `skip_both`,
n = 300 PlantNet) isolates the cause:

| Variant | skip list | PlantNet n=300 | ROUGE-L mean/median |
|---|---|---|---|
| `bnb_nf4` (baseline) | – | ~0.1 %* | 0.238 / 0.231 |
| `bnb_nf4_skip_ev` | `embed_vision` | **0.00 %** (0/300) | 0.229 / 0.214 |
| `bnb_nf4_skip_vt` | `vision_tower` | **67.67 %** (203/300) | 0.689 / 0.863 |
| `bnb_nf4_skip_both` | `vision_tower` + `embed_vision` | **69.33 %** (208/300) | 0.707 / 0.872 |
| `gptq_w4g128_da=1` (ref) | LM only | 68.8 % (full) | 0.699 / 0.863 |
| `bf16` reference | nothing | 70.6 % (full) | 0.711 / 0.867 |

*baseline number from laptop full-val run, n=2,870.

The conclusion is unambiguous: **the SigLIP vision tower is the sole
non-negotiable**. Once it stays bf16, accuracy recovers to GPTQ-level
regardless of whether the projector (`embed_vision`) is NF4'd. Skipping
the projector while keeping the vision tower NF4 does nothing. The
1.66-pp gap between `skip_vt` and `skip_both` is within the 95 % CI
for a difference of proportions at n = 300 (~±5 pp).

**Keep `vision_tower` bf16. Everything else is up for grabs.**

## Numbers — original 2-variant evidence (n=2,870 each)

PlantNet val, n = 2,870. WikiText-103, n = 200 × 512 tokens.

| Variant | Vision tower | LM | PlantNet match | WikiText PPL |
|---|---|---|---|---|
| bf16 reference | bf16 | bf16 | **70.6 %** (2027/2870) | 2,873 |
| GPTQ w4g128 da=0 | **bf16** (kept) | 4-bit | 68.4 % (1962/2870) | 3,149 |
| GPTQ w4g128 da=1 | **bf16** (kept) | 4-bit | 68.8 % (1975/2870) | 2,932 |
| **bnb NF4** | **4-bit (NF4)** | 4-bit (NF4) | **0.1 %** (2/2870) | 4,086 |

Initial conclusion from those two data points alone (GPTQ vs bnb): the
collapse is caused by quantizing at least one of `{vision_tower,
audio_tower, embed_vision, embed_audio}` — i.e. the four submodules
where bnb's scope differs from GPTQ's. Audio sub-towers are unused in
the PlantNet forward path so they were almost certainly not the
culprit, but the original 2-variant evidence could not isolate
`vision_tower` from `embed_vision` (the multimodal projector).

The WikiText delta hints at "vision-side has additional damage": NF4
PPL is 1.42× bf16 (moderate LM-side hit), GPTQ da=1 is 1.02× bf16.
The PlantNet collapse to 0.1 % is far larger than what a 1.42× PPL
could explain, so something visual must also be broken.

## Ablation — isolating vision_tower vs embed_vision (n=300)

Run on a 4090 CUDA host (PlantNet val.jsonl, `--plantnet_n 300`,
deterministic temperature):

| Variant | skip list (logical) | resolved bnb skip | PlantNet | ROUGE mean/med | wall (s) |
|---|---|---|---|---|---|
| `bnb_nf4_skip_ev` | `embed_vision` | `model.embed_vision, lm_head` | **0.00 %** (0/300) | 0.229 / 0.214 | 1022 |
| `bnb_nf4_skip_vt` | `vision_tower` | `model.vision_tower, lm_head` | **67.67 %** (203/300) | 0.689 / 0.863 | 1115 |
| `bnb_nf4_skip_both` | `vision_tower, embed_vision` | `model.vision_tower, model.embed_vision, lm_head` | **69.33 %** (208/300) | 0.707 / 0.872 | 1120 |

Read line-by-line:

- **`skip_ev`** (projector skipped, vision_tower still NF4'd) → still
  0 %. The projector is NOT the culprit. NF4'ing the projector alone
  while leaving the SigLIP encoder at NF4 doesn't help.
- **`skip_vt`** (vision_tower skipped, projector still NF4'd) →
  67.67 %, essentially recovering to GPTQ-level. The vision tower IS
  the culprit. NF4'ing the projector while keeping the vision tower
  at bf16 is essentially harmless.
- **`skip_both`** → 69.33 %, ~1.66 pp above `skip_vt`. That delta is
  inside the 95 % CI for a difference of proportions at n=300
  (~±5 pp), so it cannot be statistically distinguished from
  sampling noise. If a real effect exists, it is small — at most
  a few points on this task.

Together: **`vision_tower` bf16 is both necessary and sufficient.**

The pattern from the original 2-variant table is also clean:

- LM-only 4-bit (GPTQ): −2.3 / −1.8 pts → quantization is essentially
  free on this task.
- LM + vision_tower 4-bit (NF4): −70.5 pts → the vision tower is
  doing the heavy lifting.

The WikiText delta tells the same story from the language side: NF4
PPL is 1.42× bf16, GPTQ da=1 is 1.02× bf16. NF4's language
quantization is real but moderate; it is the vision tower that
destroys this model.

## What NF4 actually quantizes (scope analysis)

bitsandbytes' `BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_quant_type="nf4")` quantizes **every `nn.Linear` in the
model** that is not explicitly excluded via `llm_int8_skip_modules`.
We pass no skip-list (`bnb_nf4.py:78-84`), so the quantizer touches:

- `language_model.*` — 32 decoder layers × 9 Linears ≈ 288 modules
- `vision_tower.encoder.layers.*` — 26 SigLIP layers × 7 Linears (q/k/v/out + gate/up/down) ≈ 182 modules
- `vision_tower.patch_embedder.input_proj` — the patch projector. In
  Gemma 4's SigLIP variant this is implemented as `nn.Linear` (not
  Conv2d), so bnb DOES quantize it. Verified 2026-05-13:
  `model.vision_tower.patch_embedder.input_proj.weight` is U8 in the
  NF4 output.
- `embed_vision.embedding_projection` — the 1.2 M-param projector
- `audio_tower.*` Linears
- `embed_audio.*`

What stays bf16 in NF4:

- `Embedding` modules (token embeddings, position caches) — bnb does
  **not** wrap `nn.Embedding`. So `embed_tokens` (≈ 1.07 GB) stays bf16.
  Inside vision_tower this includes
  `patch_embedder.position_embedding_table` (verified bf16).
- `nn.LayerNorm` / `RMSNorm` weights and any biases — these are
  parameters but not `Linear`, so bnb skips them. The post-encoder
  LayerNorm and per-layer norms stay bf16.
- `Conv2d` would also stay bf16 (bnb's `replace_with_bnb_linear`
  doesn't touch convolutions), but Gemma 4's SigLIP doesn't use
  `Conv2d` for patch-embed — it's a Linear, and it gets NF4'd.

Net effect for our checkpoint (verified via
`inspect_quantized quantization/results/bnb_nf4`):
- 6.31 GB on disk (vs 9.51 GB bf16, vs 6.97 GB GPTQ-LM-only)
- `vision_tower` 104.8 MB at 9.35 bits/elt (vs 319.2 MB bf16 in GPTQ)
- `audio_tower` 150.4 MB at 8.02 bits/elt (vs 581.4 MB bf16 in GPTQ)
- Inside the SigLIP encoder, **every Linear is 4-bit**, including the
  patch embedder

## Why this is catastrophic for PlantNet specifically

PlantNet asks the model to identify a **species** from a single
photo. The species identity is a **fine-grained** visual judgment —
"is this leaf serrated like *Quercus rubra* or smooth like *Quercus
alba*". Two species often differ only in leaf-edge curvature or
flower-petal arrangement. The SigLIP encoder's job is to turn the
input image into a 280-token feature sequence that captures those
fine distinctions, so the projector + language model can name the
species.

When SigLIP's `Linear` layers are NF4-quantized:

1. **Layer scales degrade.** NF4 is data-free — every weight tensor
   gets quantized using a per-block (default 64-weight) statistic
   computed from the weights alone, with no awareness of the
   activation distribution. SigLIP's pretrained activations are
   tuned to the pretrained weights; the post-quant scales no longer
   match.
2. **Vision features collapse to coarse categories.** Empirically,
   the model still produces text (PPL only 1.42× bf16), but the
   visual conditioning is gone: every photo gets answered with one
   of a few "generic" plant names. The fine-grained discrimination
   that pushes a sample from "right genus" to "right species" is
   destroyed.
3. **The projector's calibration is broken.** Projector was trained
   on bf16 SigLIP outputs. NF4 SigLIP outputs are quantitatively
   different — the projector's full-param SFT pass doesn't know
   about that shift.

Note that the language side does NOT collapse. The model still
generates fluent botanical-sounding descriptions; it just attaches
them to the wrong species. WikiText PPL of 4,086 vs bf16 2,873 is
the language penalty (real but moderate); 0.1 % PlantNet is the
vision penalty (catastrophic).

## What this means for the deliverable

1. **Skip every variant that quantizes the vision tower.** The bnb
   NF4 row was useful — it identified the floor, with one data
   point — and we don't need a second.
2. **MLX VLM's default scope is correct for this task.** `mlx_vlm.convert`
   leaves `vision_tower.*` at bf16 by default and only quantizes the
   LM body + projectors. That matches GPTQ's scope, which has
   already been validated at 68–69 % PlantNet on our SFT.
3. **Any future "quantize the vision tower too" idea is now
   pre-rejected.** Downgraded to "don't try without QAT".
4. **QAT might still rescue vision-tower quantization.** Quantization-
   aware training (fake-quant fwd, bf16 grads) gives the model a
   chance to learn around the quantization noise. NF4 is the
   data-free PTQ floor; QAT is the corresponding ceiling. If the
   policy ruling allows QAT, the vision-tower-quant path becomes
   research-worthy. Until then, treat vision tower as bf16-only.

## Code: how to verify and reproduce

Three existing scripts can confirm this finding. All run on any host
(no CUDA / MLX required) by reading safetensors headers only. The
commands below were each run against `quantization/results/bnb_nf4`
and `gptq_w4g128_da0`; their real outputs are quoted where useful.

### 1. `scripts.inspect.quantized`

Per-submodule size + dtype histogram for a single model directory:

```bash
$PYTHON_BIN -m scripts.inspect.quantized \
    quantization/results/bnb_nf4

$PYTHON_BIN -m scripts.inspect.quantized \
    quantization/results/gptq_w4g128_da0
```

Look at the `Per submodule` block — `vision_tower` is ~0.32 GB in
GPTQ output, ~0.10 GB in NF4 output. The `Dtype histogram` shows the
SigLIP encoder weights as `U8` (NF4 storage) vs `BF16` (GPTQ keeps
it untouched).

Real `bnb_nf4` output:

```
Per submodule (safetensors only):
  language_model         6.03 GB   avg 13.92 bits/elt
  vision_tower          104.8 MB   avg  9.35 bits/elt
  audio_tower           150.4 MB   avg  8.02 bits/elt
  embed_vision          595.5 KB   avg  8.01 bits/elt
  embed_audio             1.2 MB   avg  8.01 bits/elt

Dtype histogram:
  F32           1575 tensors      1.1 MB   avg 32.00 bits/elt
  BF16          1426 tensors     5.16 GB   avg 16.00 bits/elt
  U8            1575 tensors     1.12 GB   avg  8.00 bits/elt
```

Real `gptq_w4g128_da0` output:

```
Per submodule (safetensors only):
  language_model         6.06 GB   avg 17.25 bits/elt
  vision_tower          319.2 MB   avg 16.00 bits/elt
  audio_tower           581.4 MB   avg 16.00 bits/elt
  embed_vision            2.2 MB   avg 16.00 bits/elt
  embed_audio             4.5 MB   avg 16.00 bits/elt

Dtype histogram:
  BF16          1676 tensors     6.04 GB   avg 16.00 bits/elt
  I32            825 tensors    898.0 MB   avg 32.00 bits/elt
  F16            275 tensors     27.8 MB   avg 16.00 bits/elt
```

(Note: GPTQ's I32 tensors are the packed 4-bit weights — GPTQ
stores 8 nibbles per int32 — and F16 tensors are the per-group
scales. The fact that vision_tower is `avg 16.00 bits/elt` means the
GPTQ packing only touches `language_model.*`, not the towers.)

### 2. `scripts.inspect.compare_sizes`

Side-by-side per-submodule diff between two model dirs:

```bash
$PYTHON_BIN -m scripts.inspect.compare_sizes \
    quantization/results/bnb_nf4 \
    quantization/results/gptq_w4g128_da0 \
    --label_a bnb_nf4 --label_b gptq_w4g128_da0
```

This is the single fastest way to see "where the 700 MB delta lives"
across two recipes. Real output:

```
  submodule                  bnb_nf4  gptq_w4g128_da0           delta
  ----------------------------------------------------------------
  audio_tower               150.4 MB        581.4 MB  +     431.0 MB
  embed_audio                 1.2 MB          4.5 MB  +       3.3 MB
  embed_vision              595.5 KB          2.2 MB  +       1.7 MB
  language_model             6.03 GB         6.06 GB  +      28.3 MB
  other                          0 B             0 B  +          0 B
  vision_tower              104.8 MB        319.2 MB  +     214.4 MB
  ----------------------------------------------------------------
  TOTAL                      6.31 GB         6.97 GB  +     678.8 MB
```

The 678.8 MB delta is almost entirely the audio_tower (+431 MB) and
vision_tower (+214 MB) being kept bf16 in GPTQ. Note that the
language_model is *bigger* in GPTQ (+28 MB) because GPTQ keeps
`embed_tokens` and `lm_head` bf16 while bnb quantizes nothing of
those (Embedding) — but the packed-int32 scales in GPTQ add slightly
more bytes than NF4's packed-uint8 + F32 scales.

### 3. `src.common.safetensors_io`

For a finer-grained ad-hoc query (e.g. "list every tensor whose name
contains `vision_tower` and is not bf16"):

```python
from pathlib import Path
from src.common.safetensors_io import enumerate_directory

for t in enumerate_directory(Path("quantization/results/bnb_nf4")):
    if "vision_tower" in t.name and t.dtype != "BF16":
        print(f"{t.dtype:<8} {t.bits_per_element:>5.1f} {t.name}")
```

### 4. `scripts.inspect.vision_dtype` (tripwire)

A single-purpose tripwire-style script that asserts the invariant
"`vision_tower.*` must be bf16 (or fp32/fp16)" AND "`vision_tower`
total on-disk bytes must be within a plausible range". Exits non-zero
on any violation. Designed as a CI hook before pushing any merged dir
to HF or the iOS bundle:

```bash
$PYTHON_BIN -m scripts.inspect.vision_dtype \
    quantization/results/<variant>
# exit 0  → vision tower is bf16, sized correctly, safe to ship
# exit 1  → vision tower has quantized tensors OR wrong size,
#           refusing to ship
# exit 2  → not a directory
```

Two guards:
1. **Dtype guard**: every tensor under `vision_tower.*` must be in
   `{BF16, F32, F16}`. Catches NF4-style U8 packing and GPTQ-style
   I32 packing inside the tower.
2. **Size guard**: total bytes under `vision_tower.*` must be within
   `[250 MB, 400 MB]` (Gemma 4 E2B's SigLIP tower is ~319 MB). Catches
   the "keys exist but were silently stripped to zero-sized stubs"
   case and the "accidental fp32 upcast" case. Override via
   `--min_bytes` / `--max_bytes` for other base models.

Real verification (all six dirs we have):

| variant | exit | message |
|---|---|---|
| `_merged_bf16` | 0 | OK: 658 tensors, 334.7 MB on disk |
| `gptq_w4g128_da0` | 0 | OK: 658 tensors, 334.7 MB on disk |
| `gptq_w4g128_da1` | 0 | OK: 658 tensors, 334.7 MB on disk |
| `gptq_w4g64_da0` | 0 | OK: 658 tensors, 334.7 MB on disk |
| `gptq_w4g128_lmhead` | 0 | OK: 658 tensors, 334.7 MB on disk |
| `bnb_nf4` | 1 | FAIL: 339 vision_tower.* tensor(s) in disallowed dtype(s) (U8) |
| `bf16_reference` | 1 | FAIL: no vision_tower.* (this dir holds eval outputs only, weights live in `_merged_bf16/`) |

The two `bf16_reference` and `bnb_nf4` failures are exactly the cases
the tripwire is designed to catch: an empty eval-only dir mistakenly
treated as a shippable bundle, and a vision-quantized variant.

Tests live in `quantization/tests/test_inspect_vision_dtype.py`
(9 tests, all pass: bf16/fp32 happy path, U8/I32 failure, missing
tower, too-small/too-large bounds, non-dir input, custom-bound
override).

### 5. Direct verification (ad-hoc one-liner)

The fastest "does this NF4 dir actually have a quantized vision
tower" check, no script required:

```bash
$PYTHON_BIN -c "
from src.common.safetensors_io import enumerate_directory
from collections import Counter
import sys
from pathlib import Path

d = Path(sys.argv[1])
dtypes = Counter()
for t in enumerate_directory(d):
    if 'vision_tower' in t.name:
        dtypes[t.dtype] += 1
print(f'vision_tower dtype histogram for {d}:')
for dt, n in dtypes.most_common():
    print(f'  {dt}: {n} tensors')
" quantization/results/bnb_nf4
```

Verified output for `bnb_nf4`:

```
vision_tower dtype histogram for quantization/results/bnb_nf4:
  BF16: 545 tensors
  F32:  339 tensors
  U8:   339 tensors
```

The 339 U8 tensors are the NF4-packed Linear weights (one per
quantized Linear); the 339 F32 tensors are the `absmax` /
`nested_absmax` / `quant_map` scale arrays bnb stores alongside each
packed weight. The 545 BF16 tensors are LayerNorm weights, biases,
the `position_embedding_table` Embedding, and pre/post-encoder norms.

For `gptq_w4g128_da0` (same command):

```
vision_tower dtype histogram for quantization/results/gptq_w4g128_da0:
  BF16: 658 tensors
```

Only BF16 — GPTQ left the vision tower entirely untouched, as
intended.

For `_merged_bf16`: only `BF16` tensors (untested but trivially true).

## Open questions

1. ~~**Is the SigLIP patch embedder Conv2d quantized by bnb?**~~
   **Resolved.** Gemma 4's SigLIP variant does NOT use a
   Conv2d patch embedder — it uses `patch_embedder.input_proj` which
   is `nn.Linear`. bnb therefore DID quantize the patch front-end
   to NF4 (verified U8 weight + F32 scales for
   `model.vision_tower.patch_embedder.input_proj.weight`). The
   `position_embedding_table` next to it stays bf16 because it is
   `nn.Embedding`. So in our NF4 output, every Linear in the visual
   front-end and every Linear in the SigLIP encoder is 4-bit; only
   norms and the position-embedding lookup survive. This sharpens
   the failure-mode story: there is no bf16 "anchor" anywhere in
   the visual signal path between pixels and the projector.
2. ~~**Would `llm_int8_skip_modules=["vision_tower"]` rescue NF4?**~~
   **Resolved.** Yes — `bnb_nf4_skip_vt` lands at 67.67 %
   PlantNet on n=300, within ~1 pp of GPTQ w4g128 da=1 (68.8 % on
   full 2,870). The recipe is **NF4 the language model + audio
   tower + audio embed, keep `vision_tower` (and tied `lm_head`) at
   bf16**. Still not iOS-deployable (no MLX kernel for NF4) but is
   now the cheapest "what does QLoRA-era 4-bit cost when scoped
   correctly" reference. See the ablation table in this doc.
3. **Does the result generalize beyond PlantNet?** VQAv2 is 0 % by
   design on the SFT'd model, so we can't use it as a vision-only
   probe. A separate "vision-only" benchmark (e.g. CIFAR-100 zero-shot
   via the SigLIP head, or VQAv2 against the *un-*SFT'd base) would
   confirm the SigLIP tower is the bottleneck. Out of deadline scope
   but a useful tech-report addendum.

## Implementation gotchas hit during the ablation

The ablation surfaced two real bnb + transformers + tied-embedding
bugs that we now guard against in `quantization/src/methods/bnb_nf4.py`.
Both are silent failure modes — the quant pipeline reports "OK", the
checkpoint saves cleanly, and the model only fails at inference time
(or worse, produces wrong-but-fluent text that looks superficially
fine). Each ate one full ablation run before being caught.

### 1. `lm_head` round-trip drops `quant_state` for tied-embedding models

Gemma 4 has `tie_word_embeddings = True` — `lm_head.weight` and
`model.embed_tokens.weight` are the same tensor. When bnb's
`replace_with_bnb_linear` is invoked with `llm_int8_skip_modules =
None`, transformers auto-adds `lm_head` to the skip list. But the
moment the caller supplies their own skip list (e.g.
`["embed_vision"]`), the auto-add is suppressed and `lm_head` gets
NF4'd. Subsequent `save_pretrained → from_pretrained` round-trip
loses `lm_head.weight.quant_state`, and the reloaded model emits
`UserWarning: FP4 quantization state not initialized` on every
forward, followed by garbage logits. The eval log fills with
"Inference failed on sample N: " (empty exception message), wall
time is ~180 ms / sample (just the failing prefill), and the result
JSON shows 0 % PlantNet with ROUGE 0.

**Fix in `BnBNF4Config`**: any caller-supplied `skip_modules` list
gets `lm_head` appended automatically (no-op for the default-None
case where transformers handles it). Tests:
`test_build_bnb_config_forwards_skip_modules_and_adds_lm_head`,
`test_build_bnb_config_does_not_duplicate_lm_head`.

### 2. `llm_int8_skip_modules` is anchored-prefix match, not substring

transformers' `should_convert_module`
(`transformers/quantizers/quantizers_utils.py`) checks each pattern
against the full dotted module name with:

```python
re.match(f"{key}\\.", full_name) or
re.match(f"{key}", full_name) or
full_name.endswith(key)
```

`re.match` only matches at the **start** of the string. For a Gemma
4 `ForConditionalGeneration`, every Linear inside the vision tower
has a full path like `model.vision_tower.encoder.layers.0.q_proj` —
which starts with `model.`, not `vision_tower`. So passing
`llm_int8_skip_modules=["vision_tower"]` is a **silent no-op**: the
config.json correctly records the skip list, but every Linear under
`vision_tower` still gets NF4'd. The `endswith` clause catches some
"leaf attr" patterns like `"q_proj"` but not parent submodule
prefixes.

We found this when the tripwire (`inspect_vision_dtype`) flagged
`skip_both` as still having a U8 vision_tower despite the skip list.

**Fix in `BnBNF4Config`**: a small `_SKIP_NAME_MAP` translates
ergonomic logical names to the actual prefix paths transformers
will match:

```python
_SKIP_NAME_MAP = {
    "vision_tower":   "model.vision_tower",
    "embed_vision":   "model.embed_vision",
    "audio_tower":    "model.audio_tower",
    "embed_audio":    "model.embed_audio",
    "language_model": "model.language_model",
    "lm_head":        "lm_head",
}
```

Callers can still pass raw paths (anything not in the map passes
through unchanged). Tests: `test_skip_names_unknown_pass_through_
unchanged`, `test_resolve_skip_names_function`.

### Tripwire as the ground truth

These two gotchas wasted ~30 min of compute before we caught them.
The cheap defense going forward is the
`inspect_vision_dtype` tripwire (this doc, section 4): it reads
safetensors headers (~5 s, no GPU) and fails fast if `vision_tower`
is anything other than bf16/fp32. **Run it on every output dir
between quant and eval.** The ablation driver
(`/tmp/opencode/run_bnb_ablation.sh`) calls it in step 2/3.

## Pointers

- `quantization/src/methods/bnb_nf4.py` — `BnBNF4Config.skip_modules`
  with logical-name resolver + `lm_head` auto-add. 12 tests in
  `quantization/tests/test_bnb_nf4.py`.
- `quantization/scripts/inspect/vision_dtype.py` — vision-tower
  bf16-clean tripwire. 9 tests in
  `quantization/tests/test_inspect_vision_dtype.py`.
- `quantization/results/bnb_nf4/eval.json` — laptop result, n=2,870
  (the original 0.1 % collapse).
- `quantization/results/bnb_nf4_skip_{ev,vt,both}/eval.json` — the
  three ablation results, n=300.
- `00-quantization-report-pub.md` — public team report, summarizes
  this finding in the recommendation section.
- `B1-sft-results.md` — full per-variant HF/CUDA numbers
  (bf16, GPTQ, NF4).
- `B2-sft-results.md` — full per-variant MLX numbers
  (the iOS-deployable candidates).
