# MLX stack design — mlx-vlm is the substrate; mlx-lm only contributes (buggy) quant cores

## TL;DR

1. **All deployable artifacts must be in mlx-vlm format.** Doesn't
   matter which route produces them (A / B.1 / B.2) — iOS runs
   `mlx-swift-lm` which is bit-compatible with `mlx_vlm.convert` output
   only. There is no other deploy substrate for Gemma 4 today.
2. **mlx-lm's Gemma 4 model class is wrong.** `mlx_lm/models/gemma4.py`
   and `gemma4_text.py` diverge from the actual on-device runtime on
   four concrete points (RMSNorm, projection layers, KV-shared K/V,
   audio handling). **Never load Gemma 4 via `mlx_lm.load`** — for ANY
   route. The result won't match iOS forward-pass math.
3. **mlx-lm's quant cores are reusable but buggy.**
   `mlx_lm/quant/{gptq, awq, dwq, dynamic_quant}.py` operate on a
   generic `nn.Module` and don't care which library built the tree.
   They CAN be used via the "hybrid flow" (load with mlx-vlm, hand
   `model.language_model` to an mlx-lm `*_quantize` core, save with
   mlx-vlm). But each core has bugs on Gemma 4 today (GPTQ NaN, AWQ
   NYI, DWQ broadcast). That makes them inferior to HF GPTQModel for a
   shipped deliverable — hence the Route B.1 priority.

The clean separation:

| What | Library | OK to use on Gemma 4? |
|---|---|---|
| Model class / forward pass / `load()` | mlx-vlm | yes — required |
| Model class / forward pass / `load()` | mlx-lm | **NO** — broken |
| Quantization bit-packing kernel | mlx-lm (via mlx-vlm) | yes — shared between both libs |
| Calibration-driven PTQ algorithms | mlx-lm `*_quantize` cores | yes, via hybrid flow + workarounds |
| iOS inference | mlx-swift-lm (mlx-vlm twin) | yes — required |

## 1. mlx-vlm is the deployment substrate

Sources of truth that pin this:

- `mlx-community/gemma-4-e2b-it-4bit/README.md` (the canonical 4-bit
  reference checkpoint we benchmark against):
  > "This model was converted to MLX format from
  > `google/gemma-4-e2b-it` using mlx-vlm version **0.4.3**."
- `external/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift:7`:
  > "// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4"

The deploy chain:

```
<any route> ──▶ <mlx-vlm-format checkpoint> ──▶ mlx-swift-lm ──▶ iOS
                ▲                                ▲
                │                                │
                └─ produced by mlx_vlm.convert  └─ Swift twin of mlx-vlm
                   (or by writing through         (constructs the SAME
                    mlx_vlm.utils.save_*)         model tree byte-for-byte)
```

This is true even for routes that don't *start* in mlx-vlm:

- **Route A** (4-bit SFT, QAT/QLoRA): training output is bf16+adapter.
  Merge to bf16, then `mlx_vlm.convert`.
- **Route B.1** (bf16 SFT → HF GPTQModel → MLX bridge): GPTQ output is
  HF format. The "bridge" step writes through mlx-vlm save path so the
  final artifact loads via `mlx_vlm.load`.
- **Route B.2** (bf16 SFT → MLX → mlx-lm quant): see hybrid flow in
  §4; ends with `mlx_vlm.utils.save_*`.

Every artifact that ships has to be `mlx_vlm.load`-compatible. That's
the universal end-state.

## 2. mlx-lm's Gemma 4 model class is wrong — never use it

`mlx_lm/models/gemma4.py` + `gemma4_text.py` implement a Gemma 4
class that diverges from `mlx_vlm/models/gemma4/language.py` (the
correct, iOS-matching one) on four concrete points:

| Layer | `mlx_lm.models.gemma4_text` | `mlx_vlm.models.gemma4.language` |
|---|---|---|
| `per_layer_model_projection` | `nn.Linear` (quantizable) | `ScaledLinear` (custom, **no `to_quantized` method**) |
| LM norm in projection path | `nn.RMSNorm` | `RMSNormZeroShift` (**no `+1` weight offset**) |
| KV-shared K/V (layers 15-34 in E2B) | `sanitize` drops them | model class allocates them (load fails on missing weights) |
| Audio handling | none | `gemma4.Model.__init__` reads `config.audio_config`; `load_model` does `config.setdefault("audio_config", {})` and **forces audio allocation** unless explicitly `null` |

Concretely:

- An `mlx_lm.quant.*` output uses mlx-lm's class layout during convert.
  When loaded via `mlx_vlm.load`, mlx-vlm constructs a *different*
  model tree and the weights don't fit. The RMSNorm-vs-RMSNormZeroShift
  mismatch alone is a math difference, not a renaming — patching keys
  doesn't help.
- An `mlx_vlm.convert` output uses mlx-vlm's class layout. Won't load
  in `mlx_lm.load` either (reverse direction).
- iOS Swift `Gemma4.swift` follows mlx-vlm exactly. The byte layout it
  expects is what `mlx_vlm.convert` (or `mlx_vlm.utils.save_*`) writes.

**Therefore: never use `mlx_lm.load` on a Gemma 4 checkpoint.** Even
if it's just "to check the file loads", you're checking against the
wrong model class and any result is misleading. The 2026-05-13
mac_mlx_lm round burned a day learning this — summarized in
[`../general/15-postmortems.md`](../general/15-postmortems.md) §2.

## 3. mlx-lm's quant cores ARE reusable (with caveats)

`mlx_lm/quant/{gptq, awq, dwq, dynamic_quant}.py` are the genuinely
interesting part of mlx-lm — they implement calibration-driven,
sensitivity-aware PTQ on top of the same flat-precision packer that
`mlx_vlm.convert` already uses. The `*_quantize()` core functions
take a generic `nn.Module` + calibration tensors:

```python
gptq_quantize(model, data, bits, group_size, fallback_bits, ...)
awq_quantize(model, inputs, awq_config, group_size, bits, ...)
dwq_quantize(...)
# dynamic_quant has no single callable — logic is inside main()
```

These DON'T care which library constructed the `nn.Module` tree —
they iterate `model.leaf_modules()` and replace `nn.Linear` /
`SwitchLinear` in place. So they're usable on a tree built by
`mlx_vlm.load` (the right tree) just as well as on one built by
`mlx_lm.load` (the wrong tree).

Status of each core on Gemma 4 (mlx-lm 0.31.3):

| Core | Status | Issue |
|---|---|---|
| `gptq` | runs, but produced NaN logits in mac_mlx_lm round | needs re-test under hybrid flow. Even when working, lacks gptqmodel's mature tricks (`desc_act`, dead-column, auto-clip, LQER) — see the missing-features discussion in [`B2-sft-results.md`](B2-sft-results.md) M4 v1 post-mortem |
| `awq` | convert fails immediately | `AWQ_MODEL_CONFIGS["gemma4"]` missing. ~5-line patch (copy `gemma3` entry, adjust for KV-shared layout) |
| `dwq` | convert fails | broadcast bug at `mlx_lm/quant/dwq.py:113` in the validation-loss path. Reproduces regardless of model tree |
| `dynamic_quant` | ran end-to-end | sensitivity-based bit allocation, not OBS-style PTQ. Different family. Useful as a research data point, not a direct replacement for HF GPTQModel |

This list is the **why-B.2-is-not-the-deliverable-priority** evidence.
The Apple research-quant toolbox is interesting but not production-grade
today. Either we patch the bugs and port the missing tricks (see the
M4 v1 post-mortem in [`B2-sft-results.md`](B2-sft-results.md)) or we
lean on the mature CUDA tools (Route B.1) for the actual deliverable.

## 4. Hybrid flow — mlx-vlm tree + mlx-lm quant core

The recipe for using an mlx-lm quant core without falling into mlx-lm's
broken model-class trap:

```python
# 1. Load using mlx-vlm so the model tree is mlx-vlm's (ScaledLinear,
#    RMSNormZeroShift, KV-shared layer layout, etc.). Weights are bf16.
from mlx_vlm import load
model, processor = load(bf16_dir)

# 2. Reach the language-model subtree. Vision/audio sibling subtrees
#    stay bf16 — the quant core never sees them.
lm = model.language_model

# 3. Call mlx-lm's *_quantize core on the subtree. It iterates leaf
#    modules and replaces nn.Linear / SwitchLinear in place. ScaledLinear
#    (in mlx-vlm's tree) is auto-skipped because it has no `to_quantized`.
from mlx_lm.quant.gptq import gptq_quantize
gptq_quantize(lm, calibration_data, bits=4, group_size=64,
              fallback_bits=6, fallback_group_size=64)

# 4. Save through mlx-vlm's save path so keys + config land in the
#    layout mlx_vlm.load expects.
from mlx_vlm.utils import save_weights, save_config
save_weights(out_path, model, donate_weights=True)
save_config(model_config, config_path=out_path / "config.json")
```

This is the foundation of Route B.2 — see
[`B2-sft-results.md`](B2-sft-results.md) for the measured outcomes.

Why this works where the 2026-05-13 mac_mlx_lm round didn't:

| Concern | mac_mlx_lm (2026-05-13) | Hybrid flow |
|---|---|---|
| Which Gemma 4 class does calibration forward run through? | mlx-lm's `gemma4_text` (subtly wrong: `nn.RMSNorm`, `nn.Linear` for `per_layer_model_projection`) | mlx-vlm's `gemma4.language` (the same class the on-device runtime uses) |
| Which Linear modules get quantized? | All `nn.Linear` in mlx-lm tree, including `per_layer_model_projection` | All `nn.Linear` in mlx-vlm tree; `ScaledLinear` is auto-skipped (no `to_quantized`) |
| KV-shared K/V handling | Sanitized away → output missing keys | Allocated in the tree → quantized like every other layer (or skipped via predicate) |
| Vision/audio modules | Stripped at convert time | Stay bf16 in the same checkpoint, ready for `mlx_vlm.load` |
| Loadable by mlx-swift-lm on iOS? | No | Yes (same byte layout as `mlx_vlm.convert` output) |

## 5. Knobs available through `mlx_vlm.convert`

The deploy-time substrate (Route A's final step, Route B.1's bridge
endpoint, Route B.2's save target) is `mlx_vlm.convert`. Its CLI
surface:

```bash
mlx_vlm.convert \
    --hf-path  <bf16-dir> \
    --mlx-path <out> \
    -q \
    --q-bits         {2, 3, 4, 6, 8}                       # flat-precision baseline
    --q-group-size   {32, 64, 128}
    --q-mode         {affine, mxfp4, nvfp4, mxfp8}
    --quant-predicate {mixed_2_6, mixed_3_4, mixed_3_5,
                       mixed_3_6, mixed_3_8, mixed_4_6,
                       mixed_4_8}                          # mixed-precision recipes
```

Internally, `mlx_vlm.convert` calls
`mlx_lm.utils.quantize_model(...)` at `mlx_vlm/convert.py:158-170` to
do the bit-packing — the kernel is shared with mlx-lm. The choice of
which modules get quantized goes through
`mlx_vlm.utils.skip_multimodal_module` (`mlx_vlm/utils.py:82-103`),
which:

- skips `vision_tower.*` / `audio_tower.*` by default
- does **NOT** skip `embed_vision.*` / `embed_audio.*` (Gemma 4's
  projectors). If the projector was SFT'd at full precision (e.g.
  `tune_projector: true`), supply a custom `quant_predicate` callable
  to also skip those — see
  `src/quantization/scripts/run/mlx_vlm_deploy_variant.py`
  for the reference recipe.

The mixed-precision recipes encode a fixed per-tensor policy:
`mixed_3_4` → most tensors at 3-bit, "use-more-bits" tensors at 4-bit
(specifically `v_proj` + `down_proj` in the first 1/8 + last 1/8 +
every-3rd middle layer, plus `lm_head` and `embed_tokens`). This is
functionally equivalent to what Unsloth's "UD" recipe does (see
`01-baselines.md`).

## 6. Recommended path by goal

### Goal: deploy ≤ 4 GB on iOS, fastest path

Stay inside `mlx_vlm.convert` flat or mixed-precision recipes. The
mlx-community 3.58 GB public reference (`mlx_vlm_g64` on the un-SFT'd
base) is the proof-of-concept. Per-variant numbers on our SFT in
`B2-sft-results.md`. No mlx-lm research-quant needed for this path.

### Goal: best deliverable quality

Route B.1 today: `bf16 SFT → HF GPTQModel (CUDA, mature tricks) →
MLX bridge`. The bridge is the only new engineering — write a function
that takes HF GPTQ output and produces an mlx_vlm.load-compatible
artifact preserving the GPTQ-quantized values. Spec TBW. The HF side
already lands 68-69 % match at n=2,870 (R2 in `B1-sft-results.md`).

### Goal: MLX-native research with deliverable quality

Route B.2 once the algorithm port lands. Use the hybrid flow above,
but only after porting at least `desc_act` (act-order) from
`gptqmodel` into `mlx_lm/quant/gptq.py`. Until then, B.2 is strictly
weaker than B.1 on quality, with the only upside being "no CUDA
dependency at quant time". See
[`B2-sft-results.md`](B2-sft-results.md) M4 / M8 for the current
status and the EoRA post-quant adapter that recovers most of the
quality drop training-free.

### Goal: pure research / sensitivity numbers, no iOS artifact

Pure-mlx-lm-flow is acceptable but **NOT** for iOS-deployable output:

1. Convert via mlx-lm — produces mlx-lm-format quant output.
2. Text-only PPL via `mlx_lm.evaluate` (this is what we did for
   `dynamic_quant` PPL = 511.7 in the mac_mlx_lm round).
3. For multimodal PlantNet eval, dequantize back to bf16 via
   `mlx_lm.utils.dequantize_model`, splice the bf16 vision from
   `_merged_bf16/`, run `scripts/run/eval.py --loader hf_bf16` on a CUDA box.

Caveat: the calibration forward pass in step 1 runs through mlx-lm's
**wrong Gemma 4 model class**, so the PPL numbers don't directly
predict iOS behavior. Useful only for relative cross-method comparison
in the same wrong frame.

## 7. File pointers

| Concept | Path |
|---|---|
| mlx-vlm Gemma 4 model class (canonical) | `mlx_vlm/models/gemma4/{gemma4, language, vision, audio}.py` |
| mlx-vlm convert entrypoint | `mlx_vlm/convert.py` (see `QUANT_RECIPES`, `mixed_quant_predicate_builder`) |
| mlx-vlm load entrypoint | `mlx_vlm/utils.py:load_model` (note `config.setdefault("audio_config", {})` at line 230) |
| mlx-lm Gemma 4 model class (BUGGY — do not use) | `mlx_lm/models/gemma4.py` + `gemma4_text.py` |
| mlx-lm quant cores (reusable via hybrid flow) | `mlx_lm/quant/{gptq, awq, dwq, dynamic_quant}.py` |
| iOS Swift VLM loader | `external/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift` |
| KV-shared sidecar injection (for bf16 loads) | `src/quantization/scripts/repair/prep_inject_kv_shared.py` |
| 2026-05-13 mac_mlx_lm post-mortem (full debug trail) | [`../general/15-postmortems.md`](../general/15-postmortems.md) §2 |
| Route picker | `00-quantization-roadmap.md` |
| Bridge / save / load compatibility lessons | `02b-mlx-torch-convert.md` |
