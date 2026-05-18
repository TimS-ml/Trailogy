# MLX ↔ PyTorch / HF transformers conversion — what we learned

## TL;DR

`transformers ≥ 5.8` + `mlx_vlm` 0.4.3 do not interoperate cleanly at
convert time without help:

1. `transformers 5.8`'s Gemma 4 attention class allocates `k_proj` /
   `v_proj` / `k_norm` / `v_norm` **only on the non-KV-shared
   layers** (E2B: 0-14). For layers 15-34, those modules don't exist
   as parameters at all.
2. `_keys_to_ignore_on_load_unexpected` adds the dead keys to a
   silent-ignore list on load — so loading a v5.5-era checkpoint that
   carries them works fine, but the parameters do not become model
   attributes.
3. Therefore `AutoModelForImageTextToText.save_pretrained()` after
   `merge_and_unload()` produces a safetensors that has NO K/V for
   layers 15-34 (HF only writes registered `nn.Parameter`s).
4. `mlx_vlm.convert` 0.4.3 constructs an mlx Gemma 4 model that
   ALWAYS allocates `k_proj` / `v_proj` / `k_norm` / `v_norm` for all
   35 layers (no `kv_shared_only` constructor switch — see the prior
   audit, mlx-vlm Python §). Its `load_weights` call is strict and
   errors with `Missing N parameters` listing those exact tensors.

Net effect: the standard "PEFT merge → mlx_vlm convert" recipe is
**broken end-to-end on the latest HF transformers**. It only worked
historically because v5.5 (and earlier) kept the K/V Linears
allocated everywhere, so save_pretrained wrote them, so mlx-vlm
found them. The bug is structural and silent: it shows up as a
convert-time failure with no hint that the cause is "your training
stack updated past the inflection point".

## Evidence chain

### What's in `unsloth/gemma-4-E2B-it` HF Hub (the bf16 source we train against)

```
$ ls <repo>/.../baseline2_qlora_bnb_nf4_skip_both.bf16-merged/
chat_template.jinja  config.json  generation_config.json
model.safetensors  processor_config.json  tokenizer_config.json
tokenizer.json
```

Inventory (counting `model.language_model.layers.{i}.self_attn.*` per
layer): **all 35 layers have q_proj, k_proj, v_proj, o_proj, q_norm,
k_norm weights present**.

```
Most-complete layer self_attn inventory: ['k_norm', 'k_proj', 'o_proj', 'q_norm', 'q_proj', 'v_proj']
All 35 layers have identical self_attn structure (no missing).
```

This checkpoint was made under the v5.5-era layout: every layer carries
its own K/V projection on disk, even though for layers 15-34 those
weights are functionally dead.

### What `transformers 5.8` does to those keys at load time

`transformers/models/gemma4/modular_gemma4.py:990` (verified by
inspection on the installed `transformers==5.8.0`):

```python
if not self.is_kv_shared_layer:
    self.k_norm = ...
    self.v_norm = ...
    self.k_proj = nn.Linear(...)
    self.v_proj = nn.Linear(...) if not use_alt else None
```

`is_kv_shared_layer = layer_idx >= num_hidden_layers - num_kv_shared_layers`.
For Gemma 4 E2B with `num_kv_shared_layers=20`, the gate is True for
layers 15-34 → those layers' attention objects have no `k_proj`
attribute at all.

Lines 1300-1305 then build `_keys_to_ignore_on_load_unexpected` for
exactly those (layer, key) pairs:

```python
self._keys_to_ignore_on_load_unexpected = []
for layer in self.model.layers:
    if layer.self_attn.is_kv_shared_layer:
        self._keys_to_ignore_on_load_unexpected.extend(
            [f"...layers.{layer.layer_idx}.self_attn.k_proj.weight",
             f"...layers.{layer.layer_idx}.self_attn.v_proj.weight",
             ...])
```

`from_pretrained` checks this list and silently drops the matching
keys from the unexpected-keys warning. **The weights are read from
the safetensors and then discarded into the void** — they have nowhere
to land in the model object.

### What `save_pretrained` writes back

`PreTrainedModel.save_pretrained` only serializes `self.state_dict()`,
which only contains the model's `nn.Parameter` and registered buffers.
Since layers 15-34's K/V projections are not allocated as Parameters,
they are NOT in `state_dict()`, and they are NOT in the output
safetensors.

Concrete diff against the baseline-2 merged dir
(`quantization/results/baseline2_qlora_bnb_nf4_skip_both.bf16-merged/model.safetensors`):

```
Layers with missing attn projections in the BASE:
  layer 15: missing ['k_proj.weight', 'v_proj.weight', ...]
  layer 16: missing ['k_norm.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight', 'q_proj.weight', ...]
  layer 17: ... [same]
  ...
  layer 34: ... [same]
```

(Layer 15 is the first KV-shared layer; layers 16-34 additionally
drop the `q_proj` / `o_proj` / `q_norm` ... entries because the
config-level Gemma 4 ClippableLinear wrapper attributes also live on
the dead `self_attn` and so they too don't become Parameters on the
v5.8 class. The total deficit vs the base checkpoint is 60 tensors —
matching mlx-vlm's error count.)

### What `mlx_vlm.convert` 0.4.3 does

`mlx_vlm/models/gemma4/language.py` constructs every Attention layer
unconditionally with `kv_shared_only=False`, so all 35 layers'
`Attention.__init__` allocates `k_proj`, `k_norm`, `v_proj`,
`v_norm`. The forward path correctly gates them behind
`if shared_kv is not None`, so they never fire at inference — but
the module objects exist, and `mx.load_weights(..., strict=True)`
fails with `Missing N parameters`.

`mlx_vlm/utils.py:323` calls `model.load_weights(list(weights.items()))`
WITHOUT explicitly setting `strict=False`. The default is strict.
That's what raises:

```
ValueError: Missing 60 parameters:
  language_model.model.layers.15.self_attn.k_norm.weight,
  language_model.model.layers.15.self_attn.k_proj.weight,
  language_model.model.layers.15.self_attn.v_proj.weight,
  ...
  language_model.model.layers.34.self_attn.v_proj.weight.
```

### Why `mlx-community/gemma-4-e2b-it-4bit` works on iPhone but our re-converted model would not

`mlx-community/gemma-4-e2b-it-4bit` was produced from a snapshot of
`google/gemma-4-e2b-it` that **still carried v5.5-era weight keys
on disk**. mlx-vlm convert saw them, loaded them into the dead
modules, and shipped them at ~7 MB / int4 of inert cost. On iPhone,
mlx-swift-lm follows the same pattern. Everyone is happy because
the disk format predates the v5.8 cleanup.

A future re-conversion path through HF transformers 5.8 erases those
keys — and that's exactly our merged dir today.

## Why this only surfaced now

| Era | Training stack | Convert input | Convert outcome |
|---|---|---|---|
| Pre-2025-12 (transformers ≤ 5.5) | unsloth + transformers 5.5 | safetensors retains dead K/V | mlx-vlm convert OK (dead K/V get loaded, never used) |
| 2026-05 (transformers 5.8 forced) | unsloth (pin overridden) + transformers 5.8 + peft 0.19 | save_pretrained writes only live params | mlx-vlm convert FAILS strict-load |
| Future (mlx-vlm gets `kv_shared_only` flag + `strict=False` on convert) | any | either layout | OK |

The mlx-vlm side already has a clean path (`kv_shared_only=True` would
make the missing tensors expected). The fix on that side is upstream
work; on our side we need a bridge that produces a save_pretrained
output that LOOKS like the v5.5-era layout (dead keys present).

## Fix strategies

### Strategy A — Tensor-level merge that preserves all base keys (PREFERRED)

Open base safetensors + adapter safetensors directly via the
`safetensors` library. For each LoRA pair `(lora_A, lora_B)`, find the
matching base tensor by name pattern, compute the delta
`(alpha / r) * B @ A`, add it in place to the base tensor, and write
the result to a new safetensors with the SAME key set as the base.
Plus the `modules_to_save` tensors (projector) just get copied over
the corresponding base tensor.

Properties:
- Never instantiates an HF model object → never goes through the v5.8
  KV-share gate → never drops dead K/V keys.
- Deterministic, dtype-explicit, no dependency on the trained-time
  vs convert-time model class agreeing.
- Handles base checkpoints with arbitrary "extra" tensors (calibration
  buffers like `input_max`/`input_min`/`output_max`/`output_min` that
  some v5.5-era distributions of Gemma 4 carry) transparently — they
  pass through untouched.

Limitations:
- Only handles standard LoRA + `modules_to_save` for the projector. No
  vision-tower layer tuning support without an extension. (We don't
  use that for baseline-2.)
- Trusts the adapter's `adapter_config.json` for `r`, `alpha`,
  `target_modules`. A misconfigured adapter would produce a wrong
  delta scale, just like merge_and_unload would.

This is the route to implement first.

### Strategy B — Post-hoc copy-back of missing tensors after `merge_and_unload`

After the existing HF-based merge, open the output safetensors and
copy the missing K/V (and other dropped) tensors from the original
base. Cheaper to implement (~50 LOC) but fragile: depends on
"which tensors HF dropped" being a stable function of the v5.8 gate,
which is technically implementation-defined and could change between
patch releases.

Acceptable as an interim band-aid; do NOT ship as the long-term path.

### Strategy C — Patch mlx-vlm to `strict=False` at convert time

The cleanest upstream fix, but requires getting changes into mlx-vlm
0.x+1, plus a `kv_shared_only` flag so the dead modules don't get
allocated at all. Out of scope for our current run.

### Strategy D — Stop going through HF entirely (full unsloth-free retrain)

Tracked separately in `A-*.md` (e.g. `A-baseline2-qlora-progress.md`)
in this dir. Doesn't fix the
existing baseline-2 adapter; would require re-running training. Useful
as an independent reference point, but the convert problem here is
about save / convert, not training, so this is the wrong layer to fix.

## Action items + verification

- [x] Implemented Strategy A as
  `src/quantization/scripts/repair/merge_safetensors.py`. Takes
  `--base` (HF repo id or local dir), `--adapter`, `--output`. Loads
  base monolithic or sharded safetensors via the `safetensors`
  library, applies LoRA deltas in fp32 (cast back to base dtype),
  replaces `modules_to_save` tensors directly, writes a monolithic
  output safetensors with the SAME key set as the base. Side-cars
  (processor / tokenizer / config) copied from adapter first, base
  as fallback.
- [x] Smoke / size check: 2011 base tensors went in → 2011 tensors
  come out. Output is **9.54 GB** (vs the HF-via-save_pretrained
  variant at 9.51 GB → the 0.03 GB delta is the recovered inert
  K/V tensors for layers 15-34). 205 LoRA deltas applied, 1
  modules_to_save (`embed_vision`) replaced. fp32-then-cast-back
  arithmetic matches the standard PEFT scale convention.
- [x] `mlx_vlm.convert -q --q-bits 4 --q-group-size 64 --q-mode affine`
  on the Strategy A output **succeeds**. Produces a 3.37 GB MLX-INT4
  directory with `2649` tensors — byte-for-byte the same tensor
  inventory as `mlx-community/gemma-4-e2b-it-4bit` (matching suffix
  counts: 1034 weight / 319 biases+scales / 232 each of input/output
  max/min / 35 layer_scalar / 12 per_dim_scale / 1 bias / 1
  position_embedding_table).
- [x] Sanity eval the bf16-merged dir at PlantNet n=200 (HF
  `hf_bf16` loader, fits on 4090): **67.50 %** species_match,
  ROUGE-L 0.6855. Within sample-noise margin of the prior bnb-NF4
  reading (69.50 % @ n=200) — 1σ at this n is ~3.2 pp, so the
  2 pp gap is not statistically meaningful. Confirms the merge is
  numerically reasonable (not a no-op, not garbage).
- [x] MLX runtime eval (mlx_vlm loader) on the Linux 4090 — UNBLOCKED.
  Root cause: `libmlx.so` is hard-linked against `libnvrtc.so.12`
  (the env-bundled `nvidia-cuda-nvrtc-cu12 12.9.86`), but the only
  CUDA headers on the host are CUDA 13 (system `/usr/local/cuda` is
  13.1; `nvidia-cuda-runtime-13.0.96` is also installed). CUDA 13's
  `cuda_fp6.hpp` / `cuda_fp4.hpp` introduce bare
  `__NV_SILENCE_DEPRECATION_BEGIN` lines at file scope that NVRTC
  12.9's preprocessor cannot expand (the macro is defined in
  `vector_types.h:113` but its definition does not reach those
  headers under NVRTC's pre-include order). Every gather / quant
  kernel JIT fails and inference returns all-zero arrays.
  Fix: stage matching CUDA 12.9 toolkit headers from the pip wheels
  `nvidia-cuda-runtime-cu12==12.9.79` + `nvidia-cuda-cccl-cu12==12.9.27`
  + `nvidia-cuda-nvcc-cu12==12.9.86`, merge their include trees, and
  point `CUDA_HOME` at the staged root. Idempotent helper:
  `src/quantization/scripts/_env/_mlx_env.sh` — sourcing it
  downloads the wheels once, stages them at
  `$MLX_ENV_ROOT/cuda12.9/`, and exports the right env vars. After
  staging, mlx-cuda compiles every kernel cleanly and inference
  produces reasonable plant descriptions on the 4090
  (~12 prompt-tps / ~115 gen-tps for 290-prompt-token + 44-gen-token
  image-text turns).
- [x] PlantNet n=200 eval landed (see results table below). The
  MLX-INT4 number is significantly worse than the bf16-merged
  reference at the same n (-45 pp species_match). This is genuine
  quantization-induced quality loss, not an eval bug: head-to-head
  on the same sample shows identical wrong outputs whether called
  through the eval loader or via the direct `mlx_vlm.generate(...)`
  API, with identical rendered prompts. Diagnostics in the next
  section. Reference comparison: the iOS-shipping
  `mlx-community/gemma-4-e2b-it-4bit` (same quant config:
  `group_size=64`, `bits=4`, `mode=affine`) emits empty / single
  newline / asterisk-spam on the same `"Describe this plant."`
  prompts — i.e. the QLoRA training **is** present in our quantized
  weights (trained answer style preserved), but fine-grained
  species discrimination collapses under 4-bit affine quant.
- [ ] PlantNet n=2870 full eval on the Strategy A bf16-merged dir
  to land a number directly comparable to baseline-1's bf16
  reference (70.63 %). At n=200 with ~3 pp noise floor we cannot
  resolve a real QLoRA vs bf16-SFT gap. Expected wall ~3 h on 4090.
- [ ] Improve the MLX-INT4 quality. The 22.5 % @ n=200 number above
  is the affine-mode floor with no calibration. Options, in order
  of expected impact:
  1. `mlx_vlm.convert --q-mode dynamic_quant --q-calibration-data <plantnet-train.jsonl>`
     — affine + per-tensor scale calibrated on the training
     distribution typically recovers 3-15 pp on fine-tuned VLMs.
  2. `--q-group-size 32` — half the group size, roughly half the
     quantization error per group; adds ~400-600 MB to the artifact
     (would push us toward but probably not past the 4 GB budget).
  3. Selective skip-modules: keep the trained `embed_vision`
     projector at bf16 (it's small — 2.2 MB — but is the layer the
     LoRA training tuned most heavily); skip-quantize the last few
     language decoder layers.
  4. 6-bit `--q-bits 6` — significantly bigger artifact (~4.5 GB),
     likely over budget but recovers most of the gap.

## Results table (all PlantNet val, same val.jsonl, eval_seed=0)

| Variant | Size GB | n | species_match | ROUGE-L mean |
|---|---:|---:|---:|---:|
| baseline-1 bf16 reference (LoRA → bf16 merge, no quant) | 9.51 | **2,870** | **70.63 %** | 0.7108 |
| baseline-1 bnb-NF4 skip vt+ev | 6.52 | 300 | 69.33 % | 0.7070 |
| baseline-2 QLoRA bnb-NF4 skip vt+ev (via *broken* HF-merge — missing 60 inert K/V) | 6.52 | 200 | 69.50 % | 0.7105 |
| baseline-2 QLoRA safetensors-merge bf16 (*correct*, all 2,011 keys) | 9.57 | 200 | 67.50 % | 0.6855 |
| baseline-2 QLoRA → MLX-INT4 g64 affine (Strategy A → mlx_vlm.convert) | **3.37** | 200 | 22.50 % | 0.3148 |

Engineering takeaway: end-to-end deploy path now produces a 3.37 GB
artifact at the iOS shipping shape. Convert-time blocker resolved,
runtime eval works on the 4090 (no Mac required).

Quality takeaway: 4-bit affine MLX quantization with no calibration
costs **−45 pp species_match** on this domain-specialized VLM (67.50 %
→ 22.50 % at n=200). The trained answer style survives but
fine-grained class discrimination does not. The iOS-shipping
`mlx-community/gemma-4-e2b-it-4bit` reference (same `bits=4` /
`group_size=64` / `mode=affine`) emits empty or whitespace-only
output on the same plant-description prompts — confirming the
adapter weights are present in our quantized model (the trained
"This appears to be X. Y is a species of …" template fires), but
the per-class features the LoRA learned are not robust to 4-bit
affine compression. Mitigation options listed in the action items
above; recommended next step is `dynamic_quant` with calibration on
the PlantNet training distribution.

## Why the -45 pp drop is real and not an eval bug

Confirmed by direct comparison:

- Eval-loader-rendered prompt: `'<bos><|turn>user\n<|image|>Describe this plant.<turn|>\n<|turn>model\n'`
- Direct-`apply_chat_template`-rendered prompt: byte-identical.
- Both paths produce identical wrong species for sample 0
  ("magic-lily" → "Common Marsh-mallow") and identical text content.

So this is not a prompt-template mismatch. The processor_config.json
top-level `size` was also patched to `{height: 960, width: 672}` via
`finetune.src.export_mlx.patch_processor_config_for_mlx_swift()` to
match the trained shape — no change in eval result.

The model genuinely identifies many plants incorrectly post-quant.
Some samples additionally degenerate into pad-token spam after the
first wrong species ("This appears to be X. Y is a species of …. Y
is a species of …. Y is a species of …. <pad><pad><pad>…" until
`max_tokens=128`). This is a known mode-collapse pattern under
heavy 4-bit affine quant: the entropy in the output distribution
collapses, the LM enters a repeat trap, and once it exhausts the
trained continuation it falls back to the pad-token argmax.

## Future-proofing

If `transformers` ever flips the `_keys_to_ignore_on_load_unexpected`
behavior back to "consume into a buffer" (unlikely), or if `mlx-vlm`
adds a `kv_shared_only=True` path (likely, upstream issue exists),
this whole bridge becomes a no-op. Strategy A is forward-compatible
in either direction.

Also: a v5.8-pure training stack (unsloth bypassed) produces an
adapter with EXACTLY the same target-module set as the unsloth path
under our version overrides (verified empirically — adapter LoRA only
covers the surviving v5.8 modules). So this convert-side bridge is
needed regardless of whether we keep or drop unsloth at training time.
