# Reference models — what we benchmark against

## TL;DR

- This doc audits public Gemma 4 E2B MLX checkpoints so custom quantization results can be compared against known external baselines.
- Only `mlx-community/gemma-4-e2b-it-4bit` is close to true 4-bit on its quantized portion, at 3.58 GB on disk.
- The UD-MLX variants use mixed 4/5/6/8-bit language-model weights, averaging about 6.1 bits per parameter rather than uniform 4-bit.
- `mlx_vlm.convert` leaves the vision and audio towers in bf16; readers should strip unused audio weights when trying to hit the 3.6 GB deployment target.

Three public 4-bit MLX quantizations of Gemma 4 E2B exist in the wild.
We need to reproduce at least one (matching `mlx-community`'s 3.58 GB
output to within tens of MB) before any custom quantization claim is
credible.

> **Key finding.** All three repos call themselves "4-bit", but only ref1
> actually averages ~4.5 bits/param on its quantized portion. ref2 and
> ref3 are mixed 4/5/6/8-bit with the LM averaging **~6.1 bits/param**.
> "UD-MLX-4bit" means "MLX mixed-precision with 4-bit as the floor",
> not a uniform 4-bit model. The ~970 MB gap between ref2 (HEAD) and
> ref3 (`9ee11f5`) is **not** a quantization-recipe difference — it's
> that `9ee11f5` is a text-only export with no vision/audio towers.

## Base model

`unsloth/gemma-4-E2B-it` — bf16 single safetensors shard.

- File size: **9.54 GB**
- Total params: **5.12 B**
- Per-submodule breakdown (measured 2026-05-12):

| Sub-module | Params | bf16 size |
|---|---|---|
| `language_model.*` | 4.65 B | 8.66 GB |
| `audio_tower.*` | 0.31 B | 0.57 GB |
| `vision_tower.*` (SigLIP) | 0.17 B | 0.31 GB |
| `embed_vision.*` (projector) | 1.2 M | ~3 MB |
| `embed_audio.*` | 1.6 M | ~3 MB |

"E2B" labeling: matformer-style nested-effective-2B architecture
(`hidden_size_per_layer_input: 256` in the config). The total is 5.1B
params, the active subset per forward pass is ~2B.

## Reference 1 — `mlx-community/gemma-4-e2b-it-4bit`

- Size: **3.58 GB** on disk (sha `99d9a53ff8`)
- Stack: `mlx_vlm.convert -q --q-bits 4` defaults (group_size=64, mode=affine)
- This is **our target**. Reproducing this number from the base bf16
  is gate `1a` in the project timeline.

**Verified (2026-05-13).** The `config.json` declares a flat global
4-bit / gs=64 with **zero per-tensor overrides**. But in practice only
**319 of 1034 weight tensors are actually quantized**:

| submodule | bytes (MB) | n layers | n quantized | eff. bits/param |
|---|---|---|---|---|
| `language_model` | 2634.4 | 565 | **317** | 4.54 |
| `audio_tower` | 609.6 | 258 | **0** (BF16) | 16.00 |
| `vision_tower` | 303.3 | 209 | **0** (BF16) | 16.00 |
| `embed_audio` (projector) | 1.3 | 1 | 1 | 4.50 |
| `embed_vision` (projector) | 0.7 | 1 | 1 | 4.50 |

So the original "open question" is answered: **`mlx_vlm.convert` quantizes
only the LM body + the two `embed_*` projectors**. It leaves the entire
`audio_tower` and `vision_tower` at BF16 (~913 MB combined). The 3.58 GB
number is **NOT** "uniform 4-bit across all weights" — it's "4-bit on
the LM + 16-bit on the towers".

Implication for our 3.6 GB target: if we strip the audio_tower (we don't
use audio for plant-id), we save ~610 MB immediately. The LM is already
under 2.7 GB at flat 4-bit.

## Reference 2 — `unsloth/gemma-4-E2B-it-UD-MLX-4bit` @ HEAD

- Size: **4.52 GB** on disk (sha `3236b6b700`)
- Stack: Unsloth Dynamic ("UD") mixed-precision MLX recipe
- `config.json` carries **317 per-tensor quantization overrides** on top
  of a 4-bit / gs=64 default

**Verified (2026-05-13) — the bit-width map of the LM.** All 317 over-
ridden tensors are in `language_model.*`:

| bits | n layers | params | bytes (MB) | eff bits/param |
|---|---|---|---|---|
| 4 | 63 | 1.02 B | 573.3 | 4.50 |
| 5 | 33 | 0.42 B | 292.0 | 5.50 |
| 6 | 11 | 2.86 B | 2327.6 | 6.50 |
| 8 | 210 | 0.32 B | 345.1 | 8.50 |

The LM averages **~6.14 bits/param** (NOT 4-bit). The bulk of the bytes
sit at 6-bit because the 11 big projection layers (the per-layer FFN
fanouts) are what dominate parameter count.

Per-submodule rollup, ref2:

| submodule | bytes (MB) | n quantized | eff bits/param |
|---|---|---|---|
| `language_model` | 3566.1 | 317 | 6.14 |
| `audio_tower` | 609.6 | 0 (BF16) | 16.00 |
| `vision_tower` | 303.3 | 0 (BF16) | 16.00 |
| `embed_audio` | 4.7 | 0 (BF16) | 16.00 |
| `embed_vision` | 2.4 | 0 (BF16) | 16.00 |

**Note vs ref1:** ref2 *leaves the two `embed_*` projectors at BF16*,
while ref1 quantizes them. ref2 puts the precision budget into the
attention path of the LM instead.

**Over the 4 GB ceiling.** Useful for understanding Unsloth's
philosophy (keep sensitive layers — attention `{q,k,v,o}_proj` and the
embedding — at higher precision), but doesn't satisfy our size target
as-is. Either replicate the older recipe at `9ee11f5` (Reference 3),
or strip multimodal towers we don't need.

## Reference 3 — `unsloth/gemma-4-E2B-it-UD-MLX-4bit` @ `9ee11f5`

- Size: **3.55 GB** on disk (sha `9ee11f5737`)
- 1236 tensors total (vs ref2's 2645) — see below

**Verified (2026-05-13) — `9ee11f5` is a text-only export.** It contains
**only the `language_model` submodule**. The `audio_tower`,
`vision_tower`, `embed_audio`, and `embed_vision` are entirely absent
from the safetensors. The LM bit-width map is virtually identical to
ref2:

| bits | n layers (ref3) | n layers (ref2) | params | bytes (MB) |
|---|---|---|---|---|
| 4 | 63 | 63 | 1.02 B | 573.3 |
| 5 | 33 | 33 | 0.42 B | 292.0 |
| 6 | 11 | 11 | 2.86 B | 2327.6 |
| 8 | **211** | **210** | 0.34 B | 359.7 |

So between `9ee11f5` and HEAD, Unsloth promoted **one** LM layer between
quantization tiers (the only LM-side difference) — and **added back the
multimodal towers**. The ~970 MB file-size gap reconciles exactly:

| submodule | ref2 − ref3 (MB) |
|---|---|
| `audio_tower` | 609.7 |
| `vision_tower` | 334.7 |
| `language_model` | 12.9 |
| `embed_audio` | 4.7 |
| `embed_vision` | 2.4 |
| **total** | **964.4** |

(matches the 964 MB reported file-size delta exactly.)

> Note: the `vision_tower` 334.7 MB above is slightly larger than the
> 303.3 MB shown in the per-submodule rollup for ref2 — the diff counts
> *all* tensor bytes (including non-Linear scalars like
> `position_embedding_table`, `layer_scalar`, `per_dim_scale`,
> calibration `input_max`/`output_min`), whereas the per-submodule
> rollup only sums `weight + scales + biases`. The full-tensor view is
> the one that has to reconcile against on-disk file size.

**Under the 4 GB ceiling.** But the only reason it fits is because it
drops the multimodal stack. For a multimodal target we'd need to
either re-attach BF16 towers (back to ~4.5 GB) or quantize the towers
ourselves (an unsupported configuration in stock `mlx_vlm.convert` —
the towers don't get a `QuantizedLinear` wrapper by default).

## What we DON'T have a reference for

- A 4-bit Gemma 4 E2B trained with QAT.
- A GPTQ Gemma 4 E2B with PlantNet-flavored calibration data.
- An AWQ Gemma 4 E2B.

Any of these we produce ourselves becomes a new data point in the eval
matrix.

## How to inspect a quantized safetensors file

The recipe (production version lives at
`src/quantization/scripts/inspect/quantized.py`):

- Pull only `config.json` and the safetensors **header** (a few hundred
  KB) via HTTP Range requests — **no weight download** required for
  remote inspection. For local files, read the safetensors header
  directly.
- Derive per-tensor `bits` from the shape ratio
  `weight.shape[-1] / scales.shape[-1] = group_size * bits / 32`.
- Build per-tensor, per-submodule, and per-bit-width rollups.
- Diff models layer-by-layer to surface scope differences.

### The shape trick (cheat-sheet)

MLX `QuantizedLinear` stores three tensors per logical layer:

```
<name>.weight   # packed U32, shape [out, in * bits / 32]
<name>.scales   # BF16,        shape [out, in / group_size]
<name>.biases   # BF16,        shape [out, in / group_size]
```

So once you know `group_size` (always 64 in these three repos),
`bits = 32 * weight.shape[-1] / scales.shape[-1] / group_size`.
A tensor with no sibling `.scales` is stored raw at its declared dtype.
