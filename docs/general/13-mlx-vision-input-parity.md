# MLX Vision Input Parity — Gemma 4 E2B on `mlx-community` + `mlx-swift-lm`

A deep-dive into a quiet bug the iOS app has been silently working
around since the first VLM build. It surfaces when you try to fine-tune
the checkpoint and deploy: train and inference end up reading two
different input distributions. Understanding it is necessary before
signing off on the finetune → MLX export pipeline.

## TL;DR

| | What it should be | What `mlx-community` ships | What `mlx-swift-lm` does |
|---|---|---|---|
| Image resize policy | Aspect-ratio-preserving, up to 2520 patches with sides divisible by 48 | (no opinion — relies on whatever `processor_config.json` says) | **Fixed-size stretch** to whatever `size:` says |
| `processor_config.json:image_processor.size` | **Absent** (Google upstream) | **`{height: 224, width: 224}`** — incorrect | reads it as the resize target |
| Resulting patch grid | Variable, e.g. 60×42 or 56×42 | 14×14 = 196 patches | 14×14 = 196 patches |
| Vision pooler behavior | Trained kernel-3 stride pooler over 60×42 → 280 cleanly-pooled tokens | n/a | **Degenerates** at 14×14: kernel collapses to 1, output is 196 real tokens + 84 zero tokens |

Two bugs compound:

- **Bug A (data)** — the `mlx-community` `processor_config.json` adds
  a `size: 224×224` field that the upstream Google checkpoint doesn't
  have. This is wrong for the trained pooler.
- **Bug B (code)** — `mlx-swift-lm`'s `Gemma4Processor.preprocess`
  does fixed-size resize without aspect-ratio preservation and without
  the 2D `image_position_ids` that the HF reference produces. Even if
  Bug A is patched, the model still sees aspect-distorted inputs.

The iOS app currently sidesteps Bug A by force-rewriting the
`processor_config.json` to `size: 960×672` after every fetch
(`scripts/fetch-gemma.sh`). It does not address Bug B; the workaround
for Bug B is to pre-stretch training images to the same 960×672 shape
in `finetune/src/prepare_plantnet.py`.

## What Gemma 4 vision is supposed to be

Gemma 4 ships a SigLIP2-style **dynamic-resolution** vision encoder,
not a fixed-square one. The reference HF implementation defines:

```
patch_size              = 16
pooling_kernel_size     = 3
max_soft_tokens         = 280
position_embedding_size = 10240
```

From these, the processor computes:

```
max_patches = max_soft_tokens × pooling_kernel_size²
            = 280 × 9 = 2520
side_mult   = pooling_kernel_size × patch_size
            = 3 × 16 = 48
```

`Gemma4ImageProcessor.preprocess` then resizes **each input image
preserving aspect ratio** so that:

1. The resized total is at most 2520 patches,
2. Both sides are multiples of 48 (so the kernel-3 pool divides cleanly).

The processor returns **three** tensors per image:

| Name | Shape | Meaning |
|---|---|---|
| `pixel_values` | `(B, max_patches, 3·patch²) = (B, 2520, 768)` | Patchified, padded with zeros if image had fewer than 2520 patches |
| `image_position_ids` | `(B, 2520, 2)` | 2D `(x, y)` grid coordinates per patch; padded patches get `(-1, -1)` |
| `num_soft_tokens_per_image` | `(B,)` int | How many real (non-padded) soft tokens this image will produce after pooling |

The vision tower's `Gemma4VisionPatchEmbedder` looks up its 2D
position embedding by the `image_position_ids`. The position embedding
table is `position_embedding_size × hidden = 10240 × 768` specifically
so it can address arbitrary 2D grids.

The `Gemma4VisionPooler` then applies a kernel-3 stride pool across
the patch grid (each pooled token absorbs 3×3 = 9 raw patches),
producing `(num_patches_h / 3) × (num_patches_w / 3)` pooled "soft
tokens" which become the language-model image tokens between `<boi>`
and `<eoi>` in the prompt.

**Concrete example** — a 960×672 input:

```
patches_h = 960 / 16 = 60
patches_w = 672 / 16 = 42
total     = 60 × 42 = 2520 patches    (= max budget, exactly)
pooled_h  = 60 / 3  = 20
pooled_w  = 42 / 3  = 14
soft tok  = 20 × 14 = 280 tokens      (= image_seq_length, exactly)
```

This is the shape Gemma 4 was trained on. Other ratios work too as
long as the math lands on multiples of 48 and stays under 2520
patches.

## Bug A — `mlx-community` invented a `size: 224×224` field

The upstream **Google** checkpoint at
`google/gemma-4-e2b-it` ships a `processor_config.json` whose
`image_processor` block has **no `size` field at all**:

```json
"image_processor": {
  "image_processor_type": "Gemma4ImageProcessor",
  "image_seq_length": 280,
  "max_soft_tokens": 280,
  "patch_size": 16,
  "pooling_kernel_size": 3,
  "resample": 3,
  "rescale_factor": 0.00392156862745098
  // NO "size" — HF processor uses aspect-ratio-preserving resize
}
```

The `mlx-community` port at `mlx-community/gemma-4-e2b-it-4bit`
adds a `size: {height: 224, width: 224}` field that does not exist
upstream:

```json
"image_processor": {
  ...
  "size": { "height": 224, "width": 224 }   // ← mlx-community injected this
}
```

`224×224` is the SigLIP1 default — **not** the SigLIP2 / Gemma 4
default. It's likely a copy-paste from a generic vision-config template
during the conversion script. There is no code path in the trained
model that expects 224×224.

**Verification:**

```bash
curl -s https://huggingface.co/google/gemma-4-e2b-it/raw/main/processor_config.json \
    | jq '.image_processor | {size, max_soft_tokens, pooling_kernel_size}'
# {
#   "size": null,
#   "max_soft_tokens": 280,
#   "pooling_kernel_size": 3
# }

curl -s https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit/raw/main/processor_config.json \
    | jq '.image_processor | {size, max_soft_tokens, pooling_kernel_size}'
# {
#   "size": { "height": 224, "width": 224 },         ← spurious
#   "max_soft_tokens": 280,
#   "pooling_kernel_size": 3
# }
```

## Bug B — `mlx-swift-lm`'s `Gemma4Processor` is a fixed-stretch resampler

The HF reference processor outputs three tensors and dynamic patch
counts. `mlx-swift-lm`'s Swift port outputs **one** tensor and assumes
a regular grid:

```swift
// MLXVLM/Models/Gemma4.swift
public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws
    -> (MLXArray, THW)
{
    var userProcessing = processing ?? UserInput.Processing()
    let targetSize = config.fixedSize          // ← reads `size` from processor_config.json
    userProcessing.resize = targetSize
    let processedImages = images.map { image in
        let processedImage = MediaProcessing.apply(image, processing: userProcessing)
        let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
        let resizedImage = MediaProcessing.resampleBicubic(srgbImage, to: targetSize)  // fixed stretch
        ...
    }
    let pixelValues = concatenated(processedImages)
    return (pixelValues, THW(images.count, Int(targetSize.height), Int(targetSize.width)))
}
```

Three things are missing relative to HF:

1. **No aspect-ratio preservation** — `resampleBicubic` is a hard
   stretch to the configured shape. A 1024×768 landscape input gets
   squashed into a 672×960 portrait frame (or whatever you configured).
2. **No `image_position_ids` output** — the tower computes positions
   internally as a regular `(H/16, W/16)` grid, so the position
   embedding lookup is meaningful only when the input is itself a
   regular full-rectangle.
3. **No dynamic `num_soft_tokens_per_image`** — the prompt builder
   reserves a fixed `image_seq_length` (280) image tokens regardless of
   the image. HF would have computed that count per image based on the
   resized grid.

**Default fallback when `size` is absent**:

```swift
public var fixedSize: CGSize {
    if let size {
        return CGSize(width: size.width, height: size.height)
    }
    // 800x800 keeps the patch count under Gemma4's 280 * 3^2 vision budget.
    return CGSize(width: 800, height: 800)
}
```

`800` is not a multiple of `48`. `800/16 = 50` patches per side,
kernel-3 pool doesn't divide cleanly. The fallback is also wrong, just
less catastrophically.

## The math: why 224×224 produces zero-padded soft tokens

This is the part that quietly breaks recognition quality even when
nothing crashes.

At 224×224 with the trained `pooling_kernel_size = 3` and
`max_soft_tokens = 280`:

```
patches_h = 224 / 16 = 14
patches_w = 224 / 16 = 14
total     = 14 × 14 = 196 patches
```

`mlx-swift-lm`'s `Gemma4VisionPooler` derives the kernel size at
runtime from `validCount / output_length`:

```swift
let kernel = Int(sqrt(Double(max(1, validCount / max(length, 1)))))
//        = Int(sqrt(196 / 280))
//        = Int(sqrt(0))                    // integer division
//        = 1
let divisor = max(kernel * kernel, 1)       // = 1
let pooledLength = max(length, 1)           // = 280

var kernelIndices = actualPositions.asType(.int32)
kernelIndices = floor(kernelIndices.asType(.float32) / Float(kernel)).asType(.int32)
let flatKernel =
    kernelIndices[0..., 0]
    + MLXArray(Int32(max(maxX / max(kernel, 1), 1)))
    * kernelIndices[0..., 1]
let weights =
    gemma4OneHot(flatKernel, numClasses: pooledLength).asType(.float32)
    / Float(divisor)
let output = einsum(
    "lL,bld->bLd", weights, pooledHiddenStates[0..., ..<validCount, 0...]
)
```

Trace through with `kernel=1`:

- `kernel = 1` ⇒ no actual pooling. Each raw patch maps 1:1 to one
  output slot.
- `flatKernel[i] = x_i + 14 · y_i` for `i ∈ [0, 196)`. This produces
  unique slot indices `0..195`.
- `gemma4OneHot(flatKernel, numClasses=280)` gives a one-hot weight
  matrix of shape `(196, 280)` — only columns 0..195 ever receive a 1,
  columns 196..279 stay all-zero.
- `einsum` produces output of shape `(B, 280, hidden)` where the first
  196 vectors carry signal and the last 84 are zero vectors.

**The language model then sees 280 image tokens** between `<boi>` and
`<eoi>` — but **84 of them (30 %) are all-zero** vectors, which is
out-of-distribution for a model trained against 280 dense vectors from
a proper kernel-3 pool over 60×42 patches.

The model still produces text (image-token padding with zeros doesn't
crash anything), but the language conditioning is degraded relative to
training. Fine-tuned LoRA weights conditioned on this degraded
distribution is exactly the trap.

## Current workaround: `scripts/fetch-gemma.sh` rewrites `processor_config.json`

The iOS app ships a post-fetch patch script that overrides Bug A:

```bash
# scripts/fetch-gemma.sh (excerpt)
TRAINED_SIZE='{"height": 960, "width": 672}'
PCFG="$DEST/processor_config.json"
python3 - "$PCFG" "$TRAINED_SIZE" <<'PY'
import json, sys
p = sys.argv[1]
trained_size = json.loads(sys.argv[2])
with open(p) as f:
    cfg = json.load(f)
ip = cfg.get("image_processor", {})
patch = {
    "do_normalize": ip.get("do_normalize", False),
    "image_mean":   ip.get("image_mean",   [0.0, 0.0, 0.0]),
    "image_std":    ip.get("image_std",    [1.0, 1.0, 1.0]),
    "size":         trained_size,           # ← force 960×672
}
...
PY
```

What this fixes:

1. **Hoists** `do_normalize` / `image_mean` / `image_std` from the
   nested `image_processor` block to the top level —
   `mlx-swift-lm`'s `Gemma4ProcessorConfiguration` Codable decoder
   reads them from the top level only.
2. **Forces** `size: 960×672` so the now-degenerate pooler kernel
   comes out as `kernel = sqrt(2520/280) = 3`, exactly matching how
   the model was trained.

The architectural fields (`image_seq_length=280`,
`pooling_kernel_size=3`, `default_output_length=280`, `patch_size=16`)
are **deliberately left alone** — they are trained values.

After the patch:

```
patches_h = 960 / 16 = 60
patches_w = 672 / 16 = 42
total     = 2520 patches
kernel    = sqrt(2520 / 280) = 3
pooled    = 20 × 14 = 280 (no zero padding)
```

## What still doesn't work after the workaround — Bug B remains

The patch only fixes the **shape** that `mlx-swift-lm` reads. It does
**not** fix the fact that `mlx-swift-lm` is a fixed-stretch resampler
without aspect-ratio preservation.

Consequences:

| Input photo | What HF would do (training) | What iOS does (inference) |
|---|---|---|
| Portrait phone photo (3:4) | Resize to ≈672×960, dynamic 280 tokens | Stretch to 672×960. **Match.** |
| Square photo (1:1) | Resize to ≈768×768, kernel divides to 256 tokens | Stretch to 672×960 (vertical squish). Mismatch in shape. |
| Landscape photo (4:3) | Resize to ≈912×672 landscape, dynamic 266 tokens | Stretch to 672×960 portrait (90° distortion). Severe mismatch. |

Recognition quality drops most on landscape photos. The clean fix is
to port the Python reference's `aspect_ratio_preserving_resize` to
Swift — non-trivial but matches training distribution.

## Implications for the LoRA finetune pipeline

This is what makes the bug acute. The training stack uses HF's
`Gemma4ImageProcessor` (via unsloth), which **does** do aspect-ratio
preservation. The deploy stack uses `mlx-swift-lm`, which does not.
Same PlantNet image, two different visual feature distributions.

Without intervention, LoRA learns conditional on aspect-preserved
features and is deployed against aspect-stretched ones — the language
adapter sees inputs from outside its training distribution.

**Mitigation now in `finetune/src/prepare_plantnet.py`**: the prep
script defaults to `--resize_to 960x672` and pre-stretches every
PlantNet image to the iOS runtime shape, writing the resized copies
under `<output_dir>/images_resized/<split>/<sid>/`. The JSONL points at
those.

This guarantees that:

- HF processor at training sees a 960×672 input → its
  aspect-ratio-preserving path becomes a no-op → 60×42=2520 patches →
  20×14=280 pooled tokens.
- `mlx-swift-lm` at iOS inference also stretches to 960×672 → same
  60×42 patch grid → same 280 pooled tokens.

Train and deploy now read the same distribution, at the cost of
distorting non-3:2-portrait inputs at training too (acceptable: as
long as both sides agree, the LoRA learns whatever invariance it
needs).

If Bug B ever gets a proper fix in `mlx-swift-lm`, pass
`--resize_to none` and let HF's processor handle aspect ratios at
training time again.

## The proper fix: port HF's processor to mlx-swift-lm

Sketch (not implemented):

1. Implement `aspectRatioPreservingResize(_ image:, patchSize:, maxPatches:, poolingKernelSize:)`
   in `MediaProcessing` — direct port of the HF reference.
2. Change `Gemma4Processor.preprocess` to:
   - Resize per the new helper.
   - Generate `imagePositionIds` (2D `(x, y)` per real patch, `(-1, -1)`
     for padding) and `numSoftTokensPerImage`.
   - Return `(MLXArray pixelValues, MLXArray imagePositionIds, [Int] numSoftTokensPerImage)`
     instead of `(MLXArray, THW)`.
3. Change `Gemma4VisionModel.callAsFunction(_ pixelValues:)` to accept
   `imagePositionIds` and pass them into `Gemma4VisionPatchEmbedder`
   instead of the internally-derived regular grid.
4. Change `Gemma4Processor.prepare(input:)` to expand the prompt's
   `<image>` placeholders with the **per-image** `numSoftTokens` count
   instead of always 280, matching the HF behavior.
5. Update `processor_config.json` handling: prefer the **upstream**
   Google JSON (no `size`), let it omit the field, and use
   aspect-ratio-preserving resize as the default.

This is the path that lets `--resize_to none` work, removes the need
for `fetch-gemma.sh` to patch `processor_config.json`, and matches the
model to its training distribution for any aspect ratio. It's also the
right upstream contribution back to `mlx-swift-lm`.

## Cross-references

- Train↔deploy export contract that enforces 960×672 on the merged
  model: [`../finetune/01-pipeline.md`](../finetune/01-pipeline.md) and
  [`../quantization/05-mlx-vlm-design.md`](../quantization/05-mlx-vlm-design.md).
- KV-shared layer parity audit (the sibling cross-stack issue):
  [`12-mlx-vlm-vs-hf-kv-sharing.md`](12-mlx-vlm-vs-hf-kv-sharing.md).
- MLX quantization debug (where the wrong forward pass surfaced):
  [`15-postmortems.md`](15-postmortems.md) §2.
