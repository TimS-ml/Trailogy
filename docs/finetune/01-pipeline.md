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

## What Problem This Pipeline Solves

The iOS app needs an MLX/VLM-format Gemma 4 checkpoint, but training happens in
the PyTorch/HF ecosystem. This pipeline adapts the model in bf16, merges the
adapter safely, and exports the result into the same multimodal shape the app
loads.

The baseline mode is deliberately conservative: LoRA updates the language side,
while the vision tower, audio tower, and projector stay frozen. Projector and
vision-tower tuning are documented separately in [`02-projector-mode.md`](02-projector-mode.md)
and [`03-vision-mode.md`](03-vision-mode.md).

## Pipeline

```text
PlantNet images
  -> prepare_plantnet.py: resize to 960x672 and write JSONL
  -> finetune.py: train bf16 LoRA with multimodal batches
  -> export_mlx.py: merge adapter into bf16 multimodal checkpoint
  -> mlx_vlm.convert: quantize/export MLX artifact
  -> iOS bundle: load through mlx-swift-lm VLM path
```

## Why The Details Matter

Three choices prevent misleading results:

| Risk | Guardrail |
|---|---|
| Language-only loaders silently drop `vision_tower` and `embed_vision`. | Merge with `AutoModelForImageTextToText`; export with `mlx_vlm`, not `mlx_lm`. |
| Train/deploy image preprocessing can diverge. | Pre-stretch training images to 960x672 and patch deploy processor config to the same size. |
| Frozen multimodal towers can accidentally become trainable. | Explicit freeze pass plus `assert_frozen` tripwire. |

The key invariant is that the model being evaluated after reload must be the
same multimodal model that was trained and exported.

## Training Format

Each example is converted into a multimodal chat record:

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "/path/to/resized.jpg"},
      {"type": "text", "text": "What plant is this?"}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "This is Quercus robur..."}
    ]}
  ]
}
```

The JSONL does not include a literal `<image>` token in the user text. The chat
template reserves image tokens from the structured image block. A literal token
would double-reserve image slots and corrupt the prompt.

## Baseline Configuration

| Field | Baseline intent |
|---|---|
| Base model | Stock multimodal Gemma 4 E2B in bf16. |
| Adapter | LoRA on language modules. |
| Vision tower | Frozen. |
| Audio tower | Frozen. |
| Projector | Frozen in this baseline; optionally tuned in projector mode. |
| Optimizer | Full-precision AdamW; 8-bit optimizers are rejected by config validation. |
| Export | Merge adapter, preserve vision weights, convert with `mlx_vlm`. |

## Output Artifacts

| Artifact | Purpose |
|---|---|
| `outputs/<run>/final_adapter/` | Saved PEFT adapter. |
| merged bf16 checkpoint | Full multimodal model after adapter merge. |
| MLX export directory | iOS-loadable model directory with `config.json`, `processor_config.json`, and safetensors. |
| `eval.json` / metrics logs | Evidence that the exported model still behaves as expected. |

## Main Lessons

1. Train in bf16 for credibility unless a run is explicitly a quantized-training
   experiment.
2. Do not use language-only model classes anywhere in the merge/export path.
3. Keep image preprocessing identical between training and iOS deployment.
4. Add tensor-inventory tripwires because loader failures can be silent.

## Related Docs

- Final anti-forgetting recipe: [`03-anti-forgetting-and-final-recipe.md`](03-anti-forgetting-and-final-recipe.md)
- Train/deploy image parity: [`../general/13-mlx-vision-input-parity.md`](../general/13-mlx-vision-input-parity.md)
- Silent loading failures: [`../general/15-postmortems.md`](../general/15-postmortems.md)
- Quantization entry point: [`../quantization/00-quantization-report-pub.md`](../quantization/00-quantization-report-pub.md)
