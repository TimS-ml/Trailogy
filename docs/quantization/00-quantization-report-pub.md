# Gemma 4 E2B — Quantization Overview

One-page summary of the quantization sweep on the SFT'd Gemma 4 E2B.
Per-method numbers and reproduction recipes live in the per-route
files; this doc is the entry point and the headline-result table.

For the MLX stack mental model (mlx-vlm as deploy substrate, mlx-lm
as quant-core provider), read [`05-mlx-vlm-design.md`](05-mlx-vlm-design.md)
first.

## Target

- **iOS Gemma 4 E2B at ≤ 4 GB on disk** (ideal ~3.5 GB).
- Loadable by `mlx-swift-lm` in `.vlm` mode.
- Minimal accuracy drop vs bf16 reference on PlantNet.

## Source model

Base `unsloth/gemma-4-E2B-it` (5.12 B params, 9.54 GB bf16), SFT'd on
PlantNet-50k English+wiki for 5 epochs with LoRA r=256 + full
projector tuning. Adapter merged into a single bf16 checkpoint at
`src/quantization/results/_merged_bf16/` — that 9.5 GB merge is the
input to every quantization method.

## Size budget

Base bf16 = **9.54 GB**, 5.12 B params.

| Sub-module | Params | bf16 size | Quant policy |
|---|---:|---:|---|
| `language_model` | 4.65 B | 8.66 GB | quantize (4-bit) |
| `audio_tower` | 0.31 B | 0.57 GB | strip (iOS unused) |
| `vision_tower` (SigLIP) | 0.17 B | 0.31 GB | bf16 always |
| `embed_vision` (projector) | 1.2 M | ~3 MB | bf16 always |
| `embed_audio` | 1.6 M | ~3 MB | strip with audio tower |

After audio strip + LM body 4-bit + vision bf16: **~3.0-3.6 GB
achievable**. Public reference `mlx-community/gemma-4-e2b-it-4bit` =
3.58 GB on the un-SFT'd base.

## Headline results

PlantNet test split, n=300 quick-test (sorted `test.jsonl` seed=0).
**M0/R0 are bf16 references on their respective frameworks**; all
quantized rows compare against the bf16 reference on the same
framework.

| # | Route | Variant | Size | PlantNet | Drop vs ref | iOS-loadable |
|---|---|---|---:|---:|---:|---|
| M0 | bf16 (MLX) | `bf16_r0_sft_aug_enwiki` | 9.5 GB | **88.3%** | — (ceiling) | yes |
| M2 | MLX-VLM affine | `mlx_vlm_g64_sft_aug_enwiki` | 3.4 GB | 83.7% | −4.7 | ✅ |
| **M8b** | **MLX + EoRA** | **M2 + EoRA r=64** | **3.6 GB** | **88.0%** | **−0.3** | ✅ |
| R0 | bf16 (HF/CUDA) | `bf16_reference` | 9.54 GB | 86.7% | — (ceiling) | no |
| R6 | HF/CUDA hybrid | `gptq_w4g64_da0_hybrid_pl_g128` | **3.41 GB** | 83.7% | −3.0 | no |
| R5 | HF (bnb NF4) | `bnb_nf4` | 6.31 GB | **0.1%** | −70.5 | no (cautionary) |

**Recommended deploy artifact: M8b — M2 + EoRA r=64 (3.6 GB, 88.0%).**

EoRA is a training-free post-quant adapter that computes a low-rank
correction per quantized linear from calibration covariance + the
weight delta. It closes the M2 → M0 gap to within statistical noise
on n=300 (within ±5 pp 95% CI). Adapter ships as a separate
safetensors alongside the quantized model; mlx-swift-lm already
supports the QLoRA forward path.

## Routes — what each one is for

| Route | Path | Purpose | Sub-4 GB? | iOS? |
|---|---|---|---|---|
| **A** | 4-bit during SFT (QLoRA/QAT) | Reference accuracy ceiling for "what does 4-bit-aware training give you" | n/a | no |
| **B.1** | bf16 SFT → HF GPTQModel + torchao hybrid | CUDA reference artifact; cross-validates B.2 numbers | ✅ (3.41 GB) | no |
| **B.2** | bf16 SFT → MLX → mlx-lm quant cores | Production deploy artifact | ✅ (3.4 GB) | ✅ |

B.1 and B.2 are independent quantization routes on the same SFT'd
bf16 merge. B.2 is the only route that produces an iOS-loadable
artifact; B.1 exists primarily as a CUDA reference to cross-validate
the MLX numbers.

## Key findings

1. **Vision tower must stay bf16.** bnb NF4 quantizes the SigLIP
   encoder to 4-bit and collapses PlantNet from 70.6% to 0.1%. The
   `vision_tower` bf16 constraint is non-negotiable on this task.
   Full ablation in [`B1-bnb-nf4-vision-collapse.md`](B1-bnb-nf4-vision-collapse.md).

2. **Affine PTQ ceiling on this model is ~84% (M3 g32).** Flat
   affine quantization with the correct scope (LM body +
   `embed_{vision,audio}` skip) lands within the 5-pt comfortable
   band. Smaller group_size gives slightly better numerics
   (g32 > g64 > g128) at +0.2 GB / group halving.

3. **GPTQ stable (M4) is NaN-free but per-row symmetric grid is too
   coarse.** Costs 22.7 pts vs the affine M2/M3 baseline. The
   parameter-golf port works as a stability layer but its
   quantization grid is wrong for a 4.4 B model at 4-bit.

4. **Calibration size dominates group_size for GPTQ quality.** Early
   laptop GPTQ runs with 54 calibration samples dropped to 47.7% on
   n=2870. Re-quanting with 256 PlantNet + 256 WikiText recovered to
   83.7% on n=300.

5. **EoRA closes the quant gap for free.** No training; just calibration
   covariance + truncated SVD on `(W_bf16 − dequant(W_q))`. r=64
   recovers 4.4 of M2's 4.7-pt gap to bf16 ceiling, at 0.2 GB adapter
   cost.

6. **HF transformers ≥ 5.8 changed Gemma 4 KV layer storage**, dropping
   the K/V tensors for KV-shared layers (15-34) on save. The
   `mlx_vlm.load` path needs a sidecar injecting zero-init K/V/k_norm
   tensors for those layers; the K/V are inert at runtime
   (KV-shared layers fetch from the global cache). Full details in
   [`02b-mlx-torch-convert.md`](02b-mlx-torch-convert.md).

## Tripwires

A variant trips if any of:

- `SIZE > 4.0 GB` — exceeds the iOS jetsam ceiling.
- `PlantNet drop > 10 pts vs same-framework bf16` — likely a scope
  error (vision tower quantized, projector regression).
- `PPL > 2× bf16` — catastrophic language damage.
- `inspect_vision_dtype` non-zero — `vision_tower.*` has anything
  other than bf16/fp32/fp16. Designed as a pre-ship CI hook.

## Per-route detail

- **B.1 (HF/CUDA):** [`B1-sft-results.md`](B1-sft-results.md) — per-variant
  GPTQ numbers, hybrid pipeline narrative.
- **B.1 tooling:** [`B1-torchao-vs-gptqmodel.md`](B1-torchao-vs-gptqmodel.md) —
  why GPTQModel handles Linears and torchao handles the packed embedding.
- **B.1 failure case:** [`B1-bnb-nf4-vision-collapse.md`](B1-bnb-nf4-vision-collapse.md) —
  why NF4 is catastrophic; the vision-tower lesson.
- **B.2 (MLX deploy):** [`B2-sft-results.md`](B2-sft-results.md) — per-variant
  MLX numbers, EoRA recipe.
- **Calibration data:** [`04-calibration-data-design.md`](04-calibration-data-design.md).
- **MLX stack:** [`05-mlx-vlm-design.md`](05-mlx-vlm-design.md).

## Public 4-bit references

From [`01-baselines.md`](01-baselines.md), on the un-SFT'd base:

| Repo | Size | LM bits | Towers |
|---|---:|---:|---|
| `mlx-community/gemma-4-e2b-it-4bit` | **3.58 GB** | 4.54 | bf16 |
| `unsloth/gemma-4-E2B-it-UD-MLX-4bit` @HEAD | 4.52 GB | 6.14 | bf16 |
| `unsloth/gemma-4-E2B-it-UD-MLX-4bit` @9ee11f5 | 3.55 GB | ~6.14 | absent (text-only) |

The 3.58 GB `mlx-community` number is the size target. Mixed-precision
Unsloth UD lands above the 4 GB ceiling unless the multimodal towers
are dropped — which we cannot, because plant-ID needs the vision tower.
