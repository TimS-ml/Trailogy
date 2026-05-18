# Gemma 4 E2B — Quantization Overview

## TL;DR

- This is the entry point for the Gemma 4 E2B quantization sweep, where the target is an iOS-loadable artifact at or below 4 GB from a 9.54 GB bf16 SFT merge.
- The recommended deploy candidate is M8b: MLX affine 4-bit g64 plus EoRA r=64, landing at 3.6 GB and 88.0 % PlantNet on the n=300 quick test.
- M8b is effectively tied with the MLX bf16 reference at 88.3 %, so readers should treat it as the current ship candidate rather than just a compression demo.
- The CUDA/HF hybrid route cross-checks the tradeoff at 3.41 GB and 83.7 %, but it is not the iOS deployment format.

One-page summary of the quantization sweep on the SFT'd Gemma 4 E2B.
Per-method numbers and reproduction recipes live in the per-route
files; this doc is the entry point and the headline-result table.

For the MLX stack mental model (mlx-vlm as deploy substrate, mlx-lm
as quant-core provider), read [`05-mlx-vlm-design.md`](05-mlx-vlm-design.md)
first.

## Why Quantization Was A Core Problem

The merged bf16 SFT model is about 9.5 GB. That is not a practical iOS artifact.
Trailogy needed a model small enough to bundle and load on-device while keeping
the plant-vision behavior gained from fine-tuning.

The target was simple:

```text
bf16 SFT model (~9.5 GB) -> iOS-loadable MLX artifact (<= 4 GB)
```

The hard part was preserving multimodal behavior. Quantizing the wrong modules
or using the wrong model tree can produce an artifact that is small, but
functionally broken.

## Headline Result

PlantNet quick test, n=300:

| Route | Variant | Size | PlantNet | iOS-loadable? | Role |
|---|---|---:|---:|---|---|
| MLX bf16 | M0 reference | 9.5 GB | 88.3% | yes | Quality ceiling. |
| MLX affine | M2 g64 | 3.4 GB | 83.7% | yes | Plain deploy quant baseline. |
| **MLX + EoRA** | **M8b, EoRA r=64** | **3.6 GB** | **88.0%** | **yes** | **Recommended candidate.** |
| HF/CUDA hybrid | R6 | 3.41 GB | 83.7% | no | Cross-check, not deployment format. |
| bnb NF4 | R5 | 6.31 GB | 0.1% | no | Failure case: vision tower quantized. |

The current ship candidate is M8b because it lands under the practical size
budget and is statistically close to the bf16 reference on the quick test.

## What Made This Non-Trivial

| Problem | Consequence | Fix |
|---|---|---|
| Vision tower quantized by generic 4-bit paths. | PlantNet collapses. | Keep vision tower in bf16/fp16; inspect dtype before shipping. |
| `mlx_lm` and `mlx_vlm` Gemma 4 trees differ. | Quantized output may not match iOS runtime. | Treat `mlx_vlm` as the deployment substrate. |
| Plain affine quantization loses several PlantNet points. | Small model but lower quality. | Add EoRA low-rank post-quant correction. |
| CUDA/HF outputs are not iOS artifacts. | Good cross-checks cannot ship directly. | Use them as reference rows, not deployment candidates. |

## Why EoRA Helped

EoRA is a training-free correction added after quantization. It estimates the
quantization error and adds back a low-rank adapter derived from calibration
statistics. In this sweep, EoRA r=64 recovered most of the plain MLX quant gap
with a modest size cost.

This matters because the project did not need only compression. It needed an
artifact that still behaved like the fine-tuned model.

## Route Summary

| Route | Purpose | Ship status |
|---|---|---|
| Route A: 4-bit-aware SFT | Research question: does training directly against a quantized representation help? | Not the default; project policy keeps normal SFT bf16 unless explicitly exploring this. |
| Route B.1: HF/CUDA PTQ | Independent quality/size cross-check using mature CUDA tools. | Useful reference, not iOS format. |
| Route B.2: MLX/VLM PTQ | Produce the actual iOS-loadable artifact. | Production path. |

## Ship Gates

An artifact is not a candidate unless it passes all of these:

- Size at or below the practical mobile budget.
- Loads through the MLX/VLM path used by iOS.
- Preserves vision weights in a non-collapsed dtype.
- PlantNet drop is within the accepted band against the same-framework bf16 reference.
- Eval uses a fixed split, fixed prompt, and deterministic generation settings.

## Where To Read Next

- MLX mental model: [`05-mlx-vlm-design.md`](05-mlx-vlm-design.md)
- Methods and eval protocol: [`02-methods-and-eval.md`](02-methods-and-eval.md)
- Full MLX results: [`B2-sft-results.md`](B2-sft-results.md)
- Vision-tower failure case: [`B1-bnb-nf4-vision-collapse.md`](B1-bnb-nf4-vision-collapse.md)
- Quantization postmortem context: [`../general/15-postmortems.md`](../general/15-postmortems.md)
