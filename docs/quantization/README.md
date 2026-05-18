# Gemma 4 E2B — Quantization Experiments

## TL;DR

- This folder is the public index for Gemma 4 E2B deploy-time quantization: the goal is an SFT'd 4-bit checkpoint under 4 GB with minimal PlantNet accuracy loss.
- The docs compare three routes: 4-bit-aware SFT, a CUDA/HF post-training path, and an MLX-native path that can produce the iOS artifact.
- The current recommended artifact comes from the MLX-native route; the CUDA/HF route remains useful as an independent accuracy and size cross-check.
- Start with `00-quantization-report-pub.md` for headline results, then use the route-specific files for reproduction details and failure analysis.

Engineering notes for the deploy-time quantization sweep. Companion to
the code in `src/quantization/`. The goal: produce a quantized SFT'd
Gemma 4 E2B under 4 GB on disk (target ~3.0-3.6 GB) with minimal
accuracy drop on PlantNet.

Route A (4-bit SFT) docs are prefixed `A-`. Route B.1 (HF/CUDA) and
B.2 (MLX deploy) live alongside.

## Files

### General (route-agnostic)

| File | Covers |
|---|---|
| [`00-quantization-report-pub.md`](00-quantization-report-pub.md) | **Overview + headline results.** Start here. Size budget, route summary, the recommended deploy artifact, and links into the per-route detail files. |
| [`00-quantization-roadmap.md`](00-quantization-roadmap.md) | **Route picker.** Two-route plan (Route A QLoRA / Route B PTQ), method matrix, decision criteria, calibration-data design, cross-backend validation plan. |
| [`01-baselines.md`](01-baselines.md) | Reference models on the **un-SFT'd** base: `mlx-community/gemma-4-e2b-it-4bit` (3.58 GB), `unsloth/gemma-4-E2B-it-UD-MLX-4bit` (HEAD 4.52 GB, `9ee11f5` 3.55 GB). |
| [`02-methods-and-eval.md`](02-methods-and-eval.md) | What each method does + how we measure it. Quick test (n=300) vs full eval. |
| [`02b-mlx-torch-convert.md`](02b-mlx-torch-convert.md) | HF transformers ≥ 5.8 ↔ `mlx_vlm.convert` interop. The KV-shared K/V missing-tensors gotcha, bridge strategies, why the "PEFT merge → mlx_vlm convert" recipe broke on the latest HF. |
| [`04-calibration-data-design.md`](04-calibration-data-design.md) | **Route-agnostic** calibration data spec (text-only / mixed-text / multimodal). Adapters for B.1 (HF GPTQModel) and B.2 (mlx-lm `*_quantize()` core via hybrid flow). |
| [`05-mlx-vlm-design.md`](05-mlx-vlm-design.md) | **MLX stack mental model.** mlx-vlm is the deploy substrate; mlx-lm's Gemma 4 model class is wrong (never `mlx_lm.load`); mlx-lm's quant cores are reusable via the hybrid flow but buggy. Read before any quant work. |

### Route A — 4-bit SFT (QAT / QLoRA during training)

| File | Covers |
|---|---|
| [`A-baseline2-qlora-progress.md`](A-baseline2-qlora-progress.md) | QLoRA baseline-2 SFT run progress notes. Companion to `B1-sft-results.md` (which tracks the PTQ side of the same plan). |

### Route B.1 — bf16 SFT → HF GPTQModel + torchao hybrid

| File | Covers |
|---|---|
| [`B1-sft-results.md`](B1-sft-results.md) | Per-variant detail for **HF/CUDA** quant runs on our SFT'd model (bf16 reference, GPTQ R1-R4, hybrid R6 first sub-4 GB artifact). |
| [`B1-torchao-vs-gptqmodel.md`](B1-torchao-vs-gptqmodel.md) | Tooling decomposition: GPTQModel handles `nn.Linear`, torchao handles the packed `embed_tokens_per_layer`. The 4.7 GB embedding size driver, the `embed_scale` bug, and the hybrid pipeline that lands sub-4 GB. |
| [`B1-bnb-nf4-vision-collapse.md`](B1-bnb-nf4-vision-collapse.md) | Why bitsandbytes NF4 is catastrophic (vision tower → 0.1 % species_match) on our SFT'd model. The "do not quantize the vision tower" lesson, with the 3-arm ablation. |

### Route B.2 — bf16 SFT → MLX → mlx-lm quantize (production deploy)

| File | Covers |
|---|---|
| [`B2-research-spec.md`](B2-research-spec.md) | Research spec for Route B.2 — what we set out to measure (calibration-data effect, EoRA rank sweep, deploy-side parity). |
| [`B2-sft-results.md`](B2-sft-results.md) | Per-variant detail for **MLX** runs on our SFT'd model — the iOS-deployable candidates. EoRA r=64 closes the gap to bf16. |
| [`B2-sft-r8a8-13k-results.md`](B2-sft-r8a8-13k-results.md) | Latest SFT'd r8/α8 step-13k results across the M-method matrix on PlantNet. Numbers on the actual deploy candidate. |

## Reading order

For a new contributor:

1. [`00-quantization-report-pub.md`](00-quantization-report-pub.md) — overview + headline results.
2. [`05-mlx-vlm-design.md`](05-mlx-vlm-design.md) — the mental model (which library does what; why mlx-lm outputs don't deploy on iOS).
3. [`02-methods-and-eval.md`](02-methods-and-eval.md) — what we test and how we measure it.
4. [`01-baselines.md`](01-baselines.md) — the three public 4-bit MLX reference checkpoints.
5. [`B1-sft-results.md`](B1-sft-results.md) + [`B2-sft-results.md`](B2-sft-results.md) — per-variant numbers on our SFT'd model.
6. [`04-calibration-data-design.md`](04-calibration-data-design.md) — the calibration design.
7. [`B1-bnb-nf4-vision-collapse.md`](B1-bnb-nf4-vision-collapse.md) — case study on the failure mode of quantizing the vision tower.

Historical context lives in
[`../general/15-postmortems.md`](../general/15-postmortems.md) §2 —
the mac_mlx_lm round, including the four architectural mismatches
between `mlx_lm` and `mlx_vlm` and the failed splice attempt.

## Related

| Location | Purpose |
|---|---|
| `src/quantization/` | Code: PTQ method wrappers + scripts + configs |
| `src/quantization/src/eval/` | Benchmark runner — unified loader interface (hf_bf16 / hf_gptq / mlx_vlm) |
| `src/finetune/src/export_mlx.py` | Production MLX export (uses `mlx_vlm.convert`; tripwires assert vision tower is preserved) |
| [`../finetune/01-pipeline.md`](../finetune/01-pipeline.md) | Where the bf16 SFT'd merged model comes from |
