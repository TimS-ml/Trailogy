# Quantization experiments

Deploy-time quantization sweep for Gemma 4 E2B. The goal is to produce
an SFT'd, quantized iOS-deployable model under 4 GB with minimal
accuracy drop.

This README is the self-contained entry point. Internal design notes
and rationale live in the team's private design-doc repo (not mirrored
here).

## Why a separate top-level directory

Most quantization methods are **post-training**: they take a trained
bf16 checkpoint and produce a quantized checkpoint, with no
modification to the training loop. Those methods live here.

**QAT (quantization-aware training)** is the exception — it modifies
the training loop, so its training-time code lives in `finetune/`
alongside the SFT machinery. The export-time half of QAT (applying the
QAT recipe to produce the final quantized artifact) lives here, in
`methods/qat_export.py`.

```
quantization/
├── README.md            (this file)
├── common/              shared utilities: model I/O, size, calibration
├── methods/             one module per quantization method
├── bridges/             format translators (HF GPTQ → MLX, etc.) — see B.1.3
├── eval/                benchmark harness (PlantNet, VQAv2, WikiText PPL, …)
├── scripts/             entry points: run_quant.py, run_eval.py, ...
├── configs/             per-method YAML configs (loaded via run_quant --config)
├── tests/               unit tests (run on Mac without CUDA)
└── results/             eval JSONs + quantized model dirs (gitignored)
```

## Methods overview

Methods that produce a real quantized artifact today:

| Module | Method | Stage | Hardware needed |
|---|---|---|---|
| `methods/mlx_vlm_baseline.py` | `mlx_vlm.convert -q --q-bits 4` (group 64) | post-training | Mac (MLX) or Linux (mlx-cuda-12) |
| `methods/mlx_vlm_groups.py` | Same, group_size 32 / 64 / 128 sweep | post-training | Mac (MLX) or Linux (mlx-cuda-12) |
| `methods/gptq.py` | GPTQ calibration-based PTQ (B.1 priority) | post-training | 4090 (CUDA) |
| `methods/bnb_nf4.py` | bitsandbytes NF4 (reference only, not MLX-deployable) | post-training | 4090 (CUDA) |
| `methods/mac_mlx_lm/{gptq,awq,dwq,dynamic_quant}.py` | mlx-lm native quants (B.2 research) | post-training | Mac (MLX) |

Method **stubs** under `methods/_stubs/` — raise `NotImplementedError`
when invoked, all blocked on external work:

| Module | Status |
|---|---|
| `methods/_stubs/awq.py` | CUDA AWQ — lower priority than GPTQ |
| `methods/_stubs/unsloth_ud.py` | Unsloth UD MLX recipe — gated on `scripts/diff_unsloth_ud.py` extraction |
| `methods/_stubs/qat_export.py` | QAT export — gated on policy ruling + QAT training-loop code |

Bridges (format translators) live in `bridges/`:

| Module | Status |
|---|---|
| `bridges/hf_gptq_to_mlx.py` | **B.1.3 deliverable** — placeholder + design spec |

See the internal design-doc on quantization methods for the per-method
rationale, expected size, and risk notes.

## Quickstart — single variant

```bash
# 1. Merge SFT LoRA into bf16 + quantize via mlx_vlm baseline.
#    On Mac (mlx required); merge-only step also works on 4090.
python -m quantization.scripts.run.quant \
    --method mlx_vlm_g64 \
    --base_model unsloth/gemma-4-E2B-it \
    --adapter outputs/plantnet-50k-lora-r256+fullproj-lr5e5/final-adapter \
    --output_dir results/mlx_vlm_g64

# 2. Eval the bf16 reference (4090) — PlantNet + VQAv2 + WikiText PPL:
python -m quantization.scripts.run.eval \
    --variant bf16_reference \
    --loader hf_bf16 \
    --model_dir results/mlx_vlm_g64.bf16-merged \
    --plantnet_val_jsonl finetune/data/val.jsonl \
    --benchmarks plantnet_val wikitext_ppl vqav2_devtest \
    --output_dir results/bf16_reference

# 3. Eval the INT4 MLX variant (Mac) — PlantNet + VQAv2 (PPL skipped):
python -m quantization.scripts.run.eval \
    --variant mlx_vlm_g64 \
    --loader mlx_vlm \
    --model_dir results/mlx_vlm_g64 \
    --plantnet_val_jsonl finetune/data/val.jsonl \
    --benchmarks plantnet_val vqav2_devtest \
    --output_dir results/mlx_vlm_g64
```

## Smoke-size eval (no full model load)

For verifying the eval framework on a 4090 without paying the cost of
loading the 9.5 GB bf16 base, use small subsets:

```bash
python -m quantization.scripts.run.eval \
    --variant smoke \
    --loader hf_bf16 \
    --model_dir <merged-bf16-dir> \
    --plantnet_val_jsonl finetune/data/val.jsonl \
    --plantnet_n 50 --vqav2_n 100 --wikitext_n 20 \
    --benchmarks plantnet_val wikitext_ppl vqav2_devtest \
    --output_dir /tmp/eval-smoke
```

## Reproducing the mlx-community baseline

Before any custom quant claim is credible, we must reproduce
`mlx-community/gemma-4-e2b-it-4bit` (3.58 GB) from the bf16 base
using our pipeline:

```bash
python -m quantization.scripts.repair.repro_mlx_community \
    --base_model unsloth/gemma-4-E2B-it \
    --output_dir results/mlx_community_repro
```

If this doesn't produce a directory within ~5% of 3.58 GB, every
downstream comparison is suspect. Treat this as the first baseline
reproduction gate.

## Data discipline — non-negotiable

Two rules every method must respect:

1. **Calibration data is from `train.jsonl`, never `val.jsonl`.**
   GPTQ (and any future method that takes a calibration set) must use
   the training split. Eval is on `val.jsonl`. Leaking eval samples
   into calibration trivially preserves eval scores while learning
   nothing. Enforced at two layers:
   - Python: `quantization.methods.gptq._reject_calibration_leak`
     raises ``CalibrationDataLeakError`` on `val.jsonl` paths.
   - Bash: ``_reject_overfit100`` in ``scripts/_env/_common.sh`` rejects
     overfit100 paths.

2. **No overfit100 data anywhere in the deliverable pipeline.**
   `plantnet-overfit100-*` files are train==eval by design — they are
   for memorization-ceiling tests only. Calibrating, training, or
   evaluating on them looks perfect and tells us nothing about
   generalization.

If you need to bypass the guards for an experiment, rename the file or
patch out the guard locally — but **don't merge that patch**. The
guards exist to catch the leak before it ships.

## Hardware

| Step | Hardware | Notes |
|---|---|---|
| Adapter merge (bf16) | CPU or 4090 | ~10 GB RAM (CPU) or VRAM (GPU) for `Gemma4ForConditionalGeneration` |
| GPTQ calibration | 4090 (CUDA) | Fits in 24 GB at bf16 |
| QAT training | 4090 (CUDA) | Same VRAM budget as normal SFT |
| MLX conversion (`mlx_vlm.convert`) | **Mac with MLX** | The MLX runtime is Apple Silicon only |
| Eval (bf16) | 4090 (CUDA) | |
| Eval (INT4 MLX) | Mac with MLX | |

The 4090 covers everything **except** the actual MLX bit-packing
conversion. Smoke-test entry points are tagged with the hardware they
need.

## Smoke tests

```bash
# Unit tests — pure Python, no model load, no CUDA needed:
pytest quantization/tests/

# 4090 smoke (any single quant method on a tiny calibration set):
bash quantization/scripts/run/smoke_4090.sh

# Mac smoke (MLX conversion of a tiny model):
bash quantization/scripts/smoke_quant_mac.sh
```

## Pointers

| Concern | File |
|---|---|
| Existing baseline MLX export | `finetune/src/export_mlx.py` |
| Safetensors header reader (reusable) | `finetune/src/export_mlx.py:_read_safetensors_header` |
| Trained vision shape constant | `finetune/src/export_mlx.py:TRAINED_VISION_SIZE` |
| SFT config dataclasses | `finetune/src/config.py` |
| Project policy on training-time quantization | `AGENTS.md` |
| Deadline + priorities | Internal dev timeline (private design-doc repo) |
