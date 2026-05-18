# Eval Results — PlantNet Species-Match Benchmark

Eval results for quantization variants of the SFT'd Gemma 4 E2B model,
evaluated on the **paper-grade PlantNet-300K test/ split** (29,880 rows,
782 species after English-name filter).

Quick test: n=300, seed=0 (`random.Random(0).sample(records, 300)`).

## Results

| # | Variant | Backend | Size | PlantNet match | ROUGE-L mean | Drop vs M0 |
|---|---|---|---|---|---|---|
| M0 | bf16 reference | mlx_vlm | 9.5 GB | **85.7 %** (257/300) | 0.831 | — |
| M0_hf | bf16 reference | hf_bf16 (MPS) | 9.5 GB | **81.0 %** (243/300) | 0.787 | -4.7 pt |
| M1 | INT4 g128 + skip embed_vision | mlx_vlm | 3.2 GB | **78.3 %** (235/300) | 0.761 | -7.4 pt |

**M0 vs M0_hf**: same bf16 weights, different inference frameworks.
The -4.7 pt gap is under investigation (possible causes: HF
transformers attention implementation, processor template differences,
or MPS vs Metal numerics). The MLX path is the deploy target.

## Source model

SFT config:
`finetune/configs/plantnet-50k-baseline-lora-r256+fullproj-lr5e5-data-aug-enwiki.yaml`

Key training settings: LoRA r=256 alpha=256, tune_projector=true,
lr=2e-4 (LoRA) / 5e-5 (projector), bf16, 5 epochs, online image
augmentation, English common-name + Wikipedia description prompts.

## How each variant was produced

### M0 — bf16 reference via mlx_vlm (no quantization)

The merged bf16 checkpoint loaded directly via `mlx_vlm.load`. Requires
a KV-shared sidecar (`prep_inject_kv_shared.py`) for mlx_vlm
compatibility.

```bash
python -m quantization.scripts.run_eval \
    --variant M0_bf16_test300 \
    --loader mlx_vlm \
    --model_dir quantization/results/_merged_bf16 \
    --plantnet_val_jsonl finetune/data/english-desc/test.jsonl \
    --plantnet_n 300 \
    --eval_seed 0 \
    --benchmarks plantnet_val \
    --output_dir quantization/results/bf16_r0_sft_aug_enwiki_test300
```

### M0_hf — bf16 reference via HF transformers (PyTorch, MPS)

Same merged bf16 checkpoint loaded via HF
`AutoModelForImageTextToText.from_pretrained` with `torch_dtype=bf16`,
`device_map=auto` (lands on MPS). Cross-validation of M0 using a
different inference stack.

Env: PyTorch 2.10 + transformers 5.8.1 + peft 0.19.1, Apple M5 Pro.

```bash
# Ran via inline script (quantization/scripts/run/eval.py has a
# scripts/inspect/ shadow that breaks stdlib on Python 3.13).
# Equivalent to:
python -m quantization.scripts.run.eval \
    --variant M0_bf16_test300_hf \
    --loader hf_bf16 \
    --model_dir quantization/results/_merged_bf16 \
    --plantnet_val_jsonl finetune/data/english-desc/test.jsonl \
    --plantnet_n 300 \
    --eval_seed 0 \
    --benchmarks plantnet_val \
    --output_dir quantization/results/bf16_r0_sft_aug_enwiki_test300_hf \
    --device_map auto
```

### M1 — INT4 affine g128 (iOS-deployable)

4-bit affine quantization with group_size=128. Vision tower, audio
tower, and projector (`embed_vision`, `embed_audio`) kept at bf16.

Quantization:

```bash
python -m quantization.scripts.run_mlx_vlm_deploy_variant \
    --src   quantization/results/_merged_bf16 \
    --dst   quantization/results/mlx_vlm_g128_sft_aug_enwiki \
    --q-bits 4 \
    --q-group-size 128 \
    --q-mode affine \
    --also-skip embed_vision embed_audio
```

Eval:

```bash
python -m quantization.scripts.run_eval \
    --variant M1_g128_test300 \
    --loader mlx_vlm \
    --model_dir quantization/results/mlx_vlm_g128_sft_aug_enwiki \
    --plantnet_val_jsonl finetune/data/english-desc/test.jsonl \
    --plantnet_n 300 \
    --eval_seed 0 \
    --benchmarks plantnet_val \
    --output_dir quantization/results/mlx_vlm_g128_sft_aug_enwiki_test300
```

## Test data generation

```bash
PYTHON=/path/to/python bash finetune/scripts/prepare_plantnet_50k.sh
```

Produces `train.jsonl` (45k), `val.jsonl` (5k, in-distribution holdout),
and `test.jsonl` (29,880, paper-grade from PlantNet-300K-data-v2/test/).
All eval numbers in this directory use `test.jsonl`.

## File layout

```
M0_bf16_test300_summary.json       Aggregated metrics (accuracy, ROUGE-L, wall time)
M0_bf16_test300_per_sample.json    300 per-case predictions with full text:
                                     - response: model generation
                                     - reference: ground truth
                                     - pred_species / ref_species: extracted names
                                     - rouge_l, species_match: auto metrics

M0_bf16_test300_hf_summary.json    Same structure, hf_bf16 loader on MPS
M0_bf16_test300_hf_per_sample.json Same structure as M0

M1_g128_test300_summary.json       Same structure as M0
M1_g128_test300_per_sample.json    Same structure as M0
```

Per-sample files are designed for LLM-as-judge evaluation: each record
contains the full `response` and `reference` text alongside automated
metrics.
