# Modal Runner

Run Trailogy LoRA / DoRA finetuning on Modal cloud GPUs.

## Prerequisites

```bash
pip install modal
modal setup
```

## HuggingFace Token

The base model (`unsloth/gemma-4-E2B-it`) is gated. Create a Modal secret:

```bash
modal secret create huggingface HF_TOKEN=hf_...
```

## How It Works

- Base image: `timsml/trailogy:latest` (all deps pre-installed)
- At image build time, clones `Trailogy` repo (`feature/sft-quant-datamix` branch)
- Training data is already in the container at `/workspace/data/mix-50k-v2/`
- Outputs/checkpoints are persisted to the `trailogy-runs` Modal volume

## Training

### Full run (1xL40S 48GB)

```bash
# Default config: r32-a32-dora-nokl-local-30k-v2mix.yaml
modal run scripts/modal_runner.py::main

# Custom config
modal run scripts/modal_runner.py::main \
    --config configs/local_sweep/r32-a32-dora-nokl-local-30k-v2mix.yaml

# With custom run name and step limit
modal run scripts/modal_runner.py::main --run-name dora-r32-test --max-steps 5000
```

This will:
1. Spin up 1xL40S (48GB VRAM)
2. Clone Trailogy repo (feature/sft-quant-datamix branch)
3. Run `python -m src.finetune --config <config>`
4. Stream logs to your terminal
5. After training: auto-eval on val sets
6. Save logs + checkpoints + eval results to the `trailogy-runs` volume
7. Download everything to `./modal-runs/` locally

### Smoke test (1xL40S, 50 steps)

```bash
modal run scripts/modal_runner.py::smoke_test
```

### Download runs manually

```bash
modal run scripts/modal_runner.py::download_runs
```

Or directly:
```bash
modal volume get trailogy-runs / ./modal-runs --force
```

## Output Files

After a run, `./modal-runs/` contains:

| Path | Description |
|------|-------------|
| `logs/<run_name>.log` | Full training log |
| `outputs/<run_name>/final-adapter/` | LoRA/DoRA adapter weights |
| `outputs/<run_name>/checkpoint-*/` | Intermediate checkpoints |
| `results/<run_name>_eval.json` | Eval results (PlantNet, MMLU, etc.) |

## Cost Estimate

| Config | Rate | Typical Duration | Cost |
|--------|------|-----------------|------|
| 1xL40S full run (30k steps) | ~$1.4/hr | ~10-14 hr | ~$15-20 |
| 1xL40S smoke test | ~$1.4/hr | ~5 min | ~$0.1 |

Container auto-terminates after training. 6-hour hard timeout on full runs.
