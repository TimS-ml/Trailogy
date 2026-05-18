# finetune/

LoRA finetuning of **Gemma 4 E2B** for the hikeCompanion iOS app, using
[unsloth](https://github.com/unslothai/unsloth) on an NVIDIA GPU.

The base model and chat template match what the iOS app loads at runtime
(`mlx-community/gemma-4-e2b-it-4bit`), so a trained LoRA adapter can be
merged + re-quantized to MLX without touching the base.

## Project rules — DO NOT REGRESS

- **Vision tower is frozen.** `FastModel.get_peft_model(..., finetune_vision_layers=False)`
  prevents LoRA injection into the vision encoder.
- **Audio tower is frozen.** Unsloth has no `finetune_audio_layers` flag yet
  (see notebook commentary), so we run an explicit post-LoRA freeze pass
  in `src/freeze.py:freeze_vision_audio_towers`. The config validator
  rejects `finetune_vision_layers: true` or `finetune_audio_layers: true`
  to guard against accidental drift.
- **iOS app does not need the audio tower.** Audio weights are stripped
  at deploy time via `../scripts/strip-gemma-audio.py`.

## Pipeline

```
PlantNet-300K images
    └─→ src/prepare_plantnet.py ──→ data/train.jsonl + data/val.jsonl
                                          │
                                          ▼
                                 src/finetune.py (unsloth FastModel + LoRA)
                                          │
                                          ▼
                                 outputs/.../final-adapter
                                          │
                                          ▼
                                 src/export_mlx.py ──→ iOS bundle
```

## Quick start

### 1. Install (NVIDIA box)

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...        # Gemma 4 is gated
huggingface-cli login
```

### 2. Prepare data (canonical PlantNet-50k)

```bash
# Download PlantNet-300K v2 from https://zenodo.org/records/10419064
# Extract so PlantNet-300K-data-v2/{train,val,test}/<species_id>/ sits next to this repo.
bash scripts/run/prepare_plantnet_50k.sh
```

### 3. Train

```bash
bash scripts/run/train.sh                                                 # default config, real training
bash scripts/run/train.sh configs/default.yaml --dry-run                  # CPU/Mac dry run
bash scripts/run/train.sh configs/my-run.yaml --no-eval                   # skip auto-eval
bash scripts/run/train.sh configs/my-run.yaml \
    --resume_from_checkpoint outputs/<prior-run>/checkpoint-1000      # resume; log → train-resume.log
```

As of 2026-05-11 every real training run writes the following artifacts
under `outputs/<run_name>/` where `<run_name> = <config-stem>_<timestamp>`:

| File | Source |
|---|---|
| `train.log` (or `train-resume.log` when resuming) | shell `tee` of the training process stdout/stderr |
| `eval.log` | shell `tee` of the auto-eval pass that runs immediately after a successful train |
| `final-adapter/` | PEFT save — language LoRA + (optionally) projector + last-N vision layers |
| `checkpoint-N/` | trainer checkpoints (every `training.save_steps`) |

Auto-eval is on by default (controlled by `cfg.eval.enabled`, default
true). It calls `python -m src.evaluate --config <same yaml>` and writes
the eval JSON summary to `results/<run_name>_eval.json` (unchanged path).
Pass `--no-eval` to `train.sh` to suppress, or set `eval.enabled: false`
in the config.

### 4. Verify (Mac, no GPU)

```bash
pytest                                    # unit tests
python -m src.finetune --config configs/default.yaml --dry-run
```

The dry-run path validates config, loads + converts data, prints
dataset stats, and skips `FastModel.from_pretrained` (which would need
CUDA).

## Layout

```
finetune/
├── configs/
│   └── default.yaml             # hyperparameters
├── data/                        # generated JSONL (gitignored)
├── scripts/
│   ├── run/                     # train/export/data-prep entry points
│   └── inspect/                 # diagnostics
├── src/
│   ├── __init__.py
│   ├── config.py                # YAML + CLI config (typed, validated)
│   ├── data.py                  # JSONL → unsloth `messages` format
│   ├── freeze.py                # vision + audio tower freeze pass (+ keep-projector variant)
│   ├── projector.py             # vision-language projector identification (modules_to_save)
│   ├── finetune.py              # main training entry point
│   ├── prepare_plantnet.py      # image discovery + JSONL emission (Latin-name variant)
│   ├── prepare_plantnet_enriched.py  # image discovery + JSONL emission (English vernacular + Wikipedia)
│   ├── evaluate.py              # eval + interactive inference
│   └── export_mlx.py            # adapter → MLX export (+ projector-changed tripwire)
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_freeze.py                       # uses fakes — runs without torch
│   ├── test_projector.py                    # projector identification + ensure_trainable
│   ├── test_export_mlx.py
│   ├── test_export_projector_tripwire.py    # silent-modules_to_save-drop detector
│   ├── test_prepare_plantnet.py
│   ├── test_evaluate_config.py              # `evaluate.py --config` defaults + override semantics
│   └── test_dry_run.py                      # subprocess CLI smoke (LoRA-only + projector)
├── pytest.ini
├── requirements.txt
└── README.md
```

## Config reference

`configs/default.yaml` is a lightly-commented schema. The dataclasses in
`src/config.py` are the source of truth — anything not in those
dataclasses is rejected by the YAML loader (with a warning).

Key knobs:

| Section / Field | Default | Notes |
|---|---|---|
| `model.base_model` | `unsloth/gemma-4-E2B-it` | Match iOS runtime. E4B works but won't fit on-device. |
| `model.max_seq_length` | 1024 | Bumping past 2048 hurts throughput on a single A100. |
| `model.load_in_4bit` | **false** | **Project policy: no QLoRA by default.** Standard runs use bf16 LoRA; explicit QLoRA comparison configs may opt in with `true`, and the validator emits a warning. The trainable-4bit tripwire in `finetune.py` still fails loudly if any `modules_to_save` projector / vision-layer copy becomes 4-bit. Inference-time 4-bit (in `evaluate.py`) is separate and unaffected — that's post-training quantization, not QLoRA. |
| `model.dtype` | `bfloat16` | Compute dtype. bf16 is the right default for Gemma 4: native on Ampere+ (A100/H100), no fp16 attention softmax overflow. Set to `float16` for V100/T4 (less stable), `float32` for full precision, or `null` to let unsloth auto-detect. |
| `lora.finetune_vision_layers` | **false** | Validator rejects true. |
| `lora.finetune_audio_layers` | **false** | Validator rejects true. |
| `lora.r` / `lora.lora_alpha` | 8 / 8 | Bump together; alpha == r is the unsloth recommendation. |
| `lora.tune_projector` | false | **(opt-in)** Unfreeze the vision-language projector (`embed_vision.*`) as full params, co-trained with the LoRA. See "Projector tuning" below. |
| `lora.projector_learning_rate` | null | LR for the projector when `tune_projector` is true. `null` → auto = `training.learning_rate / 10`. |
| `lora.tune_last_n_vision_layers` | 0 | **(opt-in)** Unfreeze the last N vision encoder layers (`vision_tower.encoder.layers.{total-N..total-1}`) as full params via PEFT `modules_to_save`. Requires `tune_projector: true`. See "Vision-tower last-N tuning" below. |
| `lora.vision_layers_learning_rate` | null | LR for the tuned vision encoder layers. `null` → auto = `training.learning_rate / 20`. |
| `training.max_steps` | null | Set to small int (e.g. 60) for a smoke test. |
| `training.num_train_epochs` | 1 | Used when `max_steps` is null. |
| `data.max_train_samples` | null | Cap for fast iteration; works alongside the prep script's own cap. |
| `eval.enabled` | **true** | Auto-eval after successful train, in `train.sh`. `false` skips. |
| `eval.max_eval_samples` | 300 | Cap val-set size during auto-eval. `null` = full val set. |
| `eval.max_new_tokens` | 256 | Generation length per sample during auto-eval. |
| `eval.use_unsloth` | **true** | Required for vision-tower models — `AutoModelForCausalLM` silently drops `vision_tower.*` on Gemma 4. |
| `eval.batch_size` | 1 | Currently the eval loop is sample-at-a-time. |
| `eval.load_in_4bit` | **false** | Inference-time 4-bit quantization of the merged model. Default off — eval runs in bf16 to match training dtype so scores reflect deployed-quality weights. This is post-training quantization (no gradients), NOT QLoRA — the no-QLoRA policy only governs training; eval 4-bit stays opt-in for memory-constrained eval boxes. Override via `--quantize` / `--no-quantize` on the CLI. |

## Projector tuning (feature/lora-plus-projector)

LoRA on the language layers re-keys/re-values the image soft tokens, but
cannot change either the visual representation (vision encoder is
frozen) or the **projection** of that representation into the
language-model hidden space (the `Gemma4MultimodalEmbedder` at
`embed_vision.*`). For VLM tuning, that projector benefits from
**full-parameter** training, not LoRA — a single Linear projection has
too little effective capacity for low-rank adaptation. This is the
LLaVA / PaliGemma standard recipe.

Enable with the new flag:

```yaml
lora:
  tune_projector: true
  projector_learning_rate: null   # auto = training.learning_rate / 10
```

Or use the reference config: `configs/plantnet-50k-projector.yaml`.

What changes mechanically when `tune_projector: true`:

1. `src.projector.find_projector_param_names(model)` introspects the
   loaded base model and identifies projector params via substring +
   boundary-dot matching. Fails loud if no projector params are found.
2. `FastModel.get_peft_model` is called with
   `modules_to_save=[<projector module names>]` so PEFT keeps those
   modules as fully trainable (saved as full tensors in the adapter
   dir, not LoRA-adapted).
3. The freeze pass uses `freeze_vision_audio_towers_keeping_projector`
   (vision encoder still frozen, audio tower still frozen, projector
   exempted from the walker).
4. A belt-and-braces `ensure_projector_trainable` pass re-flips
   `requires_grad=True` if unsloth's `modules_to_save` was silently
   dropped (logged as a WARNING when it fires).
5. `assert_frozen` is called with the projector params on the
   allowlist — still catches any other accidental unfreeze.
6. The optimizer is built with **two param groups**: LoRA params at
   `training.learning_rate`, projector params at
   `lora.projector_learning_rate` (or `training.learning_rate / 10`
   when null). bitsandbytes `AdamW8bit` if available, else
   `torch.optim.AdamW`.
7. At export time, `_assert_projector_changed_if_tuned` (in
   `export_mlx.py`) compares the merged model's projector tensors to
   the base model's. Identical bytes mean PEFT silently failed to
   restore `modules_to_save` weights — ship-stopper.

What does NOT change:

- The vision **encoder** (`vision_tower.{patch_embedder, encoder, pooler}`)
  is still frozen. The `tune_projector` flag does NOT unfreeze the
  encoder; the validator rejects `tune_projector=true` with
  `finetune_vision_layers=true`.
- The audio tower stays frozen.
- LoRA-only configs (existing `plantnet-50k-r8-a8-lr2e4.yaml` etc.)
  are unchanged. `tune_projector` defaults to false.
- The export pipeline is structurally identical; the new tripwire is a
  no-op for LoRA-only adapters (no projector tensors in the adapter →
  check skipped).

### GPU verification recipe

After landing changes that affect training, run the GPU smoke recipe
below on the NVIDIA box. The full pytest suite passes on macOS/CPU but
does not exercise the real CUDA training path.

```bash
cd src/finetune
pytest                                        # all green; same suite as Mac

# Tiny real run on the GPU box — 50 steps, verifies the projector path
python -m src.finetune --config configs/plantnet-50k-projector.yaml \
    --max_steps 50 --max_train_samples 200

# Look for these log lines confirming the path took:
#   "Projector tuning ENABLED. Identified N projector param(s) ..."
#   "Optimizer param groups: M LoRA params @ lr=2.00e-04, K projector params @ lr=2.00e-05"
#   No WARNING about "FastModel.get_peft_model did not honor modules_to_save"
#   (a warning here is OK — it means the fallback worked, but file the unsloth bug)

# Export the resulting adapter and confirm the new tripwire fires correctly:
bash scripts/run/export.sh outputs/plantnet-50k-lora-plus-projector/final-adapter \
    exports/projector-smoke 4

# Look for these log lines:
#   "Projector-changed tripwire: adapter contains N projector tensor(s) ..."
#   "Projector-changed tripwire passed: K/N projector params differ from base ..."
#   No RuntimeError "Projector-changed tripwire FIRED" (= ship-stopper if seen)
```

If full training (no `--max_steps` cap) produces an adapter that:
- Passes `pytest`
- Loads + merges without the tripwire firing
- Visibly improves plant-ID quality vs. the LoRA-only baseline

…then the projector-tune mode is validated.

### Backward compatibility

Existing LoRA-only configs and behavior are bit-identical: a
`git diff feature/finetune-unsloth feature/lora-plus-projector --
configs/plantnet-50k-r8-a8-lr2e4.yaml` is empty, and running with that
config takes exactly the same code path as before
(`freeze_vision_audio_towers`, no `modules_to_save`, single
optimizer-group, no projector tripwire).

## Vision-tower last-N tuning (feature/lora-plus-projector-plus-vision-tower)

Extends the projector path further: additionally unfreeze the **last N
layers** of `vision_tower.encoder.layers` as full parameters via PEFT's
`modules_to_save`. Gemma 4 E2B has 16 such layers; `N = 2` trains
`layers.14` and `layers.15` (the most semantic SigLIP layers).
~14M extra trainable params for `N = 2`.

Enable with:

```yaml
lora:
  tune_projector: true               # REQUIRED — validator rejects without it
  projector_learning_rate: 5.0e-5
  tune_last_n_vision_layers: 2
  vision_layers_learning_rate: null  # auto = training.learning_rate / 20
```

Reference config: `configs/plantnet-50k-lora-r256+fullproj+vision2-lr1e5.yaml`.

The validator rejects `tune_last_n_vision_layers > 0` without
`tune_projector: true` — moving the visual feature distribution while
the projector stays frozen creates a feature-space misalignment the
language LoRA cannot fix.

What changes mechanically when `tune_last_n_vision_layers > 0`:

1. `src.vision_layers.find_vision_encoder_layer_count(model)` discovers
   the layer count from the loaded base. The last-N indices are
   computed.
2. `find_last_n_vision_layer_module_names(model, n)` returns
   `vision_tower.encoder.layers.{i}` suffix strings — these are
   appended to PEFT's `modules_to_save` alongside the projector
   modules. PEFT matches them as suffix → works under any wrapping
   depth.
3. The freeze pass uses
   `freeze_vision_audio_towers_keeping_projector_and_vision_layers`
   so both the projector and the tuned vision layers are exempted
   while the rest of the vision/audio towers stay frozen.
4. Belt-and-braces `ensure_vision_layers_trainable` parallels the
   projector fallback. Logs a WARNING if PEFT silently dropped the
   wrapper.
5. `assert_frozen` extended with `tuned_vision_layer_indices` — still
   fires on any OTHER frozen-token leak.
6. The optimizer gains a **3rd param group**:
   - LoRA          @ `training.learning_rate`
   - Projector     @ `lora.projector_learning_rate` (or `LR/10`)
   - Vision layers @ `lora.vision_layers_learning_rate` (or `LR/20`)
7. At save time, `_assert_vision_layer_tensors_present_if_tuned`
   scans the adapter directory's safetensors headers and fails if any
   tuned layer index has no saved tensors.
8. At export time (`export_mlx.py`),
   `_assert_vision_layers_changed_if_tuned` snapshots base
   vision-layer bytes before adapter load and asserts at least one
   param per tuned layer index differs byte-for-byte after merge.
   Identical = PEFT silently dropped the wrapper; ship-stopper.

### Backward compatibility (vision-tower path)

`tune_last_n_vision_layers: 0` (default) → bit-identical to the
projector-only / LoRA-only paths. Every existing config takes the same
code path as before.

## Hardware

| Stage | Hardware | Notes |
|---|---|---|
| `pytest` + `--dry-run` | Any Mac/Linux | No CUDA needed. |
| Real training | NVIDIA A100/4090/3090 | QLoRA fits in ~16 GB VRAM on E2B. |
| MLX export | Apple Silicon Mac | Run after merging LoRA into base. |

## Related

- `../HikeCompanion/` — iOS app that loads the merged + MLX-converted model.
- `../scripts/strip-gemma-audio.py` — strips the audio tower from the
  on-device checkpoint (the language + vision towers are what we ship).
- `../../PlantNet-300K/` — dataset (we use the images, not the train code).
- `../../gemma4-e4b-unsloth.py` — the unsloth reference notebook this
  finetune is patterned after.
