# Model-Side Architecture Overview

## TL;DR

- The model pipeline turns a stock Gemma 4 E2B checkpoint into a supervised-finetuned, quantized artifact for the iOS app bundle.
- Data mixture, fine-tuning, and quantization run as separate workstreams but share one evaluation surface.
- The deploy target is a small on-device model bundle that stays under mobile memory limits while preserving general reasoning and plant-identification quality.
- The iOS app consumes the final artifact from its bundled `Models/Gemma/` directory.

## Three modules

Three Python packages sit side-by-side in the repo:

| Module | Purpose | Active phase |
|---|---|---|
| [`data_mix/`](../../src/data_mix/) | Multi-bucket dataset assembly (plant + cambrian + smoltalk + refuse / LLaVA / offline_qa) | Track A |
| [`finetune/`](../../src/finetune/) | SFT pipeline — LoRA + `modules_to_save`, KL/L2 anti-forgetting, ModalityAware sampler, eval | Track B |
| [`quantization/`](../../src/quantization/) | PTQ + MLX deploy quant + EoRA post-quant adapter | Track C |

All three share the same `src/` layout, `pytest.ini` shape, and
`scripts/run | inspect | repair | _env/` structure.

## Three workstreams (parallel)

```mermaid
flowchart TB
    subgraph TrackA["Track A — Data Mixture"]
        A1["v1 4-bucket: plant + cambrian + smoltalk + refuse"]
        A2["v2 LLaVA-mix + image=None text-only"]
        A3["v3 offline_qa persona bucket"]
        A4["v4 mix-50k variants (a/b/c) + 100k"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph TrackB["Track B — SFT"]
        B1["Baseline LoRA + projector + tripwires + QLoRA"]
        B2["ModalityAware sampler + multi-eval"]
        B3["KL + L2 anti-forgetting + cloud_sweep + camera-state gate"]
        B4["KL bracket + vision-last-N + SOTA recipe + v4 cloud sweep"]
        B1 --> B2 --> B3 --> B4
    end

    subgraph TrackC["Track C — Quantization"]
        C1["PTQ framework + safetensors merge + mlx_vlm linux"]
        C2["src/ refactor + YAML wiring + hybrid CUDA + hybrid MLX"]
        C3["EoRA post-quant adapter (+4.3 pp training-free)"]
        C4["v4 prefix wired into eval + EoRA polish"]
        C1 --> C2 --> C3 --> C4
    end

    TrackA -. trains .-> TrackB
    TrackB -. exports .-> TrackC
    TrackC -. ships to iOS bundle .-> ios[/Trailogy/HikeCompanion/<br/>Resources/Models/Gemma/]
```

The three tracks **share the eval surface** — same
`finetune/src/evaluate.py`, same `quantization/src/eval/plantnet.py`,
same Qwen judge, same deterministic generality eval (`plant_100 +
mmlu_50 + aime_20`). See [`10-eval-setup.md`](10-eval-setup.md) for the
shared eval contract and its caveats.

## Hardware

| Class | Role |
|---|---|
| 4090 desktop (24 GB) | Dominant SFT + CUDA-side PTQ host; `local_sweep` |
| 4090 laptop (16 GB) | `local_sweep_v2` (smaller batch) |
| 2× A100 (40 GB) | Parallel config sweeps |
| H100 / H200 cloud (remote GPU box) | `cloud_sweep`, big-batch + KL ablations, eval during training |
| Apple Silicon (M-series Mac) | MLX-native quant (Track C.2 hybrid) + EoRA calibration |

Cross-validation between CUDA and MLX eval backends works in either
direction since the from-source `mlx-cuda` fix
([`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md)).

## Core data flow — SFT (Track B)

```mermaid
sequenceDiagram
    participant CFG as YAML config
    participant FT as finetune.py
    participant PEFT as PEFT
    participant US as unsloth FastModel
    participant TR as ModalityAwareSFTTrainer
    participant REG as RegularizationState
    participant EVAL as evaluate.py
    participant FS as outputs/{run}/

    CFG->>FT: load_config()
    FT->>FT: validate (4-bit-in-trainable, 8-bit-optim,<br/>vision-without-projector, KL+modality coupling)
    FT->>US: FastModel.from_pretrained(load_in_4bit=False, bf16)
    US-->>FT: model + tokenizer

    FT->>FT: freeze_vision_audio_keep_projector(+ assert_frozen allowlist)
    FT->>PEFT: LoraConfig(r, alpha, modules_to_save)
    PEFT-->>FT: PeftModel
    FT->>FT: ensure_projector_trainable (belt-and-braces)
    FT->>FT: _assert_no_4bit_in_trainable_full_param_modules

    FT->>REG: build_regularizers(snapshot params)
    Note over REG: ~100 MB bf16 anchor for L2

    FT->>TR: ModalityAwareSFTTrainer(regularization_state, prompt_prefixes)
    Note over TR: compute_loss = parent CE + reg.compute_extra_loss()
    TR->>TR: train loop (KL teacher = disable_adapter())

    TR->>FS: save_steps → adapter checkpoint
    FT->>FT: save tripwire — scan adapter safetensors

    FT->>EVAL: auto-eval after training
    EVAL->>FS: eval.json + eval_per_sample.json
```

The hot path: `prepare_*.sh → finetune.py → export_mlx.py` chained by
`scripts/run/{train, export, sweep_eval_*}.sh`. The
`ModalityAwareSFTTrainer` is a thin subclass of TRL's `SFTTrainer`
that (a) batches by modality (no cross-modal batches), (b) wraps
`compute_loss` with `reg_state.compute_extra_loss()` when KL/L2 are
enabled, and (c) hooks `Trainer.log()` to drain a rolling reg_kl /
reg_l2 metric buffer every `logging_steps`.

### Sweep harness

```mermaid
flowchart LR
    subgraph Cloud["Remote H100/H200"]
        CSH[cloud_sweep_v4_10epoch_sweep.sh]
        CSH --> CQ[queue: configs/cloud_sweep/v4/*.yaml]
        CQ --> CRUN[per-config: train → generality_eval → wandb summary]
        CRUN --> MMD{MMLU ≥ 0.46?}
        MMD -- yes --> KEEP[record + continue]
        MMD -- no --> DROP[MMLU_DROPPED + continue]
    end
    subgraph Local["4090 desktop"]
        LSH[local_sweep_v3_3epoch_sweep.sh]
        LSH --> LQ[queue: configs/local_sweep/*.yaml]
    end
```

The cloud sweep ports the same SOTA recipe (`r=8 / α=8 / no-KL` with
vision-last-2) to the 4090 via local-sweep config. Glob-based queue —
drop a new YAML → re-launch with `STAGE_RESUME=1` and only the new
config runs.

## Core data flow — data mixture (Track A)

```mermaid
sequenceDiagram
    participant CFG as data_mix YAML
    participant BM as build_mix.sh
    participant ORC as mix.py orchestrator
    participant PS as plant_sampler
    participant CS as cambrian_sampler
    participant SS as smoltalk_sampler
    participant NB as negative_builder
    participant LS as llava_sampler
    participant OS as offline_qa_sampler
    participant SCH as schema validator
    participant OUT as data_mix/_local/{mix}/

    BM->>BM: env preflight (HF_HOME, DATASETS_DIR exist)
    BM->>CFG: yaml.safe_load
    CFG->>ORC: resolve env paths + bucket ratios

    par Per-bucket sampling
        ORC->>PS: sample(plant.n, prompt_variants=3)
    and
        ORC->>CS: sample(cambrian.n, non-plant filter)
    and
        ORC->>SS: sample(smoltalk.n, bound to dummy 960×672)
    and
        ORC->>NB: build_negatives(refuse.n, fixed template)
    and
        ORC->>LS: sample(llava.n, optional v2+)
    and
        ORC->>OS: sample(offline_qa.n, persona, v3+)
    end

    ORC->>SCH: validate (strict role alternation, image truthiness)
    SCH-->>ORC: pass / InsufficientPoolError

    ORC->>ORC: class_stratified_split (per-species val)
    ORC->>OUT: train.jsonl + val/{source}.jsonl + manifest
```

Samplers share a 960×672 stretch-resize helper that mirrors
`prepare_plantnet.py`, so visual feature distribution matches training-
time and deploy-time `mlx-swift-lm` (see
[`13-mlx-vision-input-parity.md`](13-mlx-vision-input-parity.md)).

### Mix versions

| Version | Buckets | Configs | Key change |
|---|---|---|---|
| v1 | plant + cambrian + smoltalk + refuse | `mix-200.yaml`, `mix-20k.yaml` | initial 4-bucket scaffold |
| v2 | + LLaVA, + `image=None` text-only path | `mix-200-llava.yaml` | multi-val output (per-bucket val) |
| v3 | + offline_qa persona | (folded into mix-50k variants) | tour-guide voice, no image |
| v4 | per-trail-tuned ratios | `mix-50k.yaml`, `mix-50k-b.yaml`, `mix-50k-c.yaml`, `mix-100k.yaml` | smoltalk +5pp A/B, mixc text variant |

## Core data flow — quantization (Track C)

```mermaid
flowchart TB
    subgraph SFT["SFT artifact"]
        S1["finetune/outputs/{run}/<br/>final_adapter/ + base_bf16/"]
    end

    subgraph Merge["Merge (safetensors-level)"]
        M1["scripts/repair/merge_safetensors.py<br/>(preserves dead KV keys)"]
        M2["base + adapter → merged_bf16/"]
        S1 --> M1 --> M2
    end

    subgraph RouteA["Route A: QLoRA SFT"]
        A1["unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"]
        A2["LoRA r=256 + projector"]
        A1 --> A2
        A2 --> M1
    end

    subgraph RouteB1["Route B.1: CUDA-side PTQ"]
        B1G["gptq.py + gptqmodel<br/>(desc_act, group_size 128)"]
        B1N["bnb_nf4.py<br/>(skip vt+ev mandatory)"]
        B1H["gptq_torchao_hybrid.py<br/>(int4 pack embedding, strip audio)"]
        M2 --> B1G
        M2 --> B1N
        B1G --> B1H
        B1H --> B1OUT["~2.77 GB CUDA artifact<br/>(quality reference)"]
    end

    subgraph RouteB2["Route B.2: MLX-side hybrid PTQ"]
        B2P["prep_inject_kv_shared.py<br/>(repair v5.8 KV drops)"]
        B2D["mlx_hybrid_quant.py<br/>mlx_vlm.load → mlx_lm.quant.* → save_weights"]
        B2E["eora_mlx.py<br/>(eigenspace SVD on quant error)"]
        B2R["eora_post_quant.py<br/>(WikiText calibration)"]
        B2M["merge_eora_into_mlx.py"]
        M2 --> B2P --> B2D
        B2D --> B2E
        B2E --> B2R --> B2M
        B2M --> B2OUT["~3.4 GB MLX artifact + EoRA adapter"]
    end

    subgraph Bridge["B.1 → MLX bridge (queued)"]
        BR1["bridges/hf_gptq_to_mlx.py<br/>(spec only)"]
        B1OUT -.queued.-> BR1
    end

    subgraph Eval["Eval (shared)"]
        E1["src/eval/runner.py"]
        E2["plantnet.py + wikitext_ppl.py"]
        E3["scripts/run/eval.py — multi-loader dispatch"]
        E4["LLM-as-judge (Qwen2.5-VL-72B)"]
    end

    B1OUT --> E3
    B2OUT --> E3
    M2 --> E3
    E3 --> E1
    E1 --> E2
    E2 --> E4

    B2OUT --> IOS["Trailogy iOS bundle<br/>(via fetch-gemma-finetune.sh)"]
```

**Why both routes**: Route A (QLoRA SFT on bnb-4bit base) tests the
direct-tune-on-4-bit hypothesis cheaply (~3× faster wall) and
produces a strong CUDA-side bnb-NF4 result (69.5 % @ 6.52 GB), but
the MLX deploy step collapses badly because `mlx_vlm.convert`'s
data-free affine quantizer can't recover. **Route B.2 (hybrid MLX +
EoRA) closes most of that gap** — EoRA r=64 closes +4.3 pp on M2
subset (83.7 % → 88.0 %, within bf16 noise) with zero gradient descent.

See [`../quantization/README.md`](../quantization/README.md) for the
per-route detail.

## Core data flow — eval (shared across tracks)

```mermaid
sequenceDiagram
    participant CLI as scripts/run/eval.py
    participant LOAD as eval/model_loaders.py
    participant RUN as eval/runner.py
    participant PN as eval/plantnet.py
    participant WT as eval/wikitext_ppl.py
    participant J as judge.py (Qwen2.5-VL-72B)
    participant OUT as outputs/{run}/eval.json

    CLI->>LOAD: dispatch loader (hf_bf16 / hf_gptq / hf_bnb_nf4 /<br/>hf_gptq_hybrid / mlx_vlm)
    LOAD-->>RUN: model + tokenizer + processor

    par PlantNet eval
        RUN->>PN: load val records
        PN->>PN: build_eval_prompt (+ [camera=on] for image asks)
        PN->>RUN: generate per sample
        RUN->>PN: species_match (exact) + ROUGE-L
        PN-->>RUN: per_sample.json + summary
    and Generality eval
        RUN->>RUN: plant_100 + mmlu_50 + aime_20
        Note over RUN: 170 deterministic samples
    and (Optional) WikiText PPL
        RUN->>WT: lm_eval-style ppl on language-only tree
    end

    opt LLM-as-judge
        RUN->>J: device_map=auto, Qwen2.5-VL-72B
        J-->>RUN: per-sample accuracy / richness / hallucination
    end

    RUN->>OUT: eval.json + eval_per_sample.json
```

Same `eval.py` runs on `hf_bf16`, `hf_gptq`, `hf_bnb_nf4`,
`hf_gptq_hybrid`, and `mlx_vlm` loaders — single CLI, dispatched by
loader name. The v4 `[camera=on/off]` prefix is wired into both
PlantNet eval (`300d52e`) and text-only generality domains (`e98ec18`)
so v4-trained checkpoints aren't evaluated under a v3 prompt shape.

## Anti-forgetting stack (Track B v3+)

```mermaid
flowchart LR
    subgraph Data["Data-level (Track A)"]
        D1["Bucket dilution<br/>45% plant + 30% cambrian +<br/>15% smoltalk + 10% refuse"]
    end
    subgraph Weight["Weight-level (Track B v3)"]
        W1["KLPenalty<br/>teacher = disable_adapter()<br/>no second model load"]
        W2["WeightL2Anchor<br/>bf16 snapshot at init<br/>~100 MB"]
    end
    subgraph Input["Input-level (Track B v4)"]
        I1["[camera=on/off]<br/>modality-state prefix gate<br/>(replaces v3 [task=*])"]
    end
    subgraph EvalLevel["Eval-level"]
        E1["Generality eval<br/>plant_100 + mmlu_50 + aime_20<br/>MMLU drop gate ≥ 0.46"]
    end

    D1 --> Train
    W1 --> Train
    W2 --> Train
    I1 --> Train[(SFT step)]
    Train --> E1
    E1 -. feedback .-> D1
    E1 -. feedback .-> W1
```

**Three independent attack surfaces on catastrophic forgetting**:

1. **Data**: dilute the plant signal at training-data assembly.
2. **Weight**: KL anchor (teacher = same model under
   `disable_adapter()` — zero extra GPU memory) + L2 anchor (param-
   at-init snapshot — for LoRA delta params ≈ weight decay; for
   `modules_to_save` full-rank params it's genuine EWC-style "stay
   close to pretrained").
3. **Input**: gate the fine-tune behavior on a modality-state prefix
   the model learns to associate with task. iOS app applies the
   matching prefix at inference.

Eval (the 4th surface) closes the loop with a deterministic 170-sample
generality eval after every cloud sweep run; `MMLU < 0.46` configs get
`MMLU_DROPPED` in the summary. See
[`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
for the detailed design and why rank=8/α=8/no-KL won.

## Tripwire inventory

| Tripwire | Where | Catches |
|---|---|---|
| `assert_frozen` allowlist | `freeze.py` | non-allowlisted frozen-token params un-frozen |
| `ensure_projector_trainable` / `ensure_vision_layers_trainable` | `freeze.py` | PEFT silently drops `modules_to_save` |
| `_assert_no_4bit_in_trainable_full_param_modules` | `finetune.py` | `bnb.Params4bit` in trainable (breaks projector tuning) |
| `_assert_no_8bit_optim` | config validator | `adamw_8bit` bleeding into bf16 SFT |
| Save tripwire | `finetune.py` post-save | adapter safetensors header missing tuned tensors |
| Export tripwire (vision tower) | `export_mlx.py` | `AutoModelForCausalLM` drops `vision_tower` / `embed_vision` |
| Export tripwire (projector-changed byte-diff) | `export_mlx.py` | PEFT silently fails to restore `modules_to_save` |
| `inspect_vision_dtype` | `quantization/scripts/inspect/` | `vision_tower` bf16 invariant violated |
| Orphan-tensor smoke | `save_reload_check.py` | `transformers` k_proj/v_proj drop on KV-shared layers (see [`15-postmortems.md`](15-postmortems.md) §1) |
| `_mlx_dir_has_vision_weights` | `export_mlx.py` | `mlx_lm.convert` (language-only) used instead of `mlx_vlm.convert` |
| YAML config round-trip test | `tests/test_run_quant_yaml.py` | YAML knobs that aren't actually loaded |
| `CalibrationDataLeakError` | `quantization/src/common/calibration.py` | calibration overlap with eval set |
| `prep_inject_kv_shared` preflight | `prep_inject_kv_shared.py` | `mlx_vlm.load` "Missing 60 parameters" on v5.8 saves |

## Deploy artifact → iOS bundle

The final MLX checkpoint lands at
`Trailogy/HikeCompanion/Resources/Models/Gemma/` via the iOS app's
`scripts/fetch-gemma-finetune.sh` (gated HF download). The iOS app
then:

1. Auto-backs-up the stock model to `Models/Gemma.stock/` on first
   swap (so reverting is one `mv`).
2. Patches `processor_config.json` to `size: 960×672` (the trained
   pooler shape — see [`13-mlx-vision-input-parity.md`](13-mlx-vision-input-parity.md)).
3. Loads via `mlx-swift-lm` in `.vlm` mode (~2.78 GB MLX active +
   ~3.54 GB peak during VLM prefill).
4. Emits the `[camera=on]` SFT data-prefix gate on every image Ask —
   matches the Track B v4 training-time input distribution.

The deploy size budget is locked by the **~4 GB iOS jetsam ceiling**.
`mlx-community/gemma-4-e2b-it-4bit` ≈ 3.58 GB on disk; our SFT'd MLX
g128 artifact lands ~3.3 GB; with the EoRA adapter folded in,
deployable artifact is ~3.4 GB. Quantization memory math (bytes/weight
derivation, why g128 is the sweet spot for the iOS budget) is in
[`03-memory-management.md#quantization-memory-math`](03-memory-management.md#quantization-memory-math).

## Technology stack

| Layer | Technology |
|---|---|
| Training framework | `transformers ≥ 5.8` + `peft 0.19.1` + `trl 1.4.0` + `unsloth 2026.5.2` |
| 4-bit base loader | `bitsandbytes` (Route A QLoRA only) |
| CUDA PTQ | `gptqmodel 7.0.0` (Marlin / ExLlamaV2 / Machete kernels) |
| CUDA PTQ — embedding pack | `torchao` (`IntxWeightOnlyConfig` int4 g128) |
| MLX deploy quant | `mlx_vlm` (`mlx_vlm.convert -q`) |
| MLX research quant | `mlx_lm.quant.{gptq, awq, dwq, dynamic_quant}` |
| MLX post-quant adapter | EoRA (pure MLX, no PyTorch dep) — `eora_mlx.py` |
| Optimizer | `adamw_torch_fused` (8-bit optim REJECTED at config-validate) |
| Eval — exact | species_match + ROUGE-L + WikiText PPL |
| Eval — generality | plant_100 + mmlu_50 + aime_20 (deterministic) |
| Eval — judge | Qwen2.5-VL-72B (device_map=auto, structured JSON) |
| Sweep infra | bash glob queues + wandb auto-stream + MMLU drop gate |

See [`14-package-versions-and-known-bugs.md`](14-package-versions-and-known-bugs.md)
for the tested version matrix and known bugs in each layer.

## Per-module entry points

| Module | Entry point | Inputs | Outputs |
|---|---|---|---|
| `data_mix/` | `build_mix.sh <config>` | YAML + HF/disk sources | `_local/<mix>/train.jsonl` + `val/<source>.jsonl` |
| `finetune/` | `train.sh <config>` | YAML + base HF model + data_mix output | `outputs/<run>/{final_adapter, base_bf16, eval.json, checkpoints/}` |
| `quantization/` | `quant.py --config <yaml>` | merged bf16 model | `results/<method>/<run>/model.safetensors + config.json + processor_config.json` |
| Eval (shared) | `eval.py --config <yaml> --loader <name>` | any saved model + eval set | `eval.json + eval_per_sample.json` |

## Cross-references

- iOS-side architecture: [`02-architecture-ios-app.md`](02-architecture-ios-app.md)
- Memory management (iOS) + quantization memory math: [`03-memory-management.md`](03-memory-management.md)
- Model dev timeline: [`08-dev-timeline-model.md`](08-dev-timeline-model.md)
- Per-module READMEs: [`../data_mix/README.md`](../data_mix/README.md), [`../finetune/README.md`](../finetune/README.md), [`../quantization/README.md`](../quantization/README.md)
- Eval setup: [`10-eval-setup.md`](10-eval-setup.md)
