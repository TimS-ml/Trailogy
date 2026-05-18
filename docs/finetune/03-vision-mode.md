> last edit: 2026-05-15 (path fix, 2026-05-15 prior revision under §"When vision-tower tuning should become valuable")

# Finetune Pipeline — Projector + LoRA + Vision-Tower Last-N Mode

Adds full-param fine-tuning of the **last N transformer blocks** of
the SigLIP vision encoder on top of the projector + LoRA pipeline.

This doc assumes you have read both [`01-pipeline.md`](01-pipeline.md)
and [`02-projector-mode.md`](02-projector-mode.md). The vision-tower
mode stacks on top of projector tuning — the validator requires
`cfg.lora.tune_projector: true` alongside
`cfg.lora.tune_last_n_vision_layers > 0`.

**Status:** Works correctly after the package version fix (see
the PEFT orphan-tensor bug note in the project write-up).
The first attempt (2026-05-11) mode-collapsed due to that bug;
with correct packages (2026-05-12) the mode trains and saves cleanly,
but **projector-only remains strictly better on overfit100**
(100 % vs 96 % species match — see §Empirical results).

At production scale this mode is now treated as a **default-negative
probe**, not a likely winner. See the §"When vision-tower tuning should
become valuable — revised 2026-05-15" subsection in this doc, and
`../../quantization/B1-bnb-nf4-vision-collapse.md` for the
companion PTQ-side evidence that the SigLIP tower is fragile in
both directions (training-time drift and post-training quantization).

Reference config:
`finetune/configs/plantnet-overfit100-lora-r256+fullproj+vision2-lr1e5.yaml`.

---

## What it does

Unfreezes the **last N** transformer blocks of
`vision_tower.encoder.layers` as full params via PEFT's
`modules_to_save`. Gemma 4 E2B has 16 such layers
(`Gemma4VisionConfig.num_hidden_layers = 16`); N = 2 trains
`layers.14` and `layers.15` — the most semantic / task-specific SigLIP
layers. About 14 M trainable params for N=2 (each
`Gemma4VisionEncoderLayer` carries ~7 M params: 4 attention projections
+ 2 MLP projections + norms).

Earlier vision encoder layers (0..total−N−1), `patch_embedder`, and
`pooler` stay frozen. The projector and language LoRA continue
unchanged. Audio stays frozen.

---

## Six-step wiring in `finetune.py:real_train`

```
                ┌────────────────────────────────────────────────────────────┐
                │ cfg.lora.tune_last_n_vision_layers > 0                     │
                │   (and cfg.lora.tune_projector == True — enforced)         │
                └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
1. Discover vision encoder layers       src/vision_layers.py
   • find_vision_encoder_layer_count(model)       → highest layers.{i}+1 (16)
   • find_last_n_vision_layer_module_names(...)   → ["vision_tower.encoder.layers.14",
                                                     "vision_tower.encoder.layers.15"]
   • find_vision_layer_param_names(model, n)      → list[str] full param paths
   Note: returned module suffix is the disambiguating full path
         (vision_tower.encoder.layers.N), NOT the bare leaf "layers.N",
         to avoid colliding with text decoder's model.layers.N.

                                  ▼
2. Stack `modules_to_save` with projector entries    finetune.py:peft_kwargs
   modules_to_save = (
       projector_module_names               # ["embed_vision"]
       + vision_layer_module_names          # ["vision_tower.encoder.layers.14",
                                            #  "vision_tower.encoder.layers.15"]
   )
   PEFT wraps each entry in ModulesToSaveWrapper independently.

                                  ▼
3. Token-based identification under PEFT wrapping     vision_layers.py +
                                                       finetune._is_vision_layer_param_name
   Regex (?:^|\.)vision_tower\.encoder\.layers\.(\d+)(?:\.|$)
   matches at a path-component boundary under arbitrary wrapping depth.
   Excludes `.original_module.` frozen reference copies — same dual-path
   discipline as the projector path.

                                  ▼
4. Freeze pass with combined allowlist                freeze.py:freeze_vision_audio_towers_keeping_projector_and_vision_layers
   Skips both projector params AND params under tuned vision layer
   indices when freezing. Vision encoder layers 0..(total-N-1),
   patch_embedder, pooler, and the entire audio side stay frozen.
   ensure_vision_layers_trainable is the belt-and-braces fallback
   (flips requires_grad=True if PEFT silently dropped modules_to_save).

                                  ▼
5. Three-group optimizer                              finetune.py
   AdamW8bit([
       {"params": lora_params,    "lr": training.learning_rate},      # 2e-4
       {"params": proj_params,    "lr": projector_learning_rate},     # 5e-5 (LR/10 auto)
       {"params": vision_params,  "lr": vision_layers_learning_rate}, # 1e-5 (LR/20 auto)
   ])
   Order of classification matters: vision-layer first (most specific),
   then projector, then everything else into the LoRA group.
   `vision_layers_learning_rate` is lower than the projector LR because
   pretrained SigLIP weights are more fragile than the freshly-needed
   projector adaptation (LLaVA-style late-encoder convention).

                                  ▼
6. Three layered tripwires (in addition to the existing projector ones)
   • _assert_vision_layer_tensors_present_if_tuned (post-save):
     adapter dir must contain at least one tensor per tuned layer index.
   • _snapshot_trainable_params + _assert_trainable_set_unchanged_post_train
     (immediately after trainer.train() returns): same exact set of param
     names must be trainable post-train as pre-train. Catches
     resume-from-checkpoint silently dropping modules_to_save mid-run.
   • _assert_vision_layers_changed_if_tuned (post-merge, export_mlx.py):
     byte-snapshots base vision-layer params before adapter load, then
     byte-diffs after merge_and_unload(). Each tuned layer must have at
     least one param that changed.

   All three are no-ops when tune_last_n_vision_layers == 0.
   Same bf16-safe byte-comparison helper used as the projector tripwire
   (_tensor_to_comparable_bytes — casts bf16 → fp32 only for the
   comparison representation, because tensor.numpy() rejects bf16
   directly).
```

---

## Config validator interactions (`config.py`)

- `tune_last_n_vision_layers > 0` AND `tune_projector: false` →
  validator error. (Moving visual features while the projector stays
  frozen creates an alignment problem the language LoRA can't fix.)
- `tune_last_n_vision_layers > 0` AND `finetune_vision_layers: true` →
  validator error. (The Unsloth flag injects LoRA into ALL vision
  layers, conflicting with the "selectively unfreeze last N as full
  params" semantics.)
- `tune_last_n_vision_layers < 0` → validator error.
- `vision_layers_learning_rate ≤ 0` when set → validator error.
- `vision_layers_learning_rate: null` (default) → auto-resolves to
  `training.learning_rate / 20` at runtime.

---

## Why N=2 was chosen (design rationale at the time of the experiment)

Two motivations point to "few late layers":

1. **Late-layer semantics.** In a SigLIP-style transformer, late blocks
   carry the most task-discriminative features (analogous to ResNet
   block-4 vs block-1). Tuning the last 1–2 is the standard recipe in
   late-layer encoder finetuning literature (BERT, CLIP, ViT all behave
   similarly).
2. **Bounded blast radius.** Vision encoder pre-training is expensive
   to replicate; the design wanted to preserve as much of SigLIP's
   generic discrimination capability as possible. Unfreezing only the
   last 2 bounds how far the encoder can drift from its pretrained
   state.

---

## Empirical results: overfit100 (2026-05-12, post-fix)

After fixing the PEFT orphan-tensor bug (updated unsloth, transformers,
peft), the vision-tower mode trains cleanly with no
mode collapse. However it does not outperform projector-only.

### Config

`plantnet-overfit100-lora-r256+fullproj+vision2-lr1e5`

| Field | Value |
|---|---|
| `lora.r` / `lora_alpha` | 256 / 256 (α/r = 1.0) |
| `lora.learning_rate` | 2e-4 |
| `lora.tune_projector` | true |
| `lora.projector_learning_rate` | 5e-5 |
| `lora.tune_last_n_vision_layers` | 2 (layers 14, 15) |
| `lora.vision_layers_learning_rate` | 1e-5 |
| `training.num_train_epochs` | 30 |
| Samples | 100 |

### Results

| metric | proj + LoRA + vision-2 |
|---|---|
| Saved tensors | 493 (410 LoRA + 1 projector + 82 vision layers 14–15) |
| Tripwires | All passed (save, projector tensor, vision layer tensor) |
| Final train loss | 0.00007 |
| ROUGE-L | 0.960 |
| Species match (strict) | 96 / 100 (96%) |
| Training time | ~10 min |

### Comparison with projector-only baseline

| mode | species match | ROUGE-L | training time |
|---|---|---|---|
| r256+fullproj-lr5e5 (projector-only) | **100 / 100 (100%)** | **0.9998** | **~5.5 min** |
| r256+fullproj+vision2-lr1e5 | 96 / 100 (96%) | 0.960 | ~10 min |

Vision-tower tuning **reduces** performance on 100 samples (96% vs
100% species match) and takes roughly **2x longer** due to the vision
encoder backward pass.

---

## Analysis

### The 2026-05-11 "Sedum X" mode collapse

The first attempt (2026-05-11) produced catastrophic mode collapse
where every prediction became "Sedum X" regardless of input. That
failure occurred on a package stack affected by the PEFT orphan-tensor
bug: the adapter was trained on an older transformers that had
`k_proj`/`v_proj` as `nn.Linear` in all 35 layers; on reload, PEFT
silently dropped 80 "orphan" tensors. The mode collapse was likely
caused by the **combination** of the PEFT bug and aggressive vision LR
— the bug corrupted the adapter state, and the vision encoder drifted
into a degenerate fixed point. With correct packages, no mode collapse
occurs.

### Why vision-tower tuning doesn't help on 100 samples

On 100 samples, vision-tower tuning adds parameters (82 tensors,
~14 M params) without enough data to benefit. The projector-only mode
(1.18 M projector params + LoRA) is sufficient to memorize the mapping
for 100 images — it achieves 100% species match. The additional vision
encoder capacity has nothing useful to learn and slightly hurts by
introducing noise into the pretrained SigLIP representations.

### When vision-tower tuning should become valuable

Vision-tower tuning is expected to become valuable on **larger datasets
(50k+)** where the frozen SigLIP features may not discriminate
fine-grained species well enough. At that scale, the projector alone
may not have enough capacity to remap frozen vision features into the
species-discriminative space the language model needs. The 100-sample
overfit test is too small to demonstrate this benefit — it tests
memorization, not generalization.

### When vision-tower tuning should become valuable — revised 2026-05-15

The optimistic prior above is the 2026-05-12 reading. As of 2026-05-15
the prior has been revised, and the section above is retained only as
a historical record of the earlier framing. Two pieces of evidence
flip the expected sign of `tune_last_n_vision_layers > 0` at
production scale:

1. **Gemma 4 ships with an undocumented vision decoder.** Only the
   last two transformer blocks of `vision_tower.encoder.layers` are
   safe to touch under any production training setup; earlier layers
   feed into pretrained behavior the public Gemma 4 release does not
   reveal enough about to reason over. This is why `N=2` is a hard
   design ceiling here, not just a default.

2. **The overfit100 −4 pp is the signal, not noise.** The
   reading that "n=100 is too small to demonstrate the benefit"
   (from the 2026-05-12 framing) underweights how clean the
   memorization test is. overfit100 is a pure capacity / loss-landscape
   probe: it asks whether adding 14 M trainable params makes
   memorizing 100 images easier. The fact that it makes memorization
   *harder* (96 % vs 100 % species match, ROUGE-L 0.960 vs 0.9998,
   and ~2× wall) is **direct evidence that touching the SigLIP late
   layers hurts the cross-modal mapping the projector + language
   LoRA build on top**. A capacity addition that hurts a memorization
   test is unlikely to be rescued by more data.

3. **Companion PTQ-side evidence: vision tower is fragile at deploy
   time too.** bnb-NF4 on the SigLIP encoder collapses PlantNet
   match from ~70 % → 0.1 % even on a checkpoint that never touched
   the vision side at training time. See
   `../../quantization/B1-bnb-nf4-vision-collapse.md`. The
   tower is brittle in both directions — under SFT updates and
   under post-training quantization. Treating it as a fixed sensor
   that the language side has to adapt to is the consistent
   throughline.

Implications:

- Vision-late-layer tuning is a **default-negative probe**, not a
  default-on mode. Treat any production run with
  `tune_last_n_vision_layers > 0` as a controlled experiment whose
  prior expectation is regression vs the projector-only baseline.
- Vision-side capacity expansion beyond `N=2` (deeper unfreeze,
  LoRA-on-SigLIP, high-res training) is also default-off — same
  fragility argument, no upper bound on the downside.
- **The asymmetric-risk argument is the deciding factor.** Even if
  some real upside exists from vision-late-layer tuning at 50k+
  samples, it is bounded above by the undocumented-decoder
  constraint at N=2. The downside (cross-modal misalignment) is
  bounded by neither the layer cap nor the dataset size.

This revision does not invalidate the section above as engineering
documentation — the wiring, tripwires, three-group optimizer, and
validator rules all stand. It updates the *expected sign* of running
this mode on production-scale data.

### Training time cost

The 2x training time increase comes from computing gradients through
the vision encoder backward pass (SigLIP's 16-layer transformer). This
means vision-tower tuning should only be used when there's evidence the
frozen encoder is the bottleneck — not as a default mode.

### LR selection for vision layers

Vision-tower LR selection remains important. Pretrained SigLIP weights
are fragile; the convention is to use LR/10 to LR/100 relative to the
LoRA LR. The current default (LR/20 = 1e-5 when LoRA LR = 2e-4) is
reasonable but should be validated on larger datasets. On 50k+ samples
with many more training steps, even 1e-5 could accumulate large updates
— monitoring the vision encoder's representation drift (e.g., via CKA
or linear probing) would be prudent.

---

## File references — vision-tower mode

| Concern | Path |
|---|---|
| `tune_last_n_vision_layers` / `vision_layers_learning_rate` config + validator | `src/finetune/src/config.py` |
| Vision-tower last-N tuning helpers | `src/finetune/src/vision_layers.py` |
| Projector-and-vision-layers freeze walker | `src/finetune/src/freeze.py:freeze_vision_audio_towers_keeping_projector_and_vision_layers` |
| Vision-layer save tripwire | `src/finetune/src/finetune.py:_assert_vision_layer_tensors_present_if_tuned` |
| Pre/post-train trainable-set check | `src/finetune/src/finetune.py:_snapshot_trainable_params`, `_assert_trainable_set_unchanged_post_train` |
| Three-param-group optimizer | `src/finetune/src/finetune.py:real_train` (param-group block) |
| bf16-safe tensor byte snapshot | `src/finetune/src/export_mlx.py:_tensor_to_comparable_bytes` |
| Vision-layer-changed export tripwire | `src/finetune/src/export_mlx.py:_assert_vision_layers_changed_if_tuned` |
| Vision-tower last-N config | `src/finetune/configs/plantnet-overfit100-lora-r256+fullproj+vision2-lr1e5.yaml` |
| Vision-layer identification unit tests | `src/finetune/tests/test_vision_layers.py` |
| Vision-layer export tripwire tests | `src/finetune/tests/test_export_vision_layer_tripwire.py` |
| Vision-layer save-tripwire tests | `src/finetune/tests/test_finetune_save_tripwire.py` |
| Pre/post-train trainable-set check tests | `src/finetune/tests/test_finetune_trainable_set_check.py` |

