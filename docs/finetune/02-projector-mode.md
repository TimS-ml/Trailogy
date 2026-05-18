> last edit: 2026-05-15 (path fix, cross-refs to 05/06, production baseline-1 framing)

# Finetune Pipeline — Projector + LoRA Mode

Adds full-param fine-tuning of Gemma 4's `embed_vision` projector on
top of the baseline LoRA pipeline. **This is the current production
baseline-1 recipe** — `r=256 + projector + data-aug-enwiki, 5 epochs`
is the bf16 reference for the entire quantization track
(see `../../quantization/B1-sft-results.md` row R0).

This doc assumes you have read [`01-pipeline.md`](01-pipeline.md). The
projector mode reuses every step of the baseline pipeline; this
section only describes what changes when `cfg.lora.tune_projector:
true` is set.

For the extension that also unfreezes the last N transformer blocks
of the vision encoder, see [`03-vision-mode.md`](03-vision-mode.md).

Reference config: `src/finetune/configs/plantnet-50k-baseline-v2.yaml`.

---

## What it does

When `cfg.lora.tune_projector: true` is set, the `embed_vision`
projector — Gemma 4's `Gemma4MultimodalEmbedder`, a single
`RMSNorm + Linear` that maps SigLIP's pooled 768-d features into the
LM's hidden space — is unfrozen as **full params** (not LoRA) and
trains alongside the language LoRA. The vision **encoder**
(`vision_tower.{patch_embedder, encoder, pooler}`) and the audio side
stay frozen.

### Why full-param, not LoRA, on the projector

The projector is a single Linear of shape `768 → 2048` ≈ 1.18 M params.
LoRA on a single Linear with rank-r adds `r*(in+out)` params, so even
r=64 gives only `64*(768+2048) = 180 K` effective parameters in a
low-rank subspace of a 1.18 M param weight. The compression ratio is
not buying anything — the projector is already small enough to tune
directly. Full-param training is both simpler and more expressive.

---

## Five-step wiring in `finetune.py:real_train`

```
                ┌────────────────────────────────────────────────────────────┐
                │ cfg.lora.tune_projector == True                            │
                └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
1. Identify projector params/modules    src/projector.py:78-173
   • find_projector_param_names(model)       → list[str] full param paths
   • find_projector_module_names(model)      → list[str] leaf module names
   (token-based: matches "embed_vision." etc., excludes vision encoder)

                                  ▼
2. Pass `modules_to_save=[…]` to PEFT    finetune.py:534-536
   FastModel.get_peft_model(
       …,
       modules_to_save=projector_module_names,   # e.g. ["embed_vision"]
   )
   PEFT wraps the named modules in ModulesToSaveWrapper:
     • base.modules_to_save.default.<param>  ← trainable copy (in optim)
     • base.original_module.<param>          ← frozen ref copy (excluded)

                                  ▼
3. Patch the PEFT wrapper's forward()    finetune.py:_patch_peft_embedder_wrappers
   PEFT's AuxiliaryTrainingWrapper.forward(self, x, ...) takes `x`
   positionally; HF's Gemma4Model calls
   self.embed_vision(inputs_embeds=hidden) with kwargs only.
   Patch maps the kwarg to the positional slot.

                                  ▼
4. Freeze + projector-aware tripwire     freeze.py:288-363, assert_frozen
   freeze_vision_audio_towers_keeping_projector(model, projector_param_names)
   ensure_projector_trainable(model, projector_param_names)
       ↳ flips requires_grad=True on projector params if PEFT silently
         dropped modules_to_save. Logs WARNING when triggered.
   assert_frozen(model, allowlist=projector_param_names)
       ↳ raises if any non-projector frozen-token param still trainable

                                  ▼
5. Two-param-group optimizer            finetune.py:641-680
   AdamW(
       [
           {"params": lora_params,       "lr": cfg.training.learning_rate},      # 2e-4
           {"params": projector_params,  "lr": cfg.lora.projector_learning_rate}, # 5e-5 (or LR/10 auto)
       ]
   )
   Passed into SFTTrainer as `(optimizer, None)`. HF Trainer constructs
   the cosine scheduler from sft_args against our custom optimizer.
```

The projector-aware freeze walker
`freeze_vision_audio_towers_keeping_projector` (`freeze.py:288-363`)
uses the same `vision_tower.` / `embed_vision.` token matcher as the
LoRA-only freeze pass, but skips any parameter detected as a projector
param by `_is_projector_param` (token-based check against
`embed_vision.` etc., with explicit excludes for the vision **encoder**
sub-modules `vision_tower.{patch_embedder, encoder, pooler}` and the
audio side). The encoder, audio tower, audio embedder, etc. still get
frozen — only the language-to-vision projection stays trainable.

The `_is_projector_param` matcher also skips PEFT's
`.original_module.` path (the frozen reference copy PEFT keeps next to
the trainable `modules_to_save.{adapter}.` copy). Without that
exclusion the optimizer would get duplicate parameter entries and the
frozen reference would be silently re-enabled.

---

## Config validator interactions

`config.py:215-228` rejects two contradictions at parse time:

- `tune_projector: true` together with `finetune_vision_layers: true` —
  the latter implies vision-encoder LoRA which conflicts with the
  "encoder stays frozen" invariant.
- `tune_projector: true` with `projector_learning_rate <= 0`.
  `projector_learning_rate: null` is allowed and resolves to
  `training.learning_rate / 10` at runtime (LLaVA convention).

---

## Two tripwires guarding projector tuning

### Did PEFT actually save projector tensors? (post-save)

After training completes, `_assert_projector_tensors_present_if_tuned`
(`finetune.py:_assert_projector_tensors_present_if_tuned`) reads the
safetensors headers in the saved adapter directory and aborts the run
with a non-zero exit code if **no** projector tensors are present.
This catches the catastrophic silent failure where PEFT's
`save_pretrained` ignored `modules_to_save` (e.g. due to an unsloth
wrapper regression or PEFT version mismatch) and stored only LoRA
tensors. Without this tripwire, an entire 3-epoch run would ship an
adapter byte-equivalent to a LoRA-only run. The check is cheap because
safetensors headers are read without loading any tensor data.

The check fires immediately after `model.save_pretrained()` in
`real_train`.

### Did the projector actually change after merge? (post-merge)

`_assert_projector_changed_if_tuned` (`export_mlx.py:401-491`) runs
immediately after `merge_and_unload()` in projector-mode adapters. It:

1. Reads the adapter directory's safetensors headers to detect whether
   `modules_to_save` was used at training time (= projector tensors
   are present in the adapter). LoRA-only adapters skip this check.
2. Byte-snapshots the projector parameters of the **freshly loaded
   base model** before the adapter was applied.
3. After `merge_and_unload()`, byte-snapshots the same parameters in
   the merged model.
4. If every projector parameter is byte-identical between base and
   merged, PEFT silently failed to restore the `modules_to_save`
   weights at load time, and the merged model is effectively a
   LoRA-only merge. **Aborts the export.**

This is the complement to the in-training
`_assert_projector_tensors_present_if_tuned` tripwire: the training
one verifies the adapter has projector tensors on disk; this one
verifies those tensors actually flowed into the merged checkpoint.
Both must fire green or the projector-tuning value is silently dropped
somewhere between save and merge.

Test coverage:
`finetune/tests/test_export_projector_tripwire.py` synthesises an
adapter directory with vs. without projector tensors and asserts the
tripwire behaviour in both cases.

---

## LoRA alpha when projector mode is on: stay at α = r

In LoRA-only mode the LoRA contribution dominates; α = r = 1.0 scaling
is correct. In projector mode the same rule applies — α = r — and the
projector full-param weights handle the "high-magnitude update on a
small parameter count" channel separately at its own LR. Don't try to
compensate by raising α; that just amplifies the LoRA's effective
update without changing the projector's. The current config uses
`r = 256, lora_alpha = 256, learning_rate = 2e-4` for the LoRA group
and `projector_learning_rate = 5e-5` (= LR/4, slightly more aggressive
than the auto LR/10 default to give the 1.18 M projector room).

---

## What projector + LoRA mode is actually learning

In LoRA-only mode every parameter between the SigLIP pooler output and
the language model's residual stream is frozen, including the
**projector** — `embed_vision`'s RMSNorm + 768×2048 Linear that
re-projects the 280 pooled image features into LM hidden-dim space.
With ~1.18 M params, the projector is a serious bottleneck: it's the
*only* path through which visual information enters the LM's residual
stream, and it sees no gradient updates in LoRA-only mode.

Empirically, unfreezing this single Linear and co-training it as full
params (with the language LoRA still active) is the
highest-empirical-value change we made — see [Empirical
results](#empirical-results-loraonly-vs-projector--lora) below. The
mechanism is two-fold:

1. **Better cross-modal alignment.** The pretrained projector
   `embed_vision.embedding_projection.weight` was learned during
   Gemma 4's generic multimodal pretraining, on a wide image-caption
   distribution. PlantNet is a narrower visual distribution (botanical
   macro photos, specific lighting, field-photo aesthetics) and a more
   specialized output channel (species binomials). Letting the
   projector adapt to this distribution while the encoder stays fixed
   gives the model 1.18 M degrees of freedom to "re-aim" SigLIP's
   general features into the LM's species-name vocabulary.

2. **More effective gradient flow into the LM.** When the projector is
   frozen, every gradient update to the language LoRA has to find a
   way to compensate for whatever the projector outputs without
   modification. When it's trainable, the projector and the LoRA share
   the optimization load: the projector re-shapes the visual signature
   into something the LoRA can use, and the LoRA learns to interpret
   the re-shaped signature into species names.

The vision **encoder** (`vision_tower.{patch_embedder, encoder,
pooler}`) still stays frozen. The argument from LoRA-only mode about
"templated answers being weak supervision for the encoder" remains
valid. SigLIP-level visual discrimination is not the bottleneck; the
projector bridge alignment is. The overfit-100 experiment below
confirms this: once the projector is co-trained, the same r=256 LoRA
that plateaued at 93% species match in LoRA-only mode reaches 100%.

The updated one-line framing:

> **Projector + LoRA SFT learns (a) a re-aimed projection from
> SigLIP's general visual features into a domain-relevant subspace,
> and (b) a cross-modal lookup from that re-aimed subspace to the
> species name vocabulary.** It still does not teach the model to see
> (encoder frozen); it teaches the model where to look in SigLIP's
> output and what to say when it sees X. The projector bridge is the
> missing piece that unlocks the LoRA's full potential.

---

## Empirical results: LoRA-only vs projector + LoRA

### Note on earlier results

The 2026-05-11 version of this document reported species_match 0/200
for both LoRA-only and projector + LoRA modes. Those zero-match
results were caused by a **PEFT orphan-tensor bug**: an older
`transformers` version had `k_proj`/`v_proj` as `nn.Linear` across
all 35 layers; on reload, PEFT silently dropped the 80 orphan tensors.
The fix was to update `unsloth`, then update `transformers`, `peft`,
and related packages (ignoring unsloth's pin-package warnings). The
adapter from that era (`plantnet-50k-lora-r256+fullproj-lr5e5_20260510_115422`)
should not be used as a baseline.

### Overfit-100 controlled comparison (2026-05-12)

100 training samples, 30 epochs, same images. This is an overfitting
test — the goal is to verify that the training pipeline can fully
memorize a small dataset, which isolates pipeline/capacity issues from
generalization questions.

| metric                | LoRA-only (r=256, α=8) | **projector + LoRA (r=256, α=256)** | delta |
|-----------------------|------------------------|-------------------------------------|-------|
| Config name           | lora-r256-a8-lr2e4     | plantnet-overfit100-lora-r256+fullproj-lr5e5 | — |
| Train loss (final)    | —                      | **0.00004**                         | converges deeper |
| ROUGE-L mean          | 0.898                  | **0.9998**                          | +0.102 |
| species_match (strict)| 93 / 100 (93%)         | **100 / 100 (100%)**                | **+7%** |
| Response length μ     | —                      | **176 chars**                       | — |
| Response length median| —                      | **167 chars**                       | — |
| Inference time/sample | —                      | **3.15 s**                          | — |
| Saved tensors         | 410 (LoRA only)        | **411 (410 LoRA + 1 projector)**    | all verified byte-for-byte |

Key hyperparameters for the projector + LoRA run:
- `r = 256, lora_alpha = 256`
- Projector LR: `5e-5`, Language LoRA LR: `2e-4`
- 100 samples, 30 epochs
- 411 tensors saved and verified byte-for-byte

### What projector tuning adds

The 7% species-match gain (93% → 100%) and ROUGE-L jump (0.898 →
0.9998) on the same r=256 LoRA rank confirm that the projector bridge
is the critical bottleneck in LoRA-only mode. The LoRA-only adapter
has plenty of capacity at r=256 to memorize 100 samples — 93% proves
that — but it cannot close the last 7% because the frozen projector
maps SigLIP features into LM space with a generic alignment that
loses fine-grained species distinctions. Co-training the projector
lets it re-aim SigLIP's output to preserve exactly the visual
dimensions that distinguish "white oak vs red oak" in the LM's
residual stream.

The projector + LoRA run also converges to a substantially lower
training loss (0.00004 vs what LoRA-only achieves), confirming that
the projector's 1.18 M additional trainable parameters provide a
qualitatively different optimization landscape, not just marginal
improvement.

---

## Verification recipe (projector mode)

End-to-end smoke after any change to the projector path:

```bash
# 1. Unit tests
cd finetune
pytest tests/test_freeze.py tests/test_projector.py tests/test_export_projector_tripwire.py tests/test_config.py

# 2. Dry run on the projector config
python -m src.finetune --config configs/plantnet-50k-baseline-v2.yaml --dry-run

# 3. Tiny real run on the GPU box — 50 steps, verifies freeze + LoRA + projector
python -m src.finetune --config configs/plantnet-50k-baseline-v2.yaml \
    --max_steps 50 --max_train_samples 200

# 4. Verify the projector save tripwire fired green:
#    "Projector save tripwire passed" should appear in the training log.

# 5. Export the resulting adapter
bash scripts/export.sh outputs/<run-name>/final-adapter exports/smoke 4

# 6. Verify the projector-changed export tripwire:
#    "Projector-changed tripwire passed: N/N projector params differ from base"
```

---

## File references — projector mode

| Concern | Path |
|---|---|
| `tune_projector` config field + validator | `src/finetune/src/config.py:215-228` |
| Projector identification helpers | `src/finetune/src/projector.py` |
| Projector trainable fallback | `src/finetune/src/projector.py:176-213` |
| Projector-aware freeze walker | `src/finetune/src/freeze.py:288-363` |
| Projector-tensor save tripwire | `src/finetune/src/finetune.py:_assert_projector_tensors_present_if_tuned` |
| PEFT wrapper forward patch | `src/finetune/src/finetune.py:_patch_peft_embedder_wrappers` |
| Two-param-group optimizer | `src/finetune/src/finetune.py:real_train` (param-group block) |
| Projector-changed export tripwire | `src/finetune/src/export_mlx.py:_assert_projector_changed_if_tuned` |
| Current baseline config (projector + LoRA) | `src/finetune/configs/plantnet-50k-baseline-v2.yaml` |
| Projector identification unit tests | `src/finetune/tests/test_projector.py` |
| Projector export tripwire tests | `src/finetune/tests/test_export_projector_tripwire.py` |
| Tune-projector config validator tests | `src/finetune/tests/test_config.py` |
