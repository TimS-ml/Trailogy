# Gemma 4 E2B Finetune — Pipeline & Experiment Reports

## TL;DR

- This folder documents a finetuning effort that teaches a multimodal Gemma 4 model to identify plants while preserving general assistant behavior.
- Start with the pipeline and final-recipe docs to understand the end-to-end training flow, then read the projector and vision-mode notes for optional model changes.
- The experiment notes explain why the recipe favors bf16 supervised finetuning, frozen vision/audio towers, projector tuning, small-rank LoRA, and no KL penalty.
- Later ablations show that removing the text-only camera prefix and moving to rank 16 produced the strongest recorded recipe.

Engineering notes and experiment reports for the Gemma 4 E2B bf16
finetune on PlantNet-300K.

## Files

### Core pipeline + canonical recipe

| File | What it covers |
|---|---|
| [`01-pipeline.md`](01-pipeline.md) | End-to-end baseline pipeline (data prep, bf16 LoRA training, adapter merge, MLX conversion). LoRA-only mode (default). Overfit100 + plantnet-50k empirical results. Data-format ablation (English vs Latin, with vs without wiki descriptions). Package version fix. |
| [`03-anti-forgetting-and-final-recipe.md`](03-anti-forgetting-and-final-recipe.md) | **The shipped recipe** (r=8 / α=8 / no KL, 3 epochs on mix-50k) + the anti-forgetting design that supports it: camera-state prefix gate, KL penalty + L2 weight anchor (designed but disabled), KL-rank sweep, S_step framework, offline_qa persona bucket. |

### Opt-in tuning modes

| File | What it covers |
|---|---|
| [`02-projector-mode.md`](02-projector-mode.md) | Opt-in projector tuning (`tune_projector: true`). Five-step wiring, two tripwires, projector-mode mechanism. Achieves 100% species match on overfit100. |
| [`03-vision-mode.md`](03-vision-mode.md) | Opt-in vision-tower last-N tuning (`tune_last_n_vision_layers > 0`). Six-step wiring, three additional tripwires. |

### Decision / exploration notes

| File | What it covers |
|---|---|
| [`04-gemma-tuner-investigation.md`](04-gemma-tuner-investigation.md) | Why we did not use `gemma-tuner-multimodal` — exploration finding that pushed us to the bespoke pipeline. |
| [`05-sft-sweep-plan.md`](05-sft-sweep-plan.md) | Two-stage sweep plan across r/α/dropout/KL/L2 + naming conventions, run-config matrix, and stop-conditions. |
| [`06-bnb-vs-torchao-sft.md`](06-bnb-vs-torchao-sft.md) | Decision doc: bnb 4-bit (QLoRA, train-VRAM tool) vs torchao QAT (deploy-accuracy tool). Module-level matrix for Gemma 4 VLM. |
| [`07-anti-forgetting-regularization.md`](07-anti-forgetting-regularization.md) | Deep-dive on the anti-forgetting stack: KL output-distribution penalty + L2 weight anchor + camera-state prefix gate. Designed full, shipped pared-down (see `03-anti-forgetting-and-final-recipe.md` for the shipped subset). |
| [`08-lr-and-adapter-update-magnitude.md`](08-lr-and-adapter-update-magnitude.md) | S_step framework — accounting for per-step adapter update magnitude. Why higher rank does not always mean more drift. |
| [`09-kl-is-overkill-at-small-rank.md`](09-kl-is-overkill-at-small-rank.md) | Why KL turned out unnecessary at r=8 / α=8: empirical KL-rank sweep, the "small-rank LoRA is self-anchoring" argument. |
| [`10-no-text-prefix-and-bigger-rank.md`](10-no-text-prefix-and-bigger-rank.md) | Late-stage ablation: dropping the text-side `[camera=off]` prefix + pushing rank to 16. What changes, what does not. |

## Reading order

- New to the pipeline: read `01` first.
- Why the canonical recipe is r=8 / α=8 / no KL: jump to `03`.
- Reviewing the opt-in tuning modes: `02-projector-mode.md` then
  `03-vision-mode.md`.
- Deciding "should we add 4-bit at training time": `01` then
  `06-bnb-vs-torchao-sft.md`.

## Related

| Location | Purpose |
|---|---|
| `src/finetune/scripts/plot_loss.py` | Loss-curve comparison plot generator |
| [`../general/13-mlx-vision-input-parity.md`](../general/13-mlx-vision-input-parity.md) | Companion: inference-side input bug — preprocessing parity story |
| [`../general/15-postmortems.md`](../general/15-postmortems.md) §1 | Root cause investigation of the PEFT orphan-tensor / KV-shared bug |
| [`../general/14-package-versions-and-known-bugs.md`](../general/14-package-versions-and-known-bugs.md) | Verified package versions for the working pipeline |
