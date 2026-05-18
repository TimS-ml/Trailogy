# Gemma 4 E2B Finetune — Pipeline & Experiment Reports

## TL;DR

- This folder documents a finetuning effort that teaches a multimodal Gemma 4 model to identify plants while preserving general assistant behavior.
- Start with the pipeline and final-recipe docs to understand the end-to-end training flow, then read the projector and vision-mode notes for optional model changes.
- The experiment notes explain why the recipe favors bf16 supervised finetuning, frozen vision/audio towers, projector tuning, small-rank LoRA, and no KL penalty.
- Later ablations show that removing the text-only camera prefix and moving to rank 16 produced the strongest recorded recipe.

Engineering notes and experiment reports for the Gemma 4 E2B bf16
finetune on PlantNet-300K.

## Read First

| Read | Why |
|---|---|
| [`01-pipeline.md`](01-pipeline.md) | End-to-end path from PlantNet data to MLX export. |
| [`03-anti-forgetting-and-final-recipe.md`](03-anti-forgetting-and-final-recipe.md) | Why the final recipe avoided catastrophic forgetting. |
| [`../data_mix/B-mix-50k-v2.md`](../data_mix/B-mix-50k-v2.md) | The corpus that made the recipe work. |

## Optional Detail

- [`02-projector-mode.md`](02-projector-mode.md): opt-in projector tuning.
- [`03-vision-mode.md`](03-vision-mode.md): opt-in vision-tower tuning.
- [`07-anti-forgetting-regularization.md`](07-anti-forgetting-regularization.md): deeper KL/L2 design notes.
- [`09-kl-is-overkill-at-small-rank.md`](09-kl-is-overkill-at-small-rank.md): supporting ablation for dropping KL.
- [`10-no-text-prefix-and-bigger-rank.md`](10-no-text-prefix-and-bigger-rank.md): later ablation record.

## Historical / Debug Notes

- [`04-gemma-tuner-investigation.md`](04-gemma-tuner-investigation.md)
- [`05-sft-sweep-plan.md`](05-sft-sweep-plan.md)
- [`06-bnb-vs-torchao-sft.md`](06-bnb-vs-torchao-sft.md)
- [`08-lr-and-adapter-update-magnitude.md`](08-lr-and-adapter-update-magnitude.md)

## Related

- Silent adapter failure postmortem: [`../general/15-postmortems.md`](../general/15-postmortems.md)
- Package/version notes: [`../general/14-package-versions-and-known-bugs.md`](../general/14-package-versions-and-known-bugs.md)
- Quantization result: [`../quantization/00-quantization-report-pub.md`](../quantization/00-quantization-report-pub.md)
