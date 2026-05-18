# `cloud_sweep/` — representative large-batch configs

Representative LoRA configs for the anti-forgetting trade-off: retain
general capability while learning plants from PlantNet-300K.

No batch sweep runner ships in the public tree. Launch individual configs
with `scripts/run/train.sh`.

## Configs

| yaml                                  | r    | alpha | KL    | Role                                              |
|---------------------------------------|-----:|------:|------:|---------------------------------------------------|
| `r8-a8-nokl.yaml`                     |    8 |     8 |  off  | rank=alpha=8, no KL                               |
| `r8-a8-nokl-no-text-prefix.yaml`      |    8 |     8 |  off  | no `[camera=on/off]` modality prefix              |
| `r8-a8-nokl-vision2.yaml`             |    8 |     8 |  off  | + last-2 vision-tower layers unfrozen             |
| `r8-a16-nokl.yaml`                    |    8 |    16 |  off  | alpha-scaling control (alpha/r = 2.0)             |
| `r8-a16-kl005-t2.yaml`                |    8 |    16 | 0.05  | KL anchoring at temperature 2.0                   |
| `r256-a256-kl005.yaml`                |  256 |   256 | 0.05  | high-capacity LoRA + KL — illustrates mode-collapse risk |

All configs use the same trainer, sampler, optimizer, schedule, and
projector-tuning machinery unless explicitly overridden.

## Why rank 8 / alpha 8 won

PlantNet adaptation in this setup is a small, focused additional task.
Higher-rank LoRA (r=32, r=256) reaches comparable PlantNet accuracy
but is more prone to **mode collapse** — the model starts answering
every prompt in plant-classification style, including "How are you?".
With r=8 + alpha=8 (alpha/r = 1.0), there is less capacity to overwrite
general assistant behavior, and the resulting model passes MMLU /
AIME / refusal evals while still classifying plants well.
