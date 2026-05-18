# `local_sweep/` — single-GPU training configs

Single-GPU configs using the same training pipeline as the larger-batch
configs, adjusted for smaller memory budgets.

Hardware assumption:
* `r8-a8-nokl-local.yaml`, `r8-a8-nokl-vision2-local.yaml` — 24 GB
  consumer GPU (e.g. RTX 4090 desktop).
* `r8-a8-nokl-laptop.yaml` — 16 GB consumer GPU (e.g. RTX 4090 laptop);
  uses `per_device_train_batch_size=1` + grad-accum to hit the same
  effective batch size as the desktop config.

KL anchoring (`regularization.kl_enabled: true`) is **disabled** in
every config here because the KL trainer's teacher-forward pass
roughly doubles activation memory and OOMs on the 24 GB / 16 GB
budget at the projector-tuned, 960x672 vision-encoder configuration.
Anti-forgetting here relies entirely on the data-mix ratio
and `lora_dropout` (stochastic anchoring) instead.

No batch sweep runner ships in the public tree. Launch individual configs
with `scripts/run/train.sh`.

## Configs

| yaml                                  | role                              |
|---------------------------------------|-----------------------------------|
| `r8-a8-nokl-local.yaml`               | rank-8, projector-tuned, no KL |
| `r8-a8-nokl-vision2-local.yaml`       | + last-2 vision-tower layers unfrozen |
| `r8-a8-nokl-laptop.yaml`              | 16 GB variant of the canonical config |
