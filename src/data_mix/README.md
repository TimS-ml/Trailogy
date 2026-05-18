# data_mix

Anti-overtraining mixed SFT corpus builder for the Gemma 4 E2B VLM
finetune pipeline. Produces a single JSONL drop-in for
`finetune/src/data.py::load_vision_dataset`.

See `docs/superpowers/specs/2026-05-15-data-mix-anti-overtraining-design.md`
for the bucket ratios, schema, and rationale.

## Quick start

```bash
# Optional: point storage roots at non-default locations
export HF_HOME=/path/to/big/disk/hf_cache
export DATA_MIX_IMAGE_ROOT=/path/to/big/disk/data_mix/images
export DATA_MIX_OUTPUT_ROOT=/path/to/big/disk/data_mix/mix-20k

bash scripts/build_mix.sh
```

Defaults (all unset envs) write into the repo under `data_mix/_local/`
which is gitignored.

## Tests

```bash
pytest data_mix/tests -v
```
