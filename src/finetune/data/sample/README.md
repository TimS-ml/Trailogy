# Sample data

5-record fixture used by `scripts/run/train.sh --dry-run` so the pipeline is
exercisable on a fresh clone (Mac, no GPU, no PlantNet download).

The image paths are intentionally fake — the dry-run path does not open
images, only validates the JSONL → unsloth `messages` conversion.

For real training, run `scripts/run/prepare_plantnet_50k.sh` to populate
the canonical `data/english-desc/{train,val,test}.jsonl` files.
