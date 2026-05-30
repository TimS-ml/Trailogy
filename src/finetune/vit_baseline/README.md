# ViT classification baseline

A pure-vision baseline for NA-Plantae species ID. A ViT backbone plus a linear
classification head is trained directly on the iNaturalist NA-Plantae
ImageFolder, with the language model removed entirely.

It answers one question: **how much of the plant-ID signal lives in the vision
encoder alone, before any VLM text decoding?** The val top-1 / top-5 here is
the number to compare against the VLM's plant score (`evaluate_generality.py`
plant domain) — the gap is what the LLM contributes on top of the encoder.

## Data

ImageFolder layout (the iNaturalist NA-Plantae prepared corpus):

```
<PLANT_IMAGE_ROOT>/train/<class_slug>/<id>.jpg
<PLANT_IMAGE_ROOT>/val/<class_slug>/<id>.jpg
```

- The **train** split defines the canonical 799-class label space. Val classes
  not present in train are dropped so the head dimension is fixed.
- No absolute path is committed: set `PLANT_IMAGE_ROOT` (or `--image-root`,
  or `data.image_root` in the YAML) to the dir holding `train/` and `val/`.

## Run

```bash
export PLANT_IMAGE_ROOT=/path/to/images_resized
export CUDA_VISIBLE_DEVICES=0          # the 24 GB card for full runs

# full SFT of the vision tower (default)
bash vit_baseline/scripts/train_vit.sh vit_baseline/configs/vit-base-siglip.yaml

# linear probe (backbone frozen, head only) — fast lower-bound
bash vit_baseline/scripts/train_vit.sh vit_baseline/configs/vit-base-linearprobe.yaml

# smoke run (a few steps, tiny per-class cap)
bash vit_baseline/scripts/train_vit.sh vit_baseline/configs/vit-base-siglip.yaml \
    --max-steps 20 --max-images-per-class 8 --batch-size 8
```

Outputs land in `<output_dir>/<run_name>/`: `best.pt`, `class_to_idx.json`,
`summary.json`.

## Design

- **Backbone**: any `timm` classification model via `model.backbone`. Default
  `vit_base_patch16_siglip_224` — same SigLIP family as the Gemma 4 vision
  tower, so the baseline stands in for "what the on-device encoder can do".
  The Qwen3.5 / InternVL3.5 vision towers (the other workstream) can later be
  wrapped behind the same `build_model` interface.
- **Head**: timm attaches a single linear classifier on pooled features —
  exactly the "ViT + linear head" baseline. `freeze_backbone: true` gives a
  linear probe.
- **Optimiser**: two LR groups — head at `learning_rate`, trunk at
  `backbone_learning_rate` (LLaVA-style layered LR). Norm/bias excluded from
  weight decay.
- **Precision**: bf16 autocast only. No 8-bit / 4-bit weights or optimizers,
  per the project policy that keeps SFT bake-offs comparable.
- **Metrics**: val top-1 / top-5, checkpointing the best top-1.

## Layout

```
vit_baseline/
├── config.py        # pure-Python typed config (no torch) + YAML loader
├── dataset.py       # ImageFolder scan / class map (pure) + torch Dataset
├── model.py         # timm backbone + head; freeze + param-group split
├── train.py         # training loop (CLI / -m vit_baseline.train)
├── configs/         # vit-base-siglip.yaml, vit-base-linearprobe.yaml
└── scripts/train_vit.sh
```

Pure logic (config, dataset scanning) is unit-tested CPU-only in
`tests/test_vit_baseline_config.py` and `tests/test_vit_baseline_dataset.py`.
