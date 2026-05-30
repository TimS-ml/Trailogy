"""ViT classification baseline for NA-Plantae species ID.

A pure-vision baseline: a ViT backbone + linear classification head trained
directly on the iNaturalist NA-Plantae ImageFolder, with the language model
removed entirely. It answers a single question — how much of the plant-ID
signal lives in the vision encoder alone, before any VLM text decoding.

The package mirrors the finetune pipeline conventions: a pure-Python typed
config (no torch import) so it can be unit-tested on CPU, env-var / CLI image
roots (no absolute paths in tracked files), and bf16-only training to stay
comparable with the LoRA SFT bake-offs.
"""
