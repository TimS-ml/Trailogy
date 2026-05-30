#!/bin/bash
# Launch the NA-Plantae ViT classification baseline.
#
# Usage:
#   bash vit_baseline/scripts/train_vit.sh                                  # default config
#   bash vit_baseline/scripts/train_vit.sh vit_baseline/configs/vit-base-siglip.yaml
#   bash vit_baseline/scripts/train_vit.sh <cfg> --max-steps 20             # smoke run
#
# Environment variables:
#   PLANT_IMAGE_ROOT     — ImageFolder root (dir containing train/ and val/)
#   CUDA_VISIBLE_DEVICES — GPU selection (set to the 24 GB card for full runs)
#   WANDB_MODE           — set "offline" on air-gapped boxes; sync later
#
# Logs are tee'd to <output_dir>/<run_name>/train.log by the launcher dir.

set -euo pipefail
cd "$(dirname "$0")/../../.."  # -> src/finetune

CONFIG="${1:-vit_baseline/configs/vit-base-siglip.yaml}"
if [ "$#" -gt 0 ]; then shift; fi

if [ -z "${PLANT_IMAGE_ROOT:-}" ]; then
  echo "WARN: PLANT_IMAGE_ROOT not set; relying on data.image_root in the YAML." >&2
fi

python -m vit_baseline.train --config "$CONFIG" "$@"
