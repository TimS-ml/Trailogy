#!/bin/bash
# Sequential architecture sweep for the ViT baseline.
#
# Reads a queue file (one config path per line, '#'/blank skipped) and trains
# each config in turn on a single GPU. Each run checkpoints independently, so
# an interrupted sweep keeps every finished run's best.pt + summary.json.
#
# Usage:
#   bash vit_baseline/scripts/sweep_vit.sh                                  # default queue
#   bash vit_baseline/scripts/sweep_vit.sh vit_baseline/configs/sweep_arch.txt
#
# Environment variables:
#   PLANT_IMAGE_ROOT     — ImageFolder root (train/ + val/)
#   CUDA_VISIBLE_DEVICES — GPU selection (the 24 GB card)
#
# Per-run logs: <output_dir>/<run_name>/train.log (tee'd here too).

set -uo pipefail
cd "$(dirname "$0")/../.."  # -> src/finetune

# Reduce allocator fragmentation for the high-resolution runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QUEUE="${1:-vit_baseline/configs/sweep_arch.txt}"
if [ ! -f "$QUEUE" ]; then
  echo "queue file not found: $QUEUE" >&2
  exit 1
fi
if [ -z "${PLANT_IMAGE_ROOT:-}" ]; then
  echo "ERROR: PLANT_IMAGE_ROOT not set." >&2
  exit 1
fi

echo "=== ViT arch sweep | queue=$QUEUE | $(date) ==="
while IFS= read -r cfg || [ -n "$cfg" ]; do
  case "$cfg" in
    ''|\#*) continue ;;
  esac
  run_name="$(basename "$cfg" .yaml)"
  echo ""
  echo "### [$(date +%H:%M:%S)] running $cfg -> run_name=$run_name"
  python -m vit_baseline.train --config "$cfg" --run-name "$run_name"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "### WARN: $run_name exited rc=$rc — continuing to next config" >&2
  fi
done < "$QUEUE"
echo ""
echo "=== sweep done | $(date) ==="
echo "=== summaries ==="
for s in outputs/vit-baseline/*/summary.json; do
  [ -f "$s" ] && echo "$s:" && cat "$s"
done
