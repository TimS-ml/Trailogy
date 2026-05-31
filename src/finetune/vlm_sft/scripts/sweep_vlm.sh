#!/bin/bash
# Sequential VLM vision-tower SFT sweep (Qwen3.5 / InternVL3.5).
#
# Usage:
#   bash vlm_sft/scripts/sweep_vlm.sh                          # default 6-config queue
#   bash vlm_sft/scripts/sweep_vlm.sh vlm_sft/configs_sweep.txt
#   bash vlm_sft/scripts/sweep_vlm.sh <queue> --max-steps 30   # smoke
#
# Environment variables:
#   PYTHON               — interpreter (defaults to `python`)
#   CUDA_VISIBLE_DEVICES — GPU selection (the 24 GB card)
#   HF_TOKEN             — if any backbone is gated
#   WANDB_MODE           — "offline" on air-gapped boxes
#
# Each run saves into its own output_dir; an interrupted sweep keeps finished
# runs. Reduce allocator fragmentation for the 4B + vision forward.

set -uo pipefail
cd "$(dirname "$0")/../.."  # -> src/finetune
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QUEUE="${1:-vlm_sft/configs_sweep.txt}"
if [ "$#" -gt 0 ]; then shift; fi
if [ ! -f "$QUEUE" ]; then echo "queue not found: $QUEUE" >&2; exit 1; fi

echo "=== VLM SFT sweep | queue=$QUEUE | $(date) ==="
while IFS= read -r cfg || [ -n "$cfg" ]; do
  case "$cfg" in ''|\#*) continue ;; esac
  run_name="$(basename "$cfg" .yaml)"
  echo ""
  echo "### [$(date +%H:%M:%S)] running $cfg -> $run_name"
  "${PYTHON:-python}" -m vlm_sft.train --config "$cfg" --run-name "$run_name" "$@"
  rc=$?
  [ "$rc" -ne 0 ] && echo "### WARN: $run_name exited rc=$rc — continuing" >&2
done < "$QUEUE"
echo ""
echo "=== VLM sweep done | $(date) ==="
