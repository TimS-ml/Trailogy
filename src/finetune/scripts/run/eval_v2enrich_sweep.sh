#!/bin/bash
# Run generality eval on a list of v2enrich checkpoints, print
# plant/mmlu/aime per ckpt.
#
# Methodology matches gemma4_note D2 r256 sweep doc:
#   --loader hf_bf16  (no quantization)
#   --skip_judge      (ROUGE-L fallback for open-ended buckets)
#   --prompt_prefix '[camera=on] '
#   default 5-domain: plant / mmlu / aime / refusal / llava
#
# Outputs JSON to eval/results/<run>_ckpt-<step>_generality.json
# Prints a summary table at the end.
set -uo pipefail
cd "$(dirname "$0")/../.."

RUN_DIR="${RUN_DIR:-outputs/r64-a64-nokl-local-30k-v2enrich_20260525_124049}"
STEPS="${STEPS:-20000 22000 24000 26000 28000 30000}"
BASE_MODEL="unsloth/gemma-4-E2B-it"
OUT_DIR="eval/results"

# Force-override .env's legacy PLANT_IMAGE_ROOT — on this box the
# plant val images live under data/inaturalist_na_plantae_prepared/images_resized/val/.
PLANT_IMAGE_ROOT_LOCAL="/home/tim/offline-git/App/data/inaturalist_na_plantae_prepared/images_resized/val"

mkdir -p "$OUT_DIR"

echo "=== v2enrich generality eval sweep ==="
echo "  run_dir: $RUN_DIR"
echo "  steps:   $STEPS"
echo

declare -A PLANT MMLU AIME
for step in $STEPS; do
    ckpt="$RUN_DIR/checkpoint-$step"
    if [ ! -d "$ckpt" ]; then
        echo "WARN: missing $ckpt — skip"
        continue
    fi
    stem=$(basename "$RUN_DIR")
    out_json="$OUT_DIR/${stem}_ckpt-${step}_generality.json"

    echo "--- eval step $step ---"
    python eval/evaluate_generality.py \
        --base_model "$BASE_MODEL" \
        --adapter_path "$ckpt" \
        --skip_judge \
        --prompt_prefix '[camera=on] ' \
        --plant_image_root "$PLANT_IMAGE_ROOT_LOCAL" \
        --output_file "$out_json" \
        2>&1 | tail -8 || true

    # Parse the per-domain scores from the JSON
    if [ -f "$out_json" ]; then
        scores=$(python - "$out_json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
d = r.get("domains", {})
plant = d.get("plant", {}).get("species_match_rate") or d.get("plant", {}).get("score", 0)
mmlu  = d.get("mmlu",  {}).get("accuracy")          or d.get("mmlu",  {}).get("score", 0)
aime  = d.get("aime",  {}).get("accuracy")          or d.get("aime",  {}).get("score", 0)
print(f"{plant:.3f} {mmlu:.3f} {aime:.3f}")
PY
)
        read p m a <<<"$scores"
        PLANT[$step]=$p; MMLU[$step]=$m; AIME[$step]=$a
        echo "  step=$step plant=$p mmlu=$m aime=$a"
    fi
done

echo
echo "===== v2enrich summary table ====="
printf "%-6s  %-8s  %-8s  %-8s\n" step plant mmlu aime
for step in $STEPS; do
    printf "%-6s  %-8s  %-8s  %-8s\n" "$step" "${PLANT[$step]:-?}" "${MMLU[$step]:-?}" "${AIME[$step]:-?}"
done
