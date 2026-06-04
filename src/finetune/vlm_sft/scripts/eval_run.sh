#!/bin/bash
# Full eval (plant / mmlu / aime) over every saved checkpoint of a finished
# vlm_sft run, so you can see the per-step trajectory and prune the rest.
#
# Auto-detects the run mode per checkpoint:
#   *+LoRA / LoRA-only -> checkpoint is a PEFT adapter (+ extra_trainable.pt)
#       => eval as  --base_model <orig> --adapter_path <ckpt>
#   vision-only        -> checkpoint is a full model (processing_class saved)
#       => eval as  --base_model <ckpt>
# InternVL gets --image_max_patches 1 automatically (tile cap parity).
#
# Usage:
#   PLANT_IMAGE_ROOT=.../images_resized/val \
#   bash vlm_sft/scripts/eval_run.sh outputs/<run>              # all checkpoints
#   bash vlm_sft/scripts/eval_run.sh outputs/<run> 12000 30000  # specific steps
#
# Env: PYTHON (interpreter), PLANT_IMAGE_ROOT (required for plant domain).
set -uo pipefail
cd "$(dirname "$0")/../.."  # -> src/finetune
PY="${PYTHON:-python}"

RUN="${1:?usage: eval_run.sh <output_dir> [step ...]}"; shift || true
[ -d "$RUN" ] || { echo "no such run dir: $RUN" >&2; exit 1; }

if [ "$#" -gt 0 ]; then
  CKPTS=""; for s in "$@"; do CKPTS="$CKPTS $RUN/checkpoint-$s"; done
else
  CKPTS="$(ls -d "$RUN"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)"
fi
[ -n "${CKPTS// }" ] || { echo "no checkpoints under $RUN" >&2; exit 1; }

for ck in $CKPTS; do
  [ -d "$ck" ] || { echo "skip missing $ck"; continue; }
  step="${ck##*-}"
  if [ -f "$ck/adapter_config.json" ]; then
    base="$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['base_model_name_or_path'])" "$ck/adapter_config.json")"
    mode_args=(--base_model "$base" --adapter_path "$ck")
  else
    base="$ck"; mode_args=(--base_model "$ck")
  fi
  imp=()
  if echo "$base" | grep -qi internvl || { [ -f "$ck/config.json" ] && grep -qi '"model_type": *"internvl' "$ck/config.json"; }; then
    imp=(--image_max_patches 1)
  fi
  echo "### [$(date +%H:%M)] eval $(basename "$RUN") @ step $step (base=$base)"
  "$PY" eval/evaluate_generality.py "${mode_args[@]}" \
      --domains plant mmlu aime "${imp[@]}" \
      --skip_judge --max_new_tokens 256 \
      --output_file "$RUN/eval_${step}.json" || echo "  step $step FAILED"
done

echo ""; echo "=== summary: $RUN ==="
"$PY" - "$RUN" <<'PYEOF'
import json, sys, glob, os
run = sys.argv[1]
files = sorted(glob.glob(os.path.join(run, "eval_*.json")),
               key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))
print(f"{'step':>7} {'plant':>6} {'mmlu':>6} {'aime':>6}")
for f in files:
    d = json.load(open(f)); dm = d.get("domains", d)
    step = int(f.rsplit("_", 1)[-1].split(".")[0])
    g = lambda k: dm.get(k, {}).get("score")
    fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
    print(f"{step:>7} {fmt(g('plant')):>6} {fmt(g('mmlu')):>6} {fmt(g('aime')):>6}")
PYEOF
