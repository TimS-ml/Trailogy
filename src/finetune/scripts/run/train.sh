#!/bin/bash
# Launch unsloth LoRA finetuning of Gemma 4 E2B for the hiking companion.
#
# Usage:
#   bash scripts/run/train.sh                              # default config
#   bash scripts/run/train.sh configs/small.yaml           # custom config
#   bash scripts/run/train.sh --dry-run                    # CPU/Mac sanity check
#   bash scripts/run/train.sh configs/foo.yaml --no-eval   # skip auto-eval after train
#   bash scripts/run/train.sh configs/foo.yaml \
#       --resume_from_checkpoint outputs/foo_.../checkpoint-1000
#
# As of 2026-05-11 default behaviour is:
#   - Train logs tee'd to outputs/<run_name>/{train.log,train-resume.log}
#   - After successful training, eval automatically runs on val.jsonl
#     (300 samples, --use_unsloth on, per cfg.eval) tee'd to
#     outputs/<run_name>/eval.log. Disable with `--no-eval`.
#
# Environment variables:
#   HF_TOKEN             — Hugging Face token (gated model)
#   WANDB_PROJECT        — W&B project name (set with report_to: "wandb")
#   CUDA_VISIBLE_DEVICES — GPU selection (default: all)

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
# Parse args: pull out --dry-run / --no-eval; detect --resume_from_checkpoint;
# detect --run_name so the log tee path tracks the CLI-override run_name
# instead of the yaml-default.
# ---------------------------------------------------------------------------

DRY_RUN_FLAG=""
NO_EVAL=""
IS_RESUME=""
RUN_NAME_OVERRIDE=""
ARGS=()
# Use a positional-style loop with explicit shift so we can pair
# --run_name with its value. The naive for-arg-in-$@ loop treats
# `--run_name foo` as two unrelated tokens.
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN_FLAG="--dry-run"; shift
            ;;
        --no-eval)
            NO_EVAL="1"; shift
            ;;
        --resume_from_checkpoint)
            IS_RESUME="1"; ARGS+=("$1" "$2"); shift 2
            ;;
        --resume_from_checkpoint=*)
            IS_RESUME="1"; ARGS+=("$1"); shift
            ;;
        --run_name)
            RUN_NAME_OVERRIDE="$2"; ARGS+=("$1" "$2"); shift 2
            ;;
        --run_name=*)
            RUN_NAME_OVERRIDE="${1#--run_name=}"; ARGS+=("$1"); shift
            ;;
        *)
            ARGS+=("$1"); shift
            ;;
    esac
done

CONFIG="${ARGS[0]:-configs/default.yaml}"
# Drop the positional config from the list; what remains is the
# pass-through flag tail forwarded to `python -m src.finetune`.
if [ "${#ARGS[@]}" -gt 0 ]; then
    EXTRA_ARGS=("${ARGS[@]:1}")
else
    EXTRA_ARGS=()
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && [ -z "$DRY_RUN_FLAG" ]; then
    echo "WARNING: HF_TOKEN not set. Required for gated unsloth/gemma-4-E2B-it."
    echo "  export HF_TOKEN=hf_..."
    echo ""
fi

# ---------------------------------------------------------------------------
# Compute run_name + output_dir up-front so we can mkdir + tee logs to a
# stable path. Resolution order (highest to lowest priority):
#   1. --run_name CLI override (captured above as RUN_NAME_OVERRIDE)
#   2. yaml training.run_name field
#   3. {config_stem}_{YYYYMMDD_HHMMSS}  (fallback, mirrors
#      src.finetune._generate_run_name)
# ---------------------------------------------------------------------------

if [ -n "$RUN_NAME_OVERRIDE" ]; then
    RUN_NAME="$RUN_NAME_OVERRIDE"
else
    RUN_NAME=$(python -c "
import sys, yaml
from datetime import datetime
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1])) or {}
rn = (cfg.get('training') or {}).get('run_name')
if not rn:
    stem = Path(sys.argv[1]).stem
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rn = f'{stem}_{ts}'
print(rn)
" "$CONFIG")
fi
OUTPUT_DIR="outputs/$RUN_NAME"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

echo "=== Gemma 4 hikeCompanion unsloth LoRA finetune ==="
echo "Config:     $CONFIG"
echo "Run name:   $RUN_NAME"
echo "Output dir: $OUTPUT_DIR"
echo "Mode:       ${DRY_RUN_FLAG:-real training}${IS_RESUME:+ (resume)}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-all}"
echo "Auto-eval:  $([ -n "$NO_EVAL" ] && echo 'disabled (--no-eval)' || echo 'per cfg.eval.enabled')"
echo ""

# ---------------------------------------------------------------------------
# Data sanity: check the configured train JSONL before launching a real run.
# ---------------------------------------------------------------------------

TRAIN_FILE=$(python -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print((cfg.get('data') or {}).get('train_file') or 'data/train.jsonl')
" "$CONFIG")

if [ ! -f "$TRAIN_FILE" ]; then
    if [ -n "$DRY_RUN_FLAG" ] && [ -f "data/sample/train.jsonl" ]; then
        echo "INFO: $TRAIN_FILE missing — falling back to data/sample/ for dry-run."
        EXTRA_ARGS+=(--train_file "data/sample/train.jsonl"
                     --val_file   "data/sample/val.jsonl"
                     --max_train_samples 5)
    elif [ -z "$DRY_RUN_FLAG" ]; then
        echo "ERROR: configured data.train_file not found: $TRAIN_FILE"
        echo "Run scripts/run/prepare_plantnet_50k.sh first."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Train. Dry-run skips the tee + auto-eval entirely (no artifact dir).
# ---------------------------------------------------------------------------

if [ -n "$DRY_RUN_FLAG" ]; then
    python -m src.finetune --config "$CONFIG" --run_name "$RUN_NAME" \
        $DRY_RUN_FLAG "${EXTRA_ARGS[@]}"
    exit $?
fi

mkdir -p "$OUTPUT_DIR"

# Resume → train-resume.log; otherwise → train.log.
LOG_NAME="train.log"
if [ -n "$IS_RESUME" ]; then
    LOG_NAME="train-resume.log"
fi

set -o pipefail
python -m src.finetune --config "$CONFIG" --run_name "$RUN_NAME" "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/$LOG_NAME"
TRAIN_RC=${PIPESTATUS[0]}

echo ""
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "=== Training FAILED with exit $TRAIN_RC ==="
    echo "Log: $OUTPUT_DIR/$LOG_NAME"
    exit "$TRAIN_RC"
fi

echo "=== Training complete ==="
echo "Adapter saved to: $OUTPUT_DIR/final-adapter"
echo "Train log:        $OUTPUT_DIR/$LOG_NAME"

# ---------------------------------------------------------------------------
# Auto-eval after train (default on; suppressed by --no-eval or
# `eval.enabled: false` in yaml).
# ---------------------------------------------------------------------------

if [ -n "$NO_EVAL" ]; then
    echo ""
    echo "INFO: --no-eval given; skipping auto-eval."
    exit 0
fi

EVAL_ENABLED=$(python -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print((cfg.get('eval') or {}).get('enabled', True))
" "$CONFIG")

if [ "$EVAL_ENABLED" != "True" ]; then
    echo ""
    echo "INFO: cfg.eval.enabled = $EVAL_ENABLED; skipping auto-eval."
    exit 0
fi

echo ""
echo "=== Auto-eval: $RUN_NAME ==="
python -m src.evaluate --config "$CONFIG" --run_name "$RUN_NAME" \
    --adapter_path "$OUTPUT_DIR/final-adapter" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"
EVAL_RC=${PIPESTATUS[0]}

echo ""
if [ "$EVAL_RC" -ne 0 ]; then
    echo "=== Auto-eval FAILED with exit $EVAL_RC ==="
    echo "Eval log: $OUTPUT_DIR/eval.log"
    echo "(Adapter is still saved at $OUTPUT_DIR/final-adapter; re-run manually with:"
    echo "  python -m src.evaluate --config $CONFIG --run_name $RUN_NAME)"
    exit "$EVAL_RC"
fi

echo "=== Auto-eval complete ==="
echo "Eval log:     $OUTPUT_DIR/eval.log"
echo "Eval results: results/${RUN_NAME}_eval.json"
