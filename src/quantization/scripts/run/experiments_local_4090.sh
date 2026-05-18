#!/usr/bin/env bash
# Local 4090 (24 GB VRAM) — VRAM-heavy quantization experiments.
#
# Workload partition (vs laptop 4090 script):
#   • bf16 REFERENCE eval (needs full bf16 in VRAM, peak ~12 GB)
#   • GPTQ w4g128 desc_act=True (slower, slightly better accuracy)
#   • GPTQ w4g128 desc_act=False (faster baseline)
#
# Laptop 4090 (16 GB) runs the complementary variants:
#   • GPTQ w4g64 (smaller group, larger output)
#   • GPTQ w4g128 lm_head=True (more aggressive, smaller output)
#   • bnb NF4 reference
#
# All variants write per-variant `eval.json` + `<dir>/` under
# $RESULTS_ROOT. Run `compare_runs.py` after both machines finish.
#
# Usage:
#   bash quantization/scripts/run/experiments_local_4090.sh \
#       --merged_dir /path/to/sft-merged-bf16
#
# Or with adapter (auto-merges first):
#   bash quantization/scripts/run/experiments_local_4090.sh \
#       --adapter /path/to/sft/final-adapter
#
# Environment overrides (optional):
#   RESULTS_ROOT, PLANTNET_VAL, EVAL_PLANTNET_N, EVAL_VQAV2_N,
#   EVAL_WIKITEXT_N, BASE_MODEL.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=src/quantization/scripts/_env/_common.sh
source "$SCRIPT_DIR/../_env/_common.sh"

# Arg parsing
MERGED_DIR_ARG=""
ADAPTER_ARG=""
SKIP_GPTQ_DESC_ACT_TRUE=0  # set to 1 to skip the slow variant
SKIP_BF16_REF=0            # set to 1 to skip the (long) bf16 reference eval

while [[ $# -gt 0 ]]; do
    case "$1" in
        --merged_dir) MERGED_DIR_ARG="$2"; shift 2 ;;
        --adapter)    ADAPTER_ARG="$2"; shift 2 ;;
        --skip_gptq_desc_act_true) SKIP_GPTQ_DESC_ACT_TRUE=1; shift ;;
        --skip_bf16_ref) SKIP_BF16_REF=1; shift ;;
        -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
        *) log "ERROR" "unknown arg: $1"; exit 2 ;;
    esac
done

# Decide merged_dir
if [[ -n "$MERGED_DIR_ARG" ]]; then
    MERGED_DIR="$MERGED_DIR_ARG"
elif [[ -n "$ADAPTER_ARG" ]]; then
    # Merge first so all variants share the same bf16 input.
    require_dir "$ADAPTER_ARG" "adapter"
    MERGED_DIR="$DEFAULT_RESULTS_ROOT/_merged_bf16"
    log "merge" "Merging $ADAPTER_ARG into $DEFAULT_BASE_MODEL → $MERGED_DIR"
    mkdir -p "$(dirname "$MERGED_DIR")"
    "$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from src.common.model_io import load_bf16_multimodal, save_bf16_merged
from pathlib import Path
m = load_bf16_multimodal('$DEFAULT_BASE_MODEL', adapter_path='$ADAPTER_ARG', device='cuda')
# Prefer the adapter's post-train tokenizer; required for gptqmodel /
# mlx_vlm to load the merged dir (see model_io.save_bf16_merged docs).
save_bf16_merged(m, Path('$MERGED_DIR'), processor_source='$ADAPTER_ARG')
print('OK')
"
else
    log "ERROR" "must provide --merged_dir or --adapter"
    exit 2
fi

require_dir "$MERGED_DIR" "merged_dir"
require_file "$DEFAULT_PLANTNET_VAL"   "plantnet val.jsonl"
require_file "$DEFAULT_PLANTNET_TRAIN" "plantnet train.jsonl"
_reject_overfit100 "$DEFAULT_PLANTNET_TRAIN" "PLANTNET_TRAIN"
_reject_overfit100 "$DEFAULT_PLANTNET_VAL"   "PLANTNET_VAL"
_reject_overfit100 "$MERGED_DIR"             "MERGED_DIR"

log "main" "Local 4090 experiment run starting."
log "main" "  MERGED_DIR:        $MERGED_DIR"
log "main" "  RESULTS_ROOT:      $DEFAULT_RESULTS_ROOT"
log "main" "  PLANTNET_TRAIN:    $DEFAULT_PLANTNET_TRAIN  (GPTQ calibration)"
log "main" "  PLANTNET_VAL:      $DEFAULT_PLANTNET_VAL    (eval only)"
log "main" "  Subsets: plantnet=$EVAL_PLANTNET_N, vqav2=$EVAL_VQAV2_N, wikitext=$EVAL_WIKITEXT_N"

# ----------------------------------------------------------------------
# 1. bf16 reference eval — the ground truth all variants are compared to.
# ----------------------------------------------------------------------
if [[ $SKIP_BF16_REF -eq 0 ]]; then
    log "main" "=== [1/3] bf16 reference eval ==="
    run_eval_only \
        "bf16_reference" \
        "$MERGED_DIR" \
        "hf_bf16" \
        "plantnet_val wikitext_ppl vqav2_devtest"
else
    log "main" "=== [1/3] skipped (--skip_bf16_ref) ==="
    require_file "$DEFAULT_RESULTS_ROOT/bf16_reference/eval.json" \
        "bf16_reference/eval.json (--skip_bf16_ref requires a prior run)"
fi

# ----------------------------------------------------------------------
# 2. GPTQ w4g128, desc_act=False — fast variant.
# ----------------------------------------------------------------------
log "main" "=== [2/3] GPTQ w4g128 desc_act=False ==="
run_variant \
    "gptq_w4g128_da0" \
    "gptq" \
    "$MERGED_DIR" \
    "hf_gptq" \
    --plantnet_jsonl "$DEFAULT_PLANTNET_TRAIN"

# ----------------------------------------------------------------------
# 3. GPTQ w4g128, desc_act=True — slow but more accurate variant.
# ----------------------------------------------------------------------
if [[ $SKIP_GPTQ_DESC_ACT_TRUE -eq 0 ]]; then
    log "main" "=== [3/3] GPTQ w4g128 desc_act=True ==="
    # We control desc_act via a small env-honoring override config.
    # NOTE: easiest path is a one-off Python override since run_quant.py
    # doesn't expose every QuantizeConfig field as CLI flags. We use a
    # small inline Python snippet rather than expand the CLI surface area
    # for one experiment.
    OUT_DIR="$DEFAULT_RESULTS_ROOT/gptq_w4g128_da1"
    mkdir -p "$OUT_DIR"
    log "gptq_w4g128_da1" "Running GPTQ desc_act=True via inline override"
    "$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from pathlib import Path
from src.methods.gptq import GPTQConfig, quantize
cfg = GPTQConfig(bits=4, group_size=128, desc_act=True)
quantize(Path('$MERGED_DIR'), Path('$OUT_DIR'), cfg, plantnet_jsonl='$DEFAULT_PLANTNET_TRAIN')
" 2>&1 | tee "$OUT_DIR.quant.log"
    run_eval_only \
        "gptq_w4g128_da1" \
        "$OUT_DIR" \
        "hf_gptq" \
        "plantnet_val wikitext_ppl vqav2_devtest"
else
    log "main" "=== [3/3] skipped (--skip_gptq_desc_act_true) ==="
fi

log "main" "All variants complete on local 4090. JSONs under $DEFAULT_RESULTS_ROOT/"
log "main" "Combine with laptop output via: python -m scripts.inspect.compare_runs"
