#!/usr/bin/env bash
# Laptop 4090 (16 GB VRAM) — complementary quantization experiments.
#
# Workload partition (vs local 4090 script):
#   • GPTQ w4g64 (smaller group = more scales = larger output)
#   • GPTQ w4g128 lm_head=True (more aggressive — lm_head also quantized)
#   • bnb NF4 reference (different family — QLoRA-era 4-bit)
#
# Local 4090 runs the complementary variants:
#   • bf16 reference eval
#   • GPTQ w4g128 desc_act=False
#   • GPTQ w4g128 desc_act=True
#
# Notes on VRAM:
#   • bf16 base alone = 9.5 GB.
#   • Inference (batch=1, KV cache @ seq=1024 + image tokens) peaks ~12 GB.
#     Fits in 16 GB with margin.
#   • GPTQ with offload_to_disk=True keeps quantization peak bounded.
#
# Usage:
#   bash quantization/scripts/run/experiments_laptop_4090.sh \
#       --merged_dir /path/to/sft-merged-bf16
#
# Or with adapter (auto-merges first):
#   bash quantization/scripts/run/experiments_laptop_4090.sh \
#       --adapter /path/to/sft/final-adapter
#
# Environment overrides:
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
SKIP_BNB=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --merged_dir) MERGED_DIR_ARG="$2"; shift 2 ;;
        --adapter)    ADAPTER_ARG="$2"; shift 2 ;;
        --skip_bnb)   SKIP_BNB=1; shift ;;
        -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
        *) log "ERROR" "unknown arg: $1"; exit 2 ;;
    esac
done

# Decide merged_dir
if [[ -n "$MERGED_DIR_ARG" ]]; then
    MERGED_DIR="$MERGED_DIR_ARG"
elif [[ -n "$ADAPTER_ARG" ]]; then
    require_dir "$ADAPTER_ARG" "adapter"
    MERGED_DIR="$DEFAULT_RESULTS_ROOT/_merged_bf16"
    log "merge" "Merging $ADAPTER_ARG into $DEFAULT_BASE_MODEL → $MERGED_DIR"
    mkdir -p "$(dirname "$MERGED_DIR")"
    # Use CPU merge on the laptop to be safe with VRAM (PEFT briefly
    # holds base + adapter + delta during merge_and_unload — could
    # exceed 16 GB on GPU).
    "$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from src.common.model_io import load_bf16_multimodal, save_bf16_merged
from pathlib import Path
m = load_bf16_multimodal('$DEFAULT_BASE_MODEL', adapter_path='$ADAPTER_ARG', device='cpu')
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

log "main" "Laptop 4090 experiment run starting."
log "main" "  MERGED_DIR:        $MERGED_DIR"
log "main" "  RESULTS_ROOT:      $DEFAULT_RESULTS_ROOT"
log "main" "  PLANTNET_TRAIN:    $DEFAULT_PLANTNET_TRAIN  (GPTQ calibration)"
log "main" "  PLANTNET_VAL:      $DEFAULT_PLANTNET_VAL    (eval only)"
log "main" "  Subsets: plantnet=$EVAL_PLANTNET_N, vqav2=$EVAL_VQAV2_N, wikitext=$EVAL_WIKITEXT_N"

# ----------------------------------------------------------------------
# 1. GPTQ w4g64 — smaller group, larger output, finer quantization.
# ----------------------------------------------------------------------
log "main" "=== [1/3] GPTQ w4g64 desc_act=False ==="
OUT_DIR="$DEFAULT_RESULTS_ROOT/gptq_w4g64_da0"
mkdir -p "$OUT_DIR"
"$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from pathlib import Path
from src.methods.gptq import GPTQConfig, quantize
cfg = GPTQConfig(bits=4, group_size=64, desc_act=False, lm_head=False)
quantize(Path('$MERGED_DIR'), Path('$OUT_DIR'), cfg, plantnet_jsonl='$DEFAULT_PLANTNET_TRAIN')
" 2>&1 | tee "$OUT_DIR.quant.log"
run_eval_only \
    "gptq_w4g64_da0" \
    "$OUT_DIR" \
    "hf_gptq" \
    "plantnet_val wikitext_ppl vqav2_devtest"

# ----------------------------------------------------------------------
# 2. GPTQ w4g128 lm_head=True — more aggressive (lm_head also quantized).
#    NOTE: Gemma 4 has tie_word_embeddings=True, so lm_head=True is
#    auto-downgraded to lm_head=False by _resolve_lm_head(). The result
#    is effectively gptq_w4g128_da0 — kept as a separate variant for
#    the eval matrix to record that this experiment was attempted.
# ----------------------------------------------------------------------
log "main" "=== [2/3] GPTQ w4g128 lm_head=True (auto-downgrade if tied) ==="
OUT_DIR="$DEFAULT_RESULTS_ROOT/gptq_w4g128_lmhead"
mkdir -p "$OUT_DIR"
"$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from pathlib import Path
from src.methods.gptq import GPTQConfig, quantize
cfg = GPTQConfig(bits=4, group_size=128, desc_act=False, lm_head=True)
quantize(Path('$MERGED_DIR'), Path('$OUT_DIR'), cfg, plantnet_jsonl='$DEFAULT_PLANTNET_TRAIN')
" 2>&1 | tee "$OUT_DIR.quant.log"
run_eval_only \
    "gptq_w4g128_lmhead" \
    "$OUT_DIR" \
    "hf_gptq" \
    "plantnet_val wikitext_ppl vqav2_devtest"

# ----------------------------------------------------------------------
# 3. bnb NF4 reference (low-priority; non-MLX-deployable comparison point).
# ----------------------------------------------------------------------
if [[ $SKIP_BNB -eq 0 ]]; then
    log "main" "=== [3/3] bnb NF4 reference ==="
    OUT_DIR="$DEFAULT_RESULTS_ROOT/bnb_nf4"
    mkdir -p "$OUT_DIR"
    "$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from pathlib import Path
from src.methods.bnb_nf4 import BnBNF4Config, quantize
cfg = BnBNF4Config()
quantize(Path('$MERGED_DIR'), Path('$OUT_DIR'), cfg)
" 2>&1 | tee "$OUT_DIR.quant.log"
    run_eval_only \
        "bnb_nf4" \
        "$OUT_DIR" \
        "hf_bf16" \
        "plantnet_val wikitext_ppl vqav2_devtest"
else
    log "main" "=== [3/3] skipped (--skip_bnb) ==="
fi

log "main" "All variants complete on laptop 4090. JSONs under $DEFAULT_RESULTS_ROOT/"
log "main" "Combine with local-4090 output via: python -m scripts.inspect.compare_runs"
