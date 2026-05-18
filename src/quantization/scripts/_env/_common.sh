# Shared helpers for the dual-GPU experiment scripts.
# Source this from `run_experiments_*.sh`.

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

# Python + library paths.
#
# Override PYTHON_BIN in your environment to point at the conda / venv
# python that has gptqmodel + transformers + peft installed. Example:
#   export PYTHON_BIN=/path/to/env/bin/python
#
# On Linux, gptqmodel needs a recent libstdc++ — typically the one shipped
# inside the conda env, not the system one. Set PYTHON_LIB_DIR to the
# env's lib/ to prepend it onto LD_LIBRARY_PATH (resolves GLIBCXX_3.4.x
# version mismatch). If PYTHON_LIB_DIR is unset, this script derives it
# from PYTHON_BIN.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: PYTHON_BIN is not set and no python3/python found on PATH." >&2
    echo "  Set PYTHON_BIN to your conda/venv python before running this script." >&2
    exit 2
fi
PYTHON_LIB_DIR="${PYTHON_LIB_DIR:-$(dirname "$PYTHON_BIN")/../lib}"
if [[ -d "$PYTHON_LIB_DIR" ]]; then
    if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH="$PYTHON_LIB_DIR"
    else
        export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:$LD_LIBRARY_PATH"
    fi
fi

# Default inputs — override via env or CLI args.
#
# Data-leak safety:
#   • PLANTNET_VAL is for EVAL ONLY. Never feed it to GPTQ calibration.
#   • PLANTNET_TRAIN is the calibration source (disjoint from val).
#   • OVERFIT100 paths are explicitly REJECTED — that data is for
#     memorization-ceiling tests, not for any deliverable pipeline.
DEFAULT_ADAPTER="${ADAPTER:-}"
DEFAULT_MERGED_DIR="${MERGED_DIR:-}"
DEFAULT_PLANTNET_VAL="${PLANTNET_VAL:-$REPO_ROOT/../finetune/data/val.jsonl}"
DEFAULT_PLANTNET_TRAIN="${PLANTNET_TRAIN:-$REPO_ROOT/../finetune/data/train.jsonl}"
DEFAULT_RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
DEFAULT_BASE_MODEL="${BASE_MODEL:-unsloth/gemma-4-E2B-it}"

# Reject overfit100 data anywhere in the pipeline.
_reject_overfit100() {
    local p="$1"
    local label="$2"
    if [[ "$p" == *"overfit100"* ]]; then
        log "ERROR" "$label points at overfit100 data ($p). That set is train==eval; "
        log "ERROR" "it will silently make every quant variant look perfect. Use plantnet-50k splits."
        return 1
    fi
    return 0
}

# Subset sizes for the eval harness. Override via env for full runs.
EVAL_PLANTNET_N="${EVAL_PLANTNET_N:-2870}"
EVAL_VQAV2_N="${EVAL_VQAV2_N:-1000}"
EVAL_WIKITEXT_N="${EVAL_WIKITEXT_N:-200}"

log() {
    local label="$1"; shift
    echo "[$(date '+%H:%M:%S')] [$label] $*"
}

require_dir() {
    local path="$1"
    local label="$2"
    if [[ ! -d "$path" ]]; then
        log "ERROR" "$label not found at $path"
        return 1
    fi
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "$path" ]]; then
        log "ERROR" "$label not found at $path"
        return 1
    fi
}

# Build the optional --gptq_backend argv list for run_eval.py.
#
# Reads GPTQ_BACKEND (env var). Empty → no extra flag passed (run_eval.py
# falls back to its default of "triton"). Non-empty → forwarded as
# --gptq_backend <value>. Only meaningful when the loader is hf_gptq; the
# python side ignores it for other loaders.
_gptq_backend_flags() {
    if [[ -n "${GPTQ_BACKEND:-}" ]]; then
        printf -- "--gptq_backend\n%s\n" "$GPTQ_BACKEND"
    fi
}

# Run one variant: quantize → eval → write per-variant JSON.
# Calibration (for GPTQ) uses train.jsonl; eval uses val.jsonl.
# These MUST be disjoint splits.
#
# Args:
#   $1 = variant name (results subdir under RESULTS_ROOT)
#   $2 = method name (passed to run_quant --method)
#   $3 = merged bf16 input dir
#   $4 = loader name for eval (hf_bf16 or mlx_vlm)
#   $5+ = extra args forwarded to run_quant.py
#
# Honors the GPTQ_BACKEND env var when $4 == hf_gptq — see
# _gptq_backend_flags / docs/02-methods.md.
run_variant() {
    local variant="$1"; shift
    local method="$1"; shift
    local merged="$1"; shift
    local loader="$1"; shift
    local extra_args=("$@")

    local out_dir="$DEFAULT_RESULTS_ROOT/$variant"
    log "$variant" "Starting variant ($method) → $out_dir"

    log "$variant" "[1/2] Quantizing..."
    "$PYTHON_BIN" -m scripts.run.quant \
        --method "$method" \
        --merged_dir "$merged" \
        --output_dir "$out_dir" \
        "${extra_args[@]}" \
        2>&1 | tee "$out_dir.quant.log"

    # Build eval argv. mapfile + _gptq_backend_flags keeps the optional
    # --gptq_backend flag out of the argv when GPTQ_BACKEND is unset, so
    # the existing test that asserts default-behavior is unaffected.
    local backend_flags=()
    if [[ "$loader" == "hf_gptq" ]]; then
        while IFS= read -r line; do
            backend_flags+=("$line")
        done < <(_gptq_backend_flags)
    fi

    log "$variant" "[2/2] Evaluating ($loader, val.jsonl${GPTQ_BACKEND:+, backend=$GPTQ_BACKEND})..."
    "$PYTHON_BIN" -m scripts.run.eval \
        --variant "$variant" \
        --loader "$loader" \
        --model_dir "$out_dir" \
        --plantnet_val_jsonl "$DEFAULT_PLANTNET_VAL" \
        --plantnet_n "$EVAL_PLANTNET_N" \
        --vqav2_n "$EVAL_VQAV2_N" \
        --wikitext_n "$EVAL_WIKITEXT_N" \
        --benchmarks plantnet_val wikitext_ppl vqav2_devtest \
        --output_dir "$out_dir" \
        "${backend_flags[@]}" \
        2>&1 | tee "$out_dir.eval.log"

    log "$variant" "Done. eval.json at $out_dir/eval.json"
}

# Just run eval, no quant (used for the bf16 reference).
# Honors GPTQ_BACKEND when $3 == hf_gptq.
run_eval_only() {
    local variant="$1"; shift
    local model_dir="$1"; shift
    local loader="$1"; shift
    local benchmarks="$1"; shift

    local out_dir="$DEFAULT_RESULTS_ROOT/$variant"
    mkdir -p "$out_dir"

    local backend_flags=()
    if [[ "$loader" == "hf_gptq" ]]; then
        while IFS= read -r line; do
            backend_flags+=("$line")
        done < <(_gptq_backend_flags)
    fi

    log "$variant" "Eval-only on $model_dir ($loader, benchmarks=$benchmarks${GPTQ_BACKEND:+, backend=$GPTQ_BACKEND})"
    "$PYTHON_BIN" -m scripts.run.eval \
        --variant "$variant" \
        --loader "$loader" \
        --model_dir "$model_dir" \
        --plantnet_val_jsonl "$DEFAULT_PLANTNET_VAL" \
        --plantnet_n "$EVAL_PLANTNET_N" \
        --vqav2_n "$EVAL_VQAV2_N" \
        --wikitext_n "$EVAL_WIKITEXT_N" \
        --benchmarks $benchmarks \
        --output_dir "$out_dir" \
        "${backend_flags[@]}" \
        2>&1 | tee "$out_dir.eval.log"
}
