#!/usr/bin/env bash
# Smoke tests for quantization paths that run on NVIDIA CUDA (4090).
#
# Doesn't run actual MLX conversion — that needs a Mac. Instead it
# exercises:
#   1. Safetensors header inspection on whatever bf16 checkpoint is
#      handy (pure-python, no CUDA).
#   2. GPTQ backend availability check (no model load yet, but enough
#      to confirm the chosen backend can see Gemma4ForConditionalGeneration).
#   3. LoRA merge step (the bf16-merged output that all quant methods
#      consume).
#
# Usage:
#   bash src/quantization/scripts/run/smoke_4090.sh [adapter_path]
#
# If adapter_path is omitted, only the inspection + backend checks
# run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ADAPTER="${1:-}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python"
    else
        PYTHON_BIN="${PYTHON:-python}"
    fi
fi
export PYTHON_BIN

# Preload the conda env's libstdc++ when present. bitsandbytes' CUDA
# extension and pyarrow (used by ``datasets``) both link against C++
# ABI symbols newer than the system libstdc++ ships in many distros;
# without this preload, ``import datasets`` post-``import bitsandbytes``
# fails with ``version 'CXXABI_1.3.15' not found``. The repo's
# Applying it here makes the script self-contained.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    _CONDA_LIBSTDCPP="$CONDA_PREFIX/x86_64-conda-linux-gnu/lib/libstdc++.so.6"
    if [[ -f "$_CONDA_LIBSTDCPP" && "${LD_PRELOAD:-}" != *"$_CONDA_LIBSTDCPP"* ]]; then
        export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$_CONDA_LIBSTDCPP"
    fi
    _CONDA_LIB="$CONDA_PREFIX/lib"
    if [[ -d "$_CONDA_LIB" && ":${LD_LIBRARY_PATH:-}:" != *":$_CONDA_LIB:"* ]]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_CONDA_LIB"
    fi
fi

echo "==> Quantization smoke test (4090)"
echo "    REPO_ROOT=$REPO_ROOT"
echo "    ADAPTER=${ADAPTER:-<none>}"
echo "    PYTHON_BIN=$PYTHON_BIN"
echo "    LD_PRELOAD=${LD_PRELOAD:-<none>}"

cd "$REPO_ROOT"

# 1. Pure-python checks
echo
echo "==> [1/5] Unit tests (pure python, no CUDA needed)"
if "$PYTHON_BIN" -m pytest --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pytest tests/ -q
else
    echo "    pytest-module not installed for $PYTHON_BIN; skipping."
fi

# 2. GPTQ backend check (no model load)
echo
echo "==> [2/5] GPTQ backend availability check"
"$PYTHON_BIN" -m src.methods.gptq --smoke --backend gptqmodel || \
    echo "    (gptqmodel not available — fall back to optimum or auto_gptq)"

# 3. Safetensors inspection on a known bf16 base (if cached)
echo
echo "==> [3/5] Inspect base bf16 (if HF cache present)"
BASE_DIR="$HOME/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it/snapshots"
if [[ -d "$BASE_DIR" ]]; then
    FIRST_SNAP=$(ls -1 "$BASE_DIR" | head -1)
    if [[ -n "$FIRST_SNAP" ]]; then
        "$PYTHON_BIN" -m scripts.inspect.quantized "$BASE_DIR/$FIRST_SNAP" || true
    fi
else
    echo "    HF cache for unsloth/gemma-4-E2B-it not found; skipping."
fi

# 4. Unsloth UD diff (background-style — kicks off downloads if needed)
echo
echo "==> [4/5] Unsloth UD diff (downloads 3.5-4.5 GB each, resumable)"
echo "    Run this manually when network bandwidth is available:"
echo "      \$PYTHON_BIN -m scripts.inspect.diff_unsloth_ud --three_way --output_yaml /tmp/promotion_keys.yaml"

# 5. LoRA merge smoke (requires CUDA + bf16 base on disk)
if [[ -n "$ADAPTER" ]]; then
    echo
    echo "==> [5/5] LoRA merge smoke (bf16 → safetensors, CUDA)"
    OUT_DIR="results/smoke-merge-$$"
    "$PYTHON_BIN" -m scripts.run.quant \
        --method mlx_vlm_g64 \
        --base_model unsloth/gemma-4-E2B-it \
        --adapter "$ADAPTER" \
        --output_dir "$OUT_DIR" \
        --merge_device cuda 2>&1 | tee "$OUT_DIR.log" || \
        echo "    Expected: mlx_vlm import error on non-Mac. The merge step itself should succeed."
    if [[ -d "${OUT_DIR}.bf16-merged" ]]; then
        echo
        echo "==> Inspecting merged bf16 dir:"
        "$PYTHON_BIN" -m scripts.inspect.quantized "${OUT_DIR}.bf16-merged"
    fi
else
    echo
    echo "==> [5/5] LoRA merge smoke skipped (no adapter path given)"
fi

echo
echo "==> Done."
