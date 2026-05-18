# Environment bootstrap for mlx_vlm / mlx_lm on NVIDIA Linux.
#
# Sourced (not executed) by callers that need to run mlx-cuda kernels:
#
#     source src/quantization/scripts/_env/_mlx_env.sh
#     python ...   # uses $MLX_PYTHON
#
# What this fixes
# ---------------
# Out-of-the-box some `mlx` environments cannot JIT mlx-cuda kernels
# because:
#
# 1. libmlx.so is linked against `libnvrtc.so.12` (CUDA 12 NVRTC, the
#    bundled `nvidia-cuda-nvrtc-cu12 12.9.86`).
# 2. The env only contains CUDA 13 toolkit headers
#    (`nvidia-cuda-runtime-13.0.96` shipped with torch 2.10).
# 3. CUDA 13's `cuda_fp6.hpp` and `cuda_fp4.hpp` use bare
#    `__NV_SILENCE_DEPRECATION_BEGIN` lines that NVRTC 12.9 cannot
#    preprocess; every gather/quant kernel JIT fails with
#    `error: this declaration has no storage class or type specifier`
#    and inference returns all-zero arrays.
#
# Fix: stage the matching CUDA 12.9 toolkit headers (downloaded once
# via pip into $MLX_ENV_ROOT/cuda12.9/) and point CUDA_HOME at them.
# NVRTC 12.9 + CUDA 12.9 headers parse cleanly; kernels JIT-compile;
# everything runs on the 4090.
#
# What this sets
# --------------
# - MLX_PYTHON        : absolute path to the mlx env's python
# - CUDA_HOME, CUDA_PATH : staged CUDA 12.9 toolkit root
# - LD_LIBRARY_PATH   : prepended with the env's lib/ (libstdc++,
#                      libnvrtc.so.12, etc.)
#
# The function `mlx_env_stage_cuda12_headers` is idempotent — re-running
# is cheap. Skip the stage step by setting `MLX_SKIP_CUDA12_STAGE=1`.

# --- locate env ---------------------------------------------------------
MLX_PYTHON="${MLX_PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$MLX_PYTHON" || ! -x "$MLX_PYTHON" ]]; then
    echo "ERROR: MLX_PYTHON is not set and no python3/python found on PATH." >&2
    return 1 2>/dev/null || exit 1
fi
MLX_ENV_ROOT="${MLX_ENV_ROOT:-$(cd "$(dirname "$MLX_PYTHON")/.." && pwd)}"
if [[ ! -x "$MLX_ENV_ROOT/bin/python" ]]; then
    echo "ERROR: mlx env not found at $MLX_ENV_ROOT — adjust MLX_ENV_ROOT or MLX_PYTHON." >&2
    return 1 2>/dev/null || exit 1
fi
export MLX_PYTHON="$MLX_ENV_ROOT/bin/python"

# --- stage CUDA 12.9 headers ------------------------------------------
mlx_env_stage_cuda12_headers() {
    local stage="$MLX_ENV_ROOT/cuda12.9"
    # Sentinel: existence of cuda_fp6.hpp tells us we already staged.
    if [[ -f "$stage/include/cuda_fp6.hpp" ]]; then
        return 0
    fi
    echo "[mlx_env] staging CUDA 12.9 headers at $stage (one-time)..."
    local tmp; tmp="$(mktemp -d)"
    mkdir -p "$stage/include" "$stage/bin"
    "$MLX_ENV_ROOT/bin/pip" download --no-deps --quiet -d "$tmp" \
        'nvidia-cuda-runtime-cu12==12.9.79' \
        'nvidia-cuda-cccl-cu12==12.9.27' \
        'nvidia-cuda-nvcc-cu12==12.9.86' || {
            echo "[mlx_env] ERROR: pip download failed (offline? wrong index?)" >&2
            return 1
        }
    local extract="$tmp/extract"; mkdir -p "$extract"
    for w in "$tmp"/*.whl; do
        "$MLX_ENV_ROOT/bin/python" -m zipfile -e "$w" "$extract" >/dev/null
    done
    # Merge runtime + cccl + nvcc includes into one tree (NVRTC's include
    # search path needs everything reachable from a single root).
    cp -a "$extract/nvidia/cuda_runtime/include/." "$stage/include/"
    cp -a "$extract/nvidia/cuda_cccl/include/."    "$stage/include/"
    cp -a "$extract/nvidia/cuda_nvcc/include/."    "$stage/include/"
    # nvcc binary is unused (we go through nvrtc) but stage it anyway in
    # case some future mlx-cuda call needs it.
    [[ -d "$extract/nvidia/cuda_nvcc/bin"  ]] && cp -a "$extract/nvidia/cuda_nvcc/bin/." "$stage/bin/"  || true
    [[ -d "$extract/nvidia/cuda_nvcc/nvvm" ]] && cp -a "$extract/nvidia/cuda_nvcc/nvvm"  "$stage/"     || true
    rm -rf "$tmp"
    echo "[mlx_env] staged $(ls "$stage/include" | wc -l) headers"
}

if [[ -z "${MLX_SKIP_CUDA12_STAGE:-}" ]]; then
    mlx_env_stage_cuda12_headers || return 1 2>/dev/null || exit 1
fi

# --- exports ----------------------------------------------------------
# Prefer the conda-installed CUDA 12.9 toolkit (real `targets/...`
# layout with bin/include/lib) over the legacy pip-staged header-only
# directory at $MLX_ENV_ROOT/cuda12.9. The legacy path is still
# supported as a fallback when only `pip install nvidia-cuda-*` was used.
if [[ -d "$MLX_ENV_ROOT/targets/x86_64-linux/include" \
   && -f "$MLX_ENV_ROOT/targets/x86_64-linux/include/cuda_bf16.h" ]]; then
    export CUDA_HOME="$MLX_ENV_ROOT/targets/x86_64-linux"
else
    export CUDA_HOME="$MLX_ENV_ROOT/cuda12.9"
fi
export CUDA_PATH="$CUDA_HOME"
# Env lib dir first: brings the env's libstdc++ (needed by torch in this
# env) and libnvrtc.so.12 (loaded by libmlx.so via $ORIGIN rpath, but the
# explicit path here makes ldd output legible). Also prepend the cudnn
# and nccl wheel lib dirs — required when libmlx.so was built locally
# from source on Linux; harmless for the upstream wheel which has them
# in its RPATH already.
_MLX_PREPEND_PATHS=(
    "$MLX_ENV_ROOT/lib/python3.12/site-packages/nvidia/cudnn/lib"
    "$MLX_ENV_ROOT/lib/python3.12/site-packages/nvidia/nccl/lib"
    "$MLX_ENV_ROOT/lib"
)
for _p in "${_MLX_PREPEND_PATHS[@]}"; do
    if [[ -d "$_p" ]]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$_p:"*) : ;;
            *) export LD_LIBRARY_PATH="$_p${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
done
unset _MLX_PREPEND_PATHS _p

# --- quick smoke: matmul + gather on GPU -------------------------------
# Caller can opt out with MLX_SKIP_SMOKE=1.
mlx_env_smoke() {
    "$MLX_PYTHON" -c "
import mlx.core as mx
a = mx.random.uniform(shape=(64,64)); mx.eval(a @ a)
idx = mx.array([0,1,2], dtype=mx.uint32); mx.eval(a[idx])
" >/dev/null 2>&1
}

if [[ -z "${MLX_SKIP_SMOKE:-}" ]]; then
    if ! mlx_env_smoke; then
        echo "[mlx_env] WARN: mlx-cuda smoke failed — kernels may not JIT. Check CUDA_HOME=$CUDA_HOME." >&2
    fi
fi
