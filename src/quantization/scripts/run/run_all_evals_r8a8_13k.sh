#!/usr/bin/env bash
# Run the full MLX quantization eval sweep for r8-a8-nokl-step13000.
# Sequentially (Metal serialization), logs all to one file, idempotent
# per step (skips a step if its eval.json already exists).
#
# Env:
#   MLX_PYTHON  : absolute path to a python with mlx, mlx_vlm, mlx_lm,
#                 transformers, safetensors, datasets. Defaults to
#                 `python` on $PATH.
#   HIKE_ROOT   : path to the public ML module parent (`src/`). Defaults
#                 to the parent of src/quantization.
#   SFT_NAME    : SFT result subdir under src/quantization/results.
#                 Default: r8-a8-nokl-step13000.
#
# Usage:
#   cd src/quantization
#   nohup bash scripts/run/run_all_evals_r8a8_13k.sh \
#     > /tmp/r8a8_13k_evals.log 2>&1 &
set -u
set -o pipefail

PY="${MLX_PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QDIR="$(cd "$SCRIPT_DIR/../.." && pwd)"        # src/quantization/
HIKE_ROOT="${HIKE_ROOT:-$(cd "$QDIR/.." && pwd)}"
SFT_NAME="${SFT_NAME:-r8-a8-nokl-step13000}"
# scripts.run.eval imports finetune.src.* for the PlantNet benchmark
export PYTHONPATH="$HIKE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RES="$QDIR/results/$SFT_NAME"
DATA="${EVAL_JSONL:-$HIKE_ROOT/finetune/data/english-desc/test.jsonl}"
N="${EVAL_N:-300}"

cd "$QDIR"
ts() { date '+%H:%M:%S'; }
hdr() { printf '\n========== [%s] %s ==========\n' "$(ts)" "$*"; }

# ---- 1. M0: bf16 reference (mlx_vlm on merged/) ----------------------
if [ ! -f "$RES/eval_M0_bf16.json" ]; then
  hdr "M0: bf16 reference (mlx_vlm)"
  "$PY" -m scripts.run.eval \
    --variant M0_bf16 \
    --loader mlx_vlm \
    --model_dir "$RES/merged" \
    --benchmarks plantnet_val \
    --plantnet_val_jsonl "$DATA" \
    --plantnet_n $N \
    --output_dir "$RES/_eval/M0_bf16" \
    || echo "[$(ts)] M0 eval failed"
  cp -f "$RES/_eval/M0_bf16/eval.json" "$RES/eval_M0_bf16.json" 2>/dev/null
else
  hdr "M0: SKIP (already at eval_M0_bf16.json)"
fi

# ---- 2. M2: g64 bare (no EoRA) ---------------------------------------
if [ ! -f "$RES/eval_M2_g64.json" ]; then
  hdr "M2: g64 bare"
  "$PY" -m scripts.run.eval \
    --variant M2_g64 \
    --loader mlx_vlm \
    --model_dir "$RES/mlx_g64" \
    --benchmarks plantnet_val \
    --plantnet_val_jsonl "$DATA" \
    --plantnet_n $N \
    --output_dir "$RES/_eval/M2_g64" \
    || echo "[$(ts)] M2 eval failed"
  cp -f "$RES/_eval/M2_g64/eval.json" "$RES/eval_M2_g64.json" 2>/dev/null
else
  hdr "M2: SKIP"
fi

# ---- 3. M8b-merge ----------------------------------------------------
if [ ! -f "$RES/eval_M8b_merge.json" ]; then
  hdr "M8b-merge: dequant + EoRA r=64 + requant"
  "$PY" -m scripts.run.eval \
    --variant M8b_merge \
    --loader mlx_vlm \
    --model_dir "$RES/mlx_g64_eora64_merged" \
    --benchmarks plantnet_val \
    --plantnet_val_jsonl "$DATA" \
    --plantnet_n $N \
    --output_dir "$RES/_eval/M8b_merge" \
    || echo "[$(ts)] M8b-merge eval failed"
  cp -f "$RES/_eval/M8b_merge/eval.json" "$RES/eval_M8b_merge.json" 2>/dev/null
else
  hdr "M8b-merge: SKIP"
fi

# ---- 4. M8a: r=32 (truncate r=64) ------------------------------------
if [ ! -f "$RES/eval_M8a_r32.json" ]; then
  hdr "M8a: M2 + EoRA r=32 (truncated from r=64)"
  "$PY" -m scripts.run.eval_eora_only \
    --quant-dir "$RES/mlx_g64" \
    --adapter "$RES/eora/adapters_r64.safetensors" \
    --rank 32 \
    --plantnet-jsonl "$DATA" \
    --plantnet-n $N \
    --output "$RES/eval_M8a_r32.json" \
    --variant-label "M8a_eora_r32" \
    || echo "[$(ts)] M8a eval failed"
else
  hdr "M8a: SKIP"
fi

# ---- 5. M8b: r=64 (full) ---------------------------------------------
if [ ! -f "$RES/eval_M8b_r64.json" ]; then
  hdr "M8b: M2 + EoRA r=64"
  "$PY" -m scripts.run.eval_eora_only \
    --quant-dir "$RES/mlx_g64" \
    --adapter "$RES/eora/adapters_r64.safetensors" \
    --rank 64 \
    --plantnet-jsonl "$DATA" \
    --plantnet-n $N \
    --output "$RES/eval_M8b_r64.json" \
    --variant-label "M8b_eora_r64" \
    || echo "[$(ts)] M8b eval failed"
else
  hdr "M8b: SKIP"
fi

# ---- 6. M1: g128 generate + eval -------------------------------------
if [ ! -d "$RES/mlx_g128" ]; then
  hdr "M1 generate: g128"
  "$PY" -m scripts.run.mlx_vlm_deploy_variant \
    --src "$RES/merged" \
    --dst "$RES/mlx_g128" \
    --q-bits 4 --q-group-size 128 --q-mode affine \
    --also-skip embed_vision embed_audio \
    || echo "[$(ts)] M1 generate failed"
fi
if [ ! -f "$RES/eval_M1_g128.json" ] && [ -d "$RES/mlx_g128" ]; then
  hdr "M1: g128 eval"
  "$PY" -m scripts.run.eval \
    --variant M1_g128 \
    --loader mlx_vlm \
    --model_dir "$RES/mlx_g128" \
    --benchmarks plantnet_val \
    --plantnet_val_jsonl "$DATA" \
    --plantnet_n $N \
    --output_dir "$RES/_eval/M1_g128" \
    || echo "[$(ts)] M1 eval failed"
  cp -f "$RES/_eval/M1_g128/eval.json" "$RES/eval_M1_g128.json" 2>/dev/null
fi

# ---- 7. M3: g32 generate + eval --------------------------------------
if [ ! -d "$RES/mlx_g32" ]; then
  hdr "M3 generate: g32"
  "$PY" -m scripts.run.mlx_vlm_deploy_variant \
    --src "$RES/merged" \
    --dst "$RES/mlx_g32" \
    --q-bits 4 --q-group-size 32 --q-mode affine \
    --also-skip embed_vision embed_audio \
    || echo "[$(ts)] M3 generate failed"
fi
if [ ! -f "$RES/eval_M3_g32.json" ] && [ -d "$RES/mlx_g32" ]; then
  hdr "M3: g32 eval"
  "$PY" -m scripts.run.eval \
    --variant M3_g32 \
    --loader mlx_vlm \
    --model_dir "$RES/mlx_g32" \
    --benchmarks plantnet_val \
    --plantnet_val_jsonl "$DATA" \
    --plantnet_n $N \
    --output_dir "$RES/_eval/M3_g32" \
    || echo "[$(ts)] M3 eval failed"
  cp -f "$RES/_eval/M3_g32/eval.json" "$RES/eval_M3_g32.json" 2>/dev/null
fi

# ---- 8. M6: dynamic_quant generate + eval ----------------------------
if [ ! -d "$RES/mlx_hybrid_dynamic_quant_bpw4_g128" ]; then
  hdr "M6 generate: dynamic_quant (bits=4, g=128; calib 64x512 hardcoded in driver)"
  "$PY" -m scripts.run.mlx_hybrid_quant \
    --method dynamic_quant \
    --hf-path "$RES/merged" \
    --mlx-path "$RES/mlx_hybrid_dynamic_quant_bpw4_g128" \
    --bits 4 --group-size 128 \
    || echo "[$(ts)] M6 generate failed"
fi
if [ ! -f "$RES/eval_M6_dq.json" ] && [ -d "$RES/mlx_hybrid_dynamic_quant_bpw4_g128" ]; then
  hdr "M6: dynamic_quant eval"
  "$PY" -m scripts.run.eval \
    --variant M6_dynamic_quant \
    --loader mlx_vlm \
    --model_dir "$RES/mlx_hybrid_dynamic_quant_bpw4_g128" \
    --benchmarks plantnet_val \
    --plantnet_val_jsonl "$DATA" \
    --plantnet_n $N \
    --output_dir "$RES/_eval/M6_dq" \
    || echo "[$(ts)] M6 eval failed"
  cp -f "$RES/_eval/M6_dq/eval.json" "$RES/eval_M6_dq.json" 2>/dev/null
fi

hdr "DONE — summary:"
for f in "$RES"/eval_M*.json; do
  [ -f "$f" ] || continue
  "$PY" -c "
import json, sys
with open(sys.argv[1]) as f: d = json.load(f)
p = d.get('benchmarks', {}).get('plantnet_val') or d.get('benchmarks', {}).get('plantnet') or d
v = d.get('variant', sys.argv[1].split('/')[-1])
m = p.get('species_match', 0)
sm = p.get('species_matches', 0)
n = p.get('n', 0)
rl = p.get('rouge_l_mean', 0)
print(f'{v:25s} match={m*100:5.1f}% ({sm}/{n}) ROUGE-L={rl:.3f}')
" "$f"
done
