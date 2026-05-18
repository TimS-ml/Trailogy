#!/usr/bin/env bash
# Build the 50K anti-overtraining mix. All storage roots are env-driven
# with safe in-repo defaults; operator overrides via environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_MIX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="$(cd "${DATA_MIX_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${DATA_MIX_DIR}/configs/mix-50k.yaml}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "== data_mix build =="
echo "HF_HOME              = ${HF_HOME:-<unset, using huggingface_hub default>}"
echo "DATA_MIX_IMAGE_ROOT  = ${DATA_MIX_IMAGE_ROOT:-<unset, will fall back to ${DATA_MIX_DIR}/_local/images>}"
echo "DATA_MIX_OUTPUT_ROOT = ${DATA_MIX_OUTPUT_ROOT:-<unset, will fall back to ${SRC_ROOT}/finetune/data/mix-50k>}"
echo "PLANTNET_JSONL       = ${PLANTNET_JSONL:-<unset, will fall back to ${SRC_ROOT}/finetune/data/english-desc-v2/train.jsonl>}"
echo "CONFIG               = ${CONFIG}"
echo "PYTHON_BIN           = ${PYTHON_BIN}"
echo

# Pre-flight: PlantNet JSONL must resolve
RESOLVED_PLANTNET="${PLANTNET_JSONL:-${SRC_ROOT}/finetune/data/english-desc-v2/train.jsonl}"
if [[ ! -f "${RESOLVED_PLANTNET}" ]]; then
  echo "ERROR: PlantNet JSONL not found at ${RESOLVED_PLANTNET}" >&2
  echo "       Set PLANTNET_JSONL to override." >&2
  exit 1
fi

# Run from the public ML module parent so `data_mix.src.mix` resolves.
cd "${SRC_ROOT}"
"${PYTHON_BIN}" -m data_mix.src.mix --config "${CONFIG}"
