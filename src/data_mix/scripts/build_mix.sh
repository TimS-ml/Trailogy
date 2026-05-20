#!/usr/bin/env bash
# Build a data mix. All storage roots are env-driven with safe in-repo
# defaults; operator overrides via environment.
#
# Default CONFIG = NA-Plantae-backed mix-50k.yaml. For the PlantNet-only
# 1.0 recipe, set CONFIG=$DATA_MIX_DIR/configs/mix-50k-plantnet.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_MIX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="$(cd "${DATA_MIX_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${DATA_MIX_DIR}/configs/mix-50k.yaml}"

PYTHON_BIN="${PYTHON_BIN:-python}"

_EXTERNAL_DATA_DEFAULT="$(cd "${SRC_ROOT}/.." && pwd)/data"

echo "== data_mix build =="
echo "HF_HOME              = ${HF_HOME:-<unset, using huggingface_hub default>}"
echo "TRAILOGY_DATA_ROOT   = ${TRAILOGY_DATA_ROOT:-<unset, default ${_EXTERNAL_DATA_DEFAULT}>}"
echo "DATA_MIX_IMAGE_ROOT  = ${DATA_MIX_IMAGE_ROOT:-<unset, default <data_root>/_image_cache>}"
echo "DATA_MIX_OUTPUT_ROOT = ${DATA_MIX_OUTPUT_ROOT:-<unset, default <data_root>/<config-stem>/>}"
echo "PLANTNET_JSONL       = ${PLANTNET_JSONL:-<unset; required only when CONFIG has plant.train > 0>}"
echo "CONFIG               = ${CONFIG}"
echo "PYTHON_BIN           = ${PYTHON_BIN}"
echo

# Pre-flight: PlantNet JSONL is REQUIRED only when the chosen CONFIG
# actually uses the plant bucket (plant.train > 0 OR plant.val > 0).
# NA-Plantae-only mixes skip the plant bucket entirely.
#
# We grep the CONFIG for an integer plant.{train,val} > 0 to decide.
# Operators with a yaml that uses jinja / anchors should set the env
# var BUILD_MIX_SKIP_PLANTNET_PREFLIGHT=1 to bypass this check.
if [[ "${BUILD_MIX_SKIP_PLANTNET_PREFLIGHT:-0}" != "1" ]]; then
  PLANT_USED=0
  if grep -E '^[[:space:]]*plant:' "${CONFIG}" >/dev/null 2>&1; then
    # Capture the numeric train + val from the plant: block. Naive but
    # adequate for our flat yaml schema.
    PLANT_BLOCK="$(awk '
      /^plant:[[:space:]]*$/         { in_block=1; next }
      in_block && /^[^[:space:]]/    { in_block=0 }
      in_block                       { print }
    ' "${CONFIG}")"
    if echo "${PLANT_BLOCK}" | grep -E '^[[:space:]]*(train|val):[[:space:]]*[1-9]' >/dev/null 2>&1; then
      PLANT_USED=1
    fi
  fi
  if [[ "${PLANT_USED}" == "1" ]]; then
    _DEFAULT_PLANTNET="${TRAILOGY_DATA_ROOT:-${_EXTERNAL_DATA_DEFAULT}}/english-desc-v2/train.jsonl"
    RESOLVED_PLANTNET="${PLANTNET_JSONL:-${_DEFAULT_PLANTNET}}"
    if [[ ! -f "${RESOLVED_PLANTNET}" ]]; then
      echo "ERROR: ${CONFIG} declares a non-zero plant bucket but" >&2
      echo "       PlantNet JSONL is not at ${RESOLVED_PLANTNET}" >&2
      echo "       Set PLANTNET_JSONL to override, or set" >&2
      echo "       BUILD_MIX_SKIP_PLANTNET_PREFLIGHT=1 to bypass." >&2
      exit 1
    fi
  fi
fi

# Run from the public ML module parent so `data_mix.src.mix` resolves.
cd "${SRC_ROOT}"
"${PYTHON_BIN}" -m data_mix.src.mix --config "${CONFIG}"
