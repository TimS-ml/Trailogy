#!/usr/bin/env bash
# Build the mix-50k-v3 corpus.
#
# Differences from build_mix.sh:
#   * Pins CONFIG to the v2 yaml (na_plantae 60 %, drop list applied,
#     val files frozen from the v1 mix).
#   * Pre-flight check: requires the v1 mix output to exist at
#     $TRAILOGY_DATA_ROOT/mix-50k/ because v2 hardlinks its val files
#     from there. Fails loud if the v1 dir is missing.
#   * Pre-flight check: requires the NA-Plantae prepared JSONLs to
#     exist (the v2 mix has no PlantNet bucket so $PLANTNET_JSONL is
#     not required, but the NA-Plantae prepared train/val are).
#
# Operator env vars (same as build_mix.sh):
#   PYTHON_BIN          python interpreter to invoke
#   TRAILOGY_DATA_ROOT  external data root (default: <repo>/../data)
#   DATA_MIX_OUTPUT_ROOT  override the per-config default output dir
#   HF_HOME             huggingface cache root
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_MIX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="$(cd "${DATA_MIX_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SRC_ROOT}/.." && pwd)"

CONFIG="${CONFIG:-${DATA_MIX_DIR}/configs/mix-50k-v3.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

_EXTERNAL_DATA_DEFAULT="${REPO_ROOT}/../data"
DATA_ROOT="${TRAILOGY_DATA_ROOT:-${_EXTERNAL_DATA_DEFAULT}}"
DATA_ROOT="$(cd "${DATA_ROOT}" 2>/dev/null && pwd || echo "${DATA_ROOT}")"

echo "== data_mix v3 build =="
echo "HF_HOME              = ${HF_HOME:-<unset, using huggingface_hub default>}"
echo "TRAILOGY_DATA_ROOT   = ${DATA_ROOT}"
echo "DATA_MIX_OUTPUT_ROOT = ${DATA_MIX_OUTPUT_ROOT:-<unset, default <data_root>/mix-50k-v3/>}"
echo "CONFIG               = ${CONFIG}"
echo "PYTHON_BIN           = ${PYTHON_BIN}"
echo

# Pre-flight: NA-Plantae prepared JSONLs must exist (the v2 mix is
# NA-Plantae-only).
for f in \
  "${DATA_ROOT}/inaturalist_na_plantae_prepared/train.jsonl" \
  "${DATA_ROOT}/inaturalist_na_plantae_prepared/val.jsonl"
do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: required NA-Plantae JSONL missing: ${f}" >&2
    echo "       Run data_mix.scripts.prepare_na_plantae first." >&2
    exit 1
  fi
done

# Pre-flight: freeze source must exist.
FREEZE_SRC="${DATA_ROOT}/mix-50k"
if [[ ! -d "${FREEZE_SRC}" ]]; then
  echo "ERROR: freeze_val_from source directory missing: ${FREEZE_SRC}" >&2
  echo "       Build the v1 mix first via build_mix.sh." >&2
  exit 1
fi
for name in val_plant.jsonl val_nonplant.jsonl val_negative.jsonl; do
  if [[ ! -f "${FREEZE_SRC}/${name}" ]]; then
    echo "ERROR: freeze source missing val file: ${FREEZE_SRC}/${name}" >&2
    echo "       Re-run build_mix.sh on the v1 config." >&2
    exit 1
  fi
done

# Run from the public ML module parent so `data_mix.src.mix` resolves.
cd "${SRC_ROOT}"
"${PYTHON_BIN}" -m data_mix.src.mix --config "${CONFIG}"
